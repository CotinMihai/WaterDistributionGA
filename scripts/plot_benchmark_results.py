#!/usr/bin/env python3
"""Generate benchmark convergence plots by strategy and network.

Creates:
- <output>/GA/<network>/convergence.png (mean + std + trials)
- <output>/PSO/<network>/convergence.png
- <output>/SA/<network>/convergence.png
- <output>/ALL/<network>/convergence_comparison.png

Requires convergence_data.json produced by run_benchmarks.py.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _downsample(curve, max_points: int = 2000):
    """Uniformly downsample long curves for lighter plots."""
    if len(curve) <= max_points:
        return np.asarray(curve, dtype=float)
    idx = np.linspace(0, len(curve) - 1, max_points).astype(int)
    return np.asarray(curve, dtype=float)[idx]


def _prepare_curves(trials):
    curves = [t.get('convergence', []) for t in trials]
    curves = [c for c in curves if c]
    if not curves:
        return None

    min_len = min(len(c) for c in curves)
    trimmed = [np.asarray(c[:min_len], dtype=float) for c in curves]
    return trimmed


def _plot_strategy_network(curves, out_path, title):
    min_len = min(len(c) for c in curves)
    stacked = np.vstack([c[:min_len] for c in curves])
    mean_curve = np.mean(stacked, axis=0)
    std_curve = np.std(stacked, axis=0)

    mean_curve = _downsample(mean_curve)
    std_curve = _downsample(std_curve)
    x = np.arange(len(mean_curve))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual trials faintly for context
    for c in curves:
        c_ds = _downsample(c[:min_len])
        ax.plot(np.arange(len(c_ds)), c_ds, color='#999999', alpha=0.2, linewidth=1)

    ax.plot(x, mean_curve, color='#1F77B4', linewidth=2.5, label='Mean')
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                    color='#1F77B4', alpha=0.2, label='Std dev')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Cost', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_network_comparison(net_name, strat_curves, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'GA': '#1F77B4',
        'PSO': '#2CA02C',
        'SA': '#D62728',
    }

    for strat, curves in strat_curves.items():
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        stacked = np.vstack([c[:min_len] for c in curves])
        mean_curve = np.mean(stacked, axis=0)
        mean_curve = _downsample(mean_curve)
        ax.plot(np.arange(len(mean_curve)), mean_curve,
                color=colors.get(strat, '#333333'), linewidth=2.5,
                label=strat)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Cost', fontsize=12)
    ax.set_title(f'{net_name} Network — Strategy Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results', help='Benchmark output directory')
    parser.add_argument('--output', default=None, help='Output plots directory')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output or os.path.join(input_dir, 'plots')

    data_path = os.path.join(input_dir, 'convergence_data.json')
    if not os.path.exists(data_path):
        raise SystemExit(
            f"Missing {data_path}. Re-run benchmarks to generate convergence_data.json."
        )

    with open(data_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    networks = payload.get('networks', {})
    if not networks:
        raise SystemExit('No convergence data found in JSON.')

    # Per-strategy plots
    for net_name, strats in networks.items():
        for strat, trials in strats.items():
            curves = _prepare_curves(trials)
            if curves is None:
                continue
            out_dir = os.path.join(output_dir, strat, net_name)
            _safe_mkdir(out_dir)
            out_path = os.path.join(out_dir, 'convergence.png')
            title = f'{net_name} — {strat} Convergence'
            _plot_strategy_network(curves, out_path, title)

    # Cross-strategy comparisons
    for net_name, strats in networks.items():
        strat_curves = {s: _prepare_curves(t) for s, t in strats.items()}
        out_dir = os.path.join(output_dir, 'ALL', net_name)
        _safe_mkdir(out_dir)
        out_path = os.path.join(out_dir, 'convergence_comparison.png')
        _plot_network_comparison(net_name, strat_curves, out_path)

    print(f"Plots written to: {output_dir}")


if __name__ == '__main__':
    main()
