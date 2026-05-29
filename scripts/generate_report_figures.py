#!/usr/bin/env python3
"""
Generate publication-quality figures for the WDN Optimization Final Report.

Loads results from all experimental configurations, produces:
- SOTA comparison charts
- Convergence curves (per-network & overlay)
- Boxplots & violin plots
- Sensitivity analysis plots (pop size, generation count)
- Feasibility heatmaps
- Wall-time vs quality scatter
- Budget efficiency curves
- Pressure/velocity margin distributions
- Statistical test tables

Usage:
    python scripts/generate_report_figures.py --base-dir /home/mihai/eaa --output report_figures
    python scripts/generate_report_figures.py --manifest report_manifest.json
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ─── Publication style ───────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

STRATEGY_COLORS = {
    'GA': '#2E86AB',
    'SA': '#E8451E',
    'PSO': '#44AF69',
}

NETWORK_COLORS = {
    'Two-Loop': '#5B5EA6',
    'Hanoi': '#D4A843',
    'GoYang': '#E07A5F',
}

KNOWN_BEST = {
    'Two-Loop': 419_000,
    'Hanoi': 6_081_000,
    'GoYang': 177_010_359,
}

NETWORK_ORDER = ['Two-Loop', 'Hanoi', 'GoYang']
STRATEGY_ORDER = ['GA', 'SA', 'PSO']


# ─── Data loading ────────────────────────────────────────────────────────────

def load_summary(dir_path):
    """Load summary.csv from a results directory."""
    path = os.path.join(dir_path, 'summary.csv')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_detailed(dir_path):
    """Load detailed_results.csv from a results directory."""
    path = os.path.join(dir_path, 'detailed_results.csv')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_convergence(dir_path):
    """Load convergence_data.json from a results directory."""
    path = os.path.join(dir_path, 'convergence_data.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_float(val, default=float('nan')):
    try:
        return float(str(val).replace('%', '').replace(',', ''))
    except (ValueError, TypeError):
        return default


def parse_int(val, default=0):
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return default


def infer_config_meta(label):
    """Infer trial/generation/population/core metadata from a config label."""
    if 'BASELINE' in label:
        return {
            'trials': 10,
            'gen': 800,
            'pop': 200,
            'cores': 2,
            'budget': 800 * 200,
            'label': 'Baseline',
        }

    match = re.search(r'(\d+)T_(\d+)G_(\d+)P_(\d+)C', label)
    if match:
        trials, gen, pop, cores = [int(v) for v in match.groups()]
        return {
            'trials': trials,
            'gen': gen,
            'pop': pop,
            'cores': cores,
            'budget': gen * pop,
            'label': label,
        }

    # Legacy matrix names.
    if 'POP50' in label:
        return {'trials': None, 'gen': 800, 'pop': 50, 'cores': None, 'budget': 800 * 50, 'label': label}
    if 'POP100' in label:
        return {'trials': None, 'gen': 800, 'pop': 100, 'cores': None, 'budget': 800 * 100, 'label': label}
    if 'POP400' in label:
        return {'trials': None, 'gen': 800, 'pop': 400, 'cores': None, 'budget': 800 * 400, 'label': label}
    if 'GEN200' in label:
        return {'trials': None, 'gen': 200, 'pop': 200, 'cores': None, 'budget': 200 * 200, 'label': label}
    if 'GEN400' in label:
        return {'trials': None, 'gen': 400, 'pop': 200, 'cores': None, 'budget': 400 * 200, 'label': label}
    if 'GEN1600' in label:
        return {'trials': None, 'gen': 1600, 'pop': 200, 'cores': None, 'budget': 1600 * 200, 'label': label}

    return None


def experiment_points(experiments):
    """Return experiment metadata points for report-core/fallback sensitivity plots."""
    points = []
    for label, path in experiments.items():
        meta = infer_config_meta(label)
        if not meta:
            continue
        meta = dict(meta)
        meta['name'] = label
        meta['path'] = path
        points.append(meta)

    # Drop smoke from report sensitivity figures when stronger configs exist.
    non_smoke = [p for p in points if 'SMOKE' not in p['name']]
    if len(non_smoke) >= 2:
        points = non_smoke

    return points


# ─── Core experimental configs ───────────────────────────────────────────────

def experiments_from_manifest(base_dir, manifest_path):
    """Load report-approved experiment directories from report_manifest.json."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = Path(base_dir) / manifest_file
    if not manifest_file.exists():
        return None

    with manifest_file.open(encoding='utf-8') as f:
        manifest = json.load(f)

    experiments = {}
    baseline = manifest.get('baseline', {})
    baseline_path = baseline.get('path')
    if baseline_path:
        baseline_dir = Path(baseline_path)
        if not baseline_dir.is_absolute():
            baseline_dir = Path(base_dir) / baseline_dir
        if (baseline_dir / 'summary.csv').exists():
            experiments['BASELINE (pop=200, gen=800)'] = str(baseline_dir)

    for entry in manifest.get('experiments', []):
        path = entry.get('path')
        label = entry.get('label')
        if not path or not label:
            continue
        exp_dir = Path(path)
        if not exp_dir.is_absolute():
            exp_dir = Path(base_dir) / exp_dir
        if (exp_dir / 'summary.csv').exists():
            experiments[label] = str(exp_dir)

    return experiments


def discover_experiments(base_dir):
    """
    Auto-discover experiment directories matching known patterns.
    Scans both the base_dir (for legacy runs) and base_dir/results/ (for matrix runs).
    Returns dict: {config_label: dir_path}
    """
    experiments = {}

    # Baseline
    baseline_dir = os.path.join(base_dir, 'FINAL_10T_800G_200P_2C')
    if os.path.isdir(baseline_dir):
        experiments['BASELINE (pop=200, gen=800)'] = baseline_dir

    # Named experiment dirs (legacy, in project root)
    patterns = [
        'EXP0_QUICK', 'EXP1_POP50', 'EXP1_POP100', 'EXP1_POP400',
        'EXP2_GEN200', 'EXP2_GEN400', 'EXP2_GEN1600',
        'EXP3_TWOLOOP_30T', 'EXP3_HANOI_30T', 'EXP3_GOYANG_30T',
        'EXP4_QUICK',
    ]
    for pat in patterns:
        for search_dir in [base_dir, os.path.join(base_dir, 'results')]:
            d = os.path.join(search_dir, pat)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, 'summary.csv')):
                experiments[pat] = d
                break

    # Scan results/ directory for any other experiment dirs
    results_dir = os.path.join(base_dir, 'results')
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            full = os.path.join(results_dir, name)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'summary.csv')):
                if name not in [os.path.basename(v) for v in experiments.values()]:
                    experiments[name] = full

    # Also check base dir for older run dirs
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, 'summary.csv')):
            if name not in experiments and name not in [os.path.basename(v) for v in experiments.values()]:
                experiments[name] = full

    return experiments


# ─── Figure generators ───────────────────────────────────────────────────────

def fig01_sota_comparison(baseline_summary, output_dir):
    """Bar chart: Our best cost vs SOTA per network × strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        net_rows = [r for r in baseline_summary if r['Network'] == net]
        if not net_rows:
            continue

        strats = []
        our_costs = []
        for s in STRATEGY_ORDER:
            row = next((r for r in net_rows if r['Strategy'] == s), None)
            if row:
                strats.append(s)
                our_costs.append(parse_float(row['Best_Cost']))

        x = np.arange(len(strats))
        width = 0.35
        bars_ours = ax.bar(x - width/2, our_costs,
                           width, label='Our Implementation',
                           color=[STRATEGY_COLORS[s] for s in strats],
                           edgecolor='white', linewidth=0.5)
        bars_sota = ax.bar(x + width/2, [KNOWN_BEST[net]] * len(strats),
                           width, label='SOP-WDN (SOTA)',
                           color='#CCCCCC', edgecolor='#999999', linewidth=0.5)

        ax.set_title(net, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strats)
        ax.set_ylabel('Best Cost (USD)' if idx == 0 else '')

        # Add gap% annotations
        for i, (cost, s) in enumerate(zip(our_costs, strats)):
            gap = ((cost - KNOWN_BEST[net]) / KNOWN_BEST[net]) * 100
            color = '#2E7D32' if gap < 0 else '#C62828'
            sign = '' if gap < 0 else '+'
            ax.annotate(f'{sign}{gap:.1f}%',
                       xy=(i - width/2, cost),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=8, fontweight='bold', color=color)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Comparison with SOP-WDN Paper Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig01_sota_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig02_boxplots(baseline_detailed, output_dir):
    """Boxplots: cost distribution per network × strategy (baseline run)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        net_rows = [r for r in baseline_detailed if r['Network'] == net]

        data = []
        labels = []
        colors = []
        for s in STRATEGY_ORDER:
            strat_rows = [r for r in net_rows if r['Strategy'] == s
                          and r.get('Feasible') == 'True']
            costs = [parse_float(r['Best_Cost']) for r in strat_rows]
            if costs:
                data.append(costs)
                labels.append(s)
                colors.append(STRATEGY_COLORS[s])

        if not data:
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                       widths=0.5, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='white',
                                      markeredgecolor='black', markersize=5))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        # SOTA reference line
        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle='--',
                   linewidth=1.5, alpha=0.7, label='SOTA')
        ax.set_title(net, fontweight='bold')
        ax.set_ylabel('Best Cost' if idx == 0 else '')
        if idx == 2:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Cost Distribution Across 10 Independent Trials', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig02_boxplots.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig03_convergence(baseline_dir, output_dir, all_experiments=None):
    """Convergence curves: mean ± std per network, merging data from multiple dirs."""
    # Merge convergence data from all experiment dirs
    merged_networks = {}
    dirs_to_check = [baseline_dir]
    if all_experiments:
        dirs_to_check = list(set(all_experiments.values()))

    for d in dirs_to_check:
        conv_data = load_convergence(d)
        networks = conv_data.get('networks', {})
        for net, strats in networks.items():
            if net not in merged_networks:
                merged_networks[net] = {}
            for s, trials in strats.items():
                if s not in merged_networks[net]:
                    merged_networks[net][s] = []
                merged_networks[net][s].extend(trials)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        if net not in merged_networks:
            ax.set_title(net, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            continue

        # Cap curves at 10x SOTA — trials with terminal cost beyond this are
        # penalty-blown infeasible runs and would dominate the log scale.
        cost_cap = KNOWN_BEST[net] * 10.0
        for s in STRATEGY_ORDER:
            if s not in merged_networks[net]:
                continue
            trials = merged_networks[net][s]
            curves = [t.get('convergence', []) for t in trials]
            # Filter out curves whose final value is penalty-blown (infeasible)
            feasible_curves = [c for c in curves if c and c[-1] <= cost_cap]
            n_total = len([c for c in curves if c])
            n_dropped = n_total - len(feasible_curves)
            if not feasible_curves:
                continue

            min_len = min(len(c) for c in feasible_curves)
            stacked = np.array([c[:min_len] for c in feasible_curves], dtype=float)
            # Use median + percentile band instead of mean + std for robustness
            mid = np.median(stacked, axis=0)
            lo = np.percentile(stacked, 25, axis=0)
            hi = np.percentile(stacked, 75, axis=0)

            # Downsample for cleaner plots
            max_pts = 1000
            if len(mid) > max_pts:
                idx_ds = np.linspace(0, len(mid)-1, max_pts).astype(int)
                mid = mid[idx_ds]
                lo = lo[idx_ds]
                hi = hi[idx_ds]
                x = idx_ds
            else:
                x = np.arange(len(mid))

            label = s if n_dropped == 0 else f'{s} ({n_dropped} infeasible dropped)'
            ax.plot(x, mid, color=STRATEGY_COLORS[s],
                   linewidth=2, label=label, zorder=3)
            ax.fill_between(x, lo, hi,
                           color=STRATEGY_COLORS[s], alpha=0.15)

        # SOTA reference
        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle='--',
                   linewidth=1, alpha=0.6, label='SOTA')
        ax.set_yscale('log')
        ax.set_xlabel('Generation / Iteration')
        ax.set_ylabel('Best Cost' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        # Tight y-band around SOTA so the iter-0 penalized spike does not
        # dominate the log scale.
        ax.set_ylim(KNOWN_BEST[net] * 0.7, KNOWN_BEST[net] * 5.0)
        ax.legend(fontsize=9)

    fig.suptitle('Convergence (median + IQR, feasible-final trials only)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig03_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig04_population_sensitivity(experiments, output_dir):
    """Line plot: population sensitivity at fixed generation count (G=200)."""
    points = [
        p for p in experiment_points(experiments)
        if p['gen'] == 200 and p['pop'] in {100, 150, 200} and 'BASELINE' not in p['name']
    ]
    if len(points) < 1:
        print('  ⚠ Skipping fig04: no fixed-generation population configs')
        return

    points = sorted(points, key=lambda p: p['pop'])
    x = np.arange(len(points))
    xlabels = [f"P{p['pop']}" for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        for s in STRATEGY_ORDER:
            best_costs = []
            mean_costs = []
            valid_x = []
            for pos, point in enumerate(points):
                summary = load_summary(point['path'])
                row = next((r for r in summary if r['Network'] == net and r['Strategy'] == s), None)
                if row:
                    best_costs.append(parse_float(row['Best_Cost']))
                    mean_costs.append(parse_float(row['Mean_Cost']))
                    valid_x.append(pos)

            if valid_x:
                ax.plot(valid_x, best_costs, 'o-',
                       color=STRATEGY_COLORS[s], linewidth=2,
                       markersize=6, label=f'{s} (best)')
                ax.plot(valid_x, mean_costs, 's--',
                       color=STRATEGY_COLORS[s], linewidth=1,
                       markersize=4, alpha=0.6, label=f'{s} (mean)')

        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle=':',
                   linewidth=1, alpha=0.5, label='SOTA')
        ax.set_xlabel('Population size (G=200)')
        ax.set_ylabel('Cost' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle('Population Sensitivity at Fixed Generation Count', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig04_population_sensitivity.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig05_generation_sensitivity(experiments, output_dir):
    """Line plot: generation sensitivity at fixed population size (P=100)."""
    points = [
        p for p in experiment_points(experiments)
        if p['pop'] == 100 and p['gen'] in {100, 200, 400} and 'BASELINE' not in p['name']
    ]
    if len(points) < 1:
        print('  ⚠ Skipping fig05: no fixed-population generation configs')
        return

    points = sorted(points, key=lambda p: p['gen'])
    x = np.arange(len(points))
    xlabels = [f"G{p['gen']}" for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        for s in STRATEGY_ORDER:
            best_costs = []
            mean_costs = []
            valid_x = []
            for pos, point in enumerate(points):
                summary = load_summary(point['path'])
                row = next((r for r in summary if r['Network'] == net and r['Strategy'] == s), None)
                if row:
                    best_costs.append(parse_float(row['Best_Cost']))
                    mean_costs.append(parse_float(row['Mean_Cost']))
                    valid_x.append(pos)

            if valid_x:
                ax.plot(valid_x, best_costs, 'o-',
                       color=STRATEGY_COLORS[s], linewidth=2,
                       markersize=6, label=f'{s} (best)')
                ax.plot(valid_x, mean_costs, 's--',
                       color=STRATEGY_COLORS[s], linewidth=1,
                       markersize=4, alpha=0.6, label=f'{s} (mean)')

        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle=':',
                   linewidth=1, alpha=0.5, label='SOTA')
        ax.set_xlabel('Generations (P=100)')
        ax.set_ylabel('Cost' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle('Generation Sensitivity at Fixed Population Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig05_generation_sensitivity.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig06_feasibility_heatmap(experiments, output_dir):
    """Heatmap: feasibility rate across configs × strategies × networks."""
    configs = sorted(experiments.keys())

    all_data = {}
    for label, d in experiments.items():
        summary = load_summary(d)
        for row in summary:
            key = (row['Network'], row['Strategy'])
            trials = parse_int(row.get('Trials', 10), 10)
            feasible = parse_int(row.get('Feasible_Count', 0))
            rate = (feasible / trials * 100) if trials > 0 else 0
            if label not in all_data:
                all_data[label] = {}
            all_data[label][key] = rate

    # Build matrix
    row_labels = []
    col_labels = []
    for net in NETWORK_ORDER:
        for s in STRATEGY_ORDER:
            col_labels.append(f'{net}\n{s}')

    matrix = []
    for label in configs:
        row = []
        for net in NETWORK_ORDER:
            for s in STRATEGY_ORDER:
                row.append(all_data.get(label, {}).get((net, s), float('nan')))
        matrix.append(row)
        row_labels.append(label.replace('BASELINE (pop=200, gen=800)', 'BASELINE'))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, max(4, len(configs) * 0.6 + 1)))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, ha='center')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='white' if val < 50 else 'black')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Feasibility Rate (%)')
    ax.set_title('Feasibility Rate Across All Configurations', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig06_feasibility_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig07_walltime_vs_quality(experiments, output_dir):
    """Scatter: runtime vs best cost, colored by strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        for label, d in experiments.items():
            detailed = load_detailed(d)
            net_rows = [r for r in detailed if r['Network'] == net]

            for s in STRATEGY_ORDER:
                strat_rows = [r for r in net_rows if r['Strategy'] == s]
                times = [parse_float(r.get('Wall_Time_s', 0)) for r in strat_rows]
                costs = [parse_float(r['Best_Cost']) for r in strat_rows]
                feasible = [r.get('Feasible', 'True') == 'True' for r in strat_rows]

                if times and costs:
                    feas_t = [t for t, f in zip(times, feasible) if f]
                    feas_c = [c for c, f in zip(costs, feasible) if f]
                    infeas_t = [t for t, f in zip(times, feasible) if not f]
                    infeas_c = [c for c, f in zip(costs, feasible) if not f]

                    if feas_t:
                        ax.scatter(feas_t, feas_c, c=STRATEGY_COLORS[s],
                                  s=20, alpha=0.6, label=s if label == list(experiments.keys())[0] else '')
                    if infeas_t:
                        ax.scatter(infeas_t, infeas_c, c=STRATEGY_COLORS[s],
                                  s=20, alpha=0.3, marker='x')

        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle='--',
                   linewidth=1, alpha=0.5)
        ax.set_xlabel('Wall Time (s)')
        ax.set_ylabel('Best Cost' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.set_yscale('log')
        # Clip y-axis to a sane band around SOTA so infeasible
        # penalty-blown points do not destroy the visual scale.
        ax.set_ylim(KNOWN_BEST[net] * 0.3, KNOWN_BEST[net] * 3.0)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Wall Time vs. Solution Quality', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig07_walltime_vs_quality.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig08_budget_efficiency(experiments, output_dir):
    """Line plot: best FEASIBLE cost vs total evaluations (gen × pop).
    Drops configs in which the (net, strat) had zero feasible trials."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    budget_data = {}  # {(net, strat): [(budget, best_feasible_cost), ...]}

    for label, d in experiments.items():
        meta = infer_config_meta(label)
        if meta is None:
            continue
        budget = meta['budget']
        detailed = load_detailed(d)
        for net in NETWORK_ORDER:
            for s in STRATEGY_ORDER:
                rows = [r for r in detailed if r['Network'] == net
                        and r['Strategy'] == s and r.get('Feasible') == 'True']
                if not rows:
                    continue
                best_cost = min(parse_float(r['Best_Cost']) for r in rows)
                key = (net, s)
                if key not in budget_data:
                    budget_data[key] = []
                budget_data[key].append((budget, best_cost))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        for s in STRATEGY_ORDER:
            key = (net, s)
            if key in budget_data:
                points = sorted(budget_data[key], key=lambda x: x[0])
                budgets = [p[0] for p in points]
                costs = [p[1] for p in points]
                ax.plot(budgets, costs, 'o-', color=STRATEGY_COLORS[s],
                       linewidth=2, markersize=6, label=s)

        ax.axhline(KNOWN_BEST[net], color='#E53935', linestyle=':',
                   linewidth=1, alpha=0.5, label='SOTA')
        ax.set_xlabel('Total Evaluations (gen × pop)')
        ax.set_ylabel('Best Cost' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.set_xscale('log')
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Computation Budget Efficiency (best feasible cost only)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig08_budget_efficiency.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig09_penalty_decomposition(baseline_detailed, output_dir):
    """Stacked bar: PP and VP for best trial per strategy × network."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        net_rows = [r for r in baseline_detailed if r['Network'] == net]

        strats = []
        pps = []
        vps = []
        for s in STRATEGY_ORDER:
            strat_rows = [r for r in net_rows if r['Strategy'] == s]
            if strat_rows:
                # Take the best trial (lowest cost feasible)
                feasible_rows = [r for r in strat_rows if r.get('Feasible') == 'True']
                if feasible_rows:
                    best = min(feasible_rows, key=lambda r: parse_float(r['Best_Cost']))
                else:
                    best = min(strat_rows, key=lambda r: parse_float(r['Penalized_Cost']))
                strats.append(s)
                pps.append(parse_float(best.get('PP', 1)))
                vps.append(parse_float(best.get('VP', 1)))

        if not strats:
            continue

        x = np.arange(len(strats))
        width = 0.35
        ax.bar(x - width/2, pps, width, label='PP (Pressure)',
               color='#5C6BC0', alpha=0.8)
        ax.bar(x + width/2, vps, width, label='VP (Velocity)',
               color='#FF7043', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(strats)
        ax.set_ylabel('Penalty Value' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8, alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Pressure & Velocity Penalty Decomposition (Best Trial)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig09_penalty_decomposition.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig10_pressure_margin(baseline_detailed, output_dir):
    """Violin plot: distribution of min pressure across trials."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        net_rows = [r for r in baseline_detailed if r['Network'] == net]

        data = []
        labels = []
        for s in STRATEGY_ORDER:
            strat_rows = [r for r in net_rows if r['Strategy'] == s]
            pressures = [parse_float(r.get('Min_Pressure', 0)) for r in strat_rows]
            if pressures:
                data.append(pressures)
                labels.append(s)

        if not data:
            continue

        vp = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(STRATEGY_COLORS[labels[i]])
            body.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)

        # Min pressure threshold
        min_p = 30.0 if net != 'GoYang' else 15.0
        ax.axhline(min_p, color='#E53935', linestyle='--',
                   linewidth=1.5, label=f'Min required ({min_p}m)')
        ax.set_ylabel('Min Nodal Pressure (m)' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Minimum Pressure Distribution Across Trials', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig10_pressure_margin.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig11_velocity_feasibility(experiments, output_dir):
    """Bar chart: % of trials with velocity feasibility per strategy × network."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        config_labels = []
        ga_rates = []
        sa_rates = []
        pso_rates = []

        for label, d in sorted(experiments.items()):
            detailed = load_detailed(d)
            net_rows = [r for r in detailed if r['Network'] == net]
            if not net_rows:
                continue

            short_label = label.replace('BASELINE (pop=200, gen=800)', 'BASE')
            config_labels.append(short_label[:12])

            for s, rate_list in [('GA', ga_rates), ('SA', sa_rates), ('PSO', pso_rates)]:
                strat_rows = [r for r in net_rows if r['Strategy'] == s]
                if strat_rows:
                    vf = sum(1 for r in strat_rows if r.get('Velocity_Feasible') == 'True')
                    rate_list.append(vf / len(strat_rows) * 100)
                else:
                    rate_list.append(0)

        if not config_labels:
            continue

        x = np.arange(len(config_labels))
        width = 0.25
        ax.bar(x - width, ga_rates, width, label='GA', color=STRATEGY_COLORS['GA'])
        ax.bar(x, sa_rates, width, label='SA', color=STRATEGY_COLORS['SA'])
        ax.bar(x + width, pso_rates, width, label='PSO', color=STRATEGY_COLORS['PSO'])

        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Velocity Feasibility (%)' if idx == 0 else '')
        ax.set_ylim(0, 110)
        ax.set_title(net, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Velocity Feasibility Rate', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig11_velocity_feasibility.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig12_gap_violin(experiments, output_dir):
    """Violin plot: distribution of gap-to-SOTA across 30-trial experiments."""
    exp30_dirs = {k: v for k, v in experiments.items() if '30T' in k}

    if not exp30_dirs:
        # Fallback to baseline
        exp30_dirs = {k: v for k, v in experiments.items() if 'BASELINE' in k}
        if not exp30_dirs:
            print('  ⚠ Skipping fig12: no 30-trial or baseline data')
            return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        found_data = False
        data = []
        labels = []

        for label, d in exp30_dirs.items():
            detailed = load_detailed(d)
            net_rows = [r for r in detailed if r['Network'] == net]
            if not net_rows:
                continue

            for s in STRATEGY_ORDER:
                strat_rows = [r for r in net_rows if r['Strategy'] == s]
                gaps = [((parse_float(r['Best_Cost']) - KNOWN_BEST[net]) / KNOWN_BEST[net]) * 100
                       for r in strat_rows if r.get('Feasible') == 'True']
                if gaps:
                    data.append(gaps)
                    labels.append(s)
                    found_data = True

        if not found_data:
            continue

        vp = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(STRATEGY_COLORS.get(labels[i], '#888'))
            body.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels)
        ax.axhline(0, color='#E53935', linestyle='--', linewidth=1.5, alpha=0.7, label='SOTA')
        ax.set_ylabel('Gap to SOTA (%)' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Gap-to-SOTA Distribution (Feasible Trials Only)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig12_gap_violin.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig13_summary_table(experiments, output_dir):
    """Summary ranking table as a figure."""
    # Collect best results per network × strategy across ALL experiments
    best_results = {}

    for label, d in experiments.items():
        summary = load_summary(d)
        for row in summary:
            key = (row['Network'], row['Strategy'])
            cost = parse_float(row['Best_Cost'])
            if key not in best_results or cost < best_results[key]['cost']:
                best_results[key] = {
                    'cost': cost,
                    'config': label,
                    'mean': parse_float(row.get('Mean_Cost', 0)),
                    'std': parse_float(row.get('Std_Cost', 0)),
                    'feasible': row.get('Feasible_Count', '?'),
                    'trials': row.get('Trials', '?'),
                    'time': parse_float(row.get('Mean_Time_s', 0)),
                }

    # Build table data
    cell_text = []
    row_labels = []
    for net in NETWORK_ORDER:
        for s in STRATEGY_ORDER:
            key = (net, s)
            if key in best_results:
                r = best_results[key]
                gap = ((r['cost'] - KNOWN_BEST[net]) / KNOWN_BEST[net]) * 100
                sign = '+' if gap > 0 else ''
                row_labels.append(f'{net} / {s}')
                cell_text.append([
                    f"{r['cost']:,.0f}",
                    f"{r['mean']:,.0f}",
                    f"{r['std']:,.0f}",
                    f"{r['feasible']}/{r['trials']}",
                    f"{sign}{gap:.1f}%",
                    f"{r['time']:.1f}s",
                    r['config'][:20],
                ])

    col_labels = ['Best Cost', 'Mean Cost', 'Std', 'Feasible', 'SOTA Gap', 'Avg Time', 'Config']

    fig, ax = plt.subplots(figsize=(16, max(3, len(row_labels) * 0.4 + 1)))
    ax.axis('off')

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Color code gap column
    for i, row in enumerate(cell_text):
        gap_val = parse_float(row[4])
        if gap_val < 0:
            table[i+1, 4].set_facecolor('#C8E6C9')
        else:
            table[i+1, 4].set_facecolor('#FFCDD2')

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Best Results Summary Across All Configurations', fontweight='bold',
                fontsize=14, pad=20)
    path = os.path.join(output_dir, 'fig13_summary_table.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig14_normalized_convergence(baseline_dir, output_dir, all_experiments=None):
    """Normalized convergence: % of final cost achieved vs % of budget spent."""
    # Merge convergence data from all experiment dirs
    merged_networks = {}
    dirs_to_check = [baseline_dir]
    if all_experiments:
        dirs_to_check = list(set(all_experiments.values()))
    for d in dirs_to_check:
        conv_data = load_convergence(d)
        networks = conv_data.get('networks', {})
        for net, strats in networks.items():
            if net not in merged_networks:
                merged_networks[net] = {}
            for s, trials in strats.items():
                if s not in merged_networks[net]:
                    merged_networks[net][s] = []
                merged_networks[net][s].extend(trials)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        if net not in merged_networks:
            ax.set_title(net, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            continue

        for s in STRATEGY_ORDER:
            if s not in merged_networks[net]:
                continue
            trials = merged_networks[net][s]
            curves = [t.get('convergence', []) for t in trials]
            curves = [np.array(c, dtype=float) for c in curves if c]
            if not curves:
                continue

            min_len = min(len(c) for c in curves)
            stacked = np.vstack([c[:min_len] for c in curves])
            mean_c = np.mean(stacked, axis=0)

            # Normalize: 0 = initial cost, 1 = final cost
            initial = mean_c[0]
            final = mean_c[-1]
            if abs(initial - final) < 1e-6:
                continue
            normalized = (initial - mean_c) / (initial - final)
            budget_pct = np.linspace(0, 100, len(normalized))

            ax.plot(budget_pct, normalized * 100,
                   color=STRATEGY_COLORS[s], linewidth=2, label=s)

        ax.set_xlabel('Budget Spent (%)')
        ax.set_ylabel('Cost Reduction Achieved (%)' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)

    fig.suptitle('Normalized Convergence Speed', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig14_normalized_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def fig15_runtime_comparison(experiments, output_dir):
    """Grouped bar chart: mean runtime per strategy across configs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        config_labels = []
        times_by_strat = {s: [] for s in STRATEGY_ORDER}

        for label, d in sorted(experiments.items()):
            summary = load_summary(d)
            net_rows = [r for r in summary if r['Network'] == net]
            if not net_rows:
                continue

            short = label.replace('BASELINE (pop=200, gen=800)', 'BASE')[:12]
            config_labels.append(short)
            for s in STRATEGY_ORDER:
                row = next((r for r in net_rows if r['Strategy'] == s), None)
                times_by_strat[s].append(parse_float(row.get('Mean_Time_s', 0)) if row else 0)

        if not config_labels:
            continue

        x = np.arange(len(config_labels))
        width = 0.25
        for i, s in enumerate(STRATEGY_ORDER):
            ax.bar(x + (i - 1) * width, times_by_strat[s], width,
                  label=s, color=STRATEGY_COLORS[s], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean Runtime (s)' if idx == 0 else '')
        ax.set_title(net, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Mean Runtime Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig15_runtime_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


def generate_statistical_tests(experiments, output_dir):
    """Run Wilcoxon rank-sum tests and save results."""
    from scipy import stats

    results = []
    for label, d in experiments.items():
        detailed = load_detailed(d)
        for net in NETWORK_ORDER:
            net_rows = [r for r in detailed if r['Network'] == net]
            strat_costs = {}
            for s in STRATEGY_ORDER:
                costs = [parse_float(r['Best_Cost']) for r in net_rows
                        if r['Strategy'] == s and r.get('Feasible') == 'True']
                if costs:
                    strat_costs[s] = costs

            # Pairwise tests
            pairs = [('GA', 'SA'), ('GA', 'PSO'), ('SA', 'PSO')]
            for s1, s2 in pairs:
                if s1 in strat_costs and s2 in strat_costs:
                    if len(strat_costs[s1]) >= 3 and len(strat_costs[s2]) >= 3:
                        try:
                            stat, p = stats.mannwhitneyu(strat_costs[s1], strat_costs[s2],
                                                         alternative='two-sided')
                            results.append({
                                'config': label[:30],
                                'network': net,
                                'comparison': f'{s1} vs {s2}',
                                'U_statistic': f'{stat:.1f}',
                                'p_value': f'{p:.4f}',
                                'significant': 'Yes' if p < 0.05 else 'No',
                            })
                        except Exception:
                            pass

    if results:
        path = os.path.join(output_dir, 'statistical_tests.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'  ✓ {path}')
    else:
        print('  ⚠ No statistical tests produced (insufficient data)')


def generate_latex_table(experiments, output_dir):
    """Generate a LaTeX-formatted summary table."""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Summary of Best Results Across All Configurations}')
    lines.append(r'\label{tab:summary}')
    lines.append(r'\begin{tabular}{llrrrrl}')
    lines.append(r'\hline')
    lines.append(r'Network & Strategy & Best Cost & Mean Cost & Std & Feasible & SOTA Gap \\')
    lines.append(r'\hline')

    for label, d in experiments.items():
        if 'BASELINE' not in label:
            continue
        summary = load_summary(d)
        for net in NETWORK_ORDER:
            for s in STRATEGY_ORDER:
                row = next((r for r in summary if r['Network'] == net and r['Strategy'] == s), None)
                if row:
                    cost = parse_float(row['Best_Cost'])
                    gap = ((cost - KNOWN_BEST[net]) / KNOWN_BEST[net]) * 100
                    sign = '+' if gap > 0 else ''
                    lines.append(
                        f"{net} & {s} & {cost:,.0f} & "
                        f"{parse_float(row.get('Mean_Cost', 0)):,.0f} & "
                        f"{parse_float(row.get('Std_Cost', 0)):,.0f} & "
                        f"{row.get('Feasible_Count', '?')}/{row.get('Trials', '?')} & "
                        f"{sign}{gap:.1f}\\% \\\\"
                    )

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(output_dir, 'summary_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ {path}')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate final report figures')
    parser.add_argument('--base-dir', default='/home/mihai/eaa',
                       help='Project root containing result directories')
    parser.add_argument('--output', default='report_figures',
                       help='Output directory for figures')
    parser.add_argument('--manifest', default='report_manifest.json',
                       help='Report manifest to use before auto-discovery')
    parser.add_argument('--auto-discover', action='store_true',
                       help='Ignore manifest and scan all result directories')
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = os.path.join(base_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  WDN Final Report — Figure Generator')
    print(f'{"="*60}')
    print(f'  Base dir: {base_dir}')
    print(f'  Output:   {output_dir}')

    # Discover experiments
    experiments = None
    source_desc = 'auto-discovery'
    if not args.auto_discover:
        experiments = experiments_from_manifest(base_dir, args.manifest)
        if experiments is not None:
            source_desc = f'manifest: {args.manifest}'
    if experiments is None:
        experiments = discover_experiments(base_dir)

    print(f'\n  Experiment source: {source_desc}')
    print(f'  Found {len(experiments)} experiment configurations:')
    for label, d in experiments.items():
        print(f'    • {label}: {d}')

    # Load baseline data
    baseline_dir = experiments.get('BASELINE (pop=200, gen=800)')
    if not baseline_dir:
        # Fallback to first available
        baseline_dir = next(iter(experiments.values()), None)
    if not baseline_dir:
        print('ERROR: No experiment data found!')
        sys.exit(1)

    baseline_summary = load_summary(baseline_dir)
    baseline_detailed = load_detailed(baseline_dir)

    print(f'\n  Generating figures...\n')

    # Generate all figures
    try:
        fig01_sota_comparison(baseline_summary, output_dir)
    except Exception as e:
        print(f'  ✗ fig01: {e}')

    try:
        fig02_boxplots(baseline_detailed, output_dir)
    except Exception as e:
        print(f'  ✗ fig02: {e}')

    try:
        fig03_convergence(baseline_dir, output_dir, all_experiments=experiments)
    except Exception as e:
        print(f'  ✗ fig03: {e}')

    try:
        fig04_population_sensitivity(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig04: {e}')

    try:
        fig05_generation_sensitivity(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig05: {e}')

    try:
        fig06_feasibility_heatmap(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig06: {e}')

    try:
        fig07_walltime_vs_quality(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig07: {e}')

    try:
        fig08_budget_efficiency(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig08: {e}')

    try:
        fig09_penalty_decomposition(baseline_detailed, output_dir)
    except Exception as e:
        print(f'  ✗ fig09: {e}')

    try:
        fig10_pressure_margin(baseline_detailed, output_dir)
    except Exception as e:
        print(f'  ✗ fig10: {e}')

    try:
        fig11_velocity_feasibility(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig11: {e}')

    try:
        fig12_gap_violin(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig12: {e}')

    try:
        fig13_summary_table(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig13: {e}')

    try:
        fig14_normalized_convergence(baseline_dir, output_dir, all_experiments=experiments)
    except Exception as e:
        print(f'  ✗ fig14: {e}')

    try:
        fig15_runtime_comparison(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ fig15: {e}')

    # Statistical tests
    try:
        generate_statistical_tests(experiments, output_dir)
    except ImportError:
        print('  ⚠ scipy not available, skipping statistical tests')
    except Exception as e:
        print(f'  ✗ statistical_tests: {e}')

    # LaTeX table
    try:
        generate_latex_table(experiments, output_dir)
    except Exception as e:
        print(f'  ✗ latex_table: {e}')

    print(f'\n{"="*60}')
    print(f'  ✅ All figures written to {output_dir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
