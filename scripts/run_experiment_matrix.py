#!/usr/bin/env python3
"""
Experiment Matrix Runner for WDN Optimization Benchmarks.

Defines report-ready experimental configurations, runs each one sequentially,
then aggregates results and generates report figures.

Usage:
    # Run the report-rich preset (experiments + figures):
    python scripts/run_experiment_matrix.py

    # Only run experiments (skip figure generation):
    python scripts/run_experiment_matrix.py --no-figures

    # Only generate figures from existing results:
    python scripts/run_experiment_matrix.py --figures-only

    # Run a specific config by name:
    python scripts/run_experiment_matrix.py --only REPORT_FAST_3T_100G_100P_2C

    # Use the old full matrix explicitly:
    python scripts/run_experiment_matrix.py --preset legacy --dry-run

    # Dry run (show what would be executed):
    python scripts/run_experiment_matrix.py --dry-run

    # List all configs:
    python scripts/run_experiment_matrix.py --list
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── PROJECT PATHS ───────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_SCRIPT = PROJECT_DIR / 'run_benchmarks.py'
PYTHON = PROJECT_DIR / '.venv' / 'bin' / 'python'
RESULTS_DIR = PROJECT_DIR / 'results'       # All experiment outputs go here
FIGURES_DIR = PROJECT_DIR / 'report_figures' # Generated figures go here
FIGURES_SCRIPT = PROJECT_DIR / 'scripts' / 'generate_report_figures.py'
REPORT_MANIFEST = PROJECT_DIR / 'report_manifest.json'

# ─── EXPERIMENT MATRICES ─────────────────────────────────────────────────────
# Each entry supports:
#   name, group, networks, strategies, trials, pop, gen, cores, priority, enabled
#   benchmark_args: existing run_benchmarks.py args appended to the command
#   budget: evaluation budget proxy, usually gen * pop
#   include_in_manifest: whether this completed run belongs in the final report
#   constraint_variant: optional reporting label, e.g. "core_scaling"

def _cfg(**kwargs):
    """Return a normalized experiment config."""
    config = {
        'benchmark_args': [],
        'budget': kwargs.get('gen', 0) * kwargs.get('pop', 0),
        'include_in_manifest': True,
        'constraint_variant': 'default',
        'enabled': True,
    }
    config.update(kwargs)
    return config


REPORT_CORE_MATRIX = [
    _cfg(
        name='REPORT_SMOKE_1T_20G_40P_2C',
        group='Smoke',
        networks='all',
        strategies='all',
        trials=1,
        pop=40,
        gen=20,
        cores=2,
        priority=0,
        est_minutes=10,
        include_in_manifest=False,
    ),
    _cfg(
        name='REPORT_FAST_3T_100G_100P_2C',
        group='Report Core',
        networks='all',
        strategies='all',
        trials=3,
        pop=100,
        gen=100,
        cores=2,
        priority=10,
        est_minutes=40,
    ),
    _cfg(
        name='REPORT_MED_3T_200G_100P_2C',
        group='Report Core',
        networks='all',
        strategies='all',
        trials=3,
        pop=100,
        gen=200,
        cores=2,
        priority=20,
        est_minutes=80,
    ),
]


REPORT_RICH_MATRIX = [
    _cfg(
        name='REPORT_SMOKE_1T_20G_40P_2C',
        group='Smoke',
        networks='all',
        strategies='all',
        trials=1,
        pop=40,
        gen=20,
        cores=2,
        priority=0,
        est_minutes=5,
        include_in_manifest=False,
    ),
    _cfg(
        name='REPORT_FAST_3T_100G_100P_2C',
        group='Generation Scaling',
        networks='all',
        strategies='all',
        trials=3,
        pop=100,
        gen=100,
        cores=2,
        priority=10,
        est_minutes=30,
    ),
    _cfg(
        name='REPORT_MED_3T_200G_100P_2C',
        group='Generation/Population Scaling',
        networks='all',
        strategies='all',
        trials=3,
        pop=100,
        gen=200,
        cores=2,
        priority=20,
        est_minutes=55,
    ),
    _cfg(
        name='REPORT_DEEP_3T_400G_100P_2C',
        group='Generation Scaling',
        networks='all',
        strategies='all',
        trials=3,
        pop=100,
        gen=400,
        cores=2,
        priority=30,
        est_minutes=105,
    ),
    _cfg(
        name='REPORT_POP150_3T_200G_150P_2C',
        group='Population Scaling',
        networks='all',
        strategies='all',
        trials=3,
        pop=150,
        gen=200,
        cores=2,
        priority=40,
        est_minutes=75,
    ),
    _cfg(
        name='REPORT_WIDE_3T_200G_200P_2C',
        group='Population Scaling',
        networks='all',
        strategies='all',
        trials=3,
        pop=200,
        gen=200,
        cores=2,
        priority=50,
        est_minutes=90,
    ),
]


REPORT_OPTIONAL_MATRIX = [
    _cfg(
        name='REPORT_SCALE_TL_GA_3T_100G_100P_1C',
        group='Core Scaling',
        networks='TwoLoop',
        strategies='GA',
        trials=3,
        pop=100,
        gen=100,
        cores=1,
        priority=100,
        est_minutes=20,
        constraint_variant='core_scaling',
    ),
    _cfg(
        name='REPORT_SCALE_TL_GA_3T_100G_100P_2C',
        group='Core Scaling',
        networks='TwoLoop',
        strategies='GA',
        trials=3,
        pop=100,
        gen=100,
        cores=2,
        priority=101,
        est_minutes=15,
        constraint_variant='core_scaling',
    ),
    _cfg(
        name='REPORT_SCALE_TL_GA_3T_100G_100P_4C',
        group='Core Scaling',
        networks='TwoLoop',
        strategies='GA',
        trials=3,
        pop=100,
        gen=100,
        cores=4,
        priority=102,
        est_minutes=12,
        constraint_variant='core_scaling',
    ),
    _cfg(
        name='REPORT_SCALE_TL_GA_3T_100G_100P_8C',
        group='Core Scaling',
        networks='TwoLoop',
        strategies='GA',
        trials=3,
        pop=100,
        gen=100,
        cores=8,
        priority=103,
        est_minutes=10,
        constraint_variant='core_scaling',
    ),
    _cfg(
        name='REPORT_OVERNIGHT_5T_400G_150P_2C',
        group='Overnight',
        networks='all',
        strategies='all',
        trials=5,
        pop=150,
        gen=400,
        cores=2,
        priority=200,
        est_minutes=240,
        constraint_variant='overnight',
    ),
]


LEGACY_EXPERIMENT_MATRIX = [
    {
        'name': 'EXP0_QUICK',
        'group': 'Validation',
        'networks': 'all',
        'strategies': 'all',
        'trials': 5,
        'pop': 50,
        'gen': 100,
        'cores': 2,
        'priority': 0,
        'enabled': True,
    },

    # ── Experiment 1: Population Size Sensitivity (gen=800 fixed) ─────────
    {
        'name': 'EXP1_POP50',
        'group': 'Population Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 50,
        'gen': 800,
        'cores': 2,
        'priority': 10,
        'enabled': True,
    },
    {
        'name': 'EXP1_POP100',
        'group': 'Population Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 100,
        'gen': 800,
        'cores': 2,
        'priority': 11,
        'enabled': True,
    },
    # NOTE: pop=200,gen=800 is the BASELINE (FINAL_10T_800G_200P_2C)
    {
        'name': 'EXP1_POP400',
        'group': 'Population Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 400,
        'gen': 800,
        'cores': 2,
        'priority': 12,
        'enabled': True,
    },

    # ── Experiment 2: Generation Count Sensitivity (pop=200 fixed) ────────
    {
        'name': 'EXP2_GEN200',
        'group': 'Generation Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 200,
        'gen': 200,
        'cores': 2,
        'priority': 20,
        'enabled': True,
    },
    {
        'name': 'EXP2_GEN400',
        'group': 'Generation Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 200,
        'gen': 400,
        'cores': 2,
        'priority': 21,
        'enabled': True,
    },
    # NOTE: gen=800 is the BASELINE
    {
        'name': 'EXP2_GEN1600',
        'group': 'Generation Sensitivity',
        'networks': 'all',
        'strategies': 'all',
        'trials': 10,
        'pop': 200,
        'gen': 1600,
        'cores': 2,
        'priority': 22,
        'enabled': True,
    },

    # ── Experiment 3: High-Trial Statistical Robustness ───────────────────
    {
        'name': 'EXP3_TWOLOOP_30T',
        'group': '30-Trial Robustness',
        'networks': 'TwoLoop',
        'strategies': 'all',
        'trials': 30,
        'pop': 200,
        'gen': 800,
        'cores': 2,
        'priority': 30,
        'enabled': True,
    },
    {
        'name': 'EXP3_HANOI_30T',
        'group': '30-Trial Robustness',
        'networks': 'Hanoi',
        'strategies': 'all',
        'trials': 30,
        'pop': 200,
        'gen': 800,
        'cores': 2,
        'priority': 31,
        'enabled': True,
    },
    {
        'name': 'EXP3_GOYANG_30T',
        'group': '30-Trial Robustness',
        'networks': 'GoYang',
        'strategies': 'all',
        'trials': 30,
        'pop': 200,
        'gen': 800,
        'cores': 2,
        'priority': 32,
        'enabled': True,
    },
]


PRESET_MATRICES = {
    'report-core': REPORT_CORE_MATRIX,
    'report-rich': REPORT_RICH_MATRIX,
    'report-optional': REPORT_OPTIONAL_MATRIX,
    'legacy': LEGACY_EXPERIMENT_MATRIX,
}


# ─── HELPER: Estimate runtime ────────────────────────────────────────────────

def estimate_runtime_minutes(config):
    """Rough estimate of runtime in minutes based on config parameters."""
    if 'est_minutes' in config:
        return float(config['est_minutes'])

    # Base cost per (trial × network × strategy) for gen=800, pop=200 ≈ 2 min
    base_per_combo = 2.0
    scale = (config['pop'] / 200.0) * (config['gen'] / 800.0)

    n_networks = 3 if config['networks'] == 'all' else 1
    n_strategies = 3 if config['strategies'] == 'all' else 1
    n_trials = config['trials']

    return base_per_combo * scale * n_networks * n_strategies * n_trials


def expected_summary_rows(config):
    """Return expected summary rows for a completed benchmark config."""
    return len(expected_summary_keys(config))


def expected_summary_keys(config):
    """Return expected (Network, Strategy) pairs for a completed config."""
    network_names = {
        'all': ['Two-Loop', 'Hanoi', 'GoYang'],
        'TwoLoop': ['Two-Loop'],
        'Hanoi': ['Hanoi'],
        'GoYang': ['GoYang'],
    }
    strategy_names = {
        'all': ['GA', 'SA', 'PSO'],
        'GA': ['GA'],
        'SA': ['SA'],
        'PSO': ['PSO'],
    }
    networks = network_names.get(config['networks'], [config['networks']])
    strategies = strategy_names.get(config['strategies'], [config['strategies']])
    return {(network, strategy) for network in networks for strategy in strategies}


def summary_status(config, results_base_dir):
    """Return (status, found_rows, expected_rows) for a config output."""
    summary_path = Path(results_base_dir) / config['name'] / 'summary.csv'
    expected_keys = expected_summary_keys(config)
    expected = len(expected_keys)
    if not summary_path.exists():
        return 'missing', 0, expected

    try:
        with summary_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 'invalid', 0, expected

    found_keys = {(r.get('Network'), r.get('Strategy')) for r in rows}
    found = len(found_keys & expected_keys)
    if found_keys >= expected_keys:
        return 'complete', found, expected
    return 'incomplete', found, expected


# ─── RUNNER ──────────────────────────────────────────────────────────────────

def run_experiment(config, results_base_dir):
    """Run a single experiment configuration."""
    output_dir = os.path.join(results_base_dir, config['name'])

    # Skip only if the summary has the expected network × strategy coverage.
    status, found, expected = summary_status(config, results_base_dir)
    if status == 'complete':
        print(f"  ⏭  Already completed: {config['name']} ({found}/{expected} summary rows)")
        return True
    if status in {'incomplete', 'invalid'}:
        print(f"  ↻ Re-running incomplete config: {config['name']} ({found}/{expected} summary rows)")

    cmd = [
        str(PYTHON), str(BENCHMARKS_SCRIPT),
        '--network', config['networks'],
        '--strategy', config['strategies'],
        '--trials', str(config['trials']),
        '--gen', str(config['gen']),
        '--pop', str(config['pop']),
        '--cores', str(config['cores']),
        '--output', output_dir,
    ]
    cmd.extend(config.get('benchmark_args', []))

    env = os.environ.copy()
    env.setdefault('WDN_SIMULATOR', 'epanet')
    env.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

    est = estimate_runtime_minutes(config)
    print(f"\n{'─'*60}")
    print(f"  ▶ Running: {config['name']}")
    print(f"    Group:    {config['group']}")
    print(f"    Config:   net={config['networks']}, strat={config['strategies']}, "
          f"trials={config['trials']}, pop={config['pop']}, gen={config['gen']}")
    if config.get('benchmark_args'):
        print(f"    Args:     {' '.join(config['benchmark_args'])}")
    print(f"    Output:   {output_dir}")
    print(f"    Est time: ~{est:.0f} min ({est/60:.1f} h)")
    print(f"    Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("    Note:     startup can be quiet for a few seconds while WNTR/pandas import")
    print(f"{'─'*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_DIR), env=env,
            stdout=sys.stdout, stderr=sys.stderr,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            status, found, expected = summary_status(config, results_base_dir)
            if status != 'complete':
                print(f"\n  ❌ Incomplete output: {config['name']} ({found}/{expected} summary rows)")
                return False
            print(f"\n  ✅ Completed: {config['name']} in {elapsed/60:.1f} min")
            return True
        else:
            print(f"\n  ❌ Failed: {config['name']} (exit code {result.returncode})")
            return False
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  ⚠  Interrupted: {config['name']} after {elapsed/60:.1f} min")
        raise
    except Exception as e:
        print(f"\n  ❌ Error: {config['name']}: {e}")
        return False


def aggregate_results(results_base_dir):
    """
    Build an aggregated summary CSV from all experiment outputs.
    Also copies baseline data into the aggregation.
    """
    agg_path = os.path.join(results_base_dir, 'aggregated_summary.csv')
    all_rows = []

    # Include baseline
    baseline_dir = os.path.join(str(PROJECT_DIR), 'FINAL_10T_800G_200P_2C')
    dirs_to_scan = [('BASELINE', baseline_dir)]

    # Add all experiment dirs
    for entry in sorted(os.listdir(results_base_dir)):
        full = os.path.join(results_base_dir, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, 'summary.csv')):
            dirs_to_scan.append((entry, full))

    for label, d in dirs_to_scan:
        summary_path = os.path.join(d, 'summary.csv')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['Config'] = label
                row['Config_Dir'] = d
                all_rows.append(row)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(agg_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  📊 Aggregated summary: {agg_path} ({len(all_rows)} rows)")

    # Also build aggregated detailed results
    agg_detailed_path = os.path.join(results_base_dir, 'aggregated_detailed.csv')
    all_detailed = []
    for label, d in dirs_to_scan:
        detailed_path = os.path.join(d, 'detailed_results.csv')
        if not os.path.exists(detailed_path):
            continue
        with open(detailed_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['Config'] = label
                all_detailed.append(row)

    if all_detailed:
        fieldnames = list(all_detailed[0].keys())
        with open(agg_detailed_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed)
        print(f"  📊 Aggregated detailed: {agg_detailed_path} ({len(all_detailed)} rows)")

    return agg_path


def _relative_to_project(path):
    """Return a stable manifest path relative to the project directory."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(Path(path).resolve())


def _manifest_config_entry(config, results_base_dir):
    output_dir = Path(results_base_dir) / config['name']
    summary_path = output_dir / 'summary.csv'
    if not summary_path.exists() or not config.get('include_in_manifest', False):
        return None
    status, _, _ = summary_status(config, results_base_dir)
    if status != 'complete':
        return None

    return {
        'label': config['name'],
        'path': _relative_to_project(output_dir),
        'summary_csv': _relative_to_project(summary_path),
        'detailed_csv': _relative_to_project(output_dir / 'detailed_results.csv'),
        'convergence_json': _relative_to_project(output_dir / 'convergence_data.json'),
        'group': config['group'],
        'networks': config['networks'],
        'strategies': config['strategies'],
        'trials': config['trials'],
        'gen': config['gen'],
        'pop': config['pop'],
        'cores': config['cores'],
        'budget': config.get('budget', config['gen'] * config['pop']),
        'benchmark_args': config.get('benchmark_args', []),
        'constraint_variant': config.get('constraint_variant', 'default'),
    }


def write_report_manifest(results_base_dir, manifest_configs=None):
    """Write a clean manifest of report-approved completed runs."""
    baseline_dir = PROJECT_DIR / 'FINAL_10T_800G_200P_2C'
    baseline_summary = baseline_dir / 'summary.csv'

    if manifest_configs is None:
        manifest_configs = REPORT_RICH_MATRIX
    experiments = []
    seen = set()
    for config in sorted(manifest_configs, key=lambda c: c['priority']):
        entry = _manifest_config_entry(config, results_base_dir)
        if entry and entry['label'] not in seen:
            experiments.append(entry)
            seen.add(entry['label'])

    manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'protocol': {
            'simulator': 'epanet',
            'penalty_protocol': 'current_project_protocol',
            'vp2': 0.65,
            'note': (
                'Velocity bounds are passed through existing run_benchmarks.py '
                'arguments. Pressure thresholds remain fixed in network.py.'
            ),
        },
        'baseline': {
            'label': 'BASELINE_10T_800G_200P_2C',
            'path': _relative_to_project(baseline_dir),
            'summary_csv': _relative_to_project(baseline_summary),
            'detailed_csv': _relative_to_project(baseline_dir / 'detailed_results.csv'),
            'convergence_json': _relative_to_project(baseline_dir / 'convergence_data.json'),
            'trials': 10,
            'gen': 800,
            'pop': 200,
            'budget': 160000,
            'constraint_variant': 'default',
            'exists': baseline_summary.exists(),
        },
        'experiments': experiments,
    }

    with open(REPORT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')

    print(f"  Report manifest: {REPORT_MANIFEST} ({len(experiments)} completed report runs)")
    return str(REPORT_MANIFEST)


def generate_figures(results_base_dir):
    """Run the figure generation script."""
    print(f"\n{'═'*60}")
    print(f"  Generating report figures...")
    print(f"{'═'*60}")

    cmd = [
        str(PYTHON), str(FIGURES_SCRIPT),
        '--base-dir', str(PROJECT_DIR),
        '--output', str(FIGURES_DIR),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode == 0


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def build_configs(preset, include_optional=False):
    """Return normalized configs for a preset."""
    if preset not in PRESET_MATRICES:
        raise ValueError(f"Unknown preset: {preset}")

    configs = list(PRESET_MATRICES[preset])
    if include_optional and preset in {'report-core', 'report-rich'}:
        configs.extend(REPORT_OPTIONAL_MATRIX)

    return sorted([c for c in configs if c.get('enabled', True)],
                  key=lambda x: x['priority'])


def print_matrix(configs, results_dir, preset):
    """Print the experiment matrix as a table."""
    print(f"\n{'═'*90}")
    print(f"  WDN Optimization — Experiment Matrix ({preset})")
    print(f"{'═'*90}")
    print(f"  {'#':<3} {'Name':<36} {'Group':<22} {'Net':<8} {'Strat':<6} "
          f"{'Trials':<7} {'Pop':<5} {'Gen':<6} {'Est':<6} {'Status'}")
    print(f"  {'─'*3} {'─'*36} {'─'*22} {'─'*8} {'─'*6} "
          f"{'─'*7} {'─'*5} {'─'*6} {'─'*6} {'─'*10}")

    total_min = 0
    for i, c in enumerate(configs):
        est = estimate_runtime_minutes(c)
        total_min += est

        status, found, expected = summary_status(c, results_dir)
        if status == 'complete':
            status_char = f'Done {found}/{expected}'
        elif status == 'missing':
            status_char = f'Pending 0/{expected}'
        else:
            status_char = f'Incomplete {found}/{expected}'

        print(f"  {i+1:<3} {c['name']:<36} {c['group']:<22} {c['networks']:<8} "
              f"{c['strategies']:<6} {c['trials']:<7} {c['pop']:<5} {c['gen']:<6} "
              f"{est:<6.0f} {status_char}")
        if c.get('benchmark_args'):
            print(f"      args: {' '.join(c['benchmark_args'])}")

    print(f"\n  Total estimated runtime (enabled only): "
          f"{total_min:.0f} min ({total_min/60:.1f} hours)")
    print(f"  Results directory: {results_dir}")
    print(f"  Figures directory: {FIGURES_DIR}")
    print(f"  Manifest: {REPORT_MANIFEST}")
    print(f"{'═'*90}\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run WDN Optimization Experiment Matrix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment_matrix.py --list
  python scripts/run_experiment_matrix.py --dry-run
  python scripts/run_experiment_matrix.py --preset report-rich --no-figures
  python scripts/run_experiment_matrix.py --only REPORT_SMOKE_1T_20G_40P_2C --no-figures
  python scripts/run_experiment_matrix.py --preset legacy --dry-run
  python scripts/run_experiment_matrix.py --figures-only
        """,
    )
    parser.add_argument('--preset', type=str, default='report-rich',
                       choices=sorted(PRESET_MATRICES.keys()),
                       help='Experiment preset to use (default: report-rich)')
    parser.add_argument('--include-optional', action='store_true',
                       help='Append report-optional configs to the selected report preset')
    parser.add_argument('--list', action='store_true',
                       help='List all configurations and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    parser.add_argument('--only', type=str, nargs='+',
                       help='Run only specific configs by name')
    parser.add_argument('--figures-only', action='store_true',
                       help='Skip experiments, only generate figures')
    parser.add_argument('--no-figures', action='store_true',
                       help='Skip figure generation after experiments')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                       help=f'Base directory for results (default: {RESULTS_DIR})')
    args = parser.parse_args()

    results_dir = args.results_dir
    configs = build_configs(args.preset, args.include_optional)

    # List mode
    if args.list:
        print_matrix(configs, results_dir, args.preset)
        return

    # Figures-only mode
    if args.figures_only:
        aggregate_results(results_dir)
        write_report_manifest(results_dir, configs)
        generate_figures(results_dir)
        return

    # Select configs to run
    if args.only:
        configs = [c for c in configs if c['name'] in args.only]
        if not configs:
            print(f"ERROR: No configs found matching: {args.only}")
            available = [c['name'] for c in build_configs(args.preset, args.include_optional)]
            print(f"Available: {available}")
            sys.exit(1)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Print plan
    print_matrix(configs, results_dir, args.preset)

    if args.dry_run:
        print("  DRY RUN - no experiments will be executed.\n")
        for c in configs:
            output_dir = os.path.join(results_dir, c['name'])
            status, found, expected = summary_status(c, results_dir)
            action = '[SKIP]' if status == 'complete' else '[RUN] '
            reason = 'complete' if status == 'complete' else f'{status} {found}/{expected}'
            print(f"  {action} {c['name']}: "
                  f"net={c['networks']}, pop={c['pop']}, gen={c['gen']}, "
                  f"trials={c['trials']}, cores={c['cores']} ({reason})")
            cmd_preview = [
                str(PYTHON), str(BENCHMARKS_SCRIPT),
                '--network', c['networks'],
                '--strategy', c['strategies'],
                '--trials', str(c['trials']),
                '--gen', str(c['gen']),
                '--pop', str(c['pop']),
                '--cores', str(c['cores']),
                '--output', output_dir,
                *c.get('benchmark_args', []),
            ]
            print(f"        {' '.join(cmd_preview)}")
        return

    # Run experiments
    total = len(configs)
    completed = 0
    failed = 0
    interrupted = False
    t_start = time.time()

    print(f"\n  🚀 Starting {total} experiments...")
    print(f"     Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i, config in enumerate(configs, 1):
        print(f"\n  [{i}/{total}] ", end='')
        try:
            success = run_experiment(config, results_dir)
            if success:
                completed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print(f"\n\n  ⚠  Interrupted! Completed {completed}/{total} experiments.")
            print("  Stopping without aggregation/manifest/figure generation.\n")
            interrupted = True
            break

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'═'*60}")
    print(f"  Experiment Matrix — Summary")
    print(f"{'═'*60}")
    print(f"  Completed: {completed}/{total}")
    print(f"  Failed:    {failed}")
    if interrupted:
        print(f"  Interrupted: yes")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.1f} h)")
    print(f"{'═'*60}")

    if interrupted:
        print("\n  Run interrupted. Re-run the same command when ready.")
        print("  Existing completed configs will still be skipped via summary.csv.\n")
        sys.exit(130)

    # Aggregate results
    aggregate_results(results_dir)
    write_report_manifest(results_dir, configs)

    # Generate figures
    if not args.no_figures:
        generate_figures(results_dir)

    print(f"\n  🎉 Done! Results in: {results_dir}")
    print(f"         Figures in:  {FIGURES_DIR}\n")


if __name__ == '__main__':
    main()
