#!/usr/bin/env python3
"""
Main entry point for WDN optimization benchmarking.

Usage:
    python run_benchmarks.py                    # Full benchmark (30 trials, all networks)
    python run_benchmarks.py --trials 5         # Quick test with 5 trials
    python run_benchmarks.py --network TwoLoop  # Single network
    python run_benchmarks.py --strategy GA      # Single strategy
    python run_benchmarks.py --gen 500          # Fewer generations
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from wdn_optimizer.network import two_loop_network, hanoi_network, goyang_network
    from wdn_optimizer.benchmark import run_benchmark
except ModuleNotFoundError as exc:
    if exc.name == 'wntr':
        print("Missing dependency: wntr")
        print("Use the project virtual environment interpreter:")
        print("  /home/mihai/eaa/.venv/bin/python run_benchmarks.py ...")
        print("or activate it first:")
        print("  source .venv/bin/activate")
        sys.exit(2)
    raise


def main():
    parser = argparse.ArgumentParser(
        description='WDN Optimization Benchmark Runner')

    parser.add_argument('--trials', type=int, default=30,
                        help='Number of independent trials (default 30)')
    parser.add_argument('--gen', type=int, default=2000,
                        help='Generations/iterations per trial (default 2000)')
    parser.add_argument('--pop', type=int, default=100,
                        help='Population/swarm size (default 100)')
    parser.add_argument('--network', type=str, default='all',
                        choices=['all', 'TwoLoop', 'Hanoi', 'GoYang'],
                        help='Which network to optimize (default all)')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['all', 'GA', 'SA', 'PSO'],
                        help='Which strategy to use (default all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default results)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-generation info')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum absolute velocity for feasibility in m/s (default: network profile)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum absolute velocity for feasibility in m/s (default: network profile)')
    parser.add_argument('--twoloop-vmin', type=float, default=None,
                        help='Two-Loop minimum absolute velocity override in m/s')
    parser.add_argument('--twoloop-vmax', type=float, default=None,
                        help='Two-Loop maximum absolute velocity override in m/s')
    parser.add_argument('--hanoi-vmin', type=float, default=None,
                        help='Hanoi minimum absolute velocity override in m/s')
    parser.add_argument('--hanoi-vmax', type=float, default=None,
                        help='Hanoi maximum absolute velocity override in m/s')
    parser.add_argument('--goyang-vmin', type=float, default=None,
                        help='GoYang minimum absolute velocity override in m/s')
    parser.add_argument('--goyang-vmax', type=float, default=None,
                        help='GoYang maximum absolute velocity override in m/s')
    parser.add_argument('--cores', type=int, default=None,
                        help='Number of CPU cores to use (default: all but 1)')

    args = parser.parse_args()

    # Select networks
    network_map = {
        'TwoLoop': two_loop_network,
        'Hanoi': hanoi_network,
        'GoYang': goyang_network,
    }
    if args.network == 'all':
        problems = [fn() for fn in network_map.values()]
    else:
        problems = [network_map[args.network]()]

    # Apply optional global and per-network velocity feasibility bounds.
    per_network_overrides = {
        'Two-Loop': (args.twoloop_vmin, args.twoloop_vmax),
        'Hanoi': (args.hanoi_vmin, args.hanoi_vmax),
        'GoYang': (args.goyang_vmin, args.goyang_vmax),
    }

    for p in problems:
        if args.vmin is not None:
            p.min_velocity = args.vmin
        if args.vmax is not None:
            p.max_velocity = args.vmax

        local_vmin, local_vmax = per_network_overrides.get(p.name, (None, None))
        if local_vmin is not None:
            p.min_velocity = local_vmin
        if local_vmax is not None:
            p.max_velocity = local_vmax

    # Basic bounds validation to prevent impossible configurations.
    for p in problems:
        if p.min_velocity > p.max_velocity:
            print(f"Invalid velocity bounds for {p.name}: "
                  f"vmin={p.min_velocity} > vmax={p.max_velocity}")
            sys.exit(2)

    # Select strategies
    if args.strategy == 'all':
        strategies = ['GA', 'SA', 'PSO']
    else:
        strategies = [args.strategy]

    print("═" * 60)
    print("  WDN Pipe Diameter Optimization Benchmark")
    print("═" * 60)
    if args.cores:
        n_cores = args.cores
    else:
        n_cores = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    print(f"  Networks:   {[p.name for p in problems]}")
    print(f"  Strategies: {strategies}")
    print(f"  Trials:     {args.trials}")
    print(f"  Gen/Iter:   {args.gen}")
    print(f"  Pop/Swarm:  {args.pop}")
    print(f"  Output:     {args.output}/")
    bounds_desc = ", ".join([f"{p.name}=[{p.min_velocity}, {p.max_velocity}]" for p in problems])
    print(f"  Feasible v: {bounds_desc} m/s")
    print(f"  Parallel:   {n_cores} CPU cores")
    print("═" * 60)

    recommended_ranges = {
        'Two-Loop': (0.3, 2.0),
        'Hanoi': (0.3, 2.5),
        'GoYang': (0.0, 2.0),
    }
    warnings = []
    for p in problems:
        rec_min, rec_max = recommended_ranges[p.name]
        if p.min_velocity > rec_min:
            warnings.append(
                f"{p.name}: vmin={p.min_velocity} may be too strict; typical <= {rec_min}"
            )
        if p.max_velocity < rec_max:
            warnings.append(
                f"{p.name}: vmax={p.max_velocity} may be too strict; typical >= {rec_max}"
            )

    if warnings:
        print("Warning: selected velocity bounds may cause systematic infeasibility:")
        for w in warnings:
            print(f"  - {w}")
        print("Proceeding anyway. Use per-network overrides to tune each benchmark.")
        print("═" * 60)

    results = run_benchmark(
        problems=problems,
        strategies=strategies,
        num_trials=args.trials,
        max_gen=args.gen,
        pop_size=args.pop,
        output_dir=args.output,
        verbose_trials=args.verbose,
        n_workers=args.cores,
    )

    # Print summary table
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)

    for net_name, strats in results.items():
        print(f"\n  📊 {net_name} Network")
        print(f"  {'Strategy':<8} {'Best':>14} {'Mean':>14} {'Std':>12} "
              f"{'Feasible':>8} {'Time(s)':>8}")
        print(f"  {'-'*66}")

        for strat, trials in strats.items():
            costs = [t['best_cost'] for t in trials]
            feasible = sum(1 for t in trials if t['best_feasible'])
            times = [t['wall_time'] for t in trials]

            import numpy as np
            print(f"  {strat:<8} {min(costs):>14,.0f} {np.mean(costs):>14,.0f} "
                  f"{np.std(costs):>12,.0f} {feasible:>6}/{len(trials):<2} "
                  f"{np.mean(times):>7.1f}")

    print()


if __name__ == '__main__':
    main()
