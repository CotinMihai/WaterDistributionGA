"""
Benchmark runner: executes multiple trials across strategies and networks.

Collects statistics, generates CSV output and convergence plots.
"""

import os
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from .network import WDNProblem, get_all_problems, validate_problem
from .ga_solver import GASolver
from .pso_solver import PSOSolver
from .sa_solver import SASolver


def run_single_trial(strategy: str, problem: WDNProblem,
                     seed: int, max_gen: int = 2000,
                     pop_size: int = 100, verbose: bool = False,
                     progress_callback=None, n_workers: int | None = None) -> dict:
    """
    Run a single optimization trial.

    Parameters
    ----------
    strategy : 'GA', 'SA' or 'PSO'
    problem : WDNProblem
    seed : int
    max_gen : int
    pop_size : int
    verbose : bool
    n_workers : int, optional
        Number of CPU cores for parallel evaluation.

    Returns
    -------
    result : dict from solver.solve()
    """
    if strategy == 'GA':
        solver = GASolver(problem, pop_size=pop_size, max_gen=max_gen, seed=seed, n_workers=n_workers)
    elif strategy == 'SA':
        # SA evaluates one candidate per iteration; scale iterations so the
        # total hydraulic evaluation budget is comparable to GA/PSO. (so maybe max_gen * pop_size?)
        sa_iters = max(1, int(max_gen) * int(pop_size) )
        solver = SASolver(problem, max_iter=sa_iters, seed=seed, n_workers=n_workers)
    elif strategy == 'PSO':
        solver = PSOSolver(problem, swarm_size=pop_size, max_iter=max_gen, seed=seed, n_workers=n_workers)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return solver.solve(verbose=verbose, progress_callback=progress_callback)


def _make_progress_callback(label: str, width: int = 30):
    """Create a lightweight in-terminal progress bar callback."""
    last_percent = -1

    def _callback(step: int, total: int):
        nonlocal last_percent
        if total <= 0:
            return
        percent = int((step * 100) / total)
        if percent == last_percent and step < total:
            return
        last_percent = percent

        filled = int((step * width) / total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r    {label} [{bar}] {percent:3d}% ({step}/{total})",
              end="", flush=True)
        if step >= total:
            print()

    return _callback


def _trial_rank_key(trial: dict):
    """
    Rank trials by assignment-friendly ordering:
    1) pressure-feasible first, 2) lower raw cost, 3) lower penalized cost.
    """
    return (
        0 if trial.get('best_feasible', False) else 1,
        float(trial.get('best_cost', float('inf'))),
        float(trial.get('best_penalized_cost', float('inf'))),
    )


def _calculate_paper_gap(best_cost: float, known_best: float) -> tuple[float, float]:
    """Return deviation from the paper baseline in USD and percent."""
    if known_best <= 0:
        return 0.0, 0.0
    gap_usd = best_cost - known_best
    gap_percent = (gap_usd / known_best) * 100.0
    return gap_usd, gap_percent


def _known_best_map() -> dict[str, float]:
    return {
        'Two-Loop': 419_000,
        'Hanoi': 6_081_000,
        'GoYang': 177_010_359,
    }


def run_benchmark(problems=None, strategies=None,
                  num_trials=30, max_gen=2000, pop_size=100,
                  output_dir='results', verbose_trials=False, n_workers: int | None = None):
    """
    Run the full benchmark suite.

    Parameters
    ----------
    problems : list of WDNProblem, optional
        Defaults to all three benchmarks.
    strategies : list of str, optional
        Defaults to ['GA', 'SA', 'PSO'].
    num_trials : int
        Number of independent trials per (strategy, network) pair.
    max_gen : int
        Generations/iterations per trial.
    pop_size : int
        Population/swarm size.
    output_dir : str
        Directory for output files.
    verbose_trials : bool
        Print per-generation info during trials.
    n_workers : int, optional
        Number of CPU cores to use for parallel evaluation.
        If None, uses all available cores minus 1.

    Returns
    -------
    all_results : dict
        Nested: {network_name: {strategy: [trial_results]}}
    """
    if problems is None:
        problems = get_all_problems()
    if strategies is None:
        strategies = ['GA', 'SA', 'PSO']

    for prob in problems:
        ok, errors = validate_problem(prob)
        if not ok:
            details = "; ".join(errors)
            raise ValueError(f"Invalid benchmark definition for {prob.name}: {details}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for prob in problems:
        all_results[prob.name] = {}

        for strat in strategies:
            print(f"\n{'='*60}")
            print(f"  {prob.name} — {strat}  ({num_trials} trials, "
                  f"{max_gen} gen, pop={pop_size})")
            print(f"{'='*60}")

            trial_results = []
            all_results[prob.name][strat] = trial_results
            network_seed_offset = {
                'Two-Loop': 101,
                'Hanoi': 202,
                'GoYang': 303,
            }.get(prob.name, 999)

            strategy_seed_offset = {
                'GA': 11,
                'SA': 22,
                'PSO': 33,
            }.get(strat, 0)

            for trial in range(num_trials):
                seed = trial * 1000 + network_seed_offset + strategy_seed_offset
                print(f"  Trial {trial+1}/{num_trials} (seed={seed})...",
                      end=' ', flush=True)
                t0 = time.time()
                progress_label = f"{strat} trial {trial + 1}/{num_trials}"
                progress_cb = None
                if not verbose_trials:
                    progress_cb = _make_progress_callback(progress_label)

                result = run_single_trial(strat, prob, seed=seed,
                                          max_gen=max_gen, pop_size=pop_size,
                                          verbose=verbose_trials,
                                          progress_callback=progress_cb,
                                          n_workers=n_workers)
                elapsed = time.time() - t0
                try:
                    # Convert chosen indices back to raw physical diameter attributes 
                    diams = [prob.available_diameters[i] for i in result['best_solution']]
                    # Auto-convert m to mm for standard literature readability if stored in m
                    if all(d < 2.0 for d in diams): 
                        diams = [round(d * 1000, 1) for d in diams]
                    
                    diam_str = "[" + ", ".join([str(d) for d in diams]) + "]"
                    total_len = sum(prob.pipe_lengths)
                except Exception:
                    diam_str = str(result['best_solution'])
                    total_len = "Unknown"

                feas = "✓" if result['best_feasible'] else "✗"
                min_p = result.get('min_pressure', float('nan'))
                n_p_viol = result.get('pressure_violations', -1)
                p_ok = "✓" if result.get('best_pressure_feasible', False) else "✗"
                v_ok = "✓" if result.get('best_velocity_feasible', False) else "✗"
                vmin = result.get('min_abs_velocity', float('nan'))
                vmax = result.get('max_abs_velocity', float('nan'))
                n_v_low = result.get('velocity_low_violations', -1)
                n_v_high = result.get('velocity_high_violations', -1)
                low_pipes = result.get('velocity_low_pipes', [])
                high_pipes = result.get('velocity_high_pipes', [])
                known_best = _known_best_map().get(prob.name, 0)
                gap_usd, gap_percent = _calculate_paper_gap(result['best_cost'], known_best)
                gap_sign = '+' if gap_percent > 0 else ''

                print(f"  → Final Network Cost:      {result['best_cost']:>14,.0f} USD")
                if known_best > 0:
                    if gap_percent < 0:
                        print(f"  ⭐ [BEAT SOTA] {gap_sign}{gap_percent:.2f}% ({gap_usd:+,.0f} USD) - SOTA was: {known_best:,.0f}")
                    else:
                        print(f"  → SOTA Gap:                {gap_sign}{gap_percent:.2f}% ({gap_usd:+,.0f} USD)  [SOTA: {known_best:,.0f}]")
                print(f"  → Network Physical Feasibility:   {feas} (PP: {result['PP']:.2f}, VP: {result['VP']:.2f})")
                print(f"  → Pressure Check: {p_ok} min={min_p:.2f} m, violations={n_p_viol}")
                print(f"  → Velocity Check: {v_ok} min|v|={vmin:.2f} m/s, max|v|={vmax:.2f} m/s, low/high violations={n_v_low}/{n_v_high}")
                if low_pipes or high_pipes:
                    low_preview = ", ".join(low_pipes[:6]) if low_pipes else "none"
                    high_preview = ", ".join(high_pipes[:6]) if high_pipes else "none"
                    print(f"  → Velocity violating pipes (low/high): {low_preview} / {high_preview}")
                print(f"  → Optimal Diameters Found: {diam_str} mm")
                print(f"  → Total Topographic Length:{total_len} meters")
                print(f"  → Mathematical Solve Time: {elapsed:.1f}s\n")
                
                trial_results.append(result)

                # Incremental save to prevent data loss
                try:
                    _write_summary_csv(all_results, output_dir, quiet=True)
                    _write_detailed_csv(all_results, output_dir, quiet=True)
                    _write_convergence_data(all_results, output_dir, quiet=True)
                except Exception as e:
                    print(f"  [Warning] Incremental save failed: {e}")

            try:
                _plot_convergence(all_results, output_dir, quiet=True)
                _plot_boxplots(all_results, output_dir, quiet=True)
            except Exception as e:
                print(f"  [Warning] Plotting failed: {e}")

    # Final explicit save
    _write_summary_csv(all_results, output_dir, quiet=False)
    _write_detailed_csv(all_results, output_dir, quiet=False)
    _write_convergence_data(all_results, output_dir, quiet=False)
    try:
        _plot_convergence(all_results, output_dir, quiet=False)
        _plot_boxplots(all_results, output_dir, quiet=False)
    except Exception:
        pass

    print(f"\n✅ Results written to {output_dir}/")
    return all_results


def _write_summary_csv(all_results, output_dir, quiet=False):
    """Write summary statistics to CSV."""
    path = os.path.join(output_dir, 'summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Network', 'Strategy', 'Known_Best', 'Trials',
            'Best_Cost', 'Mean_Cost', 'Std_Cost', 'Median_Cost',
            'Feasible_Count', 'Mean_Time_s',
            'Best_PP', 'Best_VP', 'Gap_USD', 'Gap_Percent', 'Beat_SOTA'
        ])

        for net_name, strats in all_results.items():
            for strat, trials in strats.items():
                costs = [t['best_cost'] for t in trials]
                feasible = [t['best_feasible'] for t in trials]
                times = [t['wall_time'] for t in trials]

                known_bests = _known_best_map()

                # Pick representative winner with feasible-first ordering.
                best_trial = min(trials, key=_trial_rank_key)
                best_reported_cost = best_trial['best_cost']
                known_best = known_bests.get(net_name, 0)

                gap_usd, gap_percent = _calculate_paper_gap(best_reported_cost, known_best)
                beat_sota = 'YES' if gap_percent < 0 else 'NO'

                writer.writerow([
                    net_name, strat,
                    known_best if known_best > 0 else 'N/A',
                    len(trials),
                    f'{best_reported_cost:.0f}',
                    f'{np.mean(costs):.0f}',
                    f'{np.std(costs):.0f}',
                    f'{np.median(costs):.0f}',
                    sum(feasible),
                    f'{np.mean(times):.1f}',
                    f'{best_trial["PP"]:.4f}',
                    f'{best_trial["VP"]:.4f}',
                    f'{gap_usd:.0f}',
                    f'{gap_percent:.2f}%',
                    beat_sota,
                ])

    if not quiet:
        print(f"  → Summary: {path}")


def _write_detailed_csv(all_results, output_dir, quiet=False):
    """Write per-trial results."""
    path = os.path.join(output_dir, 'detailed_results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Network', 'Strategy', 'Trial', 'Best_Cost',
            'Penalized_Cost', 'Feasible', 'Pressure_Feasible', 'Velocity_Feasible',
            'Min_Pressure', 'Pressure_Violations',
            'Min_Abs_Velocity', 'Max_Abs_Velocity',
            'Velocity_Low_Violations', 'Velocity_High_Violations',
            'Velocity_Low_Pipes', 'Velocity_High_Pipes',
            'PP', 'VP', 'Wall_Time_s'
        ])

        for net_name, strats in all_results.items():
            for strat, trials in strats.items():
                for i, t in enumerate(trials):
                    writer.writerow([
                        net_name, strat, i + 1,
                        f'{t["best_cost"]:.0f}',
                        f'{t["best_penalized_cost"]:.0f}',
                        t['best_feasible'],
                        t.get('best_pressure_feasible', ''),
                        t.get('best_velocity_feasible', ''),
                        f'{t.get("min_pressure", float("nan")):.4f}',
                        t.get('pressure_violations', ''),
                        f'{t.get("min_abs_velocity", float("nan")):.4f}',
                        f'{t.get("max_abs_velocity", float("nan")):.4f}',
                        t.get('velocity_low_violations', ''),
                        t.get('velocity_high_violations', ''),
                        "|".join(t.get('velocity_low_pipes', [])),
                        "|".join(t.get('velocity_high_pipes', [])),
                        f'{t["PP"]:.4f}',
                        f'{t["VP"]:.4f}',
                        f'{t["wall_time"]:.2f}',
                    ])

    if not quiet:
        print(f"  → Detailed: {path}")


def _write_convergence_data(all_results, output_dir, quiet=False):
    """Write per-trial convergence curves to JSON for post-processing."""
    path = os.path.join(output_dir, 'convergence_data.json')
    payload = {
        'networks': {},
    }

    for net_name, strats in all_results.items():
        payload['networks'][net_name] = {}
        for strat, trials in strats.items():
            payload['networks'][net_name][strat] = [
                {
                    'trial': idx + 1,
                    'convergence': t.get('convergence', []),
                }
                for idx, t in enumerate(trials)
            ]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)

    if not quiet:
        print(f"  → Convergence data: {path}")


def _plot_convergence(all_results, output_dir, quiet=False):
    """Plot convergence curves (averaged across trials)."""
    for net_name, strats in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for strat, trials in strats.items():
            # Get minimum convergence length
            min_len = min(len(t['convergence']) for t in trials)
            curves = np.array([t['convergence'][:min_len] for t in trials])

            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            gens = np.arange(min_len)

            ax.plot(gens, mean_curve, label=f'{strat} (mean)', linewidth=2)
            ax.fill_between(gens, mean_curve - std_curve,
                            mean_curve + std_curve, alpha=0.2)

        ax.set_xlabel('Generation / Iteration', fontsize=12)
        ax.set_ylabel('Best Cost', fontsize=12)
        ax.set_title(f'{net_name} Network — Convergence', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Use log scale if values span orders of magnitude
        ax.set_yscale('log')

        path = os.path.join(output_dir, f'convergence_{net_name.lower().replace("-","_")}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if not quiet:
            print(f"  → Convergence plot: {path}")


def _plot_boxplots(all_results, output_dir, quiet=False):
    """Box plots comparing strategy cost distributions."""
    for net_name, strats in all_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        data = []
        labels = []
        for strat, trials in strats.items():
            costs = [t['best_cost'] for t in trials]
            data.append(costs)
            labels.append(strat)

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        colors = ['#4A90D9', '#D94A4A', '#4CAF50', '#D9A13B', '#7E57C2']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Best Cost Found', fontsize=12)
        ax.set_title(f'{net_name} Network — Cost Distribution ({len(trials)} trials)',
                     fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        path = os.path.join(output_dir, f'boxplot_{net_name.lower().replace("-","_")}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if not quiet:
            print(f"  → Box plot: {path}")
