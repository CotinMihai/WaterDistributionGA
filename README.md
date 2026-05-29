# Water Distribution Network (WDN) Optimizer

This project benchmarks metaheuristic solvers for least-cost water distribution network design under hydraulic constraints.

It currently supports three solvers:
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Particle Swarm Optimization (PSO)

The benchmark runs against three classic networks:
- Two-Loop
- Hanoi
- GoYang

## Quick Start

Use the project virtual environment. The system Python may not have the required hydraulic dependencies.

```bash
source .venv/bin/activate
/home/mihai/eaa/.venv/bin/python run_benchmarks.py --network TwoLoop --strategy all --trials 1 --gen 20 --pop 20
```

For a full benchmark run:

```bash
/home/mihai/eaa/.venv/bin/python run_benchmarks.py --strategy all --network all --trials 30 --gen 2000 --pop 100 --output results
```

## CLI Reference

| Argument | Description | Notes |
|---|---|---|
| `--network` | Selects the benchmark topology. | `TwoLoop`, `Hanoi`, `GoYang`, or `all` |
| `--strategy` | Selects the solver set. | `GA`, `SA`, `PSO`, or `all` |
| `--trials` | Number of independent runs per network and strategy. | Default: `30` |
| `--gen` | Number of generations / iterations per trial. | Used by all solvers |
| `--pop` | Population size for GA and swarm size for PSO. | Ignored by SA |
| `--output` | Output directory for CSV files and plots. | Default: `results` |
| `--verbose` | Prints solver internals every 100 steps. | Progress bar is hidden when verbose is on |
| `--vmin` | Global minimum absolute velocity for feasibility (m/s). | Optional override |
| `--vmax` | Global maximum absolute velocity for feasibility (m/s). | Optional override |
| `--twoloop-vmin`, `--twoloop-vmax` | Two-Loop velocity bounds override. | Optional |
| `--hanoi-vmin`, `--hanoi-vmax` | Hanoi velocity bounds override. | Optional |
| `--goyang-vmin`, `--goyang-vmax` | GoYang velocity bounds override. | Optional |
| `--cores` | Number of CPU cores for parallel evaluation. | Default: all-but-one |

## What `all` Means

`--strategy all` runs all implemented strategies in sequence:
- GA
- SA
- PSO

`--network all` runs all three benchmark networks.

## How It Works

The benchmark follows these phases:

1. Build the selected water network model.
2. Generate an initial population or starting state.
3. Evaluate each candidate with WNTR hydraulics.
4. Compute cost, pressure penalty, and velocity penalty.
5. Apply the solver-specific search step.
6. Repeat until the generation or iteration limit is reached.
7. Collect the best trial for each solver and network.
8. Write summary CSV files and plots.

## Deviations From The Paper

This repo keeps the paper-style baseline intact, but it also includes a few practical deviations that are useful for benchmarking and presentation:

| Change | Type | Why it exists |
|---|---|---|
| EPANET-backed hydraulic evaluation through WNTR | implementation / speed | Faster than the native WNTR simulator when EPANET is available, while keeping the same solver interface. |
| Reused hydraulic worker pool and cached candidate evaluations | implementation / speed | Reduces repeated simulator overhead during large benchmark runs. |
| SA adaptive step scale | algorithmic tweak | Starts with larger SA moves and gradually shrinks them for later fine-tuning. |
| Earlier GA hill climbing | algorithmic tweak | Gives GA more time in the refinement phase. |
| PSO tail local search | algorithmic tweak | Refines the best particle near the end of a run. |

Recommended interpretation:
- Keep the algorithmic tweaks when you want better practical results or to highlight benchmark deviations.
- Leave them off only when you need a stricter paper-faithful baseline for comparison.

## Validation Rules

A candidate solution is considered feasible when it satisfies the hydraulic pressure requirement at all demand nodes.

The evaluator checks:
- minimum nodal pressure against the network target
- pipe velocity penalty against the target velocity
- total pipe material cost

Important: PP and VP are optimization penalties and can still be greater than 1 for feasible solutions,
because they also include soft deviations (for example over-pressure and off-target velocity).

Current feasibility policy used by this project:
- Pressure: all demand nodes must be at or above the network minimum pressure.
- Velocity: all pipes must have absolute velocity in [`vmin`, `vmax`].

Default feasibility bounds are network-specific:
- Two-Loop: `0.3` to `2.0` m/s
- Hanoi: `0.0` to `7.0` m/s
- GoYang: `0.0` to `2.0` m/s

Use `--vmin` and `--vmax` for global overrides, or per-network flags to avoid over-constraining all benchmarks with one shared bound.

The fitness value is based on the penalized objective:

$$
fitness = \frac{1}{cost \times PP_{eff} \times VP_{eff}}
$$

The system also uses temperature-based penalty annealing so early search is less rigid and later search becomes stricter.

## Winner Selection

Each trial produces one best candidate. The benchmark then compares trials and reports the winner using this rule:

- prefer the lowest-cost feasible solution
- if no feasible solution exists, use the lowest penalized cost as a fallback

The summary output reports:
- best cost
- mean cost
- standard deviation
- number of feasible trials
- average runtime

## Outputs

The benchmark writes these artifacts in the output directory:

- `summary.csv` - aggregated results per network and strategy
- `detailed_results.csv` - per-trial results
- `convergence_[network].png` - convergence plots
- `boxplot_[network].png` - cost distribution plots

## Notes

- If you launch the script with `python3 run_benchmarks.py` outside the project venv, `wntr` may be missing.
- The progress bar shows the current generation / iteration for each trial.
- `--pop` affects GA/PSO directly. SA iteration budget is scaled to keep evaluation effort comparable:
	`SA iterations = gen × pop`.
- Simulation backend is selected with `WDN_SIMULATOR` (`epanet` or `wntr`). The native WNTR simulator is usually slower; WNTR's EPANET-backed path is typically faster and is the default in this repo when available.

## Step-by-Step Guide

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a detailed explanation of the pipeline, phases, and winner validation logic.
