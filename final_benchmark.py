#!/usr/bin/env python3
"""
Presentation benchmark launcher.

Runs a small set of curated benchmark combinations so you can show how the
program behaves on different networks and solvers without manually typing a
long command each time.

Examples
--------
    python final_benchmark.py
    python final_benchmark.py --preset quick
    python final_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_BENCHMARKS = ROOT / "run_benchmarks.py"


def build_scenarios(preset: str) -> list[dict[str, object]]:
    """Return the benchmark scenarios for a given preset."""
    common = {
        "cores": 8,
    }

    scenarios_by_preset = {
        "quick": [
            {
                "label": "Two-Loop GA demo",
                "args": [
                    "--network", "TwoLoop",
                    "--strategy", "GA",
                    "--trials", "1",
                    "--gen", "8",
                    "--pop", "40",
                    "--twoloop-vmin", "0.3",
                    "--twoloop-vmax", "2.0",
                ],
            },
            {
                "label": "Hanoi PSO demo",
                "args": [
                    "--network", "Hanoi",
                    "--strategy", "PSO",
                    "--trials", "1",
                    "--gen", "6",
                    "--pop", "40",
                    "--hanoi-vmin", "0.0",
                    "--hanoi-vmax", "7.0",
                ],
            },
            {
                "label": "GoYang GA demo",
                "args": [
                    "--network", "GoYang",
                    "--strategy", "GA",
                    "--trials", "1",
                    "--gen", "6",
                    "--pop", "30",
                    "--goyang-vmin", "0.0",
                    "--goyang-vmax", "2.0",
                ],
            },
        ],
        "presentation": [
            {
                "label": "Two-Loop all strategies",
                "args": [
                    "--network", "TwoLoop",
                    "--strategy", "all",
                    "--trials", "2",
                    "--gen", "10",
                    "--pop", "50",
                    "--twoloop-vmin", "0.3",
                    "--twoloop-vmax", "2.0",
                ],
            },
            {
                "label": "Hanoi all strategies",
                "args": [
                    "--network", "Hanoi",
                    "--strategy", "all",
                    "--trials", "1",
                    "--gen", "8",
                    "--pop", "50",
                    "--hanoi-vmin", "0.0",
                    "--hanoi-vmax", "7.0",
                ],
            },
            {
                "label": "GoYang all strategies",
                "args": [
                    "--network", "GoYang",
                    "--strategy", "all",
                    "--trials", "1",
                    "--gen", "8",
                    "--pop", "50",
                    "--goyang-vmin", "0.0",
                    "--goyang-vmax", "2.0",
                ],
            },
        ],
        "strict": [
            {
                "label": "Two-Loop strict bounds",
                "args": [
                    "--network", "TwoLoop",
                    "--strategy", "GA",
                    "--trials", "1",
                    "--gen", "10",
                    "--pop", "60",
                    "--vmin", "0.3",
                    "--vmax", "2.0",
                ],
            },
            {
                "label": "All networks with defaults",
                "args": [
                    "--network", "all",
                    "--strategy", "GA",
                    "--trials", "1",
                    "--gen", "6",
                    "--pop", "40",
                ],
            },
        ],
    }

    if preset not in scenarios_by_preset:
        raise ValueError(f"Unknown preset: {preset}")

    scenarios = []
    for item in scenarios_by_preset[preset]:
        args = [str(PYTHON), str(RUN_BENCHMARKS), *item["args"]]
        args.extend(["--cores", str(common["cores"])])
        scenarios.append({"label": item["label"], "cmd": args})
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a curated set of benchmark demos for presentation.")
    parser.add_argument(
        "--preset",
        choices=["quick", "presentation", "strict"],
        default="presentation",
        help="Which scenario set to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"Python interpreter not found: {PYTHON}", file=sys.stderr)
        return 2
    if not RUN_BENCHMARKS.exists():
        print(f"run_benchmarks.py not found: {RUN_BENCHMARKS}", file=sys.stderr)
        return 2

    scenarios = build_scenarios(args.preset)

    print("═" * 72)
    print("Final Benchmark Launcher")
    print(f"Preset: {args.preset}")
    print("═" * 72)

    for index, scenario in enumerate(scenarios, start=1):
        label = scenario["label"]
        cmd = scenario["cmd"]
        print(f"\n[{index}/{len(scenarios)}] {label}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\nAll requested benchmark scenarios completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())