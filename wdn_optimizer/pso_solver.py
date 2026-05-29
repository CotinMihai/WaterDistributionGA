"""
Discrete Particle Swarm Optimization Solver (Strategy 2).

Based on modified PSO from Surco et al. (2018), cited in the SOTA survey
as a strong competitor for WDN optimization.

Key differences from the GA:
  • No encoding/decoding — works directly with integer diameter indices
  • Swarm intelligence instead of evolutionary operators
  • Continuous velocity mapped to discrete space by rounding/clamping
  • Typically converges faster initially but may lack fine-tuning
"""

import numpy as np
import time
from .hydraulics import HydraulicEvaluator


class PSOSolver:
    """
    Discrete PSO for WDN pipe diameter optimization.

    Parameters
    ----------
    problem : WDNProblem
    swarm_size : int
        Number of particles (default 100 — same budget as GA).
    max_iter : int
        Maximum iterations (default 2000 — same budget as GA).
    w_start, w_end : float
        Inertia weight linearly decreases from w_start to w_end.
        Default: 0.9 → 0.4 (promotes exploration → exploitation).
    c1, c2 : float
        Cognitive and social coefficients (default 2.0).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, problem, swarm_size=100, max_iter=2000,
                 w_start=0.7, w_end=0.4, c1=1.8, c2=1.8,
                 penalty_scale_min=0.25, seed=None, n_workers=None):
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.penalty_scale_min = penalty_scale_min
        self.n_workers = n_workers
        self.rng = np.random.default_rng(seed)

        self.evaluator = HydraulicEvaluator(problem)
        self.num_dims = problem.num_pipes
        self.num_diameters = problem.num_diameters

        # Velocity clamping: |v| ≤ Vmax
        # Use a fraction of the diameter range as the default Vmax to
        # prevent particles from making extreme jumps every iteration.
        # This keeps PSO stable in the discrete indexing space while
        # still allowing meaningful exploration.
        self.vmax = max(1.0, 0.2 * float(self.num_diameters - 1))

    def solve(self, verbose=True, progress_callback=None) -> dict:
        """
        Run the PSO and return results.

        Returns
        -------
        result : dict (same structure as GASolver.solve)
        """
        t0 = time.time()
        nd = self.num_diameters

        # --- 1. Initialize swarm ---
        # DESIGN: Positions are integers in [0, num_diameters-1].
        # Velocities are real-valued (continuous PSO mechanics applied to
        # discrete space).
        positions = self.rng.integers(0, nd,
                                       size=(self.swarm_size, self.num_dims))
        # Initialize velocities with a smaller spread to avoid large
        # early jumps; particles can still accelerate via PSO updates.
        velocities = self.rng.uniform(-0.5 * self.vmax, 0.5 * self.vmax,
                                       size=(self.swarm_size, self.num_dims))

        # Seed one particle with the maximum-diameter index for all pipes
        # (analogous to GA seeding). This ensures at least one feasible
        # / near-feasible solution exists in the initial swarm and helps
        # stabilise penalty-driven searches.
        try:
            positions[0] = nd - 1
            velocities[0] = 0.0
        except Exception:
            pass

        # Evaluate initial swarm
        evals = self.evaluator.evaluate_batch(
            positions,
            penalty_scale=self._penalty_scale_for_progress(0.0),
            n_workers=self.n_workers,
        )
        score_vals = np.array([e['fitness_raw'] for e in evals])

        # Personal bests
        pbest_pos = positions.copy()
        pbest_score = score_vals.copy()
        pbest_evals = list(evals)

        # Global best
        gbest_idx = np.argmax(score_vals)
        gbest_pos = positions[gbest_idx].copy()
        gbest_eval = evals[gbest_idx]
        gbest_score = score_vals[gbest_idx]

        convergence = [gbest_eval['cost'] if gbest_eval['feasible']
                       else gbest_eval['penalized_cost']]

        for iteration in range(1, self.max_iter + 1):
            penalty_scale = self._penalty_scale_for_progress(iteration / self.max_iter)
            # --- 2. Compute inertia weight (linear decrease) ---
            # WHY LINEAR DECREASE: Early iterations need high inertia
            # (exploration/momentum), late iterations need low inertia
            # (exploitation/fine-tuning). This is the standard approach
            # and empirically effective.
            w = self.w_start - (self.w_start - self.w_end) * (
                iteration / self.max_iter)

            # --- 3. Update velocities and positions ---
            r1 = self.rng.random(size=(self.swarm_size, self.num_dims))
            r2 = self.rng.random(size=(self.swarm_size, self.num_dims))

            # Standard PSO velocity update equation:
            #   v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            # WHY THIS EQUATION: The three terms balance:
            #   - momentum (w*v): continue current trajectory
            #   - cognitive (c1*r1*(pbest-x)): move toward personal best
            #   - social (c2*r2*(gbest-x)): move toward swarm's best
            velocities = (w * velocities
                          + self.c1 * r1 * (pbest_pos - positions)
                          + self.c2 * r2 * (gbest_pos - positions))

            # Clamp velocities
            velocities = np.clip(velocities, -self.vmax, self.vmax)

            # Update positions (continuous → discrete by rounding)
            # DESIGN: We add velocity to position, then round and clamp
            # to valid index range. This is the standard discrete PSO
            # approach — simpler and more effective than sigmoid-based
            # binary PSO for this multi-valued discrete problem.
            new_positions = np.round(positions + velocities).astype(int)
            new_positions = np.clip(new_positions, 0, nd - 1)
            positions = new_positions

            # --- 4. Evaluate ---
            evals = self.evaluator.evaluate_batch(positions, penalty_scale=penalty_scale,
                                                  n_workers=self.n_workers)
            score_vals = np.array([e['fitness_raw'] for e in evals])

            # --- 5. Update personal and global bests ---
            for i in range(self.swarm_size):
                if score_vals[i] > pbest_score[i]:
                    pbest_score[i] = score_vals[i]
                    pbest_pos[i] = positions[i].copy()
                    pbest_evals[i] = evals[i]

                    if score_vals[i] > gbest_score:
                        gbest_score = score_vals[i]
                        gbest_pos = positions[i].copy()
                        gbest_eval = evals[i]

            convergence.append(gbest_eval['cost'] if gbest_eval['feasible']
                               else gbest_eval['penalized_cost'])

            if verbose and iteration % 100 == 0:
                status = "✓" if gbest_eval['feasible'] else "✗"
                print(f"  PSO iter {iteration:4d}/{self.max_iter}: "
                      f"best_cost={gbest_eval['cost']:,.0f}  "
                      f"PP={gbest_eval['PP']:.3f}  VP={gbest_eval['VP']:.3f}  "
                      f"feasible={status}")

            if progress_callback is not None:
                progress_callback(iteration, self.max_iter)

        wall_time = time.time() - t0
        self.evaluator.close()

        return {
            'best_cost': gbest_eval['cost'],
            'best_penalized_cost': gbest_eval['penalized_cost'],
            'best_solution': gbest_pos,
            'best_feasible': gbest_eval['feasible'],
            'best_pressure_feasible': gbest_eval.get('pressure_feasible'),
            'best_velocity_feasible': gbest_eval.get('velocity_feasible'),
            'best_pressures': gbest_eval['pressures'],
            'min_pressure': gbest_eval.get('min_pressure'),
            'pressure_violations': gbest_eval.get('pressure_violations'),
            'min_abs_velocity': gbest_eval.get('min_abs_velocity'),
            'max_abs_velocity': gbest_eval.get('max_abs_velocity'),
            'velocity_low_violations': gbest_eval.get('velocity_low_violations'),
            'velocity_high_violations': gbest_eval.get('velocity_high_violations'),
            'velocity_low_pipes': gbest_eval.get('velocity_low_pipes', []),
            'velocity_high_pipes': gbest_eval.get('velocity_high_pipes', []),
            'convergence': convergence,
            'wall_time': wall_time,
            'final_generation': self.max_iter,
            'PP': gbest_eval['PP'],
            'VP': gbest_eval['VP'],
        }

    def _penalty_scale_for_progress(self, progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.penalty_scale_min + (1.0 - self.penalty_scale_min) * progress
