"""
Simulated Annealing solver for WDN optimization.

This complements GA as a second optimizer cited in
WDN literature. SA uses a temperature schedule to balance exploration and
exploitation while sharing the same hydraulic penalty model.
"""

import time
import numpy as np

from .hydraulics import HydraulicEvaluator


class SASolver:
    """Simulated Annealing for pipe diameter index optimization."""

    def __init__(self, problem, max_iter=2000, initial_temp=1.0,
                 final_temp=1e-3, step_scale=0.1,
                 penalty_scale_min=0.25, adaptive_step_scale=True,
                 seed=None, n_workers=None):
        self.problem = problem
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.step_scale = step_scale
        self.penalty_scale_min = penalty_scale_min
        self.adaptive_step_scale = adaptive_step_scale
        self.n_workers = n_workers
        self.rng = np.random.default_rng(seed)

        self.evaluator = HydraulicEvaluator(problem)
        self.num_dims = problem.num_pipes
        self.num_diameters = problem.num_diameters

    def solve(self, verbose=True, progress_callback=None):
        t0 = time.time()

        # Start from a deliberately feasible high-diameter design, then anneal down.
        current = np.full(self.num_dims, self.num_diameters - 1, dtype=int)
        current_eval = self.evaluator.evaluate(
            current,
            penalty_scale=self._penalty_scale_for_progress(0.0),
        )

        best = current.copy()
        best_eval = dict(current_eval)

        convergence = [self._objective_for_curve(best_eval)]

        for iteration in range(1, self.max_iter + 1):
            progress = iteration / self.max_iter
            temperature = self._temperature(progress)
            penalty_scale = self._penalty_scale_for_progress(progress)

            candidate = self._neighbor(current, temperature)
            cand_eval = self.evaluator.evaluate(candidate, penalty_scale=penalty_scale)

            if self._accept(current_eval, cand_eval, temperature):
                current = candidate
                current_eval = cand_eval

            if self._is_better(cand_eval, best_eval):
                best = candidate.copy()
                best_eval = dict(cand_eval)

            convergence.append(self._objective_for_curve(best_eval))

            if verbose and iteration % 100 == 0:
                status = "✓" if best_eval['feasible'] else "✗"
                print(
                    f"  SA iter {iteration:4d}/{self.max_iter}: "
                    f"best_cost={best_eval['cost']:,.0f}  "
                    f"PP={best_eval['PP']:.3f}  VP={best_eval['VP']:.3f}  "
                    f"T={temperature:.4f}  feasible={status}"
                )

            if progress_callback is not None:
                progress_callback(iteration, self.max_iter)

        wall_time = time.time() - t0
        self.evaluator.close()

        return {
            'best_cost': best_eval['cost'],
            'best_penalized_cost': best_eval['penalized_cost'],
            'best_solution': best,
            'best_feasible': best_eval['feasible'],
            'best_pressure_feasible': best_eval.get('pressure_feasible'),
            'best_velocity_feasible': best_eval.get('velocity_feasible'),
            'best_pressures': best_eval['pressures'],
            'min_pressure': best_eval.get('min_pressure'),
            'pressure_violations': best_eval.get('pressure_violations'),
            'min_abs_velocity': best_eval.get('min_abs_velocity'),
            'max_abs_velocity': best_eval.get('max_abs_velocity'),
            'velocity_low_violations': best_eval.get('velocity_low_violations'),
            'velocity_high_violations': best_eval.get('velocity_high_violations'),
            'velocity_low_pipes': best_eval.get('velocity_low_pipes', []),
            'velocity_high_pipes': best_eval.get('velocity_high_pipes', []),
            'convergence': convergence,
            'wall_time': wall_time,
            'final_generation': self.max_iter,
            'PP': best_eval['PP'],
            'VP': best_eval['VP'],
        }

    def _temperature(self, progress):
        # Geometric-like cooling in log-space for smooth decay.
        progress = float(np.clip(progress, 0.0, 1.0))
        log_t0 = np.log(max(self.initial_temp, 1e-12))
        log_tf = np.log(max(self.final_temp, 1e-12))
        return float(np.exp(log_t0 + (log_tf - log_t0) * progress))

    def _penalty_scale_for_progress(self, progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.penalty_scale_min + (1.0 - self.penalty_scale_min) * progress

    def _neighbor(self, current, temperature):
        neighbor = current.copy()

        # Higher temperature mutates more positions.
        temp_ratio = (temperature - self.final_temp) / max(self.initial_temp - self.final_temp, 1e-12)
        temp_ratio = float(np.clip(temp_ratio, 0.0, 1.0))

        if self.adaptive_step_scale:
            progress = 1.0 - temp_ratio
            effective_step_scale = 0.4 - (0.3 * progress)
        else:
            effective_step_scale = self.step_scale

        max_mutations = max(1, int(np.ceil(effective_step_scale * self.num_dims)))
        n_mut = max(1, int(np.ceil(1 + temp_ratio * (max_mutations - 1))))
        idxs = self.rng.choice(self.num_dims, size=n_mut, replace=False)

        for idx in idxs:
            step = self.rng.choice([-1, 1])
            neighbor[idx] = int(np.clip(neighbor[idx] + step, 0, self.num_diameters - 1))

        return neighbor

    def _accept(self, current_eval, candidate_eval, temperature):
        if candidate_eval['feasible'] and not current_eval['feasible']:
            return True
        if current_eval['feasible'] and not candidate_eval['feasible']:
            return False

        curr_obj = current_eval['penalized_cost']
        cand_obj = candidate_eval['penalized_cost']

        if cand_obj <= curr_obj:
            return True

        delta = cand_obj - curr_obj
        prob = np.exp(-delta / max(temperature * max(curr_obj, 1.0), 1e-12))
        return self.rng.random() < prob

    def _is_better(self, a, b):
        if a['feasible'] and not b['feasible']:
            return True
        if b['feasible'] and not a['feasible']:
            return False

        if a['feasible'] and b['feasible']:
            return a['cost'] < b['cost']

        return a['penalized_cost_raw'] < b['penalized_cost_raw']

    def _objective_for_curve(self, eval_result):
        return eval_result['cost'] if eval_result['feasible'] else eval_result['penalized_cost']
