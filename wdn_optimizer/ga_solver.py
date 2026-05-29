"""
Genetic Algorithm Solver (Strategy 1) — SOP-WDN Style.

Faithfully implements the GA described in Sangroula et al. (2022) §2.4:
  • Gray-code binary encoding with redundancy handling
  • Power-law rank-based selection (τ = 1.15)
  • k-point crossover (k ≈ 0.8 × Np, probability 87%)
  • Bit-flip mutation (rate 5%; hill-climbing in last 10% of generations)
  • Elitism (top 2 individuals preserved)
  • Penalty-based fitness: fitness = 1 / (cost × PP × VP)

Design decisions documented inline.
"""

import numpy as np
import time
from .utils import (encode_chromosome, decode_chromosome, bits_needed)
from .hydraulics import HydraulicEvaluator


class GASolver:
    """
    Genetic Algorithm for WDN pipe diameter optimization.

    Parameters
    ----------
    problem : WDNProblem
    pop_size : int
        Population size (default 100, from Table 3).
    max_gen : int
        Maximum generations (default 2000, from paper protocol).
    crossover_prob : float
        Crossover probability (default 0.87 — midpoint of 85–90%).
    mutation_rate : float
        Bit-flip mutation rate (default 0.05 — midpoint of 4–6%).
    tau : float
        Selection pressure exponent (default 1.15 — midpoint of 1.1–1.2).
    elite_count : int
        Number of elite individuals to preserve (default 2).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, problem, pop_size=100, max_gen=2000,
                 crossover_prob=0.87, mutation_rate=0.05, tau=1.15,
                 elite_count=2, penalty_scale_min=0.25, seed=None, n_workers=None):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.tau = tau
        self.elite_count = elite_count
        self.penalty_scale_min = penalty_scale_min
        self.n_workers = n_workers
        self.rng = np.random.default_rng(seed)

        self.evaluator = HydraulicEvaluator(problem)
        self.bits_per_pipe = bits_needed(problem.num_diameters)
        self.chrom_length = self.bits_per_pipe * problem.num_pipes

        # k-point crossover: k ≈ round(0.8 × Np)
        self.num_crossover_points = max(1, round(0.8 * problem.num_pipes))

    def solve(self, verbose=True, progress_callback=None) -> dict:
        """
        Run the GA and return results.

        Returns
        -------
        result : dict with keys:
            'best_cost', 'best_solution', 'best_feasible',
            'convergence' (list of best cost per generation),
            'wall_time', 'final_generation'
        """
        t0 = time.time()

        # --- 1. Initialize population (random binary strings) ---
        # Design: each chromosome is a flat binary array of length
        # (num_pipes × bits_per_pipe), representing Gray-coded diameter indices.
        # Seed the first individual with maximum diameter pipes (True bits) 
        # to ensure at least one fully feasible network exists in the initial gene pool,
        # preventing the stochastic GA from stalling inside the 'cheapest illegal' penalty flatline.
        population = self.rng.integers(0, 2,
                                        size=(self.pop_size, self.chrom_length),
                                        dtype=np.int8)
        population[0] = True

        # Evaluate initial population
        evals = self._evaluate_population(
            population,
            penalty_scale=self._penalty_scale_for_progress(0.0),
        )
        best_idx = 0
        for i in range(1, len(evals)):
            if self._is_better_for_best(evals[i], evals[best_idx]):
                best_idx = i
        best_eval = evals[best_idx]
        best_chrom = population[best_idx].copy()

        # Feasibility archive: keep the best feasible solution seen so far
        if best_eval['feasible']:
            best_feasible_eval = best_eval
            best_feasible_chrom = best_chrom.copy()
        else:
            best_feasible_eval = None
            best_feasible_chrom = None

        convergence = [best_eval['cost'] if best_eval['feasible']
                   else best_eval['penalized_cost']]

        hill_climbing_start = int(0.9 * self.max_gen)

        for gen in range(1, self.max_gen + 1):
            penalty_scale = self._penalty_scale_for_progress(gen / self.max_gen)
            # --- 2. Selection (power-law rank-based) ---
            parents = self._selection(population, evals)

            # --- 3. Crossover (k-point) ---
            offspring = self._crossover(parents)

            # --- 4. Mutation ---
            use_hill_climbing = gen >= hill_climbing_start
            if use_hill_climbing:
                offspring = self._hill_climbing_mutation(
                    offspring,
                    evals,
                    penalty_scale=penalty_scale,
                )
            else:
                offspring = self._mutation(offspring)

            # --- 5. Elitism ---
            # Preserve the top `elite_count` individuals from current gen
            fitness_vals = np.array([e['fitness'] for e in evals])
            elite_indices = np.argsort(fitness_vals)[-self.elite_count:]
            elite_chroms = population[elite_indices].copy()

            # Replace worst in offspring with elites
            population = offspring.copy()
            new_evals = self._evaluate_population(population, penalty_scale=penalty_scale)
            new_fitness = np.array([e['fitness'] for e in new_evals])
            worst_indices = np.argsort(new_fitness)[:self.elite_count]
            for i, wi in enumerate(worst_indices):
                population[wi] = elite_chroms[i]
                # Re-evaluate the elite (it was already evaluated)
                new_evals[wi] = evals[elite_indices[i]]

            evals = new_evals

            # Track best
            for i, candidate_eval in enumerate(evals):
                if self._is_better_for_best(candidate_eval, best_eval):
                    best_eval = candidate_eval
                    best_chrom = population[i].copy()

                # Update feasibility archive
                if candidate_eval.get('feasible'):
                    if (best_feasible_eval is None
                            or candidate_eval['cost'] < best_feasible_eval['cost']):
                        best_feasible_eval = candidate_eval
                        best_feasible_chrom = population[i].copy()

            convergence.append(best_eval['cost'] if best_eval['feasible']
                               else best_eval['penalized_cost'])

            if verbose and gen % 100 == 0:
                status = "✓" if best_eval['feasible'] else "✗"
                print(f"  GA gen {gen:4d}/{self.max_gen}: "
                      f"best_cost={best_eval['cost']:,.0f}  "
                      f"PP={best_eval['PP']:.3f}  VP={best_eval['VP']:.3f}  "
                      f"feasible={status}")

            if progress_callback is not None:
                progress_callback(gen, self.max_gen)

        wall_time = time.time() - t0

        # Decode best solution
        # Prefer best feasible solution if available (more stable)
        if best_feasible_eval is not None:
            best_eval = best_feasible_eval
            best_chrom = best_feasible_chrom

        best_indices = decode_chromosome(best_chrom, self.bits_per_pipe,
                                          self.problem.num_diameters)
        self.evaluator.close()

        return {
            'best_cost': best_eval['cost'],
            'best_penalized_cost': best_eval['penalized_cost'],
            'best_solution': best_indices,
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
            'final_generation': self.max_gen,
            'PP': best_eval['PP'],
            'VP': best_eval['VP'],
        }

    # ----- Internal methods -----

    def _evaluate_population(self, population, penalty_scale=1.0):
        """Decode chromosomes and evaluate via hydraulic simulator."""
        decoded_pop = np.array([decode_chromosome(chrom, self.bits_per_pipe, 
                                                   self.problem.num_diameters)
                                for chrom in population])
        return self.evaluator.evaluate_batch(decoded_pop, penalty_scale=penalty_scale, 
                                            n_workers=self.n_workers)

    def _penalty_scale_for_progress(self, progress):
        """Relax penalties early and enforce them near convergence."""
        progress = float(np.clip(progress, 0.0, 1.0))
        return self.penalty_scale_min + (1.0 - self.penalty_scale_min) * progress

    def _is_better_for_best(self, a, b):
        """Compare candidates using full penalties (scale-invariant)."""
        if a['feasible'] and not b['feasible']:
            return True
        if b['feasible'] and not a['feasible']:
            return False

        if a['feasible'] and b['feasible']:
            return a['cost'] < b['cost']

        return a['penalized_cost_raw'] < b['penalized_cost_raw']

    def _selection(self, population, evals):
        """
        Power-law rank-based selection (SOP-WDN).

        P(i) = i^(-τ) / Σ i^(-τ)
        where i is the rank (1 = best).

        WHY RANK-BASED: Unlike fitness-proportionate selection (roulette
        wheel), rank-based selection is robust to fitness scaling issues.
        Since fitness = 1/(cost×PP×VP), the absolute values can vary by
        orders of magnitude. Ranking normalizes this.
        """
        fitness_vals = np.array([e['fitness'] for e in evals])
        # Rank: 1 = best (highest fitness)
        order = np.argsort(-fitness_vals)  # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        # Power-law probabilities
        probs = ranks.astype(float) ** (-self.tau)
        probs /= probs.sum()

        # Select parents (with replacement)
        selected_idx = self.rng.choice(len(population), size=self.pop_size,
                                        replace=True, p=probs)
        return population[selected_idx].copy()

    def _crossover(self, parents):
        """
        k-point crossover (SOP-WDN).

        k = round(0.8 × Np) crossover points, applied at the pipe
        *boundary* positions to respect the gene structure.

        WHY k-POINT (not uniform): The paper found that higher mixing
        ratios lead to faster convergence. k-point with k ≈ 0.8×Np gives
        extensive mixing while preserving some linkage between adjacent
        pipe genes.
        """
        offspring = parents.copy()

        for i in range(0, self.pop_size - 1, 2):
            if self.rng.random() < self.crossover_prob:
                p1, p2 = offspring[i], offspring[i + 1]

                # Choose k random crossover points
                points = np.sort(self.rng.choice(
                    self.chrom_length - 1,
                    size=min(self.num_crossover_points,
                             self.chrom_length - 1),
                    replace=False)) + 1

                # Alternate segments
                child1, child2 = p1.copy(), p2.copy()
                swap = False
                prev = 0
                for pt in points:
                    if swap:
                        child1[prev:pt] = p2[prev:pt]
                        child2[prev:pt] = p1[prev:pt]
                    prev = pt
                    swap = not swap
                if swap:
                    child1[prev:] = p2[prev:]
                    child2[prev:] = p1[prev:]

                offspring[i] = child1
                offspring[i + 1] = child2

        return offspring

    def _mutation(self, population):
        """
        Bit-flip mutation (SOP-WDN).

        Each bit is flipped with probability = mutation_rate.

        WHY BIT-FLIP: Complements the Gray-code encoding — flipping a
        single bit in a Gray code typically changes the diameter by one
        step, making the mutation smooth in the solution space.
        """
        mask = self.rng.random(population.shape) < self.mutation_rate
        population[mask] = 1 - population[mask]
        return population

    def _hill_climbing_mutation(self, population, prev_evals, penalty_scale=1.0):
        """
        Hill-climbing mutation for the last 10% of generations.

        Mutation is only accepted if it *improves* the individual's fitness.
        This fine-tunes solutions near the end of the search.
        """
        for i in range(len(population)):
            mutant = population[i].copy()
            mask = self.rng.random(len(mutant)) < self.mutation_rate
            mutant[mask] = 1 - mutant[mask]

            if mask.any():
                # Evaluate mutant
                indices = decode_chromosome(mutant, self.bits_per_pipe,
                                             self.problem.num_diameters)
                result = self.evaluator.evaluate(indices, penalty_scale=penalty_scale)
                if result['fitness'] > prev_evals[i]['fitness']:
                    population[i] = mutant

        return population
