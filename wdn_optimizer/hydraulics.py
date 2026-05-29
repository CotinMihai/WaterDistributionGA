"""
Hydraulic evaluation wrapper around WNTR.

Design decision: We use WNTR's built-in EpanetSimulator (which calls the
EPANET C library under the hood) for hydraulic simulation.  This gives us
accurate, validated hydraulic results identical to what the SOP-WDN paper
obtains through EPANET-MATLAB.

The evaluate() function is the single interface between the optimizer and
the hydraulic solver: it takes a candidate solution (list of diameter
indices), sets the pipe diameters in the network model, runs the
simulation, and returns cost + penalties.
"""

import warnings
import numpy as np
import wntr
import copy
import logging
import os
from multiprocessing import Pool
from collections import OrderedDict

# Suppress WNTR/EPANET warnings during batch optimization
logging.getLogger('wntr').setLevel(logging.ERROR)

from .utils import compute_penalties, apply_penalty_scale, fitness, is_feasible


_CACHE_LIMIT = 4096


def _cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _compute_base_result(problem, wn, diameter_indices, simulator_backend,
                         junction_names, pipe_names, available_diameters,
                         diameter_costs, pipe_lengths):
    """Compute hydraulic outputs that do not depend on penalty scaling."""
    indices = np.asarray(diameter_indices, dtype=np.intp)
    cache_key = tuple(int(i) for i in indices)

    cost = float(np.dot(diameter_costs[indices], pipe_lengths))

    for i, pname in enumerate(pipe_names):
        pipe = wn.get_link(pname)
        pipe.diameter = float(available_diameters[int(indices[i])])

    try:
        results = _run_simulation(wn, simulator_backend)

        pressure_df = results.node['pressure']
        pressures = np.array([pressure_df[jn].iloc[0] for jn in junction_names])

        velocity_df = results.link['velocity']
        velocities = np.array([velocity_df[pn].iloc[0] for pn in pipe_names])

    except Exception:
        pressures = np.full(len(junction_names), -100.0)
        velocities = np.zeros(len(pipe_names))

    PP_raw, VP_raw = compute_penalties(
        pressures,
        problem.min_pressure,
        velocities,
        feasible_min_velocity=problem.min_velocity,
        feasible_max_velocity=problem.max_velocity,
        pressure_hard_multiplier=50.0,
        velocity_hard_multiplier=50.0,
    )
    pressure_feasible = np.all(pressures >= problem.min_pressure - 1e-3)
    v_abs = np.abs(velocities)
    velocity_feasible = np.all(v_abs >= problem.min_velocity) and np.all(v_abs <= problem.max_velocity)
    feasible = is_feasible(
        pressures,
        problem.min_pressure,
        pipe_velocities=velocities,
        min_velocity=problem.min_velocity,
        max_velocity=problem.max_velocity,
    )
    min_pressure = float(np.min(pressures)) if len(pressures) else float('nan')
    pressure_violations = int(np.sum(pressures < (problem.min_pressure - 1e-3)))
    max_velocity = float(np.max(np.abs(velocities))) if len(velocities) else 0.0
    min_velocity = float(np.min(np.abs(velocities))) if len(velocities) else 0.0
    velocity_low_violations = int(np.sum(v_abs < problem.min_velocity))
    velocity_high_violations = int(np.sum(v_abs > problem.max_velocity))
    low_idx = np.where(v_abs < problem.min_velocity)[0]
    high_idx = np.where(v_abs > problem.max_velocity)[0]
    low_pipes = [pipe_names[int(i)] for i in low_idx]
    high_pipes = [pipe_names[int(i)] for i in high_idx]

    return cache_key, {
        'cost': cost,
        'PP': PP_raw,
        'VP': VP_raw,
        'feasible': feasible,
        'pressure_feasible': bool(pressure_feasible),
        'velocity_feasible': bool(velocity_feasible),
        'pressures': pressures,
        'velocities': velocities,
        'min_pressure': min_pressure,
        'pressure_violations': pressure_violations,
        'max_abs_velocity': max_velocity,
        'min_abs_velocity': min_velocity,
        'velocity_low_violations': velocity_low_violations,
        'velocity_high_violations': velocity_high_violations,
        'velocity_low_pipes': low_pipes,
        'velocity_high_pipes': high_pipes,
    }

def _apply_penalty_result(base, penalty_scale):
    """Attach penalty-scale-dependent values to a cached base result."""
    PP_eff, VP_eff = apply_penalty_scale(base['PP'], base['VP'], penalty_scale)
    fit = fitness(base['cost'], PP_eff, VP_eff)
    penalized_cost = base['cost'] * PP_eff * VP_eff
    penalized_cost_raw = base['cost'] * base['PP'] * base['VP']
    fit_raw = fitness(base['cost'], base['PP'], base['VP'])

    result = dict(base)
    result.update({
        'PP_effective': PP_eff,
        'VP_effective': VP_eff,
        'fitness': fit,
        'fitness_raw': fit_raw,
        'penalty_scale': penalty_scale,
        'penalized_cost': penalized_cost,
        'penalized_cost_raw': penalized_cost_raw,
    })
    return result


def _evaluate_candidate(problem, wn, diameter_indices, penalty_scale, simulator_backend,
                        junction_names, pipe_names, available_diameters,
                        diameter_costs, pipe_lengths, cache=None):
    """Evaluate one candidate on a provided network object."""
    indices = np.asarray(diameter_indices, dtype=np.intp)
    cache_key = tuple(int(i) for i in indices)

    base = None
    if cache is not None:
        base = _cache_get(cache, cache_key)

    if base is None:
        _, base = _compute_base_result(
            problem,
            wn,
            indices,
            simulator_backend,
            junction_names,
            pipe_names,
            available_diameters,
            diameter_costs,
            pipe_lengths,
        )
        if cache is not None:
            _cache_put(cache, cache_key, base)

    return _apply_penalty_result(base, penalty_scale)


_WORKER_STATE = {}


def _worker_init(problem, simulator_backend):
    """Initialize one worker with its own reusable network copy."""
    global _WORKER_STATE
    _WORKER_STATE = {
        'problem': problem,
        'wn': problem.wn,
        'simulator_backend': simulator_backend,
        'junction_names': [name for name, _ in problem.wn.junctions()],
        'pipe_names': list(problem.pipe_names),
        'available_diameters': np.asarray(problem.available_diameters, dtype=float),
        'diameter_costs': np.asarray(problem.diameter_costs, dtype=float),
        'pipe_lengths': np.asarray(problem.pipe_lengths, dtype=float),
        'cache': OrderedDict(),
    }


def _worker_evaluate(args):
    diameter_indices, penalty_scale = args
    state = _WORKER_STATE
    return _evaluate_candidate(
        state['problem'],
        state['wn'],
        diameter_indices,
        penalty_scale,
        state['simulator_backend'],
        state['junction_names'],
        state['pipe_names'],
        state['available_diameters'],
        state['diameter_costs'],
        state['pipe_lengths'],
        cache=state['cache'],
    )


def _run_simulation(wn, simulator_backend):
    """Run a hydraulic simulation, falling back to WNTR if EPANET is unavailable."""
    backends = ['epanet', 'wntr'] if simulator_backend == 'epanet' else [simulator_backend]
    last_error = None

    for backend in backends:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if backend == 'wntr':
                    sim = wntr.sim.WNTRSimulator(wn)
                else:
                    sim = wntr.sim.EpanetSimulator(wn)
                return sim.run_sim()
        except Exception as exc:
            last_error = exc

    raise last_error


class HydraulicEvaluator:
    """
    Evaluates candidate WDN solutions using WNTR's hydraulic simulator.

    Parameters
    ----------
    problem : WDNProblem
        The benchmark problem definition.
    """

    def __init__(self, problem):
        self.problem = problem
        # Keep one reusable network for sequential evaluation.
        self._template_wn = problem.wn
        self._working_wn = copy.deepcopy(problem.wn)
        self._pool = None
        self._pool_workers = None
        requested_backend = os.getenv('WDN_SIMULATOR', 'epanet').strip().lower()
        self.simulator_backend = requested_backend if requested_backend in {'wntr', 'epanet'} else 'epanet'
        self._junction_names = [name for name, _ in problem.wn.junctions()]
        self._pipe_names = list(problem.pipe_names)
        self._available_diameters = np.asarray(problem.available_diameters, dtype=float)
        self._diameter_costs = np.asarray(problem.diameter_costs, dtype=float)
        self._pipe_lengths = np.asarray(problem.pipe_lengths, dtype=float)
        self._cache = OrderedDict()

    def close(self):
        """Release any cached worker pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            self._pool_workers = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def evaluate(self, diameter_indices: np.ndarray,
                 penalty_scale: float = 1.0) -> dict:
        """
        Evaluate one candidate solution.

        Parameters
        ----------
        diameter_indices : array of int, shape (num_pipes,)
            Index into problem.available_diameters for each pipe.

        Returns
        -------
        result : dict with keys:
            'cost'       : float — total pipe cost
            'PP'         : float — pressure penalty (≥1)
            'VP'         : float — velocity penalty (≥1)
            'fitness'    : float — 1 / (cost × PP × VP)
            'feasible'   : bool  — all pressures ≥ min_pressure
            'pressures'  : array — nodal pressures
            'velocities' : array — pipe velocities
            'penalized_cost' : float — cost × PP × VP
        """
        return _evaluate_candidate(
            self.problem,
            self._working_wn,
            diameter_indices,
            penalty_scale,
            self.simulator_backend,
            self._junction_names,
            self._pipe_names,
            self._available_diameters,
            self._diameter_costs,
            self._pipe_lengths,
            cache=self._cache,
        )

    def evaluate_batch(self, population: np.ndarray,
                       penalty_scale: float = 1.0,
                       n_workers: int = None) -> list[dict]:
        """
        Evaluate a batch of solutions using multiprocessing for speed.
        
        Parameters
        ----------
        population : array of shape (pop_size, num_pipes)
        penalty_scale : float
        n_workers : int
            Number of parallel processes. If None, uses CPU count - 1.
            
        Returns
        -------
        results : list of dict (one per individual)
        """
        if n_workers is None:
            n_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
        
        # Skip pool overhead for tiny batches
        if len(population) == 1 or n_workers == 1:
            return [self.evaluate(individual, penalty_scale=penalty_scale)
                    for individual in population]
        
        try:
            if self._pool is None or self._pool_workers != n_workers:
                self.close()
                self._pool = Pool(
                    processes=n_workers,
                    initializer=_worker_init,
                    initargs=(self.problem, self.simulator_backend),
                )
                self._pool_workers = n_workers

            payload = [(individual, penalty_scale) for individual in population]
            chunksize = max(1, len(payload) // (n_workers * 4))
            return self._pool.map(_worker_evaluate, payload, chunksize=chunksize)
        except Exception as e:
            print(f"  [Warning] Multiprocessing failed ({e}), using sequential eval")
            return [self.evaluate(individual, penalty_scale=penalty_scale)
                    for individual in population]
