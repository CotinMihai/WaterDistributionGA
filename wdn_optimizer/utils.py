"""
Shared utilities: Gray-code encoding/decoding, penalty functions, helpers.

Design decisions documented inline.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gray Code (Reflected Binary Code)
# ---------------------------------------------------------------------------
# WHY GRAY CODE: The SOP-WDN paper (§2.4.1) uses Gray code so that adjacent
# pipe diameters differ by only 1 bit.  This reduces the Hamming cliff
# problem — a single bit-flip mutation always maps to a *neighbouring*
# diameter, making the mutation operator much more effective for this
# discrete search space.
# ---------------------------------------------------------------------------

def int_to_gray(n: int) -> int:
    """Convert integer to Gray code."""
    return n ^ (n >> 1)


def gray_to_int(g: int) -> int:
    """Convert Gray code back to integer."""
    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def encode_chromosome(diameter_indices: np.ndarray, bits_per_pipe: int) -> np.ndarray:
    """
    Encode a solution (array of diameter indices) into a Gray-coded
    binary chromosome (flat bit array).

    Parameters
    ----------
    diameter_indices : array of int, shape (num_pipes,)
        Index into the available-diameters list for each pipe.
    bits_per_pipe : int
        Number of bits used to encode each pipe diameter.

    Returns
    -------
    chromosome : array of int (0/1), shape (num_pipes * bits_per_pipe,)
    """
    chrom = []
    for idx in diameter_indices:
        gray = int_to_gray(int(idx))
        bits = [(gray >> (bits_per_pipe - 1 - b)) & 1
                for b in range(bits_per_pipe)]
        chrom.extend(bits)
    return np.array(chrom, dtype=np.int8)


def decode_chromosome(chromosome: np.ndarray, bits_per_pipe: int,
                       num_diameters: int) -> np.ndarray:
    """
    Decode a Gray-coded binary chromosome into diameter indices.

    Handles redundant codes by mapping them to valid indices using the
    linear-scaling strategy from SOP-WDN §2.4.1: redundant codes are
    spread across valid diameters with the largest pipe getting 2×
    the allocation chance of the smallest.

    Parameters
    ----------
    chromosome : array of int (0/1)
    bits_per_pipe : int
    num_diameters : int
        Number of valid commercial diameters.

    Returns
    -------
    diameter_indices : array of int, shape (num_pipes,)
    """
    num_pipes = len(chromosome) // bits_per_pipe
    max_code = 2 ** bits_per_pipe  # total representable codes
    indices = np.empty(num_pipes, dtype=np.int32)

    for p in range(num_pipes):
        segment = chromosome[p * bits_per_pipe:(p + 1) * bits_per_pipe]
        gray_val = 0
        for b in segment:
            gray_val = (gray_val << 1) | int(b)
        int_val = gray_to_int(gray_val)

        if int_val < num_diameters:
            indices[p] = int_val
        else:
            # Redundancy handling: linearly scaled mapping
            # Larger pipes get proportionally more allocation
            indices[p] = _map_redundant(int_val, num_diameters, max_code)

    return indices


def _map_redundant(code: int, num_diameters: int, max_code: int) -> int:
    """
    Map a redundant Gray-code value to a valid diameter index.

    Uses linear scaling so that the largest diameter has ~2× the
    allocation probability of the smallest diameter (SOP-WDN §2.4.1).
    """
    # Linear weights: weight[i] = 1 + i/(num_diameters-1)  → range [1, 2]
    weights = np.array([1.0 + i / max(1, num_diameters - 1)
                        for i in range(num_diameters)])
    cumulative = np.cumsum(weights) / np.sum(weights)
    frac = (code - num_diameters) / max(1, max_code - num_diameters)
    idx = np.searchsorted(cumulative, frac)
    return int(min(idx, num_diameters - 1))


# ---------------------------------------------------------------------------
# Penalty Functions (PP, VP) and Fitness
# ---------------------------------------------------------------------------
# WHY PENALTY-BASED FITNESS: The hydraulic constraints (minimum pressure,
# velocity bounds) are *implicit* — they can only be evaluated after running
# the hydraulic simulation.  Penalty functions map constraint violations
# into cost multipliers, allowing the optimizer to compare infeasible
# solutions by their *degree* of violation.
#
# Coefficient values from Table 3 of the paper:
#   PP1=0.02  (above target — mild, since over-pressure is acceptable)
#   PP2=1.9   (below target — severe, under-pressure is dangerous)
#   VP1=0.3   (above target velocity)
#   VP2=0.06  (below target velocity)
# ---------------------------------------------------------------------------

DEFAULT_PENALTY_COEFFS = {
    'PP1': 0.02,   # pressure above target (mild)
    'PP2': 1.9,    # pressure below target (severe)
    'VP1': 0.3,    # velocity above target
    'VP2': 0.65,   # velocity below target (increased from 0.06 to 0.4 to 0.55 for stricter enforcement)
}

def compute_penalties(nodal_pressures: np.ndarray,
                      min_pressure: float,
                      pipe_velocities: np.ndarray,
                      target_velocity: float = 1.0,
                      coeffs: dict | None = None,
                      feasible_min_velocity: float = 0.3,
                      feasible_max_velocity: float | None = None,
                      pressure_hard_multiplier: float = 15.0,
                      velocity_hard_multiplier: float = 25.0) -> tuple[float, float]:
    """
    Compute the Pressure Penalty (PP) and Velocity Penalty (VP).

    Parameters
    ----------
    nodal_pressures : array of float
        Simulated pressure at each demand node (m).
    min_pressure : float
        Minimum required pressure head (m).
    pipe_velocities : array of float
        Simulated velocity in each pipe (m/s).
    target_velocity : float
        Target velocity (m/s).  Default 1.0.
    coeffs : dict, optional
        Override default penalty coefficients.
    feasible_min_velocity : float
        Hard constraint threshold for min velocity. If any pipe falls below
        this, VP is multiplied by 10 to make infeasible solutions uncompetitive.
        Defaults to 0.3 m/s (Two-Loop, Hanoi). GoYang should pass 0.0.
    feasible_max_velocity : float
        Hard constraint threshold for max velocity. If any pipe exceeds
        this, VP is multiplied by velocity_hard_multiplier. Pass None to
        disable the upper-bound hard penalty.
    pressure_hard_multiplier : float
        Multiplier applied to PP if any node violates the minimum pressure.
    velocity_hard_multiplier : float
        Multiplier applied to VP if any pipe violates the velocity bounds.

    Returns
    -------
    (PP, VP) : tuple of float
        Pressure penalty and velocity penalty (both >= 1.0).
    """
    if coeffs is None:
        coeffs = DEFAULT_PENALTY_COEFFS

    # Keep pressure penalty threshold aligned with the feasibility tolerance.
    pressure_tol = 1e-3

    nodal_pressures = np.asarray(nodal_pressures, dtype=float)
    pipe_velocities = np.asarray(pipe_velocities, dtype=float)

    # Vectorized pressure penalty computation.
    p_diff = (min_pressure - pressure_tol) - nodal_pressures
    PP = 1.0
    PP += np.sum(np.clip(p_diff, 0.0, None) * coeffs['PP2'])
    PP += np.sum(np.clip(-p_diff, 0.0, None) * coeffs['PP1'])

    pressure_violation_count = np.sum(nodal_pressures < (min_pressure - pressure_tol))
    if pressure_violation_count > 0:
        PP *= pressure_hard_multiplier

    # Vectorized velocity penalty computation.
    v_diff = np.abs(pipe_velocities) - target_velocity
    VP = 1.0
    VP += np.sum(np.clip(v_diff, 0.0, None) * coeffs['VP1'])
    VP += np.sum(np.clip(-v_diff, 0.0, None) * coeffs['VP2'])

    # Hard-violation multiplier: if any pipe violates the feasibility bounds,
    # multiply VP to make such solutions uncompetitive. This enforces the
    # hydraulic constraint more strictly than soft penalties alone.
    low_violation_count = np.sum(np.abs(pipe_velocities) < feasible_min_velocity)
    high_violation_count = 0
    if feasible_max_velocity is not None:
        high_violation_count = np.sum(np.abs(pipe_velocities) > feasible_max_velocity)
    if low_violation_count > 0 or high_violation_count > 0:
        VP *= velocity_hard_multiplier

    return PP, VP


def apply_penalty_scale(PP: float, VP: float, scale: float = 1.0) -> tuple[float, float]:
    """
    Scale penalties around 1.0 for temperature-based relaxation.

    scale=1.0 keeps penalties unchanged.
    scale<1.0 relaxes penalties (useful at high temperature).
    scale>1.0 strengthens penalties.
    """
    if scale < 0:
        scale = 0.0
    return 1.0 + (PP - 1.0) * scale, 1.0 + (VP - 1.0) * scale


def fitness(cost: float, PP: float, VP: float) -> float:
    """
    Fitness = 1 / (cost × PP × VP).

    Higher fitness = better solution.
    """
    denom = cost * PP * VP
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def is_feasible(nodal_pressures: np.ndarray,
                min_pressure: float,
                pipe_velocities: np.ndarray | None = None,
                min_velocity: float | None = None,
                max_velocity: float | None = None) -> bool:
    """
    Check hydraulic feasibility.

    Pressure feasibility is always required.
    If velocity bounds are provided, enforce them too.
    """
    pressure_ok = np.all(nodal_pressures >= min_pressure - 1e-3)
    if not pressure_ok:
        return False

    if pipe_velocities is None or min_velocity is None or max_velocity is None:
        return True

    v = np.abs(np.asarray(pipe_velocities, dtype=float))
    return bool(np.all(v >= min_velocity) and np.all(v <= max_velocity))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def bits_needed(num_diameters: int) -> int:
    """Number of bits needed to encode `num_diameters` values."""
    if num_diameters <= 1:
        return 1
    return int(np.ceil(np.log2(num_diameters)))
