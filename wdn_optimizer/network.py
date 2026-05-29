"""
Benchmark WDN network definitions.

Builds WNTR WaterNetworkModel objects for three standard benchmarks, with
all data taken directly from the Appendix tables of Sangroula et al. (2022).

Design decision: we construct the network models programmatically rather
than shipping .inp files, so the project is fully self-contained and every
parameter is traceable to the source paper.
"""

import wntr


# ═══════════════════════════════════════════════════════════════════════════
# Data class holding everything the optimizer needs
# ═══════════════════════════════════════════════════════════════════════════

class WDNProblem:
    """
    Full specification of a WDN optimization problem.

    Attributes
    ----------
    name : str
        Benchmark name.
    wn : wntr.network.WaterNetworkModel
        The WNTR network *template* (diameters will be overwritten per eval).
    pipe_names : list[str]
        Ordered pipe names (defines the solution vector indexing).
    available_diameters : list[float]
        Commercial diameters in metres (WNTR uses SI).
    diameter_costs : list[float]
        Cost per metre for each diameter (local currency).
    pipe_lengths : list[float]
        Length of each pipe in metres.
    min_pressure : float
        Minimum allowable pressure at demand nodes (m).
    reservoir_id : str
        Name of the reservoir node.
    known_best_cost : float
        Best feasible cost from the literature.
    """

    def __init__(self, name, wn, pipe_names, available_diameters,
                 diameter_costs, pipe_lengths, min_pressure,
                 reservoir_id, known_best_cost,
                 min_velocity=0.3, max_velocity=2.0):
        self.name = name
        self.wn = wn
        self.pipe_names = pipe_names
        self.available_diameters = available_diameters
        self.diameter_costs = diameter_costs
        self.pipe_lengths = pipe_lengths
        self.min_pressure = min_pressure
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.reservoir_id = reservoir_id
        self.known_best_cost = known_best_cost
        self.num_pipes = len(pipe_names)
        self.num_diameters = len(available_diameters)


# ═══════════════════════════════════════════════════════════════════════════
# 1) Two-Loop Network
# ═══════════════════════════════════════════════════════════════════════════
# 7 nodes, 8 pipes, 8 diameters → search space ~1.67×10^7
# All pipes 1000 m, Hazen-Williams C = 130
# Reservoir at node 1, head = 210 m
# Minimum pressure 30 m at all demand nodes
# 
# ═══════════════════════════════════════════════════════════════════════════

def two_loop_network() -> WDNProblem:
    wn = wntr.network.WaterNetworkModel()

    # --- Reservoir ---
    wn.add_reservoir('R1', base_head=210.0, coordinates=(0, 200))

    # --- Junctions (demand nodes) ---
    #   Node: elevation(m), demand(m³/h → converted to m³/s)
    node_data = [
        ('J2', 150.0, 100.0),
        ('J3', 160.0, 100.0),
        ('J4', 155.0, 120.0),
        ('J5', 150.0, 270.0),
        ('J6', 165.0, 330.0),
        ('J7', 160.0, 200.0),
    ]
    for nid, elev, demand_m3h in node_data:
        wn.add_junction(nid, base_demand=demand_m3h / 3600.0,
                        elevation=elev, coordinates=(0, 0))

    # --- Pipes ---
    #   (pipe_id, start_node, end_node, length, roughness)
    pipe_data = [
        ('P1', 'R1', 'J2', 1000.0, 130.0),
        ('P2', 'J2', 'J3', 1000.0, 130.0),
        ('P3', 'J2', 'J4', 1000.0, 130.0),
        ('P4', 'J4', 'J5', 1000.0, 130.0),
        ('P5', 'J4', 'J6', 1000.0, 130.0),
        ('P6', 'J6', 'J7', 1000.0, 130.0),
        ('P7', 'J3', 'J5', 1000.0, 130.0),
        ('P8', 'J5', 'J7', 1000.0, 130.0),
    ]
    pipe_names = []
    pipe_lengths = []
    for pid, sn, en, length, roughness in pipe_data:
        wn.add_pipe(pid, sn, en, length=length, diameter=0.3048,
                    roughness=roughness)
        pipe_names.append(pid)
        pipe_lengths.append(length)

    # --- Available diameters & costs  ---
    # Table 4 specifically defines 8 exact pipeline size options for this benchmark:
    diameters_in = [1, 2, 4, 6, 10, 14, 16, 18]
    costs_per_m = [2.0, 5.0, 11.0, 16.0, 32.0, 60.0, 90.0, 130.0]

    diameters_m = [d * 0.0254 for d in diameters_in]

    wn.options.hydraulic.headloss = 'H-W'
    wn.options.time.duration = 0  # steady-state

    return WDNProblem(
        name='Two-Loop',
        wn=wn,
        pipe_names=pipe_names,
        available_diameters=diameters_m,
        diameter_costs=costs_per_m,
        pipe_lengths=pipe_lengths,
        min_pressure=30.0,
        min_velocity=0.3,
        max_velocity=2.0,
        reservoir_id='R1',
        known_best_cost=419_000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2) Hanoi Network  (Fujiwara & Kang, 1990)
# ═══════════════════════════════════════════════════════════════════════════
# 32 nodes (node 1 = reservoir), 34 pipes, 6 diameters
# Search space ~2.86×10^26
# Reservoir head = 100 m, all node elevations 0 m
# Minimum pressure 30 m
# Hazen-Williams C = 130 for all pipes
# Paper reference: Tables A3–A4, Table 7
# ═══════════════════════════════════════════════════════════════════════════

def hanoi_network() -> WDNProblem:
    wn = wntr.network.WaterNetworkModel()

    # --- Reservoir (node 1) ---
    wn.add_reservoir('R1', base_head=100.0, coordinates=(0, 0))

    # --- Demands (m³/h) for nodes 2–32 (Table A3) ---
    demands_m3h = [
        890, 850, 130, 725, 1005, 1350, 550, 525, 525,
        500, 560, 940, 615, 280, 310, 865, 1345, 60,
        1275, 930, 485, 1045, 820, 170, 900, 370, 290,
        360, 360, 105, 805,
    ]
    for i, dem in enumerate(demands_m3h, start=2):
        wn.add_junction(f'J{i}', base_demand=dem / 3600.0,
                        elevation=0.0, coordinates=(i, 0))

    # --- Pipes (Table A4) ---
    pipe_connections = [
        (1, 2, 100),    (2, 3, 1350),   (3, 4, 900),    (4, 5, 1150),
        (5, 6, 1450),   (6, 7, 450),    (7, 8, 850),    (8, 9, 850),
        (9, 10, 800),   (10, 11, 950),  (11, 12, 1200), (12, 13, 3500),
        (10, 14, 800),  (14, 15, 500),  (15, 16, 550),  (17, 16, 2730),
        (18, 17, 1750), (19, 18, 800),  (3, 19, 400),   (3, 20, 2200),
        (20, 21, 1500), (21, 22, 500),  (20, 23, 2650), (23, 24, 1230),
        (24, 25, 1300), (26, 25, 850),  (27, 26, 300),  (16, 27, 750),
        (23, 28, 1500), (28, 29, 2000), (29, 30, 1600), (30, 31, 150),
        (32, 31, 860),  (25, 32, 950),
    ]
    pipe_names = []
    pipe_lengths = []
    for idx, (sn, en, length) in enumerate(pipe_connections, start=1):
        sn_id = 'R1' if sn == 1 else f'J{sn}'
        en_id = f'J{en}'
        pid = f'P{idx}'
        wn.add_pipe(pid, sn_id, en_id, length=float(length),
                    diameter=0.3048, roughness=130.0)
        pipe_names.append(pid)
        pipe_lengths.append(float(length))

    # --- Available diameters & costs (Table 7) ---
    diameters_mm = [304.8, 406.4, 508.0, 609.6, 762.0, 1016.0]
    costs_per_m  = [45.73, 70.40, 98.38, 129.33, 180.75, 278.28]
    diameters_m  = [d / 1000.0 for d in diameters_mm]

    wn.options.hydraulic.headloss = 'H-W'
    wn.options.time.duration = 0

    return WDNProblem(
        name='Hanoi',
        wn=wn,
        pipe_names=pipe_names,
        available_diameters=diameters_m,
        diameter_costs=costs_per_m,
        pipe_lengths=pipe_lengths, 
        min_pressure=30.0,
        min_velocity=0.0,
        max_velocity=7.0,
        reservoir_id='R1',
        known_best_cost=6_081_000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3) GoYang Network 
# ═══════════════════════════════════════════════════════════════════════════
# 22 nodes (node 1 = reservoir with pump), 30 pipes, 8 diameters
# Search space ~1.23×10^27
# Paper reports: reservoir at 71 m with a fixed 4.52 kW pump.
# This implementation uses an equivalent constant source head for steady-state
# diameter optimization, because detailed pump curve/power controls are not
# provided in the benchmark appendix tables.
# Minimum pressure 15 m above ground level
# 
# Paper reference: 
# ═══════════════════════════════════════════════════════════════════════════

GOYANG_PAPER_RESERVOIR_HEAD_M = 71.0
GOYANG_EQUIVALENT_SOURCE_HEAD_M = 86.0

def goyang_network() -> WDNProblem:
    wn = wntr.network.WaterNetworkModel()

    # --- Reservoir ---
    wn.add_reservoir('R1', base_head=GOYANG_EQUIVALENT_SOURCE_HEAD_M,
                     coordinates=(0, 0))

    # --- Junctions (nodes 2–22) with elevations and demands ---
    # Demand is in m³/d → convert to m³/s.
    # Values are aligned with the project paper extraction and the reference
    # EPANET model in temp.inp.

    node_data = [
        # (node_id_paper, elevation, demand_m3_per_day)
        (2,  56.4,  153.0),
        (3,  53.8,   70.5),
        (4,  54.9,   58.5),
        (5,  56.0,   75.0),
        (6,  57.0,   67.5),
        (7,  53.9,   63.0),
        (8,  54.5,   48.0),
        (9,  57.9,   42.0),
        (10, 62.1,   30.0),
        (11, 62.8,   42.0),
        (12, 58.6,   37.5),
        (13, 59.3,   37.5),
        (14, 59.8,   63.0),
        (15, 59.2,  445.5),
        (16, 53.6,  108.0),
        (17, 54.8,   79.5),
        (18, 55.1,   55.5),
        (19, 54.2,  118.5),
        (20, 54.5,  124.5),
        (21, 62.9,   31.5),
    ]
    for nid, elev, dem_d in node_data:
        wn.add_junction(f'J{nid}', base_demand=dem_d / 86400.0,
                        elevation=elev, coordinates=(nid, 0))
    # Node 22 is present in the link table and in the reference INP.
    wn.add_junction('J22', base_demand=31.5 / 86400.0,
                    elevation=55.0, coordinates=(22, 0))

    # NOTE: The paper reservoir level is 71 m and includes a fixed pump

    # --- Pipes (Table A6) ---
    pipe_connections = [
        (1, 2, 165),    (2, 3, 124),    (3, 4, 118),     (4, 5, 81),
        (5, 6, 134),    (6, 12, 135),   (12, 15, 202),   (2, 22, 135),
        (2, 21, 170),   (21, 22, 113),  (22, 20, 335),   (20, 19, 115),
        (2, 19, 345),   (19, 17, 114),  (3, 16, 103),    (16, 17, 261),
        (17, 18, 72),   (7, 18, 373),   (3, 7, 98),      (7, 8, 110),
        (4, 8, 98),     (8, 9, 246),    (5, 11, 174),    (10, 11, 102),
        (6, 10, 92),    (6, 9, 100),    (10, 13, 130),   (12, 13, 90),
        (13, 14, 185),  (15, 14, 90),
    ]

    pipe_names = []
    pipe_lengths = []
    for idx, (sn, en, length) in enumerate(pipe_connections, start=1):
        sn_id = 'R1' if sn == 1 else f'J{sn}'
        en_id = f'J{en}'
        pid = f'P{idx}'
        wn.add_pipe(pid, sn_id, en_id, length=float(length),
                    diameter=0.2, roughness=100.0)
        pipe_names.append(pid)
        pipe_lengths.append(float(length))

    # --- Available diameters & costs ---
    diameters_mm = [80, 100, 125, 150, 200, 250, 300, 350]
    costs_per_m  = [37890, 38933, 40563, 42554, 47624, 54125, 62109, 71524]
    diameters_m  = [d / 1000.0 for d in diameters_mm]

    wn.options.hydraulic.headloss = 'H-W'
    wn.options.time.duration = 0

    return WDNProblem(
        name='GoYang',
        wn=wn,
        pipe_names=pipe_names,
        available_diameters=diameters_m,
        diameter_costs=costs_per_m,
        pipe_lengths=pipe_lengths,
        min_pressure=15.0,
        min_velocity=0.0,
        max_velocity=2.0,
        reservoir_id='R1',
        known_best_cost=177_010_359.0,
    )


def get_all_problems() -> list[WDNProblem]:
    """Return all three benchmark problems."""
    return [two_loop_network(), hanoi_network(), goyang_network()]


def validate_problem(problem: WDNProblem) -> tuple[bool, list[str]]:
    """
    Validate internal consistency of a benchmark definition.

    Returns
    -------
    (ok, errors)
    """
    errors = []

    if len(problem.pipe_names) != len(problem.pipe_lengths):
        errors.append("pipe_names length does not match pipe_lengths length")

    if len(problem.available_diameters) != len(problem.diameter_costs):
        errors.append("available_diameters length does not match diameter_costs length")

    if any(d <= 0 for d in problem.available_diameters):
        errors.append("available_diameters contains non-positive values")

    if any(c <= 0 for c in problem.diameter_costs):
        errors.append("diameter_costs contains non-positive values")

    if any(l <= 0 for l in problem.pipe_lengths):
        errors.append("pipe_lengths contains non-positive values")

    # For benchmark data from the literature, diameters and costs should be nondecreasing.
    if any(problem.available_diameters[i] > problem.available_diameters[i + 1]
           for i in range(len(problem.available_diameters) - 1)):
        errors.append("available_diameters are not nondecreasing")

    if any(problem.diameter_costs[i] > problem.diameter_costs[i + 1]
           for i in range(len(problem.diameter_costs) - 1)):
        errors.append("diameter_costs are not nondecreasing")

    return len(errors) == 0, errors
