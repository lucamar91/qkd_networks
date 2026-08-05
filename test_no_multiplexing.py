"""
test_routing_scenarios_no_multiplexing.py

Same validation approach as test_routing_scenarios.py, but for the
NO-MULTIPLEXING case (multiplexing_type=None) with dmax = 111 km and the
following predicted Dijkstra-vs-physics clash:

    Scenario 1 (Baseline, 90 km):
        1A: 1 hop x 90 km
        1B: 2 hops x 45 km
        Predicted: Dijkstra picks 1A, physics picks 1B  (MISMATCH)

    Scenario 2 (Short Detour):
        2A: 2 hops x 50 km  (100 km total)
        2B: 3 hops x 35 km  (105 km total)
        Predicted: Dijkstra picks 2A, physics likely picks 2B  (MISMATCH)

    Scenario 3 (Massive Detour):
        3A: 2 hops x 100 km (200 km total)
        3B: 4 hops x 35 km  (140 km total)
        Predicted: distance penalty finally wins -> Dijkstra picks 3B,
                   which is also physically optimal  (MATCH)

    Scenario 4 (Unpruned -> Pruned, dmax = 111 km):
        4A: 1 hop x 180 km
        4B: 4 hops x 45 km
        Predicted (unpruned): Dijkstra picks 4A, SKR ~ 0 (dark-count noise)
        Predicted (pruned):   4A removed, Dijkstra forced onto 4B

Unlike test_routing_scenarios.py (which asserted the surrogate metric gets
things right), this script asserts the *predicted mismatches* -- i.e. it
checks that the clash actually shows up in the physical simulation, which
is the point of this particular test.

Run:
    python test_routing_scenarios_no_multiplexing.py

Requires repeaters_NV.py importable (same folder / PYTHONPATH).
"""

import copy
import numpy as np
import pandas as pd

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# 1. Baseline parameters -- NO MULTIPLEXING (multiplexing_type=None, as below)
# ---------------------------------------------------------------------------
# Same caveat as before: adjust these to your real thesis values. Only the
# distances and dmax were specified for this test; everything else here is
# still a placeholder split of the ~10% baseline hardware efficiency etc.
params = RepeaterParams()
params.p_det     = 0.95
params.alpha     = 0.18
params.V         = 0.95
params.nu        = 167e3
params.R_dark    = 100
params.delta_det = 100e-12
params.eta_c     = 0.8
params.P_BSM     = 0.9993
params.eta_M     = 0.46
params.T_coh     = 0.05
params.M         = None       # not used, no multiplexing

ARCHITECTURE      = 'node'    # <-- match your real setup
MULTIPLEXING_TYPE = None      # no multiplexing, as requested
DISTILLATION      = False
DISTILLATION_TIME = 'before_swap'

DMAX = 111.0  # km, new pruning threshold for this test


def prune_by_distance(net, dmax):
    """Remove edges longer than dmax [km] and rebuild probability matrices."""
    net2 = copy.copy(net)
    A2 = net.A.copy()
    A2[net.dist > dmax] = 0
    net2.A = A2
    net2.Probs_mtx, net2.P_click, net2.Probs_raw = net2._build_prob_matrix()
    return net2


def dijkstra_weight(net, path):
    """Surrogate Dijkstra weight of an arbitrary path: sum_i -log2(P_i) - log2(P_BSM)."""
    P_BSM = net.params.P_BSM
    w = 0.0
    for a, b in zip(path[:-1], path[1:]):
        p_link = net.Probs_mtx[a, b]
        if p_link <= 0:
            return np.inf
        w += -np.log2(p_link) - np.log2(P_BSM)
    return w


# ---------------------------------------------------------------------------
# 2. Graph builder: several candidate paths sharing one source and target
# ---------------------------------------------------------------------------

def build_scenario_graph(candidate_hop_lengths):
    SOURCE, TARGET = 0, 1
    edges = {}
    paths = []
    next_id = 2

    for hop_lengths in candidate_hop_lengths:
        cur = SOURCE
        nodes = [SOURCE]
        for i, length in enumerate(hop_lengths):
            nxt = TARGET if i == len(hop_lengths) - 1 else next_id
            if nxt == next_id:
                next_id += 1
            edges[(cur, nxt)] = length
            nodes.append(nxt)
            cur = nxt
        paths.append(nodes)

    n_nodes = next_id
    A = np.zeros((n_nodes, n_nodes))
    dist = np.zeros((n_nodes, n_nodes))
    for (u, v), d in edges.items():
        A[u, v] = A[v, u] = 1
        dist[u, v] = dist[v, u] = d

    return A, dist, paths


# ---------------------------------------------------------------------------
# 3. Run one scenario
# ---------------------------------------------------------------------------

def run_scenario(scenario_label, candidate_hop_lengths, params, dmax=None):
    A, dist, paths = build_scenario_graph(candidate_hop_lengths)
    net = QuantumRepeaterNetwork(
        params, A, dist, coords=None,
        architecture=ARCHITECTURE, scale='city',
        distillation=DISTILLATION, multiplexing_type=MULTIPLEXING_TYPE,
        distillation_time=DISTILLATION_TIME,
    )
    if dmax is not None:
        net = prune_by_distance(net, dmax)

    try:
        best_weight, best_path = net.optimal_path(source=0, target=1)
        selected_path = best_path
    except Exception:
        best_weight, selected_path = np.inf, None

    rows = []
    for path in paths:
        w = dijkstra_weight(net, path)
        if np.isinf(w):
            R, Q, SKR, F = 0.0, 0.5, 0.0, 0.25
        else:
            R, Q, SKR, F = net.calculate_metrics(path)
        rows.append({
            "Scenario":        scenario_label,
            "Path (node ids)": path,
            "Hops":            len(path) - 1,
            "Lengths (km)":    [round(float(net.dist[a, b]), 2) for a, b in zip(path[:-1], path[1:])],
            "Total Dist (km)": round(float(sum(net.dist[a, b] for a, b in zip(path[:-1], path[1:]))), 2),
            "Dijkstra Weight": w,
            "Fidelity":        F,
            "SKR (bit/s)":     SKR,
            "Selected?":       (selected_path is not None and path == selected_path),
        })

    df = pd.DataFrame(rows)
    max_skr = df["SKR (bit/s)"].max()
    df["Optimal SKR?"] = np.isclose(df["SKR (bit/s)"], max_skr) & (max_skr > 0)
    return df


# ---------------------------------------------------------------------------
# 4. Scenarios exactly as specified for the no-multiplexing test
# ---------------------------------------------------------------------------

scenario_defs = {
    "1: Baseline (90 km)": {
        "candidates": [
            [90],         # 1A - 1 hop, 90 km
            [45, 45],     # 1B - 2 hops, 90 km
        ],
        "dmax": None,
    },
    "2: Short Detour": {
        "candidates": [
            [50, 50],       # 2A - 2 hops, 100 km
            [35, 35, 35],   # 2B - 3 hops, 105 km
        ],
        "dmax": None,
    },
    "3: Massive Detour": {
        "candidates": [
            [100, 100],     # 3A - 2 hops, 200 km
            [35] * 4,       # 3B - 4 hops, 140 km
        ],
        "dmax": None,
    },
    "4: Unpruned": {
        "candidates": [
            [180],        # 4A - 1 hop, 180 km
            [45] * 4,     # 4B - 4 hops, 180 km
        ],
        "dmax": None,
    },
    "4: Pruned (dmax <= 111)": {
        "candidates": [
            [180],        # 4A - should be removed by pruning
            [45] * 4,     # 4B - should be forced/selected
        ],
        "dmax": DMAX,
    },
}

results = []
for label, spec in scenario_defs.items():
    df_s = run_scenario(label, spec["candidates"], params, dmax=spec["dmax"])
    results.append(df_s)

full_results = pd.concat(results, ignore_index=True)

# ---------------------------------------------------------------------------
# 5. Assertions -- check the PREDICTED clash, not "correctness"
# ---------------------------------------------------------------------------

def get(df, scenario, hops):
    row = df[(df["Scenario"] == scenario) & (df["Hops"] == hops)]
    assert len(row) == 1, f"expected exactly one match for {scenario}, hops={hops}"
    return row.iloc[0]

print("\n=== Checking predicted Dijkstra-vs-physics behaviour (no multiplexing) ===\n")
checks = []

# Scenario 1: predict Dijkstra picks 1A, but physics prefers 1B -> mismatch
r1a = get(full_results, "1: Baseline (90 km)", 1)
r1b = get(full_results, "1: Baseline (90 km)", 2)
checks.append(("1A selected by Dijkstra", bool(r1a["Selected?"])))
checks.append(("1B is physically optimal (not 1A)", bool(r1b["Optimal SKR?"]) and not bool(r1a["Optimal SKR?"])))
checks.append(("Scenario 1 clash confirmed (Dijkstra != physics)", bool(r1a["Selected?"]) and bool(r1b["Optimal SKR?"])))

# Scenario 2: predict Dijkstra picks 2A, physics likely prefers 2B -> mismatch
r2a = get(full_results, "2: Short Detour", 2)
r2b = get(full_results, "2: Short Detour", 3)
checks.append(("2A selected by Dijkstra", bool(r2a["Selected?"])))
checks.append(("2B is physically optimal (not 2A)", bool(r2b["Optimal SKR?"]) and not bool(r2a["Optimal SKR?"])))

# Scenario 3: predict distance penalty wins -> Dijkstra picks 3B, which is also optimal -> match
r3b = get(full_results, "3: Massive Detour", 4)
checks.append(("3B selected by Dijkstra (distance penalty finally dominates)", bool(r3b["Selected?"])))
checks.append(("3B is physically optimal", bool(r3b["Optimal SKR?"])))

# Scenario 4 unpruned: predict 4A selected despite ~0 SKR
r4a_un = get(full_results, "4: Unpruned", 1)
checks.append(("4A selected despite noise-dominated SKR (unpruned)", bool(r4a_un["Selected?"])))
checks.append(("4A SKR ~ 0 (unpruned)", r4a_un["SKR (bit/s)"] < 1e-3))

# Scenario 4 pruned: predict 4A removed, 4B forced/selected
r4a_pr = get(full_results, "4: Pruned (dmax <= 111)", 1)
r4b_pr = get(full_results, "4: Pruned (dmax <= 111)", 4)
checks.append(("4A not selected after pruning", not bool(r4a_pr["Selected?"])))
checks.append(("4B selected after pruning", bool(r4b_pr["Selected?"])))

all_passed = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_passed &= ok

print(f"\n{'ALL PREDICTIONS CONFIRMED' if all_passed else 'SOME PREDICTIONS DID NOT HOLD -- inspect table below'}\n")

pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print(full_results.to_string(index=False))

full_results.to_csv("routing_scenario_results_no_multiplexing.csv", index=False)
print("\nSaved full results to routing_scenario_results_no_multiplexing.csv")