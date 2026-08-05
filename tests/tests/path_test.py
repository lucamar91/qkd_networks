"""
test_routing_scenarios.py

Validates the surrogate Dijkstra routing metric (Section 3.3 / Appendix A.1
of the proposal) against the full physical simulation, for the four
artificial "toy model" scenarios described in Table 1:

    1. Baseline        - strict hop penalisation over a fixed 100 km distance
    2. Short Detour     - is a slightly-longer 2-hop path preferred over a
                           3-hop path of similar distance?
    3. Massive Detour   - does distance penalty eventually beat hop penalty?
    4. Pruning          - does the unpruned graph erroneously pick a
                           noise-dominated 250 km direct link, and does the
                           dmax = 130 km pruning threshold rescue it?

For every candidate path in every scenario we compute:
    - the surrogate Dijkstra weight (sum of -log2(P_link) - log2(P_BSM))
    - the true physical Fidelity / SKR via QuantumRepeaterNetwork.calculate_metrics
and then check whether the path the Dijkstra algorithm *would* select
(net.optimal_path) is also the path with the highest physical SKR.

Run:
    python test_routing_scenarios.py

Requires repeaters_NV.py to be importable (same folder, or on PYTHONPATH).
"""

import copy
import numpy as np
import pandas as pd

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# 1. Baseline parameters
# ---------------------------------------------------------------------------
# NOTE: the proposal specifies alpha = 0.2 dB/km and P_BSM = 0.9993 exactly,
# plus a combined "baseline hardware efficiency of 10%". That 10% is the
# product of several per-link factors in this code (p_det * eta_c * eta_M),
# so there isn't a single unique way to split it. Adjust p_det / eta_c /
# eta_M below to whatever split you actually used in the thesis -- the
# split matters for QBER (dark-count ratio) even though it doesn't change
# the raw click probability much.
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
params.M         = 20         # matches "M = 20 multiplexing" in the proposal

DMAX = 130.0  # km, pruning threshold from the proposal


def make_network(params):
    """Fresh QuantumRepeaterNetwork with no distillation / no multiplexing."""
    def _dummy(A, dist):
        return QuantumRepeaterNetwork(
            params, A, dist, coords=None,
            architecture='node', scale='city',
            distillation=False, multiplexing_type='accumulated',
        )
    return _dummy


def prune_by_distance(net, dmax):
    """
    Remove every edge longer than dmax [km] and rebuild the probability
    matrices. Direct stand-in for the dmax pruning step described in the
    proposal (which is expressed there via a k_min elementary-link-rate
    cutoff -- pruning by distance directly is equivalent for these toy
    single-length-class scenarios and is far more transparent to test).
    """
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
    """
    Build a small graph containing several parallel chains between a shared
    source (node 0) and target (node 1), one chain per entry of
    candidate_hop_lengths (a list of lists of hop distances in km).

    Returns
    -------
    A, dist   : adjacency / distance matrices
    paths     : list of node-index paths, one per candidate, in the same
                order as candidate_hop_lengths
    """
    SOURCE, TARGET = 0, 1
    edges = {}          # (u, v) -> distance
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
# 3. Run one scenario: evaluate every candidate path, compare to Dijkstra's pick
# ---------------------------------------------------------------------------

def run_scenario(scenario_label, candidate_hop_lengths, params, dmax=None):
    """
    Parameters
    ----------
    scenario_label : str            e.g. "1: Baseline"
    candidate_hop_lengths : list of list of float
                                     one entry per candidate path, e.g.
                                     [[100], [50, 50], [20]*5]
    dmax : float or None            if given, prune edges longer than dmax first

    Returns
    -------
    pd.DataFrame  one row per candidate path
    """
    A, dist, paths = build_scenario_graph(candidate_hop_lengths)
    net = QuantumRepeaterNetwork(
        params, A, dist, coords=None,
        architecture='node', scale='city',
        distillation=False, multiplexing_type='accumulated',
    )
    if dmax is not None:
        net = prune_by_distance(net, dmax)

    # What would the surrogate Dijkstra metric actually pick?
    try:
        best_weight, best_path = net.optimal_path(source=0, target=1)
        selected_path = best_path
    except Exception:
        # unreachable after pruning
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
    # physical optimum = highest SKR among candidates that are actually reachable
    max_skr = df["SKR (bit/s)"].max()
    df["Optimal SKR?"] = np.isclose(df["SKR (bit/s)"], max_skr) & (max_skr > 0)
    return df


# ---------------------------------------------------------------------------
# 4. Define the four scenarios exactly as in Table 1
# ---------------------------------------------------------------------------

scenario_defs = {
    "1: Baseline": {
        "candidates": [
            [100],          # 1A - 1 hop,  100 km
            [50, 50],       # 1B - 2 hops, 100 km
            [20] * 5,       # 1C - 5 hops, 100 km
        ],
        "dmax": None,
    },
    "2: Short Detour": {
        "candidates": [
            [60, 60],       # 2A - 2 hops, 120 km
            [35, 35, 35],   # 2B - 3 hops, 105 km
        ],
        "dmax": None,
    },
    "3: Massive Detour": {
        "candidates": [
            [120, 120],     # 3A - 2 hops, 240 km
            [40] * 4,       # 3B - 4 hops, 160 km
            [30] * 6,       # 3C - 6 hops, 180 km
        ],
        "dmax": None,
    },
    "4: Unpruned": {
        "candidates": [
            [250],          # 4A - 1 hop,  250 km
            [50] * 5,       # 4B - 5 hops, 250 km
        ],
        "dmax": None,
    },
    "4: Pruned (dmax <= 130)": {
        "candidates": [
            [250],          # 4A - 1 hop, 250 km  -> should be removed by pruning
            [50] * 5,       # 4B - 5 hops, 250 km -> should be "rescued"
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
# 5. Assertions -- turn the proposal's qualitative claims into pass/fail checks
# ---------------------------------------------------------------------------

def get(df, scenario, hops):
    row = df[(df["Scenario"] == scenario) & (df["Hops"] == hops)]
    assert len(row) == 1, f"expected exactly one match for {scenario}, hops={hops}"
    return row.iloc[0]

print("\n=== Running assertions against the proposal's claims ===\n")

checks = []

# Scenario 1: the single 100 km hop should be selected (Dijkstra) AND optimal
r = get(full_results, "1: Baseline", 1)
checks.append(("1A selected", bool(r["Selected?"])))
checks.append(("1A physically optimal", bool(r["Optimal SKR?"])))

# Scenario 2: 2A (2 hops, 120 km) should be selected over 2B (3 hops, 105 km)
r2a = get(full_results, "2: Short Detour", 2)
checks.append(("2A selected over 2B", bool(r2a["Selected?"])))

# Scenario 3: distance penalty should override hop penalty -> 3B (4 hops) selected,
# NOT 3A (2 hops, but 240 km)
r3a = get(full_results, "3: Massive Detour", 2)
r3b = get(full_results, "3: Massive Detour", 4)
checks.append(("3A (extreme 2-hop detour) NOT selected", not bool(r3a["Selected?"])))
checks.append(("3B (4-hop) selected", bool(r3b["Selected?"])))

# Scenario 4 unpruned: 4A (1 hop, 250 km) is erroneously selected despite SKR ~ 0
r4a_un = get(full_results, "4: Unpruned", 1)
checks.append(("4A selected despite being noise-dominated (unpruned)", bool(r4a_un["Selected?"])))
checks.append(("4A SKR ~ 0 (unpruned)", np.isclose(r4a_un["SKR (bit/s)"], 0.0, atol=1e-6)))

# Scenario 4 pruned: 4A should now be unreachable / not selected, 4B rescued
r4a_pr = get(full_results, "4: Pruned (dmax <= 130)", 1)
r4b_pr = get(full_results, "4: Pruned (dmax <= 130)", 5)
checks.append(("4A not selected after pruning", not bool(r4a_pr["Selected?"])))
checks.append(("4B selected after pruning (rescued)", bool(r4b_pr["Selected?"])))
checks.append(("4B is physically optimal after pruning", bool(r4b_pr["Optimal SKR?"])))

all_passed = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_passed &= ok

print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED -- inspect table below and adjust params'}\n")

pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print(full_results.to_string(index=False))

full_results.to_csv("routing_scenario_results.csv", index=False)
print("\nSaved full results to routing_scenario_results.csv")