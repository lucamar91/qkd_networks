"""
test_complex_network.py

Implements Appendix A.2 "Complex Network Validation":

    - Build a pruned N=1000 node network graph (dmax ~ 130 km cutoff).
    - Sample 100 random source-target pairs.
    - For each pair, use Yen's algorithm to find the k=20 shortest paths
      under the surrogate Dijkstra weight.
    - Run the full physical simulation (calculate_metrics) on all 20
      candidates to find the TRUE best SKR.
    - optimality_ratio = SKR(surrogate-selected path) / SKR(true best path)
    - Plot a histogram of optimality_ratio over the 100 pairs.

Run:
    python test_complex_network.py

Requires repeaters_NV.py on PYTHONPATH. If your real S2-graph builder
(network_funcs.S2_graph_definite_N) isn't importable in this environment,
the script falls back to a synthetic random-geometric graph SOLELY so the
pipeline can be exercised end to end -- swap in your real build_s2_graph(...)
call (see USE_REAL_GRAPH below) to get results that actually match your
thesis.
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork, build_s2_graph, build_density_graph

# ---------------------------------------------------------------------------
# 1. Parameters -- SAME CAVEAT AS test_routing_scenarios.py:
#    these are placeholders, replace with your real thesis values.
# ---------------------------------------------------------------------------
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
params.M         = 20

ARCHITECTURE      = 'node'          # <-- match your real setup ('node' or 'midpoint')
MULTIPLEXING_TYPE = 'accumulated'           # <-- e.g. 'accumulated' if you use M=20 multiplexing
DISTILLATION      = False
DISTILLATION_TIME = 'before_swap'

N_NODES   = 1000
DMAX_KM   = 128.0
N_PAIRS   = 100
K_PATHS   = 20
RNG_SEED  = 0

USE_REAL_GRAPH = True   # set True once network_funcs is available in your environment

# ---------------------------------------------------------------------------
# 2. Build the network
# ---------------------------------------------------------------------------

def build_network():
    if USE_REAL_GRAPH:
        try:
            A, dist, coords, scale_km = build_density_graph(N=N_NODES, rho=1, beta=2.6261, mu=0.0233)
        except Exception as e:
            raise RuntimeError(
                "Could not build the real S2 graph (network_funcs missing?). "
                "Set USE_REAL_GRAPH = False to use the synthetic fallback "
                "for pipeline testing, or make network_funcs importable."
            ) from e
    else:
        # ---- SYNTHETIC FALLBACK: for exercising the pipeline only ----
        rng = np.random.default_rng(RNG_SEED)
        coords2d = rng.uniform(0, 31, size=(N_NODES, 2))  # ~ 'city' scale, 31 km box
        G = nx.random_geometric_graph(N_NODES, radius=4.0, seed=RNG_SEED, pos={i: coords2d[i] for i in range(N_NODES)})
        A = nx.to_numpy_array(G)
        dist = np.zeros((N_NODES, N_NODES))
        for i, j in G.edges():
            d = np.linalg.norm(coords2d[i] - coords2d[j])
            dist[i, j] = dist[j, i] = d
        coords = None

    net = QuantumRepeaterNetwork(
        params, A, dist, coords=coords,
        architecture=ARCHITECTURE, scale=scale_km,
        distillation=DISTILLATION, multiplexing_type=MULTIPLEXING_TYPE,
        distillation_time=DISTILLATION_TIME,
    )
    return net


def prune_by_distance(net, dmax):
    import copy
    net2 = copy.copy(net)
    A2 = net.A.copy()
    A2[net.dist > dmax] = 0
    net2.A = A2
    net2.Probs_mtx, net2.P_click, net2.Probs_raw = net2._build_prob_matrix()
    return net2


# ---------------------------------------------------------------------------
# 3. Surrogate-weight graph (same formula as QuantumRepeaterNetwork.optimal_path)
# ---------------------------------------------------------------------------

def build_weight_graph(net):
    p = net.params
    W = np.zeros_like(net.Probs_mtx)
    nz = net.Probs_mtx > 0
    W[nz] = -np.log2(net.Probs_mtx[nz]) - np.log2(p.P_BSM)
    return nx.from_numpy_array(W)


# ---------------------------------------------------------------------------
# 4. Yen's k-shortest-paths + physical evaluation, per source-target pair
# ---------------------------------------------------------------------------

def evaluate_pair(net, G, source, target, k=K_PATHS):
    """
    Returns dict with SKR of the surrogate-selected path (rank 0) vs the
    true best SKR among the top-k surrogate candidates, or None if the
    pair is unreachable.
    """
    if not nx.has_path(G, source, target):
        return None

    try:
        path_gen = nx.shortest_simple_paths(G, source, target, weight='weight')
        candidates = list(itertools.islice(path_gen, k))
    except nx.NetworkXNoPath:
        return None

    skrs = []
    for path in candidates:
        try:
            _, _, SKR, _ = net.calculate_metrics(path)
        except Exception:
            SKR = 0.0
        skrs.append(SKR)

    skrs = np.array(skrs)
    skr_selected = skrs[0]                 # rank-0 = what the surrogate metric picks
    skr_true_best = skrs.max()

    return {
        "source": source,
        "target": target,
        "n_candidates": len(candidates),
        "SKR_selected": skr_selected,
        "SKR_true_best": skr_true_best,
    }


# ---------------------------------------------------------------------------
# 5. Run the statistical test
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    net = build_network()
    net = prune_by_distance(net, DMAX_KM)
    G = build_weight_graph(net)

    N = net.A.shape[0]
    results = []
    attempts = 0
    while len(results) < N_PAIRS and attempts < N_PAIRS * 20:
        attempts += 1
        s, t = rng.integers(0, N, size=2)
        if s == t:
            continue
        r = evaluate_pair(net, G, int(s), int(t))
        if r is not None:
            results.append(r)

    df = pd.DataFrame(results)

    # optimality ratio: only defined where the true best path has SKR > 0
    valid = df["SKR_true_best"] > 0
    df["optimality_ratio"] = np.nan
    df.loc[valid, "optimality_ratio"] = (
        df.loc[valid, "SKR_selected"] / df.loc[valid, "SKR_true_best"]
    )

    n_no_viable_path = int((~valid).sum())
    n_valid = int(valid.sum())

    print(f"Pairs sampled:                 {len(df)}")
    print(f"Pairs with no viable SKR>0 path among candidates: {n_no_viable_path}")
    print(f"Pairs used for optimality ratio: {n_valid}")
    if n_valid > 0:
        ratios = df.loc[valid, "optimality_ratio"]
        print(f"Mean optimality ratio:          {ratios.mean():.4f}")
        print(f"Median optimality ratio:        {ratios.median():.4f}")
        print(f"Fraction exactly optimal (=1):  {(ratios >= 0.9999).mean():.4f}")
        print(f"Fraction ratio >= 0.9:          {(ratios >= 0.9).mean():.4f}")

    df.to_csv("complex_network_results.csv", index=False)
    print("\nSaved per-pair results to complex_network_results.csv")

    if n_valid > 0:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df.loc[valid, "optimality_ratio"], bins=20, range=(0, 1),
                  color="#3b6ea5", edgecolor="white")
        plt.xlabel("Optimality ratio  (SKR$_{selected}$ / SKR$_{true\\ best}$)")
        plt.ylabel("Number of source-target pairs")
        plt.title(f"Surrogate routing metric optimality ratio (N={N} nodes, k={K_PATHS})")
        plt.tight_layout()
        plt.savefig("optimality_ratio_histogram.png", dpi=200)
        print("Saved histogram to optimality_ratio_histogram.png")

    return df


if __name__ == "__main__":
    main()