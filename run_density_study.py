"""
run_density_study.py

Sweep node density for a quantum repeater network (S2 graph), for two
multiplexing settings (no mux vs. M=20 accumulated mux), computing
network-level SKR / QBER / Fidelity / reachability at each density.

Pruning (by minimum elementary-link rate) is applied BEFORE metrics are
computed, since it changes which paths Dijkstra can even choose -- this is
why we prune first and then run analyze_all_paths/network metrics on the
pruned graph.

Everything is written to a CSV so the (expensive) simulation only has to be
run once; plot_density_study.py reads that CSV and makes the figures.

Assumptions made (not specified in the request -- edit if wrong):
  * params.V (source/Barrett-Kok visibility) is not set anywhere in your
    example scripts but IS required by F_link()/calculate_metrics(). Set to
    0.9 below; change PARAM_V if you have a different design value.
  * BETA / MU (S2 graph parameters) reused from your percolation study
    script (2.6261 / 0.0233).
  * Network-level metrics (SKR, QBER, F, reachability) are averaged over a
    *sample* of source nodes per graph rather than all N=1000 sources.
    With N=1000, analyze_all_paths() runs one Dijkstra per source, so all
    N sources x 30 densities x 5 reps x 2 mux cases would mean ~300,000
    Dijkstra calls on 1000-node graphs -- likely hours of runtime. Sampling
    N_SOURCES=50 sources per graph gives a statistically stable estimate
    (same idea as your 5 repetitions) while keeping runtime reasonable.
    Set N_SOURCES = N to reproduce the exact network_metrics() behaviour
    from repeaters_NV.py if you have time to let it run.
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm  # pip install tqdm --break-system-packages  (or drop the wrapper)

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork, build_density_graph

# ---------------------------------------------------------------------------
# Config -- EDIT THESE
# ---------------------------------------------------------------------------

N        = 1000
BETA     = 2.6261
MU       = 0.0233
N_REPS   = 5
N_SOURCES = 50          # sampled sources per graph, see assumptions above

DENSITIES = np.logspace(-4, 0, 30)   # nodes / km^2

KMIN_NO_MUX = 1e3        # pairs/s, pruning threshold, no multiplexing
KMIN_MUX20  = 1e5        # pairs/s, pruning threshold, M = 20 multiplexing
M_MUX       = 20

OUTPUT_CSV = "density_results.csv"

# Physical link parameters -- EDIT to match your real NV setup
PARAM_V = 0.9   # visibility, see assumptions note above

params = RepeaterParams()
params.p_det     = 0.95
params.alpha     = 0.175
params.nu        = 3.2e7
params.eta_c     = 0.8
params.eta_M     = 0.066
params.P_BSM     = 0.98
params.q_0       = 0.01
params.R_dark    = 100
params.delta_det = 100e-12
params.T_coh     = 0.05
params.V         = PARAM_V

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_network_metrics(net, n_sources, rng):
    """
    Average SKR, QBER, fidelity and reachability over a random sample of
    source nodes, mirroring QuantumRepeaterNetwork.network_metrics() but
    (a) restricted to a source subsample for tractability and
    (b) also tracking QBER, which network_metrics() does not return.
    """
    n_nodes = net.A.shape[0]
    sample_size = min(n_sources, n_nodes)
    sources = rng.choice(n_nodes, size=sample_size, replace=False)

    skrs, qbers, fids = [], [], []
    n_viable = 0
    n_total = sample_size * (n_nodes - 1)

    for src in sources:
        df = net.analyze_all_paths(int(src))
        for _, row in df.iterrows():
            path = row["path"]
            if path is None or len(path) == 0:
                continue
            skrs.append(max(row["SKR"], 0.0))
            qbers.append(row["Q"])
            fids.append(row["F"])
            if row["SKR"] > 0:
                n_viable += 1

    return {
        "avg_skr": float(np.mean(skrs)) if skrs else 0.0,
        "avg_qber": float(np.mean(qbers)) if qbers else np.nan,
        "avg_fidelity": float(np.mean(fids)) if fids else np.nan,
        "reachability": n_viable / n_total if n_total > 0 else 0.0,
    }


def run_case(mux_label, multiplexing_type, M, k_min, rng):
    """Sweep density x repetitions for one multiplexing setting."""
    rows = []
    for rho in tqdm(DENSITIES, desc=f"[{mux_label}] density sweep"):
        for rep in range(N_REPS):
            A, dist, coords, scale_km = build_density_graph(N, rho, BETA, MU)

            net = QuantumRepeaterNetwork(
                params, A, dist, coords,
                architecture="node", scale=scale_km,
                multiplexing_type=multiplexing_type, M=M,
            )

            # Prune BEFORE computing metrics -- this changes which paths
            # Dijkstra can select.
            net_pruned = net.prune_by_min_rate(k_min)

            metrics = compute_network_metrics(net_pruned, N_SOURCES, rng)

            rows.append({
                "mux_type": mux_label,
                "k_min": k_min,
                "M": M,
                "density": rho,
                "rep": rep,
                "N": N,
                "avg_skr": metrics["avg_skr"],
                "avg_qber": metrics["avg_qber"],
                "avg_fidelity": metrics["avg_fidelity"],
                "reachability": metrics["reachability"],
            })
    return rows


# ---------------------------------------------------------------------------
# Run both cases and save
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(42)

    all_rows = []
    all_rows += run_case("no_mux", multiplexing_type=None, M=1,
                          k_min=KMIN_NO_MUX, rng=rng)
    all_rows += run_case("mux20", multiplexing_type="accumulated", M=M_MUX,
                          k_min=KMIN_MUX20, rng=rng)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Total time: {time.time() - t0:.1f} s")
