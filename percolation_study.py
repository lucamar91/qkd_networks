"""
percolation_study.py

Percolation-transition study: giant-component fraction <N_gcc>/N vs. spatial
node density, for several k_min pruning thresholds, with and without
multiplexing.

Fixes N and sweeps density rho via build_density_graph (N, rho -> S2 scale
factor), repeats each (mux_type, rho) combination `n_reps` times, prunes each
realization at several k_min values, and records the giant-component
fraction for every single repetition (not just the average) so you can
recompute statistics later without regenerating any graphs.

Requires network_funcs.py (S2_graph_definite_N) to be importable.
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm  # pip install tqdm --break-system-packages   (or just delete the tqdm wrapper below)

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork, build_density_graph

# ---------------------------------------------------------------------------
# Config -- EDIT THESE
# ---------------------------------------------------------------------------

N       = 1000
BETA    = 2.6261
MU      = 0.0233
N_REPS  = 10

# TODO: set this to whatever density range actually brackets the percolation
# transition for your params (run a quick single-rep scan first if unsure).
DENSITIES = np.unique(np.concatenate([
    np.logspace(-5, np.log10(3e-4), 4, endpoint=False),   # sparse tail below all transitions
    np.logspace(np.log10(3e-4), np.log10(3e-2), 20),      # dense, covers all k_min transitions
    np.logspace(np.log10(3e-2), 0, 5),                    # sparse tail above all transitions
]))   # nodes / km^2

# k_min thresholds to compare -- pick ~4-5 per case, spanning the practical
# rate range for that mux setting (see the rate_vs_distance study we did
# earlier: M=1 tops out ~1e4 pairs/s at d=0, M=20 tops out ~1e7).
KMIN_NO_MUX = [1e1, 1e2, 1e3]
KMIN_MUX20  = [1e2, 1e3, 1e4]

OUTPUT_CSV = "percolation_results.csv"

# Physical link parameters -- EDIT to match your real NV setup
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def giant_component_fraction(A):
    """Fraction of nodes in the largest connected component of A."""
    G = nx.from_numpy_array(A)
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / A.shape[0]


def run_case(mux_label, multiplexing_type, M, kmin_values):
    """
    Sweep density x repetitions for one multiplexing setting.
    Returns a list of result-row dicts (one per rho x rep x k_min).
    """
    rows = []
    for rho in tqdm(DENSITIES, desc=f"[{mux_label}] density sweep"):
        for rep in range(N_REPS):
            A, dist, coords, scale_km = build_density_graph(N, rho, BETA, MU)

            net = QuantumRepeaterNetwork(
                params, A, dist, coords,
                architecture='node', scale=scale_km,
                multiplexing_type=multiplexing_type, M=M,
            )

            pruned_networks = net.prune_sweep(kmin_values)

            for k_min, pruned in zip(kmin_values, pruned_networks):
                frac = giant_component_fraction(pruned.A)
                rows.append({
                    "mux_type": mux_label,
                    "k_min": k_min,
                    "density": rho,
                    "rep": rep,
                    "N": N,
                    "N_gcc": frac * N,
                    "giant_fraction": frac,
                })
    return rows


# ---------------------------------------------------------------------------
# Run both cases and save
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    all_rows = []
    all_rows += run_case("no_mux", multiplexing_type=None, M=1, kmin_values=KMIN_NO_MUX)
    all_rows += run_case("mux20", multiplexing_type='accumulated', M=20, kmin_values=KMIN_MUX20)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Total time: {time.time() - t0:.1f} s")
