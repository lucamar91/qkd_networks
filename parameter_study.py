"""
parameter_study.py

Sweeps five physics parameters one-at-a-time (others held at baseline) at
FIXED density (1.0 node/km^2, i.e. 10^0), no multiplexing. Computes
network-level SKR / QBER / Fidelity / Entanglement rate / reachability at
each sweep point.

One self-contained script does the (expensive) simulation and writes a
CSV; plot_parameter_study.py reads that CSV and makes the figures.

IMPORTANT -- pruning depends on the swept parameter.
prune_by_min_rate() thresholds on rate_mtx = params.nu * self.Probs_mtx,
and Probs_mtx is rebuilt from params (p_det, eta_c, alpha, eta_M, ...) in
_build_prob_matrix(). So which elementary links survive pruning is NOT
fixed across a sweep -- e.g. a lossier fibre (higher alpha) prunes more
edges away *before* Dijkstra ever runs, which is a real effect we want to
capture, not an artifact to avoid. Concretely this means:
  1. A network object (and its pruned version) must be rebuilt fresh for
     EVERY (parameter, value, mux_case) combination -- never reused across
     sweep values. This script does that: `params` changes -> new
     QuantumRepeaterNetwork -> new .prune_by_min_rate(k_min) every time.
  2. Graph GEOMETRY (A, dist, coords from build_density_graph) does NOT
     depend on physics params -- only on density/beta/mu, which are fixed
     here -- so that part alone is built once per repetition and reused
     across the sweep. This is purely a topology-vs-physics distinction;
     it does not touch the pruning step at all.

Entanglement rate: analyze_all_paths() returns column 'R', which is
R_entanglement = raw_rate * fidelity (see calculate_metrics() in
repeaters_NV.py) -- this is what's reported as avg_ent_rate below.

Parameters swept (baseline value in parentheses):
  * P_BSM              : 0.0  -> 1.0    (baseline 0.9993)
  * T_coh              : 0.01 -> 10 s   (baseline 0.05), log-spaced
  * eta_M * eta_c       : 0.03 -> 0.50  (baseline 0.368), see note below
  * alpha (fibre loss)  : 0.14 -> 0.22 dB/km (baseline 0.18)
  * V (visibility)      : 0.8  -> 0.99  (baseline 0.95)

NOTE on total efficiency: the model needs eta_M and eta_c separately, not
just their product. This script holds eta_c fixed at the baseline (0.8)
and solves eta_M = total_eff / eta_c for each sweep point. If you'd rather
split it differently, edit `make_efficiency_params()` below.

Run `python parameter_study.py --benchmark` first to get a real time
estimate for YOUR machine before committing to a full run.
"""

import argparse
import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm  # pip install tqdm --break-system-packages

from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork, build_density_graph

# ---------------------------------------------------------------------------
# Config -- EDIT THESE
# ---------------------------------------------------------------------------

N          = 1000
BETA       = 2.6261
MU         = 0.0233
RHO_FIXED  = 1.0          # nodes/km^2, i.e. 10^0 -- fixed for this study

N_REPS     = 5           # <-- see the timing discussion; lower this if tight on time
N_SOURCES  = 500
SWEEP_POINTS = 8         # <-- max 10 per your request; drop to 6-8 if tight on time

KMIN_NO_MUX = 100         # pairs/s, pruning threshold, no multiplexing
KMIN_MUX20  = 1000        # pairs/s, pruning threshold, M = 20 multiplexing
M_MUX       = 20

OUTPUT_CSV = "parameter_results.csv"


def get_baseline_params():
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
    return params


BASE = get_baseline_params()
BASE_EFF = BASE.eta_M * BASE.eta_c   # baseline total efficiency, for reference

# ---------------------------------------------------------------------------
# Parameter sweep definitions
# Each apply_fn returns a FRESH RepeaterParams (deep copy of baseline) with
# one attribute changed -- never mutates BASE, and never gets reused across
# sweep values.
# ---------------------------------------------------------------------------

def _copy_params(overrides):
    p = copy.deepcopy(BASE)
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def make_pbsm_params(v):
    return _copy_params({"P_BSM": v})


def make_tcoh_params(v):
    return _copy_params({"T_coh": v})


def make_efficiency_params(total_eff):
    # eta_c held at baseline, eta_M solved to hit the target product.
    eta_c = BASE.eta_c
    eta_M = total_eff / eta_c
    return _copy_params({"eta_c": eta_c, "eta_M": eta_M})


def make_alpha_params(v):
    return _copy_params({"alpha": v})


def make_visibility_params(v):
    return _copy_params({"V": v})


SWEEPS = {
    "P_BSM":     (np.linspace(0.1, 1.0, SWEEP_POINTS),                       make_pbsm_params),
    "T_coh":     (np.logspace(np.log10(0.01), np.log10(10), SWEEP_POINTS),   make_tcoh_params),
    "eta_total": (np.linspace(0.03, 0.50, SWEEP_POINTS),                     make_efficiency_params),
    "alpha":     (np.linspace(0.14, 0.22, SWEEP_POINTS),                     make_alpha_params),
    "V":         (np.linspace(0.8, 0.99, SWEEP_POINTS),                      make_visibility_params),
}

MUX_CASES = [
    ("no_mux", None, 1, KMIN_NO_MUX),
]

# ---------------------------------------------------------------------------
# Metrics (averaged over a random sample of source nodes)
# ---------------------------------------------------------------------------

def compute_network_metrics(net, n_sources, rng):
    """
    Average SKR, QBER, fidelity, entanglement rate (col 'R'), and
    reachability over a random sample of source nodes. `net` must already
    be the PRUNED network -- pruning changes which paths Dijkstra can pick,
    so it has to happen before this is called.
    """
    n_nodes = net.A.shape[0]
    sample_size = min(n_sources, n_nodes)
    sources = rng.choice(n_nodes, size=sample_size, replace=False)

    skrs, qbers, fids, ent_rates = [], [], [], []
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
            ent_rates.append(row["R"])  # R_entanglement = raw_rate * fidelity
            if row["SKR"] > 0:
                n_viable += 1

    return {
        "avg_skr": float(np.mean(skrs)) if skrs else 0.0,
        "avg_qber": float(np.mean(qbers)) if qbers else np.nan,
        "avg_fidelity": float(np.mean(fids)) if fids else np.nan,
        "avg_ent_rate": float(np.mean(ent_rates)) if ent_rates else 0.0,
        "reachability": n_viable / n_total if n_total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_full_sweep(n_reps, progress=True):
    """
    Loops: rep -> (build graph geometry once) -> parameter -> sweep value
    -> mux case -> (build network + PRUNE fresh, since pruning depends on
    the current params) -> metrics.
    """
    rng = np.random.default_rng(42)
    rows = []

    total_iters = sum(len(vals) for vals, _ in SWEEPS.values()) * n_reps * len(MUX_CASES)
    pbar = tqdm(total=total_iters, desc="parameter sweep") if progress else None

    for rep in range(n_reps):
        # Graph geometry doesn't depend on physics params -- build once per rep.
        A, dist, coords, scale_km = build_density_graph(N, RHO_FIXED, BETA, MU)

        for param_label, (values, make_params_fn) in SWEEPS.items():
            for val in values:
                params = make_params_fn(val)  # fresh params for this sweep point

                for mux_label, mux_type, M, k_min in MUX_CASES:
                    net = QuantumRepeaterNetwork(
                        params, A, dist, coords,
                        architecture="node", scale=scale_km,
                        multiplexing_type=mux_type, M=M,
                    )
                    # Pruning depends on `params` (via Probs_mtx / nu), so it
                    # must be recomputed for every single (param, value, mux)
                    # combination -- this line runs fresh every iteration.
                    net_pruned = net.prune_by_min_rate(k_min)

                    metrics = compute_network_metrics(net_pruned, N_SOURCES, rng)

                    rows.append({
                        "param": param_label,
                        "param_value": float(val),
                        "mux_type": mux_label,
                        "k_min": k_min,
                        "M": M,
                        "density": RHO_FIXED,
                        "rep": rep,
                        "N": N,
                        "avg_skr": metrics["avg_skr"],
                        "avg_qber": metrics["avg_qber"],
                        "avg_fidelity": metrics["avg_fidelity"],
                        "avg_ent_rate": metrics["avg_ent_rate"],
                        "reachability": metrics["reachability"],
                    })
                    if pbar is not None:
                        pbar.update(1)
    if pbar is not None:
        pbar.close()
    return rows


# ---------------------------------------------------------------------------
# Benchmark: time ONE rep's worth of graph build + one analysis per mux
# case, extrapolate to the full configured sweep.
# ---------------------------------------------------------------------------

def benchmark():
    rng = np.random.default_rng(0)
    print("Timing one graph build + one pruned network analysis per mux case...")

    t0 = time.time()
    A, dist, coords, scale_km = build_density_graph(N, RHO_FIXED, BETA, MU)
    t_graph = time.time() - t0

    t_per_mux = []
    for mux_label, mux_type, M, k_min in MUX_CASES:
        t1 = time.time()
        net = QuantumRepeaterNetwork(
            BASE, A, dist, coords,
            architecture="node", scale=scale_km,
            multiplexing_type=mux_type, M=M,
        )
        net_pruned = net.prune_by_min_rate(k_min)
        _ = compute_network_metrics(net_pruned, N_SOURCES, rng)
        dt = time.time() - t1
        t_per_mux.append(dt)
        print(f"  [{mux_label}] one analysis: {dt:.2f} s")

    total_sweep_points = sum(len(vals) for vals, _ in SWEEPS.values())  # 5 * SWEEP_POINTS
    per_rep_time = t_graph + total_sweep_points * sum(t_per_mux)
    est_total = per_rep_time * N_REPS

    print(f"\nGraph build: {t_graph:.2f} s (once per rep)")
    print(f"Sum over mux cases per (param,value): {sum(t_per_mux):.2f} s")
    print(f"Total sweep points across all 5 params: {total_sweep_points}")
    print(f"Estimated time per rep: {per_rep_time:.1f} s")
    print(f"Estimated TOTAL time for N_REPS={N_REPS}: "
          f"{est_total:.1f} s  (~{est_total/60:.1f} min)")
    print("\nRule of thumb: total runtime scales linearly with "
          "N_REPS x SWEEP_POINTS. Halving either roughly halves the runtime.")
    return est_total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true",
                         help="Time a single graph+analysis and estimate total "
                              "runtime for the current N_REPS/SWEEP_POINTS, then exit.")
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    else:
        t0 = time.time()
        rows = run_full_sweep(N_REPS)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
        print(f"Total time: {time.time() - t0:.1f} s")
