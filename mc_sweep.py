"""
Production Monte Carlo distance sweep
======================================

Applies the conclusions from the repetitions/spacing/fitting study:

  * REPETITIONS are chosen ADAPTIVELY per distance point, not fixed.
    A small pilot batch estimates the local relative noise, then the
    confirmed 1/sqrt(N) shot-noise scaling (exponent -0.50 to -0.52,
    measured at L=0/15/30/50 km) is used to extrapolate how many total
    repeats are needed to hit TARGET_REL_PCT. This matters a lot here:
    L=0 needs ~2 repeats for 1% precision, L=50 needs ~90+.

  * SPACING is uniform at GRID_STEP_KM = 0.5 km. The measured relative
    slope of the rate curve is close to flat (~7%/km) across the whole
    0-50 km range, so non-uniform spacing buys little; 0.5 km gives
    ~3.5% change between adjacent points, comfortably above the ~1%
    per-point MC noise floor.

  * RESULTS are written to a CSV (human-readable) and a companion .npz
    (fast to reload in numpy) so the sweep only has to be run once.
"""

import os
import csv
import time
import numpy as np
import numba as nb

# ==========================================
# TUNABLE CAMPAIGN SETTINGS
# ==========================================
L_MIN_KM = 0.0
L_MAX_KM = 100.0
GRID_STEP_KM = 0.5          # recommended: flat slope -> uniform grid is fine

TARGET_REL_PCT = 1.0        # target relative precision on the rate, per point
N_PILOT = 5                 # small pilot batch to measure local noise
N_MIN = 5                   # never trust fewer than this many repeats
N_MAX = 300                 # safety cap so a bad extrapolation can't run away
CHUNK_T = 2.0                # length of a single MC run (kept fixed & safe)

OUT_CSV = 'mc_distance_sweep.csv'

# ==========================================
# PHYSICAL PARAMETERS (unchanged from earlier scripts)
# ==========================================
Rid0 = 2 * 1513
P = 4
etaDi = 0.8
eta_duty_chopper = 20 / 33
alpha = 0.18
dt = 100e-9
N_modes = 1200
t_out = N_modes * dt
lat = 100e-6


@nb.njit()
def learn_statistics(trig, sig, tout, lat):
    her = []
    del_list = []
    marker = 0
    t_mark = 0
    c_mark = 0
    c_tout = 0
    c_ok = 0
    for j in range(len(trig)):
        if trig[j] < t_mark:
            c_mark += 1
        else:
            m = marker
            t_ref = trig[j]
            while (m < len(sig)):
                if (sig[m] >= t_ref + tout):
                    t_mark = t_ref + tout
                    c_tout += 1
                    break
                elif (sig[m] <= t_ref):
                    marker = m
                    m += 1
                else:
                    her.append(t_ref)
                    del_list.append(sig[m] - t_ref)
                    m += 1
                    t_mark = sig[m] + lat
                    c_ok += 1
                    break
    return her, del_list, c_mark, c_tout, c_ok, j


def run_mc_once(L, total_time_span, seed=None, chunk_size=5.0):
    rng = np.random.default_rng(seed)
    fib = 10 ** (-alpha * L / 10)
    Rid = Rid0 * P * fib * etaDi * eta_duty_chopper
    pdt = Rid * dt

    chunk_size = min(chunk_size, total_time_span)
    n_chunks = max(1, int(round(total_time_span / chunk_size)))
    chunk_bins = int(chunk_size / dt)

    idler_parts, signal_parts = [], []
    t_mark_offset = 0
    for _ in range(n_chunks):
        li = rng.random(chunk_bins)
        idler_parts.append(np.where(li < pdt)[0] + t_mark_offset)
        ls = rng.random(chunk_bins)
        signal_parts.append(np.where(ls < pdt)[0] + t_mark_offset)
        t_mark_offset += chunk_bins

    Ctrigger = np.concatenate(idler_parts) * dt if idler_parts else np.array([])
    Csignal = np.concatenate(signal_parts) * dt if signal_parts else np.array([])

    if len(Ctrigger) == 0 or len(Csignal) == 0:
        return 0.0, 0

    tmin = max(Ctrigger.min(), Csignal.min())
    tmax = min(Ctrigger.max(), Csignal.max())
    Ctrigger = Ctrigger[(Ctrigger >= tmin) & (Ctrigger <= tmax)]
    Csignal = Csignal[(Csignal >= tmin) & (Csignal <= tmax)]
    tmeasure = (tmax - tmin) if tmax > tmin else total_time_span

    if len(Ctrigger) == 0 or len(Csignal) == 0:
        return 0.0, 0

    success, _, _, _, _, _ = learn_statistics(Ctrigger, Csignal, t_out, lat)
    RH_mc = len(success) / tmeasure
    return RH_mc, len(success)


# ==========================================
# ADAPTIVE REPEAT LOGIC
# ==========================================
def run_distance_point(L, target_rel_pct=TARGET_REL_PCT, n_pilot=N_PILOT,
                        chunk_T=CHUNK_T, n_min=N_MIN, n_max=N_MAX, seed_base=0):
    """
    Pilot a few runs at distance L, extrapolate how many total repeats
    are needed to hit target_rel_pct (using the confirmed ~1/sqrt(N)
    scaling), then run the rest and return the combined statistics.
    """
    pilot_rates = np.array([
        run_mc_once(L, chunk_T, seed=seed_base + k)[0] for k in range(n_pilot)
    ])
    pilot_mean = pilot_rates.mean()
    pilot_std = pilot_rates.std(ddof=1) if n_pilot > 1 else pilot_mean * 0.2
    rel_pilot = 100 * pilot_std / pilot_mean if pilot_mean > 0 else 100.0

    if rel_pilot <= target_rel_pct:
        N_needed = n_pilot
    else:
        N_needed = int(np.ceil(n_pilot * (rel_pilot / target_rel_pct) ** 2))
    N_needed = int(np.clip(N_needed, n_min, n_max))

    if N_needed > n_pilot:
        extra_rates = np.array([
            run_mc_once(L, chunk_T, seed=seed_base + n_pilot + k)[0]
            for k in range(N_needed - n_pilot)
        ])
        all_rates = np.concatenate([pilot_rates, extra_rates])
    else:
        all_rates = pilot_rates[:N_needed]

    mean = all_rates.mean()
    std = all_rates.std(ddof=1) if len(all_rates) > 1 else 0.0
    rel = 100 * std / mean if mean > 0 else np.nan

    return dict(L_km=L, N_repeats=len(all_rates), T_chunk_s=chunk_T,
                rate_Hz=mean, std_Hz=std, rel_pct=rel)


# ==========================================
# SWEEP + PERSISTENCE
# ==========================================
def run_sweep(L_min=L_MIN_KM, L_max=L_MAX_KM, step=GRID_STEP_KM,
              target_rel_pct=TARGET_REL_PCT, out_csv=OUT_CSV):
    L_grid = np.round(np.arange(L_min, L_max + step, step), 4)
    results = []

    t0 = time.time()
    for i, L in enumerate(L_grid):
        seed_base = int(round(L * 10000)) + 1
        res = run_distance_point(L, target_rel_pct=target_rel_pct, seed_base=seed_base)
        results.append(res)
        print(f"[{i+1:>3}/{len(L_grid)}] L={L:6.2f} km  N={res['N_repeats']:4d}  "
              f"rate={res['rate_Hz']:10.2f} Hz  rel={res['rel_pct']:5.2f}%  "
              f"elapsed={time.time()-t0:7.1f}s")

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['L_km', 'N_repeats', 'T_chunk_s', 'rate_Hz', 'std_Hz', 'rel_pct'])
        for r in results:
            writer.writerow([r['L_km'], r['N_repeats'], r['T_chunk_s'],
                              r['rate_Hz'], r['std_Hz'], r['rel_pct']])
    print(f"\nSaved {len(results)} distance points to: {out_csv}")

    npz_path = os.path.splitext(out_csv)[0] + '.npz'
    np.savez(npz_path,
              L_km=np.array([r['L_km'] for r in results]),
              N_repeats=np.array([r['N_repeats'] for r in results]),
              rate_Hz=np.array([r['rate_Hz'] for r in results]),
              std_Hz=np.array([r['std_Hz'] for r in results]),
              rel_pct=np.array([r['rel_pct'] for r in results]))
    print(f"Saved fast-reload copy to: {npz_path}")

    total_time = time.time() - t0
    print(f"\nTotal sweep time: {total_time:.1f} s over {len(L_grid)} points "
          f"({total_time/len(L_grid):.2f} s/point average)")
    return results


def load_sweep(path=OUT_CSV):
    """
    Reload a saved sweep without re-running any Monte Carlo. Example:
        data = load_sweep()
        L, rate = data['L_km'], data['rate_Hz']
    """
    npz_path = os.path.splitext(path)[0] + '.npz'
    if os.path.exists(npz_path):
        with np.load(npz_path) as f:
            return {k: f[k] for k in f.files}
    # fall back to CSV if the .npz isn't there
    data = np.genfromtxt(path, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


if __name__ == '__main__':
    run_sweep()
