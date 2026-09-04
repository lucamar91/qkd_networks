"""
Monte Carlo heralding-rate study
=================================

Three questions, three sections:

  1) STUDY 1 - How many repetitions / how much simulated time do you need
     for a stable R_H estimate at a given distance?
  2) STUDY 2 - How finely do you need to sample distance L so you don't
     miss curvature (and don't waste compute oversampling a flat region)?
  3) STUDY 3 - Can a closed-form function be fit to the Monte Carlo curve
     so you don't have to re-run it for every new distance?

Set QUICK_TEST = True for a fast sanity-check run (~1-2 min) with small
sample sizes, purely to confirm the pipeline works. Set it to False (and
bump the parameters inside each study function, see the comments marked
"PRODUCTION") before you trust the numbers for real analysis.
"""

import os
import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

QUICK_TEST = False
OUT_DIR = 'mc_study_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# PHYSICAL PARAMETERS (from your script)
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


# ==========================================
# CORE SIMULATION FUNCTIONS
# ==========================================
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


def analytical_rate(L):
    """Closed-form (no-dead-time) heralding rate. L in km, scalar or array."""
    L = np.asarray(L, dtype=float)
    fib = 10 ** (-alpha * L / 10)
    Rid = Rid0 * P * fib * etaDi * eta_duty_chopper
    pdt = Rid * dt
    pH2 = pdt * (1 - (1 - pdt) ** (t_out / dt))
    return pH2 / dt


def run_mc_once(L, total_time_span, seed=None, chunk_size=5.0):
    """
    One independent Monte Carlo realization of the heralding process at
    distance L (km), for total_time_span seconds. Returns (R_H, n_success).
    """
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
# STUDY 1: how many repetitions do you need?
#
# IMPORTANT: this fixes a single, safe, cheap simulation length (T_chunk)
# and only ever runs simulations of that length. It never scales T itself
# up -- that's what would blow up run time / array size. Instead it asks:
# if I run T_chunk-long simulations N times and average the results, how
# does the precision of that average improve with N? The expensive part
# (running max_repeats independent short simulations) happens ONCE per
# distance; every N <= max_repeats is then evaluated cheaply by bootstrap
# resampling that same pool, so you don't re-run the MC for every N.
# ==========================================
def study_1_repetitions():
    print("=" * 70)
    print("STUDY 1: how many repetitions do you need?")
    print("=" * 70)

    if QUICK_TEST:
        distances = [0, 25]
        T_chunk = 0.5          # length of ONE simulation run -- kept small & safe
        max_repeats = 30       # total independent runs actually simulated
        N_values = [1, 2, 5, 10, 20, 30]
    else:
        # PRODUCTION: widen once QUICK_TEST validates the pipeline.
        # Note T_chunk stays modest -- you get more precision by adding
        # more repeats, not by making each run longer.
        distances = [0, 15, 30, 50]
        T_chunk = 2.0
        max_repeats = 100
        N_values = [1, 2, 5, 10, 20, 50, 100]

    n_bootstrap = 500
    rng = np.random.default_rng(12345)

    fig, ax = plt.subplots(figsize=(7, 5))
    table_rows = []

    for L in distances:
        t0 = time.time()
        rates = np.empty(max_repeats)
        counts = np.empty(max_repeats)
        for k in range(max_repeats):
            r, c = run_mc_once(L, T_chunk, seed=int(L * 1000) + k)
            rates[k] = r
            counts[k] = c
        elapsed = time.time() - t0
        print(f"\nL={L:>3} km: {max_repeats} independent runs x {T_chunk}s each "
              f"took {elapsed:.1f}s total ({1000*elapsed/max_repeats:.1f} ms/run) "
              f"-- fully bounded, known cost.")

        rel_errs = []
        for N in N_values:
            if N > max_repeats:
                continue
            boot_means = np.array([
                rng.choice(rates, size=N, replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            mean_of_means = boot_means.mean()
            std_of_mean = boot_means.std(ddof=1)
            rel = 100 * std_of_mean / mean_of_means if mean_of_means > 0 else np.nan
            rel_errs.append(rel)
            table_rows.append((L, N, N * T_chunk, mean_of_means, std_of_mean,
                                rel, N * np.mean(counts)))

        rel_errs = np.array(rel_errs)
        valid_N = np.array([N for N in N_values if N <= max_repeats])
        ax.plot(valid_N, rel_errs, 'o-', label=f'L={L} km')

        mask = np.isfinite(rel_errs) & (rel_errs > 0)
        if mask.sum() >= 2:
            logN = np.log(valid_N[mask])
            logE = np.log(rel_errs[mask])
            slope, _ = np.polyfit(logN, logE, 1)
            print(f"  fitted scaling exponent vs N = {slope:+.2f} "
                  f"(shot-noise / Poisson-counting limit expects ~ -0.5)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of repeats N averaged together (each T_chunk long)')
    ax.set_ylabel('Relative std of the averaged R_H estimate (%)')
    ax.set_title('Precision of the average vs number of repeats')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'study1_variability_vs_repeats.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nSaved: {path}")

    print(f"\n{'L(km)':>6} {'N':>5} {'eff.T(s)':>10} {'mean(Hz)':>12} "
          f"{'std(Hz)':>10} {'rel(%)':>8} {'<N_succ>':>10}")
    for row in table_rows:
        print(f"{row[0]:>6} {row[1]:>5} {row[2]:>10.2f} {row[3]:>12.2f} "
              f"{row[4]:>10.2f} {row[5]:>8.2f} {row[6]:>10.1f}")

    return table_rows


# ==========================================
# STUDY 2: distance spacing
# ==========================================
def study_2_spacing(target_tol_percent=2.0, L_max=50):
    print("\n" + "=" * 70)
    print("STUDY 2: how closely do distances need to be spaced?")
    print("=" * 70)

    dL_fine = 0.05
    L_fine = np.arange(0, L_max + dL_fine, dL_fine)
    R_fine = analytical_rate(L_fine)
    dR = np.gradient(R_fine, L_fine)
    rel_slope = np.abs(dR) / np.clip(R_fine, 1e-30, None) * 100  # %/km

    if QUICK_TEST:
        mc_grid = np.arange(0, L_max + 10, 10)
        mc_T = 0.3
        mc_repeats = 3
    else:
        # PRODUCTION
        mc_grid = np.arange(0, L_max + 2, 2)
        mc_T = 5.0
        mc_repeats = 10

    mc_mean, mc_std = [], []
    for L in mc_grid:
        rates = [run_mc_once(L, mc_T, seed=int(L * 1000) + 7_000_000 + k)[0]
                 for k in range(mc_repeats)]
        mc_mean.append(np.mean(rates))
        mc_std.append(np.std(rates, ddof=1) if mc_repeats > 1 else 0.0)
    mc_mean = np.array(mc_mean)
    mc_std = np.array(mc_std)

    # Adaptive grid suggestion, walking forward in L and taking a step
    # sized so the *analytical* curve changes by ~target_tol_percent.
    # This is a fast proxy for curvature; the MC curve is expected to be
    # at least as smooth (its extra dead-time nonlinearity is monotonic),
    # so treat this as a starting point, not a guarantee.
    adaptive = [0.0]
    while adaptive[-1] < L_max:
        Lc = adaptive[-1]
        idx = min(int(Lc / dL_fine), len(rel_slope) - 1)
        slope = rel_slope[idx]
        step = target_tol_percent / slope if slope > 1e-9 else 5.0
        step = float(np.clip(step, 0.1, 5.0))
        adaptive.append(Lc + step)
    adaptive = np.array(adaptive)
    adaptive = adaptive[adaptive <= L_max]

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(L_fine, R_fine, 'r--', label='Analytical (no dead time)')
    axs[0].errorbar(mc_grid, mc_mean, yerr=mc_std, fmt='bo', capsize=3,
                     label='Monte Carlo (avg over repeats)')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Rate (Hz)')
    axs[0].set_title('Rate vs distance: analytical vs Monte Carlo')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(L_fine, rel_slope, 'k-')
    axs[1].axhline(target_tol_percent, color='gray', linestyle=':',
                    label=f'{target_tol_percent}% target tolerance')
    for Lp in adaptive:
        axs[1].axvline(Lp, color='green', alpha=0.15)
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Distance L (km)')
    axs[1].set_ylabel('Local relative slope |dR/R|/dL (%/km)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'study2_spacing.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nSuggested adaptive grid ({len(adaptive)} points) for a "
          f"step-to-step analytical change <= {target_tol_percent}%:")
    print(np.round(adaptive, 2))

    return dict(L_fine=L_fine, R_fine=R_fine, rel_slope=rel_slope,
                mc_grid=mc_grid.astype(float), mc_mean=mc_mean, mc_std=mc_std,
                adaptive=adaptive)


# ==========================================
# STUDY 3: function fitting
# ==========================================
def model_deadtime(L, tau_eff, scale):
    """Non-paralyzable dead-time correction of the analytical rate."""
    R_id = analytical_rate(L) * scale
    return R_id / (1 + R_id * tau_eff)


def model_exp(L, a, k, c):
    return a * np.exp(-k * np.asarray(L)) + c


def model_rational_fib(L, a, b):
    fib = 10 ** (-alpha * np.asarray(L) / 10)
    return a * fib / (1 + b * fib)


def study_3_fitting(study2_results):
    print("\n" + "=" * 70)
    print("STUDY 3: can a function be fit to save re-running the MC?")
    print("=" * 70)

    L = study2_results['mc_grid']
    y = study2_results['mc_mean']
    yerr = study2_results['mc_std']
    # guard against zero-uncertainty points (e.g. only 1 repeat)
    fallback = np.nanmean(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0
    yerr_safe = np.where(yerr > 0, yerr, fallback)

    models = {
        'dead_time_corrected': (model_deadtime, [1e-4, 1.0]),
        'exponential': (model_exp, [max(y.max(), 1.0), 0.1, 0.0]),
        'rational_fib': (model_rational_fib, [max(y.max(), 1.0), 1.0]),
    }

    L_dense = np.linspace(L.min(), L.max(), 300)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fit_table = []

    for name, (func, p0) in models.items():
        try:
            popt, _ = curve_fit(func, L, y, p0=p0, sigma=yerr_safe,
                                 absolute_sigma=True, maxfev=30000)
            y_pred = func(L, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            chi2 = np.sum(((y - y_pred) / yerr_safe) ** 2)
            dof = max(len(y) - len(popt), 1)
            red_chi2 = chi2 / dof
            fit_table.append((name, popt, r2, red_chi2))
            axs[0].plot(L_dense, func(L_dense, *popt), '-',
                        label=f'{name} (R²={r2:.4f})')
        except Exception as e:
            print(f"  Fit failed for {name}: {e}")

    axs[0].errorbar(L, y, yerr=yerr, fmt='ko', capsize=3, label='MC data (avg)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Distance (km)')
    axs[0].set_ylabel('Rate (Hz)')
    axs[0].set_title('Candidate fits vs Monte Carlo data')
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)

    if fit_table:
        # "best" = reduced chi2 closest to 1 (well-calibrated fit, not
        # just highest R^2, which every model can chase)
        best_name, best_popt, best_r2, best_rc = min(
            fit_table, key=lambda r: abs(r[3] - 1))
        func = models[best_name][0]
        y_pred = func(L, *best_popt)
        axs[1].errorbar(L, (y - y_pred) / yerr_safe, yerr=1.0, fmt='o', capsize=3)
        axs[1].axhline(0, color='gray', linestyle=':')
        axs[1].set_xlabel('Distance (km)')
        axs[1].set_ylabel('Residual / sigma')
        axs[1].set_title(f'Residuals: {best_name}')
        axs[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'study3_fitting.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\n{'model':<22}{'R2':>10}{'reduced_chi2':>15}   params")
    for name, popt, r2, rc in fit_table:
        print(f"{name:<22}{r2:>10.5f}{rc:>15.3f}   {np.round(popt, 6)}")

    return fit_table


if __name__ == '__main__':
    t0 = time.time()
    study_1_repetitions()
    s2 = study_2_spacing()
    study_3_fitting(s2)
    print(f"\nTotal script time: {time.time() - t0:.1f} s "
          f"(QUICK_TEST={QUICK_TEST})")
