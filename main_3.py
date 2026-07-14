import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.stats import binom


# =============================================================
# EXISTING HELPERS (unchanged, kept here so the file is self-contained)
# =============================================================
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


def binary_entropy(p):
    """Binary entropy H(p)"""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def BBPSSW(F1, F2):
    """
    BBPSSW entanglement distillation protocol.

    Given two pairs with fidelities F1, F2, perform one round of
    BBPSSW and return the output fidelity and success probability.
    """
    num = F1 * F2 + (1 - F1) * (1 - F2) / 9
    den = (F1 * F2
           + (F1 * (1 - F2) + (1 - F1) * F2) / 3
           + 5 * (1 - F1) * (1 - F2) / 9)  # This is also the probability of success
    F_out = num / den if den > 0 else 0.0
    return F_out, den


def distill_recursive(F0, level):
    """
    Nested BBPSSW purification tree.

    level = 0  -> no distillation: returns (F0, P_success=1.0, n_raw=1)
    level = d  -> combines 2**d raw pairs (in a binary tree of d rounds)
                  down to a single output pair.

    Returns
    -------
    F_final   : fidelity of the surviving pair after `level` rounds
    P_tree    : cumulative probability that the *entire* nested tree
                succeeds (every BBPSSW round along the way succeeds)
    n_raw     : number of raw (level-0) pairs consumed = 2**level
    """
    if level <= 0:
        return F0, 1.0, 1

    F_prev, P_prev, n_prev = distill_recursive(F0, level - 1)
    F_new, P_new = BBPSSW(F_prev, F_prev)

    # Need BOTH branches feeding this round to have succeeded, then this
    # round itself to succeed.
    P_total = (P_prev ** 2) * P_new
    n_total = n_prev * 2
    return F_new, P_total, n_total


def main_3():
    os.makedirs('tests', exist_ok=True)

    # =========================================================
    # HARDWARE SCENARIOS (FALLBACK VS OPTIMISTIC)
    # =========================================================
    HARDWARE = {
        'fallback': {
            'coherence_time': 4e-03,
            'transmission_to_memory_setup': 0.60,
            'memory_efficiency': 0.49,
            'multiplexing_modes': 750,
            'duty_cycle': 0.50,
            'source_heralding': 0.60,
            'prob_noise_detection': 6e-04,
            'indistinguishability': 0.95,
            'fibre_phase_stability': 0.96,
            'source_heralding_rate': 10000
        },
        'optimistic': {
            'coherence_time': 20e-03,
            'transmission_to_memory_setup': 0.81,
            'memory_efficiency': 0.60,
            'multiplexing_modes': 1200,
            'duty_cycle': 0.61,
            'source_heralding': 0.80,
            'prob_noise_detection': 5e-04,
            'indistinguishability': 0.95,
            'fibre_phase_stability': 0.96,
            'source_heralding_rate': 50000
        },

        'theoretical_ideal': {
            'coherence_time': 100e-03,  # Extended (assumes advanced dynamical decoupling)
            'transmission_to_memory_setup': 1.0,  # Perfect optical routing
            'memory_efficiency': 1.0,  # Perfect spin-wave write/read
            'multiplexing_modes': 1200,  # Max REIC multimode capacity
            'duty_cycle': 1.0,  # Continuous operation (no cryostat cooling breaks)
            'source_heralding': 1.0,  # Perfect source efficiency
            'prob_noise_detection': 1e-04,  # Low but non-zero dark counts to generate QBER
            'indistinguishability': 1.0,  # Perfect quantum interference
            'fibre_phase_stability': 1.0,  # Perfect active phase locking
            'source_heralding_rate': 50000  # Optimized clock speed
        }
    }

    MAX_LEVEL_SWEEP = 10      # distillation levels shown in the fidelity/rate-vs-level graph
    HOP_LEVELS = [0, 2, 4, 6, 8, 10]  # distillation levels shown in the vs-hops graph
    max_hops = 6

    for scenario_name, params in HARDWARE.items():
        print(f"Processing scenario: {scenario_name}...")

        # ---------------------------------------------------------
        # MAP VARIABLES TO MATH (identical to main_2)
        # ---------------------------------------------------------
        dt = 100 * 1e-09
        t_out = params['multiplexing_modes'] * dt
        lat = 100 * 1e-06

        p_d = 0.95
        etaDi = p_d
        etaDs = p_d

        P_BSM = 0.5

        eta_T_setup = params['transmission_to_memory_setup']
        eta_QM = params['memory_efficiency']
        eta_QN = params['source_heralding'] * eta_QM
        eta_map = eta_T_setup

        # ---------------------------------------------------------
        # BASE LINK FIDELITY (identical to main_2)
        # ---------------------------------------------------------
        g2 = 49.75

        g2sw = (params['source_heralding'] * eta_T_setup * eta_QM) * 1 / \
               ((params['source_heralding'] * eta_T_setup * np.sqrt(params['memory_efficiency'])) / g2 + params[
                   'prob_noise_detection']) + 1

        p10 = 0.5 * eta_QN
        p01 = p10
        p11 = 4 * p10 * p01 / (g2sw ** 2) * (1 + g2sw)

        V = params['fibre_phase_stability'] * params['indistinguishability'] * (g2sw - 1) / (g2sw + 1)

        Feff = 0.5 * (1 + V) * (p10 + p01) / (p10 + p01 + p11)
        F_link = Feff ** 2

        # ---------------------------------------------------------
        # EXACT RATE CALCULATION (MONTE CARLO) - identical to main_2
        # ---------------------------------------------------------
        pdt = params['source_heralding_rate'] * etaDi * dt

        time_span = 10
        repetition = 10
        total_sim_time = time_span * repetition

        idler_list = []
        signal_list = []
        time_mark = 0
        chunk_bins = int(time_span / dt)

        for _ in range(repetition):
            list_idler = np.random.rand(chunk_bins)
            index_idler = np.where(list_idler < pdt)[0]
            idler_list.append(index_idler + time_mark)

            list_signal = np.random.rand(chunk_bins)
            index_signal = np.where(list_signal < pdt)[0]
            signal_list.append(index_signal + time_mark)
            time_mark += chunk_bins

        Ctrigger = np.concatenate(idler_list) * dt if len(idler_list) > 0 else np.array([])
        Csignal = np.concatenate(signal_list) * dt if len(signal_list) > 0 else np.array([])

        success, _, _, _, _, _ = learn_statistics(Ctrigger, Csignal, t_out, lat)
        R_H = len(success) / total_sim_time

        # Raw single-pair rate, accumulated over the full stream (used by
        # the "accumulated" multiplexing mechanism).
        Rcoinc = params['duty_cycle'] * 0.5 * R_H * (eta_QN ** 2) * (eta_map ** 2)

        # Burst rate: rate at which a multiplexed attempt "window" occurs,
        # stripped of the per-mode efficiency terms (those now live inside
        # the binomial multiplexing probability below, for the
        # "single burst" mechanism).
        R_burst = params['duty_cycle'] * 0.5 * R_H

        # Per-mode raw-pair success probability, used for the single-burst
        # multiplexing binomial statistics.
        p_raw = (eta_QN ** 2) * (eta_map ** 2)
        M = params['multiplexing_modes']

        # ---------------------------------------------------------
        # TWO MULTIPLEXING MECHANISMS + NESTED DISTILLATION
        # ---------------------------------------------------------
        R_raw = Rcoinc  # raw single-pair rate, accumulated over the full stream
        T_coh = params['coherence_time']

        def accumulated_model(level):
            """
            Keep collecting raw successes (over an unbounded stream of
            temporal modes) until 2**level are in hand, storing early
            arrivals in quantum memory. Early pairs dephase while they
            wait for the rest of the set to arrive.
            """
            n = 2 ** level
            t_wait_avg = (n - 1) / (2 * R_raw) if R_raw > 0 else np.inf
            if np.isfinite(t_wait_avg):
                F_decohered = 0.5 + (F_link - 0.5) * np.exp(-t_wait_avg / T_coh)
            else:
                F_decohered = 0.5
            F_final, P_tree, n_raw = distill_recursive(F_decohered, level)
            R_acc = (R_raw / n) * P_tree
            return F_final, R_acc

        def single_burst_model(level):
            """
            Require all 2**level successes within a single burst of M
            temporal modes. No storage/decoherence penalty, but the
            binomial tail probability makes the rate fall off fast.
            """
            n = 2 ** level
            F_final, P_tree, n_raw = distill_recursive(F_link, level)
            P_have_enough = 1 - binom.cdf(n - 1, M, p_raw)
            R_sb = R_burst * P_have_enough * P_tree
            return F_final, R_sb

        # ===========================================================
        # GRAPH 1: initial fidelity & rate vs distillation level,
        # accumulated vs single-burst side by side
        # ===========================================================
        levels_sweep = np.arange(0, MAX_LEVEL_SWEEP + 1)
        F_acc, R_acc, F_sb, R_sb = [], [], [], []
        for lvl in levels_sweep:
            fa, ra = accumulated_model(lvl)
            fs, rs = single_burst_model(lvl)
            F_acc.append(fa); R_acc.append(ra)
            F_sb.append(fs); R_sb.append(rs)

        fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
        fig1.suptitle(f"Distillation Sweep [{scenario_name.upper()}] (Raw F_link={F_link:.4f})", fontsize=14)

        for ax, F_vals, R_vals, title in (
            (axes1[0], F_acc, R_acc, 'Accumulated'),
            (axes1[1], F_sb, R_sb, 'Single Burst'),
        ):
            color_f = 'tab:blue'
            ax.set_xlabel('Distillation Level')
            ax.set_ylabel('Initial Pair Fidelity', color=color_f)
            ax.plot(levels_sweep, F_vals, 'o-', color=color_f, label='Fidelity')
            ax.tick_params(axis='y', labelcolor=color_f)
            ax.set_xticks(levels_sweep)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            axb = ax.twinx()
            color_r = 'tab:red'
            axb.set_ylabel('Distilled Pair Rate (Hz)', color=color_r)
            axb.plot(levels_sweep, R_vals, 's--', color=color_r, label='Rate')
            axb.set_yscale('log')
            axb.tick_params(axis='y', labelcolor=color_r)

        fig1.tight_layout()
        save_path_1 = os.path.join('tests', f'distillation_fidelity_rate_{scenario_name}.png')
        fig1.savefig(save_path_1, dpi=300)
        print(f"Graph successfully saved to: {save_path_1}")
        plt.close(fig1)

        # ===========================================================
        # GRAPH 2: fidelity & SKR vs hops, multiple distillation levels,
        # Bell-Diagonal (solid) vs Werner (dashed) -- one figure per
        # multiplexing mechanism (accumulated / single burst)
        # ===========================================================
        hops_arr = np.arange(1, max_hops + 1)
        cmap = plt.get_cmap('viridis')
        level_colors = {lvl: cmap(i / max(1, len(HOP_LEVELS) - 1)) for i, lvl in enumerate(HOP_LEVELS)}

        mechanisms = (
            ('accumulated', accumulated_model),
            ('single_burst', single_burst_model),
        )

        for mech_name, mech_fn in mechanisms:
            fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
            fig2.suptitle(f"Distillation vs Hops [{scenario_name.upper()}, {mech_name.replace('_', ' ').title()}]",
                          fontsize=14)

            for lvl in HOP_LEVELS:
                F_d, R_d = mech_fn(lvl)
                T_link_d = 1.0 / R_d if R_d > 0 else np.inf

                F_BD_arr, SKR_BD_arr = [], []
                F_W_arr, SKR_W_arr = [], []

                for N in hops_arr:
                    T_total = T_link_d
                    for _ in range(2, N + 1):
                        T_total = (T_total + T_link_d) / P_BSM

                    R_hop = 1.0 / T_total if T_total > 0 else 0.0
                    R_sifted = 0.5 * R_hop

                    # Bell-Diagonal
                    F_BD = 0.5 + 0.5 * (2 * F_d - 1) ** N
                    Q_BD = 0.5 - 0.5 * (2 * F_d - 1) ** N
                    F_BD_arr.append(F_BD)
                    SKR_BD_arr.append(R_sifted * (1 - 2 * binary_entropy(Q_BD)))

                    # Werner
                    F_W = 0.75 * ((4 * F_d - 1) / 3) ** N + 0.25
                    Q_W = 0.5 - 0.5 * ((4 * F_d - 1) / 3) ** N
                    F_W_arr.append(F_W)
                    SKR_W_arr.append(R_sifted * (1 - 2 * binary_entropy(Q_W)))

                c = level_colors[lvl]
                axs2[0].plot(hops_arr, F_BD_arr, '-', color=c, label=f'BD, d={lvl}')
                axs2[0].plot(hops_arr, F_W_arr, '--', color=c, label=f'Werner, d={lvl}')

                axs2[1].plot(hops_arr, SKR_BD_arr, '-', color=c, label=f'BD, d={lvl}')
                axs2[1].plot(hops_arr, SKR_W_arr, '--', color=c, label=f'Werner, d={lvl}')

            axs2[0].axhline(0.5, color='gray', linestyle=':', label='Classical Limit')
            axs2[0].set_xlabel('Number of Links (Hops)')
            axs2[0].set_ylabel('End-to-End Fidelity')
            axs2[0].set_title('Fidelity Decay')
            axs2[0].grid(True, alpha=0.3)
            axs2[0].legend(fontsize=7, ncol=2)

            axs2[1].axhline(0.0, color='gray', linestyle=':', label='Zero SKR Boundary')
            axs2[1].set_xlabel('Number of Links (Hops)')
            axs2[1].set_ylabel('Secret Key Rate (bits/s)')
            axs2[1].set_title('SKR vs Hops (Negative = Insecure)')
            axs2[1].grid(True, alpha=0.3)
            axs2[1].legend(fontsize=7, ncol=2)

            fig2.tight_layout()
            save_path_2 = os.path.join('tests', f'distillation_hops_{mech_name}_{scenario_name}.png')
            fig2.savefig(save_path_2, dpi=300)
            print(f"Graph successfully saved to: {save_path_2}")
            plt.close(fig2)


if __name__ == '__main__':
    main_3()
