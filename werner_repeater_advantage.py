import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


class RepeaterParams:
    """
    Physical parameters for a DV quantum repeater link.
    """

    def __init__(self):
        self.p_det = None
        self.alpha = None
        self.V = None
        self.nu = None
        self.R_dark = None
        self.delta_det = None
        self.p_pair = None
        self.eta_c = None
        self.P_BSM = None
        self.T_coh = None
        self.eta_M = None
        self.M = None


params = RepeaterParams()
params.p_det = 0.95
params.alpha = 0.18
params.V = 0.95
params.nu = 167e3
params.R_dark = 100
params.delta_det = 100e-12
params.eta_c = 0.8
params.P_BSM = 0.9993
params.eta_M = 0.46
params.T_coh = 0.05


# M will be set dynamically in the plotting loop


def binary_entropy(x):
    """Binary Shannon entropy, safe at x=0 and x=1."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def build_prob_matrix(p, dist, A=1.0):
    """
    Per-link entanglement probability at distance `dist`.
    """
    half_dist = dist / 2
    P_succ_single = p.p_det * p.eta_c * 10 ** (-p.alpha * half_dist / 10) * A * p.eta_M
    p_dc = p.R_dark * p.delta_det

    P_both_succ_single = 0.5 * (P_succ_single ** 2)
    P_both_acc_single = 4 * P_succ_single * p_dc
    P_both_raw_single = P_both_succ_single + P_both_acc_single

    P_both_succ = 1 - (1 - P_both_succ_single) ** p.M
    P_both_raw = 1 - (1 - P_both_raw_single) ** p.M

    return P_both_succ, P_both_raw, P_succ_single


def F_link(p, P_succ_single):
    """
    Per-link fidelity accounting for baseline visibility V, dark
    counts, and the completely mixed state contribution of accidents.
    """
    p_dc = p.R_dark * p.delta_det
    p_sig = 0.5 * (P_succ_single ** 2)
    p_acc = 4 * P_succ_single * p_dc
    P_total = p_sig + p_acc

    if P_total == 0:
        return 0.0

    F = (0.5 * (1 + p.V) * (p_sig / P_total)) + (0.25 * (p_acc / P_total))
    return F


def werner_curve_for_distance(p, L_total, max_hops, A=1.0):
    """
    Compute the Werner-approximation end-to-end metrics for a fixed distance.
    """
    hops_arr = np.arange(1, max_hops + 1)
    rates, QBER_W = [], []

    for N in hops_arr:
        dist_per_hop = L_total / N
        P_both_succ, P_both_raw, P_succ_single = build_prob_matrix(p, dist_per_hop, A)

        T_link = 1.0 / P_both_raw
        F0 = F_link(p, P_succ_single)
        W0 = (4 * F0 - 1) / 3

        T_total = T_link
        W_current = W0

        for _ in range(2, N + 1):
            t_wait_sec = T_link / p.nu
            W_current = W_current * np.exp(-t_wait_sec / p.T_coh)
            W_current = W_current * W0
            T_total = (T_total + T_link) / p.P_BSM

        F_W = 0.75 * W_current + 0.25
        Q_W = 0.5 - 0.5 * W_current
        QBER_W.append(Q_W)

        R_raw = p.nu / T_total
        R_entanglement = R_raw * F_W
        rates.append(R_entanglement)

    return {
        'hops': hops_arr,
        'rate': np.array(rates),
        'QBER': np.array(QBER_W),
    }


def plot_werner_validation_thesis_style(p, L_list, max_hops=10, save_path=None):
    """
    Generates a 2x2 thesis-style panel showing Rate and QBER for Sequential
    and Multiplexed architectures to validate repeater advantage and noise.
    """
    print("Generating validation plots...")
    sns.set_theme(style="ticks")

    # 1. Canvas size slightly increased to give larger text breathing room
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

    # --- ROW 1: SEQUENTIAL (M = 1) ---
    p.M = 1
    for L_total, color in zip(L_list, colors):
        curve = werner_curve_for_distance(p, L_total, max_hops)
        label = f"$L = {L_total}$ km"
        axs[0, 0].plot(curve['hops'], curve['rate'], '-o', lw=1.5, markersize=4, color=color, label=label)
        axs[0, 1].plot(curve['hops'], curve['QBER'], '-o', lw=1.5, markersize=4, color=color, label=label)

    # --- ROW 2: MULTIPLEXED (M = 20) ---
    p.M = 20
    for L_total, color in zip(L_list, colors):
        curve = werner_curve_for_distance(p, L_total, max_hops)
        label = f"$L = {L_total}$ km"
        axs[1, 0].plot(curve['hops'], curve['rate'], '-o', lw=1.5, markersize=4, color=color, label=label)
        axs[1, 1].plot(curve['hops'], curve['QBER'], '-o', lw=1.5, markersize=4, color=color, label=label)

    # --- FORMATTING ---
    labels = ['(a)', '(b)', '(c)', '(d)']
    axes_flat = axs.flatten()

    for i, ax in enumerate(axes_flat):
        ax.set_box_aspect(1)
        ax.grid(False)

        # Increased font sizes for axes and ticks
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel(r'$N_{\rm links}$', fontsize=18)
        ax.set_xlim(0.5, max_hops + 0.5)

        # Shifted letters further left (-0.22) and slightly up (1.05) to clear the axis ticks
        ax.text(-0.22, 1.05, labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom')

    # Rate Panels formatting (Column 0)
    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set_yscale('log')
        ax.set_ylabel(r'$R_{\rm ent}$ (pairs/s)', fontsize=18)
        ax.legend(loc='lower right', frameon=False, fontsize=12)

    # QBER Panels formatting (Column 1)
    for ax in [axs[0, 1], axs[1, 1]]:
        ax.axhline(0.11, color='gray', linestyle=':', lw=1.5, label='Security Limit (11%)')
        ax.set_ylabel('End-to-End QBER', fontsize=16)

        # Extended Y-axis to 0.80 to ensure the legend never touches the 800 km line
        ax.set_ylim(0, 0.80)
        ax.legend(loc='upper right', frameon=False, fontsize=11)

    sns.despine()
    plt.tight_layout()

    if save_path is None:
        save_path = 'werner_validation_thesis_style.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Graph successfully saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    L_list = [50, 200, 500, 800]
    plot_werner_validation_thesis_style(params, L_list, max_hops=10)