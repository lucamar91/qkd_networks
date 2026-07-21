import numpy as np
import matplotlib.pyplot as plt
import os


class RepeaterParams:
    """
    Physical parameters for a DV quantum repeater link.
    """
    def __init__(self):
        self.p_det     = None
        self.alpha     = None
        self.V         = None
        self.nu        = None
        self.R_dark    = None
        self.delta_det = None
        self.p_pair    = None
        self.eta_c     = None
        self.P_BSM     = None
        self.T_coh     = None
        self.eta_M     = None


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
params.T_coh     = 10


def binary_entropy(x):
    """Binary Shannon entropy, safe at x=0 and x=1."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def build_prob_matrix(p, dist, A=1.0):
    """
    Per-link entanglement probability at distance `dist` (total link
    length, km). Mirrors _build_prob_matrix with no multiplexing.

    Returns
    -------
    P_both  : float   Probability both detectors click (successful
                       entanglement generation event).
    P_click : float   Single-arm click probability.
    """
    half_dist = dist / 2
    P_click = p.p_det * p.eta_c * 10 ** (-p.alpha * half_dist / 10) * A * p.eta_M
    P_both = P_click ** 2
    return P_both, P_click

"""
def F_link(p, P_click):
    p_dc  = p.R_dark * p.delta_det
    p_sig = P_click ** 2
    p_acc = 2 * P_click * (1 - P_click) * p_dc + (1 - P_click) ** 2 * p_dc ** 2
    F = 0.5 * (1 + p.V) * (p_sig / (p_sig + p_acc))
    return F
"""

def F_link(p, P_click):
    """
    Per-link fidelity accounting for baseline visibility V, dark
    counts, and the completely mixed state contribution of accidents.
    """
    p_dc = p.R_dark * p.delta_det

    # Exact Barrett-Kok double-click probabilities
    p_sig = 0.5 * (P_click ** 2)
    p_acc = 4 * P_click * p_dc

    P_total = p_sig + p_acc

    # Fidelity is the weighted sum of the true signal and the completely mixed state (0.25)
    F = (0.5 * (1 + p.V) * (p_sig / P_total)) + (0.25 * (p_acc / P_total))
    return F


def plot_werner_approximation(p, max_hops=10, dist=0.0, A=1.0, save_path=None):
    """
    Werner-state approximation of end-to-end metrics vs. number of
    hops, for a homogeneous chain at fixed per-link distance `dist`.
    """
    hops_arr = np.arange(1, max_hops + 1)

    P_both0, P_click0 = build_prob_matrix(p, dist, A)
    T_link0 = 1.0 / P_both0
    F_link0 = F_link(p, P_click0)
    W0 = (4 * F_link0 - 1) / 3

    rates, QBER_W, F_W_arr, SKR_W_arr = [], [], [], []

    for N in hops_arr:
        # --- SEPARATE RATE AND FIDELITY TRACKING ---
        T_total = T_link0  # Tracks total time for RATE (includes failures)
        W_current = W0     # Tracks Werner parameter for FIDELITY (only successful waits)

        for _ in range(2, N + 1):
            # 1. The existing state waits in memory while the NEW link is generated.
            # It only waits for the average time it takes to generate ONE successful link.
            t_wait_sec = T_link0 / p.nu
            W_current = W_current * np.exp(-t_wait_sec / p.T_coh)

            # 2. Swap the degraded existing state with the fresh new link (W0)
            W_current = W_current * W0

            # 3. Accumulate total time for the RATE (BSM failures force full restarts)
            T_total = (T_total + T_link0) / p.P_BSM

        # -------------------------------------------
        R_raw = p.nu / T_total
        R_sifted = 0.5 * R_raw
        rates.append(R_raw)

        # Calculate metrics using the correctly tracked W_current
        F_W = 0.75 * W_current + 0.25
        Q_W = 0.5 - 0.5 * W_current
        F_W_arr.append(F_W)
        QBER_W.append(Q_W)

        SKR_W = R_sifted * (1 - 2 * binary_entropy(Q_W))
        SKR_W_arr.append(SKR_W)

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Werner Approximation [L={dist} km]\n"
        f"(Base F={F_link0:.4f}, Base Rate={p.nu / T_link0:.4f} Hz)",
        fontsize=14, fontweight='bold'
    )

    axs[0].plot(hops_arr, rates, 'r-o')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of Links (Hops)')
    axs[0].set_ylabel('Rate (Hz)')
    axs[0].set_title('Generation Rate')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(hops_arr, QBER_W, 'r-o')
    axs[1].axhline(0.11, color='gray', linestyle=':', label='Security limit')
    axs[1].set_xlabel('Number of Links (Hops)')
    axs[1].set_ylabel('End-to-End QBER')
    axs[1].set_title('QBER Growth')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].plot(hops_arr, F_W_arr, 'r-o')
    axs[2].set_xlabel('Number of Links (Hops)')
    axs[2].set_ylabel('End-to-End Fidelity')
    axs[2].set_title('Fidelity Decay')
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(hops_arr, SKR_W_arr, 'r-o')
    axs[3].axhline(0.0, color='gray', linestyle=':', label='Zero SKR Boundary')
    axs[3].set_xlabel('Number of Links (Hops)')
    axs[3].set_ylabel('Secret Key Rate (bits/s)')
    axs[3].set_title('SKR vs Hops')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()

    plt.tight_layout()
    if save_path is None:
        import os
        os.makedirs('tests/tests', exist_ok=True)
        save_path = os.path.join('tests/tests', f'werner_approximation_L{dist}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Graph successfully saved to: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    plot_werner_approximation(params, max_hops=10, dist=0.0)