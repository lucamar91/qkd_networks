import os
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


# ---------------------------------------------------------
# 1. EXACT MONTE CARLO FUNCTION
# ---------------------------------------------------------
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


def main():
    os.makedirs('tests', exist_ok=True)

    # ---------------------------------------------------------
    # 2. ALL EXPERIMENTAL PARAMETERS
    # ---------------------------------------------------------
    tau_AFC = 20 * 1e-06
    tau_SW = 0.0
    eta_AFC0 = 0.6
    eta_CP = 0.8

    eta_duty_mem = (303 / 707)
    eta_duty_chopper = (20 / 33)

    NF = 8.2 * 1e-04
    T2 = 2e-03
    gamma_inhom = 5e02
    T_eff = 2 * 1e-03

    etaH = 0.4
    P = 4
    a = 195
    Rid0 = 2 * 1513
    g2 = 1 + a / P

    dt = 100 * 1e-09
    t_out = tau_AFC + tau_SW
    lat = 100 * 1e-06

    eta_T1 = 0.95
    eta_T2 = 0.45
    eta_T3 = 0.8

    etaD = 0.8
    etaDi = etaD
    etaDs = 0.8

    L = 0
    alpha = 0.3
    fib = 10 ** (-alpha * L / 10)

    # ---------------------------------------------------------
    # 3. EXACT EFFICIENCIES
    # ---------------------------------------------------------
    eta_AFC = eta_AFC0 * np.exp(-4 * tau_AFC / T2)
    eta_coh = np.exp(-(tau_SW * np.pi * gamma_inhom) ** 2 / (2 * np.log(2)))
    eta_QM = eta_AFC * (eta_CP ** 2) * eta_coh

    eta_QN = etaH * eta_T1 * eta_QM
    eta_map = etaDs * eta_T3 * eta_T2

    # ---------------------------------------------------------
    # 4. BASE LINK FIDELITY
    # ---------------------------------------------------------
    g2sw = (etaH * eta_T1 * eta_AFC * (eta_CP ** 2) * eta_coh) * 1 / ((etaH * eta_T1 * np.sqrt(eta_AFC)) / g2 + NF) + 1

    p10 = 0.5 * eta_QN
    p01 = p10
    p11 = 4 * p10 * p01 / (g2sw ** 2) * (1 + g2sw)

    Vphase = 0.95
    etaOv = 0.95
    V = Vphase * etaOv * (g2sw - 1) / (g2sw + 1)

    Feff = 0.5 * (1 + V) * (p10 + p01) / (p10 + p01 + p11)
    F_link = Feff ** 2

    # ---------------------------------------------------------
    # 5. EXACT RATE CALCULATION (MONTE CARLO -> Rcoinc)
    # ---------------------------------------------------------
    Rid = Rid0 * P * fib * etaDi * eta_duty_chopper
    pdt = Rid * dt

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

    # The final mapped rate (from the COINCIDENCES block)
    Rcoinc = eta_duty_mem * 0.5 * R_H * (eta_QN ** 2) * (eta_map ** 2)
    T_link = 1.0 / Rcoinc

    # ---------------------------------------------------------
    # 6. PROPAGATE ACROSS N HOPS
    # ---------------------------------------------------------
    max_hops = 10
    hops_arr = np.arange(1, max_hops + 1)

    rates = []
    F_BD_arr = []
    SKR_BD_arr = []
    F_W_arr = []
    SKR_W_arr = []

    P_BSM = 0.5

    for N in hops_arr:
        # Sequential Time Calculation
        T_total = T_link
        for _ in range(2, N + 1):
            T_total = (T_total + T_link) / P_BSM

        R_raw = 1.0 / T_total
        rates.append(R_raw)

        R_sifted = 0.5 * R_raw

        # Bell-Diagonal
        F_BD = 0.5 + 0.5 * (2 * F_link - 1) ** N
        Q_BD = 0.5 - 0.5 * (2 * F_link - 1) ** N
        F_BD_arr.append(F_BD)

        # Note: max(0, ...) removed to allow negative values to plot
        SKR_BD = R_sifted * (1 - 2 * binary_entropy(Q_BD))
        SKR_BD_arr.append(SKR_BD)

        # Werner
        F_W = 0.75 * ((4 * F_link - 1) / 3) ** N + 0.25
        Q_W = 0.5 - 0.5 * ((4 * F_link - 1) / 3) ** N
        F_W_arr.append(F_W)

        SKR_W = R_sifted * (1 - 2 * binary_entropy(Q_W))
        SKR_W_arr.append(SKR_W)

    # ---------------------------------------------------------
    # 7. PLOT AND SAVE GRAPHS
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Approximation Comparison (L=0 km, Base F={F_link:.4f}, Base Rate={Rcoinc:.4f} Hz)", fontsize=14)

    # Plot 1: Rate vs Hops
    axs[0].plot(hops_arr, rates, 'k-o', label='Mapped Rate ($Rcoinc$)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of Links (Hops)')
    axs[0].set_ylabel('Rate (Hz)')
    axs[0].set_title('Sequential Generation Rate')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Plot 2: Fidelity vs Hops
    axs[1].plot(hops_arr, F_BD_arr, 'b-o', label='Bell-Diagonal')
    axs[1].plot(hops_arr, F_W_arr, 'r--x', label='Werner')
    axs[1].axhline(0.5, color='gray', linestyle=':', label='Classical Limit')
    axs[1].set_xlabel('Number of Links (Hops)')
    axs[1].set_ylabel('End-to-End Fidelity')
    axs[1].set_title('Fidelity Decay')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # Plot 3: SKR vs Hops (Linear scale to show negatives)
    axs[2].plot(hops_arr, SKR_BD_arr, 'b-o', label='Bell-Diagonal')
    axs[2].plot(hops_arr, SKR_W_arr, 'r--x', label='Werner')
    axs[2].axhline(0.0, color='gray', linestyle=':', label='Zero SKR Boundary')
    axs[2].set_xlabel('Number of Links (Hops)')
    axs[2].set_ylabel('Secret Key Rate (bits/s)')
    axs[2].set_title('SKR vs Hops (Negative = Insecure)')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    save_path = os.path.join('tests', 'swap_approximations_exact.png')
    plt.savefig(save_path, dpi=300)
    print(f"Graph successfully saved to: {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


# ---------------------------------------------------------
# 1. EXACT MONTE CARLO FUNCTION
# ---------------------------------------------------------
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


import os
import numpy as np
import matplotlib.pyplot as plt

# (Assuming learn_statistics and binary_entropy are defined elsewhere in your script)

def main_2():
    os.makedirs('tests', exist_ok=True)

    # =========================================================
    # 1. HARDWARE SCENARIOS (FALLBACK, OPTIMISTIC, THEORETICAL)
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
            'coherence_time': 100e-03,          # Extended (assumes advanced dynamical decoupling)
            'transmission_to_memory_setup': 1.0,# Perfect optical routing
            'memory_efficiency': 1.0,           # Perfect spin-wave write/read
            'multiplexing_modes': 1200,         # Max REIC multimode capacity
            'duty_cycle': 1.0,                  # Continuous operation (no cryostat cooling breaks)
            'source_heralding': 1.0,            # Perfect source efficiency
            'prob_noise_detection': 1e-04,      # Low but non-zero dark counts to generate QBER
            'indistinguishability': 1.0,        # Perfect quantum interference
            'fibre_phase_stability': 1.0,       # Perfect active phase locking
            'source_heralding_rate': 50000      # Optimized clock speed
        }
    }

    max_hops = 10
    hops_arr = np.arange(1, max_hops + 1)

    for scenario_name, params in HARDWARE.items():
        print(f"Processing scenario: {scenario_name}...")

        # ---------------------------------------------------------
        # 2. MAP VARIABLES TO MATH
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
        # 3. BASE LINK FIDELITY
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
        # 4. EXACT RATE CALCULATION (MONTE CARLO)
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

        Rcoinc = params['duty_cycle'] * 0.5 * R_H * (eta_QN ** 2) * (eta_map ** 2)
        T_link = 1.0 / Rcoinc if Rcoinc > 0 else float('inf')

        # ---------------------------------------------------------
        # 5. PROPAGATE ACROSS N HOPS
        # ---------------------------------------------------------
        rates, F_BD_arr, SKR_BD_arr, F_W_arr, SKR_W_arr = [], [], [], [], []

        for N in hops_arr:
            T_total = T_link
            for _ in range(2, N + 1):
                T_total = (T_total + T_link) / P_BSM

            R_raw = 1.0 / T_total if T_total < float('inf') else 0
            rates.append(R_raw)
            R_sifted = 0.5 * R_raw

            # Bell-Diagonal
            F_BD = 0.5 + 0.5 * (2 * F_link - 1) ** N
            Q_BD = 0.5 - 0.5 * (2 * F_link - 1) ** N
            F_BD_arr.append(F_BD)

            SKR_BD = R_sifted * (1 - 2 * binary_entropy(Q_BD))
            SKR_BD_arr.append(SKR_BD)

            # Werner
            F_W = 0.75 * ((4 * F_link - 1) / 3) ** N + 0.25
            Q_W = 0.5 - 0.5 * ((4 * F_link - 1) / 3) ** N
            F_W_arr.append(F_W)
            SKR_W = R_sifted * (1 - 2 * binary_entropy(Q_W))
            SKR_W_arr.append(SKR_W)

        # ---------------------------------------------------------
        # 6. PLOT AND SAVE SEPARATE GRAPHS
        # ---------------------------------------------------------
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Swap Approximations [{scenario_name.upper()}]\n(Base F={F_link:.4f}, Base Rate={Rcoinc:.4f} Hz)",
            fontsize=14, fontweight='bold'
        )

        # Subplot 1: Rates
        axs[0].plot(hops_arr, rates, 'k-o', label='Available Link Rate')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Number of Links (Hops)')
        axs[0].set_ylabel('Rate (Hz)')
        axs[0].set_title('Sequential Generation Rate')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Subplot 2: Fidelity
        axs[1].plot(hops_arr, F_BD_arr, 'b-o', label='Bell-Diagonal')
        axs[1].plot(hops_arr, F_W_arr, 'r--x', label='Werner')
        axs[1].axhline(0.5, color='gray', linestyle=':', label='Classical Limit')
        axs[1].set_xlabel('Number of Links (Hops)')
        axs[1].set_ylabel('End-to-End Fidelity')
        axs[1].set_title('Fidelity Decay')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        # Subplot 3: SKR
        axs[2].plot(hops_arr, SKR_BD_arr, 'b-o', label='Bell-Diagonal')
        axs[2].plot(hops_arr, SKR_W_arr, 'r--x', label='Werner')
        axs[2].axhline(0.0, color='gray', linestyle=':', label='Zero SKR Boundary')
        axs[2].set_xlabel('Number of Links (Hops)')
        axs[2].set_ylabel('Secret Key Rate (bits/s)')
        axs[2].set_title('SKR vs Hops')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

        plt.tight_layout()
        save_path = os.path.join('tests', f'swap_approximations_{scenario_name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"Graph successfully saved to: {save_path}")

        plt.close(fig)

if __name__ == "__main__":
    #main()
    main_2()