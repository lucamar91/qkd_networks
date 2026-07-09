import numpy as np


def calculate_exact_script_metrics(
        tau_sw=0,  # Sweep variable: time in the spin state [s]
        t_coh=2e-3,  # T2/T_eff: effective coherence time [s]
        eta_t1=0.95,  # source-to-memory transmission
        eta_afc0=0.6,  # zero-time AFC efficiency
        eta_cp=0.8,  # control pulses efficiency
        eta_h=0.4,  # heralding efficiency
        p_pump=4,  # power of the SPDC pump [mW]
        a_g2=195,  # g2 model coefficient
        rid0=3026,  # 2 * 1513 [Hz/mW]
        noise_floor=8.2e-4,  # NF
        eta_ov=0.95,  # Indistinguishability (etaOv)
        v_phase=0.95,  # Fibre phase stability (Vphase)
        dt=100e-9,  # mode size [s]
        tau_afc=20e-6,  # AFC storage time [s]
        eta_duty_mem=303 / 707,  # Memory duty cycle
        eta_duty_chopper=20 / 33,  # SPDC chopper duty cycle
        eta_map=0.288  # etaDs (0.8) * eta_T3 (0.8) * eta_T2 (0.45)
):
    # --- 1. SPDC Rate and g2 ---
    # idler-signal cross-correlation
    g2 = 1 + a_g2 / p_pump
    # raw rate of successful heralding clicks
    rid = rid0 * p_pump * 1.0 * 0.8 * eta_duty_chopper  # fib=1.0, etaDi=0.8

    # --- 2. Memory Efficiencies ---
    t_out = tau_afc + tau_sw
    n_modes = t_out / dt

    # AFC decay and spin-wave coherence decay
    eta_afc = eta_afc0 * np.exp(-4 * tau_afc / t_coh)
    eta_coh = np.exp(-2 * (tau_sw / t_coh) ** 0.5)

    # Total QM efficiency
    eta_qm = eta_afc * (eta_cp ** 2) * eta_coh
    # Survival probability of the signal photon at the node
    eta_qn = eta_h * eta_t1 * eta_qm

    # --- 3. Heralding and Coincidence Rates ---
    pdt = rid * dt
    # Double-click probability
    p_h2 = pdt * (1 - (1 - pdt) ** n_modes)
    rh = p_h2 / dt

    # Detected coincidence rate (Hz)
    rate_coinc = eta_duty_mem * 0.5 * rh * (eta_qn ** 2) * (eta_map ** 2)

    # --- 4. Fidelity Calculation ---
    # Signal-to-noise ratio in the memory (mirroring original denominator perfectly)
    g2sw = (eta_h * eta_t1 * eta_afc * (eta_cp ** 2) * eta_coh) / \
           ((eta_h * eta_t1 * np.sqrt(eta_afc)) / g2 + noise_floor) + 1

    # Diagonal elements
    p10 = 0.5 * eta_qn
    p01 = p10
    p11 = 4 * p10 * p01 / (g2sw ** 2) * (1 + g2sw)

    # Visibility and Effective Fidelity
    visibility = v_phase * eta_ov * (g2sw - 1) / (g2sw + 1)
    f_eff = 0.5 * (1 + visibility) * (p10 + p01) / (p10 + p01 + p11)

    # Final target fidelity
    fidelity = f_eff ** 2

    return fidelity, rate_coinc


# Run with tau_sw = 0 (baseline max fidelity)
fid, rate = calculate_exact_script_metrics(tau_sw=0)
print(f"Original Script Fidelity: {fid * 100:.2f}%")
print(f"Original Script Coincidence Rate: {rate * 3600:.2f} counts/hour")