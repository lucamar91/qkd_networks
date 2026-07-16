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
#fid, rate = calculate_exact_script_metrics(tau_sw=0)
#print(f"Original Script Fidelity: {fid * 100:.2f}%")
#print(f"Original Script Coincidence Rate: {rate * 3600:.2f} counts/hour")

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time  # Imported to track execution time


# ==========================================
# 1. THE EXACT MONTE CARLO FUNCTION
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


# ==========================================
# 2. THE SIMULATION PARAMETERS
# ==========================================
Rid0 = 2 * 1513  # Base rate [Hz/mW]
P = 4  # Pump power [mW]
etaDi = 0.8  # Detector efficiency
eta_duty_chopper = 20 / 33  # Chopper duty cycle
alpha = 0.18  # Fiber loss [dB/km]

dt = 100e-09  # Mode size [s]
N_modes = 1200  # Optimistic modes
t_out = N_modes * dt  # Timeout [s]
lat = 100e-06  # Reset latency [s]

# CHUNKING PARAMETERS TO SAVE RAM
time_span_chunk = 10  # Sim span per chunk [s] (Keeps RAM usage low)
repetition = 1  # Number of chunks (10s * 100 = 1000s total simulation time)
total_time_span = time_span_chunk * repetition

distances = np.linspace(0, 50, 11)  # Sweep 0 to 50 km

mc_rates = []
analytical_rates = []

# ==========================================
# 3. RUNNING THE SWEEP
# ==========================================
print(f"Running sweep across distances (Simulating {total_time_span}s per link)...\n")

for L in distances:
    # Start the timer for this specific link
    start_time = time.time()

    # 1. Calculate physical rate arriving at the memory
    fib = 10 ** (-alpha * L / 10)
    Rid = Rid0 * P * fib * etaDi * eta_duty_chopper
    pdt = Rid * dt

    # 2. ANALYTICAL CALCULATION
    pH2_analytical = pdt * (1 - (1 - pdt) ** (t_out / dt))
    RH_analytical = pH2_analytical / dt
    analytical_rates.append(RH_analytical)

    # 3. MONTE CARLO CALCULATION (Chunked Generation)
    idler_list = []
    signal_list = []
    time_mark = 0
    chunk_bins = int(time_span_chunk / dt)

    # Loop to prevent RAM overflow
    for _ in range(repetition):
        list_idler = np.random.rand(chunk_bins)
        index_idler = np.where(list_idler < pdt)[0]
        idler_list.append(index_idler + time_mark)

        list_signal = np.random.rand(chunk_bins)
        index_signal = np.where(list_signal < pdt)[0]
        signal_list.append(index_signal + time_mark)

        time_mark += chunk_bins

    # Combine all the chunks into single arrays and convert to actual time [s]
    Ctrigger = np.concatenate(idler_list) * dt if len(idler_list) > 0 else np.array([])
    Csignal = np.concatenate(signal_list) * dt if len(signal_list) > 0 else np.array([])

    # Clean up boundary logic
    if len(Csignal) > 0 and len(Ctrigger) > 0:
        tmin = max(min(Csignal), min(Ctrigger))
        Csignal = Csignal[Csignal >= tmin]
        Ctrigger = Ctrigger[Ctrigger >= tmin]

        tmax = min(max(Csignal), max(Ctrigger))
        Csignal = Csignal[Csignal <= tmax]
        Ctrigger = Ctrigger[Ctrigger <= tmax]

        tmeasure = (tmax - tmin) if (tmax - tmin) > 0 else total_time_span
    else:
        tmeasure = total_time_span

    # Run the physical state machine
    if len(Ctrigger) > 0 and len(Csignal) > 0:
        success, _, _, _, _, _ = learn_statistics(Ctrigger, Csignal, t_out, lat)
        RH_mc = len(success) / tmeasure
    else:
        RH_mc = 0.0

    mc_rates.append(RH_mc)

    # End the timer
    end_time = time.time()
    exec_time = end_time - start_time

    print(f"L={L:>2.0f}km | MC Rate={RH_mc:>6.1f} Hz | Calc Time={exec_time:>5.2f} sec")

# ==========================================
# 4. PLOTTING THE RESULTS
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(distances, analytical_rates, 'r--', label='Analytical Formula (No Dead Time)')
plt.plot(distances, mc_rates, 'b-', label='Monte Carlo (Physical Dead Time)')
plt.title(f"Heralding Rate vs Fiber Length ($N_{{modes}}$ = {N_modes})")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Heralding Rate $R_H$ (Hz)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()