import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Import your existing framework
from repeaters_NV import RepeaterParams, QuantumRepeaterNetwork, build_density_graph

DATA_FILE = "convergence_data.npz"


def run_convergence_test():
    # 1. Setup Parameters
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

    N = 1000
    rho = 1e-2
    max_sources = 1000  # Running to 1000 to get the exact ground truth
    chosen_n = 150  # The sample size you want to highlight and extract MRE for
    num_graphs = 10  # Averaging over 10 graphs as specified

    # --- Check if data already exists ---
    if os.path.exists(DATA_FILE):
        print(f"Found existing data file '{DATA_FILE}'. Loading results directly...")
        data = np.load(DATA_FILE)
        global_rates = data['rates']
        global_skrs = data['skrs']
    else:
        print(f"No existing data found. Starting full simulation...")
        global_rates = np.zeros((num_graphs, max_sources))
        global_skrs = np.zeros((num_graphs, max_sources))

        for g_idx in range(num_graphs):
            print(f"Generating graph {g_idx + 1}/{num_graphs} at density {rho}...")

            # Build the graph
            A, dist, coords, scale_km = build_density_graph(N, rho, beta=2.6261, mu=0.0233)

            # Initialize the repeater network
            net = QuantumRepeaterNetwork(params, A, dist, coords, scale=scale_km,
                                         architecture='node', multiplexing_type='accumulated', M=20)

            # Shuffle nodes for random incremental sampling
            all_nodes = list(range(N))
            random.shuffle(all_nodes)

            cumulative_skr = 0.0
            cumulative_rate = 0.0
            n_attempted = 0

            print("  Evaluating sources...")
            for i in range(max_sources):
                source = all_nodes[i]

                df = net.analyze_all_paths(source)

                source_skr_sum = sum(max(row['SKR'], 0) for _, row in df.iterrows())
                source_rate_sum = sum(row['R'] for _, row in df.iterrows())

                cumulative_skr += source_skr_sum
                cumulative_rate += source_rate_sum
                n_attempted += (N - 1)

                global_rates[g_idx, i] = cumulative_rate / n_attempted
                global_skrs[g_idx, i] = cumulative_skr / n_attempted

        # Save the raw data so we don't have to simulate this again
        np.savez(DATA_FILE, rates=global_rates, skrs=global_skrs)
        print(f"Simulation complete. Data saved to {DATA_FILE}")

    # 2. Calculate true statistics and relative errors
    mean_rates = np.mean(global_rates, axis=0)
    std_rates = np.std(global_rates, axis=0)

    mean_skrs = np.mean(global_skrs, axis=0)
    std_skrs = np.std(global_skrs, axis=0)

    # Ground truth is the ensemble average at N=1000
    true_ensemble_rate = mean_rates[-1]
    true_ensemble_skr = mean_skrs[-1]

    epsilon = 1e-12

    # Calculate relative error of the ensemble mean compared to the true ensemble mean
    rel_error_rates = np.abs(mean_rates - true_ensemble_rate) / (true_ensemble_rate + epsilon)
    rel_error_skrs = np.abs(mean_skrs - true_ensemble_skr) / (true_ensemble_skr + epsilon)

    sources_x = np.arange(1, max_sources + 1)

    # 3. Plotting (Thesis Style)
    print("Plotting results...")
    import seaborn as sns
    sns.set_theme(style="ticks")  # Ticks style removes the seaborn background

    # Adjust figsize to give square-ish plots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

    # --- Plot a: Entanglement Rate ---
    ax1.plot(sources_x, mean_rates, color='#ff7f0e', lw=2, label='Mean')
    ax1.fill_between(sources_x, mean_rates - std_rates, mean_rates + std_rates, color='#ff7f0e', alpha=0.2)
    ax1.set_xlabel(r'$N_{\rm sources}$', fontsize=16)
    ax1.set_ylabel(r'Avg $R_{\rm ent}$ (pairs/s)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(False)  # Ensure no grid
    ax1.set_box_aspect(1)  # Force perfectly square plot
    ax1.axvline(x=chosen_n, color='gray', linestyle=':', label=f'$n={chosen_n}$')
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top')
    ax1.legend(loc='lower right', frameon=False, fontsize=14)

    # Limit the x-axis to clearly show the convergence region
    ax1.set_xlim(0, 500)

    # --- Plot b: SKR ---
    ax2.plot(sources_x, mean_skrs, color='#2ca02c', lw=2, label='Mean')
    ax2.fill_between(sources_x, mean_skrs - std_skrs, mean_skrs + std_skrs, color='#2ca02c', alpha=0.2)
    ax2.set_xlabel(r'$N_{\rm sources}$', fontsize=16)
    ax2.set_ylabel(r'Avg SKR (bits/s)', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(False)  # Ensure no grid
    ax2.set_box_aspect(1)  # Force perfectly square plot
    ax2.axvline(x=chosen_n, color='gray', linestyle=':', label=f'$n={chosen_n}$')
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top')
    ax2.legend(loc='lower right', frameon=False, fontsize=14)

    ax2.set_xlim(0, 500)

    sns.despine()  # Removes top and right borders
    plt.tight_layout()
    plt.savefig('convergence_thesis_style.png', dpi=300, bbox_inches='tight')
    print("Saved as convergence_thesis_style.png")

    # 4. Print the MRE for the thesis text
    mre_rate_150 = rel_error_rates[chosen_n - 1] * 100
    mre_skr_150 = rel_error_skrs[chosen_n - 1] * 100

    print("\n--- MRE RESULTS FOR THESIS TEXT ---")
    print(f"At N = {chosen_n} sources:")
    print(f"Mean Relative Error for R_ent : {mre_rate_150:.4f}%")
    print(f"Mean Relative Error for SKR   : {mre_skr_150:.4f}%")
    print("-----------------------------------")

    plt.show()


if __name__ == "__main__":
    run_convergence_test()