import numpy as np
import networkx as nx
import time
import random

# Import the custom functions from your environment
from network_funcs import *
from qopt_funcs import *

def optimal_path_algo(G, target, algo='serial'):
    if algo == 'serial':
        return nx.single_source_dijkstra(G, target, weight='weight')
    elif algo == 'parallel':
        return least_maximum_weight_path(G, target)
    else:
        print('wrong algo option in optimal_path_algo')
        return
# --- 1. BASIC NETWORK SETUP ---
N = 100  # Number of nodes
radius = 45
beta = 2.6261  # Parameter for S2 model
mu = 0.0233  # Parameter for S2 model
rate_min = 0  # Minimum acceptable key rate for a connection to exist

# --- 2. CALCULATE CROSSOVER DISTANCE (d_cross) ---
print("Calculating crossover distance...")
# Define lambda functions to easily evaluate CV and DV rates
func_CV = lambda dist: hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid=float('inf'))
func_DV = lambda dist: hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid=0)

# Find where the individual rates drop to zero (critical distances)
d_max = 1000
d_c_CV = bisection_solver(func_CV, 10E-06, d_max)
d_c_DV = bisection_solver(func_DV, 10E-06, d_max)

# Find where CV and DV rates intersect (crossover)
diff = lambda d: func_CV(d) - func_DV(d)
d_cross = bisection_solver(diff, 10E-06, d_max)

# Set our hybrid switching distance to the optimal crossover point
d_hybrid = d_cross
print(f"Optimal hybrid crossover distance found: {d_hybrid:.2f} km")

# --- 3. PRE-COMPUTE RATE LOOKUP TABLE ---
print("Pre-computing key rates...")
n_ds = 10000
# Create an array of possible distances up to the maximum critical distance
d_set = np.linspace(1. / n_ds, max(d_c_DV, d_c_CV), n_ds)
keyrates = np.zeros(n_ds)

# Calculate the rate for every distance in our array
for i in range(n_ds):
    keyrates[i] = hybrid_keyrate_bitpersec(state_of_the_art_params, d_set[i], d_hybrid)


def classifiying_node(G, node, current_edge_CV, current_edge_DV):
    neighbors = list(G.neighbors(node))
    # If a node didn't survive pruning and has no connections
    if len(neighbors) == 0:
        return 'isolated'

    if all(current_edge_CV[node, neighbor] for neighbor in neighbors):
        return 'CV'
    elif all(current_edge_DV[node, neighbor] for neighbor in neighbors):
        return 'DV'
    else:
        return 'hybrid'


def power_cost(G, edge_CV, edge_DV, include_true_dsp_cost=False):
    # print("\nCalculating network power consumption...")

    node_power = np.zeros(len(G.nodes()), dtype=float)

    # Base node overhead (standard computer)
    base_computer_W = 100.0
    shared_laser = 4.2

    # DV connection hardware (Laser, AM Modulator, Time Tagger)
    dv_edge_optical_W = 26.0 + 22.0
    # SNSPDs share a massive helium compressor (cryostat) supporting ~20 edges
    dv_cryostat_W = 2735.0

    # CV connection optical hardware (Laser, IQ Mod, BHD, DAC, ADC)
    cv_edge_optical_W = 5.4 + 6.8 + 40.0 + 20.0

    # CV requires heavy digital signal processing (DSP) to recover analog waveforms.
    # True cost assumes real-time processing: 0.0003 Joules/symbol * 100 MHz laser = 30,000 W.
    # Standard cost assumes offline processing using a dedicated 100 W computer.
    if include_true_dsp_cost:
        cv_dsp_W = 30000.0
    else:
        cv_dsp_W = 100.0

    cv_total_edge_W = cv_edge_optical_W + cv_dsp_W

    for node in G.nodes():
        power = base_computer_W

        # Count connections for this specific node
        neighbors = list(G.neighbors(node))
        n_CV = sum(edge_CV[node, neighbor] for neighbor in neighbors)
        n_DV = sum(edge_DV[node, neighbor] for neighbor in neighbors)

        if (n_CV + n_DV) > 0:
            power += shared_laser

        # DV specific hardware
        if n_DV > 0:
            power += (n_DV * dv_edge_optical_W)
            # Add a new cryostat for every 20 DV connections
            power += np.ceil(n_DV / 20) * dv_cryostat_W

        # CV specific hardware
        if n_CV > 0:
            power += (n_CV * cv_total_edge_W)

        node_power[node] = power

    total_power = sum(node_power)
    # print(f"Total Network Power Consumption: {total_power:,.2f} W")

    return node_power, total_power

def avg_SKR_sample(G_pruned):
    # Isolate the main connected network (giant component)
    comp_list = sorted(nx.connected_components(G_pruned), key=len, reverse=True)
    G_giant = G_pruned.subgraph(comp_list[0])
    giant_nodes = list(G_giant.nodes())

    # Limit routing tests to a subset of nodes to keep runtime manageable
    n_nodes_to_test = 20
    node_max = min(len(giant_nodes), n_nodes_to_test)
    network_rates = []

    # Route the quantum signals through the giant component
    for i in range(node_max):
        target = giant_nodes[i]

        # Use Dijkstra to find the optimal paths based on the lowest sum of inverse rates
        weights = nx.single_source_dijkstra_path_length(G_pruned, target, weight='weight')

        # Avoid double-counting node pairs by only checking sources with a lower index
        for source in range(target):
            if source in weights:
                # Convert the inverse weight back into the actual rate (bits/sec)
                actual_rate = weights[source] ** -1
                network_rates.append(actual_rate)
            else:
                # The nodes are completely disconnected
                network_rates.append(0)

    average_skr = np.average(network_rates)
    # print(f"Average Secret Key Rate for the network: {average_skr:.4f} bits/sec")
    return average_skr

def avg_SKR_giant(G_pruned, routing_mode='serial'):
    comp_list = sorted(nx.connected_components(G_pruned), key=len, reverse=True)
    G_giant = G_pruned.subgraph(comp_list[0])
    giant_nodes = list(G_giant.nodes())

    # Calculate the routing between every node in the giant component
    rates = []
    for i in range(len(giant_nodes)):
        target = giant_nodes[i]
        weights, paths = optimal_path_algo(G_pruned, target, algo=routing_mode)

        for source in giant_nodes[:i]:
            if source in weights:
                rates.append(weights[source] ** -1)
            else:
                rates.append(0)

    # Final metrics for this specific network layout
    avg_skr = np.average(rates) if rates else 0
    return avg_skr


def avg_SKR(G_pruned, routing_mode='parallel'):
    all_nodes = list(G_pruned.nodes())
    rates = []

    connected_pairs = 0
    total_pairs = 0

    for i in range(len(all_nodes)):
        target = all_nodes[i]
        weights, paths = optimal_path_algo(G_pruned, target, algo=routing_mode)

        for source in all_nodes[:i]:
            total_pairs += 1

            if source in weights and weights[source] != float('inf'):
                rates.append(weights[source] ** -1)
                connected_pairs += 1
            else:
                rates.append(0)

    # Calculate final metrics
    avg_skr = np.average(rates) if rates else 0
    reachability = connected_pairs / total_pairs if total_pairs > 0 else 0

    return avg_skr, reachability



def calculate_energy_efficiency(G, routing_mode='serial'):

    # Get the average Secret Key Rate
    average_SKR = avg_SKR(G, routing_mode=routing_mode)

    # Get the total power consumption
    _, total_power = power_cost(G, edge_CV, edge_DV, include_true_dsp_cost=False)

    # Calculate Energy Efficiency
    efficiency = average_SKR / total_power

    # Print the final benchmark metrics
    print("\n--- Network Energy Benchmark ---")
    print(f"Total Network Power Consumption: {total_power:,.2f} W")
    print(f"Average Secret Key Rate for the network: {average_SKR:.4f} bits/sec")
    print(f"Network Energy Efficiency:       {efficiency:,.6f} bits/Joule")
    print("--------------------------------")

    return efficiency


# Run the function and store the result
#network_EE = calculate_energy_efficiency(G_pruned)

def network_energy_efficiency(num_runs=5, N=100, radius=45, routing_mode='parallel', include_true_dsp_cost=False, type = 'hybrid'):
    print(f"\n========================================================")
    print(f" TOPOLOGY DIAGNOSTIC: Testing {num_runs} Unique Network Layouts ")
    print(f"========================================================")

    run_powers = []
    run_skrs = []
    run_ees = []
    run_reach = []

    for run in range(num_runs):
        # Generate a brand new raw S2 network layout
        A, Dists = S2_graph_definite_N(N, beta, mu, return_coords=False)

        # Initialize pruning and edge classification matrices
        W = np.zeros_like(A)

        if type == 'hybrid':
            edge_CV = np.zeros_like(A, dtype=bool)
            edge_DV = np.zeros_like(A, dtype=bool)

            # Prune edges and apply quantum weights
            for i in range(N):
                for j in range(i):
                    if A[i, j] == 1:
                        dij = radius * Dists[i, j]
                        idx_d = np.argmin(abs(dij - d_set))
                        h_rate = keyrates[idx_d]

                        if dij < d_c_DV and h_rate > rate_min:
                            W[i, j] = W[j, i] = h_rate ** -1

                            if dij >= d_hybrid:
                                edge_DV[i, j] = edge_DV[j, i] = True
                            else:
                                edge_CV[i, j] = edge_CV[j, i] = True
        elif type == 'CV':
            edge_CV = np.zeros_like(A, dtype=bool)
            edge_DV = np.zeros_like(A, dtype=bool)

            for i in range(N):
                for j in range(i):
                    if A[i, j] == 1:
                        dij = radius * Dists[i, j]
                        # Check physical limit for pure CV
                        if dij < d_c_CV:
                            h_rate = func_CV(dij)  # Pure CV rate
                            if h_rate > rate_min:
                                W[i, j] = W[j, i] = h_rate ** -1
                                edge_CV[i, j] = edge_CV[j, i] = True

        elif type == 'DV':
            edge_CV = np.zeros_like(A, dtype=bool)
            edge_DV = np.zeros_like(A, dtype=bool)

            for i in range(N):
                for j in range(i):
                    if A[i, j] == 1:
                        dij = radius * Dists[i, j]
                        # Check physical limit for pure DV
                        if dij < d_c_DV:
                            h_rate = func_DV(dij)  # Pure DV rate
                            if h_rate > rate_min:
                                W[i, j] = W[j, i] = h_rate ** -1
                                edge_DV[i, j] = edge_DV[j, i] = True
        else:
            raise ValueError("type must be 'hybrid', 'CV' or 'DV'")

        G_pruned = nx.from_numpy_array(W)

        # Calculate the deterministic power for this specific layout
        _, total_power = power_cost(G_pruned, edge_CV, edge_DV, include_true_dsp_cost=include_true_dsp_cost)

        avg_skr, reachability = avg_SKR(G_pruned, routing_mode=routing_mode)
        ee = avg_skr / total_power if total_power > 0 else 0

        run_powers.append(total_power)
        run_skrs.append(avg_skr)
        run_ees.append(ee)
        run_reach.append(reachability)

        print(f"Graph Layout {run + 1} -> Power: {total_power:,.0f} W | SKR: {avg_skr:,.2f} bps | EE: {ee:,.8f} bits/J")

    avg_power = np.average(run_powers)
    avg_SKR_all = np.average(run_skrs)
    avg_EE = np.average(run_ees)
    avg_reach = np.average(run_reach)
    # Analyze the variance across the different layouts
    print("\n--- AVERAGE RESULTS ACROSS ALL RUNS ---")
    print(f"Avg Power: {avg_power:,.0f} W | Avg SKR: {avg_SKR_all:,.2f} bps | Avg EE: {avg_EE:,.8f} bits/J")
    return run_powers, run_skrs, run_ees, avg_power, avg_SKR_all, avg_EE, avg_reach

if __name__ == "__main__":
    # Put your loose executable code/prints in here
    network_energy_efficiency(num_runs=5, N=500, radius=45, routing_mode='serial', type='DV')


def ee_comparison_diagnostic(N_values, radius=200, runs=3, routing_mode='serial'):
    print(f"\n{'=' * 60}")
    print(f" EE COMPARISON DIAGNOSTIC (Radius = {radius} km) ")
    print(f"{'=' * 60}")

    comparison_data = []

    for n_val in N_values:
        for _ in range(runs):
            A, Dists = S2_graph_definite_N(n_val, beta, mu, return_coords=False)
            W, edge_CV, edge_DV = np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)

            for i in range(n_val):
                for j in range(i):
                    if A[i, j] == 1:
                        dij = radius * Dists[i, j]
                        idx_d = np.argmin(abs(dij - d_set))
                        h_rate = keyrates[idx_d]
                        if dij < d_c_DV and h_rate > rate_min:
                            W[i, j] = W[j, i] = h_rate ** -1
                            if dij >= d_hybrid:
                                edge_DV[i, j] = edge_DV[j, i] = True
                            else:
                                edge_CV[i, j] = edge_CV[j, i] = True

            G = nx.from_numpy_array(W)
            _, total_power = power_cost(G, edge_CV, edge_DV, include_true_dsp_cost=True)

            if total_power == 0:
                continue

            # --- 1. GIANT COMPONENT CALCULATION ---
            comp_list = sorted(nx.connected_components(G), key=len, reverse=True)
            G_giant = G.subgraph(comp_list[0]) if comp_list else G.subgraph([])
            giant_nodes = list(G_giant.nodes())

            giant_rates = []
            for i in range(len(giant_nodes)):
                target = giant_nodes[i]
                weights, _ = optimal_path_algo(G, target, algo=routing_mode)
                for source in giant_nodes[:i]:
                    giant_rates.append(weights[source] ** -1 if source in weights else 0)
            avg_skr_giant = np.average(giant_rates) if giant_rates else 0

            # --- 2. ALL NODES CALCULATION ---
            all_nodes = list(G.nodes())
            all_rates = []
            for i in range(len(all_nodes)):
                target = all_nodes[i]
                weights, _ = optimal_path_algo(G, target, algo=routing_mode)
                for source in all_nodes[:i]:
                    all_rates.append(weights[source] ** -1 if source in weights else 0)
            avg_skr_all = np.average(all_rates) if all_rates else 0

            comparison_data.append({
                'N': n_val,
                'EE (Giant Component)': avg_skr_giant / total_power,
                'EE (All Nodes)': avg_skr_all / total_power
            })

        print(f"Completed N={n_val}")

    return comparison_data

def topology_variance_diagnostic(num_runs=5, N=100, radius=45):
    print(f"\n========================================================")
    print(f" TOPOLOGY DIAGNOSTIC: Testing {num_runs} Unique Network Layouts ")
    print(f"========================================================")

    run_powers = []
    run_skrs = []
    run_ees = []

    for run in range(num_runs):
        # Generate a brand new raw S2 network layout
        A, Dists = S2_graph_definite_N(N, beta, mu, return_coords=False)

        # Initialize pruning and edge classification matrices
        W = np.zeros_like(A)
        edge_CV = np.zeros_like(A, dtype=bool)
        edge_DV = np.zeros_like(A, dtype=bool)

        # Prune edges and apply quantum weights
        for i in range(N):
            for j in range(i):
                if A[i, j] == 1:
                    dij = radius * Dists[i, j]
                    idx_d = np.argmin(abs(dij - d_set))
                    h_rate = keyrates[idx_d]

                    if dij < d_c_DV and h_rate > rate_min:
                        W[i, j] = W[j, i] = h_rate ** -1

                        if dij >= d_hybrid:
                            edge_DV[i, j] = edge_DV[j, i] = True
                        else:
                            edge_CV[i, j] = edge_CV[j, i] = True

        G_pruned = nx.from_numpy_array(W)

        # Calculate the deterministic power for this specific layout
        _, total_power = power_cost(G_pruned, edge_CV, edge_DV, include_true_dsp_cost=False)

        # Isolate the main connected network to calculate True SKR
        comp_list = sorted(nx.connected_components(G_pruned), key=len, reverse=True)
        G_giant = G_pruned.subgraph(comp_list[0])
        giant_nodes = list(G_giant.nodes())

        # Calculate the routing between EVERY node in the giant component
        rates = []
        for i in range(len(giant_nodes)):
            target = giant_nodes[i]
            weights = nx.single_source_dijkstra_path_length(G_pruned, target, weight='weight')

            for source in giant_nodes[:i]:
                if source in weights:
                    rates.append(weights[source] ** -1)
                else:
                    rates.append(0)

        # Final metrics for this specific network layout
        avg_skr = np.average(rates) if rates else 0
        ee = avg_skr / total_power if total_power > 0 else 0

        run_powers.append(total_power)
        run_skrs.append(avg_skr)
        run_ees.append(ee)

        print(f"Graph Layout {run + 1} -> Power: {total_power:,.0f} W | SKR: {avg_skr:,.2f} bps | EE: {ee:,.8f} bits/J")

    # Analyze the variance across the different layouts
    print("\n--- VARIANCE ACROSS DIFFERENT TOPOLOGIES ---")
    print(
        f"Power Spread: {min(run_powers):,.0f} W to {max(run_powers):,.0f} W (Diff: {max(run_powers) / min(run_powers):.2f}x)")
    print(
        f"SKR Spread:   {min(run_skrs):,.2f} bps to {max(run_skrs):,.2f} bps (Diff: {max(run_skrs) / min(run_skrs):.2f}x)")
    print(
        f"EE Spread:    {min(run_ees):,.8f} bits/J to {max(run_ees):,.8f} bits/J (Diff: {max(run_ees) / min(run_ees):.2f}x)")


# Call this at the bottom of your script, passing your variables
# topology_variance_diagnostic(num_runs=10)

def diagnose_sampling_time_and_variance(G, sample_size=20, num_samples=5, routing_mode='serial'):
    print(f"\n{'=' * 60}")
    print(f" SAMPLING DIAGNOSTIC: {sample_size}-Node Sample vs True Network ")
    print(f" Routing Mode: {routing_mode.upper()}")
    print(f"{'=' * 60}")

    comp_list = sorted(nx.connected_components(G), key=len, reverse=True)
    G_giant = G.subgraph(comp_list[0])
    giant_nodes = list(G_giant.nodes())

    print(f"Nodes in Giant Component: {len(giant_nodes)}")

    print(f"\n--- Running {num_samples} Random Samples ({sample_size} nodes each) ---")
    sample_skrs = []
    sample_times = []

    for run in range(num_samples):
        start_time = time.time()

        random.shuffle(giant_nodes)
        sample_nodes = giant_nodes[:sample_size]

        rates = []
        for i in range(len(sample_nodes)):
            target = sample_nodes[i]

            # --- THE NEW FUNCTION CALL ---
            # Unpack the returned tuple into 'weights' and 'paths'
            weights, paths = optimal_path_algo(G, target, algo=routing_mode)

            for source in sample_nodes[:i]:
                if source in weights:
                    rates.append(weights[source] ** -1)
                else:
                    rates.append(0)

        sample_skr = np.average(rates)
        run_time = time.time() - start_time

        sample_skrs.append(sample_skr)
        sample_times.append(run_time)
        print(f"  Sample {run + 1}: SKR = {sample_skr:,.2f} bps | Time = {run_time:.4f} seconds")

    print("\n--- Running TRUE Network Calculation (All connected nodes) ---")
    start_time_true = time.time()

    true_rates = []
    for i in range(len(giant_nodes)):
        target = giant_nodes[i]

        # --- THE NEW FUNCTION CALL ---
        weights, paths = optimal_path_algo(G, target, algo=routing_mode)

        for source in giant_nodes[:i]:
            if source in weights:
                true_rates.append(weights[source] ** -1)
            else:
                true_rates.append(0)

    true_skr = np.average(true_rates)
    true_time = time.time() - start_time_true

    print(f"  TRUE SKR: {true_skr:,.2f} bps | Time = {true_time:.4f} seconds")

    print("\n--- FINAL ANALYSIS ---")
    skr_spread = max(sample_skrs) / min(sample_skrs) if min--(sample_skrs) > 0 else float('inf')
    avg_sample_time = np.average(sample_times)

    print(f"SKR Variance: {skr_spread:.2f}x spread (Max: {max(sample_skrs):,.0f}, Min: {min(sample_skrs):,.0f})")
    print(f"Time Savings: True run took {true_time:.4f}s. Average sample took {avg_sample_time:.4f}s.")

    return true_skr, true_time

# diagnose_sampling_time_and_variance(G_pruned, sample_size=20, num_samples=5, routing_mode='serial')