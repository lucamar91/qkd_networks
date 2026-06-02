import numpy as np
import networkx as nx

# Import the custom functions from your environment
from network_funcs import *
from qopt_funcs import *

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

# --- 4. GENERATE RAW GRAPH ---
print("Generating raw S2 network...")
# A: Adjacency matrix (1 if connected, 0 if not)
# Dists: Matrix of angular distances between nodes
A, Dists = S2_graph_definite_N(N, beta, mu, return_coords=False)

# --- 5. BUILD THE WEIGHTED GRAPH ---
print("Applying quantum weights to graph edges...")
W = np.zeros_like(A)  # Empty matrix to store our actual weights
edge_CV = np.zeros_like(A) # Tells you if edge is CV
edge_DV = np.zeros_like(A) # Tells you if edge is DV

# Loop through every possible pair of nodes
A_pruned = np.zeros_like(A)
for i in range(N):
    for j in range(i):  # Only check lower triangle (since graph is undirected)
        # If the raw graph says they are connected
        if A[i, j] == 1:

            # Calculate actual geographic distance
            dij = radius * Dists[i, j]

            # Find the closest matching distance in our pre-computed table
            idx_d = np.argmin(abs(dij - d_set))
            h_rate = keyrates[idx_d]

            # PRUNING: Only proceed if the link physically works!
            if dij < d_c_DV and h_rate > rate_min:

                # Store the INVERSE of the rate as the weight
                W[i, j] = W[j, i] = h_rate ** -1

                A_pruned[i,j] = A_pruned[j,i] = 1

                # NOW classify it as CV or DV (since we know it exists)
                if dij >= d_hybrid:
                    edge_DV[i, j] = edge_DV[j, i] = True
                else:
                    edge_CV[i, j] = edge_CV[j, i] = True

# Convert the NumPy weight matrix into a usable NetworkX graph object
G_pruned = nx.from_numpy_array(W)

print(
    f"Graph generated successfully with {G_pruned.number_of_nodes()} nodes and {G_pruned.number_of_edges()} valid quantum edges!")
print(np.array_equal(edge_CV + edge_DV, A_pruned))

def classifiying_node(G, node): # This is just to rpove that most nodes aren't just CV or DV byt hybrid
    neighbors = list(G.neighbors(node))
    if len(neighbors) == 1:
        return 'edge'
    elif all(edge_CV[node, neighbor] for neighbor in neighbors):
        return 'CV'
    elif all(edge_DV[node, neighbor] for neighbor in neighbors):
        return 'DV'
    else:
        return 'hybrid'

# Classify each node and store the results in a dictionary
node_classification = {}
for node in G_pruned.nodes():
    node_classification[node] = classifiying_node(G_pruned, node)


def power_cost(G, edge_CV, edge_DV, include_true_dsp_cost=False):
    # print("\nCalculating network power consumption...")

    node_power = np.zeros(len(G.nodes()), dtype=float)

    # Base node overhead (standard computer)
    base_computer_W = 100.0

    # DV connection hardware (Laser, AM Modulator, Time Tagger)
    dv_edge_optical_W = 4.2 + 26.0 + 22.0
    # SNSPDs share a massive helium compressor (cryostat) supporting ~20 edges
    dv_cryostat_W = 2735.0

    # CV connection optical hardware (Laser, IQ Mod, BHD, DAC, ADC)
    cv_edge_optical_W = 4.2 + 5.4 + 6.8 + 40.0 + 20.0

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

        if n_DV > 0:
            power += (n_DV * dv_edge_optical_W)
            # Add a new cryostat for every 20 DV connections
            power += np.ceil(n_DV / 20) * dv_cryostat_W

        if n_CV > 0:
            power += (n_CV * cv_total_edge_W)

        node_power[node] = power

    total_power = sum(node_power)
    # print(f"Total Network Power Consumption: {total_power:,.2f} W")

    return node_power, total_power

def avg_SKR(G_pruned):
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


def calculate_energy_efficiency(G):

    # Get the average Secret Key Rate
    average_SKR = avg_SKR(G)

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
network_EE = calculate_energy_efficiency(G_pruned)