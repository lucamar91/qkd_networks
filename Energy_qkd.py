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
for i in range(N):
    for j in range(i):  # Only check lower triangle (since graph is undirected)
        if A[i, j] == 1:  # If the raw graph says they are connected

            # Calculate actual geographic distance
            dij = radius * Dists[i, j]

            # Find the closest matching distance in our pre-computed table
            idx_d = np.argmin(abs(dij - d_set))
            h_rate = keyrates[idx_d]

            if dij >= d_hybrid:
                edge_DV[i, j] = edge_DV [j, i] = True
            else:
                edge_CV[i, j] = edge_CV[j, i] = True

            # If the distance is valid and the rate is above our minimum
            if dij < d_c_DV and h_rate > rate_min:
                # Store the INVERSE of the rate as the weight (for routing algorithms)
                W[i, j] = W[j, i] = h_rate ** -1

# Convert the NumPy weight matrix into a usable NetworkX graph object
G_weighted = nx.from_numpy_array(W)

print(
    f"Graph generated successfully with {G_weighted.number_of_nodes()} nodes and {G_weighted.number_of_edges()} valid quantum edges!")
print(np.array_equal(edge_CV + edge_DV, A))

