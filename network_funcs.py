import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import os

##### network related funcs
    
def powerlaw_rnd_var(gamma, size, x_min=1):
    u = np.random.uniform(0,1,size)
    C = (gamma-1) * x_min**(gamma-1)
    x = (u*(1-gamma)/C + (x_min)**(1-gamma) )**(1/(1-gamma))       # to get a variable power-law distributed
    return x

def uniform_unit_sphere_distribution(N):
    n=0
    coords = []
    while n < N:
        v = np.random.uniform(-1., 1., (3,))
        sq_mod = np.sum(v**2)
        if sq_mod <= 1:                      # if we include points in the cube but outside the sphere we lose the spherical symm
            v *= 1/np.sqrt(sq_mod)
            coords.append(v)
            n += 1
    return np.array(coords)

def angular_dist_in_sphere(u, v):   # basically computes length of geodesic curve bw tips of u and v, sitting on unit sphere surf
    if np.any(np.abs(u)>1) or np.any(np.abs(v)>1):
        print('WARNING: inputs are supposed to be arrays of components between -1 and 1.')
    lat1 = np.arcsin(u[2])
    lat2 = np.arcsin(v[2])
    long1 = np.arctan2(u[1], u[0])
    long2 = np.arctan2(v[1], v[0])
    cosx = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)   # see haversine formula / great-circle distance
    return np.arccos(cosx)

def S2_graph_definite_N(N, beta, mu, D=2, sample_from_file=False, return_coords=False):
    Dists = np.zeros((N,N))
    A = np.zeros((N,N))
    gamma = 2.3        # it should be in line with actual values in internet networks
    R = np.sqrt(N/4/np.pi)
    # generate network with N nodes    
    if sample_from_file:
        fname = '../mercator_data/as20000102.inf_coord'
        pool = np.loadtxt(fname)         # lines starting with '#' will be ignored (see loadtxt docs)
        np.random.shuffle(pool)          # shuffles randomly the rows of the array
        k_pool = pool[:,1]               # 2nd column contains the k's
        coords_pool = pool[:,3:]         # cols 3,4,5 contain the cartesian coordinates
        # to normalize latent vectors to unit length (not all vecs have same length!, newaxis needed for broadcasting arrays):
        coords_pool /= np.sqrt(np.sum(coords_pool**2, axis=1))[:, np.newaxis]
        coords = coords_pool[:N]
        k = k_pool[:N]
        bookmark = N                     # to keep track of the rows already extracted
    else:    
        coords = uniform_unit_sphere_distribution(N)
        k = powerlaw_rnd_var(gamma, N)
    for i in range(N):
        for j in range(i):
            toss = np.random.uniform(0,1)
            Dists[i,j] = Dists[j,i] = angular_dist_in_sphere(coords[i], coords[j])
            pij = ( 1 + ( R*Dists[i,j]/(mu*k[i]*k[j])**(1./D) )**beta )**-1
            if toss < pij:
                A[i,j] = A[j,i] = 1
    # keep only the giant component (in general there will be small isolated components)
    Graph = clean_the_dust( nx.from_numpy_array(A) )
    nodelist_clean = list( max(nx.connected_components(Graph), key=len) )
    N_old = len(nodelist_clean)
    A_old = A[nodelist_clean][:,nodelist_clean]
    Dists_old = Dists[nodelist_clean][:,nodelist_clean]
    k_old = k[nodelist_clean]
    coords_old = coords[nodelist_clean]
    # iterate until there is a single connected component containing N nodes
    while(N_old < N):
        n = N-N_old
        # i need new coords and new ks for the new nodes
        if sample_from_file:
            coords = np.vstack( (np.copy(coords_old), np.copy(coords_pool[bookmark:bookmark+n])) )
            k = np.hstack( (np.copy(k_old), np.copy(k_pool[bookmark:bookmark+n])) )
            bookmark += n
        else:
            coords = np.vstack((np.copy(coords_old), uniform_unit_sphere_distribution(n)))
            k = np.hstack( (np.copy(k_old), powerlaw_rnd_var(gamma, n)) )
        # i need to compute the new elements of A and Dists
        A = np.zeros((N,N))
        A[:N_old, :N_old] = np.copy(A_old)
        Dists = np.zeros_like(A)
        Dists[:N_old, :N_old] = np.copy(Dists_old)
        for i in range(N_old, N):                        # for every new node
            for j in range(i):                  # i compute A, Dist with all nodes alrdy in nw, new & old
                Dists[i,j] = Dists[j,i] = angular_dist_in_sphere(coords[i], coords[j])
                toss = np.random.uniform(0,1)
                pij = ( 1 + ( R*Dists[i,j]/(mu*k[i]*k[j]) )**beta )**-1
                if toss < pij:
                    A[i,j] = A[j,i] = 1
        # clean the new graph
        Graph = clean_the_dust( nx.from_numpy_array(A) )
        nodelist_clean = list( max(nx.connected_components(Graph), key=len) )
        N_old = len(nodelist_clean)
        # selecting only the elements related to the nodes in the main component
        A_old = A[nodelist_clean][:,nodelist_clean]
        Dists_old = Dists[nodelist_clean][:,nodelist_clean]
        k_old = k[nodelist_clean]
        coords_old = coords[nodelist_clean]
    plot_graph_on_sphere(coords, A, 1)
    if return_coords:
        return A, Dists, coords
    else:
        return A, Dists

def clean_the_dust(Graph):       # only takes the largest connected component of a graph
    return Graph.subgraph(max(nx.connected_components(Graph), key=len)).copy()

def plot_graph_on_sphere(cartesian_coords, adjacency_matrix, R, filename='earth', caption='',
                         bckgrnd_color = 'midnightblue', pt_color='yellow', edge_color='white'):    # ft chatgpt
    # could be improved, eg R must be the length of the coords vecs so its just messy to input it separately
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # generate cartesian coords arrays
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]
    # represent transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = R * np.outer(np.cos(u), np.sin(v))
    y_sphere = R * np.outer(np.sin(u), np.sin(v))
    z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.2)
    # represent points
    ax.scatter(x, y, z, s=50, color=pt_color)
    # represent paths between points as geodetic curves
    n_points = cartesian_coords.shape[0]
    for i in range(n_points):
        for j in range(i+1, n_points):
            if adjacency_matrix[i, j] == 1:
                phi = np.arccos(np.dot(cartesian_coords[i], cartesian_coords[j]) / (R**2))    # the angle bw the 2 vecs
                t = np.linspace(0, phi, 100)               # for the parametric curve
                x_arc = np.sin(t) * (cartesian_coords[i, 0] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 0] / np.sin(phi))
                y_arc = np.sin(t) * (cartesian_coords[i, 1] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 1] / np.sin(phi))
                z_arc = np.sin(t) * (cartesian_coords[i, 2] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 2] / np.sin(phi))
                ax.plot(x_arc, y_arc, z_arc, color=edge_color, alpha=0.5)
    # require same scale for all axes
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    #plt.rcParams['figure.figsize'] = [15,15]
    fig.set_size_inches(15,15)
    ax.set_facecolor(bckgrnd_color)
    #plt.show()            # interactive 3d mode (not compatible w savefig)
    fig.text(.5, .15, s=caption, color=edge_color, fontsize=30)
    output_dir = 'earth_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + '/' + bckgrnd_color):
        os.makedirs(output_dir + '/' + bckgrnd_color)
    filename = 'earth_frames/' + bckgrnd_color + '/' + filename + '.png'
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)
    return

'''def gif_from_list(frame_list, output_path = 'blackout.mp4'):
    with imageio.get_writer(output_path, fps=25) as writer:
        for frame in frame_list:
            writer.append_data(imageio.imread(frame))
    return'''

def dijkstra(W, source): # W is a numpy 2d array: symmetr mtx, Wij is weight of (i,j) edge # 
    # initialization:
    n = W.shape[0]
    S = [source]
    T = np.arange(n).tolist()
    T.remove(source)
    f = np.full(n, np.inf)
    f[source] = 0
    J = np.full(n, None)
    J[source] = n   # n acts as the index of an original node (not in the network)
    for i in range(n):
        if W[source, i] != 0:
            f[i] = W[source, i]
            J[i] = source
    # loop:
    while any(f[t] != np.inf for t in T):
        fT = [f[k] for k in T]
        j = T[np.argmin(fT)]
        T.remove(j)
        S.append(j)
        if T == []:
            break
        for i in [k for k in T if W[j,k]>0]:  # for all nodes in T adjacent to j
            if f[i] > f[j]+W[i,j]:
                f[i] = f[j]+W[i,j]
                J[i] = j
    return J, f

# The following is the pathfinding algorithm used for the final results. It finds the paths of edges achieving the lowest time. 
# Same inputs, outputs as nx.single_source_dijkstra
def least_maximum_weight_path(graph, source, target=None, weight='weight'):    # ft. chatgpt
    # Initialize data structures
    dist = {node: float('inf') for node in graph.nodes()}
    dist[source] = 0
    path = {source: [source]}
    heap = [(0, source)]

    # Dijkstra's algorithm
    while heap:
        current_weight, current_node = heapq.heappop(heap)
        if current_node == target:
            return dist[target], path[target]

        for neighbor in graph.neighbors(current_node):
            weight_value = graph[current_node][neighbor].get(weight, 1)
            updated_weight = max(dist[current_node], weight_value)
            if updated_weight < dist[neighbor]:
                dist[neighbor] = updated_weight
                path[neighbor] = path[current_node] + [neighbor]
                heapq.heappush(heap, (updated_weight, neighbor))

    return dist, path

# just a visualization tool to highlight the path
def visualize_least_maximum_weight_path(graph, source, target=None, weight='weight'):     # ft. chatgpt
    if target:
        _, least_max_path = least_maximum_weight_path(graph, source, target, weight)
    else:
        least_max_paths = {}
        for node in graph.nodes():
            least_max_weight, least_max_path = least_maximum_weight_path(graph, source, node, weight)
            least_max_paths[node] = (least_max_weight, least_max_path)
    pos = nx.circular_layout(graph)  # positions for all nodes
    nx.draw_networkx_nodes(graph, pos, node_size=70)
    nx.draw_networkx_edges(graph, pos, width=2)
    nx.draw_networkx_labels(graph, pos, font_size=6, font_family="sans-serif")
    edge_labels = {(u, v): d[weight] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    if target:
        path_edges = [(least_max_path[i], least_max_path[i+1]) for i in range(len(least_max_path)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2, edge_color='r')
    else:
        for node, (_, path) in least_max_paths.items():
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2, edge_color='r', alpha=0.5)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    return

# a visual tool for highlighting k-cores in a network
def visualize_graph_with_k_core(graph, k):
    # Find the k-core subgraph
    k_core_subgraph = nx.k_core(graph, k=k)

    # Plot the graph with circular layout
    pos = nx.circular_layout(graph)

    # Draw nodes and edges of the entire graph
    nx.draw(graph, pos, with_labels=True, node_color='skyblue')

    # Draw nodes and edges of the k-core subgraph in a different color
    nx.draw_networkx_nodes(k_core_subgraph, pos, node_color='red')
    nx.draw_networkx_edges(k_core_subgraph, pos, edge_color='red', width=2)

    # Show plot
    plt.axis('equal')
    plt.show()




############################################ FUNCTIONS FOR QUANTUM REPEATER NETWORKS ############################################


def optimal_quantum_relay_path(Probs_mtx, P_Bell, source, target=None): # structured like nx.single_source_dijkstra
    # subtlety here: we cannot log probs mtx because there are zero elements corresponding to non-directly connected nodes
    # however we want 0s in the same positions in W to make dijkstra work. so we use boolean masks
    W = np.zeros_like(Probs_mtx)
    mask = Probs_mtx > 0
    W[mask] = -np.log2(Probs_mtx[mask]) - np.log2(P_Bell)
    G = nx.from_numpy_array(W)
    weights, paths = nx.single_source_dijkstra(G, source, target, weight='weight')
    if target is None:
        probs = {node: np.exp(-weight + np.log2(P_Bell)) for node, weight in weights.items()}  # to avoid overcounting the Bell state success probability. exp to directly return probabilities
    else:
        probs = np.exp(-weights + np.log2(P_Bell))
    return probs, paths
    # EDIT: this model is too simple. if the probability is exponentially small, the weight is simply the sum of the lengths of the edges
    # ==> this penalizes too much the presence of a repeater: a repeaterless path is more convenient than one with a repeater! defeats the purpose

    # EDIT2: this is due to the fact that we are not modelling a quantum repeater bc we dont have quantum memories! this is called a quantum relay
    # there are models for quantum repeater chains, lets consider eg the one in shchukin et al "Waiting time in quantum repeaters with probabilistic entanglement swapping"

'''
def optimal_quantum_repeater_path(Probs_mtx, P_Bell, source, target=None): # structured like nx.single_source_dijkstra
    # subtlety here: we cannot log probs mtx because there are zero elements corresponding to non-directly connected nodes
    # however we want 0s in the same positions in W to make dijkstra work. so we use boolean masks
    W = np.zeros_like(Probs_mtx)
    mask = Probs_mtx > 0
    W[mask] = -np.log2(Probs_mtx[mask]) - np.log2(P_Bell)
    G = nx.from_numpy_array(W)
    weights, paths = nx.single_source_dijkstra(G, source, target, weight='weight')
    if target is None:     # + np.log2(P_Bell) to avoid overcounting the Bell state success probability. exp to directly return probabilities
    else:
        probs = {node: np.exp(-weight + np.log2(P_Bell)) for node, weight in weights.items()}  
        probs = np.exp(-weights + np.log2(P_Bell))
    return probs, paths
'''