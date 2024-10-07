import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import os

# This file contains the functions needed to generate, handle and visualize the networks
    
def powerlaw_rnd_var(gamma, size, x_min=1):      # returns a power-law distributed variable
    u = np.random.uniform(0,1,size)
    C = (gamma-1) * x_min**(gamma-1)
    x = (u*(1-gamma)/C + (x_min)**(1-gamma) )**(1/(1-gamma))
    return x

def uniform_unit_sphere_distribution(N):         # generates points uniformly distributed on the surface of a sphere of radius = 1
    n=0
    coords = []
    while n < N:
        v = np.random.uniform(-1., 1., (3,))
        sq_mod = np.sum(v**2)
        if sq_mod <= 1:
            v *= 1/np.sqrt(sq_mod)
            coords.append(v)
            n += 1
    return np.array(coords)

def angular_dist_in_sphere(u, v):       # assumes u, v points on the surface of a sphere, returns angular distance (= geodesic distance/radius)
    if np.any(np.abs(u)>1) or np.any(np.abs(v)>1):
        print('WARNING: inputs are supposed to be arrays of components between -1 and 1.')
    lat1 = np.arcsin(u[2])
    lat2 = np.arcsin(v[2])
    long1 = np.arctan2(u[1], u[0])
    long2 = np.arctan2(v[1], v[0])
    cosx = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(long1-long2)   # see haversine formula / great-circle distance
    return np.arccos(cosx)

def S2_graph_definite_N(N, beta, mu, D=2, return_coords=False):
    Dists = np.zeros((N,N))
    A = np.zeros((N,N))
    gamma = 2.3              # power-law distr. exponent, compatible with values found for real networks
    R = np.sqrt(N/4/np.pi)
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
    Graph = keep_giant_component( nx.from_numpy_array(A) )
    nodelist_clean = list( max(nx.connected_components(Graph), key=len) )
    N_old = len(nodelist_clean)
    A_old = A[nodelist_clean][:,nodelist_clean]
    Dists_old = Dists[nodelist_clean][:,nodelist_clean]
    k_old = k[nodelist_clean]
    coords_old = coords[nodelist_clean]
    # iterate until there is a single connected component containing N nodes
    while(N_old < N):
        n = N-N_old
        # create new coords and new ks for the new nodes
        coords = np.vstack((np.copy(coords_old), uniform_unit_sphere_distribution(n)))
        k = np.hstack( (np.copy(k_old), powerlaw_rnd_var(gamma, n)) )
        # compute the new elements of A and Dists
        A = np.zeros((N,N))
        A[:N_old, :N_old] = np.copy(A_old)
        Dists = np.zeros_like(A)
        Dists[:N_old, :N_old] = np.copy(Dists_old)
        for i in range(N_old, N):                        # for every new node
            for j in range(i):                  # it computes A, Dists with all the nodes already in the network
                Dists[i,j] = Dists[j,i] = angular_dist_in_sphere(coords[i], coords[j])
                toss = np.random.uniform(0,1)
                pij = ( 1 + ( R*Dists[i,j]/(mu*k[i]*k[j]) )**beta )**-1
                if toss < pij:
                    A[i,j] = A[j,i] = 1
        # clean the new graph
        Graph = keep_giant_component( nx.from_numpy_array(A) )
        nodelist_clean = list( max(nx.connected_components(Graph), key=len) )
        N_old = len(nodelist_clean)
        # selecting only the elements related to the nodes in the main component
        A_old = A[nodelist_clean][:,nodelist_clean]
        Dists_old = Dists[nodelist_clean][:,nodelist_clean]
        k_old = k[nodelist_clean]
        coords_old = coords[nodelist_clean]
    if return_coords:
        return A, Dists, coords
    else:
        return A, Dists

def keep_giant_component(Graph):       # only takes the largest connected component of a graph
    return Graph.subgraph(max(nx.connected_components(Graph), key=len)).copy()

# Algorithm used for finding the optimal path as explained in main text. Same inputs, outputs as nx.single_source_dijkstra
def least_maximum_weight_path(graph, source, target=None, weight='weight'):
    # Initialize data structures
    dist = {node: float('inf') for node in graph.nodes()}
    dist[source] = 0
    path = {source: [source]}
    heap = [(0, source)]

    # Path optimization
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

# Function for the visual representation of the system
def plot_graph_on_sphere(cartesian_coords, adjacency_matrix, R, filename='earth', caption='',
                         bckgrnd_color = 'midnightblue', pt_color='yellow', edge_color='white'):    # saves figure in: './earth_frames/<bckgrnd_color>/<filename>.png'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Generate coordinate arrays
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    z = cartesian_coords[:, 2]
    # Represent the sphere in transparency
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = R * np.outer(np.cos(u), np.sin(v))
    y_sphere = R * np.outer(np.sin(u), np.sin(v))
    z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.2)
    # Represent the points
    ax.scatter(x, y, z, s=50, color=pt_color)
    # Represents the arches connecting points on the surface
    n_points = cartesian_coords.shape[0]
    for i in range(n_points):
        for j in range(i+1, n_points):
            if adjacency_matrix[i, j] == 1:
                phi = np.arccos(np.dot(cartesian_coords[i], cartesian_coords[j]) / (R**2))    # the angle between the 2 vecs
                t = np.linspace(0, phi, 100)                                                  # parameter of the curve
                x_arc = np.sin(t) * (cartesian_coords[i, 0] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 0] / np.sin(phi))
                y_arc = np.sin(t) * (cartesian_coords[i, 1] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 1] / np.sin(phi))
                z_arc = np.sin(t) * (cartesian_coords[i, 2] / np.sin(phi)) + np.sin(phi - t) * (cartesian_coords[j, 2] / np.sin(phi))
                ax.plot(x_arc, y_arc, z_arc, color=edge_color, alpha=0.5)
    # set same scale for all axes
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    # save plot
    fig.set_size_inches(15,15)
    ax.set_facecolor(bckgrnd_color)
    fig.text(.5, .15, s=caption, color=edge_color, fontsize=30)
    filepath = 'earth_frames/' + bckgrnd_color + '/'
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath)
    filename = filepath + filename + '.png'
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)
    return


