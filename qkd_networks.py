import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
from network_funcs import *
from qopt_funcs import *
import networkx as nx
import time
plt.rcParams['figure.figsize'] = [9,7]
plt.rcParams.update({'font.size': 15})

# WHAT DOES THIS CODE DO? 
# It generates instances of complex networks (nw) & computes avg rate and connectivity, sweeping node densities BY CHANGING SYSTEM SIZE
# this version can also simulate a QKD network using a hybrid combination of CV and DV, switching protoc at a dist d_hybrid
# in this version im going all in w networkx: computing all the shortest paths (all_pairs_dijkstra), not just sampling
# also im simplifying the computation and storage of the results

compute_nw_rates = True      # if False only connectivity of the network is computed (to reduce the runtime)
edges_may_fail = False       # if True: PoF parameter fixes the probability of failure of an edge (for whatsoever reason)
PoF = 0.1
keyrate_algo = 'parallel'     # 'parallel' or 'serial' (dijkstra)
detection_mode = 'homodyne'    # homo-/hetero-dyne: one/both quadratures are measured by the party that drives the reconciliation
reconciliation = 'reverse'     # type of reconciliation

# params for the nw model (beta, mu inferred using Mercator):
beta = 2.6261        # beta param of S2 model; for S1 the inferred beta was 1.2437
mu = 0.0233          # mu param of S2 model; for S1 the inferred mu was 0.0294
sample_from_file = False    # if True coordinates would be sampled from an existing embedding of a real network (d-Mercator needed)

Ns = [100]
rate_min = 0
n_nodes_for_dijkstra = 20

rho_span = '_wide'              # different sets of points for different plots: '_focus' (for susceptibility), '_wide' (for anything else)
if rho_span == '_focus':
    n_iter, n_couples = 40, 10
    rhos = 0.14*10**np.linspace(-2.3,-1.9,20)
elif rho_span == '_wide':
    n_iter, n_couples = 10, 10
    rhos = 0.14*10**np.linspace(-4.,1.,50)
else:
    print('ERROR: variable \'rho_span\' must be \'_focus\' or \'_wide\'.')

def optimal_path_algo(G, target, algo='serial'):
    if algo == 'serial':
        return nx.single_source_dijkstra(G, target, weight='weight')
    elif algo == 'parallel':
        return least_maximum_weight_path(G, target)
    else:
        print('wrong algo option in optimal_path_algo')
        return

# computing critical distances, needed later for pruning (if d_hyb is 0 hybrid_keyrate_bitpersec returns DV rates, if inf it returns CV rates)
d_max = 1000
func_CV = lambda dist : hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid = float('inf'))
d_c_CV = bisection_solver(func_CV, 10E-06, d_max)            # not starting from 0 bc if T=1 there a div by 0
func_DV = lambda dist : hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid = 0)
d_c_DV = bisection_solver(func_DV, 10E-06, d_max)
diff = lambda d: func_CV(d) - func_DV(d)
d_cross = bisection_solver(diff, 10E-06, d_max)

d_hybrid = float('inf')                      # float('inf') to force CV-only, 0 to force DV-only, d_cross for hybrid CV/DV
if not edges_may_fail:
    PoF = 0.

qkd = 'hybrid'            # forcing the qkd parameter to 'hybrid'
print('Coordinates are sampled ' + ('from a pool inferred from a real network through dMercator.' if sample_from_file else 'randomly.'))
print('Rates are ' + ('' if compute_nw_rates else 'not ') + 'computed.')
if edges_may_fail:
    print('Each edge may fail with probability %.3f' % PoF)


start = time.time()
for N in Ns:
    radii = np.sqrt(N/4/np.pi/rhos)
    print('%d instances of a '%n_iter + qkd + '-QKD network with %d nodes are generated, '%N + '%d pairs of nodes are sampled per instance.'%n_couples)
    print('Evaluating for d_hybrid = %.2f km...'%d_hybrid)
    giant_ratio = np.zeros((n_iter, len(radii)))
    clustering_coeffs = np.zeros_like(giant_ratio)
    dict_of_geo_dist_lists = {radius: [] for radius in radii}
    dict_of_rate_lists_dijkstra = {radius: [] for radius in radii}
    dict_of_len_lists_dijkstra = {radius: [] for radius in radii}
    dict_of_avg_degree_distribs = {radius: {} for radius in radii}
    dict_of_avg_topological_dists = {radius: [] for radius in radii}

    # computing rates (not time-expensive, safer than loading files since relevant params may change)
    n_ds = 10000
    d_set = np.linspace(1./n_ds,max(d_c_DV,d_c_CV),n_ds)
    keyrates = np.zeros((n_ds,))
    for i in range(n_ds):
        d = d_set[i]
        keyrates[i] = hybrid_keyrate_bitpersec(state_of_the_art_params, d, d_hybrid)

    for it in range(n_iter):
        print('N=%d, instance %d'%(N,it))
        A, Dists, coords = S2_graph_definite_N(N, beta, mu, return_coords=True)
        n_nodes_giant = []                                   # nr of nodes in largest (giant) component
        clustering_list = []
        rate_sum_accum = []
        rate_sq_sum_accum = []
        for radius in radii:
            # PRUNING
            A_pruned = np.zeros_like(A)
            W = np.zeros_like(A)                    # matrix of the graph weights ie the inverse rates (s/bit)
            for i in range(N):
                for j in range(i):
                    if A[i,j] == 1:
                        dij = radius*Dists[i,j]
                        idx_d = np.argmin( abs(dij-d_set) )
                        h_rate = keyrates[idx_d]
                        if dij < d_c_DV and h_rate > rate_min:    # the transition CV/DV can cause problems, im double-checking
                            if np.random.uniform() > PoF * int(edges_may_fail):  # ie if edge doesnt fail (uniform returns 0<x<1 by default)
                                W[i,j] = W[j,i] = h_rate**-1
                                A_pruned[i,j] = A_pruned[j,i] = 1
            # uncomment the following lines to generate plots of the system (increases runtime)
            # if radius in radii[::5]:
            #     plot_graph_on_sphere(coords, A_pruned, 1, filename='inst_%d' % it + 'earth_radius%.2f' % radius, bckgrnd_color = 'white', pt_color='red', edge_color='black')
            G_pruned = nx.from_numpy_array(W)          # return a weighted graph object

            comp_list = sorted(nx.connected_components(G_pruned), key=len, reverse=True)
            G_giant = G_pruned.subgraph(comp_list[0])
            n_nodes_giant.append(nx.number_of_nodes(G_giant))
            # Correction to the clustering coefficient
            clustering_coefficients = nx.clustering(G_pruned)
            if len([v for v in dict(G_pruned.degree).values() if int(v) > 1]) != 0:
                average_clustering_coefficient = sum(clustering_coefficients.values()) / len([v for v in dict(G_pruned.degree).values() if int(v) > 1])
            else:
                average_clustering_coefficient = 0
            clustering_list.append(average_clustering_coefficient)

            # histogram of degree distr: painful thing is to sum uneven lists and keep track of everything until the end of cycle
            deg_hist = nx.degree_histogram(G_pruned)           # degree histogram: it is a list, degree is the index
            for deg in range(len(deg_hist)):
                if deg in dict_of_avg_degree_distribs[radius].keys():
                    dict_of_avg_degree_distribs[radius][deg] += float(deg_hist[deg])/n_iter
                else:
                    dict_of_avg_degree_distribs[radius][deg] = float(deg_hist[deg])/n_iter          # if its first time it occurs

            # avg geographical distances in pruned network
            bool_mask = np.array(A_pruned, dtype=bool)
            dict_of_geo_dist_lists[radius].append(np.average(radius*Dists[bool_mask]))

            # lens of shortest paths
            if compute_nw_rates:
                node_counter = 0
                node_max = min(len(list(comp_list[0])), n_nodes_for_dijkstra)
                while node_counter < node_max:
                    target = list(comp_list[0])[node_counter]   # this is done to take nodes in the same conn comp
                    weights, paths = optimal_path_algo(G_pruned, target, algo=keyrate_algo)
                    for source in range(target):
                        if nx.has_path(G_pruned, source, target):   # could i use 'weights' dict keys to check this?
                            dict_of_rate_lists_dijkstra[radius].append( weights[source]**-1 ) # dict of weights --> array of rates
                            dict_of_len_lists_dijkstra[radius].append( len(paths[source])-1 )   # the list shortest_path contains E+1 nodes, E=nr of edges
                        else:
                            dict_of_rate_lists_dijkstra[radius].append( 0 )
                    node_counter += 1

            # avg topological distance (expected logN if smallworld)
            dict_of_avg_topological_dists[radius].append(nx.average_shortest_path_length(G_giant))

        giant_ratio[it] = np.array(n_nodes_giant)/float(N)
        clustering_coeffs[it] = np.array(clustering_list)


    # saving avg's and errorbars:    ebar = 2*sigma/sqrt(N)  -->  stdev of the mean
    giant_ratio_avg = np.average(np.array(giant_ratio), axis=0)
    giant_ratio_ebar = 2*np.std(np.array(giant_ratio), axis=0, ddof=1)/np.sqrt(n_iter)

    if compute_nw_rates:
        network_rate_avg, network_rate_ebar = np.zeros_like(radii), np.zeros_like(radii)
        avg_shortest_path_lens, ebar_shortest_path_lens = np.zeros_like(radii), np.zeros_like(radii)
        avg_geo_dist, ebar_geo_dist = np.zeros_like(radii), np.zeros_like(radii)
        avg_topolog_dist, ebar_topolog_dist = np.zeros_like(radii), np.zeros_like(radii)
        for r in range(len(radii)):
            radius = radii[r]
            network_rate_avg[r] = np.average(np.array(dict_of_rate_lists_dijkstra[radius]))
            network_rate_ebar[r] = 2*np.std(np.array(dict_of_rate_lists_dijkstra[radius]), ddof=1)/np.sqrt(len(dict_of_rate_lists_dijkstra[radius]))
            avg_shortest_path_lens[r] = np.average(np.array(dict_of_len_lists_dijkstra[radius]))
            ebar_shortest_path_lens[r] = 2*np.std(np.array(dict_of_len_lists_dijkstra[radius]), ddof=1)/np.sqrt(n_iter)
            avg_geo_dist[r] = np.average(np.array(dict_of_geo_dist_lists[radius]))
            ebar_geo_dist[r] = 2*np.std(np.array(dict_of_geo_dist_lists[radius]), ddof=1)/np.sqrt(len(dict_of_geo_dist_lists[radius]))
            avg_topolog_dist[r] = np.average(np.array(dict_of_avg_topological_dists[radius]))
            ebar_topolog_dist[r] = 2*np.std(np.array(dict_of_avg_topological_dists[radius]), ddof=1)/np.sqrt(len(dict_of_avg_topological_dists[radius]))

    if qkd == 'CV' or qkd == 'DV':
        suffix = '_N%d' % N
    elif qkd == 'hybrid':
        suffix = '_N%d' % N + '_d_hyb%.0f' % d_hybrid
    suffix += '_NX'
    suffix += rho_span
    suffix += '_ratemin%.0f' % rate_min
    suffix += ('_sampled_from_real_nw' if sample_from_file else '') + ('_PoF%.2f' % PoF if edges_may_fail else '')
    suffix += '_%dnws' % n_iter
    # SAVING FILES
    if compute_nw_rates:
        # KEY RATE + DISTANCE BW NODES
        if keyrate_algo == 'serial':
            np.savetxt('outputs/out_'+qkd+'_rhos_rate' + suffix + '.dat', np.vstack((rhos, network_rate_avg, network_rate_ebar)) )
            np.savetxt('outputs/out_'+qkd+'_rhos_aspl_dijk' + suffix + '.dat', np.vstack((rhos, avg_shortest_path_lens, ebar_shortest_path_lens)) )
        elif keyrate_algo == 'parallel':
            np.savetxt('outputs/out_'+qkd+'_rhos_parK' + suffix + '.dat', np.vstack((rhos, network_rate_avg, network_rate_ebar)) )
            np.savetxt('outputs/out_'+qkd+'_rhos_aspl' + suffix + '.dat', np.vstack((rhos, avg_shortest_path_lens, ebar_shortest_path_lens)) )
    # CONNECTIVITY
    np.savetxt('outputs/out_'+qkd+'_rhos_conn' + suffix + '.dat', np.vstack((rhos,giant_ratio_avg, giant_ratio_ebar)) )
    # AVERAGE GEODETIC DISTANCE BW NODES
    np.savetxt('outputs/out_'+qkd+'_rhos_geod' + suffix + '.dat', np.vstack((rhos, avg_geo_dist, ebar_geo_dist)) )
    # AVERAGE TOPOLOGICAL DISTANCE BW NODES
    np.savetxt('outputs/out_'+qkd+'_rhos_topd' + suffix + '.dat', np.vstack((rhos, avg_topolog_dist, ebar_topolog_dist)) )
    # CLUSTERING COEFFICIENT
    clus_data = np.vstack(( rhos, np.average(np.array(clustering_coeffs), axis=0), 2*np.std(np.array(clustering_coeffs), axis=0, ddof=1)/np.sqrt(n_iter)))
    np.savetxt('outputs/out_'+qkd+'_rhos_clus' + suffix + '.dat', clus_data)
    # DEGREE DISTRIBUTION
    k_max = max( len(dict_of_avg_degree_distribs[radius].keys()) for radius in radii )
    degree_distr_array = np.zeros((k_max+1, len(rhos)))
    degree_distr_array[0] = rhos
    for r in range(len(radii)):
        degree_histo = dict_of_avg_degree_distribs[radii[r]]
        for k in degree_histo.keys():
            degree_distr_array[1+k,r] = degree_histo[k]
    np.savetxt('outputs/out_'+qkd+'_rhos_degr' + suffix + '.dat', degree_distr_array)


    ########################################## quick plot for immediate feedback ###########################################
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$\rho=\frac{N}{4\pi R^2}$ [km$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'avg % nodes in giant comp. $\frac{\langle N_{GC}\rangle}{N}$')
    ax1.plot(rhos, giant_ratio_avg, 'o--', label=r'$\langle N_{GC}\rangle/N$')
    ax1.axhline(y=1, ls='--', color = 'red', alpha=0.4)
    ax1.errorbar(rhos, giant_ratio_avg, yerr=giant_ratio_ebar, fmt=' ', color='tab:blue')
    ax1.grid()
    if compute_nw_rates:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'avg rate $\langle K \rangle$')
        ax2.plot(rhos, network_rate_avg, '^-', color='tab:orange', label=r'$\langle K \rangle$')
        ax2.errorbar(rhos, network_rate_avg, yerr=network_rate_ebar, fmt=' ', color='tab:orange')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        plt.legend()
    if qkd == 'CV':
        plt.title(qkd+', N=%d, $\varepsilon_B=$%.2f, ' % (N, eps_B) + ('coords from dMercator' if sample_from_file else 'random coords'))
    elif qkd == 'DV':
        plt.title(qkd+', N=%d, QBER=%.2f, $p_{dc}=$%.2E' % (N, state_of_the_art_params.q, p_darkcount))
    plt.savefig('outputs/out_'+qkd+'_quickplot' + suffix + '.png', dpi=300)

end = time.time()
print('Execution took %.f seconds.' % (end - start))
