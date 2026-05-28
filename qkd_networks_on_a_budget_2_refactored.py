import matplotlib.pyplot as plt
import os
import numpy as np
from network_funcs import *
from qopt_funcs import *
import networkx as nx
import time
import shutil    # to copy files at the end of the script

# WHAT DOES THIS CODE DO?
# It generates instances of complex networks (nw), sweeping node densities by changing system size.
# Computed quantities: connectivity, susceptibility, average key rate, average geodetic and topological
# distances, degree distribution, clustering coeff.
# The version of the QKD protocol used (CV/DV/hybrid) can be changed through the variable 'd_hybrid' below.
# WHAT CHANGES WRT qkd_networks.py ? We analyze the properties of the hybrid nw for different "budgets"
# of DV links.
#
# This refactored version generates each S2 network instance once, caches the base edge list,
# prunes it once per radius, ranks the candidate DV edges once per criterion, and then evaluates
# all requested DV budgets from the same ranked edge list.

Ns = [2000]                    # list of network sizes

rate_min = 0
n_nodes_for_dijkstra = 20
compute_nw_rates = False      # if False only connectivity of the network is computed (to reduce runtime)
edges_may_fail = False        # if True, an edge fails with prob PoF
PoF = 0.1                     # probability of failure of an edge
keyrate_algo = 'parallel'     # 'parallel' or 'serial' (dijkstra)
beta = 2.6261                 # \beta param of S2 model
mu = 0.0233                   # \mu param of S2 model
sample_from_file = False      # if True, coordinates are sampled from the results of d-Mercator
detection_mode = 'homodyne'   # homo-/hetero-dyne
reconciliation = 'reverse'    # type of reconciliation
budget_list = [0, 1, 4, 16, 32, 64, 128, 256, 512, 1024, np.inf]
candidate_ranking_criteria = [
    'degree',
    'centrality',
    'geodetic_distance',
    'topological_distance',
    'random',
]

####### The state-of-the-art values for the following params are usually kept fixed and assigned in qopt_funcs.py ##########
# params for the qkd rates:
alpha = state_of_the_art_params.alpha
freq = state_of_the_art_params.freq
# CV-specific
eps_B = state_of_the_art_params.eps_B
eta_source_CV = state_of_the_art_params.eta_source_CV
eta_det_CV = state_of_the_art_params.eta_det_CV
T_A = state_of_the_art_params.T_A
r_A = state_of_the_art_params.r_A
# DV-specific
eta_source_DV = state_of_the_art_params.eta_source_DV
eta_det_DV = state_of_the_art_params.eta_det_DV
R_dark = state_of_the_art_params.R_dark
deltat_det = state_of_the_art_params.deltat_det
p_darkcount = state_of_the_art_params.p_darkcount
q = state_of_the_art_params.q

# Defining the set of node densities to be simulated
rho_span = '_wide'              # providing some presets of points for the plots: '_focus', '_wide' or anything else
if rho_span == '_focus':        # for Fig. 1b
    n_iter, n_couples = 40, 10
    rhos = 0.14 * 10 ** np.linspace(-2.3, -1.9, 20)
elif rho_span == '_wide':
    n_iter, n_couples = 10, 10
    rhos = 0.14 * 10 ** np.linspace(-4., 1., 50)
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


def filename_float(x):
    return ('%.3g' % x).replace('.', 'p')


def budget_label(DV_budget):
    if np.isinf(DV_budget):
        return 'Ball'
    return 'B%d' % int(DV_budget)


def output_suffix(N, DV_budget, ranking_criterion):
    suffix = '_N%d_%s_%s' % (N, budget_label(DV_budget), ranking_criterion)
    suffix += '_sampled' if sample_from_file else ''
    suffix += '_pof%s' % filename_float(PoF) if edges_may_fail else ''
    return suffix


def init_budget_stats(radii):
    return {
        'giant_ratio': np.zeros((n_iter, len(radii))),
        'clustering_coeffs': np.zeros((n_iter, len(radii))),
        'geo_dist_lists': {radius: [] for radius in radii},
        'rate_lists_dijkstra': {radius: [] for radius in radii},
        'len_lists_dijkstra': {radius: [] for radius in radii},
        'avg_degree_distribs': {radius: {} for radius in radii},
        'avg_topological_dists': {radius: [] for radius in radii},
        'num_DV_edges': {radius: [] for radius in radii},
    }


def compute_keyrates(d_hybrid, d_c_DV, d_c_CV):
    n_ds = 10000
    d_set = np.linspace(1. / n_ds, max(d_c_DV, d_c_CV), n_ds)
    keyrates = np.zeros((n_ds,))
    for i, d in enumerate(d_set):
        keyrates[i] = hybrid_keyrate_bitpersec(state_of_the_art_params, d, d_hybrid)
    return d_set, keyrates


def relabel_edges_to_component(edges, component_nodes):
    component_nodes = sorted(component_nodes)
    node_map = {old_node: new_node for new_node, old_node in enumerate(component_nodes)}
    component_set = set(component_nodes)
    relabeled_edges = []

    for i, j, distance in edges:
        i = int(i)
        j = int(j)
        if i in component_set and j in component_set:
            relabeled_edges.append((node_map[i], node_map[j], distance))

    relabeled_edges = np.array(relabeled_edges, dtype=float)
    if relabeled_edges.size == 0:
        relabeled_edges = relabeled_edges.reshape((0, 3))
    return relabeled_edges, component_nodes


def largest_component_from_edges(N, edges):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from((int(i), int(j)) for i, j, distance in edges)
    return list(max(nx.connected_components(G), key=len))


def generate_edges_for_new_nodes(coords, k, start_node, N, beta, mu, D=2, original_refill_formula=False):
    edges = []
    R = np.sqrt(N / 4 / np.pi)

    for i in range(start_node, N):
        for j in range(i):
            distance = angular_dist_in_sphere(coords[i], coords[j])
            toss = np.random.uniform(0, 1)

            if original_refill_formula:
                denom = mu * k[i] * k[j]
            else:
                denom = (mu * k[i] * k[j]) ** (1. / D)

            pij = (1 + (R * distance / denom) ** beta) ** -1
            if toss < pij:
                edges.append((i, j, distance))

    edges = np.array(edges, dtype=float)
    if edges.size == 0:
        edges = edges.reshape((0, 3))
    return edges


def S2_graph_definite_N_edges(N, beta, mu, D=2, sample_from_file=False, return_coords=False):
    gamma = 2.3

    if sample_from_file:
        fname = '../mercator_data/as20000102.inf_coord'
        pool = np.loadtxt(fname)
        np.random.shuffle(pool)
        k_pool = pool[:, 1]
        coords_pool = pool[:, 3:]
        coords_pool /= np.sqrt(np.sum(coords_pool ** 2, axis=1))[:, np.newaxis]
        coords = coords_pool[:N]
        k = k_pool[:N]
        bookmark = N
    else:
        coords = uniform_unit_sphere_distribution(N)
        k = powerlaw_rnd_var(gamma, N)
        bookmark = N

    edges = generate_edges_for_new_nodes(
        coords,
        k,
        0,
        N,
        beta,
        mu,
        D=D,
        original_refill_formula=False
    )

    component_nodes = largest_component_from_edges(N, edges)
    edges, component_nodes = relabel_edges_to_component(edges, component_nodes)
    coords = coords[component_nodes]
    k = k[component_nodes]
    N_old = len(component_nodes)

    while N_old < N:
        n = N - N_old

        if sample_from_file:
            new_coords = coords_pool[bookmark:bookmark + n]
            new_k = k_pool[bookmark:bookmark + n]
            bookmark += n
        else:
            new_coords = uniform_unit_sphere_distribution(n)
            new_k = powerlaw_rnd_var(gamma, n)

        coords = np.vstack((coords, new_coords))
        k = np.hstack((k, new_k))
        new_edges = generate_edges_for_new_nodes(
            coords,
            k,
            N_old,
            N,
            beta,
            mu,
            D=D,
            original_refill_formula=True
        )

        if len(edges) == 0:
            edges = new_edges
        elif len(new_edges) > 0:
            edges = np.vstack((edges, new_edges))

        component_nodes = largest_component_from_edges(N, edges)
        edges, component_nodes = relabel_edges_to_component(edges, component_nodes)
        coords = coords[component_nodes]
        k = k[component_nodes]
        N_old = len(component_nodes)

    if return_coords:
        return edges, coords
    return edges


def cached_base_graph_filename(graph_dir, N, it):
    sample_suffix = '_sampled' if sample_from_file else ''
    return os.path.join(
        graph_dir,
        'base_edges_N%d_it%04d%s.npz' % (N, it, sample_suffix)
    )


def load_or_generate_base_graph(graph_dir, N, it):
    filename = cached_base_graph_filename(graph_dir, N, it)

    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=False)
        print('Loaded base graph cache: %s' % filename)
        return data['base_edges'], data['coords']

    base_edges, coords = S2_graph_definite_N_edges(
        N,
        beta,
        mu,
        sample_from_file=sample_from_file,
        return_coords=True
    )
    np.savez_compressed(filename, base_edges=base_edges, coords=coords)
    print('Saved base graph cache: %s' % filename)
    return base_edges, coords


def build_pruned_graph_arrays(base_edges, radius, d_set, keyrates):
    pruned_edges = []
    cv_edges = []
    candidate_DV_edges = []
    geo_distances = []

    for i, j, normalized_distance in base_edges:
        i = int(i)
        j = int(j)
        dij = radius * normalized_distance
        idx_d = np.argmin(abs(dij - d_set))
        h_rate = keyrates[idx_d]

        if dij < d_c_DV and h_rate > rate_min:
            if np.random.uniform() > PoF * int(edges_may_fail):
                weight = h_rate ** -1
                pruned_edges.append((i, j, weight))
                geo_distances.append(dij)

                if dij <= d_c_CV:
                    cv_edges.append((i, j, weight))
                elif dij < d_c_DV:
                    candidate_DV_edges.append((i, j, weight, dij))

    avg_geo_distance = np.average(geo_distances) if len(geo_distances) > 0 else 0

    pruned_edges = np.array(pruned_edges, dtype=float)
    cv_edges = np.array(cv_edges, dtype=float)
    candidate_DV_edges = np.array(candidate_DV_edges, dtype=float)
    if pruned_edges.size == 0:
        pruned_edges = pruned_edges.reshape((0, 3))
    if cv_edges.size == 0:
        cv_edges = cv_edges.reshape((0, 3))
    if candidate_DV_edges.size == 0:
        candidate_DV_edges = candidate_DV_edges.reshape((0, 4))

    return pruned_edges, cv_edges, candidate_DV_edges, avg_geo_distance


def rank_candidate_dv_edges(candidate_DV_edges, G_pruned, ranking_criterion):
    if ranking_criterion == 'centrality':
        edge_centrality = nx.edge_betweenness_centrality(G_pruned)
        edge_centrality = {
            tuple(sorted(edge)): c
            for edge, c in edge_centrality.items()
        }
    else:
        edge_centrality = {}

    candidate_DV_edges = [
        (min(int(i), int(j)), max(int(i), int(j)), float(weight), float(geo_distance))
        for i, j, weight, geo_distance in candidate_DV_edges
    ]

    def dv_edge_score(edge):
        i, j, weight, geo_distance = edge

        if ranking_criterion == 'centrality':
            return edge_centrality.get((i, j), 0)

        if ranking_criterion == 'random':
            return np.random.random()

        if ranking_criterion == 'geodetic_distance':
            return geo_distance

        if ranking_criterion == 'topological_distance':
            try:
                return nx.shortest_path_length(G_pruned, i, j)
            except nx.NetworkXNoPath:
                return np.inf

        if ranking_criterion == 'degree':
            return G_pruned.degree(i) * G_pruned.degree(j)

        return edge_centrality.get((i, j), 0)

    reverse_sort = ranking_criterion in ['centrality', 'random', 'degree']
    return sorted(candidate_DV_edges, key=dv_edge_score, reverse=reverse_sort)


def save_budget_outputs(
    DV_budget,
    suffix,
    radii,
    rhos,
    qkd,
    stats,
):
    giant_ratio = stats['giant_ratio']
    clustering_coeffs = stats['clustering_coeffs']
    dict_of_geo_dist_lists = stats['geo_dist_lists']
    dict_of_rate_lists_dijkstra = stats['rate_lists_dijkstra']
    dict_of_len_lists_dijkstra = stats['len_lists_dijkstra']
    dict_of_avg_degree_distribs = stats['avg_degree_distribs']
    dict_of_avg_topological_dists = stats['avg_topological_dists']
    dict_of_num_DV_edges = stats['num_DV_edges']

    giant_ratio_avg = np.average(giant_ratio, axis=0)
    giant_ratio_ebar = 2 * np.std(giant_ratio, axis=0, ddof=1) / np.sqrt(n_iter)

    avg_shortest_path_lens = np.zeros_like(radii)
    ebar_shortest_path_lens = np.zeros_like(radii)
    avg_geo_dist = np.zeros_like(radii)
    ebar_geo_dist = np.zeros_like(radii)
    avg_topolog_dist = np.zeros_like(radii)
    ebar_topolog_dist = np.zeros_like(radii)

    for r in range(len(radii)):
        radius = radii[r]

        geo_values = np.array(dict_of_geo_dist_lists[radius])
        top_values = np.array(dict_of_avg_topological_dists[radius])

        avg_geo_dist[r] = np.average(geo_values)
        ebar_geo_dist[r] = np.std(geo_values, ddof=1) / np.sqrt(len(geo_values))

        avg_topolog_dist[r] = np.average(top_values)
        ebar_topolog_dist[r] = np.std(top_values, ddof=1) / np.sqrt(len(top_values))

    if compute_nw_rates:
        network_rate_avg = np.zeros_like(radii)
        network_rate_ebar = np.zeros_like(radii)

        for r in range(len(radii)):
            radius = radii[r]

            rate_values = np.array(dict_of_rate_lists_dijkstra[radius])
            len_values = np.array(dict_of_len_lists_dijkstra[radius])

            network_rate_avg[r] = np.average(rate_values)
            network_rate_ebar[r] = np.std(rate_values, ddof=1) / np.sqrt(len(rate_values))

            avg_shortest_path_lens[r] = np.average(len_values)
            ebar_shortest_path_lens[r] = np.std(len_values, ddof=1) / np.sqrt(n_iter)

        if keyrate_algo == 'serial':
            np.savetxt('rate' + suffix + '.dat',
                       np.vstack((rhos, network_rate_avg, network_rate_ebar)))
            np.savetxt('aspl_dijk' + suffix + '.dat',
                       np.vstack((rhos, avg_shortest_path_lens, ebar_shortest_path_lens)))
        elif keyrate_algo == 'parallel':
            np.savetxt('parK' + suffix + '.dat',
                       np.vstack((rhos, network_rate_avg, network_rate_ebar)))
            np.savetxt('aspl' + suffix + '.dat',
                       np.vstack((rhos, avg_shortest_path_lens, ebar_shortest_path_lens)))

    np.savetxt('conn' + suffix + '.dat',
               np.vstack((rhos, giant_ratio_avg, giant_ratio_ebar)))

    np.savetxt('geod' + suffix + '.dat',
               np.vstack((rhos, avg_geo_dist, ebar_geo_dist)))

    np.savetxt('topd' + suffix + '.dat',
               np.vstack((rhos, avg_topolog_dist, ebar_topolog_dist)))

    clus_data = np.vstack((
        rhos,
        np.average(clustering_coeffs, axis=0),
        2 * np.std(clustering_coeffs, axis=0, ddof=1) / np.sqrt(n_iter)
    ))
    np.savetxt('clus' + suffix + '.dat', clus_data)

    k_max = max(len(dict_of_avg_degree_distribs[radius].keys()) for radius in radii)
    degree_distr_array = np.zeros((k_max + 1, len(rhos)))
    degree_distr_array[0] = rhos

    for r in range(len(radii)):
        degree_histo = dict_of_avg_degree_distribs[radii[r]]
        for k in degree_histo.keys():
            degree_distr_array[1 + k, r] = degree_histo[k]

    np.savetxt('degr' + suffix + '.dat', degree_distr_array)

    np.savetxt('budg' + suffix + '.dat',
               np.vstack((
                   rhos,
                   [np.average(dict_of_num_DV_edges[radius]) for radius in radii]
               )))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$\rho=\frac{N}{4\pi R^2}$ [km$^{-2}$]')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'avg % nodes in giant comp. $\frac{\langle N_{GC}\rangle}{N}$')
    ax1.plot(rhos, giant_ratio_avg, 'o--', label=r'$\langle N_{GC}\rangle/N$')
    ax1.axhline(y=1, ls='--', color='red', alpha=0.4)
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
        plt.title(qkd + ', N=%d, $\\epsilon_B=$%.2f, ' % (N, eps_B) +
                  ('coords from dMercator' if sample_from_file else 'random coords'))
    elif qkd == 'DV':
        plt.title(qkd + ', N=%d, QBER=%.2f, $p_{dc}=$%.2E' %
                  (N, state_of_the_art_params.q, p_darkcount))

    plt.savefig('quick' + suffix + '.png', dpi=300)
    plt.close(fig)


# computing critical distances, needed later for pruning
d_max = 1000
func_CV = lambda dist: hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid=float('inf'))
d_c_CV = bisection_solver(func_CV, 10E-06, d_max)
func_DV = lambda dist: hybrid_keyrate_bitpersec(state_of_the_art_params, dist, d_hybrid=0)
d_c_DV = bisection_solver(func_DV, 10E-06, d_max)
diff = lambda d: func_CV(d) - func_DV(d)
d_cross = bisection_solver(diff, 10E-06, d_max)

# The 'd_hybrid' variable can be changed to any value. Of particular interest are:
# float('inf') to force CV-only networks, 0 to force DV-only nws, d_cross for 'optimal' hybrid protocol
d_hybrid = d_cross

if not edges_may_fail:
    PoF = 0.

qkd = 'hybrid'
print('Coordinates are sampled ' + ('from a pool inferred from a real network through dMercator.' if sample_from_file else 'randomly.'))
print('Rates are ' + ('' if compute_nw_rates else 'not ') + 'computed.')
if edges_may_fail:
    print('Each edge may fail with probability %.3f' % PoF)


start = time.time()

base_dir = os.getcwd()
os.makedirs("budget_outputs", exist_ok=True)

for N in Ns:
    radii = np.sqrt(N / 4 / np.pi / rhos)

    print('%d instances of a ' % n_iter + qkd + '-QKD network with %d nodes are generated, ' % N +
          '%d pairs of nodes are sampled per instance to compute the rates.' % (n_nodes_for_dijkstra * N))
    print('Evaluating for d_hybrid = %.2f km...' % d_hybrid)
    print('Analysis is iterated for the following DV budgets: %s' % str(budget_list))
    print('Analysis is iterated for the following ranking criteria: %s' % str(candidate_ranking_criteria))

    folder = os.path.join("budget_outputs", "N%d" % N)
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    graph_dir = "graphs"
    os.makedirs(graph_dir, exist_ok=True)

    suffix_by_criterion_budget = {
        ranking_criterion: {
            DV_budget: output_suffix(N, DV_budget, ranking_criterion)
            for DV_budget in budget_list
        }
        for ranking_criterion in candidate_ranking_criteria
    }

    budgets_to_run_by_criterion = {}
    for ranking_criterion in candidate_ranking_criteria:
        budgets_to_run = []

        for DV_budget in budget_list:
            suffix = suffix_by_criterion_budget[ranking_criterion][DV_budget]
            budg_filename = 'budg' + suffix + '.dat'

            if os.path.exists(budg_filename):
                print('Skipping criterion=%s, DV_budget=%s: %s already exists.' %
                      (ranking_criterion, str(DV_budget), budg_filename))
            else:
                budgets_to_run.append(DV_budget)

        if len(budgets_to_run) > 0:
            budgets_to_run_by_criterion[ranking_criterion] = budgets_to_run

    if len(budgets_to_run_by_criterion) == 0:
        os.chdir(base_dir)
        continue

    stats_by_criterion_budget = {
        ranking_criterion: {
            DV_budget: init_budget_stats(radii)
            for DV_budget in budgets_to_run
        }
        for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items()
    }

    # Key rates depend only on the physical parameters and d_hybrid, so compute them once per N.
    d_set, keyrates = compute_keyrates(d_hybrid, d_c_DV, d_c_CV)

    for it in range(n_iter):
        print('N=%d, instance %d, ranking_criteria=%s' %
              (N, it, str(list(budgets_to_run_by_criterion.keys()))))

        # One starting network instance, reused for all budgets and ranking criteria.
        base_edges, coords = load_or_generate_base_graph(graph_dir, N, it)

        n_nodes_giant_by_criterion_budget = {
            ranking_criterion: {
                DV_budget: []
                for DV_budget in budgets_to_run
            }
            for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items()
        }
        clustering_list_by_criterion_budget = {
            ranking_criterion: {
                DV_budget: []
                for DV_budget in budgets_to_run
            }
            for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items()
        }

        for radius in radii:
            pruned_edges, cv_edges, candidate_DV_edges, avg_geo_distance = build_pruned_graph_arrays(
                base_edges,
                radius,
                d_set,
                keyrates
            )
            G_pruned = nx.Graph()
            G_pruned.add_nodes_from(range(N))
            G_pruned.add_weighted_edges_from(
                (int(i), int(j), float(weight))
                for i, j, weight in pruned_edges
            )

            G_CV = nx.Graph()
            G_CV.add_nodes_from(range(N))
            G_CV.add_weighted_edges_from(
                (int(i), int(j), float(weight))
                for i, j, weight in cv_edges
            )

            for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items():
                sorted_candidate_edges = rank_candidate_dv_edges(
                    candidate_DV_edges,
                    G_pruned,
                    ranking_criterion
                )

                number_possible_DV_edges = len(sorted_candidate_edges)

                for DV_budget in budgets_to_run:
                    stats = stats_by_criterion_budget[ranking_criterion][DV_budget]
                    stats['num_DV_edges'][radius].append(number_possible_DV_edges)
                    stats['geo_dist_lists'][radius].append(avg_geo_distance)

                    if np.isinf(DV_budget):
                        n_DV_to_add = number_possible_DV_edges
                    else:
                        n_DV_to_add = int(min(DV_budget, number_possible_DV_edges))

                    DV_edges = [
                        (i, j, weight)
                        for i, j, weight, geo_distance in sorted_candidate_edges[:n_DV_to_add]
                    ]

                    G_budget = G_CV.copy()
                    G_budget.add_weighted_edges_from(DV_edges)

                    comp_list = sorted(nx.connected_components(G_budget), key=len, reverse=True)

                    if len(comp_list) > 0 and len(comp_list[0]) > 0:
                        G_giant = G_budget.subgraph(comp_list[0])
                        n_nodes_giant = nx.number_of_nodes(G_giant)
                    else:
                        G_giant = G_budget
                        n_nodes_giant = 0

                    n_nodes_giant_by_criterion_budget[ranking_criterion][DV_budget].append(n_nodes_giant)

                    # Jasper's correction to nx.average_clustering
                    clustering_coefficients = nx.clustering(G_budget)
                    nodes_with_degree_gt_1 = [
                        v for v in dict(G_budget.degree).values()
                        if int(v) > 1
                    ]

                    if len(nodes_with_degree_gt_1) != 0:
                        average_clustering_coefficient = (
                            sum(clustering_coefficients.values()) /
                            len(nodes_with_degree_gt_1)
                        )
                    else:
                        average_clustering_coefficient = 0

                    clustering_list_by_criterion_budget[ranking_criterion][DV_budget].append(
                        average_clustering_coefficient
                    )

                    deg_hist = nx.degree_histogram(G_budget)
                    for deg in range(len(deg_hist)):
                        if deg in stats['avg_degree_distribs'][radius]:
                            stats['avg_degree_distribs'][radius][deg] += float(deg_hist[deg]) / n_iter
                        else:
                            stats['avg_degree_distribs'][radius][deg] = float(deg_hist[deg]) / n_iter

                    if compute_nw_rates and n_nodes_giant > 0:
                        node_counter = 0
                        giant_nodes = list(comp_list[0])
                        node_max = min(len(giant_nodes), n_nodes_for_dijkstra)

                        while node_counter < node_max:
                            target = giant_nodes[node_counter]
                            weights, paths = optimal_path_algo(G_budget, target, algo=keyrate_algo)

                            for source in range(target):
                                if source in weights:
                                    stats['rate_lists_dijkstra'][radius].append(weights[source] ** -1)
                                    stats['len_lists_dijkstra'][radius].append(len(paths[source]) - 1)
                                else:
                                    stats['rate_lists_dijkstra'][radius].append(0)

                            node_counter += 1

                    if n_nodes_giant > 1:
                        stats['avg_topological_dists'][radius].append(
                            nx.average_shortest_path_length(G_giant)
                        )
                    else:
                        stats['avg_topological_dists'][radius].append(0)

        for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items():
            for DV_budget in budgets_to_run:
                stats = stats_by_criterion_budget[ranking_criterion][DV_budget]
                stats['giant_ratio'][it] = (
                    np.array(n_nodes_giant_by_criterion_budget[ranking_criterion][DV_budget]) /
                    float(N)
                )
                stats['clustering_coeffs'][it] = np.array(
                    clustering_list_by_criterion_budget[ranking_criterion][DV_budget]
                )

    for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items():
        for DV_budget in budgets_to_run:
            save_budget_outputs(
                DV_budget,
                suffix_by_criterion_budget[ranking_criterion][DV_budget],
                radii,
                rhos,
                qkd,
                stats_by_criterion_budget[ranking_criterion][DV_budget]
            )

    os.chdir(base_dir)

'''# needed for execution on the cluster: move the already saved files in a local folder
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if os.path.abspath(output_dir) != os.path.abspath(os.path.join(output_dir, 'outputs')):
    for filename in os.listdir('.'):
        if filename.startswith('out') and os.path.isfile(filename):
            shutil.move(filename, os.path.join(output_dir, filename))
'''

end = time.time()
print('Execution took %.f seconds.' % (end - start))
