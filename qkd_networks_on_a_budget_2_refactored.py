import matplotlib.pyplot as plt
import os
import numpy as np
from network_funcs import *
from qopt_funcs import *
import networkx as nx
import time
import shutil    # to copy files at the end of the script

from budget_config import *

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

Ns = [500]                    # list of network sizes

cfg = SimulationConfig(
    rate_min=0.0,
    n_nodes_for_dijkstra=20,
    compute_nw_rates=False,
    edges_may_fail=False,
    PoF=0.1,
    keyrate_algo='parallel',
    beta=2.6261,
    mu=0.0233,
    sample_from_file=False,
)
detection_mode = 'homodyne'   # homo-/hetero-dyne
reconciliation = 'reverse'    # type of reconciliation
budget_list = cfg.budget_list
print(budget_list)
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

cfg.n_iter = n_iter




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

if not cfg.edges_may_fail:
    cfg.PoF = 0.

qkd = 'hybrid'
print('Coordinates are sampled ' + ('from a pool inferred from a real network through dMercator.' if cfg.sample_from_file else 'randomly.'))
print('Rates are ' + ('' if cfg.compute_nw_rates else 'not ') + 'computed.')
if cfg.edges_may_fail:
    print('Each edge may fail with probability %.3f' % cfg.PoF)


start = time.time()

base_dir = os.getcwd()
os.makedirs("budget_outputs", exist_ok=True)

for N in Ns:
    radii = np.sqrt(N / 4 / np.pi / rhos)

    print('%d instances of a ' % cfg.n_iter + qkd + '-QKD network with %d nodes are generated, ' % N +
          '%d pairs of nodes are sampled per instance to compute the rates.' % (cfg.n_nodes_for_dijkstra * N))
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
            DV_budget: output_suffix(N, DV_budget, ranking_criterion, cfg)
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
            DV_budget: init_budget_stats(radii, cfg.n_iter)
            for DV_budget in budgets_to_run
        }
        for ranking_criterion, budgets_to_run in budgets_to_run_by_criterion.items()
    }

    # Key rates depend only on the physical parameters and d_hybrid, so compute them once per N.
    d_set, keyrates = compute_keyrates(d_hybrid, d_c_DV, d_c_CV, state_of_the_art_params)

    for it in range(cfg.n_iter):
        print('N=%d, instance %d, ranking_criteria=%s' %
              (N, it, str(list(budgets_to_run_by_criterion.keys()))))

        # One starting network instance, reused for all budgets and ranking criteria.
        base_edges, coords = load_or_generate_base_graph(graph_dir, N, it, cfg)

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
                keyrates,
                cfg,
                d_c_DV,
                d_c_CV,
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

                    eigvals = laplacian_spectrum(G_budget, cfg.laplacian_spectrum_n_eigs)
                    positive = eigvals[eigvals > 1e-12]
                    fiedler = float(positive.min()) if positive.size else 0.0
                    stats['fiedler_values'][radius].append(fiedler)
                    stats['laplacian_spectrum_values'][radius].append(eigvals)

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
                            stats['avg_degree_distribs'][radius][deg] += float(deg_hist[deg]) / cfg.n_iter
                        else:
                            stats['avg_degree_distribs'][radius][deg] = float(deg_hist[deg]) / cfg.n_iter

                    if cfg.compute_nw_rates and n_nodes_giant > 0:
                        node_counter = 0
                        giant_nodes = list(comp_list[0])
                        node_max = min(len(giant_nodes), cfg.n_nodes_for_dijkstra)

                        while node_counter < node_max:
                            target = giant_nodes[node_counter]
                            weights, paths = optimal_path_algo(G_budget, target, algo=cfg.keyrate_algo)

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
                stats_by_criterion_budget[ranking_criterion][DV_budget],
                cfg,
                N,
                eps_B,
                p_darkcount,
                state_of_the_art_params,
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
