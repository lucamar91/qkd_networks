from dataclasses import dataclass, field
import matplotlib.pyplot as plt

import numpy as np
import scipy
from network_funcs import *
from qopt_funcs import *
import networkx as nx



@dataclass
class SimulationConfig:
    rate_min: float = 0.0
    n_nodes_for_dijkstra: int = 20
    compute_nw_rates: bool = False
    edges_may_fail: bool = False
    PoF: float = 0.1
    keyrate_algo: str = 'parallel'
    beta: float = 2.6261
    mu: float = 0.0233
    sample_from_file: bool = False
    n_iter: int = 10
    laplacian_spectrum_n_eigs: int = 100
    budget_list: list = field(default_factory=lambda: [0] + [2 ** expo for expo in range(4, 11, 1)] + [np.inf])


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


def output_suffix(N, DV_budget, ranking_criterion, cfg):
    suffix = '_N%d_%s_%s' % (N, budget_label(DV_budget), ranking_criterion)
    suffix += '_sampled' if cfg.sample_from_file else ''
    suffix += '_pof%s' % filename_float(cfg.PoF) if cfg.edges_may_fail else ''
    return suffix


def init_budget_stats(radii, n_iter):
    return {
        'giant_ratio': np.zeros((n_iter, len(radii))),
        'clustering_coeffs': np.zeros((n_iter, len(radii))),
        'geo_dist_lists': {radius: [] for radius in radii},
        'rate_lists_dijkstra': {radius: [] for radius in radii},
        'len_lists_dijkstra': {radius: [] for radius in radii},
        'avg_degree_distribs': {radius: {} for radius in radii},
        'avg_topological_dists': {radius: [] for radius in radii},
        'fiedler_values': {radius: [] for radius in radii},
        'laplacian_spectrum_values': {radius: [] for radius in radii},
        'num_DV_edges': {radius: [] for radius in radii},
    }


def compute_keyrates(d_hybrid, d_c_DV, d_c_CV, state_of_the_art_params):
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


def cached_base_graph_filename(graph_dir, N, it, cfg):
    sample_suffix = '_sampled' if cfg.sample_from_file else ''
    return os.path.join(
        graph_dir,
        'base_edges_N%d_it%04d%s.npz' % (N, it, sample_suffix)
    )


def load_or_generate_base_graph(graph_dir, N, it, cfg):
    filename = cached_base_graph_filename(graph_dir, N, it, cfg)

    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=False)
        print('Loaded base graph cache: %s' % filename)
        return data['base_edges'], data['coords']

    base_edges, coords = S2_graph_definite_N_edges(
        N,
        cfg.beta,
        cfg.mu,
        sample_from_file=cfg.sample_from_file,
        return_coords=True
    )
    np.savez_compressed(filename, base_edges=base_edges, coords=coords)
    print('Saved base graph cache: %s' % filename)
    return base_edges, coords


def build_pruned_graph_arrays(base_edges, radius, d_set, keyrates, cfg, d_c_DV, d_c_CV):
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

        if dij < d_c_DV and h_rate > cfg.rate_min:
            if np.random.uniform() > cfg.PoF * int(cfg.edges_may_fail):
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


def laplacian_spectrum(G, k):
    k = min(k, G.number_of_nodes() - 2)
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes()))
    # laplacian_spectrum = scipy.sparse.linalg.eigsh(L.toarray(), k=k, which='SM')
    laplacian_spectrum = np.linalg.eigvalsh(L.toarray())
    return laplacian_spectrum


def laplacian_spectrum_cdf(spectrum_values, n_bins=100):
    spectrum_values = [np.asarray(v, dtype=float).ravel() for v in spectrum_values]
    values = np.concatenate([v for v in spectrum_values if v.size > 0])
    values = np.real(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.zeros(n_bins, dtype=float)

    values = np.clip(values, 1e-6, 100.0)
    bin_edges = np.logspace(-6, 2, num=n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total <= 0:
        return np.zeros(n_bins, dtype=float)

    return np.cumsum(counts).astype(float) / float(total)


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

        if ranking_criterion == 'increasing_geodetic_distance' or ranking_criterion == 'decreasing_geodetic_distance':
            return geo_distance

        if ranking_criterion == 'increasing_topological_distance' or ranking_criterion == 'decreasing_topological_distance':
            try:
                return nx.shortest_path_length(G_pruned, i, j)
            except nx.NetworkXNoPath:
                return np.inf

        if ranking_criterion == 'degree':
            return G_pruned.degree(i) * G_pruned.degree(j)

        return edge_centrality.get((i, j), 0)

    reverse_sort = ranking_criterion in ['centrality', 'random', 'degree', 'decreasing_geodetic_distance', 'decreasing_topological_distance']
    return sorted(candidate_DV_edges, key=dv_edge_score, reverse=reverse_sort)


def save_budget_outputs(
    DV_budget,
    suffix,
    radii,
    rhos,
    qkd,
    stats,
    cfg,
    N,
    eps_B,
    p_darkcount,
    state_of_the_art_params,
):
    giant_ratio = stats['giant_ratio']
    clustering_coeffs = stats['clustering_coeffs']
    dict_of_geo_dist_lists = stats['geo_dist_lists']
    dict_of_rate_lists_dijkstra = stats['rate_lists_dijkstra']
    dict_of_len_lists_dijkstra = stats['len_lists_dijkstra']
    dict_of_avg_degree_distribs = stats['avg_degree_distribs']
    dict_of_avg_topological_dists = stats['avg_topological_dists']
    dict_of_fiedler_values = stats['fiedler_values']
    dict_of_laplacian_spectrum_values = stats['laplacian_spectrum_values']
    dict_of_num_DV_edges = stats['num_DV_edges']

    giant_ratio_avg = np.average(giant_ratio, axis=0)
    giant_ratio_ebar = 2 * np.std(giant_ratio, axis=0, ddof=1) / np.sqrt(cfg.n_iter)

    avg_shortest_path_lens = np.zeros_like(radii)
    ebar_shortest_path_lens = np.zeros_like(radii)
    avg_geo_dist = np.zeros_like(radii)
    ebar_geo_dist = np.zeros_like(radii)
    avg_topolog_dist = np.zeros_like(radii)
    ebar_topolog_dist = np.zeros_like(radii)
    avg_fiedler = np.zeros_like(radii)
    ebar_fiedler = np.zeros_like(radii)

    for r, radius in enumerate(radii):
        geo_values = np.array(dict_of_geo_dist_lists[radius])
        top_values = np.array(dict_of_avg_topological_dists[radius])
        fiedler_values = np.array(dict_of_fiedler_values[radius])

        avg_geo_dist[r] = np.average(geo_values)
        ebar_geo_dist[r] = np.std(geo_values, ddof=1) / np.sqrt(len(geo_values))

        avg_topolog_dist[r] = np.average(top_values)
        ebar_topolog_dist[r] = np.std(top_values, ddof=1) / np.sqrt(len(top_values))

        avg_fiedler[r] = np.average(fiedler_values)
        ebar_fiedler[r] = np.std(fiedler_values, ddof=1) / np.sqrt(len(fiedler_values))

    if cfg.compute_nw_rates:
        network_rate_avg = np.zeros_like(radii)
        network_rate_ebar = np.zeros_like(radii)

        for r, radius in enumerate(radii):

            rate_values = np.array(dict_of_rate_lists_dijkstra[radius])
            len_values = np.array(dict_of_len_lists_dijkstra[radius])

            network_rate_avg[r] = np.average(rate_values)
            network_rate_ebar[r] = np.std(rate_values, ddof=1) / np.sqrt(len(rate_values))

            avg_shortest_path_lens[r] = np.average(len_values)
            ebar_shortest_path_lens[r] = np.std(len_values, ddof=1) / np.sqrt(cfg.n_iter)

        if cfg.keyrate_algo == 'serial':
            np.savetxt('rate' + suffix + '.dat',
                       np.vstack((rhos, network_rate_avg, network_rate_ebar)))
            np.savetxt('aspl_dijk' + suffix + '.dat',
                       np.vstack((rhos, avg_shortest_path_lens, ebar_shortest_path_lens)))
        elif cfg.keyrate_algo == 'parallel':
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

    np.savetxt('fiedler' + suffix + '.dat',
               np.vstack((rhos, avg_fiedler, ebar_fiedler)))

    clus_data = np.vstack((
        rhos,
        np.average(clustering_coeffs, axis=0),
        2 * np.std(clustering_coeffs, axis=0, ddof=1) / np.sqrt(cfg.n_iter)
    ))
    np.savetxt('clus' + suffix + '.dat', clus_data)

    k_max = max(len(dict_of_avg_degree_distribs[radius].keys()) for radius in radii)
    degree_distr_array = np.zeros((k_max + 1, len(rhos)))
    degree_distr_array[0] = rhos

    for r, radius in enumerate(radii):
        degree_histo = dict_of_avg_degree_distribs[radius]
        for k in degree_histo.keys():
            degree_distr_array[1 + k, r] = degree_histo[k]

    np.savetxt('degr' + suffix + '.dat', degree_distr_array)

    n_cdf_bins = 100
    cdf_matrix = np.zeros((n_cdf_bins, len(radii)), dtype=float)
    for r in range(len(radii)):
        radius = radii[r]
        cdf_matrix[:, r] = laplacian_spectrum_cdf(
            dict_of_laplacian_spectrum_values[radius],
            n_bins=n_cdf_bins,
        )

    np.savetxt('lapcdf' + suffix + '.dat', np.vstack((rhos, cdf_matrix)))

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

    if cfg.compute_nw_rates:
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
                  ('coords from dMercator' if cfg.sample_from_file else 'random coords'))
    elif qkd == 'DV':
        plt.title(qkd + ', N=%d, QBER=%.2f, $p_{dc}=$%.2E' %
                  (N, state_of_the_art_params.q, p_darkcount))

    plt.savefig('quick' + suffix + '.png', dpi=300)
    plt.close(fig)
