"""
sequential_repeaters.py

Analysis of sequential quantum repeater chains over arbitrary network graphs,
with optional entanglement distillation.

Typical usage
-------------
from sequential_repeaters import RepeaterParams, QuantumRepeaterNetwork, build_s2_graph

params = RepeaterParams()
params.p_det     = 0.95
params.alpha     = 0.18
params.q_0       = 0.01
params.nu        = 10e6
params.R_dark    = 100 # Ner term application
params.delta_det = 100e-12
params.p_pair    = 0.05
params.eta_c     = 0.8
params.P_BSM     = 0.98
params.T_coh     = 0.05    # memory coherence time [s]

A, dist, coords = build_s2_graph(N=1000, beta=2.6261, mu=0.0233, scale='city')
net = QuantumRepeaterNetwork(params, A, dist, coords, architecture='node',
                             scale='city', distillation_type=None)

df = net.analyze_all_paths(source=0)
net.export_html(source=0, output_path="repeater_viz.html")
"""

import json
import math
import numpy as np
import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binary_entropy(p):
    """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def BBPSSW(F1, F2):
    """
    BBPSSW entanglement distillation protocol.

    Given two pairs with fidelities F1, F2, perform one round of
    BBPSSW and return the output fidelity and success probability.

    Parameters
    ----------
    F1, F2 : float   input pair fidelities (Werner state)

    Returns
    -------
    F_out : float   output fidelity
    P_suc : float   success probability
    """
    num = F1 * F2 + (1 - F1) * (1 - F2) / 9
    den = (F1 * F2
           + (F1 * (1 - F2) + (1 - F1) * F2) / 3
           + 5 * (1 - F1) * (1 - F2) / 9) # This is also the probability of success
    F_out = num / den if den > 0 else 0.0
    return F_out, den


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class RepeaterParams:
    """
    Physical parameters for a DV quantum repeater link.

    All attributes initialised to None; set them before passing to
    QuantumRepeaterNetwork.

    Attributes
    ----------
    p_det     : float  Detector efficiency (0-1).
    alpha     : float  Fibre loss [dB/km].
    q_0       : float  Baseline intrinsic QBER.
    nu        : float  Source repetition rate [Hz].
    R_dark    : float  Dark-count rate [Hz].
    delta_det : float  Detector time-gate duration [s].
    p_pair    : float  Pair-generation probability per pulse.
    eta_c     : float  Source-to-fibre coupling efficiency (0-1).
    P_BSM     : float  BSM success probability (0.5 for linear optics).
    T_coh     : float  Memory coherence time [s].  Used only when
                       distillation_type is not None.
    """

    def __init__(self):
        self.p_det     = None
        self.alpha     = None
        self.V       = None
        self.nu        = None
        self.R_dark    = None
        self.delta_det = None
        self.p_pair    = None
        self.eta_c     = None
        self.P_BSM     = None
        self.T_coh     = None   # memory coherence time [s]
        self.eta_M     = None


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

_SCALE_FACTORS = {
    'city':    10,    # max distance ~31 km
    'country': 100,   # max distance ~310 km
    'europe':  1000,  # max distance ~3100 km
}


def build_s2_graph(N, beta, mu, scale='city', D=2, sample_from_file=False):
    """
    Build an S2 random-geometric graph.

    Parameters
    ----------
    N              : int    Number of nodes.
    beta           : float  Inverse-temperature of the S2 model.
    mu             : float  Average-degree parameter.
    scale          : {'city','country','europe'} or float
    D              : int    Embedding dimension.
    sample_from_file : bool

    Returns
    -------
    A      : np.ndarray (N, N)  Binary adjacency matrix.
    dist   : np.ndarray (N, N)  Scaled distance matrix [km].
    coords : np.ndarray (N, 3)  Unit-sphere coordinates.
    """
    from network_funcs import S2_graph_definite_N

    A, dist, coords = S2_graph_definite_N(
        N, beta, mu, D=D,
        sample_from_file=sample_from_file,
        return_coords=True,
    )

    factor = _SCALE_FACTORS[scale.lower()] if isinstance(scale, str) else float(scale)
    dist   = factor * dist
    return A, dist, coords


def build_graph(n_hubs, avg_n_branches):
    """
    Build a hub-and-spoke graph with a linear backbone.

    Parameters
    ----------
    n_hubs         : int  Number of backbone hubs.
    avg_n_branches : int  Average number of leaf nodes per hub.

    Returns
    -------
    A    : np.ndarray (N, N)  Binary adjacency matrix.
    dist : np.ndarray (N, N)  Distance matrix [km].
    """
    import random
    G = nx.Graph()

    for i in range(n_hubs - 1):
        d = random.randint(150, 250) * 0.1   # 15–25 km
        G.add_edge(i, i + 1, weight=d)

    next_leaf = n_hubs
    for i in range(n_hubs):
        n_branches = max(0, random.randint(avg_n_branches - 3, avg_n_branches + 3))
        for _ in range(n_branches):
            d = random.randint(20, 50) * 0.1  # 2–5 km
            G.add_edge(i, next_leaf, weight=d)
            next_leaf += 1

    A    = nx.to_numpy_array(G, weight=None)
    dist = nx.to_numpy_array(G, weight='weight')
    return A, dist


def density_to_scale(N, rho):
    """
    Convert a spatial node density to the km scale factor used by build_s2_graph.

    The S2 graph embeds N nodes in a unit-sphere patch whose angular radius is
    normalised to 1 radian. When we want nodes to live inside a circular region
    of area A = N / rho [km²] we need:

        pi * r² = N / rho   →   r = sqrt(N / (pi * rho))

    This radius [km] is then passed as the ``scale`` argument of build_s2_graph
    so that all edge distances are expressed in km consistently.

    Parameters
    ----------
    N   : int    Number of nodes.
    rho : float  Spatial density  [nodes / km²].

    Returns
    -------
    scale_km : float  Scale factor [km / radian] for build_s2_graph.
    """
    return np.sqrt(N / (np.pi * rho))


def build_density_graph(N, rho, beta, mu, D=2, sample_from_file=False):
    """
    Build an S2 random-geometric graph at a given spatial node density.

    This is a thin wrapper around :func:`build_s2_graph` that converts
    *rho* [nodes/km²] to the appropriate km scale factor.

    Parameters
    ----------
    N   : int    Number of nodes.
    rho : float  Spatial density  [nodes / km²].
    beta, mu, D, sample_from_file : passed through to build_s2_graph.

    Returns
    -------
    A      : np.ndarray (N, N)  Binary adjacency matrix.
    dist   : np.ndarray (N, N)  Distance matrix [km].
    coords : np.ndarray (N, 3)  Unit-sphere coordinates.
    scale_km : float            Scale used (useful for QuantumRepeaterNetwork).
    """
    scale_km = density_to_scale(N, rho)
    A, dist, coords = build_s2_graph(
        N, beta, mu, scale=scale_km, D=D, sample_from_file=sample_from_file
    )
    return A, dist, coords, scale_km


def build_radius_graph(N, radius_km, beta, mu, D=2, sample_from_file=False):
    """
    Build an S2 random-geometric graph whose nodes live in a circular region
    of physical radius *radius_km* [km].  The S2 scale factor equals the
    radius directly, so edge distances are in km.

    Returns
    -------
    A, dist, coords, radius_km
    """
    A, dist, coords = build_s2_graph(N, beta, mu, scale=float(radius_km),
                                     D=D, sample_from_file=sample_from_file)
    return A, dist, coords, float(radius_km)


# ---------------------------------------------------------------------------
# Coordinate projection
# ---------------------------------------------------------------------------

def _sphere_to_2d(coords, scale_km):
    """
    Equirectangular projection of unit-sphere coords to km.

    Parameters
    ----------
    coords   : np.ndarray (N, 3)
    scale_km : float   km per radian

    Returns
    -------
    x, y : np.ndarray (N,)   positions in km
    """
    lat = np.arcsin(np.clip(coords[:, 2], -1, 1))
    lon = np.arctan2(coords[:, 1], coords[:, 0])
    return lon * scale_km, lat * scale_km


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumRepeaterNetwork:
    """
    Sequential quantum repeater analysis over a fixed network graph,
    with optional entanglement distillation.

    Parameters
    ----------
    params           : RepeaterParams
    A                : np.ndarray (N, N)  Binary adjacency matrix.
    dist             : np.ndarray (N, N)  Distance matrix [km].
    coords           : np.ndarray (N, 3) or None
                       Unit-sphere coords; required for export_html.
    architecture     : {'node', 'midpoint'}
    scale            : str or float   Scale used when building the graph.
    distillation_type : {None, 'multiplexing', 'standard'}
        None           – no distillation; plain sequential repeater.
        'multiplexing' – BBPSSW on two freshly generated pairs; total
                         time scales as T / P_suc.
        'standard'     – BBPSSW where one pair has aged for time T; total
                         time scales as 2*T / P_suc.
    """

    def __init__(self, params, A, dist, coords=None,
                 architecture='node', scale='city', multiphoton=False,
                 distillation=False, multiplexing_type=None, M = 10e4, distillation_level=1, distillation_time='before_swap', just_transmittance=False):
        self.params             = params
        self.A                  = A
        self.dist               = dist
        self.coords             = coords
        self.architecture       = architecture
        self.multiphoton        = multiphoton
        self.distillation       = distillation
        self.multiplexing_type  = multiplexing_type
        self.M                  = M
        self.distillation_level = distillation_level
        self.distillation_time  = distillation_time
        self.just_transmittance = just_transmittance

        if isinstance(scale, str):
            self.scale_km    = _SCALE_FACTORS[scale.lower()]
            self.scale_label = scale
        else:
            self.scale_km    = float(scale)
            self.scale_label = f'{scale} km'

        self.Probs_mtx, self.P_click = \
            self._build_prob_matrix()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_prob_matrix(self):
        """
        Compute per-link entanglement probability matrix and transmission
        matrices for both detector arms.
        """
        p    = self.params
        A, d = self.A, self.dist

        dist = d / 2
        P_click = p.p_det * p.eta_c * 10 ** (-p.alpha * dist / 10) * A * p.eta_M # Probability to create photon pair x it survives to midpoint beam splitter

        if self.multiplexing_type == 'accumulated':
            P_click= 1-(1-P_click)**self.M
        elif self.multiplexing_type == 'single_burst':
            from scipy.stats import binom
            m_required = 2 ** self.distillation_level
            P_click = 1 - binom.cdf(m_required - 1, self.M, P_click) # Check this
        elif self.multiplexing_type is None:
            pass
        else:
            raise ValueError("multiplexing_type must be 'accumulated', 'single_burst' or None")

        P_both = 0.5 * P_click**2
        return P_both, P_click

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def optimal_path(self, source, target=None):
        """
        Dijkstra shortest path maximising log-probability of success.

        Edge weight: w(i,j) = -log2(p_ij) - log2(P_BSM).

        Parameters
        ----------
        source : int
        target : int or None

        Returns
        -------
        weights : dict or float
        paths   : dict or list
        """
        p = self.params
        W = np.zeros_like(self.Probs_mtx)
        nz = self.Probs_mtx > 0
        W[nz] = -np.log2(self.Probs_mtx[nz]) - np.log2(p.P_BSM)
        G = nx.from_numpy_array(W)
        return nx.single_source_dijkstra(G, source, target=target, weight='weight')

    def entanglement_rate(self, total_time):
        """
        Raw entanglement generation rate [pairs/s] = nu / T.

        Parameters
        ----------
        total_time : float   Expected number of rounds T.

        Returns
        -------
        float
        """
        return self.params.nu / total_time

    def F_link(self, a, b):
        """
        Per-link QBER for link (a, b), accounting for baseline QBER,
        dark counts, and multi-photon contributions.

        Parameters
        ----------
        a, b : int   Node indices.

        Returns
        -------
        float   QBER in [0, 0.5]
        """
        p     = self.params
        P_click = self.P_click[a,b]
        p_dc = p.R_dark * p.delta_det

        # Exact Barrett-Kok double-click probabilities
        p_sig = 0.5 * (P_click ** 2)
        p_acc = 4 * P_click * p_dc

        P_total = p_sig + p_acc

        # Fidelity is the weighted sum of the true signal and the completely mixed state (0.25)
        F = (0.5 * (1 + p.V) * (p_sig / P_total)) + (0.25 * (p_acc / P_total))
        return F

    def calculate_metrics(self, path):
        """
        Compute total time T, end-to-end QBER, and SKR for *path*,
        incorporating coherence decay and optional distillation.
        """
        p = self.params

        # ── first link ───────────────────────────────────────────────
        a, b = path[0], path[1]
        T      = 1.0 / self.Probs_mtx[a, b]
        F_link = self.F_link(a, b)   # fidelity stays primary through the loop

        if self.distillation_time == 'before_swap':
            for level in range(self.distillation_level):
                if self.distillation == True and self.multiplexing_type == 'single_burst':
                    F_link_out, P_suc_link = BBPSSW(F_link, F_link)
                    T = T / P_suc_link
                    F_link = F_link_out

                elif self.distillation == True and self.multiplexing_type != 'single_burst':
                    # Sequential: Pair 1 waits for T while Pair 2 generates
                    W_aged  = (4 * F_link - 1) / 3 * np.exp(-(T / p.nu) / p.T_coh)
                    F_aged  = (1 + 3 * W_aged) / 4
                    F_link_out, P_suc_link = BBPSSW(F_aged, F_link)   # F_link is already "fresh"
                    T = (2 * T) / P_suc_link
                    F_link = F_link_out

                elif self.distillation == False:
                    pass
                else:
                    raise ValueError("distillation must be True or False")

            W = (4 * F_link - 1) / 3   # convert once, needed for the swap/decoherence stage below

            # ── subsequent links ─────────────────────────────────────
            for i in range(2, len(path)):
                a, b = path[i - 1], path[i]
                T_link = 1.0 / self.Probs_mtx[a, b]
                F_link_new = self.F_link(a, b)

                for level in range(self.distillation_level):
                    if self.distillation == True and self.multiplexing_type == 'single_burst':
                        F_link_out, P_suc_link = BBPSSW(F_link_new, F_link_new)
                        T_link = T_link / P_suc_link
                        F_link_new = F_link_out

                    elif self.distillation == True and self.multiplexing_type != 'single_burst':
                        W_aged = (4 * F_link_new - 1) / 3 * np.exp(-(T_link / p.nu) / p.T_coh)
                        F_aged = (1 + 3 * W_aged) / 4
                        F_link_out, P_suc_link = BBPSSW(F_aged, F_link_new)
                        T_link = (2 * T_link) / P_suc_link
                        F_link_new = F_link_out

                W_link = (4 * F_link_new - 1) / 3   # convert once, for the swap combination

                # PERFORM THE SWAP (Sequential Wait & Decoherence)
                T = (T + T_link) / p.P_BSM
                W = W * np.exp(-(T_link / p.nu) / p.T_coh) * W_link

        elif self.distillation_time == 'after_swap':
            W = (4 * F_link - 1) / 3   # need W to start the swap-combination chain

            # ── subsequent links ─────────────────────────────────────
            for i in range(2, len(path)):
                a, b   = path[i - 1], path[i]
                T_link = 1.0 / self.Probs_mtx[a, b]

                if self.multiplexing_type == 'single_burst':
                    m_required = 2**self.distillation_level
                    T = (T + T_link) / p.P_BSM**m_required
                else:
                    T = (T + T_link) / p.P_BSM

                F_new  = self.F_link(a, b)
                W_link = (4 * F_new - 1) / 3

                # Coherence decay of the already-stored pair during T_link, then combine
                W = W * np.exp(-(T_link / p.nu) / p.T_coh) * W_link
                F = (1 + 3 * W) / 4   # convert once, to run the distillation loop in F-space

                for level in range(self.distillation_level):
                    if self.distillation == True and self.multiplexing_type == 'single_burst':
                        # Distilling the SWAPPED pairs, which arrived simultaneously
                        F_out, P_suc = BBPSSW(F, F)
                        T = T / P_suc
                        F = F_out

                    elif self.distillation == True and self.multiplexing_type != 'single_burst':
                        # Sequential: Swapped Pair 1 waited for Swapped Pair 2
                        W_aged = (4 * F - 1) / 3 * np.exp(-(T / p.nu) / p.T_coh)
                        F_aged = (1 + 3 * W_aged) / 4
                        F_out, P_suc = BBPSSW(F_aged, F)
                        T = (2 * T) / P_suc
                        F = F_out

                    elif self.distillation == False:
                        pass
                    else:
                        raise ValueError("distillation must be True or False")

                W = (4 * F - 1) / 3   # convert back once, so the next hop's combination can use W
        else:
            raise ValueError("distillation_time must be 'before_swap' or 'after_swap'")

        QBER  = (1 - W) / 2
        F     = (1 + 3 * W) / 4
        R_raw = self.entanglement_rate(T)
        R     = 0.5 * R_raw
        H     = binary_entropy(QBER)
        SKR   = R * (1 - 2 * H)
        return R_raw, QBER, SKR, F

    def path_distances(self, path):
        """
        Physical distance of each hop along *path* [km].

        Parameters
        ----------
        path : list of int

        Returns
        -------
        list of float
        """
        return [
            float(self.dist[path[i - 1], path[i]])
            for i in range(1, len(path))
        ]

    def analyze_all_paths(self, source):
        """
        Compute SKR, QBER, and total time for every reachable destination
        from *source* using calculate_metrics.

        Parameters
        ----------
        source : int

        Returns
        -------
        df : pd.DataFrame
            Columns: 'total_time', 'Q', 'SKR', 'path'.
            Indexed by destination node index.
        """
        _, paths = self.optimal_path(source)
        paths.pop(source, None)

        records = []
        for dest, path in paths.items():
            R, Q, SKR, F = self.calculate_metrics(path)
            records.append({
                'R':          R,
                'Q':          Q,
                'SKR':        SKR,
                'path':       path,
                'F':          F,
            })

        return pd.DataFrame(records, index=list(paths.keys()))

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------
    def network_metrics(self, sources=None, verbose=True):
        N = self.A.shape[0]
        all_sources = list(range(N)) if sources is None else list(sources)

        rates_all = []  # R_raw for all pairs where a path exists
        f_all = []
        rates_skr = []  # R_raw for SKR > 0 pairs only (kept for reference)
        skrs = []  # SKR clipped to 0 for all pairs where path exists
        n_viable = 0  # pairs with SKR > 0
        n_connected = 0  # pairs where a path exists at all
        n_total = len(all_sources) * (N - 1) if sources is not None else N * (N - 1)

        per_source_results = []

        for src in all_sources:
            if verbose:
                print(f"  Computing paths from source {src} / {all_sources[-1]} …")

            df = self.analyze_all_paths(src)

            src_rates_all = []
            src_skrs = []
            src_viable = 0
            src_connected = 0

            for dest, row in df.iterrows():
                skr = row['SKR']
                r = row['R']
                path = row['path']
                F = row['F']

                if path is None or len(path) == 0:
                    continue  # no path exists, skip entirely

                # Path exists — include in rate average regardless of SKR
                rates_all.append(r)
                f_all.append(F)
                skrs.append(max(skr, 0.0))
                src_rates_all.append(r)
                src_skrs.append(max(skr, 0.0))
                src_connected += 1
                n_connected += 1

                if skr > 0:
                    rates_skr.append(r)
                    n_viable += 1
                    src_viable += 1

            per_source_results.append({
                'source': src,
                'df': df,
                'avg_rate': float(np.mean(src_rates_all)) if src_rates_all else 0.0,
                'avg_fidelity': float(np.mean(f_all)) if f_all else 0.0,
                'avg_skr': float(np.mean(src_skrs)) if src_skrs else 0.0,
                'reachability': src_viable / (N - 1) if N > 1 else 0.0,
            })

        metrics = {
            'avg_rate': float(np.mean(rates_all)) if rates_all else 0.0,
            'avg_fidelity': float(np.mean(f_all)) if f_all else 0.0,
            'avg_rate_skr': float(np.mean(rates_skr)) if rates_skr else 0.0,  # old behaviour, kept for reference
            'avg_skr': float(np.mean(skrs)) if skrs else 0.0,
            'reachability': n_viable / n_total if n_total > 0 else 0.0,
            'n_viable_pairs': n_viable,
            'n_connected_pairs': n_connected,
            'n_total_pairs': n_total,
            'per_source': per_source_results,
        }

        if verbose:
            print("\n── Network metrics ──────────────────────────────────")
            print(f"  Avg rate (all connected)  : {metrics['avg_rate']:.4e} pairs/s")
            print(f"  Avg fidelity              : {metrics['avg_fidelity']:.4e}")
            print(f"  Avg rate (SKR > 0 only)   : {metrics['avg_rate_skr']:.4e} pairs/s")
            print(f"  Avg SKR  (negatives → 0)  : {metrics['avg_skr']:.4e} bits/s")
            print(f"  Reachability              : {metrics['reachability'] * 100:.2f} %")
            print(f"  Viable pairs (SKR > 0)    : {metrics['n_viable_pairs']} / {metrics['n_total_pairs']}")
            print(f"  Connected pairs           : {metrics['n_connected_pairs']} / {metrics['n_total_pairs']}")
            print("─────────────────────────────────────────────────────\n")

        return metrics

    def export_html(self, source, output_path="repeater_viz.html"):
        """
        Export a self-contained interactive HTML visualisation.

        Nodes are coloured by SKR (log scale):
          gold  = source, green gradient = SKR > 0,
          red   = SKR ≤ 0, grey = unreachable.

        Click a node to highlight the optimal path and show per-hop
        distances, SKR, QBER, and hop count in the side panel.

        Parameters
        ----------
        source      : int   Source node index.
        output_path : str   Output file path.
        """
        if self.coords is None:
            raise ValueError("coords must be provided to use export_html.")

        print(f"Computing all paths from node {source}...")
        df = self.analyze_all_paths(source)

        x, y = _sphere_to_2d(self.coords, self.scale_km)
        N    = len(x)

        skr_by_node  = {int(i): row['SKR']       for i, row in df.iterrows()}
        path_by_node = {int(i): row['path']       for i, row in df.iterrows()}
        qber_by_node = {int(i): row['Q']          for i, row in df.iterrows()}
        time_by_node = {int(i): row['total_time'] for i, row in df.iterrows()}

        pos_skrs = [v for v in skr_by_node.values() if v > 0]
        log_min  = math.log10(min(pos_skrs)) if pos_skrs else 0.0
        log_max  = math.log10(max(pos_skrs)) if pos_skrs else 1.0

        def skr_to_green(skr):
            t = ((math.log10(skr) - log_min) / (log_max - log_min)
                 if log_max != log_min else 1.0)
            t = max(0.0, min(1.0, t))
            return f'rgb(0,{int(80 + t * 175)},{int(t * 100)})'

        node_colors, node_skr_labels = [], []
        for i in range(N):
            if i == source:
                node_colors.append('rgb(255,215,0)')
                node_skr_labels.append('SOURCE')
            elif i not in skr_by_node:
                node_colors.append('rgb(80,80,80)')
                node_skr_labels.append('Unreachable')
            elif skr_by_node[i] <= 0:
                node_colors.append('rgb(200,40,40)')
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')
            else:
                node_colors.append(skr_to_green(skr_by_node[i]))
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')

        edge_x, edge_y = [], []
        rows, cols = np.where(np.triu(self.A, k=1) > 0)
        for i, j in zip(rows, cols):
            edge_x += [float(x[i]), float(x[j]), None]
            edge_y += [float(y[i]), float(y[j]), None]

        path_data = {}
        for dest, path in path_by_node.items():
            hop_dists = self.path_distances(path)
            path_data[dest] = {
                'path':       path,
                'hop_dists':  hop_dists,
                'skr':        skr_by_node[dest],
                'qber':       qber_by_node[dest],
                'total_time': time_by_node[dest],
                'total_dist': sum(hop_dists),
            }

        # distillation label for header
        dist_label = self.distillation_type if self.distillation_type else 'none'

        js_data = {
            'source':       source,
            'x':            [float(v) for v in x],
            'y':            [float(v) for v in y],
            'node_colors':  node_colors,
            'skr_labels':   node_skr_labels,
            'edge_x':       edge_x,
            'edge_y':       edge_y,
            'path_data':    path_data,
            'scale_label':  self.scale_label,
            'architecture': self.architecture,
            'dist_label':   dist_label,
            'log_min':      log_min,
            'log_max':      log_max,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(_build_html(js_data))
        print(f"Saved → {output_path}")

    def export_html_3d(self, source, output_path="repeater_viz_3d.html",
                       sphere_radius=1.0):
        """
        Export a self-contained interactive 3D globe visualisation.

        Nodes are placed at their true unit-sphere coordinates (scaled by
        *sphere_radius*) so the topology is geographically faithful.
        Edges are drawn as straight lines in 3D space between sphere-surface
        points, which are great-circle arcs — exactly as in the original
        plot_graph_on_sphere from network_funcs.

        The globe can be rotated by dragging (Plotly orbit mode).

        Node colouring and the click-to-inspect interaction are identical to
        export_html (2D version):
          gold  = source, green gradient = SKR > 0 (log scale),
          red   = SKR ≤ 0, grey = unreachable.

        Parameters
        ----------
        source        : int    Source node index.
        output_path   : str    Output file path.
        sphere_radius : float  Visual radius of the globe (default 1.0).
        """
        if self.coords is None:
            raise ValueError("coords must be provided to use export_html_3d.")

        print(f"Computing all paths from node {source}...")
        df = self.analyze_all_paths(source)

        # Unit-sphere coords × visual radius
        cx = (self.coords[:, 0] * sphere_radius).tolist()
        cy = (self.coords[:, 1] * sphere_radius).tolist()
        cz = (self.coords[:, 2] * sphere_radius).tolist()
        N = len(cx)

        skr_by_node = {int(i): row['SKR'] for i, row in df.iterrows()}
        path_by_node = {int(i): row['path'] for i, row in df.iterrows()}
        qber_by_node = {int(i): row['Q'] for i, row in df.iterrows()}
        time_by_node = {int(i): row['total_time'] for i, row in df.iterrows()}

        pos_skrs = [v for v in skr_by_node.values() if v > 0]
        log_min = math.log10(min(pos_skrs)) if pos_skrs else 0.0
        log_max = math.log10(max(pos_skrs)) if pos_skrs else 1.0

        def skr_to_green(skr):
            t = ((math.log10(skr) - log_min) / (log_max - log_min)
                 if log_max != log_min else 1.0)
            t = max(0.0, min(1.0, t))
            return f'rgb(0,{int(80 + t * 175)},{int(t * 100)})'

        node_colors, node_skr_labels = [], []
        for i in range(N):
            if i == source:
                node_colors.append('rgb(255,215,0)')
                node_skr_labels.append('SOURCE')
            elif i not in skr_by_node:
                node_colors.append('rgb(80,80,80)')
                node_skr_labels.append('Unreachable')
            elif skr_by_node[i] <= 0:
                node_colors.append('rgb(200,40,40)')
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')
            else:
                node_colors.append(skr_to_green(skr_by_node[i]))
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')

        # Background edges: straight lines in 3D between sphere-surface points
        # (these ARE great-circle arcs on the sphere surface, same as the
        # original plot_graph_on_sphere geodesics when R=1)
        edge_x, edge_y, edge_z = [], [], []
        rows, cols = np.where(np.triu(self.A, k=1) > 0)
        for i, j in zip(rows, cols):
            edge_x += [cx[i], cx[j], None]
            edge_y += [cy[i], cy[j], None]
            edge_z += [cz[i], cz[j], None]

        # Path data for JavaScript
        path_data = {}
        for dest, path in path_by_node.items():
            hop_dists = self.path_distances(path)
            path_data[str(dest)] = {
                'path': path,
                'hop_dists': hop_dists,
                'skr': skr_by_node[dest],
                'qber': qber_by_node[dest],
                'total_time': time_by_node[dest],
                'total_dist': sum(hop_dists),
            }

        dist_label = self.distillation_type if self.distillation_type else 'none'

        js_data = {
            'source': source,
            'cx': cx,
            'cy': cy,
            'cz': cz,
            'node_colors': node_colors,
            'skr_labels': node_skr_labels,
            'edge_x': edge_x,
            'edge_y': edge_y,
            'edge_z': edge_z,
            'path_data': path_data,
            'scale_label': self.scale_label,
            'architecture': self.architecture,
            'dist_label': dist_label,
            'log_min': log_min,
            'log_max': log_max,
            'sphere_radius': sphere_radius,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(_build_html_3d(js_data))  # <--- Add the "_3d" here
        print(f"Saved → {output_path}")

    # ---------------------------------------------------------------------------
    # HTML template — 3D globe
    # ---------------------------------------------------------------------------

def _build_html_3d(d):
    """Return the full self-contained HTML string for the 3D globe view."""
    # String-key path_data so JS lookup with String(idx) always works
    d = dict(d)
    d['path_data'] = {str(k): v for k, v in d['path_data'].items()}
    data_json = json.dumps(d)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Repeater Network — 3D Globe</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:    #050810; --panel: #0b0f1a; --border: #1a2740;
    --accent:#00d4ff; --green: #00ff9d; --gold:   #ffd700;
    --red:   #ff3a3a; --text:  #c8d8e8; --dim:    #4a6080;
    --mono:  'Share Tech Mono', monospace;
    --sans:  'Rajdhani', sans-serif;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--sans);
          font-weight:300; height:100vh; display:flex; flex-direction:column; overflow:hidden; }}
  header {{ padding:11px 22px; border-bottom:1px solid var(--border);
            display:flex; align-items:baseline; gap:18px; flex-shrink:0; background:var(--panel); }}
  header h1 {{ font-family:var(--mono); font-size:13px; color:var(--accent);
               letter-spacing:0.18em; text-transform:uppercase; }}
  header span {{ font-size:11px; color:var(--dim); font-family:var(--mono); }}
  .main {{ display:flex; flex:1; overflow:hidden; }}
  .panel {{ width:290px; flex-shrink:0; border-right:1px solid var(--border);
            background:var(--panel); display:flex; flex-direction:column; overflow-y:auto; }}
  .section {{ padding:18px 20px; border-bottom:1px solid var(--border); }}
  .section h2 {{ font-family:var(--mono); font-size:9px; letter-spacing:0.22em;
                 color:var(--dim); text-transform:uppercase; margin-bottom:13px; }}
  .leg-row {{ display:flex; align-items:center; gap:9px; margin-bottom:7px; font-size:13px; }}
  .leg-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .colorbar {{ height:10px; border-radius:3px;
               background:linear-gradient(to right,rgb(0,80,0),rgb(0,255,100));
               margin:7px 0 3px; }}
  .cb-labels {{ display:flex; justify-content:space-between;
                font-family:var(--mono); font-size:10px; color:var(--dim); }}
  .hint {{ font-family:var(--mono); font-size:10px; color:var(--dim);
           line-height:1.7; margin-top:10px; }}
  #info-box {{ flex:1; padding:18px 20px; overflow-y:auto; }}
  .placeholder {{ color:var(--dim); font-size:12px; line-height:1.9; font-family:var(--mono); }}
  .placeholder::before {{ content:'> '; color:var(--accent); }}
  .info-title {{ font-family:var(--mono); font-size:9px; letter-spacing:0.22em;
                 color:var(--dim); text-transform:uppercase; margin-bottom:16px; }}
  .metric {{ margin-bottom:15px; }}
  .metric-label {{ font-size:9px; font-family:var(--mono); color:var(--dim);
                   letter-spacing:0.12em; text-transform:uppercase; margin-bottom:3px; }}
  .metric-value {{ font-family:var(--mono); font-size:17px; color:var(--green); }}
  .metric-value.bad     {{ color:var(--red); }}
  .metric-value.neutral {{ color:var(--text); }}
  .hop-list {{ margin-top:16px; }}
  .hop-list h3 {{ font-size:9px; font-family:var(--mono); color:var(--dim);
                  letter-spacing:0.12em; text-transform:uppercase; margin-bottom:9px; }}
  .hop-item {{ display:flex; align-items:center; gap:7px; margin-bottom:5px;
               font-family:var(--mono); font-size:12px; }}
  .hop-arrow {{ color:var(--accent); font-size:9px; }}
  .hop-dist  {{ color:var(--dim); margin-left:auto; }}
  #graph {{ flex:1; min-width:0; }}
</style>
</head>
<body>
<header>
  <h1>Quantum Repeater Network — 3D Globe</h1>
  <span id="hdr"></span>
</header>
<div class="main">
  <div class="panel">
    <div class="section">
      <h2>Legend</h2>
      <div class="leg-row"><div class="leg-dot" style="background:var(--gold)"></div><span>Source node</span></div>
      <div class="leg-row"><div class="leg-dot" style="background:rgb(200,40,40)"></div><span>SKR &le; 0</span></div>
      <div class="leg-row"><div class="leg-dot" style="background:rgb(80,80,80)"></div><span>Unreachable</span></div>
      <div style="margin-top:8px;">
        <div class="colorbar"></div>
        <div class="cb-labels">
          <span id="skr-min"></span><span>SKR (bit/s, log)</span><span id="skr-max"></span>
        </div>
      </div>
      <p class="hint">Drag to rotate · Scroll to zoom · Click node to inspect</p>
    </div>
    <div id="info-box"><p class="placeholder">Click a node to inspect its optimal path</p></div>
  </div>
  <div id="graph"></div>
</div>
<script>
const D = {data_json};

document.getElementById('hdr').textContent =
  `source: node ${{D.source}}  ·  scale: ${{D.scale_label}}  ·  arch: ${{D.architecture}}  ·  distillation: ${{D.dist_label}}`;
document.getElementById('skr-min').textContent = '10^' + D.log_min.toFixed(1);
document.getElementById('skr-max').textContent = '10^' + D.log_max.toFixed(1);

function fmtSKR(v) {{ return v == null ? '—' : v.toExponential(3) + ' bit/s'; }}

// ── globe wireframe (latitude/longitude lines, no surface trace) ───────────
// A surface trace blocks click events on scatter3d nodes underneath it.
// Instead we draw thin lat/lon lines as scatter3d so clicks pass through.
const R = D.sphere_radius;
const wireX = [], wireY = [], wireZ = [];
const NL = 18; // number of lat/lon lines each
for (let i = 0; i < NL; i++) {{
  const phi = i / NL * 2 * Math.PI;   // longitude lines
  for (let j = 0; j <= 60; j++) {{
    const t = j / 60 * Math.PI;
    wireX.push(R * Math.sin(t) * Math.cos(phi));
    wireY.push(R * Math.sin(t) * Math.sin(phi));
    wireZ.push(R * Math.cos(t));
  }}
  wireX.push(null); wireY.push(null); wireZ.push(null);
}}
for (let i = 1; i < NL - 1; i++) {{  // latitude lines (skip poles)
  const theta = i / NL * Math.PI;
  for (let j = 0; j <= 60; j++) {{
    const phi = j / 60 * 2 * Math.PI;
    wireX.push(R * Math.sin(theta) * Math.cos(phi));
    wireY.push(R * Math.sin(theta) * Math.sin(phi));
    wireZ.push(R * Math.cos(theta));
  }}
  wireX.push(null); wireY.push(null); wireZ.push(null);
}}

const trGlobe = {{
  type: 'scatter3d', mode: 'lines',
  x: wireX, y: wireY, z: wireZ,
  line: {{ color: 'rgba(30,60,160,0.22)', width: 1 }},
  hoverinfo: 'skip', name: 'globe',
}};

// ── background edges ──────────────────────────────────────────────────────
const trEdge = {{
  type: 'scatter3d', mode: 'lines',
  x: D.edge_x, y: D.edge_y, z: D.edge_z,
  line: {{ color:'rgba(0,180,255,0.10)', width:1 }},
  hoverinfo: 'skip', name: 'edges',
}};

// ── highlighted path — kept as a separate mutable trace (index 2) ─────────
// Seed with two identical points so the line shader never sees an empty array.
let pathTrace = {{
  type: 'scatter3d', mode: 'lines',
  x: [D.cx[D.source], D.cx[D.source]],
  y: [D.cy[D.source], D.cy[D.source]],
  z: [D.cz[D.source], D.cz[D.source]],
  line: {{ color:'rgba(0,255,157,0.95)', width:5 }},
  hoverinfo: 'skip', name: 'path',
}};

// ── nodes ─────────────────────────────────────────────────────────────────
// Store node indices in customdata so we can retrieve them on click.
const trNode = {{
  type: 'scatter3d', mode: 'markers',
  x: D.cx, y: D.cy, z: D.cz,
  marker: {{ size:4, color:D.node_colors, line:{{ width:0 }} }},
  text: D.skr_labels,
  customdata: D.cx.map((_,i) => i),
  hovertemplate: '<b>Node %{{customdata}}</b><br>SKR: %{{text}}<extra></extra>',
  name: 'nodes',
}};

const axStyle = {{
  showgrid:false, zeroline:false, showline:false,
  showticklabels:false, showbackground:false, title:'',
}};

const layout = {{
  paper_bgcolor: '#050810',
  margin: {{ t:0, b:0, l:0, r:0 }},
  scene: {{
    xaxis: axStyle, yaxis: axStyle, zaxis: axStyle,
    bgcolor: '#050810',
    camera: {{ eye:{{ x:1.6, y:1.6, z:0.8 }} }},
    aspectmode: 'cube',
  }},
  showlegend: false,
}};

// traces: 0=globe, 1=edges, 2=path, 3=nodes
const traces = [trGlobe, trEdge, pathTrace, trNode];
const graphDiv = document.getElementById('graph');
Plotly.newPlot(graphDiv, traces, layout,
  {{ scrollZoom:true, responsive:true, displayModeBar:true }});

// ── click handler ─────────────────────────────────────────────────────────
// We use Plotly.react() to update data — it diffs and patches the scene
// without destroying the camera or drag bindings, unlike restyle/update.
graphDiv.on('plotly_click', function(ev) {{
  const pt = ev.points[0];
  if (!pt || pt.data.name !== 'nodes') return;

  const idx = pt.customdata;   // integer node index stored in customdata
  if (idx === D.source) return;

  const pd = D.path_data[String(idx)];  // keys are strings after JSON round-trip

  // Snapshot current camera so Plotly.react doesn't reset the view
  const cam = graphDiv.layout.scene.camera;

  if (!pd) {{
    // Reset path trace to invisible stub, preserve camera
    traces[2] = Object.assign({{}}, pathTrace, {{
      x: [D.cx[D.source], D.cx[D.source]],
      y: [D.cy[D.source], D.cy[D.source]],
      z: [D.cz[D.source], D.cz[D.source]],
    }});
    Plotly.react(graphDiv, traces,
      Object.assign({{}}, layout, {{ scene: Object.assign({{}}, layout.scene, {{ camera: cam }}) }}));
    document.getElementById('info-box').innerHTML =
      `<p class="info-title">Node ${{idx}}</p>
       <div class="metric"><div class="metric-label">Status</div>
       <div class="metric-value bad">Unreachable</div></div>`;
    return;
  }}

  // Build 3D path polyline with null separators between hops
  const path = pd.path;
  const px=[], py=[], pz=[];
  for (let i = 0; i < path.length - 1; i++) {{
    px.push(D.cx[path[i]], D.cx[path[i+1]], null);
    py.push(D.cy[path[i]], D.cy[path[i+1]], null);
    pz.push(D.cz[path[i]], D.cz[path[i+1]], null);
  }}

  // Patch the path trace in-place and re-render, preserving camera
  traces[2] = Object.assign({{}}, pathTrace, {{ x:px, y:py, z:pz }});
  Plotly.react(graphDiv, traces,
    Object.assign({{}}, layout, {{ scene: Object.assign({{}}, layout.scene, {{ camera: cam }}) }}));

  // Build hop rows for side panel
  let hopHtml = '';
  for (let i = 0; i < path.length - 1; i++) {{
    hopHtml += `<div class="hop-item">
      <span>${{path[i]}}</span><span class="hop-arrow">──▶</span>
      <span>${{path[i+1]}}</span>
      <span class="hop-dist">${{pd.hop_dists[i].toFixed(2)}} km</span></div>`;
  }}

  document.getElementById('info-box').innerHTML = `
    <p class="info-title">Node ${{D.source}} &rarr; Node ${{idx}}</p>
    <div class="metric"><div class="metric-label">Secret Key Rate</div>
      <div class="metric-value ${{pd.skr>0?'':'bad'}}">${{fmtSKR(pd.skr)}}</div></div>
    <div class="metric"><div class="metric-label">QBER</div>
      <div class="metric-value neutral">${{(pd.qber*100).toFixed(3)}} %</div></div>
    <div class="metric"><div class="metric-label">Hops</div>
      <div class="metric-value neutral">${{path.length-1}}</div></div>
    <div class="metric"><div class="metric-label">Total distance</div>
      <div class="metric-value neutral">${{pd.total_dist.toFixed(2)}} km</div></div>
    <div class="hop-list"><h3>Path detail</h3>${{hopHtml}}</div>`;
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTML template — 2D flat map
# ---------------------------------------------------------------------------

def _build_html(d):
    data_json = json.dumps(d)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Repeater Network</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:    #050810; --panel: #0b0f1a; --border: #1a2740;
    --accent:#00d4ff; --green: #00ff9d; --gold:   #ffd700;
    --red:   #ff3a3a; --text:  #c8d8e8; --dim:    #4a6080;
    --mono:  'Share Tech Mono', monospace;
    --sans:  'Rajdhani', sans-serif;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--sans);
          font-weight:300; height:100vh; display:flex; flex-direction:column; overflow:hidden; }}
  header {{ padding:11px 22px; border-bottom:1px solid var(--border);
            display:flex; align-items:baseline; gap:18px; flex-shrink:0; background:var(--panel); }}
  header h1 {{ font-family:var(--mono); font-size:13px; color:var(--accent);
               letter-spacing:0.18em; text-transform:uppercase; }}
  header span {{ font-size:11px; color:var(--dim); font-family:var(--mono); }}
  .main {{ display:flex; flex:1; overflow:hidden; }}
  .panel {{ width:290px; flex-shrink:0; border-right:1px solid var(--border);
            background:var(--panel); display:flex; flex-direction:column; overflow-y:auto; }}
  .section {{ padding:18px 20px; border-bottom:1px solid var(--border); }}
  .section h2 {{ font-family:var(--mono); font-size:9px; letter-spacing:0.22em;
                 color:var(--dim); text-transform:uppercase; margin-bottom:13px; }}
  .leg-row {{ display:flex; align-items:center; gap:9px; margin-bottom:7px; font-size:13px; }}
  .leg-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .colorbar {{ height:10px; border-radius:3px;
               background:linear-gradient(to right,rgb(0,80,0),rgb(0,255,100));
               margin:7px 0 3px; }}
  .cb-labels {{ display:flex; justify-content:space-between;
                font-family:var(--mono); font-size:10px; color:var(--dim); }}
  #info-box {{ flex:1; padding:18px 20px; overflow-y:auto; }}
  .placeholder {{ color:var(--dim); font-size:12px; line-height:1.9; font-family:var(--mono); }}
  .placeholder::before {{ content:'> '; color:var(--accent); }}
  .info-title {{ font-family:var(--mono); font-size:9px; letter-spacing:0.22em;
                 color:var(--dim); text-transform:uppercase; margin-bottom:16px; }}
  .metric {{ margin-bottom:15px; }}
  .metric-label {{ font-size:9px; font-family:var(--mono); color:var(--dim);
                   letter-spacing:0.12em; text-transform:uppercase; margin-bottom:3px; }}
  .metric-value {{ font-family:var(--mono); font-size:17px; color:var(--green); }}
  .metric-value.bad     {{ color:var(--red); }}
  .metric-value.neutral {{ color:var(--text); }}
  .hop-list {{ margin-top:16px; }}
  .hop-list h3 {{ font-size:9px; font-family:var(--mono); color:var(--dim);
                  letter-spacing:0.12em; text-transform:uppercase; margin-bottom:9px; }}
  .hop-item {{ display:flex; align-items:center; gap:7px; margin-bottom:5px;
               font-family:var(--mono); font-size:12px; }}
  .hop-arrow {{ color:var(--accent); font-size:9px; }}
  .hop-dist  {{ color:var(--dim); margin-left:auto; }}
  #graph {{ flex:1; min-width:0; }}
</style>
</head>
<body>
<header>
  <h1>Quantum Repeater Network</h1>
  <span id="hdr"></span>
</header>
<div class="main">
  <div class="panel">
    <div class="section">
      <h2>Legend</h2>
      <div class="leg-row"><div class="leg-dot" style="background:var(--gold)"></div><span>Source node</span></div>
      <div class="leg-row"><div class="leg-dot" style="background:rgb(200,40,40)"></div><span>SKR &le; 0</span></div>
      <div class="leg-row"><div class="leg-dot" style="background:rgb(80,80,80)"></div><span>Unreachable</span></div>
      <div style="margin-top:8px;">
        <div class="colorbar"></div>
        <div class="cb-labels">
          <span id="skr-min"></span><span>SKR (bit/s, log)</span><span id="skr-max"></span>
        </div>
      </div>
    </div>
    <div id="info-box"><p class="placeholder">Click a node to inspect its optimal path</p></div>
  </div>
  <div id="graph"></div>
</div>
<script>
const D = {data_json};
document.getElementById('hdr').textContent =
  `source: node ${{D.source}}  ·  scale: ${{D.scale_label}}  ·  arch: ${{D.architecture}}  ·  distillation: ${{D.dist_label}}`;
document.getElementById('skr-min').textContent = '10^' + D.log_min.toFixed(1);
document.getElementById('skr-max').textContent = '10^' + D.log_max.toFixed(1);

function fmtSKR(v) {{ return v == null ? '—' : v.toExponential(3) + ' bit/s'; }}

const trEdge = {{ type:'scatter', mode:'lines', x:D.edge_x, y:D.edge_y,
  line:{{ color:'rgba(0,180,255,0.07)', width:0.8 }}, hoverinfo:'skip', name:'edges' }};
const trPath = {{ type:'scatter', mode:'lines', x:[], y:[],
  line:{{ color:'rgba(0,255,157,0.9)', width:3 }}, hoverinfo:'skip', name:'path' }};
const trNode = {{ type:'scatter', mode:'markers', x:D.x, y:D.y,
  marker:{{ size:6, color:D.node_colors, line:{{ width:0 }} }},
  text:D.skr_labels, customdata:D.x.map((_,i)=>i),
  hovertemplate:'<b>Node %{{customdata}}</b><br>SKR: %{{text}}<br>(%{{x:.1f}} km, %{{y:.1f}} km)<extra></extra>',
  name:'nodes' }};

const layout = {{
  paper_bgcolor:'#050810', plot_bgcolor:'#050810',
  margin:{{ t:10, b:10, l:10, r:10 }},
  xaxis:{{ title:{{ text:'km', font:{{ color:'#4a6080', size:11 }} }},
           gridcolor:'#0d1825', zerolinecolor:'#1a2740',
           tickfont:{{ color:'#4a6080', size:10 }}, color:'#4a6080' }},
  yaxis:{{ title:{{ text:'km', font:{{ color:'#4a6080', size:11 }} }},
           gridcolor:'#0d1825', zerolinecolor:'#1a2740',
           tickfont:{{ color:'#4a6080', size:10 }}, color:'#4a6080',
           scaleanchor:'x', scaleratio:1 }},
  showlegend:false, dragmode:'pan'
}};

Plotly.newPlot('graph', [trEdge, trPath, trNode], layout,
  {{ scrollZoom:true, responsive:true, displayModeBar:true,
     modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'] }});

document.getElementById('graph').on('plotly_click', function(ev) {{
  const pt = ev.points[0];
  if (!pt || pt.data.name !== 'nodes') return;
  const idx = pt.customdata;
  if (idx === D.source) return;
  const pd = D.path_data[idx];

  if (!pd) {{
    Plotly.restyle('graph', {{ x:[[]], y:[[]] }}, [1]);
    document.getElementById('info-box').innerHTML =
      `<p class="info-title">Node ${{idx}}</p>
       <div class="metric"><div class="metric-label">Status</div>
       <div class="metric-value bad">Unreachable</div></div>`;
    return;
  }}

  const path = pd.path;
  const px=[], py=[];
  for (let i=0; i<path.length-1; i++) {{
    px.push(D.x[path[i]], D.x[path[i+1]], null);
    py.push(D.y[path[i]], D.y[path[i+1]], null);
  }}
  Plotly.restyle('graph', {{ x:[px], y:[py] }}, [1]);

  let hopHtml='';
  for (let i=0; i<path.length-1; i++) {{
    hopHtml += `<div class="hop-item">
      <span>${{path[i]}}</span><span class="hop-arrow">──▶</span>
      <span>${{path[i+1]}}</span>
      <span class="hop-dist">${{pd.hop_dists[i].toFixed(2)}} km</span></div>`;
  }}

  document.getElementById('info-box').innerHTML = `
    <p class="info-title">Node ${{D.source}} &rarr; Node ${{idx}}</p>
    <div class="metric"><div class="metric-label">Secret Key Rate</div>
      <div class="metric-value ${{pd.skr>0?'':'bad'}}">${{fmtSKR(pd.skr)}}</div></div>
    <div class="metric"><div class="metric-label">QBER</div>
      <div class="metric-value neutral">${{(pd.qber*100).toFixed(3)}} %</div></div>
    <div class="metric"><div class="metric-label">Hops</div>
      <div class="metric-value neutral">${{path.length-1}}</div></div>
    <div class="metric"><div class="metric-label">Total distance</div>
      <div class="metric-value neutral">${{pd.total_dist.toFixed(2)}} km</div></div>
    <div class="hop-list"><h3>Path detail</h3>${{hopHtml}}</div>`;
}});
</script>
</body>
</html>"""
