"""
sequential_repeaters.py

Analysis of sequential quantum repeater chains over arbitrary network graphs.

Typical usage
-------------
from sequential_repeaters import RepeaterParams, QuantumRepeaterNetwork, build_s2_graph

params = RepeaterParams()
params.p_det     = 0.95
params.alpha     = 0.18
params.q_0       = 0.01 #Check!!!
params.nu        = 1e9
params.R_dark    = 100
params.delta_det = 100e-12
params.p_pair    = 0.05
params.eta_c     = 0.8
params.P_BSM     = 0.5
params.T_coh     = 0.05

A, dist, coords = build_s2_graph(N=1000, beta=2.6261, mu=0.0233, scale='city')
net = QuantumRepeaterNetwork(params, A, dist, coords, architecture='node', scale='city')

df = net.analyze_all_paths(source=0)
net.export_html(source=0, output_path="repeater_viz.html")
"""

import json
import math
import numpy as np
import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: binary entropy (used by secret_key_rate)
# ---------------------------------------------------------------------------

def binary_entropy(p):
    """Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def BBPSSW(F1, F2):
    """BBPSSW distillation protocol: output fidelity after one round and probability success"""
    num = F1 * F2 + (1 - F1) * (1 - F2) / 9
    den = F1 * F2 + (F1 * (1 - F2) + (1 - F1) * F2) / 3 + 5 * (1 - F1) * (1 - F2) / 9
    F = num / den if den > 0 else 0.0
    return F, den
# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class RepeaterParams:
    """
    Physical parameters for a discrete-variable (DV) quantum repeater link.

    All attributes are initialised to None; set them before passing the
    object to QuantumRepeaterNetwork.

    Attributes
    ----------
    p_det : float
        Single-photon detector efficiency (dimensionless, 0-1).
    alpha : float
        Fibre loss coefficient [dB/km].
    q_0 : float
        Baseline (intrinsic) QBER, i.e. the QBER in the absence of dark
        counts and multi-photon effects.
    nu : float
        Source repetition rate [Hz].
    R_dark : float
        Dark-count rate of the detectors [Hz].
    delta_det : float
        Detector time-gate duration [s].
    p_pair : float
        Probability of generating an entangled photon pair per pulse.
    eta_c : float
        Source-to-fibre coupling efficiency (dimensionless, 0-1).
    P_BSM : float
        Bell-state measurement success probability (0.5 for linear optics).
    """

    def __init__(self):
        self.p_det     = None   # detector efficiency
        self.alpha     = None   # fibre loss [dB/km]
        self.q_0       = None   # baseline QBER
        self.nu        = None   # repetition rate [Hz]
        self.R_dark    = None   # dark-count rate [Hz]
        self.delta_det = None   # time-gate duration [s]
        self.p_pair    = None   # pair-generation probability per pulse
        self.eta_c     = None   # source-to-fibre coupling efficiency
        self.P_BSM     = None   # BSM success probability
        self.P_coh     = None   # Coherence time of the memory


# ---------------------------------------------------------------------------
# Graph builder  (decoupled from the network class so any graph can be used)
# ---------------------------------------------------------------------------

# Scale factors: map a human-readable label to a km multiplier applied to the
# raw S2 chord distances so that the maximum inter-node distance is realistic.
_SCALE_FACTORS = {
    'city':    10,    # max distance ~31 km
    'country': 100,   # max distance ~310 km
    'europe':  1000,  # max distance ~3100 km
}

def build_s2_graph(N, beta, mu, scale='city', D=2, sample_from_file=False):
    """
    Build an S2 random-geometric graph and return its adjacency matrix,
    (scaled) distance matrix, and node coordinates.

    Parameters
    ----------
    N : int
        Number of nodes.
    beta : float
        Inverse-temperature parameter of the S2 model.
    mu : float
        Average-degree parameter of the S2 model.
    scale : {'city', 'country', 'europe'} or float
        Multiplier applied to the raw chord distances.
        Pass a float to use a custom scale factor.
    D : int
        Embedding dimension passed to S2_graph_definite_N.
    sample_from_file : bool
        Whether to load a pre-sampled graph.

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Binary adjacency matrix.
    dist : np.ndarray, shape (N, N)
        Scaled distance matrix [km].
    coords : np.ndarray, shape (N, 3)
        Node coordinates on the unit sphere.
    """
    from network_funcs import S2_graph_definite_N  # project-local import

    A, dist, coords = S2_graph_definite_N(
        N, beta, mu, D=D,
        sample_from_file=sample_from_file,
        return_coords=True
    )

    # Apply scale factor
    if isinstance(scale, str):
        factor = _SCALE_FACTORS[scale.lower()]
    else:
        factor = float(scale)

    dist = factor * dist

    return A, dist, coords


# ---------------------------------------------------------------------------
# Coordinate projection: unit sphere -> 2D (longitude / latitude in km)
# ---------------------------------------------------------------------------

def _sphere_to_2d(coords, scale_km):
    """
    Project unit-sphere coordinates to a 2D plane via equirectangular
    projection (longitude / latitude), then scale to kilometres so that
    distances on the plot correspond to physical distances.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates on the unit sphere.
    scale_km : float
        The km multiplier used when building the graph (e.g. 10 for 'city').
        One radian of arc on the unit sphere equals scale_km kilometres.

    Returns
    -------
    x, y : np.ndarray, shape (N,)
        2D positions in kilometres.
    """
    lat = np.arcsin(np.clip(coords[:, 2], -1, 1))   # [-pi/2, pi/2]
    lon = np.arctan2(coords[:, 1], coords[:, 0])     # [-pi,   pi  ]

    # Convert angular position to km using the same scale as the distances
    x = lon * scale_km
    y = lat * scale_km

    return x, y

def build_graph(n_hubs, avg_n_branches):
    import random
    G = nx.Graph()

    # 1. Build the Backbone
    # Hubs will be assigned IDs from 0 to (n_hubs - 1)
    for i in range(n_hubs - 1):
        dist = random.randint(150, 250) * 0.1  # 15.0 to 25.0 km
        G.add_edge(i, i+1, weight=dist)

    # 2. Build the Branches
    # Keep a running counter for unique leaf IDs so they don't overlap
    next_leaf_id = n_hubs

    for i in range(n_hubs): # Loop through ALL hubs
        # Generate random number of branches (ensure it doesn't go negative)
        n_branches = max(0, random.randint(avg_n_branches - 3, avg_n_branches + 3))

        for _ in range(n_branches):
            dist = random.randint(20, 50) * 0.1  # 2.0 to 5.0 km
            G.add_edge(i, next_leaf_id, weight=dist)
            next_leaf_id += 1 # Increment for the next unique leaf

    # 3. Extract Matrices
    A = nx.to_numpy_array(G, weight=None)          # weight=None guarantees pure 1s and 0s
    dists = nx.to_numpy_array(G, weight='weight')  # Grabs the km values

    return A, dists

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumRepeaterNetwork:
    """
    Sequential quantum repeater analysis over a fixed network graph.

    Parameters
    ----------
    params : RepeaterParams
        Physical parameters. All fields must be set before passing.
    A : np.ndarray, shape (N, N)
        Binary adjacency matrix of the network.
    dist : np.ndarray, shape (N, N)
        Distance matrix [km]. Must already be scaled.
    coords : np.ndarray, shape (N, 3)
        Unit-sphere coordinates of each node (from build_s2_graph).
        Required for export_html; pass None if not using visualisation.
    architecture : {'node', 'midpoint'}
        'node'     - source is placed at node A, detector at node B.
        'midpoint' - source is placed at the midpoint of each link.
    scale : str or float
        The scale used when building the graph. Stored so that the 2D
        coordinate projection in export_html is consistent with the
        distances in dist.
    """

    def __init__(self, params, A, dist, coords=None,
                 architecture='node', scale='city', distillation_type=None):
        self.params       = params
        self.A            = A
        self.dist         = dist
        self.coords       = coords
        self.architecture = architecture
        self.distillation_type = distillation_type

        # Resolve scale factor for coordinate projection
        if isinstance(scale, str):
            self.scale_km    = _SCALE_FACTORS[scale.lower()]
            self.scale_label = scale
        else:
            self.scale_km    = float(scale)
            self.scale_label = f'{scale} km'

        # Pre-compute the probability matrices (done once at construction)
        self.Probs_mtx, self.eta_A_mtx, self.eta_B_mtx = \
            self._build_prob_matrix()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prob_matrix(self):
        """
        Compute the per-link entanglement probability matrix and the
        individual transmission matrices for both detector arms.

        Returns
        -------
        P_ent_matrix : np.ndarray
            Probability of successful entanglement per link per round.
        eta_A_mtx : np.ndarray
            Transmissivity from source to detector A.
        eta_B_mtx : np.ndarray
            Transmissivity from source to detector B.
        """
        p = self.params
        A, dist = self.A, self.dist

        if self.architecture == 'node':
            # Source at one node; photon travels the full link distance
            dist_A = np.zeros_like(dist)
            dist_B = dist
        elif self.architecture == 'midpoint':
            # Source at midpoint; each photon travels half the link
            dist_A = dist / 2
            dist_B = dist / 2
        else:
            raise ValueError("architecture must be 'node' or 'midpoint'")

        # Transmission = coupling x detector efficiency x fibre attenuation
        # Multiplied by A to zero out non-edges
        eta_A_mtx = p.eta_c * p.p_det * 10 ** (-p.alpha * dist_A / 10) * A
        eta_B_mtx = p.eta_c * p.p_det * 10 ** (-p.alpha * dist_B / 10) * A

        P_ent_matrix = p.p_pair * eta_A_mtx * eta_B_mtx

        return P_ent_matrix, eta_A_mtx, eta_B_mtx

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def optimal_path(self, source, target=None):
        """
        Find the path(s) that maximise the secret-key rate using Dijkstra's
        algorithm on a weight graph derived from the entanglement probabilities.

        Edge weight: w(i,j) = -log2(p_ij) - log2(P_BSM), so minimising
        total weight is equivalent to maximising log-probability of success.

        Parameters
        ----------
        source : int
            Source node index.
        target : int or None
            If given, return only the path to that node.

        Returns
        -------
        weights : dict or float
        paths : dict or list
        """
        p = self.params
        W = np.zeros_like(self.Probs_mtx)
        nonzero = self.Probs_mtx > 0
        W[nonzero] = (
            -np.log2(self.Probs_mtx[nonzero])
            - np.log2(p.P_BSM)
        )
        G = nx.from_numpy_array(W)
        weights, paths = nx.single_source_dijkstra(
            G, source, target=target, weight='weight'
        )
        return weights, paths

    def entanglement_rate(self, total_time):
        """
        Entanglement generation rate [pairs/s].

        Parameters
        ----------
        total_time : float
            Expected number of rounds from sequential_time().

        Returns
        -------
        R_ent : float
        """
        return self.params.nu / total_time

    def Q_link(self, a, b):
        """
        QBER for a single link (a,b), accounting for baseline QBER,
        dark counts, and multi-photon contributions.

        Parameters
        ----------
        a, b : int
            Node indices of the link.

        Returns
        -------
        Q : float  (0-0.5)
        """
        p = self.params
        eta_A = self.eta_A_mtx[a, b]
        eta_B = self.eta_B_mtx[a, b]

        p_acc = (
            p.p_pair * eta_A * (1 - eta_B) * p.R_dark * p.delta_det
            + p.p_pair * eta_B * (1 - eta_A) * p.R_dark * p.delta_det
            + (p.R_dark * p.delta_det) ** 2
        )
        p_true = p.p_pair * eta_A * eta_B

        return (
            p_true * (p.q_0 + p.p_pair / 2) + 0.5 * p_acc
        ) / (p_true + p_acc)

    def calculate_metrics(self, path):
        a, b = path[0], path[1]
        T = 1.0 / self.Probs_mtx[a, b]
        Q = self.Q_link(a, b)
        W = 1 - 2*Q
        for i in range(2, len(path)):
            a, b = path[i - 1], path[i]
            T_link = 1.0 / self.Probs_mtx[a, b]
            T = (T + T_link) / self.params.P_BSM
            Q = self.Q_link(a, b)
            W_link = 1 - 2*Q
            W_swap = W * np.exp(-T_link/self.params.T_coh) * W_link
            if self.distillation_type == 'multiplexing':
                F1 = F2 = (1 + 3 * W_swap) / 4
                F, P_suc = BBPSSW(F1, F2)
                T = T / P_suc
                W = (4 * F - 1) / 3
            elif self.distillation_type == 'standard':
                W1 = W_swap * np.exp(-T / self.params.T_coh)
                F1 = (1 + 3 * W1) / 4
                F2 = (1 + 3 * W_swap) / 4
                F, P_suc = BBPSSW(F1, F2)
                T = 2 * T / P_suc
                W = (4 * F - 1) / 3
            elif self.distillation_type == None:
                W = W_swap
            else:
                raise ValueError("distillation_type must be 'multiplexing', 'standard', or None")

        QBER = (1 - W) / 2
        R_raw = self.entanglement_rate(T)
        R = 0.5 * R_raw
        H = binary_entropy(QBER)
        SKR = R * (1 - 2 * H)
        return T, QBER, SKR

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
        Compute SKR, QBER, and total time for every reachable node from
        *source*.

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
            T   = self.sequential_time(path)
            Q   = self.qber(path)
            R   = self.entanglement_rate(T)
            H   = binary_entropy(Q)
            SKR = R * (1 - 2 * H)
            records.append({
                'total_time': T,
                'Q':          Q,
                'SKR':        SKR,
                'path':       path,
            })

        return pd.DataFrame(records, index=list(paths.keys()))

    # ------------------------------------------------------------------
    # HTML visualisation export
    # ------------------------------------------------------------------

    def export_html(self, source, output_path="repeater_viz.html"):
        """
        Pre-compute all paths from *source* and export a self-contained
        interactive HTML visualisation using Plotly.

        Layout
        ------
        - 2D equirectangular projection of the sphere coordinates so that
          distances on screen correspond to physical km.
        - Left panel: legend and per-click info (SKR, QBER, hops, distances).
        - Graph canvas: pan with drag, zoom with scroll wheel.

        Node colours
        ------------
        - Gold  : source node
        - Green gradient (dark -> bright, log scale) : SKR > 0
        - Red   : SKR <= 0 (link established but not secret-key viable)
        - Grey  : unreachable from source

        Interaction
        -----------
        - Hover over a node to see its index and SKR.
        - Click a node to highlight the optimal path to it and populate
          the info panel with SKR, QBER, hop count, total distance, and
          per-hop distances.

        Parameters
        ----------
        source : int
            Source node index.
        output_path : str
            File path for the output HTML file.
        """
        if self.coords is None:
            raise ValueError(
                "coords must be provided to QuantumRepeaterNetwork "
                "in order to use export_html."
            )

        print(f"Computing all paths from node {source}...")
        df = self.analyze_all_paths(source)

        # -- 2D projection --
        x, y = _sphere_to_2d(self.coords, self.scale_km)
        N = len(x)

        # -- Lookup tables keyed by destination node index --
        skr_by_node  = {int(i): row['SKR']        for i, row in df.iterrows()}
        path_by_node = {int(i): row['path']        for i, row in df.iterrows()}
        qber_by_node = {int(i): row['Q']           for i, row in df.iterrows()}
        time_by_node = {int(i): row['total_time']  for i, row in df.iterrows()}

        # -- Log-scale colour mapping for positive-SKR nodes --
        pos_skrs = [v for v in skr_by_node.values() if v > 0]
        log_min = math.log10(min(pos_skrs)) if pos_skrs else 0.0
        log_max = math.log10(max(pos_skrs)) if pos_skrs else 1.0

        def skr_to_green(skr):
            """Map positive SKR value to an rgb green string (dark->bright)."""
            t = (math.log10(skr) - log_min) / (log_max - log_min) \
                if log_max != log_min else 1.0
            t = max(0.0, min(1.0, t))
            # dark green (0,80,0) -> bright green (0,255,100)
            return f'rgb(0,{int(80 + t * 175)},{int(t * 100)})'

        node_colors = []
        node_skr_labels = []
        for i in range(N):
            if i == source:
                node_colors.append('rgb(255,215,0)')        # gold
                node_skr_labels.append('SOURCE')
            elif i not in skr_by_node:
                node_colors.append('rgb(80,80,80)')          # grey
                node_skr_labels.append('Unreachable')
            elif skr_by_node[i] <= 0:
                node_colors.append('rgb(200,40,40)')         # red
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')
            else:
                node_colors.append(skr_to_green(skr_by_node[i]))
                node_skr_labels.append(f'{skr_by_node[i]:.3e} bit/s')

        # -- Background edge coordinates --
        edge_x, edge_y = [], []
        rows, cols = np.where(np.triu(self.A, k=1) > 0)
        for i, j in zip(rows, cols):
            edge_x += [float(x[i]), float(x[j]), None]
            edge_y += [float(y[i]), float(y[j]), None]

        # -- Path data for JavaScript (serialised as JSON) --
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
            'log_min':      log_min,
            'log_max':      log_max,
        }

        html = _build_html(js_data)
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

def _build_html(d):
    """Return the full self-contained HTML string for the visualisation."""

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
    --bg:       #050810;
    --panel:    #0b0f1a;
    --border:   #1a2740;
    --accent:   #00d4ff;
    --green:    #00ff9d;
    --gold:     #ffd700;
    --red:      #ff3a3a;
    --text:     #c8d8e8;
    --dim:      #4a6080;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Rajdhani', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ---- header ---- */
  header {{
    padding: 11px 22px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 18px;
    flex-shrink: 0;
    background: var(--panel);
  }}
  header h1 {{
    font-family: var(--mono);
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }}
  header span {{
    font-size: 11px;
    color: var(--dim);
    font-family: var(--mono);
  }}

  /* ---- body layout ---- */
  .main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ---- side panel ---- */
  .panel {{
    width: 290px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }}
  .section {{
    padding: 18px 20px;
    border-bottom: 1px solid var(--border);
  }}
  .section h2 {{
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.22em;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 13px;
  }}

  /* legend */
  .leg-row {{
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 7px;
    font-size: 13px;
  }}
  .leg-dot {{
    width: 11px; height: 11px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .colorbar {{
    height: 10px;
    border-radius: 3px;
    background: linear-gradient(to right, rgb(0,80,0), rgb(0,255,100));
    margin: 7px 0 3px;
  }}
  .cb-labels {{
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
  }}

  /* info box */
  #info-box {{
    flex: 1;
    padding: 18px 20px;
    overflow-y: auto;
  }}
  .placeholder {{
    color: var(--dim);
    font-size: 12px;
    line-height: 1.9;
    font-family: var(--mono);
  }}
  .placeholder::before {{
    content: '> ';
    color: var(--accent);
  }}
  .info-title {{
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.22em;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 16px;
  }}
  .metric {{ margin-bottom: 15px; }}
  .metric-label {{
    font-size: 9px;
    font-family: var(--mono);
    color: var(--dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 3px;
  }}
  .metric-value {{
    font-family: var(--mono);
    font-size: 17px;
    color: var(--green);
  }}
  .metric-value.bad     {{ color: var(--red);  }}
  .metric-value.neutral {{ color: var(--text); }}

  .hop-list {{ margin-top: 16px; }}
  .hop-list h3 {{
    font-size: 9px;
    font-family: var(--mono);
    color: var(--dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 9px;
  }}
  .hop-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
    font-family: var(--mono);
    font-size: 12px;
  }}
  .hop-arrow {{ color: var(--accent); font-size: 9px; }}
  .hop-dist  {{ color: var(--dim); margin-left: auto; }}

  /* ---- graph ---- */
  #graph {{ flex: 1; min-width: 0; }}
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
      <div class="leg-row">
        <div class="leg-dot" style="background:var(--gold)"></div>
        <span>Source node</span>
      </div>
      <div class="leg-row">
        <div class="leg-dot" style="background:rgb(200,40,40)"></div>
        <span>SKR &le; 0 &nbsp;(not viable)</span>
      </div>
      <div class="leg-row">
        <div class="leg-dot" style="background:rgb(80,80,80)"></div>
        <span>Unreachable</span>
      </div>
      <div style="margin-top:8px;">
        <div class="colorbar"></div>
        <div class="cb-labels">
          <span id="skr-min"></span>
          <span>SKR (bit/s, log)</span>
          <span id="skr-max"></span>
        </div>
      </div>
    </div>

    <div id="info-box">
      <p class="placeholder">Click a node to inspect its optimal path</p>
    </div>

  </div>

  <div id="graph"></div>
</div>

<script>
const D = {data_json};

// header
document.getElementById('hdr').textContent =
  `source: node ${{D.source}}  ·  scale: ${{D.scale_label}}  ·  architecture: ${{D.architecture}}`;

// legend labels
document.getElementById('skr-min').textContent = '10^' + D.log_min.toFixed(1);
document.getElementById('skr-max').textContent = '10^' + D.log_max.toFixed(1);

// -- scientific notation formatter --
function sci(v) {{
  if (v == null) return '—';
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = (v / Math.pow(10, e)).toFixed(2);
  return m + ' \u00d7 10\u207b' + '\u2070'.replace('0',
    String(Math.abs(e)).split('').map(c =>
      '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079'[+c]
    ).join('')
  );
}}
// simpler readable version used in info panel
function fmtSKR(v) {{
  if (v == null) return '—';
  return v.toExponential(3) + ' bit/s';
}}

// ---- Plotly traces ----

const trEdge = {{
  type: 'scatter', mode: 'lines',
  x: D.edge_x, y: D.edge_y,
  line: {{ color: 'rgba(0,180,255,0.07)', width: 0.8 }},
  hoverinfo: 'skip', name: 'edges',
}};

// highlighted path – starts empty, updated on click
const trPath = {{
  type: 'scatter', mode: 'lines',
  x: [], y: [],
  line: {{ color: 'rgba(0,255,157,0.9)', width: 3 }},
  hoverinfo: 'skip', name: 'path',
}};

const trNode = {{
  type: 'scatter', mode: 'markers',
  x: D.x, y: D.y,
  marker: {{ size: 6, color: D.node_colors, line: {{ width: 0 }} }},
  text: D.skr_labels,
  customdata: D.x.map((_, i) => i),
  hovertemplate:
    '<b>Node %{{customdata}}</b><br>' +
    'SKR: %{{text}}<br>' +
    '(%{{x:.1f}} km, %{{y:.1f}} km)<extra></extra>',
  name: 'nodes',
}};

const layout = {{
  paper_bgcolor: '#050810',
  plot_bgcolor:  '#050810',
  margin: {{ t: 10, b: 10, l: 10, r: 10 }},
  xaxis: {{
    title: {{ text: 'km', font: {{ color: '#4a6080', size: 11 }} }},
    gridcolor: '#0d1825', zerolinecolor: '#1a2740',
    tickfont: {{ color: '#4a6080', size: 10 }}, color: '#4a6080',
  }},
  yaxis: {{
    title: {{ text: 'km', font: {{ color: '#4a6080', size: 11 }} }},
    gridcolor: '#0d1825', zerolinecolor: '#1a2740',
    tickfont: {{ color: '#4a6080', size: 10 }}, color: '#4a6080',
    scaleanchor: 'x', scaleratio: 1,
  }},
  showlegend: false,
  dragmode: 'pan',
}};

Plotly.newPlot('graph', [trEdge, trPath, trNode], layout,
  {{ scrollZoom: true, responsive: true,
     displayModeBar: true,
     modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'] }});

// ---- click handler ----
document.getElementById('graph').on('plotly_click', function(ev) {{
  const pt = ev.points[0];
  if (!pt || pt.data.name !== 'nodes') return;

  const idx = pt.customdata;
  if (idx === D.source) return;

  const pd = D.path_data[idx];

  // clear path if unreachable
  if (!pd) {{
    Plotly.restyle('graph', {{ x: [[]], y: [[]] }}, [1]);
    document.getElementById('info-box').innerHTML =
      `<p class="info-title">Node ${{idx}}</p>
       <div class="metric">
         <div class="metric-label">Status</div>
         <div class="metric-value bad">Unreachable</div>
       </div>`;
    return;
  }}

  // build highlighted path coords
  const path = pd.path;
  const px = [], py = [];
  for (let i = 0; i < path.length - 1; i++) {{
    px.push(D.x[path[i]], D.x[path[i+1]], null);
    py.push(D.y[path[i]], D.y[path[i+1]], null);
  }}
  Plotly.restyle('graph', {{ x: [px], y: [py] }}, [1]);

  // build hop detail rows
  let hopHtml = '';
  for (let i = 0; i < path.length - 1; i++) {{
    hopHtml += `<div class="hop-item">
      <span>${{path[i]}}</span>
      <span class="hop-arrow">──▶</span>
      <span>${{path[i+1]}}</span>
      <span class="hop-dist">${{pd.hop_dists[i].toFixed(2)}} km</span>
    </div>`;
  }}

  const skrClass = pd.skr > 0 ? '' : 'bad';

  document.getElementById('info-box').innerHTML = `
    <p class="info-title">Node ${{D.source}} &rarr; Node ${{idx}}</p>

    <div class="metric">
      <div class="metric-label">Secret Key Rate</div>
      <div class="metric-value ${{skrClass}}">${{fmtSKR(pd.skr)}}</div>
    </div>
    <div class="metric">
      <div class="metric-label">QBER</div>
      <div class="metric-value neutral">${{(pd.qber*100).toFixed(3)}} %</div>
    </div>
    <div class="metric">
      <div class="metric-label">Hops</div>
      <div class="metric-value neutral">${{path.length - 1}}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Total distance</div>
      <div class="metric-value neutral">${{pd.total_dist.toFixed(2)}} km</div>
    </div>

    <div class="hop-list">
      <h3>Path detail</h3>
      ${{hopHtml}}
    </div>`;
}});
</script>
</body>
</html>"""
