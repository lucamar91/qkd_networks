"""
generate_viz.py

Run this script to build the network, compute all paths from a chosen
source node, and export the interactive HTML visualisation.

Usage
-----
    python generate_viz.py
"""

from sequential_repeaters_dist import RepeaterParams, QuantumRepeaterNetwork, build_s2_graph

# ── 1. Physical parameters ───────────────────────────────────────────────────

params = RepeaterParams()
params.p_det     = 0.95        # detector efficiency
params.alpha     = 0.18        # fibre loss [dB/km]
params.q_0       = 0.01        # baseline QBER
params.nu        = 1e9         # repetition rate [Hz]
params.R_dark    = 100         # dark-count rate [Hz]
params.delta_det = 100e-12     # time-gate duration [s]
params.p_pair    = 0.05        # pair-generation probability per pulse
params.eta_c     = 0.8         # source-to-fibre coupling efficiency
params.P_BSM     = 0.5         # BSM success probability (linear optics)
params.T_coh     = 0.05

# ── 2. Graph preset ───────────────────────────────────────────────────────────
# scale : 'city' | 'country' | 'europe'  (sets the physical size)
# N     : number of nodes

SCALE = 'city'
N     = 1000       # keep small for a quick test; use 1000 for full run

A, dist, coords = build_s2_graph(N=N, beta=2.6261, mu=0.0233, scale=SCALE)

# ── 3. Build network ──────────────────────────────────────────────────────────
# architecture : 'node' | 'midpoint'

net = QuantumRepeaterNetwork(
    params, A, dist, coords,
    architecture='midpoint',
    scale=SCALE, distillation_type='multiplexing'
)

# ── 4. Export ─────────────────────────────────────────────────────────────────

SOURCE = 0   # change to any node index you like

net.export_html(source=SOURCE, output_path="repeater_viz.html")
