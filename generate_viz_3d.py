"""
generate_viz_3d.py

Build the network, compute all paths from a chosen source node, and export
the interactive 3D globe visualisation.

Usage
-----
    python generate_viz_3d.py
"""

from sequential_repeaters_dist import RepeaterParams, QuantumRepeaterNetwork, build_s2_graph

# ── 1. Physical parameters ───────────────────────────────────────────────────

params = RepeaterParams()
params.p_det     = 0.95
params.alpha     = 0.18
params.q_0       = 0.01
params.nu        = 1e9
params.R_dark    = 100
params.delta_det = 100e-12
params.p_pair    = 0.05
params.eta_c     = 0.8
params.P_BSM     = 0.5
params.T_coh     = 0.05     # memory coherence time [s]

# ── 2. Graph preset ───────────────────────────────────────────────────────────
# The 3D globe uses the raw unit-sphere coordinates from build_s2_graph, so
# only S2 graphs are supported (build_graph does not produce sphere coords).

SCALE = 'city'   # 'city' | 'country' | 'europe'
N     = 200      # start small for a quick test; use 1000 for the full run

A, dist, coords = build_s2_graph(N=N, beta=2.6261, mu=0.0233, scale=SCALE)

# ── 3. Build network ──────────────────────────────────────────────────────────

net = QuantumRepeaterNetwork(
    params, A, dist, coords,
    architecture='node',
    scale=SCALE,
    distillation_type=None,   # None | 'multiplexing' | 'standard'
)

# ── 4. Export 3D globe ───────────────────────────────────────────────────────
# sphere_radius controls the visual size of the globe in the HTML scene.
# The default of 1.0 is fine; increase slightly if nodes look cramped.

SOURCE = 0

net.export_html_3d(source=SOURCE, output_path="repeater_viz_3d.html",
                   sphere_radius=1.0)

# You can also export the 2D version at the same time if you want both:
# net.export_html(source=SOURCE, output_path="repeater_viz_2d.html")
