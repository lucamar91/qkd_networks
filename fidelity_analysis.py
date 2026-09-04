"""
fidelity_analysis.py
====================
Visualisation suite for comparing quantum repeater performance
WITH vs WITHOUT entanglement distillation.

Generates a multi-panel figure:
  1. QBER vs total path distance   (scatter, colour-coded by n_hops)
  2. Fidelity vs total path distance
  3. SKR vs total path distance    (log-y)
  4. SKR ratio (dist / no-dist) vs total path distance – where >1 distillation wins
  5. SKR ratio vs MAX inter-node hop distance
  6. Distribution of per-hop distances for paths where distillation wins vs loses
  7. Fidelity improvement ratio vs max hop distance
  8. QBER improvement vs max hop distance

Run this file directly to produce the figure.  Edit the "USER CONFIG" block
at the bottom to match your network setup (or pass your own DataFrames).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, ".")  # make sure the project root is on the path

# ──────────────────────────────────────────────────────────────────────────────
# Core comparison helper
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(net_no_dist, net_dist, source: int = 0) -> pd.DataFrame:
    """
    Run analyze_all_paths for both networks and merge into a single DataFrame.

    Parameters
    ----------
    net_no_dist : QuantumRepeaterNetwork   distillation=False
    net_dist    : QuantumRepeaterNetwork   distillation=True
    source      : int

    Returns
    -------
    df : pd.DataFrame
        Columns:
          path, n_hops, total_dist_km, max_hop_km, mean_hop_km,
          Q_no, Q_dist, SKR_no, SKR_dist,
          fidelity_no, fidelity_dist,
          skr_ratio, qber_ratio, fidelity_ratio,
          dist_wins   (bool – distillation gives strictly higher SKR)
    """
    df_no   = net_no_dist.analyze_all_paths(source=source)
    df_dist = net_dist.analyze_all_paths(source=source)

    # Merge on destination index
    merged = df_no[['Q', 'SKR', 'path']].join(
        df_dist[['Q', 'SKR']], lsuffix='_no', rsuffix='_dist', how='inner'
    )
    merged['path'] = df_no['path']

    # ── path geometry ──────────────────────────────────────────────────
    def hop_dists(path):
        d = net_no_dist.dist
        return [float(d[path[i - 1], path[i]]) for i in range(1, len(path))]

    merged['hop_dist_list'] = merged['path'].apply(hop_dists)
    merged['n_hops']        = merged['hop_dist_list'].apply(len)
    merged['total_dist_km'] = merged['hop_dist_list'].apply(sum)
    merged['max_hop_km']    = merged['hop_dist_list'].apply(max)
    merged['mean_hop_km']   = merged['hop_dist_list'].apply(np.mean)

    # ── fidelity from QBER: F = (1 + 3*(1-2Q)) / 4  (Werner state) ───
    merged['fidelity_no']   = (1 + 3 * (1 - 2 * merged['Q_no']))   / 4
    merged['fidelity_dist'] = (1 + 3 * (1 - 2 * merged['Q_dist'])) / 4

    # ── ratios ────────────────────────────────────────────────────────
    eps = 1e-30
    merged['skr_ratio']      = merged['SKR_dist'] / (merged['SKR_no'] + eps)
    merged['qber_ratio']     = merged['Q_dist']   / (merged['Q_no']   + eps)
    merged['fidelity_ratio'] = merged['fidelity_dist'] / (merged['fidelity_no'] + eps)
    merged['dist_wins']      = merged['SKR_dist'] > merged['SKR_no']

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    'no_dist': '#4fc3f7',   # sky blue
    'dist':    '#ff7043',   # deep orange
    'win':     '#66bb6a',   # green
    'lose':    '#ef5350',   # red
    'neutral': '#90a4ae',
    'bg':      '#0d1117',
    'panel':   '#161b22',
    'text':    '#c9d1d9',
    'grid':    '#21262d',
}


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(_PALETTE['panel'])
    ax.tick_params(colors=_PALETTE['text'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE['grid'])
    ax.grid(color=_PALETTE['grid'], linewidth=0.6, linestyle='--')
    ax.set_title(title, color=_PALETTE['text'], fontsize=9, pad=6)
    ax.set_xlabel(xlabel, color=_PALETTE['neutral'], fontsize=8)
    ax.set_ylabel(ylabel, color=_PALETTE['neutral'], fontsize=8)


def plot_fidelity_analysis(df: pd.DataFrame,
                           save_path: str = 'fidelity_analysis.png',
                           dpi: int = 180):
    """
    Generate the 8-panel comparison figure.

    Parameters
    ----------
    df        : output of run_comparison()
    save_path : file to write (PNG)
    dpi       : output resolution
    """
    # ── colour map for n_hops ──────────────────────────────────────────
    hop_vals = df['n_hops'].values
    cmap     = plt.cm.viridis
    norm     = mcolors.Normalize(vmin=hop_vals.min(), vmax=hop_vals.max())
    hop_cols = cmap(norm(hop_vals))

    fig = plt.figure(figsize=(18, 14), facecolor=_PALETTE['bg'])
    fig.suptitle('Quantum Repeater: Distillation Fidelity Analysis',
                 color=_PALETTE['text'], fontsize=14, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── 1. QBER vs total distance ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, 'QBER vs total path distance', 'Total distance (km)', 'QBER')
    ax1.scatter(df['total_dist_km'], df['Q_no'],   s=12, alpha=0.6,
                color=_PALETTE['no_dist'], label='No distillation', lw=0)
    ax1.scatter(df['total_dist_km'], df['Q_dist'], s=12, alpha=0.6,
                color=_PALETTE['dist'],    label='Distillation', lw=0)
    ax1.axhline(0.11, color='#ffd54f', lw=1, ls='--', label='BB84 limit (11%)')
    ax1.legend(fontsize=7, facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.8)

    # ── 2. Fidelity vs total distance ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, 'Fidelity vs total path distance', 'Total distance (km)', 'Fidelity F')
    ax2.scatter(df['total_dist_km'], df['fidelity_no'],   s=12, alpha=0.6,
                color=_PALETTE['no_dist'], lw=0)
    ax2.scatter(df['total_dist_km'], df['fidelity_dist'], s=12, alpha=0.6,
                color=_PALETTE['dist'],    lw=0)
    ax2.axhline(0.5, color='#ffd54f', lw=1, ls='--', label='Classical limit')
    ax2.legend(fontsize=7, facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.8)

    # ── 3. SKR vs total distance (log) ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3, 'SKR vs total path distance', 'Total distance (km)', 'SKR (bit/s)')

    skr_no_pos   = df['SKR_no'].clip(lower=1e-20)
    skr_dist_pos = df['SKR_dist'].clip(lower=1e-20)
    ax3.scatter(df['total_dist_km'], skr_no_pos,   s=12, alpha=0.6,
                color=_PALETTE['no_dist'], lw=0)
    ax3.scatter(df['total_dist_km'], skr_dist_pos, s=12, alpha=0.6,
                color=_PALETTE['dist'],    lw=0)
    ax3.set_yscale('log')

    # ── 4. SKR ratio vs total distance ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _style_ax(ax4, 'SKR ratio vs total path distance',
              'Total distance (km)', 'SKR_dist / SKR_no')
    ratio_clipped = df['skr_ratio'].clip(1e-3, 1e3)
    win  = df['dist_wins']
    ax4.scatter(df.loc[ win, 'total_dist_km'], ratio_clipped[ win],
                s=14, alpha=0.7, color=_PALETTE['win'],  lw=0, label='Dist. wins')
    ax4.scatter(df.loc[~win, 'total_dist_km'], ratio_clipped[~win],
                s=14, alpha=0.7, color=_PALETTE['lose'], lw=0, label='Dist. loses')
    ax4.axhline(1.0, color='#ffd54f', lw=1.2, ls='--')
    ax4.set_yscale('log')
    ax4.legend(fontsize=7, facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.8)

    # ── 5. SKR ratio vs max inter-node hop ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _style_ax(ax5, 'SKR ratio vs max hop distance',
              'Max inter-node hop (km)', 'SKR_dist / SKR_no')
    ax5.scatter(df.loc[ win, 'max_hop_km'], ratio_clipped[ win],
                s=14, alpha=0.7, color=_PALETTE['win'],  lw=0, label='Dist. wins')
    ax5.scatter(df.loc[~win, 'max_hop_km'], ratio_clipped[~win],
                s=14, alpha=0.7, color=_PALETTE['lose'], lw=0, label='Dist. loses')
    ax5.axhline(1.0, color='#ffd54f', lw=1.2, ls='--')
    ax5.set_yscale('log')
    ax5.legend(fontsize=7, facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.8)

    # ── 6. Per-hop distance distributions (win vs lose) ───────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _style_ax(ax6, 'Hop-distance distribution: win vs lose',
              'Inter-node hop distance (km)', 'Density')

    hops_win  = np.concatenate(df.loc[ win, 'hop_dist_list'].tolist()) if win.any()  else []
    hops_lose = np.concatenate(df.loc[~win, 'hop_dist_list'].tolist()) if (~win).any() else []

    bins = np.linspace(0, max(df['max_hop_km']) * 1.05, 40)
    if len(hops_win) > 0:
        ax6.hist(hops_win,  bins=bins, density=True, alpha=0.6,
                 color=_PALETTE['win'],  label='Dist. wins')
    if len(hops_lose) > 0:
        ax6.hist(hops_lose, bins=bins, density=True, alpha=0.6,
                 color=_PALETTE['lose'], label='Dist. loses')
    ax6.legend(fontsize=7, facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.8)

    # ── 7. Fidelity improvement ratio vs max hop ──────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    _style_ax(ax7, 'Fidelity improvement vs max hop distance',
              'Max inter-node hop (km)', 'F_dist / F_no')
    fid_ratio = df['fidelity_ratio'].clip(0.5, 2.0)
    sc = ax7.scatter(df['max_hop_km'], fid_ratio, s=14, alpha=0.7,
                     c=hop_vals, cmap='viridis', norm=norm, lw=0)
    ax7.axhline(1.0, color='#ffd54f', lw=1.2, ls='--')
    cb = fig.colorbar(sc, ax=ax7, pad=0.01)
    cb.set_label('# hops', color=_PALETTE['neutral'], fontsize=7)
    cb.ax.yaxis.set_tick_params(color=_PALETTE['text'])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_PALETTE['text'], fontsize=7)

    # ── 8. QBER improvement vs max hop ────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    _style_ax(ax8, 'QBER ratio vs max hop distance (lower=better)',
              'Max inter-node hop (km)', 'Q_dist / Q_no')
    qber_ratio = df['qber_ratio'].clip(0.1, 10)
    sc2 = ax8.scatter(df['max_hop_km'], qber_ratio, s=14, alpha=0.7,
                      c=hop_vals, cmap='viridis', norm=norm, lw=0)
    ax8.axhline(1.0, color='#ffd54f', lw=1.2, ls='--')
    cb2 = fig.colorbar(sc2, ax=ax8, pad=0.01)
    cb2.set_label('# hops', color=_PALETTE['neutral'], fontsize=7)
    cb2.ax.yaxis.set_tick_params(color=_PALETTE['text'])
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color=_PALETTE['text'], fontsize=7)

    # ── 9. Win fraction vs n_hops bar ─────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    _style_ax(ax9, 'Distillation win-rate by hop count',
              'Number of hops', 'Fraction where dist. wins')
    hop_groups = df.groupby('n_hops')['dist_wins'].mean()
    hop_counts = df.groupby('n_hops')['dist_wins'].count()
    bars = ax9.bar(hop_groups.index, hop_groups.values,
                   color=[_PALETTE['win'] if v >= 0.5 else _PALETTE['lose']
                          for v in hop_groups.values],
                   alpha=0.8, width=0.7)
    ax9.axhline(0.5, color='#ffd54f', lw=1.2, ls='--')
    ax9.set_ylim(0, 1)
    # annotate counts
    for bar, (hop, cnt) in zip(bars, hop_counts.items()):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'n={cnt}', ha='center', va='bottom',
                 fontsize=6, color=_PALETTE['neutral'])

    # ── global legend strip ───────────────────────────────────────────
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_PALETTE['no_dist'],
               markersize=7, label='No distillation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_PALETTE['dist'],
               markersize=7, label='With distillation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_PALETTE['win'],
               markersize=7, label='Dist. wins (SKR↑)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_PALETTE['lose'],
               markersize=7, label='Dist. loses (SKR↓)'),
        Line2D([0], [0], color='#ffd54f', lw=1.5, ls='--', label='Break-even / limit'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5, fontsize=8,
               facecolor=_PALETTE['panel'], labelcolor=_PALETTE['text'],
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[✓] Figure saved → {save_path}")
    return fig


def print_summary(df: pd.DataFrame):
    """Print a quick text summary to stdout."""
    win  = df['dist_wins']
    total = len(df)
    print("\n── Distillation fidelity summary ────────────────────────────")
    print(f"  Total paths analysed        : {total}")
    print(f"  Paths where dist. wins SKR  : {win.sum()} ({100*win.mean():.1f} %)")
    print(f"  Median SKR ratio (dist/no)  : {df['skr_ratio'].median():.3f}")
    print(f"  Median fidelity ratio       : {df['fidelity_ratio'].median():.4f}")
    print(f"  Median QBER ratio           : {df['qber_ratio'].median():.4f}")

    # Threshold search: max hop above which distillation helps
    bins  = np.percentile(df['max_hop_km'], np.arange(0, 101, 10))
    print("\n  Win-rate by max-hop decile:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df['max_hop_km'] >= lo) & (df['max_hop_km'] < hi)
        if mask.sum() == 0:
            continue
        wr = df.loc[mask, 'dist_wins'].mean()
        print(f"    {lo:6.1f}–{hi:6.1f} km : {100*wr:5.1f} % win  (n={mask.sum()})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# ── USER CONFIG – edit this block to match your setup ─────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Example: small synthetic hub-and-spoke graph (no external deps) ──
    # Replace with your real network objects as needed.

    import sys
    sys.path.insert(0, '.')

    try:
        from sequential_repeaters_dist import (
            RepeaterParams, QuantumRepeaterNetwork, build_graph
        )
    except ImportError:
        print("[!] Could not import sequential_repeaters_dist.  "
              "Place this script next to that file and try again.")
        sys.exit(1)

    # ── Physical parameters ───────────────────────────────────────────
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
    params.T_coh     = 0.05   # memory coherence time [s]

    # ── Build a small test graph ──────────────────────────────────────
    A, dist = build_graph(n_hubs=8, avg_n_branches=3)

    # ── Two network objects: same graph, distillation on/off ──────────
    common = dict(
        params=params, A=A, dist=dist, coords=None,
        architecture='node', scale='city',
        multiplexing_type=None,
        distillation_level=1,
        distillation_time='before_swap',
    )

    net_no   = QuantumRepeaterNetwork(**common, distillation=False)
    net_dist = QuantumRepeaterNetwork(**common, distillation=True)

    # ── Run comparison from node 0 ────────────────────────────────────
    df = run_comparison(net_no, net_dist, source=0)
    print_summary(df)

    fig = plot_fidelity_analysis(df, save_path='fidelity_analysis.png', dpi=180)
    plt.show()
