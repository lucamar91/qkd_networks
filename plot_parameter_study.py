"""
plot_parameter_study.py

Reads parameter_results.csv (produced by parameter_study.py) and makes one
figure per swept parameter, with subplots for SKR, QBER, Fidelity,
entanglement rate, and reachability vs. that parameter's value, one line
per mux case (mean over reps, shaded +/- std).

Everything needed to re-make/re-style the plots lives in this file and the
CSV -- edit METRICS / labels / styling below as needed, no need to rerun
the (expensive) simulation.
"""

import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "parameter_results.csv"

METRICS = [
    ("avg_skr",       "Avg. SKR (1/s)",         True),   # (col, ylabel, log_y)
    ("avg_qber",      "Avg. QBER",               False),
    ("avg_fidelity",  "Avg. Fidelity",           False),
    ("avg_ent_rate",  "Avg. Entanglement Rate (1/s)", True),
    ("reachability",  "Reachability",            False),
]

PARAM_LABELS = {
    "P_BSM":    "P_BSM",
    "T_coh":     "T_coh (s)",
    "eta_total": "Total efficiency (eta_M * eta_c)",
    "alpha":     "Fibre loss alpha (dB/km)",
    "V":         "Visibility V",
}

MUX_STYLES = {
    "no_mux": dict(color="tab:blue", marker="o", label="No mux"),
}


def plot_one_parameter(df, param_label, outfile):
    sub = df[df["param"] == param_label]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))

    for ax, (col, ylabel, log_y) in zip(axes, METRICS):
        for mux_label, style in MUX_STYLES.items():
            mux_sub = sub[sub["mux_type"] == mux_label]
            grouped = mux_sub.groupby("param_value")[col].agg(["mean", "std"])
            grouped = grouped.sort_index()
            ax.errorbar(
                grouped.index, grouped["mean"], yerr=grouped["std"],
                capsize=3, **style,
            )
        ax.set_xlabel(PARAM_LABELS.get(param_label, param_label))
        ax.set_ylabel(ylabel)
        if param_label == "T_coh":
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Network metrics vs. {PARAM_LABELS.get(param_label, param_label)} "
                 f"(density fixed at 10^0 nodes/km^2)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    for param_label in df["param"].unique():
        outfile = f"parameter_study_{param_label}.png"
        plot_one_parameter(df, param_label, outfile)
