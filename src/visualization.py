"""
visualization.py — Generates publication-quality figures from benchmark results.

Reads results/joint_noise_benchmark_results.csv and writes three figures
(PNG + PDF) to figures/:

    fig1_accuracy_vs_epsilon     — Privacy-utility trade-off curve
    fig2_noise_decomposition     — Stacked variance: CPAD vs DP residual
    fig3_latency_comparison      — Plaintext / HE-only / Joint latency bars
"""

import logging
import os
import sys
import warnings

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
except ImportError:
    logging.critical("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    logging.critical("pandas not installed. Run: pip install pandas")
    sys.exit(1)

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

RESULTS_CSV = "./results/joint_noise_benchmark_results.csv"
FIGURES_DIR = "./figures"
DPI         = 300

PLAINTEXT_ACC_PCT = 90.6
HE_ONLY_ACC_PCT   = 85.0

FONT_FAMILY  = "DejaVu Serif"
LABEL_SIZE   = 13
TICK_SIZE    = 11
TITLE_SIZE   = 14
LEGEND_SIZE  = 11

C_JOINT     = "#1f78b4"
C_PLAIN     = "#33a02c"
C_HE        = "#e31a1c"
C_CPAD      = "#1f78b4"
C_DP        = "#ff7f00"
C_FREE      = "#b2df8a"
C_PT_BAR    = "#33a02c"
C_HE_BAR    = "#e31a1c"
C_JOINT_BAR = "#1f78b4"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family":       FONT_FAMILY,
        "axes.labelsize":    LABEL_SIZE,
        "axes.titlesize":    TITLE_SIZE,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "legend.fontsize":   LEGEND_SIZE,
        "figure.dpi":        100,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        log.error("CSV not found: %s. Run benchmarking.py first.", path)
        sys.exit(1)
    df = pd.read_csv(path).sort_values("epsilon").reset_index(drop=True)
    df["accuracy_pct"] = df["accuracy"] * 100
    df["var_cpad"]     = df["sigma_c"] ** 2
    df["var_extra"]    = df["sigma_extra_injected"] ** 2
    df["var_total"]    = df["sigma_total"] ** 2
    return df


def save_figure(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        log.info("Saved %s", path)


def plot_accuracy_vs_epsilon(df: pd.DataFrame) -> None:
    """Privacy-utility trade-off curve with Free-DP region shading."""
    fig, ax = plt.subplots(figsize=(8, 5))

    eps    = df["epsilon"].values
    acc    = df["accuracy_pct"].values
    is_free = df["free_dp_achieved"].values.astype(bool)

    ax.plot(eps, acc, color=C_JOINT, linewidth=2.2, marker="o",
            markersize=7, label="Joint HE + IND-CPAD + DP", zorder=5)

    if is_free.any():
        free_eps_min = float(eps[is_free].min())
        ax.axvspan(free_eps_min, eps.max() * 1.5,
                   alpha=0.12, color=C_FREE, label=f"Free DP region (ε ≥ {free_eps_min})")
        ax.axvline(free_eps_min, color=C_FREE, linewidth=1.5, linestyle=":", alpha=0.9)
        ax.annotate(
            f"Free DP\nthreshold\nε = {free_eps_min}",
            xy=(free_eps_min, float(acc[is_free][0])),
            xytext=(free_eps_min * 0.45, float(acc[is_free][0]) - 5.5),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            color="gray",
        )

    ax.axhline(PLAINTEXT_ACC_PCT, color=C_PLAIN, linewidth=1.6,
               linestyle="--", label=f"Plaintext Baseline ({PLAINTEXT_ACC_PCT}%)")
    ax.axhline(HE_ONLY_ACC_PCT,   color=C_HE,    linewidth=1.6,
               linestyle=":",  label=f"HE-Only Baseline ({HE_ONLY_ACC_PCT}%)")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget  ε  (log scale)", labelpad=8)
    ax.set_ylabel("Inference Accuracy (%)", labelpad=8)
    ax.set_title("Figure 1 · Privacy-Utility Trade-off:\nJoint IND-CPAD + Gaussian DP Mechanism", pad=12)
    ax.set_xlim(eps.min() * 0.7, eps.max() * 1.5)
    ax.set_ylim(max(0, min(acc) - 10), min(100, PLAINTEXT_ACC_PCT + 5))
    ax.set_xticks(eps)
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticklabels([f"{e:g}" for e in eps], fontsize=TICK_SIZE)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, "fig1_accuracy_vs_epsilon")
    plt.close(fig)


def plot_noise_decomposition(df: pd.DataFrame) -> None:
    """Stacked bar chart: CPAD variance + DP residual variance per ε, with σ_total overlay."""
    fig, ax = plt.subplots(figsize=(9, 5))

    eps_labels = [f"ε={e:g}" for e in df["epsilon"].values]
    x          = np.arange(len(eps_labels))
    width      = 0.55
    var_cpad   = df["var_cpad"].values
    var_extra  = df["var_extra"].values
    is_free    = df["free_dp_achieved"].values.astype(bool)

    ax.bar(x, var_cpad,  width, label="CPAD variance  σ_c² = 0.01",
           color=C_CPAD, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.bar(x, var_extra, width, bottom=var_cpad,
           label="DP residual variance  σ_extra²",
           color=C_DP,   alpha=0.85, edgecolor="white", linewidth=0.6)

    for i, free in enumerate(is_free):
        if free:
            ax.bar(x[i], var_cpad[i], width,
                   color="none", edgecolor=C_FREE, linewidth=2.5, linestyle="--", zorder=6)
            ax.text(x[i], var_cpad[i] + 0.0008, "Free\nDP ✔",
                    ha="center", va="bottom", fontsize=8.5,
                    color="#2e8b57", fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(x, np.sqrt(df["var_total"].values), color="black", linewidth=1.4,
             marker="D", markersize=5, linestyle="-.", alpha=0.6,
             label="σ_total (right axis)")
    ax2.set_ylabel("σ_total  (combined std dev)", fontsize=LABEL_SIZE - 1, labelpad=8)
    ax2.tick_params(labelsize=TICK_SIZE - 1)
    ax2.spines["right"].set_visible(True)

    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Privacy Budget  ε", labelpad=8)
    ax.set_ylabel("Noise Variance  σ²", labelpad=8)
    ax.set_title("Figure 2 · Noise Budget Decomposition:\nCryptographic (CPAD) vs. Algorithmic (DP) Variance", pad=12)

    handles = [
        mpatches.Patch(color=C_CPAD, alpha=0.85, label="CPAD variance  σ_c² = 0.01"),
        mpatches.Patch(color=C_DP,   alpha=0.85, label="DP residual variance  σ_extra²"),
        Line2D([0], [0], color="black", linewidth=1.4, linestyle="-.",
               marker="D", markersize=5, label="σ_total (right axis)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=LEGEND_SIZE - 0.5)
    fig.tight_layout()
    save_figure(fig, "fig2_noise_decomposition")
    plt.close(fig)


def plot_latency_comparison(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: plaintext / HE-only / joint latencies per ε.
    Log-scale y-axis is necessary — plaintext is ~0.001s vs ~0.5s for HE.

    # TODO: if per-run latency data is available (not just mean), add
    # error bars to the joint bars for a stronger empirical claim.
    """
    PLAINTEXT_LAT_MS = 0.001
    HE_ONLY_LAT_S    = 0.42

    eps_labels = [f"ε={e:g}" for e in df["epsilon"].values]
    n_groups   = len(eps_labels)
    x          = np.arange(n_groups)
    width      = 0.26
    offsets    = [-width, 0, width]
    joint_lats = df["avg_latency_per_sample"].values

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(x + offsets[0], [PLAINTEXT_LAT_MS] * n_groups, width,
           color=C_PT_BAR, alpha=0.80, edgecolor="white",
           label=f"Plaintext ({PLAINTEXT_LAT_MS*1000:.1f} ms/img)")
    ax.bar(x + offsets[1], [HE_ONLY_LAT_S] * n_groups, width,
           color=C_HE_BAR, alpha=0.80, edgecolor="white",
           label=f"HE-Only (avg {HE_ONLY_LAT_S:.2f} s/img)")
    bars_jnt = ax.bar(x + offsets[2], joint_lats, width,
                      color=C_JOINT_BAR, alpha=0.80, edgecolor="white",
                      label="Joint HE + CPAD + DP")

    for rect, val in zip(bars_jnt, joint_lats):
        ax.text(rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.008,
                f"{val:.3f}s", ha="center", va="bottom",
                fontsize=8, color="#1f3d6b")

    max_joint = float(joint_lats.max())
    ax.annotate(
        f"Noise injection overhead\n≈ {(max_joint - HE_ONLY_LAT_S):.3f} s vs HE-only",
        xy=(x[-1] + offsets[2], max_joint),
        xytext=(x[-1] + offsets[2] - 1.6, max_joint + 0.08),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.1),
        color="gray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Privacy Budget  ε", labelpad=8)
    ax.set_ylabel("Avg Latency per Image (seconds)", labelpad=8)
    ax.set_title("Figure 3 · Computational Overhead:\nPlaintext vs. HE-Only vs. Joint HE + IND-CPAD + DP", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
    ))
    fig.tight_layout()
    save_figure(fig, "fig3_latency_comparison")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    configure_style()

    df = load_results()
    log.info("Loaded %d rows from %s", len(df), RESULTS_CSV)
    log.info("\n%s", df[["epsilon", "accuracy_pct", "sigma_total",
                          "free_dp_achieved", "avg_latency_per_sample"]].to_string(index=False))

    log.info("Generating Figure 1 — Accuracy vs ε")
    plot_accuracy_vs_epsilon(df)

    log.info("Generating Figure 2 — Noise decomposition")
    plot_noise_decomposition(df)

    log.info("Generating Figure 3 — Latency comparison")
    plot_latency_comparison(df)

    log.info("All figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    main()
