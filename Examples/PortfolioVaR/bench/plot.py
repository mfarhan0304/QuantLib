#!/usr/bin/env python3
"""
plot.py  --  Generate all figures for the Portfolio Monte Carlo VaR report.

Usage:
    python bench/plot.py [--csv-dir DIR] [--out-dir DIR]

Defaults:
    --csv-dir  .          (directory containing the CSV files)
    --out-dir  figures/   (output directory for PNG/PDF figures)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FIGSIZE_WIDE  = (8, 4.5)
FIGSIZE_SQUARE = (5.5, 5.0)
FIGSIZE_NARROW = (6, 4)
DPI = 150

COLORS = {
    "seq":     "#2c7bb6",
    "par":     "#d7191c",
    "ideal":   "#aaaaaa",
    "dynamic": "#1a9641",
    "static":  "#d7191c",
    "guided":  "#f97f0f",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def load(csv_dir, fname):
    path = os.path.join(csv_dir, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping", file=sys.stderr)
        return None
    return pd.read_csv(path)

# ---------------------------------------------------------------------------
# Figure 1 — E1: VaR convergence vs N (log-log)
# ---------------------------------------------------------------------------
def fig_convergence(csv_dir, out_dir):
    df = load(csv_dir, "convergence_var.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, col, label in [(axes[0], "var95", "VaR(95%)"),
                           (axes[1], "var99", "VaR(99%)")]:
        ax.loglog(df["n_scenarios"].values, df[col].values, "o-",
                  color=COLORS["seq"], lw=1.8, ms=5, label=label)
        # O(1/sqrt(N)) reference line anchored at largest N
        n_ref = df["n_scenarios"].values
        y_ref = df[col].values[-1] * np.sqrt(n_ref[-1] / n_ref)
        ax.loglog(n_ref, y_ref, "--", color=COLORS["ideal"], lw=1.2,
                  label=r"$O(1/\sqrt{N})$")
        ax.set_xlabel("Number of scenarios N")
        ax.set_ylabel("VaR estimate (USD)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle("Figure 1 — Convergence of VaR estimate vs N (E1)", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "fig1_convergence.png")

# ---------------------------------------------------------------------------
# Figure 2 — E2 (Dixon): Stage timing breakdown — sequential vs parallel
# ---------------------------------------------------------------------------
def fig_stage_timing(csv_dir, out_dir):
    seq = load(csv_dir, "stage_timing_seq.csv")
    par = load(csv_dir, "stage_timing_omp.csv")
    if seq is None or par is None:
        return

    stages = ["rng_scenario_gen", "portfolio_reval", "tail_statistics"]
    labels = ["RNG /\nscenario gen", "Portfolio\nrevaluation", "Tail\nstatistics"]

    seq_t = [seq.loc[seq["stage"] == s, "time_s"].values[0] for s in stages]
    par_t = [par.loc[par["stage"] == s, "time_s"].values[0] for s in stages]

    x = np.arange(len(stages))
    w = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE_NARROW)
    bars_s = ax.bar(x - w/2, seq_t, w, label="Sequential", color=COLORS["seq"])
    bars_p = ax.bar(x + w/2, par_t, w, label="Parallel (8T)", color=COLORS["par"])

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title("Figure 2 — Stage timing: sequential vs parallel (N=1000, T=8)")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    # Annotate bars with % of total
    for bar, pct in zip(bars_s,
                        [seq.loc[seq["stage"]==s, "pct"].values[0] for s in stages]):
        if pct > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save(fig, out_dir, "fig2_stage_timing.png")

# ---------------------------------------------------------------------------
# Figure 3 — E3: Strong scaling — speedup and efficiency
# ---------------------------------------------------------------------------
def fig_strong_scaling(csv_dir, out_dir):
    df = load(csv_dir, "strong_scaling.csv")
    if df is None:
        return

    # Drop the "1_seq" row (sequential baseline label); keep numeric threads
    df = df[df["threads"] != "1_seq"].copy()
    df["threads"] = df["threads"].astype(int)
    df["efficiency"] = df["speedup"] / df["threads"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Speedup
    ax1.plot(df["threads"].values, df["speedup"].values, "o-",
             color=COLORS["par"], lw=2, ms=6, label="Measured")
    ax1.plot(df["threads"].values, df["threads"].values.astype(float), "--",
             color=COLORS["ideal"], lw=1.2, label="Ideal (linear)")
    ax1.set_xlabel("Thread count")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Strong Scaling — Speedup")
    ax1.legend()
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Efficiency
    ax2.plot(df["threads"].values, df["efficiency"].values, "s-",
             color=COLORS["par"], lw=2, ms=6)
    ax2.axhline(100, color=COLORS["ideal"], ls="--", lw=1.2, label="100% efficiency")
    ax2.set_xlabel("Thread count")
    ax2.set_ylabel("Parallel efficiency (%)")
    ax2.set_title("Strong Scaling — Efficiency")
    ax2.set_ylim(0, 115)
    ax2.legend()
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.suptitle("Figure 3 — Strong scaling (N=10 000, fixed work) (E3)", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "fig3_strong_scaling.png")

# ---------------------------------------------------------------------------
# Figure 4 — E4: Weak scaling — wall time vs threads
# ---------------------------------------------------------------------------
def fig_weak_scaling(csv_dir, out_dir):
    df = load(csv_dir, "weak_scaling.csv")
    if df is None:
        return

    t1 = df.loc[df["threads"] == 1, "wall_s"].values[0]
    df["normalised"] = df["wall_s"] / t1

    fig, ax = plt.subplots(figsize=FIGSIZE_NARROW)
    ax.plot(df["threads"].values, df["normalised"].values, "o-",
            color=COLORS["par"], lw=2, ms=6, label="Measured")
    ax.axhline(1.0, color=COLORS["ideal"], ls="--", lw=1.2,
               label="Ideal (flat)")
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Normalised wall time (T=1 baseline)")
    ax.set_title("Figure 4 — Weak scaling (1 000 scenarios/thread) (E4)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, out_dir, "fig4_weak_scaling.png")

# ---------------------------------------------------------------------------
# Figure 5 — E5: Schedule comparison
# ---------------------------------------------------------------------------
def fig_schedule(csv_dir, out_dir):
    df = load(csv_dir, "schedule_comparison.csv")
    if df is None:
        return

    labels = []
    times  = []
    effs   = []
    for _, row in df.iterrows():
        if row["schedule"] == "static":
            labels.append("static")
        else:
            labels.append(f"{row['schedule']}\nchunk={int(row['chunk'])}"
                          if row["schedule"] != "guided" else "guided")
        times.append(float(row["wall_s"]))
        effs.append(float(row["efficiency_pct"]))

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    colors = [COLORS["static"] if "static" in l else
              (COLORS["guided"] if "guided" in l else COLORS["dynamic"])
              for l in labels]

    ax1.bar(x, times, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title("Wall time by schedule")
    ax1.grid(True, axis="y", ls=":", alpha=0.5)

    ax2.bar(x, effs, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Thread efficiency (%)")
    ax2.set_title("Thread efficiency by schedule")
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle("Figure 5 — Schedule comparison (T=8, N=10 000) (E5)", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "fig5_schedule.png")

# ---------------------------------------------------------------------------
# Figure 6 — E6: Per-thread busy time histogram
# ---------------------------------------------------------------------------
def fig_thread_busy(csv_dir, out_dir):
    df = load(csv_dir, "thread_busy_8t.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_NARROW)
    ax.bar(df["thread_id"].values, df["busy_s"].values, color=COLORS["par"],
           edgecolor="white", linewidth=0.5)
    mean_t = df["busy_s"].values.mean()
    ax.axhline(mean_t, color=COLORS["ideal"], ls="--", lw=1.5,
               label=f"Mean {mean_t:.2f} s")
    ax.set_xlabel("Thread ID")
    ax.set_ylabel("Busy time (s)")
    ax.set_title("Figure 6 — Per-thread busy time (T=8, N=10 000, dynamic/16) (E6)")
    ax.set_ylim(0, df["busy_s"].values.max() * 1.15)
    spread = df["busy_s"].values.max() - df["busy_s"].values.min()
    ax.text(0.98, 0.05, f"Spread: {spread:.3f} s ({spread/mean_t*100:.1f}%)",
            transform=ax.transAxes, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, out_dir, "fig6_thread_busy.png")

# ---------------------------------------------------------------------------
# Figure 7 — E7: Cache / IPC table (static data from perf stat)
# ---------------------------------------------------------------------------
def fig_cache_table(out_dir):
    rows = [
        ("Sequential (T=1)",  "36.96B", "23.44B", "0.63", "4.69%", "37.9M"),
        ("Parallel  (T=8)",   "29.63B", "23.96B", "0.81", "2.78%", "38.1M"),
    ]
    cols = ["Config", "Cycles", "Instructions", "IPC",
            "Cache-miss rate", "Branch misses"]

    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    # Bold headers
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#2c7bb6")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Figure 7 — Hardware counters: sequential vs parallel (N=1 000) (E7)",
                 pad=12, fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, "fig7_cache_table.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default=".", help="Directory with CSV files")
    ap.add_argument("--out-dir", default="figures", help="Output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Reading CSVs from: {args.csv_dir}")
    print(f"Writing figures to: {args.out_dir}")

    fig_convergence(args.csv_dir, args.out_dir)
    fig_stage_timing(args.csv_dir, args.out_dir)
    fig_strong_scaling(args.csv_dir, args.out_dir)
    fig_weak_scaling(args.csv_dir, args.out_dir)
    fig_schedule(args.csv_dir, args.out_dir)
    fig_thread_busy(args.csv_dir, args.out_dir)
    fig_cache_table(args.out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
