#!/usr/bin/env python3
"""
Track B plots: LM Jacobian parallelization results.
Reads bench/trackB_results.csv, writes fig8_speedup.png, fig9_scaling.png,
fig10_jacfraction.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "trackB_results.csv")

rows = list(csv.DictReader(open(CSV_PATH)))
for r in rows:
    r["threads"] = int(r["threads"])
    r["wall_s"] = float(r["wall_s"])
    r["jac_s"]  = float(r["jac_s"])

models = ["HullWhite analytic", "HullWhite numerical", "G2 analytic"]
seq = {r["model"]: r for r in rows if r["mode"] == "seq"}

def par_rows(m):
    rs = [r for r in rows if r["model"] == m and r["mode"] == "par"]
    rs.sort(key=lambda x: x["threads"])
    return rs


# fig8: Wall-clock speedup bar chart at T=8
fig, ax = plt.subplots(figsize=(8, 5))
speeds = []
labels = []
for m in models:
    rs = [r for r in par_rows(m) if r["threads"] == 8]
    if rs:
        speeds.append(seq[m]["wall_s"] / rs[0]["wall_s"])
        labels.append(m)
bars = ax.bar(labels, speeds, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
for b, s in zip(bars, speeds):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
            f"{s:.2f}x", ha="center", fontsize=10)
ax.set_ylabel("Wall-clock speedup vs sequential (T=1)")
ax.set_title("fig8 — Track B: LM calibration speedup at T=8 (steps=300, g2pts=128)")
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig8_speedup.png"), dpi=150)
plt.close(fig)


# fig9: Strong scaling curve — Jacobian phase speedup vs nThreads
fig, ax = plt.subplots(figsize=(8, 5))
for m in models:
    rs = par_rows(m)
    if not rs:
        continue
    ts = [1] + [r["threads"] for r in rs]
    jac_speedups = [1.0] + [seq[m]["jac_s"] / r["jac_s"] for r in rs]
    ax.plot(ts, jac_speedups, "o-", label=m)
ideal = [1, 2, 4, 8]
ax.plot(ideal, ideal, "k--", label="ideal", alpha=0.5)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks([1, 2, 4, 8])
ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_xlabel("threads")
ax.set_ylabel("Jacobian-phase speedup")
ax.set_title("fig9 — Track B: fdjac2_parallel strong scaling")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig9_scaling.png"), dpi=150)
plt.close(fig)


# fig10: Jacobian fraction of total calibration cost (sequential baseline)
fig, ax = plt.subplots(figsize=(8, 5))
ordered = sorted(models, key=lambda m: seq[m]["wall_s"])
walls = [seq[m]["wall_s"] for m in ordered]
jacs  = [seq[m]["jac_s"] for m in ordered]
non_jacs = [w - j for w, j in zip(walls, jacs)]
ax.bar(ordered, jacs, label="Jacobian phase", color="#C44E52")
ax.bar(ordered, non_jacs, bottom=jacs, label="non-Jacobian (lmdif, lin solve, fcn at base)",
       color="#4C72B0")
for i, (w, j) in enumerate(zip(walls, jacs)):
    ax.text(i, w + max(walls) * 0.01, f"{100*j/w:.0f}% jac",
            ha="center", fontsize=9)
ax.set_ylabel("Wall time (s) — sequential")
ax.set_title("fig10 — Track B: Jacobian share of LM calibration (OMP_NUM_THREADS=1)")
ax.legend(loc="upper left")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig10_jacfraction.png"), dpi=150)
plt.close(fig)

print("wrote fig8_speedup.png, fig9_scaling.png, fig10_jacfraction.png")
