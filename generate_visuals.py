# -*- coding: utf-8 -*-
"""Generate PNG visualizations and data for markdown/HTML reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Load frontier data
with open(ROOT / "pareto_frontier.json") as f:
    archive = json.load(f)

frontier = archive["frontier"]

# Deduplicate by (name, rank_ic, turnover_stability) — some factors appear twice
seen = set()
unique_frontier = []
for f in frontier:
    key = (f["name"], f["rank_ic"], f["turnover_stability"])
    if key not in seen:
        seen.add(key)
        unique_frontier.append(f)

# Use absolute values for IC and IR (signal direction doesn't matter for Pareto)
for f in unique_frontier:
    f["abs_ic"] = abs(f["rank_ic"])
    f["abs_ir"] = abs(f["ic_ir"])

# Sort for display
unique_frontier.sort(key=lambda f: f["abs_ic"], reverse=True)

# Categorize factors
new_factors = {
    "ts_rank_returns", "decay_returns", "range_vol_adj", "sharpe_ret",
    "adv_term_struct", "close_open_decay", "cs_zscore_vol",
    "combo_range_vwap", "hl_spread_chg", "open_vwap_dev"
}
for f in unique_frontier:
    f["is_new"] = f["name"] in new_factors

N = len(unique_frontier)

# ═══════════════════════════════════════════════
# 1. Pareto Frontier 3-Metric Scatter (3 subplots)
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

pair_axes = [
    (0, "turnover_stability", "abs_ic", "Turnover Stability", "|Rank IC|"),
    (1, "turnover_stability", "abs_ir", "Turnover Stability", "|IC IR|"),
    (2, "abs_ic", "abs_ir", "|Rank IC|", "|IC IR|"),
]

for idx, (ax_idx, x_key, y_key, xlabel, ylabel) in enumerate(pair_axes):
    ax = axes[ax_idx]
    for f in unique_frontier:
        marker = "D" if f["is_new"] else "o"
        color = "#ef4444" if f["is_new"] else "#3b82f6"
        size = 120 if f["is_new"] else 70
        alpha = 0.85 if f["is_new"] else 0.55
        ax.scatter(f[x_key], f[y_key], c=color, s=size, marker=marker,
                   alpha=alpha, edgecolors="white", linewidth=0.6, zorder=3 if f["is_new"] else 2)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_title(f"{ylabel} vs {xlabel}", fontsize=12, fontweight="bold")

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#ef4444", markersize=10, label="New (this experiment)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b82f6", markersize=8, label="Existing frontier"),
]
fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.01, 0.98), fontsize=10, framealpha=0.9)

fig.suptitle("Pareto Frontier — 3-Metric Comparison (Updated with 10 New Factors)",
             fontsize=14, fontweight="bold", y=1.02)
fig.savefig(ASSETS / "pareto_frontier.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"✓ Saved pareto_frontier.png")

# ═══════════════════════════════════════════════
# 2. Top Factors Comparison — horizontal bar chart
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(16, 14), constrained_layout=True)
metric_configs = [
    (0, "abs_ic", "|Rank IC|", "Oranges"),
    (1, "abs_ir", "|IC IR|", "Blues"),
    (2, "turnover_stability", "Turnover Stability", "Greens"),
]

for idx, (ax_idx, metric, label, cmap) in enumerate(metric_configs):
    ax = axes[ax_idx]
    sorted_factors = sorted(unique_frontier, key=lambda f: f[metric], reverse=True)[:20]
    names = [f["name"] for f in sorted_factors][::-1]
    values = [f[metric] for f in sorted_factors][::-1]
    colors = ["#ef4444" if name in new_factors else "#3b82f6" for name in names][::-1]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(label, fontweight="bold")
    ax.set_title(f"Top 20 Factors — Sorted by {label}", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, alpha=0.8)

# Legend
legend_elements2 = [
    plt.Rectangle((0, 0), 1, 1, color="#ef4444", label="New this experiment"),
    plt.Rectangle((0, 0), 1, 1, color="#3b82f6", label="Existing frontier"),
]
fig.legend(handles=legend_elements2, loc="upper right", bbox_to_anchor=(1.0, 0.99), fontsize=10)
fig.suptitle("Factor Performance Ranking (All Metrics)", fontsize=14, fontweight="bold")
fig.savefig(ASSETS / "top_factors.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"✓ Saved top_factors.png")

# ═══════════════════════════════════════════════
# 3. Metric Correlation Heatmap
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))

metrics_matrix = np.array([
    [f["abs_ic"] for f in unique_frontier],
    [f["abs_ir"] for f in unique_frontier],
    [f["turnover_stability"] for f in unique_frontier],
])

corr = np.corrcoef(metrics_matrix)
metric_labels = ["|Rank IC|", "|IC IR|", "Turnover"]

im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if abs(corr[i, j]) > 0.5 else "black")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_yticklabels(metric_labels, fontsize=12)
ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Pearson r", fontsize=11)

fig.savefig(ASSETS / "metric_correlations.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"✓ Saved metric_correlations.png")

# ═══════════════════════════════════════════════
# 4. New Factor Highlights — dedicated comparison
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))

new = [f for f in unique_frontier if f["is_new"]]
new_sorted = sorted(new, key=lambda f: f["abs_ic"], reverse=True)
x = np.arange(len(new_sorted))
width = 0.25

ic_vals = [f["abs_ic"] for f in new_sorted]
ir_vals = [f["abs_ir"] for f in new_sorted]
to_vals = [f["turnover_stability"] for f in new_sorted]

bars1 = ax.bar(x - width, ic_vals, width, label="|Rank IC|", color="#ef4444", edgecolor="white")
bars2 = ax.bar(x, ir_vals, width, label="|IC IR|", color="#f59e0b", edgecolor="white")
bars3 = ax.bar(x + width, to_vals, width, label="Turnover", color="#10b981", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([f["name"] for f in new_sorted], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Metric Value", fontweight="bold")
ax.set_title("10 New Factors — Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, axis="y", alpha=0.25)

fig.savefig(ASSETS / "new_factors.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"✓ Saved new_factors.png")

# ═══════════════════════════════════════════════
# 5. Frontier History — experiment-by-experiment
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)

# Read results.tsv for experiment history
import csv
results = []
results_path = ROOT / "results.tsv"
if results_path.exists():
    with open(results_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(row)

if results:
    exp_nums = list(range(1, len(results) + 1))
    ics = [abs(float(r["rank_ic"])) for r in results]
    irs = [abs(float(r["ic_ir"])) for r in results]
    tos = [float(r["turnover"]) for r in results]

    ax = axes[0]
    ax.plot(exp_nums, ics, "o-", color="#3b82f6", markersize=4, linewidth=1, alpha=0.7, label="|Rank IC|")
    ax.axhline(y=max(ics), color="#ef4444", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("|Rank IC|", fontweight="bold", color="#3b82f6")
    ax.set_title("Experiment History — |Rank IC| Over Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(exp_nums, tos, "o-", color="#10b981", markersize=3, linewidth=1, alpha=0.5, label="Turnover")
    ax2.set_ylabel("Turnover Stability", fontweight="bold", color="#10b981")
    ax2.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.plot(exp_nums, irs, "s-", color="#f59e0b", markersize=4, linewidth=1, alpha=0.7, label="|IC IR|")
    ax.set_xlabel("Experiment Number", fontweight="bold")
    ax.set_ylabel("|IC IR|", fontweight="bold", color="#f59e0b")
    ax.set_title("Experiment History — |IC IR| Over Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

    fig.savefig(ASSETS / "experiment_history.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"✓ Saved experiment_history.png")

plt.close("all")
print("\n✅ All PNGs generated successfully!")
