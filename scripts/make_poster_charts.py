#!/usr/bin/env python3
"""Generate the two real data charts used in docs/poster.tex.

Numbers are copied verbatim from docs/nist302_ablation_master_table.md
(sections 1, 19, "Full probabilistic model comparison") and from the
sealed-test outputs/nist302_TEST_FINAL_* metrics.json files -- nothing
here is invented for the poster.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POSTER_BLUE = "#1F3A63"
POSTER_CYAN = "#12706B"
POSTER_RED = "#B0413E"

plt.rcParams.update({
    "font.size": 13,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Chart A: critical finding -- zero-shot vs fine-tuned vs +boundary continuity,
# all on the sealed test split (217 images), same evaluation script.
# ---------------------------------------------------------------------------
conditions = ["Zero-shot", "Fine-tuned", "+ Boundary\ncontinuity"]
colors = [POSTER_BLUE, POSTER_RED, POSTER_CYAN]

mae = [0.1054, 0.1416, 0.1346]
ssim = [0.4884, 0.2802, 0.3036]
orientation = [0.2764, 0.3237, 0.3090]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
specs = [
    ("MAE (lower is better)", mae, axes[0]),
    ("SSIM (higher is better)", ssim, axes[1]),
    ("Orientation error (lower is better)", orientation, axes[2]),
]
for title, values, ax in specs:
    bars = ax.bar(conditions, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

fig.suptitle("Sealed test set (217 images): fine-tuning on the approximate target regresses on real pixels; boundary continuity partially recovers it",
             fontsize=12.5, y=1.05)
fig.tight_layout()
fig.savefig("docs/poster_assets/chart_critical_finding.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart B: complete sealed-test ranking (217 images, evaluated once).
# Every row is outputs/nist302_TEST_FINAL_*/metrics.json, predictive-mean
# MAE on real held-out quality==1 pixels.
# ---------------------------------------------------------------------------
models = [
    ("RePaint (K=5)", 0.1008, "prob"),
    ("Zero-shot transfer", 0.1054, "det"),
    ("Residual DDPM (K=10)", 0.1126, "prob"),
    ("Learned ensemble", 0.1151, "prob"),
    ("+ Boundary continuity", 0.1346, "det"),
    ("CVAE, spatial (K=50)", 0.1378, "prob"),
    ("Fine-tuned (supp.+spec.)", 0.1416, "det"),
    ("DDIM-20 (K=10)", 0.1453, "prob"),
    ("Flow matching, 8 steps", 0.1545, "prob"),
    ("Latent DDPM (K=10)", 0.2111, "prob"),
    ("Latent DDPM, guided (K=10)", 0.2143, "prob"),
]
models.sort(key=lambda row: row[1])
labels = [m[0] for m in models]
values = [m[1] for m in models]
kinds = [m[2] for m in models]
bar_colors = [POSTER_BLUE if k == "det" else POSTER_CYAN for k in kinds]

fig, ax = plt.subplots(figsize=(8.6, 6.0))
y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, color=bar_colors, height=0.66)
for bar, value in zip(bars, values):
    ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}", va="center", fontsize=11)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11.5)
ax.invert_yaxis()
ax.set_xlabel("MAE on real held-out pixels, sealed test split (lower is better)")
ax.set_title("Every model, one sealed test set, evaluated once", fontsize=13.5)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(values) * 1.22)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=POSTER_BLUE, label="deterministic"),
    plt.Rectangle((0, 0), 1, 1, color=POSTER_CYAN, label="probabilistic (predictive mean)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=10.5, frameon=False)
fig.tight_layout()
fig.savefig("docs/poster_assets/chart_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart C: uncertainty quality vs. distance from verified correspondences.
# Source: docs/FINAL_REPORT.md section 6 / reconstruction_vs_hallucination.md.
# The CVAE's correlation sign-flips -- it becomes confident where it is most
# wrong -- while DDIM and RePaint degrade gracefully.
# ---------------------------------------------------------------------------
bands = ["0–2 mm", "2–5 mm", "> 5 mm"]
x = np.arange(len(bands))
series = [
    ("DDIM-20 (pixel DDPM)", [0.527, 0.262, 0.167], POSTER_BLUE, "o"),
    ("RePaint", [0.389, 0.105, 0.017], POSTER_CYAN, "s"),
    ("CVAE, spatial", [0.011, -0.149, -0.297], POSTER_RED, "^"),
]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.2, zorder=1)
for label, values, color, marker in series:
    ax.plot(x, values, marker=marker, markersize=9, linewidth=2.6, color=color, label=label, zorder=3)
    for xi, value in zip(x, values):
        ax.annotate(f"{value:+.3f}", (xi, value), textcoords="offset points",
                    xytext=(0, 11 if value >= 0 else -20), ha="center", fontsize=10.5, color=color)

ax.axhspan(-0.40, 0.0, color=POSTER_RED, alpha=0.06, zorder=0)
ax.text(2.02, -0.22, "overconfident\nwhere most wrong", fontsize=10, color=POSTER_RED,
        ha="right", va="center", style="italic")
ax.set_xticks(x)
ax.set_xticklabels(bands, fontsize=12)
ax.set_xlabel("Distance from nearest examiner-verified correspondence")
ax.set_ylabel(r"Spearman $\rho$ (predictive $\sigma$, true error)")
ax.set_title("Does predictive uncertainty stay honest away from evidence?", fontsize=13)
ax.set_ylim(-0.40, 0.62)
ax.set_xlim(-0.25, 2.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("docs/poster_assets/chart_uncertainty_distance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart D: SOCOFing deterministic baselines (Track A, 900-image validation).
# Source: docs/FINAL_REPORT.md section 3.
# ---------------------------------------------------------------------------
base_models = ["Nearest-observed\ninterpolation", "U-Net", "Gated\nconvolution"]
base_mae = [0.277, 0.176, 0.174]
base_orient = [0.815, 0.241, 0.229]
base_colors = ["#9AA5B1", POSTER_CYAN, POSTER_BLUE]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
for ax, values, title in ((axes[0], base_mae, "MAE on missing pixels"),
                          (axes[1], base_orient, "Orientation error")):
    bars = ax.bar(base_models, values, color=base_colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(title + " (lower is better)", fontsize=11.5)
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
fig.suptitle("Track A: learned models clear the non-learned floor by a wide margin", fontsize=12.5, y=1.04)
fig.tight_layout()
fig.savefig("docs/poster_assets/chart_socofing_baselines.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("wrote docs/poster_assets/chart_critical_finding.png")
print("wrote docs/poster_assets/chart_model_comparison.png")
print("wrote docs/poster_assets/chart_uncertainty_distance.png")
print("wrote docs/poster_assets/chart_socofing_baselines.png")
