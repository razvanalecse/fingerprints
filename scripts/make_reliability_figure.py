#!/usr/bin/env python3
"""Poster figure: error-coverage curve for selective reconstruction."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, CYAN, RED, GREY = "#1F3A63", "#12706B", "#B0413E", "#9AA5B1"
r = json.load(open("outputs/multisignal_reliability.json"))
COVERAGE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
cov = np.array(COVERAGE) * 100
c = r["curves"]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(cov, c["random"], color=GREY, lw=2.4, marker="o", ms=6, label="random abstention (control)")
ax.plot(cov, c["single_signal"], color=CYAN, lw=2.6, marker="s", ms=7, label="geometric disagreement")
ax.plot(cov, c["multisignal"], color=BLUE, lw=2.8, marker="D", ms=7, label="multi-signal, cross-validated")
ax.plot(cov, c["oracle"], color=RED, lw=2.2, ls="--", marker="^", ms=6, label="oracle (needs the answer)")

ax.annotate("30% lower error\nkeeping 60%", xy=(60, c["multisignal"][COVERAGE.index(0.6)]),
            xytext=(47, 0.56), fontsize=11.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
ax.set_xlabel("Share of the missing region the model still answers for (%)")
ax.set_ylabel("Orientation error where it answers")
ax.set_title("Reconstruct only what can be justified", fontsize=13.5)
ax.invert_xaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10.5, frameon=False, loc="lower left")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("docs/poster_assets/chart_reliability.png", dpi=200, bbox_inches="tight")
print("wrote docs/poster_assets/chart_reliability.png")
