#!/usr/bin/env python3
"""Poster figure: a generalist image model asked to complete a fingerprint."""
import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DL = Path("/Users/razvanalecse/Downloads")
CASES = DL / "generalist_baseline_pack" / "cases"
RED = "#B0413E"
COMPLETION_GLOB = os.environ.get("GENERALIST_COMPLETION_GLOB", "generalist_completion_*.png")


def gray(path):
    return np.asarray(Image.open(path).convert("L").resize((128, 128), Image.LANCZOS),
                      dtype=np.float32) / 255.0


seen, uniq = set(), []
for p in sorted(DL.glob(COMPLETION_GLOB)):
    h = hashlib.md5(p.read_bytes()).hexdigest()
    if h not in seen:
        seen.add(h)
        uniq.append(p)

panels = [
    ("Ground truth\n(arch: no core)", gray(CASES / "case01_rect20" / "ground_truth.png"), False),
    ("What the model was given\n(grey = hidden)", gray(CASES / "case01_fragments30" / "masked_input.png"), False),
]
for i, p in enumerate(uniq[:3], start=1):
    panels.append((f"Completion {i}\ninvented whorl core", gray(p), True))

fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.75))
for ax, (title, image, flag) in zip(axes, panels):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(flag)
        spine.set_color(RED)
        spine.set_linewidth(2.0)
    ax.set_title(title, fontsize=10, color=RED if flag else "black",
                 fontweight="bold" if flag else "normal")
fig.suptitle(
    "A general-purpose image model, same input, same prompt, five times",
    fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig("docs/poster_assets/generalist_baseline.png", dpi=200, bbox_inches="tight")
print("wrote docs/poster_assets/generalist_baseline.png")
