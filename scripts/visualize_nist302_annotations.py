#!/usr/bin/env python3
"""Visual QA for an SD 302 EFS quality map and minutiae annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

from fingerprint_reconstruction.datasets.nist302_ebts import (
    QUALITY_LABELS,
    decode_transaction,
    efs_to_pixel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--an2k2txt", type=Path, required=True)
    parser.add_argument("--ppi", type=float, required=True)
    parser.add_argument("--record-index", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transaction = decode_transaction(args.transaction, an2k2txt=args.an2k2txt)
    try:
        record = next(item for item in transaction.efs_records if item.record_index == args.record_index)
    except StopIteration as error:
        raise ValueError(f"Type-9 record {args.record_index} not found") from error
    if record.ridge_quality is None or record.ridge_quality.encoding != "UNC":
        raise ValueError("selected record has no uncompressed ridge-quality map")

    image = np.asarray(Image.open(args.image).convert("L"))
    quality = np.asarray(
        [[ord(value) - 48 for value in row] for row in record.ridge_quality.rows],
        dtype=np.uint8,
    )
    scale = args.ppi / 2540.0
    map_height, map_width = quality.shape
    extent = (
        0,
        map_width * record.ridge_quality.grid_size_0_01mm * scale,
        map_height * record.ridge_quality.grid_size_0_01mm * scale,
        0,
    )
    colors = ["#111827", "#6b7280", "#60a5fa", "#22c55e", "#f59e0b", "#ef4444"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("SD302e latent (masked)")

    quality_artist = axes[1].imshow(quality, cmap=cmap, norm=norm, origin="upper")
    axes[1].set_title(f"Official 9.308 quality map\n{quality.shape[1]}×{quality.shape[0]} cells")
    colorbar = fig.colorbar(quality_artist, ax=axes[1], fraction=0.047, pad=0.03, ticks=range(6))
    colorbar.ax.set_yticklabels([str(index) for index in QUALITY_LABELS])

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(quality, cmap=cmap, norm=norm, origin="upper", extent=extent, alpha=0.52)
    axes[2].set_title("Quality aligned in physical units")

    axes[3].imshow(image, cmap="gray", vmin=0, vmax=255)
    for minutia in record.minutiae:
        x, y = efs_to_pixel(minutia.x_0_01mm, minutia.y_0_01mm, ppi=args.ppi)
        angle = np.deg2rad(minutia.direction_degrees)
        radius = 11.0
        axes[3].plot(
            [x - radius * np.cos(angle), x + radius * np.cos(angle)],
            [y - radius * np.sin(angle), y + radius * np.sin(angle)],
            color="#00e5ff",
            linewidth=1.2,
        )
        axes[3].scatter([x], [y], s=12, facecolors="none", edgecolors="#ffea00", linewidths=0.8)
    axes[3].set_title(f"Official minutiae (n={len(record.minutiae)})")

    for axis in (axes[0], axes[2], axes[3]):
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.axis("off")
    axes[1].axis("off")
    fig.suptitle(
        f"{args.transaction.name} | assessment={record.assessment} | "
        f"EFS grid={record.ridge_quality.grid_size_0_01mm / 100:.2f} mm",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
