#!/usr/bin/env python3
"""Show how SD302e PNG images correspond to SD302h EFS annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint_reconstruction.datasets.nist302_ebts import decode_transaction, efs_to_pixel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--sd302e-root", type=Path, required=True)
    parser.add_argument("--sd302h-lffs-root", type=Path, required=True)
    parser.add_argument("--an2k2txt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _manifest_by_lffs(path: Path) -> dict[str, dict[str, str]]:
    mapping = {}
    for row in _read_rows(path):
        for filename in json.loads(row["lffs_filenames"]):
            mapping[filename] = row
    return mapping


def _select_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # Fixed ranks make the illustration deterministic while avoiding extreme
    # outliers that are visually unrepresentative of each examiner category.
    selected = []
    for assessment in ("VALUE", "LIMITED", "NOVALUE"):
        group = sorted(
            (row for row in rows if row["assessment"] == assessment),
            key=lambda row: (int(row["minutiae_count"]), row["relative_path"]),
        )
        selected.append(group[len(group) // 2])
    return selected


def main() -> None:
    args = parse_args()
    annotations = _read_rows(args.annotation_csv)
    manifest = _manifest_by_lffs(args.manifest)
    examples = _select_examples(annotations)

    colors = ["#111827", "#6b7280", "#60a5fa", "#22c55e", "#f59e0b", "#ef4444"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    fig, axes = plt.subplots(3, 4, figsize=(15, 13), constrained_layout=True)

    for row_index, annotation in enumerate(examples):
        lffs_name = Path(annotation["relative_path"]).name
        image_record = manifest[lffs_name]
        image_path = args.sd302e_root / image_record["original_masked_path"]
        lffs_path = args.sd302h_lffs_root / annotation["relative_path"]
        ppi = float(image_record["native_ppi"])
        image = np.asarray(Image.open(image_path).convert("L"))
        transaction = decode_transaction(lffs_path, an2k2txt=args.an2k2txt)
        record = transaction.efs_records[0]
        if record.ridge_quality is None:
            raise ValueError(f"missing quality map: {lffs_path}")
        quality = np.asarray(
            [[ord(value) - 48 for value in line] for line in record.ridge_quality.rows],
            dtype=np.uint8,
        )
        scale = ppi / 2540.0
        height, width = quality.shape
        extent = (
            0,
            width * record.ridge_quality.grid_size_0_01mm * scale,
            height * record.ridge_quality.grid_size_0_01mm * scale,
            0,
        )

        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"{record.assessment} — SD302e latent PNG")
        axes[row_index, 1].imshow(quality, cmap=cmap, norm=norm, origin="upper")
        axes[row_index, 1].set_title("SD302h: official 9.308 map")
        axes[row_index, 2].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 2].imshow(
            quality, cmap=cmap, norm=norm, origin="upper", extent=extent, alpha=0.52
        )
        axes[row_index, 2].set_title("PNG + physically aligned map")
        axes[row_index, 3].imshow(image, cmap="gray", vmin=0, vmax=255)
        for minutia in record.minutiae:
            x, y = efs_to_pixel(minutia.x_0_01mm, minutia.y_0_01mm, ppi=ppi)
            axes[row_index, 3].scatter(
                x, y, s=18, facecolors="none", edgecolors="#00d5ff", linewidths=1.0
            )
        axes[row_index, 3].set_title(f"SD302h: minutiae (n={len(record.minutiae)})")

        for column in (0, 2, 3):
            axes[row_index, column].set_xlim(0, image.shape[1])
            axes[row_index, column].set_ylim(image.shape[0], 0)
        for column in range(4):
            axes[row_index, column].axis("off")

    fig.suptitle(
        "SD302e is the image; SD302h supplies expert annotations for that same latent",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
