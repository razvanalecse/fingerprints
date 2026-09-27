#!/usr/bin/env python3
"""Render a deterministic sanity-check batch from the SD302e+SD302h loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from fingerprint_reconstruction.data.nist302_torch_dataset import Nist302AnnotatedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--finger-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def _first_by_assessment(dataset: Nist302AnnotatedDataset, assessment: str) -> int:
    return next(
        index for index, row in enumerate(dataset.rows) if row.assessment == assessment
    )


def main() -> None:
    args = parse_args()
    common = dict(
        manifest_path=args.manifest,
        annotation_csv=args.annotations,
        image_root=args.image_root,
        annotation_root=args.annotation_root,
        output_shape=(args.size, args.size),
        assessments=("VALUE", "LIMITED"),
        source_codes=(1,),
        finger_positions_csv=args.finger_positions,
    )
    datasets = {
        split: Nist302AnnotatedDataset(split=split, **common)
        for split in ("train", "validation", "test")
    }
    selections = [
        (split, assessment, datasets[split][_first_by_assessment(datasets[split], assessment)])
        for split in ("train", "validation", "test")
        for assessment in ("VALUE", "LIMITED")
    ]

    quality_cmap = ListedColormap(
        ["#111111", "#8c8c8c", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
    )
    quality_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), quality_cmap.N)
    fig, axes = plt.subplots(len(selections), 4, figsize=(12, 17), constrained_layout=True)
    for row_axes, (split, assessment, item) in zip(axes, selections):
        image = item["image"][0].numpy()
        quality = item["quality"][0].numpy()
        ridge = item["reliable_ridge"][0].numpy().astype(bool)
        minutiae = item["reliable_minutiae"][0].numpy().astype(bool)

        row_axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        row_axes[1].imshow(quality, cmap=quality_cmap, norm=quality_norm)
        row_axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
        row_axes[2].imshow(np.ma.masked_where(~ridge, ridge), cmap="Greens", alpha=0.45)
        row_axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
        row_axes[3].imshow(
            np.ma.masked_where(~minutiae, minutiae), cmap="Blues", alpha=0.55
        )
        subject = item["subject_id"]
        row_axes[0].set_ylabel(
            f"{split} | {assessment}\nsubject {subject}\nminutiae {int(item['minutiae_count'])}",
            fontsize=9,
        )
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])

    titles = (
        "SD302e latent image",
        "SD302h quality codes (0–5)",
        "Reliable ridge area (q≥2)",
        "Reliable minutiae area (q≥3)",
    )
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=11)
    fig.suptitle(
        "SD302e pixels joined to SD302h EFS annotations — subject-disjoint splits",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
