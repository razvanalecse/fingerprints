#!/usr/bin/env python3
"""Visualize one registered-approximate sample and both distance notions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fingerprint_reconstruction.data import Nist302RegisteredDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots={
            "sd302a": args.sd302a_root,
            "sd302b": args.sd302b_root,
            "sd302d": args.sd302d_root,
        },
        split=args.split,
    )
    item = dataset[args.index]
    panels = (
        (item["latent_image"][0], "Real latent image", "gray", None),
        (item["mask"][0], "Observed M: quality≥2", "gray", (0, 1)),
        (item["target"][0], "X_pseudo: registered exemplar", "gray", (0, 1)),
        (item["evaluation_roi"][0], "Missing evaluation ROI", "magma", (0, 1)),
        (item["distance_to_observed"][0], "Distance to observed M (px)", "viridis", None),
        (
            item["distance_to_nearest_correspondence_mm"][0],
            "Distance to nearest verified point (mm)",
            "viridis",
            None,
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, (array, title, cmap, limits) in zip(axes.flat, panels):
        values = np.asarray(array)
        artist = axis.imshow(
            values,
            cmap=cmap,
            vmin=limits[0] if limits else None,
            vmax=limits[1] if limits else None,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if limits is None and cmap != "gray":
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{item['population_label']} | {item['registration_status']} | "
        f"n={int(item['num_correspondences'])}, RMSE={float(item['affine_rmse_mm']):.3f} mm",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
