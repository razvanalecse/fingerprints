#!/usr/bin/env python3
"""Visual QA for examiner-correspondence SD302 registration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fingerprint_reconstruction.datasets.nist302_ebts import (
    decode_transaction,
    efs_to_pixel,
    match_correspondences,
)
from fingerprint_reconstruction.evaluation.registration import (
    fit_affine_transform,
    fit_similarity_transform,
    warp_image_output_to_input,
)
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=Path, required=True)
    parser.add_argument("--an2k2txt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exemplars", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _contrast(image: np.ndarray) -> np.ndarray:
    low, high = np.quantile(image, (0.01, 0.99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.float32))
    return np.asarray(pil.resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def main() -> None:
    args = parse_args()
    transaction = decode_transaction(args.comp, an2k2txt=args.an2k2txt)
    records = sorted(transaction.efs_records, key=lambda record: int(record.idc))
    if len(records) != 2:
        raise ValueError("COMP must contain exactly two EFS records")
    sources = {source.source_index: source for source in transaction.comp_sources}
    lffs_filename = sources[1].filename
    irr_stem = Path(sources[2].filename).stem

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    latent_candidates = [
        row for row in manifest_rows if lffs_filename in json.loads(row["lffs_filenames"])
    ]
    if len(latent_candidates) != 1:
        raise ValueError(f"expected one latent PNG for {lffs_filename}")
    latent_row = latent_candidates[0]

    with args.exemplars.open(newline="", encoding="utf-8") as stream:
        exemplar_candidates = [
            row for row in csv.DictReader(stream) if Path(row["relative_path"]).stem == irr_stem
        ]
    if len(exemplar_candidates) != 1:
        raise ValueError(f"expected one exemplar PNG for {irr_stem}")
    exemplar_row = exemplar_candidates[0]
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    if not exemplar_row["ppi"]:
        raise ValueError("registration QA requires declared exemplar PPI")

    latent = load_grayscale(args.latent_root / latent_row["original_masked_path"])
    exemplar = load_grayscale(
        roots[exemplar_row["dataset_part"]] / exemplar_row["relative_path"]
    )
    matches = match_correspondences(
        transaction, records[0].record_index, records[1].record_index
    )
    latent_efs_points = np.asarray(
        [[match.first.x_0_01mm, match.first.y_0_01mm] for match in matches],
        dtype=np.float64,
    )
    exemplar_efs_points = np.asarray(
        [[match.second.x_0_01mm, match.second.y_0_01mm] for match in matches],
        dtype=np.float64,
    )
    latent_points = np.asarray(
        [
            efs_to_pixel(
                match.first.x_0_01mm,
                match.first.y_0_01mm,
                ppi=float(latent_row["native_ppi"]),
                horizontal_offset_0_01mm=records[0].roi.horizontal_offset_0_01mm,
                vertical_offset_0_01mm=records[0].roi.vertical_offset_0_01mm,
            )
            for match in matches
        ]
    )
    exemplar_points = np.asarray(
        [
            efs_to_pixel(
                match.second.x_0_01mm,
                match.second.y_0_01mm,
                ppi=float(exemplar_row["ppi"]),
                horizontal_offset_0_01mm=records[1].roi.horizontal_offset_0_01mm,
                vertical_offset_0_01mm=records[1].roi.vertical_offset_0_01mm,
            )
            for match in matches
        ]
    )
    similarity_physical = fit_similarity_transform(latent_efs_points, exemplar_efs_points)
    affine_physical = fit_affine_transform(latent_efs_points, exemplar_efs_points)
    similarity = fit_similarity_transform(latent_points, exemplar_points)
    affine = fit_affine_transform(latent_points, exemplar_points)
    similarity_warp = warp_image_output_to_input(
        exemplar, similarity.matrix, output_shape=latent.shape
    )
    affine_warp = warp_image_output_to_input(exemplar, affine.matrix, output_shape=latent.shape)

    display_shape = (512, 512)
    panels = [
        _resize(_contrast(latent), display_shape),
        _resize(_contrast(exemplar), display_shape),
        _resize(_contrast(similarity_warp), display_shape),
        _resize(_contrast(affine_warp), display_shape),
    ]
    latent_display = panels[0]
    affine_display = panels[3]
    overlay = np.stack((1 - latent_display, 1 - affine_display, 1 - affine_display), axis=-1)

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    titles = (
        "SD302e latent",
        "SD302g exemplar",
        f"Similarity registered\nRMSE={similarity_physical.rmse_mm:.3f} mm",
        f"Affine registered\nRMSE={affine_physical.rmse_mm:.3f} mm",
        "Affine overlay\nred=latent, cyan=exemplar",
    )
    for axis, panel, title in zip(axes[:4], panels, titles[:4]):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
    axes[4].imshow(np.clip(overlay, 0, 1))
    axes[4].set_title(titles[4])
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{lffs_filename} | official correspondences={len(matches)} | registered_approximate",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
