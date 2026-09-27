#!/usr/bin/env python3
"""Visualize same-finger SD302 latent/exemplar pairs before registration."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from fingerprint_reconstruction.preprocessing.normalize import load_grayscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    with args.pairs.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_latent = defaultdict(list)
    for row in rows:
        by_latent[row["latent_sample_id"]].append(row)

    selected = []
    used_subjects = set()
    for split in ("train", "validation", "test"):
        for latent_id, candidates in by_latent.items():
            parts = {row["exemplar_dataset_part"] for row in candidates}
            subject = candidates[0]["subject_id"]
            if (
                candidates[0]["split"] == split
                and parts == {"sd302a", "sd302b", "sd302d"}
                and subject not in used_subjects
            ):
                selected.append(candidates)
                used_subjects.add(subject)
                break
    if len(selected) != 3:
        raise RuntimeError("could not find one complete exemplar set in every split")

    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5), constrained_layout=True)
    for row_axes, candidates in zip(axes, selected):
        first = candidates[0]
        latent = load_grayscale(
            args.latent_root / first["latent_relative_path"],
            output_shape=(args.size, args.size),
        )
        row_axes[0].imshow(latent, cmap="gray", vmin=0, vmax=1)
        row_axes[0].set_ylabel(
            f"{first['split']} | subject {first['subject_id']}\n"
            f"FGP {first['fgp']} | {first['assessment']}",
            fontsize=9,
        )
        for column, part in enumerate(("sd302a", "sd302b", "sd302d"), start=1):
            choices = [row for row in candidates if row["exemplar_dataset_part"] == part]
            # A deterministic display choice only; the benchmark table retains every candidate.
            choice = sorted(
                choices,
                key=lambda row: (
                    row["exemplar_capture"] != "roll",
                    -(int(row["exemplar_ppi"]) if row["exemplar_ppi"] else 0),
                    row["exemplar_device"],
                    row["exemplar_relative_path"],
                ),
            )[0]
            exemplar = load_grayscale(
                roots[part] / choice["exemplar_relative_path"],
                output_shape=(args.size, args.size),
            )
            row_axes[column].imshow(exemplar, cmap="gray", vmin=0, vmax=1)
            row_axes[column].text(
                0.02,
                0.98,
                f"{choice['exemplar_capture']}, {choice['exemplar_ppi'] or 'native'} ppi",
                transform=row_axes[column].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])

    for axis, title in zip(
        axes[0],
        (
            "SD302e latent",
            "SD302a challenger roll",
            "SD302b baseline exemplar",
            "SD302d auxiliary plain",
        ),
    ):
        axis.set_title(title, fontsize=11)
    fig.suptitle(
        "Same subject + same FGP, different impressions (not pixel-aligned)", fontsize=14
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
