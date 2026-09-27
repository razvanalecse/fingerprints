#!/usr/bin/env python3
"""Estimate a resolution-specific Gabor ridge bank from SD302 training targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics.ridge_frequency import estimate_local_ridge_frequency


def summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "q05": float(np.quantile(values, 0.05)),
        "q10": float(np.quantile(values, 0.10)),
        "q35": float(np.quantile(values, 0.35)),
        "median": float(np.quantile(values, 0.50)),
        "q65": float(np.quantile(values, 0.65)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-images", type=int, default=200)
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
        split="train",
        output_shape=(args.image_size, args.image_size),
    )
    count = min(args.num_images, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int)
    frequencies: list[float] = []
    accepted_images = 0
    for sequence, index in enumerate(indices, start=1):
        item = dataset[int(index)]
        result = estimate_local_ridge_frequency(
            item["target"][0].numpy(),
            item["evaluation_roi"][0].numpy().astype(bool),
            minimum_region_fraction=0.45,
        )
        if result.frequencies.size:
            accepted_images += 1
            frequencies.extend(result.frequencies.tolist())
        if sequence % 50 == 0:
            print(f"processed={sequence}/{count}", flush=True)
    values = np.asarray(frequencies, dtype=np.float64)
    if values.size < 20:
        raise RuntimeError("too few valid local ridge-frequency windows")
    statistics = summary(values)
    recommended = [statistics[key] for key in ("q10", "q35", "q65", "q90")]
    report = {
        "split": "train",
        "test_accessed": False,
        "target": "registered_approximate exemplar warped to latent frame",
        "region": "unobserved registered support (evaluation_roi)",
        "image_size": [args.image_size, args.image_size],
        "sampled_images": count,
        "images_with_valid_windows": accepted_images,
        "frequency_unit": "cycles per pixel after resize",
        "frequency": statistics,
        "recommended_gabor_frequencies": recommended,
        "selection_rule": "training-distribution quantiles [0.10, 0.35, 0.65, 0.90]",
        "estimator": {
            "patch_size": 32,
            "stride": 8,
            "minimum_region_fraction": 0.45,
            "search_interval": [0.04, 0.40],
        },
        "warning": (
            "The bank is resolution- and preprocessing-specific. It constrains local "
            "oriented spectral energy and is not an exact differentiable copy of the "
            "reported dominant-frequency metric."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
