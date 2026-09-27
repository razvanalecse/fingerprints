#!/usr/bin/env python3
"""Evaluate a classical baseline on frozen registered-approximate SD302 validation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import (
    evaluate_registered_model,
    summarize_records,
)
from fingerprint_reconstruction.models.classical import NearestObservedInpainting
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.training.trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-ridge-frequency", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
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
        output_shape=(args.image_size, args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = select_device(args.device)
    if args.checkpoint is None:
        model = NearestObservedInpainting().to(device)
        model_name = "nearest_observed_interpolation"
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_reconstruction_model(
            checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model_name = str(checkpoint["config"]["model"].get("architecture", "unet"))
    started = time.perf_counter()
    records = evaluate_registered_model(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        include_ridge_frequency=not args.skip_ridge_frequency,
    )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "model": model_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": str(device),
        "image_size": args.image_size,
        "split": args.split,
        "population": "correspondence-qualified latent subset",
        "target": "registered_approximate",
        "pixel_aligned_ground_truth": False,
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records),
        "evaluation": summarize_records(records),
        "metric_warning": (
            "Pixel metrics include residual registration and cross-impression appearance error; "
            "they are secondary to structural metrics."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = {
        key: value
        for key, value in report["evaluation"]["metrics"].items()
        if key
        in {
            "evaluation_mae",
            "evaluation_ssim_map_mean",
            "evaluation_orientation_error",
            "evaluation_ridge_frequency_relative_mae",
            "nearest_0_2mm_orientation_error",
            "nearest_2_5mm_orientation_error",
            "nearest_gt_5mm_orientation_error",
        }
    }
    print(json.dumps({"num_images": len(records), "elapsed_seconds": elapsed, "metrics": selected}, indent=2))


if __name__ == "__main__":
    main()
