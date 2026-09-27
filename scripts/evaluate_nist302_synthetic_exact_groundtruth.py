#!/usr/bin/env python3
"""Track E: exact-ground-truth missing-region recovery on synthetically degraded SD302 exemplars.

Every other NIST302 evaluation in this project is Track B (approximate
registered exemplar target) or Track C (real held-out `quality==1` pixels,
which are themselves faint *existing* traces, not genuinely absent
regions -- see docs/limitations.md). This script is the one place in the
project's NIST302 work where the missing region's true value is known
exactly: `Sd302SyntheticPartialDataset` degrades a single clean SD302
exemplar and then masks it, so the pre-mask (degraded) pixel values are
exact ground truth for the missing region by construction, not an
approximation. Works with any deterministic checkpoint built via
`build_reconstruction_model` (zero-shot SOCOFing, SD302 fine-tuned, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.nist302_synthetic_dataset import Sd302SyntheticPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import paired_ridge_frequency_error, region_image_metrics
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field, orientation_error
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-manifest", type=Path, required=True)
    parser.add_argument("--registered-manifest", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest,
        registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots,
        split=args.split,
        output_shape=(128, 128),
        base_seed=args.seed,
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        output = model.reconstruct(observed, mask)
        for index in range(len(observed)):
            reference = target[index, 0].cpu().numpy()
            estimate = output[index, 0].cpu().numpy()
            missing = (mask[index, 0] < 0.5).cpu().numpy()
            metrics = region_image_metrics(reference, estimate, missing)
            reference_field = estimate_orientation_field(reference, use_foreground_mask=False)
            estimate_field = estimate_orientation_field(estimate, use_foreground_mask=False)
            try:
                orientation = orientation_error(reference_field, estimate_field, region_mask=missing)
            except ValueError:
                orientation = float("nan")
            frequency = paired_ridge_frequency_error(reference, estimate, missing)
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "subject_id": batch["subject_id"][index],
                    "mask_family": batch["mask_family"][index],
                    "observed_fraction": float(batch["observed_fraction"][index]),
                    **{f"missing_{key}": value for key, value in metrics.items()},
                    "missing_orientation_error": orientation,
                    **{f"missing_{key}": value for key, value in frequency.items()},
                    "missing_pixels": float(missing.sum()),
                }
            )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "test_loaded": False,
        "track": "E: synthetic degradation of clean SD302 exemplars, exact missing-region ground truth by construction",
        "images": len(records),
        "elapsed_seconds": elapsed,
        "evaluation": summarize_records(records),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = {
        key: value["mean"]
        for key, value in report["evaluation"]["metrics"].items()
        if key in {"missing_mae", "missing_ssim_map_mean", "missing_orientation_error", "missing_ridge_frequency_relative_mae"}
    }
    print(json.dumps({"images": len(records), **selected}, indent=2))


if __name__ == "__main__":
    main()
