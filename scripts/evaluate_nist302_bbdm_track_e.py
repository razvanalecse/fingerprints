#!/usr/bin/env python3
"""Evaluate Brownian Bridge Diffusion on Track E exact missing-region ground truth."""

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
from fingerprint_reconstruction.models.brownian_bridge import ConditionalBrownianBridge
from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field, orientation_error
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def fill_observed(observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask * observed + (1.0 - mask) * 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-manifest", type=Path, required=True)
    parser.add_argument("--registered-manifest", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split=args.split, output_shape=(128, 128), base_seed=args.seed,
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    denoiser = DiffusionUNet(channels=tuple(checkpoint["channels_used"]), time_dim=int(checkpoint["time_dim"]), multiscale_conditioning=True, middle_attention=True, attention_heads=4, upsampling_mode="bilinear")
    model = ConditionalBrownianBridge(denoiser, s=float(checkpoint["s"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for batch in loader:
        mask = batch["mask"].to(device)
        observed_filled = fill_observed(batch["observed"].to(device), mask)
        target = batch["target"].to(device)
        samples = model.sample(observed_filled, mask, num_samples=5, steps=args.sample_steps, generator=generator)
        mean = samples.mean(dim=1)
        reference = target[0, 0].cpu().numpy()
        estimate = mean[0, 0].cpu().numpy()
        missing = (mask[0, 0] < 0.5).cpu().numpy()
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
                "sample_id": str(batch["sample_id"][0]), "subject_id": str(batch["subject_id"][0]),
                **{f"missing_{k}": v for k, v in metrics.items()}, "missing_orientation_error": orientation,
                **{f"missing_{k}": v for k, v in frequency.items()},
            }
        )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"images": len(records), "elapsed_seconds": elapsed, "sample_steps": args.sample_steps, "checkpoint": str(args.checkpoint), "test_loaded": False, "track": "E", "evaluation": summarize_records(records)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = {k: report["evaluation"]["metrics"][k]["mean"] for k in ("missing_mae", "missing_ssim_map_mean", "missing_orientation_error", "missing_ridge_frequency_relative_mae") if k in report["evaluation"]["metrics"]}
    print(json.dumps({"images": len(records), **selected}, indent=2))


if __name__ == "__main__":
    main()
