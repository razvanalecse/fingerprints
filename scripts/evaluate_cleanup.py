#!/usr/bin/env python3
"""Evaluate a checkpoint wrapped with the oriented-Gabor ridge clean-up."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.ridge_cleanup import RidgeCleanup
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.training.trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--orientation-sigma", type=float, default=6.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--start-distance", type=float, default=6.0)
    parser.add_argument("--ramp-distance", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = build_reconstruction_model(checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"]))
    base.load_state_dict(checkpoint["model_state"])
    model = RidgeCleanup(base.to(device), iterations=args.iterations,
                         orientation_sigma=args.orientation_sigma, blend=args.blend,
                         start_distance=args.start_distance, ramp_distance=args.ramp_distance)
    config = yaml.safe_load(Path("configs/gated_mps.yaml").read_text())
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest, image_root=args.image_root, split="validation", output_shape=(128, 128),
        families=[MaskFamily(v) for v in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"], base_seed=1729,
    )
    subset = Subset(dataset, range(min(args.limit or len(dataset), len(dataset))))
    records = evaluate_model(model=model, loader=DataLoader(subset, batch_size=16, shuffle=False), device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "metrics.json").write_text(json.dumps(
        {"model": "ridge_cleanup", "iterations": args.iterations, "orientation_sigma": args.orientation_sigma,
         "blend": args.blend, "evaluation": stratified_summaries(records)}, indent=2, sort_keys=True) + "\n")
    print("done")


if __name__ == "__main__":
    main()
