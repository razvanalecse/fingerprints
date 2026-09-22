#!/usr/bin/env python3
"""Re-evaluate a trained U-Net checkpoint with ROI-aware metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.training.trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = select_device(args.device)
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    records = evaluate_model(model=model, loader=loader, device=device)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = stratified_summaries(records)
    report_path = args.output / "metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["evaluation"] = summary
    report["evaluation_checkpoint"] = str(args.checkpoint)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
