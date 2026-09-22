#!/usr/bin/env python3
"""Evaluate the parameter-free Gabor ridge-continuation baseline on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.models.gabor_extension import GaborRidgeExtension
from fingerprint_reconstruction.preprocessing.masks import MaskFamily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gated_mps.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N images only")
    parser.add_argument("--iterations", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(v) for v in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    subset = Subset(dataset, range(min(args.limit or len(dataset), len(dataset))))
    loader = DataLoader(subset, batch_size=16, shuffle=False)
    model = GaborRidgeExtension(iterations=args.iterations)
    started = time.perf_counter()
    records = evaluate_model(model=model, loader=loader, device=torch.device("cpu"))
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "model": "gabor_ridge_extension",
        "parameters": 0,
        "iterations": args.iterations,
        "elapsed_seconds": elapsed,
        "evaluation": stratified_summaries(records),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    indices = [0, 7, 21, 40, 63, 88]
    figure, axes = plt.subplots(len(indices), 4, figsize=(9, 2.3 * len(indices)))
    for row, index in enumerate(indices):
        if index >= len(subset):
            continue
        item = dataset[index]
        observed, mask, target = item["observed"][None], item["mask"][None], item["target"]
        recon = model.reconstruct(observed, mask)[0]
        shown = torch.where(mask[0].bool(), observed[0], torch.full_like(observed[0], 0.72))
        for axis, image, title in zip(
            axes[row],
            (target, shown, recon, (recon - target).abs()),
            ("target", f"observed ({item['mask_family']}, {float(item['observed_fraction']):.1f})", "gabor", "abs error"),
        ):
            axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "reconstruction-preview.png", dpi=150, facecolor="white")
    print(f"done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
