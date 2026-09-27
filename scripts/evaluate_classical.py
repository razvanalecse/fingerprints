#!/usr/bin/env python3
"""Evaluate the deterministic nearest-observed interpolation lower bound."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.models.classical import NearestObservedInpainting
from fingerprint_reconstruction.preprocessing.masks import MaskFamily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model = NearestObservedInpainting()
    started = time.perf_counter()
    records = evaluate_model(model=model, loader=loader, device=torch.device("cpu"))
    elapsed = time.perf_counter() - started
    summary = stratified_summaries(records)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "model": "nearest_observed_interpolation",
        "parameters": 0,
        "split": "validation",
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records),
        "evaluation": summary,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    batch = next(iter(loader))
    reconstruction = model.reconstruct(batch["observed"][:1], batch["mask"][:1])
    mask = batch["mask"][:1]
    shown_observation = torch.where(
        mask.bool(), batch["observed"][:1], torch.full_like(batch["observed"][:1], 0.72)
    )
    images = [
        batch["target"][0, 0].numpy(),
        shown_observation[0, 0].numpy(),
        mask[0, 0].numpy(),
        reconstruction[0, 0].numpy(),
        torch.abs(reconstruction - batch["target"][:1])[0, 0].numpy(),
    ]
    titles = ["Target X", "Observed support", "Mask M", "Nearest interpolation", "Absolute error"]
    figure, axes = plt.subplots(1, 5, figsize=(15, 3))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Baseline 0 — deterministic classical interpolation")
    figure.tight_layout()
    figure.savefig(args.output / "reconstruction-preview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"elapsed_seconds": elapsed, "overall": summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
