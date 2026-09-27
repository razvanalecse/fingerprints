#!/usr/bin/env python3
"""Evaluate simple-average vs learned-combiner ensembles on real held-out pixels.

Track C (real, held-out `quality==1` pixels, never touched by any loss or
target construction, including the combiner's own training on Track E).
Reports each individual base model too, computed fresh in this same run
(same K, same seed) for a clean, directly comparable set of numbers.
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

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import paired_ridge_frequency_error, region_image_metrics
from fingerprint_reconstruction.models.ensemble import EnsembleCombiner
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field, orientation_error
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from nist302_ensemble_base_models import MODEL_ORDER, base_model_predictions, load_base_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--combiner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-model-k", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    return parser.parse_args()


def region_metrics_for(reference: np.ndarray, estimate: np.ndarray, region: np.ndarray) -> dict[str, object]:
    metrics = region_image_metrics(reference, estimate, region)
    reference_field = estimate_orientation_field(reference, use_foreground_mask=False)
    estimate_field = estimate_orientation_field(estimate, use_foreground_mask=False)
    try:
        metrics["orientation_error"] = orientation_error(reference_field, estimate_field, region_mask=region)
    except ValueError:
        metrics["orientation_error"] = float("nan")
    frequency = paired_ridge_frequency_error(reference, estimate, region)
    metrics.update(frequency)
    return metrics


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    confidence_report = json.loads(
        Path("outputs/nist302_registration/geometric_confidence_calibration_128.json").read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
        exemplar_roots=roots, split=args.split, output_shape=(128, 128),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    base_models = load_base_models(device)
    combiner_checkpoint = torch.load(args.combiner_checkpoint, map_location=device, weights_only=False)
    combiner = EnsembleCombiner(num_models=len(MODEL_ORDER)).to(device)
    combiner.load_state_dict(combiner_checkpoint["model_state"])
    combiner.eval()

    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        if int(heldout.sum()) < 5:
            continue

        predictions_dict = base_model_predictions(base_models, observed, mask, k=args.base_model_k, generator=generator)
        stacked = torch.stack([predictions_dict[name] for name in MODEL_ORDER], dim=1)
        simple_average = stacked.mean(dim=1)[0, 0].cpu().numpy()
        blended, weight_maps = combiner(observed, mask, stacked)
        learned_ensemble = blended[0, 0].cpu().numpy()
        mean_weights = weight_maps[0, :, heldout].mean(dim=1).cpu().numpy() if heldout.any() else np.full(len(MODEL_ORDER), np.nan)

        record: dict[str, object] = {
            "sample_id": str(batch["sample_id"][0]), "subject_id": str(batch["subject_id"][0]),
        }
        for name in MODEL_ORDER:
            estimate = predictions_dict[name][0, 0].cpu().numpy()
            for key, value in region_metrics_for(reference, estimate, heldout).items():
                record[f"{name}_{key}"] = value
        for key, value in region_metrics_for(reference, simple_average, heldout).items():
            record[f"simple_average_{key}"] = value
        for key, value in region_metrics_for(reference, learned_ensemble, heldout).items():
            record[f"learned_ensemble_{key}"] = value
        for name, weight in zip(MODEL_ORDER, mean_weights):
            record[f"learned_weight_{name}"] = float(weight)
        records.append(record)

    if not records:
        raise ValueError("ensemble evaluation produced no records")
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "images": len(records), "elapsed_seconds": elapsed, "base_model_k": args.base_model_k,
        "combiner_checkpoint": str(args.combiner_checkpoint), "model_order": list(MODEL_ORDER),
        "target": "real latent held-out quality=1 pixels", "test_loaded": False,
        "evaluation": summarize_records(records),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = {
        key: report["evaluation"]["metrics"][key]["mean"]
        for key in (
            "pixel_ddpm_mae", "residual_ddpm_mae", "cvae_mae", "simple_average_mae", "learned_ensemble_mae",
            "simple_average_ssim_map_mean", "learned_ensemble_ssim_map_mean",
        )
        if key in report["evaluation"]["metrics"]
    }
    print(json.dumps({"images": len(records), **selected}, indent=2))


if __name__ == "__main__":
    main()
