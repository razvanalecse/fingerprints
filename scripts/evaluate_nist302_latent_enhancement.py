#!/usr/bin/env python3
"""Evaluate SD302 enhancement on officially held-out quality-1 latent pixels."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import (
    paired_ridge_frequency_error,
    region_image_metrics,
)
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import (
    VISIBLE_LATENT_SUPPORT,
    FingerprintSupportPredictor,
    apply_support_constraint,
)
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.training.trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()

    device = select_device(args.device)
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
        output_shape=(128, 128),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_checkpoint = torch.load(
        args.support_checkpoint, map_location=device, weights_only=False
    )
    if support_checkpoint.get("target_semantics") != VISIBLE_LATENT_SUPPORT:
        raise ValueError("support checkpoint does not predict visible latent support")
    support_model = FingerprintSupportPredictor(
        channels=tuple(support_checkpoint["channels_used"])
    ).to(device)
    support_model.load_state_dict(support_checkpoint["model_state"])
    support_model.eval()

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            latent = batch["latent_image"].to(device)
            quality = batch["quality"].to(device)
            raw = model.reconstruct(observed, mask)
            support_probability = torch.maximum(
                support_model(torch.cat((observed, mask), dim=1)), mask
            )
            output = apply_support_constraint(
                raw,
                observed,
                mask,
                support_probability,
                mode=args.support_mode,
                threshold=args.support_threshold,
            )
            for index in range(len(observed)):
                reference = latent[index, 0].cpu().numpy()
                estimate = output[index, 0].cpu().numpy()
                heldout = (quality[index, 0] == 1).cpu().numpy()
                background = (quality[index, 0] == 0).cpu().numpy()
                predicted_support = (
                    support_probability[index, 0] >= args.support_threshold
                ).cpu().numpy()
                true_support = (quality[index, 0] >= 1).cpu().numpy()
                metrics = region_image_metrics(reference, estimate, heldout)
                reference_field = estimate_orientation_field(
                    reference, use_foreground_mask=False
                )
                estimate_field = estimate_orientation_field(
                    estimate, use_foreground_mask=False
                )
                try:
                    orientation = orientation_error(
                        reference_field, estimate_field, region_mask=heldout
                    )
                except ValueError:
                    orientation = float("nan")
                frequency = paired_ridge_frequency_error(reference, estimate, heldout)
                intersection = np.logical_and(predicted_support, true_support).sum()
                union = np.logical_or(predicted_support, true_support).sum()
                record: dict[str, object] = {
                    "sample_id": batch["sample_id"][index],
                    "subject_id": batch["subject_id"][index],
                    **{f"heldout_q1_{key}": value for key, value in metrics.items()},
                    "heldout_q1_orientation_error": orientation,
                    **{f"heldout_q1_{key}": value for key, value in frequency.items()},
                    "background_darkness": float(np.mean(1.0 - estimate[background])),
                    "predicted_support_iou": float(intersection / max(union, 1)),
                    "heldout_q1_pixels": float(heldout.sum()),
                    "background_pixels": float(background.sum()),
                }
                records.append(record)
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "checkpoint": str(args.checkpoint),
        "support_checkpoint": str(args.support_checkpoint),
        "support_mode": args.support_mode,
        "support_threshold": args.support_threshold,
        "split": "validation",
        "test_loaded": False,
        "ground_truth_recovery_region": "official EFS quality==1 pixels hidden from input quality>=2",
        "background_metric": "mean(1-output) on official EFS quality==0; lower is whiter",
        "elapsed_seconds": elapsed,
        "evaluation": summarize_records(records),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = {
        key: value
        for key, value in report["evaluation"]["metrics"].items()
        if key
        in {
            "heldout_q1_mae",
            "heldout_q1_ssim_map_mean",
            "heldout_q1_orientation_error",
            "heldout_q1_ridge_frequency_relative_mae",
            "background_darkness",
            "predicted_support_iou",
        }
    }
    print(json.dumps({"num_images": len(records), "metrics": selected}, indent=2))


if __name__ == "__main__":
    main()
