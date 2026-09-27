#!/usr/bin/env python3
"""Subject-cross-fitted interval calibration for the SD302 spatial CVAE.

This calibrates marginal symmetric intervals around the predictive mean. It does
not alter samples, fidelity metrics, or the spatial ranking of uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import norm
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import (
    bounded_scaled_interval_coverage,
    scaled_std_interval_coverage,
    standardized_residual_scale,
)
from fingerprint_reconstruction.models.cvae import SpatialConditionalVAE
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import (
    constrain_sample_batch,
    load_support_predictor,
    sample_in_chunks,
    support_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pixels-per-subject", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def subject_folds(subjects: list[str], folds: int, seed: int) -> dict[str, int]:
    if folds < 2 or folds > len(subjects):
        raise ValueError("folds must be between 2 and the number of subjects")
    shuffled = np.asarray(sorted(subjects), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    return {str(subject): index % folds for index, subject in enumerate(shuffled)}


def balanced_calibration_arrays(
    entries: list[dict[str, object]],
    subjects: set[str],
    *,
    pixels_per_subject: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    errors, deviations = [], []
    for subject in sorted(subjects):
        subject_entries = [entry for entry in entries if entry["subject_id"] == subject]
        subject_error = np.concatenate([entry["error"] for entry in subject_entries])
        subject_std = np.concatenate([entry["std"] for entry in subject_entries])
        if subject_error.size > pixels_per_subject:
            indices = rng.choice(subject_error.size, pixels_per_subject, replace=False)
            subject_error, subject_std = subject_error[indices], subject_std[indices]
        errors.append(subject_error)
        deviations.append(subject_std)
    return np.concatenate(errors), np.concatenate(deviations)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.pixels_per_subject <= 0:
        raise ValueError("samples must be >=2 and pixels-per-subject positive")
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "spatial":
        raise ValueError("calibration currently requires a spatial CVAE checkpoint")
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    if confidence_report["source_split"] != "validation":
        raise ValueError("geometric confidence calibration provenance is invalid")
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots=roots,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model = SpatialConditionalVAE(
        channels=tuple(checkpoint["channels_used"]),
        latent_channels=int(checkpoint["latent_dim"]),
        upsampling_mode="bilinear",
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_model = load_support_predictor(
        Path(config["model"]["support_checkpoint"]), device
    )
    generator = torch.Generator().manual_seed(args.seed)
    entries: list[dict[str, object]] = []
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        samples = sample_in_chunks(
            model,
            observed,
            mask,
            num_samples=args.samples,
            chunk_size=int(config["evaluation"]["sample_chunk_size"]),
            generator=generator,
        )
        samples = constrain_sample_batch(
            samples,
            observed,
            mask,
            support,
            mode="soft",
            threshold=float(config["evaluation"]["support_threshold"]),
        )
        samples_np = samples[:, :, 0].cpu().numpy()
        references = batch["latent_image"][:, 0].numpy()
        quality = batch["quality"][:, 0].numpy()
        for index in range(len(observed)):
            region = quality[index] == 1
            # Index the image first so NumPy cannot move the boolean-indexed
            # spatial dimensions in front of the K sample dimension.
            selected = samples_np[index][:, region]
            if selected.shape[0] != args.samples:
                raise RuntimeError("sample axis was corrupted during ROI selection")
            predictive_mean = selected.mean(axis=0)
            reference_values = references[index][region]
            entries.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "subject_id": str(batch["subject_id"][index]),
                    "mean": predictive_mean,
                    "reference": reference_values,
                    "error": np.abs(predictive_mean - reference_values),
                    "std": selected.std(axis=0, ddof=1),
                }
            )

    subjects = sorted({str(entry["subject_id"]) for entry in entries})
    fold_for = subject_folds(subjects, args.folds, args.seed)
    levels = tuple(float(value) for value in config["evaluation"]["nominal_coverages"])
    records: list[dict[str, object]] = []
    fold_scales: dict[str, dict[str, float]] = {}
    for fold in range(args.folds):
        calibration_subjects = {s for s in subjects if fold_for[s] != fold}
        error, std = balanced_calibration_arrays(
            entries,
            calibration_subjects,
            pixels_per_subject=args.pixels_per_subject,
            seed=args.seed + fold,
        )
        scales = {
            str(int(round(100 * level))): standardized_residual_scale(
                error, std, nominal_coverage=level
            )
            for level in levels
        }
        fold_scales[str(fold)] = scales
        for entry in entries:
            subject = str(entry["subject_id"])
            if fold_for[subject] != fold:
                continue
            record: dict[str, object] = {
                "sample_id": entry["sample_id"],
                "subject_id": subject,
                "fold": fold,
            }
            for level in levels:
                label = str(int(round(100 * level)))
                calibrated_coverage, calibrated_width = bounded_scaled_interval_coverage(
                    entry["mean"],
                    entry["std"],
                    entry["reference"],
                    scale=scales[label],
                )
                gaussian_scale = float(norm.ppf((1.0 + level) / 2.0))
                raw_coverage, raw_width = bounded_scaled_interval_coverage(
                    entry["mean"],
                    entry["std"],
                    entry["reference"],
                    scale=gaussian_scale,
                )
                record[f"raw_coverage_{label}"] = raw_coverage
                record[f"raw_width_{label}"] = raw_width
                record[f"calibrated_coverage_{label}"] = calibrated_coverage
                record[f"calibrated_width_{label}"] = calibrated_width
                record[f"calibrated_error_{label}"] = abs(
                    calibrated_coverage - level
                )
            records.append(record)

    # Subject means are the inferential unit; pixels and images are clustered.
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["subject_id"])].append(record)
    subject_records = []
    for subject, values in grouped.items():
        numeric = {
            key: float(np.mean([float(value[key]) for value in values]))
            for key in values[0]
            if key not in {"sample_id", "subject_id", "fold"}
        }
        subject_records.append({"subject_id": subject, **numeric})

    all_error, all_std = balanced_calibration_arrays(
        entries,
        set(subjects),
        pixels_per_subject=args.pixels_per_subject,
        seed=args.seed + 10000,
    )
    final_scales = {
        str(int(round(100 * level))): standardized_residual_scale(
            all_error, all_std, nominal_coverage=level
        )
        for level in levels
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cross_fitted_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "method": "subject-balanced cross-fitted standardized-residual scaling",
        "interpretation": (
            "marginal interval calibration only; spatial uncertainty ranking and "
            "uncertainty-error Spearman correlation are unchanged"
        ),
        "split": "validation",
        "test_loaded": False,
        "decoder_upsampling": "bilinear",
        "support_mode": "soft",
        "samples": args.samples,
        "folds": args.folds,
        "subjects": len(subjects),
        "images": len(entries),
        "pixels_per_subject_cap": args.pixels_per_subject,
        "fold_scales": fold_scales,
        "final_scales_for_future_sealed_test": final_scales,
        "cross_fitted_subject_summary": summarize_records(subject_records),
    }
    (args.output / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
