#!/usr/bin/env python3
"""Difficulty-stratified proper-scoring-rule calibration for the NIST302 DDPM.

Extends diagnose_nist302_ddpm_uncertainty_by_distance.py (Spearman only) with
three proper scoring rules that do not just check monotonic association:
sample-based CRPS, the Gneiting-Raftery interval score (at a nominal 90%
central interval), and sparsification error (AUSE). Stratifies by two
independent difficulty axes, both already present in the dataset:

- **Registration confidence** (evaluation_roi_nearest_{0_2mm,2_5mm,gt_5mm}):
  distance to the nearest examiner-verified correspondence point.
- **Mask geometry** (distance_to_observed, pixel-space distance to the
  nearest actually-observed pixel), split into per-image tertiles: this is
  the axis the registration-confidence bands do not capture -- how far a
  missing pixel is from *any* legible evidence, independent of registration
  quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import (
    interval_score,
    sample_based_crps,
    sparsification_error,
    uncertainty_error_spearman,
)
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability

REGISTRATION_BANDS = ("nearest_0_2mm", "nearest_2_5mm", "nearest_gt_5mm")
GEOMETRY_BANDS = ("geometry_near", "geometry_mid", "geometry_far")
MINIMUM_BAND_PIXELS = 10
NOMINAL_COVERAGE = 0.90
METRIC_NAMES = ("crps", "interval_score", "sparsification_error", "spearman", "pixels")
ALL_BAND_NAMES = REGISTRATION_BANDS + GEOMETRY_BANDS + ("overall",)


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
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-chunk-size", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def _band_metrics(samples_np: np.ndarray, reference: np.ndarray, band_mask: np.ndarray) -> dict[str, float] | None:
    if int(band_mask.sum()) < MINIMUM_BAND_PIXELS:
        return None
    predictive_mean = samples_np.mean(axis=0)
    predictive_std = samples_np.std(axis=0, ddof=1)
    absolute_error = np.abs(predictive_mean - reference)
    return {
        "crps": sample_based_crps(samples_np, reference, band_mask),
        "interval_score": interval_score(samples_np, reference, band_mask, nominal_coverage=NOMINAL_COVERAGE),
        "sparsification_error": sparsification_error(predictive_std[band_mask], absolute_error[band_mask]),
        "spearman": uncertainty_error_spearman(samples_np, reference, band_mask),
        "pixels": float(band_mask.sum()),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
        exemplar_roots=roots, split="validation", output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]), schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]), beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]), time_dim=int(checkpoint["time_dim"]),
        multiscale_conditioning=bool(config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=1, middle_attention=bool(config["model"]["middle_attention"]),
        attention_heads=int(config["model"]["attention_heads"]),
        upsampling_mode=str(config["model"]["upsampling_mode"]),
    )
    model = ConditionalDDPM(denoiser, scheduler).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_model = load_support_predictor(Path(config["model"]["support_checkpoint"]), device)

    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, float]] = []

    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        chunks = []
        for start in range(0, args.samples, args.sample_chunk_size):
            size = min(args.sample_chunk_size, args.samples - start)
            chunks.append(
                model.sample_ddim(
                    observed, mask, inference_steps=args.ddim_steps, num_samples=size,
                    eta=0.0, enforce_data_consistency=True, auxiliary_condition=support,
                    generator=generator,
                )
            )
        samples = torch.cat(chunks, dim=1)
        b, k, _, h, w = samples.shape
        samples = apply_support_constraint(
            samples.reshape(b * k, 1, h, w), observed.repeat_interleave(k, dim=0),
            mask.repeat_interleave(k, dim=0), support.repeat_interleave(k, dim=0),
            mode=str(config["evaluation"]["support_mode"]),
            threshold=float(config["evaluation"]["support_threshold"]),
        ).reshape(b, k, 1, h, w)
        samples_np = samples[0, :, 0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        distance_to_observed = batch["distance_to_observed"][0, 0].numpy()

        # Always emit every band__metric key (NaN when a band has too few
        # pixels in this particular image) -- summarize_records infers which
        # metrics to summarize from records[0] alone, so inconsistent
        # per-image key sets would silently drop any metric absent from the
        # very first record, even where later images do have it.
        record: dict[str, float] = {
            f"{band}__{name}": float("nan") for band in ALL_BAND_NAMES for name in METRIC_NAMES
        }
        for band in REGISTRATION_BANDS:
            band_mask = (batch[f"evaluation_roi_{band}"][0, 0].numpy() > 0.5) & heldout
            metrics = _band_metrics(samples_np, reference, band_mask)
            if metrics is not None:
                for name, value in metrics.items():
                    record[f"{band}__{name}"] = value

        heldout_distances = distance_to_observed[heldout]
        if heldout_distances.size >= 3 * MINIMUM_BAND_PIXELS:
            tertiles = np.quantile(heldout_distances, [1 / 3, 2 / 3])
            geometry_bands = {
                "geometry_near": heldout & (distance_to_observed <= tertiles[0]),
                "geometry_mid": heldout & (distance_to_observed > tertiles[0]) & (distance_to_observed <= tertiles[1]),
                "geometry_far": heldout & (distance_to_observed > tertiles[1]),
            }
            for band_name, band_mask in geometry_bands.items():
                metrics = _band_metrics(samples_np, reference, band_mask)
                if metrics is not None:
                    for name, value in metrics.items():
                        record[f"{band_name}__{name}"] = value

        overall_metrics = _band_metrics(samples_np, reference, heldout)
        if overall_metrics is not None:
            for name, value in overall_metrics.items():
                record[f"overall__{name}"] = value
        if record:
            records.append(record)

    if not records:
        raise ValueError("difficulty-stratified calibration produced no records")
    summary = summarize_records(records)
    report = {
        "num_images": len(records),
        "samples": args.samples,
        "nominal_coverage_for_interval_score": NOMINAL_COVERAGE,
        "minimum_band_pixels": MINIMUM_BAND_PIXELS,
        "registration_bands": REGISTRATION_BANDS,
        "geometry_bands": GEOMETRY_BANDS,
        "geometry_band_definition": "per-image tertiles of pixel-distance to nearest observed pixel, among held-out quality==1 pixels",
        "summary": summary,
        "test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("num_images", "samples")}, indent=2))
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
