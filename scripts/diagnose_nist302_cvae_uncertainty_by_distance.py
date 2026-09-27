#!/usr/bin/env python3
"""Measure CVAE uncertainty-error correlation by anchor-distance stratum.

Mirrors diagnose_nist302_ddpm_uncertainty_by_distance.py exactly (same
distance bands, same pooled/per-image Spearman computation); only model
loading and sampling differ since SpatialConditionalVAE has no
auxiliary_condition input and support gating is applied after sampling.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.cvae import SpatialConditionalVAE
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import (
    constrain_sample_batch,
    load_support_predictor,
    sample_in_chunks,
    support_probability,
)

BANDS = ("nearest_0_2mm", "nearest_2_5mm", "nearest_gt_5mm")


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
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    import yaml

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

    model = SpatialConditionalVAE(
        channels=tuple(checkpoint["channels_used"]),
        latent_channels=int(checkpoint["latent_dim"]),
        upsampling_mode="bilinear",
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_model = load_support_predictor(Path(config["model"]["support_checkpoint"]), device)

    generator = torch.Generator().manual_seed(args.seed)
    pooled_error: dict[str, list[float]] = defaultdict(list)
    pooled_std: dict[str, list[float]] = defaultdict(list)
    band_pixel_counts: dict[str, int] = defaultdict(int)
    per_image_correlations: dict[str, list[float]] = defaultdict(list)

    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        samples = sample_in_chunks(
            model, observed, mask,
            num_samples=args.samples,
            chunk_size=int(config["evaluation"]["sample_chunk_size"]),
            generator=generator,
        )
        samples = constrain_sample_batch(
            samples, observed, mask, support,
            mode="soft", threshold=float(config["evaluation"]["support_threshold"]),
        )
        samples_np = samples[0, :, 0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        predictive_mean = samples_np.mean(axis=0)
        predictive_std = samples_np.std(axis=0)
        absolute_error = np.abs(predictive_mean - reference)

        for band in BANDS:
            band_mask = (batch[f"evaluation_roi_{band}"][0, 0].numpy() > 0.5) & heldout
            count = int(band_mask.sum())
            if count == 0:
                continue
            band_pixel_counts[band] += count
            pooled_error[band].extend(absolute_error[band_mask].tolist())
            pooled_std[band].extend(predictive_std[band_mask].tolist())
            if count >= 20:
                rho = spearmanr(predictive_std[band_mask], absolute_error[band_mask]).statistic
                if np.isfinite(rho):
                    per_image_correlations[band].append(float(rho))

    report = {"samples": args.samples, "num_images": len(dataset)}
    for band in BANDS:
        if not pooled_error[band]:
            report[band] = "no pixels"
            continue
        error = np.asarray(pooled_error[band])
        std = np.asarray(pooled_std[band])
        pooled_rho = spearmanr(std, error).statistic
        report[band] = {
            "pixels": band_pixel_counts[band],
            "images_with_20plus_pixels": len(per_image_correlations[band]),
            "mean_absolute_error": float(error.mean()),
            "mean_predictive_std": float(std.mean()),
            "pooled_uncertainty_error_spearman": float(pooled_rho),
            "mean_per_image_spearman": (
                float(np.mean(per_image_correlations[band])) if per_image_correlations[band] else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
