#!/usr/bin/env python3
"""Does DDPM uncertainty-error correlation hold uniformly, or only near anchors?

Stratifies the held-out quality==1 pixels by distance to the nearest verified
correspondence point (fields already provided by Nist302RegisteredDataset) and
recomputes, per band, the same diagnostics used for the full-validation report:
raw coverage and the uncertainty-error Spearman correlation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability

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
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-chunk-size", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


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
    pooled_error: dict[str, list[float]] = defaultdict(list)
    pooled_std: dict[str, list[float]] = defaultdict(list)
    band_pixel_counts: dict[str, int] = defaultdict(int)
    per_image_correlations: dict[str, list[float]] = defaultdict(list)

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
        samples_np = samples[:, :, 0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        predictive_mean = samples_np[0].mean(axis=0)
        predictive_std = samples_np[0].std(axis=0, ddof=1)
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

    report = {"samples": args.samples, "ddim_steps": args.ddim_steps, "num_images": len(dataset)}
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
