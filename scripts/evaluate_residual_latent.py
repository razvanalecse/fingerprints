#!/usr/bin/env python3
"""Evaluate coarse-plus-residual latent diffusion on fixed paired cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.partial_pairs import observed_fraction_report
from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.latent_diffusion import ResidualConditionalLatentDDPM
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from evaluate_latent_reinjection import evaluate_samples
from preview_shape_constrained_latent import load_support_model
from train_latent_diffusion import load_autoencoder, load_structure_predictor
from train_residual_latent_diffusion import load_coarse_predictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-k", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=90)
    parser.add_argument("--pixel-projection-interval", type=int, default=5)
    parser.add_argument("--frequency-guidance-max-sigma", type=float, default=0.0)
    parser.add_argument("--frequency-guidance-power", type=float, default=1.0)
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    coarse = load_coarse_predictor(Path(checkpoint["coarse_checkpoint"]), device)
    structure = load_structure_predictor(
        Path(config["structure_predictor"]["checkpoint"]), device
    )
    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]), schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]), beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]), time_dim=int(checkpoint["time_dim"]),
        data_channels=autoencoder.latent_channels,
        condition_channels=2 * autoencoder.latent_channels,
        normalized_condition=True, multiscale_conditioning=True,
        auxiliary_condition_channels=4,
        middle_attention=bool(config["model"].get("middle_attention", True)),
        attention_heads=int(config["model"].get("attention_heads", 4)),
    )
    denoiser.load_state_dict(checkpoint["denoiser_state"])
    model = ResidualConditionalLatentDDPM(
        autoencoder, coarse, denoiser, scheduler,
        latent_scale=float(checkpoint["latent_scale"]),
        residual_scale=float(checkpoint["residual_scale"]),
        structure_predictor=structure,
    ).to(device).eval()
    return model, config


@torch.no_grad()
def main():
    args = parse_args()
    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest, image_root=args.image_root, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"], base_seed=seed,
        include_foreground=True,
    )
    indices = np.linspace(0, len(dataset) - 1, args.num_images).round().astype(int)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)
    records, preview, sampling_seconds = [], None, 0.0
    started_all = time.perf_counter()
    for case_index, batch in enumerate(loader):
        target = batch["target"][0, 0].to(device)
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        foreground = batch["foreground"][0, 0].to(device).bool()
        support_probability = support_model(torch.cat((observed, mask), dim=1))
        support = support_probability >= 0.5
        seed_everything(seed + 1_000_003 * case_index)
        started = time.perf_counter()
        samples = model.sample_ddim(
            observed, mask, inference_steps=args.steps, num_samples=args.samples_k,
            latent_data_consistency=True,
            pixel_projection_interval=args.pixel_projection_interval,
            frequency_guidance_max_sigma=args.frequency_guidance_max_sigma,
            frequency_guidance_power=args.frequency_guidance_power,
        )[0, :, 0]
        sampling_seconds += time.perf_counter() - started
        samples = torch.where(support[0, 0], samples, torch.ones_like(samples))
        samples = torch.where(mask[0, 0].bool(), observed[0, 0], samples)
        metrics, mean, std = evaluate_samples(samples, target, mask[0, 0], foreground)
        row = {
            "sample_id": batch["sample_id"][0], "mask_family": batch["mask_family"][0],
            "observed_fraction": float(batch["observed_fraction"][0]),
            "selected_dataset_index": int(indices[case_index]),
            **observed_fraction_report(mask[0, 0].cpu().numpy(), foreground.cpu().numpy()),
            **{f"reinjected_{name}": value for name, value in metrics.items()},
        }
        records.append(row)
        if case_index == 0:
            coarse = model.predict_coarse(observed, mask)[0, 0].cpu()
            partial = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))[0, 0].cpu()
            preview = (target.cpu(), partial, coarse, mean, samples[0].cpu(), std)
        if (case_index + 1) % 10 == 0 or case_index == 0:
            print(f"case={case_index + 1}/{args.num_images} sample={row['sample_id']} mean_mae={row['reinjected_mean_mae']:.4f}", flush=True)
    fields = (
        "mean_mae", "mean_psnr", "mean_ssim_map_mean",
        "mean_orientation_error", "mean_ridge_frequency_mae_cpx",
        "mean_ridge_period_mae_pixels", "single_mae", "best_of_k_mae",
        "uncertainty_error_spearman",
    )
    means = {name: float(np.nanmean([row[f"reinjected_{name}"] for row in records])) for name in fields}
    report = {
        "num_pairs": len(records), "steps": args.steps, "samples_k": args.samples_k,
        "pixel_projection_interval": args.pixel_projection_interval,
        "frequency_guidance_max_sigma": args.frequency_guidance_max_sigma,
        "frequency_guidance_power": args.frequency_guidance_power,
        "sampling_seconds": sampling_seconds,
        "elapsed_seconds": time.perf_counter() - started_all,
        "means": means,
        "model": "deterministic coarse predictor plus conditional latent residual diffusion",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired-per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if preview is not None:
        labels = ("Target", "Partial", "Coarse", "Residual mean", "Residual sample", "Predictive std")
        figure, axes = plt.subplots(1, 6, figsize=(18, 3))
        for index, (axis, panel, label) in enumerate(zip(axes, preview, labels)):
            axis.imshow(panel, cmap="magma" if index == 5 else "gray", vmin=0, vmax=None if index == 5 else 1)
            axis.set_title(label); axis.axis("off")
        figure.tight_layout(); figure.savefig(args.output / "preview.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
