#!/usr/bin/env python3
"""Compare raw and predicted-support-constrained latent diffusion samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.metrics import region_image_metrics
from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.latent_diffusion import ConditionalLatentDDPM
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.training.trainer import select_device
from train_latent_diffusion import load_autoencoder, load_structure_predictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-k", type=int, default=4)
    return parser.parse_args()


def load_latent_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    autoencoder = load_autoencoder(Path(checkpoint["autoencoder_checkpoint"]), device)
    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]),
        schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]),
        time_dim=int(checkpoint["time_dim"]),
        data_channels=autoencoder.latent_channels,
        condition_channels=autoencoder.latent_channels,
        normalized_condition=True,
        multiscale_conditioning=bool(config["model"].get("multiscale_conditioning", False)),
        auxiliary_condition_channels=int(config["model"].get("auxiliary_condition_channels", 0)),
        middle_attention=bool(config["model"].get("middle_attention", False)),
        attention_heads=int(config["model"].get("attention_heads", 4)),
    )
    denoiser.load_state_dict(checkpoint["denoiser_state"])
    structure_checkpoint = config.get("structure_predictor", {}).get("checkpoint")
    structure_predictor = (
        load_structure_predictor(Path(structure_checkpoint), device)
        if structure_checkpoint
        else None
    )
    model = ConditionalLatentDDPM(
        autoencoder,
        denoiser,
        scheduler,
        latent_scale=float(checkpoint["latent_scale"]),
        structure_predictor=structure_predictor,
    ).to(device).eval()
    return model, config


def load_support_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FingerprintSupportPredictor(channels=tuple(checkpoint["channels_used"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


@torch.no_grad()
def main():
    args = parse_args()
    device = select_device(args.device)
    latent, config = load_latent_model(args.latent_checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
        include_foreground=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    target = batch["target"].to(device)
    observed = batch["observed"].to(device)
    mask = batch["mask"].to(device)
    true_support = batch["foreground"].to(device).bool()
    support_probability = support_model(torch.cat((observed, mask), dim=1))
    predicted_support = support_probability >= 0.5
    raw = latent.sample_ddim(
        observed, mask, inference_steps=args.steps, num_samples=args.samples_k,
        enforce_data_consistency=True,
    )[0]
    expanded_support = predicted_support[0].expand_as(raw)
    shaped = torch.where(expanded_support, raw, torch.ones_like(raw))
    expanded_mask = mask[0].expand_as(raw).bool()
    expanded_observed = observed[0].expand_as(raw)
    shaped = torch.where(expanded_mask, expanded_observed, shaped)

    target_np = target[0, 0].cpu().numpy()
    true_support_np = true_support[0, 0].cpu().numpy()
    predicted_support_np = predicted_support[0, 0].cpu().numpy()
    missing_roi = true_support_np & ~mask[0, 0].cpu().numpy().astype(bool)
    raw_mean = raw[:, 0].mean(0).cpu().numpy()
    shaped_mean = shaped[:, 0].mean(0).cpu().numpy()
    intersection = np.logical_and(true_support_np, predicted_support_np).sum()
    union = np.logical_or(true_support_np, predicted_support_np).sum()
    report = {
        "sample_id": batch["sample_id"][0],
        "mask_family": batch["mask_family"][0],
        "observed_fraction": float(batch["observed_fraction"][0]),
        "support_iou": float(intersection / max(union, 1)),
        "raw_mean_missing_roi": region_image_metrics(target_np, raw_mean, missing_roi),
        "shaped_mean_missing_roi": region_image_metrics(target_np, shaped_mean, missing_roi),
        "raw_mean_absolute_error_full_frame": float(np.mean(np.abs(raw_mean - target_np))),
        "shaped_mean_absolute_error_full_frame": float(np.mean(np.abs(shaped_mean - target_np))),
        "constraint": "predicted support only; true support used exclusively for evaluation",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    display = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))[0, 0].cpu()
    panels = [
        target[0, 0].cpu(), display, support_probability[0, 0].cpu(),
        predicted_support[0, 0].float().cpu(), raw_mean, shaped_mean,
        raw[0, 0].cpu(), shaped[0, 0].cpu(),
    ]
    labels = [
        "Target (evaluation only)", "Partial input", "Predicted support probability",
        "Predicted support (used)", "Raw predictive mean", "Shape-constrained mean",
        "Raw latent sample", "Shape-constrained sample",
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(label); axis.axis("off")
    figure.suptitle(
        f"No target-support leakage | r={report['observed_fraction']:.2f} | "
        f"predicted-support IoU={report['support_iou']:.3f}"
    )
    figure.tight_layout(); figure.savefig(args.output / "raw-vs-shape-constrained.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
