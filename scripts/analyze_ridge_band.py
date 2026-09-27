#!/usr/bin/env python3
"""Estimate the empirical SOCOFing ridge band and audit pipeline components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.metrics.ridge_frequency import (
    estimate_local_ridge_frequency,
    paired_ridge_frequency_error,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_latent_diffusion import load_autoencoder
from train_residual_latent_diffusion import load_coarse_predictor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--coarse-checkpoint", type=Path)
    return parser.parse_args()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


@torch.no_grad()
def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split=args.split,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=seed,
        include_foreground=True,
    )
    count = min(args.num_images, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)
    autoencoder = load_autoencoder(Path(config["autoencoder"]["checkpoint"]), device)
    coarse_checkpoint = args.coarse_checkpoint or Path(config["coarse_predictor"]["checkpoint"])
    coarse = load_coarse_predictor(coarse_checkpoint, device)

    target_frequencies, target_periods = [], []
    autoencoder_errors, coarse_errors = [], []
    per_image = []
    preview = None
    for case_index, batch in enumerate(loader):
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        foreground = batch["foreground"][0, 0].cpu().numpy().astype(bool)
        missing_roi = foreground & ~mask[0, 0].cpu().numpy().astype(bool)
        reconstruction = autoencoder(target, sample_posterior=False)[0]
        coarse_image = coarse.reconstruct(observed, mask)
        target_np = target[0, 0].cpu().numpy()
        reconstruction_np = reconstruction[0, 0].cpu().numpy()
        coarse_np = coarse_image[0, 0].cpu().numpy()

        target_result = estimate_local_ridge_frequency(target_np, foreground)
        target_frequencies.extend(target_result.frequencies.tolist())
        target_periods.extend(target_result.periods_pixels.tolist())
        ae_error = paired_ridge_frequency_error(target_np, reconstruction_np, foreground)
        coarse_error = paired_ridge_frequency_error(
            target_np,
            coarse_np,
            missing_roi,
            minimum_region_fraction=0.45,
        )
        autoencoder_errors.append(ae_error)
        coarse_errors.append(coarse_error)
        per_image.append({
            "sample_id": batch["sample_id"][0],
            "mask_family": batch["mask_family"][0],
            "observed_fraction": float(batch["observed_fraction"][0]),
            "target_windows": int(target_result.frequencies.size),
            **{f"autoencoder_{key}": value for key, value in ae_error.items()},
            **{f"coarse_{key}": value for key, value in coarse_error.items()},
        })
        if preview is None:
            preview = (target_np, reconstruction_np, coarse_np, foreground, missing_roi)
        if (case_index + 1) % 50 == 0:
            print(f"audited={case_index + 1}/{count}", flush=True)

    def collect(records, key):
        return [record[key] for record in records if np.isfinite(record[key])]

    report = {
        "split": args.split,
        "num_images": count,
        "device": str(device),
        "coarse_checkpoint": str(coarse_checkpoint),
        "frequency_unit": "cycles_per_pixel_after_resize_to_128x128",
        "period_unit": "pixels_after_resize_to_128x128",
        "target_frequency": _summary(target_frequencies),
        "target_period": _summary(target_periods),
        "autoencoder_frequency_mae_cpx": _summary(collect(autoencoder_errors, "ridge_frequency_mae_cpx")),
        "autoencoder_period_mae_pixels": _summary(collect(autoencoder_errors, "ridge_period_mae_pixels")),
        "coarse_missing_frequency_mae_cpx": _summary(collect(coarse_errors, "ridge_frequency_mae_cpx")),
        "coarse_missing_period_mae_pixels": _summary(collect(coarse_errors, "ridge_period_mae_pixels")),
        "estimator": {
            "patch_size": 32,
            "stride": 8,
            "search_frequency_interval": [0.04, 0.40],
            "minimum_peak_prominence": 1.5,
        },
        "interpretation_guard": "The empirical band is resolution-specific and must not be transferred blindly to NIST images.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ridge-band-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "per-image.json").write_text(
        json.dumps(per_image, indent=2) + "\n", encoding="utf-8"
    )

    frequencies = np.asarray(target_frequencies)
    periods = np.asarray(target_periods)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(frequencies, bins=np.arange(0.03, 0.27, 0.01), color="#315a9a")
    axes[0].set(xlabel="Dominant frequency (cycles/pixel)", ylabel="Local windows", title="Empirical SOCOFing ridge band")
    axes[1].hist(periods, bins=np.arange(3.5, 25.5, 1), color="#3b8c6e")
    axes[1].set(xlabel="Dominant period (pixels)", ylabel="Local windows", title="Equivalent ridge spacing")
    labels = ["Autoencoder\n(full ROI)", "Coarse\n(missing ROI)"]
    values = [
        collect(autoencoder_errors, "ridge_frequency_mae_cpx"),
        collect(coarse_errors, "ridge_frequency_mae_cpx"),
    ]
    axes[2].boxplot(values, tick_labels=labels, showfliers=False)
    axes[2].set(ylabel="Frequency MAE (cycles/pixel)", title="Where ridge spacing is lost")
    figure.tight_layout()
    figure.savefig(args.output / "ridge-band-audit.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    if preview is not None:
        target_np, reconstruction_np, coarse_np, foreground, missing_roi = preview
        figure, axes = plt.subplots(1, 5, figsize=(15, 3))
        panels = (target_np, reconstruction_np, coarse_np, foreground, missing_roi)
        labels = ("Target", "Autoencoder", "Coarse", "Foreground", "Missing ROI")
        for axis, panel, label in zip(axes, panels, labels):
            axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
            axis.set_title(label)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(args.output / "component-preview.png", dpi=200, bbox_inches="tight")
        plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
