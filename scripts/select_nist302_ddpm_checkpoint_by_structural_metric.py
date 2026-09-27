#!/usr/bin/env python3
"""Re-select the NIST302 DDPM checkpoint by a held-out structural composite metric.

The training script selects `best-checkpoint.pt` by validation noise-
prediction MSE -- a proxy for how well the model denoises, not for
reconstruction quality. This script re-evaluates every saved per-epoch
checkpoint (`checkpoint-epoch-*.pt`, only produced by a "full" mode training
run after this session's change to save one per epoch) on a validation
subset, using a composite of the metrics this project actually reports:

    J(epoch) = z(MAE) + z(orientation_error) + z(ridge_frequency_relative_mae)

where z(.) is a z-score computed across this run's own epochs, giving each
term equal weight without an arbitrarily hand-picked lambda (the weighting
choice is fixed by construction -- equal variance-normalized weight -- before
any epoch's identity is looked at, not tuned to produce an outcome). Lower is
better for J.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics import paired_ridge_frequency_error, region_image_metrics
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field, orientation_error
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability

EPOCH_CHECKPOINT_PATTERN = re.compile(r"checkpoint-epoch-(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def build_model(checkpoint: dict, config: dict, device: torch.device) -> ConditionalDDPM:
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
    return model


@torch.no_grad()
def evaluate_checkpoint(
    model: ConditionalDDPM, support_model, loader, device, *,
    ddim_steps: int, samples: int, seed: int, support_mode: str, support_threshold: float,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    mae_values, orientation_values, frequency_values = [], [], []
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        drawn = model.sample_ddim(
            observed, mask, inference_steps=ddim_steps, num_samples=samples,
            eta=0.0, enforce_data_consistency=True, auxiliary_condition=support,
            generator=generator,
        )
        b, k, _, h, w = drawn.shape
        drawn = apply_support_constraint(
            drawn.reshape(b * k, 1, h, w), observed.repeat_interleave(k, dim=0),
            mask.repeat_interleave(k, dim=0), support.repeat_interleave(k, dim=0),
            mode=support_mode, threshold=support_threshold,
        ).reshape(b, k, 1, h, w)
        mean = drawn[0].mean(dim=0)[0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        if int(heldout.sum()) < 5:
            continue
        metrics = region_image_metrics(reference, mean, heldout)
        mae_values.append(metrics["mae"])
        try:
            reference_field = estimate_orientation_field(reference, use_foreground_mask=False)
            mean_field = estimate_orientation_field(mean, use_foreground_mask=False)
            orientation_values.append(orientation_error(reference_field, mean_field, region_mask=heldout))
        except ValueError:
            pass
        frequency = paired_ridge_frequency_error(reference, mean, heldout)
        if np.isfinite(frequency["ridge_frequency_relative_mae"]):
            frequency_values.append(frequency["ridge_frequency_relative_mae"])
    return {
        "mae": float(np.mean(mae_values)),
        "orientation_error": float(np.mean(orientation_values)) if orientation_values else float("nan"),
        "ridge_frequency_relative_mae": float(np.mean(frequency_values)) if frequency_values else float("nan"),
        "num_images": len(mae_values),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
        exemplar_roots=roots, split="validation", output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    support_model = load_support_predictor(Path(config["model"]["support_checkpoint"]), device)

    checkpoint_paths = sorted(
        args.checkpoint_dir.glob("checkpoint-epoch-*.pt"),
        key=lambda path: int(EPOCH_CHECKPOINT_PATTERN.search(path.name).group(1)),
    )
    if not checkpoint_paths:
        raise ValueError(f"no per-epoch checkpoints found in {args.checkpoint_dir}")

    per_epoch: dict[int, dict[str, float]] = {}
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(checkpoint, config, device)
        metrics = evaluate_checkpoint(
            model, support_model, loader, device,
            ddim_steps=args.ddim_steps, samples=args.samples, seed=args.seed,
            support_mode=str(config["evaluation"]["support_mode"]),
            support_threshold=float(config["evaluation"]["support_threshold"]),
        )
        epoch = int(checkpoint["epoch"])
        per_epoch[epoch] = metrics
        print(f"epoch={epoch} mae={metrics['mae']:.4f} orientation={metrics['orientation_error']:.4f} "
              f"frequency={metrics['ridge_frequency_relative_mae']:.4f}", flush=True)

    epochs = sorted(per_epoch)
    mae = np.asarray([per_epoch[e]["mae"] for e in epochs])
    orientation = np.asarray([per_epoch[e]["orientation_error"] for e in epochs])
    frequency = np.asarray([per_epoch[e]["ridge_frequency_relative_mae"] for e in epochs])

    def zscore(values: np.ndarray) -> np.ndarray:
        std = values.std(ddof=1)
        return (values - values.mean()) / std if std > 1e-8 else np.zeros_like(values)

    composite = zscore(mae) + zscore(orientation) + zscore(frequency)
    best_index = int(np.argmin(composite))
    best_epoch_by_composite = epochs[best_index]

    best_checkpoint = torch.load(
        args.checkpoint_dir / "best-checkpoint.pt", map_location=device, weights_only=False
    )
    best_epoch_by_noise_loss = int(best_checkpoint["epoch"])

    report = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "evaluation_images": args.max_images,
        "ddim_steps": args.ddim_steps,
        "samples": args.samples,
        "composite_definition": "z(mae) + z(orientation_error) + z(ridge_frequency_relative_mae), z-scored across this run's own epochs",
        "per_epoch": {
            str(e): {**per_epoch[e], "composite_z_score": float(composite[i])}
            for i, e in enumerate(epochs)
        },
        "best_epoch_by_noise_loss": best_epoch_by_noise_loss,
        "best_epoch_by_structural_composite": best_epoch_by_composite,
        "epochs_agree": best_epoch_by_noise_loss == best_epoch_by_composite,
        "test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: report[key] for key in ("best_epoch_by_noise_loss", "best_epoch_by_structural_composite", "epochs_agree")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
