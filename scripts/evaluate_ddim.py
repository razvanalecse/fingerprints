#!/usr/bin/env python3
"""Evaluate a conditional DDPM using DDIM or RePaint-style sampling."""

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

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import stratified_summaries
from fingerprint_reconstruction.metrics import (
    empirical_interval_coverage,
    pairwise_diversity_mae,
    region_image_metrics,
    uncertainty_error_spearman,
)
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask,
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=("ddim", "repaint"), default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-k", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--resampling-steps", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--selection", choices=("first", "evenly_spaced"), default="first"
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]),
        schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
    )
    model = ConditionalDDPM(
        DiffusionUNet(
            channels=tuple(checkpoint["channels_used"]),
            time_dim=int(checkpoint["time_dim"]),
        ),
        scheduler,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


def safe_orientation_error(reference, estimate, region) -> float:
    try:
        reference_field = estimate_orientation_field(reference)
        estimate_field = estimate_orientation_field(estimate, use_foreground_mask=False)
        return orientation_error(reference_field, estimate_field, region_mask=region)
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    sampler,
    steps,
    samples_k,
    eta,
    resampling_steps,
    max_images,
):
    records, preview = [], None
    sampling_seconds = 0.0
    for batch in loader:
        if max_images is not None and len(records) >= max_images:
            break
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        target = batch["target"].to(device)
        started = time.perf_counter()
        if sampler == "ddim":
            samples = model.sample_ddim(
                observed,
                mask,
                inference_steps=steps,
                num_samples=samples_k,
                eta=eta,
                enforce_data_consistency=True,
            )[0, :, 0]
        else:
            samples = model.sample_repaint(
                observed,
                mask,
                num_samples=samples_k,
                resampling_steps=resampling_steps,
                enforce_data_consistency=True,
            )[0, :, 0]
        elapsed = time.perf_counter() - started
        sampling_seconds += elapsed
        samples_np = samples.cpu().numpy()
        target_np = target[0, 0].cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy().astype(bool)
        roi = estimate_foreground_mask(target_np) & ~mask_np
        if not roi.any():
            continue
        mean_np = samples_np.mean(axis=0)
        std_np = samples_np.std(axis=0, ddof=1)
        sample_mae = np.mean(np.abs(samples_np[:, roi] - target_np[roi]), axis=1)
        best_index = int(np.argmin(sample_mae))
        outputs = {
            "single": samples_np[0],
            "mean": mean_np,
            "best_of_k": samples_np[best_index],
        }
        record = {
            "sample_id": batch["sample_id"][0],
            "mask_family": batch["mask_family"][0],
            "observed_fraction": float(batch["observed_fraction"][0]),
            "sampler": sampler,
            "inference_steps": float(
                steps if sampler == "ddim" else model.scheduler.timesteps
            ),
            "resampling_steps": float(resampling_steps if sampler == "repaint" else 1),
            "samples_k": float(samples_k),
            "sampling_seconds": elapsed,
            "seconds_per_reconstruction": elapsed / samples_k,
            "best_sample_index": float(best_index),
            "pairwise_diversity_mae": pairwise_diversity_mae(samples_np, roi),
            "mean_predictive_std": float(std_np[roi].mean()),
            "uncertainty_error_spearman": uncertainty_error_spearman(
                samples_np, target_np, roi
            ),
            "missing_roi_pixels": float(roi.sum()),
        }
        for nominal in (0.5, 0.8, 0.9):
            coverage, width = empirical_interval_coverage(
                samples_np, target_np, roi, nominal_coverage=nominal
            )
            label = int(nominal * 100)
            record[f"coverage_{label}"] = coverage
            record[f"interval_width_{label}"] = width
        for label, output in outputs.items():
            metrics = region_image_metrics(target_np, output, roi)
            record.update({f"{label}_{name}": value for name, value in metrics.items()})
            record[f"{label}_orientation_error"] = safe_orientation_error(
                target_np, output, roi
            )
        records.append(record)
        if preview is None:
            preview = (target_np, observed[0, 0].cpu().numpy(), mask_np, mean_np, std_np, samples_np)
        print(
            f"image={len(records)} sample={record['sample_id']} "
            f"seconds={elapsed:.3f} single_mae={record['single_mae']:.4f} "
            f"mean_mae={record['mean_mae']:.4f}",
            flush=True,
        )
    if not records or preview is None:
        raise ValueError("evaluation produced no valid records")
    return records, preview, sampling_seconds


def save_preview(
    preview,
    output: Path,
    *,
    sampler: str,
    steps: int,
    samples_k: int,
    resampling_steps: int,
) -> None:
    target, observed, mask, mean, std, samples = preview
    observation = np.where(mask, observed, 0.72)
    panels = [target, observation, mean, std]
    labels = ["Target X", "Observed support", "Predictive mean", "Predictive std"]
    panels.extend(samples[index] for index in range(min(4, samples_k)))
    labels.extend(f"{sampler.upper()} sample {index + 1}" for index in range(min(4, samples_k)))
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel, cmap="magma" if label == "Predictive std" else "gray", vmin=0, vmax=None if label == "Predictive std" else 1)
        axis.set_title(label); axis.axis("off")
    detail = (
        f"{steps} DDIM steps"
        if sampler == "ddim"
        else f"ancestral T=500, U={resampling_steps}"
    )
    figure.suptitle(f"Conditional {sampler.upper()}: {detail}, K={samples_k}")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, config, checkpoint = load_model(args.checkpoint, device)
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    if args.selection == "evenly_spaced":
        if args.max_images is None:
            raise ValueError("evenly_spaced selection requires --max-images")
        if not 1 <= args.max_images <= len(dataset):
            raise ValueError("max-images must be within the validation-set size")
        indices = np.linspace(0, len(dataset) - 1, args.max_images).round().astype(int)
        if np.unique(indices).size != args.max_images:
            raise RuntimeError("evenly spaced selection produced duplicate indices")
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=True)
    records, preview, sampling_seconds = evaluate(
        model,
        loader,
        device,
        sampler=args.sampler,
        steps=args.steps,
        samples_k=args.samples_k,
        eta=args.eta,
        resampling_steps=args.resampling_steps,
        max_images=args.max_images,
    )
    with (args.output / "per-image-probabilistic-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    save_preview(
        preview,
        args.output / "samples-preview.png",
        sampler=args.sampler,
        steps=args.steps,
        samples_k=args.samples_k,
        resampling_steps=args.resampling_steps,
    )
    report = {
        "sampler": args.sampler,
        "device": str(device),
        "training_timesteps": int(checkpoint["timesteps"]),
        "inference_steps": (
            args.steps if args.sampler == "ddim" else int(checkpoint["timesteps"])
        ),
        "eta": args.eta if args.sampler == "ddim" else None,
        "resampling_steps": args.resampling_steps if args.sampler == "repaint" else 1,
        "samples_k": args.samples_k,
        "num_images": len(records),
        "case_selection": args.selection,
        "sampling_seconds": sampling_seconds,
        "mean_seconds_per_image": sampling_seconds / len(records),
        "mean_seconds_per_reconstruction": sampling_seconds / (len(records) * args.samples_k),
        "warning": "best_of_k is oracle-assisted and is not single-sample performance",
        "interval_method": "central empirical sample quantiles; resolution is limited by K",
        "evaluation": stratified_summaries(records),
        "seed_state": seed_state.as_dict(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("num_images", "mean_seconds_per_image", "mean_seconds_per_reconstruction")}, indent=2))


if __name__ == "__main__":
    main()
