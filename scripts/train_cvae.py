#!/usr/bin/env python3
"""Train and evaluate the conditional variational fingerprint autoencoder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.losses.cvae import CVAELoss
from fingerprint_reconstruction.models.cvae import ConditionalVAE
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import estimate_foreground_mask
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, objective, device, beta, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            mu, logvar = model.encode_posterior(target, observed, mask)
            z = model.reparameterize(mu, logvar) if training else mu
            reconstruction = model.decode(z, observed, mask)
            result = objective(reconstruction, target, mask, mu, logvar, beta=beta)
            if training:
                if not torch.isfinite(result.total):
                    raise FloatingPointError(f"non-finite CVAE loss at batch {batch_index}")
                result.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        totals["loss"] += float(result.total.detach()) * size
        for name, value in result.components.items():
            totals[name] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty CVAE epoch")
    return {**{key: value / examples for key, value in totals.items()}, "examples": examples}


@torch.no_grad()
def probabilistic_evaluation(model, loader, device, *, samples_k, max_batches=None):
    model.eval()
    records = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        samples = model.sample(observed, mask, num_samples=samples_k)[:, :, 0]
        mean = samples.mean(dim=1)
        standard_deviation = samples.std(dim=1, unbiased=True)
        target_numpy = target[:, 0].cpu().numpy()
        mask_numpy = mask[:, 0].cpu().numpy().astype(bool)
        samples_numpy = samples.cpu().numpy()
        mean_numpy = mean.cpu().numpy()
        std_numpy = standard_deviation.cpu().numpy()
        for index in range(target.shape[0]):
            roi = estimate_foreground_mask(target_numpy[index]) & ~mask_numpy[index]
            if not roi.any():
                continue
            roi_samples = samples_numpy[index][:, roi]
            sample_errors = np.mean(
                np.abs(roi_samples - target_numpy[index][roi][None, :]),
                axis=1,
            )
            pairwise = []
            for left in range(samples_k):
                for right in range(left + 1, samples_k):
                    pairwise.append(
                        float(np.mean(np.abs(roi_samples[left] - roi_samples[right])))
                    )
            absolute_mean_error = np.abs(mean_numpy[index][roi] - target_numpy[index][roi])
            uncertainty = std_numpy[index][roi]
            correlation = spearmanr(uncertainty, absolute_mean_error).statistic
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "mask_family": batch["mask_family"][index],
                    "observed_fraction": float(batch["observed_fraction"][index]),
                    "single_sample_mae": float(sample_errors[0]),
                    "mean_reconstruction_mae": float(absolute_mean_error.mean()),
                    "best_of_k_mae": float(sample_errors.min()),
                    "pairwise_diversity_mae": float(np.mean(pairwise)),
                    "mean_predictive_std": float(uncertainty.mean()),
                    "uncertainty_error_spearman": float(correlation),
                    "missing_roi_pixels": float(roi.sum()),
                }
            )
    if not records:
        raise ValueError("probabilistic evaluation produced no valid records")
    return records


@torch.no_grad()
def save_preview(model, batch, device, output, *, samples_k, title):
    model.eval()
    target = batch["target"][:1].to(device)
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    samples = model.sample(observed, mask, num_samples=samples_k)[0, :, 0]
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=True)
    observed_display = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    panels = [target[0, 0], observed_display[0, 0], mean, std]
    labels = ["Target X", "Observed support", "Predictive mean", "Predictive std"]
    show_count = min(4, samples_k)
    panels.extend(samples[index] for index in range(show_count))
    labels.extend(f"Prior sample {index + 1}" for index in range(show_count))
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel.cpu().numpy(), cmap="magma" if label == "Predictive std" else "gray", vmin=0)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    families = [MaskFamily(value) for value in config["data"]["mask_families"]]
    dataset_kwargs = dict(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=families,
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    train_dataset = SocofingPartialDataset(split="train", **dataset_kwargs)
    validation_dataset = SocofingPartialDataset(split="validation", **dataset_kwargs)
    training = config["training"]
    if args.smoke_test:
        channels, latent_dim, epochs, batch_size = (8, 16, 32), 4, 1, 2
        train_batches, validation_batches, mode = 2, 1, "smoke_test"
    elif args.pilot:
        channels = tuple(training["pilot_channels"])
        latent_dim = int(training["pilot_latent_dim"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches = int(training["pilot_train_batches"])
        validation_batches = int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        channels = tuple(config["model"]["channels"])
        latent_dim = int(config["model"]["latent_dim"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "full"
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model = ConditionalVAE(channels=channels, latent_dim=latent_dim).to(device)
    objective = CVAELoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        beta = float(training["beta_max"]) * min(1.0, (epoch + 1) / int(training["kl_warmup_epochs"]))
        train_metrics = run_epoch(model, train_loader, objective, device, beta, optimizer, train_batches)
        validation_metrics = run_epoch(model, validation_loader, objective, device, beta, None, validation_batches)
        history.append({"epoch": epoch, "beta": beta, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "config": config, "channels_used": list(channels), "latent_dim": latent_dim, "epoch": epoch, "validation_loss": best_loss},
                args.output / "best-checkpoint.pt",
            )
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} beta={beta:.4f} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f} kl={validation_metrics['kl']:.5f}", flush=True)
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    samples_k = int(config["evaluation"]["pilot_samples"] if mode != "full" else config["evaluation"]["samples_k"][1])
    records = probabilistic_evaluation(model, validation_loader, device, samples_k=samples_k, max_batches=validation_batches)
    with (args.output / "per-image-probabilistic-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    save_preview(model, next(iter(validation_loader)), device, args.output / "samples-preview.png", samples_k=samples_k, title=f"CVAE {mode}: K={samples_k}, best epoch {best_epoch + 1}")
    report = {
        "mode": mode,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latent_dim": latent_dim,
        "samples_k": samples_k,
        "history": history,
        "probabilistic_evaluation": summarize_records(records),
        "seed_state": seed_state.as_dict(),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["probabilistic_evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
