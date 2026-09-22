#!/usr/bin/env python3
"""Train conditional diffusion in a frozen fingerprint autoencoder latent space."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.latent_diffusion import (
    ConditionalLatentDDPM,
    FingerprintAutoencoderKL,
    estimate_latent_scale,
)
from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume denoiser weights and epoch counter from a latent-diffusion checkpoint.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize compatible denoiser weights while starting a new experiment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the configured batch size (useful for conservative MPS runs).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the configured learning rate, including after a weights-only resume.",
    )
    parser.add_argument(
        "--run-epochs",
        type=int,
        help="Run at most this many epochs in the current process, while retaining the global epoch counter.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def load_autoencoder(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FingerprintAutoencoderKL(
        channels=tuple(checkpoint["channels_used"]),
        latent_channels=int(checkpoint["latent_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model


def load_structure_predictor(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FingerprintStructurePredictor(
        channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model


def run_epoch(model, loader, device, optimizer=None, max_batches=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total, examples = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = model.training_loss(target, observed, mask)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite latent diffusion loss at {batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        total += float(loss.detach()) * size
        examples += size
    if not examples:
        raise ValueError("empty latent diffusion epoch")
    return {"loss": total / examples, "examples": examples}


@torch.no_grad()
def save_preview(model, batch, device, output, *, samples_k, ddim_steps, title):
    model.eval()
    target = batch["target"][:1].to(device)
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    samples = model.sample_ddim(observed, mask, inference_steps=ddim_steps, num_samples=samples_k)[0, :, 0]
    mean, std = samples.mean(0), samples.std(0, unbiased=True)
    observed_display = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    panels = [target[0, 0], observed_display[0, 0], mean, std]
    labels = ["Target X", "Observed support", "Predictive mean", "Predictive std"]
    panels.extend(samples[index] for index in range(min(4, samples_k)))
    labels.extend(f"Latent DDIM sample {index + 1}" for index in range(min(4, samples_k)))
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel.cpu(), cmap="magma" if label == "Predictive std" else "gray", vmin=0, vmax=None if label == "Predictive std" else 1)
        axis.set_title(label); axis.axis("off")
    figure.suptitle(title); figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    common = dict(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    training, diffusion = config["training"], config["diffusion"]
    if args.smoke_test:
        channels, time_dim, timesteps = (8, 16, 32), 32, 8
        epochs, batch_size, train_batches, validation_batches, mode = 1, 4, 2, 1, "smoke_test"
    elif args.pilot:
        channels = tuple(training["pilot_channels"]); time_dim = int(training["pilot_time_dim"])
        timesteps = int(diffusion["pilot_timesteps"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches, validation_batches = int(training["pilot_train_batches"]), int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        channels = tuple(config["model"]["channels"]); time_dim = int(config["model"]["time_dim"])
        timesteps = int(diffusion["timesteps"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None; mode = "full"
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        batch_size = args.batch_size
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    autoencoder_path = Path(config["autoencoder"]["checkpoint"])
    autoencoder = load_autoencoder(autoencoder_path, device)
    latent_scale = estimate_latent_scale(autoencoder, train_loader, device, max_batches=int(config["autoencoder"]["latent_scale_batches"]))
    scheduler = DDPMScheduler(timesteps=timesteps, schedule=str(diffusion["schedule"]), beta_start=float(diffusion["beta_start"]), beta_end=float(diffusion["beta_end"]))
    denoiser = DiffusionUNet(
        channels=channels,
        time_dim=time_dim,
        data_channels=autoencoder.latent_channels,
        condition_channels=autoencoder.latent_channels,
        normalized_condition=True,
        multiscale_conditioning=bool(config["model"].get("multiscale_conditioning", False)),
        auxiliary_condition_channels=int(config["model"].get("auxiliary_condition_channels", 0)),
        middle_attention=bool(config["model"].get("middle_attention", False)),
        attention_heads=int(config["model"].get("attention_heads", 4)),
    )
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
        latent_scale=latent_scale,
        structure_predictor=structure_predictor,
    ).to(device)
    train_auxiliary_only = bool(training.get("train_auxiliary_only", False))
    if train_auxiliary_only:
        if not model.denoiser.auxiliary_condition_channels:
            raise ValueError("train_auxiliary_only requires auxiliary conditioning")
        model.denoiser.requires_grad_(False)
        model.denoiser.auxiliary_projections.requires_grad_(True)
    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(training["learning_rate"])
    )
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    trainable_parameters = [
        parameter for parameter in model.denoiser.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("denoiser has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=float(training["weight_decay"]))
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale, start_epoch = [], math.inf, -1, 0, 0
    resumed_optimizer = False
    initialization_report = None
    if args.initialize_from is not None:
        if args.resume is not None:
            raise ValueError("resume and initialize-from are mutually exclusive")
        initialization_checkpoint = torch.load(
            args.initialize_from, map_location=device, weights_only=False
        )
        incompatible = model.denoiser.load_state_dict(
            initialization_checkpoint["denoiser_state"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected:
            raise ValueError(f"unexpected initialization keys: {unexpected}")
        allowed_prefixes = (
            "condition_projections.", "middle_attention.", "auxiliary_projections."
        )
        disallowed = [key for key in missing if not key.startswith(allowed_prefixes)]
        if disallowed:
            raise ValueError(f"missing non-v2 initialization keys: {disallowed}")
        initialization_report = {
            "checkpoint": str(args.initialize_from),
            "missing_new_keys": missing,
            "unexpected_keys": unexpected,
        }
        print(f"initialized v2 from {args.initialize_from}; new_keys={len(missing)}", flush=True)
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        expected = (list(channels), time_dim, timesteps)
        actual = (
            list(resume_checkpoint["channels_used"]),
            int(resume_checkpoint["time_dim"]),
            int(resume_checkpoint["timesteps"]),
        )
        if actual != expected:
            raise ValueError(f"resume architecture mismatch: expected {expected}, got {actual}")
        model.denoiser.load_state_dict(resume_checkpoint["denoiser_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_epoch = int(resume_checkpoint.get("best_epoch", resume_checkpoint["epoch"]))
        best_loss = float(
            resume_checkpoint.get("best_validation_loss", resume_checkpoint["validation_loss"])
        )
        history = list(resume_checkpoint.get("history", []))
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            resumed_optimizer = True
        print(
            f"resuming from epoch={start_epoch} best_epoch={best_epoch + 1} "
            f"best_validation={best_loss:.6f} optimizer_state={resumed_optimizer}",
            flush=True,
        )
    if start_epoch >= epochs:
        raise ValueError(f"checkpoint already reached configured epochs={epochs}")
    stop_epoch = epochs
    if args.run_epochs is not None:
        if args.run_epochs <= 0:
            raise ValueError("run epochs must be positive")
        stop_epoch = min(epochs, start_epoch + args.run_epochs)
    started = time.perf_counter()
    for epoch in range(start_epoch, stop_epoch):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer, train_batches, float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, device, None, validation_batches)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"denoiser_state": model.denoiser.state_dict(), "config": config, "autoencoder_checkpoint": str(autoencoder_path), "channels_used": list(channels), "time_dim": time_dim, "timesteps": timesteps, "latent_scale": latent_scale, "epoch": epoch, "validation_loss": best_loss}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        torch.save(
            {
                "denoiser_state": model.denoiser.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "autoencoder_checkpoint": str(autoencoder_path),
                "channels_used": list(channels),
                "time_dim": time_dim,
                "timesteps": timesteps,
                "latent_scale": latent_scale,
                "epoch": epoch,
                "validation_loss": validation_metrics["loss"],
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "history": history,
            },
            args.output / "last-checkpoint.pt",
        )
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f}", flush=True)
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True); break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.denoiser.load_state_dict(checkpoint["denoiser_state"])
    ddim_steps = min(int(config["evaluation"]["preview_ddim_steps"]), timesteps)
    save_preview(model, next(iter(validation_loader)), device, args.output / "samples-preview.png", samples_k=int(config["evaluation"]["preview_samples"]), ddim_steps=ddim_steps, title=f"Conditional latent diffusion {mode}: T={timesteps}, DDIM={ddim_steps}, best epoch {best_epoch + 1}")
    report = {"mode": mode, "device": str(device), "timesteps": timesteps, "preview_ddim_steps": ddim_steps, "latent_scale": latent_scale, "elapsed_seconds": elapsed, "start_epoch": start_epoch, "stop_epoch": stop_epoch, "batch_size": batch_size, "learning_rate": learning_rate, "train_auxiliary_only": train_auxiliary_only, "trainable_denoiser_parameters": sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad), "resumed_optimizer": resumed_optimizer, "initialization": initialization_report, "best_epoch": best_epoch, "best_validation_loss": best_loss, "denoiser_parameters": sum(p.numel() for p in model.denoiser.parameters()), "autoencoder_parameters": sum(p.numel() for p in model.autoencoder.parameters()), "history": history, "seed_state": seed_state.as_dict()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "timesteps", "latent_scale", "elapsed_seconds", "best_epoch", "best_validation_loss", "denoiser_parameters")}, indent=2))


if __name__ == "__main__":
    main()
