#!/usr/bin/env python3
"""Train or smoke-test the deterministic SOCOFing U-Net baseline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.augmentation import AugmentationConfig
from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import run_reconstruction_epoch, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--run-epochs", type=int)
    return parser.parse_args()


def save_preview(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    output: Path,
    title: str,
) -> None:
    model.eval()
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    target = batch["target"][:1].to(device)
    with torch.no_grad():
        raw = model(torch.cat((observed, mask), dim=1))
        reconstruction = mask * observed + (1.0 - mask) * raw
    observed_display = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    consistency_error = torch.max(torch.abs(reconstruction * mask - observed * mask)).item()
    images = [
        target[0, 0].cpu().numpy(),
        observed_display[0, 0].cpu().numpy(),
        mask[0, 0].cpu().numpy(),
        raw[0, 0].cpu().numpy(),
        reconstruction[0, 0].cpu().numpy(),
        torch.abs(reconstruction - target)[0, 0].cpu().numpy(),
    ]
    titles = ["Target X", "Observed support", "Mask M", "Raw model output", "Data-consistent", "Absolute error"]
    figure, axes = plt.subplots(1, 6, figsize=(18, 3))
    for axis, image, panel_title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(f"{title}\nmax observed-pixel consistency error = {consistency_error:.2e}")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_state = seed_everything(seed)
    device = select_device(args.device)
    families = [MaskFamily(value) for value in config["data"]["mask_families"]]
    fractions = config["data"]["observed_fractions"]
    image_size = tuple(config["data"]["image_size"])
    augmentation = AugmentationConfig.from_mapping(config["data"].get("augmentation"))

    train_dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="train",
        output_shape=image_size,
        families=families,
        observed_fractions=fractions,
        base_seed=seed,
        augmentation=augmentation,
    )
    validation_dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=image_size,
        families=families,
        observed_fractions=fractions,
        base_seed=seed,
    )

    training_config = config["training"]
    if args.smoke_test:
        batch_size = 2
        channels = (8, 16, 32)
        epochs = 1
        train_max_batches = 2
        validation_max_batches = 1
        mode_name = "smoke_test"
    elif args.pilot:
        batch_size = int(training_config["batch_size"])
        channels = tuple(training_config["pilot_channels"])
        epochs = int(training_config["pilot_epochs"])
        train_max_batches = int(training_config["pilot_train_batches"])
        validation_max_batches = int(training_config["pilot_validation_batches"])
        mode_name = "pilot"
    else:
        batch_size = int(training_config["batch_size"])
        channels = tuple(config["model"]["channels"])
        epochs = int(training_config["epochs"])
        train_max_batches = None
        validation_max_batches = None
        mode_name = "full"
    if args.run_epochs is not None:
        if args.run_epochs <= 0:
            raise ValueError("run-epochs must be positive")
        epochs = min(epochs, args.run_epochs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Smoke tests must also work in restricted macOS environments where
        # PyTorch's shared-memory worker manager is unavailable.
        num_workers=0 if args.smoke_test else int(config["data"]["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = build_reconstruction_model(config["model"], channels=channels).to(device)
    if hasattr(model, "structure_predictor"):
        structure_checkpoint = torch.load(
            Path(config["model"]["structure_checkpoint"]),
            map_location=device,
            weights_only=False,
        )
        model.structure_predictor.load_state_dict(structure_checkpoint["model_state"])
        model.structure_predictor.eval().requires_grad_(False)
    if args.initialize_from is not None:
        initialization = torch.load(args.initialize_from, map_location=device, weights_only=False)
        source_channels = tuple(initialization["channels_used"])
        if source_channels != tuple(channels):
            raise ValueError(
                f"initialization channels {source_channels} do not match requested {tuple(channels)}"
            )
        if hasattr(model, "initialize_coarse_from_base_state"):
            model.initialize_coarse_from_base_state(initialization["model_state"])
        else:
            model.load_state_dict(initialization["model_state"])
        print(f"initialized from {args.initialize_from}", flush=True)
    loss_function = MaskedReconstructionLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    history = []
    best_validation_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    early_stopping_patience = int(training_config["early_stopping_patience"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_reduce_factor", 0.5)),
        patience=int(training_config.get("lr_reduce_patience", 5)),
        min_lr=float(training_config.get("minimum_learning_rate", 1e-6)),
    )
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(0)
        train_metrics = run_reconstruction_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            max_batches=train_max_batches,
        )
        validation_metrics = run_reconstruction_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
            max_batches=validation_max_batches,
        )
        scheduler.step(validation_metrics["loss"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        improvement = best_validation_loss - validation_metrics["loss"]
        if improvement > float(training_config.get("minimum_improvement", 1e-5)):
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "channels_used": list(channels),
                    "mode": mode_name,
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                },
                args.output / "best-checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} lr={learning_rate:.2e}",
            flush=True,
        )
        if not args.smoke_test and epochs_without_improvement >= early_stopping_patience:
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "channels_used": list(channels),
        "mode": mode_name,
        "epoch": epochs - 1,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    best_checkpoint = torch.load(
        args.output / "best-checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    preview_batch = next(iter(validation_loader))
    preview_title = (
        f"{config['model'].get('architecture', 'unet')} pipeline smoke test (not a trained result)"
        if args.smoke_test
        else f"{config['model'].get('architecture', 'unet')} {mode_name}: best checkpoint at epoch {best_epoch + 1}"
    )
    save_preview(
        model,
        preview_batch,
        device,
        args.output / "reconstruction-preview.png",
        preview_title,
    )
    evaluation_records = evaluate_model(
        model=model,
        loader=validation_loader,
        device=device,
        max_batches=validation_max_batches,
    )
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation_records[0]))
        writer.writeheader()
        writer.writerows(evaluation_records)
    evaluation_summary = stratified_summaries(evaluation_records)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    epoch_indices = [item["epoch"] + 1 for item in history]
    axis.plot(epoch_indices, [item["train"]["loss"] for item in history], marker="o", label="train")
    axis.plot(
        epoch_indices,
        [item["validation"]["loss"] for item in history],
        marker="o",
        label="validation",
    )
    architecture = str(config["model"].get("architecture", "unet"))
    axis.set(
        xlabel="Epoch",
        ylabel="Composite loss",
        title=f"{architecture} {mode_name} learning curve",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "training-curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    report = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "history": history,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "channels_used": list(channels),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "evaluation": evaluation_summary,
        "mode": mode_name,
        "seed_state": seed_state.as_dict(),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "architecture": architecture,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
