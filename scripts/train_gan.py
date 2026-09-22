#!/usr/bin/env python3
"""Adversarial (hinge PatchGAN) fine-tuning of a trained deterministic generator."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import evaluate_model, stratified_summaries
from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.discriminator import (
    PatchDiscriminator,
    hinge_discriminator_loss,
    hinge_generator_loss,
)
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import run_reconstruction_epoch, select_device


def save_generator(path: Path, model, config, channels, epoch: int) -> None:
    torch.save(
        {"model_state": model.state_dict(), "config": config, "channels_used": list(channels), "epoch": epoch},
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gan = config["gan"]
    seed = int(config["experiment"]["seed"])
    seed_state = seed_everything(seed)
    device = select_device(args.device)
    families = [MaskFamily(v) for v in config["data"]["mask_families"]]
    common = dict(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=families,
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=seed,
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["data"]["num_workers"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    init = torch.load(gan["init_checkpoint"], map_location="cpu", weights_only=False)
    channels = tuple(init["channels_used"])
    generator = build_reconstruction_model(config["model"], channels=channels)
    generator.load_state_dict(init["model_state"])
    generator.to(device)
    discriminator = PatchDiscriminator().to(device)
    loss_function = MaskedReconstructionLoss(**config["loss"])
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=float(gan["generator_lr"]), betas=(0.5, 0.99),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=float(gan["discriminator_lr"]), betas=(0.5, 0.99), weight_decay=0.0
    )
    lambda_adv = float(gan["lambda_adv"])
    epochs = int(config["training"]["epochs"])
    clip = float(config["training"]["gradient_clip_norm"])
    save_every = int(gan.get("save_every", 5))
    epoch_offset = int(gan.get("mask_epoch_offset", 1000))
    args.output.mkdir(parents=True, exist_ok=True)

    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch_offset + epoch)
        validation_dataset.set_epoch(0)
        generator.train()
        discriminator.train()
        totals = defaultdict(float)
        examples = 0
        for batch in train_loader:
            conditioning = batch["conditioning"].to(device)
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            size = int(target.shape[0])

            raw = generator(conditioning)
            reconstruction = mask * observed + (1.0 - mask) * raw

            d_optimizer.zero_grad(set_to_none=True)
            d_loss = hinge_discriminator_loss(
                discriminator(target, mask), discriminator(reconstruction.detach(), mask)
            )
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad(set_to_none=True)
            reconstruction_loss = loss_function(raw, target, mask)
            adversarial = hinge_generator_loss(discriminator(reconstruction, mask))
            total = reconstruction_loss.total + lambda_adv * adversarial
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite generator loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip, error_if_nonfinite=True)
            g_optimizer.step()

            totals["d_loss"] += float(d_loss.detach()) * size
            totals["g_adv"] += float(adversarial.detach()) * size
            totals["reconstruction"] += float(reconstruction_loss.total.detach()) * size
            examples += size
        train_metrics = {k: v / examples for k, v in totals.items()}
        generator.eval()
        validation_metrics = run_reconstruction_epoch(
            model=generator, loader=validation_loader, loss_function=loss_function, device=device
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"epoch={epoch + 1}/{epochs} d={train_metrics['d_loss']:.3f} g_adv={train_metrics['g_adv']:.3f} "
            f"rec_train={train_metrics['reconstruction']:.4f} rec_val={validation_metrics['loss']:.4f}",
            flush=True,
        )
        if (epoch + 1) % save_every == 0:
            save_generator(args.output / f"g-epoch-{epoch + 1:02d}.pt", generator, config, channels, epoch)
        save_generator(args.output / "best-checkpoint.pt", generator, config, channels, epoch)

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    records = evaluate_model(model=generator, loader=validation_loader, device=device)
    with (args.output / "per-image-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "history": history,
        "evaluation": stratified_summaries(records),
        "seed_state": seed_state.as_dict(),
        "gan": gan,
        "note": "best-checkpoint.pt is the LAST epoch; adversarial training has no validation-loss selection",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
