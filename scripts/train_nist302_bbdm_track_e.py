#!/usr/bin/env python3
"""Train conditional Brownian Bridge Diffusion on Track E (exact ground truth) first.

Per the plan proposing this model: BBDM should be validated where ground
truth is exact before being trained on SD302's approximately-registered
latent/exemplar pairs, since its trajectory starts *at* the conditioning
input itself (unlike DDPM, which only conditions on it) and is therefore
more sensitive to that input being genuinely correct. Trains on
Sd302SyntheticPartialDataset (Track E: synthetic degradation of clean SD302
exemplars, exact missing-region ground truth by construction).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.nist302_synthetic_dataset import Sd302SyntheticPartialDataset
from fingerprint_reconstruction.models.brownian_bridge import ConditionalBrownianBridge
from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-manifest", type=Path, required=True)
    parser.add_argument("--registered-manifest", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--max-train-images", type=int, default=1200)
    parser.add_argument("--max-validation-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def fill_observed(observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask * observed + (1.0 - mask) * 0.5


def run_epoch(model, loader, device, *, optimizer, generator, max_batches=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total, examples = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        observed_filled = fill_observed(batch["observed"].to(device), mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = model.training_loss(target, observed_filled, mask, generator=generator)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite BBDM loss at batch {batch_index}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        total += float(loss.detach()) * size
        examples += size
    if not examples:
        raise ValueError("empty BBDM epoch")
    return {"loss": total / examples, "examples": float(examples)}


@torch.no_grad()
def save_preview(model, batch, device, output, *, samples_k, steps, title):
    model.eval()
    target = batch["target"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    observed_filled = fill_observed(batch["observed"][:1].to(device), mask)
    samples = model.sample(observed_filled, mask, num_samples=samples_k, steps=steps)[0]
    mean, std = samples.mean(0), samples.std(0)
    panels = [target[0, 0], observed_filled[0, 0], mean[0], samples[0, 0], std[0]]
    labels = ["Target (exact)", "Y filled", f"BBDM mean K={samples_k}", "Sample 1", "Predictive std"]
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(panel.cpu().numpy(), cmap="magma" if label == "Predictive std" else "gray")
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_state = seed_everything(args.seed)
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    train_dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="train", output_shape=(128, 128), base_seed=args.seed,
        max_images=(4 if args.smoke_test else args.max_train_images),
    )
    validation_dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="validation", output_shape=(128, 128), base_seed=args.seed,
        max_images=(2 if args.smoke_test else args.max_validation_images),
    )
    batch_size = 2 if args.smoke_test else args.batch_size
    epochs = 1 if args.smoke_test else args.epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    channels, time_dim = ((8, 16, 32), 32) if args.smoke_test else ((32, 64, 128, 256), 128)
    denoiser = DiffusionUNet(channels=channels, time_dim=time_dim, multiscale_conditioning=True, middle_attention=True, attention_heads=4, upsampling_mode="bilinear")
    model = ConditionalBrownianBridge(denoiser, s=args.s).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, generator=generator, max_batches=(2 if args.smoke_test else None))
        validation_metrics = run_epoch(model, validation_loader, device, optimizer=None, generator=generator, max_batches=(1 if args.smoke_test else None))
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch + 1, 0
            torch.save({"model_state": model.state_dict(), "channels_used": list(channels), "time_dim": time_dim, "s": args.s, "epoch": best_epoch, "validation_loss": best_loss, "test_loaded": False, "training_target": "Track E exact synthetic ground truth"}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f}", flush=True)
        if not args.smoke_test and stale >= 4:
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_preview(model, next(iter(validation_loader)), device, args.output / "samples-preview.png", samples_k=5, steps=args.sample_steps, title=f"BBDM Track E, steps={args.sample_steps}")
    report = {"device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters()), "s": args.s, "sample_steps": args.sample_steps, "train_images": len(train_dataset), "validation_images": len(validation_dataset), "history": history, "test_loaded": False, "seed_state": seed_state.as_dict()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_loss}, indent=2))


if __name__ == "__main__":
    main()
