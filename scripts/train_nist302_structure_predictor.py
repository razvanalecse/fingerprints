#!/usr/bin/env python3
"""Fine-tune the leakage-safe structure predictor (support/orientation/coherence) on SD302.

Mirrors scripts/train_structure_predictor.py's model, loss, and training loop
exactly, computing the 5-channel supervised target (support, cos2theta,
sin2theta, coherence, valid) on the fly from `target` (X_pseudo, the
registered exemplar -- the only "complete" reference SD302 provides; the
real latent is, by definition, missing exactly the information this target
needs in unobserved regions) instead of SocofingPartialDataset's clean
image. Supports --init-checkpoint to warm-start from the SOCOFing-trained
predictor (outputs/structure_predictor_full), matching this project's usual
Protocol-3 recipe for auxiliary, non-generative sub-networks (the
autoencoder was warm-started the same way; see docs/cross_dataset_protocols.md
for why the denoiser itself is NOT warm-started this way).
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
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor, structure_loss
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def build_structure_target(target_image: torch.Tensor, structure_shape: tuple[int, int]) -> torch.Tensor:
    """Compute the 5-channel (support, x, y, coherence, valid) target from X_pseudo, per-image."""

    batch_targets = []
    for index in range(target_image.shape[0]):
        image = target_image[index, 0].cpu().numpy()
        field = estimate_orientation_field(image)
        support = torch.from_numpy(field.foreground.astype("float32"))[None, None]
        coherence = torch.from_numpy(field.coherence.astype("float32"))[None, None]
        valid = torch.from_numpy(field.valid.astype("float32"))[None, None]
        theta = torch.from_numpy(field.theta.astype("float32"))[None, None]
        weight = coherence * valid
        weight_down = F.interpolate(weight, structure_shape, mode="area")
        vector_x = F.interpolate(weight * torch.cos(2.0 * theta), structure_shape, mode="area")
        vector_y = F.interpolate(weight * torch.sin(2.0 * theta), structure_shape, mode="area")
        vector_x = vector_x / weight_down.clamp_min(1e-6)
        vector_y = vector_y / weight_down.clamp_min(1e-6)
        norm = torch.sqrt(vector_x.square() + vector_y.square()).clamp_min(1e-6)
        vector_x, vector_y = vector_x / norm, vector_y / norm
        support_down = F.interpolate(support, structure_shape, mode="area")
        coherence_down = F.interpolate(coherence * valid, structure_shape, mode="area") / F.interpolate(
            valid, structure_shape, mode="area"
        ).clamp_min(1e-6)
        valid_down = (weight_down >= 0.05).float()
        batch_targets.append(torch.cat((support_down, vector_x, vector_y, coherence_down, valid_down), dim=1))
    return torch.cat(batch_targets, dim=0)


def run_epoch(model, loader, device, structure_shape, *, optimizer=None, max_batches=None, loss_options=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "support_bce", "support_dice", "orientation", "coherence", "support_iou", "circular_error")}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        target = build_structure_target(batch["target"], structure_shape).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(observed, mask)
            loss, terms = structure_loss(prediction, target, **(loss_options or {}))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
        support = target[:, :1]
        predicted_support = prediction[:, :1] >= 0.5
        target_support = support >= 0.5
        intersection = (predicted_support & target_support).flatten(1).sum(1).float()
        union = (predicted_support | target_support).flatten(1).sum(1).float().clamp_min(1)
        dot = prediction[:, 1:2] * target[:, 1:2] + prediction[:, 2:3] * target[:, 2:3]
        weights = target[:, 4:5] * target[:, 3:4]
        circular_error = ((1.0 - dot.clamp(-1, 1)) * weights).sum() / weights.sum().clamp_min(1e-6)
        values = {"loss": loss.detach(), **{k: v.detach() for k, v in terms.items()}, "support_iou": (intersection / union).mean().detach(), "circular_error": circular_error.detach()}
        size = target.shape[0]
        for key, value in values.items():
            totals[key] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty structure-prediction epoch")
    return {key: value / examples for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_state, device = seed_everything(seed), select_device(args.device)
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    common = dict(
        manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
        exemplar_roots=roots, output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    train_dataset = Nist302RegisteredDataset(split="train", **common)
    validation_dataset = Nist302RegisteredDataset(split="validation", **common)
    training = config["training"]
    structure_shape = tuple(config["data"]["structure_size"])
    channels = (8, 16, 24, 32) if args.smoke_test else tuple(config["model"]["channels"])
    epochs, batch_size = (1, 2) if args.smoke_test else (int(training["epochs"]), int(training["batch_size"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model = FingerprintStructurePredictor(channels=channels).to(device)
    initialized_from = None
    if args.init_checkpoint is not None:
        init = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if tuple(init["channels_used"]) != channels:
            raise ValueError("init checkpoint architecture does not match this config")
        model.load_state_dict(init["model_state"])
        initialized_from = str(args.init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    options = {"orientation_weight": float(training["orientation_weight"]), "coherence_weight": float(training["coherence_weight"])}
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, device, structure_shape, optimizer=optimizer, max_batches=2 if args.smoke_test else None, loss_options=options, gradient_clip=float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, device, structure_shape, max_batches=1 if args.smoke_test else None, loss_options=options)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "channels_used": list(channels), "config": config, "epoch": epoch, "validation_loss": best_loss, "initialized_from": initialized_from, "test_loaded": False}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f} IoU={validation_metrics['support_iou']:.4f} orient={validation_metrics['circular_error']:.4f}", flush=True)
        if not args.smoke_test and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    report = {"mode": "smoke_test" if args.smoke_test else "full", "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "model_parameters": sum(p.numel() for p in model.parameters()), "initialized_from": initialized_from, "history": history, "seed_state": seed_state.as_dict(), "test_loaded": False, "target_definition": "support/orientation/coherence derived from X_pseudo (registered_approximate exemplar), not pixel-aligned ground truth"}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss}, indent=2))


if __name__ == "__main__":
    main()
