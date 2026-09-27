#!/usr/bin/env python3
"""Adapt to the SD302 domain without ever touching the pseudo-target.

The project's central finding is that fine-tuning against `X_pseudo` -- a
different impression of the same finger -- degrades reconstruction measured on
real pixels. Section 17 identified the obvious counter-move and left it open:
Track E supplies an *exact* missing-region target inside the SD302 image
domain, by degrading a clean SD302 exemplar and then masking it. Training there
gives domain adaptation with no approximate target anywhere in the loss.

Because the target is exact, this uses the plain `MaskedReconstructionLoss`.
None of `RegisteredApproximateLoss`'s machinery -- geometric confidence,
evaluation ROI, support weighting -- is needed or wanted: all of it exists to
cope with a target that cannot be trusted pixel-wise, and here it can.

The model is initialised from the same SOCOFing checkpoint every other SD302
arm starts from, so the comparison against zero-shot and against pseudo-target
fine-tuning is like for like.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.nist302_synthetic_dataset import Sd302SyntheticPartialDataset
from fingerprint_reconstruction.losses import MaskedReconstructionLoss
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--exemplar-manifest", "--registered-manifest",
                 "--sd302a-root", "--sd302b-root", "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--seed", type=int, default=1729)
    return p.parse_args()


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals, examples = {}, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model.reconstruct(observed, mask)
            result = loss_fn(prediction, target, mask)
            if not torch.isfinite(result.total):
                raise FloatingPointError(f"non-finite loss at batch {index}")
            if training:
                result.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        size = observed.shape[0]
        examples += size
        for key, value in result.components.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        totals["total"] = totals.get("total", 0.0) + float(result.total.detach()) * size
    return {k: v / max(examples, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    shape = tuple(config["data"]["image_size"])
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    datasets = {
        split: Sd302SyntheticPartialDataset(
            exemplar_manifest_path=args.exemplar_manifest,
            registered_manifest_path=args.registered_manifest,
            exemplar_roots=roots, split=split, output_shape=shape, base_seed=args.seed)
        for split in ("train", "validation")
    }
    batch_size = 2 if args.smoke_test else int(config["training"]["batch_size"])
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    init_path = Path(config["model"]["initialize_from"])
    init = torch.load(init_path, map_location=device, weights_only=False)
    channels = tuple(int(v) for v in init["channels_used"])
    model = build_reconstruction_model(config["model"], channels=channels).to(device)
    model.load_state_dict(init["model_state"])

    loss_keys = {"missing_l1_weight", "missing_mse_weight", "observed_l1_weight",
                 "orientation_weight", "orientation_window", "gradient_weight",
                 "ridge_energy_weight", "ridge_sigma", "ridge_energy_window"}
    loss_fn = MaskedReconstructionLoss(
        **{k: v for k, v in config["loss"].items() if k in loss_keys}).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]))

    epochs = 2 if args.smoke_test else (args.epochs or int(config["training"]["epochs"]))
    max_batches = 8 if args.smoke_test else None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "method-audit.json").write_text(json.dumps({
        "target": "Track E: exact missing-region ground truth by construction",
        "pseudo_target_used": False,
        "loss": "MaskedReconstructionLoss (exact target, no confidence weighting)",
        "initialization": str(init_path),
        "test_loaded": False,
    }, indent=2) + "\n", encoding="utf-8")

    history, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, loaders["train"], loss_fn, device, optimizer, max_batches)
        with torch.no_grad():
            validation = run_epoch(model, loaders["validation"], loss_fn, device, None, max_batches)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(f"epoch={epoch}/{epochs} train={train['total']:.6f} "
              f"validation={validation['total']:.6f}", flush=True)
        if validation["total"] < best:
            best = validation["total"]
            torch.save({"model_state": model.state_dict(), "channels_used": list(channels),
                        "config": config, "epoch": epoch}, args.output / "best-checkpoint.pt")
    (args.output / "training-metrics.json").write_text(json.dumps({
        "history": history, "best_validation_loss": best,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n", encoding="utf-8")
    print("best validation", best)


if __name__ == "__main__":
    main()
