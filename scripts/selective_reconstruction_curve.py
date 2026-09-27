#!/usr/bin/env python3
"""Reconstruct only what can be justified: an error-coverage curve.

A reconstruction that covers the whole missing region and is wrong in half of
it is less useful to an examiner than one that covers 60% and is right. This
script turns the ground-truth-free reliability signal (disagreement between the
trained model's orientation field and a harmonic completion of the observed
field) into an abstention rule, and measures what it buys.

Pixels are ranked by disagreement, the least reliable are dropped, and the
model's true orientation error is reported over what remains. Three rankings
are compared:

  random     drop pixels at random (the floor: error must stay flat)
  signal     drop by geometric disagreement, which needs no ground truth
  oracle     drop by the true error itself (the ceiling: unreachable)

`sparsification_error`, the gap between the signal and oracle curves, was
listed as a secondary outcome in the pre-registered protocol
(`configs/experiment_registry.yaml`) before any of this existed.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics.orientation_geometry import complete_harmonic
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device

EROSION, MIN_REGION = 3, 400
COVERAGE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--checkpoint", "--manifest", "--latent-root",
                 "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--max-images", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence["ordered_weights"]))
    if args.max_images:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()

    signals, errors = [], []
    rng = np.random.default_rng(args.seed)
    for batch in loader:
        mask = batch["mask"][0, 0].numpy(); observed_bool = mask > 0.5
        heldout = batch["quality"][0, 0].numpy() == 1
        if heldout.sum() < MIN_REGION:
            continue
        trusted = ndimage.binary_erosion(observed_bool, iterations=EROSION, border_value=0)
        if trusted.sum() < MIN_REGION:
            continue
        latent = batch["latent_image"][0, 0].numpy()
        recon = model.reconstruct(batch["observed"].to(device),
                                  batch["mask"].to(device))[0, 0].cpu().numpy()
        true_field = estimate_orientation_field(latent, use_foreground_mask=False)
        model_field = estimate_orientation_field(recon, use_foreground_mask=False)
        observed_field = estimate_orientation_field(
            np.where(observed_bool, latent, 0.72).astype(np.float32), use_foreground_mask=False)
        harmonic = np.mod(0.5 * np.arctan2(
            complete_harmonic(np.sin(2 * observed_field.theta), trusted),
            complete_harmonic(np.cos(2 * observed_field.theta), trusted)), np.pi)
        signals.append(1 - np.cos(2 * (model_field.theta - harmonic))[heldout])
        errors.append(1 - np.cos(2 * (true_field.theta - model_field.theta))[heldout])

    signal = np.concatenate(signals); error = np.concatenate(errors)
    order_signal = np.argsort(signal)          # most agreed-upon first
    order_oracle = np.argsort(error)
    order_random = rng.permutation(error.size)

    curves = {name: [] for name in ("random", "signal", "oracle")}
    for coverage in COVERAGE:
        keep = max(1, int(round(coverage * error.size)))
        for name, order in (("random", order_random), ("signal", order_signal),
                            ("oracle", order_oracle)):
            curves[name].append(float(error[order[:keep]].mean()))

    full = curves["signal"][0]
    report = {
        "pixels": int(error.size),
        "images": len(signals),
        "coverage_levels": list(COVERAGE),
        "curves": curves,
        "error_at_full_coverage": full,
        "reduction_at_60pct_coverage": float(1 - curves["signal"][COVERAGE.index(0.6)] / full),
        "sparsification_error": float(np.mean(
            np.array(curves["signal"]) - np.array(curves["oracle"]))),
        "reading": (
            "If the signal curve falls while the random curve stays flat, the "
            "model can be told in advance which parts of its own output not to "
            "trust, with no target available."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("coverage  random   signal   oracle")
    for i, c in enumerate(COVERAGE):
        print("  %4.0f%%   %.4f   %.4f   %.4f" % (
            100 * c, curves["random"][i], curves["signal"][i], curves["oracle"][i]))
    print("\nerror reduction at 60%% coverage: %.1f%%" % (100 * report["reduction_at_60pct_coverage"]))


if __name__ == "__main__":
    main()
