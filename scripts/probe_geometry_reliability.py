#!/usr/bin/env python3
"""Evaluate target-free geometric reliability signals for reconstruction.

Section 26 showed the zero-pole geometric model is bimodal: occasionally as
accurate as a trained network, catastrophic 39% of the time. That is only a
weakness if you cannot tell the two cases apart in advance. This script tests
whether two signals that need **no ground truth at all** predict which case
you are in:

  fit_cost        how well the fitted zero-pole model explains the *observed*
                  orientations. Computable from the input alone.
  disagreement    how far two independent completions of the hidden region are
                  from each other. Needs no target, only two estimators.

If either correlates with the true hidden-region error, it can be used to
select a method per image, and -- more valuably -- to attach a reliability
estimate to a reconstruction on real forensic data, where no target exists.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.metrics.orientation_geometry import (
    complete_harmonic, fit_zero_pole_best, zero_pole_theta)
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything

CONDITIONS = (("rect", MaskFamily.RANDOM_RECTANGLES, 0.50),
              ("central", MaskFamily.CENTRAL_MISSING, 0.60),
              ("irregular", MaskFamily.IRREGULAR, 0.50),
              ("fragments", MaskFamily.DISCONNECTED_FRAGMENTS, 0.30))
MISSING_FILL, EROSION = 0.72, 3


def axial(a, b, region):
    return float(np.mean(1.0 - np.cos(2.0 * (a[region] - b[region]))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", type=int, default=25)
    p.add_argument("--split", default="validation")
    p.add_argument("--seed", type=int, default=1729)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"]))
    model.load_state_dict(checkpoint["model_state"]); model.eval()

    rows = []
    for name, family, fraction in CONDITIONS:
        dataset = SocofingPartialDataset(
            manifest_path=args.manifest, image_root=args.image_root, split=args.split,
            output_shape=(128, 128), families=[family], observed_fractions=[fraction],
            base_seed=args.seed)
        dataset.set_epoch(0)
        for index in range(min(args.images, len(dataset))):
            item = dataset[index]
            truth = item["target"][0].numpy(); mask = item["mask"][0].numpy()
            observed = mask > 0.5; hidden = ~observed
            if hidden.sum() < 200:
                continue
            true_field = estimate_orientation_field(truth, use_foreground_mask=False)
            masked = np.where(observed, truth, MISSING_FILL).astype(np.float32)
            observed_field = estimate_orientation_field(masked, use_foreground_mask=False)
            trusted = ndimage.binary_erosion(observed, iterations=EROSION, border_value=0)
            if trusted.sum() < 200:
                continue
            weights = observed_field.coherence * observed_field.valid
            cos_c, sin_c = np.cos(2 * observed_field.theta), np.sin(2 * observed_field.theta)
            harmonic = np.mod(0.5 * np.arctan2(complete_harmonic(sin_c, trusted),
                                               complete_harmonic(cos_c, trusted)), np.pi)
            try:
                fit = fit_zero_pole_best(observed_field.theta, trusted, weights)
                zp = zero_pole_theta(128, 128, fit.theta0, fit.cores, fit.deltas)
                # normalise: cost is a sum of squares over the fitted sample
                fit_cost = float(fit.cost) / max(int(trusted.sum()), 1)
            except ValueError:
                continue
            neural = model.reconstruct(item["observed"][None], item["mask"][None])[0, 0].numpy()
            neural_field = estimate_orientation_field(
                np.where(observed, truth, neural).astype(np.float32), use_foreground_mask=False)

            rows.append({
                "condition": name,
                "sample_id": item["sample_id"],
                # ---- signals available WITHOUT ground truth ----
                "fit_cost": fit_cost,
                "disagree_zp_harmonic": axial(zp, harmonic, hidden),
                "disagree_zp_neural": axial(zp, neural_field.theta, hidden),
                "disagree_harmonic_neural": axial(harmonic, neural_field.theta, hidden),
                # ---- truth, used only to validate the signals ----
                "err_zero_pole": axial(true_field.theta, zp, hidden),
                "err_harmonic": axial(true_field.theta, harmonic, hidden),
                "err_neural": axial(true_field.theta, neural_field.theta, hidden),
            })

    if not rows:
        raise SystemExit("no records")
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    correlations = {}
    for signal in ("fit_cost", "disagree_zp_harmonic", "disagree_zp_neural", "disagree_harmonic_neural"):
        for target in ("err_zero_pole", "err_harmonic", "err_neural"):
            rho, p = spearmanr([r[signal] for r in rows], [r[target] for r in rows])
            correlations[f"{signal} -> {target}"] = {"spearman_rho": float(rho), "p": float(p)}
    report = {"images": len(rows), "correlations": correlations}
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: round(v["spearman_rho"], 3) for k, v in correlations.items()}, indent=2))


if __name__ == "__main__":
    main()
