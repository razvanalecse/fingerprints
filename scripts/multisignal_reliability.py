#!/usr/bin/env python3
"""A ground-truth-free reliability model for reconstructed ridge structure.

The single-signal abstention curve cuts orientation error by 27.5% at 60%
coverage, against an oracle that could cut it by 77%. Most of that gap is
because one scalar cannot express where a reconstruction fails. This fits a
reliability model on several per-pixel signals, none of which requires a
target:

  disagreement   model orientation vs harmonic completion of the observed field
  solve_norm     |v| of the harmonically completed doubled-angle vector. The
                 solve shrinks the vector exactly where the surrounding ridges
                 disagree about what belongs in the hole, so this is an
                 ambiguity measure produced for free by the linear system.
  distance       distance to the nearest observed pixel
  obs_coherence  harmonically propagated coherence of the observed field
  out_coherence  coherence of the model's own output
  frequency      local ridge frequency of the model's output

The reliability model is trained with ground truth -- that is allowed, it is
fitted once offline -- but is validated by grouped cross-validation over
subjects, so every reported number comes from subjects the model never saw, and
at inference it consumes only the signals above.
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

EROSION, MIN_REGION = 3, 400
COVERAGE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
FEATURES = ("disagreement", "solve_norm", "distance", "obs_coherence",
            "out_coherence", "frequency")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--checkpoint", "--manifest", "--latent-root",
                 "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--max-images", type=int)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


def ridge_fit(X, y, alpha=1.0):
    """Closed-form ridge on centred, scaled features. Keeps the project dependency-free."""
    centre, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - centre) / scale
    gram = Z.T @ Z + alpha * np.eye(Z.shape[1])
    weights = np.linalg.solve(gram, Z.T @ (y - y.mean()))
    return centre, scale, weights, y.mean()


def ridge_predict(model, X):
    centre, scale, weights, offset = model
    return ((X - centre) / scale) @ weights + offset


def group_folds(groups, folds, seed):
    """Split whole subjects into folds, so no subject appears in train and test."""
    unique = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignment = {name: index % folds for index, name in enumerate(unique)}
    membership = np.array([assignment[name] for name in groups])
    for fold in range(folds):
        test = membership == fold
        if test.any() and (~test).any():
            yield ~test, test


def curve(error, order):
    out = []
    for coverage in COVERAGE:
        keep = max(1, int(round(coverage * error.size)))
        out.append(float(error[order[:keep]].mean()))
    return out


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
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

    features, errors, groups = [], [], []
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
        out_field = estimate_orientation_field(recon, use_foreground_mask=False)
        obs_field = estimate_orientation_field(
            np.where(observed_bool, latent, 0.72).astype(np.float32), use_foreground_mask=False)

        cos_filled = complete_harmonic(np.cos(2 * obs_field.theta), trusted)
        sin_filled = complete_harmonic(np.sin(2 * obs_field.theta), trusted)
        norm = np.hypot(cos_filled, sin_filled)
        harmonic = np.mod(0.5 * np.arctan2(sin_filled, cos_filled), np.pi)
        coherence_filled = complete_harmonic(obs_field.coherence, trusted)
        distance = ndimage.distance_transform_edt(~trusted)

        stack = np.stack([
            (1 - np.cos(2 * (out_field.theta - harmonic)))[heldout],
            norm[heldout],
            distance[heldout],
            coherence_filled[heldout],
            out_field.coherence[heldout],
            np.abs(np.gradient(out_field.theta)[0])[heldout],
        ], axis=1)
        features.append(stack)
        errors.append((1 - np.cos(2 * (true_field.theta - out_field.theta)))[heldout])
        groups.append(np.full(int(heldout.sum()), str(batch["subject_id"][0])))

    X = np.nan_to_num(np.concatenate(features)); y = np.concatenate(errors)
    g = np.concatenate(groups)
    predicted = np.zeros_like(y)
    for train_mask, test_mask in group_folds(g, args.folds, args.seed):
        fitted = ridge_fit(X[train_mask], y[train_mask])
        predicted[test_mask] = ridge_predict(fitted, X[test_mask])

    single = X[:, 0]
    curves = {
        "random": curve(y, np.random.default_rng(args.seed).permutation(y.size)),
        "single_signal": curve(y, np.argsort(single)),
        "multisignal": curve(y, np.argsort(predicted)),
        "oracle": curve(y, np.argsort(y)),
    }
    index60 = COVERAGE.index(0.6)
    full = curves["oracle"][0]
    report = {
        "pixels": int(y.size), "subjects": int(len(set(g.tolist()))),
        "folds": args.folds, "features": list(FEATURES),
        "validation": "GroupKFold over subjects; every prediction is out-of-fold",
        "curves": curves,
        "reduction_at_60pct": {
            name: float(1 - values[index60] / full) for name, values in curves.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("coverage  random  single  multi   oracle")
    for i, c in enumerate(COVERAGE):
        print("  %4.0f%%  %.4f  %.4f  %.4f  %.4f" % (
            100 * c, curves["random"][i], curves["single_signal"][i],
            curves["multisignal"][i], curves["oracle"][i]))
    print("\nerror reduction at 60%% coverage:")
    for name in ("single_signal", "multisignal", "oracle"):
        print("  %-14s %.1f%%" % (name, 100 * report["reduction_at_60pct"][name]))


if __name__ == "__main__":
    main()
