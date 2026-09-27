#!/usr/bin/env python3
"""Paired held-out evaluation: geometry-conditioned model vs its own baseline.

Both models are run on the same images in the same loop, so the comparison is
exactly paired. They share an initialisation, a loss, a training schedule and a
confidence map; the only difference is that one additionally receives a solved
ridge-orientation field as conditioning.

Scoring is on the officially held-out `quality == 1` pixels -- never on
`X_pseudo`. The geometry model reached a slightly better validation loss, but
that loss is computed against the pseudo-target, which this project has already
shown can reverse a conclusion. Only this number counts.
"""
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.data.geometry_conditioned import GeometryConditionedDataset
from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite
from fingerprint_reconstruction.metrics import paired_ridge_frequency_error, region_image_metrics
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field, orientation_error)
from fingerprint_reconstruction.reproducibility import seed_everything

METRICS = ("mae", "ssim_map_mean", "orientation_error", "ridge_frequency_relative_mae")
HIGHER_IS_BETTER = {"mae": False, "ssim_map_mean": True,
                    "orientation_error": False, "ridge_frequency_relative_mae": False}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--geometry-checkpoint", "--baseline-checkpoint", "--manifest",
                 "--latent-root", "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int)
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


def load(path, device, auxiliary):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["config"]["model"])
    cfg["auxiliary_channels"] = auxiliary
    model = build_reconstruction_model(cfg, channels=tuple(checkpoint["channels_used"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def score(reference, estimate, region):
    out = dict(region_image_metrics(reference, estimate, region))
    ref_field = estimate_orientation_field(reference, use_foreground_mask=False)
    est_field = estimate_orientation_field(estimate, use_foreground_mask=False)
    try:
        out["orientation_error"] = orientation_error(ref_field, est_field, region_mask=region)
    except ValueError:
        out["orientation_error"] = float("nan")
    out.update(paired_ridge_frequency_error(reference, estimate, region))
    return out


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    base = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence["ordered_weights"]))
    dataset = GeometryConditionedDataset(base, Path(config["data"]["geometry_cache"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    geometry_model = load(args.geometry_checkpoint, device, GeometryConditionedDataset.GEOMETRY_CHANNELS)
    baseline_model = load(args.baseline_checkpoint, device, 0)

    rows = []
    for index, batch in enumerate(loader):
        if args.max_images is not None and index >= args.max_images:
            break
        observed = batch["observed"].to(device); mask = batch["mask"].to(device)
        geometry = batch["geometry"].to(device)
        heldout = batch["quality"][0, 0].numpy() == 1
        if heldout.sum() < 400:
            continue
        latent = batch["latent_image"][0, 0].numpy()

        raw = geometry_model(torch.cat((observed, mask, geometry), dim=1))
        geometry_estimate = (mask * observed + (1 - mask) * raw)[0, 0].cpu().numpy()
        baseline_estimate = baseline_model.reconstruct(observed, mask)[0, 0].cpu().numpy()

        row = {"sample_id": str(batch["sample_id"][0]), "subject_id": str(batch["subject_id"][0])}
        for tag, estimate in (("geometry", geometry_estimate), ("baseline", baseline_estimate)):
            for key, value in score(latent, estimate, heldout).items():
                row[f"{tag}_{key}"] = value
        rows.append(row)

    if not rows:
        raise SystemExit("no records")
    subjects = defaultdict(list)
    for row in rows:
        subjects[row["subject_id"]].append(row)
    baseline_means, geometry_means = {}, {}
    for metric in METRICS:
        baseline_means[metric] = [float(np.nanmean([r[f"baseline_{metric}"] for r in v]))
                                  for v in subjects.values()]
        geometry_means[metric] = [float(np.nanmean([r[f"geometry_{metric}"] for r in v]))
                                  for v in subjects.values()]
    results = paired_metric_suite(baseline_means, geometry_means,
                                  {m: HIGHER_IS_BETTER[m] for m in METRICS})
    report = {
        "images": len(rows), "subjects": len(subjects),
        "region": "officially held-out quality==1 pixels (never X_pseudo)",
        "improvement_sign": "positive mean_improvement = geometry conditioning is better",
        "baseline_mean": {m: float(np.mean(baseline_means[m])) for m in METRICS},
        "geometry_mean": {m: float(np.mean(geometry_means[m])) for m in METRICS},
        "paired_subject_tests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("%-28s %10s %10s %12s %10s %10s" % ("metric","baseline","geometry","improvement","cohen_dz","holm_p"))
    for m in METRICS:
        r = results[m]
        print("%-28s %10.4f %10.4f %+12.5f %10.2f %10.2e" % (
            m, report["baseline_mean"][m], report["geometry_mean"][m],
            r["mean_improvement"], r["cohen_dz"], r["wilcoxon_pvalue_holm"]))


if __name__ == "__main__":
    main()
