#!/usr/bin/env python3
"""Paired held-out comparison of any two deterministic arms.

Scoring is on the officially held-out `quality == 1` pixels, never on
`X_pseudo`. Validation loss is deliberately ignored: for the trust-weighted arm
it is computed under a different per-pixel weighting than the baseline's, so the
two numbers are not even on the same scale, and this project has already shown
that agreement with the pseudo-target can invert a conclusion.
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
from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite
from fingerprint_reconstruction.metrics import paired_ridge_frequency_error, region_image_metrics
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field, orientation_error)
from fingerprint_reconstruction.reproducibility import seed_everything

METRICS = ("mae", "ssim_map_mean", "orientation_error", "ridge_frequency_relative_mae")
HIGHER = {"mae": False, "ssim_map_mean": True,
          "orientation_error": False, "ridge_frequency_relative_mae": False}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--baseline-checkpoint", "--candidate-checkpoint", "--manifest",
                 "--latent-root", "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--candidate-name", default="candidate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_reconstruction_model(ck["config"]["model"], channels=tuple(ck["channels_used"])).to(device)
    m.load_state_dict(ck["model_state"]); m.eval(); return m


def score(reference, estimate, region):
    out = dict(region_image_metrics(reference, estimate, region))
    rf = estimate_orientation_field(reference, use_foreground_mask=False)
    ef = estimate_orientation_field(estimate, use_foreground_mask=False)
    try:
        out["orientation_error"] = orientation_error(rf, ef, region_mask=region)
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
    conf = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(conf["ordered_weights"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    baseline, candidate = load(args.baseline_checkpoint, device), load(args.candidate_checkpoint, device)

    rows = []
    for batch in loader:
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        heldout = batch["quality"][0, 0].numpy() == 1
        if heldout.sum() < 400:
            continue
        latent = batch["latent_image"][0, 0].numpy()
        row = {"sample_id": str(batch["sample_id"][0]), "subject_id": str(batch["subject_id"][0])}
        for tag, model in (("baseline", baseline), ("candidate", candidate)):
            estimate = model.reconstruct(observed, mask)[0, 0].cpu().numpy()
            for k, v in score(latent, estimate, heldout).items():
                row[f"{tag}_{k}"] = v
        rows.append(row)

    subjects = defaultdict(list)
    for r in rows:
        subjects[r["subject_id"]].append(r)
    base_m, cand_m = {}, {}
    for metric in METRICS:
        base_m[metric] = [float(np.nanmean([r[f"baseline_{metric}"] for r in v])) for v in subjects.values()]
        cand_m[metric] = [float(np.nanmean([r[f"candidate_{metric}"] for r in v])) for v in subjects.values()]
    results = paired_metric_suite(base_m, cand_m, {m: HIGHER[m] for m in METRICS})
    report = {"candidate": args.candidate_name, "images": len(rows), "subjects": len(subjects),
              "region": "officially held-out quality==1 pixels",
              "improvement_sign": "positive = candidate better",
              "baseline_mean": {m: float(np.nanmean(base_m[m])) for m in METRICS},
              "candidate_mean": {m: float(np.nanmean(cand_m[m])) for m in METRICS},
              "paired_subject_tests": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("%-26s %9s %9s %12s %9s %10s" % ("metric","baseline",args.candidate_name,"improvement","cohen_dz","holm_p"))
    for m in METRICS:
        r = results[m]
        print("%-26s %9.4f %9.4f %+12.5f %9.2f %10.2e" % (
            m, report["baseline_mean"][m], report["candidate_mean"][m],
            r["mean_improvement"], r["cohen_dz"], r["wilcoxon_pvalue_holm"]))


if __name__ == "__main__":
    main()
