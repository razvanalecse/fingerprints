#!/usr/bin/env python3
"""Compare over-smoothing artifacts at two ridge-resolving image scales.

Section 15 and section 23 established that at 128x128 the reconstruction is
more orientation-uniform, less curved, carries about half the minutiae density
of the real ridges, and is measurably more predictable under a Markov model of
orientation transitions. `docs/nist302_native_resolution_feasibility.md`
proposed a cause: at 128x128 the mean ridge period is only 6.45 pixels, far
below the 8-15 pixels needed to represent ridge phase at all.

This script tests that proposal causally by running the same recipe at two
working resolutions and asking whether the artifact shrinks.

Two things have to be handled or the comparison is meaningless:

* Minutiae density per 1000 pixels halves mechanically when you double the
  linear resolution, because the same physical area now contains four times
  the pixels. Ratios (reconstruction / real) are reported instead, which are
  scale-free.
* A Markov step of k pixels probes a different physical distance at each
  resolution. Steps are therefore scaled with the image size, so both runs
  measure transitions across the same fraction of a ridge period.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics import summarize_ridge_markov, summarize_ridge_topology
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device

MINIMUM_ROI_PIXELS = 400
BASE_STEPS = (2, 4)           # defined at 128px; scaled below
BASE_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image_size = tuple(config["data"]["image_size"])
    scale = image_size[0] / BASE_SIZE
    steps = tuple(max(1, int(round(s * scale))) for s in BASE_STEPS)

    confidence = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=image_size,
        geometric_confidence_weights=tuple(confidence["ordered_weights"]),
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[dict] = []
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        reconstruction = model.reconstruct(observed, mask)[0, 0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        if int(heldout.sum()) < MINIMUM_ROI_PIXELS * (scale ** 2):
            continue

        real_field = estimate_orientation_field(reference, use_foreground_mask=False)
        recon_field = estimate_orientation_field(reconstruction, use_foreground_mask=False)
        real_topology = summarize_ridge_topology(
            reference, heldout, real_field.theta, real_field.coherence, real_field.valid)
        recon_topology = summarize_ridge_topology(
            reconstruction, heldout, recon_field.theta, recon_field.coherence, recon_field.valid)

        row = {
            "sample_id": str(batch["sample_id"][0]),
            "subject_id": str(batch["subject_id"][0]),
            # scale-free: 1.0 would mean the reconstruction carries as much
            # ridge detail as the real latent it replaces
            "minutiae_density_ratio": (
                recon_topology.minutiae_density_per_1000px
                / real_topology.minutiae_density_per_1000px
                if real_topology.minutiae_density_per_1000px > 0 else float("nan")
            ),
            "coherence_diff": (recon_topology.orientation_coherence_mean
                               - real_topology.orientation_coherence_mean),
            "curvature_ratio": (
                recon_topology.orientation_curvature_mean
                / real_topology.orientation_curvature_mean
                if real_topology.orientation_curvature_mean > 0 else float("nan")
            ),
        }
        for base, step in zip(BASE_STEPS, steps):
            real_markov = summarize_ridge_markov(
                real_field.theta, heldout, valid=real_field.valid, step=step)
            recon_markov = summarize_ridge_markov(
                recon_field.theta, heldout, valid=recon_field.valid, step=step)
            if not np.isfinite(real_markov.transition_entropy_bits):
                continue
            row[f"entropy_diff_base{base}"] = (
                recon_markov.transition_entropy_bits - real_markov.transition_entropy_bits)
            row[f"self_rate_diff_base{base}"] = (
                recon_markov.self_transition_rate - real_markov.self_transition_rate)
        records.append(row)

    if not records:
        raise SystemExit("no records produced")

    keys = [k for k in records[0] if k not in ("sample_id", "subject_id")]
    summary = {k: float(np.nanmean([r[k] for r in records if k in r])) for k in keys}
    report = {
        "image_size": list(image_size),
        "markov_steps_px": list(steps),
        "markov_steps_equivalent_at_128px": list(BASE_STEPS),
        "checkpoint": str(args.checkpoint),
        "images": len(records),
        "note": (
            "minutiae_density_ratio and curvature_ratio are scale-free; "
            "coherence is already scale-free; Markov steps are scaled with "
            "resolution so both runs probe the same physical distance."
        ),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"image_size": list(image_size), "images": len(records), **summary}, indent=2))


if __name__ == "__main__":
    main()
