#!/usr/bin/env python3
"""Per-pixel trust in the registered pseudo-target.

The project's central finding is that fine-tuning against `X_pseudo` -- a
different impression of the same finger -- degrades reconstruction measured on
real pixels. The cause is not that `X_pseudo` is useless but that it is wrong
in places, and the loss currently treats every pixel of it as equally true.

This computes, per pixel, how much the pseudo-target should be believed:

  agreement   where the latent is actually observed, compare its ridge
              orientation with the registered exemplar's. Where two impressions
              of the same finger agree on ridge direction, the registration is
              locally sound; where they disagree, the target is registration
              noise.
  extrapolate agreement can only be *measured* where the latent is observed, so
              it is harmonically extended into the rest of the image: trust
              decays smoothly away from the evidence that established it.

The result drops straight into `RegisteredApproximateLoss`'s existing
per-pixel `geometric_confidence` slot.

No leakage: agreement is computed from `observed` (official quality >= 2),
which is the model's own input. The held-out quality == 1 evaluation pixels are
never read.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics.orientation_geometry import complete_harmonic
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field

EROSION, MISSING_FILL, MIN_TRUSTED = 3, 0.72, 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--manifest", "--latent-root", "--annotation-root",
                 "--sd302a-root", "--sd302b-root", "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    args.output.mkdir(parents=True, exist_ok=True)
    written, stats = 0, []
    for split in args.splits:
        dataset = Nist302RegisteredDataset(
            manifest_path=args.manifest, latent_root=args.latent_root,
            annotation_root=args.annotation_root, exemplar_roots=roots, split=split,
            output_shape=tuple(config["data"]["image_size"]),
            geometric_confidence_weights=tuple(confidence["ordered_weights"]))
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            sample_id = str(batch["sample_id"][0]).replace("/", "_")
            path = args.output / f"{sample_id}.npz"
            if path.exists():
                continue
            observed = batch["observed"][0, 0].numpy()
            mask = batch["mask"][0, 0].numpy() > 0.5
            pseudo = batch["target"][0, 0].numpy()
            latent_field = estimate_orientation_field(
                np.where(mask, observed, MISSING_FILL).astype(np.float32),
                use_foreground_mask=False)
            pseudo_field = estimate_orientation_field(pseudo, use_foreground_mask=False)
            trusted = ndimage.binary_erosion(mask, iterations=EROSION, border_value=0)
            if trusted.sum() < MIN_TRUSTED:
                trusted = mask
            # 1 where the two impressions agree on ridge direction, 0 where they
            # are perpendicular. Coherence gates out places where neither field
            # is meaningful.
            agreement = 0.5 * (1.0 + np.cos(2.0 * (latent_field.theta - pseudo_field.theta)))
            reliable = trusted & (latent_field.coherence > 0.1) & (pseudo_field.coherence > 0.1)
            if reliable.sum() < MIN_TRUSTED:
                trust = np.full_like(agreement, float(np.clip(agreement[trusted].mean(), 0, 1)))
            else:
                trust = np.clip(complete_harmonic(agreement, reliable), 0.0, 1.0)
            np.savez_compressed(path, trust=trust.astype(np.float16))
            stats.append(float(trust.mean())); written += 1
            if written % 200 == 0:
                print(f"{written} cached, running mean trust {np.mean(stats):.3f}", flush=True)
    print(f"done: {written} files, mean trust {np.mean(stats) if stats else float('nan'):.3f}, "
          f"min {np.min(stats) if stats else float('nan'):.3f}, max {np.max(stats) if stats else float('nan'):.3f}")


if __name__ == "__main__":
    main()
