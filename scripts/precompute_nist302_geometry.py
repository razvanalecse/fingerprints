#!/usr/bin/env python3
"""Cache a solved ridge-orientation field for every SD302 sample.

The project already has a model conditioned on *predicted* ridge structure
(`StructureConditionedGatedNetwork`, master table section 18): a learned
network guesses the orientation field and the reconstructor is conditioned on
its guess. This caches the alternative: the orientation field is **solved**
from the observed pixels by harmonic completion in doubled-angle space, with
no learning anywhere in the geometry branch.

Three channels per sample:
    cos 2*theta, sin 2*theta   the completed axial orientation field
    trust                      1 where the orientation came from observed
                               pixels, 0 where it was extrapolated

The trust channel matters: without it the model cannot tell solved geometry
from invented geometry, which is the same mistake this project spent its whole
NIST302 extension documenting.

SD302 masks are fixed per sample (they come from the official quality map, not
from random augmentation), so this cache is valid for every epoch. That is not
true of the SOCOFing pipeline, where masks are resampled per epoch.
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

EROSION = 3
MISSING_FILL = 0.72


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--latent-root", type=Path, required=True)
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--sd302a-root", type=Path, required=True)
    p.add_argument("--sd302b-root", type=Path, required=True)
    p.add_argument("--sd302d-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8"))
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    for split in args.splits:
        dataset = Nist302RegisteredDataset(
            manifest_path=args.manifest, latent_root=args.latent_root,
            annotation_root=args.annotation_root, exemplar_roots=roots, split=split,
            output_shape=tuple(config["data"]["image_size"]),
            geometric_confidence_weights=tuple(confidence["ordered_weights"]))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            sample_id = str(batch["sample_id"][0])
            target = args.output / f"{sample_id.replace('/', '_')}.npz"
            if target.exists():
                continue
            observed = batch["observed"][0, 0].numpy()
            mask = batch["mask"][0, 0].numpy() > 0.5
            filled = np.where(mask, observed, MISSING_FILL).astype(np.float32)
            field = estimate_orientation_field(filled, use_foreground_mask=False)
            trusted = ndimage.binary_erosion(mask, iterations=EROSION, border_value=0)
            if trusted.sum() < 100:
                trusted = mask
            cos_filled = complete_harmonic(np.cos(2.0 * field.theta), trusted)
            sin_filled = complete_harmonic(np.sin(2.0 * field.theta), trusted)
            norm = np.maximum(np.hypot(cos_filled, sin_filled), 1e-6)
            np.savez_compressed(
                target,
                geometry=np.stack([cos_filled / norm, sin_filled / norm,
                                   trusted.astype(np.float32)]).astype(np.float16))
            written += 1
            if written % 100 == 0:
                print(f"{written} cached", flush=True)
    print(f"done, {written} new files in {args.output}")


if __name__ == "__main__":
    main()
