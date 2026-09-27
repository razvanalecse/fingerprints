#!/usr/bin/env python3
"""Does the geometric second opinion predict model error on real latents?

On SOCOFing, the disagreement between this project's trained model and a
classical harmonic completion of the orientation field predicted the model's
own error at Spearman rho = 0.48, using no ground truth. SOCOFing is clean
synthetic data; this script asks whether the signal survives on real forensic
latents, which is the only place it would be useful.

The region is the officially held-out `quality == 1` band: pixels the model
never received, and the only place on SD302 where a real target exists. Both
the signal and the error are measured there, so the comparison is exact.

  signal (no ground truth)  axial disagreement between the model's orientation
                            field and a harmonic completion of the observed
                            field, inside the held-out band
  error  (needs the latent) axial orientation error of the model against the
                            real latent in the same band

A positive correlation means a reconstruction can be flagged as unreliable
without ever seeing a target -- which on real forensic data is the only
situation that exists.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics.orientation_geometry import complete_harmonic
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device

EROSION = 3
MIN_REGION = 400


def axial(a, b, region):
    return float(np.mean(1.0 - np.cos(2.0 * (a[region] - b[region]))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--latent-root", type=Path, required=True)
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--sd302a-root", type=Path, required=True)
    p.add_argument("--sd302b-root", type=Path, required=True)
    p.add_argument("--sd302d-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8"))
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence["ordered_weights"]))
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()

    rows = []
    for batch in loader:
        observed_t = batch["observed"].to(device); mask_t = batch["mask"].to(device)
        mask = batch["mask"][0, 0].numpy()
        observed_bool = mask > 0.5
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        if int(heldout.sum()) < MIN_REGION:
            continue
        trusted = ndimage.binary_erosion(observed_bool, iterations=EROSION, border_value=0)
        if trusted.sum() < MIN_REGION:
            continue

        latent = batch["latent_image"][0, 0].numpy()
        reconstruction = model.reconstruct(observed_t, mask_t)[0, 0].cpu().numpy()

        true_field = estimate_orientation_field(latent, use_foreground_mask=False)
        model_field = estimate_orientation_field(reconstruction, use_foreground_mask=False)
        observed_field = estimate_orientation_field(
            np.where(observed_bool, latent, 0.72).astype(np.float32), use_foreground_mask=False)
        cos_c, sin_c = np.cos(2 * observed_field.theta), np.sin(2 * observed_field.theta)
        harmonic = np.mod(0.5 * np.arctan2(complete_harmonic(sin_c, trusted),
                                           complete_harmonic(cos_c, trusted)), np.pi)

        rows.append({
            "sample_id": str(batch["sample_id"][0]),
            "subject_id": str(batch["subject_id"][0]),
            "heldout_pixels": int(heldout.sum()),
            "signal_geometric_disagreement": axial(model_field.theta, harmonic, heldout),
            "error_model_orientation": axial(true_field.theta, model_field.theta, heldout),
            "error_harmonic_orientation": axial(true_field.theta, harmonic, heldout),
        })

    if not rows:
        raise SystemExit("no records")
    signal = np.array([r["signal_geometric_disagreement"] for r in rows])
    error = np.array([r["error_model_orientation"] for r in rows])
    rho, p = spearmanr(signal, error)

    # subject-level, matching the project's clustering rule
    subjects = {}
    for r in rows:
        subjects.setdefault(r["subject_id"], []).append(r)
    s_sig = [np.mean([x["signal_geometric_disagreement"] for x in v]) for v in subjects.values()]
    s_err = [np.mean([x["error_model_orientation"] for x in v]) for v in subjects.values()]
    rho_s, p_s = spearmanr(s_sig, s_err)

    report = {
        "images": len(rows), "subjects": len(subjects),
        "region": "officially held-out quality==1 pixels",
        "per_image": {"spearman_rho": float(rho), "p": float(p)},
        "per_subject": {"spearman_rho": float(rho_s), "p": float(p_s)},
        "mean_model_orientation_error": float(error.mean()),
        "mean_harmonic_orientation_error": float(
            np.mean([r["error_harmonic_orientation"] for r in rows])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
