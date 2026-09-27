#!/usr/bin/env python3
"""Evaluate identity retention in reconstructed SOCOFing regions.

The evaluation hides part of a print, reconstructs it, and measures whether
the true finger remains first in a gallery search.

Two protocols are run on the same reconstructions, and the gap between them is
the result:

  full        the completed image, observed pixels re-injected. Most of this
              image is the probe's own observed pixels, so a high rank-1 here
              measures copying, not recovery. It is reported only as the
              control that shows why the second protocol is necessary.
  hidden      only the synthesized region, compared against the same region of
              every gallery print. This is the one that can fail, because
              nothing in it was given to the model.

Chance rank-1 is 1/gallery_size. Similarity is ridge-orientation agreement in
doubled-angle space (orientation is axial), which is alignment-free here
because every SOCOFing image sits in the same coordinate frame.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probes", type=int, default=50)
    p.add_argument("--gallery", type=int, default=600)
    p.add_argument("--observed-fraction", type=float, default=0.80)
    p.add_argument("--family", default="random_rectangles")
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1729)
    return p.parse_args()


def orientation_descriptor(image):
    """Doubled-angle unit vectors, weighted by coherence: axial and comparable."""
    field = estimate_orientation_field(image, use_foreground_mask=False)
    weight = field.coherence * field.valid
    return np.stack([np.cos(2 * field.theta) * weight,
                     np.sin(2 * field.theta) * weight], axis=0)


def similarity(a, b, region):
    """Cosine similarity of the two descriptors inside `region`."""
    x = a[:, region].ravel()
    y = b[:, region].ravel()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return -1.0
    return float(x @ y / (nx * ny))


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    dataset = SocofingPartialDataset(
        manifest_path=args.manifest, image_root=args.image_root, split=args.split,
        output_shape=(128, 128), families=[MaskFamily(args.family)],
        observed_fractions=[args.observed_fraction], base_seed=args.seed,
    )
    dataset.set_epoch(0)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    gallery_size = min(args.gallery, len(dataset))
    gallery_truth = []
    for index in range(gallery_size):
        gallery_truth.append(dataset[index]["target"][0].numpy())
    gallery_desc = [orientation_descriptor(g) for g in gallery_truth]

    rows = []
    for probe_index in range(min(args.probes, gallery_size)):
        item = dataset[probe_index]
        observed = item["observed"][None].to(device)
        mask_t = item["mask"][None].to(device)
        truth = item["target"][0].numpy()
        mask = item["mask"][0].numpy()
        hidden = mask < 0.5
        if hidden.sum() < 200:
            continue
        estimate = model.reconstruct(observed, mask_t)[0, 0].cpu().numpy()
        completed = mask * truth + (1 - mask) * estimate

        completed_desc = orientation_descriptor(completed)
        full_region = np.ones_like(hidden, dtype=bool)
        for protocol, region in (("full", full_region), ("hidden", hidden)):
            scores = np.array([similarity(completed_desc, g, region) for g in gallery_desc])
            order = np.argsort(-scores)
            rank = int(np.where(order == probe_index)[0][0]) + 1
            rows.append({
                "probe_index": probe_index,
                "protocol": protocol,
                "rank_of_true_finger": rank,
                "rank1": int(rank == 1),
                "score_true": float(scores[probe_index]),
                "score_best_impostor": float(np.max(np.delete(scores, probe_index))),
                "hidden_pixels": int(hidden.sum()),
            })

    summary = {}
    for protocol in ("full", "hidden"):
        subset = [r for r in rows if r["protocol"] == protocol]
        ranks = np.array([r["rank_of_true_finger"] for r in subset])
        summary[protocol] = {
            "probes": len(subset),
            "rank1_rate": float(np.mean(ranks == 1)),
            "rank5_rate": float(np.mean(ranks <= 5)),
            "median_rank": float(np.median(ranks)),
            "mean_score_true": float(np.mean([r["score_true"] for r in subset])),
            "mean_score_best_impostor": float(np.mean([r["score_best_impostor"] for r in subset])),
        }

    report = {
        "gallery_size": gallery_size,
        "observed_fraction": args.observed_fraction,
        "mask_family": args.family,
        "chance_rank1": 1.0 / gallery_size,
        "checkpoint": str(args.checkpoint),
        "protocols": {
            "full": "completed image; mostly the probe's own observed pixels (control)",
            "hidden": "synthesized region only; nothing here was given to the model",
        },
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-probe.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
