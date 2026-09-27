#!/usr/bin/env python3
"""Compare model rankings under approximate and real evaluation targets.

The critical finding in `docs/nist302_ablation_master_table.md` shows that for
*one* checkpoint pair, scoring against the registered approximate target
`X_pseudo` reverses the conclusion you would draw from real held-out pixels.
That comparison covers two models. This script evaluates whether ranking a
whole model family by the
convenient target, do you select the same model you would have selected using
real evidence?

Both scores come from a single pass over the same images, so the comparison is
exactly paired:

  Track B: metrics against `X_pseudo` over missing pixels inside the evaluation
           ROI -- the target this project trains against, and the only dense
           one available on real latents.
  Track C: metrics against the real latent over officially held-out
           `quality == 1` pixels, which no loss and no target construction ever
           touched.

Models are ranked under each target by subject-mean score, and the two
orderings are compared with Spearman and Kendall rank correlation. A high
correlation would mean the convenient target is a usable proxy; a low or
negative one means model selection itself is corrupted by the target choice.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics import region_image_metrics
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device

MINIMUM_REGION_PIXELS = 200
RANKED_METRICS = ("mae", "ssim_map_mean", "orientation_error")
# Lower is better for every metric except SSIM.
HIGHER_IS_BETTER = {"mae": False, "ssim_map_mean": True, "orientation_error": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def load_models(specs, device):
    models = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = build_reconstruction_model(
            checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[name] = model
    return models


def score(reference: np.ndarray, estimate: np.ndarray, region: np.ndarray) -> dict[str, float]:
    metrics = region_image_metrics(reference, estimate, region)
    reference_field = estimate_orientation_field(reference, use_foreground_mask=False)
    estimate_field = estimate_orientation_field(estimate, use_foreground_mask=False)
    try:
        metrics["orientation_error"] = orientation_error(
            reference_field, estimate_field, region_mask=region
        )
    except ValueError:
        metrics["orientation_error"] = float("nan")
    return metrics


def subject_mean_table(records, track, metric):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        value = row.get(f"{track}_{metric}")
        if value is None or not np.isfinite(value):
            continue
        grouped[row["model"]][row["subject_id"]].append(value)
    return {
        model: float(np.mean([np.mean(v) for v in subjects.values()]))
        for model, subjects in grouped.items()
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split=args.split,
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    models = load_models(args.model, device)
    records: list[dict] = []
    skipped = 0

    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        pseudo = batch["target"][0, 0].numpy()
        latent = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        roi = batch["evaluation_roi"][0, 0].numpy() > 0.5
        missing = (batch["mask"][0, 0].numpy() < 0.5) & roi
        heldout = quality == 1
        if missing.sum() < MINIMUM_REGION_PIXELS or heldout.sum() < MINIMUM_REGION_PIXELS:
            skipped += 1
            continue

        for name, model in models.items():
            estimate = model.reconstruct(observed, mask)[0, 0].cpu().numpy()
            track_b = score(pseudo, estimate, missing)
            track_c = score(latent, estimate, heldout)
            row = {
                "model": name,
                "sample_id": str(batch["sample_id"][0]),
                "subject_id": str(batch["subject_id"][0]),
            }
            row.update({f"trackB_{k}": v for k, v in track_b.items()})
            row.update({f"trackC_{k}": v for k, v in track_c.items()})
            records.append(row)

    if not records:
        raise ValueError("no records produced")

    rankings = {}
    for metric in RANKED_METRICS:
        pseudo_scores = subject_mean_table(records, "trackB", metric)
        real_scores = subject_mean_table(records, "trackC", metric)
        shared = sorted(set(pseudo_scores) & set(real_scores))
        if len(shared) < 3:
            continue
        higher = HIGHER_IS_BETTER[metric]
        # rank 1 = best model under that target
        def order(scores):
            return sorted(shared, key=lambda m: -scores[m] if higher else scores[m])

        pseudo_order = order(pseudo_scores)
        real_order = order(real_scores)
        pseudo_rank = {m: i + 1 for i, m in enumerate(pseudo_order)}
        real_rank = {m: i + 1 for i, m in enumerate(real_order)}
        rho, rho_p = spearmanr([pseudo_rank[m] for m in shared], [real_rank[m] for m in shared])
        tau, tau_p = kendalltau([pseudo_rank[m] for m in shared], [real_rank[m] for m in shared])
        rankings[metric] = {
            "models": shared,
            "pseudo_target_scores": {m: pseudo_scores[m] for m in shared},
            "real_pixel_scores": {m: real_scores[m] for m in shared},
            "pseudo_target_order_best_first": pseudo_order,
            "real_pixel_order_best_first": real_order,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "kendall_tau": float(tau),
            "kendall_p": float(tau_p),
            "same_winner": pseudo_order[0] == real_order[0],
            "winner_under_pseudo_target": pseudo_order[0],
            "winner_under_real_pixels": real_order[0],
            "rank_of_pseudo_winner_under_real_pixels": real_rank[pseudo_order[0]],
        }

    report = {
        "split": args.split,
        "test_loaded": args.split == "test",
        "models": sorted(models),
        "images_scored": len({r["sample_id"] for r in records}),
        "images_skipped_small_region": skipped,
        "track_b": "X_pseudo over missing pixels inside the evaluation ROI",
        "track_c": "real latent over officially held-out quality==1 pixels",
        "interpretation": (
            "A high positive Spearman rho would mean the convenient target ranks models "
            "the same way real evidence does, so using it for model selection is safe. "
            "A low or negative rho means the target choice corrupts model selection "
            "itself, not merely the reported score of a single checkpoint."
        ),
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".per-image.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(
        {
            metric: {
                "spearman_rho": round(value["spearman_rho"], 3),
                "winner_under_pseudo_target": value["winner_under_pseudo_target"],
                "winner_under_real_pixels": value["winner_under_real_pixels"],
                "rank_of_pseudo_winner_under_real_pixels": value["rank_of_pseudo_winner_under_real_pixels"],
            }
            for metric, value in rankings.items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
