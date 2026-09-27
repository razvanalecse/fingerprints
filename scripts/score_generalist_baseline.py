#!/usr/bin/env python3
"""Score generalist-model completions against exact ground truth.

Answers: can a general-purpose image model produce fingerprint completions
that look convincing without recovering the ridge structure that was there?

Every completion is resampled to 128x128 and has the observed pixels
re-injected before anything is measured, so the model is neither credited nor
penalised for touching the region it was given. Only the hidden region is
scored.

Three families of measurement are reported, because the project's own results
show they can disagree:

  fidelity    MAE, SSIM                  -- "is it the right image?"
  structure   orientation, ridge frequency, coherence, curvature,
              minutiae density, Markov predictability
                                         -- "is it a real ridge field?"
  agreement   spread across repeated completions of the same input
                                         -- "does the model know it is guessing?"

Pass --reference-checkpoint to run one of this project's deterministic models
on exactly the same masked inputs, so the comparison is like for like.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from fingerprint_reconstruction.metrics import (
    paired_ridge_frequency_error,
    region_image_metrics,
    summarize_ridge_markov,
    summarize_ridge_topology,
)
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    orientation_error,
)

MIN_REGION_PIXELS = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (128, 128):
        image = image.resize((128, 128), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def score_completion(truth, estimate, mask, missing) -> dict[str, float]:
    """All metrics for one completed image, restricted to the hidden region."""
    filled = mask * truth + (1.0 - mask) * estimate
    metrics = region_image_metrics(truth, filled, missing)
    row = {f"missing_{k}": v for k, v in metrics.items()}

    truth_field = estimate_orientation_field(truth, use_foreground_mask=False)
    filled_field = estimate_orientation_field(filled, use_foreground_mask=False)
    try:
        row["missing_orientation_error"] = orientation_error(
            truth_field, filled_field, region_mask=missing
        )
    except ValueError:
        row["missing_orientation_error"] = float("nan")
    row.update({
        f"missing_{k}": v
        for k, v in paired_ridge_frequency_error(truth, filled, missing).items()
    })

    truth_topology = summarize_ridge_topology(
        truth, missing, truth_field.theta, truth_field.coherence, truth_field.valid
    )
    filled_topology = summarize_ridge_topology(
        filled, missing, filled_field.theta, filled_field.coherence, filled_field.valid
    )
    row["coherence_diff"] = (
        filled_topology.orientation_coherence_mean - truth_topology.orientation_coherence_mean
    )
    row["curvature_diff"] = (
        filled_topology.orientation_curvature_mean - truth_topology.orientation_curvature_mean
    )
    row["minutiae_density_diff"] = (
        filled_topology.minutiae_density_per_1000px - truth_topology.minutiae_density_per_1000px
    )

    truth_markov = summarize_ridge_markov(
        truth_field.theta, missing, valid=truth_field.valid, step=4
    )
    filled_markov = summarize_ridge_markov(
        filled_field.theta, missing, valid=filled_field.valid, step=4
    )
    row["entropy_diff_step4"] = (
        filled_markov.transition_entropy_bits - truth_markov.transition_entropy_bits
    )
    row["self_rate_diff_step4"] = (
        filled_markov.self_transition_rate - truth_markov.self_transition_rate
    )
    return row


def load_reference_model(path: Path, device: str):
    import torch
    from fingerprint_reconstruction.models.factory import build_reconstruction_model

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cases_root = args.pack / "cases"
    if not cases_root.is_dir():
        raise SystemExit(f"no cases/ directory under {args.pack}")

    reference_model = None
    if args.reference_checkpoint is not None:
        reference_model = load_reference_model(args.reference_checkpoint, args.device)

    rows: list[dict] = []
    variability: list[dict] = []
    missing_completions: list[str] = []

    for case_dir in sorted(p for p in cases_root.iterdir() if p.is_dir()):
        truth = load_gray(case_dir / "ground_truth.png")
        mask = (load_gray(case_dir / "mask.png") > 0.5).astype(np.float32)
        missing = mask < 0.5
        if missing.sum() < MIN_REGION_PIXELS:
            continue

        completions = sorted(case_dir.glob("completion_*.png"))
        if not completions:
            missing_completions.append(case_dir.name)
        for path in completions:
            row = {"case_id": case_dir.name, "arm": "generalist", "completion": path.stem}
            row.update(score_completion(truth, load_gray(path), mask, missing))
            rows.append(row)

        if len(completions) >= 2:
            filled = [
                mask * truth + (1.0 - mask) * load_gray(p) for p in completions
            ]
            spread = [
                float(np.abs(a - b)[missing].mean()) for a, b in combinations(filled, 2)
            ]
            variability.append({
                "case_id": case_dir.name,
                "completions": len(completions),
                "mean_pairwise_mae_in_hidden_region": float(np.mean(spread)),
                "max_pairwise_mae_in_hidden_region": float(np.max(spread)),
            })

        if reference_model is not None:
            import torch

            with torch.no_grad():
                observed = torch.from_numpy(truth * mask)[None, None].to(args.device)
                mask_tensor = torch.from_numpy(mask)[None, None].to(args.device)
                estimate = reference_model.reconstruct(observed, mask_tensor)[0, 0].cpu().numpy()
            row = {
                "case_id": case_dir.name,
                "arm": "project_model",
                "completion": args.reference_checkpoint.parent.name,
            }
            row.update(score_completion(truth, estimate, mask, missing))
            rows.append(row)

    if not rows:
        raise SystemExit(
            "No completions found. Put completion_1.png ... into each case folder first.\n"
            f"Cases still empty: {len(missing_completions)}"
        )

    output = args.output or (args.pack / "scores")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per-completion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headline = (
        "missing_mae", "missing_ssim_map_mean", "missing_orientation_error",
        "missing_ridge_frequency_relative_mae", "coherence_diff", "curvature_diff",
        "minutiae_density_diff", "entropy_diff_step4", "self_rate_diff_step4",
    )
    summary: dict[str, dict[str, float]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "n": len(arm_rows),
            **{
                key: float(np.nanmean([r[key] for r in arm_rows if key in r]))
                for key in headline
            },
        }

    report = {
        "pack": str(args.pack),
        "cases_scored": len({r["case_id"] for r in rows}),
        "cases_without_completions": missing_completions,
        "scoring_rule": "observed pixels re-injected; only the hidden region is measured",
        "summary_by_arm": summary,
        "between_completion_variability": variability,
        "reading": (
            "Low MAE with large coherence_diff, negative curvature_diff and negative "
            "minutiae_density_diff is the signature of a completion that looks right "
            "and is structurally invented. High between-completion variability on an "
            "identical input means the model is guessing without signalling it."
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_by_arm": summary,
                      "cases_without_completions": len(missing_completions)}, indent=2))


if __name__ == "__main__":
    main()
