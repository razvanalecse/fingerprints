#!/usr/bin/env python3
"""Repeated-measures residual-diffusion benchmark over mask geometry and r.

Every selected fingerprint is evaluated under every compatible condition.
Within one fingerprint/family pair, masks are nested so increasing r reveals
additional pixels without changing the underlying geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from fingerprint_reconstruction.analysis.socofing_eda import read_manifest
from fingerprint_reconstruction.data.partial_pairs import (
    observed_fraction_report,
    stable_nested_mask_seed,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily, MaskGenerator
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale
from fingerprint_reconstruction.preprocessing.orientation import estimate_foreground_mask
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from evaluate_latent_reinjection import evaluate_samples
from evaluate_residual_latent import load_model
from preview_shape_constrained_latent import load_support_model


DEFAULT_FAMILIES = tuple(family for family in MaskFamily)
DEFAULT_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--samples-k", type=int, default=3)
    parser.add_argument("--pixel-projection-interval", type=int, default=5)
    parser.add_argument(
        "--families", nargs="+", choices=[item.value for item in MaskFamily],
        default=[item.value for item in DEFAULT_FAMILIES],
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument(
        "--fraction-domain",
        choices=("canvas", "fingerprint_roi"),
        default="canvas",
        help="Area on which each requested observed fraction is enforced.",
    )
    return parser.parse_args()


def compatible_fractions(family: MaskFamily, fractions: list[float]) -> list[float]:
    if family == MaskFamily.SEVERE_PARTIAL:
        return [value for value in fractions if value <= 0.30 + 1e-9]
    return fractions


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["sample_id"], row["mask_family"], row["observed_fraction"])
            for row in csv.DictReader(stream)
        }


@torch.no_grad()
def main():
    args = parse_args()
    if args.samples_k < 2:
        raise ValueError("samples-k must be at least 2 for uncertainty estimation")
    fractions = sorted({float(value) for value in args.fractions})
    if not fractions or min(fractions) <= 0.0 or max(fractions) >= 1.0:
        raise ValueError("fractions must lie strictly between zero and one")
    families = [MaskFamily(value) for value in args.families]

    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    base_seed = int(config["experiment"]["seed"])
    image_shape = tuple(int(value) for value in config["data"]["image_size"])
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not 2 <= args.num_images <= len(rows):
        raise ValueError("num-images must be between 2 and the selected split size")
    selected_indices = np.linspace(0, len(rows) - 1, args.num_images).round().astype(int)
    if len(set(selected_indices.tolist())) != args.num_images:
        raise RuntimeError("selected image indices are not unique")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "factorial-per-condition.csv"
    completed = existing_keys(csv_path)
    writer = None
    stream = None
    total_conditions = sum(
        len(compatible_fractions(family, fractions)) for family in families
    ) * args.num_images
    finished = len(completed)
    sampling_seconds = 0.0
    started_all = time.perf_counter()
    try:
        for selected_position, row_index in enumerate(selected_indices):
            row = rows[int(row_index)]
            target_np = load_grayscale(
                args.image_root / row["relative_path"], output_shape=image_shape
            )
            foreground_np = estimate_foreground_mask(target_np)
            target = torch.from_numpy(target_np).to(device)
            foreground = torch.from_numpy(foreground_np).to(device).bool()
            generator = MaskGenerator(image_shape)
            for family in families:
                family_fractions = compatible_fractions(family, fractions)
                if not family_fractions:
                    continue
                nested_seed = stable_nested_mask_seed(
                    base_seed=base_seed,
                    sample_id=row["sample_id"],
                    family=family,
                    replicate=args.replicate,
                )
                nested_kwargs = {
                    "family": family,
                    "observed_fractions": family_fractions,
                    "seed": nested_seed,
                    "num_fragments": (
                        4 if family == MaskFamily.DISCONNECTED_FRAGMENTS else
                        3 if family == MaskFamily.SEVERE_PARTIAL else None
                    ),
                }
                nested = (
                    generator.generate_nested_in_roi(roi=foreground_np, **nested_kwargs)
                    if args.fraction_domain == "fingerprint_roi"
                    else generator.generate_nested(**nested_kwargs)
                )
                previous_mask = None
                for fraction in family_fractions:
                    key = (row["sample_id"], family.value, f"{fraction:.2f}")
                    mask_np = nested[fraction].mask
                    if previous_mask is not None and np.any(previous_mask > mask_np):
                        raise RuntimeError("nested-mask invariant was violated")
                    previous_mask = mask_np
                    if key in completed:
                        continue
                    observed_np = np.where(mask_np, target_np, 0.0).astype(np.float32)
                    observed = torch.from_numpy(observed_np)[None, None].to(device)
                    mask = torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device)
                    support = support_model(torch.cat((observed, mask), dim=1)) >= 0.5
                    condition_seed = base_seed + 1_000_003 * selected_position + 10_007 * list(MaskFamily).index(family) + int(round(100 * fraction))
                    seed_everything(condition_seed)
                    started = time.perf_counter()
                    samples = model.sample_ddim(
                        observed,
                        mask,
                        inference_steps=args.steps,
                        num_samples=args.samples_k,
                        latent_data_consistency=True,
                        pixel_projection_interval=args.pixel_projection_interval,
                    )[0, :, 0]
                    sampling_seconds += time.perf_counter() - started
                    samples = torch.where(support[0, 0], samples, torch.ones_like(samples))
                    samples = torch.where(mask[0, 0].bool(), observed[0, 0], samples)
                    metrics, _, _ = evaluate_samples(
                        samples, target, mask[0, 0], foreground
                    )
                    record = {
                        "sample_id": row["sample_id"],
                        "subject_id": row["subject_id"],
                        "finger_id": row["finger_id"],
                        "selected_dataset_index": int(row_index),
                        "mask_family": family.value,
                        "observed_fraction": f"{fraction:.2f}",
                        "fraction_domain": args.fraction_domain,
                        "nested_mask_seed": nested_seed,
                        "nested_mask_set_id": nested[fraction].metadata["nested_mask_set_id"],
                        **observed_fraction_report(mask_np, foreground_np),
                        **{f"reinjected_{name}": value for name, value in metrics.items()},
                    }
                    if writer is None:
                        stream = csv_path.open(
                            "a", newline="", encoding="utf-8", buffering=1
                        )
                        writer = csv.DictWriter(stream, fieldnames=list(record))
                        if csv_path.stat().st_size == 0:
                            writer.writeheader()
                    writer.writerow(record)
                    completed.add(key)
                    finished += 1
                    if finished == 1 or finished % 10 == 0:
                        print(
                            f"condition={finished}/{total_conditions} "
                            f"sample={row['sample_id']} family={family.value} r={fraction:.2f} "
                            f"mean_mae={metrics['mean_mae']:.4f}",
                            flush=True,
                        )
    finally:
        if stream is not None:
            stream.close()

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_images": args.num_images,
        "families": [family.value for family in families],
        "fractions": fractions,
        "replicate": args.replicate,
        "nested_masks": True,
        "fraction_domain": args.fraction_domain,
        "num_conditions": finished,
        "samples_k": args.samples_k,
        "steps": args.steps,
        "pixel_projection_interval": args.pixel_projection_interval,
        "sampling_seconds_this_invocation": sampling_seconds,
        "elapsed_seconds_this_invocation": time.perf_counter() - started_all,
    }
    (args.output / "factorial-metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
