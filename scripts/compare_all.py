#!/usr/bin/env python3
"""Compare reference results with the ablation runs stored in outputs/v2.

Table A: deterministic models on all 900 validation cases.
Table B: the 90-case paired subset, adding the probabilistic models (mean of K
samples and single sample) that were only evaluated on that subset.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path("outputs")
V2 = OUT / "v2"


def key(row):
    return (row["sample_id"], row["mask_family"], round(float(row["observed_fraction"]), 2))


def load(path, prefix=None):
    """Return {key: {mae, psnr, ssim, ori}} using deterministic or prefixed probabilistic columns."""
    names = {
        None: ("missing_roi_mae", "missing_roi_psnr", "missing_roi_ssim_map_mean", "missing_roi_orientation_error"),
    }
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    if prefix is None:
        cols = names[None]
    else:
        cols = tuple(f"{prefix}_{m}" for m in ("mae", "psnr", "ssim_map_mean", "orientation_error"))
    out = {}
    for row in rows:
        try:
            out[key(row)] = {n: float(row[c]) for n, c in zip(("mae", "psnr", "ssim", "ori"), cols)}
            out[key(row)].update(
                {n: float(row[c]) for n, c in (("near_ssim", "near_missing_roi_ssim"), ("far_ori", "far_missing_roi_orientation_error"), ("far_ctr", "far_ridge_contrast_ratio")) if c in row and row[c] not in ("", "nan")}
            )
        except (KeyError, ValueError):
            continue
    return out


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def show(title, rows, keys=None):
    print(f"\n{title}")
    print(f"{'model':50s}{'n':>5s}{'MAE':>8s}{'PSNR':>8s}{'SSIM':>8s}{'ORI':>8s}{'nearSSIM':>10s}{'farORI':>8s}{'farCtr':>8s}")
    for label, data in rows:
        ks = [k for k in data if keys is None or k in keys]
        if not ks:
            print(f"{label:50s}{0:>5d}   (no matching cases)")
            continue
        cell = lambda name: mean([data[k][name] for k in ks if name in data[k]])
        extra = lambda name: f"{cell(name):10.3f}" if any(name in data[k] for k in ks) else f"{'-':>10s}"
        print(
            f"{label:50s}{len(ks):5d}{cell('mae'):8.4f}{cell('psnr'):8.2f}{cell('ssim'):8.4f}{cell('ori'):8.4f}"
            f"{extra('near_ssim')}{extra('far_ori')[2:] if False else extra('far_ori')[2:]:>8s}{extra('far_ctr')[2:]:>8s}"
        )


def main() -> None:
    reference_models = [
        ("Reference nearest-observed interpolation", load(OUT / "classical_nearest_validation/per-image-metrics.csv")),
        ("Reference U-Net", load(OUT / "unet_full_mps/per-image-metrics.csv")),
        ("Reference gated conv", load(OUT / "gated_full_mps/per-image-metrics.csv")),
    ]
    ablations = []
    for label, name in (
        ("Ablation: gated re-evaluation", "eval_gated_baseline"),
        ("Ablation: gated + augmentation", "eval_gated_v2_aug_now"),
        ("Ablation: gated + augmentation + ridge losses", "eval_gated_v2_aug_ridge_now"),
        ("Ablation: gated + ridge losses, no augmentation", "eval_gated_ridge_noaug_now"),
        ("Ablation: FFC generator", "eval_ffc_now"),
    ):
        path = V2 / name / "per-image-metrics.csv"
        if path.exists():
            ablations.append((label, load(path)))
    show("A) 900 validation cases, deterministic (missing-ROI metrics)", reference_models + ablations)

    paired_keys = set(load(OUT / "residual_latent_v1_continued_projection5_paired90/paired-per-image.csv", "reinjected_mean"))
    prob = []
    for label, directory, prefixes in (
        ("Latent diffusion v1", "latent_reinjection_paired90", (("plain", "plain"), ("reinjected", "reinjected"))),
        ("Latent diffusion v2", "latent_v2_reinjection_paired90", (("reinjected", "reinjected"),)),
        ("Latent v3 adapter", "latent_v3_adapter_reinjection_paired90", (("reinjected", "reinjected"),)),
        ("Latent v3 adapter + pixel-proj/5", "latent_v3_adapter_pixel_projection5_paired90", (("reinjected", "reinjected"),)),
        ("Residual latent v1 + proj/5", "residual_latent_v1_projection5_paired90", (("reinjected", "reinjected"),)),
        ("Residual latent v1 cont. + proj/5", "residual_latent_v1_continued_projection5_paired90", (("reinjected", "reinjected"),)),
    ):
        path = OUT / directory / "paired-per-image.csv"
        for tag, prefix in prefixes:
            prob.append((f"{label} [{tag}] mean-of-K", load(path, f"{prefix}_mean")))
            prob.append((f"{label} [{tag}] single sample", load(path, f"{prefix}_single")))
    for label, directory in (("Pixel DDIM-50", "ddim50_k5_balanced90"), ("RePaint u2", "repaint_u2_k5_balanced90")):
        path = OUT / directory / "per-image-probabilistic-metrics.csv"
        prob.append((f"{label} mean-of-K", load(path, "mean")))
        prob.append((f"{label} single sample", load(path, "single")))
    show("B) the 90 paired probabilistic cases", reference_models + ablations + prob, keys=paired_keys)


if __name__ == "__main__":
    main()
