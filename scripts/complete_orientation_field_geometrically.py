#!/usr/bin/env python3
"""Is the ridge orientation field recoverable without a neural network?

The project so far has asked a model to invent the missing *pixels*. This asks
a smaller and more answerable question: can the missing *geometry* be filled in
from the geometry around it, using nothing but a smoothness prior?

Orientation is axial, so the field is carried in doubled-angle form

    v(x, y) = (cos 2*theta, sin 2*theta)

which makes theta and theta+pi the same vector and removes the wrap-around
that makes naive angle averaging wrong. Each component is completed by solving
the Laplace equation inside the hidden region with the observed field as
Dirichlet boundary data -- the minimiser of

    integral over hole of |grad v|^2   subject to v = v_obs on the boundary

i.e. exactly the smoothness term of the variational formulation, with no
topology term and no learning. The completed vector is renormalised and the
angle read back as theta = 0.5 * atan2(s, c).

Three arms are compared in the hidden region against the true field:

  nearest    copy the orientation of the closest observed pixel (the floor)
  harmonic   the geometry-only completion described above
  neural     this project's trained reconstruction model, for reference

The observed region is eroded before being used as boundary data, because the
orientation estimator sees the flat grey fill and produces garbage within a
few pixels of the mask edge.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy import ndimage

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.metrics.orientation_geometry import (
    complete_biharmonic,
    complete_combination,
    complete_harmonic,
    complete_mirror_cascade,
    complete_polynomial,
    fit_zero_pole_best,
    zero_pole_theta,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import (
    OrientationField,
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device

CONDITIONS = (
    ("rect", MaskFamily.RANDOM_RECTANGLES, 0.50),
    ("central", MaskFamily.CENTRAL_MISSING, 0.60),
    ("irregular", MaskFamily.IRREGULAR, 0.50),
    ("fragments", MaskFamily.DISCONNECTED_FRAGMENTS, 0.30),
)
ARMS = ("nearest", "harmonic", "biharmonic", "polynomial", "zero_pole", "combination", "mirror", "neural")
MISSING_FILL = 0.72
BOUNDARY_EROSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def harmonic_complete(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Solve Laplace's equation on ~known with Dirichlet data from known."""
    height, width = values.shape
    unknown = ~known
    count = int(unknown.sum())
    if count == 0:
        return values.copy()
    index = -np.ones((height, width), dtype=np.int64)
    index[unknown] = np.arange(count)

    rows, cols, data = [], [], []
    rhs = np.zeros(count, dtype=np.float64)
    ys, xs = np.nonzero(unknown)
    for k in range(count):
        y, x = int(ys[k]), int(xs[k])
        degree = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < height and 0 <= xx < width):
                continue  # zero-flux at the image border
            degree += 1
            if known[yy, xx]:
                rhs[k] += float(values[yy, xx])
            else:
                rows.append(k); cols.append(int(index[yy, xx])); data.append(-1.0)
        rows.append(k); cols.append(k); data.append(float(degree))

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(count, count))
    solution = spla.spsolve(matrix, rhs)
    out = values.astype(np.float64).copy()
    out[unknown] = solution
    return out


def field_from_theta(theta: np.ndarray, coherence: np.ndarray) -> OrientationField:
    return OrientationField(
        theta=theta,
        coherence=coherence,
        valid=np.ones_like(theta, dtype=bool),
    )


def nearest_fill(theta: np.ndarray, known: np.ndarray) -> np.ndarray:
    _, indices = ndimage.distance_transform_edt(~known, return_indices=True)
    return theta[tuple(indices)]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[dict] = []
    for condition_name, family, fraction in CONDITIONS:
        dataset = SocofingPartialDataset(
            manifest_path=args.manifest, image_root=args.image_root, split=args.split,
            output_shape=(128, 128), families=[family], observed_fractions=[fraction],
            base_seed=args.seed,
        )
        dataset.set_epoch(0)
        for index in range(min(args.images, len(dataset))):
            item = dataset[index]
            truth = item["target"][0].numpy()
            mask = item["mask"][0].numpy()
            observed_bool = mask > 0.5
            hidden = ~observed_bool
            if hidden.sum() < 200:
                continue

            true_field = estimate_orientation_field(truth, use_foreground_mask=False)
            masked_image = np.where(observed_bool, truth, MISSING_FILL).astype(np.float32)
            observed_field = estimate_orientation_field(masked_image, use_foreground_mask=False)

            trusted = ndimage.binary_erosion(
                observed_bool, iterations=BOUNDARY_EROSION, border_value=0
            )
            if trusted.sum() < 200:
                continue

            cos_component = np.cos(2.0 * observed_field.theta)
            sin_component = np.sin(2.0 * observed_field.theta)
            harmonic_theta = np.mod(0.5 * np.arctan2(
                complete_harmonic(sin_component, trusted),
                complete_harmonic(cos_component, trusted)), np.pi)
            biharmonic_theta = np.mod(0.5 * np.arctan2(
                complete_biharmonic(sin_component, trusted, damping=1.0),
                complete_biharmonic(cos_component, trusted, damping=1.0)), np.pi)
            weights = observed_field.coherence * observed_field.valid
            polynomial_theta = np.mod(complete_polynomial(
                cos_component, sin_component, trusted, weights, degree=5), np.pi)
            try:
                fit = fit_zero_pole_best(observed_field.theta, trusted, weights)
                zero_pole_theta_map = zero_pole_theta(
                    *observed_field.theta.shape, fit.theta0, fit.cores, fit.deltas)
                combination_theta = complete_combination(
                    observed_field.theta, trusted, weights, fit)
            except ValueError:
                zero_pole_theta_map = harmonic_theta
                combination_theta = harmonic_theta

            try:
                mirror_theta, axis_x, symmetry_gap, _ = complete_mirror_cascade(
                    observed_field.theta, trusted, weights)
            except Exception:
                mirror_theta, axis_x, symmetry_gap = harmonic_theta, float("nan"), float("nan")

            nearest_theta = nearest_fill(observed_field.theta, trusted)

            estimate = model.reconstruct(
                item["observed"][None].to(device), item["mask"][None].to(device)
            )[0, 0].cpu().numpy()
            completed = np.where(observed_bool, truth, estimate).astype(np.float32)
            neural_field = estimate_orientation_field(completed, use_foreground_mask=False)

            row = {
                "condition": condition_name,
                "sample_id": item["sample_id"],
                "hidden_pixels": int(hidden.sum()),
            }
            for arm, field in (
                ("nearest", field_from_theta(nearest_theta, true_field.coherence)),
                ("harmonic", field_from_theta(harmonic_theta, true_field.coherence)),
                ("biharmonic", field_from_theta(biharmonic_theta, true_field.coherence)),
                ("polynomial", field_from_theta(polynomial_theta, true_field.coherence)),
                ("zero_pole", field_from_theta(zero_pole_theta_map, true_field.coherence)),
                ("combination", field_from_theta(combination_theta, true_field.coherence)),
                ("mirror", field_from_theta(mirror_theta, true_field.coherence)),
                ("neural", neural_field),
            ):
                try:
                    row[f"{arm}_orientation_error"] = orientation_error(
                        true_field, field, region_mask=hidden
                    )
                except ValueError:
                    row[f"{arm}_orientation_error"] = float("nan")
            records.append(row)

    if not records:
        raise SystemExit("no records produced")

    summary = {}
    for condition_name, *_ in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition_name]
        if not subset:
            continue
        summary[condition_name] = {
            "images": len(subset),
            **{
                arm: float(np.nanmean([r[f"{arm}_orientation_error"] for r in subset]))
                for arm in ARMS
            },
        }
    summary["ALL"] = {
        "images": len(records),
        **{
            arm: float(np.nanmean([r[f"{arm}_orientation_error"] for r in records]))
            for arm in ARMS
        },
    }

    report = {
        "question": "is the missing orientation field recoverable without learning?",
        "arms": {
            "nearest": "orientation of the closest observed pixel",
            "harmonic": "Laplace completion of (cos 2t, sin 2t), smoothness only, no learning",
            "neural": "this project's trained reconstruction model",
        },
        "metric": "axial orientation error in the hidden region, lower is better",
        "boundary_erosion_px": BOUNDARY_EROSION,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
