"""Reproducible SOCOFing exploratory analysis and diagnostic figures."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import label

from fingerprint_reconstruction.data.partial_pairs import build_partial_pair
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    suppress_background,
)


@dataclass(frozen=True)
class SocofingEdaSummary:
    num_images: int
    num_subjects: int
    num_fingers: int
    split_images: Mapping[str, int]
    split_subjects: Mapping[str, int]
    gender_images: Mapping[str, int]
    hand_images: Mapping[str, int]
    image_shapes: Mapping[str, int]
    source_modes: Mapping[str, int]
    intensity_mean_quantiles: Mapping[str, float]
    intensity_std_quantiles: Mapping[str, float]


def _count(values: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "sample_id",
        "relative_path",
        "subject_id",
        "finger_id",
        "gender",
        "hand",
        "width",
        "height",
        "source_mode",
        "grayscale_mean",
        "grayscale_std",
        "split",
    }
    missing = required - set(rows[0] if rows else ())
    if not rows:
        raise ValueError("manifest is empty")
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    return rows


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    probabilities = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
    return {f"q{int(probability * 100):02d}": float(value) for probability, value in zip(probabilities, result)}


def summarize_manifest(rows: Sequence[Mapping[str, str]]) -> SocofingEdaSummary:
    if not rows:
        raise ValueError("rows cannot be empty")
    subjects = {row["subject_id"] for row in rows}
    fingers = {(row["subject_id"], row["finger_id"]) for row in rows}
    split_subject_pairs = {(row["split"], row["subject_id"]) for row in rows}
    return SocofingEdaSummary(
        num_images=len(rows),
        num_subjects=len(subjects),
        num_fingers=len(fingers),
        split_images=_count([row["split"] for row in rows]),
        split_subjects=_count([split for split, _ in split_subject_pairs]),
        gender_images=_count([row["gender"] for row in rows]),
        hand_images=_count([row["hand"] for row in rows]),
        image_shapes=_count([f"{row['height']}x{row['width']}" for row in rows]),
        source_modes=_count([row["source_mode"] for row in rows]),
        intensity_mean_quantiles=_quantiles([float(row["grayscale_mean"]) / 255.0 for row in rows]),
        intensity_std_quantiles=_quantiles([float(row["grayscale_std"]) / 255.0 for row in rows]),
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_dataset_distributions(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    means = np.asarray([float(row["grayscale_mean"]) / 255.0 for row in rows])
    stds = np.asarray([float(row["grayscale_std"]) / 255.0 for row in rows])
    shapes = _count([f"{row['height']}x{row['width']}" for row in rows])
    splits = _count([row["split"] for row in rows])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(means, bins=40, color="#3366aa", alpha=0.9)
    axes[0, 0].set(title="Mean grayscale intensity", xlabel="Mean intensity [0,1]", ylabel="Images")
    axes[0, 1].hist(stds, bins=40, color="#aa6633", alpha=0.9)
    axes[0, 1].set(title="Within-image contrast", xlabel="Intensity standard deviation", ylabel="Images")

    axes[1, 0].bar(list(shapes), list(shapes.values()), color="#447744")
    axes[1, 0].set(title="Decoded image shapes", xlabel="Height × width", ylabel="Images")
    for tick in axes[1, 0].get_xticklabels():
        tick.set_rotation(20)

    order = [name for name in ("train", "validation", "test") if name in splits]
    axes[1, 1].bar(order, [splits[name] for name in order], color="#7755aa")
    axes[1, 1].set(title="Leakage-safe image split", xlabel="Split", ylabel="Images")
    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("SOCOFing original-image audit")
    figure.tight_layout()
    _save_figure(figure, output)


def plot_mask_families(image: np.ndarray, sample_id: str, output: Path, observed_fraction: float = 0.30) -> None:
    families = list(MaskFamily)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for axis, family in zip(axes[1:], families):
        ratio = min(observed_fraction, 0.30) if family == MaskFamily.SEVERE_PARTIAL else observed_fraction
        fragments = 4 if family == MaskFamily.DISCONNECTED_FRAGMENTS else None
        pair = build_partial_pair(
            image,
            sample_id=sample_id,
            family=family,
            observed_fraction=ratio,
            replicate=0,
            num_fragments=fragments,
        )
        axis.imshow(pair.observed, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{family.value}\nr={pair.metadata['actual_observed_fraction']:.3f}")
        axis.axis("off")
    figure.suptitle("Controlled partial-observation families")
    figure.tight_layout()
    _save_figure(figure, output)


def plot_orientation_diagnostic(image: np.ndarray, output: Path) -> Dict[str, float]:
    field = estimate_orientation_field(image)
    foreground = field.foreground if field.foreground is not None else field.valid
    cleaned_image = suppress_background(image, foreground)
    window = max(12, min(image.shape) // 8)
    stride = max(4, min(image.shape) // 21)
    segments: List[List[Tuple[float, float]]] = []
    accepted_blocks = 0
    candidate_blocks = 0
    half_length = 0.46 * stride
    half_window = window // 2
    for cy in range(half_window, image.shape[0] - half_window + 1, stride):
        for cx in range(half_window, image.shape[1] - half_window + 1, stride):
            y0, y1 = cy - half_window, cy + half_window
            x0, x1 = cx - half_window, cx + half_window
            block_foreground = foreground[y0:y1, x0:x1]
            if not foreground[cy, cx] or float(block_foreground.mean()) < 0.75:
                continue
            candidate_blocks += 1
            block_valid = field.valid[y0:y1, x0:x1]
            if float(block_valid.mean()) < 0.65:
                continue
            weights = field.coherence[y0:y1, x0:x1] * block_valid
            denominator = float(weights.sum())
            if denominator <= 0:
                continue
            angles = field.theta[y0:y1, x0:x1]
            c = float(np.sum(weights * np.cos(2.0 * angles)) / denominator)
            s = float(np.sum(weights * np.sin(2.0 * angles)) / denominator)
            concentration = float(np.hypot(c, s))
            if concentration < 0.25:
                continue
            theta = 0.5 * np.arctan2(s, c)
            dx, dy = half_length * np.cos(theta), half_length * np.sin(theta)
            segments.append([(cx - dx, cy - dy), (cx + dx, cy + dy)])
            accepted_blocks += 1

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cleaned_image, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(foreground.astype(float), levels=[0.5], colors="#00a6d6", linewidths=0.8)
    axes[0].set_title("Fingerprint ROI envelope")
    axes[0].axis("off")

    axes[1].imshow(cleaned_image, cmap="gray", vmin=0, vmax=1)
    if segments:
        collection = LineCollection(
            segments,
            colors="#d62728",
            linewidths=1.25,
            alpha=0.90,
        )
        axes[1].add_collection(collection)
    axes[1].set_title(f"Axial ridge tangents: window={window}, stride={stride}")
    axes[1].axis("off")

    masked_coherence = np.ma.masked_where(~foreground, field.coherence)
    heatmap = axes[2].imshow(masked_coherence, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Coherence inside foreground only")
    axes[2].axis("off")
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, label="Local coherence")
    figure.suptitle("Orientation-field quality control")
    figure.tight_layout()
    _save_figure(figure, output)
    return {
        "foreground_fraction": float(foreground.mean()),
        "valid_orientation_fraction": float(field.valid.mean()),
        "mean_valid_coherence": float(field.coherence[field.valid].mean()),
        "orientation_blocks_candidate": candidate_blocks,
        "orientation_blocks_displayed": accepted_blocks,
        "orientation_display_window": window,
        "orientation_display_stride": stride,
    }


def plot_orientation_batch_audit(
    rows: Sequence[Mapping[str, str]],
    image_root: Path,
    output: Path,
    *,
    samples_per_stratum: int = 3,
) -> Dict[str, object]:
    """Visually audit foreground masks across gender, hand, and source mode."""

    strata: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        key = (row["gender"], row["hand"], row["source_mode"])
        strata.setdefault(key, []).append(row)

    selected: List[Mapping[str, str]] = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda item: item["sample_id"])
        count = min(samples_per_stratum, len(candidates))
        indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
        selected.extend(candidates[index] for index in indices)

    columns = 6
    rows_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(15, 2.55 * rows_count))
    axes_array = np.atleast_1d(axes).ravel()
    foreground_fractions: List[float] = []
    valid_fractions: List[float] = []
    mean_coherences: List[float] = []
    for axis, record in zip(axes_array, selected):
        image = load_grayscale(Path(image_root) / record["relative_path"], output_shape=(128, 128))
        field = estimate_orientation_field(image)
        foreground = field.foreground if field.foreground is not None else field.valid
        cleaned_image = suppress_background(image, foreground)
        foreground_fractions.append(float(foreground.mean()))
        valid_fractions.append(float(field.valid.mean()))
        mean_coherences.append(float(field.coherence[field.valid].mean()))
        axis.imshow(cleaned_image, cmap="gray", vmin=0, vmax=1)
        if foreground.any() and not foreground.all():
            axis.contour(foreground.astype(float), levels=[0.5], colors="#00a6d6", linewidths=0.7)
        axis.set_title(
            f"{record['subject_id']} {record['hand'][0]}-{record['finger_id']}\n"
            f"{record['source_mode']}, fg={foreground.mean():.2f}",
            fontsize=7,
        )
        axis.axis("off")
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Stratified fingerprint-ROI audit (cyan contour)")
    figure.tight_layout()
    _save_figure(figure, output)
    return {
        "num_images": len(selected),
        "strata": len(strata),
        "foreground_fraction_quantiles": _quantiles(foreground_fractions),
        "valid_orientation_fraction_quantiles": _quantiles(valid_fractions),
        "mean_valid_coherence_quantiles": _quantiles(mean_coherences),
    }


def _mask_geometry(mask: np.ndarray) -> Tuple[float, int]:
    """Return normalized boundary length and observed-component count."""

    binary = np.asarray(mask, dtype=bool)
    transitions = np.count_nonzero(binary[:, 1:] != binary[:, :-1])
    transitions += np.count_nonzero(binary[1:, :] != binary[:-1, :])
    boundary_density = float(transitions / (2.0 * binary.size))
    _, components = label(binary)
    return boundary_density, int(components)


def plot_mask_geometry_diagnostic(image: np.ndarray, sample_id: str, output: Path) -> Dict[str, float]:
    ratios = np.arange(0.1, 0.81, 0.1)
    replicates = 20
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    maximum_area_error = 0.0
    for family in MaskFamily:
        family_ratios = ratios[ratios <= 0.3] if family == MaskFamily.SEVERE_PARTIAL else ratios
        boundary_means, boundary_ci = [], []
        component_means, component_ci = [], []
        for ratio in family_ratios:
            geometry = []
            for replicate in range(replicates):
                fragments = 4 if family == MaskFamily.DISCONNECTED_FRAGMENTS else None
                pair = build_partial_pair(
                    image,
                    sample_id=sample_id,
                    family=family,
                    observed_fraction=float(ratio),
                    replicate=replicate,
                    num_fragments=fragments,
                )
                maximum_area_error = max(
                    maximum_area_error,
                    abs(float(pair.metadata["actual_observed_fraction"]) - float(ratio)),
                )
                geometry.append(_mask_geometry(pair.mask))
            values = np.asarray(geometry, dtype=np.float64)
            boundary_means.append(float(values[:, 0].mean()))
            component_means.append(float(values[:, 1].mean()))
            scale = 1.96 / np.sqrt(replicates)
            boundary_ci.append(float(scale * values[:, 0].std(ddof=1)))
            component_ci.append(float(scale * values[:, 1].std(ddof=1)))
        axes[0].errorbar(
            family_ratios, boundary_means, yerr=boundary_ci,
            marker="o", markersize=3, capsize=2, linewidth=1.2, label=family.value,
        )
        axes[1].errorbar(
            family_ratios, component_means, yerr=component_ci,
            marker="o", markersize=3, capsize=2, linewidth=1.2, label=family.value,
        )
    axes[0].set(
        title="Mask boundary complexity",
        xlabel="Observed fraction r",
        ylabel="Boundary transitions / (2HW)",
    )
    axes[1].set(
        title="Observed-region fragmentation",
        xlabel="Observed fraction r",
        ylabel="Connected components",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(
        f"Mask geometry (mean ± 95% CI, n={replicates}); max |r_real−r|={maximum_area_error:.2e}"
    )
    figure.tight_layout()
    _save_figure(figure, output)
    return {"maximum_observed_fraction_error": maximum_area_error, "replicates": replicates}


def run_socofing_eda(
    *,
    manifest_path: Path,
    image_root: Path,
    output_directory: Path,
    sample_id: str | None = None,
) -> Dict[str, object]:
    rows = read_manifest(manifest_path)
    summary = summarize_manifest(rows)
    selected = next((row for row in rows if row["sample_id"] == sample_id), rows[0])
    image = load_grayscale(Path(image_root) / selected["relative_path"], output_shape=(128, 128))

    output_directory.mkdir(parents=True, exist_ok=True)
    plot_dataset_distributions(rows, output_directory / "socofing-distributions.png")
    plot_mask_families(image, selected["sample_id"], output_directory / "mask-families-r030.png")
    orientation = plot_orientation_diagnostic(image, output_directory / "orientation-diagnostic.png")
    orientation_batch = plot_orientation_batch_audit(
        rows,
        image_root,
        output_directory / "orientation-mask-batch-audit.png",
    )
    mask_geometry = plot_mask_geometry_diagnostic(
        image, selected["sample_id"], output_directory / "mask-geometry-diagnostics.png"
    )

    report: Dict[str, object] = {
        "summary": asdict(summary),
        "representative_sample": selected["sample_id"],
        "orientation_diagnostic": orientation,
        "orientation_batch_audit": orientation_batch,
        "mask_geometry_diagnostic": mask_geometry,
    }
    with (output_directory / "eda-summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report
