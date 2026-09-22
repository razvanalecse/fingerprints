"""Per-image evaluation with explicit observed/missing region separation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from scipy.stats import t

from fingerprint_reconstruction.metrics.image_metrics import region_image_metrics
from fingerprint_reconstruction.metrics.ridge_frequency import paired_ridge_frequency_error
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask,
    estimate_orientation_field,
    orientation_error,
)


def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> List[Dict[str, object]]:
    model.eval()
    records: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            reconstruction = model.reconstruct(observed, mask)
            target_numpy = target[:, 0].cpu().numpy()
            reconstruction_numpy = reconstruction[:, 0].cpu().numpy()
            mask_numpy = mask[:, 0].cpu().numpy().astype(bool)
            fractions = batch["observed_fraction"].cpu().numpy()
            for index in range(target_numpy.shape[0]):
                record: Dict[str, object] = {
                    "sample_id": batch["sample_id"][index],
                    "mask_family": batch["mask_family"][index],
                    "observed_fraction": float(fractions[index]),
                }
                fingerprint_roi = estimate_foreground_mask(target_numpy[index])
                reference_orientation = estimate_orientation_field(target_numpy[index])
                estimate_orientation = estimate_orientation_field(
                    reconstruction_numpy[index], use_foreground_mask=False
                )
                missing_roi = (~mask_numpy[index]) & fingerprint_roi
                orientation_support = (
                    reference_orientation.valid & estimate_orientation.valid & missing_roi
                )
                record["missing_roi_orientation_valid_pixels"] = float(
                    orientation_support.sum()
                )
                try:
                    record["missing_roi_orientation_error"] = orientation_error(
                        reference_orientation,
                        estimate_orientation,
                        region_mask=missing_roi,
                    )
                except ValueError:
                    record["missing_roi_orientation_error"] = float("nan")
                for name, region in (
                    ("missing", ~mask_numpy[index]),
                    ("observed", mask_numpy[index]),
                    ("missing_roi", (~mask_numpy[index]) & fingerprint_roi),
                    ("observed_roi", mask_numpy[index] & fingerprint_roi),
                ):
                    if region.any():
                        metrics = region_image_metrics(
                            target_numpy[index], reconstruction_numpy[index], region
                        )
                    else:
                        metrics = {
                            "mse": float("nan"),
                            "mae": float("nan"),
                            "psnr": float("nan"),
                            "ssim_map_mean": float("nan"),
                            "pixels": 0.0,
                            "max_absolute_error": float("nan"),
                        }
                    record.update({f"{name}_{key}": value for key, value in metrics.items()})
                records.append(record)
    if not records:
        raise ValueError("evaluation loader produced no records")
    return records


def summarize_records(records: List[Mapping[str, object]]) -> Dict[str, object]:
    """Return mean, SD, and t-based 95% CI for each finite numeric metric."""

    metric_names = sorted(
        key
        for key in records[0]
        if all(isinstance(record.get(key), (int, float, np.number)) for record in records)
    )
    summary: Dict[str, object] = {"num_images": len(records), "metrics": {}}
    for name in metric_names:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        metric_summary: Dict[str, float] = {
            "finite_count": float(finite.size),
            "infinite_count": float(np.isinf(values).sum()),
        }
        if finite.size:
            mean = float(finite.mean())
            standard_deviation = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            half_width = (
                float(t.ppf(0.975, finite.size - 1) * standard_deviation / np.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
            metric_summary.update(
                {
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                }
            )
        summary["metrics"][name] = metric_summary
    return summary


def evaluate_registered_model(
    *,
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    max_batches: Optional[int] = None,
    include_ridge_frequency: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate on approximate SD302 targets with predeclared reliability strata."""

    model.eval()
    records: List[Dict[str, object]] = []
    region_keys = {
        "evaluation": "evaluation_roi",
        "observed_near": "evaluation_roi_near",
        "observed_far": "evaluation_roi_far",
        "hull_inside": "evaluation_roi_anchor_inside",
        "hull_0_2mm": "evaluation_roi_anchor_0_2mm",
        "hull_2_5mm": "evaluation_roi_anchor_2_5mm",
        "hull_gt_5mm": "evaluation_roi_anchor_gt_5mm",
        "nearest_0_2mm": "evaluation_roi_nearest_0_2mm",
        "nearest_2_5mm": "evaluation_roi_nearest_2_5mm",
        "nearest_gt_5mm": "evaluation_roi_nearest_gt_5mm",
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            reconstruction = model.reconstruct(observed, mask)
            target_np = target[:, 0].cpu().numpy()
            reconstruction_np = reconstruction[:, 0].cpu().numpy()
            for index in range(target_np.shape[0]):
                reference_orientation = estimate_orientation_field(
                    target_np[index], use_foreground_mask=False
                )
                estimated_orientation = estimate_orientation_field(
                    reconstruction_np[index], use_foreground_mask=False
                )
                record: Dict[str, object] = {
                    "sample_id": batch["sample_id"][index],
                    "subject_id": batch["subject_id"][index],
                    "registration_status": batch["registration_status"][index],
                    "population_label": batch["population_label"][index],
                    "num_correspondences": int(batch["num_correspondences"][index]),
                    "affine_rmse_mm": float(batch["affine_rmse_mm"][index]),
                    "minimum_hull_coverage": float(
                        batch["minimum_hull_coverage"][index]
                    ),
                    "observed_fraction_canvas": float(mask[index].mean()),
                }
                for label, key in region_keys.items():
                    region = batch[key][index, 0].cpu().numpy().astype(bool)
                    if region.any():
                        image_metrics = region_image_metrics(
                            target_np[index], reconstruction_np[index], region
                        )
                        valid_orientation = (
                            region
                            & reference_orientation.valid
                            & estimated_orientation.valid
                        )
                        try:
                            orientation_value = orientation_error(
                                reference_orientation,
                                estimated_orientation,
                                region_mask=region,
                            )
                        except ValueError:
                            orientation_value = float("nan")
                    else:
                        image_metrics = {
                            "mse": float("nan"),
                            "mae": float("nan"),
                            "psnr": float("nan"),
                            "ssim_map_mean": float("nan"),
                            "pixels": 0.0,
                            "max_absolute_error": float("nan"),
                        }
                        valid_orientation = np.zeros(region.shape, dtype=bool)
                        orientation_value = float("nan")
                    record.update(
                        {f"{label}_{name}": value for name, value in image_metrics.items()}
                    )
                    record[f"{label}_orientation_error"] = orientation_value
                    record[f"{label}_orientation_valid_pixels"] = float(
                        valid_orientation.sum()
                    )
                if include_ridge_frequency:
                    full_region = batch["evaluation_roi"][index, 0].cpu().numpy().astype(bool)
                    frequency = paired_ridge_frequency_error(
                        target_np[index], reconstruction_np[index], full_region
                    )
                    record.update({f"evaluation_{key}": value for key, value in frequency.items()})
                records.append(record)
    if not records:
        raise ValueError("registered evaluation loader produced no records")
    return records


def stratified_summaries(records: List[Mapping[str, object]]) -> Dict[str, object]:
    """Summarize overall performance and the pre-specified mask strata."""

    result: Dict[str, object] = {"overall": summarize_records(records)}
    for key in ("observed_fraction", "mask_family"):
        groups: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            value = record[key]
            label = f"{float(value):.2f}" if key == "observed_fraction" else str(value)
            groups[label].append(record)
        result[f"by_{key}"] = {
            label: summarize_records(group) for label, group in sorted(groups.items())
        }
    return result
