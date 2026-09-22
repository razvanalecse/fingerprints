"""PyTorch dataset for confidence-filtered, approximately registered SD302 pairs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, cKDTree
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from fingerprint_reconstruction.data.nist302_torch_dataset import (
    Nist302DatasetError,
    rasterize_ridge_quality,
    resize_label_map,
)
from fingerprint_reconstruction.evaluation.registration import (
    rescale_output_to_input_transform,
    warp_image_output_to_input,
)
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale
from fingerprint_reconstruction.preprocessing.orientation import estimate_foreground_mask


class Nist302RegisteredDataset(Dataset):
    """Return ``(Y, M, X_pseudo)`` without claiming exact pixel ground truth."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        latent_root: Path,
        annotation_root: Path,
        exemplar_roots: Mapping[str, Path],
        split: str,
        output_shape: Tuple[int, int] = (256, 256),
        missing_fill_value: float = 0.0,
        use_exemplar_foreground: bool = True,
        distance_band_threshold_px: float = 8.0,
        geometric_confidence_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise Nist302DatasetError("split must be train, validation, or test")
        with Path(manifest_path).open(newline="", encoding="utf-8") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.rows:
            raise Nist302DatasetError(f"registered manifest has no rows for {split}")
        self.latent_root = Path(latent_root)
        self.annotation_root = Path(annotation_root)
        self.exemplar_roots = {key: Path(value) for key, value in exemplar_roots.items()}
        self.output_shape = tuple(int(value) for value in output_shape)
        self.missing_fill_value = float(missing_fill_value)
        self.use_exemplar_foreground = bool(use_exemplar_foreground)
        self.distance_band_threshold_px = float(distance_band_threshold_px)
        self.geometric_confidence_weights = tuple(
            float(value) for value in geometric_confidence_weights
        )
        if min(self.output_shape) <= 0 or not 0 <= self.missing_fill_value <= 1:
            raise Nist302DatasetError("invalid output shape or missing fill value")
        if self.distance_band_threshold_px <= 0:
            raise Nist302DatasetError("distance band threshold must be positive")
        if (
            len(self.geometric_confidence_weights) != 3
            or any(not 0.0 <= value <= 1.0 for value in self.geometric_confidence_weights)
            or self.geometric_confidence_weights[0] <= 0.0
        ):
            raise Nist302DatasetError(
                "geometric confidence weights must contain three values in [0,1] "
                "with a positive 0-2 mm weight"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        row = self.rows[index]
        part = row["exemplar_dataset_part"]
        if part not in self.exemplar_roots:
            raise Nist302DatasetError(f"no configured root for {part}")
        latent_path = self.latent_root / row["latent_relative_path"]
        exemplar_path = self.exemplar_roots[part] / row["exemplar_relative_path"]
        quality_path = self.annotation_root / row["quality_map_path"]
        latent_native_shape = (int(row["latent_height"]), int(row["latent_width"]))

        latent = load_grayscale(latent_path, output_shape=self.output_shape)
        exemplar_native = load_grayscale(exemplar_path)
        exemplar_shape = self.output_shape
        exemplar = load_grayscale(exemplar_path, output_shape=exemplar_shape)
        native_matrix = np.asarray(
            [
                [float(row[f"pixel_m0_{column}"]) for column in range(3)],
                [float(row[f"pixel_m1_{column}"]) for column in range(3)],
            ],
            dtype=np.float64,
        )
        resized_matrix = rescale_output_to_input_transform(
            native_matrix,
            native_output_shape=latent_native_shape,
            native_input_shape=exemplar_native.shape,
            resized_output_shape=self.output_shape,
            resized_input_shape=exemplar_shape,
        )
        target = warp_image_output_to_input(
            exemplar, resized_matrix, output_shape=self.output_shape, order=1, cval=1.0
        ).astype(np.float32)
        coverage = warp_image_output_to_input(
            np.ones(exemplar_shape, dtype=np.float32),
            resized_matrix,
            output_shape=self.output_shape,
            order=0,
            cval=0.0,
        ) >= 0.5
        if self.use_exemplar_foreground:
            exemplar_support = estimate_foreground_mask(exemplar)
            registered_support = warp_image_output_to_input(
                exemplar_support.astype(np.float32),
                resized_matrix,
                output_shape=self.output_shape,
                order=0,
                cval=0.0,
            ) >= 0.5
        else:
            registered_support = coverage.copy()

        with np.load(quality_path, allow_pickle=False) as archive:
            quality_grid = np.asarray(archive["quality"], dtype=np.uint8)
            grid_size = int(archive["grid_size_0_01mm"])
        quality_native = rasterize_ridge_quality(
            quality_grid,
            image_shape=latent_native_shape,
            ppi=float(row["latent_native_ppi"]),
            grid_size_0_01mm=grid_size,
        )
        quality = resize_label_map(quality_native, self.output_shape)
        mask = quality >= 2
        evaluation_roi = (~mask) & registered_support & coverage
        distance = distance_transform_edt(~mask).astype(np.float32)
        near_roi = evaluation_roi & (distance <= self.distance_band_threshold_px)
        far_roi = evaluation_roi & (distance > self.distance_band_threshold_px)
        anchor_points = np.asarray(
            json.loads(row["latent_correspondences_0_01mm"]), dtype=np.float64
        )
        if anchor_points.ndim != 2 or anchor_points.shape[1] != 2 or len(anchor_points) < 3:
            raise Nist302DatasetError("registered sample has fewer than three anchor points")
        native_scale = float(row["latent_native_ppi"]) / 2540.0
        native_points = anchor_points * native_scale
        resize_scale = np.array(
            [
                int(row["latent_width"]) / self.output_shape[1],
                int(row["latent_height"]) / self.output_shape[0],
            ],
            dtype=np.float64,
        )
        resized_points = (native_points - (resize_scale - 1.0) / 2.0) / resize_scale
        hull = ConvexHull(resized_points)
        hull_mask_image = Image.new(
            "1", (self.output_shape[1], self.output_shape[0]), color=0
        )
        ImageDraw.Draw(hull_mask_image).polygon(
            [tuple(point) for point in resized_points[hull.vertices]], fill=1
        )
        hull_mask = np.asarray(hull_mask_image, dtype=bool)
        millimeters_per_pixel = (
            25.4
            / float(row["latent_native_ppi"])
            * np.array(
                [
                    int(row["latent_height"]) / self.output_shape[0],
                    int(row["latent_width"]) / self.output_shape[1],
                ]
            )
        )
        anchor_distance_mm = distance_transform_edt(
            ~hull_mask, sampling=tuple(millimeters_per_pixel)
        ).astype(np.float32)
        anchor_inside = evaluation_roi & hull_mask
        anchor_0_2 = evaluation_roi & (~hull_mask) & (anchor_distance_mm <= 2.0)
        anchor_2_5 = evaluation_roi & (anchor_distance_mm > 2.0) & (anchor_distance_mm <= 5.0)
        anchor_gt_5 = evaluation_roi & (anchor_distance_mm > 5.0)
        y_native = (
            (np.arange(self.output_shape[0], dtype=np.float64) + 0.5)
            * resize_scale[1]
            - 0.5
        )
        x_native = (
            (np.arange(self.output_shape[1], dtype=np.float64) + 0.5)
            * resize_scale[0]
            - 0.5
        )
        grid_y, grid_x = np.meshgrid(y_native, x_native, indexing="ij")
        grid_efs_mm = np.column_stack(
            (
                grid_x.reshape(-1) / native_scale / 100.0,
                grid_y.reshape(-1) / native_scale / 100.0,
            )
        )
        nearest_anchor_mm = cKDTree(anchor_points / 100.0).query(
            grid_efs_mm, workers=1
        )[0].reshape(self.output_shape).astype(np.float32)
        nearest_0_2 = evaluation_roi & (nearest_anchor_mm <= 2.0)
        nearest_2_5 = evaluation_roi & (nearest_anchor_mm > 2.0) & (nearest_anchor_mm <= 5.0)
        nearest_gt_5 = evaluation_roi & (nearest_anchor_mm > 5.0)
        near_weight, middle_weight, far_weight = self.geometric_confidence_weights
        geometric_confidence = (
            near_weight * nearest_0_2.astype(np.float32)
            + middle_weight * nearest_2_5.astype(np.float32)
            + far_weight * nearest_gt_5.astype(np.float32)
        )

        latent_tensor = torch.from_numpy(latent).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        observed = torch.where(
            mask_tensor.bool(),
            latent_tensor,
            torch.full_like(latent_tensor, self.missing_fill_value),
        )
        return {
            "conditioning": torch.cat((observed, mask_tensor), dim=0),
            "observed": observed,
            "mask": mask_tensor,
            "target": target_tensor,
            "latent_image": latent_tensor,
            "quality": torch.from_numpy(quality.astype(np.int64)).unsqueeze(0),
            "warp_coverage": torch.from_numpy(coverage.astype(np.float32)).unsqueeze(0),
            "registered_support": torch.from_numpy(
                registered_support.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi": torch.from_numpy(
                evaluation_roi.astype(np.float32)
            ).unsqueeze(0),
            "distance_to_observed": torch.from_numpy(distance).unsqueeze(0),
            "evaluation_roi_near": torch.from_numpy(near_roi.astype(np.float32)).unsqueeze(0),
            "evaluation_roi_far": torch.from_numpy(far_roi.astype(np.float32)).unsqueeze(0),
            "distance_band_threshold_px": torch.tensor(self.distance_band_threshold_px),
            "correspondence_hull": torch.from_numpy(hull_mask.astype(np.float32)).unsqueeze(0),
            "distance_to_correspondence_hull_mm": torch.from_numpy(
                anchor_distance_mm
            ).unsqueeze(0),
            "evaluation_roi_anchor_inside": torch.from_numpy(
                anchor_inside.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi_anchor_0_2mm": torch.from_numpy(
                anchor_0_2.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi_anchor_2_5mm": torch.from_numpy(
                anchor_2_5.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi_anchor_gt_5mm": torch.from_numpy(
                anchor_gt_5.astype(np.float32)
            ).unsqueeze(0),
            "distance_to_nearest_correspondence_mm": torch.from_numpy(
                nearest_anchor_mm
            ).unsqueeze(0),
            "evaluation_roi_nearest_0_2mm": torch.from_numpy(
                nearest_0_2.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi_nearest_2_5mm": torch.from_numpy(
                nearest_2_5.astype(np.float32)
            ).unsqueeze(0),
            "evaluation_roi_nearest_gt_5mm": torch.from_numpy(
                nearest_gt_5.astype(np.float32)
            ).unsqueeze(0),
            "geometric_confidence": torch.from_numpy(geometric_confidence).unsqueeze(0),
            "geometric_confidence_stratum_weights": torch.tensor(
                self.geometric_confidence_weights, dtype=torch.float32
            ),
            "sample_id": "nist302-registered:"
            + (
                Path(row["relative_path"]).with_suffix("").as_posix()
                if row.get("relative_path")
                else f"{row['latent_sample_id']}::{row['exemplar_relative_path']}"
            ),
            "latent_sample_id": row["latent_sample_id"],
            "subject_id": row["subject_id"],
            "registration_status": "registered_approximate",
            "population_label": "correspondence-qualified latent subset",
            "pixel_aligned_ground_truth": False,
            "num_correspondences": torch.tensor(
                int(row["num_correspondences"]), dtype=torch.int64
            ),
            "affine_rmse_mm": torch.tensor(float(row["affine_rmse_mm"])),
            "minimum_hull_coverage": torch.tensor(
                float(row["minimum_hull_coverage"])
            ),
        }
