#!/usr/bin/env python3
"""Prepare a frozen-filter approximate-registration manifest for SD302."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exemplars", type=Path, required=True)
    parser.add_argument("--comp-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-correspondences", type=int, default=6)
    parser.add_argument("--maximum-affine-rmse-mm", type=float, default=0.30)
    parser.add_argument("--minimum-hull-coverage", type=float, default=0.01)
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    diagnostics = _read(args.diagnostics)
    manifest_rows = _read(args.latent_manifest)
    annotations = _read(args.annotations)
    exemplars = _read(args.exemplars)
    comp_annotations = _read(args.comp_annotations)

    # The physical-to-pixel transform below handles offsets algebraically, but
    # the current diagnostic CSV predates storing them per COMP. Verify the
    # official extraction globally before using the documented zero offsets.
    nonzero_offsets = [
        row
        for row in comp_annotations
        if row["roi_horizontal_offset_0_01mm"] != "0"
        or row["roi_vertical_offset_0_01mm"] != "0"
    ]
    if nonzero_offsets:
        raise ValueError("COMP ROI offsets are not zero; regenerate diagnostics with offsets")

    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    annotation_by_lffs = {Path(row["relative_path"]).name: row for row in annotations}
    exemplar_by_stem = {}
    for row in exemplars:
        stem = Path(row["relative_path"]).stem
        if stem in exemplar_by_stem:
            raise ValueError(f"duplicate exemplar stem: {stem}")
        exemplar_by_stem[stem] = row

    accepted = []
    rejection_counts: Counter[str] = Counter()
    for row in diagnostics:
        if row["determination"] != "INDIV":
            rejection_counts["determination_not_indiv"] += 1
            continue
        if int(row["num_correspondences"]) < args.minimum_correspondences:
            rejection_counts["too_few_correspondences"] += 1
            continue
        if not row["affine_rmse_mm"] or float(row["affine_rmse_mm"]) > args.maximum_affine_rmse_mm:
            rejection_counts["affine_residual"] += 1
            continue
        if float(row["minimum_hull_coverage"]) < args.minimum_hull_coverage:
            rejection_counts["hull_coverage"] += 1
            continue

        latent = manifest_by_id[row["latent_sample_id"]]
        if latent["errata_mentioned"].lower() == "true":
            rejection_counts["errata"] += 1
            continue
        annotation = annotation_by_lffs.get(row["lffs_filename"])
        if annotation is None or not annotation.get("quality_map_path"):
            rejection_counts["missing_quality_map"] += 1
            continue
        exemplar = exemplar_by_stem.get(Path(row["irr_filename"]).stem)
        if exemplar is None:
            rejection_counts["missing_exemplar_png"] += 1
            continue
        if not exemplar["ppi"]:
            rejection_counts["missing_exemplar_ppi"] += 1
            continue

        source_scale = float(latent["native_ppi"]) / 2540.0
        target_scale = float(exemplar["ppi"]) / 2540.0
        physical = np.asarray(
            [
                [float(row[f"affine_m0_{column}"]) for column in range(3)],
                [float(row[f"affine_m1_{column}"]) for column in range(3)],
            ]
        )
        pixel = physical.copy()
        pixel[:, :2] *= target_scale / source_scale
        pixel[:, 2] *= target_scale
        accepted.append(
            {
                **row,
                "latent_relative_path": latent["original_masked_path"],
                "latent_native_ppi": latent["native_ppi"],
                "latent_width": latent["width"],
                "latent_height": latent["height"],
                "quality_map_path": annotation["quality_map_path"],
                "quality_grid_size_0_01mm": annotation["quality_grid_size_0_01mm"],
                "exemplar_dataset_part": exemplar["dataset_part"],
                "exemplar_relative_path": exemplar["relative_path"],
                "exemplar_ppi": exemplar["ppi"],
                **{
                    f"pixel_m{matrix_row}_{column}": pixel[matrix_row, column]
                    for matrix_row in range(2)
                    for column in range(3)
                },
                "registration_status": "registered_approximate",
                "pixel_aligned_ground_truth": False,
            }
        )

    if not accepted:
        raise ValueError("frozen registration filter accepted no rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "registered_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)
    audit = {
        "schema_version": 1,
        "selection_split": "validation",
        "frozen_filter": {
            "determination": "INDIV",
            "minimum_correspondences": args.minimum_correspondences,
            "maximum_affine_rmse_mm": args.maximum_affine_rmse_mm,
            "minimum_hull_coverage": args.minimum_hull_coverage,
        },
        "num_diagnostics": len(diagnostics),
        "num_accepted": len(accepted),
        "accepted_by_split": dict(sorted(Counter(row["split"] for row in accepted).items())),
        "rejection_counts_first_failed_rule": dict(sorted(rejection_counts.items())),
        "target_semantics": "registered_approximate; not pixel-aligned ground truth",
        "comp_roi_offsets_verified_zero": True,
    }
    (args.output_dir / "registered_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
