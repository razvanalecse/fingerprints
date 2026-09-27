#!/usr/bin/env python3
"""Audit SD302i official correspondences before any image warping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from fingerprint_reconstruction.datasets.nist302_ebts import (
    decode_transaction,
    match_correspondences,
)
from fingerprint_reconstruction.evaluation.registration import (
    RegistrationError,
    fit_affine_transform,
    fit_similarity_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-root", type=Path, required=True)
    parser.add_argument("--an2k2txt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _correspondence_coverage(points: np.ndarray, roi) -> float:
    if len(points) < 3:
        return 0.0
    polygon = np.asarray(roi.polygon_0_01mm, dtype=np.float64)
    roi_area = (
        _polygon_area(polygon)
        if len(polygon) >= 3
        else float(roi.width_0_01mm * roi.height_0_01mm)
    )
    if roi_area <= 0:
        return 0.0
    try:
        hull_area = float(ConvexHull(points).volume)
    except QhullError:
        return 0.0
    return min(1.0, hull_area / roi_area)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    by_lffs = {}
    for manifest_row in manifest_rows:
        for filename in json.loads(manifest_row["lffs_filenames"]):
            if filename in by_lffs:
                raise ValueError(f"duplicate LFFS mapping in manifest: {filename}")
            by_lffs[filename] = manifest_row
    paths = sorted(args.comp_root.rglob("*.comp"))
    if args.max_files is not None:
        paths = paths[: args.max_files]
    rows = []
    failures = []
    for path in paths:
        try:
            transaction = decode_transaction(path, an2k2txt=args.an2k2txt)
            records = sorted(transaction.efs_records, key=lambda record: int(record.idc))
            if len(records) != 2:
                raise RegistrationError(f"expected two EFS records, found {len(records)}")
            matches = match_correspondences(
                transaction, records[0].record_index, records[1].record_index
            )
            source = np.asarray(
                [[match.first.x_0_01mm, match.first.y_0_01mm] for match in matches],
                dtype=np.float64,
            )
            target = np.asarray(
                [[match.second.x_0_01mm, match.second.y_0_01mm] for match in matches],
                dtype=np.float64,
            )
            similarity = fit_similarity_transform(source, target)
            affine = fit_affine_transform(source, target) if len(matches) >= 3 else None
            sources = {source.source_index: source for source in transaction.comp_sources}
            comparisons = transaction.examiner_comparisons.get(records[1].record_index, ())
            lffs_filename = sources[1].filename
            manifest_row = by_lffs.get(lffs_filename)
            if manifest_row is None:
                raise RegistrationError(f"LFFS missing from latent manifest: {lffs_filename}")
            latent_coverage = _correspondence_coverage(source, records[0].roi)
            exemplar_coverage = _correspondence_coverage(target, records[1].roi)
            rows.append(
                {
                    "relative_path": path.relative_to(args.comp_root).as_posix(),
                    "latent_sample_id": manifest_row["sample_id"],
                    "subject_id": manifest_row["subject_id"],
                    "split": manifest_row["split"],
                    "lffs_filename": lffs_filename,
                    "irr_filename": sources[2].filename,
                    "assessment": records[0].assessment or "",
                    "determination": comparisons[0].determination if comparisons else "",
                    "num_correspondences": len(matches),
                    "latent_hull_coverage": latent_coverage,
                    "exemplar_hull_coverage": exemplar_coverage,
                    "minimum_hull_coverage": min(latent_coverage, exemplar_coverage),
                    "latent_correspondences_0_01mm": json.dumps(source.tolist(), separators=(",", ":")),
                    "exemplar_correspondences_0_01mm": json.dumps(target.tolist(), separators=(",", ":")),
                    "similarity_rmse_mm": similarity.rmse_mm,
                    "similarity_median_mm": similarity.median_mm,
                    "similarity_p95_mm": similarity.p95_mm,
                    "affine_rmse_mm": affine.rmse_mm if affine else "",
                    "affine_median_mm": affine.median_mm if affine else "",
                    "affine_p95_mm": affine.p95_mm if affine else "",
                    **{
                        f"similarity_m{row}_{column}": similarity.matrix[row, column]
                        for row in range(2)
                        for column in range(3)
                    },
                    **(
                        {
                            f"affine_m{row}_{column}": affine.matrix[row, column]
                            for row in range(2)
                            for column in range(3)
                        }
                        if affine
                        else {
                            f"affine_m{row}_{column}": ""
                            for row in range(2)
                            for column in range(3)
                        }
                    ),
                }
            )
        except Exception as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "registration_diagnostics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "num_files_selected": len(paths),
        "num_successful": len(rows),
        "num_failures": len(failures),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "determination_counts": dict(
            sorted(Counter(row["determination"] or "MISSING" for row in rows).items())
        ),
        "correspondence_count": _summary(
            [float(row["num_correspondences"]) for row in rows]
        ),
        "similarity_rmse_mm": _summary(
            [float(row["similarity_rmse_mm"]) for row in rows]
        ),
        "affine_rmse_mm": _summary(
            [float(row["affine_rmse_mm"]) for row in rows if row["affine_rmse_mm"] != ""]
        ),
        "minimum_hull_coverage": _summary(
            [float(row["minimum_hull_coverage"]) for row in rows]
        ),
        "failures": failures,
        "interpretation": (
            "These are in-sample geometric residuals of official 9.361 correspondences, "
            "not reconstruction accuracy and not an identity-matching score."
        ),
    }
    (args.output_dir / "registration_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
