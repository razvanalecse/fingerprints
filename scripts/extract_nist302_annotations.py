#!/usr/bin/env python3
"""Extract auditable SD 302h/302i annotation summaries with NBIS an2k2txt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.datasets.nist302_ebts import decode_transaction, match_correspondences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--suffix", choices=(".lffs", ".comp"), required=True)
    parser.add_argument("--an2k2txt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Deterministically sample files across the root instead of taking a prefix.",
    )
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument(
        "--save-quality-maps",
        action="store_true",
        help="Store uncompressed 9.308 maps as compressed uint8 NPZ files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = sorted(args.input_root.rglob(f"*{args.suffix}"))
    num_files_discovered = len(inputs)
    if args.max_files is not None and args.sample_size is not None:
        raise ValueError("use only one of --max-files and --sample-size")
    if args.sample_size is not None:
        if not 0 < args.sample_size <= len(inputs):
            raise ValueError("--sample-size must be between 1 and the number of files")
        inputs = sorted(random.Random(args.seed).sample(inputs, args.sample_size))
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        inputs = inputs[: args.max_files]
    if not inputs:
        raise FileNotFoundError(f"no {args.suffix} files below {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "quality_maps"
    if args.save_quality_maps:
        maps_dir.mkdir(exist_ok=True)

    rows = []
    warning_counts: Counter[str] = Counter()
    failures = []
    for sequence, path in enumerate(inputs):
        try:
            transaction = decode_transaction(path, an2k2txt=args.an2k2txt)
        except Exception as error:  # keep a complete, auditable failure ledger
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            continue
        warning_counts.update(transaction.decoder_warnings)
        relative = path.relative_to(args.input_root).as_posix()
        sources = ";".join(
            f"{item.source_index}:{item.transaction_kind}:{item.filename}"
            for item in transaction.comp_sources
        )
        record_indices = [record.record_index for record in transaction.efs_records]
        for record in transaction.efs_records:
            quality = record.ridge_quality
            counts = quality.counts() if quality is not None else {}
            map_path = ""
            if args.save_quality_maps and quality is not None and quality.encoding == "UNC":
                array = np.asarray([[ord(value) - 48 for value in row] for row in quality.rows], dtype=np.uint8)
                destination = maps_dir / f"{sequence:05d}_{path.stem}_record{record.record_index}.npz"
                np.savez_compressed(
                    destination,
                    quality=array,
                    grid_size_0_01mm=np.uint16(quality.grid_size_0_01mm),
                )
                map_path = destination.relative_to(args.output_dir).as_posix()
            rows.append(
                {
                    "relative_path": relative,
                    "transaction_type": transaction.transaction_type or "",
                    "version": transaction.version or "",
                    "record_index": record.record_index,
                    "idc": record.idc,
                    "assessment": record.assessment or "",
                    "roi_width_0_01mm": record.roi.width_0_01mm,
                    "roi_height_0_01mm": record.roi.height_0_01mm,
                    "roi_horizontal_offset_0_01mm": record.roi.horizontal_offset_0_01mm,
                    "roi_vertical_offset_0_01mm": record.roi.vertical_offset_0_01mm,
                    "roi_polygon_points": len(record.roi.polygon_0_01mm),
                    "quality_height": quality.shape[0] if quality else 0,
                    "quality_width": quality.shape[1] if quality else 0,
                    "quality_grid_size_0_01mm": quality.grid_size_0_01mm if quality else "",
                    "quality_encoding": quality.encoding if quality else "",
                    "quality_format_recovered": record.quality_format_recovered,
                    **{f"quality_{value}_count": counts.get(value, 0) for value in range(6)},
                    "minutiae_count": len(record.minutiae),
                    "comp_feature_reference_count": len(
                        transaction.comp_feature_references.get(record.record_index, ())
                    ),
                    "matched_correspondence_count": sum(
                        len(match_correspondences(transaction, record.record_index, other))
                        for other in record_indices
                        if other != record.record_index
                    ),
                    "examiner_comparisons": ";".join(
                        f"{item.target_idc}:{item.determination}:{item.status}"
                        for item in transaction.examiner_comparisons.get(record.record_index, ())
                    ),
                    "relative_rotations": ";".join(
                        f"{item.target_idc}:{item.degrees}"
                        for item in transaction.relative_rotations.get(record.record_index, ())
                    ),
                    "comp_sources": sources,
                    "quality_map_path": map_path,
                }
            )

    csv_path = args.output_dir / "annotation_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    audit = {
        "schema_version": 1,
        "input_root": str(args.input_root.resolve()),
        "suffix": args.suffix,
        "decoder": str(args.an2k2txt.resolve()),
        "decoder_sha256": _sha256(args.an2k2txt),
        "num_files_discovered": num_files_discovered,
        "num_files_selected": len(inputs),
        "selection": (
            {"method": "random_without_replacement", "seed": args.seed}
            if args.sample_size is not None
            else {"method": "sorted_prefix" if args.max_files is not None else "all"}
        ),
        "num_files_decoded": len(inputs) - len(failures),
        "num_efs_records": len(rows),
        "warning_counts": dict(sorted(warning_counts.items())),
        "failures": failures,
    }
    (args.output_dir / "extraction_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
