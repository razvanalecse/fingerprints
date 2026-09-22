#!/usr/bin/env python3
"""Verify SD302 checksum manifests without requiring a second dataset copy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("spec must be LABEL=CHECKSUM_CSV=DATA_ROOT")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")

    rng = np.random.default_rng(args.seed)
    reports = []
    for label, csv_path, data_root in args.spec:
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or set(rows[0]) != {"filename", "sha256"}:
            raise ValueError(f"invalid checksum CSV: {csv_path}")
        paths = [data_root / row["filename"] for row in rows]
        missing = [str(path) for path in paths if not path.is_file()]
        available_indices = np.asarray(
            [index for index, path in enumerate(paths) if path.is_file()], dtype=np.int64
        )
        count = min(args.sample_size, len(available_indices))
        selected = (
            rng.choice(available_indices, size=count, replace=False).tolist() if count else []
        )
        mismatches = []
        for index in selected:
            actual = sha256_file(paths[index])
            expected = rows[index]["sha256"].lower()
            if actual != expected:
                mismatches.append(
                    {"filename": rows[index]["filename"], "expected": expected, "actual": actual}
                )
        reports.append(
            {
                "label": label,
                "checksum_csv": str(csv_path.resolve()),
                "data_root": str(data_root.resolve()),
                "declared_files": len(rows),
                "missing_files": len(missing),
                "missing_examples": missing[:20],
                "sampled_hashes": count,
                "hash_mismatches": mismatches,
                "passed": not missing and not mismatches,
            }
        )

    result = {
        "seed": args.seed,
        "sample_size_per_manifest": args.sample_size,
        "all_passed": all(report["passed"] for report in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
