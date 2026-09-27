#!/usr/bin/env python3
"""Build a deterministic catalogue of committed result artefacts.

The catalogue indexes aggregate and per-image JSON/CSV files under ``outputs``
without reading or copying fingerprint images. Paths and SHA-256 digests make
reported claims traceable to the exact experiment artefacts used to derive
them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


CATALOGUE_FIELDS = (
    "experiment",
    "relative_path",
    "format",
    "role",
    "size_bytes",
    "sha256",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_role(path: Path) -> str:
    name = path.name.lower()
    if "per-image" in name or "per_image" in name or "per-condition" in name:
        return "unit_level_metrics"
    if "audit" in name or "diagnostic" in name or "metadata" in name:
        return "audit_or_provenance"
    if "calibr" in name:
        return "calibration"
    if "statistic" in name or "comparison" in name or "paired" in name:
        return "statistical_comparison"
    if "metric" in name or "analysis" in name or "summary" in name:
        return "aggregate_metrics"
    return "other_machine_readable"


def discover_artifacts(outputs_root: Path, repository_root: Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    candidates = sorted(
        path
        for path in outputs_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    )
    for path in candidates:
        relative = path.relative_to(repository_root)
        within_outputs = path.relative_to(outputs_root)
        experiment = within_outputs.parts[0] if len(within_outputs.parts) > 1 else "_root"
        artifacts.append(
            {
                "experiment": experiment,
                "relative_path": relative.as_posix(),
                "format": path.suffix.lower().lstrip("."),
                "role": classify_role(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def write_catalogue(artifacts: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOGUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifacts)


def write_summary(
    artifacts: list[dict[str, object]],
    destination: Path,
    catalogue_path: Path,
    repository_root: Path,
) -> None:
    formats = Counter(str(item["format"]) for item in artifacts)
    roles = Counter(str(item["role"]) for item in artifacts)
    experiments = {str(item["experiment"]) for item in artifacts}
    summary = {
        "schema_version": 1,
        "scope": "machine-readable JSON/CSV artefacts under outputs/",
        "artifact_count": len(artifacts),
        "experiment_count": len(experiments),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "counts_by_format": dict(sorted(formats.items())),
        "counts_by_role": dict(sorted(roles.items())),
        "catalogue": catalogue_path.relative_to(repository_root).as_posix(),
        "rebuild_command": "python3 scripts/build_results_catalog.py",
        "contains_raw_fingerprint_images": False,
        "contains_model_weights": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository containing outputs/ and results/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    outputs_root = repository_root / "outputs"
    if not outputs_root.is_dir():
        raise SystemExit(f"outputs directory not found: {outputs_root}")

    results_root = repository_root / "results"
    catalogue_path = results_root / "artifact-index.csv"
    summary_path = results_root / "artifact-summary.json"
    artifacts = discover_artifacts(outputs_root, repository_root)
    write_catalogue(artifacts, catalogue_path)
    write_summary(artifacts, summary_path, catalogue_path, repository_root)
    print(f"Indexed {len(artifacts)} artefacts from {outputs_root}")


if __name__ == "__main__":
    main()
