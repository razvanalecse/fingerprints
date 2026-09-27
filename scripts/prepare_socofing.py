#!/usr/bin/env python3
"""Validate SOCOFing originals and create immutable research manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fingerprint_reconstruction.datasets.socofing import (
    build_socofing_manifest,
    write_socofing_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="SOCOFing root or Real directory")
    parser.add_argument("--output", type=Path, required=True, help="manifest output directory")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="record count mismatches as warnings instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_socofing_manifest(
        args.root,
        split_seed=args.seed,
        strict_expected_counts=not args.allow_incomplete,
    )
    paths = write_socofing_artifacts(manifest, args.output)
    print(json.dumps({"audit": asdict(manifest.audit), "files": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
