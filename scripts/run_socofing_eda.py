#!/usr/bin/env python3
"""Generate SOCOFing audit tables and deterministic diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingerprint_reconstruction.analysis.socofing_eda import run_socofing_eda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()
    report = run_socofing_eda(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_directory=args.output,
        sample_id=args.sample_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
