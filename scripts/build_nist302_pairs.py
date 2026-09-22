#!/usr/bin/env python3
"""Build long-form SD302 latent/exemplar candidate associations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from fingerprint_reconstruction.data import build_annotated_rows
from fingerprint_reconstruction.datasets.nist302_pairs import (
    build_latent_exemplar_pairs,
    load_exemplar_rows,
    write_pair_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--finger-positions", type=Path, required=True)
    parser.add_argument("--exemplars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    latent_rows = []
    for split in ("train", "validation", "test"):
        latent_rows.extend(
            build_annotated_rows(
                manifest_path=args.manifest,
                annotation_csv=args.annotations,
                split=split,
                assessments=("VALUE", "LIMITED"),
                source_codes=(1,),
                exclude_errata=True,
                finger_positions_csv=args.finger_positions,
            )
        )
    pairs = build_latent_exemplar_pairs(latent_rows, load_exemplar_rows(args.exemplars))
    write_pair_manifest(pairs, args.output)
    print(f"pairs={len(pairs)}")
    print(f"latent_samples={len({pair.latent_sample_id for pair in pairs})}")
    print(f"subjects={len({pair.subject_id for pair in pairs})}")
    print(f"split_pairs={dict(sorted(Counter(pair.split for pair in pairs).items()))}")
    print(f"dataset_parts={dict(sorted(Counter(pair.exemplar_dataset_part for pair in pairs).items()))}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
