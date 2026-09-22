#!/usr/bin/env python3
"""Build an auditable, subject-disjoint NIST SD 302 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from fingerprint_reconstruction.datasets.nist302 import (
    build_nist302_manifest,
    write_nist302_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd302e-root", type=Path, required=True)
    parser.add_argument("--finger-positions", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--errata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/nist302"))
    parser.add_argument("--seed", type=int, default=302)
    args = parser.parse_args()

    manifest = build_nist302_manifest(
        sd302e_root=args.sd302e_root,
        finger_positions_csv=args.finger_positions,
        exemplar_roots={
            "sd302a": args.sd302a_root,
            "sd302b": args.sd302b_root,
            "sd302d": args.sd302d_root,
        },
        errata_path=args.errata,
        seed=args.seed,
    )
    write_nist302_artifacts(manifest, args.output)
    audit = manifest.audit
    print(
        f"NIST SD302: {audit.num_latent_pngs} latent PNGs, "
        f"{audit.num_subjects} subjects, {audit.num_known_fgp} known FGP, "
        f"{audit.num_with_exemplar_candidates} with exemplar candidates"
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
