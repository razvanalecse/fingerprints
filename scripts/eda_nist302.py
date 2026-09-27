#!/usr/bin/env python3
"""Create a deterministic visual/data audit for an SD302 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fingerprint_reconstruction.preprocessing.normalize import load_grayscale


def _choose_exemplar(serialized: str) -> tuple[str, str]:
    candidates = json.loads(serialized)
    if not candidates:
        raise ValueError("record has no exemplar candidate")

    def rank(candidate: str) -> tuple[int, str]:
        if candidate.startswith("sd302b:") and "/1000/roll/" in candidate:
            return 0, candidate
        if candidate.startswith("sd302a:"):
            return 1, candidate
        if candidate.startswith("sd302b:") and "slap-segmented" in candidate:
            return 2, candidate
        if candidate.startswith("sd302d:"):
            return 3, candidate
        return 4, candidate

    selected = min(candidates, key=rank)
    return tuple(selected.split(":", maxsplit=1))  # type: ignore[return-value]


def _read(root: Path, relative: str) -> np.ndarray:
    path = root / relative
    with Image.open(path) as image:
        width, height = image.size
    scale = min(1.0, 512.0 / max(width, height))
    output_shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    return load_grayscale(path, output_shape=output_shape, interpolation="bilinear")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sd302e-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nist302_eda"))
    parser.add_argument("--seed", type=int, default=302)
    args = parser.parse_args()

    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    data = pd.read_csv(args.manifest, dtype={"subject_id": str})
    eligible = data[
        data["fgp"].notna()
        & ~data["errata_mentioned"].astype(bool)
        & (data["exemplar_paths"] != "[]")
    ].copy()
    if eligible.empty:
        raise ValueError("manifest contains no eligible paired records")

    rng = np.random.default_rng(args.seed)
    chosen = []
    # One example per development technique, then deterministic fill if needed.
    for technique in ("BP", "IN", "BT", "CA", "WT"):
        group = eligible[eligible["technique"] == technique]
        if not group.empty:
            chosen.append(group.iloc[int(rng.integers(len(group)))])
    remaining = eligible[~eligible["sample_id"].isin([row["sample_id"] for row in chosen])]
    while len(chosen) < 6 and not remaining.empty:
        index = int(rng.integers(len(remaining)))
        chosen.append(remaining.iloc[index])
        remaining = remaining.drop(remaining.index[index])

    figure, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.2 * len(chosen)))
    columns = ("Original unmasked", "Original masked", "Enhanced masked", "Same-finger exemplar")
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    for row_index, row in enumerate(chosen):
        exemplar_part, exemplar_relative = _choose_exemplar(row["exemplar_paths"])
        images = (
            _read(args.sd302e_root, row["original_unmasked_path"]),
            _read(args.sd302e_root, row["original_masked_path"]),
            _read(args.sd302e_root, row["enhanced_masked_path"]),
            _read(roots[exemplar_part], exemplar_relative),
        )
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(
            f"{row['subject_id']} | {row['technique']}\n"
            f"{int(row['native_ppi'])} PPI | FGP {int(row['fgp'])}",
            fontsize=9,
        )

    figure.suptitle(
        "NIST SD302: latent variants and associated same-finger exemplars\n"
        "Exemplars are different impressions, not pixel-aligned ground truth",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "latent-exemplar-audit.png", dpi=140, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "num_manifest_records": int(len(data)),
        "num_eligible_known_fgp_non_errata": int(len(eligible)),
        "num_subjects": int(data["subject_id"].nunique()),
        "technique_counts": {str(k): int(v) for k, v in data["technique"].value_counts().items()},
        "native_ppi_counts": {str(k): int(v) for k, v in data["native_ppi"].value_counts().sort_index().items()},
        "selected_sample_ids": [str(row["sample_id"]) for row in chosen],
        "warning": "Same-finger exemplars are not pixel-aligned reconstruction targets.",
    }
    (args.output / "eda-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output / 'latent-exemplar-audit.png'}")


if __name__ == "__main__":
    main()
