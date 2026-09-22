#!/usr/bin/env python3
"""Summarize SD302 EFS annotations without inventing a classification accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lffs-csv", type=Path, required=True)
    parser.add_argument("--comp-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q05": float(series.quantile(0.05)),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "q95": float(series.quantile(0.95)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _quality_fractions(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame[[f"quality_{index}_count" for index in range(6)]].sum()
    total = float(counts.sum())
    return {
        str(index): float(counts[f"quality_{index}_count"] / total) if total else 0.0
        for index in range(6)
    }


def main() -> None:
    args = parse_args()
    lffs = pd.read_csv(args.lffs_csv, keep_default_na=False)
    comp = pd.read_csv(args.comp_csv, keep_default_na=False)
    comp_latent = comp[comp.record_index == 3].copy()
    comp_exemplar = comp[comp.record_index == 4].copy()
    determinations = comp_exemplar.examiner_comparisons.str.split(":").str[1]
    determinations = determinations[determinations != ""]

    summary = {
        "schema_version": 1,
        "lffs": {
            "transactions": int(lffs.relative_path.nunique()),
            "assessment_counts": _distribution(lffs.assessment),
            "minutiae": _numeric_summary(lffs.minutiae_count),
            "quality_fractions_all_cells": _quality_fractions(lffs),
            "quality_format_recoveries": int(lffs.quality_format_recovered.sum()),
        },
        "comp": {
            "transactions": int(comp.relative_path.nunique()),
            "latent_assessment_counts": _distribution(comp_latent.assessment),
            "exemplar_assessment_counts": _distribution(comp_exemplar.assessment),
            "examiner_determination_counts": _distribution(determinations),
            "latent_minutiae": _numeric_summary(comp_latent.minutiae_count),
            "exemplar_minutiae": _numeric_summary(comp_exemplar.minutiae_count),
            "matched_correspondences": _numeric_summary(
                comp_latent.matched_correspondence_count
            ),
            "latent_quality_fractions_all_cells": _quality_fractions(comp_latent),
            "transactions_with_relative_rotation": int(
                (comp_latent.relative_rotations != "").sum()
            ),
        },
        "interpretation_guardrails": {
            "quality_denominator": "all 9.308 grid cells, including background=0",
            "comp_pairing": "same-source, different impressions; not pixel-aligned ground truth",
            "correspondence_count": "intersection of official 9.361 labels across Type-9 records",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotation_statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    order = ["NONPRINT", "NOVALUE", "LIMITED", "VALUE"]
    lffs_counts = lffs.assessment.value_counts().reindex(order, fill_value=0)
    comp_counts = comp_latent.assessment.value_counts().reindex(order, fill_value=0)
    x = np.arange(len(order))
    axes[0].bar(x - 0.2, lffs_counts, width=0.4, label="SD302h all LFFS")
    axes[0].bar(x + 0.2, comp_counts, width=0.4, label="SD302i COMP latent")
    axes[0].set_xticks(x, order, rotation=20)
    axes[0].set_ylabel("Number of transactions")
    axes[0].set_title("Examiner value assessment")
    axes[0].legend(fontsize=8)

    quality = pd.DataFrame(
        {
            "LFFS": _quality_fractions(lffs),
            "COMP latent": _quality_fractions(comp_latent),
        }
    ).T
    bottom = np.zeros(len(quality))
    colors = ["#111827", "#6b7280", "#60a5fa", "#22c55e", "#f59e0b", "#ef4444"]
    for index in range(6):
        values = quality[str(index)].to_numpy()
        axes[1].bar(quality.index, values, bottom=bottom, color=colors[index], label=str(index))
        bottom += values
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of all quality-map cells")
    axes[1].set_title("Official ridge-quality composition")
    axes[1].legend(title="9.308 code", ncol=2, fontsize=8)

    max_count = int(comp_latent.matched_correspondence_count.quantile(0.99))
    axes[2].hist(
        comp_latent.matched_correspondence_count.clip(upper=max_count),
        bins=range(0, max_count + 2),
        color="#2563eb",
        edgecolor="white",
    )
    axes[2].axvline(
        comp_latent.matched_correspondence_count.median(),
        color="#dc2626",
        linestyle="--",
        label=f"median={comp_latent.matched_correspondence_count.median():.0f}",
    )
    axes[2].set_xlabel("Matched official 9.361 labels per pair")
    axes[2].set_ylabel("Number of COMP transactions")
    axes[2].set_title(f"Latent–exemplar correspondences\n(values > p99={max_count} clipped)")
    axes[2].legend(fontsize=8)

    fig.suptitle("NIST SD302 annotation audit — descriptive quantities, not model accuracy")
    fig.savefig(args.output_dir / "annotation_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
