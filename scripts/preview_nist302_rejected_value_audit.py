#!/usr/bin/env python3
"""Render qualitative-only predictions for VALUE latents rejected by geometry filters."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fingerprint_reconstruction.data import Nist302AnnotatedDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor
from fingerprint_reconstruction.training.trainer import select_device


def load_reconstruction(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--zero-shot-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rows-per-page", type=int, default=4)
    args = parser.parse_args()
    with args.audit_csv.open(newline="", encoding="utf-8") as stream:
        audit_rows = list(csv.DictReader(stream))
    if any(row["split"] != "validation" or row["qualitative_only"] != "True" for row in audit_rows):
        raise ValueError("audit CSV must contain validation-only qualitative rows")
    dataset = Nist302AnnotatedDataset(
        manifest_path=args.manifest,
        annotation_csv=args.annotation_csv,
        image_root=args.image_root,
        annotation_root=args.annotation_root,
        split="validation",
        output_shape=(128, 128),
        assessments=("VALUE",),
        source_codes=(1, 2, 3, 4),
        exclude_errata=False,
    )
    by_lffs = {row.lffs_filename: index for index, row in enumerate(dataset.rows)}
    missing = [row["lffs_filename"] for row in audit_rows if row["lffs_filename"] not in by_lffs]
    if missing:
        raise ValueError(f"audit LFFS records missing from annotated dataset: {missing[:3]}")
    device = select_device(args.device)
    zero_shot = load_reconstruction(args.zero_shot_checkpoint, device)
    finetuned = load_reconstruction(args.finetuned_checkpoint, device)
    support_checkpoint = torch.load(args.support_checkpoint, map_location=device, weights_only=False)
    support = FingerprintSupportPredictor(
        channels=tuple(support_checkpoint["channels_used"])
    ).to(device)
    support.load_state_dict(support_checkpoint["model_state"])
    support.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "Real latent",
        "Y (quality>=2)",
        "Zero-shot gated",
        "Fine-tuned pilot",
        "Support-constrained pilot",
    )
    pages = math.ceil(len(audit_rows) / args.rows_per_page)
    with torch.no_grad():
        for page in range(pages):
            selected = audit_rows[page * args.rows_per_page : (page + 1) * args.rows_per_page]
            figure, axes = plt.subplots(len(selected), len(columns), figsize=(15, 3 * len(selected)), squeeze=False)
            for row_index, audit in enumerate(selected):
                item = dataset[by_lffs[audit["lffs_filename"]]]
                image = item["image"].unsqueeze(0).to(device)
                observed = item["observed"].unsqueeze(0).to(device)
                mask = item["mask"].unsqueeze(0).to(device)
                condition = torch.cat((observed, mask), dim=1)
                zero_raw = zero_shot(condition)
                fine_raw = finetuned(condition)
                zero_prediction = mask * observed + (1.0 - mask) * zero_raw
                fine_prediction = mask * observed + (1.0 - mask) * fine_raw
                support_probability = torch.maximum(support(condition), mask)
                constrained_missing = support_probability * fine_raw + (1.0 - support_probability)
                constrained = mask * observed + (1.0 - mask) * constrained_missing
                shown_observed = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
                panels = (
                    image[0, 0].cpu().numpy(),
                    shown_observed[0, 0].cpu().numpy(),
                    zero_prediction[0, 0].cpu().numpy(),
                    fine_prediction[0, 0].cpu().numpy(),
                    constrained[0, 0].cpu().numpy(),
                )
                reason = audit["sampling_stratum"].replace("_", " ")
                for column, (axis, panel, title) in enumerate(zip(axes[row_index], panels, columns)):
                    axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
                    axis.axis("off")
                    axis.set_title(title if row_index == 0 else "", fontsize=9)
                    if column == 0:
                        axis.set_ylabel(
                            f"{audit['subject_id']}\n{reason}", fontsize=8, rotation=0, labelpad=52
                        )
            figure.suptitle(
                "Rejected VALUE validation audit — qualitative only; no loss, metrics, or performance claim",
                fontsize=11,
            )
            figure.tight_layout()
            figure.savefig(
                args.output_dir / f"rejected_value_page_{page + 1:02d}.png",
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(figure)


if __name__ == "__main__":
    main()
