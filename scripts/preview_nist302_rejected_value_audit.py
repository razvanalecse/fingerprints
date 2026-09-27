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

import yaml

from fingerprint_reconstruction.data import Nist302AnnotatedDataset
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor, apply_support_constraint
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
    parser.add_argument("--ddpm-config", type=Path, required=True)
    parser.add_argument("--ddpm-checkpoint", type=Path, required=True)
    parser.add_argument("--ddpm-ddim-steps", type=int, default=20)
    parser.add_argument("--ddpm-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9173)
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
    by_path = {row.image_relative_path: index for index, row in enumerate(dataset.rows)}
    missing = [row["latent_relative_path"] for row in audit_rows if row["latent_relative_path"] not in by_path]
    if missing:
        print(
            f"skipping {len(missing)}/{len(audit_rows)} audit rows whose LFFS record is not "
            f"VALUE/LIMITED-assessed at source_code in (1,2,3,4): {missing}"
        )
        audit_rows = [row for row in audit_rows if row["latent_relative_path"] in by_path]
    device = select_device(args.device)
    zero_shot = load_reconstruction(args.zero_shot_checkpoint, device)
    finetuned = load_reconstruction(args.finetuned_checkpoint, device)
    support_checkpoint = torch.load(args.support_checkpoint, map_location=device, weights_only=False)
    support = FingerprintSupportPredictor(
        channels=tuple(support_checkpoint["channels_used"])
    ).to(device)
    support.load_state_dict(support_checkpoint["model_state"])
    support.eval()

    ddpm_config = yaml.safe_load(args.ddpm_config.read_text(encoding="utf-8"))
    ddpm_checkpoint = torch.load(args.ddpm_checkpoint, map_location=device, weights_only=False)
    scheduler = DDPMScheduler(
        timesteps=int(ddpm_checkpoint["timesteps"]),
        schedule=str(ddpm_config["diffusion"]["schedule"]),
        beta_start=float(ddpm_config["diffusion"]["beta_start"]),
        beta_end=float(ddpm_config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(ddpm_checkpoint["channels_used"]),
        time_dim=int(ddpm_checkpoint["time_dim"]),
        multiscale_conditioning=bool(ddpm_config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=1,
        middle_attention=bool(ddpm_config["model"]["middle_attention"]),
        attention_heads=int(ddpm_config["model"]["attention_heads"]),
        upsampling_mode=str(ddpm_config["model"]["upsampling_mode"]),
    )
    ddpm = ConditionalDDPM(denoiser, scheduler).to(device)
    ddpm.load_state_dict(ddpm_checkpoint["model_state"])
    ddpm.eval()
    generator = torch.Generator().manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "Real latent",
        "Y (quality>=2)",
        "Zero-shot gated",
        "Fine-tuned pilot",
        "Support-constrained pilot",
        "Predicted support",
        "DDPM predictive mean",
        "DDPM predictive std",
    )
    pages = math.ceil(len(audit_rows) / args.rows_per_page)
    with torch.no_grad():
        for page in range(pages):
            selected = audit_rows[page * args.rows_per_page : (page + 1) * args.rows_per_page]
            figure, axes = plt.subplots(len(selected), len(columns), figsize=(15, 3 * len(selected)), squeeze=False)
            for row_index, audit in enumerate(selected):
                item = dataset[by_path[audit["latent_relative_path"]]]
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

                ddpm_samples = ddpm.sample_ddim(
                    observed, mask, inference_steps=args.ddpm_ddim_steps,
                    num_samples=args.ddpm_samples, eta=0.0, enforce_data_consistency=True,
                    auxiliary_condition=support_probability, generator=generator,
                )
                count = ddpm_samples.shape[1]
                ddpm_samples = apply_support_constraint(
                    ddpm_samples.reshape(count, 1, *observed.shape[-2:]),
                    observed.repeat_interleave(count, dim=0),
                    mask.repeat_interleave(count, dim=0),
                    support_probability.repeat_interleave(count, dim=0),
                    mode="soft", threshold=0.5,
                ).reshape(1, count, 1, *observed.shape[-2:])
                ddpm_mean = ddpm_samples[0, :, 0].mean(dim=0)
                ddpm_std = ddpm_samples[0, :, 0].std(dim=0)

                panels = (
                    image[0, 0].cpu().numpy(),
                    shown_observed[0, 0].cpu().numpy(),
                    zero_prediction[0, 0].cpu().numpy(),
                    fine_prediction[0, 0].cpu().numpy(),
                    constrained[0, 0].cpu().numpy(),
                    support_probability[0, 0].cpu().numpy(),
                    ddpm_mean.cpu().numpy(),
                    ddpm_std.cpu().numpy(),
                )
                reason = audit["sampling_stratum"].replace("_", " ")
                for column, (axis, panel, title) in enumerate(zip(axes[row_index], panels, columns)):
                    axis.imshow(panel, cmap="gray" if title != "DDPM predictive std" else "magma")
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
