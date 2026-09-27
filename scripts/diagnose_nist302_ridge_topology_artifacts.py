#!/usr/bin/env python3
"""Does the DDPM reconstruction degenerate into "painted stripes"?

Motivated by an external review's observation that some reconstructions
contain coherent, near-uniform bands that reduce orientation error without
reconstructing genuine ridge topology. Orientation error alone (a mean
angular distance to a coarse reference) cannot detect this; this script
instead compares, within the same held-out `quality==1` region, the
reconstruction against the real latent pixels on three structural axes a
"painted stripe" artifact would distort: minutiae density (skeleton
crossing-number endings/bifurcations), orientation coherence (suspiciously
*higher* in a degenerate reconstruction), and orientation curvature
(suspiciously *lower* -- too locally straight).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import summarize_ridge_topology
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability

MINIMUM_ROI_PIXELS = 400  # skeleton/minutiae statistics are noisy on tiny regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
        exemplar_roots=roots, split="validation", output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]), schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]), beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]), time_dim=int(checkpoint["time_dim"]),
        multiscale_conditioning=bool(config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=1, middle_attention=bool(config["model"]["middle_attention"]),
        attention_heads=int(config["model"]["attention_heads"]),
        upsampling_mode=str(config["model"]["upsampling_mode"]),
    )
    model = ConditionalDDPM(denoiser, scheduler).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_model = load_support_predictor(Path(config["model"]["support_checkpoint"]), device)

    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, float]] = []
    skipped_small_roi = 0

    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        samples = model.sample_ddim(
            observed, mask, inference_steps=args.ddim_steps, num_samples=args.samples,
            eta=0.0, enforce_data_consistency=True, auxiliary_condition=support,
            generator=generator,
        )
        samples = apply_support_constraint(
            samples[0], observed.expand(args.samples, -1, -1, -1), mask.expand(args.samples, -1, -1, -1),
            support.expand(args.samples, -1, -1, -1),
            mode=str(config["evaluation"]["support_mode"]), threshold=float(config["evaluation"]["support_threshold"]),
        )
        reconstruction = samples.mean(dim=0)[0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout = quality == 1
        if int(heldout.sum()) < MINIMUM_ROI_PIXELS:
            skipped_small_roi += 1
            continue

        real_field = estimate_orientation_field(reference, use_foreground_mask=False)
        recon_field = estimate_orientation_field(reconstruction, use_foreground_mask=False)
        real_summary = summarize_ridge_topology(
            reference, heldout, real_field.theta, real_field.coherence, real_field.valid
        )
        recon_summary = summarize_ridge_topology(
            reconstruction, heldout, recon_field.theta, recon_field.coherence, recon_field.valid
        )
        records.append(
            {
                "sample_id": str(batch["sample_id"][0]),
                "subject_id": str(batch["subject_id"][0]),
                "real_minutiae_density": real_summary.minutiae_density_per_1000px,
                "recon_minutiae_density": recon_summary.minutiae_density_per_1000px,
                "minutiae_density_diff": recon_summary.minutiae_density_per_1000px - real_summary.minutiae_density_per_1000px,
                "real_coherence": real_summary.orientation_coherence_mean,
                "recon_coherence": recon_summary.orientation_coherence_mean,
                # Positive = reconstruction MORE uniform than real ridges (the suspicious direction).
                "coherence_diff": recon_summary.orientation_coherence_mean - real_summary.orientation_coherence_mean,
                "real_curvature": real_summary.orientation_curvature_mean,
                "recon_curvature": recon_summary.orientation_curvature_mean,
                # Negative = reconstruction LESS curved / too locally straight (the suspicious direction).
                "curvature_diff": recon_summary.orientation_curvature_mean - real_summary.orientation_curvature_mean,
            }
        )

    if not records:
        raise ValueError("ridge-topology diagnostic produced no records")
    summary = summarize_records(records)
    report = {
        "num_images": len(records),
        "images_skipped_small_roi": skipped_small_roi,
        "minimum_roi_pixels": MINIMUM_ROI_PIXELS,
        "samples": args.samples,
        "ddim_steps": args.ddim_steps,
        "interpretation": (
            "coherence_diff > 0 and curvature_diff < 0 together would indicate a "
            "'painted stripe' artifact: the reconstruction is more locally uniform "
            "and less naturally curved than real ridges in the same region."
        ),
        "summary": summary,
        "test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {k: summary["metrics"][k]["mean"] for k in ("minutiae_density_diff", "coherence_diff", "curvature_diff") if k in summary["metrics"]},
        indent=2,
    ))


if __name__ == "__main__":
    main()
