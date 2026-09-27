#!/usr/bin/env python3
"""Is the reconstructed ridge field too predictable to be real?

Section 15's ridge-topology check established *descriptively* that the DDPM
reconstruction is more locally uniform and less curved than the real ridges it
replaces. This script turns that into a statistical statement: read the
orientation field as a first-order Markov chain over quantized axial
orientation states and measure how much the next state is determined by the
current one.

Real friction ridges curve, bifurcate and terminate, all of which inject
transitions between orientation states. A generatively "painted" ridge field
does less of that, so it should show

  * lower conditional entropy  H(theta_{t+1} | theta_t)   -- more predictable
  * higher self-transition rate P(theta_{t+1} = theta_t)  -- flatter flow

Both are computed for the reconstruction and for the real latent *inside the
same held-out region*, so the small-sample entropy bias is shared by the two
arms and largely cancels in the paired difference. Two spatial steps are
reported: 2px (sub-ridge, where the orientation estimator itself smooths) and
4px (comparable to the 6.45px ridge period at this resolution).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import summarize_ridge_markov
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability

MINIMUM_ROI_PIXELS = 400  # transition counts are noisy on tiny regions
STEPS = (2, 4)
ORIENTATION_BINS = 12


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
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
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
        exemplar_roots=roots, split=args.split, output_shape=tuple(config["data"]["image_size"]),
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
            mode=str(config["evaluation"]["support_mode"]),
            threshold=float(config["evaluation"]["support_threshold"]),
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

        record: dict[str, float] = {
            "sample_id": str(batch["sample_id"][0]),
            "subject_id": str(batch["subject_id"][0]),
        }
        for step in STEPS:
            real = summarize_ridge_markov(
                real_field.theta, heldout, valid=real_field.valid,
                bins=ORIENTATION_BINS, step=step,
            )
            recon = summarize_ridge_markov(
                recon_field.theta, heldout, valid=recon_field.valid,
                bins=ORIENTATION_BINS, step=step,
            )
            if not np.isfinite(real.transition_entropy_bits) or not np.isfinite(recon.transition_entropy_bits):
                continue
            record[f"real_entropy_step{step}"] = real.transition_entropy_bits
            record[f"recon_entropy_step{step}"] = recon.transition_entropy_bits
            # Negative = reconstruction MORE predictable than real ridges (suspicious).
            record[f"entropy_diff_step{step}"] = (
                recon.transition_entropy_bits - real.transition_entropy_bits
            )
            record[f"real_self_rate_step{step}"] = real.self_transition_rate
            record[f"recon_self_rate_step{step}"] = recon.self_transition_rate
            # Positive = reconstruction's orientation changes LESS often (suspicious).
            record[f"self_rate_diff_step{step}"] = (
                recon.self_transition_rate - real.self_transition_rate
            )
            record[f"transitions_step{step}"] = float(real.transition_count)
        if len(record) > 2:
            records.append(record)

    if not records:
        raise ValueError("ridge-Markov diagnostic produced no records")
    summary = summarize_records(records)
    report = {
        "num_images": len(records),
        "images_skipped_small_roi": skipped_small_roi,
        "minimum_roi_pixels": MINIMUM_ROI_PIXELS,
        "orientation_bins": ORIENTATION_BINS,
        "steps": list(STEPS),
        "samples": args.samples,
        "ddim_steps": args.ddim_steps,
        "split": args.split,
        "interpretation": (
            "entropy_diff < 0 together with self_rate_diff > 0 means the reconstructed "
            "orientation field is more predictable from its own neighbourhood than the "
            "real ridges occupying the same region: the generative model produces flow "
            "that is too regular, which is the statistical form of the 'painted stripe' "
            "artifact measured descriptively in ablation section 15."
        ),
        "summary": summary,
        "test_loaded": args.split == "test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Per-image rows so the paired subject-level test can be run on this output.
    csv_path = args.output.with_suffix(".per-image.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selected = {
        key: summary["metrics"][key]["mean"]
        for key in summary["metrics"]
        if key.startswith(("entropy_diff", "self_rate_diff"))
    }
    print(json.dumps({"images": len(records), **selected}, indent=2))


if __name__ == "__main__":
    main()
