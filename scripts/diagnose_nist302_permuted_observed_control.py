#!/usr/bin/env python3
"""Sanity check: does the DDPM genuinely use the observed ridges, or hallucinate generically?

For each validation batch, reconstructs each image twice with the identical
diffusion noise trajectory (same generator seed):

1. **Matched** — conditioned on its own real `observed` (mask * its own
   latent image), as usual.
2. **Permuted** — conditioned on `mask * latent_image[other]`, i.e. the same
   visible-region *shape* but another sample's real ridge content standing
   in for it (batch rolled by one).

Both reconstructions are then scored against the *original* sample's own
held-out `quality==1` pixels. If the model genuinely conditions on the
specific ridges it is shown, feeding it a different finger's ridges through
the same mask should measurably hurt reconstruction quality in the
unobserved region. If matched and permuted are statistically indistinguishable,
that is evidence the model is largely producing a mask-shape-conditioned
generic completion rather than genuinely continuing the observed ridge
structure — i.e. hallucination, not reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


@torch.no_grad()
def reconstruct(model, observed, mask, support, *, ddim_steps, generator):
    reconstruction = model.sample_ddim(
        observed, mask, inference_steps=ddim_steps, num_samples=1,
        eta=0.0, enforce_data_consistency=True, auxiliary_condition=support,
        generator=generator,
    )[:, 0]
    return apply_support_constraint(reconstruction, observed, mask, support, mode="soft", threshold=0.5)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    import yaml

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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

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

    records: list[dict[str, object]] = []
    for batch in loader:
        latent = batch["latent_image"].to(device)
        mask = batch["mask"].to(device)
        quality = batch["quality"].numpy()
        subject_ids = batch["subject_id"] if "subject_id" in batch else None
        batch_size = latent.shape[0]
        if batch_size < 2:
            continue  # cannot permute a batch of one

        observed_matched = torch.where(mask.bool(), latent, torch.zeros_like(latent))
        other = torch.roll(latent, shifts=1, dims=0)
        observed_permuted = torch.where(mask.bool(), other, torch.zeros_like(latent))

        support_matched = support_probability(support_model, observed_matched, mask)
        support_permuted = support_probability(support_model, observed_permuted, mask)

        generator_matched = torch.Generator().manual_seed(args.seed)
        generator_permuted = torch.Generator().manual_seed(args.seed)
        reconstruction_matched = reconstruct(
            model, observed_matched, mask, support_matched,
            ddim_steps=args.ddim_steps, generator=generator_matched,
        )
        reconstruction_permuted = reconstruct(
            model, observed_permuted, mask, support_permuted,
            ddim_steps=args.ddim_steps, generator=generator_permuted,
        )

        reference_np = latent[:, 0].cpu().numpy()
        matched_np = reconstruction_matched[:, 0].cpu().numpy()
        permuted_np = reconstruction_permuted[:, 0].cpu().numpy()
        for index in range(batch_size):
            heldout = quality[index, 0] == 1
            if heldout.sum() < 5:
                continue
            error_matched = np.abs(matched_np[index][heldout] - reference_np[index][heldout])
            error_permuted = np.abs(permuted_np[index][heldout] - reference_np[index][heldout])
            sample_id = batch["sample_id"][index] if "sample_id" in batch else str(len(records))
            subject_id = (
                str(subject_ids[index]) if subject_ids is not None
                else sample_id.split("_")[0] if isinstance(sample_id, str) else str(index)
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "heldout_q1_mae_matched": float(error_matched.mean()),
                    "heldout_q1_mae_permuted": float(error_permuted.mean()),
                }
            )

    if not records:
        raise ValueError("permuted-observed control produced no records")

    subject_matched: dict[str, list[float]] = {}
    subject_permuted: dict[str, list[float]] = {}
    for record in records:
        subject_matched.setdefault(record["subject_id"], []).append(record["heldout_q1_mae_matched"])
        subject_permuted.setdefault(record["subject_id"], []).append(record["heldout_q1_mae_permuted"])
    subjects = sorted(subject_matched)
    matched_means = {subject: float(np.mean(subject_matched[subject])) for subject in subjects}
    permuted_means = {subject: float(np.mean(subject_permuted[subject])) for subject in subjects}

    tests = paired_metric_suite(
        {"heldout_q1_mae": [matched_means[subject] for subject in subjects]},
        {"heldout_q1_mae": [permuted_means[subject] for subject in subjects]},
        {"heldout_q1_mae": False},
    )
    report = {
        "num_images": len(records),
        "num_subjects": len(subjects),
        "ddim_steps": args.ddim_steps,
        "interpretation": (
            "positive_mean_improvement_means_permuted_is_better; a large POSITIVE "
            "value (permuted has LOWER error than matched) would be the alarming "
            "result -- it would mean feeding the wrong finger's ridges through the "
            "same mask does not hurt reconstruction, i.e. the model ignores observed "
            "ridge content. A significant NEGATIVE value (permuted is worse) is the "
            "expected, reassuring result: the model measurably suffers when given "
            "the wrong identity's ridges, i.e. it is genuinely using what it is shown."
        ),
        "tests": tests,
        "test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
