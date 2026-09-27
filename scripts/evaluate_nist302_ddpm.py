#!/usr/bin/env python3
"""Probabilistic validation of a support-conditioned NIST SD302 DDPM."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.metrics import (
    empirical_interval_coverage,
    paired_ridge_frequency_error,
    pairwise_diversity_mae,
    region_image_metrics,
    uncertainty_error_spearman,
)
from fingerprint_reconstruction.models.diffusion import (
    ConditionalDDPM,
    DDPMScheduler,
    DiffusionUNet,
)
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import code_provenance, load_support_predictor, support_probability


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
    parser.add_argument("--ddim-steps", type=int, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--sample-chunk-size", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    return parser.parse_args()


@torch.no_grad()
def sample_chunks(
    model: ConditionalDDPM,
    observed: torch.Tensor,
    mask: torch.Tensor,
    support: torch.Tensor,
    *,
    count: int,
    chunk_size: int,
    ddim_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    chunks = []
    for start in range(0, count, chunk_size):
        size = min(chunk_size, count - start)
        chunks.append(
            model.sample_ddim(
                observed,
                mask,
                inference_steps=ddim_steps,
                num_samples=size,
                eta=0.0,
                enforce_data_consistency=True,
                auxiliary_condition=support,
                generator=generator,
            )
        )
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    k_values = tuple(sorted(set(int(value) for value in args.k)))
    if not k_values or min(k_values) < 2 or args.sample_chunk_size <= 0:
        raise ValueError("K must contain integers >=2 and chunk size must be positive")
    seed_everything(args.seed)
    device = select_device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    timesteps = int(checkpoint["timesteps"])
    if not 2 <= args.ddim_steps <= timesteps:
        raise ValueError("DDIM steps must lie in [2, training timesteps]")
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots=roots,
        split=args.split,
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
    )
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("max-images must be positive")
        evaluation_dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    else:
        evaluation_dataset = dataset
    loader = DataLoader(evaluation_dataset, batch_size=1, shuffle=False, num_workers=0)
    scheduler = DDPMScheduler(
        timesteps=timesteps,
        schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]),
        beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]),
        time_dim=int(checkpoint["time_dim"]),
        multiscale_conditioning=bool(config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=1,
        middle_attention=bool(config["model"]["middle_attention"]),
        attention_heads=int(config["model"]["attention_heads"]),
        upsampling_mode=str(config["model"]["upsampling_mode"]),
    )
    model = ConditionalDDPM(denoiser, scheduler).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    support_model = load_support_predictor(
        Path(config["model"]["support_checkpoint"]), device
    )
    generator = torch.Generator().manual_seed(args.seed)
    maximum_k = max(k_values)
    nominal_coverages = tuple(float(v) for v in (0.50, 0.80, 0.90, 0.95))
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        support = support_probability(support_model, observed, mask)
        samples = sample_chunks(
            model,
            observed,
            mask,
            support,
            count=maximum_k,
            chunk_size=args.sample_chunk_size,
            ddim_steps=args.ddim_steps,
            generator=generator,
        )
        batch_size, count, _, height, width = samples.shape
        samples = apply_support_constraint(
            samples.reshape(batch_size * count, 1, height, width),
            observed.repeat_interleave(count, dim=0),
            mask.repeat_interleave(count, dim=0),
            support.repeat_interleave(count, dim=0),
            mode=str(config["evaluation"]["support_mode"]),
            threshold=float(config["evaluation"]["support_threshold"]),
        ).reshape(batch_size, count, 1, height, width)
        per_sample = samples[0, :, 0].cpu().numpy()
        reference = batch["latent_image"][0, 0].numpy()
        quality = batch["quality"][0, 0].numpy()
        heldout, background = quality == 1, quality == 0
        record: dict[str, object] = {
            "sample_id": str(batch["sample_id"][0]),
            "subject_id": str(batch["subject_id"][0]),
            "ddim_steps": args.ddim_steps,
        }
        single = region_image_metrics(reference, per_sample[0], heldout)
        record.update({f"single_heldout_q1_{name}": value for name, value in single.items()})
        record["single_background_darkness"] = float(
            np.mean(1.0 - per_sample[0][background])
        )
        for k in k_values:
            selected = per_sample[:k]
            mean = selected.mean(axis=0)
            metrics = region_image_metrics(reference, mean, heldout)
            record.update(
                {f"mean_k{k}_heldout_q1_{name}": value for name, value in metrics.items()}
            )
            per_sample_mae = np.mean(
                np.abs(selected[:, heldout] - reference[heldout][None]), axis=1
            )
            record[f"best_k{k}_heldout_q1_mae"] = float(per_sample_mae.min())
            record[f"diversity_k{k}_heldout_q1_mae"] = pairwise_diversity_mae(
                selected, heldout
            )
            record[f"uncertainty_error_spearman_k{k}"] = uncertainty_error_spearman(
                selected, reference, heldout
            )
            for nominal in nominal_coverages:
                coverage, width_value = empirical_interval_coverage(
                    selected, reference, heldout, nominal_coverage=nominal
                )
                label = int(round(100 * nominal))
                record[f"coverage_k{k}_{label}"] = coverage
                record[f"coverage_error_k{k}_{label}"] = abs(coverage - nominal)
                record[f"interval_width_k{k}_{label}"] = width_value
            std = selected.std(axis=0, ddof=1)
            record[f"mean_uncertainty_k{k}_heldout_q1"] = float(std[heldout].mean())
            record[f"mean_uncertainty_k{k}_background"] = float(std[background].mean())
            record[f"mean_k{k}_background_darkness"] = float(
                np.mean(1.0 - mean[background])
            )
        mean = per_sample.mean(axis=0)
        reference_field = estimate_orientation_field(reference, use_foreground_mask=False)
        mean_field = estimate_orientation_field(mean, use_foreground_mask=False)
        try:
            record["mean_heldout_q1_orientation_error"] = orientation_error(
                reference_field, mean_field, region_mask=heldout
            )
        except ValueError:
            record["mean_heldout_q1_orientation_error"] = float("nan")
        frequency = paired_ridge_frequency_error(reference, mean, heldout)
        record.update({f"mean_heldout_q1_{key}": value for key, value in frequency.items()})
        records.append(record)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per-image-probabilistic-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "split": "validation",
        "test_loaded": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_timesteps": timesteps,
        "ddim_steps": args.ddim_steps,
        "k_values": list(k_values),
        "evaluation_seed": args.seed,
        "images": len(records),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(records),
        "metrics": summarize_records(records),
        "provenance": code_provenance(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "method-audit.json").write_text(
        json.dumps(
            {
                "test_loaded": False,
                "target": "real latent held-out quality=1 pixels",
                "sampling": "DDIM eta=0 with seeded initial noise",
                "data_consistency": "exact on quality>=2 observed pixels",
                "support": "frozen visible-support predictor; soft composition",
                "best_of_k_not_single_sample": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
