#!/usr/bin/env python3
"""Train a pixel-wise ensemble combiner (pixel DDPM + residual DDPM + CVAE) on Track E.

Trained against Sd302SyntheticPartialDataset's exact synthetic ground truth
(Track E, docs/nist302_ablation_master_table.md section 17) rather than
`X_pseudo`, deliberately: this project's central finding is that `X_pseudo`
supervision can look like an improvement while regressing on real targets,
so training the combiner against it would risk the same trap. All three
base models are frozen; only the small EnsembleCombiner network is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.nist302_synthetic_dataset import Sd302SyntheticPartialDataset
from fingerprint_reconstruction.models.ensemble import EnsembleCombiner
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from nist302_ensemble_base_models import MODEL_ORDER, base_model_predictions, load_base_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-manifest", type=Path, required=True)
    parser.add_argument("--registered-manifest", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--base-model-k", type=int, default=5)
    parser.add_argument("--max-train-images", type=int, default=600)
    parser.add_argument("--max-validation-images", type=int, default=150)
    parser.add_argument("--seed", type=int, default=9173)
    return parser.parse_args()


def run_epoch(
    combiner, base_models, loader, device, *, k, optimizer, generator,
) -> dict[str, float]:
    training = optimizer is not None
    combiner.train(training)
    total_loss, total_average_loss, examples = 0.0, 0.0, 0
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        missing = 1.0 - mask
        with torch.no_grad():
            predictions_dict = base_model_predictions(base_models, observed, mask, k=k, generator=generator)
            predictions = torch.stack([predictions_dict[name] for name in MODEL_ORDER], dim=1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            blended, _ = combiner(observed, mask, predictions)
            error = (blended - target).abs() * missing
            loss = error.sum() / missing.sum().clamp_min(1.0)
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite ensemble combiner loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(combiner.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        with torch.no_grad():
            average_prediction = predictions.mean(dim=1)
            average_error = (average_prediction - target).abs() * missing
            average_loss = average_error.sum() / missing.sum().clamp_min(1.0)
        size = target.shape[0]
        total_loss += float(loss.detach()) * size
        total_average_loss += float(average_loss) * size
        examples += size
    if not examples:
        raise ValueError("empty ensemble-combiner epoch")
    return {"combiner_mae": total_loss / examples, "simple_average_mae": total_average_loss / examples, "examples": float(examples)}


def main() -> None:
    args = parse_args()
    seed_state = seed_everything(args.seed)
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    train_dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="train", output_shape=(128, 128), base_seed=args.seed,
        max_images=args.max_train_images,
    )
    validation_dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="validation", output_shape=(128, 128), base_seed=args.seed,
        max_images=args.max_validation_images,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    base_models = load_base_models(device)
    combiner = EnsembleCombiner(num_models=len(MODEL_ORDER)).to(device)
    optimizer = torch.optim.AdamW(combiner.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_metrics = run_epoch(combiner, base_models, train_loader, device, k=args.base_model_k, optimizer=optimizer, generator=generator)
        validation_metrics = run_epoch(combiner, base_models, validation_loader, device, k=args.base_model_k, optimizer=None, generator=generator)
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["combiner_mae"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["combiner_mae"], epoch + 1, 0
            torch.save(
                {"model_state": combiner.state_dict(), "model_order": list(MODEL_ORDER), "epoch": best_epoch, "validation_loss": best_loss, "test_loaded": False},
                args.output / "best-checkpoint.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch + 1}/{args.epochs} combiner_mae={train_metrics['combiner_mae']:.5f} "
            f"val_combiner={validation_metrics['combiner_mae']:.5f} val_simple_average={validation_metrics['simple_average_mae']:.5f}",
            flush=True,
        )
        if stale >= 3:
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    report = {
        "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_combiner_mae": best_loss,
        "train_images": len(train_dataset), "validation_images": len(validation_dataset),
        "base_model_k": args.base_model_k, "model_order": list(MODEL_ORDER),
        "training_target": "Track E exact synthetic ground truth (Sd302SyntheticPartialDataset), deliberately not X_pseudo",
        "history": history, "test_loaded": False, "seed_state": seed_state.as_dict(),
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_combiner_mae": best_loss}, indent=2))


if __name__ == "__main__":
    main()
