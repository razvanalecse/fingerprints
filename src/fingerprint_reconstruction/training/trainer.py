"""Minimal, auditable training utilities for deterministic reconstruction."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, Iterable, Mapping, Optional

import torch

from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_reconstruction_epoch(
    *,
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, object]],
    loss_function: MaskedReconstructionLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = defaultdict(float)
    examples = 0
    batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        conditioning = batch["conditioning"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        batch_size = int(target.shape[0])

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if hasattr(model, "forward_with_structure"):
                prediction, predicted_structure = model.forward_with_structure(conditioning)
            else:
                prediction, predicted_structure = model(conditioning), None
            losses = loss_function(
                prediction,
                target,
                mask,
                predicted_structure=predicted_structure,
            )
            if training:
                if not torch.isfinite(losses.total):
                    raise FloatingPointError(
                        f"non-finite training loss at batch {batch_index}; optimizer step aborted"
                    )
                losses.total.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                    )
                optimizer.step()

        totals["loss"] += float(losses.total.detach()) * batch_size
        for name, value in losses.components.items():
            totals[name] += float(value) * batch_size
        examples += batch_size
        batches += 1

        reconstruction = mask * conditioning[:, :1] + (1.0 - mask) * prediction.detach()
        absolute_error = torch.abs(reconstruction - target)
        squared_error = torch.square(reconstruction - target)
        for name, region in (("missing", 1.0 - mask), ("observed", mask)):
            totals[f"{name}_absolute_error_sum"] += float((absolute_error * region).sum())
            totals[f"{name}_squared_error_sum"] += float((squared_error * region).sum())
            totals[f"{name}_pixel_count"] += float(region.sum())

    if examples == 0:
        raise ValueError("loader produced no batches")
    result = {name: value / examples for name, value in totals.items()}
    for name in ("missing", "observed"):
        count = totals[f"{name}_pixel_count"]
        mae = totals[f"{name}_absolute_error_sum"] / count
        mse = totals[f"{name}_squared_error_sum"] / count
        result[f"{name}_mae"] = mae
        result[f"{name}_mse"] = mse
        result[f"{name}_psnr"] = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        for suffix in ("absolute_error_sum", "squared_error_sum", "pixel_count"):
            result.pop(f"{name}_{suffix}")
    result["examples"] = float(examples)
    result["batches"] = float(batches)
    return result
