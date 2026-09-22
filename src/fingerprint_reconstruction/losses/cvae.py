"""ELBO-style objective for conditional fingerprint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from fingerprint_reconstruction.losses.reconstruction import masked_mean


@dataclass(frozen=True)
class CVAELossOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]


class CVAELoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        mse_weight: float = 0.25,
        free_nats_per_dimension: float = 0.01,
    ) -> None:
        super().__init__()
        if min(l1_weight, mse_weight, free_nats_per_dimension) < 0:
            raise ValueError("loss weights and free nats must be non-negative")
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.free_nats_per_dimension = float(free_nats_per_dimension)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        beta: float,
    ) -> CVAELossOutput:
        missing = 1.0 - mask
        error = reconstruction - target
        l1 = masked_mean(error.abs(), missing)
        mse = masked_mean(error.square(), missing)
        kl_per_dimension = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
        kl = torch.clamp(
            kl_per_dimension.mean(dim=0), min=self.free_nats_per_dimension
        ).mean()
        total = self.l1_weight * l1 + self.mse_weight * mse + float(beta) * kl
        return CVAELossOutput(
            total=total,
            components={"missing_l1": l1.detach(), "missing_mse": mse.detach(), "kl": kl.detach()},
        )
