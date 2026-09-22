"""Fingerprint-aware objective for the first stage of latent diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.latent_diffusion import LatentDistribution


@dataclass(frozen=True)
class AutoencoderLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class AutoencoderKLLoss(nn.Module):
    def __init__(
        self,
        *,
        kl_weight: float = 1e-6,
        l1_weight: float = 1.0,
        mse_weight: float = 0.25,
        orientation_weight: float = 0.02,
        gradient_weight: float = 0.5,
        ridge_energy_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        self.kl_weight = float(kl_weight)
        self.reconstruction = MaskedReconstructionLoss(
            missing_l1_weight=l1_weight,
            missing_mse_weight=mse_weight,
            observed_l1_weight=0.0,
            orientation_weight=orientation_weight,
            gradient_weight=gradient_weight,
            ridge_energy_weight=ridge_energy_weight,
        )

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        distribution: LatentDistribution,
    ) -> AutoencoderLossOutput:
        full_missing_mask = torch.zeros_like(target)
        reconstruction_result = self.reconstruction(
            reconstruction, target, full_missing_mask
        )
        kl = distribution.kl()
        total = reconstruction_result.total + self.kl_weight * kl
        components = dict(reconstruction_result.components)
        components["kl"] = kl.detach()
        return AutoencoderLossOutput(total=total, components=components)
