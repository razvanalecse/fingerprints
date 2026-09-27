"""ELBO-style objective for conditional fingerprint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from fingerprint_reconstruction.losses.reconstruction import (
    RegisteredApproximateLoss,
    masked_mean,
)


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


class RegisteredApproximateCVAELoss(CVAELoss):
    """CVAE ELBO reweighted by evaluation ROI and calibrated geometric confidence.

    ``target`` is a registered_approximate pseudo-target (a different impression
    warped into the latent frame), so the reconstruction term must be restricted
    to the evaluation ROI and downweighted by distance-from-anchor confidence,
    exactly like ``RegisteredApproximateLoss`` does for the deterministic model.
    """

    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        mse_weight: float = 0.25,
        free_nats_per_dimension: float = 0.01,
        orientation_weight: float = 0.0,
        orientation_window: int = 9,
        gradient_weight: float = 0.0,
        ridge_energy_weight: float = 0.0,
        ridge_sigma: float = 1.5,
        ridge_energy_window: int = 17,
        ridge_band_weight: float = 0.0,
        ridge_band_frequencies: tuple[float, ...] = (0.18, 0.22, 0.26, 0.30),
        ridge_band_orientations: int = 8,
        ridge_band_kernel_size: int = 17,
        ridge_band_pool_size: int = 9,
        ridge_spectrum_weight: float = 0.0,
        ridge_spectrum_temperature: float = 0.25,
        support_background_weight: float = 0.0,
        support_region_weighting: bool = False,
    ) -> None:
        super().__init__(
            l1_weight=l1_weight,
            mse_weight=mse_weight,
            free_nats_per_dimension=free_nats_per_dimension,
        )
        self.registered_reconstruction = RegisteredApproximateLoss(
            missing_l1_weight=l1_weight,
            missing_mse_weight=mse_weight,
            observed_l1_weight=0.0,
            orientation_weight=orientation_weight,
            orientation_window=orientation_window,
            gradient_weight=gradient_weight,
            ridge_energy_weight=ridge_energy_weight,
            ridge_sigma=ridge_sigma,
            ridge_energy_window=ridge_energy_window,
            ridge_band_weight=ridge_band_weight,
            ridge_band_frequencies=ridge_band_frequencies,
            ridge_band_orientations=ridge_band_orientations,
            ridge_band_kernel_size=ridge_band_kernel_size,
            ridge_band_pool_size=ridge_band_pool_size,
            ridge_spectrum_weight=ridge_spectrum_weight,
            ridge_spectrum_temperature=ridge_spectrum_temperature,
            support_background_weight=support_background_weight,
            support_region_weighting=support_region_weighting,
        )

    def forward(  # type: ignore[override]
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        beta: float,
        evaluation_roi: torch.Tensor,
        geometric_confidence: torch.Tensor,
        observed: torch.Tensor | None = None,
        support_probability: torch.Tensor | None = None,
    ) -> CVAELossOutput:
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if mu.shape != logvar.shape or mu.shape[0] != reconstruction.shape[0]:
            raise ValueError("mu and logvar must have equal shape and matching batch size")
        if observed is None:
            observed = reconstruction.detach()
        reconstruction_output = self.registered_reconstruction(
            reconstruction,
            target,
            mask,
            observed=observed,
            evaluation_roi=evaluation_roi,
            geometric_confidence=geometric_confidence,
            support_probability=support_probability,
        )
        kl_per_dimension = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
        free_kl = torch.clamp(kl_per_dimension, min=self.free_nats_per_dimension)
        kl = free_kl.mean()
        kl_total_nats = free_kl.flatten(1).sum(1).mean()
        total = reconstruction_output.total + float(beta) * kl
        return CVAELossOutput(
            total=total,
            components={
                **reconstruction_output.components,
                # Backward-compatible aliases used by the existing CVAE
                # reports; registered_* remains the semantically precise name.
                "missing_l1": reconstruction_output.components["registered_l1"],
                "missing_mse": reconstruction_output.components["registered_mse"],
                "kl": kl.detach(),
                "kl_total_nats": kl_total_nats.detach(),
            },
        )
