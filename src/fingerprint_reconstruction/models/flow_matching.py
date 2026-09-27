"""Conditional residual flow matching: a straight-line alternative to DDPM sampling.

x_t = (1-t) x_0 + t x_1, with x_0 = X_coarse + sigma_0*eps (a noised coarse
reconstruction) and x_1 = X (the target). The network learns the constant
velocity v = x_1 - x_0 of this straight-line path, conditioned the same way
as this project's pixel DDPM (observed, mask, optional auxiliary support
channel). Reuses `DiffusionUNet` unchanged -- the sinusoidal time embedding
accepts any float tensor, so a continuous t in [0,1] is fed in directly
(scaled to roughly DDPM's usual timestep magnitude for embedding stability),
with no architecture change needed.

Motivated by rectified-flow/flow-matching literature (e.g. Liu et al.'s
Rectified Flow, and residual-restoration variants such as ResFlow/FlowIE)
reporting good restoration quality in as few as 4-10 sampling steps, versus
DDIM's 20+ or full ancestral DDPM's 500 -- a genuinely different sampling
paradigm from every diffusion variant already built in this project, not
just a different noise schedule.
"""

from __future__ import annotations

import torch
from torch import nn

from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet

TIME_SCALE = 1000.0  # matches this project's DDPM timestep magnitude for embedding stability


class ConditionalFlowMatching(nn.Module):
    def __init__(self, denoiser: DiffusionUNet, *, sigma_0: float = 0.05) -> None:
        super().__init__()
        if sigma_0 < 0:
            raise ValueError("sigma_0 must be non-negative")
        self.denoiser = denoiser
        self.sigma_0 = float(sigma_0)

    @staticmethod
    def _randn_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(reference)
        # A CPU generator provides identical seeded streams for CPU and MPS.
        return torch.randn(reference.shape, generator=generator, dtype=reference.dtype).to(reference.device)

    def training_loss(
        self,
        target: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coarse: torch.Tensor,
        *,
        auxiliary_condition: torch.Tensor | None = None,
        missing_weight_map: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if target.shape != coarse.shape:
            raise ValueError("target and coarse must have identical shapes")
        epsilon = self._randn_like(target, generator)
        x0 = coarse + self.sigma_0 * epsilon
        x1 = target
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t_broadcast = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1
        velocity_target = x1 - x0
        predicted_velocity = self.denoiser(xt, t * TIME_SCALE, observed, mask, auxiliary_condition)
        squared_error = (predicted_velocity - velocity_target).square()
        if missing_weight_map is None:
            return squared_error.mean()
        if missing_weight_map.shape != target.shape:
            raise ValueError("missing_weight_map must have the target shape")
        weighted = squared_error * missing_weight_map
        return weighted.sum() / missing_weight_map.sum().clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coarse: torch.Tensor,
        *,
        num_samples: int = 1,
        steps: int = 6,
        auxiliary_condition: torch.Tensor | None = None,
        enforce_data_consistency: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler-integrate dx/dt = v_theta from t=0 (noised coarse) to t=1 (target)."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = observed.shape[0]
        observed_expanded = observed.repeat_interleave(num_samples, dim=0)
        mask_expanded = mask.repeat_interleave(num_samples, dim=0)
        coarse_expanded = coarse.repeat_interleave(num_samples, dim=0)
        auxiliary_expanded = (
            auxiliary_condition.repeat_interleave(num_samples, dim=0)
            if auxiliary_condition is not None
            else None
        )
        epsilon = self._randn_like(coarse_expanded, generator)
        x = coarse_expanded + self.sigma_0 * epsilon
        dt = 1.0 / steps
        for step in range(steps):
            t_value = step / steps
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
            velocity = self.denoiser(x, t * TIME_SCALE, observed_expanded, mask_expanded, auxiliary_expanded)
            x = x + velocity * dt
        x = x.clamp(0.0, 1.0)
        if enforce_data_consistency:
            x = mask_expanded * observed_expanded + (1.0 - mask_expanded) * x
        return x.reshape(batch, num_samples, *x.shape[1:])
