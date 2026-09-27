"""Brownian Bridge Diffusion (BBDM): a direct stochastic bridge from Y to X.

Unlike standard DDPM (which bridges pure noise to the target) or this
project's flow-matching variant (a deterministic straight-line path), BBDM
constructs a genuine Brownian bridge between the *degraded/masked input*
itself and the target, per Li et al.'s formulation:

    x_t = (1 - m_t) X + m_t Y_filled + sqrt(delta_t) eps,

with m_0=0, m_1=1 (so x_0=X, x_1=Y_filled), and a variance schedule
delta_t = 2 s^2 t (1-t) that vanishes at both endpoints (the bridge is exact
at t=0 and t=1, with maximal injected noise at the midpoint). The network
predicts the injected eps; sampling reconstructs an x_0 estimate at each
step and re-applies the same bridge formula at a smaller t (a deterministic,
DDIM-style generalization -- reusing the predicted eps across steps rather
than the full stochastic reverse posterior), consistent with this project's
other few/reduced-step samplers.

Per the plan that proposed this model: it should be validated on data with
exact ground truth (Track A / Track E) before being trained on real,
approximately-registered latent/exemplar pairs, since a bridge that starts
from Y itself is more sensitive to Y being a genuinely correct anchor than
DDPM (which only conditions on Y, rather than starting the trajectory there).
"""

from __future__ import annotations

import torch
from torch import nn

from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet

TIME_SCALE = 1000.0


def bridge_schedule(t: torch.Tensor, *, s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (m_t, delta_t) for m_t=t, delta_t = 2 s^2 t (1-t)."""

    m_t = t
    delta_t = 2.0 * (s**2) * t * (1.0 - t)
    return m_t, delta_t


class ConditionalBrownianBridge(nn.Module):
    def __init__(self, denoiser: DiffusionUNet, *, s: float = 0.3, epsilon: float = 0.05) -> None:
        super().__init__()
        if s <= 0:
            raise ValueError("s must be positive")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must lie in (0, 0.5)")
        self.denoiser = denoiser
        self.s = float(s)
        self.epsilon = float(epsilon)

    @staticmethod
    def _randn_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(reference)
        # A CPU generator provides identical seeded streams for CPU and MPS.
        return torch.randn(reference.shape, generator=generator, dtype=reference.dtype).to(reference.device)

    def _sample_t(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Keep t away from the exact endpoints, where delta_t -> 0 and the
        # bridge marginal degenerates (division by near-zero downstream).
        return self.epsilon + (1.0 - 2.0 * self.epsilon) * torch.rand(batch, device=device, dtype=dtype)

    def training_loss(
        self,
        target: torch.Tensor,
        observed_filled: torch.Tensor,
        mask: torch.Tensor,
        *,
        auxiliary_condition: torch.Tensor | None = None,
        missing_weight_map: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if target.shape != observed_filled.shape:
            raise ValueError("target and observed_filled must have identical shapes")
        batch = target.shape[0]
        t = self._sample_t(batch, target.device, target.dtype)
        m_t, delta_t = bridge_schedule(t, s=self.s)
        m_t = m_t.view(-1, 1, 1, 1)
        delta_t = delta_t.view(-1, 1, 1, 1)
        epsilon = self._randn_like(target, generator)
        xt = (1.0 - m_t) * target + m_t * observed_filled + delta_t.sqrt() * epsilon
        predicted_epsilon = self.denoiser(xt, t * TIME_SCALE, observed_filled, mask, auxiliary_condition)
        squared_error = (predicted_epsilon - epsilon).square()
        if missing_weight_map is None:
            return squared_error.mean()
        if missing_weight_map.shape != target.shape:
            raise ValueError("missing_weight_map must have the target shape")
        weighted = squared_error * missing_weight_map
        return weighted.sum() / missing_weight_map.sum().clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        observed_filled: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_samples: int = 1,
        steps: int = 8,
        auxiliary_condition: torch.Tensor | None = None,
        enforce_data_consistency: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Deterministic, DDIM-style bridge sampling from t=1 (Y) down to t=0 (X)."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = observed_filled.shape[0]
        observed_expanded = observed_filled.repeat_interleave(num_samples, dim=0)
        mask_expanded = mask.repeat_interleave(num_samples, dim=0)
        auxiliary_expanded = (
            auxiliary_condition.repeat_interleave(num_samples, dim=0) if auxiliary_condition is not None else None
        )
        x = observed_expanded.clone()  # start exactly at t=1 (x_1 = Y_filled, delta_1=0)
        schedule = torch.linspace(1.0 - self.epsilon, self.epsilon, steps + 1)
        for index in range(steps):
            t_current, t_next = schedule[index], schedule[index + 1]
            t_tensor = torch.full((x.shape[0],), float(t_current), device=x.device, dtype=x.dtype)
            m_current, delta_current = bridge_schedule(t_tensor, s=self.s)
            predicted_epsilon = self.denoiser(x, t_tensor * TIME_SCALE, observed_expanded, mask_expanded, auxiliary_expanded)
            m_current_b, delta_current_b = m_current.view(-1, 1, 1, 1), delta_current.view(-1, 1, 1, 1)
            x0_hat = (x - m_current_b * observed_expanded - delta_current_b.sqrt() * predicted_epsilon) / (
                1.0 - m_current_b
            ).clamp_min(1e-4)
            t_next_tensor = torch.full((x.shape[0],), float(t_next), device=x.device, dtype=x.dtype)
            m_next, delta_next = bridge_schedule(t_next_tensor, s=self.s)
            m_next_b, delta_next_b = m_next.view(-1, 1, 1, 1), delta_next.view(-1, 1, 1, 1)
            x = (1.0 - m_next_b) * x0_hat + m_next_b * observed_expanded + delta_next_b.sqrt() * predicted_epsilon
        x = x.clamp(0.0, 1.0)
        if enforce_data_consistency:
            x = mask_expanded * observed_expanded + (1.0 - mask_expanded) * x
        return x.reshape(batch, num_samples, *x.shape[1:])
