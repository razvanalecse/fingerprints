"""RePaint forward resampling primitives.

The transition follows Algorithm 1 of Lugmayr et al. (CVPR 2022): after a
reverse transition from x_t to x_{t-1}, diffuse the result one Markov step
back to x_t before repeating the reverse transition.
"""

from __future__ import annotations

import torch

from fingerprint_reconstruction.models.diffusion.scheduler import DDPMScheduler


def repaint_forward_step(
    x_previous: torch.Tensor,
    step: int,
    scheduler: DDPMScheduler,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Sample q(x_t | x_{t-1}) for one RePaint resampling transition."""

    if not 0 < step < scheduler.timesteps:
        raise ValueError("step must be in [1, scheduler.timesteps - 1]")
    noise = torch.randn_like(x_previous) if noise is None else noise
    if noise.shape != x_previous.shape:
        raise ValueError("noise and x_previous must have identical shapes")
    alpha = scheduler.alphas[step].to(device=x_previous.device, dtype=x_previous.dtype)
    beta = scheduler.betas[step].to(device=x_previous.device, dtype=x_previous.dtype)
    return torch.sqrt(alpha) * x_previous + torch.sqrt(beta) * noise
