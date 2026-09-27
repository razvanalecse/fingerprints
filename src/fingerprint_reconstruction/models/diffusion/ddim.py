"""Explicit DDIM transitions for accelerated conditional diffusion sampling."""

from __future__ import annotations

import torch


def make_ddim_timesteps(training_timesteps: int, inference_steps: int) -> torch.Tensor:
    """Return a strictly decreasing subsequence containing T-1 and 0.

    Uniform spacing is performed in the original training-time index space.
    The denoiser therefore always receives timestep labels on which it was
    trained; no schedule is silently reparameterized.
    """

    if training_timesteps < 2:
        raise ValueError("training_timesteps must be at least two")
    if not 2 <= inference_steps <= training_timesteps:
        raise ValueError("inference_steps must be in [2, training_timesteps]")
    ascending = torch.linspace(
        0, training_timesteps - 1, inference_steps, dtype=torch.float64
    ).round().long()
    if torch.unique_consecutive(ascending).numel() != inference_steps:
        raise RuntimeError("rounded DDIM schedule contains duplicate timesteps")
    return ascending.flip(0)


def ddim_step(
    xt: torch.Tensor,
    predicted_x0: torch.Tensor,
    predicted_noise: torch.Tensor,
    *,
    alpha_bar_t: torch.Tensor,
    alpha_bar_previous: torch.Tensor,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute one DDIM update from timestep t to an arbitrary earlier step.

    The update is

        x_s = sqrt(a_s) x0_hat
              + sqrt(1-a_s-sigma^2) epsilon_theta + sigma z,

    where ``a_t`` and ``a_s`` are cumulative alpha products and

        sigma = eta sqrt((1-a_s)/(1-a_t) * (1-a_t/a_s)).

    ``eta=0`` gives deterministic DDIM conditional on the initial noise.
    """

    if eta < 0:
        raise ValueError("eta must be non-negative")
    if xt.shape != predicted_x0.shape or xt.shape != predicted_noise.shape:
        raise ValueError("xt, predicted_x0, and predicted_noise must have equal shapes")
    alpha_t = torch.as_tensor(alpha_bar_t, device=xt.device, dtype=xt.dtype)
    alpha_previous = torch.as_tensor(
        alpha_bar_previous, device=xt.device, dtype=xt.dtype
    )
    if torch.any(alpha_t <= 0) or torch.any(alpha_previous <= 0):
        raise ValueError("cumulative alphas must be positive")
    sigma = float(eta) * torch.sqrt(
        ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp_min(0.0)
        * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
    )
    direction_scale = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0))
    result = torch.sqrt(alpha_previous) * predicted_x0 + direction_scale * predicted_noise
    if eta > 0:
        result = result + sigma * (torch.randn_like(xt) if noise is None else noise)
    return result
