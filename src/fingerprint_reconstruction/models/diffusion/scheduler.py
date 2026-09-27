"""Transparent DDPM forward and reverse-process coefficients."""

from __future__ import annotations

import math

import torch
from torch import nn


def make_beta_schedule(name: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if timesteps < 2:
        raise ValueError("timesteps must be at least two")
    if name == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif name == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        s = 0.008
        alpha_bar = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()
    else:
        raise ValueError(f"unsupported beta schedule: {name}")
    if torch.any((betas <= 0) | (betas >= 1)):
        raise ValueError("all betas must lie strictly in (0,1)")
    return betas


def extract(coefficients: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    values = coefficients.gather(0, timesteps)
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        *,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
        posterior_variance = betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        self.timesteps = int(timesteps)
        for name, value in (
            ("betas", betas),
            ("alphas", alphas),
            ("alpha_bar", alpha_bar),
            ("alpha_bar_previous", alpha_bar_previous),
            ("sqrt_alpha_bar", torch.sqrt(alpha_bar)),
            ("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar)),
            ("posterior_variance", posterior_variance.clamp_min(1e-20)),
            ("posterior_mean_coef_x0", betas * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar)),
            ("posterior_mean_coef_xt", (1.0 - alpha_bar_previous) * torch.sqrt(alphas) / (1.0 - alpha_bar)),
        ):
            self.register_buffer(name, value)

    def q_sample(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0) if noise is None else noise
        xt = (
            extract(self.sqrt_alpha_bar, timesteps, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bar, timesteps, x0.shape) * noise
        )
        return xt, noise

    def predict_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return (
            xt - extract(self.sqrt_one_minus_alpha_bar, timesteps, xt.shape) * epsilon
        ) / extract(self.sqrt_alpha_bar, timesteps, xt.shape)

    def posterior(
        self, xt: torch.Tensor, timesteps: torch.Tensor, predicted_x0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            extract(self.posterior_mean_coef_x0, timesteps, xt.shape) * predicted_x0
            + extract(self.posterior_mean_coef_xt, timesteps, xt.shape) * xt
        )
        variance = extract(self.posterior_variance, timesteps, xt.shape)
        return mean, variance
