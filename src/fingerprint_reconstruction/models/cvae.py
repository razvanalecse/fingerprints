"""Conditional variational autoencoder for probabilistic fingerprint completion."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionalVAE(nn.Module):
    """Global-latent CVAE with a condition encoder and exact observed pixels.

    The posterior is ``q(z | X,Y,M)`` and the sampling prior is standard normal.
    The decoder receives ``z`` together with a multi-scale encoding of ``[Y,M]``.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = (32, 64, 128, 256),
        latent_dim: int = 64,
        logvar_bounds: Tuple[float, float] = (-12.0, 8.0),
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least three positive widths")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if logvar_bounds[0] >= logvar_bounds[1]:
            raise ValueError("invalid log-variance bounds")
        self.latent_dim = int(latent_dim)
        self.logvar_bounds = tuple(float(value) for value in logvar_bounds)

        posterior = []
        source = 3
        for width in widths:
            posterior.append(ConvBlock(source, width, stride=2))
            source = width
        self.posterior_encoder = nn.Sequential(*posterior)
        self.posterior_pool = nn.AdaptiveAvgPool2d(1)
        self.posterior_mu = nn.Linear(widths[-1], latent_dim)
        self.posterior_logvar = nn.Linear(widths[-1], latent_dim)

        condition = [ConvBlock(2, widths[0])]
        condition.extend(
            ConvBlock(source, target, stride=2)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.condition_encoder = nn.ModuleList(condition)
        self.latent_projection = nn.Linear(latent_dim, widths[-1])
        self.bottleneck = ConvBlock(2 * widths[-1], widths[-1])
        self.decoders = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.decoders.append(ConvBlock(current + skip_width, skip_width))
            current = skip_width
        self.output = nn.Conv2d(widths[0], 1, 1)

    @staticmethod
    def encode_observation(observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((mask * (2.0 * observed - 1.0), mask), dim=1)

    def encode_posterior(
        self, target: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.encode_observation(observed, mask)
        hidden = self.posterior_encoder(torch.cat((2.0 * target - 1.0, condition), dim=1))
        pooled = self.posterior_pool(hidden).flatten(1)
        mu = self.posterior_mu(pooled)
        logvar = self.posterior_logvar(pooled).clamp(*self.logvar_bounds)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(
        self, z: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encode_observation(observed, mask)
        skips = []
        for encoder in self.condition_encoder:
            hidden = encoder(hidden)
            skips.append(hidden)
        latent = self.latent_projection(z)[:, :, None, None].expand_as(hidden)
        hidden = self.bottleneck(torch.cat((hidden, latent), dim=1))
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        raw = torch.sigmoid(self.output(hidden))
        return mask * observed + (1.0 - mask) * raw

    def forward(
        self, target: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_posterior(target, observed, mask)
        reconstruction = self.decode(self.reparameterize(mu, logvar), observed, mask)
        return reconstruction, mu, logvar

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = observed.new_zeros((observed.shape[0], self.latent_dim))
        return self.decode(z, observed, mask)

    def sample(
        self, observed: torch.Tensor, mask: torch.Tensor, *, num_samples: int
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch = observed.shape[0]
        z = torch.randn(batch * num_samples, self.latent_dim, device=observed.device)
        expanded_observed = observed.repeat_interleave(num_samples, dim=0)
        expanded_mask = mask.repeat_interleave(num_samples, dim=0)
        decoded = self.decode(z, expanded_observed, expanded_mask)
        return decoded.reshape(batch, num_samples, 1, *observed.shape[-2:])
