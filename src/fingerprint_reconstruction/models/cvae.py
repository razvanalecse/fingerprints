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
        upsampling_mode: str = "nearest",
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least three positive widths")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if logvar_bounds[0] >= logvar_bounds[1]:
            raise ValueError("invalid log-variance bounds")
        if upsampling_mode not in {"nearest", "bilinear"}:
            raise ValueError("upsampling mode must be nearest or bilinear")
        self.latent_dim = int(latent_dim)
        self.logvar_bounds = tuple(float(value) for value in logvar_bounds)
        self.upsampling_mode = upsampling_mode

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
            hidden = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == "bilinear" else None,
            )
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
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch = observed.shape[0]
        shape = (batch * num_samples, self.latent_dim)
        z = (
            torch.randn(shape, generator=generator, dtype=observed.dtype).to(observed.device)
            if generator is not None
            else torch.randn(shape, device=observed.device, dtype=observed.dtype)
        )
        expanded_observed = observed.repeat_interleave(num_samples, dim=0)
        expanded_mask = mask.repeat_interleave(num_samples, dim=0)
        decoded = self.decode(z, expanded_observed, expanded_mask)
        return decoded.reshape(batch, num_samples, 1, *observed.shape[-2:])

    def latent_scalar_count(self, observed: torch.Tensor) -> int:
        return self.latent_dim


class SpatialConditionalVAE(nn.Module):
    """CVAE with a per-location latent field instead of one global latent vector.

    A single global ``z`` forces the same "how much do I know here" budget onto
    every pixel, so predictive spread cannot track local reconstruction
    difficulty (confirmed empirically: sample std was statistically uniform
    across easy and hard regions for :class:`ConditionalVAE`). Here the latent
    is a ``[C, H/8, W/8]`` grid, matching the condition encoder's bottleneck
    resolution exactly, so each spatial cell has its own posterior and can
    express local uncertainty.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = (32, 64, 128, 256),
        latent_channels: int = 8,
        logvar_bounds: Tuple[float, float] = (-12.0, 8.0),
        upsampling_mode: str = "nearest",
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least three positive widths")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if logvar_bounds[0] >= logvar_bounds[1]:
            raise ValueError("invalid log-variance bounds")
        if upsampling_mode not in {"nearest", "bilinear"}:
            raise ValueError("upsampling mode must be nearest or bilinear")
        self.latent_channels = int(latent_channels)
        self.logvar_bounds = tuple(float(value) for value in logvar_bounds)
        self.upsampling_mode = upsampling_mode

        # Same downsampling schedule as the condition encoder (stride 1, then
        # stride 2 per remaining width) so both bottlenecks share one resolution.
        posterior = [ConvBlock(3, widths[0])]
        posterior.extend(
            ConvBlock(source, target, stride=2)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.posterior_encoder = nn.Sequential(*posterior)
        self.posterior_mu = nn.Conv2d(widths[-1], latent_channels, 1)
        self.posterior_logvar = nn.Conv2d(widths[-1], latent_channels, 1)

        condition = [ConvBlock(2, widths[0])]
        condition.extend(
            ConvBlock(source, target, stride=2)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.condition_encoder = nn.ModuleList(condition)
        self.latent_projection = nn.Conv2d(latent_channels, widths[-1], 1)
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
        mu = self.posterior_mu(hidden)
        logvar = self.posterior_logvar(hidden).clamp(*self.logvar_bounds)
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
        latent = self.latent_projection(z)
        if latent.shape[-2:] != hidden.shape[-2:]:
            raise ValueError("spatial latent resolution does not match bottleneck resolution")
        hidden = self.bottleneck(torch.cat((hidden, latent), dim=1))
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == "bilinear" else None,
            )
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        raw = torch.sigmoid(self.output(hidden))
        return mask * observed + (1.0 - mask) * raw

    def forward(
        self, target: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_posterior(target, observed, mask)
        reconstruction = self.decode(self.reparameterize(mu, logvar), observed, mask)
        return reconstruction, mu, logvar

    def _bottleneck_shape(self, observed: torch.Tensor) -> tuple[int, int]:
        height, width = observed.shape[-2:]
        downsample = 2 ** (len(self.condition_encoder) - 1)
        if height % downsample or width % downsample:
            raise ValueError(f"spatial dimensions must be divisible by {downsample}")
        return height // downsample, width // downsample

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rows, cols = self._bottleneck_shape(observed)
        z = observed.new_zeros((observed.shape[0], self.latent_channels, rows, cols))
        return self.decode(z, observed, mask)

    def sample(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch = observed.shape[0]
        rows, cols = self._bottleneck_shape(observed)
        shape = (batch * num_samples, self.latent_channels, rows, cols)
        z = (
            torch.randn(shape, generator=generator, dtype=observed.dtype).to(observed.device)
            if generator is not None
            else torch.randn(shape, device=observed.device, dtype=observed.dtype)
        )
        expanded_observed = observed.repeat_interleave(num_samples, dim=0)
        expanded_mask = mask.repeat_interleave(num_samples, dim=0)
        decoded = self.decode(z, expanded_observed, expanded_mask)
        return decoded.reshape(batch, num_samples, 1, *observed.shape[-2:])

    def latent_scalar_count(self, observed: torch.Tensor) -> int:
        rows, cols = self._bottleneck_shape(observed)
        return self.latent_channels * rows * cols
