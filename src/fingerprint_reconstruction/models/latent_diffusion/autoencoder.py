"""Spatial KL autoencoder for fingerprint-aware latent diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


@dataclass(frozen=True)
class LatentDistribution:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return -0.5 * torch.mean(
            1.0 + self.logvar - self.mean.square() - self.logvar.exp()
        )


class FingerprintAutoencoderKL(nn.Module):
    """Moderately compressed spatial latent representation.

    With two channel stages, a 128x128 image becomes a 32x32 latent map. This
    conservative f=4 compression is intentional: fingerprint ridges are fine,
    phase-sensitive structures and must be validated before stronger f=8
    compression is attempted.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = (32, 64),
        latent_channels: int = 4,
        logvar_bounds: tuple[float, float] = (-12.0, 8.0),
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if not widths or any(value <= 0 for value in widths):
            raise ValueError("channels must contain positive widths")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if logvar_bounds[0] >= logvar_bounds[1]:
            raise ValueError("invalid log-variance bounds")
        self.latent_channels = int(latent_channels)
        self.compression_factor = 2 ** len(widths)
        self.logvar_bounds = tuple(float(value) for value in logvar_bounds)

        encoder = []
        source = 1
        for width in widths:
            encoder.extend(
                (
                    nn.Conv2d(source, width, 3, stride=2, padding=1),
                    ResidualBlock(width),
                    ResidualBlock(width),
                )
            )
            source = width
        self.encoder = nn.Sequential(*encoder)
        self.to_moments = nn.Conv2d(widths[-1], 2 * latent_channels, 1)

        self.from_latent = nn.Conv2d(latent_channels, widths[-1], 3, padding=1)
        decoder = []
        source = widths[-1]
        for width in reversed(widths):
            decoder.extend(
                (
                    ResidualBlock(source),
                    ResidualBlock(source),
                    nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                    nn.Conv2d(source, width, 3, padding=1),
                )
            )
            source = width
        self.decoder = nn.Sequential(*decoder)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], 1, 3, padding=1),
            nn.Tanh(),
        )

    def encode_distribution(self, image: torch.Tensor) -> LatentDistribution:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must have shape [B,1,H,W]")
        if image.shape[-2] % self.compression_factor or image.shape[-1] % self.compression_factor:
            raise ValueError("spatial dimensions must be divisible by compression_factor")
        moments = self.to_moments(self.encoder(2.0 * image - 1.0))
        mean, logvar = moments.chunk(2, dim=1)
        return LatentDistribution(mean, logvar.clamp(*self.logvar_bounds))

    def encode(self, image: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        distribution = self.encode_distribution(image)
        return distribution.sample() if sample else distribution.mode()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.latent_channels:
            raise ValueError("latent has incompatible shape")
        decoded = self.output(self.decoder(self.from_latent(latent)))
        return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

    def forward(
        self, image: torch.Tensor, *, sample_posterior: bool = True
    ) -> tuple[torch.Tensor, LatentDistribution]:
        distribution = self.encode_distribution(image)
        latent = distribution.sample() if sample_posterior else distribution.mode()
        reconstruction = self.decode(latent)
        if reconstruction.shape[-2:] != image.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        return reconstruction, distribution
