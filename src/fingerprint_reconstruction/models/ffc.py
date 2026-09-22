"""Gated encoder-decoder with fast-Fourier-convolution context blocks."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.models.gated_conv import GatedConvBlock


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SpectralTransform(nn.Module):
    """Learn image-wide context through a real FFT and inverse FFT."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        half = max(channels // 2, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, half, 1, bias=False), _norm(half), nn.SiLU()
        )
        self.spectrum = nn.Sequential(
            nn.Conv2d(2 * half, 2 * half, 1, bias=False), _norm(2 * half), nn.SiLU()
        )
        self.expand = nn.Conv2d(half, channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(inputs)
        height, width = reduced.shape[-2:]
        spectrum = torch.fft.rfft2(reduced.float(), norm="ortho")
        stacked = self.spectrum(torch.cat((spectrum.real, spectrum.imag), dim=1))
        real, imaginary = torch.chunk(stacked, 2, dim=1)
        restored = torch.fft.irfft2(
            torch.complex(real, imaginary), s=(height, width), norm="ortho"
        )
        return self.expand(reduced + restored.to(reduced.dtype))


class FFCLayer(nn.Module):
    def __init__(self, local_channels: int, global_channels: int) -> None:
        super().__init__()
        self.local_to_local = nn.Conv2d(local_channels, local_channels, 3, padding=1, bias=False)
        self.global_to_local = nn.Conv2d(global_channels, local_channels, 3, padding=1, bias=False)
        self.local_to_global = nn.Conv2d(local_channels, global_channels, 3, padding=1, bias=False)
        self.global_to_global = SpectralTransform(global_channels)
        self.norm_local = _norm(local_channels)
        self.norm_global = _norm(global_channels)

    def forward(self, local: torch.Tensor, global_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_local = self.local_to_local(local) + self.global_to_local(global_)
        new_global = self.local_to_global(local) + self.global_to_global(global_)
        return F.silu(self.norm_local(new_local)), F.silu(self.norm_global(new_global))


class FFCResidualBlock(nn.Module):
    def __init__(self, channels: int, global_ratio: float = 0.5) -> None:
        super().__init__()
        if not 0 < global_ratio < 1:
            raise ValueError("global_ratio must lie strictly inside (0, 1)")
        self.global_channels = max(int(round(channels * global_ratio)), 1)
        self.local_channels = channels - self.global_channels
        if self.local_channels < 1:
            raise ValueError("global_ratio leaves no local channels")
        self.first = FFCLayer(self.local_channels, self.global_channels)
        self.second = FFCLayer(self.local_channels, self.global_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local, global_ = torch.split(
            inputs, (self.local_channels, self.global_channels), dim=1
        )
        new_local, new_global = self.second(*self.first(local, global_))
        return inputs + torch.cat((new_local, new_global), dim=1)


class FFCFingerprintNetwork(nn.Module):
    """Mask-conditioned model with local and global spectral feature streams."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        *,
        bottleneck_blocks: int = 6,
        mid_blocks: int = 2,
        global_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least three positive widths")
        encoders = [GatedConvBlock(2, widths[0], kernel_size=5)]
        encoders.extend(
            GatedConvBlock(source, target, kernel_size=5, stride=2)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.encoders = nn.ModuleList(encoders)
        self.context = nn.Sequential(
            *[FFCResidualBlock(widths[-1], global_ratio) for _ in range(bottleneck_blocks)]
        )
        self.decoders = nn.ModuleList()
        self.mid_context = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.decoders.append(GatedConvBlock(current + skip_width, skip_width))
            self.mid_context.append(
                nn.Sequential(
                    *[FFCResidualBlock(skip_width, global_ratio) for _ in range(mid_blocks)]
                )
                if skip_width == widths[-2] and mid_blocks > 0
                else nn.Identity()
            )
            current = skip_width
        self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 4 or conditioning.shape[1] != 2:
            raise ValueError("conditioning must have shape [B, 2, H, W]")
        observed, mask = conditioning[:, :1], conditioning[:, 1:2]
        hidden = torch.cat((mask * (2.0 * observed - 1.0), mask), dim=1)
        skips = []
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
        hidden = self.context(hidden)
        for decoder, mid, skip in zip(
            self.decoders, self.mid_context, reversed(skips[:-1])
        ):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = mid(decoder(torch.cat((hidden, skip), dim=1)))
        return torch.sigmoid(self.output(hidden))

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must both have shape [B, 1, H, W]")
        raw = self(torch.cat((observed, mask), dim=1))
        return mask * observed + (1.0 - mask) * raw
