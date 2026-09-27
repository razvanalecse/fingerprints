"""Time-conditioned U-Net used by the pixel-space conditional DDPM."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.second = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.first(inputs)
        hidden = hidden + self.time(time_embedding)[:, :, None, None]
        return self.second(hidden) + self.skip(inputs)


class SpatialSelfAttention(nn.Module):
    """Residual self-attention over spatial positions with zero-init output."""

    def __init__(self, channels: int, *, heads: int = 4) -> None:
        super().__init__()
        if heads <= 0 or channels % heads:
            raise ValueError("attention heads must divide channels")
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        tokens = self.norm(inputs).flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return inputs + attended


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        *,
        channels: Sequence[int] = (32, 64, 128, 256),
        time_dim: int = 128,
        data_channels: int = 1,
        condition_channels: int = 1,
        normalized_condition: bool = False,
        multiscale_conditioning: bool = False,
        auxiliary_condition_channels: int = 0,
        middle_attention: bool = False,
        attention_heads: int = 4,
        upsampling_mode: str = "nearest",
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3:
            raise ValueError("at least three channel widths are required")
        if data_channels <= 0 or condition_channels <= 0:
            raise ValueError("data and condition channels must be positive")
        self.data_channels = int(data_channels)
        self.condition_channels = int(condition_channels)
        self.normalized_condition = bool(normalized_condition)
        self.multiscale_conditioning = bool(multiscale_conditioning)
        self.auxiliary_condition_channels = int(auxiliary_condition_channels)
        if self.auxiliary_condition_channels < 0:
            raise ValueError("auxiliary condition channels cannot be negative")
        self.middle_attention_enabled = bool(middle_attention)
        if upsampling_mode not in {"nearest", "bilinear"}:
            raise ValueError("upsampling mode must be nearest or bilinear")
        self.upsampling_mode = upsampling_mode
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.input = nn.Conv2d(
            self.data_channels + self.condition_channels + 1,
            widths[0],
            3,
            padding=1,
        )
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.condition_projections = nn.ModuleList()
        self.auxiliary_projections = nn.ModuleList()
        for index, width in enumerate(widths):
            # Each downsampler already maps to the next level's width.
            self.encoders.append(TimeResidualBlock(width, width, time_dim))
            if self.multiscale_conditioning:
                projection = nn.Conv2d(self.condition_channels + 1, width, 1)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.condition_projections.append(projection)
            if self.auxiliary_condition_channels:
                auxiliary_projection = nn.Conv2d(
                    self.auxiliary_condition_channels, width, 1
                )
                nn.init.zeros_(auxiliary_projection.weight)
                nn.init.zeros_(auxiliary_projection.bias)
                self.auxiliary_projections.append(auxiliary_projection)
            if index < len(widths) - 1:
                self.downsamples.append(nn.Conv2d(width, widths[index + 1], 4, stride=2, padding=1))
        self.middle = nn.ModuleList(
            [TimeResidualBlock(widths[-1], widths[-1], time_dim) for _ in range(2)]
        )
        self.middle_attention = (
            SpatialSelfAttention(widths[-1], heads=attention_heads)
            if self.middle_attention_enabled
            else nn.Identity()
        )
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.upsamples.append(nn.Conv2d(current, skip_width, 3, padding=1))
            self.decoders.append(TimeResidualBlock(2 * skip_width, skip_width, time_dim))
            current = skip_width
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], self.data_channels, 3, padding=1),
        )

    def forward(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        auxiliary_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xt.ndim != 4 or observed.ndim != 4 or mask.ndim != 4:
            raise ValueError("xt, observed, and mask must be BCHW tensors")
        if xt.shape[0] != observed.shape[0] or xt.shape[0] != mask.shape[0]:
            raise ValueError("xt, observed, and mask batch sizes must match")
        if xt.shape[-2:] != observed.shape[-2:] or xt.shape[-2:] != mask.shape[-2:]:
            raise ValueError("xt, observed, and mask spatial shapes must match")
        if xt.shape[1] != self.data_channels or observed.shape[1] != self.condition_channels:
            raise ValueError("xt or observed has an incompatible channel count")
        if mask.shape[1] != 1:
            raise ValueError("mask must contain one channel")
        if self.auxiliary_condition_channels:
            if auxiliary_condition is None:
                raise ValueError("this denoiser requires an auxiliary condition")
            if auxiliary_condition.ndim != 4 or auxiliary_condition.shape[:2] != (
                xt.shape[0], self.auxiliary_condition_channels
            ):
                raise ValueError("auxiliary condition has an incompatible shape")
        elif auxiliary_condition is not None:
            raise ValueError("auxiliary condition supplied to a denoiser that does not use it")
        time_embedding = self.time_embedding(timesteps)
        observed_encoded = mask * (
            observed if self.normalized_condition else 2.0 * observed - 1.0
        )
        conditioning = torch.cat((observed_encoded, mask), dim=1)
        hidden = self.input(torch.cat((xt, conditioning), dim=1))
        skips = []
        for index, encoder in enumerate(self.encoders):
            if self.multiscale_conditioning:
                scaled_condition = F.interpolate(
                    conditioning, size=hidden.shape[-2:], mode="nearest"
                )
                hidden = hidden + self.condition_projections[index](scaled_condition)
            if self.auxiliary_condition_channels:
                scaled_auxiliary = F.interpolate(
                    auxiliary_condition,
                    size=hidden.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                hidden = hidden + self.auxiliary_projections[index](scaled_auxiliary)
            hidden = encoder(hidden, time_embedding)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        for block in self.middle:
            hidden = block(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == "bilinear" else None,
            )
            hidden = upsample(hidden)
            hidden = decoder(torch.cat((hidden, skip), dim=1), time_embedding)
        return self.output(hidden)
