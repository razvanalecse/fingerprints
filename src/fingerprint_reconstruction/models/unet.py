"""Deterministic U-Net baseline conditioned on observed pixels and mask."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        normalization_groups = min(groups, out_channels)
        while out_channels % normalization_groups:
            normalization_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(normalization_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(normalization_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class FingerprintUNet(nn.Module):
    """U-Net mapping ``[Y, M]`` to a deterministic complete fingerprint.

    ``forward`` returns the unconstrained prediction. ``reconstruct`` additionally
    enforces exact agreement with every observed pixel.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        neutral_missing_encoding: bool = False,
        upsampling_mode: str = "transpose",
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("fingerprint conditioning requires exactly [Y, M]")
        if out_channels != 1:
            raise ValueError("grayscale reconstruction requires one output channel")
        if len(channels) < 2 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain at least two positive widths")

        widths = tuple(int(channel) for channel in channels)
        self.neutral_missing_encoding = bool(neutral_missing_encoding)
        if upsampling_mode not in {"transpose", "resize_conv"}:
            raise ValueError("upsampling_mode must be 'transpose' or 'resize_conv'")
        self.upsampling_mode = upsampling_mode
        encoder_inputs = (in_channels,) + widths[:-1]
        self.encoders = nn.ModuleList(
            DoubleConv(input_width, output_width)
            for input_width, output_width in zip(encoder_inputs, widths)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)

        decoder_input = widths[-1] * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_width in reversed(widths):
            if upsampling_mode == "transpose":
                upsampler = nn.ConvTranspose2d(
                    decoder_input, skip_width, kernel_size=2, stride=2
                )
            else:
                # Resize-convolution avoids uneven kernel overlap and the
                # checkerboard textures that can game orientation objectives.
                upsampler = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                    nn.Conv2d(decoder_input, skip_width, kernel_size=3, padding=1),
                )
            self.upconvs.append(upsampler)
            self.decoders.append(DoubleConv(2 * skip_width, skip_width))
            decoder_input = skip_width
        self.output = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 4 or conditioning.shape[1] != 2:
            raise ValueError("conditioning must have shape [B, 2, H, W]")
        observed, mask = conditioning[:, :1], conditioning[:, 1:2]
        if self.neutral_missing_encoding:
            # Known intensities span [-1, 1], while every unknown pixel is 0.
            # The separate mask channel keeps 0-valued known intensities
            # distinguishable from missing support.
            observed = mask * (2.0 * observed - 1.0)
            conditioning = torch.cat((observed, mask), dim=1)
        skips = []
        hidden = conditioning
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = self.pool(hidden)
        hidden = self.bottleneck(hidden)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            hidden = upconv(hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((skip, hidden), dim=1))
        return torch.sigmoid(self.output(hidden))

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must both have shape [B, 1, H, W]")
        prediction = self(torch.cat((observed, mask), dim=1))
        return mask * observed + (1.0 - mask) * prediction
