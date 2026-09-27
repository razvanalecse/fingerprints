"""Gated-convolution encoder-decoder for deterministic fingerprint inpainting."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor


class GatedConvBlock(nn.Module):
    """Learn features and a spatial/channel gate from the same input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.feature = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation
        )
        self.gate = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation
        )
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.normalization = nn.GroupNorm(groups, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.normalization(self.feature(inputs))
        return F.silu(features) * torch.sigmoid(self.gate(inputs))


class GatedFingerprintNetwork(nn.Module):
    """Mask-conditioned gated encoder-decoder with exact data consistency."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        *,
        auxiliary_channels: int = 0,
        upsampling_mode: str = "nearest",
        refine_blocks: int = 0,
    ) -> None:
        super().__init__()
        if upsampling_mode not in {"nearest", "bilinear"}:
            raise ValueError("upsampling_mode must be 'nearest' or 'bilinear'")
        if refine_blocks < 0:
            raise ValueError("refine_blocks cannot be negative")
        self.upsampling_mode = upsampling_mode
        widths = tuple(int(value) for value in channels)
        if len(widths) < 3 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least three positive widths")
        if auxiliary_channels < 0:
            raise ValueError("auxiliary_channels cannot be negative")
        self.auxiliary_channels = int(auxiliary_channels)
        encoders = [GatedConvBlock(2 + self.auxiliary_channels, widths[0], kernel_size=5)]
        encoders.extend(
            GatedConvBlock(source, target, kernel_size=5, stride=2)
            for source, target in zip(widths[:-1], widths[1:])
        )
        self.encoders = nn.ModuleList(encoders)
        self.auxiliary_projections = nn.ModuleList()
        if self.auxiliary_channels:
            for width in widths:
                projection = nn.Conv2d(self.auxiliary_channels, width, 1)
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                self.auxiliary_projections.append(projection)
        self.context = nn.Sequential(
            GatedConvBlock(widths[-1], widths[-1], dilation=2),
            GatedConvBlock(widths[-1], widths[-1], dilation=4),
        )
        self.decoders = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.decoders.append(GatedConvBlock(current + skip_width, skip_width))
            current = skip_width
        self.refine = nn.Sequential(
            *[GatedConvBlock(widths[0], widths[0]) for _ in range(int(refine_blocks))]
        )
        self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

    @staticmethod
    def _encode_condition(
        observed: torch.Tensor, mask: torch.Tensor, auxiliary: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Missing support is neutral (0), known black/white intensities map to
        # [-1, 1], and M remains an explicit second channel.
        values = (mask * (2.0 * observed - 1.0), mask)
        return torch.cat(values + (() if auxiliary is None else (auxiliary,)), dim=1)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        expected_channels = 2 + self.auxiliary_channels
        if conditioning.ndim != 4 or conditioning.shape[1] != expected_channels:
            raise ValueError(f"conditioning must have shape [B, {expected_channels}, H, W]")
        observed, mask = conditioning[:, :1], conditioning[:, 1:2]
        auxiliary = conditioning[:, 2:] if self.auxiliary_channels else None
        hidden = self._encode_condition(observed, mask, auxiliary)
        skips = []
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            if self.auxiliary_channels:
                scaled_auxiliary = F.interpolate(
                    auxiliary,
                    size=hidden.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                hidden = hidden + self.auxiliary_projections[index](scaled_auxiliary)
            skips.append(hidden)
        hidden = self.context(hidden)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            # Nearest-neighbour resize followed by a learned gated convolution
            # avoids transposed-convolution overlap and is deterministic on MPS.
            if self.upsampling_mode == "bilinear":
                hidden = F.interpolate(
                    hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        if len(self.refine):
            hidden = hidden + self.refine(hidden)
        return torch.sigmoid(self.output(hidden))

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must both have shape [B, 1, H, W]")
        raw = self(torch.cat((observed, mask), dim=1))
        return mask * observed + (1.0 - mask) * raw


class StructureConditionedGatedNetwork(nn.Module):
    """Gated coarse reconstruction conditioned on predicted ridge structure.

    The structure branch sees only ``(Y,M)`` and is frozen. Consequently no
    complete-image target can leak into inference conditioning.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        *,
        structure_channels: Sequence[int] = (16, 32, 64, 128),
        upsampling_mode: str = "nearest",
        refine_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.structure_predictor = FingerprintStructurePredictor(
            channels=structure_channels
        ).requires_grad_(False)
        self.coarse = GatedFingerprintNetwork(
            channels=channels,
            auxiliary_channels=4,
            upsampling_mode=upsampling_mode,
            refine_blocks=refine_blocks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.structure_predictor.eval()
        return self

    def _conditioning(
        self, observed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            structure = self.structure_predictor(observed, mask)
            structure = F.interpolate(
                structure,
                size=observed.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat((observed, mask, structure), dim=1), structure

    def forward_with_structure(
        self, conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conditioning.ndim != 4 or conditioning.shape[1] != 2:
            raise ValueError("conditioning must have shape [B,2,H,W]")
        observed, mask = conditioning[:, :1], conditioning[:, 1:2]
        augmented, structure = self._conditioning(observed, mask)
        return self.coarse(augmented), structure

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 4 or conditioning.shape[1] != 2:
            raise ValueError("conditioning must have shape [B,2,H,W]")
        return self.forward_with_structure(conditioning)[0]

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        augmented, _ = self._conditioning(observed, mask)
        raw = self.coarse(augmented)
        return mask * observed + (1.0 - mask) * raw

    def initialize_coarse_from_base_state(self, state: dict[str, torch.Tensor]) -> None:
        """Inflate a two-channel gated checkpoint with zero-init structure paths."""

        target = self.coarse.state_dict()
        converted = {}
        for name, target_value in target.items():
            if name not in state and name.startswith("auxiliary_projections."):
                converted[name] = target_value
                continue
            source = state[name]
            if source.shape == target_value.shape:
                converted[name] = source
            elif (
                name in {"encoders.0.feature.weight", "encoders.0.gate.weight"}
                and source.shape[1] == 2
                and target_value.shape[1] == 6
            ):
                inflated = torch.zeros_like(target_value)
                inflated[:, :2] = source
                converted[name] = inflated
            else:
                raise ValueError(f"cannot inflate parameter {name}: {source.shape} -> {target_value.shape}")
        self.coarse.load_state_dict(converted)
