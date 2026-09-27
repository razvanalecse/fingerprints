"""Leakage-safe coarse fingerprint support and orientation prediction."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.models.unet import DoubleConv
from fingerprint_reconstruction.losses.reconstruction import masked_mean


class FingerprintStructurePredictor(nn.Module):
    """Predict support, doubled-angle orientation, and coherence at latent scale."""

    def __init__(self, *, channels: Sequence[int] = (16, 32, 64, 128)) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        if len(widths) != 4 or any(value <= 0 for value in widths):
            raise ValueError("structure predictor requires four positive channel widths")
        self.channels = widths
        self.encoder1 = DoubleConv(2, widths[0])
        self.encoder2 = DoubleConv(widths[0], widths[1])
        self.encoder3 = DoubleConv(widths[1], widths[2])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(widths[2], widths[3])
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(widths[3], widths[2], 3, padding=1),
        )
        self.decoder = DoubleConv(2 * widths[2], widths[2])
        self.output = nn.Conv2d(widths[2], 4, 1)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must have shape [B,1,H,W]")
        neutral_observed = mask * (2.0 * observed - 1.0)
        inputs = torch.cat((neutral_observed, mask), dim=1)
        first = self.encoder1(inputs)
        second = self.encoder2(self.pool(first))
        third = self.encoder3(self.pool(second))
        hidden = self.bottleneck(self.pool(third))
        hidden = self.up(hidden)
        if hidden.shape[-2:] != third.shape[-2:]:
            hidden = F.interpolate(hidden, size=third.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.output(self.decoder(torch.cat((hidden, third), dim=1)))
        support = torch.sigmoid(raw[:, :1])
        orientation = F.normalize(raw[:, 1:3], dim=1, eps=1e-6)
        coherence = torch.sigmoid(raw[:, 3:4])
        return torch.cat((support, orientation, coherence), dim=1)


def structure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    orientation_weight: float = 1.0,
    coherence_weight: float = 0.25,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BCE+Dice support loss and confidence-weighted axial orientation loss."""

    if prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError("prediction must have shape [B,4,h,w]")
    if target.shape != (prediction.shape[0], 5, *prediction.shape[-2:]):
        raise ValueError("target must have shape [B,5,h,w]")
    support, target_x, target_y, target_coherence, valid = target.split(1, dim=1)
    support_probability = prediction[:, :1]
    bce = F.binary_cross_entropy(support_probability, support)
    dimensions = (1, 2, 3)
    intersection = torch.sum(support_probability * support, dim=dimensions)
    denominator = torch.sum(support_probability + support, dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
    dot = prediction[:, 1:2] * target_x + prediction[:, 2:3] * target_y
    orientation_region = valid * target_coherence
    orientation = masked_mean(1.0 - dot.clamp(-1.0, 1.0), orientation_region)
    coherence = masked_mean(
        torch.abs(prediction[:, 3:4] - target_coherence), support
    )
    total = bce + dice_loss + orientation_weight * orientation + coherence_weight * coherence
    return total, {"support_bce": bce, "support_dice": dice_loss, "orientation": orientation, "coherence": coherence}
