"""Leakage-safe anatomical support prediction from partial observations."""

from __future__ import annotations

from typing import Sequence

import torch

from fingerprint_reconstruction.models.unet import FingerprintUNet


class FingerprintSupportPredictor(FingerprintUNet):
    """Predict ``P(S=1 | Y,M)`` without accessing the complete image at inference."""

    def __init__(self, *, channels: Sequence[int] = (16, 32, 64, 128)) -> None:
        super().__init__(
            channels=channels,
            neutral_missing_encoding=True,
            upsampling_mode="resize_conv",
        )

    def predict_binary(
        self, observed: torch.Tensor, mask: torch.Tensor, *, threshold: float = 0.5
    ) -> torch.Tensor:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between zero and one")
        probability = self(torch.cat((observed, mask), dim=1))
        return probability >= threshold


def support_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pixel BCE plus soft Dice loss for a foreground/background support mask."""

    if probability.shape != target.shape:
        raise ValueError("probability and target must have identical shapes")
    if dice_weight < 0:
        raise ValueError("dice_weight must be non-negative")
    binary_cross_entropy = torch.nn.functional.binary_cross_entropy(probability, target)
    dimensions = tuple(range(1, probability.ndim))
    intersection = torch.sum(probability * target, dim=dimensions)
    denominator = torch.sum(probability + target, dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    dice_loss = 1.0 - dice.mean()
    total = binary_cross_entropy + dice_weight * dice_loss
    return total, {"bce": binary_cross_entropy, "dice_loss": dice_loss}
