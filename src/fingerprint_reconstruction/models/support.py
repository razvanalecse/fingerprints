"""Leakage-safe anatomical support prediction from partial observations."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F

from fingerprint_reconstruction.models.unet import FingerprintUNet

VISIBLE_LATENT_SUPPORT = "visible_latent_support"
REGISTERED_APPROXIMATE_FULL_SUPPORT = "registered_approximate_full_support"
SUPPORT_TARGET_MODES = {
    VISIBLE_LATENT_SUPPORT,
    REGISTERED_APPROXIMATE_FULL_SUPPORT,
}


def apply_support_constraint(
    reconstruction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    support_probability: torch.Tensor,
    *,
    mode: str = "soft",
    threshold: float = 0.5,
    feather_sigma: float = 1.5,
) -> torch.Tensor:
    """Compose a data-consistent output with white outside predicted support.

    ``feathered`` thresholds the predicted support and smooths only its boundary.
    This suppresses weak irregular probability tails without introducing the
    staircase edge of a binary hard mask.
    """

    tensors = (observed, mask, support_probability)
    if any(tensor.shape != reconstruction.shape for tensor in tensors):
        raise ValueError("all support-composition tensors must have identical shapes")
    if mode not in {"soft", "hard", "feathered"}:
        raise ValueError("support constraint mode must be soft, hard or feathered")
    if not 0.0 < threshold < 1.0:
        raise ValueError("support threshold must lie strictly between zero and one")
    if feather_sigma <= 0:
        raise ValueError("feather sigma must be positive")
    support = support_probability.to(reconstruction.dtype).clamp(0.0, 1.0)
    support = torch.maximum(support, mask.to(reconstruction.dtype))
    if mode == "hard":
        support = (support >= threshold).to(reconstruction.dtype)
    elif mode == "feathered":
        support = (support >= threshold).to(reconstruction.dtype)
        radius = max(1, int(round(3.0 * feather_sigma)))
        coordinates = torch.arange(
            -radius, radius + 1, device=support.device, dtype=support.dtype
        )
        kernel = torch.exp(-0.5 * torch.square(coordinates / feather_sigma))
        kernel = kernel / kernel.sum()
        horizontal = kernel.reshape(1, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1)
        support = F.pad(support, (radius, radius, 0, 0), mode="replicate")
        support = F.conv2d(support, horizontal)
        support = F.pad(support, (0, 0, radius, radius), mode="replicate")
        support = F.conv2d(support, vertical).clamp(0.0, 1.0)
        support = torch.maximum(support, mask.to(reconstruction.dtype))
    missing_output = support * reconstruction + (1.0 - support)
    return mask * observed + (1.0 - mask) * missing_output


def build_support_target(
    batch: dict[str, torch.Tensor], mode: str, *, dtype: torch.dtype
) -> torch.Tensor:
    """Build an explicitly named support target from an SD302 registered batch.

    ``visible_latent_support`` is the examiner-annotated latent area (EFS quality
    at least 1). ``registered_approximate_full_support`` is the union of the
    reliably observed latent area and the foreground of the geometrically
    registered exemplar.  The latter is an approximate training target and must
    never be described as pixel-exact ground truth.
    """

    if mode == VISIBLE_LATENT_SUPPORT:
        return (batch["quality"] >= 1).to(dtype=dtype)
    if mode == REGISTERED_APPROXIMATE_FULL_SUPPORT:
        observed = batch["mask"] >= 0.5
        registered = batch["registered_support"] >= 0.5
        return (observed | registered).to(dtype=dtype)
    choices = ", ".join(sorted(SUPPORT_TARGET_MODES))
    raise ValueError(f"unknown support target mode {mode!r}; expected one of: {choices}")


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
