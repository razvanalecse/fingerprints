"""Spectrally normalised PatchGAN discriminator conditioned on the observation mask."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class PatchDiscriminator(nn.Module):
    """Scores overlapping patches of ``[image, mask]``; higher means more real."""

    def __init__(self, widths: Sequence[int] = (64, 128, 256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 2
        for index, width in enumerate(widths):
            stride = 2 if index < len(widths) - 1 else 1
            layers += [spectral_norm(nn.Conv2d(channels, width, 4, stride=stride, padding=1)), nn.LeakyReLU(0.2)]
            channels = width
        layers.append(spectral_norm(nn.Conv2d(channels, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if image.shape != mask.shape or image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image and mask must both have shape [B, 1, H, W]")
        return self.net(torch.cat((image, mask), dim=1))


def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
