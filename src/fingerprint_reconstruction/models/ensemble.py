"""A small pixel-wise combiner for blending several frozen base models' predictions.

Motivated by wanting something cheaper and potentially more informative than
a plain average of already-trained models: a per-pixel convex combination
(softmax weights, always summing to 1) lets the combiner learn, e.g., "trust
the CVAE more in smooth interior regions, trust the residual DDPM more near
edges" instead of a single global weight per model. Kept intentionally
simple (a handful of conv layers) since the base models already do the hard
work; this network only has to learn *how to weight* them.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class EnsembleCombiner(nn.Module):
    def __init__(self, *, num_models: int, channels: Sequence[int] = (24, 32, 24)) -> None:
        super().__init__()
        if num_models < 2:
            raise ValueError("ensemble combiner requires at least two base models")
        widths = tuple(int(value) for value in channels)
        if len(widths) < 1 or any(value <= 0 for value in widths):
            raise ValueError("channels must contain at least one positive width")
        self.num_models = int(num_models)
        input_channels = 2 + num_models  # observed, mask, one prediction channel per model
        layers: list[nn.Module] = []
        source = input_channels
        for width in widths:
            layers.extend((nn.Conv2d(source, width, 3, padding=1), nn.GroupNorm(min(8, width), width), nn.SiLU()))
            source = width
        self.backbone = nn.Sequential(*layers)
        self.weight_head = nn.Conv2d(source, num_models, 1)

    def forward(self, observed: torch.Tensor, mask: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """``predictions`` has shape [B, num_models, 1, H, W]; returns the blended [B,1,H,W] output."""

        if predictions.ndim != 5 or predictions.shape[1] != self.num_models or predictions.shape[2] != 1:
            raise ValueError("predictions must have shape [B,num_models,1,H,W]")
        batch, _, _, height, width = predictions.shape
        stacked = predictions[:, :, 0]  # [B, num_models, H, W]
        inputs = torch.cat((observed, mask, stacked), dim=1)
        hidden = self.backbone(inputs)
        weights = torch.softmax(self.weight_head(hidden), dim=1)  # [B, num_models, H, W]
        blended = (weights * stacked).sum(dim=1, keepdim=True)
        return blended, weights
