"""Auditable model construction from experiment configuration."""

from __future__ import annotations

from typing import Mapping, Sequence

from torch import nn

from fingerprint_reconstruction.models.ffc import FFCFingerprintNetwork
from fingerprint_reconstruction.models.gated_conv import (
    GatedFingerprintNetwork,
    StructureConditionedGatedNetwork,
)
from fingerprint_reconstruction.models.unet import FingerprintUNet


def build_reconstruction_model(
    model_config: Mapping[str, object], *, channels: Sequence[int] | None = None
) -> nn.Module:
    architecture = str(model_config.get("architecture", "unet"))
    widths = tuple(channels or model_config["channels"])
    if architecture == "unet":
        return FingerprintUNet(
            channels=widths,
            neutral_missing_encoding=bool(
                model_config.get("neutral_missing_encoding", False)
            ),
            upsampling_mode=str(model_config.get("upsampling_mode", "transpose")),
        )
    if architecture == "gated_conv":
        return GatedFingerprintNetwork(
            channels=widths,
            auxiliary_channels=int(model_config.get("auxiliary_channels", 0)),
            upsampling_mode=str(model_config.get("upsampling_mode", "nearest")),
            refine_blocks=int(model_config.get("refine_blocks", 0)),
        )
    if architecture == "structure_gated_conv":
        return StructureConditionedGatedNetwork(
            channels=widths,
            structure_channels=tuple(model_config.get("structure_channels", (16, 32, 64, 128))),
            upsampling_mode=str(model_config.get("upsampling_mode", "nearest")),
            refine_blocks=int(model_config.get("refine_blocks", 0)),
        )
    if architecture == "ffc":
        return FFCFingerprintNetwork(
            channels=widths,
            bottleneck_blocks=int(model_config.get("bottleneck_blocks", 6)),
            mid_blocks=int(model_config.get("mid_blocks", 2)),
            global_ratio=float(model_config.get("global_ratio", 0.5)),
        )
    raise ValueError(f"unsupported reconstruction architecture: {architecture}")
