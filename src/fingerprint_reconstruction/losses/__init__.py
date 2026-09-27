"""Training objectives."""

from fingerprint_reconstruction.losses.reconstruction import (
    MaskedReconstructionLoss,
    RegisteredApproximateLoss,
)

__all__ = ["MaskedReconstructionLoss", "RegisteredApproximateLoss"]
