"""Neural reconstruction models."""

from fingerprint_reconstruction.models.classical import NearestObservedInpainting
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.ffc import FFCFingerprintNetwork
from fingerprint_reconstruction.models.gated_conv import GatedFingerprintNetwork
from fingerprint_reconstruction.models.gabor_extension import GaborRidgeExtension
from fingerprint_reconstruction.models.ridge_cleanup import RidgeCleanup
from fingerprint_reconstruction.models.unet import FingerprintUNet
from fingerprint_reconstruction.models.support import (
    REGISTERED_APPROXIMATE_FULL_SUPPORT,
    SUPPORT_TARGET_MODES,
    VISIBLE_LATENT_SUPPORT,
    FingerprintSupportPredictor,
    apply_support_constraint,
    build_support_target,
    support_loss,
)
from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor, structure_loss

__all__ = [
    "FingerprintUNet",
    "GatedFingerprintNetwork",
    "FFCFingerprintNetwork",
    "GaborRidgeExtension",
    "RidgeCleanup",
    "NearestObservedInpainting",
    "FingerprintSupportPredictor",
    "support_loss",
    "apply_support_constraint",
    "build_support_target",
    "VISIBLE_LATENT_SUPPORT",
    "REGISTERED_APPROXIMATE_FULL_SUPPORT",
    "SUPPORT_TARGET_MODES",
    "FingerprintStructurePredictor",
    "structure_loss",
    "build_reconstruction_model",
]
