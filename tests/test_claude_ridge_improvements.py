import numpy as np
import torch

from fingerprint_reconstruction.data.augmentation import (
    AugmentationConfig,
    augment_image,
    augmentation_rng,
)
from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.discriminator import (
    PatchDiscriminator,
    hinge_discriminator_loss,
    hinge_generator_loss,
)
from fingerprint_reconstruction.models.factory import build_reconstruction_model


def _batch():
    torch.manual_seed(0)
    target = torch.rand(2, 1, 48, 48)
    prediction = torch.rand(2, 1, 48, 48, requires_grad=True)
    mask = (torch.rand(2, 1, 48, 48) > 0.5).float()
    return prediction, target, mask


def test_band_and_fine_losses_are_differentiable_and_disabled_by_default():
    prediction, target, mask = _batch()
    base = MaskedReconstructionLoss()(prediction, target, mask).total
    same = MaskedReconstructionLoss(
        band_energy_weight=0.0, fine_energy_weight=0.0
    )(prediction, target, mask).total
    assert torch.equal(base, same)
    result = MaskedReconstructionLoss(
        band_energy_weight=0.5, fine_energy_weight=0.3
    )(prediction, target, mask)
    assert torch.isfinite(result.total)
    assert result.components["band_energy"] > 0
    assert result.components["fine_energy"] > 0
    result.total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_perfect_prediction_has_zero_band_and_fine_mismatch():
    _, target, mask = _batch()
    result = MaskedReconstructionLoss(
        band_energy_weight=1.0, fine_energy_weight=1.0
    )(target.clone(), target, mask)
    assert result.components["band_energy"] < 1e-6
    assert result.components["fine_energy"] < 1e-6


def test_gated_bilinear_refinement_and_ffc_shapes():
    conditioning = torch.rand(2, 2, 64, 64)
    refined = build_reconstruction_model(
        {
            "architecture": "gated_conv",
            "channels": [8, 16, 32, 64],
            "upsampling_mode": "bilinear",
            "refine_blocks": 2,
        }
    )
    default = build_reconstruction_model(
        {"architecture": "gated_conv", "channels": [8, 16, 32, 64]}
    )
    ffc = build_reconstruction_model(
        {
            "architecture": "ffc",
            "channels": [8, 16, 32],
            "bottleneck_blocks": 1,
            "mid_blocks": 1,
        }
    )
    assert refined(conditioning).shape == (2, 1, 64, 64)
    assert ffc(conditioning).shape == (2, 1, 64, 64)
    assert default.upsampling_mode == "nearest" and len(default.refine) == 0


def test_augmentation_is_epoch_deterministic_and_bounded():
    axis = np.arange(64)[None, :].repeat(64, axis=0)
    image = (0.5 + 0.5 * np.sin(2 * np.pi * axis / 8.0)).astype(np.float32)
    config = AugmentationConfig()
    first = augment_image(image, config, augmentation_rng(1, "sample", 3))
    repeated = augment_image(image, config, augmentation_rng(1, "sample", 3))
    next_epoch = augment_image(image, config, augmentation_rng(1, "sample", 4))
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)
    assert first.dtype == np.float32 and 0 <= first.min() <= first.max() <= 1


def test_patch_discriminator_losses_are_finite():
    image = torch.rand(2, 1, 64, 64)
    mask = torch.ones_like(image)
    discriminator = PatchDiscriminator(widths=(8, 16, 32))
    real = discriminator(image, mask)
    fake = discriminator(1.0 - image, mask)
    assert torch.isfinite(hinge_discriminator_loss(real, fake))
    assert torch.isfinite(hinge_generator_loss(fake))
