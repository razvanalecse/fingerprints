import numpy as np
import torch

from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss


def _ridge_grating(shape=(48, 48), theta=0.0, period=8.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    normal_coordinate = -np.sin(theta) * xx + np.cos(theta) * yy
    image = 0.5 + 0.45 * np.cos(2.0 * np.pi * normal_coordinate / period)
    return torch.from_numpy(image.astype(np.float32))[None, None]


def _half_mask(shape=(48, 48)):
    mask = torch.zeros(1, 1, *shape)
    mask[:, :, :, : shape[1] // 2] = 1.0  # left half observed, right half missing
    return mask


def test_boundary_continuity_disabled_by_default_matches_base_loss():
    target = _ridge_grating()
    prediction = target.clone().requires_grad_(True)
    mask = _half_mask()
    base = MaskedReconstructionLoss()(prediction, target, mask).total
    same = MaskedReconstructionLoss(boundary_continuity_weight=0.0)(prediction, target, mask).total
    assert torch.equal(base, same)


def test_boundary_continuity_is_near_zero_for_a_perfect_prediction():
    target = _ridge_grating()
    mask = _half_mask()
    result = MaskedReconstructionLoss(boundary_continuity_weight=1.0)(target.clone(), target, mask)
    assert result.components["boundary_continuity"] < 1e-4


def test_boundary_continuity_penalizes_a_phase_jump_at_the_mask_edge():
    # Vertical stripes (theta=pi/2): intensity varies along columns, so the
    # mask boundary (a vertical line splitting columns) cuts *across* ridges
    # and a phase shift on one side creates a genuine discontinuity there.
    target = _ridge_grating(theta=np.pi / 2, period=8.0)
    mask = _half_mask()
    # Phase-shifted on the missing side only: continuous prediction would
    # instead just be `target` again (phase-matched, no jump).
    shifted = torch.roll(_ridge_grating(theta=np.pi / 2, period=8.0), shifts=3, dims=-1)
    prediction = torch.where(mask.bool(), target, shifted)
    loss_fn = MaskedReconstructionLoss(boundary_continuity_weight=1.0)
    jumpy = loss_fn(prediction, target, mask).components["boundary_continuity"]
    smooth = loss_fn(target.clone(), target, mask).components["boundary_continuity"]
    assert jumpy > smooth


def test_boundary_continuity_is_differentiable_and_finite():
    target = _ridge_grating()
    mask = _half_mask()
    prediction = (target + 0.05 * torch.randn_like(target)).requires_grad_(True)
    result = MaskedReconstructionLoss(boundary_continuity_weight=0.5)(prediction, target, mask)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_boundary_continuity_rejects_negative_weight():
    import pytest

    with pytest.raises(ValueError):
        MaskedReconstructionLoss(boundary_continuity_weight=-0.1)


def test_boundary_continuity_rejects_even_band_width():
    import pytest

    with pytest.raises(ValueError):
        MaskedReconstructionLoss(boundary_continuity_weight=1.0, boundary_band_width=4)
