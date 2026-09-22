import torch

from fingerprint_reconstruction.losses import RegisteredApproximateLoss


def test_registered_loss_is_differentiable_and_ignores_outside_roi() -> None:
    prediction = torch.full((1, 1, 24, 24), 0.4, requires_grad=True)
    target = torch.zeros_like(prediction)
    observed = torch.full_like(prediction, 0.4)
    mask = torch.zeros_like(prediction)
    roi = torch.zeros_like(prediction)
    roi[:, :, 4:20, 4:20] = 1.0
    confidence = roi.clone()
    loss_function = RegisteredApproximateLoss(
        missing_l1_weight=1.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
        ridge_band_weight=0.1,
        ridge_band_frequencies=(0.15, 0.20),
        ridge_band_orientations=4,
        ridge_band_kernel_size=9,
        ridge_band_pool_size=3,
        ridge_spectrum_weight=0.2,
    )
    output = loss_function(
        prediction,
        target,
        mask,
        observed=observed,
        evaluation_roi=roi,
        geometric_confidence=confidence,
    )
    output.total.backward()
    assert torch.isfinite(output.total)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_geometric_confidence_downweights_distant_error() -> None:
    target = torch.zeros((1, 1, 8, 8))
    observed = torch.zeros_like(target)
    mask = torch.zeros_like(target)
    roi = torch.ones_like(target)
    loss_function = RegisteredApproximateLoss(
        missing_l1_weight=1.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
    )
    near_error = torch.zeros_like(target)
    near_error[:, :, :4] = 1.0
    far_error = torch.zeros_like(target)
    far_error[:, :, 4:] = 1.0
    confidence = torch.ones_like(target)
    confidence[:, :, 4:] = 0.25
    near = loss_function(
        near_error,
        target,
        mask,
        observed=observed,
        evaluation_roi=roi,
        geometric_confidence=confidence,
    )
    far = loss_function(
        far_error,
        target,
        mask,
        observed=observed,
        evaluation_roi=roi,
        geometric_confidence=confidence,
    )
    assert near.total > far.total
