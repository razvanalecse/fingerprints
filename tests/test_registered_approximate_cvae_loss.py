import torch

from fingerprint_reconstruction.losses.cvae import RegisteredApproximateCVAELoss


def _batch():
    torch.manual_seed(0)
    reconstruction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    roi = torch.ones(2, 1, 16, 16)
    confidence = torch.full((2, 1, 16, 16), 0.7)
    mu = torch.zeros(2, 8, requires_grad=True)
    logvar = torch.zeros(2, 8, requires_grad=True)
    return reconstruction, target, mask, roi, confidence, mu, logvar


def test_registered_cvae_loss_is_differentiable_and_finite():
    reconstruction, target, mask, roi, confidence, mu, logvar = _batch()
    loss_function = RegisteredApproximateCVAELoss()
    output = loss_function(
        reconstruction, target, mask, mu, logvar,
        beta=0.5, evaluation_roi=roi, geometric_confidence=confidence,
    )
    output.total.backward()
    assert torch.isfinite(output.total)
    assert reconstruction.grad is not None and torch.isfinite(reconstruction.grad).all()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_zero_confidence_removes_reconstruction_term_but_keeps_kl():
    reconstruction, target, mask, roi, _, mu, logvar = _batch()
    zero_confidence = torch.zeros_like(roi)
    loss_function = RegisteredApproximateCVAELoss(free_nats_per_dimension=0.0)
    output = loss_function(
        reconstruction, target, mask, mu, logvar,
        beta=1.0, evaluation_roi=roi, geometric_confidence=zero_confidence,
    )
    assert output.components["missing_l1"] == 0.0
    assert output.components["missing_mse"] == 0.0
    assert torch.isfinite(output.total)


def test_rejects_out_of_range_confidence():
    reconstruction, target, mask, roi, _, mu, logvar = _batch()
    bad_confidence = torch.full_like(roi, 1.5)
    loss_function = RegisteredApproximateCVAELoss()
    try:
        loss_function(
            reconstruction, target, mask, mu, logvar,
            beta=0.5, evaluation_roi=roi, geometric_confidence=bad_confidence,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_support_and_structural_terms_are_exposed():
    reconstruction, target, mask, roi, confidence, mu, logvar = _batch()
    support = torch.full_like(mask, 0.75)
    loss_function = RegisteredApproximateCVAELoss(
        l1_weight=0.25,
        mse_weight=0.05,
        orientation_weight=0.1,
        ridge_spectrum_weight=0.1,
        ridge_band_frequencies=(0.15, 0.20),
        ridge_band_orientations=2,
        ridge_band_kernel_size=9,
        ridge_band_pool_size=3,
        support_background_weight=0.2,
        support_region_weighting=True,
    )
    output = loss_function(
        reconstruction,
        target,
        mask,
        mu,
        logvar,
        beta=0.05,
        evaluation_roi=roi,
        geometric_confidence=confidence,
        observed=reconstruction.detach(),
        support_probability=support,
    )
    output.total.backward()
    assert torch.isfinite(output.total)
    assert "orientation" in output.components
    assert "ridge_spectrum" in output.components
    assert "support_background" in output.components
    assert output.components["kl_total_nats"] >= output.components["kl"]
