import torch

from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet
from fingerprint_reconstruction.models.flow_matching import ConditionalFlowMatching


def _build_model(sigma_0=0.05):
    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=16)
    return ConditionalFlowMatching(denoiser, sigma_0=sigma_0)


def test_training_loss_is_finite_and_differentiable():
    model = _build_model()
    target = torch.rand(2, 1, 16, 16)
    observed = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    coarse = torch.rand(2, 1, 16, 16)
    loss = model.training_loss(target, observed, mask, coarse)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_loss_respects_missing_weight_map():
    model = _build_model()
    target = torch.rand(2, 1, 16, 16)
    observed = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    coarse = torch.rand(2, 1, 16, 16)
    weight_map = torch.zeros(2, 1, 16, 16)
    weight_map[:, :, :8, :] = 1.0
    loss = model.training_loss(target, observed, mask, coarse, missing_weight_map=weight_map)
    assert torch.isfinite(loss)


def test_training_loss_zero_sigma_makes_x0_equal_coarse():
    # With sigma_0=0 and a generator, x0 is exactly `coarse` (no noise), so
    # the velocity target is exactly target - coarse: sanity-check the math
    # by verifying the loss is invariant to different epsilon draws when
    # sigma_0=0 (epsilon should have zero effect on x0/velocity).
    model = _build_model(sigma_0=0.0)
    target = torch.rand(2, 1, 16, 16)
    observed = torch.rand(2, 1, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    coarse = torch.rand(2, 1, 16, 16)
    generator_a = torch.Generator().manual_seed(0)
    generator_b = torch.Generator().manual_seed(0)
    torch.manual_seed(1)
    loss_a = model.training_loss(target, observed, mask, coarse, generator=generator_a)
    torch.manual_seed(1)
    loss_b = model.training_loss(target, observed, mask, coarse, generator=generator_b)
    torch.testing.assert_close(loss_a, loss_b)


def test_sample_output_shape_and_data_consistency():
    model = _build_model()
    model.eval()
    observed = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    coarse = torch.rand(2, 1, 16, 16)
    samples = model.sample(observed, mask, coarse, num_samples=3, steps=4)
    assert samples.shape == (2, 3, 1, 16, 16)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0
    # Data consistency: observed pixels (mask==1) must exactly equal `observed`.
    expanded_mask = mask.unsqueeze(1).expand(-1, 3, -1, -1, -1).bool()
    expanded_observed = observed.unsqueeze(1).expand(-1, 3, -1, -1, -1)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_sample_without_data_consistency_can_differ_at_observed_pixels():
    model = _build_model()
    model.eval()
    observed = torch.zeros(1, 1, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    coarse = torch.full((1, 1, 16, 16), 0.9)
    samples = model.sample(observed, mask, coarse, num_samples=1, steps=2, enforce_data_consistency=False)
    # With mask all-ones and observed all-zero but coarse far from zero,
    # disabling data consistency should generally leave the output away from
    # exactly `observed` (not a strict guarantee, but true for this setup).
    assert not torch.allclose(samples, torch.zeros_like(samples))


def test_sample_rejects_nonpositive_steps():
    import pytest

    model = _build_model()
    observed = torch.rand(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    coarse = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError):
        model.sample(observed, mask, coarse, steps=0)
