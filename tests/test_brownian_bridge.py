import torch

from fingerprint_reconstruction.models.brownian_bridge import ConditionalBrownianBridge, bridge_schedule
from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet


def _build_model(s=1.0):
    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=16)
    return ConditionalBrownianBridge(denoiser, s=s)


def test_bridge_schedule_endpoints_have_zero_variance():
    t = torch.tensor([0.0, 1.0])
    m_t, delta_t = bridge_schedule(t, s=1.0)
    torch.testing.assert_close(m_t, t)
    torch.testing.assert_close(delta_t, torch.zeros_like(t), atol=1e-6, rtol=0)


def test_bridge_schedule_peaks_at_midpoint():
    t = torch.tensor([0.1, 0.5, 0.9])
    _, delta_t = bridge_schedule(t, s=1.0)
    assert delta_t[1] > delta_t[0]
    assert delta_t[1] > delta_t[2]


def test_training_loss_is_finite_and_differentiable():
    model = _build_model()
    target = torch.rand(2, 1, 16, 16)
    observed_filled = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = model.training_loss(target, observed_filled, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_loss_respects_missing_weight_map():
    model = _build_model()
    target = torch.rand(2, 1, 16, 16)
    observed_filled = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    weight_map = torch.zeros(2, 1, 16, 16)
    weight_map[:, :, :8, :] = 1.0
    loss = model.training_loss(target, observed_filled, mask, missing_weight_map=weight_map)
    assert torch.isfinite(loss)


def test_sample_output_shape_and_bounds():
    model = _build_model()
    model.eval()
    observed_filled = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    samples = model.sample(observed_filled, mask, num_samples=3, steps=4)
    assert samples.shape == (2, 3, 1, 16, 16)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_sample_data_consistency_matches_observed_pixels():
    model = _build_model()
    model.eval()
    observed_filled = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    samples = model.sample(observed_filled, mask, num_samples=2, steps=4, enforce_data_consistency=True)
    expanded_mask = mask.unsqueeze(1).expand(-1, 2, -1, -1, -1).bool()
    expanded_observed = observed_filled.unsqueeze(1).expand(-1, 2, -1, -1, -1)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_sample_rejects_nonpositive_steps():
    import pytest

    model = _build_model()
    observed_filled = torch.rand(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    with pytest.raises(ValueError):
        model.sample(observed_filled, mask, steps=0)


def test_rejects_nonpositive_s():
    import pytest

    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=16)
    with pytest.raises(ValueError):
        ConditionalBrownianBridge(denoiser, s=0.0)
