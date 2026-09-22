import torch

from fingerprint_reconstruction.losses.cvae import CVAELoss
from fingerprint_reconstruction.models.cvae import ConditionalVAE


def test_cvae_shapes_sampling_and_data_consistency():
    model = ConditionalVAE(channels=(8, 16, 32), latent_dim=6)
    target = torch.rand(2, 1, 33, 35)
    observed = target.clone()
    mask = torch.zeros_like(target)
    mask[:, :, :8, :] = 1
    observed = observed * mask
    reconstruction, mu, logvar = model(target, observed, mask)
    samples = model.sample(observed, mask, num_samples=4)
    assert reconstruction.shape == target.shape
    assert mu.shape == logvar.shape == (2, 6)
    assert samples.shape == (2, 4, 1, 33, 35)
    expanded_mask = mask[:, None].expand(-1, 4, -1, -1, -1).bool()
    expanded_observed = observed[:, None].expand_as(samples)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_cvae_loss_is_finite_and_backpropagates():
    model = ConditionalVAE(channels=(8, 16, 32), latent_dim=4)
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    observed = target * mask
    reconstruction, mu, logvar = model(target, observed, mask)
    result = CVAELoss()(reconstruction, target, mask, mu, logvar, beta=0.1)
    result.total.backward()
    assert torch.isfinite(result.total)
    assert any(parameter.grad is not None for parameter in model.parameters())
