import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.latent_diffusion import (
    FingerprintAutoencoderKL,
    ResidualConditionalLatentDDPM,
    estimate_residual_scale,
)
from fingerprint_reconstruction.models.latent_diffusion.residual_model import (
    normalized_gaussian_measurement_projection,
)
from fingerprint_reconstruction.models.unet import FingerprintUNet


def build_model():
    autoencoder = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    coarse = FingerprintUNet(channels=(4, 8))
    denoiser = DiffusionUNet(
        channels=(8, 16, 32), time_dim=32, data_channels=2,
        condition_channels=4, normalized_condition=True,
    )
    return ResidualConditionalLatentDDPM(
        autoencoder, coarse, denoiser,
        DDPMScheduler(timesteps=8, schedule="cosine"),
        latent_scale=1.0, residual_scale=2.0,
    )


def test_residual_training_loss_is_finite_and_frozen():
    model = build_model()
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    loss = model.training_loss(target, target * mask, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.autoencoder.parameters())
    assert all(parameter.grad is None for parameter in model.coarse_predictor.parameters())


def test_residual_sampling_preserves_observations_with_projection():
    model = build_model()
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed); mask[:, :, :8] = 1
    samples = model.sample_ddim(
        observed, mask, inference_steps=4, num_samples=2,
        pixel_projection_interval=2,
    )
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 32, 32)
    assert torch.isfinite(samples).all()
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_residual_scale_is_positive():
    autoencoder = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    coarse = FingerprintUNet(channels=(4, 8))
    items = []
    for _ in range(4):
        target = torch.rand(1, 32, 32)
        mask = (torch.rand_like(target) > 0.5).float()
        items.append({"target": target, "observed": target * mask, "mask": mask})
    scale = estimate_residual_scale(
        autoencoder, coarse, DataLoader(items, batch_size=2), torch.device("cpu"),
        latent_scale=1.0, max_batches=2,
    )
    assert scale > 0


def test_frequency_projection_is_exact_at_zero_sigma():
    decoded = torch.rand(2, 1, 32, 32)
    observed = torch.rand_like(decoded)
    mask = (torch.rand_like(decoded) > 0.5).float()
    actual = normalized_gaussian_measurement_projection(
        decoded, observed, mask, sigma=0.0
    )
    expected = mask * observed + (1.0 - mask) * decoded
    assert torch.equal(actual, expected)


def test_frequency_projection_preserves_constant_measurement_residual():
    decoded = torch.full((1, 1, 32, 32), 0.25)
    observed = torch.full_like(decoded, 0.75)
    mask = torch.ones_like(decoded)
    projected = normalized_gaussian_measurement_projection(
        decoded, observed, mask, sigma=2.0
    )
    assert torch.allclose(projected, observed, atol=1e-6)


def test_frequency_guided_residual_sampling_is_finite_and_consistent():
    model = build_model()
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed); mask[:, :, :8] = 1
    samples = model.sample_ddim(
        observed, mask, inference_steps=4, num_samples=2,
        pixel_projection_interval=1, frequency_guidance_max_sigma=2.0,
    )
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert torch.isfinite(samples).all()
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])
