import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.latent_diffusion import (
    ConditionalLatentDDPM,
    FingerprintAutoencoderKL,
    estimate_latent_scale,
)
from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor


def build_model(timesteps=10):
    autoencoder = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    denoiser = DiffusionUNet(
        channels=(8, 16, 32),
        time_dim=32,
        data_channels=2,
        condition_channels=2,
        normalized_condition=True,
    )
    return ConditionalLatentDDPM(
        autoencoder,
        denoiser,
        DDPMScheduler(timesteps=timesteps, schedule="cosine"),
        latent_scale=1.0,
    )


def test_latent_diffusion_training_loss_is_finite():
    model = build_model()
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    loss = model.training_loss(target, target * mask, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.autoencoder.parameters())


def test_latent_diffusion_training_loss_accepts_pixel_resolution_weight_map():
    model = build_model()
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    weight_map = torch.rand_like(target)
    loss = model.training_loss(target, target * mask, mask, weight_map=weight_map)
    loss.backward()
    assert torch.isfinite(loss)


def test_latent_diffusion_zero_weight_map_is_finite_not_nan():
    model = build_model()
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    weight_map = torch.zeros_like(target)
    loss = model.training_loss(target, target * mask, mask, weight_map=weight_map)
    assert torch.isfinite(loss)


def test_latent_diffusion_weight_map_changes_loss_value():
    model = build_model()
    torch.manual_seed(0)
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    uniform_weight = torch.ones_like(target)
    skewed_weight = torch.zeros_like(target)
    skewed_weight[:, :, :16, :] = 1.0
    torch.manual_seed(1)
    loss_uniform = model.training_loss(target, target * mask, mask, weight_map=uniform_weight)
    torch.manual_seed(1)
    loss_skewed = model.training_loss(target, target * mask, mask, weight_map=skewed_weight)
    # Same noise/timestep draw (seeded identically); only the weighting
    # differs, so a real weighting effect should generally change the loss.
    assert loss_uniform.item() != loss_skewed.item()


def test_latent_ddim_sampling_preserves_observations():
    model = build_model()
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed); mask[:, :, :8] = 1
    samples = model.sample_ddim(observed, mask, inference_steps=5, num_samples=2)
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 32, 32)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_latent_ddim_sampling_with_latent_data_consistency():
    model = build_model(timesteps=8)
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed); mask[:, :, :8] = 1
    samples = model.sample_ddim(
        observed,
        mask,
        inference_steps=4,
        num_samples=2,
        latent_data_consistency=True,
    )
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 32, 32)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_decode_project_reencode_sampling_preserves_observations():
    model = build_model(timesteps=8)
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed); mask[:, :, :8] = 1
    samples = model.sample_ddim(
        observed,
        mask,
        inference_steps=4,
        num_samples=2,
        latent_data_consistency=True,
        pixel_projection_interval=1,
    )
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert torch.isfinite(samples).all()
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_latent_scale_estimation_matches_inverse_standard_deviation():
    autoencoder = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    items = [{"target": torch.rand(1, 32, 32)} for _ in range(4)]
    loader = DataLoader(items, batch_size=2)
    expected = []
    with torch.no_grad():
        for batch in loader:
            expected.append(autoencoder.encode(batch["target"], sample=False).flatten())
    expected_scale = 1.0 / torch.cat(expected).std(unbiased=True).item()
    actual = estimate_latent_scale(autoencoder, loader, torch.device("cpu"), max_batches=2)
    assert abs(actual - expected_scale) < 1e-6


def test_structure_guided_latent_diffusion_is_finite_and_frozen():
    autoencoder = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    denoiser = DiffusionUNet(
        channels=(8, 16, 32),
        time_dim=32,
        data_channels=2,
        condition_channels=2,
        normalized_condition=True,
        auxiliary_condition_channels=4,
    )
    structure = FingerprintStructurePredictor(channels=(4, 8, 16, 32))
    model = ConditionalLatentDDPM(
        autoencoder,
        denoiser,
        DDPMScheduler(timesteps=8, schedule="cosine"),
        latent_scale=1.0,
        structure_predictor=structure,
    )
    target = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(target) > 0.5).float()
    loss = model.training_loss(target, target * mask, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.structure_predictor.parameters())
