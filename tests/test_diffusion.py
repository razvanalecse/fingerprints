import torch

from fingerprint_reconstruction.models.diffusion import (
    ConditionalDDPM,
    DDPMScheduler,
    DiffusionUNet,
    ddim_step,
    make_beta_schedule,
    make_ddim_timesteps,
    repaint_forward_step,
)


def test_beta_schedules_are_valid():
    for name in ("linear", "cosine"):
        betas = make_beta_schedule(name, 100, 1e-4, 2e-2)
        assert betas.shape == (100,)
        assert torch.all((betas > 0) & (betas < 1))


def test_q_sample_matches_closed_form_endpoints_with_fixed_noise():
    scheduler = DDPMScheduler(timesteps=20, schedule="linear")
    x0 = torch.full((2, 1, 8, 8), 0.25)
    noise = torch.full_like(x0, -0.5)
    timesteps = torch.tensor([0, 19])
    xt, returned_noise = scheduler.q_sample(x0, timesteps, noise)
    expected = (
        scheduler.sqrt_alpha_bar[timesteps, None, None, None] * x0
        + scheduler.sqrt_one_minus_alpha_bar[timesteps, None, None, None] * noise
    )
    assert torch.allclose(xt, expected)
    assert torch.equal(returned_noise, noise)


def test_noise_prediction_inverts_q_sample():
    scheduler = DDPMScheduler(timesteps=20, schedule="cosine")
    x0 = torch.rand(3, 1, 8, 8) * 2 - 1
    noise = torch.randn_like(x0)
    # Avoid the final cosine step where alpha_bar is intentionally almost zero
    # and inversion in float32 is ill-conditioned.
    timesteps = torch.tensor([0, 7, 15])
    xt, _ = scheduler.q_sample(x0, timesteps, noise)
    recovered = scheduler.predict_x0(xt, timesteps, noise)
    assert torch.allclose(recovered, x0, atol=2e-5)


def test_diffusion_unet_shape_and_backward():
    model = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    xt = torch.randn(2, 1, 33, 35)
    observed = torch.rand_like(xt)
    mask = (torch.rand_like(xt) > 0.5).float()
    output = model(xt, torch.tensor([0, 9]), observed, mask)
    output.mean().backward()
    assert output.shape == xt.shape


def test_v2_zero_initialized_conditioning_preserves_v1_function():
    torch.manual_seed(7)
    baseline = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    enhanced = DiffusionUNet(
        channels=(8, 16, 32),
        time_dim=32,
        multiscale_conditioning=True,
        middle_attention=True,
        attention_heads=4,
    )
    incompatible = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    xt = torch.randn(2, 1, 32, 32)
    observed = torch.rand_like(xt)
    mask = (torch.rand_like(xt) > 0.5).float()
    timesteps = torch.tensor([1, 5])
    baseline.eval(); enhanced.eval()
    with torch.no_grad():
        expected = baseline(xt, timesteps, observed, mask)
        actual = enhanced(xt, timesteps, observed, mask)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_zero_initialized_auxiliary_condition_preserves_function():
    torch.manual_seed(11)
    baseline = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    enhanced = DiffusionUNet(
        channels=(8, 16, 32), time_dim=32, auxiliary_condition_channels=4
    )
    incompatible = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("auxiliary_projections.") for key in incompatible.missing_keys)
    xt = torch.randn(2, 1, 32, 32)
    observed = torch.rand_like(xt)
    mask = (torch.rand_like(xt) > 0.5).float()
    auxiliary = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([1, 5])
    baseline.eval(); enhanced.eval()
    with torch.no_grad():
        expected = baseline(xt, timesteps, observed, mask)
        actual = enhanced(xt, timesteps, observed, mask, auxiliary)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_ddpm_sampling_preserves_observed_pixels_exactly():
    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    model = ConditionalDDPM(denoiser, DDPMScheduler(timesteps=4, schedule="linear"))
    observed = torch.rand(1, 1, 16, 16)
    mask = torch.zeros_like(observed)
    mask[:, :, :5] = 1
    samples = model.sample(observed, mask, num_samples=2)
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 16, 16)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_ddpm_training_loss_is_finite():
    model = ConditionalDDPM(
        DiffusionUNet(channels=(8, 16, 32), time_dim=32),
        DDPMScheduler(timesteps=10, schedule="cosine"),
    )
    target = torch.rand(2, 1, 16, 16)
    mask = (torch.rand_like(target) > 0.5).float()
    loss, parts = model.training_loss(target, target * mask, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"missing_noise_mse", "observed_noise_mse"}


def test_ddim_timestep_schedule_is_strict_and_has_endpoints():
    schedule = make_ddim_timesteps(500, 50)
    assert schedule.shape == (50,)
    assert schedule[0] == 499
    assert schedule[-1] == 0
    assert torch.all(schedule[:-1] > schedule[1:])


def test_deterministic_ddim_step_reconstructs_previous_marginal_with_true_noise():
    scheduler = DDPMScheduler(timesteps=30, schedule="cosine")
    x0 = torch.rand(2, 1, 8, 8) * 2 - 1
    epsilon = torch.randn_like(x0)
    t, previous = 20, 7
    xt = torch.sqrt(scheduler.alpha_bar[t]) * x0 + torch.sqrt(
        1.0 - scheduler.alpha_bar[t]
    ) * epsilon
    result = ddim_step(
        xt,
        x0,
        epsilon,
        alpha_bar_t=scheduler.alpha_bar[t],
        alpha_bar_previous=scheduler.alpha_bar[previous],
        eta=0.0,
    )
    expected = torch.sqrt(scheduler.alpha_bar[previous]) * x0 + torch.sqrt(
        1.0 - scheduler.alpha_bar[previous]
    ) * epsilon
    assert torch.allclose(result, expected, atol=1e-6)


def test_ddim_sampling_preserves_observed_pixels_exactly():
    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    model = ConditionalDDPM(denoiser, DDPMScheduler(timesteps=10, schedule="linear"))
    observed = torch.rand(1, 1, 16, 16)
    mask = torch.zeros_like(observed)
    mask[:, :, :5] = 1
    samples = model.sample_ddim(observed, mask, inference_steps=5, num_samples=2)
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 16, 16)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_repaint_forward_step_matches_forward_markov_transition():
    scheduler = DDPMScheduler(timesteps=10, schedule="linear")
    previous = torch.full((2, 1, 4, 4), 0.25)
    noise = torch.full_like(previous, -0.5)
    result = repaint_forward_step(previous, 6, scheduler, noise)
    expected = (
        torch.sqrt(scheduler.alphas[6]) * previous
        + torch.sqrt(scheduler.betas[6]) * noise
    )
    assert torch.allclose(result, expected)


def test_repaint_sampling_preserves_observed_pixels_exactly():
    denoiser = DiffusionUNet(channels=(8, 16, 32), time_dim=32)
    model = ConditionalDDPM(denoiser, DDPMScheduler(timesteps=4, schedule="linear"))
    observed = torch.rand(1, 1, 16, 16)
    mask = torch.zeros_like(observed)
    mask[:, :, :5] = 1
    samples = model.sample_repaint(
        observed, mask, num_samples=2, resampling_steps=2
    )
    expanded_observed = observed[:, None].expand_as(samples)
    expanded_mask = mask[:, None].expand_as(samples).bool()
    assert samples.shape == (1, 2, 1, 16, 16)
    assert torch.equal(samples[expanded_mask], expanded_observed[expanded_mask])


def test_repaint_rejects_invalid_resampling_count():
    model = ConditionalDDPM(
        DiffusionUNet(channels=(8, 16, 32), time_dim=32),
        DDPMScheduler(timesteps=4, schedule="linear"),
    )
    observed = torch.rand(1, 1, 8, 8)
    mask = torch.zeros_like(observed)
    try:
        model.sample_repaint(observed, mask, resampling_steps=0)
    except ValueError as error:
        assert "resampling_steps" in str(error)
    else:
        raise AssertionError("invalid RePaint resampling count was accepted")
