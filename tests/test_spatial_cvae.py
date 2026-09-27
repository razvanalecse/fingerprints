import torch

from fingerprint_reconstruction.models.cvae import ConditionalVAE, SpatialConditionalVAE


def _model():
    return SpatialConditionalVAE(channels=(8, 16, 32), latent_channels=4)


def test_shapes_and_exact_observed_pixels():
    model = _model()
    observed = torch.rand(2, 1, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    target = torch.rand(2, 1, 32, 32)
    reconstruction, mu, logvar = model(target, observed, mask)
    assert reconstruction.shape == (2, 1, 32, 32)
    assert mu.shape == logvar.shape == (2, 4, 8, 8)  # downsample factor 4 (3 blocks: stride1,2,2)
    assert torch.equal(reconstruction * mask, observed * mask)


def test_reconstruct_is_deterministic_zero_latent():
    model = _model()
    observed = torch.rand(1, 1, 32, 32)
    mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
    first = model.reconstruct(observed, mask)
    second = model.reconstruct(observed, mask)
    assert torch.equal(first, second)


def test_samples_have_nonzero_predictive_variation():
    torch.manual_seed(0)
    model = _model()
    observed = torch.zeros(1, 1, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    samples = model.sample(observed, mask, num_samples=16)
    assert samples.shape == (1, 16, 1, 32, 32)
    std = samples.std(dim=1)[0, 0]
    assert std.mean() > 1e-4


def test_single_cell_latent_interventions_are_spatially_ordered():
    """Moving one latent impulse must move the output response in the same direction."""

    torch.manual_seed(4)
    model = _model().eval()
    observed = torch.zeros(1, 1, 32, 32)
    mask = torch.zeros_like(observed)
    baseline = torch.zeros(1, 4, 8, 8)
    upper_left = baseline.clone()
    lower_right = baseline.clone()
    upper_left[0, 0, 1, 1] = 5.0
    lower_right[0, 0, 6, 6] = 5.0
    with torch.no_grad():
        neutral = model.decode(baseline, observed, mask)
        first = (model.decode(upper_left, observed, mask) - neutral).abs()[0, 0]
        second = (model.decode(lower_right, observed, mask) - neutral).abs()[0, 0]

    def centroid(response):
        y, x = torch.meshgrid(
            torch.arange(32, dtype=response.dtype),
            torch.arange(32, dtype=response.dtype),
            indexing="ij",
        )
        mass = response.sum().clamp_min(1e-8)
        return (response * y).sum() / mass, (response * x).sum() / mass

    first_y, first_x = centroid(first)
    second_y, second_x = centroid(second)
    assert first.sum() > 0 and second.sum() > 0
    # Decoder convolutions intentionally broaden the response, but its center
    # must still move materially with the intervened latent cell.
    assert first_y + 2 < second_y
    assert first_x + 2 < second_x


def test_rejects_non_divisible_spatial_size():
    model = _model()
    observed = torch.rand(1, 1, 30, 30)
    mask = torch.zeros(1, 1, 30, 30)
    try:
        model.reconstruct(observed, mask)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gradients_flow_to_posterior_and_decoder():
    model = _model()
    observed = torch.rand(1, 1, 32, 32)
    mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
    target = torch.rand(1, 1, 32, 32)
    reconstruction, mu, logvar = model(target, observed, mask)
    ((1.0 - mask) * reconstruction).sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_fair_models_have_equal_latent_scalar_budget_at_128_pixels():
    observed = torch.zeros(1, 1, 128, 128)
    global_model = ConditionalVAE(channels=(32, 64, 128, 256), latent_dim=256)
    spatial_model = SpatialConditionalVAE(
        channels=(32, 64, 128, 256), latent_channels=1
    )
    assert global_model.latent_scalar_count(observed) == 256
    assert spatial_model.latent_scalar_count(observed) == 256


def test_seeded_sampling_is_reproducible():
    model = _model().eval()
    observed = torch.zeros(1, 1, 32, 32)
    mask = torch.zeros_like(observed)
    first = model.sample(
        observed,
        mask,
        num_samples=3,
        generator=torch.Generator().manual_seed(91),
    )
    second = model.sample(
        observed,
        mask,
        num_samples=3,
        generator=torch.Generator().manual_seed(91),
    )
    assert torch.equal(first, second)


def test_spatial_cvae_supports_bilinear_decoder_upsampling():
    model = SpatialConditionalVAE(
        channels=(8, 16, 32), latent_channels=2, upsampling_mode="bilinear"
    )
    observed = torch.rand(1, 1, 32, 32)
    mask = (torch.rand_like(observed) > 0.5).float()
    samples = model.sample(observed, mask, num_samples=2)
    assert samples.shape == (1, 2, 1, 32, 32)
    assert torch.allclose(samples[:, :, 0][..., mask[0, 0].bool()], observed[0, 0][mask[0, 0].bool()])
