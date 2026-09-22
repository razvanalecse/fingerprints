import torch

from fingerprint_reconstruction.losses.autoencoder import AutoencoderKLLoss
from fingerprint_reconstruction.models.latent_diffusion import FingerprintAutoencoderKL


def test_autoencoder_shapes_and_compression_factor():
    model = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=4)
    image = torch.rand(2, 1, 32, 32)
    distribution = model.encode_distribution(image)
    assert distribution.mean.shape == (2, 4, 8, 8)
    assert distribution.logvar.shape == distribution.mean.shape
    reconstruction, _ = model(image, sample_posterior=False)
    assert reconstruction.shape == image.shape
    assert torch.all((reconstruction >= 0) & (reconstruction <= 1))


def test_autoencoder_loss_is_finite_and_backpropagates():
    model = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    target = torch.rand(2, 1, 32, 32)
    reconstruction, distribution = model(target)
    result = AutoencoderKLLoss()(reconstruction, target, distribution)
    result.total.backward()
    assert torch.isfinite(result.total)
    assert "kl" in result.components


def test_autoencoder_rejects_nondivisible_spatial_shape():
    model = FingerprintAutoencoderKL(channels=(8, 16), latent_channels=2)
    try:
        model.encode(torch.rand(1, 1, 31, 32))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid spatial shape was accepted")
