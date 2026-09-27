import torch

from fingerprint_reconstruction.models.gated_conv import GatedFingerprintNetwork


def test_gated_network_shape_range_and_gradient():
    model = GatedFingerprintNetwork(channels=(8, 16, 32))
    conditioning = torch.rand(2, 2, 65, 71, requires_grad=True)
    output = model(conditioning)
    output.mean().backward()
    assert output.shape == (2, 1, 65, 71)
    assert torch.all((output >= 0) & (output <= 1))
    assert conditioning.grad is not None


def test_gated_network_data_consistency_is_exact():
    model = GatedFingerprintNetwork(channels=(8, 16, 32))
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed)
    mask[:, :, 4:20, 3:17] = 1
    reconstructed = model.reconstruct(observed, mask)
    assert torch.equal(reconstructed[mask.bool()], observed[mask.bool()])
