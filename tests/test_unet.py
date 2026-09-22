import torch

from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.unet import FingerprintUNet
from fingerprint_reconstruction.models.gated_conv import (
    GatedFingerprintNetwork,
    StructureConditionedGatedNetwork,
)


def test_unet_preserves_shape_and_range():
    model = FingerprintUNet(channels=(8, 16, 32))
    conditioning = torch.rand(2, 2, 65, 71)
    output = model(conditioning)
    assert output.shape == (2, 1, 65, 71)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_unet_reconstruction_is_exact_on_observed_pixels():
    model = FingerprintUNet(channels=(8, 16))
    observed = torch.rand(1, 1, 32, 32)
    mask = torch.zeros_like(observed)
    mask[:, :, :, :13] = 1.0
    reconstruction = model.reconstruct(observed, mask)
    assert torch.equal(reconstruction[mask.bool()], observed[mask.bool()])


def test_neutral_encoding_preserves_shape_and_is_differentiable():
    model = FingerprintUNet(channels=(8, 16), neutral_missing_encoding=True)
    conditioning = torch.rand(2, 2, 32, 32, requires_grad=True)
    output = model(conditioning)
    output.mean().backward()
    assert output.shape == (2, 1, 32, 32)
    assert conditioning.grad is not None


def test_resize_convolution_decoder_preserves_odd_shape():
    model = FingerprintUNet(channels=(8, 16, 32), upsampling_mode="resize_conv")
    output = model(torch.rand(2, 2, 65, 71))
    assert output.shape == (2, 1, 65, 71)


def test_masked_loss_uses_missing_region_as_primary_domain():
    target = torch.zeros(1, 1, 4, 4)
    prediction = torch.zeros_like(target)
    mask = torch.ones_like(target)
    mask[:, :, :, 2:] = 0.0
    prediction[:, :, :, 2:] = 1.0
    output = MaskedReconstructionLoss(
        missing_l1_weight=1.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
    )(prediction, target, mask)
    assert output.total.item() == 1.0
    assert output.components["observed_l1"].item() == 0.0


def test_orientation_loss_is_axial_and_prefers_matching_ridges():
    coordinates = torch.arange(64, dtype=torch.float32)
    vertical = (0.5 + 0.5 * torch.sin(coordinates / 2.0))[None, None, None, :].expand(1, 1, 64, 64)
    horizontal = vertical.transpose(-1, -2)
    mask = torch.zeros_like(vertical)
    objective = MaskedReconstructionLoss(
        missing_l1_weight=0.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
        orientation_weight=1.0,
    )
    matching = objective(vertical, vertical, mask).components["orientation"]
    orthogonal = objective(horizontal, vertical, mask).components["orientation"]
    assert matching.item() < 1e-2
    assert orthogonal.item() > 1.0


def test_orientation_loss_has_finite_gradient_on_uniform_images():
    prediction = torch.full((2, 1, 32, 32), 0.5, requires_grad=True)
    target = torch.full_like(prediction, 0.5)
    mask = torch.zeros_like(prediction)
    objective = MaskedReconstructionLoss(orientation_weight=0.2)
    loss = objective(prediction, target, mask).total
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ridge_energy_distinguishes_texture_from_flat_prediction():
    coordinates = torch.arange(64, dtype=torch.float32)
    target = (0.5 + 0.4 * torch.sin(coordinates / 1.7))[None, None, None, :].expand(1, 1, 64, 64)
    flat = torch.full_like(target, 0.5)
    mask = torch.zeros_like(target)
    objective = MaskedReconstructionLoss(
        missing_l1_weight=0.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
        ridge_energy_weight=1.0,
    )
    matching = objective(target, target, mask).components["ridge_energy"]
    flattened = objective(flat, target, mask).components["ridge_energy"]
    assert matching.item() < 1e-6
    assert flattened.item() > 0.02


def test_ridge_band_loss_prefers_correct_frequency_and_has_finite_gradients():
    coordinates = torch.arange(64, dtype=torch.float32)
    target = (0.5 + 0.4 * torch.sin(2 * torch.pi * 0.22 * coordinates))[None, None, None, :].expand(1, 1, 64, 64)
    wrong = (0.5 + 0.4 * torch.sin(2 * torch.pi * 0.10 * coordinates))[None, None, None, :].expand(1, 1, 64, 64).clone().requires_grad_(True)
    mask = torch.zeros_like(target)
    objective = MaskedReconstructionLoss(
        missing_l1_weight=0.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
        ridge_band_weight=1.0,
    )
    matching = objective(target, target, mask).components["ridge_band"]
    mismatching_output = objective(wrong, target, mask)
    mismatching_output.total.backward()
    assert matching.item() < 1e-7
    assert mismatching_output.components["ridge_band"].item() > 0.01
    assert torch.isfinite(wrong.grad).all()


def test_structure_conditioned_gated_network_preserves_observed_pixels():
    model = StructureConditionedGatedNetwork(
        channels=(8, 16, 32), structure_channels=(8, 16, 32, 64)
    )
    observed = torch.rand(2, 1, 64, 64)
    mask = (torch.rand_like(observed) > 0.5).float()
    reconstruction = model.reconstruct(observed, mask)
    assert reconstruction.shape == observed.shape
    assert torch.equal(reconstruction[mask.bool()], observed[mask.bool()])
    assert not any(parameter.requires_grad for parameter in model.structure_predictor.parameters())


def test_structure_conditioned_initialization_preserves_base_at_zero_auxiliary_paths():
    base = GatedFingerprintNetwork(channels=(8, 16, 32))
    conditioned = StructureConditionedGatedNetwork(
        channels=(8, 16, 32), structure_channels=(8, 16, 32, 64)
    )
    conditioned.initialize_coarse_from_base_state(base.state_dict())
    observed = torch.rand(1, 1, 64, 64)
    mask = (torch.rand_like(observed) > 0.5).float()
    with torch.no_grad():
        base_output = base(torch.cat((observed, mask), dim=1))
        conditioned_output = conditioned(torch.cat((observed, mask), dim=1))
    assert torch.allclose(base_output, conditioned_output, atol=1e-6)


def test_predicted_orientation_consistency_prefers_matching_tangent():
    coordinates = torch.arange(64, dtype=torch.float32)
    # Vertical intensity variation gives horizontal ridges (theta=0).
    horizontal_ridges = (0.5 + 0.4 * torch.sin(coordinates / 1.7))[None, None, :, None].expand(1, 1, 64, 64)
    vertical_ridges = horizontal_ridges.transpose(-1, -2)
    mask = torch.zeros_like(horizontal_ridges)
    structure = torch.zeros(1, 4, 16, 16)
    structure[:, 0] = 1.0
    structure[:, 1] = 1.0  # cos(2 theta), theta=0
    structure[:, 3] = 1.0
    objective = MaskedReconstructionLoss(
        missing_l1_weight=0.0,
        missing_mse_weight=0.0,
        observed_l1_weight=0.0,
        predicted_orientation_consistency_weight=1.0,
    )
    matching = objective(
        horizontal_ridges, horizontal_ridges, mask, predicted_structure=structure
    ).components["predicted_orientation_consistency"]
    orthogonal = objective(
        vertical_ridges, horizontal_ridges, mask, predicted_structure=structure
    ).components["predicted_orientation_consistency"]
    assert matching.item() < 0.05
    assert orthogonal.item() > 1.5
