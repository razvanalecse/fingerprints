import torch

from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor, structure_loss


def test_structure_predictor_outputs_latent_scale_maps():
    model = FingerprintStructurePredictor(channels=(8, 16, 32, 64))
    observed = torch.rand(2, 1, 128, 128)
    mask = (torch.rand_like(observed) > 0.5).float()
    prediction = model(observed, mask)
    assert prediction.shape == (2, 4, 32, 32)
    assert torch.all((prediction[:, :1] >= 0) & (prediction[:, :1] <= 1))
    norms = torch.linalg.vector_norm(prediction[:, 1:3], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_structure_loss_is_finite_and_differentiable():
    model = FingerprintStructurePredictor(channels=(8, 16, 32, 64))
    observed = torch.rand(2, 1, 128, 128)
    mask = (torch.rand_like(observed) > 0.5).float()
    prediction = model(observed, mask)
    target = torch.zeros(2, 5, 32, 32)
    target[:, 0] = 1
    target[:, 1] = 1
    target[:, 3] = 0.8
    target[:, 4] = 1
    loss, terms = structure_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"support_bce", "support_dice", "orientation", "coherence"}
