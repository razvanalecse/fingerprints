import torch

from fingerprint_reconstruction.models.support import FingerprintSupportPredictor, support_loss


def test_support_predictor_shape_and_probability_range():
    model = FingerprintSupportPredictor(channels=(8, 16))
    observed = torch.rand(2, 1, 32, 32)
    mask = (torch.rand_like(observed) > 0.5).float()
    probability = model(torch.cat((observed, mask), dim=1))
    assert probability.shape == observed.shape
    assert torch.all((probability >= 0) & (probability <= 1))


def test_support_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    probability = torch.sigmoid(logits)
    target = (torch.rand_like(probability) > 0.4).float()
    loss, terms = support_loss(probability, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"bce", "dice_loss"}
    assert logits.grad is not None
