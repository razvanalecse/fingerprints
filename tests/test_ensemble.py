import torch
import pytest

from fingerprint_reconstruction.models.ensemble import EnsembleCombiner


def test_ensemble_combiner_output_shape():
    model = EnsembleCombiner(num_models=3, channels=(4, 4))
    observed = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) > 0.5).float()
    predictions = torch.rand(2, 3, 1, 16, 16)
    blended, weights = model(observed, mask, predictions)
    assert blended.shape == (2, 1, 16, 16)
    assert weights.shape == (2, 3, 16, 16)


def test_ensemble_combiner_weights_sum_to_one():
    model = EnsembleCombiner(num_models=3, channels=(4,))
    observed = torch.rand(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    predictions = torch.rand(1, 3, 1, 8, 8)
    _, weights = model(observed, mask, predictions)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1, 8, 8), atol=1e-5, rtol=0)


def test_ensemble_combiner_blend_is_a_convex_combination():
    model = EnsembleCombiner(num_models=2, channels=(4,))
    observed = torch.rand(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    predictions = torch.stack((torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)), dim=1)
    blended, weights = model(observed, mask, predictions)
    # With predictions of exactly 0 and 1, the blend must lie in [0,1] (a true
    # convex combination), not overshoot outside the range of its inputs.
    assert blended.min() >= 0.0
    assert blended.max() <= 1.0


def test_ensemble_combiner_rejects_mismatched_num_models():
    model = EnsembleCombiner(num_models=3, channels=(4,))
    observed = torch.rand(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    wrong_predictions = torch.rand(1, 2, 1, 8, 8)
    with pytest.raises(ValueError):
        model(observed, mask, wrong_predictions)


def test_ensemble_combiner_rejects_too_few_models():
    with pytest.raises(ValueError):
        EnsembleCombiner(num_models=1)


def test_ensemble_combiner_is_trainable_with_backprop():
    model = EnsembleCombiner(num_models=2, channels=(4,))
    observed = torch.rand(2, 1, 10, 10)
    mask = torch.ones(2, 1, 10, 10)
    predictions = torch.rand(2, 2, 1, 10, 10, requires_grad=False)
    target = torch.rand(2, 1, 10, 10)
    blended, _ = model(observed, mask, predictions)
    loss = torch.nn.functional.mse_loss(blended, target)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
