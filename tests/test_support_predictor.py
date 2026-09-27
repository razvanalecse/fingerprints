import torch

import pytest

from fingerprint_reconstruction.models.support import (
    REGISTERED_APPROXIMATE_FULL_SUPPORT,
    VISIBLE_LATENT_SUPPORT,
    FingerprintSupportPredictor,
    apply_support_constraint,
    build_support_target,
    support_loss,
)


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


def test_visible_support_target_uses_official_quality_at_least_one():
    batch = {
        "quality": torch.tensor([[[[0, 1], [2, 5]]]]),
        "mask": torch.zeros(1, 1, 2, 2),
        "registered_support": torch.ones(1, 1, 2, 2),
    }
    target = build_support_target(batch, VISIBLE_LATENT_SUPPORT, dtype=torch.float32)
    assert torch.equal(target, torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]]))


def test_approximate_full_support_is_observed_registered_union():
    batch = {
        "quality": torch.zeros(1, 1, 2, 3, dtype=torch.long),
        "mask": torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        "registered_support": torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]]),
    }
    target = build_support_target(
        batch, REGISTERED_APPROXIMATE_FULL_SUPPORT, dtype=torch.float32
    )
    assert torch.equal(
        target, torch.tensor([[[[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]])
    )


def test_unknown_support_target_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown support target mode"):
        build_support_target({}, "ambiguous", dtype=torch.float32)


def test_support_constraint_is_white_outside_and_data_consistent():
    reconstruction = torch.full((1, 1, 2, 3), 0.2)
    observed = torch.full_like(reconstruction, 0.7)
    mask = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    support = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 0.25, 0.0]]]])
    hard = apply_support_constraint(
        reconstruction, observed, mask, support, mode="hard", threshold=0.5
    )
    assert hard[0, 0, 0, 0] == pytest.approx(0.7)
    assert hard[0, 0, 0, 1] == pytest.approx(0.2)
    assert hard[0, 0, 0, 2] == pytest.approx(1.0)
    soft = apply_support_constraint(reconstruction, observed, mask, support, mode="soft")
    assert soft[0, 0, 1, 1] == pytest.approx(0.8)


def test_feathered_support_is_smooth_and_data_consistent():
    reconstruction = torch.zeros(1, 1, 17, 17)
    observed = torch.full_like(reconstruction, 0.25)
    mask = torch.zeros_like(reconstruction)
    mask[:, :, 8, 8] = 1
    support = torch.zeros_like(reconstruction)
    support[:, :, 5:12, 5:12] = 1

    output = apply_support_constraint(
        reconstruction,
        observed,
        mask,
        support,
        mode="feathered",
        threshold=0.5,
        feather_sigma=1.5,
    )

    assert output[0, 0, 8, 8] == observed[0, 0, 8, 8]
    assert output[0, 0, 0, 0] > 0.999
    assert output[0, 0, 8, 8] < output[0, 0, 3, 8] < output[0, 0, 0, 8]


def test_feathered_support_rejects_non_positive_sigma():
    tensor = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="feather sigma"):
        apply_support_constraint(
            tensor, tensor, tensor, tensor, mode="feathered", feather_sigma=0
        )
