import torch

from fingerprint_reconstruction.models.classical import NearestObservedInpainting


def test_nearest_inpainting_preserves_observations_and_fills_domain():
    observed = torch.zeros(1, 1, 7, 7)
    mask = torch.zeros_like(observed)
    observed[0, 0, 1, 1] = 0.25
    observed[0, 0, 5, 5] = 0.75
    mask[0, 0, 1, 1] = 1.0
    mask[0, 0, 5, 5] = 1.0
    result = NearestObservedInpainting().reconstruct(observed, mask)
    assert result.shape == observed.shape
    assert torch.equal(result[mask.bool()], observed[mask.bool()])
    assert set(result.unique().tolist()) == {0.25, 0.75}


def test_nearest_inpainting_rejects_empty_observation():
    observed = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(observed)
    try:
        NearestObservedInpainting().reconstruct(observed, mask)
    except ValueError as error:
        assert "at least one observed" in str(error)
    else:
        raise AssertionError("empty observation must fail")
