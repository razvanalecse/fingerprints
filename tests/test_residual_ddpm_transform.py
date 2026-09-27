import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_nist302_residual_ddpm import from_residual_space, to_residual_space


def test_residual_roundtrip_is_exact():
    torch.manual_seed(0)
    image = torch.rand(2, 1, 8, 8)
    coarse = torch.rand(2, 1, 8, 8)
    residual = to_residual_space(image, coarse)
    recovered = from_residual_space(residual, coarse)
    assert torch.allclose(recovered, image, atol=1e-6)


def test_residual_space_is_bounded_in_unit_interval():
    torch.manual_seed(1)
    image = torch.rand(4, 1, 8, 8)
    coarse = torch.rand(4, 1, 8, 8)
    residual = to_residual_space(image, coarse)
    assert residual.min() >= 0.0 and residual.max() <= 1.0


def test_perfect_coarse_prediction_gives_neutral_residual():
    image = torch.rand(1, 1, 4, 4)
    residual = to_residual_space(image, image)
    assert torch.allclose(residual, torch.full_like(residual, 0.5), atol=1e-6)


def test_from_residual_space_clamps_out_of_range_values():
    coarse = torch.full((1, 1, 2, 2), 0.5)
    extreme_high = torch.full((1, 1, 2, 2), 5.0)
    extreme_low = torch.full((1, 1, 2, 2), -5.0)
    assert torch.all(from_residual_space(extreme_high, coarse) <= 1.0)
    assert torch.all(from_residual_space(extreme_low, coarse) >= 0.0)
