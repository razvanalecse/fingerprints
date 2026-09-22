import random

import numpy as np
import pytest

from fingerprint_reconstruction.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences():
    first_state = seed_everything(2026)
    first_python = [random.random() for _ in range(4)]
    first_numpy = np.random.random(4)

    second_state = seed_everything(2026)
    second_python = [random.random() for _ in range(4)]
    second_numpy = np.random.random(4)

    assert first_state == second_state
    assert first_python == second_python
    assert np.array_equal(first_numpy, second_numpy)
    assert first_state.python_hash_seed == "2026"


@pytest.mark.parametrize("invalid", [-1, 1.5, "7"])
def test_invalid_seed_is_rejected(invalid):
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_everything(invalid)
