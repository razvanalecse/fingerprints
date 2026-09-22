"""Utilities for deterministic experiment execution."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """Recorded seed policy for an experiment."""

    seed: int
    deterministic_torch: bool
    python_hash_seed: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int, deterministic_torch: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch when available.

    PyTorch is imported lazily so preprocessing remains usable on CPU-only
    machines without the training dependencies installed.
    """

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    return SeedState(
        seed=seed,
        deterministic_torch=deterministic_torch,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
    )
