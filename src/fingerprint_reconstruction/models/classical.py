"""Non-learned lower-bound reconstruction baselines."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn


class NearestObservedInpainting(nn.Module):
    """Fill each missing pixel from its nearest observed spatial neighbour.

    This deliberately simple interpolation is deterministic, parameter-free,
    and preserves all observed pixels exactly. It is a lower bound rather than
    a fingerprint-specific reconstruction method.
    """

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must both have shape [B, 1, H, W]")
        outputs = []
        for image_tensor, mask_tensor in zip(observed[:, 0], mask[:, 0]):
            image = image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            known = mask_tensor.detach().cpu().numpy() >= 0.5
            if not known.any():
                raise ValueError("nearest interpolation requires at least one observed pixel")
            indices = distance_transform_edt(
                ~known, return_distances=False, return_indices=True
            )
            filled = image[tuple(indices)]
            filled[known] = image[known]
            outputs.append(torch.from_numpy(filled.copy()).unsqueeze(0))
        return torch.stack(outputs, dim=0).to(device=observed.device, dtype=observed.dtype)
