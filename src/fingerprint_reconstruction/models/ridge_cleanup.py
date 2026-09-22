"""Oriented-Gabor clean-up of a reconstruction, applied to the missing region only.

Orientation comes from the network output (it holds the model's extrapolated
ridge flow, smoothed heavily so speckle does not steer the filter); frequency
and contrast come from the observed pixels, which are trustworthy. Observed
pixels are never modified.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.models.gabor_extension import EPS, GaborRidgeExtension
from fingerprint_reconstruction.preprocessing.orientation import estimate_foreground_mask


class RidgeCleanup(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        iterations: int = 2,
        orientation_sigma: float = 6.0,
        max_gain: float = 2.5,
        blend: float = 1.0,
        start_distance: float = 6.0,
        ramp_distance: float = 8.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.gabor = GaborRidgeExtension(iterations=iterations, structure_sigma=orientation_sigma)
        self.iterations = iterations
        self.max_gain = max_gain
        self.blend = blend
        self.start_distance = start_distance
        self.ramp_distance = ramp_distance

    def _clean(self, reconstruction: np.ndarray, observed: np.ndarray, known: np.ndarray) -> np.ndarray:
        if known.sum() < 64:
            return reconstruction.astype(np.float32)
        g = self.gabor
        _, _, _, frequency, amplitude = g._estimate_fields(observed, known)
        local_mean, signal, theta, _, _ = g._estimate_fields(reconstruction, np.ones_like(known))
        orientation_index = np.round((theta % math.pi) / math.pi * g.orientation_bins).astype(int) % g.orientation_bins
        frequency_index = np.clip(
            np.round(
                (frequency - g.frequency_range[0])
                / (g.frequency_range[1] - g.frequency_range[0])
                * (g.frequency_bins - 1)
            ).astype(int),
            0,
            g.frequency_bins - 1,
        )
        gather_index = torch.from_numpy(frequency_index * g.orientation_bins + orientation_index).long()[None, None]
        known_signal = torch.from_numpy(signal).float()
        known_mask = torch.from_numpy(known)
        target = torch.from_numpy(amplitude).float() / math.sqrt(2.0)
        state = known_signal.clone()
        pad = g.bank.shape[-1] // 2
        for _ in range(self.iterations):
            responses = F.conv2d(F.pad(state[None, None], (pad,) * 4, mode="reflect"), g.bank)
            filtered = torch.gather(responses, 1, gather_index)[0, 0]
            rms = torch.sqrt(F.avg_pool2d(filtered[None, None] ** 2, 7, 1, 3)[0, 0] + EPS)
            scale = torch.clamp(target / torch.clamp(rms, min=EPS), max=self.max_gain)
            state = torch.where(known_mask, known_signal, torch.clamp(filtered * scale, -1.0, 1.0))
        cleaned = np.clip(local_mean + state.numpy(), 0.0, 1.0)
        distance = distance_transform_edt(~known)
        weight = self.blend * np.clip((distance - self.start_distance) / max(self.ramp_distance, EPS), 0.0, 1.0)
        foreground = gaussian_filter(estimate_foreground_mask(reconstruction).astype(np.float64), 2.0)
        weight = weight * foreground
        blended = weight * cleaned + (1.0 - weight) * reconstruction
        return np.where(known, observed, blended).astype(np.float32)

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        base_output = self.base.reconstruct(observed, mask)
        outputs = []
        for rec, obs, msk in zip(base_output[:, 0], observed[:, 0], mask[:, 0]):
            cleaned = self._clean(
                rec.detach().cpu().numpy().astype(np.float64),
                obs.detach().cpu().numpy().astype(np.float64),
                msk.detach().cpu().numpy() >= 0.5,
            )
            outputs.append(torch.from_numpy(cleaned).unsqueeze(0))
        return torch.stack(outputs, dim=0).to(device=base_output.device, dtype=base_output.dtype)
