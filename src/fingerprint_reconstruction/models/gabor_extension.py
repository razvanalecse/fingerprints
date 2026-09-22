"""Parameter-free ridge continuation with orientation/frequency-steered Gabor filters.

Orientation and ridge frequency are estimated from observed pixels with masked
(normalised) convolution, extended into missing areas at progressively coarser
scales, and ridges are then propagated inward from the observed boundary by
iterating a steered Gabor filter bank while re-imposing the observed pixels.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion, gaussian_filter, sobel
from torch import nn
from torch.nn import functional as F

EPS = 1e-6


def _masked_smooth(field: np.ndarray, weight: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    return gaussian_filter(field * weight, sigma), gaussian_filter(weight, sigma)


def _extend(field: np.ndarray, weight: np.ndarray, sigma: float, fill: float) -> np.ndarray:
    """Normalised convolution, falling back to coarser scales where support is thin."""

    result = np.full(field.shape, fill, dtype=np.float64)
    assigned = np.zeros(field.shape, dtype=bool)
    for scale in (sigma, 2 * sigma, 4 * sigma, 8 * sigma):
        numerator, denominator = _masked_smooth(field, weight, scale)
        usable = (denominator > 0.04) & ~assigned
        result[usable] = numerator[usable] / denominator[usable]
        assigned |= usable
    return result


class GaborRidgeExtension(nn.Module):
    def __init__(
        self,
        *,
        orientation_bins: int = 16,
        frequency_bins: int = 3,
        kernel_size: int = 31,
        iterations: int = 24,
        sigma_across: float = 4.0,
        sigma_along: float = 7.0,
        structure_sigma: float = 5.0,
        min_period: float = 6.0,
        max_period: float = 12.0,
        amplitude_floor: float = 0.25,
    ) -> None:
        super().__init__()
        self.orientation_bins = orientation_bins
        self.frequency_bins = frequency_bins
        self.iterations = iterations
        self.structure_sigma = structure_sigma
        self.frequency_range = (1.0 / max_period, 1.0 / min_period)
        self.amplitude_floor = amplitude_floor
        # Sobel attenuates high frequencies: response(f) = sinc-like derivative gain * [1,2,1]/4 smoothing gain.
        axis_f = np.linspace(1e-3, 0.25, 512)
        gain = np.sin(2 * math.pi * axis_f) / (2 * math.pi * axis_f) * (1 + np.cos(2 * math.pi * axis_f)) / 2
        self._sobel_axis = axis_f
        self._sobel_response = axis_f * gain
        radius = kernel_size // 2
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        frequencies = torch.linspace(*self.frequency_range, frequency_bins)
        kernels = []
        for f_index in range(frequency_bins):
            for o_index in range(orientation_bins):
                theta = math.pi * o_index / orientation_bins
                across = xx * math.cos(theta) + yy * math.sin(theta)
                along = -xx * math.sin(theta) + yy * math.cos(theta)
                envelope = torch.exp(-0.5 * (across / sigma_across) ** 2 - 0.5 * (along / sigma_along) ** 2)
                kernel = envelope * torch.cos(2 * math.pi * frequencies[f_index] * across)
                kernel = kernel - envelope * (kernel.sum() / envelope.sum())
                kernels.append(kernel / (0.5 * envelope.sum()))
        self.register_buffer("bank", torch.stack(kernels).unsqueeze(1), persistent=False)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _estimate_fields(self, image: np.ndarray, known: np.ndarray):
        weight = known.astype(np.float64)
        local_mean = _extend(image, weight, 6.0, float(image[known].mean()))
        signal = np.where(known, image - local_mean, 0.0)
        valid = binary_erosion(known, iterations=2).astype(np.float64)
        gx = sobel(signal, axis=1) / 8.0
        gy = sobel(signal, axis=0) / 8.0
        jxx = _extend(gx * gx, valid, self.structure_sigma, 0.0)
        jyy = _extend(gy * gy, valid, self.structure_sigma, 0.0)
        jxy = _extend(gx * gy, valid, self.structure_sigma, 0.0)
        variance = _extend(signal * signal, weight, self.structure_sigma, 0.0)
        theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        frequency = np.sqrt((jxx + jyy) / (variance + EPS)) / (2 * math.pi)
        frequency = np.clip(np.interp(frequency, self._sobel_response, self._sobel_axis), *self.frequency_range)
        amplitude = np.sqrt(2.0 * variance)
        return local_mean, signal, theta, frequency, amplitude

    def _reconstruct_single(self, image: np.ndarray, known: np.ndarray) -> np.ndarray:
        local_mean, signal, theta, frequency, amplitude = self._estimate_fields(image, known)
        height, width = image.shape
        orientation_index = np.round((theta % math.pi) / math.pi * self.orientation_bins).astype(int) % self.orientation_bins
        frequency_index = np.clip(
            np.round(
                (frequency - self.frequency_range[0])
                / (self.frequency_range[1] - self.frequency_range[0])
                * (self.frequency_bins - 1)
            ).astype(int),
            0,
            self.frequency_bins - 1,
        )
        bank_index = torch.from_numpy(frequency_index * self.orientation_bins + orientation_index).long()
        known_signal = torch.from_numpy(signal).float()
        known_mask = torch.from_numpy(known)
        amplitude_t = torch.from_numpy(amplitude).float()
        state = known_signal.clone()
        pad = self.bank.shape[-1] // 2
        gather_index = bank_index.unsqueeze(0).unsqueeze(0)
        for _ in range(self.iterations):
            responses = F.conv2d(state[None, None], self.bank, padding=pad)
            filtered = torch.gather(responses, 1, gather_index)[0, 0]
            local_rms = torch.sqrt(F.avg_pool2d(filtered[None, None] ** 2, 7, 1, 3)[0, 0] + EPS)
            target = amplitude_t / math.sqrt(2.0)
            scale = target / torch.maximum(local_rms, self.amplitude_floor * target + EPS)
            state = torch.where(known_mask, known_signal, torch.clamp(filtered * scale, -1.0, 1.0))
        result = np.clip(local_mean + state.numpy(), 0.0, 1.0)
        return np.where(known, image, result).astype(np.float32)

    def reconstruct(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if observed.shape != mask.shape or observed.ndim != 4 or observed.shape[1] != 1:
            raise ValueError("observed and mask must both have shape [B, 1, H, W]")
        outputs = []
        for image_tensor, mask_tensor in zip(observed[:, 0], mask[:, 0]):
            image = image_tensor.detach().cpu().numpy().astype(np.float64)
            known = mask_tensor.detach().cpu().numpy() >= 0.5
            if known.sum() < 64:
                filled = np.full(image.shape, 0.5, dtype=np.float32)
                filled[known] = image[known]
            else:
                filled = self._reconstruct_single(image, known)
            outputs.append(torch.from_numpy(filled).unsqueeze(0))
        return torch.stack(outputs, dim=0).to(device=observed.device, dtype=observed.dtype)
