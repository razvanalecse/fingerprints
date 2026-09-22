"""Region-aware deterministic reconstruction losses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, region: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if values.shape != region.shape:
        raise ValueError("values and region must have identical shapes")
    weights = region.to(dtype=values.dtype)
    denominator = weights.sum().clamp_min(epsilon)
    return (values * weights).sum() / denominator


@dataclass(frozen=True)
class ReconstructionLossOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]


class MaskedReconstructionLoss(nn.Module):
    """L1/MSE objective reported separately on missing and observed regions."""

    def __init__(
        self,
        *,
        missing_l1_weight: float = 1.0,
        missing_mse_weight: float = 0.25,
        observed_l1_weight: float = 0.05,
        orientation_weight: float = 0.0,
        orientation_window: int = 9,
        orientation_epsilon: float = 1e-6,
        gradient_weight: float = 0.0,
        ridge_energy_weight: float = 0.0,
        ridge_sigma: float = 1.5,
        ridge_energy_window: int = 17,
        band_energy_weight: float = 0.0,
        band_sigma_low: float = 1.5,
        band_sigma_high: float = 4.0,
        fine_energy_weight: float = 0.0,
        fine_sigma: float = 0.8,
        ridge_band_weight: float = 0.0,
        ridge_band_frequencies: Sequence[float] = (0.18, 0.22, 0.26, 0.30),
        ridge_band_orientations: int = 8,
        ridge_band_kernel_size: int = 17,
        ridge_band_pool_size: int = 9,
        ridge_spectrum_weight: float = 0.0,
        ridge_spectrum_temperature: float = 0.25,
        predicted_orientation_consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        weights = (
            missing_l1_weight,
            missing_mse_weight,
            observed_l1_weight,
            orientation_weight,
            gradient_weight,
            ridge_energy_weight,
            band_energy_weight,
            fine_energy_weight,
            ridge_band_weight,
            ridge_spectrum_weight,
            predicted_orientation_consistency_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("loss weights must be non-negative and not all zero")
        self.missing_l1_weight = float(missing_l1_weight)
        self.missing_mse_weight = float(missing_mse_weight)
        self.observed_l1_weight = float(observed_l1_weight)
        if orientation_window < 3 or orientation_window % 2 == 0:
            raise ValueError("orientation_window must be an odd integer >= 3")
        self.orientation_weight = float(orientation_weight)
        self.orientation_window = int(orientation_window)
        self.orientation_epsilon = float(orientation_epsilon)
        if gradient_weight < 0 or ridge_energy_weight < 0:
            raise ValueError("structural loss weights must be non-negative")
        if ridge_sigma <= 0:
            raise ValueError("ridge_sigma must be positive")
        if ridge_energy_window < 3 or ridge_energy_window % 2 == 0:
            raise ValueError("ridge_energy_window must be an odd integer >= 3")
        self.gradient_weight = float(gradient_weight)
        self.ridge_energy_weight = float(ridge_energy_weight)
        self.ridge_sigma = float(ridge_sigma)
        self.ridge_energy_window = int(ridge_energy_window)
        if min(band_sigma_low, band_sigma_high, fine_sigma) <= 0:
            raise ValueError("band/fine sigmas must be positive")
        if band_sigma_high <= band_sigma_low:
            raise ValueError("band_sigma_high must exceed band_sigma_low")
        self.band_energy_weight = float(band_energy_weight)
        self.band_sigma_low = float(band_sigma_low)
        self.band_sigma_high = float(band_sigma_high)
        self.fine_energy_weight = float(fine_energy_weight)
        self.fine_sigma = float(fine_sigma)
        if ridge_band_weight < 0:
            raise ValueError("ridge_band_weight must be non-negative")
        frequencies = tuple(float(value) for value in ridge_band_frequencies)
        if not frequencies or any(not 0.0 < value < 0.5 for value in frequencies):
            raise ValueError("ridge-band frequencies must lie inside (0, 0.5)")
        if ridge_band_orientations < 2:
            raise ValueError("at least two ridge-band orientations are required")
        if ridge_band_kernel_size < 7 or ridge_band_kernel_size % 2 == 0:
            raise ValueError("ridge_band_kernel_size must be odd and >= 7")
        if ridge_band_pool_size < 1 or ridge_band_pool_size % 2 == 0:
            raise ValueError("ridge_band_pool_size must be odd and positive")
        if ridge_spectrum_weight < 0 or ridge_spectrum_temperature <= 0:
            raise ValueError("ridge spectrum weight must be non-negative and temperature positive")
        self.ridge_band_weight = float(ridge_band_weight)
        self.ridge_spectrum_weight = float(ridge_spectrum_weight)
        self.ridge_spectrum_temperature = float(ridge_spectrum_temperature)
        self.ridge_band_frequency_count = len(frequencies)
        self.ridge_band_orientation_count = int(ridge_band_orientations)
        self.predicted_orientation_consistency_weight = float(
            predicted_orientation_consistency_weight
        )
        self.ridge_band_pool_size = int(ridge_band_pool_size)
        real, imaginary = self._make_gabor_bank(
            frequencies,
            int(ridge_band_orientations),
            int(ridge_band_kernel_size),
        )
        self.register_buffer("ridge_band_real", real, persistent=False)
        self.register_buffer("ridge_band_imaginary", imaginary, persistent=False)

    @staticmethod
    def _make_gabor_bank(
        frequencies: Sequence[float], orientations: int, kernel_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-mean, unit-norm quadrature filters.

        Frequencies are cycles/pixel in the preprocessed image. Filter-energy
        matching is phase tolerant: it constrains ridge scale and direction,
        while pixel/noise objectives remain responsible for exact recovery.
        """

        radius = kernel_size // 2
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
        real_filters, imaginary_filters = [], []
        for frequency in frequencies:
            sigma_normal = max(1.5, 0.55 / frequency)
            sigma_tangent = 1.8 * sigma_normal
            for index in range(orientations):
                angle = math.pi * index / orientations
                normal = x * math.cos(angle) + y * math.sin(angle)
                tangent = -x * math.sin(angle) + y * math.cos(angle)
                envelope = torch.exp(
                    -0.5 * (
                        normal.square() / sigma_normal**2
                        + tangent.square() / sigma_tangent**2
                    )
                )
                phase = 2.0 * math.pi * frequency * normal
                real = envelope * torch.cos(phase)
                imaginary = envelope * torch.sin(phase)
                real = real - real.mean()
                imaginary = imaginary - imaginary.mean()
                real_filters.append(real / real.square().sum().sqrt().clamp_min(1e-8))
                imaginary_filters.append(
                    imaginary / imaginary.square().sum().sqrt().clamp_min(1e-8)
                )
        return (
            torch.stack(real_filters)[:, None],
            torch.stack(imaginary_filters)[:, None],
        )

    @staticmethod
    def _spatial_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sobel_x = image.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)
        return (
            F.conv2d(image, sobel_x, padding=1),
            F.conv2d(image, sobel_y, padding=1),
        )

    def _gradient_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        pred_x, pred_y = self._spatial_gradients(prediction)
        target_x, target_y = self._spatial_gradients(target)
        error = 0.5 * (torch.abs(pred_x - target_x) + torch.abs(pred_y - target_y))
        return masked_mean(error, region)

    def _gaussian_blur(
        self, image: torch.Tensor, sigma: float | None = None
    ) -> torch.Tensor:
        sigma = self.ridge_sigma if sigma is None else float(sigma)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
        return F.conv2d(image, kernel_2d, padding=radius)

    def _ridge_energy_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Match local ridge-scale high-pass energy without requiring exact phase."""

        pred_high = prediction - self._gaussian_blur(prediction)
        target_high = target - self._gaussian_blur(target)
        window = self.ridge_energy_window
        pred_energy = F.avg_pool2d(
            pred_high.square(), window, stride=1, padding=window // 2
        )
        target_energy = F.avg_pool2d(
            target_high.square(), window, stride=1, padding=window // 2
        )
        # Square roots put the quantity back on an intensity-amplitude scale.
        error = torch.abs(
            torch.sqrt(pred_energy + self.orientation_epsilon)
            - torch.sqrt(target_energy + self.orientation_epsilon)
        )
        return masked_mean(error, region)

    def _energy_match(
        self,
        prediction_band: torch.Tensor,
        target_band: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        window = self.ridge_energy_window
        prediction_energy = F.avg_pool2d(
            prediction_band.square(), window, stride=1, padding=window // 2
        )
        target_energy = F.avg_pool2d(
            target_band.square(), window, stride=1, padding=window // 2
        )
        error = torch.abs(
            torch.sqrt(prediction_energy + self.orientation_epsilon)
            - torch.sqrt(target_energy + self.orientation_epsilon)
        )
        return masked_mean(error, region)

    def _band_energy_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Match phase-tolerant energy in a difference-of-Gaussians ridge band."""

        def band(image: torch.Tensor) -> torch.Tensor:
            return self._gaussian_blur(image, self.band_sigma_low) - self._gaussian_blur(
                image, self.band_sigma_high
            )

        return self._energy_match(band(prediction), band(target), region)

    def _fine_energy_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Match fine-scale energy separately to expose speckle hallucination."""

        def fine(image: torch.Tensor) -> torch.Tensor:
            return image - self._gaussian_blur(image, self.fine_sigma)

        return self._energy_match(fine(prediction), fine(target), region)

    def _orientation_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Axial structure-tensor loss, invariant under theta -> theta + pi.

        The doubled-angle tensor vector ``(Jxx-Jyy, 2Jxy)`` represents local
        ridge orientation. Target tensor strength weights the loss so uniform
        background and poorly defined orientation contribute negligibly.
        """

        def tensor_vector(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            gx, gy = self._spatial_gradients(image)
            window = self.orientation_window
            jxx = F.avg_pool2d(gx.square(), window, stride=1, padding=window // 2)
            jyy = F.avg_pool2d(gy.square(), window, stride=1, padding=window // 2)
            jxy = F.avg_pool2d(gx * gy, window, stride=1, padding=window // 2)
            return jxx - jyy, 2.0 * jxy

        pred_x, pred_y = tensor_vector(prediction)
        target_x, target_y = tensor_vector(target)
        # Epsilon is inside sqrt: d(sqrt(x))/dx is singular at x=0 and caused
        # NaN gradients on textureless regions during long MPS training runs.
        pred_norm = torch.sqrt(pred_x.square() + pred_y.square() + self.orientation_epsilon)
        target_norm = torch.sqrt(target_x.square() + target_y.square() + self.orientation_epsilon)
        denominator = pred_norm * target_norm
        cosine = (pred_x * target_x + pred_y * target_y) / denominator
        circular_error = 1.0 - cosine.clamp(-1.0, 1.0)

        # Relative per-image reliability prevents high-contrast samples from
        # dominating the batch while suppressing textureless background.
        scale = target_norm.mean(dim=(-2, -1), keepdim=True).clamp_min(self.orientation_epsilon)
        reliability = (target_norm / (3.0 * scale)).clamp(0.0, 1.0)
        weights = region * reliability
        return (circular_error * weights).sum() / weights.sum().clamp_min(self.orientation_epsilon)

    def _predicted_orientation_consistency(
        self,
        prediction: torch.Tensor,
        predicted_structure: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        """Align generated ridge tangents with a leakage-safe predicted field."""

        if predicted_structure.ndim != 4 or predicted_structure.shape[1] != 4:
            raise ValueError("predicted_structure must have shape [B,4,h,w]")
        structure = F.interpolate(
            predicted_structure,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gx, gy = self._spatial_gradients(prediction)
        window = self.orientation_window
        jxx = F.avg_pool2d(gx.square(), window, stride=1, padding=window // 2)
        jyy = F.avg_pool2d(gy.square(), window, stride=1, padding=window // 2)
        jxy = F.avg_pool2d(gx * gy, window, stride=1, padding=window // 2)
        normal_x, normal_y = jxx - jyy, 2.0 * jxy
        normal_norm = torch.sqrt(
            normal_x.square() + normal_y.square() + self.orientation_epsilon
        )
        # The structure predictor emits the doubled-angle ridge tangent. A
        # pi/2 rotation from gradient normal to tangent negates both doubled-
        # angle coordinates.
        ridge_x, ridge_y = -normal_x / normal_norm, -normal_y / normal_norm
        target_x, target_y = structure[:, 1:2], structure[:, 2:3]
        target_norm = torch.sqrt(
            target_x.square() + target_y.square() + self.orientation_epsilon
        )
        target_x, target_y = target_x / target_norm, target_y / target_norm
        error = 1.0 - (ridge_x * target_x + ridge_y * target_y).clamp(-1.0, 1.0)
        reliability = structure[:, :1] * structure[:, 3:4]
        strength_scale = normal_norm.mean(dim=(-2, -1), keepdim=True).clamp_min(
            self.orientation_epsilon
        )
        strength = (normal_norm / (3.0 * strength_scale)).clamp(0.0, 1.0)
        weights = region * reliability * strength
        return (error * weights).sum() / weights.sum().clamp_min(self.orientation_epsilon)

    def _ridge_band_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Match local energy in an oriented empirical ridge-frequency bank."""

        difference = torch.abs(
            self._ridge_band_log_energy(prediction) - self._ridge_band_log_energy(target)
        )
        expanded_region = region.expand(-1, difference.shape[1], -1, -1)
        return masked_mean(difference, expanded_region)

    def _ridge_band_log_energy(self, image: torch.Tensor) -> torch.Tensor:
        """Return local phase-tolerant log energy for every frequency/orientation."""

        padding = self.ridge_band_real.shape[-1] // 2
        real = F.conv2d(image, self.ridge_band_real, padding=padding)
        imaginary = F.conv2d(image, self.ridge_band_imaginary, padding=padding)
        energy = real.square() + imaginary.square()
        pool = self.ridge_band_pool_size
        energy = F.avg_pool2d(energy, pool, stride=1, padding=pool // 2)
        return torch.log1p(energy)

    def _ridge_spectrum_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
    ) -> torch.Tensor:
        """Match the relative local distribution over empirical ridge frequencies.

        Energy is averaged over orientation before a temperature-softmax over
        frequency.  Consequently this term penalizes choosing the wrong ridge
        spacing even when total high-frequency energy is approximately correct.
        """

        def distribution(image: torch.Tensor) -> torch.Tensor:
            energy = self._ridge_band_log_energy(image)
            batch, _, height, width = energy.shape
            energy = energy.view(
                batch,
                self.ridge_band_frequency_count,
                self.ridge_band_orientation_count,
                height,
                width,
            ).mean(dim=2)
            return torch.softmax(energy / self.ridge_spectrum_temperature, dim=1)

        difference = torch.abs(distribution(prediction) - distribution(target))
        return masked_mean(difference, region.expand_as(difference))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        predicted_structure: torch.Tensor | None = None,
    ) -> ReconstructionLossOutput:
        if prediction.shape != target.shape or prediction.shape != mask.shape:
            raise ValueError("prediction, target, and mask must have identical shapes")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("inputs must have shape [B, 1, H, W]")
        missing = 1.0 - mask
        absolute_error = torch.abs(prediction - target)
        squared_error = torch.square(prediction - target)
        missing_l1 = masked_mean(absolute_error, missing)
        missing_mse = masked_mean(squared_error, missing)
        observed_l1 = masked_mean(absolute_error, mask)
        orientation = self._orientation_loss(prediction, target, missing)
        gradient = self._gradient_loss(prediction, target, missing)
        ridge_energy = self._ridge_energy_loss(prediction, target, missing)
        band_energy = (
            self._band_energy_loss(prediction, target, missing)
            if self.band_energy_weight > 0
            else prediction.new_zeros(())
        )
        fine_energy = (
            self._fine_energy_loss(prediction, target, missing)
            if self.fine_energy_weight > 0
            else prediction.new_zeros(())
        )
        ridge_band = (
            self._ridge_band_loss(prediction, target, missing)
            if self.ridge_band_weight > 0
            else prediction.new_zeros(())
        )
        ridge_spectrum = (
            self._ridge_spectrum_loss(prediction, target, missing)
            if self.ridge_spectrum_weight > 0
            else prediction.new_zeros(())
        )
        if self.predicted_orientation_consistency_weight > 0:
            if predicted_structure is None:
                raise ValueError(
                    "predicted_structure is required when consistency weight is positive"
                )
            predicted_orientation = self._predicted_orientation_consistency(
                prediction, predicted_structure, missing
            )
        else:
            predicted_orientation = prediction.new_zeros(())
        total = (
            self.missing_l1_weight * missing_l1
            + self.missing_mse_weight * missing_mse
            + self.observed_l1_weight * observed_l1
            + self.orientation_weight * orientation
            + self.gradient_weight * gradient
            + self.ridge_energy_weight * ridge_energy
            + self.band_energy_weight * band_energy
            + self.fine_energy_weight * fine_energy
            + self.ridge_band_weight * ridge_band
            + self.ridge_spectrum_weight * ridge_spectrum
            + self.predicted_orientation_consistency_weight * predicted_orientation
        )
        return ReconstructionLossOutput(
            total=total,
            components={
                "missing_l1": missing_l1.detach(),
                "missing_mse": missing_mse.detach(),
                "observed_l1": observed_l1.detach(),
                "orientation": orientation.detach(),
                "gradient": gradient.detach(),
                "ridge_energy": ridge_energy.detach(),
                "band_energy": band_energy.detach(),
                "fine_energy": fine_energy.detach(),
                "ridge_band": ridge_band.detach(),
                "ridge_spectrum": ridge_spectrum.detach(),
                "predicted_orientation_consistency": predicted_orientation.detach(),
            },
        )


class RegisteredApproximateLoss(MaskedReconstructionLoss):
    """Confidence-weighted loss for approximately registered latent/exemplar pairs.

    Unlike :class:`MaskedReconstructionLoss`, the pseudo-target is supervised only
    inside ``evaluation_roi``.  Continuous ``geometric_confidence`` weights reduce
    the influence of pixels far from examiner-verified correspondence anchors.
    The observed-data term is compared with the latent observation, never with the
    warped exemplar.  This preserves the distinction between an approximate
    structural target and exact pixel ground truth.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        observed: torch.Tensor,
        evaluation_roi: torch.Tensor,
        geometric_confidence: torch.Tensor,
    ) -> ReconstructionLossOutput:
        tensors = (target, mask, observed, evaluation_roi, geometric_confidence)
        if any(tensor.shape != prediction.shape for tensor in tensors):
            raise ValueError("all registered-loss tensors must have identical shapes")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("inputs must have shape [B, 1, H, W]")
        if torch.any(geometric_confidence < 0) or torch.any(geometric_confidence > 1):
            raise ValueError("geometric_confidence must lie in [0, 1]")

        region = evaluation_roi.to(prediction.dtype) * geometric_confidence.to(
            prediction.dtype
        )
        pseudo_absolute = torch.abs(prediction - target)
        pseudo_squared = torch.square(prediction - target)
        missing_l1 = masked_mean(pseudo_absolute, region)
        missing_mse = masked_mean(pseudo_squared, region)
        observed_l1 = masked_mean(torch.abs(prediction - observed), mask)
        orientation = self._orientation_loss(prediction, target, region)
        gradient = self._gradient_loss(prediction, target, region)
        ridge_energy = self._ridge_energy_loss(prediction, target, region)
        band_energy = (
            self._band_energy_loss(prediction, target, region)
            if self.band_energy_weight > 0
            else prediction.new_zeros(())
        )
        fine_energy = (
            self._fine_energy_loss(prediction, target, region)
            if self.fine_energy_weight > 0
            else prediction.new_zeros(())
        )
        ridge_band = (
            self._ridge_band_loss(prediction, target, region)
            if self.ridge_band_weight > 0
            else prediction.new_zeros(())
        )
        ridge_spectrum = (
            self._ridge_spectrum_loss(prediction, target, region)
            if self.ridge_spectrum_weight > 0
            else prediction.new_zeros(())
        )
        total = (
            self.missing_l1_weight * missing_l1
            + self.missing_mse_weight * missing_mse
            + self.observed_l1_weight * observed_l1
            + self.orientation_weight * orientation
            + self.gradient_weight * gradient
            + self.ridge_energy_weight * ridge_energy
            + self.band_energy_weight * band_energy
            + self.fine_energy_weight * fine_energy
            + self.ridge_band_weight * ridge_band
            + self.ridge_spectrum_weight * ridge_spectrum
        )
        return ReconstructionLossOutput(
            total=total,
            components={
                "registered_l1": missing_l1.detach(),
                "registered_mse": missing_mse.detach(),
                "observed_l1": observed_l1.detach(),
                "orientation": orientation.detach(),
                "gradient": gradient.detach(),
                "ridge_energy": ridge_energy.detach(),
                "band_energy": band_energy.detach(),
                "fine_energy": fine_energy.detach(),
                "ridge_band": ridge_band.detach(),
                "ridge_spectrum": ridge_spectrum.detach(),
                "effective_geometric_weight": region.mean().detach(),
            },
        )
