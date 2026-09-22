"""Latent diffusion of ridge-texture residuals around a deterministic coarse image."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.models.diffusion import (
    DDPMScheduler,
    DiffusionUNet,
    ddim_step,
    make_ddim_timesteps,
)
from fingerprint_reconstruction.models.latent_diffusion.autoencoder import FingerprintAutoencoderKL


def normalized_gaussian_measurement_projection(
    decoded: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Project a low-pass measurement residual without bleeding across holes."""

    if sigma <= 0:
        return mask * observed + (1.0 - mask) * decoded
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=decoded.device, dtype=decoded.dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = decoded.shape[1]
    horizontal = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    def blur(values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (radius, radius, 0, 0), mode="reflect")
        values = F.conv2d(values, horizontal, groups=channels)
        values = F.pad(values, (0, 0, radius, radius), mode="reflect")
        return F.conv2d(values, vertical, groups=channels)

    weighted_residual = blur(mask * (observed - decoded))
    available_weight = blur(mask.expand_as(decoded)).clamp_min(epsilon)
    correction = weighted_residual / available_weight
    return decoded + mask * correction


class ResidualConditionalLatentDDPM(nn.Module):
    """Model ``E(X)-E(X_coarse)`` rather than the complete image latent."""

    def __init__(
        self,
        autoencoder: FingerprintAutoencoderKL,
        coarse_predictor: nn.Module,
        denoiser: DiffusionUNet,
        scheduler: DDPMScheduler,
        *,
        latent_scale: float,
        residual_scale: float,
        structure_predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if latent_scale <= 0 or residual_scale <= 0:
            raise ValueError("latent and residual scales must be positive")
        latent_channels = autoencoder.latent_channels
        if denoiser.data_channels != latent_channels:
            raise ValueError("denoiser data channels must equal latent channels")
        if denoiser.condition_channels != 2 * latent_channels:
            raise ValueError("residual denoiser condition must concatenate observed and coarse latents")
        if structure_predictor is not None and denoiser.auxiliary_condition_channels != 4:
            raise ValueError("structure guidance requires four auxiliary channels")
        self.autoencoder = autoencoder
        self.coarse_predictor = coarse_predictor
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.structure_predictor = structure_predictor
        self.register_buffer("latent_scale", torch.tensor(float(latent_scale)))
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        for frozen in (self.autoencoder, self.coarse_predictor, self.structure_predictor):
            if frozen is not None:
                frozen.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        self.coarse_predictor.eval()
        if self.structure_predictor is not None:
            self.structure_predictor.eval()
        return self

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(image, sample=False) * self.latent_scale

    @torch.no_grad()
    def predict_coarse(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hasattr(self.coarse_predictor, "reconstruct"):
            return self.coarse_predictor.reconstruct(observed, mask)
        raw = self.coarse_predictor(torch.cat((observed, mask), dim=1))
        return mask * observed + (1.0 - mask) * raw

    @torch.no_grad()
    def encode_inputs(self, observed: torch.Tensor, mask: torch.Tensor):
        neutral = mask * observed + (1.0 - mask) * 0.5
        observed_latent = self.encode(neutral)
        coarse_pixels = self.predict_coarse(observed, mask)
        coarse_latent = self.encode(coarse_pixels)
        condition = torch.cat((observed_latent, coarse_latent), dim=1)
        latent_mask = F.interpolate(mask, size=coarse_latent.shape[-2:], mode="nearest")
        structure = (
            self.structure_predictor(observed, mask)
            if self.structure_predictor is not None
            else None
        )
        return condition, latent_mask, coarse_pixels, coarse_latent, observed_latent, structure

    def training_loss(
        self, target: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            target_latent = self.encode(target)
            condition, latent_mask, _, coarse_latent, _, structure = self.encode_inputs(
                observed, mask
            )
            residual = (target_latent - coarse_latent) * self.residual_scale
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (target.shape[0],), device=target.device
        )
        noisy, noise = self.scheduler.q_sample(residual, timesteps)
        predicted_noise = self.denoiser(
            noisy, timesteps, condition, latent_mask, structure
        )
        return torch.mean((predicted_noise - noise).square())

    @torch.no_grad()
    def sample_ddim(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        inference_steps: int = 50,
        num_samples: int = 1,
        eta: float = 0.0,
        latent_data_consistency: bool = True,
        pixel_projection_interval: int = 0,
        pixel_projection_strength: float = 1.0,
        frequency_guidance_max_sigma: float = 0.0,
        frequency_guidance_power: float = 1.0,
    ) -> torch.Tensor:
        if num_samples <= 0 or pixel_projection_interval < 0:
            raise ValueError("invalid sample count or projection interval")
        if not 0.0 <= pixel_projection_strength <= 1.0:
            raise ValueError("pixel_projection_strength must lie in [0,1]")
        if frequency_guidance_max_sigma < 0 or frequency_guidance_power <= 0:
            raise ValueError("invalid frequency-guidance schedule")
        batch = observed.shape[0]
        condition, latent_mask, _, coarse_latent, observed_latent, structure = self.encode_inputs(
            observed, mask
        )
        # Deterministic encoders are evaluated once per input. Their outputs are
        # then replicated across Monte-Carlo samples, avoiding K redundant
        # passes through the large coarse predictor and structure branch.
        condition = condition.repeat_interleave(num_samples, dim=0)
        latent_mask = latent_mask.repeat_interleave(num_samples, dim=0)
        coarse_latent = coarse_latent.repeat_interleave(num_samples, dim=0)
        observed_latent = observed_latent.repeat_interleave(num_samples, dim=0)
        if structure is not None:
            structure = structure.repeat_interleave(num_samples, dim=0)
        observed = observed.repeat_interleave(num_samples, dim=0)
        mask = mask.repeat_interleave(num_samples, dim=0)
        residual = torch.randn_like(coarse_latent)
        known_residual = (observed_latent - coarse_latent) * self.residual_scale
        known_noise = torch.randn_like(residual)
        schedule = make_ddim_timesteps(self.scheduler.timesteps, inference_steps)
        for index, step_tensor in enumerate(schedule):
            step = int(step_tensor)
            previous = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
            if latent_data_consistency:
                alpha = self.scheduler.alpha_bar[step]
                noised_known = torch.sqrt(alpha) * known_residual + torch.sqrt(1.0 - alpha) * known_noise
                residual = latent_mask * noised_known + (1.0 - latent_mask) * residual
            timesteps = torch.full(
                (residual.shape[0],), step, device=residual.device, dtype=torch.long
            )
            epsilon = self.denoiser(
                residual, timesteps, condition, latent_mask, structure
            )
            predicted_residual = self.scheduler.predict_x0(
                residual, timesteps, epsilon
            ).clamp(-6.0, 6.0)
            if pixel_projection_interval and (
                index % pixel_projection_interval == 0 or previous < 0
            ):
                full_latent = coarse_latent + predicted_residual / self.residual_scale
                decoded = self.autoencoder.decode(full_latent / self.latent_scale)
                progress = index / max(len(schedule) - 1, 1)
                sigma = frequency_guidance_max_sigma * (
                    max(0.0, 1.0 - progress) ** frequency_guidance_power
                )
                # The final projection is exact even when the schedule would
                # leave a small positive floating-point sigma.
                if previous < 0:
                    sigma = 0.0
                projected = normalized_gaussian_measurement_projection(
                    decoded, observed, mask, sigma=sigma
                )
                projected_latent = self.encode(projected)
                projected_residual = (
                    projected_latent - coarse_latent
                ) * self.residual_scale
                predicted_residual = (
                    (1.0 - pixel_projection_strength) * predicted_residual
                    + pixel_projection_strength * projected_residual
                )
            alpha_previous = (
                self.scheduler.alpha_bar[previous]
                if previous >= 0
                else residual.new_ones(())
            )
            residual = ddim_step(
                residual,
                predicted_residual,
                epsilon,
                alpha_bar_t=self.scheduler.alpha_bar[step],
                alpha_bar_previous=alpha_previous,
                eta=eta,
            )
        full_latent = coarse_latent + residual / self.residual_scale
        decoded = self.autoencoder.decode(full_latent / self.latent_scale)
        reconstruction = mask * observed + (1.0 - mask) * decoded
        return reconstruction.reshape(batch, num_samples, 1, *observed.shape[-2:])


@torch.no_grad()
def estimate_residual_scale(
    autoencoder: FingerprintAutoencoderKL,
    coarse_predictor: nn.Module,
    loader,
    device: torch.device,
    *,
    latent_scale: float,
    max_batches: int = 32,
    epsilon: float = 1e-8,
) -> float:
    """Estimate inverse standard deviation of target-minus-coarse latents."""

    values = []
    autoencoder.eval(); coarse_predictor.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        if hasattr(coarse_predictor, "reconstruct"):
            coarse = coarse_predictor.reconstruct(observed, mask)
        else:
            raw = coarse_predictor(torch.cat((observed, mask), dim=1))
            coarse = mask * observed + (1.0 - mask) * raw
        target_latent = autoencoder.encode(target, sample=False) * latent_scale
        coarse_latent = autoencoder.encode(coarse, sample=False) * latent_scale
        values.append((target_latent - coarse_latent).flatten().cpu())
    if not values:
        raise ValueError("residual-scale loader produced no batches")
    standard_deviation = torch.cat(values).std(unbiased=True).item()
    if standard_deviation <= epsilon:
        raise ValueError("residual standard deviation is zero or invalid")
    return 1.0 / standard_deviation
