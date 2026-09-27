"""Conditional diffusion over a frozen fingerprint autoencoder latent space."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fingerprint_reconstruction.losses.reconstruction import masked_mean
from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet, ddim_step, make_ddim_timesteps
from fingerprint_reconstruction.models.latent_diffusion.autoencoder import FingerprintAutoencoderKL


class ConditionalLatentDDPM(nn.Module):
    def __init__(
        self,
        autoencoder: FingerprintAutoencoderKL,
        denoiser: DiffusionUNet,
        scheduler: DDPMScheduler,
        *,
        latent_scale: float,
        structure_predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        if denoiser.data_channels != autoencoder.latent_channels:
            raise ValueError("denoiser data channels must equal autoencoder latent channels")
        if denoiser.condition_channels != autoencoder.latent_channels:
            raise ValueError("denoiser condition channels must equal autoencoder latent channels")
        self.autoencoder = autoencoder
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.structure_predictor = structure_predictor
        self.register_buffer("latent_scale", torch.tensor(float(latent_scale)))
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        if self.structure_predictor is not None:
            if self.denoiser.auxiliary_condition_channels != 4:
                raise ValueError("structure guidance requires four auxiliary channels")
            self.structure_predictor.requires_grad_(False)
            self.structure_predictor.eval()
        elif self.denoiser.auxiliary_condition_channels:
            raise ValueError("auxiliary denoiser requires a structure predictor")

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        if self.structure_predictor is not None:
            self.structure_predictor.eval()
        return self

    @torch.no_grad()
    def predict_structure(
        self, observed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | None:
        if self.structure_predictor is None:
            return None
        return self.structure_predictor(observed, mask)

    @torch.no_grad()
    def encode_target(self, image: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(image, sample=False) * self.latent_scale

    @torch.no_grad()
    def encode_condition(
        self, observed: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neutral_filled = mask * observed + (1.0 - mask) * 0.5
        condition = self.autoencoder.encode(neutral_filled, sample=False) * self.latent_scale
        latent_mask = F.interpolate(mask, size=condition.shape[-2:], mode="nearest")
        return condition, latent_mask

    def training_loss(
        self,
        target: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            latent_target = self.encode_target(target)
            condition, latent_mask = self.encode_condition(observed, mask)
            structure = self.predict_structure(observed, mask)
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (target.shape[0],), device=target.device
        )
        noisy, noise = self.scheduler.q_sample(latent_target, timesteps)
        predicted = self.denoiser(
            noisy, timesteps, condition, latent_mask, structure
        )
        # Diffusion predicts the full latent because spatial autoencoder
        # receptive fields prevent a strict pixel-mask/latent-mask equivalence.
        squared_error = (predicted - noise).square()
        if weight_map is None:
            return torch.mean(squared_error)
        # weight_map is pixel-resolution (e.g. evaluation_roi * geometric
        # confidence * visible support); downsample to the latent grid the
        # same way encode_condition downsamples the binary mask, then apply
        # it uniformly across latent channels.
        latent_weight = F.interpolate(
            weight_map, size=squared_error.shape[-2:], mode="nearest"
        )
        weighted = squared_error * latent_weight
        denominator = latent_weight.expand_as(squared_error).sum().clamp_min(1e-8)
        return weighted.sum() / denominator

    @torch.no_grad()
    def sample_ddim(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        inference_steps: int = 50,
        num_samples: int = 1,
        eta: float = 0.0,
        enforce_data_consistency: bool = True,
        latent_data_consistency: bool = False,
        pixel_projection_interval: int = 0,
        pixel_projection_strength: float = 1.0,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if pixel_projection_interval < 0:
            raise ValueError("pixel_projection_interval cannot be negative")
        if not 0.0 <= pixel_projection_strength <= 1.0:
            raise ValueError("pixel_projection_strength must lie in [0,1]")
        batch = observed.shape[0]
        observed_expanded = observed.repeat_interleave(num_samples, dim=0)
        mask_expanded = mask.repeat_interleave(num_samples, dim=0)
        condition, latent_mask = self.encode_condition(observed_expanded, mask_expanded)
        structure = self.predict_structure(observed_expanded, mask_expanded)
        latent = torch.randn_like(condition)
        known_noise = torch.randn_like(condition) if latent_data_consistency else None
        schedule = make_ddim_timesteps(self.scheduler.timesteps, inference_steps)
        for index, step_tensor in enumerate(schedule):
            step = int(step_tensor)
            previous = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
            if latent_data_consistency:
                alpha_current = self.scheduler.alpha_bar[step]
                noised_condition = (
                    torch.sqrt(alpha_current) * condition
                    + torch.sqrt(1.0 - alpha_current) * known_noise
                )
                latent = latent_mask * noised_condition + (1.0 - latent_mask) * latent
            timesteps = torch.full((latent.shape[0],), step, device=latent.device, dtype=torch.long)
            epsilon = self.denoiser(
                latent, timesteps, condition, latent_mask, structure
            )
            predicted_x0 = self.scheduler.predict_x0(latent, timesteps, epsilon).clamp(-6.0, 6.0)
            if pixel_projection_interval and (
                index % pixel_projection_interval == 0 or previous < 0
            ):
                decoded_x0 = self.autoencoder.decode(
                    predicted_x0 / self.latent_scale
                )
                projected_pixels = (
                    mask_expanded * observed_expanded
                    + (1.0 - mask_expanded) * decoded_x0
                )
                projected_latent = self.encode_target(projected_pixels)
                predicted_x0 = (
                    (1.0 - pixel_projection_strength) * predicted_x0
                    + pixel_projection_strength * projected_latent
                )
            alpha_previous = self.scheduler.alpha_bar[previous] if previous >= 0 else latent.new_ones(())
            latent = ddim_step(
                latent,
                predicted_x0,
                epsilon,
                alpha_bar_t=self.scheduler.alpha_bar[step],
                alpha_bar_previous=alpha_previous,
                eta=eta,
            )
        if latent_data_consistency:
            latent = latent_mask * condition + (1.0 - latent_mask) * latent
        decoded = self.autoencoder.decode(latent / self.latent_scale)
        reconstruction = (
            mask_expanded * observed_expanded + (1.0 - mask_expanded) * decoded
            if enforce_data_consistency
            else decoded
        )
        return reconstruction.reshape(batch, num_samples, 1, *observed.shape[-2:])


@torch.no_grad()
def estimate_latent_scale(
    autoencoder: FingerprintAutoencoderKL,
    loader,
    device: torch.device,
    *,
    max_batches: int = 64,
    epsilon: float = 1e-8,
) -> float:
    """Estimate 1/std(z) over deterministic posterior means, as in LDM scaling."""

    values = []
    autoencoder.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        values.append(autoencoder.encode(batch["target"].to(device), sample=False).flatten().cpu())
    if not values:
        raise ValueError("latent-scale loader produced no batches")
    standard_deviation = torch.cat(values).std(unbiased=True).item()
    if not standard_deviation > epsilon:
        raise ValueError("latent standard deviation is zero or invalid")
    return 1.0 / standard_deviation
