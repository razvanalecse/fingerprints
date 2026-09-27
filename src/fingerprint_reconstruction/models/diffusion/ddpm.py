"""Conditional pixel-space DDPM with observed-pixel reinjection."""

from __future__ import annotations

import torch
from torch import nn

from fingerprint_reconstruction.losses.reconstruction import masked_mean
from fingerprint_reconstruction.models.diffusion.ddim import ddim_step, make_ddim_timesteps
from fingerprint_reconstruction.models.diffusion.repaint import repaint_forward_step
from fingerprint_reconstruction.models.diffusion.scheduler import DDPMScheduler, extract
from fingerprint_reconstruction.models.diffusion.unet import DiffusionUNet


class ConditionalDDPM(nn.Module):
    def __init__(self, denoiser: DiffusionUNet, scheduler: DDPMScheduler) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler

    @staticmethod
    def _randn_like(
        reference: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(reference)
        # A CPU generator provides identical seeded streams for CPU and MPS.
        return torch.randn(
            reference.shape, generator=generator, dtype=reference.dtype
        ).to(reference.device)

    def training_loss(
        self,
        target: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        observed_weight: float = 0.05,
        missing_weight_map: torch.Tensor | None = None,
        auxiliary_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x0 = 2.0 * target - 1.0
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (target.shape[0],), device=target.device
        )
        xt, noise = self.scheduler.q_sample(x0, timesteps)
        predicted_noise = self.denoiser(
            xt, timesteps, observed, mask, auxiliary_condition
        )
        squared_error = (predicted_noise - noise).square()
        missing_region = 1.0 - mask
        if missing_weight_map is not None:
            if missing_weight_map.shape != target.shape:
                raise ValueError("missing_weight_map must have the target shape")
            if torch.any(missing_weight_map < 0):
                raise ValueError("missing_weight_map cannot contain negative weights")
            missing_region = missing_region * missing_weight_map.to(target.dtype)
        missing_loss = masked_mean(squared_error, missing_region)
        observed_loss = masked_mean(squared_error, mask)
        total = missing_loss + float(observed_weight) * observed_loss
        return total, {
            "missing_noise_mse": missing_loss.detach(),
            "observed_noise_mse": observed_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_samples: int = 1,
        enforce_data_consistency: bool = True,
        auxiliary_condition: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch = observed.shape[0]
        observed = observed.repeat_interleave(num_samples, dim=0)
        mask = mask.repeat_interleave(num_samples, dim=0)
        if auxiliary_condition is not None:
            auxiliary_condition = auxiliary_condition.repeat_interleave(
                num_samples, dim=0
            )
        x = self._randn_like(observed, generator)
        observed_x0 = 2.0 * observed - 1.0
        fixed_known_noise = self._randn_like(observed, generator)
        for step in reversed(range(self.scheduler.timesteps)):
            timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            epsilon = self.denoiser(
                x, timesteps, observed, mask, auxiliary_condition
            )
            predicted_x0 = self.scheduler.predict_x0(x, timesteps, epsilon).clamp(-1.0, 1.0)
            mean, variance = self.scheduler.posterior(x, timesteps, predicted_x0)
            x = (
                mean
                if step == 0
                else mean + torch.sqrt(variance) * self._randn_like(x, generator)
            )
            if enforce_data_consistency:
                if step == 0:
                    known = observed_x0
                else:
                    previous = torch.full_like(timesteps, step - 1)
                    known = (
                        extract(self.scheduler.sqrt_alpha_bar, previous, x.shape) * observed_x0
                        + extract(self.scheduler.sqrt_one_minus_alpha_bar, previous, x.shape)
                        * fixed_known_noise
                    )
                x = mask * known + (1.0 - mask) * x
        reconstruction = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        reconstruction = mask * observed + (1.0 - mask) * reconstruction
        return reconstruction.reshape(batch, num_samples, 1, *observed.shape[-2:])

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
        auxiliary_condition: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample over a reduced DDIM timestep sequence.

        Known pixels are reinjected at every selected noise level using one
        fixed noise realization per reconstruction. The returned image is
        finally projected exactly onto the observed data.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        schedule = make_ddim_timesteps(self.scheduler.timesteps, inference_steps)
        batch = observed.shape[0]
        observed = observed.repeat_interleave(num_samples, dim=0)
        mask = mask.repeat_interleave(num_samples, dim=0)
        if auxiliary_condition is not None:
            auxiliary_condition = auxiliary_condition.repeat_interleave(
                num_samples, dim=0
            )
        x = self._randn_like(observed, generator)
        observed_x0 = 2.0 * observed - 1.0
        fixed_known_noise = self._randn_like(observed, generator)
        for index, step_tensor in enumerate(schedule):
            step = int(step_tensor)
            previous_step = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
            timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            epsilon = self.denoiser(
                x, timesteps, observed, mask, auxiliary_condition
            )
            predicted_x0 = self.scheduler.predict_x0(x, timesteps, epsilon).clamp(-1.0, 1.0)
            alpha_t = self.scheduler.alpha_bar[step]
            alpha_previous = (
                self.scheduler.alpha_bar[previous_step]
                if previous_step >= 0
                else torch.ones((), device=x.device, dtype=x.dtype)
            )
            x = ddim_step(
                x,
                predicted_x0,
                epsilon,
                alpha_bar_t=alpha_t,
                alpha_bar_previous=alpha_previous,
                eta=eta,
            )
            if enforce_data_consistency:
                if previous_step < 0:
                    known = observed_x0
                else:
                    known = (
                        torch.sqrt(alpha_previous) * observed_x0
                        + torch.sqrt(1.0 - alpha_previous) * fixed_known_noise
                    )
                x = mask * known + (1.0 - mask) * x
        reconstruction = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        reconstruction = mask * observed + (1.0 - mask) * reconstruction
        return reconstruction.reshape(batch, num_samples, 1, *observed.shape[-2:])

    @torch.no_grad()
    def sample_repaint(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_samples: int = 1,
        resampling_steps: int = 2,
        enforce_data_consistency: bool = True,
        auxiliary_condition: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """RePaint-style ancestral sampling with U reverse/forward refinements.

        ``resampling_steps=1`` is the conditioning-only baseline described by
        Lugmayr et al.; larger U values repeatedly return from x_(t-1) to x_t
        and denoise again. Unlike the original unconditional RePaint prior,
        this fingerprint adaptation additionally conditions the denoiser on
        the observed image and binary mask.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if resampling_steps <= 0:
            raise ValueError("resampling_steps must be positive")
        batch = observed.shape[0]
        observed = observed.repeat_interleave(num_samples, dim=0)
        mask = mask.repeat_interleave(num_samples, dim=0)
        if auxiliary_condition is not None:
            auxiliary_condition = auxiliary_condition.repeat_interleave(
                num_samples, dim=0
            )
        x = self._randn_like(observed, generator)
        observed_x0 = 2.0 * observed - 1.0
        for step in reversed(range(self.scheduler.timesteps)):
            timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            refinements = resampling_steps if step > 0 else 1
            for refinement in range(refinements):
                epsilon = self.denoiser(
                    x, timesteps, observed, mask, auxiliary_condition
                )
                predicted_x0 = self.scheduler.predict_x0(x, timesteps, epsilon).clamp(
                    -1.0, 1.0
                )
                mean, variance = self.scheduler.posterior(x, timesteps, predicted_x0)
                x_previous = (
                    mean
                    if step == 0
                    else mean + torch.sqrt(variance) * self._randn_like(x, generator)
                )
                if enforce_data_consistency:
                    if step == 0:
                        known = observed_x0
                    else:
                        previous = torch.full_like(timesteps, step - 1)
                        known, _ = self.scheduler.q_sample(
                            observed_x0,
                            previous,
                            self._randn_like(observed_x0, generator),
                        )
                    x_previous = mask * known + (1.0 - mask) * x_previous
                if refinement + 1 < refinements:
                    x = repaint_forward_step(
                        x_previous,
                        step,
                        self.scheduler,
                        self._randn_like(x_previous, generator),
                    )
                else:
                    x = x_previous
        reconstruction = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        reconstruction = mask * observed + (1.0 - mask) * reconstruction
        return reconstruction.reshape(batch, num_samples, 1, *observed.shape[-2:])
