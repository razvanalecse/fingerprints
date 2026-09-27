"""Shared loader for the three NIST302 base models used by the ensemble scripts.

Not a package module (importable script, matching this project's existing
convention of e.g. `from train_nist302_cvae import ...`). Loads pixel DDPM,
residual DDPM (+ its frozen coarse model), and spatial CVAE, all frozen, and
exposes one function returning each model's support-constrained predictive
mean over K samples for a given (observed, mask) batch -- the common
interface the ensemble combiner and its evaluation both need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from fingerprint_reconstruction.models.cvae import SpatialConditionalVAE
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import apply_support_constraint
from train_nist302_cvae import constrain_sample_batch, load_support_predictor, sample_in_chunks, support_probability

MODEL_ORDER = ("pixel_ddpm", "residual_ddpm", "cvae")
PIXEL_DDPM_CONFIG = Path("configs/nist302_registered_ddpm.yaml")
PIXEL_DDPM_CHECKPOINT = Path("outputs/nist302_registered_ddpm_full/best-checkpoint.pt")
RESIDUAL_DDPM_CONFIG = Path("configs/nist302_registered_residual_ddpm.yaml")
RESIDUAL_DDPM_CHECKPOINT = Path("outputs/nist302_registered_residual_ddpm_full/best-checkpoint.pt")
CVAE_CONFIG = Path("configs/nist302_cvae_fair_spatial.yaml")
CVAE_CHECKPOINT = Path("outputs/nist302_cvae_fair_spatial_full/best-checkpoint.pt")
SUPPORT_CHECKPOINT = Path("outputs/nist302_visible_support_predictor_full/best-checkpoint.pt")


def to_residual_space(image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    return (0.5 + 0.5 * (image - coarse)).clamp(0.0, 1.0)


def from_residual_space(residual_image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    return (coarse + 2.0 * (residual_image - 0.5)).clamp(0.0, 1.0)


@dataclass
class EnsembleBaseModels:
    device: torch.device
    support_model: torch.nn.Module
    pixel_ddpm: ConditionalDDPM
    pixel_ddim_steps: int
    residual_ddpm: ConditionalDDPM
    residual_coarse: torch.nn.Module
    residual_ddim_steps: int
    cvae: SpatialConditionalVAE
    cvae_chunk_size: int
    cvae_support_mode: str
    cvae_support_threshold: float


def load_ddpm(
    config_path: Path, checkpoint_path: Path, device: torch.device, *, auxiliary_condition_channels: int = 1
) -> tuple[ConditionalDDPM, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scheduler = DDPMScheduler(
        timesteps=int(checkpoint["timesteps"]), schedule=str(config["diffusion"]["schedule"]),
        beta_start=float(config["diffusion"]["beta_start"]), beta_end=float(config["diffusion"]["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=tuple(checkpoint["channels_used"]), time_dim=int(checkpoint["time_dim"]),
        multiscale_conditioning=bool(config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=auxiliary_condition_channels,
        middle_attention=bool(config["model"]["middle_attention"]),
        attention_heads=int(config["model"]["attention_heads"]),
        upsampling_mode=str(config["model"]["upsampling_mode"]),
    )
    model = ConditionalDDPM(denoiser, scheduler).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model, config


def load_base_models(device: torch.device) -> EnsembleBaseModels:
    support_model = load_support_predictor(SUPPORT_CHECKPOINT, device)
    pixel_ddpm, pixel_config = load_ddpm(PIXEL_DDPM_CONFIG, PIXEL_DDPM_CHECKPOINT, device, auxiliary_condition_channels=1)
    residual_ddpm, residual_config = load_ddpm(
        RESIDUAL_DDPM_CONFIG, RESIDUAL_DDPM_CHECKPOINT, device, auxiliary_condition_channels=2
    )
    residual_coarse_checkpoint = torch.load(
        Path(residual_config["model"]["coarse_checkpoint"]), map_location=device, weights_only=False
    )
    residual_coarse = build_reconstruction_model(
        residual_coarse_checkpoint["config"]["model"],
        channels=tuple(residual_coarse_checkpoint["channels_used"]),
    ).to(device)
    residual_coarse.load_state_dict(residual_coarse_checkpoint["model_state"])
    residual_coarse.eval().requires_grad_(False)

    cvae_config = yaml.safe_load(CVAE_CONFIG.read_text(encoding="utf-8"))
    cvae_checkpoint = torch.load(CVAE_CHECKPOINT, map_location=device, weights_only=False)
    cvae = SpatialConditionalVAE(
        channels=tuple(cvae_checkpoint["channels_used"]),
        latent_channels=int(cvae_checkpoint["latent_dim"]),
        upsampling_mode="bilinear",
    ).to(device)
    cvae.load_state_dict(cvae_checkpoint["model_state"])
    cvae.eval().requires_grad_(False)

    return EnsembleBaseModels(
        device=device, support_model=support_model,
        pixel_ddpm=pixel_ddpm, pixel_ddim_steps=int(pixel_config["evaluation"]["ddim_steps"]),
        residual_ddpm=residual_ddpm, residual_coarse=residual_coarse,
        residual_ddim_steps=int(residual_config["evaluation"]["ddim_steps"]),
        cvae=cvae, cvae_chunk_size=int(cvae_config["evaluation"]["sample_chunk_size"]),
        cvae_support_mode=str(cvae_config["evaluation"]["support_mode"]),
        cvae_support_threshold=float(cvae_config["evaluation"]["support_threshold"]),
    )


@torch.no_grad()
def base_model_predictions(
    models: EnsembleBaseModels, observed: torch.Tensor, mask: torch.Tensor, *,
    k: int = 5, generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Return each base model's support-constrained predictive mean, shape [B,1,H,W]."""

    support = support_probability(models.support_model, observed, mask)

    pixel_samples = models.pixel_ddpm.sample_ddim(
        observed, mask, inference_steps=models.pixel_ddim_steps, num_samples=k,
        eta=0.0, enforce_data_consistency=True, auxiliary_condition=support, generator=generator,
    )
    b, kk, _, h, w = pixel_samples.shape
    pixel_samples = apply_support_constraint(
        pixel_samples.reshape(b * kk, 1, h, w), observed.repeat_interleave(kk, dim=0),
        mask.repeat_interleave(kk, dim=0), support.repeat_interleave(kk, dim=0),
        mode="soft", threshold=0.5,
    ).reshape(b, kk, 1, h, w)
    pixel_mean = pixel_samples.mean(dim=1)

    coarse = mask * observed + (1.0 - mask) * models.residual_coarse.reconstruct(observed, mask)
    observed_r = to_residual_space(observed, coarse)
    residual_auxiliary = torch.cat((support, coarse), dim=1)
    residual_samples_r = models.residual_ddpm.sample_ddim(
        observed_r, mask, inference_steps=models.residual_ddim_steps, num_samples=k,
        eta=0.0, enforce_data_consistency=True, auxiliary_condition=residual_auxiliary, generator=generator,
    )
    b, kk, _, h, w = residual_samples_r.shape
    residual_samples = from_residual_space(
        residual_samples_r, coarse.unsqueeze(1).expand(-1, kk, -1, -1, -1)
    )
    expanded_mask = mask.unsqueeze(1).expand(-1, kk, -1, -1, -1)
    expanded_observed = observed.unsqueeze(1).expand(-1, kk, -1, -1, -1)
    residual_samples = expanded_mask * expanded_observed + (1.0 - expanded_mask) * residual_samples
    residual_samples = apply_support_constraint(
        residual_samples.reshape(b * kk, 1, h, w), observed.repeat_interleave(kk, dim=0),
        mask.repeat_interleave(kk, dim=0), support.repeat_interleave(kk, dim=0),
        mode="soft", threshold=0.5,
    ).reshape(b, kk, 1, h, w)
    residual_mean = residual_samples.mean(dim=1)

    cvae_samples = sample_in_chunks(
        models.cvae, observed, mask, num_samples=k, chunk_size=models.cvae_chunk_size, generator=generator,
    )
    cvae_samples = constrain_sample_batch(
        cvae_samples, observed, mask, support,
        mode=models.cvae_support_mode, threshold=models.cvae_support_threshold,
    )
    cvae_mean = cvae_samples.mean(dim=1)

    return {"pixel_ddpm": pixel_mean, "residual_ddpm": residual_mean, "cvae": cvae_mean, "support": support}
