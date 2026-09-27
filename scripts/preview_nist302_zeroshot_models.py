#!/usr/bin/env python3
"""Visual comparison of SOCOFing-trained models on one registered SD302 latent."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.classical import NearestObservedInpainting
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.training.trainer import select_device


def _load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots={
            "sd302a": args.sd302a_root,
            "sd302b": args.sd302b_root,
            "sd302d": args.sd302d_root,
        },
        split="validation",
        output_shape=(128, 128),
    )
    item = dataset[args.index]
    observed = item["observed"].unsqueeze(0)
    mask = item["mask"].unsqueeze(0)
    device = select_device(args.device)
    models = {
        "Classical": NearestObservedInpainting().to(device),
        "U-Net zero-shot": _load_model(args.unet, device),
        "Gated zero-shot": _load_model(args.gated, device),
    }
    with torch.no_grad():
        reconstructions = {
            name: model.reconstruct(observed.to(device), mask.to(device))[0, 0]
            .cpu()
            .numpy()
            for name, model in models.items()
        }
    target = item["target"][0].numpy()
    shown_observed = np.where(item["mask"][0].numpy() > 0.5, item["latent_image"][0], 0.72)
    top = (
        (item["latent_image"][0], "Real latent"),
        (shown_observed, "Conditioning Y,M"),
        (target, "Registered X_pseudo"),
        (reconstructions["Classical"], "Classical"),
        (reconstructions["U-Net zero-shot"], "U-Net zero-shot"),
        (reconstructions["Gated zero-shot"], "Gated zero-shot"),
    )
    bottom = (
        (item["mask"][0], "Observed mask M", "gray", (0, 1)),
        (item["evaluation_roi"][0], "Evaluation ROI", "magma", (0, 1)),
        (
            item["distance_to_nearest_correspondence_mm"][0],
            "Nearest verified point (mm)",
            "viridis",
            None,
        ),
        (np.abs(reconstructions["Classical"] - target), "Classical |error|", "inferno", (0, 1)),
        (np.abs(reconstructions["U-Net zero-shot"] - target), "U-Net |error|", "inferno", (0, 1)),
        (np.abs(reconstructions["Gated zero-shot"] - target), "Gated |error|", "inferno", (0, 1)),
    )
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5), constrained_layout=True)
    for axis, (image, title) in zip(axes[0], top):
        axis.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
    for axis, (image, title, cmap, limits) in zip(axes[1], bottom):
        artist = axis.imshow(
            np.asarray(image), cmap=cmap, vmin=limits[0] if limits else None, vmax=limits[1] if limits else None
        )
        axis.set_title(title)
        if limits is None:
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "SOCOFing-trained deterministic models — zero-shot SD302 validation",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
