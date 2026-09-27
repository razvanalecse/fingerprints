#!/usr/bin/env python3
"""Visualize latent-support-constrained reconstruction without claiming full recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor
from fingerprint_reconstruction.training.trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--reconstruction-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    args = parser.parse_args()
    device = select_device(args.device)
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots={"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root},
        split="validation",
        output_shape=(128, 128),
    )
    reconstruction_checkpoint = torch.load(args.reconstruction_checkpoint, map_location=device, weights_only=False)
    reconstruction_model = build_reconstruction_model(
        reconstruction_checkpoint["config"]["model"],
        channels=tuple(reconstruction_checkpoint["channels_used"]),
    ).to(device)
    reconstruction_model.load_state_dict(reconstruction_checkpoint["model_state"])
    reconstruction_model.eval()
    support_checkpoint = torch.load(args.support_checkpoint, map_location=device, weights_only=False)
    support_model = FingerprintSupportPredictor(
        channels=tuple(support_checkpoint["channels_used"])
    ).to(device)
    support_model.load_state_dict(support_checkpoint["model_state"])
    support_model.eval()
    indices = np.linspace(0, len(dataset) - 1, args.rows).round().astype(int)
    figure, axes = plt.subplots(args.rows, 6, figsize=(17, 2.8 * args.rows), squeeze=False)
    with torch.no_grad():
        for row_index, dataset_index in enumerate(indices):
            item = dataset[int(dataset_index)]
            observed = item["observed"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device)
            raw = reconstruction_model(torch.cat((observed, mask), dim=1))
            unconstrained = mask * observed + (1.0 - mask) * raw
            support_probability = torch.maximum(
                support_model(torch.cat((observed, mask), dim=1)), mask
            )
            if not 0.0 < args.support_threshold < 1.0:
                raise ValueError("support-threshold must lie strictly between zero and one")
            # The soft probability is useful for calibration, but directly multiplying
            # by it leaks low-amplitude ridge texture into the background.  The displayed
            # constrained output therefore uses an inference-only binary decision made
            # solely from (Y, M); no target/registered-support pixels are accessed here.
            support_binary = (support_probability >= args.support_threshold).to(raw.dtype)
            support_binary = torch.maximum(support_binary, mask)
            missing_gated = support_binary * raw + (1.0 - support_binary)
            constrained = mask * observed + (1.0 - mask) * missing_gated
            shown_observed = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
            panels = (
                item["latent_image"][0].numpy(),
                shown_observed[0, 0].cpu().numpy(),
                item["target"][0].numpy(),
                unconstrained[0, 0].cpu().numpy(),
                support_binary[0, 0].cpu().numpy(),
                constrained[0, 0].cpu().numpy(),
            )
            titles = (
                "Real latent",
                "Y, M",
                "X_pseudo",
                "Full extrapolation",
                f"Predicted support (p>={args.support_threshold:g})",
                "Hard support-constrained",
            )
            for axis, panel, title in zip(axes[row_index], panels, titles):
                axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
                axis.set_title(title, fontsize=9)
                axis.axis("off")
    figure.suptitle(
        "Two distinct outputs: full-print extrapolation vs. latent-support-constrained enhancement",
        fontsize=12,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
