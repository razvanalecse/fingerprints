#!/usr/bin/env python3
"""Render an auditable preview from an intermediate or final U-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from train_unet import save_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_reconstruction_model(
        config["model"], channels=tuple(checkpoint["channels_used"])
    )
    model.load_state_dict(checkpoint["model_state"])
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    epoch = int(checkpoint["epoch"]) + 1
    validation_loss = float(checkpoint["validation_loss"])
    architecture = str(config["model"].get("architecture", "unet"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_preview(
        model,
        batch,
        torch.device("cpu"),
        args.output,
        f"{architecture}: intermediate epoch {epoch}, validation loss={validation_loss:.4f}",
    )
    print(f"rendered epoch {epoch} to {args.output}")


if __name__ == "__main__":
    main()
