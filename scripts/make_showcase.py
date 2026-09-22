#!/usr/bin/env python3
"""Clean side-by-side figure showing the progression of reconstruction quality."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.ridge_cleanup import RidgeCleanup
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask,
    estimate_orientation_field,
    orientation_error,
)

CASES = (
    ("random_rectangles", 0.8), ("central_missing", 0.6), ("irregular", 0.5),
    ("peripheral_only", 0.4), ("disconnected_fragments", 0.3), ("random_rectangles", 0.2),
)


def load_model(path: str):
    if path.startswith("cleanup:"):
        model, epoch = load_model(path[len("cleanup:"):])
        return RidgeCleanup(model).eval(), epoch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_reconstruction_model(checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), int(checkpoint.get("epoch", -1)) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="LABEL=checkpoint")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/gated_mps.yaml").read_text())
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest, image_root=args.image_root, split="validation", output_shape=(128, 128),
        families=[MaskFamily(v) for v in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"], base_seed=1729,
    )
    dataset.set_epoch(0)
    indices = []
    for family, fraction in CASES:
        found = [i for i in range(len(dataset)) if dataset.conditions[i % len(dataset.conditions)][0].value == family
                 and abs(dataset.conditions[i % len(dataset.conditions)][1] - fraction) < 1e-6]
        indices.append(found[args.skip])
    models = []
    for item in args.model:
        label, path = item.split("=", 1)
        model, epoch = load_model(path)
        models.append((f"{label}" + (f"\n(epoca {epoch})" if epoch > 0 else ""), model))
    columns = 2 + len(models)
    figure, axes = plt.subplots(len(indices), columns, figsize=(2.5 * columns, 2.85 * len(indices)))
    for row, index in enumerate(indices):
        item = dataset[index]
        observed, mask, target = item["observed"][None], item["mask"][None], item["target"][0].numpy()
        roi = estimate_foreground_mask(target)
        missing_roi = (~mask[0, 0].numpy().astype(bool)) & roi
        reference = estimate_orientation_field(target)
        shown = torch.where(mask[0].bool(), observed[0], torch.full_like(observed[0], 0.72))[0].numpy()
        panels = [("Tinta (reala)", target, None), ("Ce se vede (gri = lipsa)", shown, None)]
        with torch.no_grad():
            for label, model in models:
                out = model.reconstruct(observed, mask)[0, 0].numpy()
                mae = float(np.abs(out - target)[missing_roi].mean())
                try:
                    ori = orientation_error(reference, estimate_orientation_field(out, use_foreground_mask=False), region_mask=missing_roi)
                except ValueError:
                    ori = float("nan")
                panels.append((label, out, f"MAE {mae:.3f}  ORI {ori:.3f}"))
        for col, (title, image, note) in enumerate(panels):
            axis = axes[row, col]
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(title, fontsize=9, fontweight="bold")
            if note:
                axis.set_xlabel(note, fontsize=8)
            if col == 0:
                axis.set_ylabel(f"{item['mask_family'].replace('_', ' ')}\n{float(item['observed_fraction']):.0%} vizibil", fontsize=8)
    figure.suptitle("Evolutia reconstructiei (MAE si ORI mai mici = mai bine)", fontsize=11, y=0.998)
    figure.tight_layout()
    figure.savefig(args.output, dpi=140, facecolor="white")
    print("saved", args.output)


if __name__ == "__main__":
    main()
