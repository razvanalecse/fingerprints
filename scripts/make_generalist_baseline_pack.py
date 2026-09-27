#!/usr/bin/env python3
"""Build a hand-off pack for the general-purpose generative model baseline.

The research question is whether a generalist image model can produce
fingerprint completions that look convincing without recovering the ridge
structure that was actually there. Answering it requires exact ground truth,
so this pack is built from SOCOFing (Track A), never from a real forensic
latent -- on SD302 there is no pixel-exact target to score against, which is
the whole point of this project's critical finding.

For each case the pack contains:

  ground_truth.png     the original print. DO NOT give this to the model.
  mask.png             white = observed, black = hidden.
  masked_input.png     what the model sees, at native 128x128.
  masked_input_512.png the same input upscaled, because generalist image
                       models behave badly on 128x128 thumbnails. Scoring
                       resamples whatever comes back down to 128x128.

Observed pixels are re-injected at scoring time, so the model is neither
credited nor penalised for altering the region it was given -- the same rule
the project's own protocol has applied since the first commit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.preprocessing.masks import MaskFamily, MaskGenerator, MaskSpec

# Five conditions spanning "how much" and "where", applied to every print so
# that difficulty can be attributed to geometry rather than to the image.
CONDITIONS = (
    ("rect20", MaskFamily.RANDOM_RECTANGLES, 0.20, None),
    ("rect40", MaskFamily.RANDOM_RECTANGLES, 0.40, None),
    ("central60", MaskFamily.CENTRAL_MISSING, 0.60, None),
    ("peripheral40", MaskFamily.PERIPHERAL_ONLY, 0.40, None),
    ("fragments30", MaskFamily.DISCONNECTED_FRAGMENTS, 0.30, 3),
)

MISSING_FILL = 0.72  # flat mid-grey: unambiguous to a human and to a model

PROMPT = (
    "This is a grayscale fingerprint image. The flat grey area is missing. "
    "Complete the missing region so that the ridge lines continue naturally "
    "from the visible ridges around it. Preserve the visible pixels exactly "
    "and do not change them. Output a grayscale image of the same size, with "
    "no text, borders, or annotations."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prints", type=int, default=5)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--upscale", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def save_gray(array: np.ndarray, path: Path, size: int | None = None) -> None:
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    if size is not None:
        image = image.resize((size, size), Image.NEAREST)
    image.save(path)


def main() -> None:
    args = parse_args()
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split=args.split,
        output_shape=(128, 128),
        families=[MaskFamily.RANDOM_RECTANGLES],
        observed_fractions=[0.5],
        base_seed=args.seed,
    )
    dataset.set_epoch(0)

    # One image per distinct subject, so the five prints are five fingers.
    chosen: list[int] = []
    seen: set[str] = set()
    for index in range(len(dataset)):
        subject = dataset.rows[index]["subject_id"]
        if subject in seen:
            continue
        seen.add(subject)
        chosen.append(index)
        if len(chosen) == args.prints:
            break

    generator = MaskGenerator((128, 128))
    root = args.output
    (root / "cases").mkdir(parents=True, exist_ok=True)
    records = []

    for print_number, index in enumerate(chosen, start=1):
        item = dataset[index]
        truth = item["target"][0].numpy()
        sample_id = dataset.rows[index]["sample_id"]
        for condition_name, family, fraction, fragments in CONDITIONS:
            spec = MaskSpec(
                family=family,
                observed_fraction=fraction,
                seed=args.seed + print_number,
                num_fragments=fragments,
            )
            mask = generator.generate(spec).mask.astype(np.float32)
            masked = np.where(mask > 0.5, truth, MISSING_FILL)

            case_id = f"case{print_number:02d}_{condition_name}"
            case_dir = root / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            save_gray(truth, case_dir / "ground_truth.png")
            save_gray(mask, case_dir / "mask.png")
            save_gray(masked, case_dir / "masked_input.png")
            save_gray(masked, case_dir / f"masked_input_{args.upscale}.png", size=args.upscale)

            records.append({
                "case_id": case_id,
                "print_number": print_number,
                "sample_id": sample_id,
                "mask_family": family.value,
                "observed_fraction": fraction,
                "num_fragments": fragments if fragments is not None else "",
                "observed_pixels": int(mask.sum()),
                "missing_pixels": int((1 - mask).sum()),
            })

    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    (root / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")
    (root / "pack_info.json").write_text(json.dumps({
        "source_dataset": "SOCOFing",
        "split": args.split,
        "reason_for_socofing": (
            "Exact pixel ground truth. On NIST SD302 the only dense target is a "
            "different impression of the same finger, so a completion cannot be "
            "scored as recovery there -- see the project's critical finding."
        ),
        "cases": len(records),
        "prints": args.prints,
        "conditions": [c[0] for c in CONDITIONS],
        "missing_fill_value": MISSING_FILL,
        "scoring_rule": "observed pixels are re-injected before any metric is computed",
        "prompt": PROMPT,
    }, indent=2) + "\n", encoding="utf-8")

    readme = f"""GENERAL-PURPOSE GENERATIVE MODEL BASELINE
=========================================
{len(records)} cases: {args.prints} fingerprints x {len(CONDITIONS)} mask geometries.
Source: SOCOFing {args.split} split, 128x128, exact ground truth.

WHAT TO DO
----------
For each folder in cases/:

  1. Give the model  masked_input_{args.upscale}.png  (or masked_input.png).
     Do NOT give it ground_truth.png.
  2. Use the prompt in prompt.txt, unchanged, every single time.
  3. Ask for {3} to {5} separate completions of the SAME input. The
     disagreement between them is itself a result: if the visible half is
     identical and the hidden half comes back different each time, that is
     a direct demonstration that plausibility is not recovery.
  4. Save what comes back into the same case folder as:
         completion_1.png, completion_2.png, completion_3.png, ...
     Any size and any format PIL can read is fine; scoring resizes to 128x128.

THEN RUN
--------
  scripts/score_generalist_baseline.py --pack <this folder>

It re-injects the observed pixels, scores only the hidden region, and reports
MAE, SSIM, orientation error, ridge-frequency error, and the structural
deltas (coherence, curvature, minutiae density) that separate "looks right"
from "is right". It also reports the between-completion variability.

WHY SOCOFing AND NOT A REAL LATENT
----------------------------------
Because a completion can only be called recovery where an exact target
exists. On real forensic latents the only dense target available is a
different impression of the same finger, which is precisely the trap this
project measured. Do not present a completion of a real partial latent as
recovery of that person's ridges.
"""
    (root / "README.txt").write_text(readme, encoding="utf-8")
    print(f"wrote {len(records)} cases to {root}")


if __name__ == "__main__":
    main()
