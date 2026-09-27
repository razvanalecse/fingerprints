#!/usr/bin/env python3
"""Why is zero-shot better? Test whether fine-tuning copies the impression.

The critical finding records *that* fine-tuning against `X_pseudo` degrades
reconstruction on real pixels. It does not explain *why*. The standing
hypothesis is that the model learns the texture of that particular impression
rather than the finger's ridge structure. This tests it directly.

The discriminating region is not where the latent and the registered exemplar
agree -- there, following either one looks the same. It is where they
**disagree**. In those pixels a model that learned the finger should sit closer
to the latent, and a model that learned the target should sit closer to
`X_pseudo`.

For each held-out `quality == 1` pixel:
    d_latent = axial distance between the model's orientation and the latent
    d_pseudo = axial distance between the model's orientation and X_pseudo
and a pixel "follows the pseudo-target" when d_pseudo < d_latent.

Zero-shot never saw `X_pseudo` during training, so it provides the
what-does-chance-look-like baseline that makes the fine-tuned rate readable.
"""
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.statistical_tests import paired_comparison
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything

DISAGREE = 1.0   # axial distance above which the two impressions genuinely conflict
AGREE = 0.2      # and below which they agree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--zeroshot-checkpoint", "--finetuned-checkpoint", "--manifest",
                 "--latent-root", "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_reconstruction_model(ck["config"]["model"], channels=tuple(ck["channels_used"])).to(device)
    m.load_state_dict(ck["model_state"]); m.eval(); return m


def axial(a, b):
    return 1.0 - np.cos(2.0 * (a - b))


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    conf = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    loader = DataLoader(Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(conf["ordered_weights"])),
        batch_size=1, shuffle=False, num_workers=0)
    models = {"zeroshot": load(args.zeroshot_checkpoint, device),
              "finetuned": load(args.finetuned_checkpoint, device)}

    rows = []
    for batch in loader:
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        heldout = batch["quality"][0, 0].numpy() == 1
        if heldout.sum() < 400:
            continue
        latent = batch["latent_image"][0, 0].numpy()
        pseudo = batch["target"][0, 0].numpy()
        lat = estimate_orientation_field(latent, use_foreground_mask=False).theta
        pse = estimate_orientation_field(pseudo, use_foreground_mask=False).theta
        conflict = axial(lat, pse)
        disagree = heldout & (conflict > DISAGREE)
        agree = heldout & (conflict < AGREE)
        if disagree.sum() < 50 or agree.sum() < 50:
            continue
        row = {"sample_id": str(batch["sample_id"][0]), "subject_id": str(batch["subject_id"][0]),
               "disagree_pixels": int(disagree.sum()), "agree_pixels": int(agree.sum())}
        for name, model in models.items():
            out = model.reconstruct(observed, mask)[0, 0].cpu().numpy()
            mod = estimate_orientation_field(out, use_foreground_mask=False).theta
            d_lat, d_pse = axial(mod, lat), axial(mod, pse)
            row[f"{name}_follows_pseudo_in_conflict"] = float((d_pse < d_lat)[disagree].mean())
            row[f"{name}_error_in_conflict"] = float(d_lat[disagree].mean())
            row[f"{name}_error_in_agreement"] = float(d_lat[agree].mean())
        rows.append(row)

    subjects = defaultdict(list)
    for r in rows:
        subjects[r["subject_id"]].append(r)
    def per_subject(key):
        return [float(np.mean([r[key] for r in v])) for v in subjects.values()]

    tests = {}
    for key, higher in (("follows_pseudo_in_conflict", False),
                        ("error_in_conflict", False), ("error_in_agreement", False)):
        z, f = per_subject(f"zeroshot_{key}"), per_subject(f"finetuned_{key}")
        result = paired_comparison(z, f, higher_is_better=higher)
        tests[key] = {"zeroshot": float(np.mean(z)), "finetuned": float(np.mean(f)),
                      "cohen_dz": result.cohen_dz, "wilcoxon_p": result.wilcoxon_pvalue}

    report = {"images": len(rows), "subjects": len(subjects),
              "conflict_threshold": DISAGREE, "agreement_threshold": AGREE,
              "note": ("follows_pseudo_in_conflict is the fraction of conflicting pixels where "
                       "the model's orientation is closer to X_pseudo than to the real latent"),
              "tests": tests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output.with_suffix(".per-image.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("%d images, %d subjects\n" % (len(rows), len(subjects)))
    print("%-28s %10s %10s %9s %10s" % ("quantity", "zero-shot", "fine-tuned", "cohen_dz", "p"))
    for key, v in tests.items():
        print("%-28s %10.4f %10.4f %9.2f %10.2e" % (
            key, v["zeroshot"], v["finetuned"], v["cohen_dz"], v["wilcoxon_p"]))


if __name__ == "__main__":
    main()
