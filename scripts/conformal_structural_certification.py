#!/usr/bin/env python3
"""Distribution-free certified bounds on reconstructed ridge structure.

The 2025 conformal-prediction literature for imaging inverse problems states
its own open problem plainly: pixel-wise uncertainty intervals are of unclear
value, because what matters when recovering an image is many-pixel structure --
hallucinations, invented anatomy -- about which single-pixel statistics say
little. This project is unusually well placed to answer that, having shown with
Holm p ~ 1e-8 that pixel metrics (MAE, SSIM) improve while ridge *structure*
degrades.

So the quantity certified here is structural: the axial orientation error of
the reconstructed ridge field.

Method (split conformal):
  1. fit a reliability regressor on training subjects, using only signals that
     need no target (section 26's feature set);
  2. on separate calibration subjects, form nonconformity scores
     r = true_error - predicted_error and take the (1-alpha) quantile q;
  3. on unseen test subjects, the interval [0, predicted + q] is claimed to
     contain the true error with probability at least 1-alpha.

Everything is split by **subject**, never by pixel or image, so a subject never
appears in two roles.

An honest limit, and the reason coverage is reported per subject rather than
only pooled: conformal guarantees assume exchangeability, and pixels inside one
fingerprint are strongly correlated, so the effective sample size is closer to
the number of subjects than to the number of pixels. Pooled pixel coverage will
look reassuring even if subject-level coverage is erratic; both are reported.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.metrics.orientation_geometry import complete_harmonic
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field
from fingerprint_reconstruction.reproducibility import seed_everything

EROSION, MIN_REGION = 3, 400
ALPHAS = (0.20, 0.10, 0.05)
TOLERANCES = (0.13, 0.29, 0.50, 0.75, 1.00)  # 15, 22.5, 30, 37.5, 45 degrees


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("--config", "--checkpoint", "--manifest", "--latent-root",
                 "--annotation-root", "--sd302a-root", "--sd302b-root",
                 "--sd302d-root", "--output"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--max-images", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--patch", type=int, default=16,
                   help="certify PATCHxPATCH regions; 1 reverts to per-pixel")
    p.add_argument("--seed", type=int, default=9173)
    return p.parse_args()


def ridge_fit(X, y, alpha=1.0):
    centre, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - centre) / scale
    weights = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return centre, scale, weights, y.mean()


def ridge_predict(model, X):
    centre, scale, weights, offset = model
    return ((X - centre) / scale) @ weights + offset


@torch.no_grad()
def extract(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    conf = json.loads(Path(config["data"]["geometric_confidence_report"]).read_text())
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    dataset = Nist302RegisteredDataset(
        manifest_path=args.manifest, latent_root=args.latent_root,
        annotation_root=args.annotation_root, exemplar_roots=roots, split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        geometric_confidence_weights=tuple(conf["ordered_weights"]))
    if args.max_images:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_reconstruction_model(ck["config"]["model"],
                                       channels=tuple(ck["channels_used"])).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()

    feats, errs, groups, patch_ids = [], [], [], []
    offset = [0]
    height, width = tuple(config["data"]["image_size"])
    yy, xx = np.mgrid[:height, :width]
    patch_index = ((yy // args.patch) * (width // args.patch + 1) + (xx // args.patch))
    for batch in loader:
        mask = batch["mask"][0, 0].numpy(); obs = mask > 0.5
        heldout = batch["quality"][0, 0].numpy() == 1
        if heldout.sum() < MIN_REGION:
            continue
        trusted = ndimage.binary_erosion(obs, iterations=EROSION, border_value=0)
        if trusted.sum() < MIN_REGION:
            continue
        latent = batch["latent_image"][0, 0].numpy()
        recon = model.reconstruct(batch["observed"].to(device),
                                  batch["mask"].to(device))[0, 0].cpu().numpy()
        tf = estimate_orientation_field(latent, use_foreground_mask=False)
        of = estimate_orientation_field(recon, use_foreground_mask=False)
        ob = estimate_orientation_field(np.where(obs, latent, 0.72).astype(np.float32),
                                        use_foreground_mask=False)
        cos_f = complete_harmonic(np.cos(2 * ob.theta), trusted)
        sin_f = complete_harmonic(np.sin(2 * ob.theta), trusted)
        harmonic = np.mod(0.5 * np.arctan2(sin_f, cos_f), np.pi)
        feats.append(np.stack([
            (1 - np.cos(2 * (of.theta - harmonic)))[heldout],
            np.hypot(cos_f, sin_f)[heldout],
            ndimage.distance_transform_edt(~trusted)[heldout],
            complete_harmonic(ob.coherence, trusted)[heldout],
            of.coherence[heldout],
            np.abs(np.gradient(of.theta)[0])[heldout],
        ], axis=1))
        errs.append((1 - np.cos(2 * (tf.theta - of.theta)))[heldout])
        groups.append(np.full(int(heldout.sum()), str(batch["subject_id"][0])))
        patch_ids.append(patch_index[heldout] + offset[0])
        offset[0] += patch_index.max() + 1
    return (np.nan_to_num(np.concatenate(feats)), np.concatenate(errs),
            np.concatenate(groups), np.concatenate(patch_ids))


def main():
    args = parse_args()
    seed_everything(args.seed)
    X, y, g, pid = extract(args)
    if args.patch > 1:
        # Certify regions, not pixels: averaging inside a patch collapses the
        # per-pixel noise that made the pixel-level interval useless, and it is
        # also the quantity the conformal imaging literature argues actually
        # matters (many-pixel structure, not single pixels).
        unique, inverse = np.unique(pid, return_inverse=True)
        counts = np.bincount(inverse)
        keep = counts >= max(8, args.patch * args.patch // 4)
        agg_X = np.stack([np.bincount(inverse, w, len(unique)) / counts for w in X.T], axis=1)
        agg_y = np.bincount(inverse, y, len(unique)) / counts
        first = np.zeros(len(unique), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(inverse))[::-1]
        agg_g = g[first]
        X, y, g = agg_X[keep], agg_y[keep], agg_g[keep]
        print(f"certifying {len(y)} regions of up to {args.patch}x{args.patch} px")
    subjects = np.array(sorted(set(g.tolist())))
    rng = np.random.default_rng(args.seed); rng.shuffle(subjects)
    n = len(subjects)
    fit_s = set(subjects[: n // 3])
    cal_s = set(subjects[n // 3: 2 * n // 3])
    test_s = set(subjects[2 * n // 3:])
    fit = np.isin(g, list(fit_s)); cal = np.isin(g, list(cal_s)); test = np.isin(g, list(test_s))

    model = ridge_fit(X[fit], y[fit])
    cal_pred, test_pred = ridge_predict(model, X[cal]), ridge_predict(model, X[test])
    scores = y[cal] - cal_pred

    results = {}
    for alpha in ALPHAS:
        q = float(np.quantile(scores, 1 - alpha, method="higher"))
        upper = test_pred + q
        covered = y[test] <= upper
        per_subject = []
        for s in sorted(test_s):
            m = g[test] == s
            if m.any():
                per_subject.append(float(covered[m].mean()))
        entry = {
            "quantile": q,
            "target_coverage": 1 - alpha,
            "pooled_pixel_coverage": float(covered.mean()),
            "subject_coverage_mean": float(np.mean(per_subject)),
            "subject_coverage_min": float(np.min(per_subject)),
            "subject_coverage_max": float(np.max(per_subject)),
            "mean_certified_bound": float(upper.mean()),
            "certified_region_fraction": {},
        }
        for tol in TOLERANCES:
            sel = upper <= tol
            entry["certified_region_fraction"][str(tol)] = {
                "fraction_of_pixels_certified": float(sel.mean()),
                "true_error_inside": float(y[test][sel].mean()) if sel.any() else float("nan"),
                "true_error_outside": float(y[test][~sel].mean()) if (~sel).any() else float("nan"),
            }
        results[f"alpha={alpha}"] = entry

    report = {
        "pixels": int(y.size), "subjects": int(n),
        "split_by": "subject; fit/calibrate/test disjoint",
        "fit_subjects": len(fit_s), "calibration_subjects": len(cal_s), "test_subjects": len(test_s),
        "certified_quantity": "axial orientation error of the reconstructed ridge field",
        "mean_true_error_on_test": float(y[test].mean()),
        "results": results,
        "caveat": ("Conformal coverage assumes exchangeability. Pixels inside one print are "
                   "strongly correlated, so pooled pixel coverage overstates the effective "
                   "sample size; subject-level spread is the honest read."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("mean true orientation error on test subjects: %.4f" % report["mean_true_error_on_test"])
    print()
    print("%-9s %8s %10s %10s %10s %10s" % ("target","quantile","pooled","subj mean","subj min","subj max"))
    for key, e in results.items():
        print("%-9s %8.4f %10.3f %10.3f %10.3f %10.3f" % (
            key, e["quantile"], e["pooled_pixel_coverage"], e["subject_coverage_mean"],
            e["subject_coverage_min"], e["subject_coverage_max"]))
    print()
    e = results["alpha=0.1"]
    print("at 90% target, certified regions (tolerance shown as the equivalent angle):")
    for tol, v in e["certified_region_fraction"].items():
        degrees = np.degrees(0.5 * np.arccos(np.clip(1 - float(tol), -1, 1)))
        print("  bound<=%-5s (%4.1f deg)  %5.1f%% certified | true err inside %s vs outside %.3f" % (
            tol, degrees, 100 * v["fraction_of_pixels_certified"],
            ("%.3f" % v["true_error_inside"]) if v["true_error_inside"] == v["true_error_inside"] else "  -  ",
            v["true_error_outside"]))


if __name__ == "__main__":
    main()
