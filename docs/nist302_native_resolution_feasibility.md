# Native-resolution patches: feasibility analysis (point 1 of the external-review plan)

Date: 2026-09-23. This is a **quantified feasibility analysis, not an
implementation** — the concrete numbers below are new (computed this
session from the real manifest), but no patch-based training pipeline was
built. This document exists so a future continuation of this project has a
precise starting point instead of re-deriving these numbers from scratch.

## The core problem, quantified

Every NIST302 model in this project resizes the **whole** latent image down
to 128×128 before doing anything else. The native images are much larger:

| Quantity | Mean | Median | p90 | Range |
|---|---:|---:|---:|---|
| Native width (px) | 936 | — | — | 283-2109 |
| Native height (px) | 1030 | — | — | 325-2309 |
| Native PPI | 1161 | — | — | 1042-1433 |
| Linear downsampling factor (width) | **7.31x** | 7.06x | 9.98x | — |

A 7.3x average linear downsampling is a ~53x reduction in pixel count. The
consequence for ridge structure specifically: this project's own ridge-
frequency estimator measures a mean period of **6.45 pixels** at 128px
(`outputs/nist302_registration/ridge_band_train_128.json`). Back-calculating
through the mean downsampling factor gives an estimated **native ridge
period of ≈47 pixels (≈1.03mm)** — roughly double the textbook fingerprint
ridge period (~0.45-0.5mm, i.e. ≈23px at this dataset's mean PPI).

**This discrepancy should be stated honestly, not smoothed over:** it may
reflect (a) a genuine property of degraded/partial forensic latents (broken
or faint ridge lines can make a local-frequency estimator lock onto a
coarser "beat" pattern rather than the true ridge period), or (b) the same
scale-dependent estimator artifact already documented in
`docs/nist302_256px_frequency_check.md` (frequency did not scale the way a
clean resize would predict between 128px and 256px either). Either reading
supports the same conclusion below, so this uncertainty does not block the
recommendation, but a future continuation should resolve it (e.g. by
measuring frequency directly on native-resolution crops, not by back-
calculating through the resize factor as done here).

## What this means for the 128px representation

Whether the true native period is ≈23px (textbook) or ≈47px (back-
calculated), **at 128×128 the ridge period is compressed to just 6.45
pixels on average** — only 2-3x oversampled relative to the Nyquist minimum
needed to represent an oscillatory signal at all, and far below the "8-15
pixels per period" an external review specifically flagged as the target
for faithfully representing ridge phase, curvature, and bifurcations. This
is a strong, quantified candidate explanation for a pattern already
documented elsewhere in this project: section 15 of the ablation master
table found the DDPM's reconstructions are measurably more orientation-
uniform, less curved, and have roughly half the minutiae density of real
ridges in the same region — exactly the kind of detail loss expected when a
model is asked to synthesize fine ridge topology from a representation that
barely resolves the ridge period to begin with.

## What native-resolution patches would need (not built this session)

Following the external review's proposal: 128 or 256px patches with overlap,
extracted directly from native-resolution images (no whole-image resize),
would give **≈18-40 pixels per ridge period** even under the more
conservative (larger) period estimate — comfortably inside the target range,
without needing to resolve the textbook-vs-back-calculated discrepancy
above. Concretely, this would require:

1. **A patch-extraction dataset layer**, replacing `Nist302RegisteredDataset`'s
   whole-image resize with: (a) native-resolution loading, (b) a patch
   sampling strategy (e.g. grid with overlap, or evaluation-ROI-weighted
   sampling so patches aren't dominated by background), (c) the affine (or
   TPS, per section 16) registration applied at native resolution before
   cropping, since the current pipeline warps *after* resizing.
2. **Re-deriving every resolution-dependent calibration at patch scale**:
   the Gabor ridge-frequency bank (already shown to not transfer cleanly
   between 128px and 256px), the geometric-confidence distance bands, the
   support predictor, and the structure predictor (section on the two-stage
   model) would all need retraining or recalibration on patches, not reused
   as-is.
3. **A stitching/aggregation step for evaluation**, since this project's
   held-out `quality==1` metric requires assembling patch-level predictions
   back into whole-image coordinates to compare against the existing
   evaluation-ROI and distance-band infrastructure — nontrivial with
   overlapping patches (needs a blending rule) and registration-warped
   coordinates.
4. **New training runs for every model family** (deterministic, CVAE, DDPM,
   residual DDPM, latent diffusion) at patch scale, each needing its own
   smoke test, pilot, and full run, following this project's established
   (and, this session, repeatedly necessary) practice of testing new
   pipelines on small subsets before committing to full training.

Realistically, item 1 alone is comparable in scope to this session's latent-
diffusion buildout (~3-5 hours); items 2-4 multiply that across every
resolution-dependent component and every model family. This is a multi-day
undertaking, not a same-day extension — which is why it was not attempted
this session, unlike the other three items from the same external-review
plan (ridge-topology diagnostics, non-rigid registration, the two-stage
structure-guided model), all of which reused enough existing infrastructure
to be feasible in hours rather than days.

## Recommendation

If this project continues, native-resolution patches are the single
highest-leverage remaining item — the quantified ~7x downsampling and its
plausible link to the section-15 "painted stripe" artifact make a stronger,
more specific case for it than a generic "resolution might matter" concern.
It should be scoped as its own phase, starting with item 1 above (the patch
dataset layer) and a single deterministic-model pilot, before committing to
recalibrating every downstream component.
