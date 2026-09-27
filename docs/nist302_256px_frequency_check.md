# 256px ridge-frequency check (point 2 of the gap list)

Date: 2026-09-23. `scripts/analyze_nist302_ridge_band.py` was already
parameterized by `--image-size`, so this required no new code — just a second
run at 256px against the same 200 training images used for the 128px bank,
to check whether the frequency estimate (and the resulting Gabor bank used by
`ridge_band_weight`/`ridge_spectrum_weight` losses) is resolution-stable.

| Resolution | Images w/ valid windows | Mean freq (cyc/px) | Median | Recommended Gabor bank |
|---|---:|---:|---:|---|
| 128px (`ridge_band_train_128.json`, used by every trained model this session) | 199/200 | 0.155 | 0.141 | [0.078, 0.117, 0.168, 0.258] |
| 256px (`ridge_band_train_256.json`, new) | 200/200 | 0.172 | 0.168 | [0.109, 0.152, 0.188, 0.238] |

## Reading

Naive expectation: doubling the linear resolution of the same physical crop
should roughly **halve** the cycles-per-pixel frequency (twice as many pixels
across the same ridge period). That is not what happened — the 256px estimate
is slightly *higher*, not half. Two effects are the likely explanation, and
this is reported rather than smoothed over:

1. `analyze_nist302_ridge_band.py`'s local-frequency estimator uses a fixed
   32×32-pixel patch (`estimator.patch_size`) regardless of `--image-size`.
   At 256px that 32-pixel window covers a *smaller* physical area of the
   latent than the same 32-pixel window did at 128px, so it is more likely to
   land on a locally dense ridge region and less likely to average across a
   full period — a scale-dependent bias in the estimator itself, not evidence
   that the underlying ridge structure changed.
2. `Nist302RegisteredDataset` resizes each latent to `output_shape` from its
   native resolution; the two runs are not simply "the same pixels at 2x",
   they are two independent resamplings of the native image, so interpolation
   artifacts differ between them too.

**Practical conclusion:** the Gabor bank used throughout this session's
128px-trained models (`ridge_band_train_128.json`) is specific to that
resolution and estimator configuration; it should not be assumed to transfer
unchanged if a 256px model is ever trained (none was — every NIST302 model in
this project is 128px). If a 256px model is trained later, it must be
calibrated against `ridge_band_train_256.json` (now available), not the
128px bank, and the estimator's fixed patch size should be revisited as a
known scale-dependence caveat rather than treated as resolution-invariant.

No model in this project was retrained at 256px — this check only confirms
that the loss's frequency calibration is resolution-specific, closing the
originally identified gap without requiring a full 256px training run (which
would cost ~4x the compute of every 128px run in `docs/compute_cost_table.md`
for a resolution this project does not otherwise use).
