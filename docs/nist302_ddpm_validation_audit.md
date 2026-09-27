# NIST SD302 conditional DDPM: validation audit

This document records validation-only decisions for the conditional pixel-space
DDPM. The NIST SD302 test split was not loaded during training, sampler selection,
or the comparisons below. Targets are examiner-assisted
`registered_approximate` exemplars and are not pixel-aligned ground truth.

## Method

The denoiser is conditioned on the observed latent image, its reliable-ridge
mask, and a frozen visible-support probability map. Noise supervision in the
missing region is weighted by the intersection of the evaluation ROI,
distance-to-correspondence geometric confidence, and predicted visible support.
Observed pixels are re-injected exactly during sampling. The decoder uses
bilinear upsampling and the final output uses a soft support constraint.

## Pilot sampler ablation

The same pilot checkpoint, the same first 64 validation images, the same 7
subjects, and K=10 samples were used for every row. These results select a
sampler; they are not final estimates of generalization.

| Metric | DDIM-10 | DDIM-20 | DDIM-50 |
|---|---:|---:|---:|
| Seconds / image | 0.18683 | 0.32769 | 0.75234 |
| Predictive-mean MAE | 0.16827 | **0.16415** | 0.17557 |
| Predictive-mean SSIM | 0.24573 | **0.26031** | 0.23919 |
| Best-of-10 MAE | 0.17002 | **0.16302** | 0.17551 |
| Orientation error | 0.28748 | 0.28669 | **0.28563** |
| Ridge-frequency relative MAE | 0.55938 | **0.55926** | 0.57103 |
| Pairwise diversity MAE | **0.11450** | 0.07903 | 0.09166 |
| Nominal-90% empirical coverage | **0.41262** | 0.26274 | 0.26203 |
| Uncertainty-error Spearman rho | 0.12214 | 0.11145 | **0.21649** |
| Background darkness | **0.06017** | 0.06050 | 0.06266 |

DDIM-20 is selected as the primary fidelity/time configuration. DDIM-50 is
retained as an uncertainty-localization ablation because it gives the largest
positive uncertainty-error correlation, despite being about 2.3 times slower
than DDIM-20 and worse on image fidelity. DDIM-10 gives more diversity and wider
intervals, but worse reconstruction fidelity.

At subject level, DDIM-20 had lower predictive-mean and best-of-K MAE and higher
SSIM than both alternatives for all seven pilot subjects. The corresponding
unadjusted paired tests were small, but no comparison survived Holm correction
across the full metric family because the pilot contains only seven subjects.
Accordingly, this is a validation engineering decision, not confirmatory
evidence for a scientific superiority claim.

## Interpretation of uncertainty

Positive uncertainty-error correlation means that pixels with greater sample
dispersion tend, on average, to have greater absolute reconstruction error. It
does not imply calibrated probabilities. Coverage remains far below its nominal
level for every uncalibrated sampler. Conversely, the CVAE's negative
correlation means it is often confident in locations where it is wrong and
variable in locations where error is smaller. This distinction between
localization and calibration must be preserved in the paper.

## Reproducibility caveat

Seeds, configuration, splits, per-image metrics, and checkpoints are saved.
PyTorch reports that `index_put_with_accumulate_mps` lacks a deterministic MPS
implementation, so Metal runs are controlled but not guaranteed bit-for-bit
identical. Final estimates should include repeated seeds or, at minimum, a
repeatability audit of the selected configuration.
