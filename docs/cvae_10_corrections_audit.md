# Audit of the ten CVAE corrections

Date: 2026-09-23

Scope: NIST SD302 `registered_approximate` train/validation pairs. The test split
remained sealed throughout this audit. The registered exemplar is an approximate
structural reference, not pixel-aligned ground truth.

## Implemented corrections

1. **Ridge-frequency reporting.** Ridge-frequency results are expanded into
   numeric columns (`mae_cpx`, relative MAE, period MAE, valid windows), rather
   than serializing a dictionary into one CSV cell.
2. **Visible-support conditioning.** The frozen predictor trained for the exact
   target `quality >= 1` is used during both training and sampling. Its target
   semantics are checked when the checkpoint is loaded.
3. **Support-aware reconstruction loss.** The registered reconstruction region
   is weighted by support probability and a separate background-darkness penalty
   discourages hallucinated rectangular foreground.
4. **Fingerprint-structural losses.** Orientation, gradient/ridge-energy,
   ridge-band and ridge-spectrum terms are available and enabled in the fair
   comparison configs. Ridge-band frequencies come only from the train split.
5. **Separated estimands.** Evaluation reports a single stochastic sample,
   the K-sample predictive mean and best-of-K independently. Best-of-K is never
   labeled as single-sample performance.
6. **Calibration.** Empirical central-interval coverage, absolute calibration
   error, interval width and uncertainty-error Spearman correlation are reported.
7. **K sensitivity.** Full validation uses fixed seeds and K = 5, 10, 20, 50.
8. **Fair architecture comparison.** Global and spatial arms are trained from
   scratch with identical channels/loss/data. Both use exactly 256 latent scalars:
   global `z in R^256`; spatial `z in R^(1x16x16)`.
9. **Spatial-locality test.** A latent intervention test verifies that perturbing
   opposite spatial latent cells moves the response centroid in the corresponding
   image direction. A mere non-zero variance check is retained only as a weaker
   auxiliary test.
10. **Paired subject-level inference.** Model comparisons aggregate image metrics
    by subject, then run paired tests, effect sizes, confidence intervals and Holm
    correction. Diversity and interval width remain descriptive because neither
    has a universally monotone definition of “better”.

## Engineering verification

- Test suite: 211 passed, 2 intentionally skipped.
- Smoke runs: global and spatial CVAE completed on Apple MPS.
- Pilot training: 3 epochs, 64 train batches/epoch, validation-only selection.
- Full validation evaluation: 287 images, 29 subjects, fixed sampling seed;
  K = 5, 10, 20, 50.
- Test split loaded: false.

## Pilot findings on full validation

These are model-selection diagnostics, not final test results.

- At K=50, the spatial CVAE improves predictive-mean MAE by about 0.00655 and
  predictive-mean SSIM by about 0.02283 relative to the fair global CVAE.
- Spatial best-of-50 MAE improves by about 0.01210.
- Spatial ridge-frequency relative MAE improves by about 0.18834.
- The global CVAE is better for a single stochastic sample by about 0.00760 MAE.
- Spatial diversity is materially larger, but uncertainty-error correlation is
  still negative and interval coverage remains far below nominal. Therefore the
  current CVAE is not uncertainty-calibrated and H8 is not supported by this pilot.
- Holm-adjusted subject-level Wilcoxon tests favor spatial for predictive mean,
  best-of-50, SSIM, orientation, ridge frequency and coverage error. They favor
  global for single-sample MAE.

## Decision gate

The spatial arm is the stronger probabilistic candidate, while the global arm is
the stronger single-sample control. Both should be retained for the full fair
experiment. Uncertainty calibration must be treated as an unresolved research
problem; it must not be claimed from visually plausible samples alone.

## Full fair validation experiment

Both arms were subsequently trained on all 1,213 registered-approximate training
pairs with validation-only checkpoint selection. The global arm stopped early at
epoch 14 and selected epoch 9; the spatial arm completed 20 epochs and selected
epoch 16. Both retained the identical 256-scalar latent budget.

| Validation metric | Global CVAE | Spatial CVAE |
|---|---:|---:|
| Single-sample MAE | 0.16052 | 0.15917 |
| Predictive-mean MAE, K=50 | 0.15626 | 0.13934 |
| Predictive-mean SSIM, K=50 | 0.27619 | 0.37548 |
| Best-of-50 MAE | 0.13633 | 0.13077 |
| Orientation error | 0.32184 | 0.29477 |
| Ridge-frequency relative MAE | 0.72627 | 0.63194 |
| Pairwise diversity MAE, K=50 | 0.03827 | 0.08787 |
| Empirical coverage of nominal 90% interval, K=50 | 0.22164 | 0.56092 |
| Uncertainty-error Spearman rho, K=50 | -0.13173 | -0.15598 |
| Background darkness | 0.02081 | 0.01908 |

Subject-level paired inference (29 validation subjects; 28 for ridge frequency)
favored the spatial model after Holm correction for predictive-mean MAE, SSIM,
best-of-50 MAE, orientation error, ridge-frequency error, background darkness and
coverage error. The single-sample MAE difference was not significant
(`p_Holm = 0.417`).

The spatial model is therefore selected for subsequent probabilistic work, while
the global model remains a fair ablation/control. This is a validation-only model
selection decision. Neither model supports a claim of calibrated local
uncertainty: interval coverage is still below nominal and pixel-wise uncertainty
is negatively associated with absolute error. The preview also shows a serrated
support-boundary artifact that should be addressed before final figures.

## Boundary and uncertainty follow-up

The serrated boundary was traced primarily to nearest-neighbor decoder
upsampling. A validation-only inference ablation loaded the identical epoch-16
spatial checkpoint and changed only interpolation to bilinear. Relative to
nearest-neighbor decoding, bilinear decoding produced:

- mean-K50 MAE: 0.13934 -> 0.13845;
- mean-K50 SSIM: 0.37548 -> 0.40016;
- best-of-50 MAE: 0.13077 -> 0.12190;
- ridge-frequency relative MAE: 0.63194 -> 0.51738;
- orientation error: 0.29477 -> 0.29437.

The paired subject-level comparison favored bilinear decoding after Holm
correction for single-sample MAE, predictive-mean MAE, SSIM, best-of-50 MAE and
ridge-frequency error. Orientation was statistically unchanged. A thresholded
Gaussian-feathered support reduced background darkness but harmed fidelity and
was therefore rejected. Bilinear + soft support is the selected validation
configuration.

Uncertainty intervals were additionally calibrated using five-fold
subject-cross-fitted standardized residual scaling, with equal pixel caps per
calibration subject. Nominal 90% marginal coverage increased from 0.510 to
0.922, but the mean bounded interval width increased from 0.218 to 0.851 on the
entire [0,1] intensity range. At 95% coverage the bounded width was 0.921. The
required final validation scale factors were approximately 1.72, 6.00, 19.93
and 50.27 for 50%, 80%, 90% and 95% intervals. Thus post-hoc scaling can repair
marginal coverage only by producing nearly uninformative intervals; it does not
repair negative spatial uncertainty-error correlation. This calibration is
retained as a diagnostic, not selected as a practically useful uncertainty
method.
