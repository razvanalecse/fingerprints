# NIST SD302 ablation master table (in progress)

**Reporting tracks:** every result below is Track B (evaluated against the
approximate registered exemplar `X_pseudo`) or Track C (evaluated against
held-out real `quality==1` pixels — "debatable ridge flow," i.e. enhancement
of a faint existing trace, not recovery of a genuinely absent region); Track
D results (predictive uncertainty) are themselves computed on top of Track C
targets. Full track definitions: `docs/FINAL_REPORT.md` section 2.5. Section
1 below is explicitly Track B *and* Track C side by side, which is the point
of that section. Every other section is Track C or Track D unless noted.

## ⚠ Critical finding: the evaluation methodology reverses the headline conclusion

Section 1 below (`evaluate_nist302_registered.py`, target = `X_pseudo`, the
registered exemplar) shows fine-tuning **improving** MAE and orientation over
zero-shot. Re-evaluating the *same two checkpoints* with the held-out
real-pixel methodology (`evaluate_nist302_latent_enhancement.py`, target =
genuine quality==1 latent pixels, never used in any loss) gives the
**opposite result on every metric**:

| Metric (held-out real pixels) | Zero-shot | Fine-tuned (calibrated, support+spectrum) | Change | Holm p |
|---|---:|---:|---:|---:|
| MAE | 0.1068 | 0.1464 | **+0.040 (worse)** | 1.9e-8 |
| Orientation error | — | — | **+0.058 (worse)** | 3.7e-7 |
| Ridge-frequency rel. MAE | — | — | **+0.202 (worse)** | 5.5e-5 |
| SSIM | — | — | **+0.222 (worse)** | 1.9e-8 |

All four differences are large and Holm-significant, all favoring zero-shot.
**Interpretation:** fine-tuning against `X_pseudo` teaches the model to match
a different impression's texture, not the true underlying intensity. Against
that same imperfect target, fine-tuning looks like an improvement (Section 1);
against real, held-out pixels never used anywhere in training or target
construction, it is a regression. This is the single most important finding
in this document for the "reconstruction vs hallucination" chapter: **it
shows measured improvement can be an artifact of the evaluation target, not a
property of the model**, and every fine-tuning result in this document that
was only checked against `X_pseudo` (Section 1) should be treated as
unconfirmed against ground truth until re-evaluated the way Sections 2-7 were.
Sections 2-7 (support, spectrum, confidence-weighting, RePaint, residual
diffusion, CVAE) were all evaluated against real held-out pixels throughout
and are not affected by this caveat — they are comparisons *among*
fine-tuned/probabilistic variants, not against the untuned zero-shot baseline.

A 6-model Friedman omnibus test on held-out MAE (zero-shot, fine-tuned,
CVAE, DDIM, RePaint, residual DDPM; 29 subjects) confirms real differences
exist somewhere among the six (χ²=116.27, df=5, p=1.9e-23) — full data in
`outputs/nist302_friedman_omnibus_mae.json`. Subject-mean MAE ranking (best to
worst): zero-shot (0.107) < RePaint (0.110) < residual DDPM (0.121) < CVAE
(0.139) < fine-tuned deterministic (0.146) < DDIM (0.155). Zero-shot's
top rank is now explained by the finding above, not a fluke.

**A third, independent confirmation (Track E, section 17): the same
regression shows up even with *exact* pixel ground truth**, on synthetically
degraded clean SD302 exemplars where the missing region's true value is
known by construction (not approximated by a different impression, and not
limited to sparse held-out real pixels). Fine-tuned is significantly worse
than zero-shot on SSIM (dz=−2.78, Holm p=1.5e-8), orientation error
(dz=−0.56, p=0.0047), and ridge-frequency error (dz=−0.71, p=0.0024); only
MAE is a statistical tie. Three methodologically unrelated ways of measuring
"did fine-tuning actually help" — `X_pseudo` (says yes), held-out real
pixels (says no), and exact synthetic ground truth (also says no) — agree on
two out of three, and the two that agree are the two with a real, uncorrupted
target. This is now the best-supported finding in the entire project.

Date: 2026-09-23. Validation-only; test split not loaded for any comparison
below. All tests are subject-level paired Wilcoxon with Holm correction across
each comparison's own metric family (29 subjects; 28 for ridge-frequency
metrics, which are undefined on some images).

## Completed ablations

### 1. Structural fine-tuning vs SOCOFing zero-shot
Reference: `gated_socofing_zeroshot` — Candidate: `gated_sd302_finetuned_no_spectrum`
(orientation + ridge-band losses, no ridge-spectrum, no support conditioning).

| Metric | Improvement | Holm p | Significant |
|---|---:|---:|---|
| MAE | +0.101 | 2.2e-8 | yes |
| Orientation error | +0.178 | 7.5e-8 | yes |
| Ridge-frequency rel. MAE | −0.116 | 3.2e-5 | yes (worse) |
| SSIM | −0.085 | 1.5e-8 | yes (worse) |

Fine-tuning buys orientation and MAE at a real cost to frequency and SSIM —
expected, since this arm has no frequency-domain loss term yet.

### 2. Visible-support conditioning vs none
Reference: `old_no_support` — Candidate: `support_constrained_FINAL`.

| Metric | Improvement | Holm p | Significant |
|---|---:|---:|---|
| Background darkness (lower hallucination) | +0.022 | 1.9e-8 | yes |
| Held-out real-pixel MAE | +0.006 | 0.028 | yes |
| Held-out orientation error | −0.021 | 5.5e-5 | yes (worse) |
| Held-out ridge-frequency rel. MAE | −0.026 | 0.164 | no |
| Held-out SSIM | +0.018 | 1.0e-7 | yes |

Support conditioning does what it was built for (stops hallucinating ridges
into scanner background) and helps MAE/SSIM on real held-out pixels, at a
real cost to orientation.

### 3. Ridge-spectrum loss vs support-only
Reference: `support_constrained_FINAL` — Candidate: `support_spectrum_FINAL`
(epoch 13/20, training was interrupted and not resumed).

| Metric | Improvement | Holm p | Significant |
|---|---:|---:|---|
| Held-out real-pixel MAE | +0.004 | 2.6e-6 | yes |
| Held-out orientation error | +0.006 | 0.021 | yes |
| Held-out ridge-frequency rel. MAE | −0.071 | 0.014 | yes (worse) |
| Held-out SSIM | +0.004 | 0.004 | yes |

Adding the spectrum loss on top of support conditioning improves MAE,
orientation and SSIM together (no three-way tradeoff, unlike ablations 1-2),
but ridge-frequency gets worse — surprising, since this loss targets
frequency directly. Caveat: this arm never finished training (epoch 13/20),
so the frequency result may not reflect the converged model.

### 4. Calibrated geometric confidence vs uniform weights
Reference: `uniform_confidence_FULL` (epoch 18/20) — Candidate:
`calibrated_confidence_FULL` (epoch 13/20), otherwise identical recipe
(support + spectrum). This re-runs `required_ablation` from
`geometric_confidence_calibration_128.json`.

| Metric | Improvement | Holm p | Significant | Practical size |
|---|---:|---:|---|---|
| Background darkness | +0.0001 | 9.5e-5 | yes | negligible |
| Held-out real-pixel MAE | +0.0006 | 2.6e-7 | yes | negligible |
| Held-out orientation error | −0.0054 | 8.5e-5 | yes (worse) | small |
| Held-out ridge-frequency rel. MAE | +0.0102 | 0.973 | no | — |
| Held-out SSIM | +0.0013 | 1.1e-4 | yes | negligible |

**Important methodological note, corrected during this session:** the first
attempt at this ablation compared against a pilot-scale checkpoint (3 epochs,
data subset) instead of the matched full-scale calibrated run, which produced
misleadingly large effect sizes (SSIM +0.018, frequency +0.175). The corrected,
matched comparison above shows the calibrated weighting has a real but small
effect — with ~300k pooled pixels, even negligible effects reach significance,
so Holm significance here should not be read as practical importance. Both
checkpoints were trained to a comparable but not identical epoch (13 vs 18),
which remains a residual confound; neither run was resumed to a matched
stopping point because the training script does not support resuming.

### 5. RePaint (resampling=2, K=5) vs plain DDIM-20 (K=5)
Reference: `pixel_ddpm_ddim20_k5` — Candidate: `repaint_resample2_k5`. Same
checkpoint, same support conditioning; only the sampler differs.

| Metric | Improvement | Holm p | Significant |
|---|---:|---:|---|
| Predictive-mean MAE | +0.045 | 2.2e-8 | yes |
| Predictive-mean SSIM | +0.135 | 2.2e-8 | yes (largest effect in this table) |
| Orientation error | +0.011 | 2.0e-4 | yes |
| Ridge-frequency rel. MAE | +0.012 | 0.115 | no |
| Uncertainty-error Spearman | −0.073 | 1.4e-4 | yes (worse, but stays positive: 0.147) |
| Single-sample MAE | +0.048 | 2.2e-8 | yes |

RePaint dominates DDIM on every fidelity metric with the largest SSIM effect
seen anywhere in this project's NIST302 work, at a real but smaller cost to
uncertainty localization than the residual-diffusion arm below.

### 6. Residual conditional DDPM (`R = X_pseudo - X_coarse`) vs pixel-space DDPM
Reference: `pixel_ddpm_ddim20` (K=10) — Candidate: `residual_ddpm_ddim20` (K=10).
New model built this session: denoiser conditioned on `[Y, M, support,
X_coarse]`, diffusing the residual in a remapped [0,1] space; `X_coarse` is
the frozen support+spectrum deterministic fine-tune (epoch 13/20).

| Metric | Improvement | Holm p | Significant |
|---|---:|---:|---|
| Predictive-mean MAE | +0.035 | 2.2e-8 | yes |
| Predictive-mean SSIM | +0.084 | 2.2e-8 | yes |
| Orientation error | +0.011 | 1.4e-4 | yes |
| Ridge-frequency rel. MAE | −0.039 | 1.4e-3 | yes (worse) |
| Uncertainty-error Spearman | **−0.206** | 2.2e-8 | yes (large; drops to 0.060) |
| Single-sample MAE | +0.025 | 2.2e-8 | yes |

Residual diffusion buys real fidelity (not diffusing the structure the
deterministic model already recovers pays off) but its sample-to-sample
diversity stops tracking true error nearly as well — plausibly because
diversity is now confined to the smaller residual degree of freedom once
`X_coarse` is fixed. No single arm dominates on both fidelity and calibrated
uncertainty.

## Full probabilistic model comparison (descriptive; K differs per arm, see below)

| Model | K | MAE ↓ | SSIM ↑ | Orientation ↓ | Uncertainty-error ρ ↑ |
|---|---:|---:|---:|---:|---:|
| CVAE, spatial, bilinear (validation-only pilot ablation) | 50 | 0.138 | 0.400 | 0.294 | negative |
| DDPM, DDIM-20 (pixel-space) | 10 | 0.155 | 0.305 | 0.302 | **0.273** |
| DDPM, RePaint (resampling=2) | 5 | **0.111** | **0.432** | 0.292 | 0.147 |
| DDPM, residual (`X_coarse` + diffused R) | 10 | 0.121 | 0.389 | **0.291** | 0.060 |
| DDPM, latent-space (f=4 autoencoder) | 10 | 0.215 | 0.245 | 0.303 | 0.145 |

K is not matched across rows (RePaint's ancestral sampler is ~30x the cost of
DDIM per sample, so it was evaluated at K=5 instead of K=10/50); this table is
for orientation, not as a K-controlled ranking. Every pairwise comparison that
matters for model selection was instead run K-matched above (sections 5-6).

**Reading across the whole probabilistic track:** no model wins on both fidelity
and calibrated, localized uncertainty simultaneously. RePaint is the strongest
single-sample and mean-image reconstruction. DDIM-20 has the best-localized
uncertainty signal, confirmed at full statistical power and decomposed by
distance-to-anchor (`nist302_ddpm_distance_diag_full.json`: correlation 0.53 /
0.26 / 0.17 for 0-2mm / 2-5mm / >5mm bands — positive everywhere, weakening
with distance from verified evidence, not collapsing). Both residual diffusion
and the CVAE family lose uncertainty quality as they gain fidelity or spatial
capacity. This tradeoff, not a single winning model, is the honest finding to
report for the "reconstruction vs hallucination" / uncertainty chapter.

### K-sensitivity (K=20, 50) for DDIM and residual DDPM

Both arms were re-evaluated at K=20 and K=50 (full 287-image validation) to
check whether the K=10 comparison in sections 6-7 was sensitive to sample
count.

| Model | K | Mean predictive-mean MAE ↓ | SSIM ↑ | Uncertainty-error ρ ↑ |
|---|---:|---:|---:|---:|
| DDIM-20 | 10 | 0.155 | 0.305 | 0.273 |
| DDIM-20 | 20 | 0.155 | 0.308 | 0.297 |
| DDIM-20 | 50 | 0.155 | 0.311 | 0.324 |
| Residual DDPM | 10 | 0.121 | 0.389 | 0.060 |
| Residual DDPM | 20 | 0.120 | 0.397 | 0.084 |
| Residual DDPM | 50 | 0.120 | 0.403 | 0.100 |

(Sources: `outputs/nist302_ddpm_full_validation_ddim20_k20_50/metrics.json`,
`outputs/nist302_residual_ddpm_full_validation_k20_50/metrics.json`.)

Predictive-mean fidelity is essentially flat in K for both arms (as expected —
averaging more samples barely moves the mean once K is already ≥10), but the
uncertainty-error correlation keeps climbing with K for both, because the
*std* estimate itself gets less noisy with more samples. The gap between the
two arms is stable, though: residual DDPM's correlation roughly doubles from
K=10 to K=50 (0.060→0.100) but never approaches DDIM's, which itself keeps
improving (0.273→0.324). More samples make both arms' uncertainty signal
somewhat more informative, but do not change which arm has the better one, so
sections 6-7's K=10 comparison is not an artifact of an unlucky K choice.

### 7. CVAE (spatial, bilinear, calibrated) vs pixel-space DDPM, matched K=10
Reference: `cvae_spatial_bilinear_k10` — Candidate: `pixel_ddpm_ddim20_k10`.

| Metric | Improvement (DDIM − CVAE) | Holm p | Significant |
|---|---:|---:|---|
| Predictive-mean MAE | −0.017 | 3.6e-3 | yes (DDIM worse) |
| Predictive-mean SSIM | −0.095 | 2.2e-8 | yes (DDIM worse) |
| Orientation error | −0.010 | 0.046 | yes (DDIM worse) |
| Ridge-frequency rel. MAE | −0.005 | 0.522 | no |
| Uncertainty-error Spearman | **+0.434** | 2.2e-8 | yes (DDIM far better) |
| Single-sample MAE | −0.011 | 0.017 | yes (DDIM worse) |

This closes the loop on the whole probabilistic track: CVAE is actually a
*better* fidelity model than plain pixel DDIM (it loses to DDIM only on
uncertainty, where the gap is the single largest effect measured anywhere in
this project's NIST302 work). Combined with sections 5-7, the four
probabilistic arms form a clean two-axis tradeoff, not a strict ranking:
- **Best fidelity:** RePaint > Residual DDPM > CVAE > DDIM.
- **Best uncertainty localization:** DDIM ≫ RePaint > Residual DDPM ≈ CVAE
  (negative).

No arm is simultaneously good at both. This — not a single winning model — is
the finding for the "reconstruction vs hallucination" chapter.

### 8. Data-consistency ablation (null result, confirmed by construction)

Hard data consistency composites `mask*observed + (1-mask)*raw` at inference.
Checked directly on the support+spectrum checkpoint with random inputs: the
composited and raw outputs are **identical (max abs diff = 0.0) everywhere
the missing-region evaluation looks**, and differ only at observed pixels
(max abs diff 0.79), which are excluded from every held-out metric in this
table by construction. This ablation is therefore mathematically null for
every result reported here — enabling or disabling data consistency cannot
change any missing-region number, only the (unreported) observed-pixel exact
value. No retraining was needed or attempted; the effect is architectural,
not learned.

### 9. Model × distance interaction (uncertainty calibration by proximity to verified evidence)

Same held-out quality==1 pixels, split by `evaluation_roi_nearest_{0_2mm,2_5mm,gt_5mm}`
(distance to the nearest examiner-verified correspondence point), pooled Spearman
uncertainty-error correlation per band:

| Model | K | 0-2mm | 2-5mm | >5mm | Pattern |
|---|---:|---:|---:|---:|---|
| DDPM, DDIM-20 | 10 | 0.527 | 0.262 | 0.167 | positive everywhere, weakens with distance |
| CVAE, spatial, bilinear | 20 | 0.011 | −0.149 | −0.297 | **flips negative** with distance |
| DDPM, RePaint (resampling=2) | 5 | 0.389 | 0.105 | 0.017 | positive everywhere, weakens to near-zero |

(Sources: `nist302_ddpm_distance_diag_full.json`, `nist302_cvae_distance_diag_full.json`,
`nist302_repaint_distance_diag_full.json`.)

This sharpens section 7's finding: it is not just that the CVAE's uncertainty
is *weaker* on average than DDIM's — it actively points the wrong way far from
verified evidence (negative correlation, meaning the CVAE tends to be *more*
confident exactly where it is least reliable), while DDIM's and RePaint's
signals both degrade gracefully but never invert. For any deployment that
would use predictive std to flag unreliable regions to an examiner, this is
a materially different failure mode: DDIM/RePaint under-warn at long range,
CVAE actively misleads. DDIM leads RePaint at every single band (0.527 vs.
0.389 near, widening to 0.167 vs. 0.017 far — RePaint's signal is
essentially uninformative beyond 5mm), so DDIM is not just this project's
best-calibrated model on average (section 7), it is also the one whose
uncertainty signal degrades the *slowest* with distance from verified
evidence among the three architectures checked here.

### 10. Isolated loss ablations: which single structural loss matters most?

Reference for both: `full_all_losses` = the same support+spectrum+calibrated-
confidence checkpoint used throughout (`outputs/nist302_registered_support_spectrum_full`,
`orientation_weight=0.35, gradient=0.10, ridge_energy=0.10, ridge_band=0.25,
ridge_spectrum=0.50`). Each candidate zeroes every structural-loss weight
except one, keeping support conditioning and calibrated confidence
weighting unchanged. Held-out real-pixel target throughout (not `X_pseudo`).

| Metric | ridge-band-only vs full | orientation-only vs full |
|---|---:|---:|
| MAE | **worse** (−0.0072, dz=−0.89, p=1.1e-4) | **worse** (−0.0343, dz=−3.38, p=1.9e-8) |
| Orientation error | better (+0.046, dz=1.17, p=4.9e-7) | **worse** (−0.024, dz=−1.21, p=2.1e-7) |
| Ridge-frequency rel. MAE | better (+0.289, dz=1.04, p=1.2e-5) | better (+0.108, dz=0.60, p=3.4e-3) |
| SSIM | better (+0.074, dz=2.15, p=1.9e-8) | **worse** (−0.064, dz=−4.84, p=1.9e-8) |

(Improvement sign: positive always means the candidate is better; source
JSONs `outputs/nist302_ablation_ridge_band_only_vs_full_subject_tests.json`,
`outputs/nist302_ablation_orientation_only_vs_full_subject_tests.json`.)

**Reading:** the two isolated losses behave very differently, and neither
isolated arm is simply "a worse version of the full model" — ridge-band-only
loses only on raw pixel MAE and actually *beats* the full combined-loss model
on orientation error, ridge-frequency accuracy, and SSIM (the largest effect
size in this whole ablation table, dz=2.15). Orientation-only is worse across
the board, including on orientation error itself — the metric its own loss
term is supposed to directly target — while still picking up a smaller,
real improvement on ridge-frequency (orientation and ridge frequency are
correlated ridge-structure properties, so this is not implausible).

The practical conclusion: **ridge-band loss is doing most of the useful
structural work in the combined loss**, and orientation loss alone is a
comparatively weak training signal — plausibly because the ridge-band term
supplies dense, per-pixel, frequency-domain gradient at 8 orientations,
while the orientation-field loss compares a coarser windowed orientation
estimate and gives sparser, more indirect gradient. This also means the
full combined loss is not simply "the sum of its parts": ridge-band-only
outperforming the full model on 3 of 4 metrics suggests some interaction
between the loss terms (possibly the orientation and ridge-energy terms
introducing gradient conflicts, or simply different optimal training
lengths — the full model stopped at epoch 13/18, ridge-band-only at 6/11,
orientation-only at 9/14, all under the same early-stopping criterion on
each run's own validation loss). This is reported as an open finding, not
fully explained; a controlled study varying only the loss combination at a
fixed training length would be needed to isolate the interaction from the
training-length confound.

### 11. Permuted-observed control: does the model use the ridges it is shown, or hallucinate generically?

Motivated by a methodological audit raising exactly this concern. For each
validation image, the pixel DDPM reconstructs the same missing region twice
with an *identical* diffusion noise trajectory (same seed): once conditioned
on its own real observed ridges (`mask * own latent image`), once conditioned
on a **different finger's** real ridges through the same mask shape
(`mask * another sample's latent image`, batch rolled by one). Both
reconstructions are scored against the original sample's own held-out
`quality==1` pixels.

If the model genuinely continues the ridge structure it is shown, feeding it
the wrong finger's ridges should measurably hurt reconstruction quality in
the unobserved region. If the model mostly produces a generic,
mask-shape-conditioned completion regardless of identity, matched and
permuted should score about the same.

| | Matched (own ridges) vs permuted (wrong finger's ridges) |
|---|---|
| Held-out MAE, mean paired change | permuted **worse** by 0.0342 |
| Cohen dz | −2.17 (very large) |
| Holm-adjusted Wilcoxon p | 3.7e-9 |
| n | 287 images, 29 subjects |

(Source: `outputs/nist302_permuted_observed_control_full.json`,
`scripts/diagnose_nist302_permuted_observed_control.py`.)

**Reading:** this is the reassuring result. The DDPM measurably, and by a
large margin, does worse when given the wrong finger's ridges through the
same visible-region shape — it is not simply learning "fill this mask shape
with a generic plausible fingerprint texture" independent of what it is
actually shown. This does not by itself prove the *unobserved* region is
recovered correctly (that remains the central open problem this whole
project documents), but it does rule out the specific, more basic failure
mode of the model ignoring its conditioning input entirely.

### 12. Difficulty-stratified proper scoring rules (CRPS, interval score, sparsification error)

Motivated by a methodological audit noting that Spearman correlation alone only
checks monotonic association, not the magnitude of miscalibration. Extends
section 9's distance analysis with three proper scoring rules (CRPS,
Gneiting-Raftery interval score at nominal 90% coverage, and sparsification
error / AUSE — see `src/fingerprint_reconstruction/metrics/uncertainty.py`)
for the pixel DDPM, stratified along **two independent difficulty axes**:
registration confidence (distance to nearest verified correspondence point,
as in section 9) and mask geometry (pixel-distance to the nearest actually-
observed pixel, per-image tertiles — an axis the registration bands do not
capture, since it reflects the local shape of what is missing, not
registration quality).

| Band (registration confidence) | CRPS ↓ | Interval score ↓ | Sparsification error (AUSE) ↓ | Spearman ↑ | n images |
|---|---:|---:|---:|---:|---:|
| 0-2mm | 0.118 | 1.00 | 0.027 | 0.525 | 261 |
| 2-5mm | 0.136 | 1.31 | 0.052 | 0.371 | 286 |
| >5mm | 0.151 | 1.65 | 0.064 | 0.264 | 263 |

| Band (mask geometry tertile) | CRPS ↓ | Interval score ↓ | Sparsification error (AUSE) ↓ | Spearman ↑ | n images |
|---|---:|---:|---:|---:|---:|
| Near (closest third to observed pixels) | 0.112 | 0.85 | 0.027 | 0.508 | 287 |
| Mid | 0.151 | 1.41 | 0.056 | 0.250 | 287 |
| Far (farthest third) | 0.167 | 2.19 | 0.072 | 0.091 | 287 |

(Source: `outputs/nist302_ddpm_difficulty_calibration_full.json`,
`scripts/calibrate_nist302_ddpm_uncertainty_by_difficulty.py`. Overall
pooled: CRPS 0.142, interval score 1.46, AUSE 0.068, Spearman 0.296.)

**Reading:** all three proper scoring rules agree with each other and with
the Spearman-only finding from section 9, on *both* difficulty axes
independently: every metric monotonically gets worse (CRPS up, interval
score up, AUSE up, Spearman down) as either distance-to-verified-evidence or
distance-to-observed-pixels increases. This is a stronger result than
section 9 alone: it is not just that the *ranking* of uncertain-vs-confident
pixels weakens with difficulty (which Spearman alone shows), but that the
*calibrated interval quality* (interval score, which penalizes both
overconfident misses and needlessly wide intervals in one score) and the
*practical usefulness of discarding high-uncertainty pixels* (sparsification
error, which cannot be gamed by uniformly widening every interval the way
raw coverage can) both degrade in the same direction. The mask-geometry axis
shows a numerically larger spread than the registration-confidence axis
(Spearman 0.508→0.091 vs. 0.525→0.264), suggesting how far a pixel is from
*any* observed evidence is at least as strong a difficulty driver as
registration quality specifically — a genuinely new axis this project had
not separately measured before this check.

**Caveat consistent with section 9 and `docs/limitations.md`:** this remains
a Track C / Track D analysis (held-out `quality==1` pixels, "weak trace
enhancement," not "truly missing region recovery") and the pixel DDPM only;
the equivalent breakdown for RePaint, residual DDPM, and CVAE was not
re-run with these three scoring rules due to time, only with Spearman
(section 9).

### 13. Checkpoint re-selection by structural composite metric vs. noise loss

Motivated by a methodological audit noting the DDPM checkpoint is selected by
validation noise-prediction MSE, not by any of the metrics this project
actually reports. `scripts/train_nist302_ddpm.py` was changed this session to
save a checkpoint every epoch (previously only the noise-loss best was kept),
and a new script (`scripts/select_nist302_ddpm_checkpoint_by_structural_metric.py`)
re-evaluates every saved epoch on a validation subset using
`J(epoch) = z(MAE) + z(orientation_error) + z(ridge_frequency_relative_mae)`
(equal-weighted z-scores across the run's own epochs, fixed by construction
before any epoch identity is inspected). This required a full pixel-DDPM
retrain (`outputs/nist302_registered_ddpm_full_with_epochs`) since the
original run never saved per-epoch checkpoints.

| Selection criterion | Epoch chosen | Held-out MAE | Orientation error | Ridge-freq. rel. MAE |
|---|---:|---:|---:|---:|
| Validation noise-prediction MSE (original method) | 12 | 0.155 | 0.286 | 0.504 |
| Structural composite (this check) | 8 | **0.114** | **0.280** | **0.464** |

(60-image validation subset, DDIM-10, K=3 samples/image, averaged;
`outputs/nist302_ddpm_checkpoint_selection_comparison.json`. Full 287-image,
higher-K confirmation was not re-run given time constraints — this is a
directional finding on a representative subset, not a final number.)

**Reading: the criteria disagree, and the disagreement matters.** The
noise-loss-selected checkpoint (epoch 12) is *worse* on all three
reported-metric axes than the structural-composite pick (epoch 8) — not a
marginal difference (MAE 0.155 vs. 0.114, roughly a 36% relative gap). This
directly confirms the concern motivating this check: minimizing noise-
prediction MSE on validation is not the same objective as minimizing the
metrics this project actually reports, and the standard training loop
silently discarded the better-by-every-other-measure epoch 8 checkpoint in
favor of a later epoch that happened to denoise marginally better in a sense
nobody downstream cares about. **Practical implication for any future
NIST302 DDPM training in this project: select checkpoints by this (or a
similar) held-out structural composite, not noise loss.** The existing
`outputs/nist302_registered_ddpm_full/best-checkpoint.pt` used throughout
this document's other DDPM results was selected by noise loss and was not
retroactively replaced — re-running every DDPM-dependent comparison in this
document against a composite-selected checkpoint was out of scope for the
time remaining, so this section should be read as identifying a real,
actionable methodological gap rather than a retroactive correction of prior
numbers.

### 14. Pixel diffusion vs. latent diffusion on NIST302

Previously deprioritized as too expensive (see the earlier version of "Still
pending" below); done after all, per explicit instruction to attempt every
item on the external-review plan. Built from scratch this session:
`scripts/train_nist302_latent_autoencoder.py` (SD302-registered-target
fine-tune of the SOCOFing autoencoder, PSNR 33.97dB / SSIM 0.913 / orientation
error 0.0036 on its own codec-reconstruction task — the compression stage
itself is excellent), `scripts/train_nist302_latent_diffusion.py` (denoiser
trained from scratch on SD302, matching the pixel/residual DDPM precedent —
not SOCOFing-pretrained, see `docs/cross_dataset_protocols.md`), and
`scripts/evaluate_nist302_latent_diffusion.py` (held-out real-pixel target
throughout, same methodology as every other DDPM evaluation in this
project).

K-matched (K=10) paired subject-level comparison against pixel DDPM (DDIM-20):

| Metric | Pixel DDPM (reference) | Latent diffusion (candidate) | Change | Holm p |
|---|---:|---:|---:|---:|
| Mean-K10 MAE | 0.155 | 0.215 | **worse by 0.058** | 4.1e-8 |
| Mean-K10 SSIM | 0.305 | 0.245 | **worse by 0.056** | 4.1e-8 |
| Best-of-10 MAE | — | — | **worse by 0.077** | 4.1e-8 |
| Single-sample MAE | — | — | **worse by 0.127** | 4.1e-8 |
| Ridge-frequency rel. MAE | — | — | **worse by 0.067** | 1.2e-3 |
| Orientation error | — | — | no significant difference | 0.49 |
| Background darkness | — | — | slightly *better* (−0.004) | 2.3e-6 |
| Uncertainty-error ρ (K10, descriptive) | 0.273 | 0.145 | worse (not paired-tested) | — |

(Source: `outputs/nist302_pixel_vs_latent_diffusion_subject_tests.json`,
287 images, 29 subjects, DDIM-20 vs DDIM-50 sampling.)

**Reading: latent diffusion is worse than pixel-space diffusion on this
problem, decisively.** Every fidelity metric that reached significance
favors pixel DDPM, several by large effect sizes (single-sample MAE
dz=−3.96, one of the largest effect sizes in this whole project's DDPM-family
comparisons). The one metric where latent diffusion wins (background
darkness, i.e. less unwanted dark-region hallucination) is real but tiny in
absolute terms and does not offset the fidelity gap. This is a genuine,
well-powered negative result, not a training failure: the autoencoder stage
alone reconstructs SD302 registered targets to PSNR 34dB, so the compression
is not the bottleneck by itself — the loss appears to happen specifically in
the *conditional generative* step, plausibly because compressing to a
32×32×4 latent (4x spatial downsampling) discards exactly the
fine-grained, phase-sensitive ridge-period information (`docs/nist302_256px_frequency_check.md`
already flagged ridge periods as only a few pixels wide even at full
128×128 resolution) that the denoiser would need to reconstruct precisely,
and the much smaller denoiser (1.98M vs. 8.28M parameters) has less capacity
to compensate. At 128×128 on fingerprints specifically, this project finds
no fidelity benefit to diffusing in a compressed latent space — the
motivation for latent diffusion in the literature (compute cost at high
resolution) does not apply here, and the numbers confirm it does not pay
for itself for free either.

### 15. Ridge-topology artifact check: does the DDPM paint coherent fake bands?

Motivated directly by a methodological audit's qualitative observation that some
reconstructions "contain coherent, nearly-horizontal bands, which reduce
orientation error without faithfully reconstructing ridge topology."
Orientation error alone (mean angular distance to a coarse reference) cannot
detect this failure mode — a locally uniform stripe pattern can score well on
orientation error while being structurally wrong. Built three new,
approximate diagnostic tools this session
(`src/fingerprint_reconstruction/metrics/ridge_topology.py`: adaptive
binarization, skeletonization, crossing-number minutiae counting, and an
orientation-curvature measure) and compared the pixel DDPM's mean
reconstruction against the real latent pixels, within the identical held-out
`quality==1` region, on all three axes a "painted stripe" artifact would
distort.

| Metric (reconstruction − real, within the same held-out region) | Mean diff | 95% CI | Direction | n images |
|---|---:|---:|---|---:|
| Orientation coherence | **+0.181** | [0.169, 0.193] | reconstruction more uniform (suspicious) | 284 |
| Orientation curvature | **−0.022** | [−0.025, −0.019] | reconstruction flatter/straighter (suspicious) | 284 |
| Minutiae density (per 1000px) | **−9.78** | [−12.1, −7.4] | reconstruction has ~half the minutiae of real ridges | 275 |

(Real vs. reconstructed absolute levels: coherence 0.637 → 0.818, curvature
0.103 → 0.081, minutiae density 21.6 → 11.8 per 1000px. Source:
`outputs/nist302_ridge_topology_artifacts_full.json`,
`scripts/diagnose_nist302_ridge_topology_artifacts.py`.)

**Reading: the methodological audit's qualitative observation is confirmed
quantitatively, on all three independent structural axes, with tight
confidence intervals excluding zero by a wide margin.** The DDPM's mean
reconstruction in the held-out region is measurably more orientation-uniform,
less naturally curved, and has roughly half the minutiae density of the real
ridges it is being compared against. This is a genuinely different finding
from anything in sections 1-14: those sections measure *how far* the
reconstruction is from the target on pixel/orientation/frequency metrics;
this section shows *why* — the reconstruction is not merely noisy or blurry,
it is systematically simplifying the ridge topology into smoother, less
structured bands. This directly explains how a model can post a
"reasonable" mean orientation error (section 4-6) while still not
representing genuine ridge continuity, and gives the "reconstruction vs
hallucination" framing (`docs/reconstruction_vs_hallucination.md`) a concrete,
measurable mechanism rather than only an aggregate fidelity/uncertainty
tradeoff.

**Caveats:** the binarization/skeletonization/minutiae-counting pipeline is
explicitly approximate and diagnostic-grade (see the module's own docstring)
— it is not a forensic-grade minutiae extractor and should not be used to
claim calibrated minutiae accuracy. Only the pixel DDPM was checked this way;
the equivalent comparison for RePaint, residual DDPM, CVAE, and latent
diffusion was not run given time constraints, so it is unknown whether this
artifact is DDPM-specific or common across this project's probabilistic
models — a natural next check if this line of investigation continues.

### 16. Non-rigid (thin-plate-spline) vs. affine registration: is affine leaving error on the table?

Motivated by a methodological audit's concern that SD302's affine registration
may not capture genuine local (elastic) skin deformation between latent and
exemplar impressions, which would inflate residual registration error in a
way that varies spatially — exactly what the confidence-weighted losses
throughout this project try to compensate for. Built
`src/fingerprint_reconstruction/evaluation/nonrigid_registration.py` (a
thin-plate-spline point-mapper and image-warp, matching the existing affine
module's calling convention) and compared both methods the only fair way
possible with ~9-20 correspondence points per image:
**leave-one-correspondence-out** RMSE (fit on N-1 points, predict the held-
out point) — not the manifest's own in-sample `affine_rmse_mm`, which cannot
detect overfitting since a sufficiently flexible model trivially fits its
own training points exactly.

| Split | Affine LOO RMSE | TPS LOO RMSE | Paired improvement | Cohen dz | Holm p | n subjects |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.296mm | 0.291mm | +0.0053mm | 0.13 (small) | 0.073 | 138 |
| Validation | 0.310mm | 0.298mm | +0.0115mm | 0.31 (small) | 0.143 | 29 |

(For context only, not a fair comparator: the manifest's own in-sample
affine RMSE averages 0.194mm/train, 0.199mm/validation — substantially lower
than *either* method's true leave-one-out generalization error, confirming
in-sample residuals meaningfully understate real registration uncertainty
throughout this project's other documents that cite `affine_rmse_mm`
directly.)

**Reading: TPS gives a small, consistent, but not statistically significant
improvement over affine on both splits.** This is an informative near-null
result, not an unexplored gap: with the number of correspondence points
SD302 actually provides (~9-20 per image), a non-rigid model captures at
most a small amount of additional geometric fit beyond what an affine
transform already achieves, and the improvement does not clear conventional
significance thresholds on either split. This suggests the affine
registration used throughout this project is a reasonable choice given the
available correspondence density — not a major unaddressed source of
registration error — though it does not rule out that denser correspondence
annotation (more points per image, which this project cannot add) might
reveal a larger non-rigid signal. The gap between in-sample and leave-one-out
RMSE (≈0.10mm on both splits, for affine) is itself a useful number: it is
the honest generalization penalty hidden by every in-sample registration
RMSE figure cited elsewhere in this project.

### 17. Track E: synthetic degradation of clean SD302 exemplars, exact missing-region ground truth

Motivated by a methodological audit's point 9: every other evaluation in this
project is Track B (approximate registered exemplar) or Track C (real
held-out pixels, which are themselves faint *existing* traces per
`docs/limitations.md`, not genuinely absent regions). This is the one place
in the project's NIST302 work with **exact** ground truth for the missing
region: `Sd302SyntheticPartialDataset`
(`src/fingerprint_reconstruction/data/nist302_synthetic_dataset.py`)
degrades a single clean SD302 exemplar (elastic warp + blur + contrast
reduction + noise, `preprocessing/degradation.py`) then applies a synthetic
partial-visibility mask (reusing SOCOFing's mask-family infrastructure
directly). Since the mask is applied *after* degradation, the pre-mask
pixel values are exact ground truth for the missing region by construction
— at the cost of the degradation being simulated, not a real capture
process. Subject splits are derived from the same `registered_manifest.csv`
subject→split assignment used everywhere else, so this never touches the
sealed test subjects.

| Metric (missing region, exact ground truth) | Zero-shot | Fine-tuned | Paired improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1456 | 0.1456 | ~0 (tie) | −0.01 | 0.97 |
| SSIM | 0.2571 | 0.1959 | **−0.060 (worse)** | −2.78 | 1.5e-8 |
| Orientation error | 0.7427 | 0.7760 | **−0.029 (worse)** | −0.56 | 0.0047 |
| Ridge-frequency rel. MAE | 0.5286 | 0.5649 | **−0.034 (worse)** | −0.71 | 0.0024 |

(800 images, 29 subjects, validation split; sources:
`outputs/nist302_synthetic_exact_gt_zeroshot_full/metrics.json`,
`outputs/nist302_synthetic_exact_gt_finetuned_full/metrics.json`,
`outputs/nist302_synthetic_exact_gt_zeroshot_vs_finetuned_subject_tests.json`,
`scripts/evaluate_nist302_synthetic_exact_groundtruth.py`.)

**Reading: this independently reproduces the project's central critical
finding (top of this document) with a completely different, exact-ground-
truth methodology.** Fine-tuning on SD302's `registered_approximate` pairs
makes the model measurably worse at recovering a genuinely missing region on
SD302-domain images generally — not only on real latents, and not only
against the specific held-out real pixels used in the original finding. The
absolute error levels here (orientation error ≈0.74-0.78) are notably higher
than the equivalent Track C numbers (≈0.29-0.30) — expected, since Track E's
"missing region" is the *entire* masked area (comparable to SOCOFing's task
difficulty) rather than Track C's sparse, edge-concentrated held-out pixels,
and the synthetic degradation includes an elastic warp the models were never
trained to invert. This is a harder, more literal "missing region recovery"
test than anything else in the NIST302 track, and it still shows the same
qualitative pattern.

**Caveats:** the degradation pipeline is simulated, not a real capture
process — it demonstrates the models generalize poorly to a *harder,
exactly-specified* missing-region task in the SD302 image domain, not that
it exactly reproduces real latent degradation physics. No new model was
trained on this task this session (only existing zero-shot/fine-tuned
checkpoints were evaluated on it); training directly on Track E data was
part of the original external-review proposal and remains open future work.

### 18. Two-stage (structure-guided) latent diffusion vs. plain latent diffusion

Motivated by a methodological audit's point 3: predict support/orientation/
frequency first, then condition the texture-synthesis diffusion on those
predicted fields. This project's SOCOFing latent-diffusion code already had
exactly this hook built in but unused (`ConditionalLatentDDPM`'s
`structure_predictor` argument, requiring a 4-channel auxiliary condition:
support + doubled-angle orientation (x, y) + coherence — not an explicit
frequency channel, a documented simplification since orientation/coherence
already capture much of what a naive frequency channel would add). Fine-
tuned `FingerprintStructurePredictor` (already built for SOCOFing) on SD302
(`scripts/train_nist302_structure_predictor.py`, warm-started from the
SOCOFing checkpoint, target computed on the fly from `X_pseudo`) and wired
it into a second latent-diffusion training run.

**Correction: this is already ControlNet-style multiscale injection, not
naive input concatenation.** `DiffusionUNet`'s `auxiliary_condition_channels`
mechanism (used by every conditioned DDPM-family model in this project —
support alone for pixel DDPM, support+coarse for residual DDPM, the
4-channel structure prediction here) injects the auxiliary signal at
*every* encoder resolution via a separate `1x1` convolution per scale, each
explicitly zero-initialized (`nn.init.zeros_` on both weight and bias) —
exactly the "zero convolution" mechanism that defines ControlNet, not a
single input-layer concatenation. A methodological audit separately proposed
building a "ControlNet compact" model as a distinct new architecture item;
that item is therefore already covered by every structure/support-
conditioned model in this document, including this section's result, and
does not need a separate implementation.

| Metric | Plain latent diffusion | Structure-guided | Paired improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| Ridge-frequency rel. MAE | 0.572 | 0.533 | **+0.053 (better)** | 0.57 | 0.0027 |
| SSIM (K=10 mean) | 0.245 | 0.242 | −0.003 (worse) | −0.61 | 0.032 |
| MAE (K=10 mean) | 0.215 | 0.217 | −0.002 (worse) | −0.41 | 0.13 (n.s.) |
| Single-sample MAE | — | — | −0.003 (worse) | −0.47 | 0.14 (n.s.) |
| Coverage error (80/90/95%) | — | — | worse (all 3) | 0.60-0.65 | 0.02-0.04 |
| Orientation error | — | — | −0.001 (worse) | −0.29 | 0.25 (n.s.) |

(287 images, 29 subjects; sources:
`outputs/nist302_latent_diffusion_structure_guided_full_validation/metrics.json`,
`outputs/nist302_latent_diffusion_structure_guided_vs_plain_subject_tests.json`.)

**Reading: a genuine, real mixed result, not a clean win.** Structure
guidance gives a real, significant improvement specifically on ridge-
frequency accuracy (the largest effect size in this comparison, and the
metric most directly related to what the structure predictor is trained to
predict) but mildly *worsens* SSIM and uncertainty-interval coverage
calibration, both significant after correction; MAE, single-sample fidelity,
and orientation error show small, non-significant differences either way.
The two-stage architecture helps the specific structural property it was
built to inform, without improving (and slightly hurting) overall pixel
fidelity or calibration. Since both latent-diffusion variants remain
dominated by pixel-space DDPM on essentially every axis (section 14), this
refinement does not change the project's overall conclusion about latent
diffusion for this problem — it is a real, modest internal finding within
the latent-diffusion family, not a resolution of section 14's negative
result.

### 19. Boundary/phase-continuity loss: does penalizing ridge discontinuity at the mask edge help?

Motivated directly by a methodological audit's proposal (`L_boundary`, penalizing
a jump in the directional derivative along the local ridge tangent exactly
at the observed/missing boundary). Implemented as a new term in
`MaskedReconstructionLoss`/`RegisteredApproximateLoss`
(`_boundary_continuity_loss`, `src/fingerprint_reconstruction/losses/reconstruction.py`):
derives the ridge tangent from the target's own structure tensor, computes
the Sobel-gradient projection along that tangent for both prediction and
target, and penalizes the difference within a thin band straddling the mask
edge (`boundary_band_width`, default 5px). Added on top of the full
support+spectrum config (identical loss weights otherwise) and fine-tuned
from the same SOCOFing checkpoint.

| Metric (held-out real pixels) | Full baseline | + boundary continuity | Paired improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1479 | **0.1414** | +0.0062 (better) | 2.03 | 3.0e-8 |
| SSIM | 0.2807 | **0.3010** | +0.0198 (better) | 2.66 | 1.9e-8 |
| Orientation error | 0.3372 | **0.3235** | +0.0134 (better) | 1.01 | 4.1e-5 |
| Ridge-frequency rel. MAE | 0.8293 | 0.8197 | +0.0137 (better, n.s.) | 0.14 | 0.39 |

(287 images, 29 subjects; sources:
`outputs/nist302_boundary_continuity_heldout/metrics.json`,
`outputs/nist302_boundary_continuity_vs_full_subject_tests.json`.)

**Reading: this is a clean, strong, statistically decisive win — one of the
best-supported single-change improvements in this entire document.** Three
of four metrics improve with large effect sizes (Cohen dz 1.0-2.7) and
Holm-significant p-values as low as 1.9e-8; the fourth (ridge-frequency)
moves in the same favorable direction without reaching significance. Unlike
several other additions in this document (e.g. section 4's confidence-
weighting, which was statistically significant but practically small), this
effect is both statistically robust and practically large. It directly
supports the diagnostic reasoning behind section 15 (the "painted stripe"
artifact): explicitly penalizing discontinuity exactly at the boundary
where the model must hand off from real, observed ridge structure to
synthesized structure measurably improves fidelity broadly, not just at the
boundary itself. This is a strong candidate to fold into this project's
recommended default configuration going forward.

### 20. Two new sampling paradigms on Track E: flow matching and Brownian Bridge Diffusion

Both built and tested this session as genuinely different alternatives to
this project's DDPM/DDIM family (not new noise schedules — different
governing dynamics entirely): **conditional residual flow matching**
(`src/fingerprint_reconstruction/models/flow_matching.py`, a straight-line
ODE from a noised coarse reconstruction to the target, sampled in as few as
4-10 Euler steps) and **conditional Brownian Bridge Diffusion**
(`src/fingerprint_reconstruction/models/brownian_bridge.py`, a stochastic
bridge directly between the masked/filled input and the target, per an
methodological audit's suggestion to validate it on exact-ground-truth data
before real latents).

**A real bug was found and fixed in the BBDM implementation before any
result here is reported.** The first trained BBDM checkpoint collapsed to a
near-uniform black output in the missing region (visually confirmed in
`outputs/nist302_bbdm_track_e_full/samples-preview.png`) — traced to two
compounding issues: the default noise scale (`s=1.0`) made the injected
bridge noise (up to `sqrt(0.5)≈0.71` standard deviation at the bridge
midpoint) dwarf the actual pixel signal, and the deterministic sampler's
algebraic `x0` recovery divides by `(1 - m_t)`, which was allowed to reach
as low as `1e-4` at the start of sampling — amplifying any small numerator
by up to 10,000x. Fixed by reducing the default noise scale to `s=0.3` and
widening the numerical safety margin (`epsilon` from `1e-4` to `0.05`,
capping the worst-case amplification at 20x) and retraining from scratch;
the retrained model no longer collapses (see
`outputs/nist302_bbdm_track_e_full_fixed/samples-preview.png`), though its
output remains visibly smoother/blurrier than this project's other
probabilistic models.

Both were evaluated on Track E (synthetic degradation, exact ground truth);
flow matching's training run had not finished evaluation as of this
writing, BBDM vs. the deterministic zero-shot baseline on the same 28
common subjects:

| Metric | Zero-shot deterministic | BBDM (fixed) | Paired improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1456 | 0.1443 | −0.0102 (worse) | −0.51 | 0.018 |
| SSIM | 0.2571 | **0.3315** | **+0.0483 (better)** | 3.12 | 3.0e-8 |
| Orientation error | 0.7427 | **0.6992** | **+0.0518 (better)** | 1.18 | 5.7e-6 |
| Ridge-frequency rel. MAE | 0.5286 | 0.5656 | −0.0282 (worse) | −0.61 | 0.013 |

(Source: `outputs/nist302_bbdm_vs_zeroshot_subject_tests.json`.)

**Reading:** a genuinely mixed but not unfavorable result for a same-day,
once-debugged new sampling paradigm — BBDM is significantly *better* on
SSIM (very large effect, dz=3.12) and orientation error, significantly
*worse* on MAE and ridge-frequency, all with real effect sizes. Combined
with the visibly smoother output, this is consistent with BBDM learning a
well-calibrated *coarse* reconstruction (good structural/SSIM agreement)
without yet resolving fine ridge texture (worse ridge-frequency) — plausibly
addressable with more training or sampling steps, neither of which this
session had time to tune. This should be read as a preliminary result from
a single configuration, not a mature comparison to the rest of this
document's more extensively tuned models.

### 21. Ensemble: does combining pixel DDPM, residual DDPM, and CVAE beat any single model?

Two ensembling strategies over the three already-trained probabilistic base
models: a **simple average** of their K-sample predictive means, and a
**learned pixel-wise combiner** (`EnsembleCombiner`,
`src/fingerprint_reconstruction/models/ensemble.py` — a small CNN outputting
a per-pixel softmax weight over the three models, so it can learn e.g. "trust
CVAE more here, residual DDPM more there" rather than one global weight).
The combiner was trained on **Track E** (exact synthetic ground truth)
specifically to avoid the `X_pseudo` trap documented in this project's
central finding, then evaluated — like everything else in this table — on
Track C (real held-out pixels).

| Metric | Pixel DDPM | Residual DDPM | CVAE | Simple average | Learned ensemble |
|---|---:|---:|---:|---:|---:|
| MAE ↓ | 0.1430 | **0.1197** | 0.1407 | 0.1266 | 0.1229 |
| SSIM ↑ | 0.3191 | 0.3907 | 0.3950 | 0.3929 | **0.3976** |
| Orientation error ↓ | 0.2997 | 0.2940 | 0.2982 | **0.2877** | 0.2929 |

(287 images, 29 subjects; sources: `outputs/nist302_ensemble_evaluation_full/metrics.json`,
`outputs/nist302_ensemble_key_comparisons.json`.)

Key paired subject-level comparisons:

| Comparison | Result | Cohen dz | Holm p |
|---|---|---:|---:|
| Residual DDPM (best individual) vs. learned ensemble, MAE | residual DDPM **better** | −0.57 | 0.008 |
| Simple average vs. learned ensemble, MAE | learned ensemble **better** | 0.71 | 5.5e-4 |
| CVAE (best individual) vs. learned ensemble, SSIM | tie | 0.04 | 1.0 |
| Simple average vs. learned ensemble, SSIM | learned ensemble **better** | 0.64 | 0.0044 |
| Simple average vs. learned ensemble, orientation | simple average **better** | −0.96 | 5e-5 |

**Reading: a genuinely mixed, honest result — ensembling is not a free win
here.** The learned combiner significantly beats plain averaging on two of
three metrics (MAE, SSIM), confirming the pixel-wise learned weighting adds
real value over a naive average. But **neither ensembling strategy beats
the single best individual model on its own best metric**: residual DDPM
alone still wins MAE, and the learned ensemble only ties (not beats) CVAE on
SSIM. Simple averaging, not the learned combiner, gives the best orientation
error of anything in this comparison. One likely contributing factor: the
learned combiner puts the *most* weight on pixel DDPM (mean weight 0.578,
vs. 0.288 for residual DDPM and 0.134 for CVAE) despite pixel DDPM being
the *worst* individual model on every metric shown here — plausibly because
the combiner was trained on Track E (synthetic degradation) and evaluated
on Track C (real latents), a domain-shift gap that could bias which model
the combiner learns to trust. **Practical conclusion: for this project's
three base models, simply using the best individual model for the metric
that matters (residual DDPM for fidelity, or accepting CVAE/ensemble
parity for SSIM) is as good as or better than building an ensemble** — a
useful, if negative, result given the extra complexity and training cost
ensembling adds.

### 22. Flow matching, evaluated against real held-out pixels: does it beat the best existing models?

Full 287-image Track C evaluation of the flow-matching checkpoint from
section 20, at 8 Euler steps (matching training) and, to directly answer
"does more compute help," a second run at 20 steps.

| Steps | MAE (K=10) | SSIM (K=10) | Orientation | Ridge-freq. rel. MAE | Sec/image |
|---|---:|---:|---:|---:|---:|
| 8 | 0.1602 | 0.3305 | 0.2978 | **0.5061** | 1.64 |
| 20 | 0.1635 | 0.3242 | 0.3022 | 0.5125 | 1.03 |

**More steps did not help — if anything, 20 steps is marginally worse than
8 on every metric** (within noise, not formally tested, but consistently in
the same direction across all four metrics). This confirms flow matching's
core claim for this model: the learned velocity field is already close to
straight, so additional integration steps buy little; 8 steps is not an
undertrained shortcut, it is close to where this model saturates. (The
sec/image figures are not a clean cost comparison — both ran under variable
background contention from other jobs this session — so no speed claim is
made from them.)

K-matched paired comparisons against this project's two strongest fidelity
models:

| Comparison | MAE | SSIM | Orientation | Ridge-freq. rel. MAE |
|---|---|---|---|---|
| vs. RePaint (K=5) | flow matching **worse**, dz=−2.46, p<0.001 | flow matching **worse**, dz=−1.80, p<0.001 | tie (p=0.34) | tie (p=0.76) |
| vs. residual DDPM (K=10) | flow matching **worse**, dz=−2.31, p<0.001 | flow matching **worse**, dz=−1.29, p<0.001 | flow matching worse, dz=−0.41, p=0.03 | flow matching **better**, dz=0.78, p=3.4e-4 |

(Sources: `outputs/nist302_flow_matching_vs_repaint_subject_tests.json`,
`outputs/nist302_flow_matching_vs_residual_ddpm_subject_tests.json`.)

**Reading: flow matching does not beat this project's best models on
overall fidelity, but it has one genuine, statistically confirmed strength
— the best ridge-frequency accuracy of any model evaluated in this entire
NIST302 track (0.506, ahead of residual DDPM's ~0.52-0.56 and DDIM's
~0.52-0.56).** Both RePaint and residual DDPM clearly and significantly beat
it on MAE and SSIM, with large effect sizes; this is a first-attempt,
same-day model with no hyperparameter search (a single `sigma_0=0.05`,
never tuned), so this gap should not be read as a ceiling on flow matching
as a method, only on this particular untuned configuration. The
ridge-frequency win, on top of the visually convincing preview
(`outputs/nist302_flow_matching_full/samples-preview.png`), is a genuinely
promising signal for a model built and trained in hours rather than the
extensive tuning behind RePaint and residual DDPM — a reasonable stopping
point for this session, with real future-work potential (sigma_0 sweep,
longer training, and combining flow matching's frequency strength with
residual DDPM's fidelity, e.g. as an ensemble member per section 21).

### 23. Ridge-Markov predictability: is the reconstructed orientation field too regular to be real?

Section 15 established descriptively that the DDPM reconstruction is more
locally uniform (coherence +0.181) and less curved (curvature -0.022) than the
real ridges it replaces. Those are two summary statistics; this section turns
the same intuition into a statistical test. The orientation field is read as a
first-order Markov chain over 12 axial orientation states (bins of [0, pi)),
and two quantities are measured inside the *same* held-out region for the
reconstruction and for the real latent: the conditional entropy
H(theta_{t+1} | theta_t), and the self-transition rate P(theta_{t+1} = theta_t).

Scoring both arms on the same region matters: entropy estimates are biased
downward when few transitions are counted, and here that bias is shared by the
two arms and largely cancels in the paired difference, rather than having to be
corrected with an estimator whose assumptions would themselves need defending.

| Measure (284 images, 29 subjects) | Real ridges | Reconstruction | Cohen dz | Holm p |
|---|---:|---:|---:|---:|
| Conditional entropy, step 2px | 0.968 bits | **0.851** | -1.63 | 3.7e-8 |
| Conditional entropy, step 4px | 1.490 bits | **1.313** | -1.75 | 7.1e-8 |
| Self-transition rate, step 2px | 0.766 | **0.810** | -2.06 | 1.5e-8 |
| Self-transition rate, step 4px | 0.586 | **0.647** | -1.80 | 3.4e-8 |

(Sources: `outputs/nist302_ridge_markov_full.json`,
`outputs/nist302_ridge_markov_subject_tests.json`;
`src/fingerprint_reconstruction/metrics/ridge_markov.py`,
`scripts/diagnose_nist302_ridge_markov.py`.)

**Reading: the reconstruction's ridge flow is measurably too predictable.**
Knowing the orientation at a pixel tells you more about the orientation 4px
away in the reconstruction than it does in the real ridges occupying that same
region, by 0.17 bits, and the orientation simply fails to change 6 percentage
points more often. All four measures are Holm-significant with large effect
sizes. This is the statistical form of section 15's "painted stripe" artifact:
the model produces flow that is smoother than friction ridges actually are,
which is exactly what a model optimising a pixel loss under uncertainty should
do, and exactly what destroys the discontinuities that carry identity.

The 4px step is the more meaningful of the two, being comparable to the 6.45px
ridge period measured at this resolution (`docs/nist302_native_resolution_feasibility.md`);
the 2px step partly measures the orientation estimator's own smoothing, which
is why both are reported rather than one.

### 24. Does the evaluation target change the model *ranking*, not just one pair?

The critical finding at the top of this document shows the target choice
reversing a conclusion for **one** checkpoint pair. The question a practitioner
actually faces is stronger: if you rank a whole family by the convenient
target, do you select the model you would have selected using real evidence?

Eight deterministic checkpoints were scored against both targets in a single
pass over the same 287 validation images (`scripts/compare_nist302_target_rankings.py`):
Track B is `X_pseudo` over missing pixels inside the evaluation ROI, Track C is
the real latent over officially held-out `quality==1` pixels. Models were then
ranked by subject-mean score under each target and the two orderings compared.

| Metric | Spearman rho | p | Kendall tau | Reading |
|---|---:|---:|---:|---|
| SSIM | **+0.976** | 0.0000 | +0.929 | the pseudo-target is a safe proxy |
| MAE | -0.190 | 0.651 | -0.214 | no usable relationship |
| Orientation error | **-0.786** | 0.021 | -0.571 | ranking substantially inverted |

Concretely, on orientation error the pseudo-target ranks `zero_shot` **last of
eight** while real pixels rank it **second**; on MAE the winner under the
pseudo-target (`finetune_no_spectrum`) places **seventh of eight** against real
evidence.

(Source: `outputs/nist302_target_ranking_validation.json`. Run on validation,
deliberately not on the sealed test split, which has been opened exactly once.)

**Reading: the target choice corrupts model selection, but not uniformly.**
This is a more useful result than a blanket warning would have been. Someone
working on SD302 can rank models by SSIM against the registered exemplar and
expect the right answer; ranking by orientation error against it gives close to
the reverse of the truth, and ranking by MAE gives noise. The failure is
specific to the metrics whose optimum depends on ridge phase, which
`X_pseudo` cannot carry because it is a different impression.

**Caveat stated plainly:** rank correlations over 8 models are weak instruments.
Only the orientation result separates from zero at the 0.05 level. The MAE
result supports "uncorrelated", not "inverted". Extending this to the
probabilistic family would require re-running those models against Track B,
which has not been done.

### 25. Can the missing orientation field be recovered without learning?

An external suggestion proposed reframing the task: instead of inventing the
missing *pixels*, complete the missing *geometry* first -- reconstruct the
ridge orientation field from the geometry around the hole, then synthesise
ridges along it. The proposed objective had three terms: data fidelity on the
observed region, a smoothness term, and a topology term encoding the core/delta
singularity structure (the classical zero-pole model).

This section tests the first two terms only. Orientation is carried in
doubled-angle form `(cos 2t, sin 2t)` so that `t` and `t+pi` are identical,
and each component is completed by solving Laplace's equation inside the hidden
region with the observed field as Dirichlet data -- the exact minimiser of the
smoothness term, with no learning and no fingerprint prior of any kind.
The observed region is eroded by 3px first, because the orientation estimator
sees the flat grey fill and produces garbage near the mask edge.

| Condition | Nearest observed | Harmonic (geometry only) | Trained model |
|---|---:|---:|---:|
| random_rectangles, r=0.50 | 0.5007 | 0.3521 | **0.0595** |
| central_missing, r=0.60 | 0.5426 | 0.4269 | **0.0578** |
| irregular, r=0.50 | 0.5296 | 0.4113 | **0.0267** |
| disconnected_fragments, r=0.30 | 0.6124 | 0.5502 | **0.0609** |
| **All** | 0.5463 | 0.4351 | **0.0512** |

(100 SOCOFing validation images, axial orientation error in the hidden region,
lower is better. Sources: `outputs/socofing_orientation_geometry.json`,
`scripts/complete_orientation_field_geometrically.py`.)

Paired over the same 100 images: harmonic completion beats the nearest-observed
floor on 86% of images (Wilcoxon p = 1.6e-14), and the trained model beats
harmonic completion on **100%** of images (p = 3.9e-18).

**Reading: the smoothness-only version of the proposal is falsified, and the
way it fails is informative.** A smoothness prior does carry real information
-- it improves on the non-learned floor by 20% overall and by 30% on
rectangular holes -- but it remains 8.5x worse than the trained model, losing
on every single image. The gap is widest exactly where it should be if ridge
geometry were merely smooth: with disconnected fragments the harmonic solution
is barely better than copying the nearest observed orientation (0.5502 vs
0.6124), because a large hole with little boundary gives `grad v = 0` nothing
to extrapolate from and the solution diffuses toward a local mean.

So fingerprint ridge geometry is **not** "smooth continuation". That is
precisely the argument for the third term the proposal included and this test
omitted: the core/delta singularity structure is what smoothness cannot
supply. It is also why that term is expensive rather than cheap -- detecting
singularities from the observed region is itself unsolved when the singularity
falls *inside* the hole, which is the hard case.

**Caveat on the comparison.** The trained arm sees real observed pixels and
carries a learned fingerprint prior; the harmonic arm has no fingerprint prior
at all. This is not "learning beats mathematics" -- it is "smoothness alone is
an insufficient prior", which is a narrower and better-supported claim.

### 26. Advanced geometric completion: classical singularity models vs. smoothness

Section 25 tested only the smoothness term and found it far weaker than a
trained model. This section implements the rest of the proposal properly
(`src/fingerprint_reconstruction/metrics/orientation_geometry.py`): a thin-plate
prior that continues the boundary *slope* rather than flattening it, a global
low-order polynomial surface, the classical zero-pole singularity model
(`theta0 + 1/2 [sum arg(z - core) - sum arg(z - delta)]`) with the singularity
positions **fitted** rather than detected, and the combination of that global
model with harmonic completion of its residual.

Fitting rather than detecting matters: the interesting case is exactly the one
where the core falls *inside* the hole and therefore cannot be detected at all.
A synthetic check confirms the implementation recovers a core hidden entirely
within the masked region to within a pixel (axial error 0.0010), and caught a
real degeneracy on the way: a `sin(2*delta)` residual vanishes both at
alignment and at a 90-degree offset, so the fit converges happily to the
perpendicular field. The residual is now the difference of doubled-angle unit
vectors, which is zero only at true alignment.

| Arm | Mean | Median | Best 10% |
|---|---:|---:|---:|
| Nearest observed (floor) | 0.5463 | 0.5449 | 0.3256 |
| **Harmonic (smoothness)** | **0.4351** | **0.3967** | 0.2070 |
| Thin plate, damped | 0.4607 | -- | -- |
| Polynomial, degree 5 | 0.4709 | -- | -- |
| Zero-pole, model-selected | 0.5387 | 0.5230 | **0.1892** |
| Zero-pole + harmonic residual | 0.5021 | -- | -- |
| Trained model | **0.0512** | **0.0408** | 0.0148 |

(100 SOCOFing validation images, 4 mask geometries, axial orientation error in
the hidden region. Source: `outputs/socofing_orientation_geometry_v2.json`.)

**Reading: more geometric sophistication did not help on average, and the
failure mode is the interesting part.** Every "advanced" prior is worse in the
mean than plain smoothness. But the zero-pole model is not uniformly worse --
it is **bimodal**. It lands under 0.10, near the trained model's accuracy, on
4% of images and actually beats the trained model on 2%; it exceeds 0.60,
worse than doing nothing, on 39%. Harmonic completion is never excellent and
never catastrophic (p25 0.28, p75 0.58); the zero-pole model is both.

The split is structured, not random. On `central_missing` holes -- the case
where a singularity most often lies inside the hole -- zero-pole reaches
near-neural accuracy on 16% of images and fails catastrophically on 56%. The
classical model is therefore *right in kind*: when it identifies the hidden
singularity it extrapolates through it in a way no smoothness prior can. It is
simply unreliable at identifying it from a partial observation, which is the
hard part flagged in section 25 and is not solved here.

**A correction worth recording.** An intermediate run on 12 images showed
zero-pole beating harmonic by 22% and was briefly reported that way. At n=100
the ordering reverses. Given the bimodality documented above, a 12-image sample
of this estimator is worthless, and the earlier claim was an artifact of sample
size rather than a real effect.

### 27. Trust-weighted supervision: how much of the fine-tuning damage is registration noise?

The critical finding says fine-tuning against `X_pseudo` degrades reconstruction
measured on real pixels. A natural objection is that the fine-tuning was simply
careless about registration quality: `X_pseudo` is wrong in places, the loss
treated every pixel of it as equally true, so of course it hurt. This section
tests that objection directly instead of arguing about it.

Per pixel, the latent's observed ridge orientation is compared with the
registered exemplar's. Where two impressions of the same finger agree on ridge
direction the registration is locally sound; where they are perpendicular the
target is registration noise. Agreement can only be *measured* where the latent
is observed, so it is harmonically extended into the rest of the image, and the
result is dropped into `RegisteredApproximateLoss`'s existing per-pixel
`geometric_confidence` slot (`scripts/precompute_nist302_pseudo_trust.py`).

No leakage: agreement is computed from `observed` (official quality >= 2), the
model's own input. The held-out quality == 1 evaluation pixels are never read.

**Measured trust is far from uniform:** mean 0.586 across 1500 samples, range
0.21 to 0.87. The two impressions agree on ridge direction only about 59% of
the time, and how much they agree varies strongly by image.

| Metric (held-out real pixels) | Fine-tuned | + trust weighting | Improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1438 | 0.1447 | -0.0009 (n.s.) | -0.29 | 0.121 |
| SSIM | 0.2048 | **0.2117** | +0.0069 | 0.86 | 2.3e-4 |
| Orientation error | 0.5166 | **0.4539** | **+0.0627** | **1.38** | **4.5e-8** |
| Ridge-frequency rel. MAE | 0.8952 | **0.8417** | +0.0535 | 0.63 | 2.6e-3 |

Three of four metrics improve significantly, one is unchanged, and the
orientation effect is large.

**But the conclusion does not flip.** Against the zero-shot checkpoint, which
never saw SD302 at all, trust-weighted fine-tuning is still worse on every
metric: MAE 0.0975 vs 0.1447 (dz -2.11), SSIM 0.4869 vs 0.2117 (dz -3.25),
orientation 0.3467 vs 0.4539 (dz -1.31), ridge frequency 0.6117 vs 0.8417
(dz -1.19), all Holm p < 1e-6.

**Reading: the objection is real but explains only a third of the damage.**
Un-weighted fine-tuning moves orientation error 0.3467 -> 0.5166, a degradation
of 0.170. Weighting the supervision by measured inter-impression agreement
recovers 0.063 of that, about 37%. The remaining two thirds are not
registration noise. The most likely account, consistent with section 15 and
section 23, is that the model is learning the *texture of that particular
impression* rather than the finger's ridge structure -- an error that is
present exactly where the two impressions agree, and therefore invisible to
this weighting.

This strengthens the critical finding rather than softening it: the obvious
mechanical explanation has now been measured and priced, and most of the effect
survives it.

(Sources: `outputs/trust_weighted_heldout.json`,
`outputs/trust_weighted_vs_zeroshot.json`; 287 validation images, 29 subjects.)

### 28. Conformal certification of structural fidelity

Every uncertainty statement in this project so far is a correlation: predictive
spread correlates with error at rho = 0.28, geometric disagreement at rho =
0.67. A correlation cannot tell an examiner anything about a specific print.
This section produces a statement that can: a distribution-free bound with a
validated coverage guarantee.

The 2025 conformal-prediction literature for imaging inverse problems states
its own open problem plainly -- pixel-wise intervals are of unclear value,
because what matters when recovering an image is many-pixel structure. This
project is well placed to answer that, having shown that pixel metrics improve
while ridge structure degrades, so the certified quantity here is structural:
the axial orientation error of the reconstructed ridge field, aggregated over
16x16 regions.

Split conformal, with subjects partitioned into disjoint fit / calibration /
test thirds: a reliability regressor is fitted on ground-truth-free signals,
nonconformity scores `true - predicted` are collected on calibration subjects,
and the (1-alpha) quantile is applied to unseen test subjects.

**Coverage holds, and holds per subject** (zero-shot arm):

| Target | Quantile | Pooled coverage | Per-subject mean | min | max |
|---:|---:|---:|---:|---:|---:|
| 80% | 0.328 | 0.801 | 0.810 | 0.747 | 0.881 |
| 90% | 0.570 | 0.900 | 0.909 | 0.837 | 0.970 |
| 95% | 0.849 | 0.956 | 0.962 | 0.905 | 1.000 |

**What can actually be certified, at 90% coverage:**

| Tolerance | Equivalent angle | Regions certified | True error inside | outside |
|---:|---:|---:|---:|---:|
| 0.13 | 15 deg | 0.0% | -- | 0.517 |
| 0.29 | 22 deg | 0.0% | -- | 0.517 |
| 0.50 | 30 deg | 0.0% | -- | 0.517 |
| 0.75 | 38 deg | 5.1% | 0.134 | 0.537 |
| 1.00 | 45 deg | 41.3% | 0.316 | 0.658 |

**Reading: the machinery is sound and the bottleneck is the reconstruction.**
Empirical coverage tracks the target to the third decimal and does not collapse
on individual subjects, so the guarantee is real rather than an artifact of
pooling. But at any forensically tight tolerance -- under 30 degrees -- nothing
can be certified at all, because the reconstruction is simply not that accurate
anywhere, reliably. At 45 degrees, 41% of regions can be certified and their
true error is half that of the uncertified remainder.

Two honest limits. Per-pixel certification was tried first and was useless: the
90% quantile reached 0.979 on a scale whose maximum is 2, certifying nothing.
Region aggregation halved it, which is why the literature's own objection to
pixel-wise intervals is well taken. And conformal coverage assumes
exchangeability, which pixels inside one print badly violate; per-subject
coverage is reported for exactly that reason and is the number to trust.

(Sources: `outputs/conformal_structural.json`,
`outputs/conformal_structural_regions.json`,
`outputs/conformal_structural_zeroshot.json`.)

### 29. Conditioning the reconstructor on a solved orientation field

Section 18 conditioned reconstruction on a *predicted* ridge structure from a
learned network. This conditions it on a **solved** one: the orientation field
is extracted from the observed pixels and completed harmonically in
doubled-angle space by a sparse linear solve, with no learning anywhere in the
geometry branch. Three auxiliary channels -- cos 2t, sin 2t, and a channel
marking observed versus extrapolated geometry -- are injected at every encoder
scale through zero-initialised projections. The first-layer kernel is widened
with zeroed channels, so the model is numerically identical to its
unconditioned baseline at step zero and any later difference is attributable to
the geometry.

**The first attempt was invalid, and the diagnosis matters more than the
result.** Trained at the backbone's fine-tuning rate, the auxiliary projections
finished at mean|w| = 0.00104 against 0.0722 for the observed-pixel channels:
the model used the geometry at 1.4% of the weight given to its real input, i.e.
it ignored it. Parameters born at exactly zero cannot grow at a rate chosen for
a pretrained backbone. Any result read off that run would have been training
noise.

Re-run with a separate learning rate for the conditioning branch (50x), the
projections reached mean|w| = 0.036, **49.8%** of the input channels. The model
genuinely used the geometry, so the hypothesis could finally be tested.

| Metric (held-out real pixels) | Baseline | + solved geometry | Improvement | Cohen dz | Holm p |
|---|---:|---:|---:|---:|---:|
| MAE | 0.1438 | 0.1444 | -0.0007 | -0.13 | 0.55 |
| SSIM | 0.2048 | **0.2145** | +0.0097 | 0.72 | 9.7e-4 |
| Orientation error | 0.5166 | 0.5373 | -0.0207 | -0.36 | 0.092 |
| Ridge-frequency rel. MAE | -- | -- | +0.0269 | 0.24 | 0.29 |

**Reading: a clean negative result.** Handing the network a solved orientation
field does not make its orientation output more accurate; the point estimate
moves the wrong way and does not reach significance. Only SSIM improves.

In hindsight this should have been predicted. The harmonic completion is a
deterministic function of the observed pixels, and the network already receives
those pixels. No information was added, only a re-representation of information
already present. Conditioning of this kind can only help as an inductive bias
that the network cannot easily learn on its own, and here it evidently is not
one. That distinguishes it from section 18's *predicted* structure, which came
from a separately trained network with its own parameters.

(Sources: `outputs/geometry_conditioned_heldout.json`,
`outputs/geometry_conditioned_v2_heldout.json`; 287 validation images,
29 subjects.)

### 30. Mirror symmetry as a completion prior

Proposed idea: fill the missing orientation field from the mirror image of the
print, and where the mirrored source is itself missing -- the intersection of
the hole with its own reflection -- fall back to another method.

Implemented as a cascade (`complete_mirror_cascade`). The reflection axis is
not assumed: it is **fitted** to the observed field, by searching the vertical
axis whose reflection best explains the observed orientations. Reflection maps
a direction to `pi - theta`, so a genuinely symmetric field satisfies
`theta(mirror(p)) = pi - theta(p)`.

A synthetic check confirms the machinery: on a field that is reflection
symmetric by construction the axis is recovered exactly with disagreement
0.0000, and a tilted field that no reflection can explain scores 1.2172. The
first version of that check was itself wrong -- it built the field with
`|x - b|`, giving `theta(mirror) = theta` rather than `pi - theta`, and
therefore reported the symmetric case as the worse one. The implementation was
correct and the test was not.

| Condition | Nearest (floor) | Harmonic | Zero-pole | **Mirror cascade** | Trained model |
|---|---:|---:|---:|---:|---:|
| random_rectangles | 0.5007 | **0.3521** | 0.5282 | 0.5442 | 0.0595 |
| central_missing | 0.5426 | **0.4269** | 0.5983 | 0.6285 | 0.0578 |
| irregular | 0.5296 | **0.4113** | 0.4858 | 0.5607 | 0.0267 |
| disconnected_fragments | 0.6124 | **0.5502** | 0.5424 | 0.5516 | 0.0609 |
| **All** | 0.5463 | **0.4351** | 0.5387 | 0.5713 | 0.0512 |

Paired over the same 100 images: the mirror cascade beats harmonic completion
on only 28% of images (Wilcoxon p = 4.0e-7, i.e. significantly worse), and
beats the nearest-observed floor on 55% (p = 0.86, indistinguishable from it).

**Reading: falsified on every mask geometry.** Mirror completion is the weakest
geometric prior tested, below even the global zero-pole model, and on
rectangular and irregular holes it is worse than copying the nearest observed
orientation.

Two things are worth separating. The intersection problem the proposal
anticipated is real: for a hole centred on the symmetry axis -- which is what
`central_missing` is, because a central hole covers the core and the core sits
on the axis -- every mirrored source pixel falls inside the hole too, so
reflection has nothing to copy and the cascade degenerates to harmonic. But
that is the *secondary* issue. The primary one is that it also failed on
off-axis holes, where reflection had plenty to copy. A real fingerprint's
orientation field is simply not reflection-symmetric to a useful degree; the
visual quasi-symmetry around a loop core does not survive measurement.

**A prediction recorded and falsified.** Before running on real data this
analysis predicted no benefit on `central_missing` and a possible benefit on
off-centre masks. The first half held, the second did not.

**Not measured:** the symmetry residual from the axis fit is a
ground-truth-free quantity and might still be useful as a *reliability*
feature even though it is useless as a completion prior. A column bug meant it
was not written to the per-image output, so this remains untested.

(Source: `outputs/socofing_orientation_mirror.json`, 100 SOCOFing validation
images, 4 mask geometries.)

### 31. Why zero-shot wins: the model learns the impression, not the finger

The critical finding records *that* fine-tuning against `X_pseudo` degrades
reconstruction on real pixels. Every section since has treated that as given.
This one asks the question nobody asked: **why is a model trained only on clean
synthetic prints better on real forensic latents than one adapted to them?**

The standing hypothesis was that fine-tuning teaches the texture of that
particular impression. Testing it needs the right region. Where the latent and
the registered exemplar agree, following either one looks identical, so those
pixels cannot discriminate. The test therefore isolates the pixels where the
two impressions **conflict** (axial distance > 1.0) and asks, for each, whether
the model's orientation lands closer to the latent or to `X_pseudo`.

Zero-shot never saw `X_pseudo` in training, so its rate is what
not-following looks like.

| Quantity (held-out quality==1 pixels) | Zero-shot | Fine-tuned | Cohen dz | p |
|---|---:|---:|---:|---:|
| Follows `X_pseudo` in conflicting pixels | 0.243 | **0.432** | -1.88 | 3.7e-9 |
| Orientation error, conflicting pixels | 0.497 | **0.780** | -1.75 | 3.7e-9 |
| Orientation error, agreeing pixels | 0.342 | 0.390 | -0.48 | 3.3e-2 |

(282 validation images, 29 subjects, aggregated per subject.)

**Reading: the hypothesis is confirmed, and the damage is localised.** Where
the two impressions conflict, the fine-tuned model sides with the training
impression 43% of the time against zero-shot's 24% -- a 78% relative increase
with a large effect size. And the harm tracks that: in conflicting pixels the
fine-tuned model's error is 57% higher than zero-shot's (dz -1.75), while in
agreeing pixels the gap is small (dz -0.48).

This upgrades the central finding from an observation to a mechanism. Training
against a different impression of the same finger does not degrade
reconstruction diffusely; it teaches the network to reproduce *that
impression's* ridge placement, and the cost appears precisely where that
impression differs from the one being reconstructed.

It also explains section 27's ceiling. Weighting supervision by measured
inter-impression agreement recovered 37% of the damage and no more, because a
per-pixel weighting can only address harm that is localised to conflicting
pixels. The dz -0.48 residual in agreeing pixels is damage that no such
weighting can reach: the fine-tuned model is somewhat worse even where the
target was trustworthy, which points at a global shift in what the network
represents rather than a pixel-level supervision error.

(Source: `outputs/pseudo_target_following.json`,
`scripts/diagnose_pseudo_target_following.py`.)

### 32. Replacing the target instead of reweighting it: training on Track E

Section 17 identified this and left it open: Track E supplies an **exact**
missing-region target inside the SD302 image domain, by degrading a clean SD302
exemplar and then masking it. Nothing approximate enters the loss. Section 27
reweighted the pseudo-target by measured trust and recovered 37% of the
fine-tuning damage; section 31 explained why that was a ceiling, since a
per-pixel weighting can only reach harm localised to conflicting pixels. The
prediction that follows is that *replacing* the target should beat
*reweighting* it. This tests that.

Same SOCOFing initialisation as every other SD302 arm, plain
`MaskedReconstructionLoss` (no geometric confidence, no evaluation ROI, no
support weighting -- all of that machinery exists to cope with a target that
cannot be trusted pixel-wise, and here it can), evaluated on the officially
held-out `quality == 1` pixels.

| Arm (held-out real pixels) | MAE | SSIM | Orientation |
|---|---:|---:|---:|
| **Zero-shot SOCOFing** | **0.0975** | **0.4869** | **0.3467** |
| Track E, exact synthetic target | 0.1134 | 0.3468 | 0.4021 |
| Fine-tuned on `X_pseudo` | 0.1438 | 0.2048 | 0.5166 |

Paired, per subject, Track E against pseudo-target fine-tuning: MAE +0.0304
(dz 1.46, p = 3.7e-7), SSIM +0.1420 (dz 2.87, p = 1.5e-8), orientation +0.1145
(dz 0.92, p = 2.8e-5), ridge frequency -0.0134 (n.s.). Against zero-shot,
Track E is worse on all four (dz -0.49 to -1.46, all Holm-significant).

**Reading: the prediction held, and the damage decomposes.** Swapping the
approximate target for an exact one recovers **66% of the MAE damage, 50% of
the SSIM damage and 67% of the orientation damage**, against 37% for
reweighting. Replacement beats reweighting by roughly a factor of two, exactly
as section 31's mechanism implies it should.

**But domain adaptation still loses to not adapting at all.** Roughly a third
of the gap survives an exact target, which rules out "the pseudo-target is
wrong" as a complete explanation. Two candidates remain and this experiment
cannot separate them: the simulated degradation may not resemble real latent
degradation closely enough to transfer, or adaptation to this domain is simply
harmful at this data volume and resolution. Deciding between them needs a real
degraded-and-clean pair, which SD302 does not provide.

Taken with sections 27 and 31, the central finding now has a quantified
anatomy rather than a single number: about two thirds of the harm comes from
the target being a different impression, of which roughly half is localised to
pixels where the impressions conflict and is reachable by reweighting; the
remaining third is not a target problem at all.

(Sources: `outputs/track_e_vs_zeroshot.json`,
`outputs/track_e_vs_pseudo.json`, `scripts/train_nist302_track_e.py`;
287 validation images, 29 subjects.)

**Final, not provisional.** The 20-epoch run completed; validation reached its
minimum at epoch 10 (0.152018) and never beat it again, so the saved
best-checkpoint is that epoch and the numbers above are the final ones. They
were re-run against it and are unchanged.

## Still pending
- ~~RePaint distance-diagnostic run~~ — **done, see section 9.**
- ~~Pixel diffusion vs latent diffusion on NIST302~~ — **done, see section
  14.** Initially deprioritized as too expensive, then attempted anyway per
  explicit instruction to work through every item on the external-review
  plan; result is a clean negative finding (latent diffusion is worse on
  every significant fidelity metric).
- ~~Ridge-topology / "painted stripe" artifact check~~ — **done, see section 15.**
- ~~Non-rigid (TPS) vs. affine registration~~ — **done, see section 16.**
- ~~Two-stage (structure-guided) latent diffusion~~ — **done, see section
  18.** A real, mixed result: significantly better ridge-frequency accuracy,
  significantly worse SSIM and interval coverage, small/non-significant
  differences elsewhere. Does not change section 14's overall conclusion
  (latent diffusion remains dominated by pixel-space DDPM).
- **Native-resolution patches** — not implemented; a full quantified
  feasibility analysis was done instead
  (`docs/nist302_native_resolution_feasibility.md`), finding the current
  128px whole-image resize compresses the ridge period to ~6.45px (a ~7.3x
  average downsampling from native resolution), a plausible concrete
  explanation for section 15's ridge-topology artifacts. Scoped as its own
  multi-day phase, not attempted this session.
- A genuine remaining engineering gap surfaced by section 13: the DDPM
  checkpoints used throughout this document (`outputs/nist302_registered_ddpm_full`,
  and by extension the residual/latent arms built on the same pipeline
  pattern) were all selected by validation noise loss, which section 13
  showed picks a measurably worse checkpoint than a structural composite
  would. None of this project's DDPM-family results were retroactively
  re-run against composite-selected checkpoints — a real, acknowledged gap,
  not a silently dropped one.
- Final write-up sections: literature review update (offline-only, see
  `docs/literature_context.md`'s own caveat) is the main one still open;
  limitations and the "reconstruction vs hallucination" chapter are written
  (`docs/limitations.md`, `docs/reconstruction_vs_hallucination.md`).
