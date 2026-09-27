# Limitations

Date: 2026-09-23. This document collects every methodological limitation
identified across the project rather than scattering them across individual
result docs, so a reader of the final report sees the full picture in one
place. It is organized by where the limitation enters the pipeline: data,
target/evaluation, training, and hardware/compute.

## 1. Data limitations

- **`registered_approximate` is not pixel-aligned ground truth.** SD302's
  pseudo-target is a *different impression* of the same finger, geometrically
  registered into the latent's coordinate frame via an affine transform fit
  to examiner-verified correspondence points. It shares structure with the
  true latent (same ridge topology, roughly the same pose) but not exact
  pixel intensities — two impressions of the same finger differ in pressure,
  skin distortion, and noise even after alignment. Every metric computed
  against `X_pseudo` inherits this approximation; see the critical finding in
  `docs/nist302_ablation_master_table.md` for direct evidence of how much
  this can matter (fine-tuning looks like an improvement against `X_pseudo`
  and a regression against real held-out pixels).
- **The manifest's own `affine_rmse_mm` understates true registration error
  by roughly 0.10mm** (section 16 of the master table): it is an in-sample
  residual, not cross-validated, and a leave-one-correspondence-out estimate
  is consistently higher (~0.30mm vs. ~0.19-0.20mm in-sample) on both train
  and validation splits. A non-rigid (thin-plate-spline) alternative was
  built and compared the same way — it improves leave-one-out RMSE only
  slightly and not significantly (Holm p=0.07-0.14), so affine is a
  reasonable choice given SD302's correspondence density, but any use of
  `affine_rmse_mm` elsewhere in this project (e.g. as a rejection criterion
  or confidence proxy) should be understood as an optimistic lower bound on
  true registration error, not a calibrated estimate of it.
- **Held-out quality==1 pixels are the most rigorous target used, but "held
  out" does not mean "genuinely absent."** `quality>=2` is used as observed
  input, `quality==1` as the held-out evaluation target — but quality==1
  means "debatable ridge flow" (a faint, ambiguous trace NIST's examiners
  could not confidently score), not "no ridge information present at all."
  Every held-out-pixel number in this project — the critical finding, the
  probabilistic-model comparisons, the distance-stratified uncertainty
  diagnostics, all of it — is therefore best read as measuring **enhancement
  of a weak existing trace**, not **synthesis of a truly missing region**.
  Fully obliterated latent regions have no ridge annotation at all and are
  not evaluated by any metric in this project. This is a real, load-bearing
  caveat on every "reconstruction" claim in this report, not just a
  footnote: nowhere in the NIST302 track has this project actually measured
  recovery of ridges with zero observable trace, only enhancement of ridges
  that were faintly, ambiguously present. Held-out quality==1 pixels are
  also concentrated near the edges of legible latent regions by
  construction, which is a second, separate bias (the evaluation is
  strongest in the hardest, most ambiguous part of the image — conservative,
  not necessarily representative of typical latent quality).
- **Only 200 subjects, 1,213 train / 287 validation registered pairs.** This
  is a small dataset by deep-learning standards, particularly for the
  probabilistic models (CVAE, DDPM variants) trained from a SOCOFing
  initialization or from scratch. Subject-level statistics (29 validation
  subjects) are used throughout specifically to avoid pseudo-replication from
  multiple images per subject, but 29 subjects still gives limited power for
  small effect sizes — several ablations in the master table report
  statistically significant but practically small effects for exactly this
  reason (Section 4, confidence-weighting).
- **SD303 was never received or used.** All "cross-dataset" claims in
  `docs/cross_dataset_protocols.md` are SOCOFing↔SD302 only; generalization
  to a third, independently-collected dataset is untested.

## 2. Target / evaluation limitations

- **The test split was never loaded for any result in this project.**
  Every number reported anywhere in this codebase (SOCOFing and NIST302) is
  train/validation only, by design, to keep the test split sealed for a
  final, one-shot evaluation. This means every comparison, ranking, and
  "best model" claim in the ablation tables is a *model-selection* result,
  not a *generalization* result — the final report must re-run the winning
  configuration(s) on the test split once, and only once, before claiming a
  final number.
- **Geometric confidence weights and the Gabor ridge-frequency bank are
  calibrated on the *validation* split** (frozen before touching test), not
  train, because the calibration source (`nist302_classical_128_validation`)
  needed already-registered pairs and a witness model. This is a smaller
  deviation from strict train/val/test separation than it first appears —
  the weights are frozen and never revised from outcomes — but it means
  validation is not purely held-out with respect to *every* modeling choice,
  only with respect to loss gradients and model selection.
- **Best-of-K is reported as a separate, clearly labeled estimand and never
  conflated with single-sample or predictive-mean performance** (per
  `docs/cvae_10_corrections_audit.md` point 5), but a reader skimming only
  headline numbers could still mistake a best-of-50 figure for a realistic
  single-shot reconstruction quality — this must be stated explicitly
  wherever K-sweep numbers are quoted in the final report.
- **The 256px ridge-frequency check (`docs/nist302_256px_frequency_check.md`)
  found the loss's Gabor calibration is resolution-specific** (does not
  scale the naively-expected way between 128px and 256px). Since every
  trained NIST302 model in this project is 128px, this is a documented,
  closed gap, not a live bug — but it means the structural losses as
  currently calibrated cannot be assumed to transfer to a higher-resolution
  model without recalibration.

## 3. Modeling / training limitations

- **No single probabilistic model wins on both fidelity and calibrated
  uncertainty.** This is reported as the central finding (see
  `docs/nist302_ablation_master_table.md` sections 7-9), not hidden as a
  limitation, but it does mean there is no model in this project ready to be
  called "the" NIST302 reconstruction model without specifying which axis
  (best mean-image quality: RePaint; best localized uncertainty: DDIM) the
  claim is about.
- **Only the deterministic arm is actually protocol-3 pretrained.** Checked
  directly against training-script code: `finetune_nist302_registered.py` is
  the only NIST302 training script with a working checkpoint warm-start
  (`config["model"]["initialize_from"]`, loading the SOCOFing gated model).
  The spatial CVAE, pixel DDPM, and residual DDPM were all trained from a
  random initialization directly on SD302 — neither DDPM script has any
  warm-start mechanism at all. An earlier version of `docs/cross_dataset_protocols.md`
  described the DDPM arms as protocol-3 complete; that was incorrect and has
  been corrected there. Every NIST302 probabilistic-model result in this
  report except the deterministic baseline should be read as "trained from
  scratch on SD302's 1,213 pairs," not "pretrained on SOCOFing."
- **RePaint's fidelity advantage has a real, unamortized latency cost**
  (~35-60x DDIM-20 per sample, `docs/compute_cost_table.md`), which is why it
  was evaluated at K=5 rather than K=10/20/50 like the other probabilistic
  arms — its numbers in the full-comparison table are not K-matched against
  the rest, only against DDIM at K=5 specifically (section 5, K-matched).
- **The CVAE's uncertainty is not just weaker than DDIM's but sign-flips with
  distance from verified evidence** (section 9 of the master table): pooled
  Spearman correlation goes from ~0 near anchors to strongly negative far
  from them. Practically, this means CVAE predictive std cannot safely be
  used to flag unreliable regions to a human examiner without further work —
  it can be actively misleading in the region an examiner would trust it
  most.
- **Isolated loss ablations (orientation-only, ridge-band-only) are complete**
  (section 10 of `docs/nist302_ablation_master_table.md`) and show the two
  losses are not interchangeable: ridge-band-only nearly matches or beats the
  full combined-loss model on 3 of 4 held-out metrics, while orientation-only
  is worse across the board, including on orientation error itself. The two
  runs were early-stopped at different epoch counts under an identical
  validation-loss criterion (13/18, 6/11, 9/14 for full/ridge-band-only/
  orientation-only respectively), which is a plausible partial confound for
  the loss-interaction interpretation and is not fully disentangled.
- **Every DDPM checkpoint in this project was selected by validation noise-
  prediction MSE, which this session confirmed is measurably not the same
  as selecting for the metrics actually reported.** A direct check
  (section 13 of the master table) re-evaluated every epoch of a pixel-DDPM
  retrain by a held-out structural composite (MAE + orientation error +
  ridge-frequency error) instead: it picked a different epoch than noise
  loss did, and that epoch was better on *all three* metrics by a large
  margin (MAE 0.114 vs. 0.155). `outputs/nist302_registered_ddpm_full/best-checkpoint.pt`
  — used throughout this project's other DDPM results — was never
  retroactively re-selected this way; every DDPM number in this report
  should be read with the caveat that a better checkpoint from the same
  training run plausibly exists and was not the one evaluated.
- **Pixel diffusion beats latent diffusion on NIST302, decisively** (section
  14 of the master table) — latent-space diffusion (f=4 autoencoder, 1.98M-
  parameter denoiser) is worse than pixel-space DDPM on every fidelity
  metric that reached significance (MAE, SSIM, best-of-K, single-sample,
  ridge-frequency), several by large effect sizes, and worse on uncertainty
  calibration too. The compression stage alone is excellent (PSNR 34dB); the
  loss happens in the conditional generative step, plausibly because a 4x
  spatial downsampling discards ridge-period information that matters at
  this resolution. This was initially deprioritized as too expensive, then
  run anyway; the negative result is now a documented finding, not a gap.

## 4. Hardware / compute limitations

- **No VRAM/peak-memory measurement exists for any run.** `torch.mps` does
  not expose CUDA-equivalent peak-allocation APIs, and no polling
  instrumentation was added this session; any memory figure in the final
  report requires new instrumentation, not a re-read of existing logs
  (`docs/compute_cost_table.md`).
- **All training/inference timings were measured under variable, often
  heavy GPU contention** (up to 6 concurrent jobs sharing one Apple Silicon
  GPU at points in this project), and are reported as such where the
  contention level is known; a few figures (e.g. the residual DDPM's
  training wall-clock) are explicitly flagged as needing a solo re-measurement
  before being cited as a clean per-model cost.
- **No `statsmodels` availability (offline environment).** The originally
  planned "Friedman or mixed-effects" statistical comparison used
  `scipy.stats.friedmanchisquare` as the feasible substitute; a proper
  subject-as-random-effect mixed model was not fit anywhere in this project.
  This is noted directly in the relevant JSON outputs' `note` fields, not
  only here.
- **No internet access in this environment.** The literature context in
  `docs/literature_context.md` is written from training-time knowledge, not
  verified against current citations or fetched sources; it should be
  treated as an orientation, not a citation-checked literature review.
