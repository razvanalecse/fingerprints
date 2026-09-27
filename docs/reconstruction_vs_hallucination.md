# Reconstruction vs. hallucination: the central finding

Date: 2026-09-23. This chapter compiles findings that are scattered across
`docs/nist302_ablation_master_table.md` (its top critical-finding box and
sections 5-9) into one standalone narrative, since it is the single most
important thread running through the whole NIST SD302 extension of this
project. Every number below is sourced from that document or the JSON files
it cites; nothing new is computed here.

## The question

A model shown a partial, degraded latent fingerprint and asked to fill in the
missing structure faces a fundamental ambiguity: for any given observed
partial print, there are many plausible completions consistent with generic
ridge-flow physics, and only one of them is the print that was actually
there. A model that is confidently wrong is more dangerous, in a forensic
context, than one that visibly admits uncertainty. This project therefore
tracks two axes for every probabilistic model, not one:

1. **Fidelity** — how close is the reconstruction (single sample, or the
   predictive mean over K samples) to the real print?
2. **Calibrated, localized uncertainty** — does the model's own predictive
   spread (across K samples) actually track where it is likely to be wrong?

A model that is good on axis 1 alone is not trustworthy by itself: it can be
smoothly, confidently wrong exactly where the print is most ambiguous, which
is indistinguishable from hallucination unless axis 2 is also checked.

## Finding 1: measured "improvement" can be an artifact of the target, not the model

Before comparing models to each other, this project found that the *choice of
evaluation target* can flip a headline conclusion outright. Fine-tuning the
deterministic model on SD302 (support+spectrum losses, calibrated confidence
weighting) improves every metric when evaluated against `X_pseudo` — the
approximately-registered exemplar from a *different impression* of the same
finger:

| Metric | Zero-shot vs fine-tuned, against `X_pseudo` |
|---|---|
| MAE | fine-tuning improves |
| Orientation error | fine-tuning improves |

But re-evaluated against real, held-out `quality==1` latent pixels — genuine
pixels that were never used in any loss or target construction — the
*identical two checkpoints* show fine-tuning is **worse on all four metrics**:

| Metric (held-out real pixels) | Zero-shot | Fine-tuned | Change | Holm p |
|---|---:|---:|---:|---:|
| MAE | 0.1068 | 0.1464 | +0.040 (worse) | 1.9e-8 |
| Orientation error | — | — | +0.058 (worse) | 3.7e-7 |
| Ridge-frequency rel. MAE | — | — | +0.202 (worse) | 5.5e-5 |
| SSIM | — | — | +0.222 (worse) | 1.9e-8 |

**Interpretation:** fine-tuning against `X_pseudo` teaches the model to
reproduce a different impression's specific texture — plausible-looking,
locally consistent, and wrong. This is hallucination in its purest form: the
model becomes *more* confident and *more* fluent while moving further from
the truth, and a target-consistency metric alone cannot detect it. Only
comparison against pixels that were genuinely never touched by training
reveals the regression. Every subsequent comparison in this project (support
conditioning, ridge-spectrum loss, confidence weighting, RePaint vs. DDIM,
residual vs. pixel DDPM, CVAE vs. DDIM) was deliberately run against the
held-out real-pixel target specifically because of this finding, and is not
subject to the same caveat.

## Finding 2: no single probabilistic model wins on both axes

Across the four probabilistic architectures evaluated on NIST302 (CVAE
spatial, DDPM/DDIM, DDPM/RePaint, residual-space DDPM), fidelity and
uncertainty calibration trade off against each other, not together:

| Model | K | MAE ↓ | SSIM ↑ | Orientation ↓ | Uncertainty-error ρ ↑ |
|---|---:|---:|---:|---:|---:|
| CVAE, spatial, bilinear | 50 | 0.138 | 0.400 | 0.294 | negative |
| DDPM, DDIM-20 | 10 | 0.155 | 0.305 | 0.302 | **0.273** |
| DDPM, RePaint (resampling=2) | 5 | **0.111** | **0.432** | 0.292 | 0.147 |
| DDPM, residual (`X_coarse` + diffused R) | 10 | 0.121 | 0.389 | **0.291** | 0.060 |

- **Best fidelity:** RePaint > Residual DDPM > CVAE > DDIM.
- **Best uncertainty localization:** DDIM ≫ RePaint > Residual DDPM ≈ CVAE
  (negative).

A K-matched, statistically powered head-to-head (CVAE vs. DDIM at K=10,
subject-level paired Wilcoxon, Holm-corrected) makes the tradeoff precise
rather than merely descriptive: CVAE is significantly *better* on every
fidelity metric (MAE −0.017, SSIM −0.095, orientation −0.010, all Holm
p < 0.05) and significantly *worse* on uncertainty (+0.434 Spearman
improvement for DDIM — the single largest effect size measured anywhere in
this project's NIST302 work, p = 2.2e-8).

RePaint's fidelity win is real but not free: its ancestral sampler with
per-step resampling refinements costs roughly 35-60x more per sample than
DDIM-20 (`docs/compute_cost_table.md`), which is also why it was evaluated at
K=5 rather than the K=10-50 used for the other arms — any claim that RePaint
"wins" needs this latency cost stated alongside it, not as a footnote.

A fifth arm, latent-space diffusion, was added later (section 14 of the
master table) and does not change this picture — it simply loses on both
axes. K-matched against pixel DDIM at K=10, it is significantly worse on
every fidelity metric that reached significance (MAE, SSIM, best-of-K,
single-sample, ridge-frequency; several large effect sizes, e.g. single-
sample MAE dz=−3.96) and descriptively worse on uncertainty too (Spearman
0.145 vs. DDIM's 0.273). Compressing to a 32×32×4 latent before diffusing
loses fidelity at this resolution without buying anything back on the
uncertainty axis either — it is dominated by pixel-space DDPM on both
questions this chapter asks, not a genuine fifth point on the tradeoff
frontier.

## Finding 3: uncertainty quality itself degrades differently across architectures with distance from ground truth

Stratifying held-out pixels by distance to the nearest examiner-verified
correspondence point (the only pixels in SD302 with an explicit, geometric
reliability signal) shows the two mechanisms above are not the whole story —
*how* uncertainty degrades with distance from verified evidence differs by
architecture:

| Model | 0-2mm | 2-5mm | >5mm | Pattern |
|---|---:|---:|---:|---|
| DDPM, DDIM-20 | 0.527 | 0.262 | 0.167 | positive everywhere, weakens gracefully |
| DDPM, RePaint | 0.389 | 0.105 | 0.017 | positive everywhere, weakens to near-zero |
| CVAE, spatial, bilinear | 0.011 | −0.149 | −0.297 | **flips negative** far from evidence |

DDIM leads RePaint at every band, so DDIM is not just the best-calibrated
model on average (Finding 2), its uncertainty signal also degrades the
slowest with distance from verified evidence among the three architectures
checked this way (section 9 of `docs/nist302_ablation_master_table.md`).

This is a materially different failure mode from Finding 2's aggregate
tradeoff. It is not merely that the CVAE's uncertainty is weaker on average —
far from verified evidence, the CVAE tends to be systematically *more*
confident exactly where it is least reliable, i.e. it actively misleads a
downstream consumer of its predictive std in the region where that signal
would matter most (regions the model is extrapolating into, farthest from
anything actually observed or verified). DDIM's signal, by contrast, only
weakens — it never inverts. A deployment that used predictive std to route
uncertain regions to a human examiner would fail safely with DDIM (under-
flagging at long range) and fail dangerously with the CVAE (confidently
wrong flags at exactly the range that matters).

DDIM's graceful-weakening pattern was re-checked with three proper scoring
rules beyond Spearman (CRPS, interval score, sparsification error — section
12 of the master table), on a *second*, independent difficulty axis too
(distance to the nearest actually-*observed* pixel, not just to a verified
correspondence point). All three scores degrade monotonically in the same
direction on both axes (e.g. Spearman 0.508→0.091 near-to-far on the
mask-geometry axis), which rules out Spearman-only association being a
misleadingly optimistic single-number summary — the calibrated interval
quality and the practical value of discarding uncertain pixels both degrade
for real, not just their rank-correlation.

## Finding 4: the model does use the ridges it is shown, but this alone does not prove the missing region is recovered correctly

A natural, sharper version of the hallucination worry is: does the model
even use the specific ridges it is conditioned on, or does it just learn to
paint a generic, plausible-looking fingerprint texture into whatever shape
the mask has, regardless of identity? This was tested directly (section 11
of `docs/nist302_ablation_master_table.md`): the same missing region was
reconstructed twice with identical diffusion randomness, once conditioned on
the sample's own real ridges and once on a *different finger's* ridges
through the same mask shape, then both scored against the original held-out
pixels. Using the wrong finger's ridges made reconstruction significantly
worse (MAE +0.034, Cohen dz=−2.17, p=3.7e-9, 287 images) — the model is not
ignoring its conditioning input.

This closes off one specific failure mode but not the central one. It shows
the model is sensitive to *which* ridges it is shown; it does not show that
what it produces in the truly unobserved region is a *correct* continuation
of those ridges, only that it is not identity-blind. Combined with the
`quality==1` caveat (the held-out target is itself a faint, ambiguous trace,
not a fully absent region — see `docs/limitations.md`), the honest summary
is: this project has evidence the models are doing *something* real with
the observed evidence, not pure hallucination, but has not established that
what they produce for genuinely unobserved ridges is topologically correct.

## The honest conclusion

There is no single "best NIST302 reconstruction model" produced by this
project, and that absence is itself the finding, not a gap to be closed by
more tuning. The choice of model should depend on what the reconstruction
will be used for:

- If the goal is the best-looking single or mean reconstruction and
  uncertainty will not be consumed downstream, RePaint wins on fidelity,
  at a real and substantial inference-time cost.
- If the goal is to also flag *where* a reconstruction is unreliable —
  which is the only responsible way to present a hallucination-prone
  reconstruction in a forensic context — DDIM's uncertainty signal is the
  only one of the four architectures tested that degrades gracefully rather
  than inverting, even though its raw fidelity is the weakest of the four.
- The CVAE and residual-DDPM arms occupy a middle ground on fidelity but
  should not be used anywhere a predictive-std-based reliability signal is
  load-bearing, given Finding 3.

Any future work on this project should treat "make one model win on both
axes simultaneously" as the actual open research problem, not an assumed
default of using whichever model has the best mean-image score.
