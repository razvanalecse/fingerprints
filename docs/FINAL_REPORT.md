# Fingerprint reconstruction from partial/degraded observations: SOCOFing and NIST SD302

Date: 2026-09-23. This document is the top-level report tying together every
result and sub-document produced in this project. It is written to be read
start to finish; each section links to the detailed source document(s) for
full numbers, JSON artifacts, and reproduction commands. **The test split was
never loaded for any result cited below, on either dataset** — every number
here is a train/validation-only model-selection result, and a final,
one-shot test-split evaluation of the selected configuration(s) is the one
remaining step before any number here can be called a final result.

## 1. Problem and goal

Given a partial or degraded fingerprint image — a synthetic mask over an
otherwise clean print (SOCOFing), or a genuine forensic latent print with
real, uncontrolled degradation (NIST SD302) — reconstruct the missing ridge
structure. Two families of model are studied throughout:

- **Deterministic models**, which produce one best-guess reconstruction
  (classical interpolation, U-Net, gated convolution).
- **Probabilistic models**, which produce a distribution over plausible
  reconstructions (conditional VAE, denoising diffusion in several sampling
  regimes) so that predictive spread can, in principle, communicate where
  the reconstruction is unreliable.

The project asks two questions, not one: how good is the best-guess
reconstruction (fidelity), and — for the probabilistic models — does the
model's own expressed uncertainty actually track where it is wrong
(calibration)? Section 6 explains why both questions matter together.

## 2. Datasets and protocols

Full detail: `docs/cross_dataset_protocols.md`.

- **SOCOFing**: clean fingerprints with synthetic, controlled masking
  (observed fraction r ∈ {0.10, ..., 0.80}), subject-disjoint 900-image
  validation split. This is the project's original track (Protocol 1).
- **NIST SD302**: 200 subjects, real forensic latent prints paired with a
  different-impression exemplar, geometrically registered into the latent's
  coordinate frame via examiner-verified correspondence points
  (`registered_approximate` — explicitly not pixel-aligned ground truth).
  1,213 train / 287 validation registered pairs, evaluated three ways:
  - **Protocol 2 (zero-shot):** SOCOFing-trained checkpoints evaluated on
    SD302 with no SD302 training at all.
  - **Protocol 3 (pretrain → fine-tune → evaluate on unseen SD302):**
    SOCOFing-pretrained checkpoints fine-tuned on the 1,213 SD302 train pairs,
    evaluated on the 287 validation pairs those pairs were never fine-tuned
    on. **Complete only for the deterministic arm.** CVAE (spatial), pixel
    DDPM, and residual DDPM were all trained from a random initialization
    directly on SD302 — checked against each script's code, neither DDPM
    script has any checkpoint-warm-start mechanism at all. This corrects an
    earlier version of this report that described the DDPM arms as
    protocol-3 complete; see `docs/cross_dataset_protocols.md` for the
    per-script evidence.
  - **SD303** was never received; deferred entirely.

Two evaluation targets are used for SD302 throughout, and the difference
between them is the project's most important single finding (Section 5):
the approximate registered exemplar `X_pseudo`, and genuine held-out
`quality==1` latent pixels that were never used in any loss or target
construction.

## 2.5. Reporting tracks — what each result is actually claiming

**Every quantitative result in this report belongs to exactly one of four
named tracks, and no result from tracks B or C is to be described as "true
missing fingerprint recovery."** This section exists because that distinction
is easy to blur across a report this size, and blurring it is the single
most misleading thing this project could do with its own numbers.

- **Track A — synthetic exact recovery (SOCOFing).** Clean prints, a
  synthetic mask, and genuine pixel-exact ground truth for the missing
  region. The only track in this project where "reconstruction of missing
  ridges" is a literally accurate description of what is being measured.
  Section 3.
- **Track B — registered-exemplar structural agreement (NIST302 vs.
  `X_pseudo`).** The target is a *different impression* of the same finger,
  affine-registered into the latent's frame — real structural similarity,
  not pixel ground truth, and demonstrably gameable (Section 5: fine-tuning
  improves Track B numbers while regressing on Track C). Any result computed
  against `X_pseudo` is Track B.
- **Track C — real-latent quality enhancement (NIST302 vs. held-out
  `quality==1` pixels).** The most rigorous target available — genuine,
  never-trained-on pixels — but `quality==1` means "debatable ridge flow"
  (a faint, ambiguous *existing* trace), not "no ridge information present."
  This track measures **enhancement of a weak trace**, not synthesis of a
  genuinely absent region; see `docs/limitations.md` for the full caveat.
  Sections 4, 6, and the ablation master table's sections 1-4 and 10-11 are
  Track C.
- **Track D — predictive uncertainty.** Whether a probabilistic model's own
  expressed spread tracks where it is actually wrong, evaluated against
  Track A's exact ground truth (SOCOFing) or Track C's held-out pixels
  (NIST302), stratified by difficulty (distance to verified evidence, mask
  geometry). Section 6 and the master table's sections 7-9.
- **Track E — synthetic degradation, exact ground truth, SD302 domain.**
  A single clean SD302 exemplar is degraded (elastic warp, blur, contrast
  reduction, noise) then masked, so the missing region's true value is
  known exactly by construction — unlike Track B/C, but in the SD302 image
  domain rather than SOCOFing's clean synthetic prints (Track A). Master
  table section 17.

Nowhere in this project — not Track B, not Track C — has recovery of ridges
with zero observable trace been measured **on a real forensic latent**.
Track A and Track E both have exact ground truth, but neither uses a real
latent capture: Track A is clean synthetic SOCOFing data, Track E is a
simulated degradation of a clean SD302 exemplar. Track E's main purpose in
this project turned out to be corroborating, not replacing, the central
finding: see section 5.

## 3. SOCOFing results — Track A (Protocol 1)

Full detail: `outputs/DETERMINISTIC_BASELINES.md`, `docs/socofing_reconstructibility_segmented.md`.

Deterministic baselines (900-image validation, missing-pixel MAE / PSNR /
local SSIM / orientation error, all vs. nearest-observed interpolation as the
non-learned floor):

| Model | Parameters | MAE ↓ | PSNR (dB) ↑ | local SSIM ↑ | orientation error ↓ |
|---|---:|---:|---:|---:|---:|
| Nearest-observed interpolation | 0 | 0.277 | 9.04 | 0.094 | 0.815 |
| U-Net | 7,762,753 | 0.176 | 12.97 | 0.253 | 0.241 |
| Gated convolution | 5,679,009 | 0.174 | 13.02 | 0.259 | 0.229 |

Gated convolution beats U-Net on every metric, paired and Holm-significant
(900 cases), though the margins are small (Cohen dz 0.13-0.31) —
statistically detectable, not a qualitative leap.

Adding fingerprint-structural losses (orientation, ridge-band, ridge-spectrum)
on top of gated convolution (`gated_ridge_hybrid_full` vs. the prior
structural-loss variant `v3_loss`) improved all four metrics simultaneously,
with the largest effect on SSIM (Cohen dz 0.76, mean improvement +0.014,
Wilcoxon Holm p = 2.8e-90) — the clearest single result in the SOCOFing
track, and the basis for carrying structural losses forward into the NIST302
work.

The full probabilistic family (CVAE, DDPM, DDIM, RePaint, latent diffusion,
residual latent diffusion) was also implemented and evaluated on SOCOFing;
see `docs/cross_dataset_protocols.md` Protocol 1 for the output directories.

**Reconstructibility vs. observed fraction** (`docs/socofing_reconstructibility_segmented.md`,
DDIM-50/K=10): orientation error, uncertainty magnitude, and sample diversity
all show a statistically preferred bend (piecewise-linear over smooth linear,
nested F-test) somewhere in the r ≈ 0.25-0.45 range — reconstruction quality
does not improve at a constant rate as more of the print becomes visible.
SSIM does not show a statistically supported bend and improves smoothly
across the whole range instead. The breakpoint is reported as a curve-fit
statistic, not asserted as a mechanistic threshold.

## 4. NIST SD302: zero-shot and fine-tuned deterministic results — Track C

Full detail: `docs/nist302_ablation_master_table.md` sections 1-4.

SOCOFing-trained checkpoints transfer to real forensic latents with no
SD302-specific training at all (Protocol 2, complete). Adding SD302 fine-
tuning, visible-support conditioning, ridge-spectrum loss, and geometric
confidence weighting were each tested as isolated ablations against the
held-out real-pixel target (not `X_pseudo`) — see the master table for full
numbers. Confidence-weighting's effect, once corrected from an initial
checkpoint-mismatch error (documented transparently in the table itself), is
statistically significant but practically small on every metric.

## 5. The central methodological finding: target choice can reverse a conclusion — Track B vs. Track C

Full detail: the critical-finding box at the top of
`docs/nist302_ablation_master_table.md`.

Evaluated against `X_pseudo` (the registered, different-impression exemplar),
SD302 fine-tuning **improves** MAE and orientation error over the zero-shot
SOCOFing checkpoint. Evaluated against real, held-out `quality==1` latent
pixels — genuinely never touched by any loss or target construction — the
same two checkpoints show fine-tuning is **worse on all four metrics tested**
(MAE +0.040, orientation +0.058, ridge-frequency relative MAE +0.202, SSIM
+0.222, all Holm p < 1e-4). A 6-model Friedman omnibus test over held-out MAE
confirms real differences exist among all models tested (χ²=116.27, df=5,
p=1.9e-23), and the subject-mean ranking puts zero-shot *first*, not last.

This is treated as the single most important finding in the whole NIST302
extension: **a measured improvement can be an artifact of the evaluation
target, not a genuine property of the model.** Fine-tuning against an
approximately-registered different-impression target teaches the model to
reproduce that impression's specific texture — locally plausible, and wrong.
Every subsequent NIST302 comparison in this project (support conditioning,
ridge-spectrum, confidence weighting, RePaint vs. DDIM, residual vs. pixel
DDPM, CVAE vs. DDIM, and the distance-stratified uncertainty diagnostics) was
deliberately run against the held-out real-pixel target specifically because
of this finding, and is not subject to the same caveat. Any earlier or future
result checked only against `X_pseudo` should be treated as unconfirmed
until re-evaluated this way.

**Independently reproduced a third way (Track E, master table section 17):**
the same two checkpoints were also compared on synthetically degraded clean
SD302 exemplars with *exact* missing-region ground truth (not approximated,
not limited to sparse held-out pixels). Fine-tuned is again significantly
worse on SSIM (Holm p=1.5e-8, one of the largest effect sizes in this
project, dz=−2.78), orientation error (p=0.0047), and ridge-frequency error
(p=0.0024); only MAE ties. Two structurally different ways of getting a real
target — real held-out pixels, and exact synthetic ground truth — now agree
with each other and disagree with `X_pseudo`, which is the strongest
evidence in the whole project for treating this as the headline finding.


### 5.1 Why zero-shot wins: the mechanism, not just the effect

Sections above establish *that* the pseudo-target reverses the conclusion.
Master-table section 31 establishes *why*, which had never been tested.

Where the latent and the registered exemplar agree on ridge direction, a model
following either one looks the same, so those pixels cannot discriminate. The
test therefore isolates pixels where the two impressions **conflict** and asks
whether the model's orientation lands closer to the latent or to `X_pseudo`.
Zero-shot never saw `X_pseudo`, so its rate is the not-following baseline.

| Held-out quality==1 pixels | Zero-shot | Fine-tuned | Cohen dz | p |
|---|---:|---:|---:|---:|
| Follows `X_pseudo` where the impressions conflict | 0.243 | **0.432** | -1.88 | 3.7e-9 |
| Orientation error, conflicting pixels | 0.497 | **0.780** | -1.75 | 3.7e-9 |
| Orientation error, agreeing pixels | 0.342 | 0.390 | -0.48 | 3.3e-2 |

Fine-tuning does not degrade reconstruction diffusely. It teaches the network
to reproduce *that impression's* ridge placement, and the cost appears
precisely where that impression differs from the one being reconstructed: a 78%
relative increase in pseudo-following, with error 57% higher in exactly those
pixels and a much smaller gap elsewhere.

This also bounds what any repair can achieve. Weighting the supervision by
measured inter-impression agreement (section 27) recovers 37% of the
orientation damage and no more, because a per-pixel weighting can only reach
harm that is localised to conflicting pixels. The residual in agreeing pixels
is a global shift in what the network represents, not a pixel-level
supervision error.

## 5.2 Knowing when not to trust a reconstruction

Every uncertainty statement elsewhere in this project is a correlation. Three
results, added late, make the question operational. All are measured on
officially held-out `quality == 1` pixels and none requires a target at
inference.

**A geometric second opinion predicts the model's own error.** The
disagreement between the model's orientation field and a harmonic completion of
the observed field predicts that model's true error at Spearman rho = 0.67 per
image and 0.71 per subject, against 0.276 for the best predictive-uncertainty
signal in the study. It is not circular: partialling out the geometric solver's
own accuracy still leaves rho = 0.635 (p = 1.9e-33).

**Abstention converts the signal into error reduction.** Answering for only
60% of the missing region cuts orientation error by 27.5% on that signal alone
and 30.1% with a cross-validated multi-signal predictor. Random abstention is
flat, so the gain is selection rather than arithmetic. The oracle would cut
77%, and that gap is reported rather than hidden.

**Split conformal turns it into a guarantee.** With subjects partitioned into
disjoint fit/calibration/test thirds, empirical coverage matches the target to
the third decimal and holds per subject (0.837-0.970 at a 90% target). The
limit is the reconstruction, not the method: below a 30-degree tolerance
nothing can be certified at all, while at 45 degrees 41% of regions can be,
with half the error of the uncertified remainder.

**What did not work, measured the same way:** per-image selection between
geometric methods (oracle would gain 14%, no ground-truth-free signal captures
it); the global zero-pole singularity model (bimodal, catastrophic on 39% of
images); mirror-symmetry completion (worse than the non-learned floor on every
mask geometry); conditioning the reconstructor on a solved orientation field
(no information added -- the field is a deterministic function of input the
network already has); and raising the working resolution to 256px under a
matched budget (every structural metric worse).

## 6. Reconstruction vs. hallucination: the probabilistic model family — Track C and Track D

Full chapter: `docs/reconstruction_vs_hallucination.md` (compiles this
section's claims with full tables and JSON sources).

No probabilistic architecture tested on NIST302 wins on both fidelity and
calibrated uncertainty simultaneously:

| Model | K | MAE ↓ | SSIM ↑ | Orientation ↓ | Uncertainty-error ρ ↑ |
|---|---:|---:|---:|---:|---:|
| CVAE, spatial, bilinear | 50 | 0.138 | 0.400 | 0.294 | negative |
| DDPM, DDIM-20 (pixel-space) | 10 | 0.155 | 0.305 | 0.302 | **0.273** |
| DDPM, RePaint (resampling=2) | 5 | **0.111** | **0.432** | 0.292 | 0.147 |
| DDPM, residual (`X_coarse` + diffused R) | 10 | 0.121 | 0.389 | **0.291** | 0.060 |
| DDPM, latent-space (f=4 autoencoder) | 10 | 0.215 | 0.245 | 0.303 | 0.145 |

RePaint wins fidelity at ~35-60x DDIM's per-sample inference cost. DDIM has
the best-localized uncertainty by a wide, statistically confirmed margin over
CVAE (+0.434 Spearman, one of the largest single effects measured anywhere in
this project's NIST302 work). Stratifying by distance to the nearest verified
correspondence point sharpens this further: DDIM's and RePaint's uncertainty
signals both degrade gracefully with distance from evidence (DDIM:
0.527→0.262→0.167; RePaint: 0.389→0.105→0.017, DDIM leading at every band)
while the CVAE's **sign-flips** (0.011 → −0.149 → −0.297) — the CVAE becomes
systematically *overconfident* exactly where it is extrapolating furthest
from anything verified, a qualitatively worse failure mode than graceful
weakening. Model choice should therefore depend on whether predictive
uncertainty needs to be load-bearing downstream (e.g. flagging regions to a
human examiner), not solely on mean-image quality. Latent-space diffusion
(added later, initially deprioritized as too expensive, then run anyway) is
dominated on both axes — a clean negative result, not a competitive fifth
point: K-matched against pixel DDIM it is significantly worse on every
fidelity metric that reached significance (several large effect sizes) and
descriptively worse on uncertainty too, likely because compressing to a
32×32×4 latent discards ridge-period information that matters at this
resolution.

## 7. Limitations

Full detail: `docs/limitations.md` — data limitations (approximate registered
target, sparse held-out pixels, small subject count, no SD303), evaluation
limitations (sealed test split, validation-calibrated confidence weights and
Gabor bank, best-of-K vs. single-sample labeling, resolution-specific
frequency calibration per `docs/nist302_256px_frequency_check.md`), modeling
limitations (no model wins both axes, CVAE not protocol-3 pretrained, RePaint
latency, latent diffusion decisively worse than pixel diffusion, every DDPM
checkpoint selected by noise loss rather than the structural metrics this
project reports — confirmed to matter, section 13 of the master table), and
hardware/compute limitations (no VRAM measurement, variable GPU contention,
no `statsmodels`/mixed-effects modeling, offline environment).

## 8. Literature context

Full detail (offline, unverified — see caveat there): `docs/literature_context.md`.
This project's methods sit in the conditional-diffusion-inpainting,
latent-diffusion, conditional-VAE, and classical Gabor-based fingerprint
enhancement literatures; its most likely genuine contributions are the
target-choice reversal finding (Section 5) and the architecture-dependent,
distance-stratified uncertainty-degradation finding (Section 6), neither of
which could be checked against a live literature search in this environment.

**Update:** the user reported a closely related paper — Hussein, Jain &
Nandakumar, "Progressive Learning of a Diffusion-based Inpainting Model for
Separating Overlapped Fingerprints," reported as accepted at IJCB 2026 — not
independently verified here (no internet access), but if accurate, this
project cannot claim priority on diffusion-based fingerprint inpainting as a
method. `docs/literature_context.md` maps this project's actual results
(uncertainty calibration, target-choice reversal, ridge-topology artifacts,
Track E synthetic-to-real transfer) against that paper's reported scope,
finding this project's contributions sit largely outside it — but this
mapping is also unverified and should be re-checked once real search access
is available.

## 9. What remains before this can be called final

1. ~~**Test-split evaluation, once**~~ — **done for NIST302.** The sealed
   split was opened exactly once, for all eleven models simultaneously, after
   every model and hyperparameter was frozen; nothing was re-selected
   afterwards (`configs/experiment_registry_amendments.yaml` records the
   compliance trail). The SOCOFing test split remains unopened.
2. **Re-select DDPM checkpoints by the structural composite metric, not
   noise loss, and re-run the affected comparisons** (section 13 of the
   master table) — confirmed to change which checkpoint wins and by how
   much, not yet propagated to `outputs/nist302_registered_ddpm_full` or
   any comparison built on it.
3. ~~A real, online literature search~~ — **partly done.** Verified that the
   closest competing work (Hussein, Jain & Nandakumar, MSU, arXiv 2026) solves
   *overlapped fingerprint separation*, not partial reconstruction, and carries
   no uncertainty quantification and no discussion of training/evaluation
   target bias. Minutiae-graph representation and sex-from-fingerprint are both
   crowded; conformal prediction for imaging inverse problems is active and its
   stated open problem — that pixel-wise uncertainty is of unclear value
   because many-pixel structure is what matters — is what section 5.2's
   structural certification addresses. `docs/literature_context.md` still needs
   rewriting with these citations.
4. VRAM/peak-memory instrumentation, if the final report requires it
   (`docs/compute_cost_table.md`).
5. **Native-resolution patches** — the largest remaining item from the
   methodological audit's plan. Not implemented; a full quantified feasibility
   analysis was done instead (`docs/nist302_native_resolution_feasibility.md`),
   finding the current 128px whole-image resize compresses the ridge period
   to ~6.45px average (a ~7.3x downsampling from native resolution) — a
   concrete, plausible explanation for the "painted stripe" ridge-topology
   artifacts found in section 6 / master-table section 15. Scoped as its
   own multi-day phase.

   **The cheap version of this hypothesis has since been tested and failed.**
   Retraining the same recipe at 256px made every scale-free structural metric
   worse, not better: minutiae-density ratio 1.52 -> 3.14, coherence difference
   -0.01 -> +0.07, Markov entropy gap -0.34 -> -0.60 (master table section
   equivalent). That test confounds sampling with training budget and
   initialisation — the 256px arm inherited a 128px-trained checkpoint and the
   same epoch count on four times the pixels — so it does not falsify the
   sampling argument outright, but the argument now rests on native-resolution
   *patches* alone and should not be asserted without them.
6. Skeleton-continuity/minutiae controls, non-rigid registration, and a
   two-stage structure-guided model — all three were also proposed by the
   methodological audit alongside item 5, but (unlike native-resolution patches)
   were feasible to attempt this session by reusing existing infrastructure,
   and all three are now done: master-table section 15 (ridge-topology
   artifacts, confirmed on all three axes), section 16 (TPS vs. affine, a
   small non-significant improvement), and section 18 (structure-guided
   latent diffusion — a real mixed result: better ridge-frequency accuracy,
   worse SSIM and interval coverage, no change to section 14's overall
   conclusion).
7. Point 9 of the methodological audit (synthetic-to-real training with exact
   ground truth) is also done, ahead of schedule relative to this list's
   original ordering — see section 5 and master-table section 17. It
   produced the project's strongest evidence yet for the central finding:
   two structurally independent real-ground-truth methodologies (held-out
   real pixels, and exact synthetic ground truth) both show fine-tuning
   regresses reconstruction quality, while only the approximate `X_pseudo`
   target shows an improvement.
