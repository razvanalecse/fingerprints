# Literature context (offline, unverified)

Date: 2026-09-23. **This is not a citation-checked literature review.** This
environment has no internet access, so nothing below was looked up, fetched,
or verified against a current source; it is written from general training-time
knowledge of the field to give the final report orientation on where this
project's methods and findings sit relative to prior work. Before submission,
every claim here should be checked against an actual current search — treat
this as a first draft to verify, not a citable section as-is.

## Where this project's methods come from

- **Conditional image inpainting via denoising diffusion.** The pixel-space
  conditional DDPM, DDIM sampling, and RePaint-style resampled inpainting used
  throughout this project follow the general line of work from Ho et al.
  (DDPM), Song et al. (DDIM, deterministic accelerated sampling), and
  Lugmayr et al. (RePaint, unconditional-prior resampling for inpainting) —
  the fingerprint-specific adaptation here is conditioning the denoiser
  directly on the observed pixels and mask (and, for the residual variant, on
  a frozen coarse deterministic reconstruction), rather than working with an
  unconditional prior and a separate inpainting mask trick as in the original
  RePaint formulation.
- **Latent diffusion.** The SOCOFing-side latent-diffusion and residual
  latent-diffusion models follow the general autoencoder-plus-diffusion-in-
  latent-space idea popularized by Rombach et al. (Stable Diffusion / latent
  diffusion models), motivated there primarily by compute cost at high
  resolution — this project's images are small (128×128) so the primary
  motivation for adopting it here would be architectural comparison, not
  necessity, which is part of why the NIST302 extension of this arm was
  deprioritized under the time constraint (see `docs/limitations.md`).
- **Conditional VAEs for structured uncertainty.** The global- and spatial-
  latent CVAE architectures follow the conditional VAE line (Sohn et al. and
  follow-on work); the spatial (per-location latent grid) variant built this
  session is a direct response to a concrete, measured failure of the global
  variant (uniform predictive std across easy/hard regions) rather than
  literature-driven from the start — it is closer in spirit to the general
  observation in generative modeling that a single global latent code is an
  information bottleneck poorly suited to expressing spatially localized
  uncertainty, which shows up in segmentation and other dense-prediction
  probabilistic literature (e.g. probabilistic U-Net style spatial-latent
  approaches) more than in the original CVAE inpainting literature itself.
- **Fingerprint reconstruction from partial/latent prints.** The task framing
  (reconstruct a full, useable fingerprint from a partial or degraded
  observation) sits in the latent-fingerprint enhancement and fingerprint
  reconstruction/synthesis literature; the specific choice to hold out real
  `quality==1` pixels for evaluation, and to distinguish `X_pseudo` from true
  pixel ground truth throughout, is a direct methodological response to how
  NIST distributes SD302 (paired latent/exemplar impressions with
  examiner-verified correspondence points, not pixel-registered ground
  truth), not something imported from a specific paper.
- **Ridge orientation, ridge-frequency, and Gabor-based structural losses.**
  The orientation-field, ridge-band, and ridge-spectrum losses used to steer
  reconstructions toward plausible ridge structure draw on the classical
  fingerprint enhancement literature (Gabor filtering conditioned on local
  ridge orientation and frequency, in the tradition of Hong, Wan, and Jain's
  enhancement algorithm and its many descendants), adapted here as
  differentiable training losses rather than a post-hoc enhancement filter.

## Directly competing prior work (user-reported, not independently verified)

**Hussein, Jain, and Nandakumar, "Progressive Learning of a Diffusion-based
Inpainting Model for Separating Overlapped Fingerprints," reported as
accepted at IJCB 2026.** This was surfaced by the user, not found by this
environment (no internet access here) — it has not been read or verified
against a primary source, only recorded as reported: (1) starts from a
fingerprint-prior diffusion model, (2) trains it to complete partial prints,
(3) adds problem-specific conditioning afterward, (4) uses latent diffusion
with LoRA adapters, (5) extends the U-Net input with the partial image and
mask, (6) trains progressively from easy to hard. If accurately reported,
this directly overlaps with this project's core method (diffusion inpainting
for fingerprints) and means **this project cannot claim to be first to apply
diffusion-based inpainting to fingerprints** — that framing should be
dropped from any write-up.

**What is not covered by that paper, per the same report, and where this
project's actual results land:**

| Claimed open space | This project's coverage |
|---|---|
| Uncertainty-aware reconstruction | CVAE, DDPM/DDIM/RePaint/residual/latent-diffusion families (master table sections 5-7, 14) |
| Uncertainty calibration | CRPS, interval score, sparsification error (section 12); coverage/interval-width calibration (`calibrate_nist302_ddpm_uncertainty.py`) |
| Performance vs. observed-fraction relationship | `docs/socofing_reconstructibility_segmented.md` (segmented regression, SOCOFing) |
| Fragment/mask geometry | Section 12's mask-geometry difficulty axis (distance to nearest observed pixel, independent of registration confidence) |
| Synthetic-to-real transfer on SD302 | Track E, section 17 (synthetic degradation with exact ground truth) |
| Uncertainty vs. distance to verified information | Sections 9 and 12 (registration-confidence and mask-geometry distance bands) |
| Ground-truth recovery vs. structural plausibility | The Track A-E reporting system (`FINAL_REPORT.md` section 2.5) and section 15's ridge-topology artifact check, which explicitly separates "looks locally plausible" from "is topologically correct" |

If the report about this paper is accurate, this project's actual
contribution is not "first diffusion inpainting for fingerprints" but this
specific list — which this session had already substantially built out
*before* this paper was surfaced, for reasons internal to the project (the
target-choice critical finding, the ridge-topology artifact check, and the
permuted-observed control were all motivated by direct empirical concerns,
not by positioning against this or any other paper).

**A detailed 12-section alternative architecture proposal was also
suggested** (a three-stage "Progressive Structure-Conditioned Ridge
Diffusion": structural-prior pretraining → synthetic partial inpainting →
real-latent LoRA adaptation, at ridge-aware patch resolution, with
ControlNet-style multiscale structural injection, a boundary/phase-
continuity loss, and conditional flow-matching or Brownian-bridge
alternatives to standard DDPM). Status per component, updated as each was
checked against this project's actual code and results:

- **Residual diffusion**, **a structure predictor conditioning the
  denoiser**, and **the permuted-Y control experiment** were already built
  before this proposal was seen (master table sections 6, 11, 18).
- **ControlNet-style multiscale structural injection is already present,
  not a simpler substitute for it**: `DiffusionUNet`'s
  `auxiliary_condition_channels` mechanism, used by every conditioned
  DDPM-family model in this project, injects its conditioning signal at
  every encoder resolution through a separate zero-initialized 1x1
  convolution per scale — the defining "zero convolution" mechanism of
  ControlNet itself (master table section 18's correction).
- **A boundary/phase-continuity loss**, **conditional residual flow
  matching**, and **conditional Brownian Bridge Diffusion** were built and
  tested this session (`src/fingerprint_reconstruction/losses/reconstruction.py`'s
  `_boundary_continuity_loss`, `src/fingerprint_reconstruction/models/flow_matching.py`,
  `src/fingerprint_reconstruction/models/brownian_bridge.py`) — see the
  master table for results once training completes.
- **Ridge-aware native-resolution patches** remain the one component not
  attempted at full scale: the multi-day pipeline rebuild is scoped in
  `docs/nist302_native_resolution_feasibility.md` and not attempted end-to-
  end, though a same-day pilot patch dataset
  (`Sd302SyntheticPatchDataset`) was built to give a directional answer
  without the full rebuild.

## What is genuinely new in this project (as far as can be assessed offline)

- The direct, controlled demonstration that fine-tuning against an
  approximately-registered cross-impression target can *look like* an
  improvement while being a *regression* against real held-out pixels (the
  critical finding at the top of `docs/nist302_ablation_master_table.md`) is
  a specific, quantified instance of a known general risk in weak/proxy
  supervision, but this project does not know of (and could not check for,
  offline) a prior paper making exactly this measurement on latent
  fingerprints specifically.
- The distance-to-verified-evidence stratification of uncertainty
  calibration (section 9 of the master table, and
  `docs/nist302_repaint_distance_diag_full.json` /
  `nist302_cvae_distance_diag_full.json` / `nist302_ddpm_distance_diag_full.json`)
  — showing that different probabilistic architectures' uncertainty signals
  degrade differently (DDIM gracefully, CVAE by sign-flipping) as a function
  of geometric distance from ground-truth anchor points — is, as far as this
  offline assessment can tell, a project-specific empirical contribution
  rather than a replication of an existing benchmark.

## Explicit caveat

Given the offline constraint, this section cannot rule out that some of the
above is already established more rigorously elsewhere, nor can it supply
correct citation keys, years, or venue information. Anyone taking this report
forward should treat every claim above as "needs a real search," not as
settled prior art.
