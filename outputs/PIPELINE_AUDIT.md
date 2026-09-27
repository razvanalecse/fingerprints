# Pipeline audit — 2026-09-21

This document records what is implemented, what has executable evidence, and
what remains absent. A stage is **complete** only when code, tests, configuration,
and an inspectable result all exist.

## Status against the requested 18-stage pipeline

| Stage | Status | Implemented evidence | Missing before completion |
|---|---|---|---|
| 1. Literature review / novelty | Not complete | Project scope in `README.md` | Systematic search, screened bibliography, comparison matrix, novelty claims review |
| 2. Dataset analysis | Partial | SOCOFing strict parser, SHA-256 audit, 6,000-image manifest | NIST SD302/303 intentionally postponed by user; formal SOCOFing datasheet still needed |
| 3. Mathematical formulation | Partial | Mask semantics, axial orientation error and reconstruction losses encoded in source/tests | One versioned mathematical protocol document |
| 4. Experimental protocol / leakage | Partial | Subject-grouped deterministic splits and per-image manifest | Frozen experiment matrix, preregistered primary outcomes, multiplicity plan |
| 5. Preprocessing / MaskGenerator | Implemented | Eight exact-area mask families, normalization, alpha-aware decoding, foreground/orientation code | Final parameter freeze after visual audit |
| 6. EDA | Implemented for SOCOFing | Dataset distributions, mask diagnostics, foreground/orientation audits | NIST EDA deferred with datasets |
| 7. Deterministic baselines | Complete for the three preregistered families | Classical interpolation, 50-epoch U-Net and 50-epoch gated-convolution model evaluated on the same 900 validation cases with paired tests | Ridge-aware ablation is rejected in its current form; test split remains locked |
| 8. CVAE | Initial benchmark complete | Conditional posterior, standard-normal prior, KL warm-up/free-bits, K=10 sampling, diversity and uncertainty-error evaluation on 900 validation cases | K sensitivity and calibration/coverage remain part of Stage 12 |
| 9. Conditional DDPM | Limited-compute validation complete | `outputs/ddpm_limited_full_mps`, `outputs/ddim_50step_k10_validation` | Explicit scheduler and time-conditioned U-Net; 500-step training, 50-step DDIM evaluation with K=10 on 900 validation images; full-resolution and multi-seed confirmation remain |
| 10. RePaint-style sampling | Limited-compute subject-distributed validation complete | `outputs/repaint_u2_k5_balanced90`, `outputs/ddim50_vs_repaint_u2_paired_balanced90.json` | Paper-aligned reinjection/resampling; U=2 improves fidelity and orientation over DDIM-50 on 90 distinct subjects but reduces diversity and costs ~18x more; all-900 and multi-seed confirmation remain |
| 11. Latent diffusion | Not started | — | Autoencoder validation and latent diffusion |
| 12. Uncertainty quantification | Not started | — | K-sample mean/variance, calibration, coverage, error correlation |
| 13. Cross-dataset | Blocked by deliberate dataset deferral | — | SD302/303 acquisition and domain-specific protocol |
| 14. Ablations | Begun | Pixel-only U-Net; orientation/decoder experiments preserved | Controlled matched-seed ablation table |
| 15. Statistical testing | Not started | Per-image metric table exists | Paired tests, mixed-effects models, effect sizes, multiplicity correction |
| 16. Final figures/tables | Partial | EDA and U-Net learning/performance plots | Model comparison, uncertainty, calibration, compute tables |
| 17. Interpretation | Not started | Only provisional diagnostic notes | Interpretation after frozen experiments |
| 18. Paper writing | Not started | — | Full manuscript |

## What is actually reproducible now

1. SOCOFing preparation creates an atomic CSV/JSONL manifest and audit report.
2. The split is grouped by subject, preventing the same subject from appearing
   in train and validation/test.
3. Every mask uses `M=1` for observed pixels and has exactly
   `round(r*H*W)` observed pixels, subject to the declared severe-mask domain.
4. Foreground and axial ridge-orientation estimation have automated tests and
   visual audits.
5. The deterministic U-Net consumes `[Y,M]` and data consistency copies every
   observed pixel exactly.
6. The completed pixel-loss U-Net run contains its configuration, checkpoint,
   per-image metrics and plots. It is a valid negative baseline: it converges
   but blurs missing ridge structure.
7. The current ridge-aware work is an ablation under development, not a final
   result. Failed/intermediate outputs are kept separate and must not enter the
   headline result table.

## Immediate stage gates

### Gate A — freeze Stages 1–6

- write the literature matrix and formal mathematical protocol;
- produce a frozen experiment registry with primary metrics and seeds;
- verify and document the exact subject counts in every split;
- label NIST work explicitly as deferred, not completed.

### Gate B — close deterministic baselines

- classical nearest-observed interpolation baseline — **complete**;
- pixel-loss U-Net;
- gated-convolution baseline — **complete**;
- ridge-aware U-Net only if it improves orientation/frequency metrics without
  checkerboard or broad-band artifacts;
- matched validation inputs and paired confidence intervals.

### Gate C — probabilistic models

Only after Gates A and B: CVAE, conditional DDPM, RePaint, uncertainty and
best-of-K/expected-performance analysis.

## Current methodological correction

`data consistency` and `conditioning strength` are not the same claim. The
former is verified exactly on observed pixels. It does not prove that a U-Net
uses those pixels to infer the missing region. That must be measured with
controlled perturbation/occlusion tests and performance as a function of `r`.
