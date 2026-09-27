# Cross-dataset protocols (point 6)

Date: 2026-09-23. Three protocols were specified; SD303 is explicitly
deferred until it is received and its documentation verified, per standing
instruction.

## Protocol 1 — SOCOFing → SOCOFing

Train and evaluate entirely within SOCOFing, synthetic masks, subject-disjoint
split. This is the project's original deterministic + probabilistic track.

- Deterministic: `outputs/DETERMINISTIC_BASELINES.md` (classical, U-Net, Gated).
- Structural-loss variant (this session's contribution): `v3_loss`, and the
  Codex-built hybrid combining it with ridge-spectrum loss
  (`outputs/gated_claude_hybrid_full/vs_v3_loss_paired.json`), which beat
  `v3_loss` on all four primary metrics simultaneously.
- Probabilistic: cVAE, DDPM, DDIM, RePaint, latent diffusion, residual latent
  diffusion — all implemented and evaluated on SOCOFing (`outputs/cvae_full_mps`,
  `outputs/ddim_50step_k10_validation`, `outputs/repaint_u2_k5_balanced90`,
  `outputs/residual_latent_diffusion_v1*`).
- Reconstructibility vs r, segmented regression: `docs/socofing_reconstructibility_segmented.md`.

**Status: complete**, test split still sealed.

## Protocol 2 — SOCOFing → SD302, zero-shot

Take the SOCOFing-trained Gated/U-Net checkpoints unchanged and evaluate
directly on SD302 `registered_approximate` validation, no SD302 training data
used at all.

- `outputs/nist302_gated_zeroshot_validation`, `outputs/nist302_unet_zeroshot_validation`.
- Headline numbers vs classical NIST302 baseline:
  `outputs/nist302_registered_model_comparison/paired_subject_tests.json`.

**Status: complete**, test split still sealed.

## Protocol 3 — SOCOFing pretraining → SD302 fine-tuning → SD302 unseen validation

**Correction (2026-09-23): this section previously overstated which arms
actually follow the pretrain-then-finetune recipe.** Checked directly against
each training script's code (`config["model"]["initialize_from"]` is the only
mechanism any script uses to warm-start from a checkpoint):

- **Deterministic (genuinely protocol 3):** `scripts/finetune_nist302_registered.py`
  reads `config["model"]["initialize_from"]` and loads
  `outputs/gated_full_mps/best-checkpoint.pt` (the SOCOFing-trained gated
  model) before fine-tuning on the 1,213 SD302 train pairs. This is the one
  arm where "SOCOFing pretrain → SD302 fine-tune" is literally true:
  `outputs/nist302_registered_finetune_no_spectrum_full`,
  `outputs/nist302_registered_support_spectrum_full`,
  `outputs/nist302_registered_support_spectrum_uniform_confidence_full`.
- **CVAE:** `scripts/train_nist302_cvae.py` *supports* an optional
  `initialize_from`, but the spatial arm that was ultimately selected
  (`outputs/nist302_registered_cvae_spatial_full`) was **not** given one —
  it trained from a random initialization directly on SD302 (the SOCOFing
  CVAE checkpoint was tried only for the global-latent variant; the spatial
  architecture is structurally incompatible with those weights). Not
  protocol 3 in the pretrain-then-finetune sense.
- **DDPM (pixel and residual): also not protocol 3.**
  `scripts/train_nist302_ddpm.py` and `scripts/train_nist302_residual_ddpm.py`
  have **no `initialize_from`/resume mechanism at all** — neither script's
  `parse_args()` nor its config schema has any path to warm-start the
  denoiser from a checkpoint. `outputs/nist302_registered_ddpm_full` and
  `outputs/nist302_registered_residual_ddpm_full` were trained **from a
  random initialization directly on the 1,213 SD302 train pairs**, the same
  as the spatial CVAE. The only SOCOFing-derived component either script
  uses is the frozen visible-support predictor (an auxiliary conditioning
  signal, not a pretrained denoiser) and, for the residual arm, the frozen
  coarse deterministic model (itself SD302 fine-tuned, per above — not
  SOCOFing-frozen either). This was previously mislabeled here as complete
  protocol-3 coverage; it was not, and every place in this project's docs
  that describes the DDPM arms as "pretrained then fine-tuned" should be
  read as "trained from scratch on SD302" instead.
- Full ablation table: `docs/nist302_ablation_master_table.md`.

**Status: complete only for the deterministic arm.** CVAE, pixel DDPM, and
residual DDPM are all SD302-only training runs (random initialization), not
pretrain-then-finetune, and should not be described as protocol 3 anywhere
in the final report without this caveat attached.

Test split remains locked for all three protocols; no test-set number appears
anywhere in this project's outputs.

## SD303

Deferred. No SD303 data has been received; nothing in this codebase
references it.
