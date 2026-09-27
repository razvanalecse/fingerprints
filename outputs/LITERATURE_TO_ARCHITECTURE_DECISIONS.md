# Literature-to-architecture decisions

This note records which published ideas motivate each implementation change. It
does not claim novelty or transfer of published results to our datasets.

## Evidence already acted on

1. **RePaint** — Lugmayr et al., 2022, *Inpainting using Denoising Diffusion
   Probabilistic Models*. RePaint conditions an unconditional DDPM during reverse
   diffusion by repeatedly sampling/reinjecting known regions. Our latent
   reinjection is an explicitly labelled approximation because an autoencoder
   latent mask is not pixel-exact. On 90 paired SOCOFing validation cases it
   reduced predictive-mean MAE from 0.3216 to 0.2647 and improved orientation
   error from 0.4167 to 0.3480.
   Source: https://arxiv.org/abs/2201.09865

2. **Latent Diffusion Models** — Rombach et al., 2022. LDM separates perceptual
   compression from diffusion and uses attention for flexible conditioning. We
   first validated the fingerprint autoencoder independently, then added
   zero-initialized multi-scale conditioning and bottleneck self-attention.
   Source: https://arxiv.org/abs/2112.10752

3. **Palette** — Saharia et al., 2022. Palette studies conditional diffusion for
   inpainting and other image-to-image problems, reports an important role for
   self-attention, and studies L1 versus L2 denoising losses. This motivates a
   controlled attention ablation and a future L1/L2 noise-loss ablation, not a
   direct assumption of improvement.
   Source: https://arxiv.org/abs/2111.05826

4. **DDRM / DPS** — Kawar et al., 2022; Chung et al., 2023. These works formulate
   restoration as posterior sampling constrained by the measurement operator.
   They motivate our planned decode-project-reencode sampler, where the pixel
   observation operator is applied during sampling rather than only at the end.
   Sources: https://papers.neurips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html
   and https://arxiv.org/abs/2209.14687

## Fingerprint-specific structural guidance

5. **FingerNet** — Li, Feng, and Kuo, 2018, *Deep convolutional neural network
   for latent fingerprint enhancement*. The network shares features between an
   enhancement branch and an orientation branch. This supports a multi-task
   structure predictor, but our predictor must receive only `(Y,M)` at inference.
   Source: https://doi.org/10.1016/j.image.2017.08.010

6. **ID Preserving GAN for Partial Latent Fingerprint Reconstruction** — Dabouei
   et al., BTAS 2018. The generator predicts additional orientation/frequency
   maps to discourage erroneous filling and false minutiae in large missing
   regions. Our planned diffusion variant will condition on predicted axial
   orientation `(cos(2 theta), sin(2 theta))`, confidence, and support. We will
   not use identity/matching losses in the primary reconstruction objective.
   Source: https://arxiv.org/abs/1808.00035

7. **Generative Convolutional Networks for Latent Fingerprint Reconstruction** —
   Svoboda, Monti, and Bronstein, 2017. This is prior work on predicting missing
   ridge patterns in corrupted/partial latent fingerprints and prevents us from
   claiming that generative partial-fingerprint reconstruction itself is novel.
   Source: https://arxiv.org/abs/1705.01707

8. **Progressive GAN latent fingerprint enhancement** — Huang et al., 2020.
   Jointly representing enhancement and orientation reinforces the need to
   evaluate orientation fields separately from pixel metrics.
   Source: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w48/Huang_Latent_Fingerprint_Image_Enhancement_Based_on_Progressive_Generative_Adversarial_Network_CVPRW_2020_paper.pdf

## Texture and topology candidates

9. **Diffusion Texture Prior Model** — Ye et al., CVPR 2024. DTPM combines an
   initial predictor with a diffusion texture prior. This motivates a future
   coarse-structure predictor followed by diffusion of ridge-scale residuals,
   rather than asking diffusion to generate global support and texture jointly.
   Source: https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Learning_Diffusion_Texture_Priors_for_Image_Restoration_CVPR_2024_paper.html

10. **clDice** — Shit et al., CVPR 2021. Differentiable soft skeletonization can
    preserve connectivity in tubular segmentation. Fingerprint ridges are dense,
    alternating line structures rather than sparse vessels, so clDice is only a
    candidate ablation after binarization robustness is demonstrated.
    Source: https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf

11. **Frequency-Guided Posterior Sampling** — Thaker, Goyal, and Vidal, ICCV
    2025. FGPS progressively incorporates measurement frequencies during
    posterior sampling. Fingerprint ridges make a low-to-high frequency
    curriculum especially relevant, but transfer is not assumed: it will be a
    separate sampler ablation against identical seeds and cases.
    Source: https://openaccess.thecvf.com/content/ICCV2025/html/Thaker_Frequency-Guided_Posterior_Sampling_for_Diffusion-Based_Image_Restoration_ICCV_2025_paper.html

12. **A Unified Conditional Framework for Diffusion-based Image Restoration** —
    Zhang et al., 2023. A lightweight network first predicts guidance and the
    diffusion model learns a residual conditioned throughout its blocks. This
    independently supports our planned coarse predictor plus ridge-texture
    residual diffusion experiment.
    Source: https://arxiv.org/abs/2305.20049

13. **Spectral Filter Predictor for Progressive Latent Fingerprint
    Restoration** — Liu et al., 2024. This fingerprint-specific work motivates
    treating ridge frequency as an explicit signal and warns that spatial
    downsampling may alias friction ridges. We will therefore estimate the
    frequency target robustly before adding it as a fifth structural channel.
    Source: https://ieeexplore.ieee.org/document/10526230

## Completed structural and sampling ablations

- The leakage-safe structure predictor receives only `(Y,M)` and predicts
  support, doubled-angle orientation, and coherence at latent scale. Its best
  validation support IoU was 0.956 and weighted axial error was 0.0895.
- Full-backbone structural fine-tuning slightly degraded several image metrics.
  This negative result is retained.
- Freezing v2 and learning only zero-initialized structural adapters improved
  all primary paired metrics over v2 on 90 cases: mean MAE by 0.0143, PSNR by
  0.483 dB, SSIM by 0.00254, and orientation error by 0.0449 (Holm-corrected
  tests significant for all four).
- Decode-project-reencode every five DDIM steps further improved MAE by 0.0116,
  PSNR by 0.372 dB, and SSIM by 0.0203 over latent reinjection, at about 24%
  extra sampling time. Orientation error did not improve significantly.
- The protocol now records both canvas and fingerprint-ROI observed fractions.
  Across the first 90 paired cases, their mean absolute difference was 0.0872
  and the maximum was 0.2650, showing that nominal canvas `r` is not an adequate
  reconstructibility variable by itself.
- Coarse-plus-residual latent diffusion is now implemented as
  `R_z = s_r [E(X)-E(X_coarse)]`, with a frozen Gated predictor supplying
  `X_coarse`. On the same 90 paired cases, it improved over the previous best
  projected sampler by 0.0293 mean MAE, 0.825 dB PSNR, 0.0771 SSIM, and 0.0688
  orientation error. Holm-corrected paired tests were significant for every
  primary metric; effect sizes were `d_z=1.206` for MAE and `d_z=1.296` for
  SSIM. This supports structure/texture factorization on SOCOFing but does not
  establish transfer to real latent fingerprints.
- Four lower-learning-rate continuation epochs produced a Pareto trade-off:
  orientation error and best-of-K improved, while mean MAE changed slightly in
  the wrong direction and not significantly. Both checkpoints are retained.
- An orientation-steered quadrature Gabor loss was then calibrated from the
  empirical SOCOFing ridge band (median frequency 0.242 cycles/pixel at the
  128x128 working resolution). Applied to the deterministic coarse predictor,
  it reduced ridge-period error by 59.3% and orientation error from 0.2289 to
  0.1812, while slightly worsening MAE and PSNR. This is retained as a
  structure--pixel trade-off, not reported as a universal improvement.
- Replacing the original coarse predictor in residual latent diffusion with
  this Gabor-trained predictor was evaluated on the same 90 cases and random
  seeds. Mean MAE changed by +0.00185 (paired Wilcoxon `p=0.120`, not
  significant), whereas SSIM improved by 0.00723 (`p=9.11e-5`) and axial
  orientation error improved by 0.05543 (`p=3.68e-10`); orientation improved
  in 87.8% of cases. Best-of-5 MAE and uncertainty--error Spearman correlation
  also improved. A separate 20-case, 25-step, K=3 control found essentially no
  ridge-period change (1.2510 versus 1.2472 pixels, `p=0.984`) but retained a
  significant orientation improvement (`p=0.00730`). The evidence therefore
  supports improved ridge geometry rather than mere matching of ridge spacing.
  It does not show exact recovery of highly unobserved ground truth.
- A nested repeated-measures pilot now evaluates 12 validation fingerprints
  under all eight observed fractions and seven complete mask families (672
  complete factorial conditions, plus 36 severe-partial conditions). Mask
  geometry is fixed while `r` grows. Friedman tests show large observed-fraction
  effects for MAE (`W=0.937`), SSIM (`W=0.942`), and orientation error
  (`W=0.832`), all `p<2.6e-12`. Mask geometry also has significant effects,
  especially for SSIM (`W=0.769`) and orientation (`W=0.543`). These are pilot
  subject-level results and require a larger held-out confirmatory run.
- Segmented regression on the orientation curve selected `r=0.50` in 83.9% of
  subject bootstraps and improved BIC by 18.3 relative to a linear trend. This
  is evidence for a possible change in slope, not yet proof of a universal
  reconstructibility threshold. MAE and SSIM did not support a segmented model.
- Canvas-area `r` was shown to differ systematically from the fraction of actual
  fingerprint ROI observed. A separate RQ6 control therefore enforced equal
  ROI information for central-only and peripheral-only masks (12 subjects, all
  eight fractions). Peripheral-only observations had lower average MAE
  (`p=0.0161`) and substantially lower orientation error (`p=0.000977`) when
  averaged across `r`; SSIM showed no overall difference (`p=0.791`). Per-r
  effects crossed over for MAE/SSIM at larger `r`, so the data do not support a
  blanket claim that central information is always more informative.
- Pairwise sample diversity had a negative within-subject association with
  observed fraction in the ROI-controlled experiment (mean subject Spearman
  `-0.677`, Wilcoxon `p=0.000488`). Because this pilot used only `K=3`, it
  supports H10 provisionally and must be confirmed with `K>=20`.
- Predictive intervals for the Gabor-coarse residual model were calibrated with
  `K=20` on 30 validation subjects and evaluated on 30 unseen test subjects.
  Raw sample quantiles were strongly under-dispersed (test coverage 0.284,
  0.487, and 0.572 for nominal 50%, 80%, and 90% intervals). Scale multipliers
  learned only on validation subjects improved test coverage to 0.476, 0.772,
  and 0.887. Thus spatial uncertainty is informative but its raw magnitude is
  not calibrated; calibrated intervals, clipped to the valid intensity domain
  `[0,1]`, must be reported separately from raw generative quantiles.

## Ordered implementation plan

1. Keep latent reinjection as the validated sampler default.
2. Retain multi-scale conditioning/attention only where paired tests support it.
3. Extend the validated structure predictor with ridge frequency only after a
   robust target-quality audit.
4. Keep the validated adapter-only structural conditioning; do not unfreeze the
   backbone by default.
5. Add decoded-x0 structural supervision as an ablation, not to the base loss.
6. Tune decode-project-reencode interval/strength and report its compute cost.
7. Test deterministic coarse prediction plus diffusion texture residuals.
8. Test low-to-high frequency-guided sampling.
9. Test L1 versus L2 noise prediction and topology/ridge-frequency losses.

Every item must be evaluated with subject-disjoint splits, missing-ROI metrics,
paired cases, confidence intervals, effect sizes, and multiple-comparison control.
