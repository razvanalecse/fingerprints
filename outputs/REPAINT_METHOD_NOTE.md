# RePaint-style fingerprint reconstruction: literature-grounded implementation note

## Primary sources

- Ho, Jain, and Abbeel (2020), *Denoising Diffusion Probabilistic Models*,
  NeurIPS 2020: <https://arxiv.org/abs/2006.11239>
- Song, Meng, and Ermon (2021), *Denoising Diffusion Implicit Models*, ICLR
  2021: <https://arxiv.org/abs/2010.02502>
- Lugmayr et al. (2022), *RePaint: Inpainting using Denoising Diffusion
  Probabilistic Models*, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html>

## Implemented transition

At reverse time `t`, the missing region is sampled from the learned DDPM
posterior. The observed region is independently sampled at the corresponding
forward-process noise level and the two regions are composed using the binary
observation mask. For `U > 1`, the composed `x_(t-1)` is diffused one Markov
step back to `x_t`,

`x_t = sqrt(alpha_t) x_(t-1) + sqrt(beta_t) epsilon`,

and the reverse transition is repeated. This is the resampling mechanism in
Algorithm 1 of Lugmayr et al.; its purpose is to give the model additional
opportunities to harmonize the generated and known regions before proceeding
to the next noise level.

## Declared adaptation

Original RePaint applies inference-time conditioning to an unconditional DDPM.
This project applies the same known-region composition and resampling mechanism
to a fingerprint DDPM already conditioned on `(Y, M)`. Therefore, the
experiment estimates the incremental effect of RePaint resampling beyond
explicit mask conditioning; it is not claimed to be an exact replication of
the original network setup.

`U=1` is the conditioning-only ancestral baseline. `U=2` adds one
forward/reverse resampling cycle at every non-terminal timestep. Observed
pixels are projected exactly onto `Y` in the returned reconstruction.

## Interpretation constraint

Sharper or better-harmonized ridges do not imply recovery of the true missing
fingerprint. Results must continue to distinguish ground-truth fidelity,
structural plausibility, and predictive uncertainty.
