# Final figures

This directory contains publication-ready aggregate plots. No figure here
contains a NIST fingerprint image.

- `socofing_baselines.png`: deterministic Track-A comparison.
- `nist302_model_mae.svg`: held-out real-pixel MAE with the source split caveat
  included directly in the figure.
- `uncertainty_vs_anchor_distance.png`: uncertainty localization by distance
  from examiner-verified evidence.
- `selective_reconstruction.png`: error-coverage curve under abstention.

Permitted examples include metric curves, confidence intervals, calibration
plots, ablation summaries, sampling-time comparisons, and SOCOFing examples if
its licence permits the intended redistribution. Do not commit figures that
display NIST SD302 latent or exemplar impressions unless the applicable NIST
terms have been checked and explicitly permit that use.

Every figure should have a caption or adjacent metadata identifying the script,
input artefacts, split, seed, and metric definition used to generate it.
