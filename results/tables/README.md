# Final tables

This directory contains compact, manuscript-ready tables derived from the
per-run artefacts indexed in `../artifact-index.csv`.

- `socofing_deterministic_baselines.csv`: controlled Track-A baselines.
- `nist302_heldout_evaluation_217.csv`: held-out real-pixel metrics with source
  split metadata preserved explicitly.
- `nist302_k_sensitivity.csv`: expected, best-of-K, diversity, and uncertainty
  behavior as K changes.
- `compute_cost.csv`: measured training and inference costs.
- `residual_mask_family_summary.csv`: aggregate results for every mask family.
- `residual_uncertainty_k_scaling.csv`: fidelity, calibration, diversity, and
  cost for K in {5, 10, 20, 50}.
- `reports/` contains the complete table-bearing baseline, ablation, compute,
  and reconstructibility reports.
- `table_index.csv` inventories every final table/report and its provenance.

Each final table should document:

- dataset and split;
- unit of analysis and sample size;
- model/checkpoint selection rule;
- observed-fraction and mask-family strata;
- mean or median with 95% confidence interval;
- paired effect size and multiplicity-adjusted p-value where applicable;
- exact source artefact paths under `outputs/`.

Do not copy raw dataset records or per-subject biometric metadata here.
