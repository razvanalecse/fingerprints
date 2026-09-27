# Statistical results

This directory contains frozen inferential analyses selected without using a
final test set.

- `selected_paired_tests.csv` consolidates the principal subject/case-paired
  model comparisons and retains the exact source JSON for every row.
- `nist302_friedman_omnibus.json` records the six-model repeated-measures
  omnibus MAE test.

Reports should include the statistical unit, pairing/grouping structure,
distributional checks, test statistic, 95% confidence interval, effect size,
multiplicity correction, and the exact source artefacts. Subject or finger
should be treated as a grouping variable whenever repeated observations would
otherwise violate independence.

Thresholds, registration filters, and hyperparameters must be calibrated on
training/validation data. The final test set is opened only after the analysis
plan and model selection are frozen.
