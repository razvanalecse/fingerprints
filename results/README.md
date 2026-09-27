# Curated research results

This directory is the publication-facing layer of the repository. It is kept
separate from `outputs/`, which contains the machine-readable artefacts emitted
by individual experimental runs.

## Directory contract

- `artifact-index.csv` inventories every committed JSON/CSV metric artefact in
  `outputs/`, including its SHA-256 digest for traceability.
- `artifact-summary.json` records aggregate counts and the command used to
  rebuild the catalogue.
- `tables/` contains manuscript-ready aggregate tables. Per-image measurements
  remain under `outputs/` so that the analysis can be reproduced.
- `figures/` contains publication-ready aggregate or synthetic-data figures.
- `statistics/` contains frozen inferential results: paired tests, confidence
  intervals, effect sizes, multiplicity corrections, and model specifications.

Rebuild the catalogue from the repository root with:

```bash
python3 scripts/build_results_catalog.py
```

## Data-governance rule

No raw fingerprint image, model checkpoint, or identifiable NIST SD302 latent
may be added here. Figures derived from real forensic impressions must remain
local unless redistribution is explicitly permitted by the applicable dataset
agreement. Aggregate metrics and plots that do not reveal source impressions
may be committed.

## Interpretation rule

The project distinguishes three different targets:

1. **Ground-truth recovery**: similarity to a known held-out region, available
   exactly for synthetic masking and only approximately for registered NIST
   exemplar/latent pairs.
2. **Structural plausibility**: ridge-orientation, frequency, topology, and
   boundary continuity, which do not prove recovery of the true missing ridge
   pattern.
3. **Predictive uncertainty**: sample dispersion and calibration relative to
   observed reconstruction error.

Best-of-K scores must never be reported as single-sample predictive
performance. Results based on registered NIST pseudo-targets must be labelled
`registered_approximate` and stratified by geometric confidence where possible.
