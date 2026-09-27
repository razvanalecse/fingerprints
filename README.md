# From Fragments to Structure

Research code for uncertainty-aware reconstruction of partial fingerprints using
deterministic and probabilistic generative models.

## Scope

This repository is restricted to academic experiments on public or synthetic
datasets. It does not contain tooling for bypassing authentication, fabricating
physical biometric artefacts, or impersonating individuals.

The implementation is intentionally incremental. Every stage adds tests and a
reproducible configuration before model training is introduced.

## Current stage

- deterministic experiment seeding;
- leakage-safe grouped split construction;
- exact-area synthetic mask generation for eight mask families;
- strict SOCOFing filename parsing and original/derived-file separation;
- image decoding, intensity audit, SHA-256 dataset fingerprinting, and atomic manifests;
- alpha-aware SOCOFing decoding and foreground/axial-orientation diagnostics;
- deterministic U-Net baseline conditioned on `[Y, M]`;
- parameter-free nearest-observed interpolation lower bound;
- missing-region reconstruction losses and exact observed-pixel consistency;
- ROI-restricted axial orientation error and paired model tests;
- reproducible PyTorch smoke-training with checkpoints and visual previews;
- unit tests for geometry, determinism, and observed-fraction semantics.

## U-Net smoke test

```bash
MPLCONFIGDIR=work/matplotlib .venv/bin/python scripts/train_unet.py \
  --config configs/unet.yaml \
  --manifest data/processed/socofing/socofing_manifest.csv \
  --image-root data/raw/socofing/SOCOFing/Real \
  --output outputs/unet_smoke \
  --smoke-test
```

The smoke test validates data loading, forward/backward passes, checkpointing,
and data consistency. It is deliberately not reported as model performance.

## Apple GPU / MPS pilot

The macOS 26 host requires a current Python/PyTorch environment. The isolated
`.venv-mps` environment uses Python 3.12 and PyTorch with Metal support:

```bash
MPLCONFIGDIR=work/matplotlib PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv-mps/bin/python scripts/train_unet.py \
  --config configs/unet.yaml \
  --manifest data/processed/socofing/socofing_manifest.csv \
  --image-root data/raw/socofing/SOCOFing/Real \
  --output outputs/unet_pilot_mps \
  --device mps --pilot
```

When executed in an environment without Metal device access, run this command in a local shell
because the sandbox does not expose the Metal device.

## Prepare SOCOFing originals

```bash
.venv/bin/python scripts/prepare_socofing.py \
  --root /absolute/path/to/SOCOFing \
  --output data/processed/socofing
```

The command fails if the expected 6,000 originals, 600 subjects, or ten fingers
per subject are not present. `--allow-incomplete` is reserved for development
fixtures and explicit subset studies; count mismatches are then recorded in the
audit report.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

Dataset files are deliberately excluded from version control. NIST datasets must
be obtained under the terms shown by the official NIST download service.

## Results and traceability

`outputs/` stores machine-readable artefacts from individual runs. The curated
[`results/`](results/README.md) layer contains the publication-facing catalogue,
populated final tables, aggregate figures, and frozen statistical analyses.
Read the [`results summary`](results/RESULTS_SUMMARY.md), or rebuild both the
curated outputs and artefact catalogue with:

```bash
python3 scripts/build_curated_results.py
python3 scripts/build_results_catalog.py
```

Raw fingerprints, NIST-derived image previews, and model checkpoints are not
included in either directory.

## License and responsible use

This project is released under the custom
[Academic Research Use License v1.0](LICENSE). It may be used for
non-commercial academic research, education, reproducibility, and scientific
evaluation on public, properly licensed, or synthetic datasets.

The license prohibits operational biometric identification or authentication,
impersonation, unauthorized security testing, fabrication of biometric
artefacts, redistribution of third-party datasets, and representing a plausible
generation as a person's true missing fingerprint. This is a research-only
license and is not an OSI-approved open-source license.
