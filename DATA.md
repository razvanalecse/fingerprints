# Data, and what is not in this repository

## Not included

**No fingerprint images of any kind are committed here, and no model weights.**

- `data/` — SOCOFing and NIST SD302. Neither is redistributed. SOCOFing is a
  public research dataset; **NIST Special Database 302 is distributed by NIST
  under its own terms and contains real forensic latent prints from real
  people.** Obtain both from their sources.
- `*.pt` — 330 checkpoints, 6.5 GB in the working tree.
- Rendered previews under `outputs/` and several figures under
  `docs/poster_assets/` — many of them display real SD302 latents. They are
  excluded by `.gitignore` rather than deleted, so re-enabling them is a
  one-line change if your data agreement allows it. The compiled poster PDF
  embeds the same images and is excluded for the same reason; `docs/poster.tex`
  is committed and rebuilds it once the assets are present.

## Included

- All source, scripts, configs and tests.
- All documentation, including `docs/nist302_ablation_master_table.md`
  (32 ablation sections) and `docs/FINAL_REPORT.md`.
- **All metrics.** Every `.json` and `.csv` under `outputs/` is committed, about
  40 MB, so each number in the write-ups can be traced to the run that produced
  it without re-running anything.

## Reproducing

Paths to the raw data are passed on the command line, not hard-coded; see the
`--sd302a-root` / `--latent-root` style arguments in `scripts/`. The exact
roots used are recorded in `data/processed/nist302/nist302_audit.json`, which
is itself not committed.

## A note on scope

This project studies reconstruction as an inverse problem and as a test of
evaluation methodology. It is not an identification system, and its central
result is a caution about how reconstruction quality is measured, not a
capability claim. A generated ridge field must not be described as recovery of
an individual's true fingerprint.
