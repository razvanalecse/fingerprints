# Superseded poster pipeline

The poster was first built with a Node/PPTX pipeline; it is now built from
`docs/poster.tex` with LaTeX. The `.mjs` builders and their layout JSON are
kept here because they are project code and the layout files record how the
earlier version was composed.

The generated artefacts that went with them -- PPTX candidates, PNG renders,
inspection dumps, roughly 31 MB -- are not committed. Neither is the older
rendered poster under `artifacts/`, which embeds real SD302 latents.
