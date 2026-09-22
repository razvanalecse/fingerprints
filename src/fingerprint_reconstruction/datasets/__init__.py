"""Dataset-specific readers and manifest builders."""

from .socofing import (
    SocofingAudit,
    SocofingManifest,
    SocofingRecord,
    build_socofing_manifest,
    parse_socofing_filename,
    write_socofing_artifacts,
)
from .nist302 import (
    Nist302Audit,
    Nist302LatentRecord,
    Nist302Manifest,
    build_nist302_manifest,
    parse_latent_png_filename,
    write_nist302_artifacts,
)
from .nist302_pairs import (
    Nist302ExemplarRow,
    Nist302LatentExemplarPair,
    Nist302PairError,
    build_latent_exemplar_pairs,
    load_exemplar_rows,
    write_pair_manifest,
)

__all__ = [
    "SocofingAudit",
    "SocofingManifest",
    "SocofingRecord",
    "build_socofing_manifest",
    "parse_socofing_filename",
    "write_socofing_artifacts",
    "Nist302Audit",
    "Nist302LatentRecord",
    "Nist302Manifest",
    "build_nist302_manifest",
    "parse_latent_png_filename",
    "write_nist302_artifacts",
    "Nist302ExemplarRow",
    "Nist302LatentExemplarPair",
    "Nist302PairError",
    "build_latent_exemplar_pairs",
    "load_exemplar_rows",
    "write_pair_manifest",
]
