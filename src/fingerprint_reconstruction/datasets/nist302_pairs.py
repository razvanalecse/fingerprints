"""Leakage-safe SD302 latent/exemplar association records.

Associations are identity- and finger-consistent but never declared pixel-
aligned.  Registration and its acceptance criteria belong to a later stage.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from fingerprint_reconstruction.data.nist302_torch_dataset import Nist302AnnotatedRow


class Nist302PairError(ValueError):
    """Raised when an SD302 association violates an explicit contract."""


@dataclass(frozen=True)
class Nist302ExemplarRow:
    dataset_part: str
    relative_path: str
    subject_id: str
    device: str
    ppi: Optional[int]
    capture: str
    fgp: int


@dataclass(frozen=True)
class Nist302LatentExemplarPair:
    pair_id: str
    latent_sample_id: str
    subject_id: str
    split: str
    fgp: int
    latent_relative_path: str
    lffs_filename: str
    assessment: str
    exemplar_dataset_part: str
    exemplar_relative_path: str
    exemplar_device: str
    exemplar_ppi: Optional[int]
    exemplar_capture: str
    pixel_aligned_ground_truth: bool = False


def load_exemplar_rows(path: Path) -> Tuple[Nist302ExemplarRow, ...]:
    """Load and validate the exemplar artifact produced by prepare_nist302.py."""

    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            fgp = int(raw["fgp"])
            if not 1 <= fgp <= 10:
                raise Nist302PairError(f"non-distal exemplar FGP: {fgp}")
            ppi_text = raw["ppi"].strip()
            rows.append(
                Nist302ExemplarRow(
                    dataset_part=raw["dataset_part"],
                    relative_path=raw["relative_path"],
                    subject_id=raw["subject_id"],
                    device=raw["device"],
                    ppi=int(ppi_text) if ppi_text else None,
                    capture=raw["capture"],
                    fgp=fgp,
                )
            )
    if not rows:
        raise Nist302PairError("exemplar manifest is empty")
    keys = [(row.dataset_part, row.relative_path) for row in rows]
    if len(keys) != len(set(keys)):
        raise Nist302PairError("duplicate exemplar paths in manifest")
    return tuple(rows)


def build_latent_exemplar_pairs(
    latent_rows: Iterable[Nist302AnnotatedRow],
    exemplar_rows: Iterable[Nist302ExemplarRow],
    *,
    dataset_parts: Sequence[str] = ("sd302a", "sd302b", "sd302d"),
    captures: Sequence[str] = ("roll", "plain", "slap"),
    require_candidate: bool = True,
) -> Tuple[Nist302LatentExemplarPair, ...]:
    """Return every declared same-subject, same-FGP candidate association."""

    allowed_parts = set(dataset_parts)
    allowed_captures = set(captures)
    index: Dict[Tuple[str, int], list[Nist302ExemplarRow]] = {}
    for exemplar in exemplar_rows:
        if exemplar.dataset_part not in allowed_parts or exemplar.capture not in allowed_captures:
            continue
        index.setdefault((exemplar.subject_id, exemplar.fgp), []).append(exemplar)

    pairs = []
    missing = []
    for latent in latent_rows:
        if latent.fgp is None:
            continue
        candidates = sorted(
            index.get((latent.subject_id, latent.fgp), ()),
            key=lambda row: (
                row.dataset_part,
                row.capture,
                -(row.ppi or 0),
                row.device,
                row.relative_path,
            ),
        )
        if not candidates:
            missing.append(latent.sample_id)
            continue
        for candidate in candidates:
            exemplar_key = f"{candidate.dataset_part}:{candidate.relative_path}"
            pairs.append(
                Nist302LatentExemplarPair(
                    pair_id=f"{latent.sample_id}::exemplar::{exemplar_key}",
                    latent_sample_id=latent.sample_id,
                    subject_id=latent.subject_id,
                    split=latent.split,
                    fgp=latent.fgp,
                    latent_relative_path=latent.image_relative_path,
                    lffs_filename=latent.lffs_filename,
                    assessment=latent.assessment,
                    exemplar_dataset_part=candidate.dataset_part,
                    exemplar_relative_path=candidate.relative_path,
                    exemplar_device=candidate.device,
                    exemplar_ppi=candidate.ppi,
                    exemplar_capture=candidate.capture,
                )
            )
    if require_candidate and missing:
        raise Nist302PairError(
            f"{len(missing)} known-FGP latent samples have no exemplar candidate; "
            f"first: {missing[:3]}"
        )
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise Nist302PairError("latent/exemplar pair identifiers are not unique")
    return tuple(pairs)


def write_pair_manifest(pairs: Sequence[Nist302LatentExemplarPair], path: Path) -> None:
    """Write the long-form association table without absolute machine paths."""

    if not pairs:
        raise Nist302PairError("cannot write an empty pair manifest")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(pair) for pair in pairs]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
