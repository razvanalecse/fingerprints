"""Auditable NIST SD 302 discovery and latent/exemplar manifest creation.

The SD 302 latent PNGs and exemplar PNGs are *different impressions*.  A
same-finger association is therefore useful for identity/structure evaluation,
but it is not pixel-aligned reconstruction ground truth.  This module records
that distinction explicitly and never infers a finger position from subject ID
or hand alone.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from fingerprint_reconstruction.data.splits import GroupRecord, make_group_split


LATENT_PNG_NAME = re.compile(
    r"^(?P<subject>\d{8})_(?P<activity>[0-9A-Z]+)_(?P<hand>[LRX])_"
    r"(?P<encounter>[0-9A-Z]+)_(?P<technique>[A-Z]+)_"
    r"(?P<digitizer>[0-9A-Z]+)_(?P<ppi>\d+)PPI_(?P<bits>\d+)BPC_"
    r"(?P<channels>\d+)CH_LP(?P<latent_number>\d+)_"
    r"(?P<source>[1-4])\.png$"
)

LFFS_NAME = re.compile(
    r"^(?P<subject>\d{8})_(?P<activity>[0-9A-Z]+)_(?P<hand>[LRX])_"
    r"(?P<encounter>[0-9A-Z]+)_(?P<technique>[A-Z]+)_"
    r"(?P<digitizer>[0-9A-Z]+)_(?P<ppi>\d+)PPI_(?P<bits>\d+)BPC_"
    r"(?P<channels>\d+)CH_LP(?P<latent_number>\d+)-"
    r"(?P<impression>\d+)_(?P<source>[1-4])\.lffs$"
)

EXEMPLAR_WITH_PPI = re.compile(
    r"^(?P<subject>\d{8})_(?P<device>[A-Z])_(?P<ppi>\d+)_"
    r"(?P<capture>roll|plain|slap)_(?P<fgp>\d{2})\.png$"
)
EXEMPLAR_CHALLENGER = re.compile(
    r"^(?P<subject>\d{8})_(?P<device>[A-Z])_"
    r"(?P<capture>roll)_(?P<fgp>\d{2})\.png$"
)

MANIFEST_SCHEMA_VERSION = 1


class Nist302ValidationError(ValueError):
    """Raised when an SD 302 file violates a declared data contract."""


@dataclass(frozen=True)
class ParsedLatentName:
    subject_id: str
    activity: str
    hand: str
    encounter: str
    technique: str
    digitizer: str
    ppi: int
    bits_per_channel: int
    channels: int
    latent_number: int
    source_code: int

    @property
    def canonical_lift_key(self) -> str:
        """Key shared by full-resolution PNG and standardized 1000-PPI LFFS.

        Resolution and bit depth are deliberately omitted because the LFFS
        release standardizes them.  The PNG release has no impression index;
        multiple LFFS impressions therefore remain an explicit ambiguity.
        """

        return ":".join(
            (
                self.subject_id,
                self.activity,
                self.hand,
                self.encounter,
                self.technique,
                self.digitizer,
                str(self.channels),
                str(self.latent_number),
                str(self.source_code),
            )
        )


@dataclass(frozen=True)
class FingerPositionEntry:
    filename: str
    canonical_lift_key: str
    impression: int
    fgp: Optional[int]


@dataclass(frozen=True)
class ExemplarRecord:
    dataset_part: str
    relative_path: str
    subject_id: str
    device: str
    ppi: Optional[int]
    capture: str
    fgp: int


@dataclass(frozen=True)
class Nist302LatentRecord:
    sample_id: str
    subject_id: str
    hand: str
    activity: str
    encounter: str
    technique: str
    digitizer: str
    native_ppi: int
    native_bits_per_channel: int
    channels: int
    latent_number: int
    source_code: int
    original_masked_path: str
    original_unmasked_path: Optional[str]
    enhanced_masked_path: Optional[str]
    enhanced_unmasked_path: Optional[str]
    width: int
    height: int
    fgp: Optional[int]
    fgp_status: str
    lffs_filenames: Tuple[str, ...]
    errata_mentioned: bool
    exemplar_paths: Tuple[str, ...]
    pixel_aligned_ground_truth: bool
    split: str


@dataclass(frozen=True)
class Nist302Audit:
    schema_version: int
    dataset_name: str
    roots: Mapping[str, str]
    num_latent_pngs: int
    num_subjects: int
    num_known_fgp: int
    num_unknown_fgp: int
    num_ambiguous_fgp: int
    num_errata_mentions: int
    num_with_exemplar_candidates: int
    fgp_counts: Mapping[str, int]
    source_code_counts: Mapping[str, int]
    split_subject_counts: Mapping[str, int]
    split_sample_counts: Mapping[str, int]
    unmatched_png_names: Tuple[str, ...]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class Nist302Manifest:
    records: Tuple[Nist302LatentRecord, ...]
    exemplars: Tuple[ExemplarRecord, ...]
    audit: Nist302Audit


def parse_latent_png_filename(filename: str) -> ParsedLatentName:
    match = LATENT_PNG_NAME.fullmatch(Path(filename).name)
    if match is None:
        raise Nist302ValidationError(f"invalid SD302 latent PNG filename: {filename}")
    groups = match.groupdict()
    return ParsedLatentName(
        subject_id=groups["subject"],
        activity=groups["activity"],
        hand=groups["hand"],
        encounter=groups["encounter"],
        technique=groups["technique"],
        digitizer=groups["digitizer"],
        ppi=int(groups["ppi"]),
        bits_per_channel=int(groups["bits"]),
        channels=int(groups["channels"]),
        latent_number=int(groups["latent_number"]),
        source_code=int(groups["source"]),
    )


def parse_lffs_finger_position_filename(filename: str) -> Tuple[ParsedLatentName, int]:
    match = LFFS_NAME.fullmatch(Path(filename).name)
    if match is None:
        raise Nist302ValidationError(f"invalid SD302 LFFS filename: {filename}")
    groups = match.groupdict()
    parsed = ParsedLatentName(
        subject_id=groups["subject"],
        activity=groups["activity"],
        hand=groups["hand"],
        encounter=groups["encounter"],
        technique=groups["technique"],
        digitizer=groups["digitizer"],
        ppi=int(groups["ppi"]),
        bits_per_channel=int(groups["bits"]),
        channels=int(groups["channels"]),
        latent_number=int(groups["latent_number"]),
        source_code=int(groups["source"]),
    )
    return parsed, int(groups["impression"])


def load_finger_positions(path: Path) -> Mapping[str, Tuple[FingerPositionEntry, ...]]:
    grouped: Dict[str, list[FingerPositionEntry]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filename", "fgp"]:
            raise Nist302ValidationError(
                f"unexpected finger_positions schema {reader.fieldnames!r}"
            )
        for row in reader:
            parsed, impression = parse_lffs_finger_position_filename(row["filename"])
            raw_fgp = row["fgp"].strip()
            fgp = None if raw_fgp == "NA" else int(raw_fgp)
            if fgp is not None and not 1 <= fgp <= 10:
                raise Nist302ValidationError(f"non-distal FGP in finger_positions: {fgp}")
            grouped[parsed.canonical_lift_key].append(
                FingerPositionEntry(row["filename"], parsed.canonical_lift_key, impression, fgp)
            )
    return {key: tuple(sorted(values, key=lambda item: item.filename)) for key, values in grouped.items()}


def parse_exemplar_filename(filename: str, *, dataset_part: str, relative_path: str) -> ExemplarRecord:
    name = Path(filename).name
    match = EXEMPLAR_WITH_PPI.fullmatch(name)
    if match is None:
        match = EXEMPLAR_CHALLENGER.fullmatch(name)
    if match is None:
        raise Nist302ValidationError(f"invalid SD302 exemplar filename: {filename}")
    groups = match.groupdict()
    return ExemplarRecord(
        dataset_part=dataset_part,
        relative_path=relative_path,
        subject_id=groups["subject"],
        device=groups["device"],
        ppi=int(groups["ppi"]) if groups.get("ppi") else None,
        capture=groups["capture"],
        fgp=int(groups["fgp"]),
    )


def discover_exemplars(roots: Mapping[str, Path]) -> Tuple[ExemplarRecord, ...]:
    records = []
    for dataset_part, root_value in sorted(roots.items()):
        root = Path(root_value).resolve()
        if not root.is_dir():
            raise Nist302ValidationError(f"missing exemplar root for {dataset_part}: {root}")
        for path in sorted(root.rglob("*.png")):
            # Unsegmented slap images use FGP 13--15 and are not single-finger exemplars.
            try:
                record = parse_exemplar_filename(
                    path.name, dataset_part=dataset_part, relative_path=path.relative_to(root).as_posix()
                )
            except Nist302ValidationError:
                continue
            if 1 <= record.fgp <= 10:
                records.append(record)
    return tuple(records)


def _read_errata_png_names(path: Optional[Path]) -> frozenset[str]:
    if path is None:
        return frozenset()
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return frozenset(re.findall(r"\b\d{8}_[A-Za-z0-9_]+_LP\d+_\d+\.png\b", text))


def _variant_path(root: Path, variant: str, filename: str, subject: str) -> Optional[str]:
    path = root / "latent" / "png" / Path(variant) / "full_resolution" / subject / filename
    return path.relative_to(root).as_posix() if path.is_file() else None


def _resolve_fgp(entries: Sequence[FingerPositionEntry]) -> Tuple[Optional[int], str]:
    known = {entry.fgp for entry in entries if entry.fgp is not None}
    if len(known) == 1 and all(entry.fgp in known for entry in entries):
        return next(iter(known)), "known"
    if not known:
        return None, "unknown"
    return None, "ambiguous_multiple_impressions"


def build_nist302_manifest(
    *,
    sd302e_root: Path,
    finger_positions_csv: Path,
    exemplar_roots: Mapping[str, Path],
    errata_path: Optional[Path] = None,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 302,
) -> Nist302Manifest:
    """Build a subject-disjoint manifest from the original masked PNG release."""

    latent_root = Path(sd302e_root).resolve()
    base = latent_root / "latent" / "png" / "original" / "masked" / "full_resolution"
    if not base.is_dir():
        raise Nist302ValidationError(f"missing SD302e original/masked/full_resolution: {base}")

    positions = load_finger_positions(Path(finger_positions_csv))
    exemplars = discover_exemplars(exemplar_roots)
    exemplars_by_finger: Dict[Tuple[str, int], list[ExemplarRecord]] = defaultdict(list)
    for exemplar in exemplars:
        exemplars_by_finger[(exemplar.subject_id, exemplar.fgp)].append(exemplar)
    errata_names = _read_errata_png_names(errata_path)

    staged = []
    unmatched = []
    for path in sorted(base.rglob("*.png")):
        parsed = parse_latent_png_filename(path.name)
        entries = positions.get(parsed.canonical_lift_key, ())
        if not entries:
            unmatched.append(path.name)
        fgp, fgp_status = _resolve_fgp(entries)
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise Nist302ValidationError(f"cannot decode latent {path}: {error}") from error

        candidates = exemplars_by_finger.get((parsed.subject_id, fgp), []) if fgp else []
        candidate_paths = tuple(
            f"{item.dataset_part}:{item.relative_path}"
            for item in sorted(candidates, key=lambda item: (item.dataset_part, item.relative_path))
        )
        staged.append(
            dict(
                sample_id=f"nist302:{parsed.canonical_lift_key}",
                subject_id=parsed.subject_id,
                hand=parsed.hand,
                activity=parsed.activity,
                encounter=parsed.encounter,
                technique=parsed.technique,
                digitizer=parsed.digitizer,
                native_ppi=parsed.ppi,
                native_bits_per_channel=parsed.bits_per_channel,
                channels=parsed.channels,
                latent_number=parsed.latent_number,
                source_code=parsed.source_code,
                original_masked_path=path.relative_to(latent_root).as_posix(),
                original_unmasked_path=_variant_path(latent_root, "original/unmasked", path.name, parsed.subject_id),
                enhanced_masked_path=_variant_path(latent_root, "enhanced/masked", path.name, parsed.subject_id),
                enhanced_unmasked_path=_variant_path(latent_root, "enhanced/unmasked", path.name, parsed.subject_id),
                width=width,
                height=height,
                fgp=fgp,
                fgp_status=fgp_status if entries else "no_lffs_mapping",
                lffs_filenames=tuple(entry.filename for entry in entries),
                errata_mentioned=path.name in errata_names,
                exemplar_paths=candidate_paths,
                pixel_aligned_ground_truth=False,
            )
        )

    split = make_group_split(
        [GroupRecord(item["sample_id"], item["subject_id"], str(item["fgp"] or "unknown")) for item in staged],
        group_level="subject",
        fractions=fractions,
        seed=seed,
    )
    records = tuple(
        Nist302LatentRecord(**item, split=split.assignments[item["sample_id"]]) for item in staged
    )

    fgp_counts = Counter(str(record.fgp) if record.fgp is not None else record.fgp_status for record in records)
    split_subjects: Dict[str, set[str]] = defaultdict(set)
    split_samples = Counter(record.split for record in records)
    for record in records:
        split_subjects[record.split].add(record.subject_id)
    warnings = [
        "Latent/exemplar associations are same-finger, different-impression pairs; pixel_aligned_ground_truth is false.",
        "LFFS may describe multiple impressions in one PNG; conflicting/partial FGP annotations are marked ambiguous.",
        "Any filename mentioned in errata is flagged for manual review, not silently corrected or removed.",
    ]
    if unmatched:
        warnings.append(f"{len(unmatched)} PNG files have no canonical LFFS finger-position mapping.")

    audit = Nist302Audit(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_name="NIST SD 302",
        roots={"sd302e": str(latent_root), **{key: str(Path(value).resolve()) for key, value in exemplar_roots.items()}},
        num_latent_pngs=len(records),
        num_subjects=len({record.subject_id for record in records}),
        num_known_fgp=sum(record.fgp_status == "known" for record in records),
        num_unknown_fgp=sum(record.fgp_status in {"unknown", "no_lffs_mapping"} for record in records),
        num_ambiguous_fgp=sum(record.fgp_status == "ambiguous_multiple_impressions" for record in records),
        num_errata_mentions=sum(record.errata_mentioned for record in records),
        num_with_exemplar_candidates=sum(bool(record.exemplar_paths) for record in records),
        fgp_counts=dict(sorted(fgp_counts.items())),
        source_code_counts=dict(sorted(Counter(str(record.source_code) for record in records).items())),
        split_subject_counts={key: len(value) for key, value in sorted(split_subjects.items())},
        split_sample_counts=dict(sorted(split_samples.items())),
        unmatched_png_names=tuple(sorted(unmatched)),
        warnings=tuple(warnings),
    )
    return Nist302Manifest(records, exemplars, audit)


def write_nist302_artifacts(manifest: Nist302Manifest, output_dir: Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = [asdict(record) for record in manifest.records]
    fieldnames = list(records[0]) if records else []
    with (output / "nist302_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("lffs_filenames", "exemplar_paths"):
                row[key] = json.dumps(row[key], separators=(",", ":"))
            writer.writerow(row)
    exemplar_rows = [asdict(record) for record in manifest.exemplars]
    if exemplar_rows:
        with (output / "nist302_exemplars.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exemplar_rows[0]))
            writer.writeheader()
            writer.writerows(exemplar_rows)
    (output / "nist302_audit.json").write_text(
        json.dumps(asdict(manifest.audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
