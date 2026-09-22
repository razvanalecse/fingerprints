"""SOCOFing discovery, validation, and reproducible manifest creation.

Only the original ``Real`` subset is accepted by the primary research pipeline.
The synthetically altered files distributed with SOCOFing are derivatives of
those originals and must never be independently split across partitions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from fingerprint_reconstruction.data.splits import GroupRecord, make_group_split


SOCOfING_FILENAME = re.compile(
    r"^(?P<subject>[0-9]+)__(?P<gender>[MF])_"
    r"(?P<hand>Left|Right)_"
    r"(?P<finger>thumb|index|middle|ring|little)_finger"
    r"(?P<suffix>[^.]*)\.(?P<extension>bmp)$",
    flags=re.IGNORECASE,
)

FINGER_ORDER = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "little": 5}
HAND_ORDER = {"Left": 0, "Right": 1}
MANIFEST_SCHEMA_VERSION = 1


class SocofingValidationError(ValueError):
    """Raised when SOCOFing violates a declared data contract."""


@dataclass(frozen=True)
class ParsedSocofingName:
    subject_id: str
    gender: str
    hand: str
    finger: str
    suffix: str
    extension: str

    @property
    def finger_id(self) -> str:
        return f"{self.hand.lower()}_{self.finger}"

    @property
    def is_original(self) -> bool:
        return self.suffix == ""


@dataclass(frozen=True)
class SocofingRecord:
    sample_id: str
    relative_path: str
    subject_id: str
    finger_id: str
    gender: str
    hand: str
    finger: str
    width: int
    height: int
    source_mode: str
    grayscale_min: int
    grayscale_max: int
    grayscale_mean: float
    grayscale_std: float
    file_size_bytes: int
    sha256: str
    split: Optional[str] = None


@dataclass(frozen=True)
class SocofingAudit:
    schema_version: int
    dataset_name: str
    dataset_root: str
    real_directory: str
    num_images: int
    num_subjects: int
    num_unique_fingers: int
    image_shapes: Mapping[str, int]
    source_modes: Mapping[str, int]
    gender_counts: Mapping[str, int]
    hand_counts: Mapping[str, int]
    split_image_counts: Mapping[str, int]
    split_subject_counts: Mapping[str, int]
    dataset_sha256: str
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class SocofingManifest:
    records: Tuple[SocofingRecord, ...]
    audit: SocofingAudit


def parse_socofing_filename(filename: str, *, require_original: bool = True) -> ParsedSocofingName:
    """Parse a SOCOFing filename without relying on its parent directory."""

    match = SOCOfING_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        raise SocofingValidationError(f"invalid SOCOFing filename: {filename}")

    groups = match.groupdict()
    parsed = ParsedSocofingName(
        subject_id=f"{int(groups['subject']):06d}",
        gender=groups["gender"].upper(),
        hand=groups["hand"].capitalize(),
        finger=groups["finger"].lower(),
        suffix=groups["suffix"],
        extension=groups["extension"].lower(),
    )
    if require_original and not parsed.is_original:
        raise SocofingValidationError(
            f"derived/altered file is not allowed in the original subset: {filename}"
        )
    return parsed


def _find_real_directory(root: Path) -> Path:
    candidates = (root, root / "Real", root / "SOCOFing" / "Real")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.BMP")):
            return candidate.resolve()
        if candidate.is_dir() and any(candidate.glob("*.bmp")):
            return candidate.resolve()
    raise SocofingValidationError(
        "could not locate the SOCOFing Real directory; expected ROOT/Real, "
        "ROOT/SOCOFing/Real, or a directory containing original BMP files"
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample_id(parsed: ParsedSocofingName) -> str:
    return f"socofing:{parsed.subject_id}:{parsed.finger_id}"


def _read_record(path: Path, real_directory: Path) -> SocofingRecord:
    parsed = parse_socofing_filename(path.name, require_original=True)
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            source_mode = image.mode
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                gray_image = Image.alpha_composite(background, rgba).convert("L")
            else:
                gray_image = image.convert("L")
            gray = np.asarray(gray_image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as error:
        raise SocofingValidationError(f"cannot decode image {path}: {error}") from error

    if gray.shape != (height, width) or width <= 0 or height <= 0:
        raise SocofingValidationError(f"invalid decoded dimensions for {path}")

    return SocofingRecord(
        sample_id=_stable_sample_id(parsed),
        relative_path=path.relative_to(real_directory).as_posix(),
        subject_id=parsed.subject_id,
        finger_id=parsed.finger_id,
        gender=parsed.gender,
        hand=parsed.hand,
        finger=parsed.finger,
        width=width,
        height=height,
        source_mode=source_mode,
        grayscale_min=int(gray.min()),
        grayscale_max=int(gray.max()),
        grayscale_mean=float(gray.mean(dtype=np.float64)),
        grayscale_std=float(gray.std(dtype=np.float64)),
        file_size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _counter(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _dataset_digest(records: Sequence[SocofingRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_identity_contract(records: Sequence[SocofingRecord]) -> None:
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted({x for x in sample_ids if sample_ids.count(x) > 1})
        raise SocofingValidationError(
            f"duplicate subject/finger originals detected: {duplicates[:5]}"
        )

    subject_gender: Dict[str, set] = {}
    for record in records:
        subject_gender.setdefault(record.subject_id, set()).add(record.gender)
    inconsistent = [subject for subject, genders in subject_gender.items() if len(genders) > 1]
    if inconsistent:
        raise SocofingValidationError(
            f"inconsistent gender labels within subjects: {inconsistent[:5]}"
        )


def build_socofing_manifest(
    root: Path,
    *,
    split_fractions: Sequence[float] = (0.70, 0.15, 0.15),
    split_seed: int = 1729,
    expected_images: Optional[int] = 6000,
    expected_subjects: Optional[int] = 600,
    expected_fingers_per_subject: Optional[int] = 10,
    strict_expected_counts: bool = True,
) -> SocofingManifest:
    """Discover originals, validate them, and assign leakage-safe splits."""

    root = Path(root).expanduser().resolve()
    real_directory = _find_real_directory(root)
    paths = sorted(
        (path for path in real_directory.iterdir() if path.suffix.lower() == ".bmp"),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        raise SocofingValidationError(f"no BMP files found in {real_directory}")

    records = [_read_record(path, real_directory) for path in paths]
    records.sort(
        key=lambda record: (
            int(record.subject_id),
            HAND_ORDER[record.hand],
            FINGER_ORDER[record.finger],
        )
    )
    _validate_identity_contract(records)

    subjects = sorted({record.subject_id for record in records})
    warnings: List[str] = []
    observed_shapes = _counter(f"{record.height}x{record.width}" for record in records)
    observed_modes = _counter(record.source_mode for record in records)
    if len(observed_shapes) > 1:
        warnings.append(f"multiple decoded image shapes observed: {observed_shapes}")
    if len(observed_modes) > 1:
        warnings.append(f"multiple decoded source modes observed: {observed_modes}")
    expectations = (
        (expected_images, len(records), "images"),
        (expected_subjects, len(subjects), "subjects"),
    )
    for expected, actual, label in expectations:
        if expected is not None and expected != actual:
            message = f"expected {expected} {label}, found {actual}"
            if strict_expected_counts:
                raise SocofingValidationError(message)
            warnings.append(message)

    if expected_fingers_per_subject is not None:
        finger_counts = _counter(record.subject_id for record in records)
        incomplete = {
            subject: count
            for subject, count in finger_counts.items()
            if count != expected_fingers_per_subject
        }
        if incomplete:
            message = (
                f"{len(incomplete)} subjects do not have "
                f"{expected_fingers_per_subject} original fingers"
            )
            if strict_expected_counts:
                raise SocofingValidationError(message)
            warnings.append(message)

    grouped = [
        GroupRecord(record.sample_id, record.subject_id, record.finger_id)
        for record in records
    ]
    split = make_group_split(
        grouped,
        group_level="subject",
        fractions=split_fractions,
        seed=split_seed,
    )
    records = [
        SocofingRecord(**{**asdict(record), "split": split.assignments[record.sample_id]})
        for record in records
    ]

    split_subjects: Dict[str, set] = {"train": set(), "validation": set(), "test": set()}
    for record in records:
        assert record.split is not None
        split_subjects[record.split].add(record.subject_id)

    audit = SocofingAudit(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_name="SOCOFing",
        dataset_root=str(root),
        real_directory=str(real_directory),
        num_images=len(records),
        num_subjects=len(subjects),
        num_unique_fingers=len({(r.subject_id, r.finger_id) for r in records}),
        image_shapes=observed_shapes,
        source_modes=observed_modes,
        gender_counts=_counter(r.gender for r in records),
        hand_counts=_counter(r.hand for r in records),
        split_image_counts=_counter(str(r.split) for r in records),
        split_subject_counts={k: len(v) for k, v in split_subjects.items()},
        dataset_sha256=_dataset_digest(records),
        warnings=tuple(warnings),
    )
    return SocofingManifest(tuple(records), audit)


def _atomic_text_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_socofing_artifacts(manifest: SocofingManifest, output_directory: Path) -> Dict[str, Path]:
    """Write stable CSV/JSONL records and an audit report atomically."""

    output_directory = Path(output_directory).expanduser().resolve()
    csv_path = output_directory / "socofing_manifest.csv"
    jsonl_path = output_directory / "socofing_manifest.jsonl"
    audit_path = output_directory / "socofing_audit.json"
    fields = list(asdict(manifest.records[0]).keys())

    def write_csv(stream) -> None:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in manifest.records:
            writer.writerow(asdict(record))

    def write_jsonl(stream) -> None:
        for record in manifest.records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def write_audit(stream) -> None:
        json.dump(asdict(manifest.audit), stream, indent=2, sort_keys=True)
        stream.write("\n")

    _atomic_text_write(csv_path, write_csv)
    _atomic_text_write(jsonl_path, write_jsonl)
    _atomic_text_write(audit_path, write_audit)
    return {"csv": csv_path, "jsonl": jsonl_path, "audit": audit_path}
