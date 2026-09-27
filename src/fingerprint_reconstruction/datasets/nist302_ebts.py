"""Parse SD 302 LFFS/COMP annotations through NIST's ``an2k2txt``.

SD 302h and SD 302i are ANSI/NIST-ITL transactions, not ordinary image
archives.  This module intentionally delegates binary transaction decoding to
NBIS and parses its documented formatted-text representation.  It does not
guess byte offsets in the original records.

Coordinates in EFS Type-9 fields are expressed in 0.01 mm units.  They must
not be interpreted as pixel coordinates without using the associated image
resolution and ROI offsets.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


FORMATTED_LINE = re.compile(
    r"^(?P<record>\d+)\.(?P<field_index>\d+)\.(?P<subfield>\d+)\."
    r"(?P<item>\d+) \[(?P<tag>\d+\.\d+)\]=(?P<value>.*?)(?:\x1f)?$"
)

QUALITY_LABELS: Mapping[int, str] = {
    0: "background",
    1: "debatable_ridge_flow",
    2: "definitive_ridge_flow_debatable_minutiae",
    3: "definitive_minutiae_debatable_ridge_edges",
    4: "definitive_ridge_edges_debatable_pores",
    5: "definitive_pores",
}


class Nist302EbtsError(ValueError):
    """Raised when an SD 302 transaction or decoded field is inconsistent."""


@dataclass(frozen=True)
class FormattedFieldItem:
    record_index: int
    field_index: int
    subfield_index: int
    item_index: int
    tag: str
    value: str


@dataclass(frozen=True)
class EfsRoi:
    width_0_01mm: int
    height_0_01mm: int
    horizontal_offset_0_01mm: int
    vertical_offset_0_01mm: int
    polygon_0_01mm: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class RidgeQualityMap:
    grid_size_0_01mm: int
    encoding: str
    rows: Tuple[str, ...]

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def counts(self) -> Mapping[int, int]:
        return {
            value: sum(row.count(str(value)) for row in self.rows)
            for value in QUALITY_LABELS
        }


@dataclass(frozen=True)
class EfsMinutia:
    x_0_01mm: int
    y_0_01mm: int
    direction_degrees: int
    kind: str
    position_uncertainty_radius_0_01mm: Optional[int]
    direction_uncertainty_degrees: Optional[int]


@dataclass(frozen=True)
class EfsRecord:
    record_index: int
    idc: str
    roi: EfsRoi
    assessment: Optional[str]
    ridge_quality: Optional[RidgeQualityMap]
    quality_format_recovered: bool
    minutiae: Tuple[EfsMinutia, ...]


@dataclass(frozen=True)
class CompSource:
    source_index: int
    transaction_kind: str
    record_index: int
    identifier: str
    image_index: int
    record_types: str
    filename: str
    dataset: str


@dataclass(frozen=True)
class CompFeatureReference:
    label: str
    feature_kind: str
    field_number: int
    feature_index: int
    x_0_01mm: int
    y_0_01mm: int


@dataclass(frozen=True)
class ExaminerComparison:
    target_idc: int
    determination: str
    status: str
    complex_comparison: bool


@dataclass(frozen=True)
class RelativeRotation:
    target_idc: int
    degrees: int


@dataclass(frozen=True)
class MatchedCorrespondence:
    label: str
    first: CompFeatureReference
    second: CompFeatureReference


@dataclass(frozen=True)
class Nist302Transaction:
    path: Optional[str]
    transaction_type: Optional[str]
    version: Optional[str]
    fields: Tuple[FormattedFieldItem, ...]
    efs_records: Tuple[EfsRecord, ...]
    comp_sources: Tuple[CompSource, ...]
    comp_feature_references: Mapping[int, Tuple[CompFeatureReference, ...]]
    examiner_comparisons: Mapping[int, Tuple[ExaminerComparison, ...]]
    relative_rotations: Mapping[int, Tuple[RelativeRotation, ...]]
    decoder_warnings: Tuple[str, ...] = ()


def parse_formatted_text(text: str, *, source_path: Optional[str] = None) -> Nist302Transaction:
    """Parse output produced by NBIS ``an2k2txt``.

    Unknown fields are retained in :attr:`Nist302Transaction.fields`, so the
    adapter is loss-aware when NIST adds optional EFS fields.
    """

    fields = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        match = FORMATTED_LINE.fullmatch(raw_line)
        if match is None:
            if raw_line.strip():
                raise Nist302EbtsError(
                    f"unrecognized an2k2txt line {line_number}: {raw_line[:120]!r}"
                )
            continue
        groups = match.groupdict()
        fields.append(
            FormattedFieldItem(
                record_index=int(groups["record"]),
                field_index=int(groups["field_index"]),
                subfield_index=int(groups["subfield"]),
                item_index=int(groups["item"]),
                tag=groups["tag"],
                value=groups["value"],
            )
        )
    if not fields:
        raise Nist302EbtsError("an2k2txt output contains no fields")

    records = _group_by_record(fields)
    transaction_type = _first_value(records.get(1, ()), "1.004")
    version = _first_value(records.get(1, ()), "1.002")
    efs_records = tuple(
        _parse_efs_record(record_index, record_fields)
        for record_index, record_fields in sorted(records.items())
        if any(item.tag == "9.001" for item in record_fields)
    )
    type2 = records.get(2, ())
    comp_sources = _parse_comp_sources(type2)
    feature_refs = {
        index: _parse_comp_feature_references(record_fields)
        for index, record_fields in sorted(records.items())
        if any(item.tag == "9.361" for item in record_fields)
    }
    comparisons = {
        index: _parse_examiner_comparisons(record_fields)
        for index, record_fields in sorted(records.items())
        if any(item.tag == "9.362" for item in record_fields)
    }
    rotations = {
        index: _parse_relative_rotations(record_fields)
        for index, record_fields in sorted(records.items())
        if any(item.tag == "9.363" for item in record_fields)
    }
    return Nist302Transaction(
        path=source_path,
        transaction_type=transaction_type,
        version=version,
        fields=tuple(fields),
        efs_records=efs_records,
        comp_sources=comp_sources,
        comp_feature_references=feature_refs,
        examiner_comparisons=comparisons,
        relative_rotations=rotations,
    )


def decode_transaction(path: Path, *, an2k2txt: Path) -> Nist302Transaction:
    """Decode one LFFS/COMP transaction in an isolated temporary directory."""

    source = Path(path).resolve()
    decoder = Path(an2k2txt).resolve()
    if not source.is_file():
        raise Nist302EbtsError(f"transaction not found: {source}")
    if not decoder.is_file():
        raise Nist302EbtsError(f"an2k2txt not found: {decoder}")
    with tempfile.TemporaryDirectory(prefix="sd302-an2k-") as directory:
        output = Path(directory) / "transaction.txt"
        process = subprocess.run(
            (str(decoder), str(source), str(output)),
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not output.is_file():
            raise Nist302EbtsError(
                f"an2k2txt failed for {source.name} (exit {process.returncode}): "
                f"{process.stderr.strip()}"
            )
        transaction = parse_formatted_text(
            output.read_text(encoding="utf-8", errors="strict"),
            source_path=str(source),
        )
    warnings = tuple(
        line.strip()
        for line in (process.stdout + "\n" + process.stderr).splitlines()
        if line.strip().startswith("WARNING")
    )
    return Nist302Transaction(
        **{**transaction.__dict__, "decoder_warnings": warnings}
    )


def efs_to_pixel(
    x_0_01mm: int,
    y_0_01mm: int,
    *,
    ppi: float,
    horizontal_offset_0_01mm: int = 0,
    vertical_offset_0_01mm: int = 0,
) -> Tuple[float, float]:
    """Convert ROI-relative EFS coordinates to full-image pixel coordinates.

    ANSI/NIST EFS coordinates have their origin at the ROI's upper-left corner;
    X grows rightward and Y grows downward.  The ROI offsets are therefore
    added when coordinates are placed in an uncropped source image.  Leave
    both offsets at zero for an image already cropped to the EFS ROI.
    """

    if ppi <= 0:
        raise Nist302EbtsError(f"ppi must be positive, got {ppi}")
    pixels_per_0_01mm = ppi / 2540.0
    return (
        (x_0_01mm + horizontal_offset_0_01mm) * pixels_per_0_01mm,
        (y_0_01mm + vertical_offset_0_01mm) * pixels_per_0_01mm,
    )


def match_correspondences(
    transaction: Nist302Transaction,
    first_record_index: int,
    second_record_index: int,
) -> Tuple[MatchedCorrespondence, ...]:
    """Pair official 9.361 features by their transaction-wide label."""

    first = {
        item.label: item
        for item in transaction.comp_feature_references.get(first_record_index, ())
    }
    second = {
        item.label: item
        for item in transaction.comp_feature_references.get(second_record_index, ())
    }
    return tuple(
        MatchedCorrespondence(label, first[label], second[label])
        for label in sorted(first.keys() & second.keys())
    )


def _group_by_record(
    fields: Iterable[FormattedFieldItem],
) -> Mapping[int, Tuple[FormattedFieldItem, ...]]:
    grouped: Dict[int, list[FormattedFieldItem]] = {}
    for item in fields:
        grouped.setdefault(item.record_index, []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _subfields(
    fields: Sequence[FormattedFieldItem], tag: str
) -> Tuple[Tuple[str, ...], ...]:
    selected = [item for item in fields if item.tag == tag]
    grouped: Dict[int, list[FormattedFieldItem]] = {}
    for item in selected:
        grouped.setdefault(item.subfield_index, []).append(item)
    return tuple(
        tuple(item.value for item in sorted(values, key=lambda entry: entry.item_index))
        for _, values in sorted(grouped.items())
    )


def _first_value(fields: Sequence[FormattedFieldItem], tag: str) -> Optional[str]:
    groups = _subfields(fields, tag)
    return groups[0][0] if groups and groups[0] else None


def _parse_int(value: str, *, field: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise Nist302EbtsError(f"non-integer value in {field}: {value!r}") from error


def _optional_int(value: str, *, field: str) -> Optional[int]:
    return None if value == "" else _parse_int(value, field=field)


def _parse_roi(fields: Sequence[FormattedFieldItem]) -> EfsRoi:
    values = _subfields(fields, "9.300")
    if len(values) != 1 or len(values[0]) < 2:
        raise Nist302EbtsError("Type-9 record has no valid 9.300 ROI")
    items = values[0]
    polygon: Tuple[Tuple[int, int], ...] = ()
    if len(items) >= 5 and items[4]:
        points = []
        for point in items[4].split("-"):
            try:
                x, y = point.split(",", maxsplit=1)
            except ValueError as error:
                raise Nist302EbtsError(f"invalid ROI point {point!r}") from error
            points.append((_parse_int(x, field="9.300"), _parse_int(y, field="9.300")))
        polygon = tuple(points)
    return EfsRoi(
        width_0_01mm=_parse_int(items[0], field="9.300"),
        height_0_01mm=_parse_int(items[1], field="9.300"),
        horizontal_offset_0_01mm=_parse_int(items[2], field="9.300") if len(items) >= 3 else 0,
        vertical_offset_0_01mm=_parse_int(items[3], field="9.300") if len(items) >= 4 else 0,
        polygon_0_01mm=polygon,
    )


def _parse_quality_map(
    fields: Sequence[FormattedFieldItem],
) -> Tuple[Optional[RidgeQualityMap], bool]:
    row_groups = _subfields(fields, "9.308")
    format_groups = _subfields(fields, "9.309")
    recovered = False
    if row_groups and not format_groups:
        # Eleven files in the 2026 SD302h release place ``GSZ, UNC, row`` in
        # one 9.308 subfield instead of emitting 9.309.  Recover only this
        # narrow, self-validating pattern and expose the recovery in metadata.
        candidates = [
            (index, group)
            for index, group in enumerate(row_groups)
            if len(group) == 3 and group[0].isdigit() and group[1] in {"UNC", "RLE"}
        ]
        if len(candidates) == 1:
            index, group = candidates[0]
            format_groups = ((group[0], group[1]),)
            row_groups = row_groups[:index] + ((group[2],),) + row_groups[index + 1 :]
            recovered = True
    if len(format_groups) == 1 and len(format_groups[0]) > 2:
        group = format_groups[0]
        pairs = tuple(group[index : index + 2] for index in range(0, len(group), 2))
        if len(group) % 2 == 0 and len(set(pairs)) == 1:
            format_groups = (pairs[0],)
            recovered = True
    rows = tuple(group[0] for group in row_groups if group)
    if not rows and not format_groups:
        return None, recovered
    if not rows or len(format_groups) != 1 or len(format_groups[0]) != 2:
        raise Nist302EbtsError("9.308 ridge-quality data and 9.309 format must coexist")
    grid_size = _parse_int(format_groups[0][0], field="9.309")
    encoding = format_groups[0][1]
    if not 1 <= grid_size <= 41:
        raise Nist302EbtsError(f"invalid 9.309 grid size: {grid_size}")
    if encoding not in {"UNC", "RLE"}:
        raise Nist302EbtsError(f"unsupported 9.309 encoding: {encoding!r}")
    if encoding == "UNC":
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise Nist302EbtsError("non-rectangular uncompressed 9.308 map")
        invalid = set("".join(rows)) - {str(value) for value in QUALITY_LABELS}
        if invalid:
            raise Nist302EbtsError(f"invalid ridge-quality codes: {sorted(invalid)}")
    return RidgeQualityMap(grid_size, encoding, rows), recovered


def _parse_minutiae(fields: Sequence[FormattedFieldItem]) -> Tuple[EfsMinutia, ...]:
    result = []
    for items in _subfields(fields, "9.331"):
        if len(items) < 4:
            raise Nist302EbtsError("9.331 minutia has fewer than four items")
        direction = _parse_int(items[2], field="9.331")
        if not 0 <= direction <= 359:
            raise Nist302EbtsError(f"invalid minutia direction: {direction}")
        result.append(
            EfsMinutia(
                x_0_01mm=_parse_int(items[0], field="9.331"),
                y_0_01mm=_parse_int(items[1], field="9.331"),
                direction_degrees=direction,
                kind=items[3],
                position_uncertainty_radius_0_01mm=(
                    _optional_int(items[4], field="9.331") if len(items) >= 5 else None
                ),
                direction_uncertainty_degrees=(
                    _optional_int(items[5], field="9.331") if len(items) >= 6 else None
                ),
            )
        )
    return tuple(result)


def _parse_efs_record(record_index: int, fields: Sequence[FormattedFieldItem]) -> EfsRecord:
    quality, recovered = _parse_quality_map(fields)
    return EfsRecord(
        record_index=record_index,
        idc=_first_value(fields, "9.002") or "",
        roi=_parse_roi(fields),
        assessment=_first_value(fields, "9.353"),
        ridge_quality=quality,
        quality_format_recovered=recovered,
        minutiae=_parse_minutiae(fields),
    )


def _parse_comp_sources(fields: Sequence[FormattedFieldItem]) -> Tuple[CompSource, ...]:
    result = []
    for items in _subfields(fields, "2.1406"):
        if len(items) != 8:
            raise Nist302EbtsError(f"2.1406 source has {len(items)} items, expected 8")
        result.append(
            CompSource(
                source_index=_parse_int(items[0], field="2.1406"),
                transaction_kind=items[1],
                record_index=_parse_int(items[2], field="2.1406"),
                identifier=items[3],
                image_index=_parse_int(items[4], field="2.1406"),
                record_types=items[5],
                filename=items[6],
                dataset=items[7],
            )
        )
    return tuple(result)


def _parse_comp_feature_references(
    fields: Sequence[FormattedFieldItem],
) -> Tuple[CompFeatureReference, ...]:
    result = []
    labels = set()
    for items in _subfields(fields, "9.361"):
        if len(items) != 6:
            raise Nist302EbtsError(f"9.361 reference has {len(items)} items, expected 6")
        if items[0] in labels:
            raise Nist302EbtsError(f"duplicate 9.361 correspondence label: {items[0]!r}")
        labels.add(items[0])
        result.append(
            CompFeatureReference(
                label=items[0],
                feature_kind=items[1],
                field_number=_parse_int(items[2], field="9.361"),
                feature_index=_parse_int(items[3], field="9.361"),
                x_0_01mm=_parse_int(items[4], field="9.361"),
                y_0_01mm=_parse_int(items[5], field="9.361"),
            )
        )
    return tuple(result)


def _parse_examiner_comparisons(
    fields: Sequence[FormattedFieldItem],
) -> Tuple[ExaminerComparison, ...]:
    result = []
    for items in _subfields(fields, "9.362"):
        if len(items) < 3:
            raise Nist302EbtsError(f"9.362 comparison has {len(items)} items, expected >=3")
        result.append(
            ExaminerComparison(
                target_idc=_parse_int(items[0], field="9.362"),
                determination=items[1],
                status=items[2],
                complex_comparison=len(items) >= 9 and items[8] == "COMPLEX",
            )
        )
    return tuple(result)


def _parse_relative_rotations(
    fields: Sequence[FormattedFieldItem],
) -> Tuple[RelativeRotation, ...]:
    result = []
    for items in _subfields(fields, "9.363"):
        if len(items) != 2:
            raise Nist302EbtsError(f"9.363 rotation has {len(items)} items, expected 2")
        degrees = _parse_int(items[1], field="9.363")
        if not -179 <= degrees <= 180:
            raise Nist302EbtsError(f"9.363 rotation outside [-179, 180]: {degrees}")
        result.append(
            RelativeRotation(
                target_idc=_parse_int(items[0], field="9.363"),
                degrees=degrees,
            )
        )
    return tuple(result)
