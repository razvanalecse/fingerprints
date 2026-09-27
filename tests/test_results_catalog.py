import csv
import json
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_results_catalog.py"
SPEC = spec_from_file_location("build_results_catalog", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_catalogue_is_deterministic_and_excludes_images(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    run = repository / "outputs" / "experiment_a"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"mae": 0.1}\n', encoding="utf-8")
    (run / "per-image-metrics.csv").write_text("id,mae\n1,0.1\n", encoding="utf-8")
    (run / "preview.png").write_bytes(b"not-an-image")

    first = MODULE.discover_artifacts(repository / "outputs", repository)
    second = MODULE.discover_artifacts(repository / "outputs", repository)

    assert first == second
    assert [item["relative_path"] for item in first] == [
        "outputs/experiment_a/metrics.json",
        "outputs/experiment_a/per-image-metrics.csv",
    ]
    assert first[0]["role"] == "aggregate_metrics"
    assert first[1]["role"] == "unit_level_metrics"
    assert all(len(str(item["sha256"])) == 64 for item in first)


def test_catalogue_and_summary_are_machine_readable(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    outputs = repository / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "calibration.json").write_text("{}\n", encoding="utf-8")

    artifacts = MODULE.discover_artifacts(outputs, repository)
    catalogue = repository / "results" / "artifact-index.csv"
    summary = repository / "results" / "artifact-summary.json"
    MODULE.write_catalogue(artifacts, catalogue)
    MODULE.write_summary(artifacts, summary, catalogue, repository)

    with catalogue.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(summary.read_text(encoding="utf-8"))

    assert rows[0]["relative_path"] == "outputs/calibration.json"
    assert payload["artifact_count"] == 1
    assert payload["contains_raw_fingerprint_images"] is False
    assert payload["contains_model_weights"] is False
