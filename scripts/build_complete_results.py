#!/usr/bin/env python3
"""Build the complete public results package without redistributing biometrics.

This script generates aggregate figures from committed metrics, copies the
complete set of statistical result JSON files, and assembles final tables and
table-bearing reports under ``results/``. It intentionally excludes previews
that contain NIST latent or exemplar images.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
STATISTICS = RESULTS / "statistics"


def run_script(script: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/fingerprint-results-matplotlib")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".csv", ".json", ".md", ".svg"}:
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        shutil.copyfile(source, destination)


def generate_existing_analyses() -> list[dict[str, str]]:
    run_script("build_curated_results.py")
    figure_rows: list[dict[str, str]] = []
    deterministic_runs = [
        ("unet_full_mps", "U-Net", "socofing_unet"),
        ("gated_full_mps", "Gated convolution", "socofing_gated"),
        ("gated_ridge_hybrid_full", "Gated + structural losses", "socofing_gated_structural"),
    ]
    for run, label, destination in deterministic_runs:
        output = FIGURES / destination
        run_script(
            "plot_unet_results.py",
            "--metrics", str(ROOT / "outputs" / run / "metrics.json"),
            "--output", str(output),
            "--label", label,
        )
        for filename, category in [
            ("training-curve.png", "training"),
            ("performance-vs-observed-fraction.png", "observed_fraction"),
            ("performance-by-mask-family.png", "mask_geometry"),
        ]:
            figure_rows.append({"figure": f"{destination}/{filename}", "category": category, "source": f"outputs/{run}/metrics.json"})

    run_script(
        "plot_ddim_results.py",
        "--input", str(ROOT / "outputs/ddim_50step_k10_validation/per-image-probabilistic-metrics.csv"),
        "--output", str(FIGURES / "ddim_trends_vs_observed_fraction.png"),
    )
    figure_rows.append({"figure": "ddim_trends_vs_observed_fraction.png", "category": "observed_fraction_and_uncertainty", "source": "outputs/ddim_50step_k10_validation/per-image-probabilistic-metrics.csv"})

    run_script(
        "plot_sampler_comparison.py",
        "--ddim", str(ROOT / "outputs/ddim50_k5_balanced90/per-image-probabilistic-metrics.csv"),
        "--repaint", str(ROOT / "outputs/repaint_u2_k5_balanced90/per-image-probabilistic-metrics.csv"),
        "--output", str(FIGURES / "ddim_vs_repaint.png"),
    )
    figure_rows.append({"figure": "ddim_vs_repaint.png", "category": "sampler_quality_uncertainty_cost", "source": "outputs/ddim50_k5_balanced90; outputs/repaint_u2_k5_balanced90"})

    with tempfile.TemporaryDirectory(prefix="fingerprint-results-") as temporary_text:
        temporary = Path(temporary_text)
        analyses = [
            ("factorial", "outputs/residual_factorial_nested12_k3_s25/factorial-per-condition.csv", 302),
            ("rq6_central_vs_peripheral", "outputs/rq6_central_vs_peripheral_roi_nested12_k3_s25/factorial-per-condition.csv", 303),
        ]
        for name, source, seed in analyses:
            output = temporary / name
            run_script(
                "analyze_factorial_benchmark.py",
                "--input", str(ROOT / source),
                "--output", str(output),
                "--bootstrap", "2000",
                "--seed", str(seed),
            )
            for filename, category in [
                ("geometry-by-r.png", "mask_geometry_by_observed_fraction"),
                ("marginal-performance-vs-r.png", "reconstructibility"),
                ("nominal-vs-effective-roi-fraction.png", "mask_semantics"),
            ]:
                destination = FIGURES / f"{name}_{filename}"
                copy(output / filename, destination)
                figure_rows.append({"figure": destination.name, "category": category, "source": source})
            copy(output / "factorial-analysis.json", STATISTICS / f"{name}_analysis.json")

        k_output = temporary / "k"
        k_arguments: list[str] = []
        for k in (5, 10, 20, 50):
            k_arguments.extend(["--run", str(k), str(ROOT / f"outputs/residual_uncertainty_k{k}_paired30")])
        run_script("analyze_uncertainty_k.py", *k_arguments, "--output", str(k_output))
        copy(k_output / "k-scaling.png", FIGURES / "monte_carlo_k_scaling.png")
        copy(k_output / "k-scaling.csv", TABLES / "residual_uncertainty_k_scaling.csv")
        copy(k_output / "k-scaling.json", STATISTICS / "residual_uncertainty_k_scaling.json")
        figure_rows.append({"figure": "monte_carlo_k_scaling.png", "category": "sample_count_calibration_cost", "source": "outputs/residual_uncertainty_k{5,10,20,50}_paired30"})

        strata_output = temporary / "strata"
        strata_source = "outputs/residual_factorial_nested12_k3_s25/factorial-per-condition.csv"
        run_script(
            "analyze_residual_strata.py",
            "--input", str(ROOT / strata_source),
            "--output", str(strata_output),
        )
        for filename, category in [
            ("metrics-vs-effective-roi-fraction.png", "effective_information"),
            ("metrics-by-mask-family.png", "mask_geometry"),
        ]:
            copy(strata_output / filename, FIGURES / filename)
            figure_rows.append({"figure": filename, "category": category, "source": strata_source})
        copy(strata_output / "mask-family-summary.csv", TABLES / "residual_mask_family_summary.csv")
        copy(strata_output / "stratified-summary.json", STATISTICS / "residual_stratified_summary.json")

    distance_source = "outputs/nist302_registered_baseline_validation/per-image-metrics.csv"
    run_script(
        "plot_nist302_distance_strata.py",
        "--metrics", str(ROOT / distance_source),
        "--output", str(FIGURES / "nist302_registered_distance_strata.png"),
    )
    figure_rows.append({"figure": "nist302_registered_distance_strata.png", "category": "registration_distance", "source": distance_source})
    return figure_rows


def load_metrics(path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for key in keys:
        payload = payload[key]
    return payload


def mean(metrics: dict[str, Any], key: str) -> float:
    value = metrics[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def generate_additional_figures() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    colors = {"zero": "#1F3A63", "fine": "#B0413E", "boundary": "#12706B"}
    sources = {
        "zero": "outputs/nist302_TEST_FINAL_zeroshot_deterministic/metrics.json",
        "fine": "outputs/nist302_TEST_FINAL_finetuned_support_spectrum/metrics.json",
        "boundary": "outputs/nist302_TEST_FINAL_boundary_continuity/metrics.json",
    }
    labels = {"zero": "Zero-shot", "fine": "Fine-tuned", "boundary": "+ Boundary continuity"}
    metrics = {key: load_metrics(source, ("evaluation", "metrics")) for key, source in sources.items()}
    specifications = [
        ("heldout_q1_mae", "MAE", False),
        ("heldout_q1_ssim_map_mean", "SSIM", True),
        ("heldout_q1_orientation_error", "Orientation error", False),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (metric, title, higher_better) in zip(axes, specifications):
        values = [mean(metrics[key], metric) for key in sources]
        bars = axis.bar([labels[key] for key in sources], values, color=[colors[key] for key in sources])
        axis.set_title(f"{title} ({'higher' if higher_better else 'lower'} is better)")
        axis.tick_params(axis="x", rotation=15)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    figure.suptitle("NIST SD302 held-out real-pixel evaluation (217 images)")
    figure.text(0.5, 0.01, "Split/test flags differ across source artefacts; see results table provenance.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    destination = FIGURES / "nist302_target_regression_and_boundary_recovery.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    rows.append({"figure": destination.name, "category": "evaluation_target_ablation", "source": "; ".join(sources.values())})

    distribution_specs = [
        ("Zero-shot", "outputs/nist302_TEST_FINAL_zeroshot_deterministic/per-image-metrics.csv", "heldout_q1_mae"),
        ("RePaint", "outputs/nist302_TEST_FINAL_repaint/per-image-probabilistic-metrics.csv", "mean_k5_heldout_q1_mae"),
        ("Residual DDPM", "outputs/nist302_TEST_FINAL_residual_ddpm/per-image-probabilistic-metrics.csv", "mean_k10_heldout_q1_mae"),
        ("CVAE", "outputs/nist302_TEST_FINAL_cvae_spatial/per-image-probabilistic-metrics.csv", "mean_k50_heldout_q1_mae"),
        ("DDIM-20", "outputs/nist302_TEST_FINAL_ddim/per-image-probabilistic-metrics.csv", "mean_k10_heldout_q1_mae"),
    ]
    distributions: list[np.ndarray] = []
    for _, source, field in distribution_specs:
        with (ROOT / source).open(newline="", encoding="utf-8") as handle:
            values = [float(row[field]) for row in csv.DictReader(handle) if row[field] not in {"", "nan"}]
        distributions.append(np.asarray(values, dtype=float))
    figure, axis = plt.subplots(figsize=(10, 5.6))
    box = axis.boxplot(distributions, tick_labels=[item[0] for item in distribution_specs], patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#1F3A63", "#12706B", "#2E8B57", "#6A4C93", "#B0413E"]):
        patch.set_facecolor(color); patch.set_alpha(0.82)
    axis.set_ylabel("Held-out quality==1 MAE")
    axis.set_title("Distribution of per-image reconstruction error")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    destination = FIGURES / "nist302_mae_distributions.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    rows.append({"figure": destination.name, "category": "metric_distributions", "source": "; ".join(item[1] for item in distribution_specs)})

    ddim = load_metrics("outputs/nist302_TEST_FINAL_ddim/metrics.json", ("metrics", "metrics"))
    residual = load_metrics("outputs/nist302_TEST_FINAL_residual_ddpm/metrics.json", ("metrics", "metrics"))
    k_values = np.asarray([10, 20, 50])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for payload, label, color in [(ddim, "DDIM-20", "#1F3A63"), (residual, "Residual DDPM", "#12706B")]:
        axes[0].plot(k_values, [mean(payload, f"mean_k{k}_heldout_q1_mae") for k in k_values], "o-", label=label, color=color)
        axes[1].plot(k_values, [mean(payload, f"uncertainty_error_spearman_k{k}") for k in k_values], "o-", label=label, color=color)
        axes[2].plot(k_values, [mean(payload, f"best_k{k}_heldout_q1_mae") for k in k_values], "o-", label=label, color=color)
    titles = ["Predictive-mean MAE", "Uncertainty-error Spearman", "Best-of-K MAE (oracle diagnostic)"]
    for axis, title in zip(axes, titles):
        axis.set_title(title); axis.set_xlabel("K"); axis.set_xticks(k_values); axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Effect of Monte-Carlo sample count K")
    figure.tight_layout()
    destination = FIGURES / "nist302_k_sensitivity.png"
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    rows.append({"figure": destination.name, "category": "sample_count_ablation", "source": "outputs/nist302_TEST_FINAL_ddim/metrics.json; outputs/nist302_TEST_FINAL_residual_ddpm/metrics.json"})
    return rows


def copy_safe_showcases() -> list[dict[str, str]]:
    mappings = {
        "generalist_baseline.png": "socofing_generalist_baseline.png",
        "grid_socofing.png": "socofing_reconstruction_grid.png",
        "mask_families.png": "socofing_mask_families.png",
        "repaint_socofing.png": "socofing_repaint_samples.png",
    }
    rows = []
    for source_name, destination_name in mappings.items():
        copy(ROOT / "docs/poster_assets" / source_name, FIGURES / destination_name)
        rows.append({"figure": destination_name, "category": "qualitative_socofing", "source": f"docs/poster_assets/{source_name}"})
    return rows


def copy_statistical_sources() -> list[dict[str, str]]:
    destination = STATISTICS / "source_json"
    destination.mkdir(parents=True, exist_ok=True)
    keywords = (
        "test", "comparison", "paired", "omnibus", "calibr", "distance_diag",
        "reliability", "reconstructibility", "segmented", "conformal",
        "pseudo_target", "permuted_observed", "target_ranking", "selective",
    )
    selected = sorted(
        path for path in (ROOT / "outputs").glob("*.json")
        if any(keyword in path.name.lower() for keyword in keywords)
    )
    rows = []
    for source in selected:
        copy(source, destination / source.name)
        rows.append({"file": f"source_json/{source.name}", "source": source.relative_to(ROOT).as_posix()})
    return rows


def copy_table_reports() -> list[dict[str, str]]:
    report_sources = [
        ROOT / "outputs/DETERMINISTIC_BASELINES.md",
        ROOT / "docs/nist302_ablation_master_table.md",
        ROOT / "docs/compute_cost_table.md",
        ROOT / "docs/socofing_reconstructibility_segmented.md",
    ]
    destination = TABLES / "reports"
    rows = []
    for source in report_sources:
        copy(source, destination / source.name)
        rows.append({"file": f"reports/{source.name}", "source": source.relative_to(ROOT).as_posix()})
    return rows


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    STATISTICS.mkdir(parents=True, exist_ok=True)
    figures = generate_existing_analyses() + generate_additional_figures() + copy_safe_showcases()
    existing_curated = [
        {"figure": "socofing_baselines.png", "category": "model_comparison", "source": "outputs/{classical_nearest_validation,unet_full_mps,gated_full_mps}/metrics.json"},
        {"figure": "nist302_model_mae.svg", "category": "model_comparison", "source": "results/tables/nist302_heldout_evaluation_217.csv"},
        {"figure": "uncertainty_vs_anchor_distance.png", "category": "uncertainty", "source": "outputs/nist302_{ddpm,repaint,cvae}_distance_diag_full.json"},
        {"figure": "selective_reconstruction.png", "category": "selective_prediction", "source": "outputs/multisignal_reliability.json"},
    ]
    write_index(FIGURES / "figure_index.csv", sorted(figures + existing_curated, key=lambda row: row["figure"]))
    statistical_sources = copy_statistical_sources()
    write_index(STATISTICS / "statistics_index.csv", statistical_sources)
    table_reports = copy_table_reports()
    table_files = sorted(path for path in TABLES.glob("*.csv") if path.name != "table_index.csv")
    table_rows = table_reports + [
        {"file": path.name, "source": "generated by scripts/build_complete_results.py or scripts/build_curated_results.py"}
        for path in table_files
    ]
    write_index(TABLES / "table_index.csv", sorted(table_rows, key=lambda row: row["file"]))
    print(f"Complete results package: {len(figures) + len(existing_curated)} figures, {len(table_rows)} tables/reports, {len(statistical_sources)} statistical source files")


if __name__ == "__main__":
    main()
