#!/usr/bin/env python3
"""Build publication-facing tables and aggregate figures from run artefacts.

The builder reads only JSON metrics already stored under ``outputs/``. It does
not read, copy, or render fingerprint images. Outputs are deterministic CSV,
JSON, Markdown, and SVG files under ``results/``.
"""

from __future__ import annotations

import csv
import json
import shutil
from html import escape
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUTS = REPOSITORY / "outputs"
RESULTS = REPOSITORY / "results"


def load(relative_path: str) -> dict[str, Any]:
    return json.loads((REPOSITORY / relative_path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_mean(metrics: dict[str, Any], key: str) -> float | str:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean", "")
    return "" if value is None else value


def metric_ci(metrics: dict[str, Any], key: str, bound: str) -> float | str:
    value = metrics.get(key)
    return value.get(bound, "") if isinstance(value, dict) else ""


def build_socofing_table() -> None:
    configurations = [
        ("Nearest-observed interpolation", "outputs/classical_nearest_validation/metrics.json", "parameters"),
        ("U-Net", "outputs/unet_full_mps/metrics.json", "model_parameters"),
        ("Gated convolution", "outputs/gated_full_mps/metrics.json", "model_parameters"),
    ]
    rows: list[dict[str, Any]] = []
    for model, source, parameter_key in configurations:
        payload = load(source)
        metrics = payload["evaluation"]["overall"]["metrics"]
        rows.append(
            {
                "model": model,
                "parameters": payload.get(parameter_key, 0),
                "n_images": int(metrics["missing_roi_mae"]["finite_count"]),
                "missing_roi_mae": metric_mean(metrics, "missing_roi_mae"),
                "missing_roi_psnr_db": metric_mean(metrics, "missing_roi_psnr"),
                "missing_roi_ssim": metric_mean(metrics, "missing_roi_ssim_map_mean"),
                "missing_roi_orientation_error": metric_mean(metrics, "missing_roi_orientation_error"),
                "observed_mae": metric_mean(metrics, "observed_mae"),
                "source": source,
            }
        )
    write_csv(
        RESULTS / "tables" / "socofing_deterministic_baselines.csv",
        rows,
        list(rows[0]),
    )


def nist_row(
    model: str,
    family: str,
    source: str,
    metrics_path: tuple[str, ...],
    keys: dict[str, str],
    k: int | str,
) -> dict[str, Any]:
    payload = load(source)
    metrics: dict[str, Any] = payload
    for component in metrics_path:
        metrics = metrics[component]
    mae_key = keys["mae"]
    n_images = payload.get("images") or payload.get("validation_images")
    if not n_images:
        candidate = metrics.get(mae_key, {})
        n_images = candidate.get("finite_count", "") if isinstance(candidate, dict) else ""
    return {
        "model": model,
        "family": family,
        "K": k,
        "n_images": n_images,
        "mae_mean": metric_mean(metrics, mae_key),
        "mae_ci95_low": metric_ci(metrics, mae_key, "ci95_low"),
        "mae_ci95_high": metric_ci(metrics, mae_key, "ci95_high"),
        "psnr_db_mean": metric_mean(metrics, keys["psnr"]),
        "ssim_mean": metric_mean(metrics, keys["ssim"]),
        "orientation_error_mean": metric_mean(metrics, keys["orientation"]),
        "ridge_frequency_relative_mae": metric_mean(metrics, keys["frequency"]),
        "uncertainty_error_spearman": metric_mean(metrics, keys.get("uncertainty", "")),
        "reported_split": payload.get("split", "not_recorded"),
        "test_loaded": payload.get("test_loaded", "not_recorded"),
        "source": source,
    }


def build_nist302_table() -> list[dict[str, Any]]:
    deterministic_keys = {
        "mae": "heldout_q1_mae",
        "psnr": "heldout_q1_psnr",
        "ssim": "heldout_q1_ssim_map_mean",
        "orientation": "heldout_q1_orientation_error",
        "frequency": "heldout_q1_ridge_frequency_relative_mae",
    }
    probabilistic_keys = lambda k: {
        "mae": f"mean_k{k}_heldout_q1_mae",
        "psnr": f"mean_k{k}_heldout_q1_psnr",
        "ssim": f"mean_k{k}_heldout_q1_ssim_map_mean",
        "orientation": "mean_heldout_q1_orientation_error",
        "frequency": "mean_heldout_q1_ridge_frequency_relative_mae",
        "uncertainty": f"uncertainty_error_spearman_k{k}",
    }
    specs = [
        ("RePaint", "probabilistic", "outputs/nist302_TEST_FINAL_repaint/metrics.json", ("metrics", "metrics"), probabilistic_keys(5), 5),
        ("Zero-shot gated convolution", "deterministic", "outputs/nist302_TEST_FINAL_zeroshot_deterministic/metrics.json", ("evaluation", "metrics"), deterministic_keys, "1"),
        ("Residual DDPM", "probabilistic", "outputs/nist302_TEST_FINAL_residual_ddpm/metrics.json", ("metrics", "metrics"), probabilistic_keys(10), 10),
        ("Boundary-continuity model", "deterministic", "outputs/nist302_TEST_FINAL_boundary_continuity/metrics.json", ("evaluation", "metrics"), deterministic_keys, "1"),
        ("Spatial CVAE", "probabilistic", "outputs/nist302_TEST_FINAL_cvae_spatial/metrics.json", ("probabilistic_evaluation", "metrics"), probabilistic_keys(50), 50),
        ("Fine-tuned support+spectrum", "deterministic", "outputs/nist302_TEST_FINAL_finetuned_support_spectrum/metrics.json", ("evaluation", "metrics"), deterministic_keys, "1"),
        ("Pixel DDPM / DDIM-20", "probabilistic", "outputs/nist302_TEST_FINAL_ddim/metrics.json", ("metrics", "metrics"), probabilistic_keys(10), 10),
        ("Flow matching", "probabilistic", "outputs/nist302_TEST_FINAL_flow_matching/metrics.json", ("metrics", "metrics"), probabilistic_keys(10), 10),
        ("Latent DDPM", "probabilistic", "outputs/nist302_TEST_FINAL_latent_diffusion_plain/metrics.json", ("metrics", "metrics"), probabilistic_keys(10), 10),
        ("Latent DDPM, structure-guided", "probabilistic", "outputs/nist302_TEST_FINAL_latent_diffusion_structure_guided/metrics.json", ("metrics", "metrics"), probabilistic_keys(10), 10),
    ]
    rows = [nist_row(*spec) for spec in specs]

    ensemble_source = "outputs/nist302_TEST_FINAL_ensemble/metrics.json"
    ensemble_payload = load(ensemble_source)
    ensemble_metrics = ensemble_payload["evaluation"]["metrics"]
    rows.insert(
        3,
        {
            "model": "Learned ensemble",
            "family": "probabilistic ensemble",
            "K": ensemble_payload.get("base_model_k", "not_recorded"),
            "n_images": ensemble_payload.get("images", ""),
            "mae_mean": metric_mean(ensemble_metrics, "learned_ensemble_mae"),
            "mae_ci95_low": metric_ci(ensemble_metrics, "learned_ensemble_mae", "ci95_low"),
            "mae_ci95_high": metric_ci(ensemble_metrics, "learned_ensemble_mae", "ci95_high"),
            "psnr_db_mean": metric_mean(ensemble_metrics, "learned_ensemble_psnr"),
            "ssim_mean": metric_mean(ensemble_metrics, "learned_ensemble_ssim_map_mean"),
            "orientation_error_mean": metric_mean(ensemble_metrics, "learned_ensemble_orientation_error"),
            "ridge_frequency_relative_mae": metric_mean(ensemble_metrics, "learned_ensemble_ridge_frequency_relative_mae"),
            "uncertainty_error_spearman": "",
            "reported_split": ensemble_payload.get("split", "not_recorded"),
            "test_loaded": ensemble_payload.get("test_loaded", "not_recorded"),
            "source": ensemble_source,
        },
    )
    write_csv(RESULTS / "tables" / "nist302_heldout_evaluation_217.csv", rows, list(rows[0]))
    return rows


def build_k_sensitivity_table() -> None:
    rows: list[dict[str, Any]] = []
    for model, source in [
        ("Pixel DDPM / DDIM-20", "outputs/nist302_TEST_FINAL_ddim/metrics.json"),
        ("Residual DDPM", "outputs/nist302_TEST_FINAL_residual_ddpm/metrics.json"),
    ]:
        metrics = load(source)["metrics"]["metrics"]
        for k in (10, 20, 50):
            rows.append(
                {
                    "model": model,
                    "K": k,
                    "predictive_mean_mae": metric_mean(metrics, f"mean_k{k}_heldout_q1_mae"),
                    "predictive_mean_ssim": metric_mean(metrics, f"mean_k{k}_heldout_q1_ssim_map_mean"),
                    "best_of_k_mae": metric_mean(metrics, f"best_k{k}_heldout_q1_mae"),
                    "diversity_mae": metric_mean(metrics, f"diversity_k{k}_heldout_q1_mae"),
                    "uncertainty_error_spearman": metric_mean(metrics, f"uncertainty_error_spearman_k{k}"),
                    "source": source,
                }
            )
    write_csv(RESULTS / "tables" / "nist302_k_sensitivity.csv", rows, list(rows[0]))


def build_compute_table() -> None:
    rows = [
        {"model": "Gated convolution (SOCOFing)", "parameters": 5679009, "epochs": "50", "training_minutes": 30.0, "K": "", "steps": "", "seconds_per_image": "", "measurement_note": "solo"},
        {"model": "Spatial CVAE", "parameters": 4898129, "epochs": "20", "training_minutes": 17.9, "K": "", "steps": "", "seconds_per_image": "", "measurement_note": "solo"},
        {"model": "Pixel DDPM", "parameters": 8280897, "epochs": "15/25", "training_minutes": 16.3, "K": 10, "steps": 20, "seconds_per_image": 0.626, "measurement_note": "DDIM evaluation; solo"},
        {"model": "Residual DDPM", "parameters": 8281377, "epochs": "17/25", "training_minutes": 30.2, "K": 10, "steps": 20, "seconds_per_image": 1.104, "measurement_note": "training had GPU contention"},
        {"model": "RePaint", "parameters": 8280897, "epochs": "uses pixel-DDPM checkpoint", "training_minutes": "", "K": 5, "steps": "500x2", "seconds_per_image": 19.35, "measurement_note": "ancestral sampling; solo"},
    ]
    write_csv(RESULTS / "tables" / "compute_cost.csv", rows, list(rows[0]))


def build_statistics_table() -> None:
    sources = [
        ("SOCOFing: gated vs U-Net", "outputs/unet_vs_gated_paired_tests.json", "metrics"),
        ("NIST302: fine-tuned vs zero-shot", "outputs/nist302_zeroshot_vs_finetuned_soft_subject_tests.json", "tests"),
        ("NIST302: RePaint vs DDIM", "outputs/nist302_ddim_vs_repaint_subject_tests.json", "tests"),
        ("NIST302: DDIM vs CVAE", "outputs/nist302_cvae_vs_ddim_subject_tests.json", "tests"),
        ("NIST302: residual DDPM vs pixel DDPM", "outputs/nist302_pixel_vs_residual_ddpm_subject_tests.json", "tests"),
        ("NIST302 synthetic exact GT: fine-tuned vs zero-shot", "outputs/nist302_synthetic_exact_gt_zeroshot_vs_finetuned_subject_tests.json", "tests"),
    ]
    rows: list[dict[str, Any]] = []
    for comparison, source, test_key in sources:
        payload = load(source)
        for metric, result in payload[test_key].items():
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n": result.get("n", ""),
                    "mean_improvement": result.get("mean_improvement", ""),
                    "ci95_low": result.get("ci95_low", ""),
                    "ci95_high": result.get("ci95_high", ""),
                    "cohen_dz": result.get("cohen_dz", ""),
                    "paired_t_pvalue": result.get("paired_t_pvalue", ""),
                    "wilcoxon_pvalue_holm": result.get("wilcoxon_pvalue_holm", ""),
                    "source": source,
                }
            )
    write_csv(RESULTS / "statistics" / "selected_paired_tests.csv", rows, list(rows[0]))

    omnibus = load("outputs/nist302_friedman_omnibus_mae.json")
    summary = {
        key: omnibus[key]
        for key in ("metric", "n_subjects", "models", "friedman_chi2", "df", "friedman_p_value", "note")
    }
    summary["source"] = "outputs/nist302_friedman_omnibus_mae.json"
    (RESULTS / "statistics" / "nist302_friedman_omnibus.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def horizontal_bar_svg(rows: list[dict[str, Any]], destination: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["mae_mean"]))
    width, left, right, top, row_height = 1100, 330, 100, 100, 48
    height = top + len(ordered) * row_height + 85
    maximum = max(float(row["mae_mean"]) for row in ordered) * 1.10
    plot_width = width - left - right
    palette = {"deterministic": "#1F3A63", "probabilistic": "#12706B", "probabilistic ensemble": "#6A4C93"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1e293b}.title{font-size:24px;font-weight:700}.sub{font-size:15px;fill:#475569}.label{font-size:15px}.value{font-size:14px;font-weight:700}</style>',
        '<text class="title" x="30" y="38">NIST SD302: held-out real-pixel MAE</text>',
        '<text class="sub" x="30" y="66">217 images; lower is better. Split/test flags are inconsistent across source artefacts; see the CSV.</text>',
    ]
    for index, row in enumerate(ordered):
        y = top + index * row_height
        value = float(row["mae_mean"])
        bar_width = value / maximum * plot_width
        color = palette.get(str(row["family"]), "#64748B")
        parts.extend(
            [
                f'<text class="label" x="{left - 12}" y="{y + 22}" text-anchor="end">{escape(str(row["model"]))}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="30" rx="3" fill="{color}"/>',
                f'<text class="value" x="{left + bar_width + 9:.1f}" y="{y + 21}">{value:.4f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="sub" x="{left}" y="{height - 28}">MAE on held-out quality==1 pixels</text>',
            f'<rect x="{width - 330}" y="20" width="14" height="14" fill="#1F3A63"/><text class="sub" x="{width - 309}" y="32">deterministic</text>',
            f'<rect x="{width - 190}" y="20" width="14" height="14" fill="#12706B"/><text class="sub" x="{width - 169}" y="32">probabilistic</text>',
            "</svg>",
        ]
    )
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_figures(nist_rows: list[dict[str, Any]]) -> None:
    figure_root = RESULTS / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    horizontal_bar_svg(nist_rows, figure_root / "nist302_model_mae.svg")
    safe_aggregate_figures = {
        "chart_socofing_baselines.png": "socofing_baselines.png",
        "chart_uncertainty_distance.png": "uncertainty_vs_anchor_distance.png",
        "chart_reliability.png": "selective_reconstruction.png",
    }
    for source_name, destination_name in safe_aggregate_figures.items():
        shutil.copyfile(
            REPOSITORY / "docs" / "poster_assets" / source_name,
            figure_root / destination_name,
        )


def main() -> None:
    build_socofing_table()
    nist_rows = build_nist302_table()
    build_k_sensitivity_table()
    build_compute_table()
    build_statistics_table()
    build_figures(nist_rows)
    print("Built curated tables, statistical summaries, and aggregate figures in results/")


if __name__ == "__main__":
    main()
