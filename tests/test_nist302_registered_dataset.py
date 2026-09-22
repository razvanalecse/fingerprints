import csv
from pathlib import Path

import numpy as np
from PIL import Image

from fingerprint_reconstruction.data import Nist302RegisteredDataset


def test_registered_dataset_returns_explicit_approximate_target(tmp_path: Path) -> None:
    latent_root = tmp_path / "latent"
    exemplar_root = tmp_path / "exemplar"
    annotation_root = tmp_path / "annotations"
    latent_root.mkdir()
    exemplar_root.mkdir()
    (annotation_root / "quality_maps").mkdir(parents=True)
    image = np.arange(25, dtype=np.uint8).reshape(5, 5) * 10
    Image.fromarray(image).save(latent_root / "latent.png")
    Image.fromarray(image).save(exemplar_root / "exemplar.png")
    np.savez_compressed(
        annotation_root / "quality_maps" / "map.npz",
        quality=np.array([[2, 2], [0, 0]], dtype=np.uint8),
        grid_size_0_01mm=np.uint16(2),
    )
    row = {
        "split": "test",
        "exemplar_dataset_part": "sd302b",
        "exemplar_relative_path": "exemplar.png",
        "latent_relative_path": "latent.png",
        "latent_height": "5",
        "latent_width": "5",
        "latent_native_ppi": "2540",
        "quality_map_path": "quality_maps/map.npz",
        "latent_sample_id": "latent-1",
        "subject_id": "00000001",
        "num_correspondences": "8",
        "affine_rmse_mm": "0.1",
        "minimum_hull_coverage": "0.02",
        "latent_correspondences_0_01mm": "[[0,0],[4,0],[0,4],[4,4]]",
        "pixel_m0_0": "1",
        "pixel_m0_1": "0",
        "pixel_m0_2": "0",
        "pixel_m1_0": "0",
        "pixel_m1_1": "1",
        "pixel_m1_2": "0",
    }
    manifest = tmp_path / "registered.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    dataset = Nist302RegisteredDataset(
        manifest_path=manifest,
        latent_root=latent_root,
        annotation_root=annotation_root,
        exemplar_roots={"sd302b": exemplar_root},
        split="test",
        output_shape=(5, 5),
        use_exemplar_foreground=False,
        geometric_confidence_weights=(1.0, 0.8, 0.6),
    )
    item = dataset[0]
    assert item["conditioning"].shape == (2, 5, 5)
    assert item["target"].shape == (1, 5, 5)
    assert item["evaluation_roi"].sum() > 0
    assert np.array_equal(
        item["evaluation_roi"].numpy().astype(bool),
        (item["evaluation_roi_near"] + item["evaluation_roi_far"]).numpy().astype(bool),
    )
    anchor_partition = (
        item["evaluation_roi_anchor_inside"]
        + item["evaluation_roi_anchor_0_2mm"]
        + item["evaluation_roi_anchor_2_5mm"]
        + item["evaluation_roi_anchor_gt_5mm"]
    )
    assert np.array_equal(
        item["evaluation_roi"].numpy().astype(bool),
        anchor_partition.numpy().astype(bool),
    )
    nearest_partition = (
        item["evaluation_roi_nearest_0_2mm"]
        + item["evaluation_roi_nearest_2_5mm"]
        + item["evaluation_roi_nearest_gt_5mm"]
    )
    assert np.array_equal(
        item["evaluation_roi"].numpy().astype(bool),
        nearest_partition.numpy().astype(bool),
    )
    assert item["pixel_aligned_ground_truth"] is False
    assert item["registration_status"] == "registered_approximate"
    confidence = item["geometric_confidence"].numpy()
    assert np.all(confidence[item["evaluation_roi_nearest_0_2mm"].numpy().astype(bool)] == 1.0)
    assert np.all(confidence[item["evaluation_roi_nearest_2_5mm"].numpy().astype(bool)] == 0.8)
    assert np.all(confidence[item["evaluation_roi_nearest_gt_5mm"].numpy().astype(bool)] == 0.6)
    assert np.all(confidence[~item["evaluation_roi"].numpy().astype(bool)] == 0.0)
