"""Tests for ml.report.paper_inference_maps."""

from __future__ import annotations

import json
from pathlib import Path

from ml.report.paper_inference_maps import PaperInferenceMap, patch_ground_resolution


def test_patch_ground_resolution_reads_lambert93_bbox(tmp_path: Path) -> None:
    """A 1374 m bounding box over a 128 px patch yields ~10.73 m/px."""
    # Square patch of side 1374 m (the real size of patch 40175 in Lambert-93).
    side = 1374.0
    geo = {
        "type": "FeatureCollection",
        "crs": {"properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
        "features": [
            {
                "properties": {"ID_PATCH": 40175},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side], [0.0, 0.0]]
                    ],
                },
            }
        ],
    }
    (tmp_path / "metadata.geojson").write_text(json.dumps(geo))
    mpp = patch_ground_resolution("40175", tmp_path)
    assert abs(mpp - side / 128) < 1e-6
    assert 10.7 < mpp < 10.8  # NOT the nominal 10.0


def test_patch_ground_resolution_unknown_patch_falls_back(tmp_path: Path) -> None:
    """An unknown patch id falls back to the nominal 10 m, not a crash."""
    geo = {"type": "FeatureCollection", "features": []}
    (tmp_path / "metadata.geojson").write_text(json.dumps(geo))
    assert patch_ground_resolution("99999", tmp_path) == 10.0


def test_paper_inference_map_accuracy() -> None:
    """parcel_accuracy is n_correct / n_parcels, 0 when there are no parcels."""
    m = PaperInferenceMap(
        patch_id="40175",
        path=Path("x.png"),
        n_correct=73,
        n_parcels=76,
        n_out_of_scope=2,
        ground_resolution_m=10.74,
    )
    assert abs(m.parcel_accuracy - 73 / 76) < 1e-9
    empty = PaperInferenceMap(
        patch_id="0",
        path=Path("x.png"),
        n_correct=0,
        n_parcels=0,
        n_out_of_scope=0,
        ground_resolution_m=10.0,
    )
    assert empty.parcel_accuracy == 0.0
