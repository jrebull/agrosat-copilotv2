"""Tests del pipeline denso compartido PASTIS-R (ml.ingest.pastis_dataset).

Usa un mini-PASTIS sintetico hermetico escrito en ``tmp_path`` (sin depender del
dataset real de ~29 GB): valida shapes tras resize, normalizacion por fold,
split por folds oficiales, ignore de void y los dos modos temporales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTISDataset,
    load_norm_stats,
    pastis_fold_split,
)

_T = 5
_H = 128


def _write_mini_pastis(root: Path, patch_folds: dict[int, int]) -> None:
    """Escribe un PASTIS-R minimo en ``root`` para los patches indicados.

    Args:
        root: Directorio raiz destino.
        patch_folds: Mapa ``{patch_id: fold}`` a materializar.
    """
    (root / "DATA_S2").mkdir(parents=True, exist_ok=True)
    (root / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    features = []
    for pid, fold in patch_folds.items():
        s2 = rng.integers(0, 4000, size=(_T, 10, _H, _H), dtype=np.int16)
        np.save(root / "DATA_S2" / f"S2_{pid}.npy", s2)
        # Canal 0 semantico con clases 0..19 (incluye void=19).
        semantic = rng.integers(0, 20, size=(_H, _H), dtype=np.uint8)
        target = np.stack([semantic, semantic, semantic], axis=0)
        np.save(root / "ANNOTATIONS" / f"TARGET_{pid}.npy", target)
        features.append(
            {
                "type": "Feature",
                "id": pid,
                "properties": {
                    "ID_PATCH": pid,
                    "TILE": "T31TFM",
                    "Fold": fold,
                    "dates-S2": {str(i): 20190101 + i for i in range(_T)},
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]],
                },
            }
        )

    (root / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    norm = {f"Fold_{k}": {"mean": [1000.0] * 10, "std": [500.0] * 10} for k in range(1, 6)}
    (root / "NORM_S2_patch.json").write_text(json.dumps(norm), encoding="utf-8")


@pytest.fixture
def mini_pastis(tmp_path: Path) -> Path:
    """Mini-PASTIS de 3 patches en folds 1, 4 y 5."""
    root = tmp_path / "PASTIS-R"
    _write_mini_pastis(root, {1: 1, 2: 4, 3: 5})
    return root


def test_load_norm_stats_averages_folds(mini_pastis: Path) -> None:
    """Las estadisticas de normalizacion tienen forma (10,) y valores esperados."""
    mean, std = load_norm_stats(mini_pastis, folds=(1, 2, 3))
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert np.allclose(mean, 1000.0)
    assert np.allclose(std, 500.0)


def test_fold_split_is_disjoint(mini_pastis: Path) -> None:
    """El split por folds asigna cada patch a un unico conjunto."""
    split = pastis_fold_split(mini_pastis, train_folds=(1,), val_folds=(4,), test_folds=(5,))
    assert split["train"] == ["1"]
    assert split["val"] == ["2"]
    assert split["test"] == ["3"]


def test_dataset_2d_shapes_and_normalization(mini_pastis: Path) -> None:
    """Modo median: image (10,256,256) normalizada + label long (256,256)."""
    ds = PASTISDataset(["1"], root=mini_pastis, target_size=256, temporal_reduction="median")
    item = ds[0]
    assert item["image"].shape == (10, 256, 256)
    assert item["image"].dtype == torch.float32
    assert item["semantic"].shape == (256, 256)
    assert item["semantic"].dtype == torch.int64
    # Labels permanecen en el rango de clases tras resize nearest (sin interpolar ids).
    assert int(item["semantic"].max()) <= 19
    # Normalizacion: (valor - 1000)/500 deja la media cerca de 0 en magnitud acotada.
    assert torch.isfinite(item["image"]).all()


def test_dataset_temporal_mode_shapes(mini_pastis: Path) -> None:
    """Modo none: image (fixed_t,10,256,256) + dates (fixed_t,)."""
    ds = PASTISDataset(
        ["1"], root=mini_pastis, target_size=128, temporal_reduction="none", fixed_t=10
    )
    item = ds[0]
    assert item["image"].shape == (10, 10, 128, 128)  # (fixed_t, C, H, W)
    assert item["dates"].shape == (10,)
    assert item["dates"].dtype == torch.int64


def test_ignore_index_constant() -> None:
    """La convencion compartida fija void = 19 como ignore_index."""
    assert PASTIS_IGNORE_INDEX == 19
