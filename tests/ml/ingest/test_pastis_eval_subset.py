"""Tests para ``ml.ingest.pastis_eval_subset``.

Nunca golpea el filesystem real de PASTIS-R: cada test sintetiza un mini
PASTIS en ``tmp_path`` con un puado de patches numpy mock y un
``metadata.geojson`` minimo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.ingest import pastis_eval_subset
from ml.ingest.pastis_eval_subset import (
    _md5_file,
    build_pastis_eval_subset,
)

# ---------------------------------------------------------------------------
# Fixtures: mini PASTIS-R sintetico en tmp_path
# ---------------------------------------------------------------------------


def _make_metadata(features: list[dict]) -> dict:
    """Construye un GeoJSON FeatureCollection PASTIS-R-like.

    Args:
        features: Lista de feature dicts.

    Returns:
        Dict GeoJSON FeatureCollection.
    """
    return {"type": "FeatureCollection", "features": features}


def _make_feature(
    patch_id: int,
    tile: str,
    fold: int,
    bbox_x: tuple[float, float] = (650000.0, 650100.0),
    bbox_y: tuple[float, float] = (6850000.0, 6850100.0),
) -> dict:
    """Construye una feature PASTIS-R-like en EPSG:2154.

    Args:
        patch_id: ID del patch.
        tile: Codigo TILE Sentinel-2.
        fold: Fold cross-val (1..5).
        bbox_x: (xmin, xmax) en EPSG:2154.
        bbox_y: (ymin, ymax) en EPSG:2154.

    Returns:
        Dict GeoJSON Feature.
    """
    x0, x1 = bbox_x
    y0, y1 = bbox_y
    return {
        "type": "Feature",
        "id": patch_id,
        "properties": {
            "ID_PATCH": patch_id,
            "TILE": tile,
            "Fold": fold,
            "dates-S2": {"0": 20180101, "1": 20180115},
        },
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [x0, y0],
                    ]
                ]
            ],
        },
    }


def _write_patch(
    root: Path,
    patch_id: int,
    target: np.ndarray,
    s2: np.ndarray | None = None,
) -> None:
    """Persiste los .npy de un patch en la estructura PASTIS-R.

    Args:
        root: Raiz PASTIS-R.
        patch_id: ID del patch.
        target: ndarray (3, H, W) uint8 con semantic/instance/zone.
        s2: ndarray (T, 10, H, W) o None (generado sintetico si None).
    """
    (root / "DATA_S2").mkdir(parents=True, exist_ok=True)
    (root / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)
    np.save(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", target.astype(np.uint8))
    if s2 is None:
        H, W = target.shape[-2], target.shape[-1]
        # Sintetizamos imagery (no test-data prod): el modulo bajo prueba la
        # acepta sin saber distinguir; aqui es input, no output sintetico.
        rng = np.random.default_rng(patch_id)
        s2 = rng.integers(0, 5000, size=(2, 10, H, W), dtype=np.int16)
    np.save(root / "DATA_S2" / f"S2_{patch_id}.npy", s2)


def _build_target(class_id: int, instance_id: int, h: int = 8, w: int = 8) -> np.ndarray:
    """Target uniforme con una sola instancia de la clase indicada.

    Args:
        class_id: Clase semantica (1..18).
        instance_id: ID de instancia.
        h: Altura.
        w: Ancho.

    Returns:
        ndarray (3, H, W) uint8.
    """
    semantic = np.full((h, w), class_id, dtype=np.uint8)
    instance = np.full((h, w), instance_id, dtype=np.uint8)
    zone = np.zeros((h, w), dtype=np.uint8)
    return np.stack([semantic, instance, zone], axis=0)


@pytest.fixture
def mini_pastis_root(tmp_path: Path) -> Path:
    """Construye un mini PASTIS-R sintetico con 18 clases x 2 patches = 36 instancias.

    Cada patch contiene 2 instancias de clases distintas para forzar que la
    enumeracion produzca al menos 36 parcelas (suficiente para stratify_by=class
    con minimo >= 1 por clase).

    Args:
        tmp_path: Directorio temporal pytest.

    Returns:
        Ruta a la raiz PASTIS-R sintetica.
    """
    root = tmp_path / "PASTIS-R"
    root.mkdir(parents=True, exist_ok=True)

    features: list[dict] = []
    # 18 clases x 2 patches por clase = 36 parcelas
    patch_id = 1
    for cls in range(1, 19):
        for rep in range(2):
            tile = f"T31TFM" if rep == 0 else "T31TFL"
            fold = ((cls + rep) % 5) + 1
            features.append(_make_feature(patch_id, tile, fold))
            target = _build_target(cls, instance_id=cls, h=8, w=8)
            _write_patch(root, patch_id, target)
            patch_id += 1

    metadata = _make_metadata(features)
    (root / "metadata.geojson").write_text(json.dumps(metadata), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_raises_if_pastis_missing(tmp_path: Path) -> None:
    """Si la raiz PASTIS-R no existe, debe lanzar FileNotFoundError."""
    missing_root = tmp_path / "nope"
    out = tmp_path / "subset.parquet"
    with pytest.raises(FileNotFoundError) as excinfo:
        build_pastis_eval_subset(
            output_path=out, pastis_root=missing_root, overwrite=True
        )
    assert "PASTIS-R not found" in str(excinfo.value)
    assert "dvc pull" in str(excinfo.value) or "zenodo" in str(excinfo.value).lower()


def test_raises_if_data_s2_missing(tmp_path: Path) -> None:
    """Si metadata.geojson existe pero DATA_S2/ no, tambien lanza."""
    root = tmp_path / "PASTIS-R"
    root.mkdir()
    (root / "metadata.geojson").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_pastis_eval_subset(
            output_path=tmp_path / "subset.parquet",
            pastis_root=root,
            overwrite=True,
        )


def test_build_subset_shape_and_schema(mini_pastis_root: Path, tmp_path: Path) -> None:
    """Con el mini PASTIS sintetico genera subset con schema canonico."""
    out = tmp_path / "subset.parquet"
    result = build_pastis_eval_subset(
        output_path=out,
        n_samples=36,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        stratify_by="class",
        save_imagery=False,
    )
    assert result == out
    assert out.exists()

    df = pl.read_parquet(out)
    expected_cols = {
        "parcel_id",
        "patch_id",
        "instance_id",
        "class_id",
        "class_name",
        "tile",
        "fold",
        "lon",
        "lat",
        "n_pixels",
    }
    assert set(df.columns) == expected_cols
    assert df.height > 0
    assert df.height <= 36
    # class_id en rango 1..18
    cls_vals = df["class_id"].to_list()
    assert all(1 <= c <= 18 for c in cls_vals)


def test_parcel_id_is_utf8_composite(mini_pastis_root: Path, tmp_path: Path) -> None:
    """parcel_id debe ser Utf8 con el esquema {patch_id}_{instance_id}."""
    out = tmp_path / "subset.parquet"
    build_pastis_eval_subset(
        output_path=out,
        n_samples=18,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        save_imagery=False,
    )
    df = pl.read_parquet(out)
    assert df.schema["parcel_id"] == pl.Utf8
    for row in df.iter_rows(named=True):
        expected = f"{row['patch_id']}_{row['instance_id']}"
        assert row["parcel_id"] == expected


def test_determinism_md5(mini_pastis_root: Path, tmp_path: Path) -> None:
    """Mismo seed + mismo input => mismo MD5 del parquet."""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"

    build_pastis_eval_subset(
        output_path=out_a,
        n_samples=36,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        save_imagery=False,
    )
    build_pastis_eval_subset(
        output_path=out_b,
        n_samples=36,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        save_imagery=False,
    )
    assert _md5_file(out_a) == _md5_file(out_b)


def test_stratification_min_per_class(mini_pastis_root: Path, tmp_path: Path) -> None:
    """Con stratify_by='class' y n_samples=1024 cada clase recibe >= max(8, n//36) muestras
    si hay disponibilidad (en el fixture hay 2 instancias por clase, por lo que el
    threshold efectivo es min(2, max(8, 1024//36))).
    """
    out = tmp_path / "subset.parquet"
    build_pastis_eval_subset(
        output_path=out,
        n_samples=1024,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        stratify_by="class",
        save_imagery=False,
    )
    df = pl.read_parquet(out)
    counts = df.group_by("class_id").agg(pl.len().alias("n")).sort("class_id")
    # Cada clase disponible (las 18) debe estar presente con al menos la cantidad
    # disponible en el catalogo (2 en este fixture).
    assert counts.height == 18
    for n in counts["n"].to_list():
        assert n >= 2


def test_skip_existing_without_overwrite(mini_pastis_root: Path, tmp_path: Path) -> None:
    """Si overwrite=False y el archivo existe, no regenera (MD5 inalterado)."""
    out = tmp_path / "subset.parquet"
    build_pastis_eval_subset(
        output_path=out,
        n_samples=36,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        save_imagery=False,
    )
    md5_before = _md5_file(out)
    mtime_before = out.stat().st_mtime_ns

    # Segundo run con overwrite=False y seed distinto: debe NO tocar archivo.
    build_pastis_eval_subset(
        output_path=out,
        n_samples=36,
        seed=999,
        pastis_root=mini_pastis_root,
        overwrite=False,
        save_imagery=False,
    )
    assert _md5_file(out) == md5_before
    assert out.stat().st_mtime_ns == mtime_before


def test_imagery_blob_emitted(mini_pastis_root: Path, tmp_path: Path) -> None:
    """Con save_imagery=True se materializa <output>.imagery.parquet con bandas B02..B12."""
    out = tmp_path / "subset.parquet"
    build_pastis_eval_subset(
        output_path=out,
        n_samples=18,
        seed=42,
        pastis_root=mini_pastis_root,
        overwrite=True,
        save_imagery=True,
    )
    imagery_path = out.with_suffix(out.suffix + ".imagery.parquet")
    assert imagery_path.exists()
    img = pl.read_parquet(imagery_path)
    assert "parcel_id" in img.columns
    assert "t_index" in img.columns
    for band in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"):
        assert f"band_{band}" in img.columns
    # Al menos 1 fila por parcela seleccionada * T(=2 en fixture)
    parcels = pl.read_parquet(out)
    assert img.height == parcels.height * 2


def test_cli_main(mini_pastis_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """El CLI imprime un JSON con output + md5 y exit code 0."""
    out = tmp_path / "subset.parquet"
    rc = pastis_eval_subset.main(
        [
            "--output",
            str(out),
            "--n-samples",
            "18",
            "--seed",
            "42",
            "--pastis-root",
            str(mini_pastis_root),
            "--no-imagery",
            "--overwrite",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.strip()
    # structlog tambien escribe a stdout; el JSON del CLI es la ultima linea.
    last_line = [line for line in captured.splitlines() if line.startswith("{")][-1]
    payload = json.loads(last_line)
    assert payload["output"] == str(out)
    assert len(payload["md5"]) == 32
