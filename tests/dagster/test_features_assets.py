"""Tests Dagster — assets US-016 ``parcel_features_fused``,
``parcel_splits_spatial_kfold``, ``parcel_features_scaler``.

Cubre:

- Materializacion en-memoria con ``dagster.materialize`` y el fixture demo.
- Lineage: ``AssetKey`` deps declaradas correctamente por cada asset.

Notas:

- El asset Dagster real construye el frame con `build_fused_features` SIN
  inyectar frames (deja los bloques en None por GEE-not-available en CI).
  El test verifica que la materializacion no rompe y produce un parquet
  con el contrato de 191 cols.
- Los tests de splits y scaler usan `pytest.importorskip("h3")`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from dagster import AssetKey, materialize_to_memory


def _maybe_copy_fixture_to_cwd(tmp_path: Path) -> Path:
    """Copia el fixture demo a `tmp_path` y devuelve la cwd configurada.

    Los assets US-016 leen `data/test_fixtures/parcels_demo_3regions.parquet`
    relativo al cwd. Para aislar materializaciones, copiamos el fixture.
    """
    repo_root = Path(__file__).resolve().parents[2]
    fixture_src = repo_root / "data" / "test_fixtures" / "parcels_demo_3regions.parquet"
    if not fixture_src.exists():
        pytest.skip("Fixture demo no encontrado en data/test_fixtures/.")
    target_dir = tmp_path / "data" / "test_fixtures"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "parcels_demo_3regions.parquet"
    target.write_bytes(fixture_src.read_bytes())
    return tmp_path


def test_parcel_features_fused_materializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`parcel_features_fused` materializa el parquet (sin GEE/FarSLIP)."""
    from dagster_project.assets.features import parcel_features_fused

    cwd = _maybe_copy_fixture_to_cwd(tmp_path)
    monkeypatch.chdir(cwd)
    result = materialize_to_memory([parcel_features_fused])
    assert result.success
    out_path = cwd / "data" / "features" / "features_fused_v1.parquet"
    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert df.height == 9  # 9 parcelas en el fixture
    assert df.width == 2 + 189


def test_parcel_splits_materializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`parcel_splits_spatial_kfold` materializa los 5 folds en parquet."""
    pytest.importorskip("h3", reason="US-016 spatial split requiere h3.")
    from dagster_project.assets.features import parcel_splits_spatial_kfold

    cwd = _maybe_copy_fixture_to_cwd(tmp_path)
    monkeypatch.chdir(cwd)
    result = materialize_to_memory([parcel_splits_spatial_kfold])
    assert result.success
    splits_dir = cwd / "data" / "splits" / "spatial_kfold_v1"
    # Al menos un fold debe existir.
    assert any((splits_dir / f"fold_{i}").exists() for i in range(5))


def test_parcel_scaler_materializes_after_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`parcel_features_scaler` ejecuta tras fused + splits y persiste joblib."""
    pytest.importorskip("h3", reason="US-016 splits requiere h3.")
    from dagster_project.assets.features import (
        parcel_features_fused,
        parcel_features_scaler,
        parcel_splits_spatial_kfold,
    )

    cwd = _maybe_copy_fixture_to_cwd(tmp_path)
    monkeypatch.chdir(cwd)
    # Materializamos los 3 assets en orden de dependencias.
    result = materialize_to_memory(
        [parcel_features_fused, parcel_splits_spatial_kfold, parcel_features_scaler]
    )
    assert result.success
    scaler_path = cwd / "artifacts" / "scaler_v1.pkl"
    assert scaler_path.exists()


def test_lineage_deps_correct() -> None:
    """Los 3 assets declaran las dependencias upstream esperadas."""
    from dagster_project.assets.features import (
        parcel_features_fused,
        parcel_features_scaler,
        parcel_splits_spatial_kfold,
    )

    fused_deps = set(parcel_features_fused.dependency_keys)
    assert AssetKey("alphaearth_embeddings_italy") in fused_deps
    assert AssetKey("features_parcels") in fused_deps
    assert AssetKey("parcels") in fused_deps

    splits_deps = set(parcel_splits_spatial_kfold.dependency_keys)
    assert AssetKey("parcels") in splits_deps

    scaler_deps = set(parcel_features_scaler.dependency_keys)
    assert AssetKey("parcel_features_fused") in scaler_deps
    assert AssetKey("parcel_splits_spatial_kfold") in scaler_deps
