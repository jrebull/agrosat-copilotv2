"""Tests Dagster — US-017 assets ``sentinel2_crops_256`` y ``farslip_embeddings_italy``.

Cubre:

- Materialización smoke de ``sentinel2_crops_256`` mockeando el builder
  ``ml.farslip.dataset.build_farslip_pairs`` para devolver un DataFrame
  Polars mínimo (3 filas sintéticas).
- Lineage: ``farslip_embeddings_italy.dependency_keys`` contiene
  ``AssetKey(["sentinel2_crops_256"])``.
- Graceful skip cuando el extractor FarSLIP falla por GCS creds ausentes
  (``DefaultCredentialsError``) — el asset reporta
  ``status="skipped_no_gcs"`` sin romper la materialización.
- Registro en ``dagster_project.definitions``: ambos AssetKeys aparecen
  en el ``AssetGraph`` global.

Notas:

- Si el subagente ml-engineer no ha aterrizado todavía ``ml.farslip``,
  los tests que ejecutan el asset usan ``sys.modules`` para inyectar
  stubs antes del import; los tests de introspección puramente Dagster
  funcionan sin necesidad del módulo ML.
- Toda materialización corre en ``materialize_to_memory`` con
  ``partition_key="pianura_padana"``; los otros 2 ROIs se ejercitan en
  bulk extraction real (fuera de tests, en VM L4 spot).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest


def _ensure_ml_farslip_dataset_stub(monkeypatch: pytest.MonkeyPatch, manifest_df: Any) -> None:
    """Inserta un stub de ``ml.farslip.dataset.build_farslip_pairs`` en sys.modules.

    Necesario porque el módulo real lo crea el subagente ml-engineer en
    paralelo. El stub devuelve ``manifest_df`` cuando se invoca y simula
    el side-effect de escribir ``data/farslip_pairs/{roi}/manifest.parquet``
    para que el asset downstream pueda leerlo.

    Args:
        monkeypatch: fixture pytest para restaurar sys.modules tras el test.
        manifest_df: ``polars.DataFrame`` que el stub devolverá.
    """
    # Inyectar/reemplazar stub siempre — el modulo real puede haber sido
    # importado transitivamente (ml.extractors.farslip_extractor importa
    # ml.farslip.distill), dejando ml.farslip.dataset cacheado en sys.modules.
    if "ml.farslip" not in sys.modules:
        ml_farslip = types.ModuleType("ml.farslip")
        monkeypatch.setitem(sys.modules, "ml.farslip", ml_farslip)

    ml_farslip_dataset = types.ModuleType("ml.farslip.dataset")

    def _stub_build_farslip_pairs(*args: Any, **kwargs: Any) -> Any:
        # Side-effect: escribe manifest.parquet en disco para que
        # farslip_embeddings_italy pueda leerlo aguas abajo.
        roi = kwargs.get("rois", ("pianura_padana",))[0]
        output_root = Path(kwargs.get("output_root", Path("data/farslip_pairs")))
        roi_dir = output_root / roi
        roi_dir.mkdir(parents=True, exist_ok=True)
        manifest_df.write_parquet(roi_dir / "manifest.parquet")
        return manifest_df

    ml_farslip_dataset.build_farslip_pairs = _stub_build_farslip_pairs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.farslip.dataset", ml_farslip_dataset)


def _ensure_ml_extractors_farslip_stub(
    monkeypatch: pytest.MonkeyPatch,
    raise_on_init: Exception | None = None,
) -> None:
    """Inserta un stub de ``ml.extractors.farslip_extractor.FarSLIPExtractor``.

    Args:
        monkeypatch: fixture pytest.
        raise_on_init: si no es None, ``FarSLIPExtractor.__init__`` lanza
            la excepción dada (simula DefaultCredentialsError).
    """
    if "ml.extractors.farslip_extractor" in sys.modules:
        monkeypatch.delitem(sys.modules, "ml.extractors.farslip_extractor")

    module = types.ModuleType("ml.extractors.farslip_extractor")

    class _StubFarSLIPExtractor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if raise_on_init is not None:
                raise raise_on_init
            self._dim = 512

        def load_crops_batch(self, paths: list[Any]) -> Any:
            import torch

            return torch.zeros((len(paths), 4, 256, 256), dtype=torch.float32)

        def extract_embeddings(self, crops: Any) -> Any:
            import torch

            return torch.zeros((crops.shape[0], self._dim), dtype=torch.float32)

    module.FarSLIPExtractor = _StubFarSLIPExtractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.extractors.farslip_extractor", module)


def _build_minimal_manifest(n: int = 3) -> Any:
    """Construye un DataFrame Polars mínimo válido para tests smoke."""
    import polars as pl

    return pl.DataFrame(
        {
            "crop_path": [f"data/farslip_pairs/pianura_padana/crops/c{i}.tif" for i in range(n)],
            "crop_doy": [120, 180, 240][:n],
            "crop_year": [2024] * n,
            "cap_class": ["mais", "frumento", "vite"][:n],
            "region": ["pianura_padana"] * n,
            "text_it": ["mais", "frumento", "vite"][:n],
            "text_es": ["maíz", "trigo", "vid"][:n],
            "text_en": ["maize", "wheat", "vine"][:n],
            "lat": [45.0, 45.1, 45.2][:n],
            "lon": [9.0, 9.1, 9.2][:n],
        }
    )


def test_sentinel2_crops_256_materializes_one_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sentinel2_crops_256`` materializa una partition con manifest mínimo."""
    from dagster import materialize_to_memory

    monkeypatch.chdir(tmp_path)
    manifest = _build_minimal_manifest(n=3)
    _ensure_ml_farslip_dataset_stub(monkeypatch, manifest)

    from dagster_project.assets.sentinel2_crops import sentinel2_crops_256

    result = materialize_to_memory(
        [sentinel2_crops_256],
        partition_key="pianura_padana",
    )
    assert result.success

    # Verificar que el manifest se escribió en disco (side-effect del stub).
    expected_manifest = tmp_path / "data" / "farslip_pairs" / "pianura_padana" / "manifest.parquet"
    assert expected_manifest.exists()

    # Verificar metadata keys presentes en el evento de materialización.
    materializations = result.get_asset_materialization_events()
    assert len(materializations) == 1
    mat = materializations[0].materialization
    metadata_keys = set(mat.metadata.keys())
    required_keys = {
        "roi",
        "n_pairs",
        "min_doy",
        "max_doy",
        "n_classes",
        "output_path",
        "data_version",
        "code_version",
    }
    assert required_keys.issubset(metadata_keys), (
        f"missing metadata keys: {required_keys - metadata_keys}"
    )


def test_farslip_embeddings_italy_depends_on_crops_asset() -> None:
    """Lineage: ``farslip_embeddings_italy`` declara dep en ``sentinel2_crops_256``."""
    from dagster import AssetKey

    from dagster_project.assets.farslip import farslip_embeddings_italy

    deps = set(farslip_embeddings_italy.dependency_keys)
    assert AssetKey(["sentinel2_crops_256"]) in deps, (
        f"esperado AssetKey('sentinel2_crops_256') en deps, encontrado: {deps}"
    )


def test_farslip_embeddings_italy_skips_on_missing_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si ``FarSLIPExtractor`` lanza ``DefaultCredentialsError`` → skip limpio."""
    from dagster import build_asset_context

    monkeypatch.chdir(tmp_path)
    manifest = _build_minimal_manifest(n=3)

    # Pre-escribir el manifest upstream (simulamos que sentinel2_crops_256 corrió).
    roi_dir = tmp_path / "data" / "farslip_pairs" / "pianura_padana"
    roi_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(roi_dir / "manifest.parquet")

    # Stub que simula DefaultCredentialsError de google.auth.
    fake_exc = type("DefaultCredentialsError", (Exception,), {})("no GCS creds in CI")
    _ensure_ml_extractors_farslip_stub(monkeypatch, raise_on_init=fake_exc)

    from dagster_project.assets.farslip import farslip_embeddings_italy

    context = build_asset_context(partition_key="pianura_padana")
    result = farslip_embeddings_italy(context)

    # MaterializeResult devuelto con metadata de skip — no excepción.
    assert result is not None
    metadata = result.metadata or {}
    status = metadata.get("status")
    assert status is not None
    # MetadataValue.text expone .value; comparamos defensivamente.
    status_value = getattr(status, "value", status)
    assert status_value == "skipped_no_gcs"


def test_assets_registered_in_definitions() -> None:
    """Ambos AssetKeys aparecen en ``defs.get_asset_graph().asset_keys``."""
    from dagster import AssetKey

    from dagster_project.definitions import defs

    asset_keys = set(defs.resolve_asset_graph().get_all_asset_keys())
    assert AssetKey(["sentinel2_crops_256"]) in asset_keys
    assert AssetKey(["farslip_embeddings_italy"]) in asset_keys
