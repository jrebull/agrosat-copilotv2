"""Tests Dagster — US-022b-B re-materializacion del pipeline FarSLIP.

Cubre:

- ``farslip_embeddings_consolidated`` produce
  ``data/farslip/embeddings_pastis.parquet`` (gate B-4 del plan).
- Skipea limpio cuando no hay parquets upstream (CI sin GCS).
- Lineage explicito: ``farslip_embeddings_italy.deps`` incluye el modelo
  ``farslip_clip_italy_v1`` (B-5 del plan).
- ``farslip_pairs_italy`` y ``farslip_clip_italy_v1`` AssetSpec presentes en
  el AssetGraph global (lineage UI).
- ``farslip_full_pipeline_job`` definido y selecciona el subgrafo correcto.
- Resource ``mlflow`` registrado en ``Definitions.resources``.
- Tags MLflow presentes en metadata de los assets nuevos.

Notas:

- MLflow se mockea via ``build_asset_context(resources={"mlflow": MagicMock()})``
  para no requerir un servidor real en CI.
- Los AssetSpec externos no se materializan; solo se inspecciona su presencia
  en el ``AssetGraph``.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _write_fake_partition(embeddings_root: Path, roi: str, year: int, n_rows: int = 5) -> Path:
    """Escribe un parquet de embeddings sintetico en el layout del asset upstream.

    Args:
        embeddings_root: ``data/farslip_embeddings/`` raiz.
        roi: nombre de la ROI italiana.
        year: anio del crop.
        n_rows: filas sinteticas.

    Returns:
        ``Path`` al parquet escrito.
    """
    import polars as pl

    out_dir = embeddings_root / roi / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embeddings.parquet"

    df = pl.DataFrame(
        {
            "crop_id": [f"{1000 + i}" for i in range(n_rows)],
            "embedding": [[0.1 * j for j in range(512)] for _ in range(n_rows)],
            "crop_doy": [120 + i * 10 for i in range(n_rows)],
            "cap_class": ["mais"] * n_rows,
        }
    )
    df.write_parquet(out_path, compression="zstd")
    return out_path


def test_farslip_embeddings_consolidated_skips_when_no_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin parquets upstream: skip limpio con status="skipped_no_upstream"."""
    from dagster import build_asset_context

    monkeypatch.chdir(tmp_path)

    from dagster_project.assets.farslip_pipeline import (
        farslip_embeddings_consolidated,
    )

    context = build_asset_context(resources={"mlflow": MagicMock()})
    result = farslip_embeddings_consolidated(context)

    assert result is not None
    metadata = result.metadata or {}
    status = metadata.get("status")
    assert status is not None
    status_value = getattr(status, "value", status)
    assert status_value == "skipped_no_upstream"

    rows = metadata.get("rows")
    rows_value = getattr(rows, "value", rows)
    assert rows_value == 0


def test_farslip_embeddings_consolidated_writes_canonical_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con upstream poblado: escribe data/farslip/embeddings_pastis.parquet (B-4)."""
    import polars as pl
    from dagster import build_asset_context

    monkeypatch.chdir(tmp_path)

    # Simula upstream materializado: 3 ROIs x 1 year.
    embeddings_root = tmp_path / "data" / "farslip_embeddings"
    for roi in ("pianura_padana", "toscana", "puglia"):
        _write_fake_partition(embeddings_root, roi, 2024, n_rows=4)

    from dagster_project.assets.farslip_pipeline import (
        DATA_FARSLIP_CONSOLIDATED_PATH,
        farslip_embeddings_consolidated,
    )

    mlflow_mock = MagicMock()
    context = build_asset_context(resources={"mlflow": mlflow_mock})
    result = farslip_embeddings_consolidated(context)

    # El parquet consolidado debe existir en la ruta canonica consumida por
    # ml/features/fusion.py (B-4).
    assert DATA_FARSLIP_CONSOLIDATED_PATH.exists(), (
        f"esperado parquet en {DATA_FARSLIP_CONSOLIDATED_PATH}, cwd={Path.cwd()}"
    )

    df = pl.read_parquet(DATA_FARSLIP_CONSOLIDATED_PATH)
    assert df.height == 12  # 3 rois x 4 rows
    assert "region" in df.columns
    assert "embedding" in df.columns
    assert "parcel_id" in df.columns  # cast de crop_id

    regions = set(df["region"].to_list())
    assert regions == {"pianura_padana", "toscana", "puglia"}

    # Metadata: rows + tags B-5.
    metadata = result.metadata or {}
    rows = getattr(metadata.get("rows"), "value", None)
    assert rows == 12

    data_version = getattr(metadata.get("data_version"), "value", None)
    assert data_version == "farslip-embeddings-italy-v1"

    model_version = getattr(metadata.get("model_version"), "value", None)
    assert model_version == "farslip-student-italy-v1"

    pairs_version = getattr(metadata.get("pairs_version"), "value", None)
    assert pairs_version == "farslip-pairs-italy-v1"

    # MLflow debe haber sido llamado para tags + metrics (B-5).
    assert mlflow_mock.log_metric.called
    assert mlflow_mock.log_param.called
    assert mlflow_mock.set_tag.called


def test_farslip_embeddings_italy_declares_model_dep() -> None:
    """Lineage B-5: ``farslip_embeddings_italy`` depende del modelo MLflow."""
    from dagster import AssetKey

    from dagster_project.assets.farslip import farslip_embeddings_italy

    deps = set(farslip_embeddings_italy.dependency_keys)
    assert AssetKey(["farslip_clip_italy_v1"]) in deps, (
        f"esperado AssetKey('farslip_clip_italy_v1') en deps, encontrado: {deps}"
    )


def test_farslip_consolidated_declares_lineage_to_model_and_extraction() -> None:
    """``farslip_embeddings_consolidated`` depende del extraction asset + modelo."""
    from dagster import AssetKey

    from dagster_project.assets.farslip_pipeline import (
        farslip_embeddings_consolidated,
    )

    deps = set(farslip_embeddings_consolidated.dependency_keys)
    assert AssetKey(["farslip_embeddings_italy"]) in deps
    assert AssetKey(["farslip_clip_italy_v1"]) in deps


def test_lineage_specs_registered_in_definitions() -> None:
    """Los AssetSpec externos aparecen en el AssetGraph global."""
    from dagster import AssetKey

    from dagster_project.definitions import defs

    asset_keys = set(defs.resolve_asset_graph().get_all_asset_keys())
    # Materializables.
    assert AssetKey(["sentinel2_crops_256"]) in asset_keys
    assert AssetKey(["farslip_embeddings_italy"]) in asset_keys
    assert AssetKey(["farslip_embeddings_consolidated"]) in asset_keys
    # External specs (lineage del paper).
    assert AssetKey(["farslip_pairs_italy"]) in asset_keys
    assert AssetKey(["farslip_clip_italy_v1"]) in asset_keys


def test_mlflow_resource_registered() -> None:
    """``defs.resources["mlflow"]`` esta presente para los assets B-5."""
    from dagster_project.definitions import defs

    resource_keys = set(defs.resources.keys())
    assert "mlflow" in resource_keys, (
        f"esperado resource 'mlflow' en defs.resources, encontrado: {resource_keys}"
    )


def test_farslip_full_pipeline_job_defined() -> None:
    """``farslip_full_pipeline_job`` selecciona el subgrafo materializable."""
    from dagster_project.definitions import defs
    from dagster_project.jobs import farslip_full_pipeline_job

    job = defs.resolve_job_def("farslip_full_pipeline_job")
    assert job is not None
    assert job.name == "farslip_full_pipeline_job"

    # Tags US-022b (B-5).
    assert farslip_full_pipeline_job.tags.get("us") == "US-022b"
    assert farslip_full_pipeline_job.tags.get("pipeline") == "farslip"


def test_farslip_pairs_italy_spec_tags() -> None:
    """``farslip_pairs_italy_spec`` carga el tag DVC B-5."""
    from dagster_project.assets.farslip_pipeline import farslip_pairs_italy_spec

    metadata = farslip_pairs_italy_spec.metadata or {}
    data_version_meta: Any = metadata.get("data_version")
    data_version = getattr(data_version_meta, "value", data_version_meta)
    assert data_version == "farslip-pairs-italy-v1"


def test_farslip_clip_italy_v1_spec_tags() -> None:
    """``farslip_clip_italy_v1_spec`` carga el tag DVC del student + Registry URI."""
    from dagster_project.assets.farslip_pipeline import (
        FARSLIP_REGISTRY_URI,
        farslip_clip_italy_v1_spec,
    )

    metadata = farslip_clip_italy_v1_spec.metadata or {}

    data_version = getattr(metadata.get("data_version"), "value", None)
    assert data_version == "farslip-student-italy-v1"

    registry_uri = getattr(metadata.get("registry_uri"), "value", None)
    assert registry_uri == FARSLIP_REGISTRY_URI
    assert registry_uri == "models:/farslip-clip-italy-v1/Production"


def test_consolidated_path_matches_fusion_contract() -> None:
    """B-4: la ruta canonica debe coincidir con ``ml.features.fusion._DEFAULT_FARSLIP_PATH``."""
    from dagster_project.assets.farslip_pipeline import (
        DATA_FARSLIP_CONSOLIDATED_PATH,
    )
    from ml.features.fusion import _DEFAULT_FARSLIP_PATH

    assert DATA_FARSLIP_CONSOLIDATED_PATH == _DEFAULT_FARSLIP_PATH, (
        f"Contrato roto: asset escribe {DATA_FARSLIP_CONSOLIDATED_PATH} pero "
        f"fusion.py lee {_DEFAULT_FARSLIP_PATH}"
    )
