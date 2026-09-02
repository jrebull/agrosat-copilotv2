"""Tests Dagster — asset ``drift_check`` (US-060).

Cubre:

- Materializacion con un subset REAL del parquet FarSLIP copiado al cwd ->
  ``MaterializeResult`` con ``drift_score``, ``data_version``, ``code_version``,
  y escritura del HTML en ``data/monitoring/drift/``.
- Skip graceful sin parquet upstream -> ``status="skipped_no_upstream"``,
  ``rows=0``, sin excepcion (CI sin secrets debe pasar).
- Trigger de la alerta cuando ``drift_score > 0.3`` (mock del ``drift_notifier``).
- ``drift_check`` registrado en ``Definitions`` con sus deps de ingesta y el
  schedule semanal.

No usa red ni GCS: ``_upload_to_gcs`` degrada a local sin credenciales.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from dagster import build_asset_context

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FARSLIP_PARQUET = _REPO_ROOT / "data" / "farslip" / "embeddings_pastis.parquet"


def _stage_real_subset(tmp_path: Path, n_rows: int = 600) -> Path:
    """Copia un subset REAL del parquet FarSLIP al cwd del test.

    El asset lee ``data/farslip/embeddings_pastis.parquet`` relativo al cwd; se
    materializa un subset real (no sintetico) para aislar el test. Skipea si el
    parquet real no esta disponible (DVC sin pull).
    """
    if not _FARSLIP_PARQUET.exists():
        pytest.skip(f"Parquet real ausente (DVC sin pull): {_FARSLIP_PARQUET}")
    df = pl.read_parquet(_FARSLIP_PARQUET).head(n_rows)
    target_dir = tmp_path / "data" / "farslip"
    target_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target_dir / "embeddings_pastis.parquet")
    return tmp_path


def _context() -> object:
    """Construye un context con mocks de ``mlflow`` y ``drift_notifier``."""
    notifier = MagicMock()
    notifier.send.return_value = False
    return build_asset_context(resources={"mlflow": MagicMock(), "drift_notifier": notifier})


def test_drift_check_materializes_with_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``drift_check`` materializa sobre datos reales con lineage completo."""
    from dagster_project.assets.drift import DRIFT_REPORT_DIR, drift_check

    cwd = _stage_real_subset(tmp_path)
    monkeypatch.chdir(cwd)
    context = _context()
    result = drift_check(context)

    metadata = result.metadata or {}
    status = getattr(metadata.get("status"), "value", None)
    assert status == "ok"

    drift_score = getattr(metadata.get("drift_score"), "value", None)
    assert isinstance(drift_score, float)
    assert 0.0 <= drift_score <= 1.0

    data_version = getattr(metadata.get("data_version"), "value", None)
    assert isinstance(data_version, str) and data_version

    code_version = getattr(metadata.get("code_version"), "value", None)
    assert isinstance(code_version, str) and code_version

    n_embedding_dims = getattr(metadata.get("n_embedding_dims"), "value", None)
    assert n_embedding_dims == 64  # contrato AlphaEarth 64-dim

    # El HTML semanal debe existir localmente (sin GCS).
    report_dir = cwd / DRIFT_REPORT_DIR
    htmls = list(report_dir.glob("report_*.html"))
    assert htmls, f"esperado report_*.html en {report_dir}"
    assert htmls[0].stat().st_size > 0


def test_drift_check_alert_triggers_notifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con el contraste Plan B (clase mayoritaria) el drift supera 0.3 y alerta."""
    from dagster_project.assets.drift import drift_check

    cwd = _stage_real_subset(tmp_path)
    monkeypatch.chdir(cwd)
    notifier = MagicMock()
    notifier.send.return_value = False
    context = build_asset_context(resources={"mlflow": MagicMock(), "drift_notifier": notifier})
    result = drift_check(context)

    metadata = result.metadata or {}
    alert = getattr(metadata.get("alert_triggered"), "value", None)
    drift_score = getattr(metadata.get("drift_score"), "value", 0.0)
    # El contraste Plan B (ref = no-mayoritaria, current = mayoritaria) es un
    # cambio de distribucion REAL fuerte -> alerta esperada.
    assert alert is True
    assert drift_score > 0.3
    notifier.send.assert_called_once()


def test_drift_check_skips_without_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin parquet upstream -> status skipped_no_upstream, rows=0, sin excepcion."""
    from dagster_project.assets.drift import drift_check

    # cwd vacio: no existe data/farslip/embeddings_pastis.parquet.
    monkeypatch.chdir(tmp_path)
    context = _context()
    result = drift_check(context)

    metadata = result.metadata or {}
    status = getattr(metadata.get("status"), "value", None)
    rows = getattr(metadata.get("rows"), "value", None)
    assert status == "skipped_no_upstream"
    assert rows == 0


def test_drift_check_registered_in_definitions() -> None:
    """``drift_check`` + deps de ingesta + schedule semanal en ``Definitions``."""
    from dagster import AssetKey

    from dagster_project.definitions import defs

    graph = defs.resolve_asset_graph()
    keys = {".".join(k.path) for k in graph.get_all_asset_keys()}
    assert "drift_check" in keys

    deps = {".".join(k.path) for k in graph.get(AssetKey(["drift_check"])).parent_keys}
    assert "farslip_embeddings_consolidated" in deps
    assert "parcel_features_fused" in deps

    assert "drift_notifier" in defs.resources

    schedule_names = {s.name for s in defs.schedules}
    assert "drift_check_weekly_schedule" in schedule_names
    sched = next(s for s in defs.schedules if s.name == "drift_check_weekly_schedule")
    assert sched.cron_schedule == "0 6 * * 1"
