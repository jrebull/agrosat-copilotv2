"""Tests de ml.utils.import_runs_from_filestore (US-025).

Valida el parseo del file store y la importacion idempotente contra un tracking
store de archivos temporal (``file:./tmp_mlruns``), sin Docker ni Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.utils.import_runs_from_filestore import (
    _read_run_dir,
    import_runs_from_filestore,
)


def _write_run_dir(base: Path, run_id: str, run_name: str) -> Path:
    """Crea un run dir sintetico con el formato del file store de MLflow."""
    run_dir = base / run_id
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "tags").mkdir()
    (run_dir / "meta.yaml").write_text(
        "start_time: 1780283074733\n"
        "end_time: 1780284969133\n"
        "status: 3\n"
        f"run_name: {run_name}\n"
        "user_id: tester\n",
        encoding="utf-8",
    )
    (run_dir / "params" / "epochs").write_text("30", encoding="utf-8")
    (run_dir / "params" / "batch_size").write_text("16", encoding="utf-8")
    # best metric (single point) + serie por epoch (3 puntos), con linea vacia final.
    (run_dir / "metrics" / "best_val_miou").write_text("1780286907592 0.6253 0\n", encoding="utf-8")
    (run_dir / "metrics" / "val_miou").write_text(
        "1780283135349 0.026 0\n1780283198773 0.082 1\n1780283261321 0.155 2\n\n",
        encoding="utf-8",
    )
    (run_dir / "tags" / "mlflow.runName").write_text(run_name, encoding="utf-8")
    (run_dir / "tags" / "code_version").write_text("3b2c8c8b", encoding="utf-8")
    (run_dir / "tags" / "data_version").write_text("data/PASTIS-R@untracked", encoding="utf-8")
    return run_dir


def test_read_run_dir_parses_all_sections(tmp_path: Path) -> None:
    """_read_run_dir extrae meta, params, tags y series de metricas."""
    src = tmp_path / "exp"
    run_dir = _write_run_dir(src, "abc123", "alt-tsvit-pheno-v1")
    parsed = _read_run_dir(run_dir)
    assert parsed["meta"]["run_name"] == "alt-tsvit-pheno-v1"
    assert parsed["params"]["epochs"] == "30"
    assert parsed["tags"]["code_version"] == "3b2c8c8b"
    assert parsed["tags"]["data_version"] == "data/PASTIS-R@untracked"
    # La serie val_miou tiene 3 puntos (la linea vacia final se ignora).
    assert len(parsed["metrics"]["val_miou"]) == 3
    assert len(parsed["metrics"]["best_val_miou"]) == 1


def test_read_run_dir_missing_meta_raises(tmp_path: Path) -> None:
    """Falta meta.yaml -> FileNotFoundError."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        _read_run_dir(tmp_path / "empty")


def test_import_is_idempotent(tmp_path: Path) -> None:
    """Importar dos veces no duplica: la 2da corrida hace SKIP."""
    import mlflow

    src = tmp_path / "exp"
    _write_run_dir(src, "run-aaa", "alt-tsvit-v1")
    _write_run_dir(src, "run-bbb", "alt-tsvit-pheno-v1")

    dest = f"file:{(tmp_path / 'dest_mlruns').as_posix()}"
    mlflow.set_tracking_uri(dest)
    exp_id = mlflow.create_experiment("agrosat-segmentation")

    run_ids = ["run-aaa", "run-bbb"]
    first = import_runs_from_filestore(
        src, run_ids, exp_id, tracking_uri=dest, upload_artifacts=False
    )
    # Primera corrida crea ambos.
    assert all(v is not None for v in first.values())

    second = import_runs_from_filestore(
        src, run_ids, exp_id, tracking_uri=dest, upload_artifacts=False
    )
    # Segunda corrida los detecta y omite (None = skip).
    assert all(v is None for v in second.values())

    # Solo hay 2 runs en el experimento (no 4).
    runs = mlflow.search_runs(experiment_ids=[exp_id])
    assert len(runs) == 2


def test_import_preserves_metrics_and_version_tags(tmp_path: Path) -> None:
    """El run importado conserva best_val_miou y los tags de version (rubrica)."""
    import mlflow

    src = tmp_path / "exp"
    _write_run_dir(src, "run-ccc", "alt-tsvit-pheno-v1")
    dest = f"file:{(tmp_path / 'dest2').as_posix()}"
    mlflow.set_tracking_uri(dest)
    exp_id = mlflow.create_experiment("agrosat-segmentation")

    import_runs_from_filestore(src, ["run-ccc"], exp_id, tracking_uri=dest)

    runs = mlflow.search_runs(experiment_ids=[exp_id])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["metrics.best_val_miou"] == pytest.approx(0.6253)
    assert row["tags.code_version"] == "3b2c8c8b"
    assert row["tags.data_version"] == "data/PASTIS-R@untracked"
    assert row["status"] == "FINISHED"


def test_import_missing_run_id_is_skipped(tmp_path: Path) -> None:
    """Un run_id que no existe en el file store se omite con None."""
    import mlflow

    src = tmp_path / "exp"
    _write_run_dir(src, "run-real", "alt-tsvit-v1")
    dest = f"file:{(tmp_path / 'dest3').as_posix()}"
    mlflow.set_tracking_uri(dest)
    exp_id = mlflow.create_experiment("agrosat-segmentation")

    result = import_runs_from_filestore(
        src, ["run-real", "run-inexistente"], exp_id, tracking_uri=dest
    )
    assert result["run-inexistente"] is None
    assert result["run-real"] is not None
