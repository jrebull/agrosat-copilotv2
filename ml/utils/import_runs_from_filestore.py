"""Imports MLflow runs from a local file store into the MLflow server (Postgres).

Use case (US-025): the real TSViT runs were trained on the L4 VM and their
tracking ended up in a file store ``./mlruns/<exp>/`` (file format), while the
project's local MLflow server is a Docker with a Postgres backend
(``http://localhost:5010``). The two stores are different: the server does not
read the ``./mlruns/`` folder. This module reconstructs each run from the file
store in the server's experiment via :class:`mlflow.tracking.MlflowClient`,
preserving params, tags (including ``code_version`` and ``data_version``,
required by rule 10 of ``CLAUDE.md``), the per-epoch metric series and the
original timestamps.

Permanent operational tool (not an ad-hoc ``scripts/_*.py`` script): the run
selection is done by an *allowlist* of ``run_id`` so as not to drag in smokes or
abandoned attempts that share a ``run_name`` with the real ones. The import is
idempotent: re-running detects the already-present ``run_name`` and skips it.

CLI usage::

    python -m ml.utils.import_runs_from_filestore \\
        --src-experiment-dir mlruns/965679031955557780 \\
        --run-ids 3955879d26e4498a860517c10867d672,63aacbec1ffb45d493d15ceb63d73210 \\
        --dest-experiment-id 7
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

# MLflow >= 3.13 raises on file:// stores unless explicitly allowed; this module
# exists precisely to read legacy ./mlruns file stores, so opt in up front.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)

#: Map from file store status (integer) to the MLflow status string.
_STATUS_MAP = {"1": "RUNNING", "2": "SCHEDULED", "3": "FINISHED", "4": "FAILED", "5": "KILLED"}

#: Tags that are NOT copied: ``mlflow.runName`` is injected by ``create_run`` via
#: ``run_name=`` (copying it again would duplicate it).
_SKIP_TAGS = frozenset({"mlflow.runName"})


@contextlib.contextmanager
def _silence_mlflow_url() -> Iterator[None]:
    """Suppresses the MLflow URL ``print`` (it contains an emoji).

    ``MlflowClient.set_terminated``/``create_run`` write to ``stdout`` a line
    decorated with an emoji (``\\U0001f3c3``); on Windows consoles with cp1252
    encoding that raises ``UnicodeEncodeError``. Redirects ``stdout`` to a buffer
    during the call so that the structured log (structlog, on ``stderr``) stays
    visible without breaking the import.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _read_run_dir(run_dir: Path) -> dict:
    """Parses a file store run directory into a structured dict.

    Args:
        run_dir: Path to the run directory (``<exp>/<run_id>/``).

    Returns:
        Dict with ``meta`` (start_time, end_time, status, run_name, user_id),
        ``params`` (dict), ``tags`` (dict) and ``metrics`` (dict
        ``name -> list[(value, timestamp_ms, step)]`` with the per-epoch series).

    Raises:
        FileNotFoundError: if ``meta.yaml`` is missing.
    """
    meta_path = run_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.yaml does not exist in {run_dir}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    params: dict[str, str] = {}
    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for p in params_dir.iterdir():
            if p.is_file():
                params[p.name] = p.read_text(encoding="utf-8").strip()

    tags: dict[str, str] = {}
    tags_dir = run_dir / "tags"
    if tags_dir.is_dir():
        for t in tags_dir.iterdir():
            if t.is_file():
                tags[t.name] = t.read_text(encoding="utf-8").strip()

    metrics: dict[str, list[tuple[float, int, int]]] = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        for m in metrics_dir.iterdir():
            if not m.is_file():
                continue
            points: list[tuple[float, int, int]] = []
            for line in m.read_text(encoding="utf-8").splitlines():
                if not line.strip():  # guards against empty lines / CRLF on Windows
                    continue
                ts, val, step = line.split()
                points.append((float(val), int(ts), int(step)))
            metrics[m.name] = points

    return {"meta": meta, "params": params, "tags": tags, "metrics": metrics}


def _run_exists(client: MlflowClient, experiment_id: str, run_name: str) -> str | None:
    """Returns the existing run_id with that ``run_name`` in the experiment, or None.

    The match is by ``run_name`` (not by ``run_id``) because ``create_run``
    generates a new id in Postgres: the file store id is not preserved.

    Args:
        client: MLflow client pointing to the destination server.
        experiment_id: Id of the destination experiment.
        run_name: Name of the run to search for.

    Returns:
        The ``run_id`` of the first match, or ``None`` if it does not exist.
    """
    existing = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    return str(existing[0].info.run_id) if existing else None


def import_run(
    client: MlflowClient,
    run_dir: Path,
    dest_experiment_id: str,
    *,
    recreate: bool = False,
    upload_artifacts: bool = False,
) -> str | None:
    """Reconstructs a file store run in the server's destination experiment.

    Idempotent: if a run with the same ``run_name`` already exists in the
    destination, it skips it (or deletes and recreates it if ``recreate=True``).
    The creation is wrapped in a try/except that deletes the run if something
    fails before ``set_terminated``, so as not to leave an incomplete ``RUNNING``
    run that would block future runs.

    Args:
        client: MLflow client pointing to the destination server.
        run_dir: Run directory in the file store.
        dest_experiment_id: Id of the destination experiment.
        recreate: If ``True`` and the run already exists, it deletes and recreates
            it.
        upload_artifacts: If ``True`` uploads ``best.pt`` via ``log_artifact``
            (requires proxied-artifacts enabled on the server).

    Returns:
        The created ``run_id``, or ``None`` if it was skipped (already existed and
        not recreate).
    """
    from mlflow.entities import Metric, Param

    parsed = _read_run_dir(run_dir)
    meta = parsed["meta"]
    run_name = meta["run_name"]

    existing_id = _run_exists(client, dest_experiment_id, run_name)
    if existing_id is not None:
        if not recreate:
            logger.info(
                "import_run_skip", run_name=run_name, reason="ya existe", existing=existing_id[:12]
            )
            return None
        client.delete_run(existing_id)
        logger.info("import_run_deleted_for_recreate", run_name=run_name, deleted=existing_id[:12])

    # Tags to copy (excludes mlflow.runName, which create_run sets via run_name=).
    run_tags = {k: v for k, v in parsed["tags"].items() if k not in _SKIP_TAGS}

    with _silence_mlflow_url():
        run = client.create_run(
            experiment_id=dest_experiment_id,
            start_time=int(meta["start_time"]),
            run_name=run_name,
            tags=run_tags,
        )
    run_id = str(run.info.run_id)
    try:
        # Params in a single batch.
        params = [Param(k, str(v)) for k, v in parsed["params"].items()]
        # Metrics: all the per-epoch series, preserving timestamp and step.
        metric_entities: list[Metric] = []
        for name, points in parsed["metrics"].items():
            for value, ts, step in points:
                metric_entities.append(Metric(name, value, ts, step))
        # log_batch accepts <1000 metrics / <100 params per call.
        client.log_batch(run_id, metrics=metric_entities, params=params, tags=[])

        if upload_artifacts:
            best_pt = run_dir / "artifacts" / "checkpoint" / "best.pt"
            if best_pt.is_file():
                with _silence_mlflow_url():
                    client.log_artifact(run_id, str(best_pt), artifact_path="checkpoint")
                logger.info("import_run_artifact_uploaded", run_name=run_name, file="best.pt")

        status = _STATUS_MAP.get(str(meta.get("status")), "FINISHED")
        with _silence_mlflow_url():
            client.set_terminated(run_id, status=status, end_time=int(meta["end_time"]))
    except Exception:
        # Do not leave an incomplete RUNNING run: delete it to keep idempotency.
        client.delete_run(run_id)
        logger.error("import_run_failed_rolled_back", run_name=run_name, run_id=run_id[:12])
        raise

    logger.info(
        "import_run_done",
        run_name=run_name,
        run_id=run_id[:12],
        n_metrics=len(metric_entities),
        n_params=len(params),
        artifacts=upload_artifacts,
    )
    return run_id


def import_runs_from_filestore(
    src_experiment_dir: Path | str,
    run_ids: list[str],
    dest_experiment_id: str,
    *,
    tracking_uri: str | None = None,
    recreate: bool = False,
    upload_artifacts: bool = False,
) -> dict[str, str | None]:
    """Imports an allowlist of runs from a file store to the MLflow server.

    Args:
        src_experiment_dir: Directory of the experiment in the file store
            (``mlruns/<exp_id>/``).
        run_ids: Allowlist of ``run_id`` to import (the rest are ignored).
        dest_experiment_id: Id of the destination experiment on the server.
        tracking_uri: URI of the server; if ``None`` uses
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        recreate: Recreate already-existing runs instead of skipping them.
        upload_artifacts: Upload ``best.pt`` (requires proxied-artifacts).

    Returns:
        Dict ``source_run_id -> destination_run_id`` (``None`` if skipped).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from ml.utils.mlflow_utils import resolve_tracking_uri

    uri = tracking_uri or resolve_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    src = Path(src_experiment_dir)
    result: dict[str, str | None] = {}
    for rid in run_ids:
        run_dir = src / rid
        if not run_dir.is_dir():
            logger.warning("import_run_missing_dir", run_id=rid, path=str(run_dir))
            result[rid] = None
            continue
        result[rid] = import_run(
            client,
            run_dir,
            dest_experiment_id,
            recreate=recreate,
            upload_artifacts=upload_artifacts,
        )
    logger.info("import_runs_summary", uri=uri, dest_exp=dest_experiment_id, imported=result)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Importa runs MLflow de un file store local al servidor Postgres."
    )
    parser.add_argument("--src-experiment-dir", required=True)
    parser.add_argument(
        "--run-ids", required=True, help="Lista separada por comas de run_id (allowlist)."
    )
    parser.add_argument("--dest-experiment-id", required=True)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--upload-artifacts", action="store_true")
    args = parser.parse_args(argv)

    import_runs_from_filestore(
        args.src_experiment_dir,
        [r.strip() for r in args.run_ids.split(",") if r.strip()],
        args.dest_experiment_id,
        tracking_uri=args.tracking_uri,
        recreate=args.recreate,
        upload_artifacts=args.upload_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
