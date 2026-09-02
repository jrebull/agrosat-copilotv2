"""MLflow tracking utilities for the project experiments (US-019).

Centralizes opening MLflow runs with the mandatory tags
``data_version`` and ``code_version`` (CLAUDE.md rule 10). Reuses
:mod:`ml.utils.git_meta` to resolve the git SHA and the DVC hash instead
of re-implementing the ``subprocess`` call (DRY, decision D7).

Tracking URI resolution (decision D8/D14, AC-14):

1. Explicit ``override`` passed by the caller.
2. The ``MLFLOW_TRACKING_URI`` environment variable.
3. The local Docker MLflow server ``http://localhost:5010`` if it responds
   to ``/health`` within the timeout.
4. ``file:./mlruns`` as fallback so a dev without Docker running (or the
   CI) is not blocked.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import mlflow
import structlog

from ml.utils.git_meta import dvc_data_version, git_sha

if TYPE_CHECKING:  # pragma: no cover - only for type annotations
    from mlflow import ActiveRun

logger = structlog.get_logger(__name__)

__all__ = [
    "resolve_tracking_uri",
    "server_is_reachable",
    "track_experiment",
]

_DEFAULT_SERVER_URL = "http://localhost:5010"
_DEFAULT_FILE_STORE = "file:./mlruns"
_HEALTH_TIMEOUT_S = 2.0


def server_is_reachable(server_url: str, *, timeout: float = _HEALTH_TIMEOUT_S) -> bool:
    """Probe via HTTP the ``/health`` endpoint of the MLflow server.

    Args:
        server_url: Base URL of the MLflow server (without ``/health``).
        timeout: Timeout in seconds of the HTTP request.

    Returns:
        ``True`` if ``/health`` responds with a 2xx code within the
        timeout; ``False`` on any network error or timeout.
    """
    health_url = f"{server_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:  # noqa: S310 - local dev URL
            status: int = response.status
            return 200 <= status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Backward-compatible alias (internal modules use it).
_server_is_reachable = server_is_reachable


def resolve_tracking_uri(
    override: str | None = None,
    *,
    server_url: str = _DEFAULT_SERVER_URL,
    probe_server: bool = True,
) -> str:
    """Resolve the MLflow tracking URI with gradual fallback.

    Priority: ``override`` > ``$MLFLOW_TRACKING_URI`` > ``server_url`` (if it
    responds to ``/health``) > ``file:./mlruns``.

    Args:
        override: Explicit URI; if passed, it is used without further checks.
        server_url: URL of the local Docker MLflow server to probe.
        probe_server: If ``True`` (default) probes ``/health`` before
            choosing ``server_url``; if the server does not respond,
            degrades to the file store with a ``log.warning``. If ``False``
            the server is not contacted (useful for deterministic tests).

    Returns:
        The resolved tracking URI as a string.
    """
    if override:
        return override

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if not probe_server:
        return _DEFAULT_FILE_STORE

    if _server_is_reachable(server_url):
        logger.info("mlflow_tracking_uri_resolved", uri=server_url, source="docker_server")
        return server_url

    logger.warning(
        "mlflow_server_unreachable_fallback_file_store",
        server_url=server_url,
        fallback=_DEFAULT_FILE_STORE,
        note="Servidor MLflow Docker no responde; usar `make mlflow-up` para levantarlo.",
    )
    return _DEFAULT_FILE_STORE


@contextmanager
def track_experiment(
    experiment_name: str,
    *,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    dvc_path: str | None = None,
    probe_server: bool = True,
) -> Iterator[ActiveRun]:
    """Context manager that opens an MLflow run with versioning tags.

    Resolves the tracking URI, sets the experiment, opens a run and injects
    the tags ``code_version`` (git SHA via :func:`git_sha`) and
    ``data_version`` (DVC hash via :func:`dvc_data_version` if ``dvc_path``
    is passed, or ``"untracked"`` otherwise).

    Args:
        experiment_name: Name of the MLflow experiment (created if it does
            not exist).
        run_name: Human-readable name of the run; ``None`` lets MLflow
            generate a random one.
        tracking_uri: Override of the tracking URI; if ``None`` it is
            delegated to :func:`resolve_tracking_uri`.
        dvc_path: Path to the DVC-tracked dataset to resolve the
            ``data_version``. If ``None`` the tag stays as
            ``"untracked"``.
        probe_server: Forwarded to :func:`resolve_tracking_uri`; set it
            to ``False`` in tests to avoid contacting the Docker server.

    Yields:
        The active :class:`mlflow.ActiveRun`, to log params, metrics
        and artifacts within the ``with`` block.
    """
    resolved_uri = resolve_tracking_uri(tracking_uri, probe_server=probe_server)
    # MLflow 3.x puts the file store (file:./mlruns) in "maintenance mode" and raises
    # an exception unless it is explicitly enabled. For the project's local/Colab flow
    # the file store is sufficient, so we allow it.
    if resolved_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    code_version = git_sha()
    data_version = dvc_data_version(dvc_path) if dvc_path else "untracked"

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tag("code_version", code_version)
        mlflow.set_tag("data_version", data_version)
        logger.info(
            "mlflow_run_started",
            experiment=experiment_name,
            run_name=run_name,
            run_id=active_run.info.run_id,
            tracking_uri=resolved_uri,
            code_version=code_version,
            data_version=data_version,
        )
        yield active_run
