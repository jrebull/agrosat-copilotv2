"""Ingest the 3 FarSLIP H100 training logs into the local MLflow Docker (:5010).

The H100 VM cannot run Docker (nested virtualization disabled on the Azure GPU
VM), so the runs were not registered in MLflow during training; their metrics
live in plain logs. This script parses those logs (brought to local by scp) and
registers one MLflow run per variant with the config params and per-epoch
metrics, so the lineage is centralized in the local MLflow server.

Usage (from repo root, MLflow Docker up):
    poetry run python scripts/ingest_farslip_logs_to_mlflow.py
"""

from __future__ import annotations

import re
from pathlib import Path

import mlflow
import structlog

logger = structlog.get_logger(__name__)

_TRACKING_URI = "http://localhost:5010"
_EXPERIMENT = "farslip"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _REPO_ROOT / "reports" / "farslip" / "logs"

#: log file -> MLflow run name (the variant identifier reported in the handoffs).
_RUNS: dict[str, str] = {
    "4band-pheno.log": "farslip-pheno-fix-v1",
    "baseline-rgb.log": "farslip-baseline-rgb",
    "baseline-nir.log": "farslip-baseline-nir",
}

#: Config keys parsed from the "starting farslip training" line.
_CONFIG_KEYS = (
    "band_selection",
    "batch_size",
    "epochs",
    "lr",
    "n_in_channels",
    "proto_source",
    "rois",
    "seed",
)
_METRIC_KEYS = ("loss_aux", "loss_cls", "loss_patch", "loss_total")


def _parse_kv(line: str, keys: tuple[str, ...]) -> dict[str, str]:
    """Extract ``key=value`` tokens for the given keys from a structlog line."""
    out: dict[str, str] = {}
    for k in keys:
        m = re.search(rf"\b{k}=([^\s]+)", line)
        if m:
            out[k] = m.group(1)
    return out


def _parse_log(path: Path) -> tuple[dict[str, str], list[dict[str, float]]]:
    """Parse a FarSLIP log into (config_params, per_epoch_metrics).

    Returns the config dict from the start line and a list of per-epoch metric
    dicts (one per ``epoch done`` line, ordered by epoch).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    config: dict[str, str] = {}
    epochs: list[dict[str, float]] = []
    for line in text.splitlines():
        if "starting farslip training" in line:
            config = _parse_kv(line, _CONFIG_KEYS)
        elif "epoch done" in line:
            kv = _parse_kv(line, ("epoch", *_METRIC_KEYS))
            if "epoch" in kv:
                epochs.append(
                    {
                        "epoch": int(kv["epoch"]),
                        **{m: float(kv[m]) for m in _METRIC_KEYS if m in kv},
                    }
                )
    epochs.sort(key=lambda e: e["epoch"])
    return config, epochs


def main() -> int:
    """Register the 3 FarSLIP runs in MLflow from their logs. Returns exit code."""
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment(_EXPERIMENT)

    for log_name, run_name in _RUNS.items():
        log_path = _LOG_DIR / log_name
        if not log_path.exists():
            logger.warning("log_missing", path=str(log_path))
            continue
        config, epochs = _parse_log(log_path)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("source", "h100_log_ingest")
            mlflow.set_tag("note", "trained on H100 VM; logged from file (no Docker on VM)")
            mlflow.log_params(config)
            for ep in epochs:
                step = int(ep["epoch"])
                for m in _METRIC_KEYS:
                    if m in ep:
                        mlflow.log_metric(m, ep[m], step=step)
            mlflow.log_artifact(str(log_path), artifact_path="training_log")
            logger.info(
                "run_registered", run_name=run_name, n_epochs=len(epochs), params=len(config)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
