"""AC-3 gate of US-017 / US-016b — audit of the farslip_pairs dataset.

CLI ``python -m ml.farslip.dataset_audit`` that validates:

1. ``n_pairs >= 30000`` (exit 0). ``20000 <= n_pairs < 30000`` => warning + exit 0.
2. ``n_pairs < 20000`` => exit 1.
3. Balance ``min(n_per_roi) / max(n_per_roi) >= 0.20`` => exit 0; else exit 1.

Used by ``make farslip-dataset-check``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import structlog

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the dataset_audit CLI. poetry add typer") from exc

_log = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=False)


MIN_HARD = 20_000
MIN_OK = 30_000
BALANCE_THRESHOLD = 0.20


def audit_dataset(
    dataset_root: Path = Path("data/farslip_pairs"),
    min_hard: int = MIN_HARD,
    min_ok: int = MIN_OK,
    balance_threshold: float = BALANCE_THRESHOLD,
) -> int:
    """Audit the dataset and return an exit code (0 = OK, 1 = hard failure).

    Args:
        dataset_root: root containing ``{roi}/manifest.parquet``.
        min_hard: hard threshold (below => exit 1).
        min_ok: OK threshold (in [hard, ok) => warning).
        balance_threshold: minimum acceptable min/max per-ROI ratio.
    """
    if not dataset_root.exists():
        _log.error("dataset_root no existe", path=str(dataset_root))
        return 1

    manifests = sorted(dataset_root.glob("*/manifest.parquet"))
    if not manifests:
        _log.error("no se encontro ningun manifest", root=str(dataset_root))
        return 1

    per_roi_counts: dict[str, int] = {}
    total = 0
    for m in manifests:
        roi = m.parent.name
        df = pl.read_parquet(m)
        count = df.height
        per_roi_counts[roi] = count
        total += count

    _log.info("totals", total=total, per_roi=per_roi_counts)

    if total < min_hard:
        _log.error(
            "dataset por debajo del umbral hard",
            total=total,
            min_hard=min_hard,
        )
        return 1
    if total < min_ok:
        _log.warning(
            "dataset por debajo de umbral OK (continuando)",
            total=total,
            min_ok=min_ok,
        )

    counts = list(per_roi_counts.values())
    if max(counts) == 0:
        _log.error("max count es 0; data inconsistente")
        return 1
    ratio = min(counts) / max(counts)
    if ratio < balance_threshold:
        _log.error(
            "balance min/max por ROI por debajo de umbral",
            ratio=ratio,
            threshold=balance_threshold,
            per_roi=per_roi_counts,
        )
        return 1
    _log.info("audit OK", total=total, balance_ratio=ratio)
    return 0


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Option(help="Raiz del dataset farslip_pairs")] = Path(
        "data/farslip_pairs"
    ),
    min_hard: Annotated[int, typer.Option(help="Umbral hard exit 1")] = MIN_HARD,
    min_ok: Annotated[int, typer.Option(help="Umbral warning")] = MIN_OK,
    balance_threshold: Annotated[
        float, typer.Option(help="Ratio min/max ROI aceptable")
    ] = BALANCE_THRESHOLD,
) -> None:
    """CLI entrypoint — exit code consumed by make."""
    code = audit_dataset(
        dataset_root=dataset_root,
        min_hard=min_hard,
        min_ok=min_ok,
        balance_threshold=balance_threshold,
    )
    raise typer.Exit(code=code)


if __name__ == "__main__":  # pragma: no cover
    app()
