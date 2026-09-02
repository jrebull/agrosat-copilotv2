"""DEPRECATED (US-030) — fold-4 metric export for ``Avance4.Equipo17.ipynb``.

> **OBSOLETO**: este script exportaba metricas medidas en el **fold-4** (set de
> SELECCION, NO held-out) de los modelos del equipo (DeepLabv3+, TSViT,
> TSViT-pheno). US-030 reemplaza esa comparativa por el re-score
> **apples-to-apples sobre el fold-5 held-out** de los 6 segmentadores reales.
> Los valores fold-4 de abajo (``deeplabv3plus`` mIoU 0.2709, ``tsvit`` 0.6215,
> ``tsvit-pheno`` 0.6253) quedan **invalidados**: mezclaban convenciones por
> modelo (18 vs 20 clases, 128 vs 256 px, 10 vs 3 bandas) y un fold de seleccion,
> por lo que NO eran comparables (ver ``docs/us-planning/us-030.md`` 10, AC-5).

La fuente de verdad de la tabla A4 es ahora
``reports/segmentation/metrics/model_comparison_fold5.csv``, generada por:

    from ml.eval.dense_metrics import rescore_all_checkpoints
    from ml.eval.comparison import build_fold5_comparison_table

    df = rescore_all_checkpoints(fold=5)              # 6 best.pt, held-out fold-5
    build_fold5_comparison_table(df)                  # escribe el CSV consolidado

Este modulo se conserva SOLO como referencia historica de los valores fold-4 y
para reproducir los parquets antiguos si una rama heredada los necesita. Ejecutar
``main()`` ahora aborta con un mensaje de migracion salvo que se pase
``--force-legacy`` explicitamente.

Uso (re-score fold-5, recomendado)::

    poetry run python -c "from ml.eval.dense_metrics import rescore_all_checkpoints; \\
        from ml.eval.comparison import build_fold5_comparison_table; \\
        build_fold5_comparison_table(rescore_all_checkpoints(fold=5))"

Uso (legacy fold-4, desaconsejado)::

    poetry run python scripts/export_avance4_metrics_us025.py --force-legacy
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

# OBSOLETO (fold-4, set de SELECCION). Conservado solo como referencia historica.
# La comparativa vigente vive en model_comparison_fold5.csv (fold-5 held-out).
#
# ``train_time_s`` es el wall-clock real del run de entrenamiento medido en el
# server MLflow local (Docker Postgres :5010, experimento 7
# ``agrosat-segmentation``): ``end_time - start_time`` del run FINISHED.
#   - deeplabv3plus (mobilenet, 18 clases, run 1c1e4f6f): 44676.4 s (~12.4 h).
#   - tsvit          (run 24f70756, batch 16, 30 epochs):  1894.4 s (~31.6 min, RTX 4070).
#   - tsvit-pheno    (run 0eef8a60, batch 16, 30 epochs):  1915.4 s (~31.9 min, RTX 4070).
_ROWS_LEGACY_FOLD4 = [
    {
        "model": "deeplabv3plus",
        "miou": 0.2709,
        "f1_macro": 0.3864,
        "pixel_accuracy": 0.6743,
        "miou_grouped": 0.4682,
        "f1_macro_grouped": 0.6009,
        "pixel_accuracy_grouped": 0.8018,
        "train_time_s": 44676.4,
        "epochs": 15,
        "n_train": None,
        "n_val": 482,
        "n_trainable_params": None,
        "target_size": 128,
        "device": "cuda",
    },
    {
        "model": "tsvit",
        "miou": 0.6215,
        "f1_macro": 0.7473,
        "pixel_accuracy": 0.8724,
        "miou_grouped": None,
        "f1_macro_grouped": None,
        "pixel_accuracy_grouped": None,
        "train_time_s": 1894.4,
        "epochs": 30,
        "n_train": None,
        "n_val": 482,
        "n_trainable_params": None,
        "target_size": 128,
        "device": "cuda",
    },
    {
        "model": "tsvit-pheno",
        "miou": 0.6253,
        "f1_macro": 0.7500,
        "pixel_accuracy": 0.8759,
        "miou_grouped": None,
        "f1_macro_grouped": None,
        "pixel_accuracy_grouped": None,
        "train_time_s": 1915.4,
        "epochs": 30,
        "n_train": None,
        "n_val": 482,
        "n_trainable_params": None,
        "target_size": 128,
        "device": "cuda",
    },
]

# Explicit schema: prevents Polars from inferring Null on columns with only None.
_SCHEMA = {
    "model": pl.Utf8,
    "miou": pl.Float64,
    "f1_macro": pl.Float64,
    "pixel_accuracy": pl.Float64,
    "miou_grouped": pl.Float64,
    "f1_macro_grouped": pl.Float64,
    "pixel_accuracy_grouped": pl.Float64,
    "train_time_s": pl.Float64,
    "epochs": pl.Int64,
    "n_train": pl.Int64,
    "n_val": pl.Int64,
    "n_trainable_params": pl.Int64,
    "target_size": pl.Int64,
    "device": pl.Utf8,
}

_MIGRATION_NOTE = (
    "scripts/export_avance4_metrics_us025.py esta OBSOLETO (US-030): los valores "
    "fold-4 ya no son la comparativa vigente. Genera el fold-5 held-out con "
    "rescore_all_checkpoints(fold=5) + build_fold5_comparison_table(...). "
    "Para reproducir los parquets fold-4 heredados, pasa --force-legacy."
)


def write_legacy_fold4_parquets() -> tuple[Path, Path]:
    """Write the obsolete fold-4 parquets (DeepLabv3+ and TSViT).

    Reproduces the historical fold-4 parquets consumed by the old consolidation
    cell of ``Avance4.Equipo17.ipynb``. Kept only for legacy branches; the
    current comparison is ``model_comparison_fold5.csv`` (US-030).

    Returns:
        Tuple ``(deeplab_parquet, tsvit_parquet)`` of the written paths.
    """
    out_dir = Path("reports/segmentation/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(_ROWS_LEGACY_FOLD4, schema=_SCHEMA)
    deeplab = df.filter(pl.col("model") == "deeplabv3plus")
    tsvit = df.filter(pl.col("model").is_in(["tsvit", "tsvit-pheno"]))

    p_deeplab = out_dir / "model_comparison_avance4_deeplabv3plus.parquet"
    p_tsvit = out_dir / "model_comparison_avance4_tsvit.parquet"
    deeplab.write_parquet(p_deeplab)
    tsvit.write_parquet(p_tsvit)

    logger.warning(
        "legacy_fold4_parquets_written",
        deeplab=str(p_deeplab),
        tsvit=str(p_tsvit),
        note="fold-4 obsoleto; usar model_comparison_fold5.csv (US-030)",
    )
    return p_deeplab, p_tsvit


def main(argv: list[str] | None = None) -> int:
    """Abort with a migration note unless ``--force-legacy`` is passed.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if the legacy parquets were written; ``2`` if aborted (obsolete).
    """
    args = sys.argv[1:] if argv is None else argv
    if "--force-legacy" not in args:
        logger.warning("export_avance4_deprecated", note=_MIGRATION_NOTE)
        return 2

    write_legacy_fold4_parquets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
