"""Dagster assets — US-022b-C training of temporal models.

Declares the assets that train TempCNN and InceptionTime over the phenology FE
of the US-018 subset with 5-fold spatial CV, reusing
:func:`ml.train.phenology_models.train_temporal_model`.

The training does NOT run automatically from a schedule (infra decision of
US-022b-A: on-demand training via ``make reencuadre-notebook-full`` or manual
invocation of the asset). The assets are materialized when the user explicitly
requests it from the UI or the CLI. This avoids spending GPU by accident.

Mapping to acceptance criteria (docs/us-planning/us-022b.md §3.3):

- **C-3**: ``phenology_model_tempcnn`` + ``phenology_model_inceptiontime`` train
  the two models with the same spatial CV.
- **C-4**: ``temporal_models_comparison`` consolidates the metrics and reports
  the delta vs tabular baseline in ``reports/baseline/phenology_models.csv``.
- **MLflow**: each asset records params (model_kind, n_epochs, batch_size,
  device, n_parcels, n_classes), per-epoch metrics, OOF metrics and the
  state_dict of the last-fold model as an artifact.

Lineage:

::

    feature_selection_parcels_subset.parquet (data fixture US-018)
        |
        +-> phenology_model_tempcnn        (MLflow run + state_dict artifact)
        +-> phenology_model_inceptiontime  (MLflow run + state_dict artifact)
                |
                +-> temporal_models_comparison  (reports/baseline/phenology_models.csv)
"""

from pathlib import Path

import polars as pl
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

#: Canonical path of the features subset (US-018 + US-015).
_FEATURES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "test_fixtures"
    / "feature_selection_parcels_subset.parquet"
)

#: Canonical path of the comparison table produced by
#: ``temporal_models_comparison`` and consumed by notebook 05.
_REPORTS_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "baseline" / "phenology_models.csv"
)


def _train_one(
    context: AssetExecutionContext,
    *,
    model_kind: str,
    n_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, float | int | str]:
    """Internal helper: train a temporal model and return a serializable dict.

    Lazy import of ``train_temporal_model`` to avoid bringing up torch when only
    the asset graph is validated (`dagster definitions validate`).
    """
    from ml.train.phenology_models import train_temporal_model

    result = train_temporal_model(
        features_path=_FEATURES_PATH,
        model_kind=model_kind,  # type: ignore[arg-type]
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        # mlflow_uri is injected from the mlflow resource if present.
        mlflow_uri=None,
    )
    context.log.info(
        f"trained {model_kind}: F1-macro={result.f1_macro:.4f} "
        f"mIoU={result.miou:.4f} t={result.train_time_s:.1f}s "
        f"n_parcels={result.n_parcels} n_classes={result.n_classes}"
    )
    return {
        "model_kind": result.model_kind,
        "f1_macro": float(result.f1_macro),
        "f1_weighted": float(result.f1_weighted),
        "miou": float(result.miou),
        "cohen_kappa": float(result.cohen_kappa),
        "train_time_s": float(result.train_time_s),
        "n_parcels": int(result.n_parcels),
        "n_classes": int(result.n_classes),
        "mlflow_run_id": result.mlflow_run_id or "",
    }


@asset(
    group_name="phenology_models",
    description=(
        "TempCNN (Pelletier et al. 2019) nativo sobre la FE fenologica del "
        "subset US-018 con spatial CV 5-fold. Output: dict con metricas OOF "
        "y referencia al run MLflow."
    ),
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "model_arch": "ml.models.temporal.TempCNN",
    },
)
def phenology_model_tempcnn(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Train TempCNN over the phenology FE with 5-fold spatial CV."""
    metrics = _train_one(
        context,
        model_kind="tempcnn",
        n_epochs=30,
        batch_size=256,
        device="auto",
        seed=42,
    )
    return MaterializeResult(
        metadata={
            "f1_macro": MetadataValue.float(metrics["f1_macro"]),  # type: ignore[arg-type]
            "miou": MetadataValue.float(metrics["miou"]),  # type: ignore[arg-type]
            "train_time_s": MetadataValue.float(metrics["train_time_s"]),  # type: ignore[arg-type]
            "n_parcels": MetadataValue.int(metrics["n_parcels"]),  # type: ignore[arg-type]
            "mlflow_run_id": MetadataValue.text(str(metrics["mlflow_run_id"])),
            "data_version": MetadataValue.text("us018-phenology-subset"),
            "model_kind": MetadataValue.text("tempcnn"),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )


@asset(
    group_name="phenology_models",
    description=(
        "InceptionTime (Fawaz et al. 2020) nativo sobre la FE fenologica del "
        "subset US-018 con spatial CV 5-fold. Output: dict con metricas OOF "
        "y referencia al run MLflow."
    ),
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "model_arch": "ml.models.temporal.InceptionTime",
    },
)
def phenology_model_inceptiontime(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Train InceptionTime over the phenology FE with 5-fold spatial CV."""
    metrics = _train_one(
        context,
        model_kind="inceptiontime",
        n_epochs=30,
        batch_size=256,
        device="auto",
        seed=42,
    )
    return MaterializeResult(
        metadata={
            "f1_macro": MetadataValue.float(metrics["f1_macro"]),  # type: ignore[arg-type]
            "miou": MetadataValue.float(metrics["miou"]),  # type: ignore[arg-type]
            "train_time_s": MetadataValue.float(metrics["train_time_s"]),  # type: ignore[arg-type]
            "n_parcels": MetadataValue.int(metrics["n_parcels"]),  # type: ignore[arg-type]
            "mlflow_run_id": MetadataValue.text(str(metrics["mlflow_run_id"])),
            "data_version": MetadataValue.text("us018-phenology-subset"),
            "model_kind": MetadataValue.text("inceptiontime"),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )


@asset(
    group_name="phenology_models",
    description=(
        "Tabla comparativa de modelos temporales (TempCNN + InceptionTime) "
        "vs baseline tabular 0.32. Lee las metricas de los dos assets "
        "anteriores via MLflow y persiste reports/baseline/phenology_models.csv."
    ),
    deps=[phenology_model_tempcnn, phenology_model_inceptiontime],
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "output_path": str(_REPORTS_PATH),
    },
)
def temporal_models_comparison(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Persist the temporal-model comparison table to CSV.

    This retraining version of the asset is deliberately simple: it invokes
    ``train_temporal_model`` again for the two models and consolidates. To avoid
    retraining on each materialization, a later version may read the metrics from
    MLflow via ``mlflow.search_runs``; for now simplicity wins.
    """
    rows = []
    for model_kind in ("tempcnn", "inceptiontime"):
        metrics = _train_one(
            context,
            model_kind=model_kind,
            n_epochs=30,
            batch_size=256,
            device="auto",
            seed=42,
        )
        rows.append(
            {
                "model": model_kind,
                "f1_macro": round(metrics["f1_macro"], 4),  # type: ignore[arg-type]
                "f1_weighted": round(metrics["f1_weighted"], 4),  # type: ignore[arg-type]
                "miou": round(metrics["miou"], 4),  # type: ignore[arg-type]
                "cohen_kappa": round(metrics["cohen_kappa"], 4),  # type: ignore[arg-type]
                "train_time_s": round(metrics["train_time_s"], 2),  # type: ignore[arg-type]
                "n_parcels": metrics["n_parcels"],
                "delta_vs_baseline": round(float(metrics["f1_macro"]) - 0.32, 4),
            }
        )

    table = pl.DataFrame(rows).sort("f1_macro", descending=True)
    _REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(_REPORTS_PATH)
    context.log.info(f"Tabla comparativa persistida: {_REPORTS_PATH}")

    best = table.row(0, named=True)
    return MaterializeResult(
        metadata={
            "best_model": MetadataValue.text(best["model"]),
            "best_f1_macro": MetadataValue.float(best["f1_macro"]),
            "best_delta_vs_baseline": MetadataValue.float(best["delta_vs_baseline"]),
            "table_path": MetadataValue.path(str(_REPORTS_PATH)),
            "preview": MetadataValue.md(table.to_pandas().to_markdown(index=False)),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )
