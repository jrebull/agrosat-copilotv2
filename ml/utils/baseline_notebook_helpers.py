"""DRY helpers for the `notebooks/baseline/*.ipynb` notebooks.

Centralizes the patterns repeated across the 6 baseline notebooks
(04_baseline, 04b_baseline, 04c_baseline, 04_farslip_eval_pastis,
05_reencuadre_fenologico, Avance3.Equipo17) so each notebook is left as a
composition of calls + markdown + display, without inline code.

Covers:

- :func:`load_or_build_fused_features` — loads fused features with auto-build.
  If `data/features/features_fused_pastis.parquet` does not exist (nor its
  legacy variant `_italy`), it builds from
  `data/processed/pastis_parcels_full.geoparquet` with
  :func:`ml.features.fusion.build_fused_features`.
- :func:`load_features_dataset_with_meta` — safe alias of the US-018 subset.
- :func:`load_base_plus_alphaearth_2018_2019` — base + AlphaEarth 2018
  (`ae18_NN`) + AlphaEarth 2019 (`ae19_NN`), the winning scenario of the
  ablation (`base_plus_ae18_ae19`).
- :func:`train_baseline_three_models` — trains RF + XGB + LGBM with spatial CV.
- :func:`build_model_comparison_table` — Polars DataFrame with 5 metrics x N models.
- :func:`materialize_phenology_text_if_missing` — auto-generates pheno_text block.
- :func:`materialize_s2_anchors_if_missing` — auto-generates S2 anchors block.
- :func:`materialize_spectral_signature_if_missing` — auto-generates spectral signature.
- :func:`materialize_pastis_eval_subset_if_missing` — auto-generates PASTIS subset.
- :func:`materialize_remoteclip_if_missing` — auto-generates RemoteCLIP embeddings.
- :func:`run_ablation_and_persist` — runs feature_ablation + persists table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
import structlog

from ml.utils.dataset_paths import resolve_dataset_path
from ml.utils.parcel_id import canonical_parcel_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.train.phenology_models import TemporalModelResult

logger = structlog.get_logger(__name__)

__all__ = [
    "ModelComparisonRow",
    "build_model_comparison_table",
    "load_base_plus_alphaearth_2018_2019",
    "load_features_dataset_with_meta",
    "load_or_build_fused_features",
    "load_temporal_result_from_mlflow",
    "materialize_pastis_eval_subset_if_missing",
    "materialize_phenology_text_if_missing",
    "materialize_remoteclip_if_missing",
    "materialize_s2_anchors_if_missing",
    "materialize_spectral_signature_if_missing",
    "run_ablation_and_persist",
    "train_baseline_three_models",
]


# ---------------------------------------------------------------------------
# Feature loading / construction.
# ---------------------------------------------------------------------------


_DEFAULT_SUBSET_PATH = Path("data/test_fixtures/feature_selection_parcels_subset.parquet")
_DEFAULT_PARCELS_PATH = Path("data/processed/pastis_parcels_full.geoparquet")
_DEFAULT_FUSED_PATH = Path("data/features/features_fused_pastis.parquet")
_DEFAULT_AE18_PATH = Path("data/cache/gee/alphaearth_parcels_parcels_2018_85951.parquet")
_DEFAULT_AE19_PATH = Path("data/cache/gee/alphaearth_parcels_pastis_parcels_2019_85951.parquet")


def load_features_dataset_with_meta(
    path: Path | str = _DEFAULT_SUBSET_PATH,
    *,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
) -> pl.DataFrame:
    """Load the US-018 features subset and attach real metadata.

    The original subset (`feature_selection_parcels_subset.parquet`) carries
    `parcel_id`, `year`, `class_id` and spectral/phenology features, but does NOT
    carry `patch_id` nor `class_name`. This function does a LEFT JOIN with
    `pastis_parcels_full.geoparquet` to append those columns needed for the
    spatial CV and the reports.

    Args:
        path: Path to the features parquet (default US-018 subset).
        parcels_geoparquet: Path to the full PASTIS-R parcels geoparquet.

    Returns:
        Polars DataFrame with `parcel_id` Utf8, `year`, `class_id`,
        `class_name`, `patch_id`, `fold`, plus all the feature columns.

    Raises:
        FileNotFoundError: if either of the two files does not exist.
    """
    features_path = Path(path)
    parcels_path = Path(parcels_geoparquet)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found at {features_path}. "
            "Run the EPIC 3 pipelines (US-013..US-018) first."
        )
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"Parcels geoparquet not found at {parcels_path}. Run `make build-parcels-geoparquet`."
        )

    import geopandas as gpd

    features = pl.read_parquet(features_path)
    features = canonical_parcel_id(features)

    parcels_gdf = gpd.read_parquet(parcels_path)
    _candidate_meta = (
        "parcel_id",
        "patch_id",
        "instance_id",
        "class_name",
        "fold",
        "area_m2",
        "n_pixels",
    )
    meta_cols = [c for c in _candidate_meta if c in parcels_gdf.columns]
    parcels_meta = pl.from_pandas(parcels_gdf[meta_cols])
    parcels_meta = canonical_parcel_id(parcels_meta)

    # If the features parquet already carries any of the meta columns (by
    # construction of the US-018 subset), we drop them from `parcels_meta` before
    # the join. Otherwise Polars suffixes with `_right` and those numeric
    # columns (patch_id, fold, n_pixels) end up in matrix X as features
    # — spatial leakage that the SHAP of 04_baseline exposed.
    overlap = [c for c in parcels_meta.columns if c != "parcel_id" and c in features.columns]
    if overlap:
        parcels_meta = parcels_meta.drop(overlap)
        logger.info("features_meta_overlap_dropped", overlap=overlap)

    enriched = features.join(parcels_meta, on="parcel_id", how="left")
    logger.info(
        "features_loaded_with_meta",
        features_shape=features.shape,
        enriched_shape=enriched.shape,
    )
    return enriched


def load_base_plus_alphaearth_2018_2019(
    *,
    features_path: Path | str = _DEFAULT_SUBSET_PATH,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
    alphaearth_2018_path: Path | str = _DEFAULT_AE18_PATH,
    alphaearth_2019_path: Path | str = _DEFAULT_AE19_PATH,
) -> pl.DataFrame:
    """Load the winning scenario: base + AlphaEarth 2018 + AlphaEarth 2019.

    Starts from the US-018 subset with real metadata (185 base features) and
    appends the two annual AlphaEarth embeddings of 64 dimensions each: 2018
    (columns ``ae18_00..ae18_63``) and 2019 (columns ``ae19_00..ae19_63``),
    joining by ``parcel_id`` (1:1 join, same universe of 85951 parcels). The
    result is the ``base_plus_ae18_ae19`` scenario that maximized the F1-macro
    in the scenario ablation.

    The AlphaEarth parquets carry the dimensions as ``dim_00..dim_63``; they are
    renamed to ``ae18_NN`` / ``ae19_NN`` so both years coexist in the same
    features matrix without name collision.

    Args:
        features_path: Path to the base features parquet (US-018 subset).
        parcels_geoparquet: Full PASTIS-R parcels geoparquet (metadata).
        alphaearth_2018_path: AlphaEarth 2018 parquet with ``dim_NN``.
        alphaearth_2019_path: AlphaEarth 2019 parquet with ``dim_NN``.

    Returns:
        Polars DataFrame with the base features + 64 ``ae18_NN`` columns + 64
        ``ae19_NN`` columns, plus the metadata (``parcel_id``, ``class_id``,
        ``patch_id``, ``class_name``, ``fold``).

    Raises:
        FileNotFoundError: if any of the AlphaEarth parquets is missing.
    """
    base = load_features_dataset_with_meta(
        path=features_path, parcels_geoparquet=parcels_geoparquet
    )
    base = canonical_parcel_id(base)

    def _load_alphaearth(path: Path | str, prefix: str) -> pl.DataFrame:
        ae_path = Path(path)
        if not ae_path.exists():
            raise FileNotFoundError(
                f"AlphaEarth parquet not found at {ae_path}. "
                "Run the GEE pipeline (US-012) or `dvc pull` the cache."
            )
        ae = canonical_parcel_id(pl.read_parquet(ae_path))
        dim_cols = [c for c in ae.columns if c.startswith("dim_")]
        rename = {c: f"{prefix}_{c.removeprefix('dim_')}" for c in dim_cols}
        return ae.select(["parcel_id", *dim_cols]).rename(rename)

    ae18 = _load_alphaearth(alphaearth_2018_path, "ae18")
    ae19 = _load_alphaearth(alphaearth_2019_path, "ae19")

    enriched = base.join(ae18, on="parcel_id", how="left").join(ae19, on="parcel_id", how="left")
    n_ae18 = sum(1 for c in enriched.columns if c.startswith("ae18_"))
    n_ae19 = sum(1 for c in enriched.columns if c.startswith("ae19_"))
    n_null_ae18 = int(enriched.select(pl.col("ae18_00").is_null().sum()).item())
    n_null_ae19 = int(enriched.select(pl.col("ae19_00").is_null().sum()).item())
    logger.info(
        "base_plus_ae18_ae19_loaded",
        base_shape=base.shape,
        enriched_shape=enriched.shape,
        n_ae18=n_ae18,
        n_ae19=n_ae19,
        n_null_ae18=n_null_ae18,
        n_null_ae19=n_null_ae19,
    )
    return enriched


def load_or_build_fused_features(
    output_path: Path | str = _DEFAULT_FUSED_PATH,
    *,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
    year: int = 2023,
    overwrite: bool = False,
    include_farslip: bool = True,
    include_phenology_text: bool = False,
    include_spectral_signature: bool = False,
) -> pl.DataFrame:
    """Load the fused features parquet; build it if it does not exist.

    If `output_path` exists and `overwrite=False`, it reads and returns it.
    Otherwise it invokes :func:`ml.features.fusion.build_fused_features` over the
    full parcels and persists the result.

    Args:
        output_path: Path to the fused parquet (canonical default
            `data/features/features_fused_pastis.parquet`; when using the default
            it is resolved via :func:`resolve_dataset_path`, which falls back to
            the legacy `_italy` if it is already materialized on disk). The
            content is French PASTIS-R, not Italian.
        parcels_geoparquet: Full PASTIS-R parcels geoparquet.
        year: Reference year for the GEE samplings.
        overwrite: If True regenerates the parquet even if it exists.
        include_farslip: If True includes the FarSLIP block.
        include_phenology_text: If True includes the pheno_text block.
        include_spectral_signature: If True includes the spectral signature.

    Returns:
        Polars DataFrame with all the feature columns.

    Raises:
        FileNotFoundError: if the parcels geoparquet does not exist.
    """
    # Read: if the canonical default was used, resolve to the existing
    # variant (`_pastis` or legacy `_italy`). If the caller passed an explicit
    # path, it is respected as-is.
    if output_path is _DEFAULT_FUSED_PATH:
        output = resolve_dataset_path(_DEFAULT_FUSED_PATH)
    else:
        output = Path(output_path)
    if output.exists() and not overwrite:
        logger.info("fused_features_cache_hit", path=str(output))
        return pl.read_parquet(output)

    parcels_path = Path(parcels_geoparquet)
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels geoparquet not found at {parcels_path}.")

    import geopandas as gpd

    from ml.features.fusion import build_fused_features

    parcels = gpd.read_parquet(parcels_path)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if "year" not in parcels.columns:
        parcels["year"] = year

    logger.info(
        "fused_features_building",
        n_parcels=len(parcels),
        year=year,
        include_farslip=include_farslip,
        include_phenology_text=include_phenology_text,
        include_spectral_signature=include_spectral_signature,
    )
    fused = build_fused_features(
        parcels=parcels,
        year=year,
        include_farslip=include_farslip,
        include_phenology_text=include_phenology_text,
        include_spectral_signature=include_spectral_signature,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fused.write_parquet(output)
    logger.info("fused_features_persisted", path=str(output), shape=fused.shape)
    return fused


# ---------------------------------------------------------------------------
# Training of the 3 baseline models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparisonRow:
    """One row of the model comparison table."""

    model: str
    f1_macro: float
    f1_weighted: float
    miou: float
    accuracy: float
    cohen_kappa: float
    train_time_s: float
    n_features: int
    n_samples: int


def train_baseline_three_models(
    df: pl.DataFrame,
    *,
    models: tuple[Literal["rf", "xgb", "lgbm"], ...] = ("rf", "xgb", "lgbm"),
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> list[ModelComparisonRow]:
    """Train the 3 tabular baseline models with spatial CV and return metrics.

    Args:
        df: DataFrame with features + `parcel_id`, `class_id`, `patch_id`.
        models: Tuple of models to train. Supports `"rf"`, `"xgb"`, `"lgbm"`.
        k_folds: Folds of the spatial CV.
        buffer_km: Anti-leakage buffer in km.
        random_state: Seed.

    Returns:
        List of :class:`ModelComparisonRow` with metrics + train_time.
    """
    import time

    from ml.train.baseline import train_one_model

    rows: list[ModelComparisonRow] = []
    for model_kind in models:
        t0 = time.perf_counter()
        result = train_one_model(
            df,
            model=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            random_state=random_state,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            ModelComparisonRow(
                model=model_kind,
                f1_macro=float(result.metrics["f1_macro"]),
                f1_weighted=float(result.metrics["f1_weighted"]),
                miou=float(result.metrics["miou"]),
                accuracy=float(result.metrics["accuracy"]),
                cohen_kappa=float(result.metrics["cohen_kappa"]),
                train_time_s=elapsed,
                n_features=len(result.feature_cols),
                n_samples=df.height,
            )
        )
        logger.info(
            "baseline_model_done",
            model=model_kind,
            f1_macro=round(rows[-1].f1_macro, 4),
            train_time_s=round(elapsed, 1),
        )
    return rows


def build_model_comparison_table(
    rows: Sequence[ModelComparisonRow],
    *,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """Convert comparison rows into a Polars DataFrame and optionally persist it.

    Args:
        rows: List of :class:`ModelComparisonRow`.
        output_path: If not None, persists as parquet at that path.

    Returns:
        Polars DataFrame sorted by `f1_macro` descending.
    """
    table = pl.DataFrame(
        [
            {
                "model": r.model,
                "f1_macro": round(r.f1_macro, 4),
                "f1_weighted": round(r.f1_weighted, 4),
                "miou": round(r.miou, 4),
                "accuracy": round(r.accuracy, 4),
                "cohen_kappa": round(r.cohen_kappa, 4),
                "train_time_s": round(r.train_time_s, 1),
                "n_features": r.n_features,
                "n_samples": r.n_samples,
            }
            for r in rows
        ]
    ).sort("f1_macro", descending=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out)
        logger.info("model_comparison_persisted", path=str(out))
    return table


def load_temporal_result_from_mlflow(
    model_kind: Literal["tempcnn", "inceptiontime"],
    *,
    experiment_name: str = "baseline-05-reencuadre",
    tracking_uri: str = "http://localhost:5010",
) -> TemporalModelResult:
    """Reconstruct a TemporalModelResult from an already-finished MLflow run.

    Avoids re-training TempCNN/InceptionTime when there is already a run with
    recorded metrics. Reads the most recent run with `params.model_kind` equal to
    `model_kind` and `status=FINISHED`, and reconstructs the output dataclass
    using the `oof_*` metrics and `params.n_classes`.

    Args:
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI of the tracking server.

    Returns:
        :class:`ml.train.phenology_models.TemporalModelResult` with the
        reconstructed out-of-fold metrics, empty ``y_true_oof`` and
        ``y_pred_oof`` and ``checkpoint_path`` pointing to the artifact if
        available.

    Raises:
        ValueError: if there is no FINISHED run of the requested kind.
    """
    import mlflow

    from ml.train.phenology_models import TemporalModelResult

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"experiment `{experiment_name}` does not exist at {tracking_uri}.")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.model_kind = '{model_kind}' and attributes.status = 'FINISHED'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise ValueError(
            f"no FINISHED runs with model_kind=`{model_kind}` in `{experiment_name}`. "
            "Re-train with train_temporal_model or adjust the query."
        )
    run = runs[0]
    metrics = run.data.metrics
    params = run.data.params

    n_parcels = int(params.get("n_parcels", 0))
    n_classes = int(params.get("n_classes", 18))
    train_time_s = float(metrics.get("train_time_s", 0.0))

    logger.info(
        "temporal_result_loaded_from_mlflow",
        model_kind=model_kind,
        run_id=run.info.run_id[:12],
        oof_f1_macro=round(float(metrics.get("oof_f1_macro", 0.0)), 4),
    )

    return TemporalModelResult(
        model_kind=model_kind,
        f1_macro=float(metrics.get("oof_f1_macro", 0.0)),
        f1_weighted=float(metrics.get("oof_f1_weighted", 0.0)),
        miou=float(metrics.get("oof_miou", 0.0)),
        cohen_kappa=float(metrics.get("oof_cohen_kappa", 0.0)),
        train_time_s=train_time_s,
        n_parcels=n_parcels,
        n_classes=n_classes,
        mlflow_run_id=run.info.run_id,
    )


# ---------------------------------------------------------------------------
# Auto-materialization of optional blocks.
# ---------------------------------------------------------------------------


def materialize_phenology_text_if_missing(
    parcels_features_path: Path | str,
    *,
    output_path: Path | str = Path("data/features/phenology_text_pastis.parquet"),
    enforce_api_key: bool = True,
    max_parcels: int | None = None,
) -> Path:
    """Materialize the `pheno_text_*` block if the parquet does not exist.

    Wrapper over :func:`ml.utils.phenology_text.materialize_phenology_text`.
    Idempotent: if `output_path` exists, it does not call Gemini.

    Args:
        parcels_features_path: Path to the features parquet with `parcel_id`,
            `class_id` and temporal NDVI columns.
        output_path: Destination path of the block (parquet).
        enforce_api_key: If True (default) raise RuntimeError without Gemini.
        max_parcels: Limits the number of parcels (None = all).

    Returns:
        Path of the generated or existing parquet.
    """
    from ml.utils.phenology_text import materialize_phenology_text

    return materialize_phenology_text(
        parcels_features_path=parcels_features_path,
        output_path=Path(output_path),
        enforce_api_key=enforce_api_key,
        max_parcels=max_parcels,
        overwrite=False,
    )


def materialize_s2_anchors_if_missing(
    parcels_geoparquet: Path | str,
    *,
    output_path: Path | str = Path("data/features/s2_anchors_pastis.parquet"),
    year: int = 2023,
    phenology_anchors_path: Path | str | None = None,
) -> Path:
    """Materialize the `{anchor}_b04..b08` block if the parquet does not exist.

    Wrapper over :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`.

    Args:
        parcels_geoparquet: Full PASTIS-R parcels geoparquet.
        output_path: Destination path of the S2 anchors block.
        year: Year for the GEE sampling.
        phenology_anchors_path: Optional parquet with per-parcel calendar anchors
            (schema: ``parcel_id, sog_doy, peak_doy, senescence_doy``). If
            provided, the sampler uses per-parcel specific DOY and avoids the
            ``phenology_anchors_fallback_static`` warning. Generate with
            :func:`ml.ingest.pastis_phenology_anchors.build_pastis_phenology_anchors`
            for PASTIS-R.

    Returns:
        Path of the generated or existing parquet.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("s2_anchors_cache_hit", path=str(output))
        return output

    import geopandas as gpd

    from ml.ingest.s2_anchor_sampler import sample_s2_anchors_for_parcels

    parcels = gpd.read_parquet(Path(parcels_geoparquet))
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if "year" not in parcels.columns:
        parcels["year"] = year

    anchors_path: Path | None = (
        Path(phenology_anchors_path) if phenology_anchors_path is not None else None
    )

    return sample_s2_anchors_for_parcels(
        parcels=parcels,
        year=year,
        output_path=output,
        phenology_anchors_path=anchors_path,
    )


def materialize_spectral_signature_if_missing(
    *,
    s2_anchors_path: Path | str = Path("data/features/s2_anchors_pastis.parquet"),
    output_path: Path | str = Path("data/features/spectral_signature_pastis.parquet"),
    descriptor: Literal["rep", "sam", "redge_moments"] = "rep",
) -> Path:
    """Materialize the spectral signature if it does not exist, from already-sampled S2 anchors.

    Args:
        s2_anchors_path: Path to the S2 anchors parquet (must exist; if not,
            invoke `materialize_s2_anchors_if_missing` first).
        output_path: Destination path of the `spectral_signature_*` block.
        descriptor: Descriptor type (default `"rep"`, Frampton 2013).

    Returns:
        Path of the generated or existing parquet.

    Raises:
        FileNotFoundError: if the S2 anchors are not on disk.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("spectral_signature_cache_hit", path=str(output))
        return output

    # Read the S2 anchors block: resolve to the existing variant
    # (`_pastis` canonical or legacy `_italy`) to avoid re-sampling if the
    # artifact is already on disk under the inherited name.
    anchors_path = resolve_dataset_path(s2_anchors_path)
    if not anchors_path.exists():
        raise FileNotFoundError(
            f"S2 anchors not found at {anchors_path}. Run materialize_s2_anchors_if_missing first."
        )

    from ml.features.spectral_signature import SpectralSignatureFeatures

    anchors = pl.read_parquet(anchors_path)
    anchors = canonical_parcel_id(anchors)
    transformer = SpectralSignatureFeatures(descriptor=descriptor)
    signature = transformer.fit_transform(anchors)

    output.parent.mkdir(parents=True, exist_ok=True)
    signature.write_parquet(output)
    logger.info(
        "spectral_signature_persisted",
        path=str(output),
        shape=signature.shape,
        descriptor=descriptor,
    )
    return output


def materialize_pastis_eval_subset_if_missing(
    *,
    output_path: Path | str = Path("data/test_fixtures/pastis_eval_subset.parquet"),
    n_samples: int = 1024,
) -> Path:
    """Materialize the real PASTIS-R subset if it does not exist.

    Wrapper over :func:`ml.ingest.pastis_eval_subset.build_pastis_eval_subset`.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("pastis_eval_subset_cache_hit", path=str(output))
        return output

    from ml.ingest.pastis_eval_subset import build_pastis_eval_subset

    return build_pastis_eval_subset(
        output_path=output,
        n_samples=n_samples,
        overwrite=False,
        save_imagery=True,
    )


def materialize_remoteclip_if_missing(
    *,
    pastis_eval_subset_path: Path | str = Path("data/test_fixtures/pastis_eval_subset.parquet"),
    imagery_path: Path | str = Path("data/test_fixtures/pastis_eval_subset.imagery.parquet"),
    output_path: Path | str = Path("data/farslip/remoteclip_embeddings_pastis.parquet"),
) -> Path:
    """Materialize RemoteCLIP embeddings over the PASTIS subset if they do not exist."""
    output = Path(output_path)
    if output.exists():
        logger.info("remoteclip_cache_hit", path=str(output))
        return output

    from ml.ingest.remoteclip_extractor import extract_remoteclip_embeddings

    return extract_remoteclip_embeddings(
        pastis_eval_subset_path=Path(pastis_eval_subset_path),
        imagery_path=Path(imagery_path),
        output_path=output,
    )


# ---------------------------------------------------------------------------
# Ablation runner that persists table + figures.
# ---------------------------------------------------------------------------


def run_ablation_and_persist(
    df: pl.DataFrame,
    *,
    output_dir: Path | str = Path("reports/baseline/feature_ablation"),
    models: tuple[Literal["rf", "xgb", "lgbm"], ...] = ("xgb",),
    k_folds: int = 5,
    buffer_km: float = 1.0,
    max_samples: int | None = None,
) -> tuple[pl.DataFrame, Path]:
    """Run feature_ablation + persist the parquet/csv/md table.

    Args:
        df: Fused DataFrame (must include the columns to be ablated).
        output_dir: Destination folder.
        models: Models to ablate.
        k_folds: Folds of the CV.
        buffer_km: Anti-leakage buffer.
        max_samples: Uniform subsample for CI/dev. None = all.

    Returns:
        Tuple `(polars_table, parquet_path)`.
    """
    from ml.eval.feature_ablation import (
        build_default_feature_sets,
        export_ablation_table,
        run_feature_ablation,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_default_feature_sets(df.columns)
    results = run_feature_ablation(
        df=df,
        feature_sets=feature_sets,
        models=models,
        max_samples=max_samples,
        k_folds=k_folds,
        buffer_km=buffer_km,
    )

    stem = out_dir / "ablation_table"
    export_ablation_table(results, stem)
    parquet_path = stem.with_suffix(".parquet")
    table = pl.DataFrame(
        [
            {
                "feature_set": r.feature_set,
                "model": r.model_kind,
                "n_features": r.n_features,
                "f1_macro": r.f1_macro,
                "f1_weighted": r.f1_weighted,
                "miou": r.miou,
                "delta_vs_full": r.delta_vs_full,
            }
            for r in results
        ]
    )
    table.write_parquet(parquet_path)
    logger.info(
        "ablation_persisted",
        parquet=str(parquet_path),
        n_rows=table.height,
        models=models,
    )
    return table, parquet_path
