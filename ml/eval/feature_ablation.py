"""Feature-block ablation for the baseline (US-022b-C).

Builds the **comparative matrix** of the phenological reframing:

- ``full`` — all available features.
- ``no_geom`` — drops ``geom_area_ha``, ``geom_perimeter_m``,
  ``geom_elongation`` (geographic proxies, candidates for spatial leakage).
- ``no_geom_no_era5_srtm`` — additionally drops the ERA5 blocks
  (24 cols) and SRTM (3 cols), redundant with AlphaEarth (which already
  encodes them internally).
- ``alphaearth_only`` — only the 64 dimensions ``ae_00..ae_63``.
- ``phenology_only`` — only the 8 phenological features + 24 FFT
  (NDVI/NDWI/EVI).

Canonical decisions (plan ``docs/us-planning/us-022b.md`` §6.2):

- **D-ARQ-1**: does NOT rewrite ``fusion.py`` or ``temporal_features.py`` — it
  consumes them. Receives the already-fused DataFrame or the baseline parquet.
- **Same spatial CV 5-fold** for all sets (thanks to the cache
  keyed by ``n_rows + k + buffer + seed`` of ``_build_cv_splits``).
- **Reuses** ``ml.train.baseline.train_one_model`` (does not reinvent training).
- **delta_vs_full** is computed as ``F1-macro(set) - F1-macro(full)``
  for the same ``model_kind``. ``full`` is the mandatory reference; if
  it is not in ``feature_sets`` a ``ValueError`` is raised.

The ``export_ablation_table`` export produces CSV + Markdown ready for the
notebook ``05_reencuadre_fenologico.ipynb`` and for the closing of Avance 4.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

#: Canonical pattern to detect AlphaEarth / DINOv3-derived columns. Accepts
#: ``ae_NN``, ``ae_NNN``, ``emb_NN``, ``emb_NNN``, ``alphaearth_NN[N]``,
#: ``dim_NN[N]`` and the variants with the year embedded in the prefix
#: (``ae18_NN``, ``ae19_NN``) produced by ``load_base_plus_alphaearth_2018_2019``.
#: Generalized in US-023-preview v2: the previous narrow pattern (exact len
#: 5 or 6) discarded ``emb_00`` (FarSLIP-style), ``ae_063`` (3 digits) and the
#: two AlphaEarth years (``ae18_``/``ae19_``) of the winning scenario of 04.
_AE_COL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?:ae|emb|alphaearth|dim)\d{0,4}_\d{2,3}$")

__all__ = [
    "FeatureAblationResult",
    "build_default_feature_sets",
    "export_ablation_table",
    "run_feature_ablation",
]

#: Models supported by the ablation (the temporal ones go through
#: :mod:`ml.train.phenology_models`; here only the fast tabular ones are
#: accepted so the N x M matrix runs on CPU without GPU).
SupportedModel = Literal["rf", "xgb", "lgbm", "tempcnn", "inceptiontime"]

#: Metadata columns that are never features.
_META_COLS: frozenset[str] = frozenset(
    {
        "parcel_id",
        "year",
        "patch_id",
        "instance_id",
        "class_id",
        "class_name",
        "fold",
        "n_pixels",
        "area_m2",
        "geometry",
    }
)


@dataclass(frozen=True)
class FeatureAblationResult:
    """Result of a (feature_set, model) in the ablation matrix.

    Attributes:
        feature_set: Set label (``full``, ``no_geom``, ...).
        model_kind: Applied model (``rf``, ``xgb``, ``tempcnn``,
            ``inceptiontime``).
        f1_macro: Out-of-fold F1-macro of the spatial CV.
        f1_weighted: Weighted F1.
        miou: mIoU (macro Jaccard).
        n_features: Number of effective features (those that existed in the
            DataFrame and were numeric; the requested ones that did not exist are
            ignored with a warning).
        delta_vs_full: ``f1_macro(set) - f1_macro(full)`` for the same
            model. ``nan`` if the set is ``full`` itself or if ``full`` is not
            present.
    """

    feature_set: str
    model_kind: SupportedModel
    f1_macro: float
    f1_weighted: float
    miou: float
    n_features: int
    delta_vs_full: float


# ---------------------------------------------------------------------------
# Default feature sets (when the caller does not provide their own).
# ---------------------------------------------------------------------------


def build_default_feature_sets(
    available_cols: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Builds the 5 canonical sets from the present columns.

    Args:
        available_cols: All the columns present in the fused
            DataFrame (output of :func:`ml.features.fusion.build_fused_features`
            or the US-018 subset).

    Returns:
        Mapping ``{set_name: (cols,)}`` with the 5 canonical sets + optional
        sets depending on available cols:

        - ``full``: all numeric features (excluding metadata).
        - ``no_geom``: ``full`` without ``geom_*``.
        - ``no_geom_no_era5_srtm``: ``no_geom`` without ``era5_*`` nor ``srtm_*``.
        - ``alphaearth_only``: only ``ae_*`` or ``dim_*`` (accepts both
          names).
        - ``phenology_only``: 8 phenological cols + 24 FFT
          (``{idx}_fft_amp_k``, ``{idx}_fft_phase_k`` for
          ``idx in {NDVI, NDWI, EVI}``).
        - ``with_farslip`` / ``farslip_only``: only if there are
          ``farslip_NNN`` cols (US-022-c / US-023-preview P2).
        - ``with_pheno_text`` / ``pheno_text_only``: only if there are
          ``pheno_text_NNN`` cols (US-022b-D / US-023-preview P4).
        - ``with_spectral_signature`` / ``spectral_signature_only``: only
          if there are ``spectral_signature_NNN`` cols (US-023-preview P5).
        - ``geom_only``: only if there are ``geom_*`` cols — quantitative
          test of spatial leakage (US-023-preview P3).
    """
    cols = [c for c in available_cols if c not in _META_COLS]

    full = tuple(cols)
    no_geom = tuple(c for c in cols if not c.startswith("geom_"))
    no_geom_no_era5_srtm = tuple(
        c for c in no_geom if not c.startswith("era5_") and not c.startswith("srtm_")
    )
    # Detect AlphaEarth columns with generalized pattern: accepts `ae_NN`,
    # `ae_NNN`, `emb_NN`, `emb_NNN`, `alphaearth_NN[N]` and `dim_NN[N]`. The
    # previous pattern (exact len 5 or 6) discarded `emb_00` (US-023-preview
    # P4) and blocked columns with 3 digits for encoders > 100 dims.
    ae_cols = tuple(c for c in cols if _AE_COL_PATTERN.match(c))
    if not ae_cols:
        logger.warning(
            "ae_cols_empty",
            n_total_cols=len(cols),
            note=(
                "No se detectaron columnas AlphaEarth con el patron "
                "(ae|emb|alphaearth|dim)_NN[N]; `alphaearth_only` quedara vacio."
            ),
        )
    pheno_cols_known = {
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    }
    fft_cols = tuple(c for c in cols if "_fft_amp_" in c or "_fft_phase_" in c)
    pheno_cols = tuple(c for c in cols if c in pheno_cols_known) + fft_cols
    # Optional blocks (US-017 FarSLIP, US-022b-D phenological semantic branch,
    # US-023-preview P5 spectral signature). The FarSLIP filter uses the
    # canonical pattern `farslip_NNN` (3 digits) and discards `farslip_emb_NNN` to
    # avoid collisions when both prefixes coexist transiently.
    farslip_cols = tuple(
        c for c in cols if c.startswith("farslip_") and not c.startswith("farslip_emb_")
    )
    pheno_text_cols = tuple(c for c in cols if c.startswith("pheno_text_"))
    geom_cols = tuple(c for c in cols if c.startswith("geom_"))
    spectral_signature_cols = tuple(c for c in cols if c.startswith("spectral_signature_"))

    sets: dict[str, tuple[str, ...]] = {
        "full": full,
        "no_geom": no_geom,
        "no_geom_no_era5_srtm": no_geom_no_era5_srtm,
        "alphaearth_only": ae_cols,
        "phenology_only": pheno_cols,
    }
    # Only adds the with_* / *_only sets if the corresponding columns
    # are materialized in the DataFrame (graceful degradation).
    if farslip_cols:
        sets["with_farslip"] = pheno_cols + farslip_cols
        sets["farslip_only"] = farslip_cols
    if pheno_text_cols:
        sets["with_pheno_text"] = pheno_cols + pheno_text_cols
        sets["pheno_text_only"] = pheno_text_cols
    if spectral_signature_cols:
        sets["with_spectral_signature"] = pheno_cols + spectral_signature_cols
        sets["spectral_signature_only"] = spectral_signature_cols
    # `geom_only` is added when there are `geom_*` cols: quantitative test of
    # spatial leakage. The null hypothesis is F1-macro(`geom_only`) < 0.10
    # (pure geometry provides no class signal).
    if geom_cols:
        sets["geom_only"] = geom_cols
    return sets


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def run_feature_ablation(
    features_path: Path | str | None = None,
    *,
    df: pl.DataFrame | None = None,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    models: tuple[SupportedModel, ...] = ("xgb",),
    max_samples: int | None = None,
    seed: int = 42,
    k_folds: int = 5,
    buffer_km: float = 1.0,
) -> list[FeatureAblationResult]:
    """Runs the ablation: trains each model on each feature set.

    For each pair ``(feature_set, model)`` it trains a baseline with the same
    cached spatial CV 5-fold and records F1-macro + F1-weighted + mIoU +
    ``n_features``. The ``delta_vs_full`` is computed at the end, once
    all the runs have finished.

    Args:
        features_path: Path to the fused-features parquet. If ``df`` is
            passed, it is ignored.
        df: Already-loaded Polars DataFrame.
        feature_sets: Mapping ``{name: (cols,)}`` with the sets to
            ablate. It must include the ``"full"`` key. If ``None`` they are
            built with :func:`build_default_feature_sets`.
        models: Models to apply. ``"rf"`` and ``"xgb"`` go to
            :func:`ml.train.baseline.train_one_model`; ``"tempcnn"`` and
            ``"inceptiontime"`` go to
            :func:`ml.train.phenology_models.train_temporal_model` (only
            if the set includes at least one reconstructible temporal index).
        max_samples: Deterministic uniform subsample (CI/dev). ``None`` =
            full dataset.
        seed: Deterministic seed.
        k_folds: Number of spatial CV folds.
        buffer_km: Anti-leakage buffer in km.

    Returns:
        List of :class:`FeatureAblationResult`, one per pair
        ``(set, model)`` with enough samples.

    Raises:
        ValueError: if ``df`` and ``features_path`` are both ``None`` or if
            ``feature_sets`` does not contain the ``"full"`` key.
    """
    if df is None:
        if features_path is None:
            raise ValueError("You must pass `features_path` or `df`.")
        df = pl.read_parquet(Path(features_path))

    if max_samples is not None and max_samples > 0 and df.height > max_samples:
        df = df.sample(n=max_samples, seed=seed, with_replacement=False)
        logger.info("ablation_subsampled", max_samples=max_samples, n=df.height)

    if feature_sets is None:
        feature_sets = build_default_feature_sets(df.columns)
    if "full" not in feature_sets:
        raise ValueError(
            "`feature_sets` must include the 'full' key (reference for delta_vs_full)."
        )

    logger.info(
        "ablation_start",
        n_sets=len(feature_sets),
        models=models,
        n_rows=df.height,
    )

    raw_results: list[FeatureAblationResult] = []
    for set_name, requested_cols in feature_sets.items():
        present_cols = tuple(c for c in requested_cols if c in df.columns)
        missing = [c for c in requested_cols if c not in df.columns]
        if missing:
            logger.warning(
                "ablation_missing_cols",
                feature_set=set_name,
                n_missing=len(missing),
                first_missing=missing[:5],
            )
        if not present_cols:
            logger.warning("ablation_set_empty", feature_set=set_name)
            for model_kind in models:
                raw_results.append(
                    FeatureAblationResult(
                        feature_set=set_name,
                        model_kind=model_kind,
                        f1_macro=float("nan"),
                        f1_weighted=float("nan"),
                        miou=float("nan"),
                        n_features=0,
                        delta_vs_full=float("nan"),
                    )
                )
            continue

        meta_cols = [c for c in ("parcel_id", "year", "patch_id", "class_id") if c in df.columns]
        # We keep only metadata + the requested columns. Extra _META_COLS
        # like `fold`, `instance_id` are preserved if they exist (train_one_model
        # ignores them because they are not numeric or are in its blacklist).
        keep = meta_cols + [c for c in present_cols if c not in meta_cols]
        subset_df = df.select(keep)

        for model_kind in models:
            try:
                f1_macro, f1_weighted, miou, n_feats = _train_single(
                    subset_df,
                    model_kind=model_kind,
                    k_folds=k_folds,
                    buffer_km=buffer_km,
                    seed=seed,
                )
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - safety net
                logger.warning(
                    "ablation_train_failed",
                    feature_set=set_name,
                    model_kind=model_kind,
                    error=str(exc),
                )
                f1_macro = f1_weighted = miou = float("nan")
                n_feats = len(present_cols)
            raw_results.append(
                FeatureAblationResult(
                    feature_set=set_name,
                    model_kind=model_kind,
                    f1_macro=f1_macro,
                    f1_weighted=f1_weighted,
                    miou=miou,
                    n_features=n_feats,
                    delta_vs_full=float("nan"),  # filled in the second pass
                )
            )
            logger.info(
                "ablation_cell_done",
                feature_set=set_name,
                model_kind=model_kind,
                f1_macro=round(f1_macro, 4) if not np.isnan(f1_macro) else None,
                n_features=n_feats,
            )

    # Second pass: fill delta_vs_full per model.
    f1_full_by_model: dict[str, float] = {}
    for r in raw_results:
        if r.feature_set == "full":
            f1_full_by_model[r.model_kind] = r.f1_macro

    results: list[FeatureAblationResult] = []
    for r in raw_results:
        ref = f1_full_by_model.get(r.model_kind)
        if r.feature_set == "full" or ref is None or np.isnan(ref) or np.isnan(r.f1_macro):
            delta = float("nan")
        else:
            delta = r.f1_macro - ref
        results.append(
            FeatureAblationResult(
                feature_set=r.feature_set,
                model_kind=r.model_kind,
                f1_macro=r.f1_macro,
                f1_weighted=r.f1_weighted,
                miou=r.miou,
                n_features=r.n_features,
                delta_vs_full=delta,
            )
        )

    logger.info("ablation_done", n_rows_output=len(results))
    return results


def export_ablation_table(
    results: Sequence[FeatureAblationResult],
    output_path: Path | str,
) -> tuple[Path, Path]:
    """Persists the ablation table to CSV + Markdown.

    Args:
        results: List of :class:`FeatureAblationResult`.
        output_path: Destination path (without extension or ``.csv``).
            ``<stem>.csv`` and ``<stem>.md`` are generated in the same directory.

    Returns:
        Tuple ``(csv_path, md_path)`` with the written paths.
    """
    csv_path = Path(output_path).with_suffix(".csv")
    md_path = Path(output_path).with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = pl.DataFrame(
        [
            {
                "feature_set": r.feature_set,
                "model": r.model_kind,
                "n_features": r.n_features,
                "f1_macro": round(r.f1_macro, 4),
                "f1_weighted": round(r.f1_weighted, 4),
                "miou": round(r.miou, 4),
                "delta_vs_full": (
                    round(r.delta_vs_full, 4) if not np.isnan(r.delta_vs_full) else None
                ),
            }
            for r in results
        ],
        schema={
            "feature_set": pl.Utf8,
            "model": pl.Utf8,
            "n_features": pl.Int64,
            "f1_macro": pl.Float64,
            "f1_weighted": pl.Float64,
            "miou": pl.Float64,
            "delta_vs_full": pl.Float64,
        },
    )
    table.write_csv(csv_path)
    md_body = (
        "# Ablation de features — reencuadre fenologico (US-022b-C)\n\n"
        + table.to_pandas().to_markdown(index=False)
        + "\n"
    )
    md_path.write_text(md_body, encoding="utf-8")
    logger.info("ablation_table_exported", csv=str(csv_path), md=str(md_path))
    return csv_path, md_path


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _train_single(
    df: pl.DataFrame,
    *,
    model_kind: SupportedModel,
    k_folds: int,
    buffer_km: float,
    seed: int,
) -> tuple[float, float, float, int]:
    """Trains a model and returns ``(f1_macro, f1_weighted, miou, n_feats)``.

    Deferred import: breaks the cycle ``baseline -> eval.metrics`` and
    ``eval.__init__ -> feature_ablation``.
    """
    if model_kind in ("rf", "xgb", "lgbm"):
        from ml.train.baseline import train_one_model

        tabular_result = train_one_model(
            df,
            model=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            random_state=seed,
        )
        return (
            float(tabular_result.metrics["f1_macro"]),
            float(tabular_result.metrics["f1_weighted"]),
            float(tabular_result.metrics["miou"]),
            len(tabular_result.feature_cols),
        )
    if model_kind in ("tempcnn", "inceptiontime"):
        from ml.train.phenology_models import train_temporal_model

        temporal_result = train_temporal_model(
            df=df,
            model_kind=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            seed=seed,
            n_epochs=10,
            batch_size=128,
        )
        return (
            float(temporal_result.f1_macro),
            float(temporal_result.f1_weighted),
            float(temporal_result.miou),
            int(temporal_result.n_classes),
        )
    raise ValueError(f"`model_kind` not supported: {model_kind!r}.")
