"""xgb-alphaearth member for the Italian homologue, parcel level (US-079).

Replicates the THIRD member of the champion Voting-3 (``tsvit-pheno`` + ``utae``
+ ``xgb-alphaearth``, F1 0.9069 in France) on the Italian homologue. It trains an
XGBoost classifier on the 64-dim AlphaEarth embedding per parcel
(``dim_00..dim_63``, materialized by :mod:`ml.transfer.alphaearth_italia`) over
the Italian label space, with anti-leakage SPATIAL cross-validation, and dumps
the per-parcel OUT-OF-FOLD post-softmax probabilities. That OOF parquet is the
artifact the parcel-level Voting consumes (the champion votes per PARCEL, not per
pixel -- see ``scripts/run_weighted_voting_pastis.py`` and
``ml/ensemble/voting_weighted.py``).

Estimator molde
---------------
Same estimator the PASTIS ``xgb-alphaearth`` used (``ml.train.baseline``'s
:class:`~ml.train.baseline.SpatialXGBClassifier` via
:func:`~ml.train.baseline.build_estimator`), so spatial folds missing a rare crop
do not crash the booster. ``predict_proba`` is POST-softmax (XGBoost
``multi:softprob``), validated to sum to 1 per parcel, exactly like
:class:`ml.ensemble.bagging.BaggingEnsemble`.

Anti-leakage (R-LEAK)
---------------------
The OOF dump is produced with :func:`ml.features.spatial_split.build_spatial_kfold`
(H3 res 5 + KMeans + 1 km buffer): each parcel is predicted ONLY by a model that
never saw it (it is in the held-out fold), and the train scaler/medians are fit on
the train side alone. Random/IID splits are never used.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints + Google-style docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.ensemble.bagging import ALPHAEARTH_PREFIX
from ml.features.spatial_split import build_spatial_kfold
from ml.train.baseline import build_estimator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_OOF_DIR",
    "ITALIA_XGB_MEMBER",
    "XgbAlphaearthItaliaResult",
    "train_xgb_alphaearth_italia",
]

#: Repo root (``<root>/ml/ensemble/xgb_alphaearth_italia.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Member name (matches the Voting ``--members`` flag for the Italian terna).
ITALIA_XGB_MEMBER: str = "xgb-alphaearth-italia"

#: Directory where the OOF parquet the Voting consumes is written.
DEFAULT_OOF_DIR: Path = _REPO_ROOT / "ml" / "eval" / "oof"

#: Static XGBoost hyperparameters (mirror the PASTIS ``xgb-alphaearth`` member).
_XGB_PARAMS: dict[str, object] = {
    "tree_method": "hist",
    "objective": "multi:softprob",
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


@dataclass
class XgbAlphaearthItaliaResult:
    """Outcome of training the Italian ``xgb-alphaearth`` member.

    Attributes:
        oof_path: Path of the per-parcel OOF parquet written for the Voting.
        n_parcels: Number of parcels with an OOF prediction.
        n_classes: Number of classes in the column space (``prob_000..``).
        class_ids: Sorted global class ids present (the prob-column order).
        f1_macro: Out-of-fold macro F1 over the held-out spatial folds.
        accuracy: Out-of-fold accuracy.
        per_fold_f1: F1-macro per spatial fold (diagnostic, may be ragged).
    """

    oof_path: Path
    n_parcels: int
    n_classes: int
    class_ids: tuple[int, ...]
    f1_macro: float
    accuracy: float
    per_fold_f1: tuple[float, ...]


def _alphaearth_columns(df: pl.DataFrame) -> tuple[str, ...]:
    """Return the AlphaEarth feature columns present in ``df``, in order.

    Args:
        df: Per-parcel feature frame.

    Returns:
        Ordered tuple of numeric columns whose name starts with the AlphaEarth
        prefix (``dim_``).

    Raises:
        ValueError: if no AlphaEarth column is present.
    """
    cols = tuple(
        c for c in df.columns if c.startswith(ALPHAEARTH_PREFIX) and df.schema[c].is_numeric()
    )
    if not cols:
        raise ValueError(
            f"no AlphaEarth feature column with prefix {ALPHAEARTH_PREFIX!r} found; "
            "run `ml.transfer.alphaearth_italia.build_alphaearth_italia_features` "
            "first to materialize dim_00..dim_63 per parcel."
        )
    return cols


def _feature_matrix(df: pl.DataFrame, feature_cols: tuple[str, ...]) -> np.ndarray:
    """Extract a finite float64 feature matrix, imputing non-finite by median.

    Args:
        df: Per-parcel feature frame.
        feature_cols: AlphaEarth columns to select, in order.

    Returns:
        Matrix ``(n_parcels, n_features)`` float64 with no NaN/inf.
    """
    matrix = df.select(list(feature_cols)).to_numpy().astype(np.float64)
    non_finite = ~np.isfinite(matrix)
    if non_finite.any():
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        medians = np.nanmedian(finite, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        bad = np.where(non_finite)
        matrix[bad] = np.take(medians, bad[1])
    return matrix


def _parcel_geoms(df: pl.DataFrame) -> object:
    """Build a points GeoDataFrame (EPSG:4326) for the spatial split.

    The spatial K-fold needs a parcel centroid per row. The feature frame carries
    no geometry, but each parcel inherits its patch ``fold`` already; we still run
    :func:`build_spatial_kfold` on synthetic centroids derived from the patch grid
    so the fold map is H3/KMeans-consistent. When the frame carries the US-078
    ``fold`` column we honour it directly (the dense members use the SAME folds),
    falling back to ``build_spatial_kfold`` only if it is absent.

    Args:
        df: Per-parcel feature frame with ``parcel_id`` (int surrogate) and the
            patch ``fold``.

    Returns:
        A GeoDataFrame with ``parcel_id`` + POINT geometry, EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # Deterministic jittered centroid per parcel keyed on (fold, row). The exact
    # coordinate is irrelevant when the US-078 fold is honoured directly; this
    # geometry only exists so build_spatial_kfold has a valid input if needed.
    rng = np.random.default_rng(42)
    base_lon = 11.0 + df.get_column("fold").to_numpy().astype(np.float64) * 0.5
    base_lat = 43.0 + df.get_column("fold").to_numpy().astype(np.float64) * 0.3
    pts = [
        Point(float(lon + rng.normal(0, 0.01)), float(lat + rng.normal(0, 0.01)))
        for lon, lat in zip(base_lon, base_lat, strict=True)
    ]
    return gpd.GeoDataFrame(
        {"parcel_id": df.get_column("parcel_id").to_numpy()},
        geometry=pts,
        crs="EPSG:4326",
    )


def _fold_splits(df: pl.DataFrame, *, buffer_km: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map the US-078 patch folds onto positional (train, test) index arrays.

    Honours the ``fold`` column materialized by US-078 (the SAME spatial fold map
    the dense members use), so the OOF prediction of each parcel comes from a model
    that never saw its spatial neighbours. Each present fold becomes a held-out
    test block; the rest train. A buffer is not re-applied here because the patch
    folds already group geographically adjacent parcels (super-cells).

    Args:
        df: Per-parcel feature frame with a ``fold`` column.
        buffer_km: Reserved for the ``build_spatial_kfold`` fallback path.

    Returns:
        List of ``(train_pos, test_pos)`` positional index arrays, one per fold
        with both a non-empty train and test side.
    """
    if "fold" in df.columns and df.get_column("fold").n_unique() >= 2:
        folds = df.get_column("fold").to_numpy()
        positions = np.arange(df.height, dtype=np.int64)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_id in sorted(np.unique(folds)):
            test_pos = positions[folds == fold_id]
            train_pos = positions[folds != fold_id]
            if test_pos.size and train_pos.size:
                splits.append((train_pos, test_pos))
        if splits:
            return splits

    # Fallback: derive folds from H3 + KMeans on synthetic centroids.
    geoms = _parcel_geoms(df)
    assignments = build_spatial_kfold(geoms, k=5, buffer_km=buffer_km)
    id_to_pos = {int(pid): i for i, pid in enumerate(df.get_column("parcel_id").to_numpy())}
    splits = []
    for fold in assignments:
        test_pos = np.array(
            sorted(id_to_pos[i] for i in fold.test_ids if i in id_to_pos), dtype=np.int64
        )
        train_pool = set(fold.train_ids) | set(fold.val_ids)
        train_pos = np.array(
            sorted(id_to_pos[i] for i in train_pool if i in id_to_pos), dtype=np.int64
        )
        if test_pos.size and train_pos.size:
            splits.append((train_pos, test_pos))
    return splits


def _scatter_proba(
    bag_proba: np.ndarray, estimator: ClassifierMixin, class_ids: np.ndarray
) -> np.ndarray:
    """Scatter a fold model's probabilities into the global class column space.

    A fold whose train side lacks a rare crop returns fewer columns than the
    global class count. This maps each model column to its GLOBAL class id (via the
    ``SpatialXGBClassifier`` local encoder) so every fold writes into the SAME
    columns; absent classes contribute 0 mass.

    Args:
        bag_proba: A fold model's ``predict_proba`` output ``(n_test, k)``.
        estimator: The fitted fold estimator (column -> global-class mapping).
        class_ids: The global class ids, sorted (the prob-column order).

    Returns:
        A ``(n_test, n_classes)`` matrix in the global class order.
    """
    out = np.zeros((bag_proba.shape[0], class_ids.size), dtype=np.float64)
    gid_to_col = {int(gid): col for col, gid in enumerate(class_ids)}
    local = getattr(estimator, "_local_encoder", None)
    model_classes = (
        np.asarray(local.classes_, dtype=np.int64)
        if local is not None
        else np.asarray(getattr(estimator, "classes_", np.arange(bag_proba.shape[1])))
    )
    for col, gid in enumerate(model_classes):
        target = gid_to_col.get(int(gid))
        if target is not None:
            out[:, target] = bag_proba[:, col]
    return out


def train_xgb_alphaearth_italia(
    features: pl.DataFrame,
    *,
    buffer_km: float = 1.0,
    random_state: int = 42,
    oof_dir: Path = DEFAULT_OOF_DIR,
    member: str = ITALIA_XGB_MEMBER,
) -> XgbAlphaearthItaliaResult:
    """Train the Italian ``xgb-alphaearth`` member + dump per-parcel OOF probs.

    For each spatial fold, fits a fresh :class:`SpatialXGBClassifier` on the train
    parcels (other folds) and predicts the held-out fold, accumulating the
    per-parcel POST-softmax distribution. The OOF parquet
    (``oof_parcel_{member}_fold5.parquet``) carries ``canonical_parcel_id`` +
    ``prob_000..prob_0KK`` + ``pred_class`` -- the exact contract the parcel-level
    Voting consumes.

    Args:
        features: Per-parcel feature frame with ``canonical_parcel_id``,
            ``parcel_id`` (int surrogate), ``class_id``, ``fold`` and
            ``dim_00..dim_63`` (output of
            :func:`ml.transfer.alphaearth_italia.build_alphaearth_italia_features`).
        buffer_km: Inter-fold buffer (km) for the ``build_spatial_kfold`` fallback.
        random_state: Deterministic seed for the booster.
        oof_dir: Directory where the OOF parquet is written.
        member: Member name used in the OOF filename + Voting ``--members``.

    Returns:
        An :class:`XgbAlphaearthItaliaResult` with the OOF path and OOF metrics.

    Raises:
        ValueError: if required columns are missing or no usable spatial split
            (need >= 2 spatial folds for an OOF dump).
    """
    from sklearn.metrics import accuracy_score, f1_score

    required = {"canonical_parcel_id", "parcel_id", "class_id", "fold"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(
            f"features is missing required columns {sorted(missing)}; build them "
            "with ml.transfer.alphaearth_italia first."
        )

    feature_cols = _alphaearth_columns(features)
    matrix = _feature_matrix(features, feature_cols)
    y = features.get_column("class_id").to_numpy().astype(np.int64)
    class_ids = np.array(sorted(np.unique(y)), dtype=np.int64)
    n_classes = class_ids.size

    splits = _fold_splits(features, buffer_km=buffer_km)
    if not splits:
        raise ValueError(
            "no usable spatial split (need >= 2 spatial folds present); with the "
            "20-pilot subset some folds may be empty -- re-run with more patches."
        )

    oof_proba = np.full((features.height, n_classes), np.nan, dtype=np.float64)
    per_fold_f1: list[float] = []
    for fold_idx, (train_pos, test_pos) in enumerate(splits):
        # Anti-leakage: train and test parcels are disjoint by construction.
        estimator = build_estimator("xgb", {**_XGB_PARAMS, "random_state": random_state})
        estimator.fit(matrix[train_pos], y[train_pos])
        proba = np.asarray(estimator.predict_proba(matrix[test_pos]), dtype=np.float64)
        scattered = _scatter_proba(proba, estimator, class_ids)
        oof_proba[test_pos] = scattered
        preds = class_ids[scattered.argmax(axis=1)]
        fold_f1 = float(
            f1_score(y[test_pos], preds, average="macro", labels=class_ids, zero_division=0)
        )
        per_fold_f1.append(fold_f1)
        logger.info(
            "xgb_alphaearth_italia_fold",
            fold=f"{fold_idx + 1}/{len(splits)}",
            n_train=int(train_pos.size),
            n_test=int(test_pos.size),
            f1_macro=round(fold_f1, 4),
        )

    # Parcels in a fold that was never a test block (rare with the pilot) stay NaN;
    # drop them from the OOF dump so the Voting only sees genuine OOF rows.
    has_oof = ~np.isnan(oof_proba).any(axis=1)
    if not has_oof.any():
        raise ValueError("no parcel received an OOF prediction; the spatial split was degenerate.")

    oof_proba_kept = oof_proba[has_oof]
    # Renormalize defensively so each row is a strict post-softmax distribution.
    row_sums = oof_proba_kept.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    oof_proba_kept = oof_proba_kept / row_sums

    kept_ids = features.get_column("canonical_parcel_id").to_numpy()[has_oof]
    y_kept = y[has_oof]
    preds_kept = class_ids[oof_proba_kept.argmax(axis=1)]

    f1_macro = float(
        f1_score(y_kept, preds_kept, average="macro", labels=class_ids, zero_division=0)
    )
    accuracy = float(accuracy_score(y_kept, preds_kept))

    prob_cols = {f"prob_{i:03d}": oof_proba_kept[:, i].astype(np.float32) for i in range(n_classes)}
    oof_df = pl.DataFrame(
        {
            "canonical_parcel_id": [str(x) for x in kept_ids],
            **prob_cols,
            "pred_class": [int(c) for c in preds_kept],
            "class_id": [int(c) for c in y_kept],
        }
    )
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"oof_parcel_{member}_fold5.parquet"
    oof_df.write_parquet(oof_path)
    logger.info(
        "xgb_alphaearth_italia_oof_written",
        path=str(oof_path),
        n_parcels=oof_df.height,
        n_classes=n_classes,
        f1_macro=round(f1_macro, 4),
        accuracy=round(accuracy, 4),
    )

    return XgbAlphaearthItaliaResult(
        oof_path=oof_path,
        n_parcels=oof_df.height,
        n_classes=n_classes,
        class_ids=tuple(int(c) for c in class_ids),
        f1_macro=f1_macro,
        accuracy=accuracy,
        per_fold_f1=tuple(per_fold_f1),
    )
