"""E2 Bagging ensemble: bootstrap-aggregated XGBoost-AlphaEarth at parcel level.

This is the tabular ensemble of US-040 (EPIC 6). It trains ``n_bags`` XGBoost
classifiers, each on an independent bootstrap resample of the **parcel** training
set, over the AlphaEarth Satellite Embedding V1 Annual v1.1 64-dim vector
(columns ``dim_000..dim_063``) materialized per parcel in
``data/features/features_fused_pastis.parquet``. The bag predictions are averaged
(soft-vote of POST-softmax probabilities) into a single per-parcel distribution.

It inherits the shared contract from :class:`ml.ensemble.base.EnsembleModel`
(probability validation, fold-5-only evaluation, MLflow logging) and reuses the
baseline estimator (:class:`ml.train.baseline.SpatialXGBClassifier` via
:func:`ml.train.baseline.build_estimator`) so the bagging member is the SAME
XGBoost-AlphaEarth used in US-019, only resampled.

Anti-leakage invariants (plan Section 9, R-LEAK):

1. **The training set excludes the held-out fold-5.** Every bootstrap is drawn
   ONLY from folds 1-4 of ``df_tabular``; fold-5 parcels are never seen during
   ``fit``. Reporting happens on fold-5 via :meth:`EnsembleModel.evaluate`.
2. **Spatial CV, never random.** The Optuna objective evaluates a candidate
   XGBoost with :func:`ml.train.baseline.evaluate_with_spatial_cv` (H3 res 5 +
   KMeans + 1 km buffer); there is no random/IID split of the spatial data.
3. **Probabilities, not logits.** :meth:`predict_proba` averages POST-softmax
   ``predict_proba`` outputs and validates the mean is a distribution via
   :meth:`EnsembleModel.validate_probs`.

Bagging reduces the variance of a single XGBoost (bootstrap aggregating, Breiman
1996) but does not close the gap with the dense temporal models, so the expected
fold-5 F1-macro is ~0.55-0.60 (plan Section 1.2), below the 0.6253 individual
best -- which is the documented role of the tabular member.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints and Google-style docstrings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog
from sklearn.preprocessing import LabelEncoder

from ml.ensemble.base import EnsembleModel
from ml.train.baseline import (
    build_estimator,
    evaluate_with_spatial_cv,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import optuna
    from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "ALPHAEARTH_PREFIX",
    "BaggingEnsemble",
]

#: Column prefix of the AlphaEarth Satellite Embedding 64-dim vector per parcel
#: (``dim_000 .. dim_063``) in ``features_fused_pastis.parquet``.
ALPHAEARTH_PREFIX: str = "dim_"

#: Number of agronomic classes in the harness 18-class space (prob_000..prob_017).
_NUM_CLASSES: int = 18

#: The single held-out fold; every bootstrap excludes it (anti-leakage).
_HELD_OUT_FOLD: int = EnsembleModel.HELD_OUT_FOLD

#: Static XGBoost hyperparameters shared by every bag (the search-space keys in
#: :meth:`BaggingEnsemble._suggest_params` override the dynamic ones).
_XGB_STATIC_PARAMS: dict[str, object] = {
    "tree_method": "hist",
    "objective": "multi:softprob",
}


class BaggingEnsemble(EnsembleModel):
    """Bootstrap-aggregated XGBoost-AlphaEarth ensemble at parcel granularity.

    ``fit`` draws ``n_bags`` bootstrap resamples (distinct seeds) of the parcel
    training set (folds 1-4, fold-5 excluded), tunes the shared XGBoost
    hyperparameters with Optuna over spatial CV, and fits one XGBoost-AlphaEarth
    per bag on its resample. ``predict_proba`` averages the per-bag
    ``predict_proba`` into a single post-softmax distribution per parcel.

    Attributes:
        n_bags: Number of bootstrap members.
        n_trials: Number of Optuna trials for the shared hyperparameters.
        feature_cols: AlphaEarth feature column names used as the X matrix
            (set after :meth:`fit`).
        best_params: Best XGBoost hyperparameters found by Optuna (persisted).
        study: The Optuna study (best params, trials) after :meth:`fit`.
        bags: The fitted per-bag estimators after :meth:`fit`.
        label_encoder: Shared ``LabelEncoder`` so every bag's probability
            columns align to the same global class order.
    """

    def __init__(
        self,
        *,
        n_bags: int = 10,
        n_trials: int = 30,
        feature_prefix: str = ALPHAEARTH_PREFIX,
        n_spatial_folds: int = 5,
        buffer_km: float = 1.0,
        **kw: Any,
    ) -> None:
        """Initialize the bagging ensemble.

        Args:
            n_bags: Number of bootstrap members (default 10). Must be >= 2 so the
                ensemble actually aggregates more than one resample.
            n_trials: Number of Optuna trials for the shared XGBoost
                hyperparameters (default 30). Must be >= 1.
            feature_prefix: Column prefix selecting the AlphaEarth 64-dim vector
                (default ``"dim_"`` -> ``dim_000..dim_063``).
            n_spatial_folds: Number of spatial CV folds used inside the Optuna
                objective (default 5), via ``build_spatial_kfold``.
            buffer_km: Anti-leakage inter-fold buffer in km for the spatial CV
                (default 1.0).
            **kw: Forwarded to :class:`EnsembleModel` (``oof_dir``,
                ``random_state``).

        Raises:
            ValueError: if ``n_bags < 2`` or ``n_trials < 1``.
        """
        super().__init__(**kw)
        if n_bags < 2:
            raise ValueError(f"n_bags must be >= 2 to aggregate; received {n_bags}.")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; received {n_trials}.")
        self.n_bags = n_bags
        self.n_trials = n_trials
        self.feature_prefix = feature_prefix
        self.n_spatial_folds = n_spatial_folds
        self.buffer_km = buffer_km

        self.feature_cols: tuple[str, ...] = ()
        self.best_params: dict[str, object] = {}
        self.study: optuna.study.Study | None = None
        self.bags: list[ClassifierMixin] = []
        self.label_encoder: LabelEncoder | None = None
        self._bag_seeds: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # Feature / label plumbing.
    # ------------------------------------------------------------------

    def _alphaearth_columns(self, df: pl.DataFrame) -> tuple[str, ...]:
        """Return the AlphaEarth feature columns present in ``df``, in order.

        Args:
            df: Tabular parcel DataFrame.

        Returns:
            Ordered tuple of numeric columns whose name starts with
            :attr:`feature_prefix`.

        Raises:
            ValueError: if no AlphaEarth column is present.
        """
        cols = tuple(
            c for c in df.columns if c.startswith(self.feature_prefix) and df.schema[c].is_numeric()
        )
        if not cols:
            raise ValueError(
                f"no AlphaEarth feature column with prefix {self.feature_prefix!r} "
                f"found in df_tabular (columns sampled: {df.columns[:8]}...). The "
                "Bagging member trains XGBoost on the AlphaEarth 64-dim vector."
            )
        return cols

    @staticmethod
    def _feature_matrix(df: pl.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        """Extract the feature matrix as a finite ``float64`` array.

        Non-finite values (``NaN``/``+-inf`` carried by some spectral ratios) are
        imputed with the per-column median so XGBoost never sees them.

        Args:
            df: Tabular parcel DataFrame.
            feature_cols: Columns to select, in order.

        Returns:
            Matrix ``(n_parcels, n_features)`` of dtype float64 without NaN/inf.
        """
        matrix = df.select(list(feature_cols)).to_numpy().astype(np.float64)
        non_finite = ~np.isfinite(matrix)
        if non_finite.any():
            finite = np.where(np.isfinite(matrix), matrix, np.nan)
            medians = np.nanmedian(finite, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            bad_idx = np.where(non_finite)
            matrix[bad_idx] = np.take(medians, bad_idx[1])
        return matrix

    def _training_pool(self, df_tabular: pl.DataFrame) -> pl.DataFrame:
        """Return the bootstrap source: folds 1-4, with fold-5 removed.

        Anti-leakage: the held-out fold-5 must never feed a bootstrap. When
        ``df_tabular`` carries no ``fold`` column (already pre-filtered by the
        caller) it is returned untouched.

        Args:
            df_tabular: Tabular parcel DataFrame with ``parcel_id``, ``class_id``
                and (optionally) ``fold``.

        Returns:
            The subset of ``df_tabular`` whose ``fold != 5`` (or the input as-is
            if there is no ``fold`` column).

        Raises:
            ValueError: if the pool is empty after dropping fold-5.
        """
        if "fold" not in df_tabular.columns:
            logger.warning(
                "bagging_no_fold_column",
                note="df_tabular lacks `fold`; assuming the caller already "
                "excluded fold-5 (held-out).",
            )
            pool = df_tabular
        else:
            pool = df_tabular.filter(pl.col("fold") != _HELD_OUT_FOLD)
        if pool.height == 0:
            raise ValueError(
                "the training pool is empty after dropping fold-5; df_tabular must "
                "carry folds 1-4 (fold-5 is the held-out test set)."
            )
        # Drop rows without a usable label (defensive).
        if "class_id" in pool.columns:
            pool = pool.filter(pl.col("class_id").is_not_null())
        return pool

    # ------------------------------------------------------------------
    # Bootstrap sampling (anti-leakage: distinct seeds -> diverse bags).
    # ------------------------------------------------------------------

    def bootstrap_indices(self, n_rows: int) -> list[np.ndarray]:
        """Generate ``n_bags`` bootstrap index arrays with DISTINCT seeds.

        Each array is a size-``n_rows`` sample drawn WITH replacement from
        ``range(n_rows)`` using ``np.random.default_rng(random_state + bag)``, so
        every bag sees a different resample (the source of bagging's variance
        reduction and the diversity asserted by the tests).

        Args:
            n_rows: Number of rows in the training pool.

        Returns:
            A list of ``n_bags`` ``int64`` arrays of length ``n_rows``.

        Raises:
            ValueError: if ``n_rows <= 0``.
        """
        if n_rows <= 0:
            raise ValueError(f"n_rows must be > 0 to bootstrap; received {n_rows}.")
        self._bag_seeds = tuple(self.random_state + bag for bag in range(self.n_bags))
        indices: list[np.ndarray] = []
        for seed in self._bag_seeds:
            rng = np.random.default_rng(seed)
            indices.append(rng.integers(0, n_rows, size=n_rows, dtype=np.int64))
        return indices

    # ------------------------------------------------------------------
    # Optuna tuning (spatial CV objective; persists best params).
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_params(trial: optuna.trial.Trial) -> dict[str, object]:
        """Sample the XGBoost-AlphaEarth search space for one Optuna trial.

        The space matches the baseline's light grid intent (depth, learning
        rate, subsampling) widened for continuous Optuna search.

        Args:
            trial: The Optuna trial proposing the hyperparameters.

        Returns:
            A dictionary of XGBoost constructor hyperparameters.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    def tune(self, pool: pl.DataFrame) -> optuna.study.Study:
        """Run the Optuna study that selects the shared XGBoost hyperparameters.

        The objective trains a candidate XGBoost-AlphaEarth and scores it with
        :func:`ml.train.baseline.evaluate_with_spatial_cv` (spatial CV, never
        random) on the training pool, maximizing the out-of-fold F1-macro. The
        best params are stored in :attr:`best_params` and reused by every bag.

        Args:
            pool: The training pool (folds 1-4), with ``parcel_id``,
                ``class_id`` and the AlphaEarth feature columns.

        Returns:
            The completed Optuna study (``best_params`` populated, one
            ``FrozenTrial`` per ``n_trials``).
        """
        import optuna

        # The CV needs only the metadata keys it understands + the AE features.
        cv_df = self._cv_frame(pool)

        def objective(trial: optuna.trial.Trial) -> float:
            params = dict(_XGB_STATIC_PARAMS)
            params.update(self._suggest_params(trial))
            params["random_state"] = self.random_state

            def factory() -> ClassifierMixin:
                return build_estimator("xgb", dict(params))

            cv_metrics, _, _ = evaluate_with_spatial_cv(
                cv_df,
                factory,
                k_folds=self.n_spatial_folds,
                buffer_km=self.buffer_km,
                random_state=self.random_state,
            )
            f1_mean, _ = cv_metrics.get("f1_macro", (float("nan"), float("nan")))
            # Optuna minimizes NaN poorly; map a degenerate fold to 0.0.
            return 0.0 if np.isnan(f1_mean) else float(f1_mean)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = dict(_XGB_STATIC_PARAMS)
        self.best_params.update(study.best_params)
        self.best_params["random_state"] = self.random_state
        self.study = study
        logger.info(
            "bagging_optuna_done",
            n_trials=len(study.trials),
            best_f1_macro=round(float(study.best_value), 4),
            best_params=study.best_params,
        )
        return study

    @staticmethod
    def _cv_frame(pool: pl.DataFrame) -> pl.DataFrame:
        """Build the minimal DataFrame the baseline spatial CV consumes.

        ``evaluate_with_spatial_cv`` needs ``parcel_id``, ``class_id``,
        (optionally) ``patch_id`` for the centroid lookup, and the numeric
        feature columns. ``parcel_id`` is reset to a positional integer so the
        baseline's synthetic-geometry path stays consistent.

        Args:
            pool: The training pool (folds 1-4).

        Returns:
            A DataFrame with ``parcel_id`` (positional int), the AlphaEarth
            features, ``class_id`` and ``patch_id`` if present.
        """
        keep_meta = [c for c in ("class_id", "patch_id") if c in pool.columns]
        feature_cols = [c for c in pool.columns if c.startswith(ALPHAEARTH_PREFIX)]
        frame = pool.select(feature_cols + keep_meta)
        return frame.with_row_index(name="parcel_id").with_columns(
            pl.col("parcel_id").cast(pl.Int64)
        )

    # ------------------------------------------------------------------
    # Fit / predict.
    # ------------------------------------------------------------------

    def fit(self, df_tabular: pl.DataFrame) -> BaggingEnsemble:  # type: ignore[override]
        """Fit ``n_bags`` XGBoost-AlphaEarth bootstraps + Optuna tuning.

        Steps:

        1. Build the training pool (folds 1-4; fold-5 excluded -- anti-leakage).
        2. Tune the shared XGBoost hyperparameters with Optuna over spatial CV.
        3. Encode ``class_id`` to a contiguous label space shared by every bag.
        4. For each of ``n_bags`` bootstrap resamples (distinct seeds), fit a
           fresh XGBoost-AlphaEarth with the tuned params.

        Args:
            df_tabular: Tabular parcel DataFrame with ``parcel_id``,
                ``class_id``, a ``fold`` column (1-5) and the AlphaEarth 64-dim
                feature columns (``dim_000..dim_063``).

        Returns:
            ``self`` (fitted), for chaining.

        Raises:
            ValueError: if ``class_id`` is missing, no AlphaEarth column is
                present, or the training pool is empty after dropping fold-5.
        """
        if "class_id" not in df_tabular.columns:
            raise ValueError("df_tabular must carry a `class_id` column.")

        pool = self._training_pool(df_tabular)
        self.feature_cols = self._alphaearth_columns(pool)

        # 1) Tune shared hyperparameters (spatial CV, persisted best params).
        self.tune(pool)

        # 2) Shared label encoder so every bag's predict_proba columns align.
        y_raw = pool.get_column("class_id").to_numpy().astype(np.int64)
        self.label_encoder = LabelEncoder().fit(y_raw)
        y_encoded = self.label_encoder.transform(y_raw).astype(np.int64)

        matrix = self._feature_matrix(pool, self.feature_cols)
        boot_indices = self.bootstrap_indices(pool.height)

        # 3) Fit one XGBoost-AlphaEarth per bootstrap resample.
        self.bags = []
        for bag, idx in enumerate(boot_indices):
            estimator = build_estimator("xgb", dict(self.best_params))
            estimator.fit(matrix[idx], y_encoded[idx])
            self.bags.append(estimator)
            logger.debug(
                "bagging_bag_fitted",
                bag=bag,
                seed=self._bag_seeds[bag],
                n_unique_rows=int(np.unique(idx).size),
            )

        logger.info(
            "bagging_fitted",
            n_bags=self.n_bags,
            n_train=pool.height,
            n_features=len(self.feature_cols),
            n_classes=int(self.label_encoder.classes_.size),
        )
        return self

    @staticmethod
    def _bag_global_classes(estimator: ClassifierMixin, n_cols: int) -> np.ndarray:
        """Return the global (shared-encoder) class id of each proba column.

        :class:`ml.train.baseline.SpatialXGBClassifier` re-encodes labels to a
        local contiguous ``[0, k)`` space per ``fit`` and does NOT override
        ``predict_proba``, so its columns follow the LOCAL order. The local
        encoder's ``classes_`` are exactly the global-encoded labels we passed in
        (each bag was fitted on ``y_encoded`` from the shared
        :attr:`label_encoder`), so column ``j`` maps to global id
        ``_local_encoder.classes_[j]``. A plain ``XGBClassifier`` exposes the
        same mapping through ``classes_``; the identity fallback covers any other
        estimator whose columns are already global.

        Args:
            estimator: A fitted bag estimator.
            n_cols: Number of columns in the bag's ``predict_proba`` output.

        Returns:
            An ``int64`` array ``(n_cols,)`` of global class ids, one per column.
        """
        local = getattr(estimator, "_local_encoder", None)
        if local is not None:
            return np.asarray(local.classes_, dtype=np.int64)
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            return np.asarray(classes, dtype=np.int64)
        return np.arange(n_cols, dtype=np.int64)

    def _align_to_full_classes(
        self, bag_proba: np.ndarray, estimator: ClassifierMixin
    ) -> np.ndarray:
        """Scatter a bag's probabilities into the full 18-class column space.

        A bag fitted on a bootstrap that lacks a rare class returns fewer columns
        than 18 (its ``classes_`` is a subset). This maps each bag column to its
        GLOBAL class id (via :meth:`_bag_global_classes`) so all bags average over
        the SAME 18 columns; absent classes contribute 0 mass.

        Args:
            bag_proba: A bag's ``predict_proba`` output ``(n_parcels, k)`` where
                ``k`` is the bag's own class count.
            estimator: The bag estimator that produced ``bag_proba`` (used to
                recover the column -> global-class mapping).

        Returns:
            A ``(n_parcels, 18)`` matrix in the canonical class order.
        """
        out = np.zeros((bag_proba.shape[0], _NUM_CLASSES), dtype=np.float64)
        global_ids = self._bag_global_classes(estimator, bag_proba.shape[1])
        for col, gid in enumerate(global_ids):
            if 0 <= int(gid) < _NUM_CLASSES:
                out[:, int(gid)] = bag_proba[:, col]
        return out

    def predict_proba(self, df_parcels: pl.DataFrame) -> np.ndarray:  # type: ignore[override]
        """Return the mean per-parcel probability over the bags (post-softmax).

        Each bag predicts the AlphaEarth feature matrix of ``df_parcels``; the
        per-bag ``predict_proba`` outputs are averaged into a single distribution
        per parcel and validated as post-softmax (sum-to-1) before being
        returned. Averaging PROBABILITIES (never logits) is the anti-leakage
        convention.

        Args:
            df_parcels: Tabular parcel DataFrame carrying the same AlphaEarth
                feature columns used in :meth:`fit` (typically the fold-5
                parcels for evaluation).

        Returns:
            A ``(n_parcels, 18)`` ``float64`` array summing to 1 per row, in the
            canonical class order.

        Raises:
            RuntimeError: if called before :meth:`fit`.
        """
        if not self.bags or self.label_encoder is None:
            raise RuntimeError("BaggingEnsemble.predict_proba called before fit().")

        matrix = self._feature_matrix(df_parcels, self.feature_cols)
        acc = np.zeros((matrix.shape[0], _NUM_CLASSES), dtype=np.float64)
        for estimator in self.bags:
            bag_proba = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
            acc += self._align_to_full_classes(bag_proba, estimator)
        mean_proba = acc / float(len(self.bags))

        # Renormalize defensively (alignment can drop mass for absent classes)
        # so the row is a strict distribution, then assert post-softmax.
        row_sums = mean_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        mean_proba = mean_proba / row_sums
        self.validate_probs(mean_proba, class_axis=-1, name="bagging_mean_proba")
        return np.asarray(mean_proba, dtype=np.float64)
