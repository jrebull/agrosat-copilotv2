"""E3 Stacking heterogeneo a nivel parcela (US-040, EPIC 6, subtarea C).

The stacking ensemble combines three HETEROGENEOUS base learners at the parcel
granularity through a trained meta-learner:

- ``tsvit-pheno`` and ``utae``: dense temporal segmenters whose per-pixel softmax
  was reduced to parcel probabilities by US-031 (via
  :func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs`, ``method="mean"`` --
  the same reconciliation reused here through the base class).
- ``xgb-alphaearth``: a tabular XGBoost over AlphaEarth embeddings, already at the
  parcel level.

Each base learner contributes its post-softmax ``(n_parcels, 18)`` OOF matrix; the
meta-learner (multinomial logistic regression by default, or XGBoost) is trained
on the horizontal concatenation of those matrices (``3 x 18 = 54`` meta-features)
to learn how to weight the heterogeneous members per class.

Anti-leakage (R-LEAK -- the single most important rubric criterion). This module
is the most leakage-sensitive of the four ensembles because the meta-learner is a
*second* model fitted on top of the base learners. Three invariants are enforced
programmatically, never by convention:

1. **The meta-learner sees OOF predictions ONLY.** Its training features are the
   ``prob_*`` columns of the US-031 OOF parquets -- predictions a base learner
   produced for parcels it was NOT trained on (held-out fold-5). No raw features,
   no logits, no in-fold predictions enter the meta.
2. **Spatial cross-validation, never random.** Because US-031 only dumped fold-5
   (a single held-out fold), a strict 5-fold OOF CV is impossible; the
   anti-leakage substitute is a GEOGRAPHIC partition of the fold-5 parcels via
   :func:`ml.features.spatial_split.build_spatial_kfold` (H3 res 5 + KMeans +
   1 km buffer). For each spatial sub-fold ``k`` the meta is trained on the OOF
   rows of the OTHER sub-folds and evaluated on ``k``;
   :meth:`EnsembleModel.assert_oof_only` hard-fails any train/eval parcel overlap.
   This is documented as R-OOF-DEPTH.
3. **Probabilities, not logits; report fold-5 only.** Every base matrix is
   validated post-softmax via :meth:`EnsembleModel.validate_probs`; evaluation
   goes through :meth:`EnsembleModel.evaluate` which rejects any fold but 5.

The ground truth is NOT stored inside the OOF parquet (the US-031 dump discards
the target), so the per-parcel PASTIS semantic18 labels are supplied separately
(``gt_labels``: a ``canonical_parcel_id`` + ``label`` frame) to train and evaluate
the meta-learner.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints and Google-style docstrings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import structlog

from ml.ensemble.base import EnsembleModel
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_BASE_MEMBERS",
    "MetaKind",
    "StackingEnsemble",
]

#: Default heterogeneous base learners (2 dense reduced-to-parcel + 1 tabular).
DEFAULT_BASE_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: Meta-learner family: multinomial logistic regression or XGBoost.
MetaKind = Literal["xgb", "logreg"]

#: Canonical key column shared by every parcel OOF frame and the GT frame.
_KEY: str = "canonical_parcel_id"

#: GT label column carried by the per-parcel PASTIS semantic18 frame.
_LABEL: str = "label"

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = len(PROB_COLUMNS)


class StackingEnsemble(EnsembleModel):
    """E3: heterogeneous parcel-level stacking with an OOF-only meta-learner.

    The meta-learner is trained EXCLUSIVELY on the US-031 out-of-fold parcel
    probabilities of the base learners, with spatial cross-validation over
    geographic sub-folds of fold-5 (R-OOF-DEPTH). It never sees a base learner's
    in-fold prediction, never sees logits, and is only ever reported on fold-5.

    Attributes:
        base_members: Ordered base-learner names whose OOF parcel probabilities
            feed the meta-learner.
        meta: ``"logreg"`` (default, multinomial logistic regression) or
            ``"xgb"`` (gradient-boosted trees) meta-learner family.
        n_spatial_folds: Number of geographic sub-folds of fold-5 used for the
            OOF cross-validation of the meta-learner.
        buffer_km: Inter-fold exclusion buffer in km for the spatial split.
        meta_model_: The meta-learner refit on ALL OOF rows after the spatial CV
            (populated by :meth:`fit`); ``None`` before fitting.
        meta_classes_: The class labels the meta-learner can emit, in the column
            order of :meth:`predict_proba` (populated by :meth:`fit`).
        oof_cv_metrics_: The aggregated spatial-CV metrics of the meta-learner
            (mean over sub-folds), the leakage-free estimate of its quality.
    """

    def __init__(
        self,
        base_members: Sequence[str] = DEFAULT_BASE_MEMBERS,
        *,
        meta: MetaKind = "logreg",
        n_spatial_folds: int = 5,
        buffer_km: float = 1.0,
        **kw: object,
    ) -> None:
        """Initialize the stacking ensemble.

        Args:
            base_members: Base-learner names (default
                :data:`DEFAULT_BASE_MEMBERS`). Each must have a parcel-space OOF
                parquet ``oof_parcel_{member}_fold5.parquet`` under ``oof_dir``.
            meta: Meta-learner family, ``"logreg"`` (default) or ``"xgb"``.
            n_spatial_folds: Number of geographic sub-folds of fold-5 for the
                meta-learner's OOF cross-validation (default 5).
            buffer_km: Inter-fold exclusion buffer in km (default 1.0).
            **kw: Forwarded to :class:`EnsembleModel` (``oof_dir``,
                ``random_state``).

        Raises:
            ValueError: if ``base_members`` is empty, ``meta`` is unknown or
                ``n_spatial_folds < 2``.
        """
        super().__init__(**kw)  # type: ignore[arg-type]
        if not base_members:
            raise ValueError("base_members must list at least one base learner.")
        if meta not in ("xgb", "logreg"):
            raise ValueError(f"invalid meta: {meta!r}; use 'xgb' or 'logreg'.")
        if n_spatial_folds < 2:
            raise ValueError(
                f"n_spatial_folds must be >= 2 for a spatial CV; got {n_spatial_folds}."
            )
        self.base_members: tuple[str, ...] = tuple(base_members)
        self.meta: MetaKind = meta
        self.n_spatial_folds = int(n_spatial_folds)
        self.buffer_km = float(buffer_km)

        self.meta_model_: ClassifierMixin | None = None
        self.meta_classes_: np.ndarray | None = None
        self.oof_cv_metrics_: dict[str, float] = {}
        #: Canonical parcel ids aligned with the meta feature matrix (fit order).
        self._fit_parcel_ids: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Meta-feature assembly (OOF-only, post-softmax).
    # ------------------------------------------------------------------

    def build_meta_features(
        self,
        *,
        gt_labels: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray | None]:
        """Assemble the OOF-only meta feature matrix from the base learners.

        Loads each base learner's parcel-space OOF parquet, validates it is
        post-softmax (anti-leakage R-LEAK), inner-joins all members on
        ``canonical_parcel_id`` (so every meta row is supported by EVERY base
        learner) and lays the ``prob_000..prob_017`` columns side by side. The
        result is the ``(n_parcels, n_members * 18)`` meta-feature matrix; the
        only data the meta-learner is ever allowed to see.

        When ``gt_labels`` is given, it is inner-joined too and the aligned
        label vector is returned (the GT lives OUTSIDE the OOF parquet: the
        US-031 dump discards the target).

        Args:
            gt_labels: Optional per-parcel ground-truth frame with
                ``canonical_parcel_id`` + ``label`` (PASTIS semantic18). When
                ``None`` only the features and ids are returned (``y`` is
                ``None``).

        Returns:
            Tuple ``(keys_df, x_meta, y)`` where ``keys_df`` is a one-column
            Polars frame of the aligned ``canonical_parcel_id`` (fit/predict
            order), ``x_meta`` is the ``float64`` meta-feature matrix and ``y``
            is the aligned ``int64`` label vector (or ``None`` when no GT was
            provided).

        Raises:
            ValueError: if a base member's OOF frame is not post-softmax or the
                join leaves no common parcels; FileNotFoundError if an OOF
                parquet is missing (run ``dvc pull``).
        """
        frames = self.load_oof_members(self.base_members, space="parcel")

        joined: pl.DataFrame | None = None
        member_prob_cols: dict[str, list[str]] = {}
        for member in self.base_members:
            frame = canonical_parcel_id(frames[member], col=_KEY)
            # Validate post-softmax BEFORE renaming (anti-leakage: probs, not logits).
            EnsembleModel.parcel_probs_matrix(frame)
            renamed = {col: f"{member}__{col}" for col in PROB_COLUMNS}
            member_prob_cols[member] = list(renamed.values())
            sub = frame.select([_KEY, *PROB_COLUMNS]).rename(renamed)
            joined = sub if joined is None else joined.join(sub, on=_KEY, how="inner")

        assert joined is not None  # base_members is non-empty (checked in __init__)
        joined = joined.sort(_KEY)

        y: np.ndarray | None = None
        if gt_labels is not None:
            gt = self._prepare_gt(gt_labels)
            joined = joined.join(gt, on=_KEY, how="inner").sort(_KEY)
            if joined.height == 0:
                raise ValueError(
                    "no parcels remain after joining the base OOF with gt_labels; "
                    "check that canonical_parcel_id namespaces match."
                )
            y = joined[_LABEL].to_numpy().astype(np.int64)

        if joined.height == 0:
            raise ValueError(
                "the base learners share no common parcel after the inner join; "
                "verify the OOF parquets cover the same fold-5 parcels."
            )

        feature_cols: list[str] = []
        for member in self.base_members:
            feature_cols.extend(member_prob_cols[member])
        x_meta = joined.select(feature_cols).to_numpy().astype(np.float64)
        keys_df = joined.select(_KEY)
        logger.debug(
            "stacking_meta_features_built",
            n_parcels=keys_df.height,
            n_members=len(self.base_members),
            n_meta_features=x_meta.shape[1],
            has_gt=y is not None,
        )
        return keys_df, x_meta, y

    @staticmethod
    def _prepare_gt(gt_labels: pl.DataFrame) -> pl.DataFrame:
        """Validate and normalize the per-parcel ground-truth frame.

        Args:
            gt_labels: Frame with ``canonical_parcel_id`` + ``label``.

        Returns:
            The frame with a canonical Utf8 key and an ``int64`` ``label``.

        Raises:
            ValueError: if either required column is missing.
        """
        for col in (_KEY, _LABEL):
            if col not in gt_labels.columns:
                raise ValueError(
                    f"gt_labels must carry the `{col}` column (per-parcel PASTIS "
                    "semantic18 ground truth lives OUTSIDE the OOF parquet)."
                )
        gt = canonical_parcel_id(gt_labels, col=_KEY)
        return gt.select([_KEY, pl.col(_LABEL).cast(pl.Int64)])

    # ------------------------------------------------------------------
    # Spatial sub-fold mapping (R-OOF-DEPTH).
    # ------------------------------------------------------------------

    def _subfolds_by_canonical_id(
        self,
        parcel_geoms: pl.DataFrame,
        keys_df: pl.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Map the fold-5 spatial sub-folds onto positional meta-row indices.

        Builds a GeoDataFrame from ``parcel_geoms`` (a ``canonical_parcel_id`` +
        ``geometry`` frame), assigns each parcel a synthetic integer
        ``parcel_id`` equal to its position in the meta matrix, calls
        :meth:`EnsembleModel.spatial_subfolds` (delegating to
        :func:`ml.features.spatial_split.build_spatial_kfold`) and converts each
        fold's ``test_ids`` (the held-out spatial block) into ``(train_pos,
        test_pos)`` positional index arrays over the meta rows.

        Args:
            parcel_geoms: Per-parcel geometry frame with ``canonical_parcel_id``
                and a ``geometry`` column (WKT/WKB or shapely objects) in
                EPSG:4326.
            keys_df: One-column frame of the meta-row ``canonical_parcel_id`` in
                fit order (from :meth:`build_meta_features`).

        Returns:
            List of ``(train_pos, test_pos)`` positional index tuples, one per
            non-empty spatial sub-fold. Train and test are DISJOINT by
            construction (the buffer excludes border parcels from both).

        Raises:
            ValueError: if no parcel geometry matches the meta rows.
        """
        geoms = canonical_parcel_id(parcel_geoms, col=_KEY)
        keys = keys_df[_KEY].to_list()
        pos_by_key = {k: i for i, k in enumerate(keys)}

        gdf = self._to_geodataframe(geoms)
        gdf = gdf[gdf[_KEY].isin(pos_by_key)].copy()
        if len(gdf) == 0:
            raise ValueError(
                "no parcel geometry matches the meta rows; check that "
                "parcel_geoms uses the same canonical_parcel_id namespace."
            )
        # Synthetic integer parcel_id = positional index in the meta matrix.
        positions = np.array([pos_by_key[k] for k in gdf[_KEY]], dtype=np.int64)
        gdf = gdf.assign(parcel_id=positions)

        assignments = self.spatial_subfolds(
            gdf,
            n_folds=self.n_spatial_folds,
            buffer_km=self.buffer_km,
        )

        n_rows = keys_df.height
        all_pos = np.arange(n_rows, dtype=np.int64)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in assignments:
            test_pos = np.array(sorted(fold.test_ids), dtype=np.int64)
            train_pos = np.array(sorted(set(fold.train_ids) | set(fold.val_ids)), dtype=np.int64)
            # Defensive bounds + disjointness (the buffer already guarantees it).
            test_pos = test_pos[np.isin(test_pos, all_pos)]
            train_pos = train_pos[np.isin(train_pos, all_pos)]
            if test_pos.size == 0 or train_pos.size == 0:
                continue
            splits.append((train_pos, test_pos))

        if not splits:
            raise ValueError(
                "the spatial sub-folds of fold-5 produced no usable train/test "
                "split; reduce n_spatial_folds or buffer_km."
            )
        return splits

    @staticmethod
    def _to_geodataframe(geoms: pl.DataFrame):  # type: ignore[no-untyped-def]
        """Coerce a Polars geometry frame to a GeoDataFrame in EPSG:4326.

        Accepts a ``geometry`` column of shapely objects, WKT strings or WKB
        bytes. The geometry parsing is delegated to geopandas/shapely so the
        spatial split sees real polygons/points.

        Args:
            geoms: Polars frame with ``canonical_parcel_id`` + ``geometry``.

        Returns:
            A ``geopandas.GeoDataFrame`` with ``canonical_parcel_id`` and an
            active ``geometry`` in EPSG:4326.

        Raises:
            ValueError: if the ``geometry`` column is missing or unparseable.
        """
        import geopandas as gpd
        from shapely import wkb, wkt
        from shapely.geometry.base import BaseGeometry

        if "geometry" not in geoms.columns:
            raise ValueError("parcel_geoms must carry a `geometry` column.")
        ids = geoms[_KEY].to_list()
        raw = geoms["geometry"].to_list()
        parsed: list[BaseGeometry] = []
        for value in raw:
            if value is None:
                raise ValueError("parcel_geoms has a null geometry.")
            if isinstance(value, BaseGeometry):
                parsed.append(value)
            elif isinstance(value, bytes | bytearray):
                parsed.append(wkb.loads(bytes(value)))
            elif isinstance(value, str):
                parsed.append(wkt.loads(value))
            else:  # pragma: no cover - defensive, unsupported geometry encoding
                raise ValueError(
                    f"unsupported geometry encoding: {type(value)!r}; pass shapely, WKT or WKB."
                )
        return gpd.GeoDataFrame({_KEY: ids}, geometry=parsed, crs="EPSG:4326")

    # ------------------------------------------------------------------
    # Meta-learner construction.
    # ------------------------------------------------------------------

    def _build_meta_learner(self) -> ClassifierMixin:
        """Instantiate the meta-learner (logistic regression or XGBoost).

        Returns:
            An unfitted sklearn-compatible classifier exposing ``predict_proba``.
        """
        if self.meta == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        # XGBoost meta-learner. Uses the project's SpatialXGBClassifier so a
        # spatial sub-fold missing rare classes does not crash with "Invalid
        # classes inferred"; it re-encodes labels locally per fit.
        from ml.train.baseline import SpatialXGBClassifier

        return SpatialXGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softprob",
            random_state=self.random_state,
        )

    @staticmethod
    def _proba_to_global(proba_local: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
        """Expand a meta-learner's local proba to the full 18-class space.

        A meta-learner trained on a sub-fold may only observe a subset of the 18
        classes; its ``predict_proba`` then has fewer columns. This scatters the
        local columns back to their global class index so every prediction lives
        in a fixed ``(n, 18)`` post-softmax space.

        Args:
            proba_local: ``(n, n_local_classes)`` proba from the meta-learner.
            model_classes: The class ids of the local columns, in order.

        Returns:
            A ``(n, 18)`` array in the global class space, summing to 1 per row.
        """
        n = proba_local.shape[0]
        full = np.zeros((n, _NUM_CLASSES), dtype=np.float64)
        for local_idx, cls in enumerate(model_classes):
            cls_int = int(cls)
            if 0 <= cls_int < _NUM_CLASSES:
                full[:, cls_int] = proba_local[:, local_idx]
        row_sums: np.ndarray = full.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        normalized: np.ndarray = full / row_sums
        return normalized

    # ------------------------------------------------------------------
    # Fit (anti-leakage core).
    # ------------------------------------------------------------------

    def fit(
        self,
        parcel_geoms: pl.DataFrame,
        *,
        gt_labels: pl.DataFrame,
    ) -> StackingEnsemble:
        """Fit the meta-learner on OOF predictions ONLY, with spatial CV.

        Pipeline (every step is leakage-guarded):

        1. Assemble the OOF-only meta features (post-softmax ``prob_*`` of the
           base learners, inner-joined on ``canonical_parcel_id``) and the
           aligned GT labels (GT lives OUTSIDE the OOF parquet).
        2. Partition the fold-5 parcels into geographic sub-folds with
           :func:`build_spatial_kfold` (NEVER a random split; R-OOF-DEPTH).
        3. For each sub-fold ``k``: train a fresh meta-learner on the OOF rows
           of the OTHER sub-folds and predict on ``k``;
           :meth:`assert_oof_only` hard-fails any train/eval parcel overlap.
           Aggregate the per-sub-fold F1-macro/accuracy (the leakage-free
           estimate stored in :attr:`oof_cv_metrics_`).
        4. Refit the meta-learner on ALL OOF rows for downstream
           :meth:`predict_proba` (this final fit still only ever saw OOF base
           predictions -- no raw features, no logits).

        Args:
            parcel_geoms: Per-parcel geometry frame with ``canonical_parcel_id``
                and a ``geometry`` column (shapely/WKT/WKB) in EPSG:4326, used to
                build the geographic sub-folds.
            gt_labels: Per-parcel PASTIS semantic18 ground truth with
                ``canonical_parcel_id`` + ``label`` (kept out of the OOF dump),
                used to train and score the meta-learner.

        Returns:
            ``self`` for chaining, with :attr:`meta_model_`,
            :attr:`meta_classes_` and :attr:`oof_cv_metrics_` populated.

        Raises:
            ValueError: if the OOF/GT join is empty, the base frames are not
                post-softmax, or the spatial sub-folds produce no usable split.
        """
        keys_df, x_meta, y = self.build_meta_features(gt_labels=gt_labels)
        if y is None:  # pragma: no cover - build_meta_features returns y when GT given
            raise ValueError("gt_labels did not yield aligned labels.")

        splits = self._subfolds_by_canonical_id(parcel_geoms, keys_df)
        keys = keys_df[_KEY].to_list()

        per_fold: list[dict[str, float]] = []
        for fold_idx, (train_pos, test_pos) in enumerate(splits):
            # Anti-leakage HARD GUARD: train and eval parcels must be disjoint.
            train_ids = [keys[i] for i in train_pos]
            eval_ids = [keys[i] for i in test_pos]
            EnsembleModel.assert_oof_only(
                train_ids, eval_ids, context=f"stacking sub-fold {fold_idx}"
            )

            model = self._build_meta_learner()
            model.fit(x_meta[train_pos], y[train_pos])
            proba_local = np.asarray(model.predict_proba(x_meta[test_pos]))
            proba_full = self._proba_to_global(proba_local, np.asarray(model.classes_))
            preds = proba_full.argmax(axis=1)
            fold_metrics = EnsembleModel.compute_metrics(
                y[test_pos], preds, num_classes=_NUM_CLASSES, ignore_index=None
            )
            per_fold.append(fold_metrics)
            logger.info(
                "stacking_subfold_done",
                fold=f"{fold_idx + 1}/{len(splits)}",
                n_train=int(train_pos.size),
                n_test=int(test_pos.size),
                f1_macro=round(fold_metrics["f1_macro"], 4),
            )

        self.oof_cv_metrics_ = _aggregate_metrics(per_fold)

        # Final refit on ALL OOF rows (still OOF-only: base predictions, not
        # raw features). This is the model used by predict_proba downstream.
        final_model = self._build_meta_learner()
        final_model.fit(x_meta, y)
        self.meta_model_ = final_model
        self.meta_classes_ = np.asarray(final_model.classes_)
        self._fit_parcel_ids = tuple(keys)

        logger.info(
            "stacking_fit_done",
            meta=self.meta,
            n_members=len(self.base_members),
            n_parcels=keys_df.height,
            n_subfolds=len(splits),
            f1_macro_oof=round(self.oof_cv_metrics_.get("f1_macro", float("nan")), 4),
        )
        return self

    # ------------------------------------------------------------------
    # Predict.
    # ------------------------------------------------------------------

    def predict_proba(self, parcel_ids: Sequence[str] | None = None) -> np.ndarray:
        """Return per-parcel post-softmax probabilities ``(n_parcels, 18)``.

        Reassembles the OOF-only meta features (the SAME post-softmax base
        ``prob_*`` columns used in :meth:`fit`) and runs the refit meta-learner.
        The output is expanded to the full 18-class space and validated as a
        post-softmax distribution.

        Args:
            parcel_ids: Optional subset of ``canonical_parcel_id`` to predict
                (in the returned row order). When ``None`` every parcel shared
                by the base learners is predicted (the fit order).

        Returns:
            A ``(n_parcels, 18)`` ``float64`` array summing to 1 per row.

        Raises:
            RuntimeError: if called before :meth:`fit`.
            ValueError: if a requested parcel id is absent from the joined base
                OOF.
        """
        if self.meta_model_ is None or self.meta_classes_ is None:
            raise RuntimeError("call fit(...) before predict_proba(...).")

        keys_df, x_meta, _ = self.build_meta_features(gt_labels=None)
        keys = keys_df[_KEY].to_list()

        if parcel_ids is not None:
            pos_by_key = {k: i for i, k in enumerate(keys)}
            requested = [str(p) for p in parcel_ids]
            missing = [p for p in requested if p not in pos_by_key]
            if missing:
                raise ValueError(
                    f"{len(missing)} requested parcel id(s) are not in the joined "
                    f"base OOF (e.g. {missing[:5]})."
                )
            order = np.array([pos_by_key[p] for p in requested], dtype=np.int64)
            x_meta = x_meta[order]

        proba_local = np.asarray(self.meta_model_.predict_proba(x_meta))
        proba_full = self._proba_to_global(proba_local, self.meta_classes_)
        return EnsembleModel.validate_probs(proba_full, class_axis=-1, name="stacking_proba")


def _aggregate_metrics(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Average the per-sub-fold metrics into a single mean estimate.

    Args:
        per_fold: List of ``{"f1_macro": ..., "accuracy": ...}`` dicts.

    Returns:
        ``{"f1_macro": mean, "accuracy": mean}``; ``nan`` if there were no
        sub-folds.
    """
    if not per_fold:
        return {"f1_macro": float("nan"), "accuracy": float("nan")}
    keys = per_fold[0].keys()
    return {key: float(np.mean([fold[key] for fold in per_fold])) for key in keys}
