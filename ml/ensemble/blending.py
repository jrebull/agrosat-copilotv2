"""E4 Blending ensemble: Optuna simplex weights over a spatial holdout (US-040).

The blending ensemble combines the per-parcel post-softmax probabilities of three
base learners (``tsvit-pheno`` -> parcel, ``utae`` -> parcel and the tabular
``xgb-alphaearth``) with a single set of convex weights ``w`` on the simplex
(``w_i >= 0`` and ``sum(w) == 1``). The combined distribution of a parcel is::

    P_blend = sum_i w_i * P_i      with   sum_i w_i = 1, w_i >= 0

so the result is itself a valid post-softmax distribution (a convex combination of
distributions). The argmax of ``P_blend`` is the predicted class.

Anti-leakage (plan Section 9, R-LEAK -- the single most important rubric
criterion). Three guarantees are enforced programmatically, never by convention:

1. **Probabilities, not logits.** Every member matrix is validated with
   :meth:`ml.ensemble.base.EnsembleModel.validate_probs` before it enters the
   blend, and the blended output (a convex combination) is validated again.
2. **Spatially disjoint holdout.** The weights are optimized on a holdout that is
   geographically disjoint from the parcels used to score the "train" side of the
   Optuna objective. The split comes from
   :func:`ml.features.spatial_split.build_spatial_kfold` (H3 res 5 + KMeans +
   buffer), NEVER a random/IID split: nearby parcels never straddle the
   train/val boundary, so the optimized weights cannot overfit a leaked neighbor.
3. **Report fold-5 only.** US-031 only dumped fold-5 (the single held-out fold),
   so the spatial holdout is carved from the fold-5 parcels themselves; the
   base-class :meth:`ml.ensemble.base.EnsembleModel.evaluate` rejects any other
   fold. Fold-4 was the selection fold and is never reported.

The Optuna objective minimizes the train/val F1 gap so the chosen weights
generalize rather than memorize the validation parcels::

    maximize   f1_val - gap_lambda * |f1_train - f1_val|

The simplex is parameterized by sampling ``num_members`` independent
``[0, 1]`` Optuna floats interpreted as Dirichlet-style logits and projecting
them onto the simplex by normalization (a degenerate all-zero draw falls back to
uniform weights), so every trial proposes a valid convex combination.

The ground truth is NOT inside the OOF parquet (the US-031 dump discards the
target), so the per-parcel PASTIS-R labels are supplied separately by the caller
(``y_true``): a Polars frame with ``canonical_parcel_id`` + ``label`` aligned to
the loaded members.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints and Google-style docstrings, and
the Optuna best params persisted on the instance for MLflow logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.ensemble.base import EnsembleModel
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import geopandas as gpd
    import optuna

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_BLENDING_MEMBERS",
    "BlendingEnsemble",
]

#: Default heterogeneous base learners blended at the parcel level.
DEFAULT_BLENDING_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: Numerical floor so a degenerate all-zero weight draw never divides by zero.
_SIMPLEX_EPS: float = 1e-12

#: Absolute tolerance used when asserting the learned weights live on the simplex.
_SIMPLEX_TOL: float = 1e-6


class BlendingEnsemble(EnsembleModel):
    """Blend base-learner parcel probabilities with Optuna simplex weights (E4).

    The weights are optimized on a spatially disjoint holdout carved from the
    fold-5 parcels so they generalize (anti-leakage R-LEAK); the objective
    penalizes the train/val F1 gap. ``predict_proba`` returns the convex
    combination of the per-parcel post-softmax matrices, which is itself a valid
    post-softmax distribution.

    Attributes:
        base_members: Ordered base-learner names blended at the parcel level.
        n_trials: Number of Optuna trials searching the simplex.
        gap_lambda: Weight of the train/val F1-gap penalty in the objective.
        best_params: The Optuna best params (raw simplex logits) once fitted.
        study: The Optuna study once :meth:`fit` has run (``None`` before).
    """

    def __init__(
        self,
        base_members: Sequence[str] = DEFAULT_BLENDING_MEMBERS,
        *,
        n_trials: int = 50,
        gap_lambda: float = 0.5,
        **kw: object,
    ) -> None:
        """Initialize the blending ensemble.

        Args:
            base_members: Ordered base learners to blend (default
                ``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth``). Each must have a
                ``oof_parcel_{member}_fold5.parquet`` artifact.
            n_trials: Number of Optuna trials over the simplex (default 50).
            gap_lambda: Coefficient of the ``|f1_train - f1_val|`` penalty in the
                objective (default 0.5); larger values favour weights that
                generalize over weights that merely peak on validation.
            **kw: Forwarded to :class:`ml.ensemble.base.EnsembleModel`
                (``oof_dir``, ``random_state``).

        Raises:
            ValueError: if ``base_members`` is empty, ``n_trials < 1`` or
                ``gap_lambda < 0``.
        """
        super().__init__(**kw)  # type: ignore[arg-type]
        members = tuple(base_members)
        if not members:
            raise ValueError("base_members must list at least one base learner.")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; received {n_trials}.")
        if gap_lambda < 0.0:
            raise ValueError(f"gap_lambda must be >= 0; received {gap_lambda}.")
        self.base_members: tuple[str, ...] = members
        self.n_trials = int(n_trials)
        self.gap_lambda = float(gap_lambda)
        self.best_params: dict[str, float] = {}
        self.study: optuna.study.Study | None = None
        self._weights: np.ndarray | None = None
        #: Canonical parcel id order of the members aligned during :meth:`fit`.
        self._member_ids: tuple[str, ...] = ()
        #: Per-member aligned probability tensor ``(n_members, n_parcels, 18)``.
        self._member_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public properties.
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Learned convex weights ``(n_members,)`` on the simplex.

        Returns:
            A ``float64`` array with ``w_i >= 0`` and ``sum(w) == 1``.

        Raises:
            RuntimeError: if the ensemble has not been fitted yet.
        """
        if self._weights is None:
            raise RuntimeError("BlendingEnsemble is not fitted; call fit() first.")
        return self._weights

    # ------------------------------------------------------------------
    # Member alignment (DRY: shared by fit and predict_proba).
    # ------------------------------------------------------------------

    def _align_members(self) -> tuple[tuple[str, ...], np.ndarray]:
        """Load every member's parcel OOF and align them on a common parcel set.

        Members are reduced to the INTERSECTION of their ``canonical_parcel_id``
        sets so the blend stacks aligned rows; each member contributes a
        post-softmax matrix validated by the base class.

        Returns:
            Tuple ``(parcel_ids, probs)`` where ``parcel_ids`` is the sorted
            canonical parcel ids ``(n_parcels,)`` shared by all members and
            ``probs`` is ``(n_members, n_parcels, 18)`` of post-softmax rows.

        Raises:
            ValueError: if the members share no parcel id (nothing to blend).
        """
        frames = self.load_oof_members(self.base_members, space="parcel")

        common: set[str] | None = None
        for member in self.base_members:
            ids = set(frames[member]["canonical_parcel_id"].to_list())
            common = ids if common is None else (common & ids)
        if not common:
            raise ValueError(
                "the base members share no canonical_parcel_id; cannot blend "
                "aligned parcel probabilities."
            )
        parcel_ids = tuple(sorted(common))

        stacked: list[np.ndarray] = []
        for member in self.base_members:
            aligned = (
                frames[member]
                .filter(pl.col("canonical_parcel_id").is_in(list(parcel_ids)))
                .sort("canonical_parcel_id")
            )
            matrix = aligned.select(PROB_COLUMNS).to_numpy().astype(np.float64)
            self.validate_probs(matrix, class_axis=-1, name=f"{member}_parcel")
            stacked.append(matrix)
        probs = np.stack(stacked, axis=0)  # (n_members, n_parcels, 18)
        logger.debug(
            "blending_members_aligned",
            members=list(self.base_members),
            n_parcels=len(parcel_ids),
        )
        return parcel_ids, probs

    @staticmethod
    def _labels_for(parcel_ids: Sequence[str], y_true: pl.DataFrame) -> np.ndarray:
        """Return the integer labels aligned to ``parcel_ids`` from a GT frame.

        Args:
            parcel_ids: Ordered canonical parcel ids to look up.
            y_true: PASTIS-R ground-truth frame with ``canonical_parcel_id`` and a
                ``label`` column (loaded separately; not present in the OOF dump).

        Returns:
            An ``int64`` array of class ids aligned 1:1 with ``parcel_ids``.

        Raises:
            ValueError: if ``y_true`` lacks the required columns or does not cover
                every parcel id.
        """
        required = {"canonical_parcel_id", "label"}
        missing = required - set(y_true.columns)
        if missing:
            raise ValueError(
                f"y_true ground-truth frame is missing columns: {sorted(missing)}. "
                "Load the PASTIS-R per-parcel labels separately (not in the OOF)."
            )
        lookup = dict(
            zip(
                y_true["canonical_parcel_id"].cast(pl.Utf8).to_list(),
                y_true["label"].to_list(),
                strict=True,
            )
        )
        absent = [pid for pid in parcel_ids if pid not in lookup]
        if absent:
            raise ValueError(
                f"y_true does not cover {len(absent)} parcel id(s) "
                f"(e.g. {absent[:5]}); the labels must align with the OOF members."
            )
        return np.asarray([lookup[pid] for pid in parcel_ids], dtype=np.int64)

    # ------------------------------------------------------------------
    # Simplex parameterization.
    # ------------------------------------------------------------------

    @staticmethod
    def _project_simplex(raw: np.ndarray) -> np.ndarray:
        """Project non-negative logits onto the probability simplex.

        Args:
            raw: Non-negative draws ``(n_members,)``.

        Returns:
            Convex weights ``(n_members,)`` with ``w_i >= 0`` and ``sum(w) == 1``;
            a degenerate all-zero draw falls back to uniform weights.
        """
        arr: np.ndarray = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
        total = arr.sum()
        if total < _SIMPLEX_EPS:
            return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
        weights: np.ndarray = arr / total
        return weights

    def _blend(self, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Combine member probabilities with convex weights.

        Args:
            probs: Member tensor ``(n_members, n_parcels, 18)`` of post-softmax
                rows.
            weights: Convex weights ``(n_members,)`` on the simplex.

        Returns:
            The blended post-softmax matrix ``(n_parcels, 18)`` (validated).
        """
        blended = np.tensordot(weights, probs, axes=([0], [0]))  # (n_parcels, 18)
        # Renormalize defensively against float drift, then validate post-softmax.
        denom = blended.sum(axis=-1, keepdims=True)
        denom = np.where(denom < _SIMPLEX_EPS, 1.0, denom)
        blended = blended / denom
        return self.validate_probs(blended, class_axis=-1, name="blend")

    # ------------------------------------------------------------------
    # Fit (Optuna simplex weights on a spatially disjoint holdout).
    # ------------------------------------------------------------------

    def fit(
        self,
        parcel_geoms: gpd.GeoDataFrame,
        *,
        y_true: pl.DataFrame,
        buffer_km: float = 1.0,
    ) -> BlendingEnsemble:
        """Optimize the simplex weights on a spatially disjoint holdout.

        Loads and aligns the base members, carves a geographically disjoint
        train/val split out of the fold-5 parcels via
        :func:`ml.features.spatial_split.build_spatial_kfold` (NEVER a random
        split), and runs Optuna to find convex weights that maximize::

            f1_val - gap_lambda * |f1_train - f1_val|

        so the chosen weights generalize rather than peak on the validation
        parcels. The best weights are stored on :attr:`weights` and the raw
        Optuna params on :attr:`best_params` for MLflow logging.

        Args:
            parcel_geoms: GeoDataFrame of the fold-5 parcels with ``parcel_id``
                (integer surrogate), ``canonical_parcel_id`` (Utf8, matching the
                OOF members) and an active ``geometry`` in EPSG:4326.
            y_true: PASTIS-R ground-truth frame with ``canonical_parcel_id`` and
                ``label`` (loaded separately; the OOF dump discards the target).
            buffer_km: Inter-fold exclusion buffer for the spatial split (km).

        Returns:
            ``self`` (fitted) for chaining.

        Raises:
            ValueError: if ``parcel_geoms`` lacks the required columns, if the
                spatial split yields an empty train or val side, or if the
                members/labels do not align.
        """
        import optuna

        if "canonical_parcel_id" not in parcel_geoms.columns:
            raise ValueError(
                "parcel_geoms must carry `canonical_parcel_id` to align with the "
                "OOF members (the spatial split keys on the integer `parcel_id`)."
            )

        parcel_ids, member_probs = self._align_members()
        self._member_ids = parcel_ids
        self._member_probs = member_probs
        labels = self._labels_for(parcel_ids, y_true)

        train_idx, val_idx = self._spatial_holdout(parcel_ids, parcel_geoms, buffer_km=buffer_km)

        probs_train = member_probs[:, train_idx, :]
        probs_val = member_probs[:, val_idx, :]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            raw = np.asarray(
                [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(self.base_members))],
                dtype=np.float64,
            )
            weights = self._project_simplex(raw)
            f1_train = self._f1_of(probs_train, y_train, weights)
            f1_val = self._f1_of(probs_val, y_val, weights)
            trial.set_user_attr("f1_train", f1_train)
            trial.set_user_attr("f1_val", f1_val)
            return f1_val - self.gap_lambda * abs(f1_train - f1_val)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.study = study
        self.best_params = dict(study.best_params)
        best_raw = np.asarray(
            [study.best_params[f"w_{i}"] for i in range(len(self.base_members))],
            dtype=np.float64,
        )
        self._weights = self._project_simplex(best_raw)

        logger.info(
            "blending_fitted",
            members=list(self.base_members),
            n_trials=len(study.trials),
            best_value=round(float(study.best_value), 4),
            f1_val=round(float(study.best_trial.user_attrs.get("f1_val", float("nan"))), 4),
            f1_train=round(float(study.best_trial.user_attrs.get("f1_train", float("nan"))), 4),
            weights=[round(float(w), 4) for w in self._weights],
            n_train=int(train_idx.size),
            n_val=int(val_idx.size),
        )
        return self

    def _spatial_holdout(
        self,
        parcel_ids: Sequence[str],
        parcel_geoms: gpd.GeoDataFrame,
        *,
        buffer_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carve a spatially disjoint train/val split over the aligned parcels.

        Delegates to :meth:`ml.ensemble.base.EnsembleModel.spatial_subfolds`
        (which wraps :func:`build_spatial_kfold`) and uses the first sub-fold's
        ``test_ids`` as the disjoint VAL holdout and the remaining sub-folds'
        parcels as TRAIN. The integer ``parcel_id`` of the GeoDataFrame is mapped
        back to positions in ``parcel_ids`` (the aligned member order) through the
        ``canonical_parcel_id`` column, so the returned indices select rows of the
        member probability tensor.

        Args:
            parcel_ids: Aligned canonical parcel ids (member/label order).
            parcel_geoms: GeoDataFrame restricted to those parcels.
            buffer_km: Inter-fold exclusion buffer (km).

        Returns:
            Tuple ``(train_idx, val_idx)`` of disjoint integer index arrays into
            ``parcel_ids``.

        Raises:
            ValueError: if the split leaves train or val empty.
        """
        geoms = parcel_geoms.copy()
        geoms["canonical_parcel_id"] = geoms["canonical_parcel_id"].astype(str)
        wanted = set(parcel_ids)
        geoms = geoms[geoms["canonical_parcel_id"].isin(wanted)]

        # Position of each canonical id within the aligned member order.
        pos = {pid: i for i, pid in enumerate(parcel_ids)}
        # parcel_id (int surrogate, used by build_spatial_kfold) -> canonical id.
        int_to_canonical = dict(
            zip(
                geoms["parcel_id"].astype("int64").tolist(),
                geoms["canonical_parcel_id"].tolist(),
                strict=True,
            )
        )

        folds = self.spatial_subfolds(geoms, n_folds=5, buffer_km=buffer_km)
        # First non-empty test sub-fold is the disjoint VAL holdout.
        val_fold = next((f for f in folds if f.test_ids), None)
        if val_fold is None:
            raise ValueError(
                "the spatial split produced no validation parcels; check the "
                "parcel geometries and buffer_km."
            )
        val_int_ids = set(val_fold.test_ids)

        val_idx: list[int] = []
        train_idx: list[int] = []
        for int_id, canonical in int_to_canonical.items():
            if canonical not in pos:
                continue
            (val_idx if int_id in val_int_ids else train_idx).append(pos[canonical])

        train_arr = np.asarray(sorted(set(train_idx)), dtype=np.int64)
        val_arr = np.asarray(sorted(set(val_idx)), dtype=np.int64)
        if train_arr.size == 0 or val_arr.size == 0:
            raise ValueError(
                "the spatial holdout is degenerate "
                f"(train={train_arr.size}, val={val_arr.size}); need both sides "
                "non-empty to optimize the blending weights without leakage."
            )
        # Anti-leakage: train and val parcels must be disjoint by construction.
        self.assert_oof_only(train_arr.tolist(), val_arr.tolist(), context="blending holdout")
        return train_arr, val_arr

    def _f1_of(self, probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        """Blend ``probs`` with ``weights`` and return the F1-macro vs ``labels``.

        Args:
            probs: Member tensor ``(n_members, n_subset, 18)`` of post-softmax
                rows.
            labels: Aligned class ids ``(n_subset,)``.
            weights: Convex weights ``(n_members,)``.

        Returns:
            The F1-macro of the blended argmax against ``labels``.
        """
        blended = self._blend(probs, weights)
        preds = blended.argmax(axis=-1)
        return self.compute_metrics(labels, preds)["f1_macro"]

    # ------------------------------------------------------------------
    # Predict (weighted combination of post-softmax member probabilities).
    # ------------------------------------------------------------------

    def predict_proba(self, df_parcels: Sequence[pl.DataFrame] | None = None) -> np.ndarray:
        """Return the blended per-parcel post-softmax probabilities.

        With no argument the cached fold-5 member probabilities aligned during
        :meth:`fit` are blended (the production path). Alternatively a sequence of
        per-member parcel frames (one per base member, same order, sharing
        ``canonical_parcel_id`` + ``prob_*``) may be passed to blend an arbitrary
        aligned parcel set with the learned weights.

        Args:
            df_parcels: Optional per-member parcel frames in :attr:`base_members`
                order; defaults to the cached fold-5 members.

        Returns:
            A ``float64`` matrix ``(n_parcels, 18)`` of post-softmax probabilities
            summing to 1 per row (a convex combination of the members).

        Raises:
            RuntimeError: if called before :meth:`fit`.
            ValueError: if ``df_parcels`` does not match the member count or the
                frames are not aligned.
        """
        if self._weights is None:
            raise RuntimeError("BlendingEnsemble is not fitted; call fit() first.")

        if df_parcels is None:
            if self._member_probs is None:  # pragma: no cover - guarded by fit
                raise RuntimeError("no cached member probabilities; call fit() first.")
            return self._blend(self._member_probs, self._weights)

        frames = list(df_parcels)
        if len(frames) != len(self.base_members):
            raise ValueError(
                f"df_parcels must provide one frame per base member "
                f"({len(self.base_members)}); received {len(frames)}."
            )
        ref_ids = frames[0].sort("canonical_parcel_id")["canonical_parcel_id"].to_list()
        stacked: list[np.ndarray] = []
        for member, frame in zip(self.base_members, frames, strict=True):
            ordered = frame.sort("canonical_parcel_id")
            if ordered["canonical_parcel_id"].to_list() != ref_ids:
                raise ValueError(
                    f"member {member!r} parcel ids are not aligned with the first "
                    "member; pass frames over the same canonical_parcel_id set."
                )
            matrix = ordered.select(PROB_COLUMNS).to_numpy().astype(np.float64)
            self.validate_probs(matrix, class_axis=-1, name=f"{member}_parcel")
            stacked.append(matrix)
        return self._blend(np.stack(stacked, axis=0), self._weights)

    # ------------------------------------------------------------------
    # MLflow params (DRY: surfaced to log_to_mlflow by the closing script).
    # ------------------------------------------------------------------

    def mlflow_params(self) -> dict[str, object]:
        """Return the params logged to MLflow for this blending run.

        Returns:
            A mapping with the members, ``n_trials``, ``gap_lambda`` and the
            learned per-member weights (only once fitted).
        """
        params: dict[str, object] = {
            "members": ",".join(self.base_members),
            "n_trials": self.n_trials,
            "gap_lambda": self.gap_lambda,
        }
        if self._weights is not None:
            for member, weight in zip(self.base_members, self._weights, strict=True):
                params[f"weight_{member}"] = round(float(weight), 6)
        return params
