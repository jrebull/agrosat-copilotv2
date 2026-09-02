"""Abstract base for the four rubric ensembles (US-040, EPIC 6).

This module freezes the shared contract every ensemble in ``ml/ensemble/``
inherits from: Voting (pixel), Bagging (parcel), Stacking (parcel) and Blending
(parcel). It centralizes the pieces that would otherwise be duplicated across the
four ensembles (DRY): loading the US-031 out-of-fold (OOF) probabilities,
validating that those probabilities are POST-softmax (never logits), reducing a
dense softmax to parcel-level probabilities, computing F1-macro / accuracy, and
logging a run to MLflow with the mandatory versioning tags.

Anti-leakage invariants (plan Section 9, R-LEAK -- the single most important
rubric criterion). The base class enforces all three so a subclass cannot break
them by accident:

1. **Report fold-5 only.** :attr:`EnsembleModel.HELD_OUT_FOLD` is fixed at ``5``
   and :meth:`EnsembleModel.evaluate` raises ``ValueError`` for any other fold.
   Fold-4 was the model-selection fold and is NEVER reported.
2. **Probabilities, not logits.** Every probability matrix entering the
   ensemble must be post-softmax: non-negative and summing to 1 over the class
   axis. :meth:`EnsembleModel.validate_probs` rejects logits.
3. **Meta-learner sees OOF only.** The stacking meta-learner trains exclusively
   on OOF predictions. :meth:`EnsembleModel.spatial_subfolds` partitions the
   fold-5 parcels geographically (via ``build_spatial_kfold``) so a meta row is
   never built from a base learner that saw that parcel, and the
   :meth:`EnsembleModel.assert_oof_only` helper hard-fails any train/eval index
   overlap.

The ground truth is NOT stored inside the OOF parquet (the dump discards the
target). Therefore :meth:`EnsembleModel.evaluate` receives ``y_true`` from the
caller (a per-parcel label DataFrame or a per-patch ground-truth map dict). This
keeps the base testable without loading PASTIS-R or the ~1.5 GB OOF blobs.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints and Google-style docstrings
everywhere.
"""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import structlog

from ml.eval.oof.parquet_io import read_softmax_parquet
from ml.utils.mlflow_utils import track_experiment
from ml.utils.parcel_reconcile import PROB_COLUMNS, pixel_to_parcel_probs

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Mapping, Sequence

    import geopandas as gpd

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_OOF_DIR",
    "ENSEMBLE_EXPERIMENT",
    "EnsembleModel",
    "Space",
]

#: Space in which an ensemble operates: dense pixel maps or reduced parcels.
Space = Literal["pixel", "parcel"]

#: Default directory holding the US-031 OOF parquet artifacts.
DEFAULT_OOF_DIR: str = "ml/eval/oof"

#: MLflow experiment name for all four ensembles (server :5010).
ENSEMBLE_EXPERIMENT: str = "ensemble"

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = len(PROB_COLUMNS)

#: Absolute tolerance when asserting a probability row sums to 1.
_SUM_TO_ONE_TOL: float = 1e-4

#: DVC path forwarded to ``track_experiment`` so it can resolve ``data_version``.
_OOF_DVC_PATH: str = "ml/eval/oof"


class EnsembleModel(abc.ABC):
    """Abstract base for the 4 rubric ensembles. Reports ALWAYS on fold-5.

    Subclasses implement :meth:`fit` and :meth:`predict_proba`; the base provides
    OOF loading, probability validation, pixel->parcel reduction, fold-5-only
    evaluation, spatial sub-fold helpers (anti-leakage) and MLflow logging.

    Attributes:
        HELD_OUT_FOLD: The only fold whose metrics may be reported (``5``).
            Fold-4 was the selection fold and is never evaluated.
        oof_dir: Directory holding the US-031 OOF parquet files.
        random_state: Deterministic seed shared by every ensemble.
    """

    #: Report fold; fold-4 was selection -> NEVER reported (anti-leakage core).
    HELD_OUT_FOLD: int = 5

    def __init__(
        self,
        *,
        oof_dir: Path | str = DEFAULT_OOF_DIR,
        random_state: int = 42,
    ) -> None:
        """Initialize the shared ensemble state.

        Args:
            oof_dir: Directory holding the US-031 OOF parquet artifacts
                (``oof_{model}_fold5.parquet`` and
                ``oof_parcel_{model}_fold5.parquet``).
            random_state: Deterministic seed forwarded to bootstrap sampling,
                spatial folds and Optuna so every ensemble is reproducible.
        """
        self.oof_dir = Path(oof_dir)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # OOF loading (DRY: shared by the four ensembles).
    # ------------------------------------------------------------------

    def oof_path(self, member: str, *, space: Space) -> Path:
        """Return the OOF parquet path of a member in the requested space.

        Args:
            member: Base learner name (e.g. ``"tsvit-pheno"``, ``"utae"``).
            space: ``"pixel"`` -> ``oof_{member}_fold5.parquet`` (dense softmax);
                ``"parcel"`` -> ``oof_parcel_{member}_fold5.parquet``
                (``prob_000..prob_017``).

        Returns:
            The absolute/relative path under :attr:`oof_dir`.

        Raises:
            ValueError: if ``space`` is neither ``"pixel"`` nor ``"parcel"``.
        """
        if space == "pixel":
            return self.oof_dir / f"oof_{member}_fold{self.HELD_OUT_FOLD}.parquet"
        if space == "parcel":
            return self.oof_dir / f"oof_parcel_{member}_fold{self.HELD_OUT_FOLD}.parquet"
        raise ValueError(f"invalid space: {space!r}; use 'pixel' or 'parcel'.")

    def load_oof_members(
        self,
        members: Sequence[str],
        *,
        space: Space,
    ) -> dict[str, pl.DataFrame]:
        """Load the OOF parquet for each member in the requested space.

        - ``space="pixel"``: each value is the DataFrame returned by
          :func:`ml.eval.oof.parquet_io.read_softmax_parquet`, with the dense
          ``softmax`` ``(18, 128, 128)`` reconstructed per patch.
        - ``space="parcel"``: each value is the per-parcel DataFrame with
          ``canonical_parcel_id`` + ``prob_000..prob_017`` (already reconciled
          by US-031 via :func:`pixel_to_parcel_probs`).

        Args:
            members: Ordered base-learner names to load.
            space: ``"pixel"`` or ``"parcel"`` (see above).

        Returns:
            Mapping ``{member: DataFrame}`` preserving the input order.

        Raises:
            FileNotFoundError: if any member's OOF parquet is absent (run
                ``dvc pull ml/eval/oof`` to fetch them; do NOT regenerate).
            ValueError: if ``space`` is invalid.
        """
        out: dict[str, pl.DataFrame] = {}
        for member in members:
            path = self.oof_path(member, space=space)
            if not path.exists():
                raise FileNotFoundError(
                    f"OOF parquet not found for member {member!r} "
                    f"({space} space): {path}. Run `dvc pull {self.oof_dir}` to "
                    "fetch the US-031 OOF artifacts; do not regenerate them."
                )
            if space == "pixel":
                out[member] = read_softmax_parquet(path)
            else:
                out[member] = pl.read_parquet(path)
            logger.debug(
                "oof_member_loaded",
                member=member,
                space=space,
                path=str(path),
                n_rows=out[member].height,
            )
        return out

    # ------------------------------------------------------------------
    # Anti-leakage helpers (probabilities, not logits).
    # ------------------------------------------------------------------

    @staticmethod
    def validate_probs(
        probs: np.ndarray,
        *,
        class_axis: int = -1,
        name: str = "probs",
        tol: float = _SUM_TO_ONE_TOL,
    ) -> np.ndarray:
        """Assert a probability array is post-softmax (anti-leakage R-LEAK).

        A valid ensemble input must be non-negative and sum to 1 over the class
        axis: logits (which can be negative and never normalize) are rejected.
        This is the programmatic guard that the ensembles average PROBABILITIES,
        never logits.

        Args:
            probs: Probability array. The class axis may be the last axis
                (parcel table, shape ``(..., C)``) or any other via
                ``class_axis`` (e.g. ``0`` for a dense ``(C, H, W)`` softmax).
            class_axis: Axis along which classes are laid out.
            name: Human-readable name used in the error message.
            tol: Absolute tolerance for the sum-to-one check.

        Returns:
            ``probs`` unchanged (so callers can wrap it inline).

        Raises:
            ValueError: if any value is negative or NaN, or any class-axis slice
                does not sum to 1 within ``tol`` (i.e. the input is not a valid
                post-softmax distribution -- most likely raw logits).
        """
        arr = np.asarray(probs, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"{name} is empty; expected post-softmax probabilities.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values (NaN/inf); not a softmax.")
        if (arr < -tol).any():
            raise ValueError(
                f"{name} has negative values (min={float(arr.min()):.4g}); "
                "ensembles average POST-softmax probabilities, never logits."
            )
        sums = arr.sum(axis=class_axis)
        if not np.allclose(sums, 1.0, atol=tol):
            worst = float(np.abs(sums - 1.0).max())
            raise ValueError(
                f"{name} does not sum to 1 over the class axis "
                f"(max |sum-1|={worst:.4g} > tol={tol:g}); the input looks like "
                "logits, not a post-softmax distribution. Apply softmax first."
            )
        return probs

    @staticmethod
    def parcel_probs_matrix(df: pl.DataFrame) -> np.ndarray:
        """Extract the ``(n_parcels, 18)`` probability matrix from a parcel frame.

        Reads the canonical ``prob_000..prob_017`` columns in order and validates
        the result is post-softmax.

        Args:
            df: Parcel-level DataFrame with the ``prob_*`` columns.

        Returns:
            A ``float64`` array ``(n_parcels, 18)`` summing to 1 per row.

        Raises:
            ValueError: if any ``prob_*`` column is missing or the rows are not a
                valid post-softmax distribution.
        """
        missing = [c for c in PROB_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"parcel frame is missing prob columns: {missing}.")
        matrix = df.select(PROB_COLUMNS).to_numpy().astype(np.float64)
        return EnsembleModel.validate_probs(matrix, class_axis=-1, name="parcel_probs")

    def reduce_pixel_to_parcel(
        self,
        probs_18: np.ndarray,
        parcel_ids: np.ndarray,
        *,
        patch_id: str | int,
        method: Literal["mean", "mode"] = "mean",
    ) -> pl.DataFrame:
        """Reduce a dense softmax to parcel-level probabilities (reuse, DRY).

        Thin wrapper over :func:`ml.utils.parcel_reconcile.pixel_to_parcel_probs`
        so every parcel-space ensemble shares the SAME reconciliation (mean of
        post-softmax probabilities within the parcel geometry, PASTIS-R ~98%
        purity; the margin-pixel caveat is documented in plan Section 2.5). The
        input softmax is validated as post-softmax before reduction.

        Args:
            probs_18: Post-softmax map ``(18, H, W)``.
            parcel_ids: Per-pixel local ParcelIDs ``(H, W)`` (``0`` = Background).
            patch_id: PASTIS-R patch id used to build the canonical key.
            method: ``"mean"`` (default) or ``"mode"`` reduction.

        Returns:
            A parcel-level Polars DataFrame (see ``pixel_to_parcel_probs``).
        """
        self.validate_probs(probs_18, class_axis=0, name="pixel_softmax")
        return pixel_to_parcel_probs(probs_18, parcel_ids, patch_id=patch_id, method=method)

    # ------------------------------------------------------------------
    # Spatial sub-folds (anti-leakage: meta-learner sees OOF only).
    # ------------------------------------------------------------------

    def spatial_subfolds(
        self,
        parcel_geoms: gpd.GeoDataFrame,
        *,
        n_folds: int = 5,
        buffer_km: float = 1.0,
    ) -> list[FoldAssignment]:
        """Partition the fold-5 parcels into geographic sub-folds (R-OOF-DEPTH).

        US-031 only dumped fold-5 (the single held-out fold), so a strict 5-fold
        OOF CV is impossible. The anti-leakage substitute is a geographic
        partition of the fold-5 parcels themselves: the stacking meta-learner
        trains on the OOF predictions of the OTHER sub-folds and is evaluated on
        the held-out sub-fold, so a base learner never contributes a meta feature
        for a parcel it would have to predict in the held-out sub-fold. Delegates
        to :func:`ml.features.spatial_split.build_spatial_kfold` (H3 res 5 +
        KMeans + buffer), NEVER a random/IID split.

        Args:
            parcel_geoms: GeoDataFrame of the fold-5 parcels with ``parcel_id``
                and an active ``geometry`` in EPSG:4326.
            n_folds: Number of geographic sub-folds (default 5).
            buffer_km: Inter-fold exclusion buffer in km (default 1.0).

        Returns:
            A list of ``FoldAssignment`` (disjoint train/val/test parcel ids).
        """
        from ml.features.spatial_split import build_spatial_kfold

        return build_spatial_kfold(
            parcel_geoms,
            k=n_folds,
            buffer_km=buffer_km,
            random_state=self.random_state,
        )

    @staticmethod
    def assert_oof_only(
        train_ids: Sequence[object],
        eval_ids: Sequence[object],
        *,
        context: str = "meta-learner",
    ) -> None:
        """Hard-fail if the meta train and eval parcel sets overlap (R-LEAK).

        The stacking meta-learner must be trained on parcels DISJOINT from the
        parcels it is evaluated on, so the OOF guarantee holds. Any intersection
        means a base learner's prediction for an eval parcel leaked into the meta
        training set.

        Args:
            train_ids: Parcel ids the meta-learner was trained on.
            eval_ids: Parcel ids the meta-learner is evaluated on.
            context: Label used in the error message.

        Raises:
            ValueError: if ``set(train_ids) & set(eval_ids)`` is non-empty.
        """
        overlap = set(map(str, train_ids)) & set(map(str, eval_ids))
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"{context} leakage: {len(overlap)} parcel(s) appear in BOTH the "
                f"meta train and eval sets (e.g. {sample}). The meta-learner must "
                "see OOF predictions only (disjoint sub-folds)."
            )

    # ------------------------------------------------------------------
    # Metrics (DRY: F1-macro + accuracy used by every ensemble).
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        num_classes: int = _NUM_CLASSES,
        ignore_index: int | None = 255,
    ) -> dict[str, float]:
        """Compute F1-macro and accuracy from hard labels (DRY).

        Reuses :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` so the
        ensemble metrics are identical to the segmentation harness (apples to
        apples with the individual models). Works for both flat parcel-label
        vectors and flattened pixel maps.

        Args:
            y_true: Ground-truth class ids (any shape; flattened internally).
            y_pred: Predicted class ids (same number of elements as ``y_true``).
            num_classes: Number of classes ``C`` (default 18).
            ignore_index: Label excluded from the confusion and the macro
                (default 255 = harness ignore). ``None`` keeps every label.

        Returns:
            ``{"f1_macro": float, "accuracy": float}`` in ``[0, 1]``.

        Raises:
            ValueError: if ``y_true`` and ``y_pred`` differ in element count.
        """
        from ml.eval.dense_metrics import DenseConfusionAccumulator

        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
        if yt.shape != yp.shape:
            raise ValueError(
                f"y_true ({yt.size}) and y_pred ({yp.size}) must have the same number of elements."
            )
        acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
        acc.update(yp, yt)
        derived = acc.compute()
        return {
            "f1_macro": float(derived["f1_macro"]),
            "accuracy": float(derived["pixel_accuracy"]),
        }

    # ------------------------------------------------------------------
    # Abstract contract (implemented by the four ensembles).
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, *args: object, **kwargs: object) -> EnsembleModel:
        """Fit the ensemble.

        Voting is parameter-free (it only averages OOF probabilities) and may
        return ``self`` without doing work; Bagging fits ``n_bags`` XGBoost
        bootstraps; Stacking fits a meta-learner on OOF sub-folds; Blending
        optimizes simplex weights with Optuna on a disjoint holdout.

        Returns:
            ``self`` for chaining.
        """

    @abc.abstractmethod
    def predict_proba(self, *args: object, **kwargs: object) -> np.ndarray:
        """Return post-softmax probabilities (sum-to-1 over the class axis).

        The output is a probability tensor, NOT logits and NOT hard labels:
        Voting returns dense ``(N, 18, H, W)`` or ``(18, H, W)`` per patch;
        the parcel ensembles return ``(n_parcels, 18)``. Implementations MUST
        guarantee :meth:`validate_probs` passes on their output.

        Returns:
            A ``numpy.ndarray`` of post-softmax probabilities.
        """

    # ------------------------------------------------------------------
    # Evaluation (anti-leakage: fold-5 only).
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        y_true: ParcelLabels,
        y_pred: np.ndarray | None = None,
        proba: np.ndarray | None = None,
        fold: int = HELD_OUT_FOLD,
    ) -> dict[str, float]:
        """Compute F1-macro / accuracy ON fold-5 ONLY (anti-leakage R-LEAK).

        The first thing this method does is reject any ``fold != HELD_OUT_FOLD``
        with ``ValueError`` -- fold-4 was the selection set and reporting on it
        would be a leak. The ground truth is NOT inside the OOF parquet (the dump
        discards the target), so ``y_true`` is supplied by the caller; predicted
        labels come from ``y_pred`` (already argmaxed) or are derived from
        ``proba`` via argmax over the class axis (validated as post-softmax).

        Args:
            y_true: Ground-truth labels. Either a flat ``numpy.ndarray`` /
                sequence of class ids aligned with the predictions, or a
                ``polars.DataFrame`` with ``canonical_parcel_id`` and a ``label``
                column (joined against ``proba``/``y_pred`` order is the caller's
                responsibility -- pass an aligned vector for the parcel case).
            y_pred: Optional hard predicted labels aligned with ``y_true``.
            proba: Optional post-softmax probabilities ``(N, 18)`` (or dense
                ``(..., 18, H, W)``); the argmax over the class axis is used when
                ``y_pred`` is not given. Validated as post-softmax.
            fold: Fold to evaluate. MUST be ``HELD_OUT_FOLD`` (5); any other
                value raises ``ValueError``.

        Returns:
            ``{"f1_macro": float, "accuracy": float}`` computed on fold-5.

        Raises:
            ValueError: if ``fold != HELD_OUT_FOLD``; if neither ``y_pred`` nor
                ``proba`` is given; or if ``proba`` is not post-softmax.
        """
        if fold != self.HELD_OUT_FOLD:
            raise ValueError(
                f"evaluate is fold-{self.HELD_OUT_FOLD}-only (anti-leakage): "
                f"fold={fold} is forbidden. Fold-4 was the SELECTION fold and "
                "must never be reported; only the held-out fold-5 is valid."
            )

        labels = self._labels_array(y_true)

        if y_pred is None and proba is None:
            raise ValueError("evaluate needs either `y_pred` or `proba`.")
        if y_pred is None:
            assert proba is not None  # narrowed by the guard above
            arr = np.asarray(proba, dtype=np.float64)
            class_axis = 1 if arr.ndim >= 2 else -1
            self.validate_probs(arr, class_axis=class_axis, name="proba")
            preds = arr.argmax(axis=class_axis)
        else:
            preds = np.asarray(y_pred)

        metrics = self.compute_metrics(labels, preds)
        logger.info(
            "ensemble_evaluate",
            ensemble=type(self).__name__,
            fold=fold,
            f1_macro=round(metrics["f1_macro"], 4),
            accuracy=round(metrics["accuracy"], 4),
            n=int(labels.size),
        )
        return metrics

    @staticmethod
    def _labels_array(y_true: ParcelLabels) -> np.ndarray:
        """Coerce ``y_true`` (DataFrame ``label`` column or sequence) to a vector."""
        if isinstance(y_true, pl.DataFrame):
            if "label" not in y_true.columns:
                raise ValueError(
                    "y_true DataFrame must carry a `label` column with the "
                    "ground-truth class id per parcel."
                )
            return y_true["label"].to_numpy()
        return np.asarray(y_true).reshape(-1)

    # ------------------------------------------------------------------
    # MLflow logging (DRY: one run per ensemble, mandatory tags).
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        metrics: dict[str, float],
        *,
        run_name: str,
        params: Mapping[str, object] | None = None,
        chosen: bool = False,
        inference_time_s: float | None = None,
        tracking_uri: str | None = None,
        probe_server: bool = True,
    ) -> None:
        """Log one ensemble run to the ``ensemble`` MLflow experiment (:5010).

        Opens a run via :func:`ml.utils.mlflow_utils.track_experiment`, which
        injects the mandatory ``code_version`` (git SHA) and ``data_version``
        (DVC hash of the OOF dir) tags, then logs the metrics, the optional
        params, the inference time and the ``chosen_model`` tag for the elected
        ensemble (rubric Selection criterion).

        Args:
            metrics: Metrics to log (e.g. ``f1_macro``, ``accuracy``); each key
                is suffixed with ``_fold5`` to make the held-out fold explicit.
            run_name: MLflow run name (e.g. ``"e1-voting"``).
            params: Optional hyper-parameters to log (members, ``n_bags``,
                ``n_trials``, weights, ...).
            chosen: If ``True``, tags the run with ``chosen_model=<run_name>``.
            inference_time_s: Optional wall-clock inference time to log.
            tracking_uri: Override of the MLflow tracking URI (default resolves
                to the Docker server on :5010, then the file store).
            probe_server: Forwarded to ``track_experiment``; set ``False`` in
                tests to avoid contacting the server.
        """
        import mlflow

        with track_experiment(
            ENSEMBLE_EXPERIMENT,
            run_name=run_name,
            tracking_uri=tracking_uri,
            dvc_path=_OOF_DVC_PATH,
            probe_server=probe_server,
        ):
            mlflow.set_tag("ensemble", run_name)
            if chosen:
                mlflow.set_tag("chosen_model", run_name)
            if params:
                mlflow.log_params(dict(params))
            for key, value in metrics.items():
                metric_name = key if key.endswith("_fold5") else f"{key}_fold5"
                mlflow.log_metric(metric_name, float(value))
            if inference_time_s is not None:
                mlflow.log_metric("inference_time_s", float(inference_time_s))
            logger.info(
                "ensemble_logged_to_mlflow",
                run_name=run_name,
                chosen=chosen,
                metrics=list(metrics),
            )

    @staticmethod
    def timed_predict(fn: object, *args: object, **kwargs: object) -> tuple[np.ndarray, float]:
        """Run a predict callable and return ``(output, wall_clock_seconds)``.

        Convenience for the comparison table's ``inference_time_s`` column so the
        four ensembles measure inference identically.

        Args:
            fn: Callable returning a ``numpy.ndarray`` (e.g. ``predict_proba``).
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            Tuple ``(output, elapsed_seconds)``.
        """
        if not callable(fn):
            raise TypeError("fn must be callable.")
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, time.perf_counter() - start


if TYPE_CHECKING:  # pragma: no cover - type aliases referencing optional deps
    from ml.features.spatial_split import FoldAssignment

    ParcelLabels = pl.DataFrame | np.ndarray | Sequence[int]
else:  # runtime fallbacks so annotations evaluate lazily without hard imports
    ParcelLabels = object
    FoldAssignment = object
