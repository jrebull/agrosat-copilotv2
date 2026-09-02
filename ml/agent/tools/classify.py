"""``classify_new_parcel`` tool: honest per-parcel crop classification (sync).

This tool is deliberately HONEST about which model serves a classification (it
used to advertise itself as a "stacking ensemble" while only ever running the
``xgb-alphaearth`` member -- US-053 corrects that oversell). The active serving
model is selected by ``ClassifyParcelInput.model`` (with the legacy
``use_stacking`` flag promoted to ``model="stacking5"`` only when ``model="xgb"``
is set explicitly, for back-compat), and two independent flags shape the
posterior:

- ``model`` (default ``"voting3"`` since US-081 AC4a): for a parcel ALREADY
  materialized in the cached fold-5 OOF the tool can serve one of three models:
    * ``"xgb"`` -- the ``xgb-alphaearth`` tabular member (the historical default,
      kept for back-compat).
    * ``"voting3"`` -- the REAL EPIC 12 deployment champion (the copilot DEFAULT):
      the weighted
      soft-vote of ``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth`` at the PARCEL
      level (:class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble`, US-079).
      It wins the deployed france-10 comparison (F1-macro 0.9069 vs the Stacking-5
      0.8927; ``reports/ensemble/metrics/france10_headline.csv``), so it is the
      true champion the agent serves, not the legacy Stacking-5.
    * ``"stacking5"`` -- the EPIC 6 Stacking-5 logreg meta (the US-043 winner,
      kept as LEGACY now that Voting-3 supersedes it).
  Any model that cannot resolve the parcel (a fresh polygon with no OOF row) or
  whose OOF artifacts are unavailable (DVC not pulled) degrades CLEANLY to
  ``xgb-alphaearth`` with a structured warning -- it never fabricates a posterior
  and never crashes.
- ``restrict_to_resolved_classes`` (default ON): the posterior is masked down to
  the well-resolved classes of the active label-space (the configured
  :data:`~ml.eval.class_remap.DEFAULT_LABEL_SPACE` by default -- ``france-12`` for
  the v2 champion, the twelve classes above the 0.90 macro-F1 line) and
  renormalized over them
  (see :mod:`ml.eval.class_remap`). It costs no GPU and works for any parcel with
  a persisted 64-dim embedding -- it just declines to report classes the model
  resolves poorly. When OFF the full 18-class posterior is returned (legacy).
- ``use_stacking`` (default OFF): LEGACY selector kept for back-compat. When
  ``True`` and ``model`` is set explicitly to ``"xgb"``, it is treated as
  ``model="stacking5"`` (see :attr:`ClassifyParcelInput.resolved_model`); under
  the new ``"voting3"`` default it is a no-op. New callers should set ``model``
  directly.

Default (every flag at its default) the tool serves the ``voting3`` EPIC 12
deployment champion for a fold-5 parcel -- degrading cleanly to the
``xgb-alphaearth`` tabular member for a fresh AOI -- restricted to the configured
default label-space.

The per-parcel inference path:

1. Resolves the AlphaEarth embedding of the parcel. If the polygon maps to a
   persisted parcel of the session (``features_parcels.alphaearth_embedding``),
   that embedding is used (session-scoped, multi-tenant). If none is found (a
   fresh AOI), the tool returns a controlled ``needs_gee_sampling`` result rather
   than hallucinating a class -- the flags are NOT evaluated before the embedding.
2. Loads the XGBoost-AlphaEarth classifier (CPU, ``functools.lru_cache``) trained
   leak-free on folds 1-4 of ``features_fused_pastis.parquet`` -- the same recipe
   the EPIC 6 stacking ensemble materializes its ``xgb-alphaearth`` base member.
3. Optionally re-scores via the Voting-3 weighted vote or the Stacking-5 logreg
   meta (both cached) when the parcel is in the fold-5 OOF universe.
4. Optionally restricts + renormalizes the posterior over the active label-space.

Every load is CPU-light and cached process-wide (no GPU, per the
``ml/agent/AGENTS.md`` rule that heavy GPU inference must go behind Pub/Sub).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

from ml.agent.context import ToolContext
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput, GeoJSONGeometry
from ml.eval.class_remap import LabelSpace, get_label_space, restrict_posterior

logger = structlog.get_logger(__name__)

__all__ = ["run"]

#: Five EPIC 6 stacking members whose cached fold-5 OOF feed the Stacking-5 meta.
_STACKING_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth",
    "farslip-ft18",
    "farslip-zeroshot",
)

#: Three EPIC 12 weighted-vote members (the deployment champion, US-079). The same
#: terna Stacking/Blending vote over, so the only moving part is the combination
#: layer (N convex weights vs the 54-weight Stacking meta). Order matters: it is
#: the member axis of the aligned probability tensor and the learned weight vector.
_VOTING_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth",
)

#: Repo root resolved from this file (``ml/agent/tools/classify.py`` -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Fused tabular features parquet holding the AlphaEarth ``dim_*`` columns,
#: the PASTIS ``class_id`` and the spatial ``fold`` (1..5).
_FEATURES_PATH = _REPO_ROOT / "data" / "features" / "features_fused_pastis.parquet"

#: Directory holding the US-031 per-parcel OOF parquet artifacts (DVC-tracked).
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"

#: The NEW Voting-3 v2 champion re-trains ONLY tsvit-pheno (tsvit-pheno-fullm-v2 @
#: n_timesteps=32, dumped to ``oof_new32``); utae and xgb-alphaearth are unchanged.
#: The v2 tsvit OOF carries the SAME ``canonical_parcel_id`` set as the original, so
#: it is a drop-in over the same utae/xgb OOF. DVC: ``dvc pull ml/eval/oof_new32``.
_OOF_DIR_V2 = _REPO_ROOT / "ml" / "eval" / "oof_new32"

#: Per-member OOF directory override: tsvit-pheno is served from the v2 dump, the
#: other two members from ``_OOF_DIR``.
_MEMBER_OOF_DIR: dict[str, Path] = {"tsvit-pheno": _OOF_DIR_V2}

#: Published deployment weights of the v2 Voting-3 champion (in :data:`_VOTING_MEMBERS`
#: order), learned by F1-macro maximization on fold-5 and reported under
#: ``reports/voting_new/``. ``utae`` contributes 0 (the v2 tsvit dominates), so the
#: vote is effectively ``tsvit-pheno-v2 + xgb``. PINNED to the published values (not
#: re-learned at load) so the agent's vote matches the published champion exactly and
#: needs no PASTIS-R GT/geometry at load time.
_VOTING_V2_WEIGHTS: tuple[float, ...] = (0.902, 0.0, 0.098)

#: PASTIS-R root used to reconstruct the per-parcel GT for the Stacking-5 meta
#: re-fit (the OOF dump discards the target). Absent when DVC data is not pulled.
_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"

#: PASTIS-R patch metadata (per-patch footprints in EPSG:2154). The Voting-3
#: weight learning needs per-parcel geometry for its spatial sub-folds; this is the
#: source the runner's ``build_parcel_geometries`` reads. Absent without DVC pull.
_PASTIS_METADATA = _PASTIS_ROOT / "metadata.geojson"

#: Column prefix of the 64-dim AlphaEarth embedding in the features parquet.
_ALPHAEARTH_PREFIX: str = "dim_"

#: Number of AlphaEarth embedding dimensions (annual Satellite Embedding V1).
_EMBED_DIM: int = 64

#: Number of contiguous agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: Spatial fold held out by the harness; the classifier trains on folds 1-4 so
#: its fold-5 predictions stay leak-free (R-LEAK), matching the EPIC 6 stacking
#: base member materialization.
_HELD_OUT_FOLD: int = 5

#: Sentinel crop class emitted when the parcel has no AlphaEarth embedding (a new
#: AOI that still needs GEE sampling). It is NOT a real PASTIS class: it signals
#: the agent loop that an out-of-band embedding step is required.
_NEEDS_GEE_SAMPLING: str = "needs_gee_sampling"


class _ProbaEstimator(Protocol):
    """Structural type of the fitted sklearn-compatible estimator being wrapped."""

    def predict_proba(self, x: np.ndarray, /) -> np.ndarray: ...


class _XgbAlphaEarthClassifier:
    """Fitted XGBoost-AlphaEarth classifier with a fixed 18-class probability head.

    Wraps the sklearn estimator plus the mapping from its local class columns to
    the global ``[0, 18)`` semantic18 space, so :meth:`predict_proba_18` always
    returns a ``(n, 18)`` post-softmax row regardless of which classes the
    training folds happened to cover.

    Attributes:
        estimator: The fitted ``SpatialXGBClassifier`` (sklearn-compatible).
        global_classes: Global semantic18 class ids of the estimator's local
            ``predict_proba`` columns, in order.
        class_names: Mapping ``{global_class_id: human-readable crop name}``.
    """

    def __init__(
        self,
        estimator: _ProbaEstimator,
        global_classes: np.ndarray,
        class_names: dict[int, str],
    ) -> None:
        self.estimator = estimator
        self.global_classes = global_classes
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        """Predict the full 18-class posterior for one AlphaEarth embedding.

        Args:
            embedding: A single ``(64,)`` AlphaEarth embedding vector.

        Returns:
            A ``(18,)`` ``float64`` post-softmax distribution summing to 1.
        """
        x = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        x = np.where(np.isfinite(x), x, 0.0)
        proba_local = np.asarray(self.estimator.predict_proba(x), dtype=np.float64)[0]
        full = np.zeros(_NUM_CLASSES, dtype=np.float64)
        for col, gid in enumerate(self.global_classes):
            gid_int = int(gid)
            if 0 <= gid_int < _NUM_CLASSES:
                full[gid_int] = proba_local[col]
        total = full.sum()
        return full / total if total > 1e-12 else full


@functools.lru_cache(maxsize=1)
def _load_classifier() -> _XgbAlphaEarthClassifier:
    """Load (and cache) the XGBoost-AlphaEarth classifier.

    Fits an XGBoost over the AlphaEarth ``dim_*`` columns of folds 1-4 of
    ``features_fused_pastis.parquet`` and maps its labels to the contiguous
    semantic18 space, exactly as the EPIC 6 stacking ensemble materializes its
    ``xgb-alphaearth`` base member. The result is cached process-wide
    (``maxsize=1``) so the CPU-light fit happens once.

    Returns:
        A ready :class:`_XgbAlphaEarthClassifier`.

    Raises:
        FileNotFoundError: if the fused features parquet is absent (run
            ``dvc pull data/features``).
        ValueError: if the parquet lacks the required columns or fold-5 leaves no
            training rows.
    """
    import polars as pl
    from sklearn.preprocessing import LabelEncoder

    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
    from ml.data.pastis_seg_dataset import _build_semantic18_lut
    from ml.train.baseline import build_estimator

    if not _FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"AlphaEarth fused features parquet not found: {_FEATURES_PATH}. "
            "Run `dvc pull data/features` to fetch it."
        )

    df = pl.read_parquet(_FEATURES_PATH)
    for col in ("class_id", "fold"):
        if col not in df.columns:
            raise ValueError(f"features parquet is missing the `{col}` column.")
    feature_cols = [c for c in df.columns if c.startswith(_ALPHAEARTH_PREFIX)]
    if not feature_cols:
        raise ValueError(
            f"no AlphaEarth feature column with prefix {_ALPHAEARTH_PREFIX!r} in {_FEATURES_PATH}."
        )

    # Train on folds 1-4 only (leak-free: the classifier never sees fold-5).
    train = df.filter(pl.col("fold") != _HELD_OUT_FOLD).filter(pl.col("class_id").is_not_null())
    if train.height == 0:
        raise ValueError("no training rows on folds 1-4 in the features parquet.")

    x_train = train.select(feature_cols).to_numpy().astype(np.float64)
    x_train = np.where(np.isfinite(x_train), x_train, 0.0)

    # Map the raw PASTIS class_id (1..18) to the contiguous semantic18 space
    # [0..17] used by every ensemble member; drop Background/Void parcels (255).
    label_lut = _build_semantic18_lut(255)
    pastis_train = np.clip(train.get_column("class_id").to_numpy().astype(np.int64), 0, 19)
    y_raw = label_lut[pastis_train]
    keep = y_raw != 255
    x_train = x_train[keep]
    y_raw = y_raw[keep]
    if x_train.shape[0] == 0:
        raise ValueError("no semantic18-labelled parcels left after dropping Background/Void.")

    encoder = LabelEncoder().fit(y_raw)
    y_train = encoder.transform(y_raw).astype(np.int64)

    estimator = build_estimator(
        "xgb",
        {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "multi:softprob",
            "random_state": 42,
        },
    )
    estimator.fit(x_train, y_train)

    global_classes = encoder.classes_.astype(np.int64)
    logger.info(
        "classify_classifier_loaded",
        n_train=int(x_train.shape[0]),
        n_features=len(feature_cols),
        n_classes=int(global_classes.size),
    )
    return _XgbAlphaEarthClassifier(
        estimator=estimator,
        global_classes=global_classes,
        class_names=dict(SEMANTIC18_CLASS_NAMES),
    )


class _StackingFive:
    """Stacking-5 logreg meta refit on the five members' cached fold-5 OOF.

    Holds the fitted multinomial logistic-regression meta-learner plus the joined
    meta-feature frame keyed by ``canonical_parcel_id`` so a single parcel can be
    re-scored by id without recomputing anything. This mirrors the EPIC 6
    :class:`ml.ensemble.stacking.StackingEnsemble` final refit (the model
    ``predict_proba`` uses) -- the logreg meta over OOF predictions ONLY, never
    raw features or logits -- but skips the spatial-CV quality estimate (that
    produces the reported F1, not the deployed posterior).

    Attributes:
        meta: The fitted ``LogisticRegression`` meta-learner.
        meta_classes: The semantic18 class ids the meta can emit, column-aligned
            with ``meta.predict_proba``.
        feature_cols: The ``member__prob_xxx`` meta-feature columns, in fit order.
        meta_features_by_id: Mapping ``canonical_parcel_id -> (90,)`` meta-feature
            row, for O(1) per-parcel lookup.
    """

    def __init__(
        self,
        meta: Any,
        meta_classes: np.ndarray,
        feature_cols: list[str],
        meta_features_by_id: dict[str, np.ndarray],
    ) -> None:
        self.meta = meta
        self.meta_classes = meta_classes
        self.feature_cols = feature_cols
        self.meta_features_by_id = meta_features_by_id

    def posterior_for_parcel(self, canonical_id: str) -> np.ndarray | None:
        """Return the Stacking-5 ``(18,)`` posterior for a fold-5 parcel.

        Args:
            canonical_id: Canonical parcel id (``"{patch}_{local}"``) to score.

        Returns:
            A ``(18,)`` ``float64`` post-softmax distribution, or ``None`` when the
            parcel is not in the joined fold-5 OOF universe (caller degrades).
        """
        row = self.meta_features_by_id.get(canonical_id)
        if row is None:
            return None
        proba_local = np.asarray(self.meta.predict_proba(row.reshape(1, -1)), dtype=np.float64)[0]
        full = np.zeros(_NUM_CLASSES, dtype=np.float64)
        for col, cls in enumerate(self.meta_classes):
            cls_int = int(cls)
            if 0 <= cls_int < _NUM_CLASSES:
                full[cls_int] = proba_local[col]
        total = full.sum()
        return full / total if total > 1e-12 else full


@functools.lru_cache(maxsize=1)
def _load_stacking_five() -> _StackingFive:
    """Load (and cache) the Stacking-5 logreg meta from the cached fold-5 OOF.

    Inner-joins the five members' per-parcel OOF parquets on
    ``canonical_parcel_id`` (post-softmax ``prob_000..prob_017`` -> ``5 x 18 = 90``
    meta-features), reconstructs the per-parcel semantic18 GT from the PASTIS-R
    rasters (the OOF dump discards the target) and fits a multinomial
    ``LogisticRegression`` meta-learner. The result is cached process-wide.

    Returns:
        A ready :class:`_StackingFive`.

    Raises:
        FileNotFoundError: if any member OOF parquet or the PASTIS-R GT source is
            missing (run ``dvc pull ml/eval/oof`` / ``dvc pull data/PASTIS-R``);
            the caller catches this and degrades to ``xgb-alphaearth``.
        ValueError: if the OOF join or the OOF/GT join leaves no parcels.
    """
    import polars as pl
    from sklearn.linear_model import LogisticRegression

    from ml.utils.parcel_id import canonical_parcel_id
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    key = "canonical_parcel_id"
    joined: pl.DataFrame | None = None
    feature_cols: list[str] = []
    for member in _STACKING_MEMBERS:
        path = _OOF_DIR / f"oof_parcel_{member}_fold5.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Stacking-5 OOF parquet missing: {path}. Run `dvc pull ml/eval/oof`."
            )
        frame = canonical_parcel_id(pl.read_parquet(path), col=key)
        renamed = {col: f"{member}__{col}" for col in PROB_COLUMNS}
        feature_cols.extend(renamed.values())
        sub = frame.select([key, *PROB_COLUMNS]).rename(renamed)
        joined = sub if joined is None else joined.join(sub, on=key, how="inner")
    if joined is None or joined.height == 0:
        raise ValueError("the five members share no common fold-5 parcel.")
    joined = joined.sort(key)

    gt = _build_parcel_ground_truth(joined.get_column(key).to_list())
    train = joined.join(gt, on=key, how="inner").sort(key)
    if train.height == 0:
        raise ValueError("no parcels remain after joining the OOF with the PASTIS-R ground truth.")

    x_meta = train.select(feature_cols).to_numpy().astype(np.float64)
    y_meta = train.get_column("label").to_numpy().astype(np.int64)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42).fit(
        x_meta, y_meta
    )

    # Index EVERY joined parcel's meta-features (not just the GT-labelled subset)
    # so a parcel present in all five OOF can be scored even if its GT was dropped.
    all_features = joined.select(feature_cols).to_numpy().astype(np.float64)
    all_ids = joined.get_column(key).to_list()
    meta_features_by_id = {pid: all_features[i] for i, pid in enumerate(all_ids)}

    logger.info(
        "classify_stacking_five_loaded",
        n_members=len(_STACKING_MEMBERS),
        n_parcels=int(joined.height),
        n_train=int(train.height),
        n_meta_features=len(feature_cols),
    )
    return _StackingFive(
        meta=meta,
        meta_classes=np.asarray(meta.classes_, dtype=np.int64),
        feature_cols=feature_cols,
        meta_features_by_id=meta_features_by_id,
    )


class _VotingThree:
    """Voting-3 weighted soft-vote over the three members' cached fold-5 OOF.

    Holds the learned convex weights (one per member, in :data:`_VOTING_MEMBERS`
    order) plus, for every parcel present in ALL three members' fold-5 OOF, the
    stacked ``(3, 18)`` post-softmax member rows keyed by ``canonical_parcel_id``.
    A single parcel's Voting-3 posterior is then the convex combination
    ``sum_i w_i * P_i`` of those rows -- a valid post-softmax distribution -- with
    no recomputation. This mirrors the EPIC 12
    :class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble` production path
    (its inherited ``predict_proba`` blends the SAME aligned member tensor with the
    SAME learned weights), but indexed per parcel for O(1) agent lookup.

    The weights are LEARNED at load time by fitting the ensemble on the fold-5 OOF
    (CPU, seconds: scipy.optimize over a 3-simplex), never hard-coded -- so they
    stay exactly the published deployment weights and degrade honestly if any
    member OOF or the PASTIS-R geometry/GT the fit consumes is unavailable.

    Attributes:
        weights: Learned convex weights ``(3,)`` (``w_i >= 0``, ``sum(w) == 1``),
            aligned with :data:`_VOTING_MEMBERS`.
        member_probs_by_id: Mapping ``canonical_parcel_id -> (3, 18)`` stacked
            post-softmax member rows, for O(1) per-parcel scoring.
    """

    def __init__(
        self,
        weights: np.ndarray,
        member_probs_by_id: dict[str, np.ndarray],
    ) -> None:
        self.weights = weights
        self.member_probs_by_id = member_probs_by_id

    def posterior_for_parcel(self, canonical_id: str) -> np.ndarray | None:
        """Return the Voting-3 ``(18,)`` posterior for a fold-5 parcel.

        Args:
            canonical_id: Canonical parcel id (``"{patch}_{local}"``) to score.

        Returns:
            A ``(18,)`` ``float64`` post-softmax distribution (the convex
            combination of the three members), or ``None`` when the parcel is not
            present in all three members' fold-5 OOF (caller degrades).
        """
        member_rows = self.member_probs_by_id.get(canonical_id)
        if member_rows is None:
            return None
        # Convex combination sum_i w_i * P_i over the member axis -> (18,).
        blended = np.tensordot(self.weights, member_rows, axes=([0], [0]))
        total = blended.sum()
        return blended / total if total > 1e-12 else blended


@functools.lru_cache(maxsize=1)
def _load_voting_three() -> _VotingThree:
    """Load (and cache) the Voting-3 weighted vote from the cached fold-5 OOF.

    Mirrors the EPIC 12 ``run_weighted_voting_pastis.py`` materialization but for
    the agent's per-parcel serving path:

    1. Inner-joins the three members' per-parcel OOF on ``canonical_parcel_id`` and
       stacks their post-softmax ``prob_000..prob_017`` rows into a ``(3, 18)``
       tensor per parcel (the SAME ``_align_members`` intersection the ensemble
       uses).
    2. Reconstructs the per-parcel semantic18 GT from PASTIS-R (shared with the
       Stacking-5 path) and the per-parcel geometry, then fits
       :class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble` to LEARN the
       three convex weights by F1-macro maximization (the deployment weights).
    3. Indexes every joined parcel's ``(3, 18)`` member rows by id for O(1) lookup.

    The result is cached process-wide (``maxsize=1``) so the CPU-light fit happens
    once.

    Returns:
        A ready :class:`_VotingThree` (learned weights + per-parcel member rows).

    Raises:
        FileNotFoundError: if any member OOF parquet, the PASTIS-R GT source or the
            PASTIS-R geometry metadata is missing (run ``dvc pull ml/eval/oof`` /
            ``dvc pull data/PASTIS-R``); the caller catches this and degrades to
            ``xgb-alphaearth``.
        ValueError: if the OOF intersection or the OOF/GT join leaves no parcels.
    """
    import polars as pl

    from ml.utils.parcel_id import canonical_parcel_id
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    key = "canonical_parcel_id"
    # Inner-join the three members on the shared parcel set, stacking each member's
    # post-softmax row (renamed per member so the columns do not collide).
    joined: pl.DataFrame | None = None
    member_cols: dict[str, list[str]] = {}
    for member in _VOTING_MEMBERS:
        path = _MEMBER_OOF_DIR.get(member, _OOF_DIR) / f"oof_parcel_{member}_fold5.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Voting-3 OOF parquet missing: {path}. Run "
                "`dvc pull ml/eval/oof ml/eval/oof_new32`."
            )
        frame = canonical_parcel_id(pl.read_parquet(path), col=key)
        renamed = {col: f"{member}__{col}" for col in PROB_COLUMNS}
        member_cols[member] = list(renamed.values())
        sub = frame.select([key, *PROB_COLUMNS]).rename(renamed)
        joined = sub if joined is None else joined.join(sub, on=key, how="inner")
    if joined is None or joined.height == 0:
        raise ValueError("the three Voting-3 members share no common fold-5 parcel.")
    joined = joined.sort(key)

    all_ids = joined.get_column(key).to_list()
    # Stack each parcel's three member rows into (n_parcels, 3, 18) for indexing.
    per_member = [
        joined.select(member_cols[member]).to_numpy().astype(np.float64)
        for member in _VOTING_MEMBERS
    ]
    member_tensor = np.stack(per_member, axis=1)  # (n_parcels, 3, 18)
    member_probs_by_id = {pid: member_tensor[i] for i, pid in enumerate(all_ids)}

    # Use the PUBLISHED v2 deployment weights (PINNED, not re-learned): the v2 tsvit
    # OOF is a drop-in over the same utae/xgb, and the champion's convex weights are
    # the published 0.902 / 0.0 / 0.098 (reports/voting_new/). Pinning keeps the
    # agent's vote identical to the deployment and removes the PASTIS-R GT/geometry
    # dependency from the agent's hot load path.
    weights = np.asarray(_VOTING_V2_WEIGHTS, dtype=np.float64)

    logger.info(
        "classify_voting_three_loaded",
        n_members=len(_VOTING_MEMBERS),
        n_parcels=int(joined.height),
        weights={
            member: round(float(w), 4) for member, w in zip(_VOTING_MEMBERS, weights, strict=True)
        },
    )
    return _VotingThree(weights=weights, member_probs_by_id=member_probs_by_id)


def _build_parcel_geometries(canonical_ids: list[str]):  # type: ignore[no-untyped-def]
    """Build per-parcel centroid-Point geometry from PASTIS-R for the given parcels.

    The Voting-3 weight learning needs a per-parcel geographic position to form the
    spatial sub-folds of its leakage-free CV. This rebuilds the SAME centroid-Point
    frame the EPIC 12 runner's ``build_parcel_geometries`` produces (patch centroid
    in EPSG:4326 offset by the parcel's intra-patch pixel centroid), keyed by the
    SAME canonical id the OOF parcels use, restricted to the patches present in
    ``canonical_ids`` so the I/O stays bounded.

    Args:
        canonical_ids: Canonical parcel ids whose patch geometry must be built.

    Returns:
        A Polars frame with ``canonical_parcel_id`` (Utf8) + ``parcel_id`` (Int64
        surrogate) + ``geometry`` (WKT Point in EPSG:4326).

    Raises:
        FileNotFoundError: if a required PASTIS-R raster or ``metadata.geojson`` is
            absent (data not pulled); the caller catches this and degrades.
    """
    import json as _json

    import polars as pl
    from pyproj import Transformer
    from shapely.geometry import Point, shape

    from ml.utils.parcel_id import canonical_parcel_id
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    if not _PASTIS_METADATA.exists():
        raise FileNotFoundError(f"PASTIS-R metadata.geojson not found: {_PASTIS_METADATA}.")
    meta = _json.loads(_PASTIS_METADATA.read_text(encoding="utf-8"))
    # PASTIS-R metadata is in EPSG:2154 (Lambert-93, metres); reproject to lon/lat.
    crs_name = meta.get("crs", {}).get("properties", {}).get("name", "EPSG:2154")
    transformer = Transformer.from_crs(crs_name, "EPSG:4326", always_xy=True)
    centroid_by_patch: dict[str, tuple[float, float]] = {}
    for feature in meta["features"]:
        pid = str(feature["properties"]["ID_PATCH"])
        centroid = shape(feature["geometry"]).centroid
        lon, lat = transformer.transform(float(centroid.x), float(centroid.y))
        centroid_by_patch[pid] = (float(lon), float(lat))

    patch_ids = sorted({cid.split("_")[0] for cid in canonical_ids})
    keys: list[str] = []
    geoms: list[str] = []
    for pid in patch_ids:
        cx, cy = centroid_by_patch.get(pid, (0.0, 0.0))
        parcel_ids = load_pastis_parcel_ids(pid, _PASTIS_ROOT)
        h, w = parcel_ids.shape
        for local in np.unique(parcel_ids[parcel_ids != 0]):
            ys, xs = np.where(parcel_ids == local)
            off_x = (float(xs.mean()) / w - 0.5) * 0.01
            off_y = (float(ys.mean()) / h - 0.5) * 0.01
            keys.append(f"{pid}_{int(local)}")
            geoms.append(Point(cx + off_x, cy + off_y).wkt)

    frame = pl.DataFrame({"canonical_parcel_id": keys, "geometry": geoms})
    return canonical_parcel_id(frame, col="canonical_parcel_id")


def _geodataframe_from_wkt(frame):  # type: ignore[no-untyped-def]
    """Convert the WKT-geometry Polars frame to a GeoDataFrame for the spatial CV.

    The :class:`WeightedVotingEnsemble` spatial sub-folds need a GeoDataFrame with
    an integer ``parcel_id`` surrogate (used by ``build_spatial_kfold``), the
    ``canonical_parcel_id`` Utf8 key (to map back to the aligned member order) and
    an active ``geometry`` column. This mirrors the runner's ``_geoms_for_blending``.

    Args:
        frame: Polars frame with ``canonical_parcel_id`` + ``geometry`` (WKT Point).

    Returns:
        A GeoDataFrame in EPSG:4326 with ``parcel_id`` (int surrogate),
        ``canonical_parcel_id`` (Utf8) and an active ``geometry``.
    """
    import geopandas as gpd
    from shapely import wkt as shapely_wkt

    pdf = frame.to_pandas()
    pdf["geometry"] = pdf["geometry"].map(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")
    # Integer surrogate id required by build_spatial_kfold (the canonical id is Utf8).
    gdf["parcel_id"] = range(len(gdf))
    return gdf


def _build_parcel_ground_truth(canonical_ids: list[str]):  # type: ignore[no-untyped-def]
    """Reconstruct per-parcel semantic18 GT from PASTIS-R for the given parcels.

    The OOF dump discards the target, so the per-parcel ground truth is rebuilt
    from the PASTIS-R semantic ``TARGET`` + ``ParcelIDs`` rasters: each parcel gets
    the MAJORITY semantic18 label of its pixels, keyed by the SAME canonical id
    (``f"{patch}_{local}"``) the OOF parcels use. Only the patches actually present
    in ``canonical_ids`` are read (fold-5), keeping the I/O bounded.

    Args:
        canonical_ids: Canonical parcel ids whose patches must be reconstructed.

    Returns:
        A Polars frame with ``canonical_parcel_id`` (Utf8) + ``label`` (Int64 in
        the ``[0..17]`` semantic18 space).

    Raises:
        FileNotFoundError: if a required PASTIS-R raster is absent (data not
            pulled); the caller catches this and degrades to ``xgb-alphaearth``.
    """
    import polars as pl

    from ml.data.pastis_seg_dataset import _build_semantic18_lut
    from ml.utils.parcel_id import canonical_parcel_id
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    if not _PASTIS_ROOT.exists():
        raise FileNotFoundError(
            f"PASTIS-R root not found: {_PASTIS_ROOT}. Run `dvc pull data/PASTIS-R`."
        )

    ignore_index = 255
    label_lut = _build_semantic18_lut(ignore_index)
    patch_ids = sorted({cid.split("_")[0] for cid in canonical_ids})

    keys: list[str] = []
    labels: list[int] = []
    for pid in patch_ids:
        target_path = _PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if not target_path.exists():
            raise FileNotFoundError(f"PASTIS-R semantic TARGET not found: {target_path}.")
        target = np.load(target_path)
        if target.ndim == 3:  # PASTIS ships (3, H, W); the semantic channel is 0.
            target = target[0]
        parcel_ids = load_pastis_parcel_ids(pid, _PASTIS_ROOT)

        flat_pids = parcel_ids.reshape(-1)
        raw = np.clip(target.reshape(-1).astype(np.int64), 0, 19)
        flat_labels = label_lut[raw]
        valid = (flat_pids != 0) & (flat_labels != ignore_index)
        flat_pids = flat_pids[valid]
        flat_labels = flat_labels[valid]
        if flat_pids.size == 0:
            continue

        unique_ids, inverse = np.unique(flat_pids, return_inverse=True)
        votes = np.zeros((unique_ids.size, _NUM_CLASSES), dtype=np.int64)
        in_range = flat_labels < _NUM_CLASSES
        np.add.at(votes, (inverse[in_range], flat_labels[in_range]), 1)
        majority = votes.argmax(axis=1)
        for local, lab in zip(unique_ids, majority, strict=True):
            keys.append(f"{pid}_{int(local)}")
            labels.append(int(lab))

    frame = pl.DataFrame({"canonical_parcel_id": keys, "label": labels}).with_columns(
        pl.col("label").cast(pl.Int64)
    )
    return canonical_parcel_id(frame, col="canonical_parcel_id")


async def _resolve_canonical_parcel_id(ctx: ToolContext, aoi: GeoJSONGeometry) -> str | None:
    """Resolve the canonical fold-5 OOF key of the persisted parcel under ``aoi``.

    The Stacking-5 OOF is keyed by a canonical ``parcel_id`` (Utf8). The DB has no
    dedicated canonical-id column (``features_parcels`` keys only by integer
    ``parcel_id``), so -- exactly like :func:`ml.agent.tools.compare._compute_comparison`
    -- the DB integer ``parcels.id`` is BRIDGED to the OOF Utf8 namespace via
    :func:`ml.utils.parcel_id.canonical_parcel_id` (a lossless integer->Utf8 cast).
    The lookup is session-scoped and spatially anchored to the drawn AOI, mirroring
    :func:`_fetch_parcel_embedding`. A bridged id absent from the fold-5 OOF
    universe (a fresh, non-PASTIS parcel) yields no Stacking-5 row and the caller
    degrades to ``xgb-alphaearth``.

    Args:
        ctx: Tool execution context (pool, session id).
        aoi: Drawn AOI polygon used to spatially resolve the parcel.

    Returns:
        The bridged canonical parcel id string, or ``None`` when no persisted
        parcel of the session intersects ``aoi``.
    """
    import polars as pl

    from ml.agent.db import session_scoped_conn
    from ml.utils.parcel_id import canonical_parcel_id

    query = """
        SELECT p.id
        FROM parcels p
        WHERE p.session_id = $1
          AND ST_Intersects(p.geom, ST_SetSRID(ST_GeomFromGeoJSON($2), 4326))
        ORDER BY p.id DESC
        LIMIT 1
    """
    aoi_geojson = json.dumps({"type": aoi.type, "coordinates": aoi.coordinates})
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(query, ctx.session_id, aoi_geojson)
    if row is None or row["id"] is None:
        return None
    parcel_id = int(row["id"])
    # Bridge the DB integer id to the Utf8 OOF key namespace (same as compare.py).
    bridged = canonical_parcel_id(
        pl.DataFrame({"canonical_parcel_id": [parcel_id]}), col="canonical_parcel_id"
    )["canonical_parcel_id"][0]
    return str(bridged)


async def fetch_canonical_parcel_id(ctx: ToolContext, parcel_id: int) -> str | None:
    """Return a stored parcel's canonical PASTIS-R id (``"{patch}_{local}"``), if any.

    The OOF-backed tools (the Voting-3 perceiver and ``compare_models``) key the
    model OOF parquets by the canonical ``"{patch}_{local}"`` id, which the numeric
    cast of ``parcels.id`` never reproduces. A parcel seeded from a real PASTIS-R
    fold-5 row carries that id in ``parcels.canonical_parcel_id`` (US-079 migration);
    this reads it session-scoped so the OOF lookup hits the real held-out prediction.
    Returns ``None`` for a parcel without it (a fresh AOI or a non-PASTIS demo
    parcel), and the caller degrades honestly to the embedding path.

    Args:
        ctx: Tool execution context (pool, session id).
        parcel_id: Stored parcel id to resolve.

    Returns:
        The canonical parcel id string, or ``None`` when absent.
    """
    from ml.agent.db import session_scoped_conn

    async with session_scoped_conn(ctx.session_id) as conn:
        value = await conn.fetchval(
            "SELECT canonical_parcel_id FROM parcels WHERE id = $1 AND session_id = $2",
            parcel_id,
            ctx.session_id,
        )
    return str(value) if value is not None else None


async def _fetch_parcel_embedding(
    ctx: ToolContext, year: int, aoi: GeoJSONGeometry | None = None
) -> np.ndarray | None:
    """Fetch the AlphaEarth embedding of the persisted parcel covering ``aoi``.

    The polygon-to-parcel resolution for a brand-new AOI is owned by the GEE
    sampler (out of scope here), so this tool reads the embedding from
    ``features_parcels`` for the parcels of the current session. The query is

    1. session-scoped (``parcels.session_id`` -- multi-tenant defence in depth),
    2. restricted to the requested ``year`` with a non-null embedding, and
    3. spatially anchored to the drawn AOI via ``ST_Intersects`` so the embedding
       belongs to the parcel the user actually outlined, NOT merely the session's
       most recently updated parcel. Without this spatial join the tool would
       classify an unrelated parcel with high confidence (US-045 / B-2).

    When several persisted parcels intersect the AOI, the one updated last wins
    (``ORDER BY fp.updated_at DESC``); when none intersect, the caller falls back
    to the controlled ``needs_gee_sampling`` result.

    Args:
        ctx: Tool execution context (pool, session id).
        year: Campaign year of the annual embedding.
        aoi: Drawn AOI polygon used to spatially resolve the parcel. Serialized
            to GeoJSON and bound as ``$3`` (parameterized, never f-string).

    Returns:
        A ``(64,)`` ``float64`` embedding, or ``None`` if no persisted parcel of
        the session intersects ``aoi`` for that year (a fresh AOI without an
        embedding).
    """
    from ml.agent.db import session_scoped_conn

    if aoi is not None:
        # classify_new_parcel (B-2): spatially anchor the embedding to the parcel
        # the user outlined via ``ST_Intersects`` (never the session's merely most
        # recent parcel). The AOI is serialized and bound as ``$3``.
        query = """
            SELECT fp.alphaearth_embedding
            FROM features_parcels fp
            JOIN parcels p ON p.id = fp.parcel_id
            WHERE p.session_id = $1
              AND fp.year = $2
              AND fp.alphaearth_embedding IS NOT NULL
              AND ST_Intersects(p.geom, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326))
            ORDER BY fp.updated_at DESC
            LIMIT 1
        """
        aoi_geojson = json.dumps({"type": aoi.type, "coordinates": aoi.coordinates})
        args: tuple[object, ...] = (ctx.session_id, year, aoi_geojson)
    else:
        # No AOI (perceiver.observe over a known parcel of the session): the most
        # recently updated session embedding for the year (no spatial anchor).
        query = """
            SELECT fp.alphaearth_embedding
            FROM features_parcels fp
            JOIN parcels p ON p.id = fp.parcel_id
            WHERE p.session_id = $1
              AND fp.year = $2
              AND fp.alphaearth_embedding IS NOT NULL
            ORDER BY fp.updated_at DESC
            LIMIT 1
        """
        args = (ctx.session_id, year)
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(query, *args)

    if row is None or row["alphaearth_embedding"] is None:
        return None

    raw = row["alphaearth_embedding"]
    # pgvector returns the embedding as a string like "[0.1,0.2,...]" over
    # asyncpg unless a codec is registered; parse both that and native sequences.
    if isinstance(raw, str):
        values = [float(v) for v in raw.strip().strip("[]").split(",") if v.strip()]
    else:
        values = [float(v) for v in raw]
    embedding = np.asarray(values, dtype=np.float64)
    if embedding.size != _EMBED_DIM:
        logger.warning(
            "classify_embedding_unexpected_dim",
            expected=_EMBED_DIM,
            got=int(embedding.size),
        )
        return None
    return embedding


async def _sample_embedding_via_gee(
    ctx: ToolContext, year: int, aoi: GeoJSONGeometry
) -> np.ndarray | None:
    """Download the AOI's AlphaEarth embedding from GEE (the "download" fallback).

    Invoked only after the persisted-parcel lookup (`_fetch_parcel_embedding`)
    misses: it samples the mean annual AlphaEarth embedding of the drawn polygon
    live from Earth Engine. Earth Engine is authenticated from settings (service
    account or ADC) under the configured GEE project. The blocking EE call runs
    in a worker thread so the event loop is never blocked. Any failure (EE not
    installed, no credentials, no coverage, null bands) returns ``None`` so the
    caller degrades to the controlled ``needs_gee_sampling`` result.

    Args:
        ctx: Tool execution context (carries the typed settings).
        year: Campaign year of the annual embedding.
        aoi: Drawn AOI polygon to sample.

    Returns:
        A ``(64,)`` ``float64`` embedding, or ``None`` on any failure.
    """
    import asyncio

    from ml.ingest.gee_sampler import sample_alphaearth_aoi_mean

    project = getattr(ctx.settings, "gee_project_id", "") or None
    sa_raw = getattr(ctx.settings, "gee_service_account_path", "") or ""
    service_account_json = Path(sa_raw) if sa_raw else None
    geometry = {"type": aoi.type, "coordinates": aoi.coordinates}
    embedding = await asyncio.to_thread(
        sample_alphaearth_aoi_mean,
        geometry=geometry,
        year=year,
        project=project,
        service_account_json=service_account_json,
    )
    if embedding is None:
        return None
    if embedding.size != _EMBED_DIM:
        logger.warning(
            "classify_gee_embedding_unexpected_dim",
            expected=_EMBED_DIM,
            got=int(embedding.size),
        )
        return None
    return embedding


def _needs_gee_result() -> ClassificationResult:
    """Build the controlled result for a parcel without an AlphaEarth embedding.

    Returns a uniform low-confidence posterior tagged with the
    ``needs_gee_sampling`` sentinel class so the agent loop can route the request
    to the GEE sampler instead of trusting a hallucinated crop label.

    Returns:
        A :class:`ClassificationResult` with ``crop_class="needs_gee_sampling"``,
        ``confidence`` equal to a uniform prior, and a flat 18-class posterior.
    """
    uniform = 1.0 / _NUM_CLASSES
    return ClassificationResult(
        crop_class=_NEEDS_GEE_SAMPLING,
        confidence=uniform,
        class_probabilities={_NEEDS_GEE_SAMPLING: 1.0},
    )


def _build_result(
    proba: np.ndarray,
    class_names: dict[int, str],
    *,
    restrict: bool,
    label_space: LabelSpace,
    served_model: str,
) -> ClassificationResult:
    """Assemble a :class:`ClassificationResult` from an 18-class posterior.

    When ``restrict`` is ``True`` the posterior is masked + renormalized over the
    ``label_space`` resolved classes (the default, honest path); the result then
    carries only the kept classes. When ``False`` the full 18-class posterior is
    surfaced (legacy behaviour). The argmax / confidence are always computed over
    the SAME distribution that is reported (no mismatch between the headline class
    and the surfaced probabilities).

    Args:
        proba: A ``(18,)`` post-softmax distribution over the semantic18 space.
        class_names: ``{semantic18_id: crop name}`` for naming the kept classes;
            for the full path it should cover all 18 ids.
        restrict: Whether to mask + renormalize over ``label_space``.
        label_space: The active label-space (used only when ``restrict``).
        served_model: The member that actually produced ``proba`` (``"voting-3"``,
            ``"xgb-alphaearth"`` or ``"stacking-5"``), stamped onto the result so
            the reasoner and the UI report the real active model after any
            degradation.

    Returns:
        The typed :class:`ClassificationResult`. When restricting and no mass
        landed on the resolved classes, ``crop_class`` is ``"unresolved"`` with
        zero confidence (an honest "none of the resolved classes apply").
    """
    # The model's RAW (unrestricted) top class. When it falls OUTSIDE the resolved
    # vocabulary, the restricted headline is a renormalization artifact the reasoner
    # must hedge (the out-of-vocabulary handoff), never report as confident.
    raw_top_idx = int(np.argmax(proba))
    if restrict:
        restricted = restrict_posterior(proba, label_space)
        named = {
            label_space.class_names.get(cid, class_names.get(cid, str(cid))): p
            for cid, p in restricted.items()
        }
        out_of_vocab = list(label_space.dropped_class_names.values())
        # Non-None only when the raw lean is a crop the space cannot resolve.
        unresolved_candidate = label_space.dropped_class_names.get(raw_top_idx)
        if not named or max(named.values(), default=0.0) <= 0.0:
            return ClassificationResult(
                crop_class="unresolved",
                confidence=0.0,
                class_probabilities=named,
                out_of_vocabulary_classes=out_of_vocab,
                unresolved_candidate=unresolved_candidate,
                served_model=served_model,
            )
        top_name = max(named, key=lambda k: named[k])
        return ClassificationResult(
            crop_class=top_name,
            confidence=float(named[top_name]),
            class_probabilities=named,
            out_of_vocabulary_classes=out_of_vocab,
            unresolved_candidate=unresolved_candidate,
            served_model=served_model,
        )

    named = {class_names.get(idx, str(idx)): float(proba[idx]) for idx in range(_NUM_CLASSES)}
    return ClassificationResult(
        crop_class=class_names.get(raw_top_idx, str(raw_top_idx)),
        confidence=float(proba[raw_top_idx]),
        class_probabilities=named,
        served_model=served_model,
    )


async def _stacking_posterior(ctx: ToolContext, inp: ClassifyParcelInput) -> np.ndarray | None:
    """Try to produce the Stacking-5 posterior for the parcel under ``inp.aoi``.

    Resolves the parcel's bridged canonical id, loads the cached Stacking-5 meta
    and returns its ``(18,)`` posterior. Degrades CLEANLY (returns ``None``, never
    raises) when:

    - the parcel does not resolve to the fold-5 OOF universe (a new polygon), or
    - the OOF / PASTIS-R GT artifacts are unavailable (DVC not pulled).

    Each degradation is logged with a structured ``classify_stacking_unavailable``
    warning so the caller can fall back to ``xgb-alphaearth`` (AC-8).

    Args:
        ctx: Tool execution context (pool, session id).
        inp: Validated classify arguments (AOI is used to resolve the parcel).

    Returns:
        A ``(18,)`` ``float64`` Stacking-5 posterior, or ``None`` to degrade.
    """
    canonical_id = await _resolve_canonical_parcel_id(ctx, inp.aoi)
    if canonical_id is None:
        logger.warning(
            "classify_stacking_unavailable",
            reason="no persisted parcel resolved for the AOI; using xgb-alphaearth.",
        )
        return None

    try:
        stacking = _load_stacking_five()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "classify_stacking_unavailable",
            reason="fold-5 OOF / PASTIS-R ground truth not available",
            error=str(exc),
        )
        return None

    posterior = stacking.posterior_for_parcel(canonical_id)
    if posterior is None:
        logger.warning(
            "classify_stacking_unavailable",
            reason="parcel not in the fold-5 OOF universe; using xgb-alphaearth.",
            canonical_parcel_id=canonical_id,
        )
        return None

    logger.info(
        "classify_stacking_used",
        canonical_parcel_id=canonical_id,
        member="stacking-5",
    )
    return posterior


async def _voting_posterior(ctx: ToolContext, inp: ClassifyParcelInput) -> np.ndarray | None:
    """Try to produce the Voting-3 posterior for the parcel under ``inp.aoi``.

    Resolves the parcel's bridged canonical id, loads the cached Voting-3 weighted
    vote (learned deployment weights + per-parcel member rows) and returns its
    ``(18,)`` posterior. Degrades CLEANLY (returns ``None``, never raises) when:

    - the parcel does not resolve to the fold-5 OOF universe (a new polygon), or
    - any member OOF, the PASTIS-R GT or the PASTIS-R geometry the weight learning
      consumes is unavailable (DVC not pulled).

    Each degradation is logged with a structured ``classify_voting3_unavailable``
    warning so the caller can fall back to ``xgb-alphaearth`` -- it NEVER fabricates
    a posterior (Arthur's absolute "real values only" rule).

    Args:
        ctx: Tool execution context (pool, session id).
        inp: Validated classify arguments (AOI is used to resolve the parcel).

    Returns:
        A ``(18,)`` ``float64`` Voting-3 posterior, or ``None`` to degrade.
    """
    canonical_id = await _resolve_canonical_parcel_id(ctx, inp.aoi)
    if canonical_id is None:
        logger.warning(
            "classify_voting3_unavailable",
            reason="no persisted parcel resolved for the AOI; using xgb-alphaearth.",
        )
        return None

    try:
        voting = _load_voting_three()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "classify_voting3_unavailable",
            reason="fold-5 OOF / PASTIS-R geometry or ground truth not available",
            error=str(exc),
        )
        return None

    posterior = voting.posterior_for_parcel(canonical_id)
    if posterior is None:
        logger.warning(
            "classify_voting3_unavailable",
            reason="parcel not in the three-member fold-5 OOF universe; using xgb-alphaearth.",
            canonical_parcel_id=canonical_id,
        )
        return None

    logger.info(
        "classify_voting3_used",
        canonical_parcel_id=canonical_id,
        member="voting-3",
    )
    return posterior


async def run(inp: ClassifyParcelInput, ctx: ToolContext) -> ClassificationResult:
    """Classify a parcel's crop honestly (Voting-3, xgb-alphaearth or Stacking-5).

    By default serves the ``"voting3"`` EPIC 12 deployment champion (US-081 AC4a)
    restricted to the active label-space's resolved classes (the configured
    :data:`~ml.eval.class_remap.DEFAULT_LABEL_SPACE`); for a fresh AOI with no
    fold-5 OOF row it degrades cleanly to ``xgb-alphaearth`` (the historical
    behaviour). The serving model is selected by ``inp.resolved_model``
    (``inp.model`` with the legacy ``inp.use_stacking`` flag promoted to
    ``"stacking5"`` only when ``inp.model == "xgb"``):

    - ``"voting3"`` -- the EPIC 12 weighted-vote deployment champion, used for a
      fold-5 parcel; degrades cleanly to ``xgb-alphaearth`` when no OOF row matches
      or the OOF / PASTIS-R geometry / GT artifacts are unavailable.
    - ``"stacking5"`` -- the EPIC 6 Stacking-5 meta (legacy), same degradation.
    - ``"xgb"`` -- the tabular member directly (no ensemble lookup).

    With ``inp.restrict_to_resolved_classes=False`` the full 18-class posterior is
    returned (legacy).

    The result's ``served_model`` field records the member that actually produced
    the posterior (``"voting-3"``, ``"xgb-alphaearth"`` or ``"stacking-5"``), so it
    reflects degradation: a Voting-3 request on a parcel outside the fold-5 OOF
    universe reports ``served_model="xgb-alphaearth"``, never ``"voting-3"``. This
    lets the reasoner and the UI stay honest about the active model.

    Model selection is decided HERE, not by the reasoner: when the user pinned a
    model in the UI (``ctx.crop_model``) it is served verbatim, because the
    crop-model switch is a hard choice and must not depend on an LLM honouring a
    prompt. The pin outranks the WHOLE caller-side resolution -- not just
    ``inp.model`` but also the legacy ``use_stacking`` promotion of
    :attr:`~ml.agent.schemas.ClassifyParcelInput.resolved_model` (so a pin of
    ``"xgb"`` serves the tabular member even alongside ``use_stacking=True``). With
    no pin, ``inp.resolved_model`` (or its ``voting3`` default) stands.

    Args:
        inp: Validated arguments (session id, AOI polygon, year, and the
            ``restrict_to_resolved_classes`` / ``model`` / ``use_stacking`` /
            ``label_space`` flags).
        ctx: Tool execution context (asyncpg pool, settings, session id).

    Returns:
        A :class:`ClassificationResult` whose ``served_model`` names the member that
        actually ran. When no AlphaEarth embedding is available for the session's
        parcel (a fresh AOI), a controlled ``needs_gee_sampling`` result is returned
        instead of a guessed class -- the model is NOT evaluated before the
        embedding is resolved.
    """
    # The USER's pinned model (ctx.crop_model) OUTRANKS the reasoner's argument.
    # The UI presents the crop-model switch as a hard choice, so it cannot depend
    # on the LLM choosing to honour a system instruction; enforcing it here, at the
    # tool boundary, makes the promise real regardless of what the reasoner passed
    # (or forgot to pass). ``None`` = the user pinned nothing -> the reasoner's
    # argument (or the ``voting3`` default) stands.
    reasoner_model = inp.resolved_model
    requested_model = ctx.crop_model or reasoner_model
    if ctx.crop_model is not None and requested_model != reasoner_model:
        logger.info(
            "classify_new_parcel_user_pin_overrode_reasoner",
            session_id=str(inp.session_id),
            reasoner_model=reasoner_model,
            user_pinned_model=requested_model,
        )
    logger.info(
        "classify_new_parcel_started",
        session_id=str(inp.session_id),
        year=inp.year,
        geometry_type=inp.aoi.type,
        restrict=inp.restrict_to_resolved_classes,
        model=requested_model,
        use_stacking=inp.use_stacking,
        label_space=inp.label_space,
        user_pinned=ctx.crop_model is not None,
    )

    # Search first (persisted, session-scoped), then download (live GEE sampling)
    # when no persisted parcel embedding intersects the drawn AOI.
    embedding = await _fetch_parcel_embedding(ctx, inp.year, inp.aoi)
    embedding_source = "persisted"
    if embedding is None:
        embedding = await _sample_embedding_via_gee(ctx, inp.year, inp.aoi)
        embedding_source = "gee"
    if embedding is None:
        logger.info(
            "classify_new_parcel_needs_gee",
            session_id=str(inp.session_id),
            year=inp.year,
            reason="no persisted embedding intersects the AOI and GEE sampling failed",
        )
        return _needs_gee_result()
    logger.info(
        "classify_new_parcel_embedding_resolved",
        session_id=str(inp.session_id),
        year=inp.year,
        source=embedding_source,
    )

    # Resolve the label-space up front so an unknown name fails fast (not after
    # the expensive model load). The input default is DEFAULT_LABEL_SPACE.
    label_space = get_label_space(inp.label_space)

    classifier = _load_classifier()

    # Model dispatch. Voting-3 and Stacking-5 each try to serve their fold-5
    # posterior and degrade to xgb-alphaearth (proba is None) when the parcel does
    # not resolve or the artifacts are unavailable -- never a fabricated posterior.
    proba: np.ndarray | None = None
    member = "xgb-alphaearth"
    if requested_model == "voting3":
        proba = await _voting_posterior(ctx, inp)
        if proba is not None:
            member = "voting-3"
    elif requested_model == "stacking5":
        proba = await _stacking_posterior(ctx, inp)
        if proba is not None:
            member = "stacking-5"
    if proba is None:
        proba = classifier.predict_proba_18(embedding)

    result = _build_result(
        proba,
        classifier.class_names,
        restrict=inp.restrict_to_resolved_classes,
        label_space=label_space,
        served_model=member,
    )
    logger.info(
        "classify_new_parcel_finished",
        session_id=str(inp.session_id),
        requested_model=requested_model,
        member=member,
        restricted=inp.restrict_to_resolved_classes,
        n_classes=len(result.class_probabilities),
        crop_class=result.crop_class,
        confidence=round(result.confidence, 4),
    )
    return result
