"""US-040 closure: run the four rubric ensembles + comparison table + figures.

This is the single serialized closing point of US-040 (plan Section 5, Phase 4).
It instantiates the four mandatory ensembles (Voting / Bagging / Stacking /
Blending), consumes the US-031 OOF artifacts and the PASTIS-R fold-5 ground
truth, builds the best-individual-vs-4-ensembles comparison table, emits the >=4
interpretable figures and logs one MLflow run per ensemble to the ``ensemble``
experiment on the Docker server ``:5010`` with the mandatory ``data_version`` +
``code_version`` tags (and the ``chosen_model`` tag on the winner).

Anti-leakage (R-LEAK -- the single most important rubric criterion). Documented
and enforced here as well as in the ensembles themselves:

1. **Report fold-5 ONLY, never fold-4.** ``--fold`` defaults to 5 and ANY other
   value is rejected before any work runs (fold-4 was the selection fold). Every
   metric reported by this script comes from
   :meth:`ml.ensemble.base.EnsembleModel.evaluate`, which is fold-5-only.
2. **Probabilities, not logits.** The four ensembles average POST-softmax OOF
   probabilities (US-031 dumps post-softmax); the figures revalidate the
   probabilities as post-softmax before any ROC/PR plot.
3. **Meta-learner sees OOF only.** Stacking trains its meta-learner exclusively
   on the OOF parcel probabilities with spatial sub-folds of fold-5.
4. **Blending holdout spatially disjoint.** Blending optimizes its simplex
   weights on a geographically disjoint holdout carved from fold-5.

The ground truth is NOT inside the OOF parquet (the US-031 dump discards the
target), so it is reconstructed here from PASTIS-R: the per-parcel semantic18
label is the majority vote of the semantic TARGET pixels inside each parcel's
ParcelIDs geometry (see :func:`build_parcel_ground_truth`).

xgb-alphaearth caveat (R-TAB-FEATURES). US-031 only dumped the parcel OOF of the
dense members (``tsvit-pheno``, ``utae``, ``unet``, ``deeplabv3plus``,
``segformer``, ``anysat``). There is NO ``oof_parcel_xgb-alphaearth_fold5.parquet``.
Bagging trains its XGBoost directly from ``data/features/features_fused_pastis.parquet``
(it does not need a parcel OOF), but Stacking and Blending list ``xgb-alphaearth``
as a base member. This script handles the absence GRACEFULLY: if the
xgb-alphaearth parcel OOF is missing, it either (a) MATERIALIZES it from the
tabular features by fitting an XGBoost-AlphaEarth on folds 1-4 and predicting the
fold-5 parcels (``--materialize-xgb``, the default), or (b) DEGRADES Stacking /
Blending to the two dense parcel members (``--no-materialize-xgb``), logging the
decision either way. The materialized parcel OOF is its fold-5 prediction, so the
anti-leakage holds (the XGBoost never saw fold-5).

Usage (from repo root, once the OOF + PASTIS-R are available locally):

    poetry run python scripts/run_us040_ensembles.py \\
        --oof-dir ml/eval/oof \\
        --pastis-root data/PASTIS-R \\
        --features data/features/features_fused_pastis.parquet \\
        --out-dir reports/ensemble \\
        --fold 5

Add ``--no-use-mlflow`` for a dry run without contacting the MLflow server, or
``--n-trials-bagging`` / ``--n-trials-blending`` to bound the Optuna budget.
(Typer collapses the single ``run`` command into the root, so no subcommand is
needed; ``--fold`` other than 5 is rejected immediately for anti-leakage.)

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, ``typer`` for the CLI, type hints and
Google-style docstrings; visible prose Spanish, identifiers English; no emojis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
import typer

from ml.eval.ensemble_figures import (
    build_comparison_table,
    confusion_norm_abs,
    pr_per_class,
    roc_ovr_per_class,
    spatial_residuals,
)
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="US-040 ensembles closing run.")

#: The only fold whose metrics may be reported (anti-leakage R-LEAK).
HELD_OUT_FOLD: int = 5

#: Individual baseline reference (TSViT-pheno F1-macro fold-5, plan Section 1.2).
INDIVIDUAL_BASELINE_NAME: str = "TSViT-pheno (individual)"
INDIVIDUAL_BASELINE_F1: float = 0.6253

#: Dense voting members (R-VOTE: TSViT base not dumped -> third voter = U-Net).
VOTING_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "unet")

#: Heterogeneous parcel base members of stacking/blending.
PARCEL_BASE_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: The tabular member that may need materialization (no US-031 parcel OOF).
XGB_MEMBER: str = "xgb-alphaearth"

#: Canonical key column shared by every parcel frame.
_KEY: str = "canonical_parcel_id"

#: Per-class probability prefix in the materialized parcel OOF.
_ALPHAEARTH_PREFIX: str = "dim_"

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18


# ---------------------------------------------------------------------------
# Ground truth + geometry from PASTIS-R (the OOF dump discards the target).
# ---------------------------------------------------------------------------


def build_parcel_ground_truth(
    patch_ids: Sequence[str | int],
    pastis_root: Path,
    *,
    ignore_index: int = 255,
) -> pl.DataFrame:
    """Reconstruct the per-parcel semantic18 ground truth from PASTIS-R.

    For each held-out patch, reads the semantic ``TARGET`` and the ParcelIDs
    raster, and assigns each parcel the MAJORITY semantic18 label of its pixels
    (the OOF dump uses the same parcels via ``pixel_to_parcel_probs``). The
    canonical key matches the OOF parcels: ``f"{patch_id}_{parcel_raster_id}"``.

    The raw PASTIS-R semantic channel uses the 20-class convention (``0`` =
    Background, ``1..18`` = agronomic crops, ``19`` = Void). The dense OOF members
    and the segmentation harness, however, live in the contiguous ``semantic18``
    space ``[0..17]`` (Background/Void dropped, agronomic ``c`` shifted to
    ``c - 1``). This builder applies the SAME ``semantic18`` LUT that
    :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset` uses (the source
    of truth Voting evaluates against), so the per-parcel GT is class-aligned with
    every ensemble's predictions (without it the labels are off by one).

    Args:
        patch_ids: Held-out (fold-5) PASTIS-R patch ids to reconstruct.
        pastis_root: Root of the PASTIS-R dataset (contains ``ANNOTATIONS/``).
        ignore_index: Semantic ignore label (Background/Void) excluded from the
            majority vote (default 255).

    Returns:
        A Polars DataFrame with ``canonical_parcel_id`` (Utf8) + ``label``
        (Int64, in the ``[0..17]`` semantic18 space), one row per parcel, sorted
        by the key.

    Raises:
        FileNotFoundError: if a patch's TARGET or ParcelIDs raster is missing.
    """
    from ml.data.pastis_seg_dataset import _build_semantic18_lut
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    # The contiguous semantic18 LUT (PASTIS 0..19 -> [0..17] U {ignore}), shared
    # with the dataset so the parcel GT matches the dense OOF / Voting GT exactly.
    label_lut = _build_semantic18_lut(ignore_index)

    root = Path(pastis_root)
    keys: list[str] = []
    labels: list[int] = []
    for raw_pid in patch_ids:
        pid = str(raw_pid)
        target_path = root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if not target_path.exists():
            raise FileNotFoundError(f"PASTIS-R semantic TARGET not found: {target_path}.")
        target = np.load(target_path)
        if target.ndim == 3:  # PASTIS ships (3, H, W); the semantic channel is 0.
            target = target[0]
        parcel_ids = load_pastis_parcel_ids(pid, root)

        flat_pids = parcel_ids.reshape(-1)
        # Map raw PASTIS labels (0..19) to the contiguous semantic18 space before
        # voting; Background/Void become ``ignore_index`` and are dropped below.
        raw = np.clip(target.reshape(-1).astype(np.int64), 0, 19)
        flat_labels = label_lut[raw]
        valid = (flat_pids != 0) & (flat_labels != ignore_index)
        flat_pids = flat_pids[valid]
        flat_labels = flat_labels[valid]
        if flat_pids.size == 0:
            continue

        unique_ids, inverse = np.unique(flat_pids, return_inverse=True)
        # Majority semantic18 label per parcel via a (n_parcels, num_classes) vote.
        votes = np.zeros((unique_ids.size, _NUM_CLASSES), dtype=np.int64)
        in_range = flat_labels < _NUM_CLASSES
        np.add.at(votes, (inverse[in_range], flat_labels[in_range]), 1)
        majority = votes.argmax(axis=1)
        for local, lab in zip(unique_ids, majority, strict=True):
            keys.append(f"{pid}_{int(local)}")
            labels.append(int(lab))

    frame = pl.DataFrame({_KEY: keys, "label": labels}).with_columns(pl.col("label").cast(pl.Int64))
    frame = canonical_parcel_id(frame, col=_KEY).sort(_KEY)
    logger.info("parcel_ground_truth_built", n_parcels=frame.height, n_patches=len(patch_ids))
    return frame


def build_parcel_geometries(
    patch_ids: Sequence[str | int],
    pastis_root: Path,
) -> pl.DataFrame:
    """Build a per-parcel geometry frame (centroid Point) from PASTIS-R rasters.

    PASTIS-R patches are 128x128 pixel tiles whose true polygon footprint lives
    in ``metadata.geojson``; the per-parcel geographic position used for the
    spatial sub-folds and the residual map is the patch centroid offset by the
    parcel's pixel centroid (so parcels of the same patch are co-located but
    distinct, and parcels of different patches are spatially separated). This is
    enough for ``build_spatial_kfold`` (H3 res 5) to form geographic blocks.

    Args:
        patch_ids: Held-out PASTIS-R patch ids.
        pastis_root: Root of the PASTIS-R dataset.

    Returns:
        A Polars DataFrame with ``canonical_parcel_id`` (Utf8) + ``geometry``
        (WKT Point in EPSG:4326), one row per parcel.

    Raises:
        FileNotFoundError: if ``metadata.geojson`` is missing.
    """
    from pyproj import Transformer
    from shapely.geometry import Point, shape

    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    root = Path(pastis_root)
    meta_path = root / "metadata.geojson"
    if not meta_path.exists():
        raise FileNotFoundError(f"PASTIS-R metadata.geojson not found: {meta_path}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # PASTIS-R metadata.geojson is in EPSG:2154 (Lambert-93, metres); H3 / the
    # spatial split need lon/lat (EPSG:4326), so reproject every patch centroid.
    crs_name = meta.get("crs", {}).get("properties", {}).get("name", "EPSG:2154")
    transformer = Transformer.from_crs(crs_name, "EPSG:4326", always_xy=True)
    centroid_by_patch: dict[str, tuple[float, float]] = {}
    for feature in meta["features"]:
        pid = str(feature["properties"]["ID_PATCH"])
        centroid = shape(feature["geometry"]).centroid
        lon, lat = transformer.transform(float(centroid.x), float(centroid.y))
        centroid_by_patch[pid] = (float(lon), float(lat))

    keys: list[str] = []
    geoms: list[str] = []
    for raw_pid in patch_ids:
        pid = str(raw_pid)
        cx, cy = centroid_by_patch.get(pid, (0.0, 0.0))
        parcel_ids = load_pastis_parcel_ids(pid, root)
        h, w = parcel_ids.shape
        unique_ids = np.unique(parcel_ids[parcel_ids != 0])
        for local in unique_ids:
            ys, xs = np.where(parcel_ids == local)
            # Small intra-patch offset (degrees) from the pixel centroid so the
            # parcels of one patch are co-located but separable.
            off_x = (float(xs.mean()) / w - 0.5) * 0.01
            off_y = (float(ys.mean()) / h - 0.5) * 0.01
            keys.append(f"{pid}_{int(local)}")
            geoms.append(Point(cx + off_x, cy + off_y).wkt)

    frame = pl.DataFrame({_KEY: keys, "geometry": geoms})
    frame = canonical_parcel_id(frame, col=_KEY)
    logger.info("parcel_geometries_built", n_parcels=frame.height)
    return frame


# ---------------------------------------------------------------------------
# Canonical parcel id reconciliation between the tabular and dense spaces.
# ---------------------------------------------------------------------------
#
# The tabular features (``features_fused_pastis.parquet``) key each parcel by its
# PASTIS instance id (``TARGET[1]`` channel, the sequential 1..N used by
# ``vectorize_pastis_parcels.py``), so its ``parcel_id`` column already encodes
# ``f"{patch_id}_{instance_id}"`` (e.g. ``"10003_1"``). The dense OOF dump and the
# ground truth instead key parcels by the SEPARATE ``ParcelIDs_<patch>.npy`` raster
# (e.g. ``"10003_1103071"``), which is the canonical schema of
# :mod:`ml.utils.parcel_reconcile`. The two id spaces are spatially 1:1 but
# numerically disjoint, so every tabular key must be translated to the canonical
# ParcelIDs space before it can align with the GT / dense OOF members. Doing this
# in ONE place keeps Bagging, Stacking and Blending on the same canonical key.


def _instance_to_parcel_id_map(
    patch_id: str | int,
    pastis_root: Path,
) -> dict[int, int]:
    """Map a patch's instance ids (``TARGET[1]``) to its ParcelIDs raster ids.

    PASTIS-R ships two per-pixel id rasters for the same parcels: the instance
    channel ``TARGET[1]`` (sequential ``1..N``, used by the tabular features) and
    the ``ParcelIDs_<patch>.npy`` raster (the canonical ids used by the dense OOF
    and the ground truth). They are spatially co-registered, so each instance id
    maps to the ParcelIDs value that dominates its pixels (a 1:1 correspondence in
    PASTIS-R). This bridges the tabular key space to the canonical one.

    Args:
        patch_id: PASTIS-R patch identifier.
        pastis_root: Root of the PASTIS-R dataset (contains ``ANNOTATIONS/``).

    Returns:
        A dict ``{instance_id: parcel_raster_id}`` for every parcel of the patch.

    Raises:
        FileNotFoundError: if the patch's TARGET or ParcelIDs raster is missing.
    """
    # Promoted to the library module so the FarSLIP members (ml/ensemble/*) can
    # reuse the SAME bridge without importing this script. Kept as a thin wrapper
    # for backward compatibility with this script's existing callers.
    from ml.utils.parcel_reconcile import instance_to_parcel_id_map

    return instance_to_parcel_id_map(patch_id, pastis_root)


def tabular_parcel_keys(
    df_tabular: pl.DataFrame,
    pastis_root: Path,
) -> list[str]:
    """Build canonical ParcelIDs keys for the rows of a tabular parcel frame.

    Translates each tabular row (keyed by ``patch_id`` + ``instance_id`` in the
    instance-id space) into the canonical ``f"{patch_id}_{parcel_raster_id}"`` key
    used by the dense OOF members and the ground truth, so Bagging / Stacking /
    Blending all align on a single canonical key. The ``patch_id`` column is the
    true patch (never re-derived from ``parcel_id`` to avoid double-prefixing).

    Args:
        df_tabular: Tabular parcel frame with ``patch_id`` and ``instance_id``
            columns (``features_fused_pastis.parquet`` schema).
        pastis_root: Root of the PASTIS-R dataset.

    Returns:
        A list of canonical ``canonical_parcel_id`` strings, one per row of
        ``df_tabular`` (row order preserved).

    Raises:
        ValueError: if ``patch_id`` or ``instance_id`` is missing, or an
            ``instance_id`` cannot be mapped to a ParcelIDs raster value.
    """
    for col in ("patch_id", "instance_id"):
        if col not in df_tabular.columns:
            raise ValueError(f"tabular features are missing the `{col}` column.")

    patch_ids = df_tabular.get_column("patch_id").to_list()
    instance_ids = df_tabular.get_column("instance_id").to_list()

    cache: dict[str, dict[int, int]] = {}
    keys: list[str] = []
    for patch, inst in zip(patch_ids, instance_ids, strict=True):
        pid = str(patch)
        if pid not in cache:
            cache[pid] = _instance_to_parcel_id_map(pid, pastis_root)
        raster_id = cache[pid].get(int(inst))
        if raster_id is None:
            raise ValueError(
                f"instance id {inst} of patch {pid} has no ParcelIDs raster match; "
                "the tabular features and PASTIS-R rasters are out of sync."
            )
        keys.append(f"{pid}_{raster_id}")
    logger.info("tabular_parcel_keys_built", n_rows=len(keys), n_patches=len(cache))
    return keys


def _remap_tabular_class_id(df_tabular: pl.DataFrame) -> pl.DataFrame:
    """Remap a tabular frame's ``class_id`` to the contiguous semantic18 space.

    The tabular features carry the raw PASTIS ``class_id`` (``1..18``); the dense
    OOF members, the ground truth and every ensemble live in the contiguous
    ``semantic18`` space ``[0..17]`` (agronomic ``c`` shifted to ``c - 1``,
    Background/Void dropped). This applies the SAME LUT as
    :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`, so the Bagging
    member (which learns whatever ``class_id`` it is handed) predicts in the same
    class columns as the rest. Rows whose class maps to the ignore label
    (Background/Void) are dropped.

    Args:
        df_tabular: Tabular parcel frame with a ``class_id`` column.

    Returns:
        The frame with ``class_id`` in the ``[0..17]`` semantic18 space and the
        Background/Void rows removed. Frames without ``class_id`` pass through.
    """
    if "class_id" not in df_tabular.columns:
        return df_tabular
    from ml.data.pastis_seg_dataset import _build_semantic18_lut

    lut = _build_semantic18_lut(255)
    pastis = np.clip(df_tabular.get_column("class_id").to_numpy().astype(np.int64), 0, 19)
    mapped = lut[pastis]
    remapped = df_tabular.with_columns(pl.Series("class_id", mapped, dtype=pl.Int64))
    return remapped.filter(pl.col("class_id") != 255)


# ---------------------------------------------------------------------------
# xgb-alphaearth parcel OOF: materialize from tabular features when absent.
# ---------------------------------------------------------------------------


def materialize_xgb_parcel_oof(
    features_path: Path,
    *,
    out_path: Path,
    pastis_root: Path,
    random_state: int = 42,
) -> Path:
    """Materialize the missing ``oof_parcel_xgb-alphaearth_fold5.parquet``.

    US-031 never dumped the parcel OOF of the tabular XGBoost-AlphaEarth member.
    This rebuilds it leak-free: fit an XGBoost-AlphaEarth on folds 1-4 of
    ``features_fused_pastis.parquet`` and predict the fold-5 parcels, writing the
    post-softmax ``prob_000..prob_017`` per parcel. Because the model never sees
    fold-5, the result is a true held-out OOF prediction (anti-leakage).

    The parcel keys are translated to the canonical ParcelIDs space via
    :func:`tabular_parcel_keys` so this materialized OOF aligns with the dense
    members and the ground truth (the tabular features key parcels by the
    instance-id space, NOT the ParcelIDs raster).

    Args:
        features_path: ``data/features/features_fused_pastis.parquet`` with
            ``parcel_id``, ``patch_id``, ``instance_id``, ``class_id``, ``fold``
            and the AlphaEarth ``dim_000..dim_063`` columns.
        out_path: Destination ``oof_parcel_xgb-alphaearth_fold5.parquet``.
        pastis_root: PASTIS-R root, used to translate instance ids to canonical
            ParcelIDs keys.
        random_state: Deterministic seed.

    Returns:
        The :class:`pathlib.Path` of the written parquet.

    Raises:
        ValueError: if the features parquet lacks the required columns or fold-5
            has no parcels.
    """
    from sklearn.preprocessing import LabelEncoder

    from ml.data.pastis_seg_dataset import _build_semantic18_lut
    from ml.train.baseline import build_estimator

    df = pl.read_parquet(features_path)
    for col in ("patch_id", "class_id", "fold"):
        if col not in df.columns:
            raise ValueError(f"features parquet is missing the `{col}` column.")
    feature_cols = [c for c in df.columns if c.startswith(_ALPHAEARTH_PREFIX)]
    if not feature_cols:
        raise ValueError(
            f"no AlphaEarth feature column with prefix {_ALPHAEARTH_PREFIX!r} in {features_path}."
        )

    train = df.filter(pl.col("fold") != HELD_OUT_FOLD).filter(pl.col("class_id").is_not_null())
    test = df.filter(pl.col("fold") == HELD_OUT_FOLD)
    if test.height == 0:
        raise ValueError("fold-5 has no parcels in the features parquet.")

    # The features carry the raw PASTIS class_id (1..18); map it to the contiguous
    # semantic18 space [0..17] (the dense OOF / GT space) before fitting, so the
    # materialized probabilities land in the SAME class columns as every member.
    label_lut = _build_semantic18_lut(255)
    x_train = train.select(feature_cols).to_numpy().astype(np.float64)
    x_train = np.where(np.isfinite(x_train), x_train, 0.0)
    pastis_train = np.clip(train.get_column("class_id").to_numpy().astype(np.int64), 0, 19)
    y_raw = label_lut[pastis_train]
    keep = y_raw != 255  # drop Background/Void parcels (no semantic18 class).
    x_train = x_train[keep]
    y_raw = y_raw[keep]
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
            "random_state": random_state,
        },
    )
    estimator.fit(x_train, y_train)

    x_test = test.select(feature_cols).to_numpy().astype(np.float64)
    x_test = np.where(np.isfinite(x_test), x_test, 0.0)
    proba_local = np.asarray(estimator.predict_proba(x_test), dtype=np.float64)

    # Scatter to the global 18-class space using the shared encoder classes.
    global_classes = encoder.classes_.astype(np.int64)
    full = np.zeros((proba_local.shape[0], _NUM_CLASSES), dtype=np.float64)
    for col, gid in enumerate(global_classes):
        if 0 <= int(gid) < _NUM_CLASSES:
            full[:, int(gid)] = proba_local[:, col]
    row_sums = full.sum(axis=1, keepdims=True)
    full = full / np.where(row_sums < 1e-12, 1.0, row_sums)

    keys = tabular_parcel_keys(test, pastis_root)
    data: dict[str, object] = {_KEY: keys}
    for c, name in enumerate(PROB_COLUMNS):
        data[name] = full[:, c].astype(np.float32)
    data["pred_class"] = full.argmax(axis=1).astype(np.int64)
    data["n_pixels"] = np.full(full.shape[0], -1, dtype=np.int64)  # tabular: no pixels.
    frame = canonical_parcel_id(pl.DataFrame(data), col=_KEY).sort(_KEY)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    logger.info(
        "xgb_parcel_oof_materialized",
        n_parcels=frame.height,
        n_train=train.height,
        path=str(out_path),
    )
    return out_path


def resolve_parcel_members(
    oof_dir: Path,
    *,
    materialize_xgb: bool,
    features_path: Path | None,
    pastis_root: Path,
    random_state: int,
) -> tuple[str, ...]:
    """Resolve the parcel base members, handling the missing xgb-alphaearth OOF.

    Args:
        oof_dir: OOF directory holding ``oof_parcel_{member}_fold5.parquet``.
        materialize_xgb: If ``True`` and the xgb parcel OOF is missing, build it
            from ``features_path``; if ``False`` degrade to the dense members.
        features_path: Tabular features parquet (required to materialize).
        pastis_root: PASTIS-R root (forwarded for canonical key translation).
        random_state: Seed forwarded to the materialization.

    Returns:
        The resolved ordered member tuple for stacking/blending (either the full
        :data:`PARCEL_BASE_MEMBERS` or the two dense members when degraded).
    """
    xgb_oof = oof_dir / f"oof_parcel_{XGB_MEMBER}_fold5.parquet"
    if xgb_oof.exists():
        logger.info("xgb_parcel_oof_present", path=str(xgb_oof))
        return PARCEL_BASE_MEMBERS

    if materialize_xgb and features_path is not None and Path(features_path).exists():
        logger.warning(
            "xgb_parcel_oof_missing_materializing",
            note="oof_parcel_xgb-alphaearth_fold5.parquet absent (US-031 did not "
            "dump it); materializing from tabular features (folds 1-4 -> fold-5).",
        )
        materialize_xgb_parcel_oof(
            Path(features_path),
            out_path=xgb_oof,
            pastis_root=pastis_root,
            random_state=random_state,
        )
        return PARCEL_BASE_MEMBERS

    degraded = tuple(m for m in PARCEL_BASE_MEMBERS if m != XGB_MEMBER)
    logger.warning(
        "xgb_parcel_oof_missing_degraded",
        members=degraded,
        note="oof_parcel_xgb-alphaearth_fold5.parquet absent and not materialized; "
        "stacking/blending degrade to the dense parcel members (R-TAB-FEATURES).",
    )
    return degraded


# ---------------------------------------------------------------------------
# Ensemble runners (each returns metrics + the figure inputs).
# ---------------------------------------------------------------------------


def _aligned_labels(parcel_ids: Sequence[str], gt_labels: pl.DataFrame) -> np.ndarray:
    """Align the GT labels to a parcel id order (subset + reorder).

    Args:
        parcel_ids: Canonical parcel ids (ParcelIDs space) to look up, in order.
        gt_labels: Per-parcel GT frame with ``canonical_parcel_id`` + ``label``.

    Returns:
        An ``int64`` array of labels aligned 1:1 with ``parcel_ids``.

    Raises:
        KeyError: if any parcel id is absent from the GT, with a diagnostic that
            surfaces a canonical-id namespace mismatch (instance-id vs ParcelIDs)
            instead of a bare missing-key error.
    """
    lookup = dict(
        zip(
            gt_labels[_KEY].cast(pl.Utf8).to_list(),
            gt_labels["label"].to_list(),
            strict=True,
        )
    )
    missing = [str(p) for p in parcel_ids if str(p) not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} parcel id(s) are absent from the GT lookup "
            f"(e.g. {missing[:5]}); the canonical_parcel_id namespaces do not "
            "match (the GT keys by the ParcelIDs raster -- translate tabular "
            "instance-id keys via tabular_parcel_keys before aligning)."
        )
    return np.asarray([lookup[str(p)] for p in parcel_ids], dtype=np.int64)


# ---------------------------------------------------------------------------
# Typer command.
# ---------------------------------------------------------------------------


@app.command()
def run(
    oof_dir: Path = typer.Option(Path("ml/eval/oof"), help="US-031 OOF directory."),
    pastis_root: Path = typer.Option(
        Path("data/PASTIS-R"), help="PASTIS-R root (ground truth + geometry)."
    ),
    features: Path = typer.Option(
        Path("data/features/features_fused_pastis.parquet"),
        help="Tabular AlphaEarth features for Bagging + xgb materialization.",
    ),
    out_dir: Path = typer.Option(
        Path("reports/ensemble"), help="Output dir for figures + comparison table."
    ),
    fold: int = typer.Option(HELD_OUT_FOLD, help="Report fold; MUST be 5 (anti-leakage)."),
    n_bags: int = typer.Option(10, help="Bagging bootstrap count."),
    n_trials_bagging: int = typer.Option(30, help="Bagging Optuna trials."),
    n_trials_blending: int = typer.Option(50, help="Blending Optuna trials."),
    materialize_xgb: bool = typer.Option(
        True, help="Materialize the missing xgb-alphaearth parcel OOF from features."
    ),
    use_mlflow: bool = typer.Option(True, help="Log one MLflow run per ensemble (:5010)."),
    random_state: int = typer.Option(42, help="Deterministic seed."),
) -> None:
    """Run the four ensembles, build the comparison table and the figures.

    Rejects any ``fold != 5`` immediately (anti-leakage R-LEAK). Writes the
    comparison table (parquet + CSV) and the >=4 interpretable figures under
    ``out_dir`` and logs one MLflow run per ensemble with the ``chosen_model``
    tag on the winner.
    """
    if fold != HELD_OUT_FOLD:
        raise typer.BadParameter(
            f"--fold must be {HELD_OUT_FOLD} (anti-leakage): fold-4 was the SELECTION "
            "fold and must never be reported."
        )

    figures_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from ml.ensemble.bagging import BaggingEnsemble
    from ml.ensemble.base import EnsembleModel
    from ml.ensemble.blending import BlendingEnsemble
    from ml.ensemble.stacking import StackingEnsemble
    from ml.ensemble.voting import VotingEnsemble

    # Resolve fold-5 patch ids + GT + geometry (GT is NOT in the OOF dump).
    parcel_gt = build_parcel_ground_truth(_fold5_patch_ids(oof_dir), pastis_root)
    parcel_geoms = build_parcel_geometries(_fold5_patch_ids(oof_dir), pastis_root)

    parcel_members = resolve_parcel_members(
        oof_dir,
        materialize_xgb=materialize_xgb,
        features_path=features,
        pastis_root=pastis_root,
        random_state=random_state,
    )

    results: dict[str, dict[str, float]] = {
        INDIVIDUAL_BASELINE_NAME: {
            "f1_macro": INDIVIDUAL_BASELINE_F1,
            "accuracy": float("nan"),
            "inference_time_s": float("nan"),
        }
    }

    # E1 Voting (pixel) -------------------------------------------------------
    voting = VotingEnsemble(VOTING_MEMBERS, oof_dir=oof_dir, random_state=random_state)
    patch_ids = _fold5_patch_ids(oof_dir)
    _proba_voting, t_vote = EnsembleModel.timed_predict(voting.predict_proba, patch_ids)
    voting_metrics = voting.evaluate_patches(patch_ids, fold=fold)
    voting_metrics["inference_time_s"] = t_vote
    results["E1 Voting (pixel)"] = voting_metrics
    if use_mlflow:
        voting.log_to_mlflow(
            voting_metrics,
            run_name="e1-voting",
            params={"members": ",".join(VOTING_MEMBERS)},
            inference_time_s=t_vote,
        )

    # E2 Bagging (parcel, tabular) -------------------------------------------
    # Remap the raw PASTIS class_id (1..18) to the contiguous semantic18 space
    # [0..17] so Bagging's predicted columns match the dense OOF / GT class space
    # (Bagging is class-agnostic: it learns whatever class_id it is handed).
    df_tabular = _remap_tabular_class_id(pl.read_parquet(features))
    bagging = BaggingEnsemble(
        n_bags=n_bags, n_trials=n_trials_bagging, oof_dir=oof_dir, random_state=random_state
    ).fit(df_tabular)
    fold5_tabular = df_tabular.filter(pl.col("fold") == HELD_OUT_FOLD)
    proba_bag, t_bag = EnsembleModel.timed_predict(bagging.predict_proba, fold5_tabular)
    # Translate the tabular instance-id keys to the canonical ParcelIDs space so
    # they align with the GT (the dense OOF / GT key by the ParcelIDs raster).
    bag_keys = tabular_parcel_keys(fold5_tabular, pastis_root)
    bag_labels = _aligned_labels(bag_keys, parcel_gt)
    bag_metrics = bagging.evaluate(y_true=bag_labels, proba=proba_bag, fold=fold)
    bag_metrics["inference_time_s"] = t_bag
    results["E2 Bagging (parcela)"] = bag_metrics
    if use_mlflow:
        bagging.log_to_mlflow(
            bag_metrics,
            run_name="e2-bagging",
            params={"n_bags": n_bags, "n_trials": n_trials_bagging, **bagging.best_params},
            inference_time_s=t_bag,
        )

    # E3 Stacking (parcel, OOF-only meta) ------------------------------------
    stacking = StackingEnsemble(
        parcel_members, meta="logreg", oof_dir=oof_dir, random_state=random_state
    ).fit(parcel_geoms, gt_labels=parcel_gt)
    proba_stack, t_stack = EnsembleModel.timed_predict(stacking.predict_proba)
    stack_keys, _, _ = stacking.build_meta_features(gt_labels=None)
    stack_labels = _aligned_labels(stack_keys[_KEY].to_list(), parcel_gt)
    stack_metrics = stacking.evaluate(y_true=stack_labels, proba=proba_stack, fold=fold)
    stack_metrics["inference_time_s"] = t_stack
    results["E3 Stacking (parcela)"] = stack_metrics
    if use_mlflow:
        stacking.log_to_mlflow(
            stack_metrics,
            run_name="e3-stacking",
            params={"members": ",".join(parcel_members), "meta": "logreg"},
            inference_time_s=t_stack,
        )

    # E4 Blending (parcel, Optuna simplex) -----------------------------------
    geoms_gdf = _geoms_for_blending(parcel_geoms)
    blending = BlendingEnsemble(
        parcel_members, n_trials=n_trials_blending, oof_dir=oof_dir, random_state=random_state
    ).fit(geoms_gdf, y_true=parcel_gt)
    proba_blend, t_blend = EnsembleModel.timed_predict(blending.predict_proba)
    blend_labels = _aligned_labels(blending._member_ids, parcel_gt)
    blend_metrics = blending.evaluate(y_true=blend_labels, proba=proba_blend, fold=fold)
    blend_metrics["inference_time_s"] = t_blend
    results["E4 Blending (parcela)"] = blend_metrics
    if use_mlflow:
        blending.log_to_mlflow(
            blend_metrics,
            run_name="e4-blending",
            params=blending.mlflow_params(),
            inference_time_s=t_blend,
        )

    # Comparison table + chosen model ----------------------------------------
    table = build_comparison_table(results)
    table_parquet = metrics_dir / "comparison_us040.parquet"
    table_csv = metrics_dir / "comparison_us040.csv"
    table.write_parquet(table_parquet)
    table.write_csv(table_csv)
    chosen_row = table.filter(pl.col("chosen"))
    chosen_model = chosen_row["model"][0] if chosen_row.height else INDIVIDUAL_BASELINE_NAME

    # Re-tag the chosen ensemble in MLflow (if it is one of the four). This is a
    # cosmetic post-step (the comparison CSV already marks the chosen model and the
    # per-ensemble runs are already logged): a failure here (e.g. the VM MLflow
    # artifact-root scheme rejecting the upload) must NOT abort the pipeline before
    # the figures are emitted.
    if use_mlflow:
        try:
            _retag_chosen(chosen_model, results, oof_dir, random_state)
        except Exception as exc:  # noqa: BLE001 - cosmetic retag, never fatal
            logger.warning("retag_chosen_failed_skipping", chosen=chosen_model, error=str(exc))

    # Figures (>=4 interpretable) on the BEST ensemble's predictions ---------
    _emit_figures(
        chosen_model,
        results,
        figures_dir,
        proba_stack=proba_stack,
        stack_keys=stack_keys,
        stack_labels=stack_labels,
        proba_blend=proba_blend,
        blend_ids=blending._member_ids,
        blend_labels=blend_labels,
        parcel_geoms=parcel_geoms,
    )

    logger.info("us040_run_done", chosen=chosen_model, table=str(table_csv))
    sys.stdout.buffer.write(table.write_csv().encode("utf-8"))


def _fold5_patch_ids(oof_dir: Path) -> list[str]:
    """Read the fold-5 patch ids from a dense member's pixel OOF manifest.

    The patch ids are the same across members (US-031 scored the same fold-5
    patches), so the first available dense member's parquet is read for its
    ``patch_id`` column.

    Args:
        oof_dir: OOF directory.

    Returns:
        The ordered fold-5 patch ids.

    Raises:
        FileNotFoundError: if no dense member pixel OOF exists.
    """
    for member in VOTING_MEMBERS:
        path = oof_dir / f"oof_{member}_fold{HELD_OUT_FOLD}.parquet"
        if path.exists():
            frame = pl.read_parquet(path, columns=["patch_id"])
            return frame["patch_id"].cast(pl.Utf8).to_list()
    raise FileNotFoundError(f"no dense pixel OOF found in {oof_dir}; run `dvc pull {oof_dir}`.")


def _geoms_for_blending(parcel_geoms: pl.DataFrame):  # type: ignore[no-untyped-def]
    """Convert the parcel geometry frame to the GeoDataFrame blending expects.

    Blending requires a GeoDataFrame with an integer ``parcel_id`` surrogate, the
    ``canonical_parcel_id`` and an active geometry in EPSG:4326.

    Args:
        parcel_geoms: Polars frame with ``canonical_parcel_id`` + WKT geometry.

    Returns:
        A ``geopandas.GeoDataFrame`` ready for ``BlendingEnsemble.fit``.
    """
    import geopandas as gpd
    from shapely import wkt

    pdf = parcel_geoms.to_pandas()
    pdf["geometry"] = pdf["geometry"].map(wkt.loads)
    gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    gdf["parcel_id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)
    return gdf


def _retag_chosen(
    chosen_model: str,
    results: dict[str, dict[str, float]],
    oof_dir: Path,
    random_state: int,
) -> None:
    """Open a tiny MLflow run tagging the chosen ensemble (Selection criterion).

    Args:
        chosen_model: The model elected by :func:`build_comparison_table`.
        results: The per-model metrics (to log the chosen run's metrics).
        oof_dir: OOF dir (forwarded for the data_version tag).
        random_state: Seed (unused beyond reproducibility symmetry).
    """
    run_map = {
        "E1 Voting (pixel)": "e1-voting",
        "E2 Bagging (parcela)": "e2-bagging",
        "E3 Stacking (parcela)": "e3-stacking",
        "E4 Blending (parcela)": "e4-blending",
    }
    run_name = run_map.get(chosen_model)
    if run_name is None:
        logger.info("chosen_is_individual_baseline", chosen=chosen_model)
        return
    from ml.ensemble.base import EnsembleModel

    EnsembleModel(oof_dir=oof_dir, random_state=random_state).log_to_mlflow(  # type: ignore[abstract]
        results[chosen_model],
        run_name=f"{run_name}-chosen",
        chosen=True,
    )


def _emit_figures(
    chosen_model: str,
    results: dict[str, dict[str, float]],
    figures_dir: Path,
    *,
    proba_stack: np.ndarray,
    stack_keys: pl.DataFrame,
    stack_labels: np.ndarray,
    proba_blend: np.ndarray,
    blend_ids: Sequence[str],
    blend_labels: np.ndarray,
    parcel_geoms: pl.DataFrame,
) -> None:
    """Emit the >=4 interpretable figures on the best parcel ensemble.

    The confusion / ROC / PR / spatial-residual figures are computed on the
    Stacking parcel predictions (typically the best of the four, plan Section
    1.2); the comparison barplot is implicit in the table. All inputs are fold-5
    only (anti-leakage). Uses PASTIS class names when available.

    Args:
        chosen_model: Name of the elected model (titles only).
        results: Per-model metrics (unused beyond context logging).
        figures_dir: Output directory for the PNGs.
        proba_stack: Stacking post-softmax ``(n, 18)`` (fold-5).
        stack_keys: Stacking parcel id frame aligned with ``proba_stack``.
        stack_labels: Stacking ground-truth labels aligned with ``proba_stack``.
        proba_blend: Blending post-softmax ``(n, 18)`` (fold-5).
        blend_ids: Blending parcel ids aligned with ``proba_blend``.
        blend_labels: Blending ground-truth labels aligned with ``proba_blend``.
        parcel_geoms: Per-parcel geometry frame (for the residual map).
    """
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES

    y_pred_stack = proba_stack.argmax(axis=1)
    confusion_norm_abs(
        stack_labels,
        y_pred_stack,
        labels=PASTIS_R_CLASSES,
        out_path=figures_dir / "confusion_stacking.png",
        model="E3 Stacking",
    )
    roc_ovr_per_class(
        stack_labels,
        proba_stack,
        labels=PASTIS_R_CLASSES,
        out_path=figures_dir / "roc_ovr_stacking.png",
        model="E3 Stacking",
    )
    pr_per_class(
        stack_labels,
        proba_stack,
        labels=PASTIS_R_CLASSES,
        out_path=figures_dir / "pr_stacking.png",
        model="E3 Stacking",
    )

    # Residual map on the blending predictions (geometry aligned by id order).
    geom_lookup = dict(
        zip(parcel_geoms[_KEY].to_list(), parcel_geoms["geometry"].to_list(), strict=True)
    )
    residual_geoms = pl.DataFrame(
        {
            _KEY: list(blend_ids),
            "geometry": [geom_lookup[str(p)] for p in blend_ids],
        }
    )
    spatial_residuals(
        residual_geoms,
        blend_labels,
        proba_blend.argmax(axis=1),
        out_path=figures_dir / "spatial_residuals_blending.png",
        model="E4 Blending",
    )
    logger.info("us040_figures_emitted", chosen=chosen_model, dir=str(figures_dir))


if __name__ == "__main__":
    app()
