"""US-079 A/B ablation comparison: warm-start vs no-warm-start (+ original).

Compares the per-class behaviour of the three Italian dense fine-tunes that share
ONE pipeline and differ only in the head init:

- **A_warmstart** (``ablA_warmstart-tsvit-pheno``): the conserved head rows are
  warm-started from the PASTIS (Atlantic France) head -- the kept-class flag of
  :mod:`ml.transfer.italia_label_space`.
- **B_nowarmstart** (``ablB_nowarmstart-tsvit-pheno``): every head row starts at
  its random init (``--no-warm-start``).
- **original** (``us079_v2-tsvit-pheno``): the reference run (its own hyper-params)
  for context, not part of the strict A/B.

The deliverable answers Arthur's hypothesis: does the Atlantic-France prior HURT
the CONSERVED Mediterranean classes? If the no-warm-start arm (B) beats the
warm-start arm (A) on the MAJORITY of the conserved classes, the prior is a
liability and the kept-class flag should be dropped for this domain.

Everything reuses the existing harness so the numbers are apples-to-apples:

- :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` accumulates the pixel
  confusion (the SAME accumulator the EPIC 5 segmentation eval uses).
- :func:`ml.eval.transfer_italia_eval.per_class_f1` derives per-class F1 from that
  confusion; precision/recall/support are read off the same matrix here.
- :func:`ml.transfer.finetune_italia.load_italia_patches` loads the held-out
  ``TARGET_<id>.npy`` masks; :func:`ml.ensemble.voting_italia.load_member_softmax`
  loads each run's ``test_softmax.npz`` (keyed by ``str(patch_id)``, ``(K, H, W)``
  post-softmax).
- :class:`ml.transfer.italia_label_space.ItaliaLabelSpace` carries the conserved
  flag and the fine -> coarse collapse.

Honesty
-------
Only REAL values are produced: a run whose ``test_softmax.npz`` is absent is
reported as missing and OMITTED from every table/figure (never fabricated). The
metrics are computed on the intersection of the run's predicted patches and the
on-disk masks, so a partial dump still scores its real coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.eval.transfer_italia_eval import build_coarse_label_space, per_class_f1
from ml.transfer.italia_label_space import (
    CONSERVED_LEAF_TO_PASTIS,
    ItaliaLabelSpace,
    build_italia_label_space,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = structlog.get_logger(__name__)

__all__ = [
    "PerClassScores",
    "RunPerClass",
    "compare_runs",
    "conserved_class_ids",
    "discard_curve_compare",
    "load_run_masks",
    "per_class_scores_from_softmax",
    "warm_start_ablation_verdict",
]


class PerClassScores:
    """Per-class dense scores of one run (a thin row container).

    Attributes:
        f1: ``{class_id: f1}`` for the classes with ground-truth support.
        precision: ``{class_id: precision}``.
        recall: ``{class_id: recall}``.
        support: ``{class_id: pixel_support}``.
    """

    __slots__ = ("f1", "precision", "recall", "support")

    def __init__(
        self,
        f1: dict[int, float],
        precision: dict[int, float],
        recall: dict[int, float],
        support: dict[int, int],
    ) -> None:
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.support = support


class RunPerClass:
    """A scored run: its per-class scores plus its identity / coverage.

    Attributes:
        name: The run alias (e.g. ``"A_warmstart"``).
        level: ``"fine"`` or ``"coarse"`` (the granularity the scores live at).
        scores: The :class:`PerClassScores` at ``level``.
        id_to_name: ``{class_id: class_name}`` at ``level`` (fine leaf or coarse
            bucket name).
        n_patches: Number of patches actually scored (intersection of preds and
            masks).
    """

    __slots__ = ("id_to_name", "level", "n_patches", "name", "scores")

    def __init__(
        self,
        name: str,
        level: str,
        scores: PerClassScores,
        id_to_name: dict[int, str],
        n_patches: int,
    ) -> None:
        self.name = name
        self.level = level
        self.scores = scores
        self.id_to_name = id_to_name
        self.n_patches = n_patches


def load_run_masks(
    dataset_root: Path,
    *,
    n_timesteps: int = 10,
) -> dict[int, np.ndarray]:
    """Load every on-disk Italian ``TARGET_<id>.npy`` mask keyed by patch id.

    The masks are split-agnostic (a run's ``test_softmax.npz`` already restricts
    which patches are scored); loading all of them lets a run with any held-out
    fold be scored against the real ground truth on its own patch set.

    Args:
        dataset_root: The US-078 homologue dataset root (``data/pastis_italia_2018``).
        n_timesteps: Equispaced dates forwarded to ``load_italia_patches`` (only the
            masks are used, so the value does not affect the result; kept for the
            loader contract).

    Returns:
        ``{patch_id: (H, W)}`` int64 ground-truth class masks (0 = background).
    """
    from ml.transfer.finetune_italia import load_italia_patches

    patches = load_italia_patches(italia_root=dataset_root, n_timesteps=n_timesteps)
    masks = {pid: patches.masks[i].astype(np.int64) for i, pid in enumerate(patches.patch_ids)}
    logger.info("us079_run_masks_loaded", n_masks=len(masks), root=str(dataset_root))
    return masks


def _collapse_to_coarse(arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map a fine-id array onto its coarse ids through ``lut``.

    Args:
        arr: An int array of fine class ids in ``[0, num_fine)``.
        lut: The fine-id -> coarse-id lookup table (``build_coarse_label_space``).

    Returns:
        The same-shaped array of coarse ids.
    """
    flat = arr.reshape(-1)
    flat = np.clip(flat, 0, lut.shape[0] - 1)
    return lut[flat].reshape(arr.shape)


def per_class_scores_from_softmax(
    softmax_npz_path: Path,
    masks_by_patch: Mapping[int, np.ndarray],
    *,
    label_space: ItaliaLabelSpace,
    level: str = "fine",
) -> tuple[PerClassScores, dict[int, str], int]:
    """Compute per-class F1 / precision / recall / support of one run.

    Loads the run's dense post-softmax maps (``test_softmax.npz``, keyed by
    ``str(patch_id)``, ``(K, H, W)``), argmaxes them into dense class maps,
    accumulates the pixel confusion against the matching ground-truth masks (the
    intersection of the run's patches and ``masks_by_patch``), optionally collapses
    BOTH preds and target to the coarse PASTIS-shared buckets, and derives the
    per-class scores from the accumulated confusion. The background id is excluded
    (``ignore_index``), matching the segmentation harness.

    Args:
        softmax_npz_path: Path to the run's ``test_softmax.npz``.
        masks_by_patch: ``{patch_id: (H, W)}`` ground-truth fine class masks.
        label_space: The Italian fine label space.
        level: ``"fine"`` (Italian leaves) or ``"coarse"`` (PASTIS-shared buckets).

    Returns:
        ``(scores, id_to_name, n_patches)`` where ``id_to_name`` maps the class id
        at ``level`` to its name and ``n_patches`` is the number of scored patches.

    Raises:
        FileNotFoundError: if the ``.npz`` is absent.
        ValueError: for an unsupported ``level`` or an empty intersection.
    """
    from ml.ensemble.voting_italia import load_member_softmax

    if level not in ("fine", "coarse"):
        raise ValueError(f"level must be 'fine' or 'coarse', got {level!r}")

    preds = load_member_softmax(softmax_npz_path.stem, Path(softmax_npz_path))
    shared = sorted(set(preds.probs_by_patch) & set(masks_by_patch))
    if not shared:
        raise ValueError(
            f"no shared patch between {softmax_npz_path} and the on-disk masks; "
            "the run's test_softmax.npz patch ids do not match the dataset."
        )

    bg = label_space.background_id
    if level == "fine":
        num_classes = label_space.num_classes
        ignore_index = bg
        id_to_name = label_space.id_to_leaf()
        lut = None
    else:
        lut, coarse_names = build_coarse_label_space(label_space)
        num_classes = len(coarse_names)
        ignore_index = 0  # coarse background id is 0 by construction.
        id_to_name = dict(coarse_names)

    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    for pid in shared:
        pred_map = preds.probs_by_patch[pid].argmax(axis=0).astype(np.int64)
        target = np.asarray(masks_by_patch[pid], dtype=np.int64)
        if lut is not None:
            pred_map = _collapse_to_coarse(pred_map, lut)
            target = _collapse_to_coarse(target, lut)
        acc.update(pred_map, target)

    confusion = acc.confusion_matrix().astype(np.float64)
    scores = _scores_from_confusion(confusion, ignore_index=ignore_index)
    logger.info(
        "us079_run_scored",
        npz=str(softmax_npz_path),
        level=level,
        n_patches=len(shared),
        n_classes_with_support=len(scores.f1),
    )
    return scores, id_to_name, len(shared)


def _scores_from_confusion(confusion: np.ndarray, *, ignore_index: int | None) -> PerClassScores:
    """Derive per-class F1 / precision / recall / support from a confusion matrix.

    Rows = ground truth, cols = prediction. F1 reuses
    :func:`ml.eval.transfer_italia_eval.per_class_f1` (single source of truth);
    precision / recall / support are read off the same matrix so they stay
    consistent with that F1 (``f1 = 2PR/(P+R)`` holds exactly).

    Args:
        confusion: The ``(C, C)`` pixel confusion matrix.
        ignore_index: Class excluded from the result (background).

    Returns:
        A :class:`PerClassScores` over the classes with support (>0).
    """
    diag = np.diag(confusion)
    row_sum = confusion.sum(axis=1)  # ground-truth support per class
    col_sum = confusion.sum(axis=0)  # predictions per class
    precision_all = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
    recall_all = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)

    f1 = per_class_f1(confusion, ignore_index=ignore_index)
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    support: dict[int, int] = {}
    for cid in f1:
        precision[cid] = float(precision_all[cid])
        recall[cid] = float(recall_all[cid])
        support[cid] = int(row_sum[cid])
    return PerClassScores(f1=f1, precision=precision, recall=recall, support=support)


def compare_runs(
    runs: Mapping[str, Path],
    dataset_root: Path,
    *,
    level: str = "fine",
    n_timesteps: int = 10,
    masks_by_patch: Mapping[int, np.ndarray] | None = None,
) -> tuple[pl.DataFrame, dict[str, RunPerClass]]:
    """Score 2+ runs side by side and return a per-class comparison table.

    Each run is scored independently (only REAL ``test_softmax.npz`` files are
    loaded); a run whose path is missing is logged and skipped (never fabricated).
    The returned table has one row per class present in ANY scored run, with the
    F1 of every scored run side by side plus the pairwise delta
    ``B_nowarmstart - A_warmstart`` (when both arms are present) and
    ``run - original`` deltas.

    Args:
        runs: ``{alias: path}`` where ``path`` is the run's ``test_softmax.npz`` or
            the directory containing it. Canonical aliases:
            ``{"A_warmstart", "B_nowarmstart", "original"}``.
        dataset_root: The US-078 homologue dataset root (for the masks).
        level: ``"fine"`` or ``"coarse"``.
        n_timesteps: Forwarded to the mask loader.
        masks_by_patch: Pre-loaded masks (optional); loaded from ``dataset_root``
            when ``None``.

    Returns:
        ``(table, scored)`` where ``table`` is a Polars DataFrame keyed by
        ``class_id`` / ``class_name`` / ``is_conserved`` / ``support`` with an
        ``f1_<alias>`` column per scored run (and the deltas), and ``scored`` is
        ``{alias: RunPerClass}`` for the runs that produced real scores.
    """
    if masks_by_patch is None:
        masks_by_patch = load_run_masks(dataset_root, n_timesteps=n_timesteps)
    label_space = build_italia_label_space(italia_root=dataset_root)
    conserved_ids = conserved_class_ids(label_space, level=level)

    scored: dict[str, RunPerClass] = {}
    for alias, raw_path in runs.items():
        npz_path = _resolve_softmax_path(raw_path)
        if npz_path is None:
            logger.warning(
                "us079_run_pending",
                alias=alias,
                path=str(raw_path),
                note="test_softmax.npz absent; arm omitted (not fabricated).",
            )
            continue
        scores, id_to_name, n_patches = per_class_scores_from_softmax(
            npz_path, masks_by_patch, label_space=label_space, level=level
        )
        scored[alias] = RunPerClass(alias, level, scores, id_to_name, n_patches)

    table = _build_comparison_table(scored, conserved_ids=conserved_ids)
    logger.info(
        "us079_compare_runs",
        level=level,
        scored_runs=list(scored),
        n_classes=table.height,
    )
    return table, scored


def _resolve_softmax_path(raw_path: Path) -> Path | None:
    """Resolve a run reference to its ``test_softmax.npz`` (or ``None`` if absent).

    Accepts either the ``.npz`` file directly or a directory holding it. Returns
    ``None`` when nothing usable exists (the caller reports the arm as pending).

    Args:
        raw_path: A ``.npz`` path or a run directory.

    Returns:
        The resolved ``test_softmax.npz`` path, or ``None``.
    """
    path = Path(raw_path)
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "test_softmax.npz"
        return candidate if candidate.is_file() else None
    # A non-existent .npz path (the run has not finished yet).
    return None


def _build_comparison_table(
    scored: Mapping[str, RunPerClass],
    *,
    conserved_ids: set[int],
) -> pl.DataFrame:
    """Assemble the per-class side-by-side table from the scored runs.

    Args:
        scored: ``{alias: RunPerClass}`` of the runs with real scores.
        conserved_ids: The class ids flagged conserved at this level.

    Returns:
        A Polars DataFrame with ``class_id``, ``class_name``, ``is_conserved``,
        ``support``, an ``f1_<alias>`` / ``precision_<alias>`` / ``recall_<alias>``
        column per run, and the ``delta_b_minus_a`` / ``delta_*_minus_original``
        columns when the relevant arms are present. One row per class present in
        any run, sorted by ``support`` descending.
    """
    if not scored:
        return pl.DataFrame(
            schema={
                "class_id": pl.Int64,
                "class_name": pl.Utf8,
                "is_conserved": pl.Boolean,
                "support": pl.Int64,
            }
        )

    all_ids = sorted({cid for run in scored.values() for cid in run.scores.f1})
    id_to_name: dict[int, str] = {}
    support: dict[int, int] = {}
    for run in scored.values():
        for cid, name in run.id_to_name.items():
            id_to_name.setdefault(cid, name)
        for cid, sup in run.scores.support.items():
            support[cid] = max(support.get(cid, 0), sup)

    rows: list[dict[str, object]] = []
    for cid in all_ids:
        row: dict[str, object] = {
            "class_id": cid,
            "class_name": id_to_name.get(cid, str(cid)),
            "is_conserved": cid in conserved_ids,
            "support": support.get(cid, 0),
        }
        for alias, run in scored.items():
            row[f"f1_{alias}"] = round(float(run.scores.f1.get(cid, 0.0)), 4)
            row[f"precision_{alias}"] = round(float(run.scores.precision.get(cid, 0.0)), 4)
            row[f"recall_{alias}"] = round(float(run.scores.recall.get(cid, 0.0)), 4)
        rows.append(row)

    table = pl.DataFrame(rows)
    table = _add_delta_columns(table, scored)
    return table.sort("support", descending=True)


def _add_delta_columns(table: pl.DataFrame, scored: Mapping[str, RunPerClass]) -> pl.DataFrame:
    """Add the A/B and vs-original delta columns when both arms are present.

    Args:
        table: The per-class table with the ``f1_<alias>`` columns.
        scored: The scored runs (to know which arms exist).

    Returns:
        The table with ``delta_b_minus_a`` (B no-warmstart minus A warmstart) and
        ``delta_<arm>_minus_original`` columns added where applicable.
    """
    has_a = "A_warmstart" in scored
    has_b = "B_nowarmstart" in scored
    has_orig = "original" in scored
    exprs: list[pl.Expr] = []
    if has_a and has_b:
        exprs.append(
            (pl.col("f1_B_nowarmstart") - pl.col("f1_A_warmstart"))
            .round(4)
            .alias("delta_b_minus_a")
        )
    if has_orig:
        for arm in ("A_warmstart", "B_nowarmstart"):
            if arm in scored:
                exprs.append(
                    (pl.col(f"f1_{arm}") - pl.col("f1_original"))
                    .round(4)
                    .alias(f"delta_{arm}_minus_original")
                )
    return table.with_columns(exprs) if exprs else table


def conserved_class_ids(
    label_space: ItaliaLabelSpace,
    *,
    level: str = "fine",
    conserved: Mapping[str, str] = CONSERVED_LEAF_TO_PASTIS,
) -> set[int]:
    """Return the class ids flagged CONSERVED at the requested granularity.

    At ``fine`` level a conserved id is the dense id of a leaf in ``conserved``
    (the kept-class flag). At ``coarse`` level it is the coarse-bucket id that ANY
    conserved leaf collapses to (those buckets are PASTIS-shared by construction).

    Args:
        label_space: The Italian fine label space.
        level: ``"fine"`` or ``"coarse"``.
        conserved: The Italian-leaf -> PASTIS-name crosswalk (the kept-class flag).

    Returns:
        The set of conserved class ids at ``level``.
    """
    fine_ids = {
        label_space.index[leaf]
        for leaf in label_space.conserved
        if leaf in conserved and leaf in label_space.index
    }
    if level == "fine":
        return fine_ids
    lut, _ = build_coarse_label_space(label_space)
    return {int(lut[cid]) for cid in fine_ids if 0 <= cid < lut.shape[0]}


def warm_start_ablation_verdict(
    table: pl.DataFrame,
    *,
    eps: float = 0.0,
) -> dict[str, object]:
    """Decide whether the PASTIS warm-start HURTS the conserved classes.

    Restricts the comparison to the CONSERVED classes (``is_conserved``) that BOTH
    arms scored, then compares the no-warm-start arm (B) against the warm-start arm
    (A). The verdict ``warm_start_hurts_conserved`` is ``True`` when B beats A on the
    MAJORITY of the conserved classes (strictly more wins than losses), i.e. the
    Atlantic-France prior is a net liability for the Mediterranean conserved crops --
    Arthur's hypothesis.

    Args:
        table: The per-class comparison table from :func:`compare_runs` (must carry
            ``f1_A_warmstart`` and ``f1_B_nowarmstart``).
        eps: Minimum F1 gap to count a class as a win/loss (a ``|delta| <= eps``
            class is a tie and does not count toward either side). Default 0.

    Returns:
        A dict with ``available`` (False when an arm is missing -> the rest are
        ``None``), ``n_conserved_compared``, ``mean_f1_warmstart``,
        ``mean_f1_nowarmstart``, ``mean_delta_b_minus_a``, ``n_conserved_improved``
        (B > A), ``n_conserved_worsened`` (B < A), ``n_conserved_tied``,
        ``warm_start_hurts_conserved`` and ``per_conserved`` (the per-class rows).
    """
    pending = {
        "available": False,
        "reason": "needs both f1_A_warmstart and f1_B_nowarmstart columns "
        "(the no-warm-start arm B is still training); pass --run-b to complete the A/B.",
        "n_conserved_compared": 0,
        "mean_f1_warmstart": None,
        "mean_f1_nowarmstart": None,
        "mean_delta_b_minus_a": None,
        "n_conserved_improved": None,
        "n_conserved_worsened": None,
        "n_conserved_tied": None,
        "warm_start_hurts_conserved": None,
        "per_conserved": [],
    }
    if "f1_A_warmstart" not in table.columns or "f1_B_nowarmstart" not in table.columns:
        logger.info("us079_verdict_pending", columns=table.columns)
        return pending

    conserved = table.filter(pl.col("is_conserved")).sort("support", descending=True)
    if conserved.height == 0:
        return {**pending, "available": True, "reason": "no conserved class with support"}

    a = conserved["f1_A_warmstart"].to_numpy()
    b = conserved["f1_B_nowarmstart"].to_numpy()
    delta = b - a
    improved = int(np.sum(delta > eps))
    worsened = int(np.sum(delta < -eps))
    tied = int(conserved.height - improved - worsened)
    verdict = improved > worsened

    per_conserved = conserved.select(
        ["class_id", "class_name", "support", "f1_A_warmstart", "f1_B_nowarmstart"]
    ).with_columns(
        (pl.col("f1_B_nowarmstart") - pl.col("f1_A_warmstart")).round(4).alias("delta_b_minus_a")
    )
    result = {
        "available": True,
        "reason": "ok",
        "n_conserved_compared": conserved.height,
        "mean_f1_warmstart": round(float(np.mean(a)), 4),
        "mean_f1_nowarmstart": round(float(np.mean(b)), 4),
        "mean_delta_b_minus_a": round(float(np.mean(delta)), 4),
        "n_conserved_improved": improved,
        "n_conserved_worsened": worsened,
        "n_conserved_tied": tied,
        "warm_start_hurts_conserved": bool(verdict),
        "per_conserved": per_conserved.to_dicts(),
    }
    logger.info(
        "us079_warm_start_verdict",
        n_conserved=result["n_conserved_compared"],
        mean_f1_warmstart=result["mean_f1_warmstart"],
        mean_f1_nowarmstart=result["mean_f1_nowarmstart"],
        improved=improved,
        worsened=worsened,
        warm_start_hurts_conserved=result["warm_start_hurts_conserved"],
    )
    return result


def discard_curve_compare(
    scored: Mapping[str, RunPerClass],
) -> dict[str, list[dict[str, object]]]:
    """Overlay the honest discard curve (top-N by F1) of every scored run.

    For each run, ranks its classes by per-class F1 (descending) and reports the
    macro F1 over each top-``n`` prefix -- the SAME discard-curve semantics as
    :func:`ml.eval.transfer_italia_eval.discard_curve`, here computed directly from
    the run's per-class F1 so the three arms can be plotted on one axis.

    Args:
        scored: ``{alias: RunPerClass}`` of the runs with real scores.

    Returns:
        ``{alias: [{"n_classes", "macro_f1", "classes"}, ...]}`` -- one curve per
        run (empty for a run with no class support).
    """
    curves: dict[str, list[dict[str, object]]] = {}
    for alias, run in scored.items():
        ranked = sorted(run.scores.f1.items(), key=lambda kv: kv[1], reverse=True)
        curve: list[dict[str, object]] = []
        for n in range(1, len(ranked) + 1):
            top = ranked[:n]
            macro = float(np.mean([f1 for _, f1 in top]))
            curve.append(
                {
                    "n_classes": n,
                    "macro_f1": round(macro, 4),
                    "classes": [run.id_to_name.get(cid, str(cid)) for cid, _ in top],
                }
            )
        curves[alias] = curve
    logger.info(
        "us079_discard_curve_compare",
        runs={a: len(c) for a, c in curves.items()},
    )
    return curves
