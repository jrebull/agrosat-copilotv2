"""Pure curriculum logic for the full incremental FarSLIP fine-tune (US-036-a).

This module isolates the GPU-free, testable logic of the cardinality curriculum
that drives :mod:`scripts.run_us036a_farslip_full_incremental`:

    - :func:`cardinality_ranking`: PASTIS class_ids ordered by descending pixel
      cardinality (the EDA golden order),
    - :func:`class_ids_for_step`: the active class_ids at a curriculum step
      (step 0 = the 4 dominant classes; ``+step_size`` per step, always a
      superset of the previous step, clamped to 18),
    - :func:`select_step_prototypes`: filters the US-033 ``(18, 384)`` prototype
      matrix down to the step's classes (PASTIS direct, no CAP expansion), with
      row order matching the step's class_ids so it aligns with the loss
      ``category_id`` indexing,
    - :class:`StepMetrics`: the per-step evaluation summary (per-class F1/IoU,
      macro-F1, number of well-resolved classes),
    - :func:`stop_criterion`: the four stop reasons that end the curriculum.

Scope (critical, ordered by the user 2026-06-07): ONLY real French PASTIS-R.
No Italian / synthetic / placeholder data, no ``expand_to_cap`` / CAP bridge,
no ``data/farslip_pairs``. PASTIS classes are used directly (1..18) and
``n_regions`` is always 1, so the prototype selection never produces a region
cross-product (the row count equals the step's class count, never 32 or 96).

Project convention: ``numpy`` only at the data boundary; logging via
``structlog``; no pandas; type hints everywhere; docstrings in English.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

from ml.farslip.pastis_pair_dataset import INCREMENTAL_CURRICULUM

logger = structlog.get_logger(__name__)

#: Number of agronomic PASTIS classes (1..18); the curriculum cap.
_N_PASTIS_CROPS: int = 18

#: Default base classes of the first curriculum step (the 4 dominant crops:
#: Meadow=1, Corn=3, Soft winter wheat=2, Grapevine=8).
_DEFAULT_BASE: int = 4

#: Default step size (classes added per curriculum step).
_DEFAULT_STEP_SIZE: int = 2

#: MiniLM prototype dimension of the US-033 parquet (rows are 384-dim).
_MINILM_DIM: int = 384


def cardinality_ranking() -> list[int]:
    """Return PASTIS class ids ordered by descending pixel cardinality.

    Golden order (EDA "Cardinalidad de cultivos PASTIS"): the 4 dominant classes
    first (Meadow=1, Corn=3, Soft winter wheat=2, Grapevine=8), then by
    descending n_pixels down to the tail (<200 px: Potatoes=13, Sorghum=18).

    The order is the single source of truth shared with the US-036 dataset
    builder (:data:`ml.farslip.pastis_pair_dataset.INCREMENTAL_CURRICULUM`), so a
    sample's ``category_id`` (index into ``active_classes(n)``) and the prototype
    row order stay aligned.

    Returns:
        The 18 PASTIS class_ids in descending-cardinality order.
    """
    ranking = list(INCREMENTAL_CURRICULUM)
    if len(ranking) != _N_PASTIS_CROPS or sorted(ranking) != list(range(1, 19)):
        raise ValueError(
            "INCREMENTAL_CURRICULUM must be a permutation of PASTIS class_ids "
            f"1..18; received {ranking}."
        )
    return ranking


def class_ids_for_step(
    step_idx: int,
    *,
    step_size: int = _DEFAULT_STEP_SIZE,
    base: int = _DEFAULT_BASE,
    ranking: list[int] | None = None,
) -> list[int]:
    """Return the PASTIS class ids active at curriculum step ``step_idx``.

    Step 0 = the ``base`` (default 4) dominant classes; each subsequent step adds
    ``step_size`` (default 2) more from the cardinality ranking. The result is
    always a superset of the previous step and is clamped to the 18 PASTIS crops
    (so the last reachable step is the full 18-class set).

    Args:
        step_idx: zero-based curriculum step index.
        step_size: classes added per step (default 2; plan B uses 4).
        base: classes active at step 0 (default 4 dominant).
        ranking: optional class-id ranking override (default
            :func:`cardinality_ranking`).

    Returns:
        The active PASTIS class_ids at ``step_idx``, in ranking order.

    Raises:
        ValueError: if ``step_idx`` is negative, or ``step_size``/``base`` are
            not positive.
    """
    if step_idx < 0:
        raise ValueError(f"step_idx must be >= 0, received {step_idx}.")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, received {step_size}.")
    if base < 1:
        raise ValueError(f"base must be >= 1, received {base}.")
    order = ranking if ranking is not None else cardinality_ranking()
    n_active = min(base + step_idx * step_size, len(order))
    return order[:n_active]


def n_steps(
    *,
    step_size: int = _DEFAULT_STEP_SIZE,
    base: int = _DEFAULT_BASE,
    max_classes: int = _N_PASTIS_CROPS,
) -> int:
    """Return the number of curriculum steps to reach ``max_classes``.

    The count includes step 0 (the ``base`` classes) and every ``+step_size``
    step up to and including the one that first reaches (or clamps to)
    ``max_classes``.

    Args:
        step_size: classes added per step.
        base: classes at step 0.
        max_classes: curriculum cap (default 18).

    Returns:
        The number of steps ``k`` such that ``class_ids_for_step(k-1)`` first
        reaches ``min(max_classes, 18)``.

    Raises:
        ValueError: if ``step_size``/``base`` are not positive or
            ``max_classes`` is out of ``[base, 18]``.
    """
    if step_size < 1 or base < 1:
        raise ValueError("step_size and base must be >= 1.")
    cap = min(int(max_classes), _N_PASTIS_CROPS)
    if cap < base:
        raise ValueError(f"max_classes ({max_classes}) must be >= base ({base}).")
    remaining = cap - base
    extra_steps = (remaining + step_size - 1) // step_size
    return 1 + extra_steps


def select_step_prototypes(
    proto_18: np.ndarray,
    class_ids_all: list[int],
    class_ids_step: list[int],
) -> np.ndarray:
    """Select the US-033 prototypes for the step's classes (n_regions=1).

    Filters the ``(18, 384)`` parquet matrix by ``class_ids_step`` (PASTIS
    direct, no CAP expansion). Returns ``(len(class_ids_step), 384)``; row order
    matches ``class_ids_step`` so it aligns with the loss ``category_id``
    indexing (``category_id`` = index of the dominant class inside the step's
    active set). It NEVER calls Gemini and NEVER regenerates the parquet: it only
    reorders/filters the in-memory matrix.

    Args:
        proto_18: ``(18, 384)`` float matrix loaded from the US-033 parquet (one
            row per PASTIS class, in ``class_ids_all`` order).
        class_ids_all: the class_ids of ``proto_18`` rows (the parquet order, as
            returned by ``load_class_prototype_embeddings``).
        class_ids_step: the active class_ids of the step, in curriculum order.

    Returns:
        ``(len(class_ids_step), 384)`` float32 matrix; ``row r`` is the US-033
        prototype of ``class_ids_step[r]``.

    Raises:
        ValueError: if ``proto_18`` is not 2-D with as many rows as
            ``class_ids_all``, if its dimension is not 384, or if any step class
            is missing from ``class_ids_all``.
    """
    matrix = np.asarray(proto_18, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"proto_18 must be 2-D (N, 384); received shape {matrix.shape}.")
    if matrix.shape[0] != len(class_ids_all):
        raise ValueError(
            f"proto_18 rows ({matrix.shape[0]}) must equal len(class_ids_all) "
            f"({len(class_ids_all)})."
        )
    if matrix.shape[1] != _MINILM_DIM:
        raise ValueError(f"proto_18 dim ({matrix.shape[1]}) must be {_MINILM_DIM} (MiniLM).")
    row_of: dict[int, int] = {int(cid): row for row, cid in enumerate(class_ids_all)}
    missing = [c for c in class_ids_step if c not in row_of]
    if missing:
        raise ValueError(
            f"prototype matrix is missing step class_ids {missing}; available={sorted(row_of)}."
        )
    rows = [row_of[c] for c in class_ids_step]
    selected = np.ascontiguousarray(matrix[rows], dtype=np.float32)
    logger.info(
        "step_prototypes_selected",
        proto_source="pastis_direct",
        n_regions=1,
        n_protos=int(selected.shape[0]),
        proto_dim_in=int(selected.shape[1]),
        class_ids_step=list(class_ids_step),
    )
    return selected


@dataclass
class StepMetrics:
    """Per-step evaluation summary of the incremental curriculum.

    Attributes:
        n_classes: number of active classes at this step.
        class_ids: active PASTIS class_ids (curriculum order).
        per_class_f1: ``{class_id: F1}`` of the resulting student.
        per_class_iou: ``{class_id: IoU}`` of the resulting student.
        macro_f1: mean F1 over the step's classes.
        macro_iou: mean IoU over the step's classes.
        n_eval: number of validation pairs the metrics were computed on.
        f1_well_resolved: F1 threshold for "well-resolved" (default 0.50).
    """

    n_classes: int
    class_ids: list[int]
    per_class_f1: dict[int, float] = field(default_factory=dict)
    per_class_iou: dict[int, float] = field(default_factory=dict)
    macro_f1: float = 0.0
    macro_iou: float = 0.0
    n_eval: int = 0
    f1_well_resolved: float = 0.50

    @property
    def n_classes_well_resolved(self) -> int:
        """Number of step classes with ``F1 >= f1_well_resolved``."""
        return sum(
            1 for cid in self.class_ids if self.per_class_f1.get(cid, 0.0) >= self.f1_well_resolved
        )

    def new_class_ids(self, prev: StepMetrics | None) -> list[int]:
        """Return the class_ids introduced at this step versus ``prev``.

        Args:
            prev: the previous step's metrics (``None`` for step 0).

        Returns:
            The class_ids present in ``self`` but not in ``prev`` (all of
            ``self.class_ids`` when ``prev`` is ``None``).
        """
        if prev is None:
            return list(self.class_ids)
        prev_set = set(prev.class_ids)
        return [c for c in self.class_ids if c not in prev_set]

    def macro_f1_over(self, class_ids: list[int]) -> float:
        """Mean F1 restricted to ``class_ids`` (0.0 if the subset is empty)."""
        scores = [self.per_class_f1.get(c, 0.0) for c in class_ids]
        return float(np.mean(scores)) if scores else 0.0


def compute_macro(per_class: dict[int, float], class_ids: list[int]) -> float:
    """Return the mean of ``per_class`` over ``class_ids`` (0.0 if empty)."""
    scores = [per_class.get(c, 0.0) for c in class_ids]
    return float(np.mean(scores)) if scores else 0.0


def stop_criterion(
    metrics_curr: StepMetrics,
    metrics_prev: StepMetrics | None,
    *,
    f1_well_resolved: float = 0.50,
    f1_new_unacceptable: float = 0.30,
    prev_degradation_margin: float = 0.05,
    max_classes: int = _N_PASTIS_CROPS,
) -> tuple[bool, str]:
    """Decide whether to stop the curriculum after a step.

    Stop reasons (any triggers a stop, evaluated in this order):

    1. ``new_classes_unacceptable``: none of the classes NEW to this step reaches
       ``F1 >= f1_new_unacceptable`` (the contrastive cannot anchor the new
       crops; adding more would only degrade the rest).
    2. ``prev_classes_degraded``: the macro-F1 over the PREVIOUS step's classes,
       measured on the current student, drops more than ``prev_degradation_margin``
       below the previous step's own macro-F1 over those classes (catastrophic
       forgetting of already-learned crops).
    3. ``max_classes_reached``: the full ``max_classes`` set is active and the
       step converged (success; nothing left to add).

    Note on order: ``new_classes_unacceptable`` and ``prev_classes_degraded`` are
    checked BEFORE ``max_classes_reached`` so that a final 18-class step that
    actually collapses is reported honestly (degraded / unacceptable) rather than
    masked as a success. ``budget_exhausted`` is NOT decided here: it is the
    orchestrator's responsibility (time / step cap), since it depends on wall
    clock, not on metrics.

    Args:
        metrics_curr: the current step's metrics.
        metrics_prev: the previous step's metrics (``None`` for step 0).
        f1_well_resolved: F1 threshold for "well-resolved" (passed to
            ``StepMetrics`` consumers; not used directly here).
        f1_new_unacceptable: min F1 a new class must reach to be acceptable.
        prev_degradation_margin: max tolerated absolute drop of the previous
            classes' macro-F1.
        max_classes: curriculum cap (default 18).

    Returns:
        ``(stop, reason)`` where ``reason`` is one of ``new_classes_unacceptable``,
        ``prev_classes_degraded``, ``max_classes_reached`` or ``continue``.
    """
    new_ids = metrics_curr.new_class_ids(metrics_prev)
    # (1) New classes unacceptable: not a single new crop reaches the floor.
    if new_ids:
        best_new_f1 = max(metrics_curr.per_class_f1.get(c, 0.0) for c in new_ids)
        if best_new_f1 < f1_new_unacceptable:
            logger.info(
                "stop_criterion_triggered",
                reason="new_classes_unacceptable",
                best_new_f1=best_new_f1,
                f1_new_unacceptable=f1_new_unacceptable,
                new_class_ids=new_ids,
            )
            return True, "new_classes_unacceptable"

    # (2) Previous classes degraded: catastrophic forgetting beyond the margin.
    if metrics_prev is not None:
        prev_ids = metrics_prev.class_ids
        prev_macro_then = metrics_prev.macro_f1_over(prev_ids)
        prev_macro_now = metrics_curr.macro_f1_over(prev_ids)
        drop = prev_macro_then - prev_macro_now
        if drop > prev_degradation_margin:
            logger.info(
                "stop_criterion_triggered",
                reason="prev_classes_degraded",
                prev_macro_then=prev_macro_then,
                prev_macro_now=prev_macro_now,
                drop=drop,
                prev_degradation_margin=prev_degradation_margin,
            )
            return True, "prev_classes_degraded"

    # (3) Reached the cap and converged: success.
    if metrics_curr.n_classes >= min(int(max_classes), _N_PASTIS_CROPS):
        logger.info(
            "stop_criterion_triggered",
            reason="max_classes_reached",
            n_classes=metrics_curr.n_classes,
        )
        return True, "max_classes_reached"

    return False, "continue"


__all__ = [
    "StepMetrics",
    "cardinality_ranking",
    "class_ids_for_step",
    "compute_macro",
    "n_steps",
    "select_step_prototypes",
    "stop_criterion",
]
