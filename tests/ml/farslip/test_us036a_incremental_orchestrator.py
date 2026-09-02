"""Tests for the US-036-a incremental FarSLIP curriculum (logic + orchestrator).

Zero GPU, zero network, zero real dataset, zero Gemini: the heavy
:class:`FarSLIPDistillationTrainer`, the US-036 dataset builder
(``create_incremental_dataset``) and the per-class eval are MOCKED; the US-033
parquet is synthesized as a distinguishable ``(18, 384)`` matrix. The tests
verify the LOGIC of the curriculum (ranking, steps, prototype selection, stop
criterion) and the ORCHESTRATOR (chained init strict=False, Italian-data
rejection, the productive epoch floor, the eval contract), never an actual run.

Section map (docs/us-planning/us-036-a.md section 6):
    6.1 ranking and steps (AC-3),
    6.2 PASTIS-direct prototypes per step (AC-5),
    6.3 chained init strict=False (AC-4),
    6.4 stop criterion (AC-8),
    6.5 scope guards (AC-1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

import scripts.run_us036a_farslip_full_incremental as orch
from ml.farslip.incremental_curriculum import (
    StepMetrics,
    cardinality_ranking,
    class_ids_for_step,
    n_steps,
    select_step_prototypes,
    stop_criterion,
)

_GOLDEN_RANKING = [1, 3, 2, 8, 4, 14, 5, 16, 10, 6, 15, 7, 11, 9, 12, 17, 13, 18]


def _synthetic_proto_18() -> tuple[np.ndarray, list[int]]:
    """Build a distinguishable ``(18, 384)`` matrix and its class_ids 1..18.

    Row ``r`` is the constant vector ``class_id`` so a selected row is trivially
    identifiable by its value (``proto[r] == class_ids[r]`` everywhere). The
    class_ids are NOT in 1..18 ascending order on purpose (a shuffled order), so
    the test proves ``select_step_prototypes`` maps by class_id, not by position.
    """
    class_ids = list(_GOLDEN_RANKING)  # shuffled order (not 1..18 ascending)
    proto = np.zeros((18, 384), dtype=np.float32)
    for row, cid in enumerate(class_ids):
        proto[row, :] = float(cid)
    return proto, class_ids


# ---------------------------------------------------------------------------
# 6.1 Ranking and steps (AC-3).
# ---------------------------------------------------------------------------


def test_cardinality_ranking_golden() -> None:
    """The ranking equals the EDA golden order (18-class permutation)."""
    assert cardinality_ranking() == _GOLDEN_RANKING


def test_step_0_is_four_dominant() -> None:
    """Step 0 = the 4 dominant classes [1, 3, 2, 8]."""
    ids = class_ids_for_step(0)
    assert ids == [1, 3, 2, 8]
    assert len(ids) == 4


def test_steps_are_supersets_monotone() -> None:
    """Each step is a superset of the previous and grows by step_size; clamps 18."""
    prev: list[int] = []
    for k in range(20):  # well past the cap
        ids = class_ids_for_step(k, step_size=2, base=4)
        assert set(prev).issubset(set(ids)), f"step {k} not a superset of {k - 1}"
        if len(ids) < 18:
            assert len(ids) == 4 + k * 2
        prev = ids
    # The last reachable step is exactly the 18 classes (clamped).
    assert class_ids_for_step(7, step_size=2, base=4) == _GOLDEN_RANKING
    assert class_ids_for_step(50, step_size=2, base=4) == _GOLDEN_RANKING


def test_step_clamp_never_exceeds_18() -> None:
    """A huge step index never returns more than the 18 PASTIS crops."""
    assert len(class_ids_for_step(999, step_size=4, base=4)) == 18


def test_step_size_4_plan_b() -> None:
    """Plan B (+4): 4 -> 8 -> 12 -> 16 -> 18 (clamped)."""
    assert len(class_ids_for_step(0, step_size=4)) == 4
    assert len(class_ids_for_step(1, step_size=4)) == 8
    assert len(class_ids_for_step(2, step_size=4)) == 12
    assert len(class_ids_for_step(3, step_size=4)) == 16
    assert class_ids_for_step(4, step_size=4) == _GOLDEN_RANKING


def test_n_steps_counts() -> None:
    """n_steps covers step 0 plus every +step_size up to the cap (inclusive)."""
    assert n_steps(step_size=2, base=4, max_classes=18) == 8  # 4,6,...,18
    assert n_steps(step_size=4, base=4, max_classes=18) == 5  # 4,8,12,16,18
    assert n_steps(step_size=2, base=4, max_classes=10) == 4  # 4,6,8,10


def test_class_ids_for_step_validates() -> None:
    """Negative index / non-positive params raise."""
    with pytest.raises(ValueError, match="step_idx"):
        class_ids_for_step(-1)
    with pytest.raises(ValueError, match="step_size"):
        class_ids_for_step(0, step_size=0)
    with pytest.raises(ValueError, match="base"):
        class_ids_for_step(0, base=0)


# ---------------------------------------------------------------------------
# 6.2 PASTIS-direct prototypes per step (AC-5).
# ---------------------------------------------------------------------------


def test_select_step_prototypes_shape_and_order() -> None:
    """Selecting [1,3,2,8] yields (4, 384) with rows == those class_ids."""
    proto_18, class_ids_all = _synthetic_proto_18()
    step_ids = [1, 3, 2, 8]
    selected = select_step_prototypes(proto_18, class_ids_all, step_ids)
    assert selected.shape == (4, 384)  # NOT 32, NOT 96 (n_regions=1, no CAP)
    # Row r is the constant-vector of class_ids[r] (filter by class_id, in order).
    for r, cid in enumerate(step_ids):
        assert np.allclose(selected[r], float(cid)), f"row {r} != class {cid}"


def test_select_step_prototypes_respects_arbitrary_order() -> None:
    """Row order follows class_ids_step exactly (R-PROTO-ALIGN guard)."""
    proto_18, class_ids_all = _synthetic_proto_18()
    step_ids = [8, 1, 3]  # arbitrary order
    selected = select_step_prototypes(proto_18, class_ids_all, step_ids)
    assert selected.shape == (3, 384)
    assert np.allclose(selected[0], 8.0)
    assert np.allclose(selected[1], 1.0)
    assert np.allclose(selected[2], 3.0)


def test_select_step_prototypes_no_gemini_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prototype selection never calls Gemini / regenerates the parquet."""
    import ml.features.phenology_class_prototypes as proto_mod

    calls = {"generate": 0, "encode": 0}

    def _boom_generate(*_a: Any, **_k: Any) -> Any:
        calls["generate"] += 1
        raise AssertionError("generate_class_prototypes must not be called")

    def _boom_encode(*_a: Any, **_k: Any) -> Any:
        calls["encode"] += 1
        raise AssertionError("_encode_descriptions must not be called")

    monkeypatch.setattr(proto_mod, "generate_class_prototypes", _boom_generate)
    monkeypatch.setattr(proto_mod, "_encode_descriptions", _boom_encode)

    proto_18, class_ids_all = _synthetic_proto_18()
    select_step_prototypes(proto_18, class_ids_all, [1, 3, 2, 8])
    assert calls == {"generate": 0, "encode": 0}


def test_select_step_prototypes_missing_class_raises() -> None:
    """A step class absent from the parquet is a hard error, not a silent drop."""
    proto_18, class_ids_all = _synthetic_proto_18()
    with pytest.raises(ValueError, match="missing step class_ids"):
        select_step_prototypes(proto_18, class_ids_all, [1, 99])


def test_select_step_prototypes_bad_shape_raises() -> None:
    """A non-384 prototype dim is rejected."""
    bad = np.zeros((18, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="384"):
        select_step_prototypes(bad, list(range(1, 19)), [1, 2])


# ---------------------------------------------------------------------------
# 6.4 Stop criterion (AC-8).
# ---------------------------------------------------------------------------


def _metrics(
    class_ids: list[int],
    f1: dict[int, float],
    *,
    iou: dict[int, float] | None = None,
) -> StepMetrics:
    """Build a StepMetrics from per-class F1 (IoU defaults to F1 for the test)."""
    iou_map = iou if iou is not None else dict(f1)
    macro = float(np.mean([f1.get(c, 0.0) for c in class_ids])) if class_ids else 0.0
    return StepMetrics(
        n_classes=len(class_ids),
        class_ids=class_ids,
        per_class_f1=f1,
        per_class_iou=iou_map,
        macro_f1=macro,
        n_eval=100,
    )


def test_stop_max_classes_reached() -> None:
    """18 healthy classes -> max_classes_reached (success)."""
    ids = _GOLDEN_RANKING
    prev = _metrics(ids[:16], {c: 0.7 for c in ids[:16]})
    curr = _metrics(ids, {c: 0.7 for c in ids})
    assert stop_criterion(curr, prev) == (True, "max_classes_reached")


def test_stop_prev_classes_degraded() -> None:
    """Macro-F1 of the previous classes drops > 0.05 -> prev_classes_degraded."""
    ids_prev = [1, 3, 2, 8]
    ids_curr = [1, 3, 2, 8, 4, 14]
    prev = _metrics(ids_prev, {c: 0.70 for c in ids_prev})
    # New classes are fine (so the unacceptable check passes), but the previous
    # classes collapse from 0.70 to 0.50 (drop 0.20 > 0.05).
    curr_f1 = {c: 0.50 for c in ids_prev}
    curr_f1.update({4: 0.60, 14: 0.60})
    curr = _metrics(ids_curr, curr_f1)
    assert stop_criterion(curr, prev) == (True, "prev_classes_degraded")


def test_stop_new_classes_unacceptable() -> None:
    """No new class reaches F1 >= 0.30 -> new_classes_unacceptable."""
    ids_prev = [1, 3, 2, 8]
    ids_curr = [1, 3, 2, 8, 4, 14]
    prev = _metrics(ids_prev, {c: 0.70 for c in ids_prev})
    curr_f1 = {c: 0.70 for c in ids_prev}
    curr_f1.update({4: 0.10, 14: 0.05})  # both new classes below the floor
    curr = _metrics(ids_curr, curr_f1)
    assert stop_criterion(curr, prev) == (True, "new_classes_unacceptable")


def test_continue_when_healthy() -> None:
    """New classes OK and previous stable -> (False, continue)."""
    ids_prev = [1, 3, 2, 8]
    ids_curr = [1, 3, 2, 8, 4, 14]
    prev = _metrics(ids_prev, {c: 0.70 for c in ids_prev})
    curr_f1 = {c: 0.70 for c in ids_prev}
    curr_f1.update({4: 0.55, 14: 0.52})
    curr = _metrics(ids_curr, curr_f1)
    assert stop_criterion(curr, prev) == (False, "continue")


def test_stop_priority_degraded_over_max_classes() -> None:
    """A collapsed final 18-class step is reported degraded, not masked success."""
    ids = _GOLDEN_RANKING
    prev = _metrics(ids[:16], {c: 0.70 for c in ids[:16]})
    # 18 classes but the previous-16 macro collapses -> degraded wins over max.
    curr_f1 = {c: 0.40 for c in ids[:16]}
    curr_f1.update({13: 0.40, 18: 0.40})
    curr = _metrics(ids, curr_f1)
    assert stop_criterion(curr, prev) == (True, "prev_classes_degraded")


def test_n_classes_well_resolved() -> None:
    """Well-resolved count uses the F1 >= 0.50 threshold per class."""
    ids = [1, 3, 2, 8]
    m = _metrics(ids, {1: 0.80, 3: 0.55, 2: 0.40, 8: 0.49})
    assert m.n_classes_well_resolved == 2  # only 1 and 3 clear 0.50


# ---------------------------------------------------------------------------
# 6.5 Scope guards (AC-1).
# ---------------------------------------------------------------------------


def test_rejects_italian_dataset_root() -> None:
    """Pointing at the Italian/synthetic data root is a hard ValueError."""
    with pytest.raises(ValueError, match="farslip_pairs"):
        orch._validate_pastis_root(Path("data/farslip_pairs"))
    with pytest.raises(ValueError, match="farslip_pairs"):
        orch._validate_pastis_root(Path("data") / "farslip_pairs")


def test_accepts_real_pastis_root() -> None:
    """A real PASTIS-R root passes the guard."""
    orch._validate_pastis_root(Path("data/PASTIS-R"))  # no raise


def test_run_rejects_smoke_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """epochs_per_step below 20 is rejected (productive run, not a smoke)."""
    with pytest.raises(ValueError, match="productive floor"):
        orch.run_incremental_curriculum(
            pastis_root=Path("data/PASTIS-R"),
            epochs_per_step=2,
        )


def test_run_rejects_val_fold_leakage() -> None:
    """val_folds overlapping train folds is spatial-CV leakage -> ValueError."""
    with pytest.raises(ValueError, match="leakage"):
        orch.run_incremental_curriculum(
            pastis_root=Path("data/PASTIS-R"),
            epochs_per_step=20,
            folds=(1, 2, 3),
            val_folds=(2,),
        )


# ---------------------------------------------------------------------------
# 6.3 Chained init strict=False + full orchestration (AC-4) with mocks.
# ---------------------------------------------------------------------------


class _SpyStudent:
    """Records load_state_dict calls (the warm-start observation point)."""

    def __init__(self) -> None:
        self.load_calls: list[tuple[Any, bool]] = []

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        self.load_calls.append((state_dict, strict))

        class _Incompatible:
            missing_keys: ClassVar[list[str]] = []
            unexpected_keys: ClassVar[list[str]] = []

        return _Incompatible()

    def parameters(self) -> Any:
        yield torch.zeros(1)


class _SpyTrainer:
    """Lightweight stand-in for the FarSLIP trainer (no CLIP / GPU / MLflow)."""

    instances: ClassVar[list[_SpyTrainer]] = []

    def __init__(self, config: Any, dataset: Any = None) -> None:
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cpu")
        self.student = _SpyStudent()
        self._text_prototypes: torch.Tensor | None = None
        self.set_prototypes_called = False
        type(self).instances.append(self)

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        # Mimic the 384 -> 768 reprojection so eval sees a 768-dim bank.
        self.set_prototypes_called = True
        n = prototypes.shape[0]
        self._text_prototypes = torch.randn(n, 768)

    def train(self) -> dict[str, float]:
        # Persist the last-epoch checkpoint the orchestrator copies to best.
        from safetensors.torch import save_file

        epochs = int(self.config.n_epochs)
        out = self.config.output_dir / f"student_epoch_{epochs - 1}.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2, 2)}, str(out))
        return {"loss_cls": 0.5, "loss_total": 1.0, "loss_patch": 0.3, "loss_aux": 0.2}

    def save_student(self, format: str = "safetensors", suffix: str | None = None) -> str:
        from safetensors.torch import save_file

        out = self.config.output_dir / f"student_{suffix}.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2, 2)}, str(out))
        return str(out)


def _wire_orchestrator_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    per_step_f1: dict[int, dict[int, float]] | None = None,
) -> None:
    """Patch the dataset builder, trainer, prototype loader, eval and MLflow.

    Args:
        monkeypatch: pytest fixture.
        per_step_f1: optional ``{n_classes: {class_id: f1}}`` to drive the stop
            criterion through ``eval_per_class``; defaults to all-healthy 0.7.
    """
    _SpyTrainer.instances = []

    proto_18, ids_all = _synthetic_proto_18()

    def _fake_load_protos(_path: Any = None) -> tuple[np.ndarray, list[int]]:
        return proto_18, ids_all

    def _fake_create(n_classes: int, **_kwargs: Any) -> tuple[Any, int, int, Any]:
        ds = [
            {
                "image": torch.rand(4, 224, 224),
                "region_id": torch.tensor(0, dtype=torch.long),
                "category_id": torch.tensor(0, dtype=torch.long),
            }
        ]
        return ds, 1, n_classes, torch.from_numpy(proto_18[:n_classes]).float()

    def _fake_eval(
        student: Any,
        val_dataset: Any,
        class_ids: list[int],
        prototypes: torch.Tensor,
        **_kwargs: Any,
    ) -> StepMetrics:
        f1_map = (per_step_f1 or {}).get(len(class_ids), {c: 0.70 for c in class_ids})
        return _metrics(class_ids, {c: f1_map.get(c, 0.70) for c in class_ids})

    def _fake_load_state_dict(_ckpt: Path) -> dict[str, torch.Tensor]:
        return {"w": torch.zeros(2, 2)}

    monkeypatch.setattr(orch, "load_class_prototype_embeddings", _fake_load_protos)
    monkeypatch.setattr(orch, "create_incremental_dataset", _fake_create)
    monkeypatch.setattr(orch, "FarSLIPDistillationTrainer", _SpyTrainer)
    monkeypatch.setattr(orch, "eval_per_class", _fake_eval)
    monkeypatch.setattr(orch, "_load_student_state_dict", _fake_load_state_dict)
    monkeypatch.setattr(orch, "_log_step_run", lambda **_kw: None)
    monkeypatch.setattr(orch, "propagate_seed", lambda _seed: None)


def test_step0_inits_from_teacher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Step 0 does NOT warm-start (teacher CLIP default); step 1 does."""
    _wire_orchestrator_mocks(monkeypatch)
    results = orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=8,  # 3 steps: 4, 6, 8 -> quick
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    assert results[0].init_from == "teacher_clip"
    assert results[0].n_classes == 4
    # Step 0 trainer student got no load; step 1 student did, with strict=False.
    step0_trainer, step1_trainer = _SpyTrainer.instances[0], _SpyTrainer.instances[1]
    assert step0_trainer.student.load_calls == []
    assert len(step1_trainer.student.load_calls) == 1
    _state, strict = step1_trainer.student.load_calls[0]
    assert strict is False, "chained init must use strict=False"


def test_stepk_inits_from_prev_best(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each step k>0 warm-starts from the previous step's best checkpoint."""
    _wire_orchestrator_mocks(monkeypatch)
    results = orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=8,
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    # init_from of step 1 / 2 is the best.safetensors of the previous step.
    assert results[1].init_from == str(results[0].best_ckpt)
    assert results[2].init_from == str(results[1].best_ckpt)
    assert results[0].best_ckpt.name == "best.safetensors"
    assert (tmp_path / "04cls" / "best.safetensors").exists()
    assert (tmp_path / "06cls" / "best.safetensors").exists()
    assert (tmp_path / "08cls" / "best.safetensors").exists()


def test_run_n_regions_is_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The per-step trainer cfg uses n_regions=1 and n_categories=N_k."""
    _wire_orchestrator_mocks(monkeypatch)
    orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=8,
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    seen = [(t.config.n_regions, t.config.n_categories) for t in _SpyTrainer.instances]
    assert seen == [(1, 4), (1, 6), (1, 8)]
    # All trainers ran with 4 input channels and the productive epoch count.
    assert all(t.config.n_in_channels == 4 for t in _SpyTrainer.instances)
    assert all(t.config.n_epochs == 20 for t in _SpyTrainer.instances)
    assert all(t.set_prototypes_called for t in _SpyTrainer.instances)


def test_run_stops_on_unacceptable_new_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The curriculum breaks early when a step's new classes are unacceptable."""
    # At 6 classes, the 2 new ones (4, 14) collapse below 0.30 -> stop at step 1.
    _wire_orchestrator_mocks(
        monkeypatch,
        per_step_f1={
            4: {1: 0.7, 3: 0.7, 2: 0.7, 8: 0.7},
            6: {1: 0.7, 3: 0.7, 2: 0.7, 8: 0.7, 4: 0.10, 14: 0.05},
        },
    )
    results = orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=18,
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    assert len(results) == 2  # stopped at the 6-class step
    assert results[-1].stop_reason == "new_classes_unacceptable"


def test_run_budget_exhausted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A zero wall-clock budget stops after the first step (budget_exhausted)."""
    _wire_orchestrator_mocks(monkeypatch)
    results = orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=18,
        folds=(1, 2, 3),
        val_folds=(4,),
        time_cap_hours=0.0,  # budget already spent before step 1
    )
    # Step 0 always runs (the gate requires a prior result); step 1 is gated.
    assert len(results) == 1
    assert results[-1].stop_reason == "budget_exhausted"


def test_metrics_table_rows_shape() -> None:
    """The per-class artifact rows carry class_id/f1/iou/well_resolved."""
    m = _metrics([1, 3], {1: 0.8, 3: 0.4})
    rows = orch._metrics_table_rows(m)
    assert [r["class_id"] for r in rows] == [1, 3]
    assert rows[0]["well_resolved"] is True
    assert rows[1]["well_resolved"] is False


def test_select_winner_empty() -> None:
    """No steps -> no winner."""
    assert orch._select_winner([]) is None


def test_run_selects_winner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The winner is the step with the most well-resolved classes."""
    _wire_orchestrator_mocks(
        monkeypatch,
        per_step_f1={
            4: {1: 0.8, 3: 0.8, 2: 0.8, 8: 0.8},  # 4 well-resolved
            6: {1: 0.8, 3: 0.8, 2: 0.8, 8: 0.8, 4: 0.6, 14: 0.6},  # 6 well-resolved
            8: {1: 0.8, 3: 0.8, 2: 0.8, 8: 0.8, 4: 0.6, 14: 0.6, 5: 0.1, 16: 0.1},
        },
    )
    results = orch.run_incremental_curriculum(
        pastis_root=Path("data/PASTIS-R"),
        output_root=tmp_path,
        epochs_per_step=20,
        step_size=2,
        max_classes=8,
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    winner = orch._select_winner(results)
    assert winner is not None
    assert winner.n_classes == 6  # 6 well-resolved beats 4 and 8 (8 has 2 collapsed)
    assert winner.metrics.n_classes_well_resolved == 6


# ---------------------------------------------------------------------------
# Per-step MLflow run is logged AND closed (AC-10, no RUNNING gotcha).
# ---------------------------------------------------------------------------


def test_log_step_run_closes_run(tmp_path: Path) -> None:
    """A per-step run lands in MLflow with per-class metrics and is CLOSED."""
    mlflow = pytest.importorskip("mlflow")

    store = tmp_path / "mlruns.db"
    uri = f"sqlite:///{store.as_posix()}"
    metrics = _metrics([1, 3, 2, 8], {1: 0.8, 3: 0.6, 2: 0.4, 8: 0.55})
    orch._log_step_run(
        mlflow_uri=uri,
        run_name="farslip-full-incr-04cls",
        n_classes=4,
        class_ids=[1, 3, 2, 8],
        metrics=metrics,
        init_from="teacher_clip",
        epochs=20,
        step_size=2,
        dominance_ratio=3.0,
        folds=(1, 2, 3),
        val_folds=(4,),
        pastis_root=Path("data/PASTIS-R"),
        train_metrics={"loss_cls": 0.5, "loss_total": 1.0},
        stop_reason="continue",
    )

    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("farslip")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    # The run is CLOSED (FINISHED), never left RUNNING (ml/AGENTS.md gotcha).
    assert run.info.status == "FINISHED"
    assert run.data.params["n_classes"] == "4"
    assert run.data.params["n_regions"] == "1"
    assert run.data.params["proto_source"] == "pastis_direct"
    assert run.data.metrics["f1_class_1"] == pytest.approx(0.8)
    assert run.data.metrics["macro_f1"] == pytest.approx(metrics.macro_f1)
    assert run.data.tags["us"] == "US-036-a"


def test_log_step_run_survives_bad_uri(tmp_path: Path) -> None:
    """An unreachable MLflow server degrades to a warning, never raising."""
    pytest.importorskip("mlflow")
    metrics = _metrics([1, 3], {1: 0.8, 3: 0.6})
    # Must NOT raise even if the tracking backend is invalid.
    orch._log_step_run(
        mlflow_uri="http://127.0.0.1:1",  # nothing listening
        run_name="farslip-full-incr-02cls",
        n_classes=2,
        class_ids=[1, 3],
        metrics=metrics,
        init_from="teacher_clip",
        epochs=20,
        step_size=2,
        dominance_ratio=3.0,
        folds=(1, 2, 3),
        val_folds=(4,),
        pastis_root=Path("data/PASTIS-R"),
        train_metrics={"loss_cls": 0.5},
        stop_reason="continue",
    )


# ---------------------------------------------------------------------------
# eval_per_class with a mock student + prototypes (AC-7).
# ---------------------------------------------------------------------------


class _DeterministicStudent(torch.nn.Module):
    """A student whose CLS token is a one-hot of the (fed) true class.

    For a batch of images we cannot read the label, so this stub returns a fixed
    CLS per call mapped from a queue of true classes, letting the test assert the
    cosine-argmax classification produces the expected confusion. It is enough to
    prove ``eval_per_class`` derives F1/IoU from CLS<->prototype cosine.
    """

    def __init__(self, cls_rows: torch.Tensor) -> None:
        super().__init__()
        self._cls_rows = cls_rows  # (n_total, 768)
        self._cursor = 0
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor, **_kw: Any) -> Any:
        b = pixel_values.shape[0]
        cls = self._cls_rows[self._cursor : self._cursor + b]
        self._cursor += b
        seq = torch.zeros(b, 2, cls.shape[1])
        seq[:, 0, :] = cls

        class _Out:
            last_hidden_state = seq

        return _Out()


def test_eval_per_class_perfect_classification() -> None:
    """A student whose CLS equals its class prototype yields F1=IoU=1.0."""
    class_ids = [1, 3, 2, 8]
    # Orthonormal-ish 768-dim prototypes (one per class).
    protos = torch.eye(4, 768)
    # 8 pairs, true classes cycling 0,1,2,3,0,1,2,3; CLS == the true prototype.
    true_idx = [0, 1, 2, 3, 0, 1, 2, 3]
    cls_rows = torch.stack([protos[i] for i in true_idx])
    student = _DeterministicStudent(cls_rows)
    val_ds = [
        {
            "image": torch.rand(4, 8, 8),
            "region_id": torch.tensor(0),
            "category_id": torch.tensor(i, dtype=torch.long),
        }
        for i in true_idx
    ]
    metrics = orch.eval_per_class(
        student, val_ds, class_ids, protos, device=torch.device("cpu"), batch_size=4
    )
    assert metrics.n_eval == 8
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.macro_iou == pytest.approx(1.0)
    assert metrics.n_classes_well_resolved == 4
    for cid in class_ids:
        assert metrics.per_class_f1[cid] == pytest.approx(1.0)


def test_eval_per_class_partial_confusion() -> None:
    """A misclassified pair lowers the affected classes' F1/IoU below 1.0."""
    class_ids = [1, 3, 2, 8]
    protos = torch.eye(4, 768)
    true_idx = [0, 1, 2, 3]
    cls_rows = torch.stack([protos[i] for i in true_idx])
    # Corrupt the CLS of the first pair so it points at class index 1, not 0.
    cls_rows[0] = protos[1]
    student = _DeterministicStudent(cls_rows)
    val_ds = [
        {
            "image": torch.rand(4, 8, 8),
            "region_id": torch.tensor(0),
            "category_id": torch.tensor(i, dtype=torch.long),
        }
        for i in true_idx
    ]
    metrics = orch.eval_per_class(
        student, val_ds, class_ids, protos, device=torch.device("cpu"), batch_size=2
    )
    assert metrics.n_eval == 4
    # Class 1 (idx 0): one FN -> F1 = 0. Class 3 (idx 1): one FP -> F1 < 1.
    assert metrics.per_class_f1[1] == pytest.approx(0.0)
    assert metrics.per_class_f1[3] < 1.0
    # Classes 2 and 8 are untouched.
    assert metrics.per_class_f1[2] == pytest.approx(1.0)
    assert metrics.per_class_f1[8] == pytest.approx(1.0)


def test_eval_per_class_rejects_proto_mismatch() -> None:
    """A prototype bank with the wrong row count is rejected."""
    student = _DeterministicStudent(torch.zeros(1, 768))
    with pytest.raises(ValueError, match="must equal len"):
        orch.eval_per_class(
            student,
            [],
            [1, 3, 2],
            torch.eye(2, 768),  # 2 rows, 3 class_ids
            device=torch.device("cpu"),
        )
