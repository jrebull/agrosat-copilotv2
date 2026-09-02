"""Tests for the US-036-a v2 faithful FarSLIP integration (trainer + orchestrator).

Zero GPU, zero network, zero real PASTIS, zero Gemma: the heavy
:class:`FarSLIPDistillationTrainer` CLIP load is bypassed by injecting a tiny CPU
``CLIPVisionModel`` stub, the region-category dataset is a light in-memory mock,
and MLflow uses a temporary SQLite store. The tests verify the T4 integration:

    * ``step_faithful_v2`` does ONE CPU step on a mock batch and yields finite
      ``loss_glo`` + ``loss_loc`` > 0 (AC-2/AC-3 integration).
    * the RAW PASTIS ``region_cat_ids`` -> ``[0, C)`` mapping is correct and
      fails fast on an out-of-bank id (the v1 trap fixed).
    * ``region_visual = student_cls[region_to_patch]`` gathers each region's
      patch CLS correctly (paper Section 4.3 / R-REGION-CROP).
    * ``lambda_loc=0`` makes ``loss_total == loss_glo`` (ablation, AC-3).
    * overlapping train/val folds raise (spatial-CV anti-leakage, AC-8).
    * the v1 path (``supervision="dominant"``) still works (no regression).
    * the orchestrator rejects the Italian root, fold overlap and missing
      captions, builds the trainer in ``region_category`` mode, and the MLflow
      run is CLOSED (no RUNNING gotcha).

Section map: docs/us-planning/us-036-a-v2-faithful.md sec. 6 (6.4 + integration).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
import torch.nn as nn

import scripts.run_us036a_v2_farslip_faithful as orch
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig

_HIDDEN = 32  # tiny CLS dim for fast CPU tests (the real ViT-B/16 is 768)


# ---------------------------------------------------------------------------
# Tiny CPU CLIP stub so the trainer builds without HF / GPU.
# ---------------------------------------------------------------------------


class _StubVisionConfig:
    """Minimal CLIP-vision config exposing the hidden size the trainer reads."""

    def __init__(self, hidden_size: int = _HIDDEN) -> None:
        self.hidden_size = hidden_size


class _StubVisionOut:
    """Mimics ``CLIPVisionModelOutput.last_hidden_state`` ``(B, 1+P, D)``."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _StubVisionModel(nn.Module):
    """A tiny CLIP-vision stand-in: a Conv2d patch_embed + a CLS-producing head.

    It exposes ``embeddings.patch_embedding`` (a 3-channel Conv2d, so the trainer
    adapts it to 4) and a ``config.hidden_size``, and returns a
    ``last_hidden_state`` of shape ``(B, 1+P, D)`` with a CLS token at position 0
    that depends on the input pixels (so gradients flow and a smoke step learns).
    """

    def __init__(self, hidden_size: int = _HIDDEN) -> None:
        super().__init__()
        self.config = _StubVisionConfig(hidden_size)

        class _Embeddings(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embedding = nn.Conv2d(
                    3, hidden_size, kernel_size=16, stride=16, bias=False
                )

        self.embeddings = _Embeddings()
        self._proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> _StubVisionOut:
        # (B, D, H', W') -> (B, P, D) tokens; CLS = projected mean over patches.
        feats = self.embeddings.patch_embedding(pixel_values)
        tokens = feats.flatten(2).transpose(1, 2)  # (B, P, D)
        cls = self._proj(tokens.mean(dim=1, keepdim=True))  # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+P, D)
        return _StubVisionOut(seq)


def _build_stub_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    supervision: str = "region_category",
    lambda_loc: float = 1.0,
    use_global_caption_loss: bool = True,
    n_categories: int = 4,
) -> FarSLIPDistillationTrainer:
    """Build a FarSLIP trainer whose CLIP load is the tiny CPU stub.

    Patches ``_load_models`` so no HF download happens; the teacher/student are
    distinct :class:`_StubVisionModel` instances on CPU. ``_patch_student_proj``
    still adapts the student to 4 channels via the real code path.
    """

    def _fake_load(self: FarSLIPDistillationTrainer) -> None:
        teacher = _StubVisionModel(_HIDDEN)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = _StubVisionModel(_HIDDEN)
        for p in student.parameters():
            p.requires_grad_(True)
        student.train()
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)

    monkeypatch.setattr(FarSLIPDistillationTrainer, "_load_models", _fake_load)

    cfg = FarSLIPTrainerConfig(
        output_dir=tmp_path,
        device="cpu",
        n_in_channels=4,
        n_categories=n_categories,
        supervision=supervision,  # type: ignore[arg-type]
        lambda_loc=lambda_loc,
        use_global_caption_loss=use_global_caption_loss,
    )
    return FarSLIPDistillationTrainer(cfg)


def _set_category_bank(trainer: FarSLIPDistillationTrainer, class_ids: list[int]) -> None:
    """Inject an identity-ish 768->hidden category bank in 768-passthrough mode."""
    bank = torch.randn(len(class_ids), _HIDDEN)
    trainer.set_category_prototypes(bank, class_ids)


# ---------------------------------------------------------------------------
# step_faithful_v2: one CPU step, finite positive L_glo + L_loc.
# ---------------------------------------------------------------------------


def test_step_faithful_v2_finite_positive_losses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One v2 step on a mock batch yields finite loss_glo + loss_loc > 0."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    class_ids = [1, 3, 2, 8]  # RAW PASTIS ids; bank order = canonical
    _set_category_bank(trainer, class_ids)

    images = torch.rand(2, 4, 32, 32)  # B=2 patches
    region_cat_ids = torch.tensor([1, 3, 2, 8], dtype=torch.long)  # RAW PASTIS
    region_to_patch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    caption_cls = torch.randn(2, _HIDDEN)  # already in CLS dim

    losses = trainer.step_faithful_v2(
        images, region_cat_ids, region_to_patch, caption_cls=caption_cls
    )
    assert set(losses) == {"loss_total", "loss_glo", "loss_loc"}
    assert torch.isfinite(losses["loss_glo"]) and losses["loss_glo"].item() > 0.0
    assert torch.isfinite(losses["loss_loc"]) and losses["loss_loc"].item() > 0.0
    assert torch.isfinite(losses["loss_total"])
    # Backward flows to the student.
    losses["loss_total"].backward()
    grads = [p.grad for p in trainer.student.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no student gradient produced by step_faithful_v2"
    assert all(torch.isfinite(g).all() for g in grads)


def test_step_faithful_v2_total_is_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """loss_total == loss_glo + lambda_loc * loss_loc (combine_losses contract)."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch, lambda_loc=0.5)
    _set_category_bank(trainer, [1, 3, 2, 8])
    images = torch.rand(2, 4, 32, 32)
    losses = trainer.step_faithful_v2(
        images,
        torch.tensor([1, 3, 2, 8]),
        torch.tensor([0, 0, 1, 1]),
        caption_cls=torch.randn(2, _HIDDEN),
    )
    expected = losses["loss_glo"] + 0.5 * losses["loss_loc"]
    assert torch.allclose(losses["loss_total"], expected, atol=1e-6)


def test_step_faithful_v2_lambda_zero_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lambda_loc=0 -> loss_total == loss_glo exactly (L_loc ablation, AC-3)."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch, lambda_loc=0.0)
    _set_category_bank(trainer, [1, 3, 2, 8])
    images = torch.rand(2, 4, 32, 32)
    losses = trainer.step_faithful_v2(
        images,
        torch.tensor([1, 3, 2, 8]),
        torch.tensor([0, 0, 1, 1]),
        caption_cls=torch.randn(2, _HIDDEN),
    )
    assert torch.allclose(losses["loss_total"], losses["loss_glo"], atol=1e-7)


def test_step_faithful_v2_no_caption_zeroes_glo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without caption_cls, L_glo is 0 and loss_total == lambda_loc * loss_loc."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch, lambda_loc=1.0)
    _set_category_bank(trainer, [1, 3, 2, 8])
    images = torch.rand(2, 4, 32, 32)
    losses = trainer.step_faithful_v2(
        images, torch.tensor([1, 3, 2, 8]), torch.tensor([0, 0, 1, 1]), caption_cls=None
    )
    assert losses["loss_glo"].item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(losses["loss_total"], losses["loss_loc"], atol=1e-6)


# ---------------------------------------------------------------------------
# PASTIS id -> [0, C) mapping (the v1 trap fixed).
# ---------------------------------------------------------------------------


def test_pastis_to_category_mapping_is_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAW PASTIS ids map to the canonical bank index, NOT to the raw value."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    class_ids = [1, 3, 2, 8]  # canonical order -> indices 0,1,2,3
    _set_category_bank(trainer, class_ids)
    raw = torch.tensor([8, 2, 3, 1], dtype=torch.long)
    mapped = trainer._map_region_cat_ids(raw)
    # class 8 -> idx 3, 2 -> 2, 3 -> 1, 1 -> 0.
    assert mapped.tolist() == [3, 2, 1, 0]


def test_pastis_to_category_unknown_id_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A region id absent from the bank fails fast (no silent drop of P(i))."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    _set_category_bank(trainer, [1, 3, 2, 8])
    with pytest.raises(ValueError, match="absent from the category bank"):
        trainer._map_region_cat_ids(torch.tensor([1, 99]))


def test_set_category_prototypes_validates_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mismatched / duplicate ids vs rows are rejected."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must match"):
        trainer.set_category_prototypes(torch.randn(3, _HIDDEN), [1, 2])
    with pytest.raises(ValueError, match="unique"):
        trainer.set_category_prototypes(torch.randn(3, _HIDDEN), [1, 1, 2])


# ---------------------------------------------------------------------------
# region_visual = student_cls[region_to_patch] (R-REGION-CROP gather).
# ---------------------------------------------------------------------------


def test_region_gather_shares_patch_cls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All regions of a patch get that patch's CLS (gather by region_to_patch).

    Captures the MPCL ``region_visual`` argument via a spy and checks it equals
    the student CLS gathered by ``region_to_patch`` (3 regions: 2 from patch 0,
    1 from patch 1).
    """
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    _set_category_bank(trainer, [1, 3, 2])

    captured: dict[str, torch.Tensor] = {}
    orig_forward = trainer._mpcl_loss.forward

    def _spy(region_visual: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:
        captured["region_visual"] = region_visual.detach().clone()
        return orig_forward(region_visual, *a, **k)

    monkeypatch.setattr(trainer._mpcl_loss, "forward", _spy)

    images = torch.rand(2, 4, 32, 32)
    region_to_patch = torch.tensor([0, 0, 1], dtype=torch.long)
    trainer.step_faithful_v2(images, torch.tensor([1, 3, 2]), region_to_patch, caption_cls=None)

    # Recompute the expected gather from the same deterministic student.
    with torch.no_grad():
        out = trainer.student(pixel_values=images, output_hidden_states=False)
        student_cls = out.last_hidden_state[:, 0, :]
        expected = student_cls[region_to_patch]
    assert captured["region_visual"].shape == (3, _HIDDEN)
    assert torch.allclose(captured["region_visual"], expected, atol=1e-5)
    # Regions 0 and 1 share patch-0 CLS; region 2 is patch-1 CLS.
    assert torch.allclose(captured["region_visual"][0], captured["region_visual"][1])
    assert not torch.allclose(captured["region_visual"][0], captured["region_visual"][2])


def test_step_faithful_v2_region_to_patch_out_of_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A region pointing past the batch raises (collate contract guard)."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    _set_category_bank(trainer, [1, 3])
    images = torch.rand(2, 4, 32, 32)
    with pytest.raises(ValueError, match="region_to_patch out of range"):
        trainer.step_faithful_v2(
            images, torch.tensor([1, 3]), torch.tensor([0, 5]), caption_cls=None
        )


def test_step_faithful_v2_requires_category_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling the v2 step without the category bank raises explicitly."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="category prototypes not initialized"):
        trainer.step_faithful_v2(torch.rand(1, 4, 32, 32), torch.tensor([1]), torch.tensor([0]))


# ---------------------------------------------------------------------------
# v1 path non-regression (supervision="dominant").
# ---------------------------------------------------------------------------


def test_v1_path_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """supervision='dominant' (v1) keeps step() + the dominant batch contract."""
    trainer = _build_stub_trainer(tmp_path, monkeypatch, supervision="dominant", n_categories=3)
    trainer._cls_loss.n_regions = 1
    trainer._cls_loss.n_categories = 3
    trainer.set_text_prototypes(torch.randn(3, _HIDDEN))
    out = trainer.step(
        torch.rand(2, 4, 32, 32),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
    )
    assert set(out) == {"loss_total", "loss_patch", "loss_cls", "loss_aux"}
    assert torch.isfinite(out["loss_total"])
    out["loss_total"].backward()


def test_forward_batch_dispatches_by_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_forward_batch routes the collated batch to the right forward per mode."""
    # v2 batch -> step_faithful_v2 keys.
    trainer_v2 = _build_stub_trainer(tmp_path / "v2", monkeypatch)
    _set_category_bank(trainer_v2, [1, 3])
    batch_v2 = {
        "images": torch.rand(2, 4, 32, 32),
        "region_cat_ids": torch.tensor([1, 3]),
        "region_to_patch": torch.tensor([0, 1]),
        "captions": ["a", "b"],
        "caption_cls": torch.randn(2, _HIDDEN),
    }
    out_v2 = trainer_v2._forward_batch(batch_v2)
    assert "loss_loc" in out_v2

    # v1 batch -> step keys.
    trainer_v1 = _build_stub_trainer(
        tmp_path / "v1", monkeypatch, supervision="dominant", n_categories=3
    )
    trainer_v1._cls_loss.n_regions = 1
    trainer_v1._cls_loss.n_categories = 3
    trainer_v1.set_text_prototypes(torch.randn(3, _HIDDEN))
    batch_v1 = {
        "image": torch.rand(2, 4, 32, 32),
        "region_id": torch.tensor([0, 0]),
        "category_id": torch.tensor([0, 1]),
    }
    out_v1 = trainer_v1._forward_batch(batch_v1)
    assert "loss_cls" in out_v1


# ---------------------------------------------------------------------------
# Spatial-CV anti-leakage at the orchestrator (AC-8).
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_train_eval_fold_overlap() -> None:
    """val_folds overlapping train folds raises (assert_disjoint_folds)."""
    with pytest.raises(ValueError, match="disjoint"):
        orch.run_faithful_v2(
            pastis_root=Path("data/PASTIS-R"),
            captions_path=Path("data/farslip/pastis_captions.parquet"),
            folds=(1, 2, 3),
            val_folds=(2,),
        )


def test_orchestrator_rejects_italian_root() -> None:
    """Pointing at the Italian/synthetic data root is a hard ValueError."""
    with pytest.raises(ValueError, match="farslip_pairs"):
        orch.run_faithful_v2(
            pastis_root=Path("data/farslip_pairs"),
            captions_path=Path("data/farslip/pastis_captions.parquet"),
        )


def test_validate_pastis_root_accepts_real() -> None:
    """A real PASTIS-R root passes the guard."""
    orch._validate_pastis_root(Path("data/PASTIS-R"))


# ---------------------------------------------------------------------------
# Orchestrator end-to-end with mocks (dataset, trainer, eval, captions).
# ---------------------------------------------------------------------------


class _MockRegionDataset:
    """In-memory region-category dataset mimicking RegionCategoryPairDataset.

    Exposes ``_samples`` (the (patch_id, regions) list the orchestrator inspects),
    ``mean_regions_per_patch`` and ``__getitem__``/``__len__`` so eval iterates it.
    """

    def __init__(self, samples: list[tuple[str, list[tuple[int, int]]]]) -> None:
        self._samples = samples
        total = sum(len(r) for _p, r in samples)
        self.mean_regions_per_patch = total / len(samples) if samples else 0.0

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid, regions = self._samples[idx]
        return {
            "image": torch.rand(4, 32, 32),
            "patch_id": pid,
            "caption": f"caption {pid}",
            "region_cat_ids": torch.tensor([c for _i, c in regions], dtype=torch.long),
        }


class _SpyFaithfulTrainer:
    """Lightweight stand-in for the FarSLIP trainer (no CLIP / GPU / MLflow)."""

    instances: ClassVar[list[_SpyFaithfulTrainer]] = []

    def __init__(self, config: Any, dataset: Any = None) -> None:
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cpu")
        self.student = nn.Linear(1, 1)
        self._category_prototypes: torch.Tensor | None = None
        self.set_category_called = False
        type(self).instances.append(self)

    def set_category_prototypes(self, prototypes: torch.Tensor, pastis_class_ids: Any) -> None:
        self.set_category_called = True
        self._category_prototypes = torch.randn(prototypes.shape[0], 32)

    def train(self, dataloader: Any = None) -> dict[str, float]:
        from safetensors.torch import save_file

        epochs = int(self.config.n_epochs)
        out = self.config.output_dir / f"student_epoch_{epochs - 1}.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2, 2)}, str(out))
        return {"loss_total": 1.0, "loss_glo": 0.6, "loss_loc": 0.4}

    def save_student(self, format: str = "safetensors", suffix: str | None = None) -> str:
        from safetensors.torch import save_file

        out = self.config.output_dir / f"student_{suffix}.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2, 2)}, str(out))
        return str(out)


def _wire_orchestrator_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captions: dict[str, str] | None = None,
    train_samples: list[tuple[str, list[tuple[int, int]]]] | None = None,
    val_samples: list[tuple[str, list[tuple[int, int]]]] | None = None,
) -> None:
    """Patch captions loader, dataset, trainer, prototypes, eval and MLflow."""
    _SpyFaithfulTrainer.instances = []
    caps = captions if captions is not None else {"1": "a", "2": "b", "3": "c", "4": "d"}
    tr = train_samples or [("1", [(10, 1), (11, 3)]), ("2", [(20, 2)])]
    vl = val_samples or [("3", [(30, 1)]), ("4", [(40, 3), (41, 3)])]

    def _fake_load_captions(_path: Any = None) -> dict[str, str]:
        return caps

    seq = iter([_MockRegionDataset(tr), _MockRegionDataset(vl)])

    def _fake_dataset(*_a: Any, **_k: Any) -> _MockRegionDataset:
        return next(seq)

    def _fake_protos(
        _path: Any, active_class_ids: tuple[int, ...]
    ) -> tuple[torch.Tensor, list[int]]:
        return torch.randn(len(active_class_ids), 384), list(active_class_ids)

    def _fake_eval(
        student: Any,
        val_dataset: Any,
        category_prototypes: torch.Tensor,
        class_ids: list[int],
        **_k: Any,
    ) -> orch.FaithfulRunResult:
        return orch.FaithfulRunResult(
            supervision="region_category",
            n_categories=len(class_ids),
            class_ids=list(class_ids),
            per_class_f1={c: 0.6 for c in class_ids},
            per_class_iou={c: 0.5 for c in class_ids},
            macro_f1=0.6,
            macro_iou=0.5,
            n_eval=2,
            n_classes_well_resolved=len(class_ids),
            best_ckpt=Path(),
            mean_regions_per_patch=0.0,
        )

    monkeypatch.setattr(orch, "load_captions", _fake_load_captions)
    monkeypatch.setattr(orch, "RegionCategoryPairDataset", _fake_dataset)
    monkeypatch.setattr(orch, "FarSLIPDistillationTrainer", _SpyFaithfulTrainer)
    monkeypatch.setattr(orch, "_category_prototypes", _fake_protos)
    monkeypatch.setattr(orch, "eval_per_class_v2", _fake_eval)
    monkeypatch.setattr(orch, "_log_faithful_run", lambda **_kw: None)
    monkeypatch.setattr(orch, "propagate_seed", lambda _seed: None)
    monkeypatch.setattr(orch, "_require_captions_for_dataset", lambda *a, **k: None)


def test_run_faithful_v2_builds_region_category_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The orchestrator instantiates the trainer in region_category mode."""
    _wire_orchestrator_mocks(monkeypatch)
    result = orch.run_faithful_v2(
        pastis_root=Path("data/PASTIS-R"),
        captions_path=Path("data/farslip/pastis_captions.parquet"),
        output_dir=tmp_path,
        n_epochs=2,
        batch_size=2,
        active_class_ids=(1, 3, 2),
        folds=(1, 2, 3),
        val_folds=(4,),
    )
    assert len(_SpyFaithfulTrainer.instances) == 1
    trainer = _SpyFaithfulTrainer.instances[0]
    assert trainer.config.supervision == "region_category"
    assert trainer.config.n_in_channels == 4
    assert trainer.config.lambda_loc == 1.0
    assert trainer.set_category_called
    assert result.supervision == "region_category"
    assert result.best_ckpt.name == "best.safetensors"
    assert (tmp_path / "best.safetensors").exists()
    assert result.mean_regions_per_patch > 1.0


def test_run_faithful_v2_dominant_v1_ablation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--supervision dominant_v1 maps to the trainer's v1 'dominant' mode."""
    _wire_orchestrator_mocks(monkeypatch)
    orch.run_faithful_v2(
        pastis_root=Path("data/PASTIS-R"),
        captions_path=Path("data/farslip/pastis_captions.parquet"),
        output_dir=tmp_path,
        supervision="dominant_v1",
        n_epochs=2,
        batch_size=2,
        active_class_ids=(1, 3, 2),
    )
    assert _SpyFaithfulTrainer.instances[0].config.supervision == "dominant"


def test_run_faithful_v2_empty_captions_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No captions loaded -> fail fast (Phase A not run / parquet missing)."""
    _wire_orchestrator_mocks(monkeypatch, captions={})
    with pytest.raises(ValueError, match="no captions loaded"):
        orch.run_faithful_v2(
            pastis_root=Path("data/PASTIS-R"),
            captions_path=Path("data/farslip/pastis_captions.parquet"),
            output_dir=tmp_path,
            n_epochs=2,
        )


def test_require_captions_for_dataset_detects_missing(tmp_path: Path) -> None:
    """A kept patch without a caption is surfaced eagerly (not at __getitem__)."""
    ds = _MockRegionDataset([("1", [(10, 1)]), ("99", [(20, 2)])])
    with pytest.raises(ValueError, match="have no caption"):
        orch._require_captions_for_dataset(ds, {"1": "a"}, "train")


# ---------------------------------------------------------------------------
# eval_per_class_v2 with a deterministic student (AC-9 comparable to v1).
# ---------------------------------------------------------------------------


class _DeterministicStudent(nn.Module):
    """A student whose CLS equals a queued prototype, for confusion assertions."""

    def __init__(self, cls_rows: torch.Tensor) -> None:
        super().__init__()
        self._cls_rows = cls_rows
        self._cursor = 0
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor, **_kw: Any) -> Any:
        b = pixel_values.shape[0]
        cls = self._cls_rows[self._cursor : self._cursor + b]
        self._cursor += b
        seq = torch.zeros(b, 2, cls.shape[1])
        seq[:, 0, :] = cls

        class _Out:
            last_hidden_state = seq

        return _Out()


def test_eval_per_class_v2_perfect(tmp_path: Path) -> None:
    """A student whose CLS equals its category prototype yields F1=IoU=1.0."""
    class_ids = [1, 3, 2, 8]
    protos = torch.eye(4, 16)
    # 4 patches, majority categories 1,3,2,8 -> indices 0,1,2,3; CLS == proto.
    samples = [
        ("p0", [(1, 1), (2, 1)]),  # majority 1 -> idx 0
        ("p1", [(3, 3)]),  # 3 -> idx 1
        ("p2", [(4, 2), (5, 2)]),  # 2 -> idx 2
        ("p3", [(6, 8)]),  # 8 -> idx 3
    ]
    ds = _MockRegionDataset(samples)
    cls_rows = torch.stack([protos[i] for i in (0, 1, 2, 3)])
    student = _DeterministicStudent(cls_rows)
    result = orch.eval_per_class_v2(
        student, ds, protos, class_ids, device=torch.device("cpu"), batch_size=2
    )
    assert result.n_eval == 4
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.macro_iou == pytest.approx(1.0)
    assert result.n_classes_well_resolved == 4


def test_eval_per_class_v2_partial_confusion(tmp_path: Path) -> None:
    """A misclassified patch lowers the affected categories' F1 below 1.0."""
    class_ids = [1, 3, 2, 8]
    protos = torch.eye(4, 16)
    samples = [
        ("p0", [(1, 1)]),  # true idx 0
        ("p1", [(3, 3)]),  # true idx 1
        ("p2", [(4, 2)]),  # true idx 2
        ("p3", [(6, 8)]),  # true idx 3
    ]
    ds = _MockRegionDataset(samples)
    cls_rows = torch.stack([protos[i] for i in (0, 1, 2, 3)])
    cls_rows[0] = protos[1]  # patch 0 now points at class index 1
    student = _DeterministicStudent(cls_rows)
    result = orch.eval_per_class_v2(
        student, ds, protos, class_ids, device=torch.device("cpu"), batch_size=4
    )
    assert result.per_class_f1[1] == pytest.approx(0.0)  # one FN
    assert result.per_class_f1[3] < 1.0  # one FP
    assert result.per_class_f1[2] == pytest.approx(1.0)
    assert result.per_class_f1[8] == pytest.approx(1.0)


def test_eval_per_class_v2_rejects_proto_mismatch() -> None:
    """A bank with the wrong row count is rejected."""
    student = _DeterministicStudent(torch.zeros(1, 16))
    with pytest.raises(ValueError, match="must equal"):
        orch.eval_per_class_v2(student, _MockRegionDataset([]), torch.eye(2, 16), [1, 3, 2])


def test_patch_majority_category_tie_break() -> None:
    """Majority category; ties broken by the smaller PASTIS id (deterministic)."""
    assert orch._patch_majority_category([(1, 5), (2, 5), (3, 2)]) == 5
    # Tie between 2 and 7 (one each) -> smaller id 2.
    assert orch._patch_majority_category([(1, 7), (2, 2)]) == 2


# ---------------------------------------------------------------------------
# v1 vs v2 table + closed MLflow run (AC-9, AC-10).
# ---------------------------------------------------------------------------


def _result(class_ids: list[int], f1: dict[int, float]) -> orch.FaithfulRunResult:
    return orch.FaithfulRunResult(
        supervision="region_category",
        n_categories=len(class_ids),
        class_ids=class_ids,
        per_class_f1=f1,
        per_class_iou={c: f1[c] * 0.9 for c in class_ids},
        macro_f1=float(np.mean([f1[c] for c in class_ids])),
        macro_iou=float(np.mean([f1[c] * 0.9 for c in class_ids])),
        n_eval=10,
        n_classes_well_resolved=sum(1 for c in class_ids if f1[c] >= 0.5),
        best_ckpt=Path("checkpoints/farslip/faithful_v2/best.safetensors"),
        mean_regions_per_patch=2.3,
    )


def test_v1_vs_v2_table_rows_with_delta() -> None:
    """The comparison table carries v2 metrics and per-class delta vs v1."""
    result = _result([1, 3], {1: 0.7, 3: 0.4})
    rows = orch._v1_vs_v2_table_rows(result, {1: 0.5, 3: 0.5})
    assert [r["class_id"] for r in rows] == [1, 3]
    assert rows[0]["delta_f1"] == pytest.approx(0.2)
    assert rows[1]["delta_f1"] == pytest.approx(-0.1)
    assert rows[0]["well_resolved_v2"] is True
    assert rows[1]["well_resolved_v2"] is False


def test_v1_vs_v2_table_rows_without_v1() -> None:
    """Without a v1 baseline the delta columns are None (honest, still emitted)."""
    rows = orch._v1_vs_v2_table_rows(_result([1], {1: 0.6}), None)
    assert rows[0]["f1_v1"] is None
    assert rows[0]["delta_f1"] is None


def test_log_faithful_run_closes_run(tmp_path: Path) -> None:
    """The faithful-v2 run lands in MLflow with metrics and is CLOSED (FINISHED)."""
    mlflow = pytest.importorskip("mlflow")
    store = tmp_path / "mlruns.db"
    uri = f"sqlite:///{store.as_posix()}"
    result = _result([1, 3, 2], {1: 0.8, 3: 0.6, 2: 0.4})
    orch._log_faithful_run(
        mlflow_uri=uri,
        run_name="farslip-faithful-v2",
        result=result,
        supervision="region_category",
        lambda_loc=1.0,
        temperature=0.07,
        use_global_caption_loss=True,
        folds=(1, 2, 3),
        val_folds=(4,),
        pastis_root=Path("data/PASTIS-R"),
        captions_path=Path("data/farslip/pastis_captions.parquet"),
        caption_model="gemma4:31b-it-q8_0",
        prompt_version="v2",
        train_metrics={"loss_total": 1.0, "loss_glo": 0.6, "loss_loc": 0.4},
        v1_per_class_f1={1: 0.5, 3: 0.5, 2: 0.5},
    )
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("farslip")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"  # never RUNNING (ml/AGENTS.md gotcha)
    assert run.data.params["supervision"] == "region_category"
    assert run.data.params["lambda_loc"] == "1.0"
    assert run.data.metrics["f1_class_1"] == pytest.approx(0.8)
    assert run.data.metrics["macro_f1"] == pytest.approx(result.macro_f1)
    assert run.data.tags["us"] == "US-036-a-v2"


def test_log_faithful_run_survives_bad_uri(tmp_path: Path) -> None:
    """An unreachable MLflow server degrades to a warning, never raising."""
    pytest.importorskip("mlflow")
    orch._log_faithful_run(
        mlflow_uri="http://127.0.0.1:1",
        run_name="farslip-faithful-v2",
        result=_result([1], {1: 0.6}),
        supervision="region_category",
        lambda_loc=1.0,
        temperature=0.07,
        use_global_caption_loss=True,
        folds=(1, 2, 3),
        val_folds=(4,),
        pastis_root=Path("data/PASTIS-R"),
        captions_path=Path("data/farslip/pastis_captions.parquet"),
        caption_model="gemma4:31b-it-q8_0",
        prompt_version="v2",
        train_metrics={"loss_total": 1.0},
        v1_per_class_f1=None,
    )


# ---------------------------------------------------------------------------
# CLI parse helpers.
# ---------------------------------------------------------------------------


def test_parse_class_ids_validates_range() -> None:
    """class_ids out of [1, 18] are rejected; valid ones parse in order."""
    assert orch._parse_class_ids("1,3,2,8") == (1, 3, 2, 8)
    with pytest.raises(Exception, match="out of"):
        orch._parse_class_ids("1,19")


def test_parse_folds_roundtrip() -> None:
    """Comma-separated folds parse to an ordered tuple."""
    assert orch._parse_folds("1,2,3") == (1, 2, 3)
