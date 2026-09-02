"""Mechanics tests of the incremental Stage-1 -> Stage-2 protocol (US-036 ml/B).

Covers ``scripts/train_incremental.py``:

7.  ``test_strict_false_transfers_encoder``: a real Stage-1 student is saved,
    a real Stage-2 student loads it with ``strict=False`` and its
    ``patch_embedding`` + one encoder block become IDENTICAL to Stage-1; the
    text prototypes are NOT in the loaded ``state_dict`` (they are rebuilt).
8.  ``test_prototypes_rebuilt_per_stage``: after the load, ``set_text_prototypes``
    leaves a bank of 18 prototypes (not 4).
9.  ``test_smoke_one_step_per_stage``: on CPU, one ``step()`` per stage returns a
    finite loss > 0 and the per-stage ``n_categories`` is correct.
10. ``test_from_scratch_skips_warmstart``: with ``from_scratch=True`` Stage-2 does
    NOT call ``load_state_dict`` on the Stage-1 checkpoint (and the complement:
    with ``from_scratch=False`` it DOES).

The strict/prototype/smoke tests use the REAL trainer (so the weight-transfer
assertion is meaningful) on ``device="cpu"`` with a tiny batch; they skip cleanly
when ``openai/clip-vit-base-patch16`` cannot be loaded. The ``from_scratch`` test
mocks the trainer and ``create_incremental_dataset`` so it exercises the
orchestration branch without loading CLIP or touching PASTIS / the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from scripts.train_incremental import (
    _load_student_state_dict,
    _parse_folds,
    _run_stage,
)


def _hf_available() -> bool:
    """Heuristic: ``transformers`` installed and ``CLIPVisionModel`` importable."""
    try:
        from transformers import CLIPVisionModel  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _build_trainer(out_dir: Path, n_categories: int) -> FarSLIPDistillationTrainer:
    """Builds a real CPU trainer for ``n_categories`` (n_regions=1, 4 channels).

    Args:
        out_dir: checkpoint output directory.
        n_categories: number of active PASTIS classes for the stage.

    Returns:
        A ready :class:`FarSLIPDistillationTrainer` on CPU.
    """
    cfg = FarSLIPTrainerConfig(
        teacher_model_id="openai/clip-vit-base-patch16",
        dataset_root=Path("data/PASTIS-R"),
        output_dir=out_dir,
        n_epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        device="cpu",
        n_in_channels=4,
        n_regions=1,
        n_categories=n_categories,
    )
    return FarSLIPDistillationTrainer(cfg, dataset=None)


@pytest.fixture(scope="module")
def stage1_trainer(
    tmp_path_factory: pytest.TempPathFactory,
) -> FarSLIPDistillationTrainer:
    """Real Stage-1 trainer (4 categories) on CPU; skips if CLIP is unavailable."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    out_dir = tmp_path_factory.mktemp("stage1")
    try:
        return _build_trainer(out_dir, n_categories=4)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo cargar CLIP: {exc}")


@pytest.fixture(scope="module")
def stage2_trainer(
    tmp_path_factory: pytest.TempPathFactory,
) -> FarSLIPDistillationTrainer:
    """Real Stage-2 trainer (18 categories) on CPU; skips if CLIP is unavailable."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    out_dir = tmp_path_factory.mktemp("stage2")
    try:
        return _build_trainer(out_dir, n_categories=18)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo cargar CLIP: {exc}")


# ---------------------------------------------------------------------------
# Test 7: strict=False transfers the encoder, NOT the prototypes.
# ---------------------------------------------------------------------------


def test_strict_false_transfers_encoder(
    stage1_trainer: FarSLIPDistillationTrainer,
    stage2_trainer: FarSLIPDistillationTrainer,
    tmp_path: Path,
) -> None:
    """Stage-2 warm-start (strict=False) copies the encoder, not the prototypes.

    Verifies AC-6 / R-STRICT: after ``load_state_dict(strict=False)`` the
    ``patch_embedding`` and one transformer block of Stage-2 equal Stage-1, the
    ``missing_keys`` / ``unexpected_keys`` are empty (full encoder transfer with
    no phantom keys), and ``_text_prototypes`` is NOT in the saved ``state_dict``.
    """
    # Mutate Stage-1 weights so they differ from the fresh Stage-2 init, making
    # the transfer observable (both start as a teacher deepcopy otherwise).
    with torch.no_grad():
        for p in stage1_trainer.student.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    stage1_trainer.config.output_dir = tmp_path
    ckpt_path = Path(stage1_trainer.save_student(format="safetensors", suffix="best"))

    saved_state = _load_student_state_dict(ckpt_path)
    # The text prototypes live OUTSIDE the student state_dict (plain attribute),
    # so they must never appear in the persisted checkpoint.
    assert not any("text_prototype" in k for k in saved_state), (
        "text prototypes leaked into the student state_dict"
    )
    assert not any(k.startswith("_proto_proj_w") for k in saved_state)

    # Pre-load: Stage-2 patch_embedding differs from the mutated Stage-1.
    pe_key = "embeddings.patch_embedding.weight"
    block_key = "encoder.layers.0.self_attn.q_proj.weight"
    assert pe_key in saved_state and block_key in saved_state
    assert not torch.allclose(stage2_trainer.student.state_dict()[pe_key], saved_state[pe_key])

    incompatible = stage2_trainer.student.load_state_dict(saved_state, strict=False)
    assert list(incompatible.missing_keys) == [], (
        f"unexpected missing_keys after warm-start: {incompatible.missing_keys}"
    )
    assert list(incompatible.unexpected_keys) == [], (
        f"unexpected unexpected_keys after warm-start: {incompatible.unexpected_keys}"
    )

    post = stage2_trainer.student.state_dict()
    assert torch.allclose(post[pe_key], saved_state[pe_key], atol=1e-7), (
        "patch_embedding not transferred from Stage-1"
    )
    assert torch.allclose(post[block_key], saved_state[block_key], atol=1e-7), (
        "encoder block not transferred from Stage-1"
    )


# ---------------------------------------------------------------------------
# Test 8: prototypes are rebuilt per stage (4 -> 18).
# ---------------------------------------------------------------------------


def test_prototypes_rebuilt_per_stage(
    stage1_trainer: FarSLIPDistillationTrainer,
    stage2_trainer: FarSLIPDistillationTrainer,
) -> None:
    """Stage-1 holds 4 prototypes; after the encoder load Stage-2 rebuilds 18.

    The prototype bank size follows ``set_text_prototypes`` per stage, NOT the
    transferred encoder. We inject random MiniLM-384 protos so the test does not
    depend on the US-033 parquet.
    """
    proto_4 = torch.randn(4, 384)
    stage1_trainer.set_text_prototypes(proto_4)
    assert stage1_trainer._text_prototypes is not None
    assert stage1_trainer._text_prototypes.shape[0] == 4

    # Stage-2: rebuild with 18 prototypes (encoder may already be warm-started).
    proto_18 = torch.randn(18, 384)
    stage2_trainer.set_text_prototypes(proto_18)
    assert stage2_trainer._text_prototypes is not None
    assert stage2_trainer._text_prototypes.shape[0] == 18, (
        "Stage-2 prototype bank must hold 18, not 4 (rebuilt per stage)"
    )


# ---------------------------------------------------------------------------
# Test 9: smoke one step per stage on CPU.
# ---------------------------------------------------------------------------


def test_smoke_one_step_per_stage(
    stage1_trainer: FarSLIPDistillationTrainer,
    stage2_trainer: FarSLIPDistillationTrainer,
) -> None:
    """One ``step()`` per stage yields a finite loss > 0 with the right n_categories."""
    torch.manual_seed(42)

    # Stage-1: 4 categories.
    stage1_trainer.set_text_prototypes(torch.randn(4, 384))
    assert stage1_trainer.config.n_categories == 4
    images = torch.rand(2, 4, 224, 224)
    region_ids = torch.zeros(2, dtype=torch.long)
    cat_ids = torch.tensor([0, 3], dtype=torch.long)  # valid in [0, 3]
    out1 = stage1_trainer.step(images, region_ids, cat_ids)
    loss1 = float(out1["loss_total"].detach().cpu().item())
    assert torch.isfinite(out1["loss_total"]).all()
    assert loss1 > 0.0

    # Stage-2: 18 categories.
    stage2_trainer.set_text_prototypes(torch.randn(18, 384))
    assert stage2_trainer.config.n_categories == 18
    cat_ids_18 = torch.tensor([0, 17], dtype=torch.long)  # valid in [0, 17]
    out2 = stage2_trainer.step(images, region_ids, cat_ids_18)
    loss2 = float(out2["loss_total"].detach().cpu().item())
    assert torch.isfinite(out2["loss_total"]).all()
    assert loss2 > 0.0


# ---------------------------------------------------------------------------
# Test 10: --from-scratch skips the warm-start.
# ---------------------------------------------------------------------------


class _SpyTrainer:
    """Lightweight stand-in for :class:`FarSLIPDistillationTrainer`.

    Records whether ``load_state_dict`` was called so the orchestration branch
    (warm-start vs from-scratch) can be asserted without loading CLIP/GPU.
    """

    def __init__(self, config: FarSLIPTrainerConfig, dataset: Any = None) -> None:
        self.config = config
        self.dataset = dataset
        self.load_called = False
        self.set_prototypes_called = False
        self.train_called = False
        self.student = self  # so trainer.student.load_state_dict resolves here

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        self.load_called = True

        class _Incompatible:
            missing_keys: ClassVar[list[str]] = []
            unexpected_keys: ClassVar[list[str]] = []

        return _Incompatible()

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        self.set_prototypes_called = True

    def train(self) -> dict[str, float]:
        self.train_called = True
        return {"loss_total": 1.0, "loss_cls": 1.0, "loss_patch": 0.5, "loss_aux": 0.1}

    def save_student(self, format: str = "safetensors", suffix: str | None = None) -> str:
        return str(self.config.output_dir / f"student_{suffix}.safetensors")


@pytest.fixture()
def patched_run_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Patches ``create_incremental_dataset`` + the trainer in train_incremental.

    Returns a holder that captures the constructed ``_SpyTrainer`` so tests can
    assert on ``load_called`` after invoking :func:`_run_stage`.
    """
    import scripts.train_incremental as ti

    proto = torch.randn(18, 384)
    mini_ds = [
        {
            "image": torch.rand(4, 224, 224),
            "region_id": torch.tensor(0, dtype=torch.long),
            "category_id": torch.tensor(0, dtype=torch.long),
        }
        for _ in range(2)
    ]

    def _fake_create(n_classes: int, **_kwargs: Any) -> tuple[Any, int, int, torch.Tensor]:
        return mini_ds, 1, n_classes, proto[:n_classes]

    captured: dict[str, Any] = {}

    def _fake_trainer(config: FarSLIPTrainerConfig, dataset: Any = None) -> _SpyTrainer:
        spy = _SpyTrainer(config, dataset)
        captured["trainer"] = spy
        return spy

    monkeypatch.setattr(ti, "create_incremental_dataset", _fake_create)
    monkeypatch.setattr(ti, "FarSLIPDistillationTrainer", _fake_trainer)

    # A real checkpoint file so the warm-start branch reaches load_state_dict.
    ckpt = tmp_path / "stage1_best.safetensors"
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(2, 2)}, str(ckpt))
    captured["ckpt"] = ckpt
    captured["module"] = ti
    return captured


def test_from_scratch_skips_warmstart(patched_run_stage: dict[str, Any], tmp_path: Path) -> None:
    """``from_scratch=True`` must NOT call ``load_state_dict`` on the ckpt."""
    stage_dir = tmp_path / "stage2"
    stage_dir.mkdir()
    _run_stage(
        stage=2,
        n_classes=18,
        epochs=1,
        batch_size=2,
        lr=1e-5,
        seed=42,
        folds=(1, 2, 3),
        ratio=3.0,
        pastis_root=Path("data/PASTIS-R"),
        stage_dir=stage_dir,
        run_name="poc-stage2",
        time_cap_hours=4.0,
        from_scratch=True,
        warm_start_ckpt=patched_run_stage["ckpt"],
    )
    spy: _SpyTrainer = patched_run_stage["trainer"]
    assert spy.load_called is False, "from_scratch=True must skip the warm-start"
    assert spy.set_prototypes_called is True
    assert spy.train_called is True


def test_warm_start_loads_when_not_from_scratch(
    patched_run_stage: dict[str, Any], tmp_path: Path
) -> None:
    """Complement of test 10: ``from_scratch=False`` DOES call ``load_state_dict``."""
    stage_dir = tmp_path / "stage2"
    stage_dir.mkdir()
    _run_stage(
        stage=2,
        n_classes=18,
        epochs=1,
        batch_size=2,
        lr=1e-5,
        seed=42,
        folds=(1, 2, 3),
        ratio=3.0,
        pastis_root=Path("data/PASTIS-R"),
        stage_dir=stage_dir,
        run_name="poc-stage2",
        time_cap_hours=4.0,
        from_scratch=False,
        warm_start_ckpt=patched_run_stage["ckpt"],
    )
    spy: _SpyTrainer = patched_run_stage["trainer"]
    assert spy.load_called is True, "from_scratch=False must apply the warm-start"


# ---------------------------------------------------------------------------
# _parse_folds: valid / empty / non-integer (no CLIP, no GPU).
# ---------------------------------------------------------------------------


def test_parse_folds_valid() -> None:
    """A comma-separated fold string parses into an int tuple in order."""
    assert _parse_folds("1,2,3") == (1, 2, 3)
    # Surrounding whitespace and trailing separators are tolerated.
    assert _parse_folds(" 4 , 5 ") == (4, 5)
    assert _parse_folds("2,,3,") == (2, 3)


def test_parse_folds_empty_raises() -> None:
    """An empty (or separator-only) fold string is a BadParameter."""
    import typer

    with pytest.raises(typer.BadParameter, match="empty"):
        _parse_folds(",,")


def test_parse_folds_non_integer_raises() -> None:
    """A non-integer token is a BadParameter, not a silent drop."""
    import typer

    with pytest.raises(typer.BadParameter, match="comma-separated ints"):
        _parse_folds("1,x,3")


# ---------------------------------------------------------------------------
# _load_student_state_dict: .pt branch / missing file / bad suffix.
# ---------------------------------------------------------------------------


def test_load_student_state_dict_pt(tmp_path: Path) -> None:
    """A ``.pt`` checkpoint loads via ``torch.load(weights_only=True)``."""
    ckpt = tmp_path / "student.pt"
    state = {"embeddings.patch_embedding.weight": torch.ones(2, 2)}
    torch.save(state, ckpt)
    loaded = _load_student_state_dict(ckpt)
    assert set(loaded) == set(state)
    assert torch.allclose(loaded["embeddings.patch_embedding.weight"], torch.ones(2, 2))


def test_load_student_state_dict_safetensors(tmp_path: Path) -> None:
    """A ``.safetensors`` checkpoint loads via ``safetensors.torch.load_file``."""
    from safetensors.torch import save_file

    ckpt = tmp_path / "student.safetensors"
    save_file({"w": torch.zeros(2, 2)}, str(ckpt))
    loaded = _load_student_state_dict(ckpt)
    assert torch.allclose(loaded["w"], torch.zeros(2, 2))


def test_load_student_state_dict_missing_raises(tmp_path: Path) -> None:
    """A non-existent checkpoint raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError, match="stage-1 checkpoint not found"):
        _load_student_state_dict(tmp_path / "nope.safetensors")


def test_load_student_state_dict_bad_suffix_raises(tmp_path: Path) -> None:
    """An unsupported suffix raises ``ValueError`` (no silent guess)."""
    ckpt = tmp_path / "student.bin"
    ckpt.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        _load_student_state_dict(ckpt)


# ---------------------------------------------------------------------------
# train(): full Stage-1 -> warm-start -> Stage-2 orchestration (AC-2 / AC-3).
# ---------------------------------------------------------------------------


class _CkptSpyTrainer(_SpyTrainer):
    """``_SpyTrainer`` that actually writes the per-epoch checkpoint file.

    ``_run_stage`` looks for ``student_epoch_{epochs-1}.safetensors`` in the stage
    dir to hand to the next stage; writing it lets ``train()`` run end-to-end
    (Stage-1 -> warm-start Stage-2) without the time-cap fallback or real CLIP.
    """

    n_categories_seen: ClassVar[list[int]] = []
    instances: ClassVar[list[_SpyTrainer]] = []

    def __init__(self, config: FarSLIPTrainerConfig, dataset: Any = None) -> None:
        super().__init__(config, dataset)
        type(self).n_categories_seen.append(config.n_categories)
        type(self).instances.append(self)

    def train(self) -> dict[str, float]:
        # Persist the epoch file the orchestrator expects (POC default 2 epochs
        # -> student_epoch_1.safetensors).
        from safetensors.torch import save_file

        epochs = int(self.config.n_epochs)
        out = self.config.output_dir / f"student_epoch_{epochs - 1}.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeddings.patch_embedding.weight": torch.zeros(2, 2)}, str(out))
        return super().train()


def test_train_runs_both_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``train()`` orchestrates Stage-1 (4) -> warm-start -> Stage-2 (18).

    Mocks ``create_incremental_dataset`` and the trainer so the command exercises
    the full orchestration branch (folds parsing, seed, both stage dirs, warm-start
    of Stage-2) on CPU without CLIP/PASTIS. Asserts AC-2/AC-3: Stage-1 uses 4
    categories, Stage-2 uses 18, and Stage-2 warm-starts (load_state_dict called).
    """
    import scripts.train_incremental as ti

    _CkptSpyTrainer.n_categories_seen = []
    _CkptSpyTrainer.instances = []

    proto = torch.randn(18, 384)

    def _fake_create(n_classes: int, **_kwargs: Any) -> tuple[Any, int, int, torch.Tensor]:
        ds = [
            {
                "image": torch.rand(4, 224, 224),
                "region_id": torch.tensor(0, dtype=torch.long),
                "category_id": torch.tensor(0, dtype=torch.long),
            }
        ]
        return ds, 1, n_classes, proto[:n_classes]

    monkeypatch.setattr(ti, "create_incremental_dataset", _fake_create)
    monkeypatch.setattr(ti, "FarSLIPDistillationTrainer", _CkptSpyTrainer)
    # Avoid touching a real MLflow server.
    monkeypatch.setattr(ti, "propagate_seed", lambda seed: None)

    ti.train(
        run_name="poc-unit",
        stage1_classes=4,
        stage2_classes=18,
        epochs_per_stage=2,
        batch_size=2,
        lr=1e-5,
        seed=42,
        folds="1,2,3",
        ratio=3.0,
        pastis_root=Path("data/PASTIS-R"),
        output_dir=tmp_path,
        from_scratch=False,
        time_cap_hours=4.0,
        mlflow_uri="file://unused",
    )

    # AC-2 / AC-3: Stage-1 saw 4 categories, Stage-2 saw 18.
    assert _CkptSpyTrainer.n_categories_seen == [4, 18]
    # Both stage dirs exist with their epoch checkpoint.
    assert (tmp_path / "poc-unit" / "stage1" / "student_epoch_1.safetensors").exists()
    assert (tmp_path / "poc-unit" / "stage2" / "student_epoch_1.safetensors").exists()
    # Stage-2 (2nd trainer) warm-started; Stage-1 (1st) did not.
    stage1_spy, stage2_spy = _CkptSpyTrainer.instances
    assert stage1_spy.load_called is False
    assert stage2_spy.load_called is True
    assert stage2_spy.set_prototypes_called is True


def test_train_from_scratch_skips_stage2_warmstart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``train(--from-scratch)`` builds both stages but Stage-2 skips the warm-start."""
    import scripts.train_incremental as ti

    _CkptSpyTrainer.n_categories_seen = []
    _CkptSpyTrainer.instances = []

    proto = torch.randn(18, 384)

    def _fake_create(n_classes: int, **_kwargs: Any) -> tuple[Any, int, int, torch.Tensor]:
        ds = [
            {
                "image": torch.rand(4, 224, 224),
                "region_id": torch.tensor(0, dtype=torch.long),
                "category_id": torch.tensor(0, dtype=torch.long),
            }
        ]
        return ds, 1, n_classes, proto[:n_classes]

    monkeypatch.setattr(ti, "create_incremental_dataset", _fake_create)
    monkeypatch.setattr(ti, "FarSLIPDistillationTrainer", _CkptSpyTrainer)
    monkeypatch.setattr(ti, "propagate_seed", lambda seed: None)

    ti.train(
        run_name="poc-scratch",
        stage1_classes=4,
        stage2_classes=18,
        epochs_per_stage=2,
        batch_size=2,
        lr=1e-5,
        seed=42,
        folds="1,2,3",
        ratio=3.0,
        pastis_root=Path("data/PASTIS-R"),
        output_dir=tmp_path,
        from_scratch=True,
        time_cap_hours=4.0,
        mlflow_uri="file://unused",
    )

    assert _CkptSpyTrainer.n_categories_seen == [4, 18]
    _stage1_spy, stage2_spy = _CkptSpyTrainer.instances
    # The fallback control (AC-9): Stage-2 must NOT warm-start from Stage-1.
    assert stage2_spy.load_called is False
