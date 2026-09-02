"""Typer CLI for the n-class incremental FarSLIP protocol over real PASTIS-R (US-036).

Orchestrates the two-stage incremental protocol on top of the **real French
PASTIS-R** dataset, validating the *mechanics* of staged training with a short
proof-of-concept (2 epochs per stage by default):

    Stage-1 (4 classes):  create_incremental_dataset(4) -> trainer(n_categories=4)
                          -> set_text_prototypes(proto_4) -> train(2 ep)
                          -> save best student to <run>/stage1/.
    Stage-2 (18 classes): trainer(n_categories=18);
                          if NOT --from-scratch: student.load_state_dict(
                              best_stage1, strict=False)  # transfers the ENCODER,
                              NOT the text prototypes (those live outside the
                              state_dict and are rebuilt per stage);
                          set_text_prototypes(proto_18) -> train(2 ep)
                          -> save best student to <run>/stage2/.

Reuse, NOT modification: the heavy machinery already exists in US-034 and is
consumed by composition:

    - :class:`ml.farslip.distill.FarSLIPDistillationTrainer` (AdamW bf16 loop,
      ``set_text_prototypes`` 384->768 frozen orthogonal lift, ``save_student``
      that persists ONLY the vision encoder state_dict, its own MLflow
      ``start_run``/``end_run`` lifecycle with ``data_version`` + ``code_version``),
    - :func:`ml.farslip.pastis_pair_dataset.create_incremental_dataset` (frozen
      contract from US-036 ml/A: real PASTIS-R peak-NDVI pairs, 3:1 filter,
      curriculum, direct US-033 prototypes — NO ``expand_to_cap``).

Scope (critical, ordered by the user 2026-06-07): ONLY real French PASTIS-R.
No Italian / synthetic / placeholder data, no ``data/farslip_pairs``, no
``expand_to_cap`` / CAP bridge. ``n_regions`` is always 1; the prototypes come
DIRECTLY from ``create_incremental_dataset`` (``proto_active``).

MLflow: each stage is ONE fully-closed run in experiment ``farslip`` on the
Docker server ``:5010`` (the trainer owns the ``start_run``/``end_run`` lifecycle
in a ``finally`` block, so runs never stay ``RUNNING``; the subprocess gotcha
does not apply here because the run is closed inside the same process). The
``stage`` / ``n_categories`` / ``from_scratch`` / ``seed`` metadata is logged as
MLflow params via ``cfg.extra_params``; ``data_version`` (PASTIS-R) +
``code_version`` (git SHA) are tagged by the trainer.

Project convention: ``structlog`` (no ``print``); type hints everywhere;
docstrings in English; checkpoints under the relative path
``checkpoints/farslip-incremental/<run>/stage{1,2}/`` (lands on ``F:`` on the VM).

Typical usage on the H100::

    poetry run python -m scripts.train_incremental train \\
        --run-name farslip-incremental-poc \\
        --stage1-classes 4 --stage2-classes 18 \\
        --epochs-per-stage 2 --batch-size 64 --lr 1e-5 --seed 42 \\
        --folds 1,2,3 --ratio 3.0 \\
        --pastis-root data/PASTIS-R \\
        --output-dir checkpoints/farslip-incremental
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the train CLI. poetry add typer") from exc

from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.farslip.pastis_pair_dataset import create_incremental_dataset
from ml.utils.seed import propagate_seed

_log = structlog.get_logger(__name__)

#: Default MLflow tracking server (Docker on :5010); the lineage lives here, NOT
#: in ``./mlruns``. Overridable with ``--mlflow-uri`` (e.g. a local SQLite file
#: for CI/dry-runs).
_DEFAULT_MLFLOW_URI = "http://localhost:5010"

#: Number of Sentinel-2 input channels of the composite (B02, B03, B04, B08).
_N_IN_CHANNELS = 4


def _parse_folds(folds: str) -> tuple[int, ...]:
    """Parses a comma-separated fold string into a tuple of ints.

    Args:
        folds: comma-separated PASTIS fold ids (e.g. ``"1,2,3"``).

    Returns:
        Tuple of fold ids in declaration order.

    Raises:
        typer.BadParameter: if any token is not a valid integer or the result
            is empty.
    """
    try:
        parsed = tuple(int(tok) for tok in folds.split(",") if tok.strip() != "")
    except ValueError as exc:
        raise typer.BadParameter(f"folds must be comma-separated ints: {folds!r}") from exc
    if not parsed:
        raise typer.BadParameter(f"folds is empty: {folds!r}")
    return parsed


def _load_student_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Loads a student state_dict from a ``.safetensors`` or ``.pt`` checkpoint.

    ``save_student`` writes safetensors by default (US-034); ``.pt`` is the
    PyTorch fallback. We dispatch on the suffix: safetensors via
    :func:`safetensors.torch.load_file`, ``.pt`` via ``torch.load`` with
    ``weights_only=True`` (no pickled code is executed).

    Args:
        checkpoint: path to the Stage-1 best student weights.

    Returns:
        The student vision-encoder ``state_dict`` (CPU tensors).

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
        ValueError: if the suffix is neither ``.safetensors`` nor ``.pt``.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"stage-1 checkpoint not found: {checkpoint}")
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint))
    if checkpoint.suffix == ".pt":
        return torch.load(checkpoint, weights_only=True, map_location="cpu")
    raise ValueError(
        f"unsupported checkpoint format {checkpoint.suffix!r}; expected .safetensors or .pt"
    )


def _run_stage(
    *,
    stage: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    folds: tuple[int, ...],
    ratio: float,
    pastis_root: Path,
    stage_dir: Path,
    run_name: str,
    time_cap_hours: float,
    from_scratch: bool,
    warm_start_ckpt: Path | None,
) -> tuple[dict[str, float], Path, list[str], list[str]]:
    """Builds the trainer for one stage, optionally warm-starts it, and trains.

    Stage-2 warm-start (``warm_start_ckpt`` set and not ``from_scratch``) loads
    the Stage-1 best student via ``load_state_dict(strict=False)``: this
    transfers the 100% of the vision encoder (``patch_embedding`` 4-channel +
    transformer blocks + CLS), NOT the text prototypes — those are NOT in the
    student ``state_dict`` (they are injected via ``set_text_prototypes`` and
    stored as a plain attribute), so they are rebuilt per stage. The
    ``missing_keys`` / ``unexpected_keys`` of the load are logged (R-STRICT
    guard: a silently-ignored key would mean a phantom warm-start).

    Args:
        stage: 1 or 2 (only for logging / MLflow params).
        n_classes: active PASTIS classes for this stage (4 or 18).
        epochs: epochs to train this stage (POC default 2).
        batch_size: DataLoader batch size.
        lr: AdamW learning rate.
        seed: determinism seed.
        folds: PASTIS folds (spatial CV).
        ratio: 3:1 Meadow filter ratio.
        pastis_root: PASTIS-R root (drives MLflow ``data_version``).
        stage_dir: output directory for this stage's checkpoints.
        run_name: MLflow run name for this stage.
        time_cap_hours: hard cap forwarded to the trainer.
        from_scratch: if ``True``, skip the warm-start even if a Stage-1
            checkpoint is provided (control for the fallback gate AC-9).
        warm_start_ckpt: Stage-1 best checkpoint to warm-start from (Stage-2
            only); ``None`` for Stage-1.

    Returns:
        Tuple ``(metrics, best_checkpoint_path, missing_keys, unexpected_keys)``.
        ``missing_keys`` / ``unexpected_keys`` are empty when no warm-start runs.
    """
    dataset, n_regions, n_categories, proto_active = create_incremental_dataset(
        n_classes,
        root=pastis_root,
        folds=folds,
        ratio=ratio,
        seed=seed,
    )
    _log.info(
        "stage dataset ready",
        stage=stage,
        n_classes=n_classes,
        n_regions=n_regions,
        n_categories=n_categories,
        n_samples=len(dataset),
        proto_shape=tuple(proto_active.shape),
    )

    cfg = FarSLIPTrainerConfig(
        dataset_root=pastis_root,  # drives MLflow data_version -> PASTIS-R
        output_dir=stage_dir,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        time_cap_hours=time_cap_hours,
        n_in_channels=_N_IN_CHANNELS,
        n_regions=n_regions,  # always 1 for PASTIS
        n_categories=n_categories,
        mlflow_run_name=run_name,
        # Logged verbatim as MLflow params (the trainer also tags data_version +
        # code_version). These give the run the stage/n_categories/from_scratch/
        # seed context the plan requires, without touching distill.py.
        extra_params={
            "stage": stage,
            "n_categories": n_categories,
            "n_regions": n_regions,
            "from_scratch": from_scratch,
            "incremental_seed": seed,
            "folds": ",".join(str(f) for f in folds),
            "ratio": ratio,
            "dataset": "pastis_r_real",
        },
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=dataset)

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if warm_start_ckpt is not None and not from_scratch:
        state_dict = _load_student_state_dict(warm_start_ckpt)
        incompatible = trainer.student.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        _log.info(
            "stage-2 warm-start applied (strict=False)",
            checkpoint=str(warm_start_ckpt),
            n_missing_keys=len(missing_keys),
            n_unexpected_keys=len(unexpected_keys),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
    else:
        _log.info(
            "stage trained without warm-start",
            stage=stage,
            from_scratch=from_scratch,
            had_checkpoint=warm_start_ckpt is not None,
        )

    # Prototypes are rebuilt per stage: the encoder transfers, the bank does not.
    trainer.set_text_prototypes(proto_active)

    metrics = trainer.train()

    # ``save_student`` writes one safetensors per epoch during the loop; the
    # "best" we hand to the next stage / DVC is the LAST epoch (POC: no early
    # stopping — convergence is read from the per-epoch MLflow curve, AC-8).
    best_ckpt = stage_dir / f"student_epoch_{epochs - 1}.safetensors"
    if not best_ckpt.exists():
        # Defensive: if the loop hit the time cap before the last epoch, fall
        # back to a final explicit save so the next stage has weights to load.
        trainer.save_student(format="safetensors", suffix="best")
        best_ckpt = stage_dir / "student_best.safetensors"
    _log.info(
        "stage done",
        stage=stage,
        best_checkpoint=str(best_ckpt),
        **{k: v for k, v in metrics.items()},
    )
    return metrics, best_ckpt, missing_keys, unexpected_keys


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def train(
    run_name: Annotated[
        str, typer.Option(help="Nombre base del run MLflow")
    ] = "farslip-incremental-poc",
    stage1_classes: Annotated[int, typer.Option(help="Clases en Stage-1")] = 4,
    stage2_classes: Annotated[int, typer.Option(help="Clases en Stage-2")] = 18,
    epochs_per_stage: Annotated[int, typer.Option(help="Epochs por stage (POC=2)")] = 2,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    folds: Annotated[str, typer.Option(help="Folds PASTIS, coma-separados")] = "1,2,3",
    ratio: Annotated[float, typer.Option(help="Ratio filtro 3:1 Meadow")] = 3.0,
    pastis_root: Annotated[Path, typer.Option(help="Raiz PASTIS-R")] = Path("data/PASTIS-R"),
    output_dir: Annotated[Path, typer.Option(help="Dir checkpoints (cae en F: en la VM)")] = Path(
        "checkpoints/farslip-incremental"
    ),
    from_scratch: Annotated[
        bool,
        typer.Option(help="Stage-2 SIN warm-start (control del gate de fallback)"),
    ] = False,
    time_cap_hours: Annotated[float, typer.Option(help="Hard cap horas")] = 4.0,
    mlflow_uri: Annotated[
        str,
        typer.Option(help="MLflow tracking URI (Docker :5010; SQLite file:// para CI)"),
    ] = _DEFAULT_MLFLOW_URI,
) -> None:
    """Entrena el protocolo incremental 4->18 sobre PASTIS-R real (POC 2ep/stage).

    Stage-1: ``create_incremental_dataset(stage1_classes)`` -> trainer
    (``n_categories=stage1_classes``) -> ``epochs_per_stage`` epochs. Stage-2:
    trainer (``n_categories=stage2_classes``); si NO ``--from-scratch``,
    ``load_state_dict(stage1_best, strict=False)`` (transfiere el encoder, no los
    prototipos); ``set_text_prototypes(proto_18)``; ``epochs_per_stage`` epochs.
    Dos runs MLflow (``:5010``, experiment ``farslip``) con ``stage`` +
    ``n_categories`` + ``from_scratch`` + ``seed`` (params) y ``data_version`` +
    ``code_version`` (tags); ambos CERRADOS. Checkpoints en
    ``<output_dir>/<run_name>/stage{1,2}/``.

    Args:
        run_name: MLflow run base name (stage suffix appended per stage).
        stage1_classes: active classes in Stage-1 (default 4).
        stage2_classes: active classes in Stage-2 (default 18).
        epochs_per_stage: epochs per stage (POC default 2).
        batch_size: DataLoader batch size.
        lr: AdamW learning rate.
        seed: determinism seed.
        folds: comma-separated PASTIS folds (spatial CV).
        ratio: 3:1 Meadow dominance filter ratio.
        pastis_root: PASTIS-R root.
        output_dir: checkpoints root (relative -> lands on F: on the VM).
        from_scratch: if set, Stage-2 skips the warm-start (fallback control).
        time_cap_hours: hard cap forwarded to each stage trainer.
        mlflow_uri: MLflow tracking URI.
    """
    propagate_seed(seed)
    parsed_folds = _parse_folds(folds)

    # Point MLflow at the :5010 Docker server (lineage lives there, not ./mlruns).
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
    except ImportError:  # pragma: no cover - mlflow optional in dry-runs
        _log.warning("mlflow not installed; training runs without remote tracking")

    run_root = output_dir / run_name
    stage1_dir = run_root / "stage1"
    stage2_dir = run_root / "stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    _log.info(
        "incremental protocol start",
        run_name=run_name,
        stage1_classes=stage1_classes,
        stage2_classes=stage2_classes,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        folds=list(parsed_folds),
        ratio=ratio,
        from_scratch=from_scratch,
        pastis_root=str(pastis_root),
        output_dir=str(run_root),
        device="cuda" if torch.cuda.is_available() else "cpu",
        mlflow_uri=mlflow_uri,
    )

    start = time.monotonic()

    # ---- STAGE-1 (4 classes) -------------------------------------------------
    metrics1, best_stage1, _, _ = _run_stage(
        stage=1,
        n_classes=stage1_classes,
        epochs=epochs_per_stage,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        folds=parsed_folds,
        ratio=ratio,
        pastis_root=pastis_root,
        stage_dir=stage1_dir,
        run_name=f"{run_name}-stage1",
        time_cap_hours=time_cap_hours,
        from_scratch=False,  # Stage-1 has no warm-start by definition
        warm_start_ckpt=None,
    )

    # ---- STAGE-2 (18 classes, warm-start unless --from-scratch) --------------
    metrics2, best_stage2, missing, unexpected = _run_stage(
        stage=2,
        n_classes=stage2_classes,
        epochs=epochs_per_stage,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        folds=parsed_folds,
        ratio=ratio,
        pastis_root=pastis_root,
        stage_dir=stage2_dir,
        run_name=f"{run_name}-stage2",
        time_cap_hours=time_cap_hours,
        from_scratch=from_scratch,
        warm_start_ckpt=best_stage1,
    )

    elapsed_h = (time.monotonic() - start) / 3600.0
    _log.info(
        "incremental protocol done",
        run_name=run_name,
        elapsed_hours=round(elapsed_h, 4),
        from_scratch=from_scratch,
        best_stage1=str(best_stage1),
        best_stage2=str(best_stage2),
        warm_start_missing_keys=len(missing),
        warm_start_unexpected_keys=len(unexpected),
        stage1_loss_cls=metrics1.get("loss_cls"),
        stage2_loss_cls=metrics2.get("loss_cls"),
        stage1_loss_total=metrics1.get("loss_total"),
        stage2_loss_total=metrics2.get("loss_total"),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
