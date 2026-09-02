"""Productive orchestrator of the TSViT-pheno Full-M retrain (US-039).

Re-trains TSViT with the **SAME Full-M capacity as the base US-038**
(``alt-tsvit-fullm-v1``) PLUS the phenology contrastive branch activated with
``lambda_contrast=0.3`` (Wen et al. 2025, "Phenology Description is All You
Need!", ISPRS J. Photogrammetry RS 228, eq. 15-16), over the **real French
PASTIS-R** dataset (folds train=(1,2,3), val=(4); fold-5 reserved held-out). Run
MLflow ``alt-tsvit-pheno-fullm-v1`` (``:5010``, experiment
``agrosat-segmentation``).

Reuse, NOT modification (write-set disjoint from US-025 / US-038):

    - :func:`ml.train.train_segmentation.build_and_train` (IMPORTED, invoked with
      ``model_kind="tsvit-pheno"``, ``lambda_contrast=0.3`` and the Full-M
      capacity kwargs): builds the temporal ``PASTISSegmentationDataset``, the
      Full-M :func:`ml.models.tsvit_wrapper.build_tsvit`, instantiates
      :class:`ml.models.pheno_semantic_branch.PhenoSemanticBranch` and adds
      ``0.3 * phenology_contrastive_loss`` to the Dice+CE inside ``_run_epoch``.
      This script does NOT reimplement the training loop, the loss, or the
      prototype loading.
    - :data:`ml.models.tsvit_wrapper.TSVIT_FULLM_CONFIG` (IMPORTED): the SINGLE
      source of truth for the Full-M capacity (US-038). ``CFG_FULL_TSVIT`` is
      derived from it so the capacity that is TRAINED here is byte-identical to
      the base ``tsvit`` run and the re-score reconstruction (R4, R-HARNESS).
    - ``ml.eval.dense_metrics.rescore_all_checkpoints`` (IMPORTED): the US-030
      fold-5 harness that re-scores the best checkpoint apples-to-apples.

Scope (critical, directive v8): ONLY real French PASTIS-R. No Italian /
synthetic / placeholder data. Pointing ``data_root`` at ``data/farslip_pairs``
(the US-034/035 Italian/synthetic root) is rejected with a clear ``ValueError``.
The US-033 prototypes parquet is read-only (never regenerated, Gemini is NEVER
called: ``build_and_train`` loads it via ``PhenoSemanticBranch`` /
``load_class_prototype_embeddings``).

**HONESTY (US-039 risk #1)**: in the SUPERVISED setting the model already learns
the temporal crop signatures from the dense labels, so the contrastive branch
adds LITTLE marginal value. The historical ``+0.004`` mIoU (fold-4) is NOISE,
not an improvement. This orchestrator reports the Delta vs the base US-038
``tsvit`` EXACTLY as it comes out (positive, ~0 or negative) and frames it as a
controlled ablation. The real value of phenology is in the self-supervised /
zero-shot FarSLIP (US-036-a), NOT here. The delta table carries that note.

Project convention: ``structlog`` (no ``print``); type hints everywhere;
docstrings in English, prose in Spanish; no emojis; checkpoints under the
relative path ``checkpoints/segmentation/tsvit-pheno-fullm-v1/`` (lands on ``F:``
on the H100 VM via ``--ckpt-dir``).

Usage (from repo root, when the GPU is free, AFTER US-036-a and US-038):
    poetry run python scripts/run_us039_tsvit_pheno_fullm.py run \\
        --device cuda --num-workers 10 \\
        --ckpt-dir F:/checkpoints/segmentation/tsvit-pheno-fullm-v1
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import structlog
import typer

from ml.models.tsvit_wrapper import TSVIT_FULLM_CONFIG

logger = structlog.get_logger(__name__)

#: Repo root resolved from this file (``scripts/<this>.py`` -> repo). Anchors the
#: default checkpoint dir and the honest delta-table output regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: MLflow run name fixed by the v8 criteria (AC-4). Selects the Full-M pheno run.
_RUN_NAME = "alt-tsvit-pheno-fullm-v1"

#: Registry key / checkpoint slug of the Full-M pheno checkpoint (AC-7, R10).
#: COEXISTS with the historical L4 ``tsvit-pheno`` (US-030), never overwrites it.
_REGISTRY_KEY = "tsvit-pheno-fullm"

#: Weight of the phenology contrastive term, fixed by the v8 criteria (AC-1).
_LAMBDA_CONTRAST = 0.3

#: MLflow tracking URI default (Docker server, gotcha two-stores: the lineage
#: lives on the server, NOT in ./mlruns). The H100 VM may use a native SQLite
#: store and ingest afterwards (US-034/035 pattern).
_DEFAULT_MLFLOW_URI = "http://localhost:5010"

#: Forbidden Italian/synthetic data root (US-034/035 path, discarded here).
_FORBIDDEN_ROOT_NAME = "farslip_pairs"

#: Official PASTIS-R folds. Train=(1,2,3), val=(4); fold-5 reserved held-out for
#: the re-score (NEVER in train/val -> no leakage, US-039 R7).
_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3)
_VAL_FOLDS: tuple[int, ...] = (4,)
_HELDOUT_FOLD = 5

#: Default delta-table path (small derived report -> Git, not DVC; AC-6).
_DEFAULT_DELTA_CSV = (
    _REPO_ROOT / "reports" / "segmentation" / "metrics" / "tsvit_pheno_vs_base_fold5.csv"
)

#: The honest framing note that travels with the delta table AND the handoff (R1).
_HONESTY_NOTE = (
    "Delta esperado ~0.3%, NO 5%; en supervisado la fenologia aporta poco "
    "(saturacion supervisada); el valor real esta en el contrastivo self-sup "
    "de FarSLIP US-036-a, NO aqui. Delta < 0 es resultado valido."
)


#: Full-M TSViT config for ``build_and_train`` (SINGLE source of truth for the
#: capacity = US-038 ``TSVIT_FULLM_CONFIG``; the train hyper-parameters mirror the
#: US-038 base run so the ONLY controlled difference is ``use_phenology=True`` +
#: ``lambda_contrast=0.3`` -- the definition of an apples-to-apples ablation,
#: US-039 §2). ``build_and_train`` fixes ``img_size=128``, ``patch_size=8``,
#: ``mlp_ratio=4``, ``semantic_dim=384`` internally, so only the capacity knobs it
#: exposes (``n_timesteps``/``dim``/``depth_*``/``heads``/``dim_head``) plus the
#: train schedule are passed here.
CFG_FULL_TSVIT: dict[str, Any] = {
    # --- Full-M capacity (from US-038 TSVIT_FULLM_CONFIG, the single truth) ---
    "n_timesteps": TSVIT_FULLM_CONFIG["n_timesteps"],  # 64 (>= PASTIS T_max ~61)
    "dim": TSVIT_FULLM_CONFIG["dim"],  # 192
    "depth_temporal": TSVIT_FULLM_CONFIG["depth_temporal"],  # 6
    "depth_spatial": TSVIT_FULLM_CONFIG["depth_spatial"],  # 6
    "heads": TSVIT_FULLM_CONFIG["heads"],  # 6
    "dim_head": TSVIT_FULLM_CONFIG["dim_head"],  # 64
    # --- Train schedule (mirrors US-038 base run; H100 96GB) ---
    "epochs": 40,
    "batch_size": 20,
    "lr": 1e-3,
    "train_folds": _TRAIN_FOLDS,
    "val_folds": _VAL_FOLDS,
    "target": "semantic18",
}


def _validate_data_root(data_root: Path | None) -> None:
    """Reject the Italian/synthetic root; require a real PASTIS-R root.

    The US-034/035 Italian crops lived under ``data/farslip_pairs``. US-039 is
    PASTIS-R-only: pointing the run at that directory (or any path whose name
    contains it) is a scope violation and fails fast (US-039 R6,
    ``test_rejects_synthetic_root``).

    Args:
        data_root: Candidate PASTIS-R root, or ``None`` to use the dataset
            default (which is the real PASTIS-R).

    Raises:
        ValueError: if ``data_root`` points at the forbidden synthetic root.
    """
    if data_root is None:
        return
    parts = {p.lower() for p in data_root.parts}
    if _FORBIDDEN_ROOT_NAME in data_root.name.lower() or _FORBIDDEN_ROOT_NAME in parts:
        raise ValueError(
            f"data_root {data_root!s} points at the Italian/synthetic "
            f"'{_FORBIDDEN_ROOT_NAME}' data (US-034/035, discarded). US-039 is "
            "PASTIS-R-only: pass a real PASTIS-R root (e.g. data/PASTIS-R)."
        )


def build_pheno_vs_base_table(
    *,
    base_miou: float,
    base_f1_macro: float,
    pheno_miou: float,
    pheno_f1_macro: float,
    fold: int = _HELDOUT_FOLD,
    out_path: Path | None = None,
) -> pl.DataFrame:
    """Build the honest pheno-vs-base fold-5 delta table (AC-6).

    Produces a 2-row table (base US-038 ``tsvit`` Full-M, pheno US-039 Full-M)
    over the SAME held-out fold-5, with the signed deltas and the explicit honesty
    note. The delta is ``pheno - base``: positive, ~0 or negative are ALL valid
    and reported as-is (US-039 R1, R2). A ``delta_miou >= +0.03`` is to be treated
    with suspicion (leak/seed/config), not celebrated -- the prose of the handoff
    must say so; this function only records the number honestly.

    Args:
        base_miou: Fold-5 mIoU of the base US-038 ``tsvit`` Full-M (no phenology).
        base_f1_macro: Fold-5 F1-macro of the base US-038 ``tsvit`` Full-M.
        pheno_miou: Fold-5 mIoU of this US-039 ``tsvit-pheno-fullm``.
        pheno_f1_macro: Fold-5 F1-macro of this US-039 ``tsvit-pheno-fullm``.
        fold: Held-out fold scored (5 = official held-out, NOT the fold-4
            selection set).
        out_path: If given, write the table as CSV (small derived report -> Git,
            not DVC).

    Returns:
        A 2-row Polars DataFrame with columns ``model``, ``fold``, ``miou``,
        ``f1_macro``, ``delta_miou``, ``delta_f1_macro``, ``note``. The base row
        carries null deltas; the pheno row carries the signed deltas.
    """
    delta_miou = pheno_miou - base_miou
    delta_f1 = pheno_f1_macro - base_f1_macro
    table = pl.DataFrame(
        [
            {
                "model": "tsvit-fullm-base",
                "fold": fold,
                "miou": base_miou,
                "f1_macro": base_f1_macro,
                "delta_miou": None,
                "delta_f1_macro": None,
                "note": _HONESTY_NOTE,
            },
            {
                "model": "tsvit-pheno-fullm",
                "fold": fold,
                "miou": pheno_miou,
                "f1_macro": pheno_f1_macro,
                "delta_miou": delta_miou,
                "delta_f1_macro": delta_f1,
                "note": _HONESTY_NOTE,
            },
        ],
        schema={
            "model": pl.Utf8,
            "fold": pl.Int64,
            "miou": pl.Float64,
            "f1_macro": pl.Float64,
            "delta_miou": pl.Float64,
            "delta_f1_macro": pl.Float64,
            "note": pl.Utf8,
        },
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(out_path)
        logger.info(
            "pheno_vs_base_table_written",
            path=str(out_path),
            delta_miou=round(delta_miou, 4),
            delta_f1_macro=round(delta_f1, 4),
        )
    return table


def _rescore_pheno_fold5(
    *,
    device: str,
    rescore_fn: Callable[..., pl.DataFrame] | None = None,
) -> dict[str, float]:
    """Re-score the Full-M pheno + base on the held-out fold-5 (US-030 harness).

    Invokes ``rescore_all_checkpoints(fold=5)`` (the US-030 harness, apples-to-
    apples 18-class 128 NEAREST) and extracts the rows of the base US-038
    ``tsvit`` (Full-M, no phenology) and this US-039 ``tsvit-pheno-fullm``. The
    harness reconstructs both with the Full-M ``model_kwargs`` (R-HARNESS) so the
    ``best.pt`` loads without shape mismatch.

    Args:
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` for the harness inference.
        rescore_fn: Injection seam for the US-030 harness (tests pass a mock);
            ``None`` uses ``ml.eval.dense_metrics.rescore_all_checkpoints``.

    Returns:
        ``{"base_miou", "base_f1_macro", "pheno_miou", "pheno_f1_macro"}`` of the
        fold-5 re-score (NaN-free; missing rows yield 0.0 with a warning).
    """
    if rescore_fn is None:
        from ml.eval.dense_metrics import rescore_all_checkpoints

        rescore_fn = rescore_all_checkpoints

    df = rescore_fn(fold=_HELDOUT_FOLD, device=device, skip_missing=True)

    def _metric(model_name: str, column: str) -> float:
        rows = df.filter(pl.col("model") == model_name)
        if rows.height == 0:
            logger.warning("rescore_row_missing", model=model_name, column=column)
            return 0.0
        value = rows.get_column(column).to_list()[0]
        return float(value) if value is not None else 0.0

    return {
        "base_miou": _metric("tsvit", "miou"),
        "base_f1_macro": _metric("tsvit", "f1_macro"),
        "pheno_miou": _metric(_REGISTRY_KEY, "miou"),
        "pheno_f1_macro": _metric(_REGISTRY_KEY, "f1_macro"),
    }


def run_tsvit_pheno_full(
    *,
    cfg_full: dict[str, Any] | None = None,
    lambda_contrast: float = _LAMBDA_CONTRAST,
    run_name: str = _RUN_NAME,
    ckpt_dir: str | Path | None = None,
    data_root: Path | None = None,
    mlflow_uri: str | None = _DEFAULT_MLFLOW_URI,
    device: str = "auto",
    num_workers: int = 0,
    resume: bool = True,
    rescore_fold5: bool = True,
    delta_csv: Path | None = None,
    train_fn: Callable[..., dict[str, float]] | None = None,
    rescore_fn: Callable[..., pl.DataFrame] | None = None,
) -> dict[str, float]:
    """Retrain TSViT with the US-038 Full-M config plus the phenology branch.

    Reuses :func:`ml.train.train_segmentation.build_and_train` with
    ``model_kind="tsvit-pheno"`` and ``lambda_contrast=0.3``; it does NOT
    reimplement the training loop, the contrastive loss, or the prototype loading
    (``build_and_train`` instantiates :class:`PhenoSemanticBranch`, which reads the
    DVC-tracked US-033 parquet read-only -- Gemini is NEVER called). After
    training, optionally re-scores the best checkpoint on the held-out fold-5 with
    the US-030 harness and builds the honest ``pheno_vs_base`` delta table.

    The capacity (``n_timesteps``/``dim``/``depth_*``/``heads``/``dim_head``) flows
    from ``cfg_full`` (defaults to :data:`CFG_FULL_TSVIT`, derived from
    :data:`ml.models.tsvit_wrapper.TSVIT_FULLM_CONFIG`), so it is byte-identical to
    the base US-038 ``tsvit`` run -- the ONLY controlled difference is the
    phenology branch (apples-to-apples ablation, US-039 §2).

    Args:
        cfg_full: Full TSViT config (single source of truth). ``None`` uses
            :data:`CFG_FULL_TSVIT`. Must NOT carry ``lambda_contrast`` (passed
            separately) nor ``model_kind`` (fixed to ``"tsvit-pheno"``).
        lambda_contrast: Weight of the phenology contrastive term (0.3, fixed by
            the v8 criteria).
        run_name: MLflow run name (``"alt-tsvit-pheno-fullm-v1"``).
        ckpt_dir: Checkpoint directory; default
            ``checkpoints/segmentation/tsvit-pheno-fullm-v1`` (relative; lands on
            ``F:`` on the VM via the CLI override).
        data_root: PASTIS-R root override; ``None`` uses the dataset default.
            Rejected if it points at the synthetic ``farslip_pairs`` root.
        mlflow_uri: MLflow tracking URI override (Docker ``:5010`` default).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        num_workers: ``DataLoader`` workers (0 on Windows/CI).
        resume: Resume from ``last.pt`` if present (H100 window may be cut).
        rescore_fold5: If ``True`` re-scores the best on fold-5 via the US-030
            harness and writes the ``pheno_vs_base`` delta table.
        delta_csv: Output path of the delta table; ``None`` uses
            :data:`_DEFAULT_DELTA_CSV`.
        train_fn: Injection seam for ``build_and_train`` (tests pass a mock);
            ``None`` imports the real one.
        rescore_fn: Injection seam for the US-030 harness (tests pass a mock).

    Returns:
        A dict with the best validation (fold-4) metrics ``{"val_miou",
        "val_f1_macro", "val_pixel_acc"}`` and, if ``rescore_fold5`` is ``True``,
        the fold-5 re-score metrics and the honest deltas (``fold5_base_miou``,
        ``fold5_pheno_miou``, ``delta_miou``, ``delta_f1_macro``).

    Raises:
        ValueError: if ``data_root`` is the forbidden synthetic root, or if
            ``cfg_full`` smuggles ``lambda_contrast`` / ``model_kind``.
    """
    _validate_data_root(data_root)

    cfg = dict(cfg_full) if cfg_full is not None else dict(CFG_FULL_TSVIT)
    for forbidden in ("lambda_contrast", "model_kind"):
        if forbidden in cfg:
            raise ValueError(
                f"cfg_full must not contain {forbidden!r}: it is passed "
                "explicitly to keep the ablation contract single-sourced."
            )

    if train_fn is None:
        from ml.train.train_segmentation import build_and_train

        train_fn = build_and_train

    resolved_ckpt_dir = (
        Path(ckpt_dir)
        if ckpt_dir is not None
        else _REPO_ROOT / "checkpoints" / "segmentation" / "tsvit-pheno-fullm-v1"
    )

    logger.info(
        "us039_train_start",
        run_name=run_name,
        lambda_contrast=lambda_contrast,
        ckpt_dir=str(resolved_ckpt_dir),
        device=device,
        **{k: cfg[k] for k in ("n_timesteps", "dim", "epochs", "batch_size")},
    )

    best = train_fn(
        "tsvit-pheno",
        lambda_contrast=lambda_contrast,
        mlflow_run_name=run_name,
        mlflow_uri=mlflow_uri,
        ckpt_dir=resolved_ckpt_dir,
        device=device,
        num_workers=num_workers,
        resume=resume,
        **cfg,
    )

    result: dict[str, float] = {
        "val_miou": float(best.get("miou", 0.0)),
        "val_f1_macro": float(best.get("f1_macro", 0.0)),
        "val_pixel_acc": float(best.get("pixel_acc", 0.0)),
    }

    if rescore_fold5:
        scored = _rescore_pheno_fold5(device=device, rescore_fn=rescore_fn)
        out_path = delta_csv if delta_csv is not None else _DEFAULT_DELTA_CSV
        build_pheno_vs_base_table(
            base_miou=scored["base_miou"],
            base_f1_macro=scored["base_f1_macro"],
            pheno_miou=scored["pheno_miou"],
            pheno_f1_macro=scored["pheno_f1_macro"],
            out_path=out_path,
        )
        result.update(
            {
                "fold5_base_miou": scored["base_miou"],
                "fold5_pheno_miou": scored["pheno_miou"],
                "delta_miou": scored["pheno_miou"] - scored["base_miou"],
                "delta_f1_macro": (scored["pheno_f1_macro"] - scored["base_f1_macro"]),
            }
        )

    logger.info("us039_done", run_name=run_name, **result)
    return result


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    num_workers: Annotated[
        int, typer.Option(help="Workers del DataLoader (10-12 en la H100).")
    ] = 0,
    ckpt_dir: Annotated[
        str | None,
        typer.Option(help="Dir checkpoints (cae en F: en la VM H100)."),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option(help="Raiz PASTIS-R (frances real); rechaza farslip_pairs."),
    ] = None,
    mlflow_uri: Annotated[
        str, typer.Option(help="MLflow tracking URI (Docker :5010).")
    ] = _DEFAULT_MLFLOW_URI,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Reanudar desde last.pt."),
    ] = True,
    rescore_fold5: Annotated[
        bool,
        typer.Option(
            "--rescore/--no-rescore",
            help="Re-score fold-5 + tabla delta honesta tras el run.",
        ),
    ] = True,
) -> None:
    """Lanza el retrain TSViT-pheno Full-M (lambda=0.3) sobre PASTIS-R real.

    Reusa ``build_and_train("tsvit-pheno", lambda_contrast=0.3, **CFG_FULL_TSVIT)``
    (capacidad Full-M de US-038), entrena con la rama contrastiva fenologica
    activada, y tras converger re-score el fold-5 held-out con el harness US-030 y
    construye la tabla honesta ``tsvit_pheno_vs_base_fold5.csv``. NO reimplementa
    el loop ni regenera los prototipos US-033 (Gemini NUNCA se llama).
    """
    result = run_tsvit_pheno_full(
        device=device,
        num_workers=num_workers,
        ckpt_dir=ckpt_dir,
        data_root=data_root,
        mlflow_uri=mlflow_uri,
        resume=resume,
        rescore_fold5=rescore_fold5,
    )
    logger.info("us039_cli_done", **{k: round(v, 4) for k, v in result.items()})


if __name__ == "__main__":
    app()
