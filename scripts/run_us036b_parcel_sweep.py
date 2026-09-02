"""Parcel-level FarSLIP sweep over N classes {4,6,8,10,12} (US-036-b).

Trains FarSLIP at the PARCEL grain (each parcel is its own crop -> its own CLS,
breaking the 1-CLS-per-patch ceiling) for an increasing number of classes, and
evaluates polygon-to-polygon (per parcel, not per patch). The output is the curve
N vs macro-F1 (parcel): the real ceiling by cardinality.

Per N (= ``active_classes(N)``, the frequency-ordered curriculum):
  1. Build train/val ``ParcelCropDataset`` (spatial CV, anti-leak).
  2. Inject the per-parcel phenology captions (Gemma, diverse) for ``L_glo``.
  3. Train ``L_total = L_glo + lambda_loc * L_loc`` with the parcel collate
     (``region_to_patch = arange(B)`` -> each parcel uses its own CLS).
  4. Evaluate per parcel (``eval_per_parcel``): macro-F1/IoU at the parcel grain.
  5. Persist the best checkpoint + an MLflow run (data_version + code_version).

Reuses the faithful-v2 machinery: ``FarSLIPDistillationTrainer``,
``_category_prototypes``, ``eval_per_parcel``, ``encode_captions_minilm`` /
``make_caption_collate`` (with ``id_key="parcel_ids"``). Real PASTIS-R only;
captions from the local Gemma (cost $0). Conventions: Polars, structlog, type
hints, English docstrings, Spanish prose, no emojis.

Usage (on the H100 VM, env ``agrosat``)::

    python -m scripts.run_us036b_parcel_sweep run \
        --pastis-root data/PASTIS-R \
        --captions-path data/farslip/parcel_phenology_captions.parquet \
        --n-values 4,6,8,10,12 --epochs 30 --batch-size 256
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import polars as pl
import structlog
import torch
import typer

import scripts.run_us036a_v2_farslip_faithful as faithful
from ml.farslip.caption_encoder import encode_captions_minilm, make_caption_collate
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.farslip.parcel_crop_dataset import ParcelCropDataset, collate_parcel_batch
from ml.farslip.pastis_pair_dataset import active_classes
from ml.features.parcel_phenology_captions import load_parcel_captions
from ml.ingest.pastis_loader import PASTIS_R_CLASSES
from ml.utils.seed import propagate_seed

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_MLFLOW_URI = "http://localhost:5010"
_N_IN_CHANNELS = 4


def run_one_n(
    n_classes: int,
    *,
    captions: dict[str, str],
    pastis_root: Path,
    output_root: Path,
    train_folds: tuple[int, ...],
    val_folds: tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    min_area_px: int,
    use_global_caption_loss: bool,
    time_cap_hours: float,
    max_patches: int | None,
    require_caption: bool,
    dominance_ratio: float | None = None,
):
    """Train + eval the parcel-level model for ONE N and return its result.

    Args:
        n_classes: number of classes (uses ``active_classes(n_classes)``).
        captions: ``{parcel_id: description}`` per-parcel captions.
        pastis_root: real PASTIS-R root.
        output_root: parent dir for ``{N:02d}cls/best.safetensors``.
        train_folds/val_folds: official spatial-CV folds (disjoint).
        epochs/batch_size/lr/seed: training hparams.
        min_area_px: minimum parcel area.
        use_global_caption_loss: toggle ``L_glo`` (per-parcel caption InfoNCE).
        time_cap_hours: hard cap forwarded to the trainer.
        max_patches: cap patches scanned (smoke).
        require_caption: keep only captioned parcels (use with a balanced
            caption SAMPLE so empty-caption parcels do not dilute ``L_glo``).
        dominance_ratio: if not ``None``, apply the per-patch 3:1 Meadow filter
            before extracting parcels (A/B the imbalance guard). Default ``None``.

    Returns:
        The :class:`FaithfulRunResult` with parcel-grain metrics.
    """
    from torch.utils.data import DataLoader

    faithful._validate_pastis_root(pastis_root)
    active = active_classes(n_classes)
    propagate_seed(seed)

    output_dir = output_root / f"{n_classes:02d}cls"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"farslip-parcel-n{n_classes}"

    train_ds = ParcelCropDataset(
        captions,
        root=pastis_root,
        folds=train_folds,
        active_class_ids=active,
        min_area_px=min_area_px,
        max_patches=max_patches,
        seed=seed,
        require_caption=require_caption,
        dominance_ratio=dominance_ratio,
    )
    val_ds = ParcelCropDataset(
        captions,
        root=pastis_root,
        folds=val_folds,
        active_class_ids=active,
        min_area_px=min_area_px,
        max_patches=max_patches,
        seed=seed,
        require_caption=require_caption,
        dominance_ratio=dominance_ratio,
    )
    logger.info(
        "parcel_sweep_n_start",
        n_classes=n_classes,
        active=list(active),
        n_train=len(train_ds),
        n_val=len(val_ds),
        run_name=run_name,
    )

    cfg = FarSLIPTrainerConfig(
        dataset_root=pastis_root,
        output_dir=output_dir,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        time_cap_hours=time_cap_hours,
        n_in_channels=_N_IN_CHANNELS,
        n_categories=len(active),
        supervision="region_category",
        use_global_caption_loss=use_global_caption_loss,
        mlflow_run_name=run_name,
        extra_params={
            "us": "US-036-b",
            "grain": "parcel",
            "n_classes": n_classes,
            "dataset": "pastis_r_real",
            "dominance_ratio": ("off" if dominance_ratio is None else f"{dominance_ratio:g}"),
        },
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=train_ds)
    bank, class_ids = faithful._category_prototypes(None, active)
    trainer.set_category_prototypes(bank, class_ids)

    if use_global_caption_loss:
        caption_embeddings = encode_captions_minilm(captions, device=trainer.device.type)
        collate_fn = make_caption_collate(
            collate_parcel_batch, caption_embeddings, id_key="parcel_ids"
        )
    else:
        collate_fn = collate_parcel_batch

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=trainer.device.type == "cuda",
        collate_fn=collate_fn,
    )
    start = time.monotonic()
    train_metrics = trainer.train(loader)

    best_ckpt = output_dir / "best.safetensors"
    last_epoch_ckpt = output_dir / f"student_epoch_{epochs - 1}.safetensors"
    if last_epoch_ckpt.exists():
        best_ckpt.write_bytes(last_epoch_ckpt.read_bytes())
    else:
        best_ckpt = Path(trainer.save_student(format="safetensors", suffix="best"))

    proto_eval = trainer._category_prototypes
    result = faithful.eval_per_parcel(
        trainer.student,
        val_ds,
        proto_eval,
        class_ids,
        device=trainer.device,
        batch_size=batch_size,
    )
    result.best_ckpt = best_ckpt
    result.train_metrics = train_metrics
    logger.info(
        "parcel_sweep_n_done",
        n_classes=n_classes,
        macro_f1=round(result.macro_f1, 4),
        macro_iou=round(result.macro_iou, 4),
        n_well=result.n_classes_well_resolved,
        elapsed_min=round((time.monotonic() - start) / 60.0, 1),
    )
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_parcel_sweep(
    *,
    pastis_root: Path,
    captions_path: Path,
    n_values: tuple[int, ...] = (4, 6, 8, 10, 12),
    train_folds: tuple[int, ...] = (1, 2, 3),
    val_folds: tuple[int, ...] = (4,),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-5,
    seed: int = 42,
    min_area_px: int = 16,
    use_global_caption_loss: bool = True,
    time_cap_hours: float = 8.0,
    output_root: Path = Path("checkpoints/farslip/parcel"),
    metrics_out: Path = Path("reports/farslip/metrics/parcel_sweep.csv"),
    max_patches: int | None = None,
    require_caption: bool = False,
    dominance_ratio: float | None = None,
) -> list:
    """Run the full N-class parcel-level sweep and write the N vs F1 curve.

    Args:
        pastis_root: real PASTIS-R root.
        captions_path: per-parcel phenology captions parquet.
        n_values: cardinalities to sweep (frequency-ordered curriculum).
        train_folds/val_folds: official spatial-CV folds.
        epochs/batch_size/lr/seed: training hparams.
        min_area_px: minimum parcel area.
        use_global_caption_loss: toggle ``L_glo``.
        time_cap_hours: hard cap per run.
        output_root: checkpoints parent dir.
        metrics_out: output CSV with the N vs macro-F1 curve.
        max_patches: cap patches (smoke).
        require_caption: keep only captioned parcels (validation with a sample).
        dominance_ratio: if not ``None``, apply the per-patch 3:1 Meadow filter
            before extracting parcels (A/B the imbalance guard). The metric CSV
            records the value so the with/without curves are self-describing.

    Returns:
        List of :class:`FaithfulRunResult`, one per N.
    """
    captions = load_parcel_captions(captions_path)
    if not captions:
        raise ValueError(
            f"no per-parcel captions in {captions_path}; run the caption generation phase first."
        )
    results = []
    rows = []
    for n in n_values:
        result = run_one_n(
            n,
            captions=captions,
            pastis_root=pastis_root,
            output_root=output_root,
            train_folds=train_folds,
            val_folds=val_folds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            min_area_px=min_area_px,
            use_global_caption_loss=use_global_caption_loss,
            time_cap_hours=time_cap_hours,
            max_patches=max_patches,
            require_caption=require_caption,
            dominance_ratio=dominance_ratio,
        )
        results.append(result)
        rows.append(
            {
                "n_classes": n,
                "macro_f1": round(result.macro_f1, 4),
                "macro_iou": round(result.macro_iou, 4),
                "n_well_resolved": result.n_classes_well_resolved,
                "n_eval_parcels": result.n_eval,
                "dominance_ratio": (-1.0 if dominance_ratio is None else float(dominance_ratio)),
                "best_ckpt": str(result.best_ckpt),
            }
        )
        # Persist the partial curve after each N (resilience between runs).
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(metrics_out)
        logger.info("parcel_sweep_partial_written", n_done=len(rows), path=str(metrics_out))

    logger.info("parcel_sweep_done", n_values=list(n_values), metrics=str(metrics_out))
    return results


def _parse_ints(s: str) -> tuple[int, ...]:
    """Parse a comma-separated int list."""
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


@app.callback()
def _root() -> None:
    """AgroSatCopilot US-036-b: parcel-level FarSLIP N-class sweep (H100)."""


@app.command()
def run(
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    captions_path: Annotated[Path, typer.Option("--captions-path")] = Path(
        "data/farslip/parcel_phenology_captions.parquet"
    ),
    n_values: Annotated[str, typer.Option("--n-values")] = "4,6,8,10,12",
    train_folds: Annotated[str, typer.Option("--train-folds")] = "1,2,3",
    val_folds: Annotated[str, typer.Option("--val-folds")] = "4",
    epochs: Annotated[int, typer.Option("--epochs")] = 30,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 256,
    lr: Annotated[float, typer.Option("--lr")] = 1e-5,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    min_area_px: Annotated[int, typer.Option("--min-area-px")] = 16,
    no_global_caption_loss: Annotated[bool, typer.Option("--no-global-caption-loss")] = False,
    time_cap_hours: Annotated[float, typer.Option("--time-cap-hours")] = 8.0,
    max_patches: Annotated[int, typer.Option("--max-patches")] = 0,
    metrics_out: Annotated[Path, typer.Option("--metrics-out")] = Path(
        "reports/farslip/metrics/parcel_sweep.csv"
    ),
    require_caption: Annotated[
        bool,
        typer.Option(
            "--require-caption",
            help="keep only captioned parcels (use with a balanced SAMPLE)",
        ),
    ] = False,
    dominance_ratio: Annotated[
        float,
        typer.Option(
            "--dominance-ratio",
            help=(
                "per-patch 3:1 Meadow filter ratio (e.g. 3.0); <=0 disables it "
                "(default, parcel grain handles imbalance). A/B against the "
                "unfiltered sweep."
            ),
        ),
    ] = 0.0,
) -> None:
    """Run the parcel-level N-class sweep and write the N vs macro-F1 curve."""
    results = run_parcel_sweep(
        pastis_root=pastis_root,
        captions_path=captions_path,
        n_values=_parse_ints(n_values),
        train_folds=_parse_ints(train_folds),
        val_folds=_parse_ints(val_folds),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        min_area_px=min_area_px,
        use_global_caption_loss=not no_global_caption_loss,
        time_cap_hours=time_cap_hours,
        metrics_out=metrics_out,
        max_patches=max_patches if max_patches > 0 else None,
        require_caption=require_caption,
        dominance_ratio=dominance_ratio if dominance_ratio > 0 else None,
    )
    summary = {
        PASTIS_R_CLASSES.get(0, "summary"): "parcel-sweep",
        "results": [
            {
                "n_classes": r.n_categories,
                "macro_f1": round(r.macro_f1, 4),
                "n_well": r.n_classes_well_resolved,
            }
            for r in results
        ],
    }
    typer.echo(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    app()
