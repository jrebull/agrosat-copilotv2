"""US-041 closing run -- Ensamble E-a: TSViT-pheno + FarSLIP dual-head fusion.

Orchestrates the E-a pipeline on the held-out fold-5 (anti-leakage):

    1. (optional) Re-dump OOF for ``tsvit-pheno-fullm`` (Fase 1, R-OOF-FULLM): the
       ``ml/eval/oof/`` manifest only has the L4 ``tsvit-pheno-v1``; E-a needs the
       strong Full-M segmenter (US-039).
    2. Build the 18 VISUAL class prototypes (mean CLS-768 per class over TRAIN
       parcels) from the parcel-level FarSLIP student (US-036-b).
    3. Fit the convex coefficient ``alpha`` on geographic OOF sub-folds of fold-5
       (``DualHeadFusionHead.fit``).
    4. Evaluate the fused map on fold-5 vs TSViT-pheno alone (HONEST ablation,
       R-HONEST-GAIN: the phenology branch adds ~0.3 %, not 5 %).
    5. Log one MLflow run ``ensemble-Ea-tsvit-pheno-farslip`` (tags ``data_version``
       + ``code_version``) and persist the fused OOF parquet for US-042 (E-b).

Reuses the US-040 orchestrator helpers (``build_parcel_geometries``,
``_fold5_patch_ids``, ``_geoms_for_blending``) and ``DualHeadFusionHead``. Real
PASTIS-R French data only; checkpoints/OOF use relative paths (land on F: on the
VM). Conventions: Polars, numpy/torch at the boundary, structlog, typer, type
hints, English docstrings, Spanish prose, no emojis.

Usage (on the H100 VM, env ``agrosat``)::

    python -m scripts.run_us041_ea run \
        --farslip-checkpoint checkpoints/farslip/parcel/04cls/best.safetensors \
        --redump-oof          # only the first time, to materialize tsvit-pheno-fullm
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import structlog
import typer

from ml.ensemble.base import EnsembleModel
from ml.ensemble.dual_head_fusion import (
    DEFAULT_FARSLIP_CHECKPOINT,
    DEFAULT_TSVIT_MEMBER,
    DualHeadFusionHead,
)

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help="US-041 E-a closing run.")

_HELD_OUT_FOLD = EnsembleModel.HELD_OUT_FOLD
_EA_RUN_NAME = "ensemble-Ea-tsvit-pheno-farslip"


def _redump_tsvit_fullm_oof(
    *, oof_dir: Path, pastis_root: Path, device: str, max_patches: int | None
) -> None:
    """Re-dump the OOF of ``tsvit-pheno-fullm`` only (Fase 1, R-OOF-FULLM).

    Passes a single-entry registry to :func:`ml.eval.oof.dump_oof.dump_oof` so the
    historical L4 ``oof_tsvit-pheno_fold5.parquet`` is NOT overwritten; the Full-M
    is written under ``oof_tsvit-pheno-fullm_fold5.parquet``.

    Args:
        oof_dir: OOF output directory.
        pastis_root: PASTIS-R root.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        max_patches: optional cap (smoke).
    """
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.oof.dump_oof import dump_oof

    member = DEFAULT_TSVIT_MEMBER
    if member not in CHECKPOINT_REGISTRY:
        raise typer.BadParameter(
            f"{member!r} is not in CHECKPOINT_REGISTRY; cannot re-dump its OOF."
        )
    single = {member: CHECKPOINT_REGISTRY[member]}
    logger.info("ea_redump_oof_start", member=member, fold=_HELD_OUT_FOLD)
    dump_oof(
        single,
        fold=_HELD_OUT_FOLD,
        out_dir=oof_dir,
        data_root=pastis_root,
        device=device,
        max_patches=max_patches,
        skip_missing=False,
        write_parcel=True,
    )
    logger.info("ea_redump_oof_done", member=member)


def _build_train_prototype_dataset(
    *, pastis_root: Path, n_classes: int, train_folds: tuple[int, ...]
):
    """Build the TRAIN-fold ParcelCropDataset for the visual prototypes."""
    from ml.farslip.parcel_crop_dataset import ParcelCropDataset
    from ml.farslip.pastis_pair_dataset import active_classes

    return ParcelCropDataset(
        captions={},
        root=pastis_root,
        folds=train_folds,
        active_class_ids=active_classes(n_classes),
    )


def _baseline_tsvit_metrics(head: DualHeadFusionHead, patch_ids: list[str]) -> dict[str, float]:
    """Score TSViT-pheno alone on fold-5 (the honest baseline for the ablation).

    Reuses the head's loaded OOF index + ground-truth loader: predicts the dense
    argmax of the TSViT member only (alpha=1 equivalent) and scores it with the
    same harness metric as the fused model.

    Args:
        head: a fitted (or loadable) :class:`DualHeadFusionHead`.
        patch_ids: fold-5 patch ids.

    Returns:
        ``{"f1_macro": float, "accuracy": float}`` for TSViT-pheno alone.
    """
    tsvit_index = head._load_tsvit_index()
    yt, yp = [], []
    for pid in patch_ids:
        p_t = head._tsvit_map(pid, tsvit_index)
        yp.append(p_t.argmax(axis=0))
        yt.append(head._ground_truth_patch(pid))
    return head.compute_metrics(
        np.concatenate([a.reshape(-1) for a in yt]),
        np.concatenate([a.reshape(-1) for a in yp]),
        num_classes=head.n_classes,
    )


@app.command()
def run(
    farslip_checkpoint: Annotated[Path, typer.Option("--farslip-checkpoint")] = Path(
        DEFAULT_FARSLIP_CHECKPOINT
    ),
    oof_dir: Annotated[Path, typer.Option("--oof-dir")] = Path("ml/eval/oof"),
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("reports/ensemble"),
    n_classes: Annotated[int, typer.Option("--n-classes")] = 18,
    train_folds: Annotated[str, typer.Option("--train-folds")] = "1,2,3",
    n_spatial_folds: Annotated[int, typer.Option("--n-spatial-folds")] = 5,
    device: Annotated[str, typer.Option("--device")] = "auto",
    redump_oof: Annotated[
        bool, typer.Option("--redump-oof", help="re-dump tsvit-pheno-fullm OOF first")
    ] = False,
    max_patches: Annotated[int, typer.Option("--max-patches")] = 0,
    use_mlflow: Annotated[bool, typer.Option("--use-mlflow/--no-mlflow")] = True,
    random_state: Annotated[int, typer.Option("--random-state")] = 42,
) -> None:
    """Run the E-a dual-head fusion closing pipeline on fold-5."""
    from scripts.run_us040_ensembles import (
        _fold5_patch_ids,
        _geoms_for_blending,
        build_parcel_geometries,
    )

    cap = max_patches if max_patches > 0 else None
    if redump_oof:
        _redump_tsvit_fullm_oof(
            oof_dir=oof_dir, pastis_root=pastis_root, device=device, max_patches=cap
        )

    patch_ids = _fold5_patch_ids(oof_dir)
    if cap is not None:
        patch_ids = patch_ids[:cap]
    logger.info("ea_run_start", n_patches=len(patch_ids), n_classes=n_classes)

    parcel_geoms = build_parcel_geometries(patch_ids, pastis_root)
    geoms_gdf = _geoms_for_blending(parcel_geoms)

    train_tuple = tuple(int(x) for x in train_folds.split(",") if x.strip())
    prototype_ds = _build_train_prototype_dataset(
        pastis_root=pastis_root, n_classes=n_classes, train_folds=train_tuple
    )

    head = DualHeadFusionHead(
        tsvit_member=DEFAULT_TSVIT_MEMBER,
        farslip_checkpoint=farslip_checkpoint,
        n_classes=n_classes,
        n_spatial_folds=n_spatial_folds,
        device=device,
        data_root=pastis_root,
        oof_dir=oof_dir,
        random_state=random_state,
    )
    t0 = time.monotonic()
    head.fit(patch_ids, geoms_gdf, prototype_dataset=prototype_ds)
    fit_s = time.monotonic() - t0

    ea_metrics = head.evaluate_patches(patch_ids)
    tsvit_metrics = _baseline_tsvit_metrics(head, patch_ids)
    gain = ea_metrics["f1_macro"] - tsvit_metrics["f1_macro"]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "alpha": head.alpha,
        "ea": ea_metrics,
        "tsvit_pheno_fullm_alone": tsvit_metrics,
        "f1_macro_gain": gain,
        "n_patches": len(patch_ids),
        "fit_seconds": round(fit_s, 1),
    }
    (out_dir / "us041_ea_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "ea_run_done",
        alpha=round(head.alpha, 4),
        ea_f1=round(ea_metrics["f1_macro"], 4),
        tsvit_f1=round(tsvit_metrics["f1_macro"], 4),
        gain=round(gain, 4),
    )

    # Persist fused OOF (parcel-level) for US-042 E-b.
    _persist_fused_parcel_oof(head, patch_ids, oof_dir=oof_dir)

    if use_mlflow:
        params = head.mlflow_params()
        params.update({"train_folds": train_folds, "n_patches": len(patch_ids)})
        head.log_to_mlflow(
            {
                "f1_macro": ea_metrics["f1_macro"],
                "accuracy": ea_metrics["accuracy"],
                "tsvit_f1_macro": tsvit_metrics["f1_macro"],
                "f1_macro_gain": gain,
            },
            run_name=_EA_RUN_NAME,
            params=params,
            inference_time_s=fit_s,
        )

    typer.echo(json.dumps(summary, ensure_ascii=False))


def _persist_fused_parcel_oof(
    head: DualHeadFusionHead, patch_ids: list[str], *, oof_dir: Path
) -> Path:
    """Reduce the fused dense maps to parcel-level and write the E-a OOF parquet.

    Produces ``oof_parcel_ea-fusion_fold5.parquet`` (canonical_parcel_id + prob_*)
    consumed by US-042 (E-b) as a base member. Uses the shared dense->parcel
    reconciliation (``reduce_pixel_to_parcel``).

    Args:
        head: fitted :class:`DualHeadFusionHead`.
        patch_ids: fold-5 patch ids.
        oof_dir: OOF directory.

    Returns:
        The written parquet path.
    """
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    frames: list[pl.DataFrame] = []
    root = head._resolved_root()
    for pid in patch_ids:
        fused = head.predict_proba([pid])  # (18,128,128)
        parcel_ids_map = load_pastis_parcel_ids(pid, root)
        frames.append(head.reduce_pixel_to_parcel(fused, parcel_ids_map, patch_id=pid))
    table = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    out_path = oof_dir / f"oof_parcel_ea-fusion_fold{_HELD_OUT_FOLD}.parquet"
    table.write_parquet(out_path)
    logger.info("ea_fused_oof_written", path=str(out_path), n_parcels=table.height)
    return out_path


if __name__ == "__main__":
    app()
