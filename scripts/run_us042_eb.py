"""US-042 closing run -- Ensamble E-b: E-a + AlphaEarth via XGBoost stacking.

Extends the E-a fusion (US-041) with the AlphaEarth FM tabular space as a third
stacking member, to test HONESTLY whether the FM embedding adds complementary
signal over the dense + contrastive path. Pipeline on fold-5 (anti-leakage):

    1. Build the 2018+2019 AVERAGED AlphaEarth features parquet (R-AE-AVG) and
       materialize the leak-free ``oof_parcel_xgb-alphaearth_fold5.parquet`` with
       ``materialize_xgb_parcel_oof`` (US-040, reused).
    2. Reduce the E-a dense fusion OOF (US-041 sidecar) to parcel-level via
       ``pixel_to_parcel_probs(method='mean')`` -> ``oof_parcel_ea-fusion_fold5``.
    3. Fit ``StackingEnsemble`` (member-generic, NO class change) over the parcel
       OOF of the members, meta-learner trained on OOF spatial sub-folds of fold-5.
    4. Evaluate on fold-5 and compare HONESTLY E-a vs E-b (18-class; the HCAT-6
       0.6535 figure is a DIFFERENT axis, never mixed -- R-LABEL-SPACE).
    5. Log MLflow ``ensemble-Eb-plus-alphaearth`` (``data_version`` + ``code_version``).

Reuses US-040 helpers (``build_parcel_ground_truth``, ``build_parcel_geometries``,
``_fold5_patch_ids``, ``materialize_xgb_parcel_oof``), ``StackingEnsemble``,
``pixel_to_parcel_probs`` and the new ``build_avg_features_for_xgb``. Real
PASTIS-R + real AlphaEarth only. Conventions: Polars, numpy at the boundary,
structlog, typer, type hints, English docstrings, Spanish prose, no emojis.

Usage (env ``agrosat``)::

    python -m scripts.run_us042_eb run --meta logreg
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import structlog
import typer

from ml.ensemble.base import EnsembleModel
from ml.ensemble.stacking import StackingEnsemble

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help="US-042 E-b closing run.")

_HELD_OUT_FOLD = EnsembleModel.HELD_OUT_FOLD
_EB_RUN_NAME = "ensemble-Eb-plus-alphaearth"

#: Default AlphaEarth year parquets (verified on disk; 100% parcel overlap).
_AE_2018 = "data/cache/gee/alphaearth_parcels_parcels_2018_85951.parquet"
_AE_2019 = "data/cache/gee/alphaearth_parcels_pastis_parcels_2019_85951.parquet"

#: E-b stacking members: E-a fusion (denso reducido a parcela) + XGB-AlphaEarth.
_EB_MEMBERS: tuple[str, ...] = ("ea-fusion", "xgb-alphaearth")


def _reduce_ea_sidecar_to_parcel(*, oof_dir: Path, pastis_root: Path) -> Path:
    """Reduce the E-a dense fusion OOF to parcel-level (``oof_parcel_ea-fusion``).

    US-041 writes the dense fusion OOF ``oof_ea_fusion_fold5.parquet`` (pixel
    softmax per patch). E-b needs it at parcel granularity to join with the
    tabular member; this reduces it with the shared ``pixel_to_parcel_probs``
    (mean of post-softmax within the parcel, PASTIS purity ~98%).

    Args:
        oof_dir: OOF directory.
        pastis_root: PASTIS-R root (for the per-pixel ParcelIDs raster).

    Returns:
        The written ``oof_parcel_ea-fusion_fold5.parquet`` path.

    Raises:
        FileNotFoundError: if the E-a dense OOF sidecar is absent (run US-041
            first: ``python -m scripts.run_us041_ea run``).
    """
    from ml.eval.oof.parquet_io import read_softmax_parquet
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids, pixel_to_parcel_probs

    dense_path = oof_dir / f"oof_ea_fusion_fold{_HELD_OUT_FOLD}.parquet"
    parcel_path = oof_dir / f"oof_parcel_ea-fusion_fold{_HELD_OUT_FOLD}.parquet"
    if parcel_path.is_file():
        logger.info("ea_parcel_sidecar_exists", path=str(parcel_path))
        return parcel_path
    if not dense_path.is_file():
        raise FileNotFoundError(
            f"E-a dense OOF not found: {dense_path}. Run US-041 first "
            "(python -m scripts.run_us041_ea run) to produce it."
        )
    frame = read_softmax_parquet(dense_path)
    rows: list[pl.DataFrame] = []
    for pid, sm in zip(frame["patch_id"].to_list(), frame["softmax"].to_list(), strict=True):
        if sm is None:
            continue
        parcel_ids_map = load_pastis_parcel_ids(str(pid), pastis_root)
        rows.append(
            pixel_to_parcel_probs(np.asarray(sm), parcel_ids_map, patch_id=str(pid), method="mean")
        )
    table = pl.concat(rows, how="vertical") if rows else pl.DataFrame()
    table.write_parquet(parcel_path)
    logger.info("ea_parcel_sidecar_written", path=str(parcel_path), n_parcels=table.height)
    return parcel_path


@app.command()
def run(
    oof_dir: Annotated[Path, typer.Option("--oof-dir")] = Path("ml/eval/oof"),
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    fused_features: Annotated[Path, typer.Option("--fused-features")] = Path(
        "data/features/features_fused_pastis.parquet"
    ),
    out_dir: Annotated[Path, typer.Option("--out-dir")] = Path("reports/ensemble"),
    meta: Annotated[str, typer.Option("--meta", help="logreg | xgb")] = "logreg",
    n_spatial_folds: Annotated[int, typer.Option("--n-spatial-folds")] = 5,
    materialize_xgb: Annotated[bool, typer.Option("--materialize-xgb/--no-materialize-xgb")] = True,
    use_mlflow: Annotated[bool, typer.Option("--use-mlflow/--no-mlflow")] = True,
    random_state: Annotated[int, typer.Option("--random-state")] = 42,
) -> None:
    """Run the E-b stacking closing pipeline (E-a + AlphaEarth) on fold-5."""
    from ml.features.alphaearth_multiyear import build_avg_features_for_xgb
    from scripts.run_us040_ensembles import (
        _fold5_patch_ids,
        build_parcel_geometries,
        build_parcel_ground_truth,
        materialize_xgb_parcel_oof,
    )

    patch_ids = _fold5_patch_ids(oof_dir)
    logger.info("eb_run_start", n_patches=len(patch_ids), meta=meta)

    # Fase 1a: averaged AlphaEarth features + leak-free XGB OOF (R-AE-AVG).
    if materialize_xgb:
        avg_feats = build_avg_features_for_xgb(
            [_AE_2018, _AE_2019],
            fused_features,
            out_path=Path("data/features/features_xgb_alphaearth_avg_2018_2019.parquet"),
        )
        materialize_xgb_parcel_oof(
            avg_feats,
            out_path=oof_dir / f"oof_parcel_xgb-alphaearth_fold{_HELD_OUT_FOLD}.parquet",
            pastis_root=pastis_root,
            random_state=random_state,
        )

    # Fase 1b: reduce the E-a dense fusion OOF to parcel (R-EA-MISSING dependency).
    _reduce_ea_sidecar_to_parcel(oof_dir=oof_dir, pastis_root=pastis_root)

    # Fase 2-3: stacking (member-generic, no class change) on OOF spatial sub-folds.
    parcel_gt = build_parcel_ground_truth(patch_ids, pastis_root)
    parcel_geoms = build_parcel_geometries(patch_ids, pastis_root)

    stack = StackingEnsemble(
        base_members=_EB_MEMBERS,
        meta=meta,  # type: ignore[arg-type]
        n_spatial_folds=n_spatial_folds,
        oof_dir=oof_dir,
        random_state=random_state,
    )
    stack.fit(parcel_geoms, gt_labels=parcel_gt)
    eb_metrics = dict(stack.oof_cv_metrics_)

    # Honest comparison vs E-a alone (the parcel-reduced E-a OOF, single member).
    ea_metrics = _ea_alone_metrics(
        oof_dir=oof_dir,
        parcel_geoms=parcel_geoms,
        parcel_gt=parcel_gt,
        n_spatial_folds=n_spatial_folds,
        random_state=random_state,
    )
    gain = eb_metrics.get("f1_macro", 0.0) - ea_metrics.get("f1_macro", 0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "members": list(_EB_MEMBERS),
        "meta": meta,
        "eb_oof_cv": eb_metrics,
        "ea_alone_oof_cv": ea_metrics,
        "f1_macro_gain_vs_ea": gain,
        "label_space": "semantic18 (18 clases); HCAT-6 0.6535 es otro eje, no comparable",
        "n_patches": len(patch_ids),
    }
    (out_dir / "us042_eb_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "eb_run_done",
        eb_f1=round(eb_metrics.get("f1_macro", 0.0), 4),
        ea_f1=round(ea_metrics.get("f1_macro", 0.0), 4),
        gain=round(gain, 4),
    )

    if use_mlflow:
        stack.log_to_mlflow(
            {
                "f1_macro": eb_metrics.get("f1_macro", 0.0),
                "accuracy": eb_metrics.get("accuracy", 0.0),
                "ea_alone_f1_macro": ea_metrics.get("f1_macro", 0.0),
                "f1_macro_gain_vs_ea": gain,
            },
            run_name=_EB_RUN_NAME,
            params={
                "base_members": ",".join(_EB_MEMBERS),
                "meta": meta,
                "n_spatial_folds": n_spatial_folds,
                "alphaearth_years": "2018,2019",
                "alphaearth_agg": "mean",
                "label_space": "semantic18",
            },
        )

    typer.echo(json.dumps(summary, ensure_ascii=False))


def _ea_alone_metrics(
    *,
    oof_dir: Path,
    parcel_geoms: pl.DataFrame,
    parcel_gt: pl.DataFrame,
    n_spatial_folds: int,
    random_state: int,
) -> dict[str, float]:
    """OOF spatial-CV metrics of E-a ALONE (single-member 'stacking' baseline).

    Scores the parcel-reduced E-a fusion OOF on the SAME spatial sub-folds as
    E-b, so the E-a vs E-b comparison is apples-to-apples. Uses a single-member
    StackingEnsemble: the meta-learner over one member is the honest "E-a alone"
    reference at the parcel grain.

    Args:
        oof_dir: OOF directory.
        parcel_geoms: fold-5 parcel geometry frame.
        parcel_gt: fold-5 parcel ground-truth labels.
        n_spatial_folds: geographic sub-folds.
        random_state: seed.

    Returns:
        ``oof_cv_metrics_`` of the single-member (E-a) stacking.
    """
    ea_only = StackingEnsemble(
        base_members=("ea-fusion",),
        meta="logreg",
        n_spatial_folds=n_spatial_folds,
        oof_dir=oof_dir,
        random_state=random_state,
    )
    ea_only.fit(parcel_geoms, gt_labels=parcel_gt)
    return dict(ea_only.oof_cv_metrics_)


if __name__ == "__main__":
    app()
