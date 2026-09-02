"""Runner: build the PASTIS-homologous Italy 2018 dataset (US-078).

Orchestrates the :mod:`ml.data.eurocrops_pastis_builder` pipeline end-to-end for
a pilot of ``--n-patches`` patches, downloading REAL Sentinel-2 L2A texture from
Sentinel Hub (one ORBIT tile per patch, season ``--season``), rasterising the
dense class mask on the tile grid, and persisting the PASTIS layout under
``--out``. The run is INCREMENTAL: each patch is written as it completes and a
re-run resumes by skipping patches already on disk (the SH disk cache plus the
artefact check guarantee no request is wasted).

The whole generation is logged to MLflow (Docker server :5010, falling back to a
local file store when it is down) with the mandatory ``data_version`` +
``code_version`` tags and the pilot metrics: number of patches, mean coverage,
mean dates, mean NDVI std, residual cloud and the consumed SH request count.

This is CPU/network only (no GPU): it runs on the laptop. Example:

    poetry run python -m scripts.build_italia_pastis --n-patches 20 \
        --season 2018-03-01..2018-10-31 --min-support 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from backend.app.core.config import get_settings
from ml.data.eurocrops_pastis_builder import (
    DEFAULT_OUT_DIR,
    PatchPlan,
    PatchResult,
    download_patch_series,
    load_labeled_polygons,
    load_patch_result_stats,
    patch_artifacts_exist,
    rasterize_patch_mask,
    save_pastis_format,
    select_dense_patches,
    write_class_mapping_doc,
    write_metadata,
)
from ml.ingest.sh_client import sh_client_from_settings
from ml.utils.mlflow_utils import track_experiment

logger = structlog.get_logger(__name__)

#: PASTIS reference NDVI spatial std (texture target ~0.2; verified on PASTIS-R).
_PASTIS_NDVI_STD: float = 0.181


def _parse_season(season: str) -> tuple[str, str]:
    """Parse a ``YYYY-MM-DD..YYYY-MM-DD`` season into ``(date_from, date_to)``.

    Args:
        season: The season string, e.g. ``"2018-03-01..2018-10-31"``.

    Returns:
        The ``(date_from, date_to)`` ISO date pair.

    Raises:
        ValueError: if the season is not a ``from..to`` pair.
    """
    start, _, end = season.partition("..")
    if not start or not end:
        raise ValueError(f"--season must be 'YYYY-MM-DD..YYYY-MM-DD'; got {season!r}")
    return start, end


def _summarise(results: list[tuple[PatchPlan, PatchResult]], n_requests: int) -> dict[str, object]:
    """Aggregate the per-patch results into the pilot GATE report.

    Args:
        results: ``(plan, result)`` pairs for the written patches.
        n_requests: Total Sentinel Hub Process API requests issued this run.

    Returns:
        A JSON-serialisable summary with the GATE figures.
    """
    if not results:
        return {"n_patches": 0, "n_requests": n_requests}
    covs = [r.coverage for _, r in results]
    dates = [r.n_dates for _, r in results]
    ndvi = [r.ndvi_std for _, r in results]
    cloud = [r.residual_cloud for _, r in results]
    # Per-class support aggregated over all patches (pixel counts).
    support: dict[int, int] = {}
    classes_seen: set[int] = set()
    for _, r in results:
        classes_seen.update(r.class_support.keys())
        for cid, n in r.class_support.items():
            support[cid] = support.get(cid, 0) + n
    return {
        "n_patches": len(results),
        "mean_coverage": float(sum(covs) / len(covs)),
        "min_coverage": float(min(covs)),
        "max_coverage": float(max(covs)),
        "mean_dates": float(sum(dates) / len(dates)),
        "min_dates": int(min(dates)),
        "max_dates": int(max(dates)),
        "mean_ndvi_std": float(sum(ndvi) / len(ndvi)),
        "pastis_ndvi_std": _PASTIS_NDVI_STD,
        "mean_residual_cloud": float(sum(cloud) / len(cloud)),
        "n_classes_present": len(classes_seen),
        "classes_present": sorted(classes_seen),
        "class_support_pixels": {str(k): v for k, v in sorted(support.items())},
        "n_requests": n_requests,
    }


def run(
    *,
    n_patches: int,
    season: str,
    min_support: int,
    n_frames: int,
    max_cloud: float,
    out_dir: Path,
    parcels_parquet: Path | None = None,
    mapping_csv: Path | None = None,
    region_prefix: str = "it",
    reverse: bool = False,
) -> dict[str, object]:
    """Build the homologue dataset and return the pilot summary.

    Args:
        n_patches: Number of dense patches to build.
        season: ``from..to`` season string.
        min_support: Minimum parcel count for a class to keep its own id.
        n_frames: Max temporal frames per patch (ORBIT request).
        max_cloud: Max scene cloud cover (scene gate; SCL masks per pixel).
        out_dir: Dataset output root.
        parcels_parquet: EuroCrops parcels parquet to label (e.g.
            ``de4_2023.parquet`` for Lower Saxony 2023); defaults to the Italy 2018
            reference.
        mapping_csv: EuroCrops crosswalk CSV (``eurocrops.csv``); defaults to the
            Italy mapping.
        region_prefix: NUTS prefix selecting the crosswalk region (``"it"``,
            ``"de4"``, ...).

    Returns:
        The pilot GATE summary dict.
    """
    date_from, date_to = _parse_season(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_kwargs: dict[str, object] = {"min_support": min_support, "region_prefix": region_prefix}
    if parcels_parquet is not None:
        label_kwargs["parcels_parquet"] = parcels_parquet
    if mapping_csv is not None:
        label_kwargs["mapping_csv"] = mapping_csv
    gdf, class_table = load_labeled_polygons(**label_kwargs)
    plans = select_dense_patches(gdf, n_patches=n_patches)
    if reverse:
        # Parallel download: a second worker walks the SAME deterministic patch
        # list back-to-front so the two processes converge in the middle. The
        # ``patch_artifacts_exist`` resume guard makes any overlap idempotent (the
        # later writer just re-saves an identical patch), so no lock is needed.
        plans = list(reversed(plans))
    write_class_mapping_doc(out_dir, class_table)

    client = sh_client_from_settings(get_settings())

    results: list[tuple[PatchPlan, PatchResult]] = []
    n_requests = 0
    for plan in plans:
        if patch_artifacts_exist(out_dir, plan.patch_id):
            logger.info("patch_resumed_from_disk", patch_id=plan.patch_id)
            results.append((plan, load_patch_result_stats(out_dir, plan)))
            continue
        try:
            patch_stack = download_patch_series(
                client,
                plan,
                date_from=date_from,
                date_to=date_to,
                n_frames=n_frames,
                max_cloud=max_cloud,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad patch is logged, skipped
            logger.warning("patch_download_error", patch_id=plan.patch_id, error=str(exc))
            continue
        n_requests += 1  # exactly one Process API tile request per attempted patch
        if patch_stack is None:
            continue
        mask = rasterize_patch_mask(gdf, patch_stack)
        result = save_pastis_format(
            out_dir, plan, patch_stack, mask, date_from=date_from, date_to=date_to
        )
        result.requests = 1
        results.append((plan, result))

    write_metadata(out_dir, class_table, results)
    summary = _summarise(results, n_requests)
    (out_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, run the build, log to MLflow, print summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--season", default="2018-03-01..2018-10-31")
    parser.add_argument("--min-support", type=int, default=500)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--parcels-parquet",
        type=Path,
        default=None,
        help="EuroCrops parcels parquet to label (e.g. de4_2023.parquet for Lower "
        "Saxony); defaults to the Italy 2018 reference.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help="EuroCrops crosswalk CSV (eurocrops.csv); defaults to the Italy mapping.",
    )
    parser.add_argument(
        "--region-prefix",
        default="it",
        help="NUTS prefix selecting the crosswalk region (it, de4, nl, ...).",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Walk the patch list back-to-front (a 2nd parallel worker uses this so "
        "the two converge in the middle; resume makes overlap idempotent).",
    )
    args = parser.parse_args(argv)

    with track_experiment(
        "us078_italia_pastis_homologo",
        run_name=f"pilot_{args.n_patches}patches",
        dvc_path=str(args.out),
    ) as run_ctx:
        import mlflow

        mlflow.log_params(
            {
                "n_patches_requested": args.n_patches,
                "season": args.season,
                "min_support": args.min_support,
                "n_frames": args.n_frames,
                "max_cloud": args.max_cloud,
            }
        )
        summary = run(
            n_patches=args.n_patches,
            season=args.season,
            min_support=args.min_support,
            n_frames=args.n_frames,
            max_cloud=args.max_cloud,
            out_dir=args.out,
            parcels_parquet=args.parcels_parquet,
            mapping_csv=args.mapping_csv,
            region_prefix=args.region_prefix,
            reverse=args.reverse,
        )
        scalar = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar)
        mlflow.log_dict(summary, "pilot_summary.json")
        run_id = run_ctx.info.run_id

    logger.info(
        "build_italia_pastis_done",
        run_id=run_id,
        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
    )
    print(json.dumps({"mlflow_run_id": run_id, **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
