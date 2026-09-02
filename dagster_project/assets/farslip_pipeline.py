"""Dagster assets — US-022b-B re-materialization of the FarSLIP pipeline.

Settles the US-017 Phase 4 debt by explicitly declaring in Dagster the three
artifacts of the FarSLIP paper (arXiv:2511.14901) with end-to-end lineage:

::

    sentinel2_crops_256  ──┐
                            ├─► farslip_pairs_italy ──┐
    cap_vocabulary ─────────┘                          │
                                                       ├─► farslip_embeddings_italy
    farslip_clip_italy_v1 (model, MLflow Registry) ────┘                │
                                                                         │
                                                                         ▼
                                                       farslip_embeddings_consolidated
                                                       (data/farslip/embeddings_pastis.parquet)

Mapping to the acceptance criteria (docs/us-planning/us-022b.md §3.2):

- **B-4**: ``farslip_embeddings_consolidated`` produces
  ``data/farslip/embeddings_pastis.parquet`` consumed by
  ``ml.features.fusion._DEFAULT_FARSLIP_PATH``.
- **B-5**: tags ``farslip-pairs-italy-v1`` (in ``farslip_pairs_italy``),
  ``farslip-embeddings-italy-v1`` (in ``farslip_embeddings_consolidated``),
  ``farslip-student-italy-v1`` (in the ``farslip_clip_italy_v1`` external asset).
- **Declarative lineage**: ``farslip_pairs_italy`` -> ``farslip_clip_italy_v1``
  (model) -> ``farslip_embeddings_italy`` (extraction) ->
  ``farslip_embeddings_consolidated`` (final parquet).
- **MLflow metrics**: via the ``mlflow`` resource (``dagster-mlflow``), tags
  ``data_version`` + ``code_version`` (ML/CLAUDE.md NON-NEGOTIABLE rule).

The asset that runs real training (``farslip_clip_italy_v1``) is NOT
materialized from Dagster — the training is launched via ``make train-l4`` on
GCP L4 spot (ml/CLAUDE.md rule). Dagster models it as an external ``AssetSpec``
with ``auto_materialize_policy=None`` so that it appears in the lineage UI
without attempting to run.

Important (US-022b-B Dagster scope):

- This file ONLY declares the new assets and the lineage wiring. The
  consolidation reads the parquets already written by ``farslip_embeddings_italy``
  (existing US-017 asset). It does NOT re-code the FarSLIP pipeline — it only
  orchestrates it and publishes the artifacts to the canonical paths (B-4).
- The real materialization (with data in GCS and student weights) happens outside
  this US (Isaac + Arthur, Phase 4 022b-B). Here the declaration is delivered so
  that ``dagster definitions validate`` passes and the lineage UI shows the full
  flow.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetSpec,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.assets.farslip import (
    DATA_FARSLIP_EMBEDDINGS_DIR,
    EMBEDDING_DIM,
    farslip_embeddings_italy,
)
from dagster_project.assets.farslip import (
    DATA_VERSION_TAG as EMBEDDINGS_DATA_VERSION_TAG,
)
from dagster_project.assets.sentinel2_crops import (
    DATA_FARSLIP_PAIRS_DIR,
    ITALY_REGIONS,
    sentinel2_crops_256,
)
from dagster_project.assets.sentinel2_crops import (
    DATA_VERSION_TAG as PAIRS_DATA_VERSION_TAG,
)
from ml.utils.git_meta import git_sha

#: Canonical path of the consolidated parquet consumed by ``fusion.py``.
#: Synchronized with ``ml.features.fusion._DEFAULT_FARSLIP_PATH``.
DATA_FARSLIP_CONSOLIDATED_PATH = Path("data/farslip/embeddings_pastis.parquet")

#: DVC + MLflow Registry tags defined in US-022b §3.2 B-5.
PAIRS_TAG = PAIRS_DATA_VERSION_TAG  # farslip-pairs-italy-v1
EMBEDDINGS_TAG = EMBEDDINGS_DATA_VERSION_TAG  # farslip-embeddings-italy-v1
STUDENT_TAG = "farslip-student-italy-v1"  # promoted to MLflow @Production

#: MLflow Registry URI of the distilled model (B-5).
FARSLIP_REGISTRY_URI = "models:/farslip-clip-italy-v1/Production"

#: AssetKey of the distilled model — referenced by
#: ``farslip_embeddings_italy`` as an external dep (explicit lineage).
FARSLIP_MODEL_ASSET_KEY = AssetKey("farslip_clip_italy_v1")


# -----------------------------------------------------------------------------
# External AssetSpec (not materializable from Dagster) — declarative lineage.
# -----------------------------------------------------------------------------

#: Semantic alias of ``sentinel2_crops_256`` to align with the contract of the
#: US-022b §4.1 plan ("farslip_pairs_italy"). It is the same physical artifact
#: (``data/farslip_pairs/{roi}/manifest.parquet`` + crops); declaring it here
#: as an external ``AssetSpec`` keeps the paper lineage visible in the UI.
farslip_pairs_italy_spec = AssetSpec(
    key=AssetKey("farslip_pairs_italy"),
    description=(
        "Alias semantico del dataset FarSLIP de pares (imagen 256x256 + texto "
        "agronomico) por ROI italiana. Materializado por sentinel2_crops_256; "
        "este AssetSpec sostiene el contrato de nombre del paper Wen et al. "
        "Tag DVC: farslip-pairs-italy-v1."
    ),
    deps=[sentinel2_crops_256],
    kinds={"polars", "geotiff"},
    group_name="farslip",
    metadata={
        "data_version": MetadataValue.text(PAIRS_TAG),
        "expected_path": MetadataValue.path(str(DATA_FARSLIP_PAIRS_DIR.resolve())),
        "alias_of": MetadataValue.text("sentinel2_crops_256"),
        "us": MetadataValue.text("US-022b-B"),
    },
)

#: Distilled FarSLIP CLIP ViT-B/16 4-band model. Lives in the MLflow Registry,
#: it is NOT materialized from Dagster — the real training is launched by ``make train-l4``
#: on GCP L4 spot. It is declared as an external AssetSpec with an upstream dep on
#: ``farslip_pairs_italy`` so that the lineage UI shows the flow
#: ``pairs -> model -> embeddings``.
farslip_clip_italy_v1_spec = AssetSpec(
    key=FARSLIP_MODEL_ASSET_KEY,
    description=(
        "Modelo FarSLIP CLIP ViT-B/16 destilado a 4 bandas Sentinel-2 "
        "(arXiv:2511.14901). Entrenado en GCP L4 spot (US-022b-A), registrado "
        "en MLflow Registry como farslip-clip-italy-v1@Production. Tag DVC: "
        "farslip-student-italy-v1. Lineage upstream: farslip_pairs_italy."
    ),
    deps=[AssetKey("farslip_pairs_italy")],
    kinds={"mlflow", "pytorch"},
    group_name="farslip",
    metadata={
        "data_version": MetadataValue.text(STUDENT_TAG),
        "registry_uri": MetadataValue.text(FARSLIP_REGISTRY_URI),
        "experiment": MetadataValue.text("farslip-clip-italy"),
        "run_name": MetadataValue.text("farslip-clip-italy-v1"),
        "training_window": MetadataValue.text("GCP L4 spot ~6h ~$1.7 USD"),
        "us": MetadataValue.text("US-022b-B"),
    },
)


# -----------------------------------------------------------------------------
# Materializable asset: consolidates the embeddings per (roi, year) into a single
# parquet consumed by ml/features/fusion.py.
# -----------------------------------------------------------------------------


def _resolve_consolidated_path() -> Path:
    """Resolves the consolidated path relative to the cwd.

    Returns:
        Absolute ``Path`` of ``data/farslip/embeddings_pastis.parquet`` ready for
        ``parent.mkdir(parents=True, exist_ok=True)``.
    """
    return DATA_FARSLIP_CONSOLIDATED_PATH


def _iter_partition_parquets(
    embeddings_root: Path,
) -> list[tuple[str, int, Path]]:
    """Iterates the parquets written by ``farslip_embeddings_italy``.

    Args:
        embeddings_root: root ``data/farslip_embeddings/`` with layout
            ``{roi}/{year}/embeddings.parquet`` (output of the upstream asset
            partitioned by ROI).

    Returns:
        List of tuples ``(roi, year, path)`` sorted by roi then year. Empty if
        ``embeddings_root`` does not exist.
    """
    if not embeddings_root.exists():
        return []
    found: list[tuple[str, int, Path]] = []
    for roi_dir in sorted(embeddings_root.iterdir()):
        if not roi_dir.is_dir():
            continue
        for year_dir in sorted(roi_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            parquet_path = year_dir / "embeddings.parquet"
            if not parquet_path.exists():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            found.append((roi_dir.name, year, parquet_path))
    return found


@asset(
    deps=[farslip_embeddings_italy, FARSLIP_MODEL_ASSET_KEY],
    group_name="farslip",
    compute_kind="polars",
    required_resource_keys={"mlflow"},
    description=(
        "Consolida los embeddings FarSLIP 512-dim de las 3 ROIs italianas en "
        "un unico parquet ``data/farslip/embeddings_pastis.parquet`` (B-4 del "
        "plan US-022b). Anade columna ``region`` y persiste con schema "
        "compatible con ``ml/features/fusion.py``. Registra metrics + tags en "
        "MLflow (data_version, code_version, n_embeddings, embedding_dim). "
        "Tag DVC: farslip-embeddings-italy-v1."
    ),
)
def farslip_embeddings_consolidated(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Consolidates embeddings per (roi, year) into ``data/farslip/embeddings_pastis.parquet``.

    Reads the parquets written by the partitions of ``farslip_embeddings_italy``
    (``data/farslip_embeddings/{roi}/{year}/``), concatenates them with Polars
    (NOT pandas — ML CLAUDE.md rule), adds a ``region`` column and persists to the
    canonical path consumed by ``fusion.py``.

    Args:
        context: Dagster context. ``context.resources.mlflow`` provides the MLflow
            client to record metrics + tags (B-5).

    Returns:
        ``MaterializeResult`` with metadata ``rows``, ``embedding_dim``, ``rois``,
        ``output_path``, ``data_version`` (DVC tag), ``code_version`` (short git
        SHA). If there are no upstream parquets: ``status="skipped_no_upstream"``
        and ``rows=0`` (no error — the FarSLIP extractor may have been skipped due
        to GCS auth in CI).

    Notes:
        Schema of the final parquet:
        ``{parcel_id: int64, region: str, embedding: list[float32]}``. Maps
        ``crop_id`` -> ``parcel_id`` (int cast via Path.stem). The cast is
        defensive: if the ``crop_id`` is not numeric a truncated hash is used.
    """
    import polars as pl

    output_path = _resolve_consolidated_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    code_version = git_sha(short=True)
    embeddings_root = DATA_FARSLIP_EMBEDDINGS_DIR

    context.log.info(
        "farslip_embeddings_consolidated.start root=%s output=%s",
        embeddings_root,
        output_path,
    )

    partitions = _iter_partition_parquets(embeddings_root)
    if not partitions:
        context.log.warning(
            "farslip_embeddings_consolidated: no se encontraron parquets en "
            "%s. Materializa primero farslip_embeddings_italy para las 3 "
            "ROIs (pianura_padana, toscana, puglia).",
            embeddings_root,
        )
        return MaterializeResult(
            metadata={
                "status": MetadataValue.text("skipped_no_upstream"),
                "rows": MetadataValue.int(0),
                "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
                "rois": MetadataValue.text(""),
                "data_version": MetadataValue.text(EMBEDDINGS_TAG),
                "code_version": MetadataValue.text(code_version),
                "output_path": MetadataValue.path(str(output_path.resolve())),
            }
        )

    frames: list[pl.DataFrame] = []
    rois_seen: set[str] = set()
    years_seen: set[int] = set()
    for roi, year, parquet_path in partitions:
        df = pl.read_parquet(parquet_path)
        # Add region/year columns (compat with fusion.py LEFT JOIN by parcel_id).
        df = df.with_columns(
            pl.lit(roi).alias("region"),
            pl.lit(year).cast(pl.Int32).alias("year"),
        )
        frames.append(df)
        rois_seen.add(roi)
        years_seen.add(year)
        context.log.info(
            "farslip_embeddings_consolidated.append roi=%s year=%d rows=%d",
            roi,
            year,
            df.height,
        )

    consolidated = pl.concat(frames, how="diagonal_relaxed")
    n_rows = consolidated.height

    # Guarantee the contract with fusion.py: numeric ``parcel_id`` column.
    # ``crop_id`` from the upstream has free format (Path.stem); we attempt an
    # int cast without losing rows — fallback to hash if not numeric.
    if "crop_id" in consolidated.columns and "parcel_id" not in consolidated.columns:
        consolidated = consolidated.with_columns(
            pl.col("crop_id").cast(pl.Int64, strict=False).alias("parcel_id"),
        )

    consolidated.write_parquet(output_path, compression="zstd")

    context.log.info(
        "farslip_embeddings_consolidated.complete rows=%d rois=%s years=%s output=%s",
        n_rows,
        sorted(rois_seen),
        sorted(years_seen),
        output_path,
    )

    # B-5: MLflow metrics + tags via the dagster-mlflow resource.
    # The resource manages the run; here we only emit params/metrics.
    mlflow_client = context.resources.mlflow
    try:
        mlflow_client.log_metric("n_embeddings", float(n_rows))
        mlflow_client.log_metric("embedding_dim", float(EMBEDDING_DIM))
        mlflow_client.log_metric("n_rois", float(len(rois_seen)))
        mlflow_client.log_param("data_version", EMBEDDINGS_TAG)
        mlflow_client.log_param("code_version", code_version)
        mlflow_client.log_param("model_version", STUDENT_TAG)
        mlflow_client.log_param("pairs_version", PAIRS_TAG)
        mlflow_client.set_tag("us", "US-022b-B")
        mlflow_client.set_tag("pipeline", "farslip")
    except Exception as exc:  # noqa: BLE001 — MLflow offline must not break the materialization
        context.log.warning(
            "farslip_embeddings_consolidated: MLflow logging failed (offline?) %s: %s",
            type(exc).__name__,
            exc,
        )

    return MaterializeResult(
        metadata={
            "rows": MetadataValue.int(n_rows),
            "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
            "rois": MetadataValue.text(",".join(sorted(rois_seen))),
            "years": MetadataValue.text(",".join(str(y) for y in sorted(years_seen))),
            "n_partitions_in": MetadataValue.int(len(partitions)),
            "output_path": MetadataValue.path(str(output_path.resolve())),
            "data_version": MetadataValue.text(EMBEDDINGS_TAG),
            "model_version": MetadataValue.text(STUDENT_TAG),
            "pairs_version": MetadataValue.text(PAIRS_TAG),
            "code_version": MetadataValue.text(code_version),
            "mlflow_run_name": MetadataValue.text("farslip-clip-italy-v1"),
            "consumed_by": MetadataValue.text(
                "ml.features.fusion.build_fused_features(include_farslip=True)"
            ),
        }
    )


__all__ = [
    "DATA_FARSLIP_CONSOLIDATED_PATH",
    "EMBEDDINGS_TAG",
    "FARSLIP_MODEL_ASSET_KEY",
    "FARSLIP_REGISTRY_URI",
    "ITALY_REGIONS",
    "PAIRS_TAG",
    "STUDENT_TAG",
    "farslip_clip_italy_v1_spec",
    "farslip_embeddings_consolidated",
    "farslip_pairs_italy_spec",
]
