"""Dagster assets for US-017 — Bulk extraction of FarSLIP embeddings.

Materializes the downstream asset ``farslip_embeddings_italy`` partitioned
by Italian ROI. For each partition:

1. Reads the manifest ``data/farslip_pairs/{roi}/manifest.parquet`` (output
   of the upstream asset ``sentinel2_crops_256``) with Polars.
2. Instantiates the ``FarSLIPExtractor`` loading student weights from
   ``gs://agrosat-models/farslip/farslip-clip-italy-v1/`` (local cache in
   ``~/.cache/agrosat/farslip/``).
3. Iterates the crops in batches (default 32) and extracts 512-dim embeddings.
4. Persists to ``data/farslip_embeddings/{roi}/{year}/embeddings.parquet``
   with schema ``{crop_id: str, embedding: list[float32], crop_doy: int,
   cap_class: str}``.

Lineage declared via ``deps=[sentinel2_crops_256]`` so that Dagster
automatically materializes the upstream if it is not fresh.

Production (NOT available in CI nor local dev without GCS creds):
    A ``GCSResource`` would be injected from ``dagster_project/resources/``
    to authenticate the download of weights from the MLflow Model Registry and
    to persist embeddings to ``gs://agrosat-features/farslip/`` via
    DVC remote. In US-017 the extractor manages GCS internally
    (local cache) — the formal injection of the resource is left for US-025.

Smoke / local dev:
    If ``FarSLIPExtractor`` fails because GCS is not accessible (creds
    absent, offline) the materialization returns a ``MaterializeResult``
    with ``status="skipped_no_gcs"`` and a warning — it does NOT fail. This allows
    `make check` and CI without GCS secrets to pass the Dagster smoke.

MLflow integration (documented, not implemented in US-017):
    The extractor reads the ``farslip-clip-italy-v1`` run from the Model Registry
    and applies the tags ``data_version=farslip-pairs-italy-v1`` +
    ``code_version=<git_sha>`` to the persisted embeddings. An MLflow run per
    asset could also be materialized for explicit tracking
    of the bulk extraction; in US-017 it is deferred to US-025 when the
    SegFormer head consumes these embeddings.
"""

from pathlib import Path

from dagster import (
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    MetadataValue,
    asset,
)

from dagster_project.assets.sentinel2_crops import (
    DATA_FARSLIP_PAIRS_DIR,
    ITALY_REGIONS,
    sentinel2_crops_256,
)
from ml.utils.gcs_errors import is_gcs_auth_error
from ml.utils.git_meta import git_sha

#: External dep on the distilled model (US-022b-B B-5). Imported by AssetKey
#: instead of symbol to avoid a circular import with farslip_pipeline.py.
_FARSLIP_MODEL_KEY = AssetKey("farslip_clip_italy_v1")

#: Paths relative to the cwd. Persistence partitioned by ROI and year (year
#: derived from the manifest, not hardcoded — Q4 fix).
DATA_FARSLIP_EMBEDDINGS_DIR = Path("data/farslip_embeddings")

#: URI of the FarSLIP student in GCS (MLflow artifact + DVC tag).
#: In CI without GCS creds the extractor falls back to local cache or raises
#: DefaultCredentialsError; in that case the asset reports skipped.
DEFAULT_WEIGHTS_URI = "gs://agrosat-models/farslip/farslip-clip-italy-v1/"

#: Version tag of the embeddings bank (DVC).
DATA_VERSION_TAG = "farslip-embeddings-italy-v1"

#: Dimension of the student embedding (CLIP ViT-B/16 projection head).
EMBEDDING_DIM = 512

#: Batch size for bulk extraction (fits comfortably in L4 24 GB).
EXTRACTION_BATCH_SIZE = 32

#: Fallback year if the manifest does not expose ``crop_year``. Only used when
#: the manifest lacks the column; the real years from the manifest take priority.
FALLBACK_YEAR = 2024


def _skipped_result(context: AssetExecutionContext, reason: str, roi: str) -> MaterializeResult:
    """Builds a uniform skip MaterializeResult for expected failures.

    Args:
        context: Dagster context to emit a warning.
        reason: skip reason (included in metadata + log).
        roi: partition key of the active ROI.

    Returns:
        MaterializeResult with ``status="skipped_no_gcs"`` and useful metadata.
    """
    context.log.warning(
        "farslip_embeddings_italy roi=%s SKIPPED: %s",
        roi,
        reason,
    )
    return MaterializeResult(
        metadata={
            "roi": MetadataValue.text(roi),
            "status": MetadataValue.text("skipped_no_gcs"),
            "reason": MetadataValue.text(reason),
            "n_embeddings": MetadataValue.int(0),
            "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
            "data_version": MetadataValue.text(DATA_VERSION_TAG),
            "code_version": MetadataValue.text(git_sha(short=True)),
        }
    )


@asset(
    deps=[sentinel2_crops_256, _FARSLIP_MODEL_KEY],
    partitions_def=ITALY_REGIONS,
    group_name="farslip",
    compute_kind="python",
    description=(
        "Bulk extraction de embeddings FarSLIP 512-dim sobre los crops "
        "Sentinel-2 256x256 de la ROI activa. Persiste a "
        "data/farslip_embeddings/{roi}/{year}/embeddings.parquet. "
        "Lineage US-022b-B: depende del modelo farslip_clip_italy_v1 "
        "(MLflow Registry @Production)."
    ),
)
def farslip_embeddings_italy(context: AssetExecutionContext) -> MaterializeResult:
    """Materializes 512-dim FarSLIP embeddings per Italian ROI.

    Args:
        context: Dagster context. ``context.partition_key`` indicates the ROI.

    Returns:
        ``MaterializeResult`` with metadata ``n_embeddings``,
        ``embedding_dim=512``, ``output_path``, ``roi``, ``year``,
        ``data_version`` (DVC tag), ``code_version`` (short git SHA).
        If GCS is not accessible: metadata ``status="skipped_no_gcs"``.

    Raises:
        FileNotFoundError: if the upstream manifest does not exist (message
            tells the user to materialize ``sentinel2_crops_256`` first).
    """
    import polars as pl

    roi = context.partition_key
    manifest_path = DATA_FARSLIP_PAIRS_DIR / roi / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"upstream manifest missing: {manifest_path}. Materialize "
            "sentinel2_crops_256 first with partition_key={roi}."
        )

    context.log.info(
        "farslip_embeddings_italy.start roi=%s manifest=%s",
        roi,
        manifest_path,
    )

    # Polars (NON-NEGOTIABLE: no pandas). LazyFrame for large volumes.
    manifest = pl.read_parquet(manifest_path)
    n_crops = manifest.height

    if n_crops == 0:
        context.log.warning(
            "farslip_embeddings_italy roi=%s: manifest vacío, nada que extraer",
            roi,
        )
        return _skipped_result(context, "manifest empty", roi)

    # Load the extractor — fails cleanly if GCS is not accessible.
    try:
        from ml.extractors.farslip_extractor import (  # type: ignore[import-not-found]
            FarSLIPExtractor,
        )

        extractor = FarSLIPExtractor(weights_uri=DEFAULT_WEIGHTS_URI)
    except ImportError as exc:
        return _skipped_result(
            context,
            f"FarSLIPExtractor no instalable todavia ({exc})",
            roi,
        )
    except Exception as exc:
        if is_gcs_auth_error(exc):
            return _skipped_result(context, f"GCS auth failed: {type(exc).__name__}", roi)
        # Real AttributeError, KeyError, ValueError bubble up — they are bugs.
        raise

    # Year derived from the manifest (Q4 fix): if the column exists, we group
    # by crop_year and write one partition per (roi, year). If it does not
    # exist, we fall back to the documented FALLBACK_YEAR.
    has_year_col = "crop_year" in manifest.columns
    if has_year_col:
        years_present = sorted(set(int(y) for y in manifest["crop_year"].to_list()))
    else:
        years_present = [FALLBACK_YEAR]
        context.log.warning("manifest sin crop_year; usando FALLBACK_YEAR=%d", FALLBACK_YEAR)

    # Bulk extraction in batches. Q9 fix: we accumulate embedding tensors
    # in a single columnar structure (np.ndarray + list[str/int]) and at the end
    # serialize with pl.DataFrame({...}) in bulk — without an intermediate
    # list[dict] nor per-row Python overhead. For 30k pairs x 512 floats this drops
    # from ~60 MB of Python dicts to ~60 MB of native arrays (allocated only once).
    import numpy as np

    output_paths_by_year: dict[int, Path] = {}

    crop_paths = manifest["crop_path"].to_list()
    crop_doys = manifest["crop_doy"].to_list() if "crop_doy" in manifest.columns else [0] * n_crops
    cap_classes = (
        manifest["cap_class"].to_list() if "cap_class" in manifest.columns else [""] * n_crops
    )
    crop_years = (
        [int(y) for y in manifest["crop_year"].to_list()]
        if has_year_col
        else [FALLBACK_YEAR] * n_crops
    )

    # Buffers per year: manifest indices whose crop_year matches.
    indices_by_year: dict[int, list[int]] = {y: [] for y in years_present}
    embeddings_buffer: list[np.ndarray] = []
    valid_indices: list[int] = []  # manifest indices that produced an embedding

    for start in range(0, n_crops, EXTRACTION_BATCH_SIZE):
        end = min(start + EXTRACTION_BATCH_SIZE, n_crops)
        batch_paths = crop_paths[start:end]

        try:
            batch_tensor = extractor.load_crops_batch(batch_paths)
            embeddings = extractor.extract_embeddings(batch_tensor)
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            context.log.error(
                "farslip_embeddings_italy roi=%s batch=%d-%d failed: %s",
                roi,
                start,
                end,
                exc,
            )
            continue

        # Single tensor -> single numpy array (1 CPU copy, not list[float]).
        batch_np = embeddings.detach().cpu().numpy().astype(np.float32)
        embeddings_buffer.append(batch_np)
        for offset in range(end - start):
            global_idx = start + offset
            valid_indices.append(global_idx)
            indices_by_year.setdefault(crop_years[global_idx], []).append(len(valid_indices) - 1)

    if not embeddings_buffer:
        return _skipped_result(context, "todos los batches fallaron", roi)

    all_embeddings = np.concatenate(embeddings_buffer, axis=0)  # (N, 512)
    n_embeddings = all_embeddings.shape[0]

    for year, local_idxs in indices_by_year.items():
        if not local_idxs:
            continue
        output_dir = DATA_FARSLIP_EMBEDDINGS_DIR / roi / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "embeddings.parquet"

        # Global manifest indices for this year.
        manifest_idxs = [valid_indices[li] for li in local_idxs]
        year_df = pl.DataFrame(
            {
                "crop_id": [Path(str(crop_paths[mi])).stem for mi in manifest_idxs],
                "embedding": [all_embeddings[li].tolist() for li in local_idxs],
                "crop_doy": [int(crop_doys[mi]) for mi in manifest_idxs],
                "cap_class": [str(cap_classes[mi]) for mi in manifest_idxs],
            }
        )
        year_df.write_parquet(output_path, compression="zstd")
        output_paths_by_year[year] = output_path

    code_version = git_sha(short=True)
    context.log.info(
        "farslip_embeddings_italy.complete roi=%s n_embeddings=%d years=%s",
        roi,
        n_embeddings,
        sorted(output_paths_by_year.keys()),
    )

    from dagster import MetadataValue as _MV

    metadata: dict[str, _MV] = {
        "roi": MetadataValue.text(roi),
        "years": MetadataValue.text(
            ",".join(str(y) for y in sorted(output_paths_by_year.keys())) or "none"
        ),
        "n_embeddings": MetadataValue.int(n_embeddings),
        "embedding_dim": MetadataValue.int(EMBEDDING_DIM),
        "batch_size": MetadataValue.int(EXTRACTION_BATCH_SIZE),
        "data_version": MetadataValue.text(DATA_VERSION_TAG),
        # B-5 US-022b: tag of the distilled model (MLflow Registry @Production).
        "model_version": MetadataValue.text("farslip-student-italy-v1"),
        "code_version": MetadataValue.text(code_version),
    }
    if output_paths_by_year:
        # "Main" path = first year (keeps compat with tests).
        first_year = sorted(output_paths_by_year.keys())[0]
        metadata["output_path"] = MetadataValue.path(
            str(output_paths_by_year[first_year].resolve())
        )
    return MaterializeResult(metadata=metadata)
