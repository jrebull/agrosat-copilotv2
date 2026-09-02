"""Populate ``rag_documents`` from the REAL PASTIS-R corpus (US-046).

Builds the Spatial-RAG *lite* corpus from three real, DVC-tracked artefacts
(nothing is synthesised):

1. ``data/farslip/parcel_phenology_captions.parquet`` -- per-parcel Spanish
   phenology descriptions (``description``) keyed by the composite ``parcel_id``
   (e.g. ``"10000_1"``). This is the document text.
2. ``data/features/features_fused_winning_pastis.parquet`` -- the AlphaEarth 64-dim
   annual embedding (``dim_00..dim_63``) and the spatial ``fold``, joined to the
   captions by ``parcel_id``. This is the document vector.
3. ``data/PASTIS-R/metadata.geojson`` -- the canonical PASTIS patch geometries
   (``ID_PATCH`` -> MultiPolygon, EPSG:2154). The document geometry is the
   *centroid of the parcel's patch polygon*, reprojected to EPSG:4326 by PostGIS
   at insert time.

Geometry note (documented trade-off): PASTIS ships geometry per *patch*, not per
*parcel*. ``data/reference/pastis_tiles_dissolved.geojson`` only carries 4
tile-level dissolved polygons, too coarse for the ``ST_DWithin`` pre-filter, so
this ingest uses the finer per-patch geometry from ``data/PASTIS-R/metadata.geojson``
and stores each document at its patch centroid. The composite ``parcel_id`` keeps
the within-patch identity. The corpus is real PASTIS-R (France); the RAG-lite layer
grounds the agent's *reasoning over this corpus*, not over the Italy demo AOI.

The script is idempotent: it ``TRUNCATE``s ``rag_documents`` before inserting, so
re-runs converge to the same corpus. By default it ingests every captioned parcel.

Fold note (verified against the real data): ``parcel_phenology_captions.parquet``
only covers spatial folds 1-4 (the held-out fold-5 has no captions), so the
captioned corpus is the 16,012 fold-1..4 parcels. Defaulting to "fold-5 only"
would therefore yield an empty corpus; the default here is the full captioned set.
Pass ``--fold N`` to restrict to a single fold (e.g. ``--fold 4``) or ``--limit N``
to cap the row count for a quick demo subset.

Usage::

    poetry run python scripts/ingest_rag_documents.py            # all captioned
    poetry run python scripts/ingest_rag_documents.py --fold 4   # one fold only
    poetry run python scripts/ingest_rag_documents.py --limit 500

Connects through ``ml.agent.db.get_pool`` (reads ``DATABASE_URL`` from settings).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import polars as pl
import structlog

from ml.agent.db import close_pool, get_pool
from ml.agent.rag import SOURCE_PHENOLOGY_CAPTION, ingest_rag_documents

logger = structlog.get_logger(__name__)

#: Repo root resolved from this file (``scripts/ingest_rag_documents.py`` -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Real corpus artefacts (DVC-tracked).
_CAPTIONS_PATH = _REPO_ROOT / "data" / "farslip" / "parcel_phenology_captions.parquet"
_FEATURES_PATH = _REPO_ROOT / "data" / "features" / "features_fused_winning_pastis.parquet"
_PATCH_GEOJSON_PATH = _REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"

#: Column prefix of the AlphaEarth 64-dim embedding in the features parquet.
_ALPHAEARTH_PREFIX: str = "dim_"

#: Number of AlphaEarth embedding dimensions.
_EMBED_DIM: int = 64

#: SRID of the PASTIS patch geometries (Lambert-93 / RGF93).
_PASTIS_SRID: int = 2154

#: Batch size for inserts (keeps a single transaction's payload bounded).
_BATCH_SIZE: int = 1000


def _load_patch_geometries() -> dict[int, str]:
    """Load the PASTIS patch geometries as ``{ID_PATCH: geojson_str}``.

    Each value is the GeoJSON geometry string of the patch polygon in EPSG:2154;
    PostGIS reprojects it to 4326 and reduces it to a centroid at insert time.

    Returns:
        Mapping of integer patch id to its GeoJSON geometry string.

    Raises:
        FileNotFoundError: if the PASTIS metadata geojson is absent.
    """
    if not _PATCH_GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"PASTIS patch geometries not found: {_PATCH_GEOJSON_PATH}. "
            "Run `dvc pull data/PASTIS-R` to fetch them."
        )
    with _PATCH_GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        collection = json.load(handle)

    geom_by_patch: dict[int, str] = {}
    for feature in collection.get("features", []):
        props = feature.get("properties", {})
        patch_id = props.get("ID_PATCH")
        geometry = feature.get("geometry")
        if patch_id is None or geometry is None:
            continue
        geom_by_patch[int(patch_id)] = json.dumps(geometry)
    logger.info("ingest_rag_patch_geoms_loaded", n_patches=len(geom_by_patch))
    return geom_by_patch


def _patch_id_from_parcel_id(parcel_id: str) -> int | None:
    """Derive the integer patch id from a composite ``"<patch>_<local>"`` id.

    Args:
        parcel_id: Composite corpus parcel id, e.g. ``"10000_1"``.

    Returns:
        The integer patch id (``10000``), or ``None`` when it cannot be parsed.
    """
    prefix = parcel_id.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def _build_documents(
    *,
    fold: int | None,
    limit: int | None,
) -> list[dict]:
    """Join captions + AlphaEarth embeddings + patch geometry into corpus docs.

    Parcels without a matching embedding or without a patch geometry are skipped
    with a warning (no fabricated data, per the US-046 contract).

    Args:
        fold: Restrict to a single spatial fold when given; ``None`` keeps every
            captioned fold (the captioned corpus spans folds 1-4 only).
        limit: Optional hard cap on the number of documents.

    Returns:
        A list of document dicts ready for :func:`ml.agent.rag.ingest_rag_documents`.

    Raises:
        FileNotFoundError: if a corpus parquet is missing.
        ValueError: if a required column is absent from the parquets.
    """
    for path in (_CAPTIONS_PATH, _FEATURES_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"corpus artefact not found: {path}. Run `dvc pull` to fetch it."
            )

    captions = pl.read_parquet(
        _CAPTIONS_PATH, columns=["parcel_id", "patch_id", "class_id", "description"]
    )
    if "description" not in captions.columns:
        raise ValueError("captions parquet is missing the `description` column.")

    feature_cols = [f"{_ALPHAEARTH_PREFIX}{i:02d}" for i in range(_EMBED_DIM)]
    features = pl.read_parquet(_FEATURES_PATH)
    missing = [c for c in (*feature_cols, "parcel_id", "fold") if c not in features.columns]
    if missing:
        raise ValueError(f"features parquet is missing columns: {missing}")

    # Keep one embedding row per parcel_id (the corpus is per-parcel; a parcel may
    # appear once per year, so collapse to the first occurrence deterministically).
    features = features.select(["parcel_id", "fold", *feature_cols]).unique(
        subset=["parcel_id"], keep="first"
    )

    # Left join keeps every captioned parcel; embeddings are attached where present.
    joined = captions.join(features, on="parcel_id", how="left")
    if fold is not None:
        joined = joined.filter(pl.col("fold") == fold)

    geom_by_patch = _load_patch_geometries()

    documents: list[dict] = []
    skipped_no_embedding = 0
    skipped_no_geom = 0
    for row in joined.iter_rows(named=True):
        parcel_id = str(row["parcel_id"])
        description = row.get("description")
        if not description:
            continue

        embedding = [row[col] for col in feature_cols]
        if any(value is None for value in embedding):
            skipped_no_embedding += 1
            logger.warning("ingest_rag_skip_no_embedding", parcel_id=parcel_id)
            continue

        patch_id = _patch_id_from_parcel_id(parcel_id)
        geom_geojson = geom_by_patch.get(patch_id) if patch_id is not None else None
        if geom_geojson is None:
            skipped_no_geom += 1
            logger.warning("ingest_rag_skip_no_geom", parcel_id=parcel_id, patch_id=patch_id)
            continue

        documents.append(
            {
                "parcel_id": parcel_id,
                "content": str(description),
                "source": SOURCE_PHENOLOGY_CAPTION,
                "embedding": [float(v) for v in embedding],
                "geom_geojson": geom_geojson,
                "geom_srid": _PASTIS_SRID,
            }
        )
        if limit is not None and len(documents) >= limit:
            break

    logger.info(
        "ingest_rag_documents_built",
        n_documents=len(documents),
        skipped_no_embedding=skipped_no_embedding,
        skipped_no_geom=skipped_no_geom,
        fold=fold,
        limit=limit,
    )
    return documents


async def _ingest(documents: list[dict]) -> int:
    """Truncate ``rag_documents`` and batch-insert the corpus (idempotent).

    Args:
        documents: Document dicts to insert.

    Returns:
        The number of rows inserted.
    """
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Idempotency: every run rebuilds the corpus from scratch. RESTART
            # IDENTITY resets the BIGSERIAL so ids stay stable across re-ingests.
            await conn.execute("TRUNCATE TABLE rag_documents RESTART IDENTITY")
            for start in range(0, len(documents), _BATCH_SIZE):
                batch = documents[start : start + _BATCH_SIZE]
                inserted += await ingest_rag_documents(conn, batch)
    return inserted


async def main(argv: list[str] | None = None) -> int:
    """Async entry point: build the corpus and load it into ``rag_documents``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Exit code: ``0`` on success, ``1`` on a missing-artefact / DB error.
    """
    parser = argparse.ArgumentParser(
        description="Ingest the real PASTIS-R phenology corpus into rag_documents."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Restrict to a single spatial fold (captioned corpus spans folds 1-4); "
        "default ingests every captioned parcel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of documents ingested (default: no cap).",
    )
    args = parser.parse_args(argv)

    try:
        documents = _build_documents(fold=args.fold, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("ingest_rag_build_failed", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not documents:
        print("no documents to ingest (check fold filter / corpus availability)")
        return 0

    try:
        inserted = await _ingest(documents)
    except Exception as exc:  # noqa: BLE001 - report any DB failure to the operator
        logger.error("ingest_rag_insert_failed", error=str(exc))
        print(f"ERROR: failed to ingest rag_documents ({exc})", file=sys.stderr)
        return 1
    finally:
        await close_pool()

    logger.info("ingest_rag_done", inserted=inserted)
    print(f"ingested {inserted} rag_documents (source={SOURCE_PHENOLOGY_CAPTION})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
