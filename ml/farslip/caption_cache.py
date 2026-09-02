"""Caption cache materialization for FarSLIP ``L_glo`` (US-036-a v2, T1).

Materializes the per-patch global captions of the faithful FarSLIP redesign to
``data/farslip/pastis_captions.parquet`` (DVC-tracked) so the training loop reads
the parquet instead of re-calling Gemma. Generation is **idempotent**: with
``resume=True`` patches already present in the parquet are not regenerated and the
Gemma client is not invoked for them.

The parquet schema (see :data:`CAPTIONS_SCHEMA`):
    - ``patch_id`` (Utf8): PASTIS patch identifier.
    - ``caption_glo`` (Utf8): the global caption ``L_glo`` (Gemma 4 multimodal).
    - ``caption_model`` (Utf8): the Ollama model tag.
    - ``prompt_version`` (Utf8): prompt template version (anti cache poisoning).
    - ``tile`` (Utf8): Sentinel-2 MGRS tile.
    - ``composite_date`` (Utf8): ``YYYYMMDD`` of the peak-NDVI composite.
    - ``present_class_ids`` (List[Int64]): active crop classes present in the patch.
    - ``n_regions`` (Int64): number of agronomic parcels (instances) in the patch.
    - ``clases`` (Utf8): human-readable present class names (for quick auditing).
    - ``gen_seconds`` (Float64): Gemma generation wall-time per caption.

Anti data-leakage: :func:`audit_captions` regex-scans ``caption_glo`` for the
forbidden patterns (numeric NDVI/index, AlphaEarth, "la clase es"/"the class
is", "satellite embedding"); zero matches means the cache is clean (AC-4).

Conventions: Polars (no pandas), ``structlog``, type hints, English docstrings.
The Gemma client and PASTIS disk reads are injected so this module is fully
mockeable in tests (no network / no real PASTIS required).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

from ml.farslip.caption_generator import (
    PROMPT_VERSION,
    _spatial_composition,
    generate_caption,
)
from ml.farslip.pastis_pair_dataset import peak_ndvi_composite
from ml.ingest.pastis_loader import (
    PASTIS_CLASS_MAP,
    load_pastis_patch,
    pastis_patch_index,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.farslip.caption_generator import GemmaCaptionClient

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_DEFAULT_CAPTIONS_PATH = _REPO_ROOT / "data" / "farslip" / "pastis_captions.parquet"
_DEFAULT_PROTO_PATH = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"

#: Non-agronomic classes excluded from the present-classes / parcels accounting.
_BACKGROUND_CLASS: int = 0
_VOID_CLASS: int = 19

#: Parquet schema of the captions cache (public contract for T4).
CAPTIONS_SCHEMA: dict[str, pl.DataType] = {
    "patch_id": pl.Utf8(),
    "caption_glo": pl.Utf8(),
    "caption_model": pl.Utf8(),
    "prompt_version": pl.Utf8(),
    "tile": pl.Utf8(),
    "composite_date": pl.Utf8(),
    "present_class_ids": pl.List(pl.Int64()),
    "n_regions": pl.Int64(),
    "clases": pl.Utf8(),
    "gen_seconds": pl.Float64(),
}

#: Anti-leakage regex patterns (AC-4 / Section 1.2). Any match is a leak.
_LEAKAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "ndvi_numeric": re.compile(r"(?i)ndvi\s*[=:]\s*[-+]?\d"),
    "alphaearth": re.compile(r"(?i)alphaearth"),
    "satellite_embedding": re.compile(r"(?i)satellite\s+embedding"),
    "la_clase_es": re.compile(r"(?i)la\s+clase\s+es\b"),
    "the_class_is": re.compile(r"(?i)the\s+class\s+is\b"),
}


def load_typical_phenology(path: Path = _DEFAULT_PROTO_PATH) -> dict[str, str]:
    """Loads the ``{class_name: typical phenology description}`` map (US-033).

    Reads the DVC-tracked prototypes parquet and returns the per-class textual
    phenology descriptions used to enrich the caption prompt. Only LEADS/FILTERS
    the parquet; it never regenerates it.

    Args:
        path: prototypes parquet (default the US-033 DVC-tracked one).

    Returns:
        ``{class_name: description}``. Empty dict if the parquet is absent.
    """
    if not path.exists():
        logger.warning("typical_phenology_missing", path=str(path))
        return {}
    df = pl.read_parquet(path, columns=["class_name", "description"])
    return {str(row["class_name"]): str(row["description"]) for row in df.iter_rows(named=True)}


def _patch_present_classes(semantic: np.ndarray, active_class_ids: tuple[int, ...]) -> list[int]:
    """Active crop class_ids present in the patch (excludes background/void).

    Args:
        semantic: ``(H, W)`` PASTIS semantic mask.
        active_class_ids: active PASTIS class_ids.

    Returns:
        Sorted list of present active class_ids.
    """
    present = set(np.unique(semantic).astype(int).tolist())
    present.discard(_BACKGROUND_CLASS)
    present.discard(_VOID_CLASS)
    return sorted(present & set(active_class_ids))


def _patch_n_parcels(instance: np.ndarray | None, semantic: np.ndarray) -> int:
    """Counts agronomic parcels (instances) on a crop pixel in the patch.

    Uses the instance/ParcelIDs channel when present (number of distinct
    non-zero instance ids that overlap a crop pixel); otherwise falls back to
    the number of present crop classes as a coarse lower bound.

    Args:
        instance: ``(H, W)`` ParcelIDs instance mask, or None.
        semantic: ``(H, W)`` PASTIS semantic mask.

    Returns:
        Number of agronomic parcels (>= 0).
    """
    crop_mask = (semantic != _BACKGROUND_CLASS) & (semantic != _VOID_CLASS)
    if instance is None:
        return int(np.unique(semantic[crop_mask]).size) if crop_mask.any() else 0
    inst_on_crop = instance[crop_mask]
    inst_on_crop = inst_on_crop[inst_on_crop != 0]
    return int(np.unique(inst_on_crop).size)


def _patch_total_area_px(semantic: np.ndarray) -> int:
    """Total cropped area in pixels (excludes background/void).

    Args:
        semantic: ``(H, W)`` PASTIS semantic mask.

    Returns:
        Number of crop pixels.
    """
    crop_mask = (semantic != _BACKGROUND_CLASS) & (semantic != _VOID_CLASS)
    return int(crop_mask.sum())


def _composite_date(dates_s2: Sequence[int], t_star: int) -> str:
    """Returns the ``YYYYMMDD`` string of the composite timestep, or empty.

    Args:
        dates_s2: per-timestep ``YYYYMMDD`` integer dates.
        t_star: composite timestep index.

    Returns:
        ``YYYYMMDD`` string, or ``""`` if unavailable.
    """
    if 0 <= t_star < len(dates_s2):
        return str(int(dates_s2[t_star]))
    return ""


def _peak_ndvi_t_star(s2: np.ndarray) -> int:
    """Re-derives the peak-NDVI timestep index used by ``peak_ndvi_composite``.

    Mirrors the argmax logic of
    :func:`ml.farslip.pastis_pair_dataset.peak_ndvi_composite` so the composite
    date reported in the parquet matches the composite the model sees.

    Args:
        s2: int16 ``(T, 10, H, W)`` PASTIS tensor.

    Returns:
        Index ``t*`` of the peak mean-NDVI timestep.
    """
    s2f = s2.astype(np.float32)
    red = s2f[:, 2]
    nir = s2f[:, 6]
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(np.abs(denom) > 1e-6, (nir - red) / denom, np.nan)
    ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
    import warnings

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_ndvi = np.nanmean(ndvi.reshape(ndvi.shape[0], -1), axis=1)
    mean_ndvi = np.where(np.isfinite(mean_ndvi), mean_ndvi, -np.inf)
    return int(np.argmax(mean_ndvi))


def _empty_captions_frame() -> pl.DataFrame:
    """Returns an empty captions DataFrame with the canonical schema.

    Returns:
        Empty :class:`polars.DataFrame` matching :data:`CAPTIONS_SCHEMA`.
    """
    return pl.DataFrame(schema=CAPTIONS_SCHEMA)


def _patch_ids_for_folds(pastis_root: Path, folds: Sequence[int]) -> list[str]:
    """Lists PASTIS patch_ids whose official fold is in ``folds``.

    Args:
        pastis_root: PASTIS-R root.
        folds: official PASTIS folds (spatial CV).

    Returns:
        Sorted list of patch_id strings in the requested folds.
    """
    index = pastis_patch_index(pastis_root / "metadata.geojson")
    if index.is_empty():
        return []
    wanted = set(int(f) for f in folds)
    selected = index.filter(pl.col("Fold").is_in(list(wanted)))
    return sorted(selected["patch_id"].to_list(), key=lambda p: int(p))


def _flush_captions(
    existing: pl.DataFrame, new_rows: Sequence[dict[str, object]], out_path: Path
) -> pl.DataFrame:
    """Atomically writes ``existing + new_rows`` to ``out_path`` and returns it.

    Writes to a sibling ``.tmp`` first and replaces, so a crash mid-write never
    corrupts the parquet that ``resume`` reads. Returns the merged frame so the
    caller can promote it to the new ``existing`` baseline after a flush.

    Args:
        existing: rows already persisted (the resume baseline).
        new_rows: caption rows accumulated since the last flush.
        out_path: destination captions parquet.

    Returns:
        The merged :class:`polars.DataFrame` now persisted at ``out_path``.
    """
    if new_rows:
        new_frame = pl.DataFrame(list(new_rows), schema=CAPTIONS_SCHEMA)
        merged = pl.concat([existing, new_frame], how="vertical")
    else:
        merged = existing
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    merged.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    return merged


def generate_captions_parquet(
    pastis_root: Path,
    out_path: Path,
    folds: Sequence[int],
    client: GemmaCaptionClient,
    prototype_path: Path | None = None,
    prompt_version: str = PROMPT_VERSION,
    resume: bool = True,
    active_class_ids: tuple[int, ...] = tuple(range(1, 19)),
    side: int = 896,
    flush_every: int = 25,
) -> Path:
    """Materializes the per-patch global captions parquet (idempotent, crash-safe).

    For every patch in ``folds`` it builds the peak-NDVI composite, derives the
    caption inputs (present classes, spatial composition, parcel count/area,
    MGRS tile, composite date, typical phenology), calls Gemma via ``client`` and
    appends the row. The parquet is flushed to disk every ``flush_every`` new
    captions (atomic ``.tmp`` replace), so a crash (SSH/tunnel drop, OOM, kill)
    loses at most ``flush_every`` captions. With ``resume=True`` patches already
    in ``out_path`` are skipped and the client is NOT invoked for them, so a
    re-launch continues from the last flush.

    Args:
        pastis_root: PASTIS-R root.
        out_path: output captions parquet path.
        folds: official PASTIS folds to caption (spatial CV).
        client: :class:`GemmaCaptionClient` (mockeable).
        prototype_path: override of the US-033 prototypes parquet.
        prompt_version: prompt template version stamp.
        resume: if True, do not regenerate captions already present.
        active_class_ids: active PASTIS crop classes (default 1..18).
        side: PNG side in px for the image Gemma sees (default 896).
        flush_every: persist the parquet every N new captions (crash safety).

    Returns:
        ``out_path`` of the written parquet.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and out_path.exists():
        existing = pl.read_parquet(out_path)
        done: set[str] = set(existing["patch_id"].to_list())
    else:
        existing = _empty_captions_frame()
        done = set()

    typical_phenology = load_typical_phenology(prototype_path or _DEFAULT_PROTO_PATH)
    patch_ids = _patch_ids_for_folds(pastis_root, folds)

    new_rows: list[dict[str, object]] = []
    n_new_total = 0
    n_skipped = 0
    for pid in patch_ids:
        if pid in done:
            n_skipped += 1
            continue
        patch = load_pastis_patch(pid, root=pastis_root, load_annotations=True)
        semantic = patch.get("semantic")
        if semantic is None:
            logger.warning("caption_patch_no_semantic", patch_id=pid)
            continue
        semantic = np.asarray(semantic)
        s2 = np.asarray(patch["s2"])

        composite = peak_ndvi_composite(s2)
        t_star = _peak_ndvi_t_star(s2)
        present_ids = _patch_present_classes(semantic, active_class_ids)
        if not present_ids:
            logger.info("caption_patch_no_active_class", patch_id=pid)
            continue
        present_names = [PASTIS_CLASS_MAP.get(cid, f"clase {cid}") for cid in present_ids]
        spatial = _spatial_composition(semantic, tuple(active_class_ids))
        n_parcels = _patch_n_parcels(patch.get("instance"), semantic)
        total_area = _patch_total_area_px(semantic)
        tile = _patch_tile(pastis_root, pid)
        date_str = _composite_date(patch.get("dates_s2") or [], t_star)

        caption, gen_seconds = generate_caption(
            composite,
            present_class_names=present_names,
            spatial_composition=spatial,
            n_parcels=n_parcels,
            total_area_px=total_area,
            tile_mgrs=tile,
            composite_date=date_str,
            typical_phenology=typical_phenology,
            client=client,
            side=side,
        )
        new_rows.append(
            {
                "patch_id": pid,
                "caption_glo": caption,
                "caption_model": client.model,
                "prompt_version": prompt_version,
                "tile": tile,
                "composite_date": date_str,
                "present_class_ids": present_ids,
                "n_regions": n_parcels,
                "clases": ", ".join(present_names),
                "gen_seconds": gen_seconds,
            }
        )
        if flush_every > 0 and len(new_rows) >= flush_every:
            existing = _flush_captions(existing, new_rows, out_path)
            n_new_total += len(new_rows)
            done.update(str(r["patch_id"]) for r in new_rows)
            new_rows = []
            logger.info(
                "captions_parquet_flush",
                path=str(out_path),
                n_persisted=existing.height,
                n_new_total=n_new_total,
            )

    merged = _flush_captions(existing, new_rows, out_path)
    n_new_total += len(new_rows)
    logger.info(
        "captions_parquet_written",
        path=str(out_path),
        n_total=merged.height,
        n_new=n_new_total,
        n_skipped=n_skipped,
        prompt_version=prompt_version,
    )
    return out_path


def _patch_tile(pastis_root: Path, patch_id: str) -> str:
    """Reads the MGRS tile of a patch from ``metadata.geojson``.

    Args:
        pastis_root: PASTIS-R root.
        patch_id: patch identifier.

    Returns:
        MGRS tile string, or ``""`` if unavailable.
    """
    index = pastis_patch_index(pastis_root / "metadata.geojson")
    if index.is_empty():
        return ""
    match = index.filter(pl.col("patch_id") == str(patch_id))
    if match.is_empty():
        return ""
    return str(match["TILE"][0])


def load_captions(path: Path = _DEFAULT_CAPTIONS_PATH) -> dict[str, str]:
    """Loads the captions parquet as a ``{patch_id: caption_glo}`` map.

    Args:
        path: captions parquet path.

    Returns:
        ``{patch_id: caption_glo}``. Empty dict if the parquet is absent.
    """
    if not path.exists():
        logger.warning("captions_parquet_missing", path=str(path))
        return {}
    df = pl.read_parquet(path, columns=["patch_id", "caption_glo"])
    return {str(row["patch_id"]): str(row["caption_glo"]) for row in df.iter_rows(named=True)}


def audit_captions(path: Path = _DEFAULT_CAPTIONS_PATH) -> dict[str, int]:
    """Counts anti-leakage pattern matches across the captions parquet.

    Scans every ``caption_glo`` for the forbidden patterns (numeric NDVI/index,
    AlphaEarth, satellite embedding, "la clase es", "the class is"). A clean
    cache returns all-zero counts (AC-4).

    Args:
        path: captions parquet path.

    Returns:
        ``{pattern_name: n_matches}`` with one entry per forbidden pattern.

    Raises:
        FileNotFoundError: if the parquet does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"captions parquet not found: {path}")
    df = pl.read_parquet(path, columns=["caption_glo"])
    counts: dict[str, int] = dict.fromkeys(_LEAKAGE_PATTERNS, 0)
    for row in df.iter_rows(named=True):
        text = str(row["caption_glo"])
        for name, pattern in _LEAKAGE_PATTERNS.items():
            counts[name] += len(pattern.findall(text))
    total = sum(counts.values())
    logger.info("audit_captions", path=str(path), total_leaks=total, **counts)
    return counts
