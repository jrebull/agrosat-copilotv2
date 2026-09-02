"""Per-class phenological prototypes for the semantic branch of TSViT.

Implements the input of the method by Wen et al. (2025), "Phenology
Description is All You Need!" (ISPRS J. Photogrammetry RS 228): instead of
a per-parcel description, a **mean NDVI curve per crop class** is built
and an LLM (Gemini 3.5 Flash) generates the textual phenological
description of each class. These 18 descriptions cover 100% of
the classes (unlike the per-parcel 60x18 subset that covers ~0.72% of
the dense pixels), so they are the correct input to contrastively align
the visual features of dense segmentation with the semantic prototype
of each pixel's class (paper Fig. 1, Table 2).

Flow:
    1. ``compute_class_mean_ndvi_curves``: scans the PASTIS-R patches
       ``DATA_S2/S2_*.npy``, computes NDVI per pixel, groups it by the
       pixel's semantic class (``ANNOTATIONS/TARGET_*.npy`` channel 0) and
       averages over a regular temporal grid indexed by DOY (the acquisition
       dates are irregular per patch, from ``metadata.geojson``).
    2. ``generate_class_prototypes``: for each of the 18 classes, calls
       :func:`ml.features.phenology_description.generate_phenology_description`
       with the mean curve and the class name as ``crop_type_hint``, then
       encodes the text into an embedding with ``all-MiniLM-L6-v2`` (384-dim,
       the same encoder as the existing per-parcel pheno_text).

The output is ``data/features/phenology_class_prototypes_pastis.parquet`` with
18 rows ``class_id, class_name, ndvi_curve, description, emb_000..emb_383``.

Regeneration (US-033 AC-8):
    The parquet is already materialized and DVC-tracked; it is NOT regenerated
    in normal operation. The SHA256 description cache
    (``data/cache/phenology_descriptions/{key}.json``, keyed by
    ``(parcel_id, curve, model, "prompt_v1")``) means cached curves do not
    re-call Gemini, so a re-run is deterministic and free for unchanged
    inputs. To refresh it (only with a valid key or an injected client; cost
    ~$0.0018 for 18 descriptions at temperature=0)::

        # Requires GEMINI_API_KEY / GOOGLE_GENAI_USE_VERTEXAI + project, or
        # ml.features.phenology_description.set_llm_client(...) for an offline
        # client. After regenerating: dvc add <parquet> + commit the .dvc.
        poetry run python -m ml.features.phenology_class_prototypes \
            --pastis-root data/PASTIS-R \
            --output data/features/phenology_class_prototypes_pastis.parquet

    Changing ``PROMPT_TEMPLATE`` requires bumping the ``b"prompt_v1"`` literal
    in :func:`ml.features.phenology_description._hash_curve` to avoid silent
    cache poisoning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_CLASS_MAP_PATH = _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"

#: Default US-078 Mediterranean homologue dataset root (Italy 2018).
_DEFAULT_ITALIA_ROOT = _REPO_ROOT / "data" / "pastis_italia_2018"
#: Default output for the Italian per-class prototypes (39 Mediterranean crops).
_DEFAULT_ITALIA_OUTPUT = (
    _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_italia.parquet"
)

#: Mediterranean agronomic context appended to each Italian class hint so Gemini
#: grounds the description in the Italy 2018 calendar (durum harvest in June,
#: perennial olive/vineyard, displaced season) instead of the Bretagne/PASTIS one.
#: It is a static, factual qualifier (NOT a fabricated number): it only nudges the
#: crop-type hint of block 2 of the prompt; the NDVI curve itself is the REAL
#: Italian one computed from the patches.
_ITALIA_CONTEXT_HINT = "cultivo mediterraneo en Italia 2018"

#: Band indices in the PASTIS-R .npy files (standard 10-band S2 order:
#: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12). NDVI uses B4 (red) and B8 (NIR).
_BAND_B4 = 2
_BAND_B8 = 6

#: Number of regular temporal bins (DOY 1..365) over which the curve is
#: averaged. 37 matches the 10-day grid of the Wen paper.
_N_TIME_BINS = 37

#: Useful classes: 1..18 (excludes 0 Background and 19 Void). The prototype is
#: generated only for the 18 benchmark crops.
_CROP_CLASS_IDS: tuple[int, ...] = tuple(range(1, 19))

#: Text encoder -> 384-dim embedding. Same model as the existing per-parcel
#: pheno_text, to keep coherence of the semantic space.
_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_DIM = 384


def load_class_names(path: Path = _CLASS_MAP_PATH) -> dict[int, str]:
    """Loads the ``class_id -> name`` map of the 18 PASTIS classes.

    Args:
        path: Path to ``pastis_class_mapping.json``.

    Returns:
        Dictionary ``{1: "Meadow", 2: "Soft winter wheat", ...}``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data["classes"]
    out: dict[int, str] = {}
    for k, v in classes.items():
        name = v["name"] if isinstance(v, dict) else v
        out[int(k)] = name
    return out


def _patch_dates_doy(metadata_path: Path) -> dict[int, np.ndarray]:
    """Returns ``{patch_id: DOY array (T,)}`` from ``metadata.geojson``.

    The dates come as ``YYYYMMDD`` integers in the ``dates-S2`` field
    (dict indexed by timestep). They are converted to day-of-year (1..366).

    Args:
        metadata_path: Path to ``metadata.geojson``.

    Returns:
        Map from patch_id to a DOY vector aligned with the temporal axis of
        the corresponding ``.npy``.

    Note:
        It is parsed as flat JSON (``json.load``), NOT with
        ``geopandas.read_file``: only the dates are needed
        (``properties.dates-S2``), not the Polygon geometries. Loading the
        2433 geometries with geopandas is ~100x slower and may hang the
        process; the raw JSON is read in ~0.1s.
    """
    geojson = json.loads(metadata_path.read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        pid = int(props["ID_PATCH"])
        dates_raw = props["dates-S2"]
        if isinstance(dates_raw, str):
            dates_raw = json.loads(dates_raw)
        # Order by timestep index (keys "0".."T-1").
        ymd = [int(dates_raw[str(i)]) for i in range(len(dates_raw))]
        doy = np.array([_ymd_to_doy(v) for v in ymd], dtype=np.int32)
        out[pid] = doy
    return out


def _ymd_to_doy(ymd: int) -> int:
    """Converts a ``YYYYMMDD`` integer to day-of-year (1..366)."""
    from datetime import date

    year = ymd // 10000
    month = (ymd % 10000) // 100
    day = ymd % 100
    return date(year, month, day).timetuple().tm_yday


def compute_class_mean_ndvi_curves(
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    *,
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
) -> dict[int, np.ndarray]:
    """Computes the mean NDVI curve per class over a regular DOY grid.

    For each patch it loads ``S2_<pid>.npy`` ``(T,10,H,W)`` and the
    semantic mask ``TARGET_<pid>.npy`` channel 0 ``(H,W)``. It computes NDVI
    per pixel-time, accumulates the sum and the count per class in each DOY
    bin, and at the end divides to obtain the mean. The int16 reflectances are
    scaled to [0,1] by dividing by 10000 (S2 L2A scale).

    Args:
        pastis_root: Root of the PASTIS-R dataset.
        n_time_bins: Number of regular DOY bins (1..365).
        max_patches: If given, limits the scan (for smoke/tests).

    Returns:
        ``{class_id: curve (n_time_bins,)}`` with NaN in bins without observation.
    """
    s2_dir = pastis_root / "DATA_S2"
    ann_dir = pastis_root / "ANNOTATIONS"
    dates_by_patch = _patch_dates_doy(pastis_root / "metadata.geojson")

    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    # Per-class accumulators: sum and count in each temporal bin.
    sums = {c: np.zeros(n_time_bins, dtype=np.float64) for c in _CROP_CLASS_IDS}
    counts = {c: np.zeros(n_time_bins, dtype=np.int64) for c in _CROP_CLASS_IDS}

    s2_paths = sorted(s2_dir.glob("S2_*.npy"))
    if max_patches is not None:
        s2_paths = s2_paths[:max_patches]

    for s2_path in s2_paths:
        pid = int(s2_path.stem.split("_")[1])
        doy = dates_by_patch.get(pid)
        if doy is None:
            continue
        s2 = np.load(s2_path).astype(np.float32) / 10000.0  # (T,10,H,W)
        target = np.load(ann_dir / f"TARGET_{pid}.npy")[0]  # (H,W) semantic
        b4 = s2[:, _BAND_B4]  # (T,H,W)
        b8 = s2[:, _BAND_B8]
        denom = b8 + b4
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom > 1e-6, (b8 - b4) / denom, np.nan)  # (T,H,W)
        # NDVI valid in [-1, 1]; out-of-range values are artifacts of
        # clouds/shadows or a near-zero denominator (not masked in PASTIS).
        ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
        bin_idx = np.clip(np.digitize(doy, bin_edges) - 1, 0, n_time_bins - 1)
        for c in _CROP_CLASS_IDS:
            class_mask = target == c  # (H,W)
            if not class_mask.any():
                continue
            # Mean NDVI of the class at each timestep -> (T,)
            ndvi_class = ndvi[:, class_mask]  # (T, n_pix_class)
            per_t = np.nanmean(ndvi_class, axis=1)  # (T,)
            valid = np.isfinite(per_t)
            np.add.at(sums[c], bin_idx[valid], per_t[valid])
            np.add.at(counts[c], bin_idx[valid], 1)

    curves: dict[int, np.ndarray] = {}
    for c in _CROP_CLASS_IDS:
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(counts[c] > 0, sums[c] / counts[c], np.nan)
        curves[c] = curve
    logger.info(
        "class_mean_ndvi_curves_computed",
        n_patches=len(s2_paths),
        n_classes=len(curves),
        n_time_bins=n_time_bins,
    )
    return curves


def load_italia_class_names(
    italia_root: Path = _DEFAULT_ITALIA_ROOT,
) -> dict[int, str]:
    """Loads the ``class_id -> hcat4_name`` map of the Italian dense classes.

    Reads the US-078 builder's ``class_mapping.json`` (or ``class_table.parquet``
    as a fallback), which materialises the contiguous Italian ids ``[1, K]``
    (id 0 = background). These are the SAME ids burnt into the dense
    ``TARGET_<id>.npy`` masks, so the prototype matrix row-aligns 1:1 with the
    pixel labels of the contrastive loss.

    Args:
        italia_root: The US-078 homologue dataset root.

    Returns:
        Dictionary ``{1: "olive", 2: "vineyards_wine_vine_rebland_grapes", ...}``
        (background id 0 excluded).

    Raises:
        FileNotFoundError: if neither ``class_mapping.json`` nor
            ``class_table.parquet`` is present (run the US-078 builder first).
    """
    mapping_path = italia_root / "class_mapping.json"
    if mapping_path.is_file():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        return {int(c["class_id"]): str(c["hcat4_name"]) for c in data["classes"]}
    table_path = italia_root / "class_table.parquet"
    if table_path.is_file():
        table = pl.read_parquet(table_path).sort("class_id")
        return {
            int(cid): str(name)
            for cid, name in zip(
                table["class_id"].to_list(),
                table["hcat4_name"].to_list(),
                strict=True,
            )
        }
    raise FileNotFoundError(
        f"no class_mapping.json / class_table.parquet under {italia_root}; run the "
        "US-078 builder (scripts/build_italia_pastis.py) first."
    )


def compute_italia_class_mean_ndvi_curves(
    italia_root: Path = _DEFAULT_ITALIA_ROOT,
    *,
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
) -> dict[int, np.ndarray]:
    """Computes the mean NDVI curve per Italian class over a regular DOY grid.

    Mediterranean analogue of :func:`compute_class_mean_ndvi_curves`. The US-078
    homologue stores, per patch, ``DATA_S2/S2_<id>.npy (T, 10, 128, 128)`` (int16
    DN, scaled ``/10000``), the dense semantic mask ``ANNOTATIONS/TARGET_<id>.npy
    (128, 128)`` and the per-frame day-of-year vector ``ANNOTATIONS/dates_<id>.npy
    (T,)`` -- unlike PASTIS the DOY lives next to the patch, so no
    ``metadata.geojson`` is parsed. NDVI is accumulated per pixel-time and grouped
    by the pixel's class into a regular DOY grid, then averaged. The curve is the
    REAL Italian phenology (durum peaking earlier, perennial olive/vine, displaced
    season), NOT the French PASTIS one.

    Args:
        italia_root: The US-078 homologue dataset root.
        n_time_bins: Number of regular DOY bins (1..365), matched to the PASTIS
            grid so the two prototype banks live in a comparable temporal frame.
        max_patches: If given, limits the scan (for smoke/tests on the pilot).

    Returns:
        ``{class_id: curve (n_time_bins,)}`` for every Italian crop id present on
        disk, with NaN in bins without observation. Classes never seen in the
        scanned patches are absent (reported by the caller).

    Raises:
        FileNotFoundError: if the dataset root or its ``DATA_S2`` are absent.
    """
    s2_dir = italia_root / "DATA_S2"
    ann_dir = italia_root / "ANNOTATIONS"
    if not s2_dir.is_dir():
        raise FileNotFoundError(
            f"homologue dataset incomplete under {italia_root} (need DATA_S2/); run "
            "scripts/build_italia_pastis.py first."
        )

    class_names = load_italia_class_names(italia_root)
    crop_ids = tuple(sorted(class_names))  # background id 0 not in the map

    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    sums = {c: np.zeros(n_time_bins, dtype=np.float64) for c in crop_ids}
    counts = {c: np.zeros(n_time_bins, dtype=np.int64) for c in crop_ids}

    s2_paths = sorted(s2_dir.glob("S2_*.npy"), key=lambda p: int(p.stem.split("_", 1)[1]))
    if max_patches is not None:
        s2_paths = s2_paths[:max_patches]

    seen_classes: set[int] = set()
    for s2_path in s2_paths:
        pid = int(s2_path.stem.split("_", 1)[1])
        target_path = ann_dir / f"TARGET_{pid}.npy"
        date_path = ann_dir / f"dates_{pid}.npy"
        if not target_path.is_file():
            continue
        s2 = np.load(s2_path).astype(np.float32) / 10000.0  # (T,10,H,W)
        target = np.load(target_path)  # (H,W) semantic class
        doy = (
            np.load(date_path).astype(np.int64)
            if date_path.is_file()
            else np.linspace(1, 365, s2.shape[0]).astype(np.int64)
        )
        b4 = s2[:, _BAND_B4]  # (T,H,W)
        b8 = s2[:, _BAND_B8]
        denom = b8 + b4
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom > 1e-6, (b8 - b4) / denom, np.nan)
        # NDVI valid in [-1, 1]; out-of-range values are cloud/shadow artifacts
        # (the homologue SCL-masks per pixel but residual haze can survive).
        ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
        bin_idx = np.clip(np.digitize(doy, bin_edges) - 1, 0, n_time_bins - 1)
        for c in crop_ids:
            class_mask = target == c  # (H,W)
            if not class_mask.any():
                continue
            seen_classes.add(c)
            ndvi_class = ndvi[:, class_mask]  # (T, n_pix_class)
            with warnings.catch_warnings():
                # A timestep fully clouded for this class is an all-NaN slice;
                # nanmean warns but the NaN is dropped by the finite mask below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                per_t = np.nanmean(ndvi_class, axis=1)  # (T,)
            valid = np.isfinite(per_t)
            np.add.at(sums[c], bin_idx[valid], per_t[valid])
            np.add.at(counts[c], bin_idx[valid], 1)

    curves: dict[int, np.ndarray] = {}
    for c in crop_ids:
        if c not in seen_classes:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(counts[c] > 0, sums[c] / counts[c], np.nan)
        curves[c] = curve
    logger.info(
        "italia_class_mean_ndvi_curves_computed",
        n_patches=len(s2_paths),
        n_classes_present=len(curves),
        n_classes_total=len(crop_ids),
        n_time_bins=n_time_bins,
    )
    return curves


def _encode_descriptions(descriptions: Sequence[str]) -> np.ndarray:
    """Encodes a list of descriptions into L2-norm 384-dim embeddings.

    Uses ``all-MiniLM-L6-v2`` (same encoder as the per-parcel pheno_text).

    Args:
        descriptions: List of texts.

    Returns:
        ``(len, 384)`` float32 matrix, L2-normalized per row.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_SENTENCE_MODEL)
    emb = model.encode(
        list(descriptions),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def generate_class_prototypes(
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    *,
    output_path: Path = _DEFAULT_OUTPUT,
    model: str = "gemini-3.5-flash",
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
) -> Path:
    """Generates the 18 per-class phenological prototypes and persists them.

    Full pipeline: mean NDVI curve per class -> per-class Gemini description
    (3-block prompt Wen et al. Fig. 2, with the class name as
    ``crop_type_hint``) -> 384-dim embedding. 100% coverage of the 18
    classes.

    Args:
        pastis_root: PASTIS-R root.
        output_path: Output parquet (18 rows).
        model: LLM model for the descriptions.
        n_time_bins: DOY bins of the mean curve.
        max_patches: Limits the NDVI scan (smoke/tests).

    Returns:
        ``Path`` of the written parquet with columns ``class_id, class_name,
        ndvi_curve (list), description, emb_000..emb_383``.
    """
    from ml.features.phenology_description import (
        generate_phenology_description,
    )

    class_names = load_class_names()
    curves = compute_class_mean_ndvi_curves(
        pastis_root, n_time_bins=n_time_bins, max_patches=max_patches
    )
    # Representative DOY of each bin (center), to pass to the generator.
    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    bin_doy = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.int32)

    rows: list[dict[str, object]] = []
    descriptions: list[str] = []
    for c in _CROP_CLASS_IDS:
        curve = curves[c]
        name = class_names.get(c, f"class_{c}")
        desc = generate_phenology_description(
            ndvi_curve=curve,
            doy=bin_doy,
            parcel_id=f"class_{c}",
            crop_type_hint=name,
            model=model,
        )
        descriptions.append(desc)
        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "ndvi_curve": curve.tolist(),
                "description": desc,
            }
        )
        logger.info("class_prototype_generated", class_id=c, class_name=name)

    embeddings = _encode_descriptions(descriptions)
    for i, row in enumerate(rows):
        for j in range(_EMB_DIM):
            row[f"emb_{j:03d}"] = float(embeddings[i, j])

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(
        "class_prototypes_persisted",
        path=str(output_path),
        n_classes=df.height,
        emb_dim=_EMB_DIM,
    )
    return output_path


def generate_italia_class_prototypes(
    italia_root: Path = _DEFAULT_ITALIA_ROOT,
    *,
    output_path: Path = _DEFAULT_ITALIA_OUTPUT,
    model: str = "gemini-3.5-flash",
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
    context_hint: str = _ITALIA_CONTEXT_HINT,
) -> Path:
    """Generates the Italian per-class phenological prototypes and persists them.

    Mediterranean homologue of :func:`generate_class_prototypes`: REAL mean NDVI
    curve per Italian class (from the US-078 patches) -> per-class Gemini
    description (3-block prompt Wen et al. Fig. 2, with the Italian HCAT4 class
    name + Mediterranean context as ``crop_type_hint``) -> 384-dim ``all-MiniLM-
    L6-v2`` embedding. This fixes the US-079 root cause B: the TSViT-pheno semantic
    branch must align Italian pixels with ITALIAN prototypes (olive perennial,
    durum harvested in June, displaced vine), not the Bretagne/PASTIS calendar.

    Only the classes actually present in the scanned patches get a row (their curve
    is real); absent classes are reported and skipped (NEVER fabricated). The
    output is row-indexed by the dense ``class_id`` so it maps 1:1 onto the
    ``TARGET_<id>.npy`` labels of the contrastive loss.

    Args:
        italia_root: The US-078 homologue dataset root.
        output_path: Output parquet (one row per present Italian class).
        model: LLM model for the descriptions (Gemini 3.5 Flash; needs creds).
        n_time_bins: DOY bins of the mean curve (matched to the PASTIS grid).
        max_patches: Limits the NDVI scan (smoke/tests on the 20-patch pilot).
        context_hint: Mediterranean qualifier appended to each class hint so the
            description is grounded in the Italy 2018 calendar.

    Returns:
        ``Path`` of the written parquet with columns ``class_id, class_name,
        ndvi_curve (list), description, emb_000..emb_383``.

    Raises:
        RuntimeError: if no Italian class curve could be computed (empty dataset).
    """
    from ml.features.phenology_description import (
        generate_phenology_description,
    )

    class_names = load_italia_class_names(italia_root)
    curves = compute_italia_class_mean_ndvi_curves(
        italia_root, n_time_bins=n_time_bins, max_patches=max_patches
    )
    if not curves:
        raise RuntimeError(
            f"no Italian class NDVI curve computed under {italia_root}; the patches "
            "carry no labelled crop pixel. Cannot fabricate prototypes."
        )
    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    bin_doy = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.int32)

    rows: list[dict[str, object]] = []
    descriptions: list[str] = []
    for c in sorted(curves):
        curve = curves[c]
        name = class_names.get(c, f"class_{c}")
        # The crop hint marries the Italian class name with the Mediterranean
        # context so Gemini does not default to a temperate-European calendar.
        crop_hint = f"{name} ({context_hint})"
        desc = generate_phenology_description(
            ndvi_curve=curve,
            doy=bin_doy,
            parcel_id=f"italia_class_{c}",
            crop_type_hint=crop_hint,
            model=model,
        )
        descriptions.append(desc)
        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "ndvi_curve": curve.tolist(),
                "description": desc,
            }
        )
        logger.info("italia_class_prototype_generated", class_id=c, class_name=name)

    embeddings = _encode_descriptions(descriptions)
    for i, row in enumerate(rows):
        for j in range(_EMB_DIM):
            row[f"emb_{j:03d}"] = float(embeddings[i, j])

    df = pl.DataFrame(rows).sort("class_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(
        "italia_class_prototypes_persisted",
        path=str(output_path),
        n_classes=df.height,
        emb_dim=_EMB_DIM,
    )
    return output_path


def load_class_prototype_embeddings(
    path: Path = _DEFAULT_OUTPUT,
) -> tuple[np.ndarray, list[int]]:
    """Loads the prototype matrix ``(18, 384)`` and its class_ids.

    Helper for TSViT training: the model indexes this matrix by
    each pixel's class to obtain the target semantic prototype of the
    contrastive alignment.

    Args:
        path: Prototypes parquet.

    Returns:
        ``(prototypes (18,384) float32, sorted class_ids)``.
    """
    df = pl.read_parquet(path)
    emb_cols = [f"emb_{j:03d}" for j in range(_EMB_DIM)]
    prototypes = df.select(emb_cols).to_numpy().astype(np.float32)
    class_ids = df["class_id"].to_list()
    return prototypes, class_ids


def load_class_prototype_matrix_by_id(
    path: Path,
    *,
    num_classes: int,
) -> np.ndarray:
    """Loads the prototypes as a dense ``(num_classes, 384)`` ROW-INDEXED matrix.

    Unlike :func:`load_class_prototype_embeddings` (which returns the rows in
    parquet order plus their ids), this returns a matrix whose row ``k`` IS the
    prototype of class id ``k``, so it can be indexed directly by the dense pixel
    label in :func:`ml.models.pheno_semantic_branch.phenology_contrastive_loss`
    (which does ``feats @ protos.t()`` and ``cross_entropy(logits, labels)`` with
    ``labels`` = the pixel class ids). Class ids absent from the parquet (e.g. the
    background row 0, or an Italian class never seen in the scanned patches) stay
    as a zero row -- never indexed because the contrastive loss only scores valid,
    in-range, non-ignored pixels.

    Args:
        path: The per-class prototypes parquet (``class_id`` + ``emb_000..383``).
        num_classes: The dense head size ``K`` (= ``label_space.num_classes``,
            background included). The returned matrix is ``(K, 384)``.

    Returns:
        A ``(num_classes, 384)`` float32 matrix, row ``k`` = prototype of class
        id ``k`` (zero row for an absent id).
    """
    df = pl.read_parquet(path)
    emb_cols = [f"emb_{j:03d}" for j in range(_EMB_DIM)]
    matrix = np.zeros((num_classes, _EMB_DIM), dtype=np.float32)
    embs = df.select(emb_cols).to_numpy().astype(np.float32)
    for row_i, cid in enumerate(df["class_id"].to_list()):
        cid_int = int(cid)
        if 0 <= cid_int < num_classes:
            matrix[cid_int] = embs[row_i]
    return matrix


def _build_arg_parser():  # pragma: no cover - CLI thin wrapper.
    import argparse

    p = argparse.ArgumentParser(
        description="Genera los 18 prototipos fenologicos por clase (Wen 2025)."
    )
    p.add_argument("--pastis-root", type=Path, default=_DEFAULT_PASTIS_ROOT)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--n-time-bins", type=int, default=_N_TIME_BINS)
    p.add_argument("--max-patches", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = _build_arg_parser().parse_args(argv)
    out = generate_class_prototypes(
        args.pastis_root,
        output_path=args.output,
        model=args.model,
        n_time_bins=args.n_time_bins,
        max_patches=args.max_patches,
    )
    logger.info("done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
