"""Loading of PASTIS-R patches from disk into Numpy / Polars structures.

PASTIS-R (Sainte-Fare-Garnot et al. 2021) delivers multitemporal Sentinel-2
as `(T, 10, 128, 128)` int16 tensors in `DATA_S2/S2_<patch_id>.npy`, panoptic
annotations `(3, 128, 128)` uint8 in `ANNOTATIONS/TARGET_<patch_id>.npy`,
per-patch metadata in `metadata.geojson` (EPSG:2154) and per-fold statistics
in `NORM_S2_patch.json`.

This module exposes lightweight helpers to load 1 patch or iterate over
several, and to convert a sample to an ergonomic long-format `pl.DataFrame`
for EDA.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PASTIS_S2_BANDS: list[str] = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]
"""Canonical order of the 10 Sentinel-2 bands kept in PASTIS-R.

The atmospheric bands B01, B09, B10 are excluded in the original dataset.
"""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_CLASS_MAP_PATH = _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"


def _load_class_mapping(path: Path = _CLASS_MAP_PATH) -> dict[int, str]:
    """Load the table of canonical PASTIS-R classes (20 classes).

    Args:
        path: Path to `pastis_class_mapping.json`.

    Returns:
        Dictionary `{class_id: name}` with the 20 classes (0..19).
    """
    if not path.exists():
        # Minimal fallback mapping if the JSON is not found (degraded mode)
        return {0: "Background", 19: "Void label"}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    classes = raw.get("classes", {})
    return {int(k): v["name"] for k, v in classes.items()}


def _load_groupings(path: Path = _CLASS_MAP_PATH) -> dict[str, dict[int, str]]:
    """Load the derived groupings (phenological_cycle, agronomic_group, etc.).

    Args:
        path: Path to `pastis_class_mapping.json`.

    Returns:
        Dictionary `{grouping_name: {class_id: group_name}}`. Empty if it does
        not exist.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    groupings = raw.get("groupings", {})
    out: dict[str, dict[int, str]] = {}
    for name, payload in groupings.items():
        mp = payload.get("map", {}) if isinstance(payload, dict) else {}
        if mp:
            out[name] = {int(k): v for k, v in mp.items()}
    return out


PASTIS_CLASS_MAP: dict[int, str] = _load_class_mapping()
"""Mapping `class_id -> readable name` loaded from `data/reference/pastis_class_mapping.json`."""

PASTIS_R_CLASSES: dict[int, str] = PASTIS_CLASS_MAP
"""Semantic alias required by US-011 (public signature of the plan)."""

PASTIS_R_GROUPINGS: dict[str, dict[int, str]] = _load_groupings()
"""Derived groupings (`phenological_cycle`, `agronomic_group`, `cereals_winter_vs_spring`)."""


def load_pastis_patch(
    patch_id: str | int,
    root: Path | None = None,
    load_annotations: bool = True,
) -> dict[str, Any]:
    """Load a PASTIS-R patch (S2 + annotations + dates + fold).

    Args:
        patch_id: Patch identifier (numeric or string without extension).
        root: Dataset root (default `data/PASTIS-R/`).
        load_annotations: If True attempts to load `TARGET_<id>.npy`.

    Returns:
        Dictionary with keys:
            - `s2`: ndarray int16 shape (T, 10, 128, 128)
            - `semantic`: ndarray uint8 (128, 128) | None — channel 0 of TARGET
            - `instance`: ndarray uint8 (128, 128) | None — channel 1 of TARGET
            - `zone`: ndarray uint8 (128, 128) | None — channel 2 of TARGET
            - `dates_s2`: list[int] — sorted YYYYMMDD dates
            - `patch_id`: str
            - `fold`: int (1..5) if available in metadata, otherwise None

    Raises:
        FileNotFoundError: If the patch's S2 file does not exist.
    """
    root = root or _DEFAULT_ROOT
    pid = str(patch_id)

    s2_path = root / "DATA_S2" / f"S2_{pid}.npy"
    if not s2_path.exists():
        raise FileNotFoundError(f"PASTIS S2 patch not found: {s2_path}")
    s2 = np.load(s2_path)

    semantic = instance = zone = None
    if load_annotations:
        tgt_path = root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if tgt_path.exists():
            tgt = np.load(tgt_path)
            # shape (3, 128, 128): semantic, instance, zone
            semantic = tgt[0]
            instance = tgt[1] if tgt.shape[0] > 1 else None
            zone = tgt[2] if tgt.shape[0] > 2 else None

    dates_s2: list[int] = []
    fold: int | None = None
    metadata_path = root / "metadata.geojson"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as fh:
            md = json.load(fh)
        for feat in md.get("features", []):
            props = feat.get("properties", {})
            if str(props.get("ID_PATCH")) == pid:
                dates_raw = props.get("dates-S2", {})
                if isinstance(dates_raw, dict):
                    dates_s2 = [
                        int(v) for _, v in sorted(dates_raw.items(), key=lambda kv: int(kv[0]))
                    ]
                fold_val = props.get("Fold")
                if fold_val is not None:
                    fold = int(fold_val)
                break

    return {
        "s2": s2,
        "semantic": semantic,
        "instance": instance,
        "zone": zone,
        "dates_s2": dates_s2,
        "patch_id": pid,
        "fold": fold,
    }


def iter_pastis_patches(
    patch_ids: list[str | int],
    root: Path | None = None,
    load_annotations: bool = True,
) -> Iterator[dict[str, Any]]:
    """Iterate over multiple patches without loading them all into memory.

    Args:
        patch_ids: List of identifiers.
        root: Dataset root.
        load_annotations: Whether to load TARGETs.

    Yields:
        Dictionary per patch with the same structure as `load_pastis_patch`.
    """
    for pid in patch_ids:
        try:
            yield load_pastis_patch(pid, root=root, load_annotations=load_annotations)
        except FileNotFoundError:
            continue


def pastis_to_polars(
    patch_ids: list[str | int],
    bands: list[str] | None = None,
    root: Path | None = None,
    include_labels: bool = True,
    include_dates: bool = True,
    pixel_stride: int = 1,
) -> pl.DataFrame:
    """Convert a list of PASTIS-R patches to a long-format `pl.DataFrame`.

    Resulting columns:
        `patch_id, t, date, y, x, band, value, class_id, class_name, fold`

    To avoid OOM with high N, `pixel_stride > 1` is allowed, which samples
    1 of every `stride` pixels per band and per t.

    Args:
        patch_ids: Identifiers of patches to load.
        bands: Subset of bands to include (default all PASTIS_S2_BANDS).
        root: Dataset root.
        include_labels: Whether to add `class_id`, `class_name` columns from TARGET[0].
        include_dates: Whether to add a `date` (YYYYMMDD) column parsed from the metadata.
        pixel_stride: Spatial subsampling stride (1 = all pixels).

    Returns:
        Long-format Polars DataFrame. Empty if no patch could be loaded.
    """
    selected = bands or PASTIS_S2_BANDS
    band_idx = [PASTIS_S2_BANDS.index(b) for b in selected]
    frames: list[pl.DataFrame] = []

    for patch in iter_pastis_patches(patch_ids, root=root, load_annotations=include_labels):
        s2 = patch["s2"]
        T, _, H, W = s2.shape
        dates = patch.get("dates_s2") or [0] * T
        ys = np.arange(0, H, pixel_stride)
        xs = np.arange(0, W, pixel_stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy_flat = yy.ravel()
        xx_flat = xx.ravel()
        n_pix = yy_flat.size

        for t in range(T):
            for bi, bname in zip(band_idx, selected, strict=True):
                vals = s2[t, bi, ys[:, None], xs[None, :]].ravel().astype(np.int32)
                date_val = int(dates[t]) if include_dates and t < len(dates) else 0
                cls_arr: np.ndarray | None = None
                if include_labels and patch.get("semantic") is not None:
                    cls_arr = patch["semantic"][ys[:, None], xs[None, :]].ravel()
                frame = pl.DataFrame(
                    {
                        "patch_id": [patch["patch_id"]] * n_pix,
                        "t": [t] * n_pix,
                        "date": [date_val] * n_pix,
                        "y": yy_flat,
                        "x": xx_flat,
                        "band": [bname] * n_pix,
                        "value": vals,
                        "class_id": (
                            cls_arr.astype(np.int16)
                            if cls_arr is not None
                            else np.zeros(n_pix, dtype=np.int16)
                        ),
                        "fold": [patch.get("fold") or 0] * n_pix,
                    }
                )
                frames.append(frame)

    if not frames:
        return pl.DataFrame(
            schema={
                "patch_id": pl.Utf8,
                "t": pl.Int64,
                "date": pl.Int64,
                "y": pl.Int64,
                "x": pl.Int64,
                "band": pl.Utf8,
                "value": pl.Int32,
                "class_id": pl.Int16,
                "fold": pl.Int64,
            }
        )

    df = pl.concat(frames, how="vertical_relaxed")
    if include_labels:
        name_map = PASTIS_CLASS_MAP
        df = df.with_columns(
            pl.col("class_id")
            .cast(pl.Int64)
            .replace_strict(name_map, default="unknown")
            .alias("class_name")
        )
    return df


def _multipolygon_centroid_2154(coordinates: list[Any]) -> tuple[float, float]:
    """Compute the EPSG:2154 centroid of a GeoJSON MultiPolygon geometry.

    Minimal implementation without a shapely dependency so the unit test can
    run even if shapely is not installed. If shapely is available it is
    delegated to (more exact over polygons with holes).

    Args:
        coordinates: GeoJSON MultiPolygon-style coordinate list:
            `[[[ [x,y], ... ]], ...]`.

    Returns:
        Tuple `(x, y)` in EPSG:2154.
    """
    try:
        from shapely.geometry import MultiPolygon, shape  # type: ignore[import-untyped]

        geom = shape({"type": "MultiPolygon", "coordinates": coordinates})
        if isinstance(geom, MultiPolygon):
            c = geom.centroid
            return float(c.x), float(c.y)
    except Exception:  # noqa: BLE001, S110
        # Intentional silent fallback: shapely missing or invalid geometry.
        # We fall back to the manual computation with the vertices of the first
        # ring (loop below). No log signal because shapely is optional.
        pass

    # Fallback: arithmetic mean of all vertices of the first ring
    xs: list[float] = []
    ys: list[float] = []
    for poly in coordinates:
        if not poly:
            continue
        outer = poly[0]
        for pt in outer:
            if len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    if not xs:
        return 0.0, 0.0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def pastis_patch_coords(
    metadata_geojson: Path,
    target_crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """Read PASTIS-R `metadata.geojson` (EPSG:2154) and reproject the centroids.

    Iterates over each feature, computes the MultiPolygon centroid in EPSG:2154
    (Lambert-93) and reprojects it to the target CRS (default EPSG:4326).

    Args:
        metadata_geojson: Path to the `metadata.geojson` distributed with PASTIS-R.
        target_crs: Target CRS for `lon`/`lat`. Default `EPSG:4326`.

    Returns:
        Polars DataFrame with columns `patch_id, lon, lat, tile, fold`. If the
        file does not exist or contains no features returns an empty DataFrame.
    """
    schema: dict[str, Any] = {
        "patch_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "tile": pl.Utf8,
        "fold": pl.Int64,
    }
    if not metadata_geojson.exists():
        return pl.DataFrame(schema=schema)

    with metadata_geojson.open(encoding="utf-8") as fh:
        gj = json.load(fh)

    features = gj.get("features", [])
    if not features:
        return pl.DataFrame(schema=schema)

    try:
        from pyproj import Transformer  # type: ignore[import-untyped]

        transformer = Transformer.from_crs("EPSG:2154", target_crs, always_xy=True)
    except Exception:  # noqa: BLE001
        transformer = None

    rows: list[dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        if geom.get("type") != "MultiPolygon":
            continue
        coords = geom.get("coordinates", [])
        cx, cy = _multipolygon_centroid_2154(coords)
        if transformer is not None:
            lon, lat = transformer.transform(cx, cy)
        else:
            # Without pyproj we return coords without reprojecting (degraded mode).
            lon, lat = cx, cy
        pid_raw = feat.get("id") or props.get("ID_PATCH")
        pid = str(pid_raw) if pid_raw is not None else ""
        fold_raw = props.get("Fold")
        rows.append(
            {
                "patch_id": pid,
                "lon": float(lon),
                "lat": float(lat),
                "tile": str(props.get("TILE", "")),
                "fold": int(fold_raw) if fold_raw is not None else 0,
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def pastis_patch_index(metadata_geojson: Path | None = None) -> pl.DataFrame:
    """Read `metadata.geojson` and return a flat index of PASTIS-R patches.

    Useful for stratified sampling by `(TILE, Fold)` before loading each
    patch's actual `.npy`. Lighter than `pastis_patch_coords` (does not
    reproject geometries) and more semantic than parsing the GeoJSON inline in
    each notebook.

    Args:
        metadata_geojson: Path to PASTIS-R's `metadata.geojson`. If None,
            uses `_DEFAULT_ROOT / "metadata.geojson"` (relative to the repo).

    Returns:
        Polars DataFrame with columns `patch_id` (str), `TILE` (str), `Fold`
        (int64). Empty if the file does not exist.
    """
    schema: dict[str, Any] = {
        "patch_id": pl.Utf8,
        "TILE": pl.Utf8,
        "Fold": pl.Int64,
    }
    meta = metadata_geojson or (_DEFAULT_ROOT / "metadata.geojson")
    if not meta.exists():
        return pl.DataFrame(schema=schema)

    with meta.open(encoding="utf-8") as fh:
        gj = json.load(fh)

    rows: list[dict[str, Any]] = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        pid_raw = feat.get("id") or props.get("ID_PATCH")
        if pid_raw is None:
            continue
        fold_raw = props.get("Fold")
        rows.append(
            {
                "patch_id": str(pid_raw),
                "TILE": str(props.get("TILE", "")),
                "Fold": int(fold_raw) if fold_raw is not None else 0,
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def pastis_pixel_labels(
    patch_id: str,
    root: Path | None = None,
    sample_per_patch: int | None = None,
    seed: int = 42,
    exclude_classes: tuple[int, ...] = (0, 19),
) -> pl.DataFrame:
    """Load `TARGET_{patch_id}.npy[0]` and georeference each pixel to (lon, lat) 4326.

    Uses the MultiPolygon bbox in `metadata.geojson` projected to EPSG:4326 as
    spatial reference and generates a regular 128x128 grid over that bbox. It
    is a sufficient approximation for EDA (exact alignment would require the
    GeoTIFF `transform` that PASTIS-R does not distribute with the `.npy`).

    Args:
        patch_id: Patch identifier.
        root: Dataset root (default `data/PASTIS-R/`).
        sample_per_patch: If not None, randomly samples N pixels.
        seed: Seed for reproducibility.
        exclude_classes: Classes to filter before the sample (default bg and void).

    Returns:
        Polars DataFrame with columns `px_id, patch_id, lon, lat, class_id,
        class_name`. Empty if the file does not exist.
    """
    schema: dict[str, Any] = {
        "px_id": pl.Utf8,
        "patch_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "class_id": pl.Int16,
        "class_name": pl.Utf8,
    }
    root = root or _DEFAULT_ROOT
    pid = str(patch_id)
    tgt_path = root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
    metadata_path = root / "metadata.geojson"
    if not tgt_path.exists():
        return pl.DataFrame(schema=schema)

    target = np.load(tgt_path)
    semantic = target[0] if target.ndim == 3 else target

    # Recover the 4326 bbox from metadata.geojson
    lon_min = lat_min = lon_max = lat_max = None
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as fh:
            md = json.load(fh)
        try:
            from pyproj import Transformer  # type: ignore[import-untyped]

            transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        except Exception:  # noqa: BLE001
            transformer = None
        for feat in md.get("features", []):
            if (
                str(feat.get("id")) == pid
                or str((feat.get("properties") or {}).get("ID_PATCH")) == pid
            ):
                coords = (feat.get("geometry") or {}).get("coordinates", [])
                xs: list[float] = []
                ys: list[float] = []
                for poly in coords:
                    if not poly:
                        continue
                    for pt in poly[0]:
                        xs.append(float(pt[0]))
                        ys.append(float(pt[1]))
                if xs and ys and transformer is not None:
                    lon1, lat1 = transformer.transform(min(xs), min(ys))
                    lon2, lat2 = transformer.transform(max(xs), max(ys))
                    lon_min, lat_min = min(lon1, lon2), min(lat1, lat2)
                    lon_max, lat_max = max(lon1, lon2), max(lat1, lat2)
                elif xs and ys:
                    lon_min, lat_min, lon_max, lat_max = (
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    )
                break

    if lon_min is None or lat_min is None or lon_max is None or lat_max is None:
        # Fallback: dummy bbox centered at (0,0) — only ensures the shape
        # is correct without asserting geographic correctness.
        lon_min, lat_min, lon_max, lat_max = 0.0, 0.0, 1.0, 1.0

    H, W = semantic.shape
    lon_axis = np.linspace(lon_min, lon_max, W)
    lat_axis = np.linspace(lat_max, lat_min, H)  # rows top->bottom
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)

    cls_flat = semantic.ravel().astype(np.int16)
    lon_flat = lon_grid.ravel().astype(np.float64)
    lat_flat = lat_grid.ravel().astype(np.float64)
    idx = np.arange(cls_flat.size)

    if exclude_classes:
        mask = ~np.isin(cls_flat, np.asarray(exclude_classes, dtype=np.int16))
        cls_flat = cls_flat[mask]
        lon_flat = lon_flat[mask]
        lat_flat = lat_flat[mask]
        idx = idx[mask]

    if sample_per_patch is not None and cls_flat.size > sample_per_patch:
        rng = np.random.default_rng(seed)
        choice = rng.choice(cls_flat.size, size=sample_per_patch, replace=False)
        cls_flat = cls_flat[choice]
        lon_flat = lon_flat[choice]
        lat_flat = lat_flat[choice]
        idx = idx[choice]

    if cls_flat.size == 0:
        return pl.DataFrame(schema=schema)

    px_ids = [f"{pid}_{int(i)}" for i in idx]
    cls_names = [PASTIS_CLASS_MAP.get(int(c), "unknown") for c in cls_flat]
    return pl.DataFrame(
        {
            "px_id": px_ids,
            "patch_id": [pid] * cls_flat.size,
            "lon": lon_flat,
            "lat": lat_flat,
            "class_id": cls_flat,
            "class_name": cls_names,
        },
        schema=schema,
    )
