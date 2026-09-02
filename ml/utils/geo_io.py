"""Centralized geospatial I/O: reading/writing of COG/GeoTIFF (US-017+).

Move here the helpers that touch rasterio so they are not duplicated in
`ml/farslip/`, `ml/ingest/`, `dagster_project/assets/`, etc.

Public API:
    - ``write_crop_tiff(arr, out_path)``: writes ``(C,H,W)`` uint16 as a simple
      GeoTIFF (identity transform for synthetic fixtures).
    - ``read_crop_tiff(path, n_expected_bands)``: reads a TIFF and validates shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

_log = structlog.get_logger(__name__)


def write_crop_tiff(arr: np.ndarray, out_path: Path) -> Path:
    """Write a crop ``(C, H, W)`` uint16 as a GeoTIFF without real georeference.

    For synthetic crops / tests we do not need a CRS. In production the GEE
    builder produces georeferenced COGs; this helper applies an identity
    transform (Affine.identity) to keep compatibility with rasterio. If
    ``rasterio`` is not installed, it falls back to ``.npy`` (path with changed
    suffix) for CPU tests without GDAL.

    Args:
        arr: array ``(C, H, W)`` dtype uint16.
        out_path: destination path of the ``.tif``. Parent is created
            automatically.

    Returns:
        Path actually written (may be ``.npy`` if rasterio is absent).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.transform import from_origin  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover
        fallback = out_path.with_suffix(".npy")
        np.save(fallback, arr)
        return fallback
    c, h, w = arr.shape
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype="uint16",
        transform=transform,
    ) as dst:
        dst.write(arr)
    return out_path


def read_crop_tiff(path: Path, n_expected_bands: int | None = None) -> np.ndarray:
    """Read a TIFF and return an array ``(C, H, W)``.

    Args:
        path: path to the ``.tif``.
        n_expected_bands: if passed, validates that the file has that number of
            bands; otherwise it raises ``ValueError``.

    Returns:
        ``np.ndarray`` shape ``(C, H, W)``.

    Raises:
        ImportError: if ``rasterio`` is not installed.
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``n_expected_bands`` does not match.
    """
    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio required for read_crop_tiff") from exc
    if not path.exists():
        raise FileNotFoundError(f"TIFF missing: {path}")
    with rasterio.open(path) as src:
        arr: np.ndarray = src.read()
    if n_expected_bands is not None and arr.shape[0] != n_expected_bands:
        raise ValueError(f"TIFF {path} has {arr.shape[0]} bands; expected {n_expected_bands}")
    return np.asarray(arr)


__all__ = ["read_crop_tiff", "write_crop_tiff"]
