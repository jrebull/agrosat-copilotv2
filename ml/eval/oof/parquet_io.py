"""Parquet (de)serialization of per-pixel softmax OOF rows (US-031).

Isolates the on-disk format of the per-pixel softmax dump so it can be tested
without loading any checkpoint. Each patch row carries a dense softmax
``(num_classes, size, size)`` and an argmax ``pred (size, size)``; storing those
3D arrays in a columnar parquet requires flattening them to fixed-length lists
and recording the shapes so the reader can reconstruct the tensors.

Format decisions (plan Section 7, R-SIZE):

- ``softmax`` -> flat ``List(Float16)`` of length ``num_classes * size * size``
  (float16 halves the size with negligible loss for a softmax).
- ``pred`` -> flat ``List(Int8)`` of length ``size * size`` (18 classes fit in
  int8; ignore/argmax values stay < 128).
- Compression is **zstd** (parquet level), which compresses correlated
  probability arrays well.
- The reconstruction shapes live in the parquet **key-value metadata**
  (``oof_num_classes`` / ``oof_size`` / ``oof_softmax_dtype``), so
  :func:`read_softmax_parquet` rebuilds ``(C, H, W)`` / ``(H, W)`` arrays without
  any external schema.

Polars is the public I/O surface (project convention: never pandas); pyarrow is
used only to attach the file-level metadata Polars does not expose directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

__all__ = [
    "META_NUM_CLASSES",
    "META_SIZE",
    "META_SOFTMAX_DTYPE",
    "read_softmax_parquet",
    "write_softmax_parquet",
]

#: Parquet key-value metadata key holding the class count of the softmax arrays.
META_NUM_CLASSES = b"oof_num_classes"
#: Parquet key-value metadata key holding the spatial side of the softmax arrays.
META_SIZE = b"oof_size"
#: Parquet key-value metadata key holding the stored softmax dtype (float16/32).
META_SOFTMAX_DTYPE = b"oof_softmax_dtype"

#: Columns whose values are flat arrays reconstructed on read.
_SOFTMAX_COL = "softmax"
_PRED_COL = "pred"

#: numpy dtype per stored softmax dtype string.
_DTYPE_MAP: dict[str, type[np.floating]] = {
    "float16": np.float16,
    "float32": np.float32,
}


def write_softmax_parquet(
    rows: list[dict[str, Any]],
    path: Path | str,
    *,
    num_classes: int = 18,
    size: int = 128,
    dtype: str = "float16",
    compression: str = "zstd",
) -> None:
    """Write per-patch softmax rows to a zstd parquet with flattened arrays.

    Each row must contain at least ``softmax`` (a ``(num_classes, size, size)``
    array) and ``pred`` (a ``(size, size)`` integer array); the remaining scalar
    fields (``patch_id``, ``fold``, ``held_out``, ``model``, ``status``,
    ``code_version``, ``data_version``, ...) are written verbatim. The 3D
    ``softmax`` and 2D ``pred`` are flattened to fixed-length lists; the
    reconstruction shapes are stored in the parquet key-value metadata.

    Rows whose ``softmax``/``pred`` is ``None`` (e.g. a ``status="missing"``
    model) are written with empty lists, so a single parquet can hold the
    missing-checkpoint sentinel alongside scored patches.

    Args:
        rows: List of per-patch row dicts (see above).
        path: Output ``.parquet`` path.
        num_classes: Class count ``C`` of each softmax (default 18).
        size: Spatial side ``H == W`` of each map (default 128).
        dtype: Stored softmax dtype, ``"float16"`` (default) or ``"float32"``.
        compression: Parquet compression codec (default ``"zstd"``).

    Raises:
        ValueError: if ``dtype`` is unsupported or a ``softmax``/``pred`` array
            does not match the declared ``(num_classes, size, size)`` shape.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"unsupported softmax dtype: {dtype!r}; use float16/float32.")
    np_dtype = _DTYPE_MAP[dtype]
    expected_softmax = (num_classes, size, size)
    expected_pred = (size, size)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        record = {k: v for k, v in row.items() if k not in (_SOFTMAX_COL, _PRED_COL)}
        softmax = row.get(_SOFTMAX_COL)
        pred = row.get(_PRED_COL)

        if softmax is None:
            record[_SOFTMAX_COL] = []
        else:
            arr = np.asarray(softmax)
            if arr.shape != expected_softmax:
                raise ValueError(f"softmax shape {arr.shape} != expected {expected_softmax}.")
            record[_SOFTMAX_COL] = arr.astype(np_dtype).reshape(-1).tolist()

        if pred is None:
            record[_PRED_COL] = []
        else:
            parr = np.asarray(pred)
            if parr.shape != expected_pred:
                raise ValueError(f"pred shape {parr.shape} != expected {expected_pred}.")
            record[_PRED_COL] = parr.astype(np.int8).reshape(-1).tolist()

        out_rows.append(record)

    schema_overrides = {
        _SOFTMAX_COL: pl.List(pl.Float16 if dtype == "float16" else pl.Float32),
        _PRED_COL: pl.List(pl.Int8),
    }
    frame = pl.DataFrame(out_rows, schema_overrides=schema_overrides)

    table = frame.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta[META_NUM_CLASSES] = str(num_classes).encode()
    meta[META_SIZE] = str(size).encode()
    meta[META_SOFTMAX_DTYPE] = dtype.encode()
    table = table.replace_schema_metadata(meta)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression=compression)


def read_softmax_parquet(path: Path | str) -> pl.DataFrame:
    """Read a softmax parquet, reconstructing the dense arrays.

    Reverses :func:`write_softmax_parquet`: reads the flat ``softmax`` /
    ``pred`` lists and the shape metadata, then rebuilds an object column where
    each entry is the reconstructed ``(C, H, W)`` softmax (numpy, original stored
    dtype) and the ``(H, W)`` ``pred`` (numpy int8). Missing-checkpoint rows
    (empty lists) reconstruct to ``None``.

    Args:
        path: Input ``.parquet`` path written by :func:`write_softmax_parquet`.

    Returns:
        A Polars DataFrame with every scalar column intact and the ``softmax`` /
        ``pred`` columns reconstructed as object columns of numpy arrays
        (``None`` for rows without a payload).

    Raises:
        ValueError: if the file lacks the shape metadata.
    """
    in_path = Path(path)
    table = pq.read_table(in_path)
    meta = table.schema.metadata or {}
    if META_NUM_CLASSES not in meta or META_SIZE not in meta:
        raise ValueError(
            f"{in_path} lacks OOF shape metadata; was it written by write_softmax_parquet?"
        )
    num_classes = int(meta[META_NUM_CLASSES])
    size = int(meta[META_SIZE])
    dtype = meta.get(META_SOFTMAX_DTYPE, b"float16").decode()
    np_dtype = _DTYPE_MAP.get(dtype, np.float16)

    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)

    softmax_arrays: list[np.ndarray | None] = []
    pred_arrays: list[np.ndarray | None] = []
    for sm, pr in zip(frame[_SOFTMAX_COL].to_list(), frame[_PRED_COL].to_list(), strict=True):
        if sm:
            softmax_arrays.append(np.asarray(sm, dtype=np_dtype).reshape(num_classes, size, size))
        else:
            softmax_arrays.append(None)
        if pr:
            pred_arrays.append(np.asarray(pr, dtype=np.int8).reshape(size, size))
        else:
            pred_arrays.append(None)

    return frame.with_columns(
        pl.Series(_SOFTMAX_COL, softmax_arrays, dtype=pl.Object),
        pl.Series(_PRED_COL, pred_arrays, dtype=pl.Object),
    )
