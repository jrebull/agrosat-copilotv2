"""Texture-aware ensemble TL: feed TSViT/U-TAE REAL Sentinel Hub crops.

The pilot in :mod:`ml.transfer.ensemble_full_tl` fed the dense champion members a
single pixel tiled into a flat patch (no texture), and their contribution was
marginal (+0.013 at k=10). The diagnosed cause: TSViT/U-TAE draw their signal
from the spatial TEXTURE of a real patch, which the per-parcel pixel does not
carry. This module closes that gap with real raster crops pulled from the
Sentinel Hub Process API (:mod:`ml.ingest.sh_client`, CDSE-authenticated):

    For a small subset of EuroCropsML parcels (per-parcel centroid available in
    the AlphaEarth parquet), download a temporal stack of real ``16x16`` L2A
    crops, feed THOSE through the dense members as extractors, and re-run the
    same few-shot transfer comparison (ensemble vs xgb-alphaearth, same k, same
    test). If the dense members now lift transfer materially more than the
    pixel-tiled run did, that confirms texture was the missing input.

Honesty
-------
- Real network + quota cost: each (parcel, window) is one Process API request, so
  the subset is small by default.
- Crops that come back empty (cloud, no acquisition) are skipped; a parcel with
  fewer than two usable frames is dropped and counted.
- The comparison is matched: the SAME parcels, the SAME few-shot budget, the
  SAME target test. Only the dense members' INPUT changes (real crop vs tiled
  pixel). The verdict is whatever the numbers say.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from ml.transfer.ensemble_full_tl import (
    _N_TIMESTEPS,
    _extract_dense_features,
    _fewshot_curve,
)

logger = structlog.get_logger(__name__)

__all__ = ["TextureTLResult", "build_season_windows", "fit_patch_to_timesteps", "run_texture_tl"]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_TRANSFER_DIR: Path = _REPO_ROOT / "data" / "transfer"
_OUT_DIR: Path = _TRANSFER_DIR / "ensemble_texture_tl"
_HCAT3_CSV: Path = _REPO_ROOT / "data" / "reference" / "eurocrops_hcat3.csv"
_ALPHAEARTH_COLS: tuple[str, ...] = tuple(f"dim_{i:02d}" for i in range(64))
_MIN_LEAF_SUPPORT: int = 10
_RANDOM_STATE: int = 42

#: On-disk cache of downloaded Sentinel Hub patches, keyed by region + coordinate.
#: A SH request is paid quota; caching every downloaded parcel means a re-run never
#: re-pays for a parcel it already fetched. Lives OUTSIDE the DVC-tracked data dirs
#: (heavy, derived) -- gitignored.
_PATCH_CACHE_DIR: Path = _TRANSFER_DIR / "_sh_patch_cache"


def _coord_key(lon: float, lat: float) -> str:
    """Stable filename key for a parcel centroid (6 decimals ~ 0.1 m precision)."""
    return f"{lon:.6f}_{lat:.6f}"


@dataclass
class TextureTLResult:
    """Few-shot transfer F1 with REAL-texture crops vs the tabular baseline."""

    per_k: pl.DataFrame
    summary: dict[str, object] = field(default_factory=dict)


def build_season_windows(year: int = 2021) -> list[tuple[str, str]]:
    """Monthly date windows spanning the growing season (May-September).

    Args:
        year: Campaign year.

    Returns:
        List of ``(date_from, date_to)`` ISO-date windows, one per month.
    """
    months = [(5, 1, 28), (6, 1, 28), (6, 15, 30), (7, 1, 28), (7, 15, 31), (8, 1, 28), (9, 1, 28)]
    return [(f"{year}-{m:02d}-{d0:02d}", f"{year}-{m:02d}-{d1:02d}") for (m, d0, d1) in months]


def fit_patch_to_timesteps(stack: np.ndarray, n: int = _N_TIMESTEPS) -> np.ndarray:
    """Resample a real crop stack ``(T, B, H, W)`` to exactly ``n`` timesteps.

    Equispaced index subsample when ``T > n``; repeat the last frame when
    ``T < n`` (TSViT-fullm needs exactly its trained length; U-TAE is flexible but
    one length keeps a stackable batch).

    Args:
        stack: Real crop stack ``(T, B, H, W)`` float32.
        n: Target timestep count.

    Returns:
        A ``(n, B, H, W)`` float32 stack.
    """
    t = stack.shape[0]
    if t == n:
        return stack
    if t > n:
        idx = np.linspace(0, t - 1, n).round().astype(int)
        subsampled: np.ndarray = stack[idx]
        return subsampled
    pad = np.repeat(stack[-1:], n - t, axis=0)
    return np.concatenate([stack, pad], axis=0)


def _load_hcat_name_map() -> dict[int, str]:
    """Return ``{hcat_code: hcat_leaf_name}`` from the HCAT v3 reference CSV."""
    if not _HCAT3_CSV.is_file():
        raise FileNotFoundError(f"HCAT v3 reference missing at {_HCAT3_CSV}")
    h = pl.read_csv(_HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8})
    return {int(c): str(n) for c, n in zip(h["HCAT3_code"], h["HCAT3_name"], strict=True)}


@dataclass
class _RegionTexture:
    """AlphaEarth annual + real-texture patches + leaf for a region's parcels."""

    annual: np.ndarray
    patches: list[np.ndarray]
    leaf: np.ndarray
    n_downloaded: int
    n_dropped: int


def _load_region_texture(
    region: str,
    *,
    sh_client: object,
    windows: list[tuple[str, str]],
    max_parcels: int,
    size: int,
    max_cloud: float,
    seed: int,
    stratify_keep: set[str] | None = None,
    per_class: int | None = None,
) -> _RegionTexture:
    """Download real-texture series for a region's parcels and pair with AlphaEarth.

    Reads the AlphaEarth parquet (64-dim + ``lon``/``lat`` + ``hcat_code``),
    applies the support guard + cap, then pulls a real crop stack per parcel from
    Sentinel Hub. Parcels with insufficient cloud-free frames are dropped.

    Args:
        region: EuroCropsML region key.
        sh_client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        windows: Season date windows.
        max_parcels: Parcel cap (small -- each parcel is several API calls).
        size: Patch side in pixels.
        max_cloud: Max cloud cover per window.
        seed: Sampling seed.

    Returns:
        A :class:`_RegionTexture`.
    """
    parquet = _TRANSFER_DIR / f"eurocropsml_alphaearth_{region}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"EuroCropsML parquet missing at {parquet}")
    name_map = _load_hcat_name_map()
    df = pl.read_parquet(
        parquet, columns=["lon", "lat", "hcat_code", *_ALPHAEARTH_COLS]
    ).with_columns(
        pl.col("hcat_code")
        .cast(pl.Int64)
        .replace_strict(name_map, default="unknown_hcat", return_dtype=pl.Utf8)
        .alias("leaf")
    )
    counts = df.group_by("leaf").len().filter(pl.col("len") >= _MIN_LEAF_SUPPORT)
    df = df.filter(pl.col("leaf").is_in(counts["leaf"].to_list()))
    if stratify_keep is not None and per_class is not None:
        # Representative per-class draw (rare leaves not starved). Restricts to the
        # label-space and caps each leaf at ``per_class``.
        from ml.transfer.finetune_baltico import stratified_parcel_sample

        picked = stratified_parcel_sample(
            df["leaf"].to_list(), keep=stratify_keep, per_class=per_class, seed=seed
        )
        df = df[picked]
    elif df.height > max_parcels:
        df = df.sample(n=max_parcels, seed=seed, shuffle=True)

    lons = df["lon"].to_list()
    lats = df["lat"].to_list()
    leaves = df["leaf"].to_list()
    annual_mat = df.select(_ALPHAEARTH_COLS).to_numpy().astype(np.float64)

    # One ORBIT request per parcel (whole season multi-temporal). EVERY downloaded
    # patch is CACHED to disk keyed by (region, lon, lat, size) so a parcel is
    # NEVER re-downloaded across runs -- a Sentinel Hub request is paid quota, and
    # re-fetching the same parcel wastes it. On a re-run only the parcels still
    # missing from the cache hit the network.
    coords = list(zip(lons, lats, strict=True))
    date_from, date_to = windows[0][0], windows[-1][1]
    cache_dir = _PATCH_CACHE_DIR / f"{region}_s{size}_c{int(max_cloud)}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[int, np.ndarray | None] = {}
    to_fetch: list[int] = []
    for i, (lon, lat) in enumerate(coords):
        cpath = cache_dir / f"{_coord_key(lon, lat)}.npz"
        if cpath.is_file():
            try:
                cached[i] = np.load(cpath)["stack"]
            except Exception:  # noqa: BLE001 -- corrupt cache entry re-fetched
                to_fetch.append(i)
        else:
            to_fetch.append(i)
    logger.info(
        "texture_cache_status",
        region=region,
        n_cached=len(cached),
        n_to_fetch=len(to_fetch),
        n_total=len(coords),
    )

    if to_fetch:
        fetch_coords = [coords[i] for i in to_fetch]
        fetched = sh_client.parcel_series_batch(  # type: ignore[attr-defined]
            fetch_coords,
            date_from=date_from,
            date_to=date_to,
            size=size,
            max_cloud=max_cloud,
            max_workers=2,
        )
        for j, i in enumerate(to_fetch):
            stack = fetched[j]
            cpath = cache_dir / f"{_coord_key(coords[i][0], coords[i][1])}.npz"
            if stack is not None:
                # Cache the raw stack BEFORE the timestep fit, so the cache is
                # reusable regardless of n_timesteps; persisted immediately so a
                # crash mid-batch keeps every parcel already paid for.
                np.savez_compressed(cpath, stack=stack)
                cached[i] = stack
            else:
                # Persist an empty marker so a cloud-failed parcel is not re-tried
                # every run (it would just fail again and waste the request).
                np.savez_compressed(cpath, stack=np.zeros((0,), dtype=np.float32))
                cached[i] = None

    annual_rows: list[np.ndarray] = []
    patch_rows: list[np.ndarray] = []
    leaf_rows: list[str] = []
    n_downloaded = 0
    n_dropped = 0
    for i in range(len(coords)):
        stack = cached.get(i)
        if stack is None or stack.size == 0:
            n_dropped += 1
            continue
        patch_rows.append(fit_patch_to_timesteps(stack))
        annual_rows.append(annual_mat[i])
        leaf_rows.append(leaves[i])
        n_downloaded += 1

    logger.info(
        "texture_region_loaded",
        region=region,
        n_downloaded=n_downloaded,
        n_dropped=n_dropped,
        n_from_cache=len(coords) - len(to_fetch),
    )
    return _RegionTexture(
        annual=np.asarray(annual_rows, dtype=np.float64),
        patches=patch_rows,
        leaf=np.asarray(leaf_rows),
        n_downloaded=n_downloaded,
        n_dropped=n_dropped,
    )


def run_texture_tl(
    *,
    source: str = "latvia",
    target: str = "estonia",
    sh_client: object,
    max_parcels_per_region: int = 80,
    dense_members: tuple[str, ...] = ("tsvit-pheno-fullm", "utae"),
    ks: tuple[int, ...] = (1, 5, 10),
    seeds: tuple[int, ...] = (0, 1, 2),
    size: int = 16,
    max_cloud: float = 40.0,
    year: int = 2021,
    device: str = "auto",
    seed: int = _RANDOM_STATE,
) -> TextureTLResult:
    """Few-shot transfer with REAL-texture crops: ensemble vs xgb-alphaearth.

    Downloads real Sentinel Hub crop stacks for a small parcel subset of both
    regions, feeds them through the dense members as extractors, concatenates with
    AlphaEarth, and runs the matched few-shot comparison (ensemble vs baseline,
    same k, same test).

    Args:
        source: Source region key.
        target: Target region key.
        sh_client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        max_parcels_per_region: Small cap (each parcel is several API calls).
        dense_members: Champion dense members to use as extractors.
        ks: Few-shot budgets.
        seeds: Seeds to average over.
        size: Crop side in pixels.
        max_cloud: Max cloud cover per window.
        year: Season year for the windows.
        device: Dense-forward device.
        seed: Sampling seed.

    Returns:
        A :class:`TextureTLResult`.

    Raises:
        ValueError: if the regions share no leaf class with downloaded crops.
    """
    windows = build_season_windows(year)
    reg_src = _load_region_texture(
        source,
        sh_client=sh_client,
        windows=windows,
        max_parcels=max_parcels_per_region,
        size=size,
        max_cloud=max_cloud,
        seed=seed,
    )
    reg_tgt = _load_region_texture(
        target,
        sh_client=sh_client,
        windows=windows,
        max_parcels=max_parcels_per_region,
        size=size,
        max_cloud=max_cloud,
        seed=seed,
    )

    shared = sorted(set(reg_src.leaf.tolist()) & set(reg_tgt.leaf.tolist()))
    if not shared:
        raise ValueError(f"{source!r} and {target!r} share no leaf with crops.")
    cls_id = {c: i for i, c in enumerate(shared)}
    keep = set(shared)
    n_classes = len(shared)

    def _prep(reg: _RegionTexture) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        mask = np.array([leaf in keep for leaf in reg.leaf], dtype=bool)
        annual = reg.annual[mask]
        y = np.array([cls_id[c] for c in reg.leaf[mask]], dtype=np.int64)
        patches = [p for p, m in zip(reg.patches, mask, strict=True) if m]
        return annual, y, patches

    a_src, y_src, p_src = _prep(reg_src)
    a_tgt, y_tgt, p_tgt = _prep(reg_tgt)

    dense_src = _extract_dense_features(p_src, dense_members, device=device)
    dense_tgt = _extract_dense_features(p_tgt, dense_members, device=device)
    ens_src = np.concatenate([a_src, *[dense_src[m] for m in dense_members]], axis=1)
    ens_tgt = np.concatenate([a_tgt, *[dense_tgt[m] for m in dense_members]], axis=1)

    rows = _fewshot_curve(
        a_src, y_src, a_tgt, y_tgt, ks=ks, n_classes=n_classes, seeds=seeds, label="baseline"
    ) + _fewshot_curve(
        ens_src, y_src, ens_tgt, y_tgt, ks=ks, n_classes=n_classes, seeds=seeds, label="ensemble"
    )
    per_k = pl.DataFrame(rows)
    agg = (
        per_k.group_by(["feature_set", "k"])
        .agg(
            pl.col("macro_f1").mean().alias("f1_mean"),
            pl.col("macro_f1").std(ddof=0).fill_null(0.0).alias("f1_std"),
        )
        .sort(["k", "feature_set"])
    )
    deltas: dict[int, float] = {}
    for k in ks:
        b = agg.filter((pl.col("feature_set") == "baseline") & (pl.col("k") == k))
        e = agg.filter((pl.col("feature_set") == "ensemble") & (pl.col("k") == k))
        if b.height and e.height:
            deltas[int(k)] = round(float(e["f1_mean"][0] - b["f1_mean"][0]), 4)

    summary: dict[str, object] = {
        "source": source,
        "target": target,
        "n_source_parcels": int(a_src.shape[0]),
        "n_target_parcels": int(a_tgt.shape[0]),
        "n_source_dropped": reg_src.n_dropped,
        "n_target_dropped": reg_tgt.n_dropped,
        "n_shared_leaves": n_classes,
        "dense_members": list(dense_members),
        "crop_size": size,
        "baseline_dim": int(a_src.shape[1]),
        "ensemble_dim": int(ens_src.shape[1]),
        "ks": list(ks),
        "seeds": list(seeds),
        "curve": agg.to_dicts(),
        "delta_ensemble_minus_baseline_by_k": deltas,
        "note": (
            "Dense members fed REAL Sentinel Hub crops (texture), not a tiled "
            "pixel; compare these deltas against ensemble_full_tl (pixel-tiled)."
        ),
    }
    logger.info(
        "texture_tl_done",
        source=source,
        target=target,
        deltas=deltas,
        n_shared_leaves=n_classes,
    )
    return TextureTLResult(per_k=per_k, summary=summary)


def save_outputs(result: TextureTLResult, out_dir: Path = _OUT_DIR) -> None:
    """Persist the few-shot curve and JSON summary to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_k.write_parquet(out_dir / "ensemble_texture_tl_per_k.parquet")
    (out_dir / "ensemble_texture_tl_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("texture_tl_saved", out_dir=str(out_dir))
