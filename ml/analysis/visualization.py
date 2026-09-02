"""Visualization utilities for Sentinel-2 EDA.

- `stretch_2_98`: 2-98 percentile stretch per band for RGB visualization.
- `folium_rois`: folium map with the 4 ROIs from `config/rois.yaml`.
- `plot_band_grid`: grid of histograms per band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def stretch_2_98(arr: np.ndarray) -> np.ndarray:
    """Apply a 2-98 percentile stretch band by band.

    Accepts 2D arrays `(H, W)` or 3D `(C, H, W)` / `(H, W, C)`.

    Args:
        arr: Numeric array to stretch.

    Returns:
        `float32` array with values in [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [2.0, 98.0])
        out = (arr - lo) / max(hi - lo, 1e-6)
        stretched: np.ndarray = np.clip(out, 0.0, 1.0)
        return stretched
    if arr.ndim == 3:
        # Detect layout (C, H, W) vs (H, W, C) by the axis size
        if arr.shape[0] <= 12 and arr.shape[0] < arr.shape[-1]:
            channels = arr.shape[0]
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(channels):
                lo, hi = np.percentile(arr[c], [2.0, 98.0])
                out[c] = np.clip((arr[c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return out
        else:
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(arr.shape[-1]):
                lo, hi = np.percentile(arr[..., c], [2.0, 98.0])
                out[..., c] = np.clip((arr[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return out
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def folium_rois(rois_yaml: Path, output_path: Path) -> Any:
    """Generate a folium map with all ROIs declared in the YAML.

    Args:
        rois_yaml: Path to `config/rois.yaml`.
        output_path: Path to save the final HTML.

    Returns:
        `folium.Map` object already saved to disk.
    """
    import folium  # type: ignore[import-untyped]
    import yaml

    with rois_yaml.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    rois = cfg.get("rois", [])
    if not rois:
        m = folium.Map(location=[42.0, 10.0], zoom_start=5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        return m

    # Center on approximate global bbox
    lats: list[float] = []
    lons: list[float] = []
    for r in rois:
        bbox = r.get("bbox", [])
        if len(bbox) == 4:
            lons.extend([bbox[0], bbox[2]])
            lats.extend([bbox[1], bbox[3]])
    if lats and lons:
        center = [float(np.mean(lats)), float(np.mean(lons))]
    else:
        center = [42.0, 10.0]

    m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, roi in enumerate(rois):
        bbox = roi.get("bbox")
        name = roi.get("name", f"roi_{i}")
        color = palette[i % len(palette)]
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.15,
                popup=folium.Popup(
                    f"<b>{name}</b><br>region: {roi.get('region', 'n/a')}<br>"
                    f"crops: {', '.join(roi.get('crops', []))}",
                    max_width=300,
                ),
            ).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return m


def plot_band_grid(
    df: pl.DataFrame,
    output_path: Path,
    dpi: int = 200,
    band_col: str = "band",
    value_col: str = "value",
    bands: list[str] | None = None,
) -> None:
    """Generate a 2x5 grid of histograms, one per Sentinel-2 band.

    Args:
        df: Long-format DataFrame with columns `band_col, value_col`.
        output_path: Path to the output PNG.
        dpi: Resolution (default 200).
        band_col: Band column name.
        value_col: Value column name.
        bands: Band subset (default 10 PASTIS).
    """
    import matplotlib.pyplot as plt

    if bands is None:
        bands = [
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

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=dpi)
    axes = axes.flatten()
    for i, band in enumerate(bands):
        ax = axes[i]
        vals = (
            df.filter(pl.col(band_col) == band)
            .select(value_col)
            .to_series()
            .drop_nulls()
            .to_numpy()
        )
        if vals.size:
            ax.hist(vals, bins=80, color="#1f77b4", alpha=0.85)
            ax.set_title(band, fontsize=11)
            ax.set_yscale("log")
        else:
            ax.set_visible(False)
    for j in range(len(bands), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribuciones por banda Sentinel-2 (log y)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _categorical_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Cyclic tab20 palette with `n` RGBA colors."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def tsne_scatter(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """2D scatter colored by class with grouped legend.

    Args:
        emb_2d: Array shape `(n, 2)` with t-SNE coords.
        labels: Array shape `(n,)` with categorical labels.
        title: Plot title.
        out_path: Output PNG path.
        dpi: Resolution.

    Returns:
        Matplotlib figure (already saved to disk if out_path is provided).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    if emb_2d.size == 0 or len(labels) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
    else:
        uniq = list(dict.fromkeys(labels.tolist()))
        colors = _categorical_palette(len(uniq))
        for cls, color in zip(uniq, colors, strict=True):
            mask = labels == cls
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=4,
                alpha=0.6,
                c=[color],
                label=str(cls),
            )
        ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def umap_scatter(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """2D UMAP scatter colored by class.

    Same parameters as `tsne_scatter`. Identical style for comparability.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    if emb_2d.size == 0 or len(labels) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
    else:
        uniq = list(dict.fromkeys(labels.tolist()))
        colors = _categorical_palette(len(uniq))
        for cls, color in zip(uniq, colors, strict=True):
            mask = labels == cls
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=4,
                alpha=0.6,
                c=[color],
                label=str(cls),
            )
        ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def correlation_heatmap(
    corr_df: pl.DataFrame,
    out_path: Path,
    threshold: float = 0.7,
    dpi: int = 200,
) -> Any:
    """Heatmap of the 64x64 correlation matrix.

    Accepts the long-format output of `correlation_matrix()` and rebuilds the
    symmetric matrix to plot.

    Args:
        corr_df: Long DataFrame with columns `dim_i, dim_j, pearson|spearman`.
        out_path: PNG path.
        threshold: Reference line (annotates cells with |corr| > threshold).
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10), dpi=dpi)
    if corr_df.is_empty():
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    method = "pearson" if "pearson" in corr_df.columns else "spearman"
    dims = sorted(set(corr_df["dim_i"].to_list()) | set(corr_df["dim_j"].to_list()))
    idx = {d: i for i, d in enumerate(dims)}
    n = len(dims)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for row in corr_df.iter_rows(named=True):
        i = idx[row["dim_i"]]
        j = idx[row["dim_j"]]
        val = row[method]
        if val is None:
            continue
        mat[i, j] = float(val)
        mat[j, i] = float(val)

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(0, n, 4))
    ax.set_xticklabels([dims[i] for i in range(0, n, 4)], rotation=90, fontsize=7)
    ax.set_yticks(range(0, n, 4))
    ax.set_yticklabels([dims[i] for i in range(0, n, 4)], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label=method)
    redundant = int(((np.abs(mat) > threshold) & (np.abs(mat) < 0.999)).sum() // 2)
    ax.set_title(f"Matriz correlacion 64x64 ({method}) — {redundant} pares |r|>{threshold}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def qq_grid(
    stats_df: pl.DataFrame,
    out_path: Path,
    ncols: int = 8,
    dpi: int = 200,
) -> Any:
    """Grid of approximate QQ plots per dimension.

    Since the exact QQ requires the raw values, this grid plots a comparison
    normal curve `(z_score, value_aprox)` derived from the statistics
    (mean, std). For exact visualization use `scipy.stats.probplot` in the
    notebook with the raw arrays.

    Args:
        stats_df: Output of `qq_test_dims()` with columns `dim, mean, std,
            skewness, kurtosis, shapiro_pvalue`.
        out_path: PNG path.
        ncols: Grid columns (default 8 -> 64 dims = 8x8).
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    n = stats_df.height
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 1.8 * nrows), dpi=dpi)
    axes_flat = np.asarray(axes).flatten()
    z = np.linspace(-3, 3, 50)
    for i, row in enumerate(stats_df.iter_rows(named=True)):
        ax = axes_flat[i]
        mean = row.get("mean", 0.0) or 0.0
        std = row.get("std", 1.0) or 1.0
        pval = row.get("shapiro_pvalue", float("nan"))
        ax.plot(z, z * std + mean, lw=0.8, color="#1f77b4")
        ax.plot(z, z, lw=0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{row['dim']} p={pval:.2g}", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("QQ plots aproximados vs N(0,1) por dimensión AlphaEarth", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def pairplot_by_class(
    df: pl.DataFrame,
    features: list[str],
    class_col: str,
    out_path: Path,
    subsample_per_class: int = 2000,
    seed: int = 42,
    top_classes: int = 10,
    dpi: int = 200,
) -> Any:
    """Seaborn pairplot conditioned by class with class cap and subsampling.

    Caps classes to `top_classes` by frequency to avoid saturating seaborn.
    Applies `.to_pandas()` as an edge adapter solely for `seaborn.pairplot`,
    which does not accept Polars.

    Args:
        df: Polars DataFrame with `features` and `class_col`.
        features: Columns to use as pairplot axes.
        class_col: Categorical column for `hue`.
        out_path: Output PNG path.
        subsample_per_class: Maximum rows per class to plot.
        seed: Sampling seed.
        top_classes: Maximum number of classes (the most frequent) to keep.
        dpi: PNG resolution (default 200).

    Returns:
        `seaborn.PairGrid` object already saved to disk.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in [*features, class_col] if c not in df.columns]
    if df.is_empty() or missing:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos para pairplot", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    freq = (
        df.group_by(class_col)
        .agg(pl.len().alias("__count"))
        .sort("__count", descending=True)
        .head(top_classes)
    )
    keep_classes = freq[class_col].to_list()
    sub = df.filter(pl.col(class_col).is_in(keep_classes))

    frames: list[pl.DataFrame] = []
    for cls in keep_classes:
        df_c = sub.filter(pl.col(class_col) == cls)
        if df_c.is_empty():
            continue
        if df_c.height > subsample_per_class:
            df_c = df_c.sample(n=subsample_per_class, seed=seed, shuffle=True)
        frames.append(df_c)
    if not frames:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos para pairplot", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    sub_sampled = pl.concat(frames, how="vertical_relaxed")
    pdf = sub_sampled.select([*features, class_col]).drop_nulls().to_pandas()
    if pdf.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos para pairplot", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    import seaborn as sns

    grid = sns.pairplot(
        pdf,
        vars=features,
        hue=class_col,
        corner=True,
        plot_kws={"s": 6, "alpha": 0.5, "edgecolor": "none"},
        diag_kind="hist",
    )
    grid.fig.suptitle(f"Pairplot por {class_col} (top-{len(keep_classes)})", y=1.02)
    grid.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(grid.fig)
    return grid


def vif_barplot(
    vif_df: pl.DataFrame,
    out_path: Path,
    threshold_warn: float = 5.0,
    threshold_drop: float = 10.0,
    dpi: int = 200,
) -> Any:
    """Horizontal VIF barplot with threshold lines at 5 and 10.

    Args:
        vif_df: Output of `correlations.vif_table()` with columns
            `feature, vif, status`.
        out_path: PNG path.
        threshold_warn: Yellow warning line (default 5).
        threshold_drop: Red critical line (default 10).
        dpi: Resolution.

    Returns:
        Matplotlib figure (closed after savefig).
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    if vif_df.is_empty():
        ax.text(0.5, 0.5, "Sin datos VIF", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    # Replace inf with a large sentinel for plotting
    sub = vif_df.with_columns(
        pl.when(pl.col("vif").is_infinite()).then(1e3).otherwise(pl.col("vif")).alias("vif_plot")
    ).sort("vif_plot", descending=False)
    feats = sub["feature"].to_list()
    vals = sub["vif_plot"].to_numpy()
    statuses = sub["status"].to_list()
    palette = {
        "ok": "#2ca02c",
        "warning": "#ff7f0e",
        "drop": "#d62728",
        "dropped_near_perfect_corr": "#7f7f7f",
    }
    colors = [palette.get(s, "#1f77b4") for s in statuses]
    ax.barh(feats, vals, color=colors, edgecolor="black", linewidth=0.4)
    warn_label = f"warn={threshold_warn}"
    drop_label = f"drop={threshold_drop}"
    ax.axvline(threshold_warn, color="orange", linestyle="--", linewidth=1.0, label=warn_label)
    ax.axvline(threshold_drop, color="red", linestyle="--", linewidth=1.0, label=drop_label)
    ax.set_xscale("symlog")
    ax.set_xlabel("VIF (escala simlog)")
    ax.set_title("Variance Inflation Factor por feature")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def dual_axis_precip_ndvi(
    df_era5: pl.DataFrame,
    df_ndvi: pl.DataFrame,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """Dual-axis plot: ERA5 precip bars + annual max NDVI line per ROI.

    Args:
        df_era5: Polars DataFrame with `year, roi_name, precip_mm`.
        df_ndvi: Polars DataFrame with `year, roi_name, ndvi_max`.
        out_path: PNG path.
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    if df_era5.is_empty() or df_ndvi.is_empty():
        ax.text(0.5, 0.5, "Sin datos ERA5 / NDVI", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    merged = df_era5.join(df_ndvi, on=["year", "roi_name"], how="inner").sort(["roi_name", "year"])
    if merged.is_empty():
        ax.text(0.5, 0.5, "ERA5 y NDVI no se cruzan", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    rois = merged["roi_name"].unique().to_list()
    palette = _categorical_palette(len(rois))
    ax2 = ax.twinx()
    width = 0.8 / max(1, len(rois))
    years = sorted(merged["year"].unique().to_list())
    xpos = np.arange(len(years))
    for i, roi in enumerate(rois):
        sub = merged.filter(pl.col("roi_name") == roi).sort("year")
        precip = sub["precip_mm"].to_numpy()
        ndvi = sub["ndvi_max"].to_numpy()
        sub_years = sub["year"].to_list()
        offset = (i - (len(rois) - 1) / 2.0) * width
        bar_x = np.array([years.index(y) for y in sub_years]) + offset
        ax.bar(
            bar_x,
            precip,
            width=width,
            color=palette[i],
            alpha=0.65,
            edgecolor="black",
            linewidth=0.3,
            label=f"{roi} precip",
        )
        line_x = np.array([years.index(y) for y in sub_years])
        ax2.plot(line_x, ndvi, marker="o", color=palette[i], linewidth=1.5, label=f"{roi} NDVI")

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitacion anual (mm)")
    ax2.set_ylabel("NDVI maximo anual")
    ax.set_title("ERA5 precip anual vs NDVI maximo por ROI")
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def acf_grid_by_class(
    acf_df: pl.DataFrame,
    out_path: Path,
    ncols: int = 3,
    dpi: int = 200,
) -> Any:
    """Grid of ACF plots aggregated by class (mean + 95% CI).

    Args:
        acf_df: Output of `correlations.acf_pacf_per_parcel()` with columns
            `parcel_id, class_name, lag, acf, pacf`.
        out_path: PNG path.
        ncols: Grid columns (default 3).
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if acf_df.is_empty():
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos ACF", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    agg = (
        acf_df.group_by(["class_name", "lag"])
        .agg(
            [
                pl.col("acf").mean().alias("acf_mean"),
                pl.col("acf").std().fill_null(0.0).alias("acf_std"),
                pl.len().alias("n"),
            ]
        )
        .sort(["class_name", "lag"])
    )
    classes = sorted(agg["class_name"].unique().to_list())
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), dpi=dpi)
    axes_flat = np.asarray(axes).flatten() if nrows * ncols > 1 else np.array([axes])
    for i, cls in enumerate(classes):
        ax = axes_flat[i]
        sub = agg.filter(pl.col("class_name") == cls).sort("lag")
        lags = sub["lag"].to_numpy()
        mean = sub["acf_mean"].to_numpy()
        std = sub["acf_std"].to_numpy()
        n = sub["n"].to_numpy()
        # Approximate 95% CI with normal: mean +/- 1.96 * std / sqrt(n)
        denom = np.sqrt(np.maximum(n, 1))
        ci = 1.96 * std / denom
        ax.bar(lags, mean, color="#1f77b4", alpha=0.85)
        ax.errorbar(lags, mean, yerr=ci, fmt="none", ecolor="black", elinewidth=0.6, capsize=2)
        ax.axhline(0.0, color="black", linewidth=0.4)
        ax.set_title(cls, fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(axis="both", labelsize=6)
    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("ACF agregado por clase (media + IC95%)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def dtw_centroids_plot(
    km_model: Any,
    df_ts: pl.DataFrame,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """Plot of the DTW centroids of the fitted `TimeSeriesKMeans`.

    Args:
        km_model: Fitted `tslearn.clustering.TimeSeriesKMeans` model with
            `cluster_centers_` accessible. If None, a placeholder is drawn.
        df_ts: Polars DataFrame with `parcel_id, class_name, cluster_id` to
            annotate the size of each cluster.
        out_path: PNG path.
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if km_model is None or not hasattr(km_model, "cluster_centers_"):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin modelo DTW", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    centers = np.asarray(km_model.cluster_centers_)
    n_clusters = centers.shape[0]
    counts: dict[int, int] = {}
    if not df_ts.is_empty() and "cluster_id" in df_ts.columns:
        cnt_df = df_ts.group_by("cluster_id").agg(pl.len().alias("n"))
        for r in cnt_df.iter_rows(named=True):
            counts[int(r["cluster_id"])] = int(r["n"])
    nrows = int(np.ceil(n_clusters / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 2.4 * nrows), dpi=dpi)
    axes_flat = np.asarray(axes).flatten() if n_clusters > 1 else np.array([axes])
    palette = _categorical_palette(n_clusters)
    for k in range(n_clusters):
        ax = axes_flat[k]
        center = centers[k].squeeze()
        ax.plot(center, color=palette[k], linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.4)
        ax.set_title(f"Cluster {k} (n={counts.get(k, 0)})", fontsize=10)
        ax.set_xlabel("Mes (resampleo)")
        ax.set_ylabel("NDVI z-score")
        ax.tick_params(axis="both", labelsize=7)
    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Centroides DTW por cluster (NDVI z-normalizado)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def cross_region_scatter(
    consistency_df: pl.DataFrame,
    out_path: Path,
    annotate_top_k: int = 10,
    dpi: int = 200,
) -> Any:
    """Scatter `importance_italia` vs `importance_francia` with identity line.

    Args:
        consistency_df: Output of `cross_region_consistency()`.
        out_path: PNG path.
        annotate_top_k: How many top dims to label with their name.
        dpi: Resolution.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    if consistency_df.is_empty():
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    x = consistency_df["importance_italia"].to_numpy()
    y = consistency_df["importance_francia"].to_numpy()
    dims = consistency_df["dim"].to_list()
    consistent = consistency_df["consistente_top10"].to_list()

    colors = ["#d62728" if c else "#1f77b4" for c in consistent]
    ax.scatter(x, y, c=colors, s=40, alpha=0.75, edgecolors="black", linewidths=0.3)
    max_val = float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0]))
    ax.plot([0, max_val], [0, max_val], "k--", lw=0.8, alpha=0.6, label="Identidad")

    # Annotate top-K by sum of importance
    sorted_idx = np.argsort(-(x + y))[:annotate_top_k]
    for i in sorted_idx:
        ax.annotate(
            dims[i],
            (x[i], y[i]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Importance Italia (Dynamic World)")
    ax.set_ylabel("Importance Francia (PASTIS-R)")
    ax.set_title("Consistencia cross-region de top dimensiones AlphaEarth")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig
