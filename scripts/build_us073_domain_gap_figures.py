"""Build the FR/ES domain-gap figures of the multi-region paper section (US-073).

Closes the two figures that were blocked (B-073-1, B-073-2) pending real ES data,
now that GEE is authenticated and the Sen4AgriNet subset (FR tile 31TCJ + ES tile
31TCG) is on disk:

- ``paper/figures/us-073/domain_gap_umap.{png,svg}`` -- the FR/ES UMAP of the
  64-dim AlphaEarth Satellite Embedding (v1.1) sampled at real Sen4AgriNet pixel
  centroids, projected with a SINGLE joint UMAP so the per-macro FR/ES separation
  IS the domain gap (B-073-1). Panel FR (PASTIS-R region) vs panel ES (Catalonia).
- ``paper/figures/us-073/ndvi_phenology_offset.{png,svg}`` -- per-macro Sentinel-2
  zonal-mean NDVI temporal curves for FR vs ES, evidencing the seasonal sowing /
  harvest offset (latitude + climate) that is the qualitative half of the gap
  (B-073-2). The dense transfer mIoU collapse is the quantitative half (table
  ``sen4agrinet_domain_gap.tex``).

Every value is REAL (GEE pulls cached under ``data/cache/gee/``); a class or AOI
with no S2 coverage is skipped and reported, never fabricated.

Attributions (also in the figure footers / captions):
- AlphaEarth: Brown/Khanna et al., "AlphaEarth Foundations",
  GEE ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` (v1.1), CC-BY-4.0.
- Sen4AgriNet (ES + FR labels/patches): Sykas et al. 2022, CC-BY-SA-4.0.
- PASTIS-R (FR source dataset of the dense model): Garnot & Landrieu, ICCV 2021.
- Sentinel-2 SR Harmonized: Copernicus / ESA.

Project conventions: Polars, structlog, type hints, English docstrings, Spanish
visible prose in figures, no emojis, never fabricate a missing value.

Usage::

    python -m scripts.build_us073_domain_gap_figures
    python -m scripts.build_us073_domain_gap_figures --dpi 200 --max-per-class 50
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")  # headless deterministic raster output

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
import typer

from ml.transfer import sen4agrinet_domain_gap as dg

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

FIGURES_DIR = Path("paper/figures/us-073")
DATA_OUT_DIR = Path("data/transfer")

#: Supported figure languages. The English variant is the canonical base file
#: (``<stem>.png``/``.svg``); Spanish is emitted with the ``_es`` suffix.
LANGS: tuple[str, ...] = ("en", "es")

#: File-stem suffix per language (English = no suffix = canonical base file).
_LANG_SUFFIX: dict[str, str] = {"en": "", "es": "_es"}

#: AlphaEarth attribution stamped on every UMAP figure footer (per language).
_ATTRIB: dict[str, str] = {
    "en": (
        "AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0) | "
        "Sen4AgriNet (Sykas et al. 2022, CC-BY-SA-4.0) | PASTIS-R (Garnot & Landrieu 2021)"
    ),
    "es": (
        "AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0) | "
        "Sen4AgriNet (Sykas et al. 2022, CC-BY-SA-4.0) | PASTIS-R (Garnot & Landrieu 2021)"
    ),
}

#: Sentinel-2 NDVI attribution stamped on the phenology figure footer (per language).
_NDVI_ATTRIB: dict[str, str] = {
    "en": (
        "Sentinel-2 SR Harmonized (Copernicus/ESA), QA60 cloud mask | "
        "Sen4AgriNet (CC-BY-SA-4.0). Real 2019 series, no fabricated values."
    ),
    "es": (
        "Sentinel-2 SR Harmonized (Copernicus/ESA), mascara de nubes QA60 | "
        "Sen4AgriNet (CC-BY-SA-4.0). Series reales 2019, sin valores fabricados."
    ),
}

#: Human macro labels per language (for the figure legends).
MACRO_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "cereals": "Cereals",
        "legumes_fodder": "Legumes/fodder",
        "oilseed_industrial": "Oilseed/industrial",
        "potato": "Potato",
        "vineyard": "Vineyard",
        "grassland": "Grassland",
        "sugar_beet": "Sugar beet",
        "soybean": "Soybean",
        "orchard": "Orchard",
        "vegetables": "Vegetables",
    },
    "es": {
        "cereals": "Cereales",
        "legumes_fodder": "Leguminosas/forraje",
        "oilseed_industrial": "Oleaginosas/industrial",
        "potato": "Patata",
        "vineyard": "Vinedo",
        "grassland": "Pradera",
        "sugar_beet": "Remolacha",
        "soybean": "Soja",
        "orchard": "Frutales",
        "vegetables": "Hortalizas",
    },
}

#: All visible chart text (titles, axes, panel headings) per language.
_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "umap_panel_fr": "France (PASTIS-R, tile 31TCJ, lat ~44N)",
        "umap_panel_es": "Catalonia (Sen4AgriNet, tile 31TCG, lat ~41.7N)",
        "umap_xlabel": "UMAP-1",
        "umap_ylabel": "UMAP-2",
        "umap_suptitle": ("Domain gap France<->Catalonia: AlphaEarth (joint UMAP, 64-dim)"),
        "ndvi_ylabel": "Zonal-mean NDVI",
        "ndvi_xlabel": "Day of year (DOY)",
        "ndvi_suptitle": (
            "Phenological offset France vs Catalonia: Sentinel-2 NDVI per macro-class (2019)"
        ),
    },
    "es": {
        "umap_panel_fr": "Francia (PASTIS-R, tile 31TCJ, lat ~44N)",
        "umap_panel_es": "Cataluna (Sen4AgriNet, tile 31TCG, lat ~41.7N)",
        "umap_xlabel": "UMAP-1",
        "umap_ylabel": "UMAP-2",
        "umap_suptitle": (
            "Brecha de dominio Francia<->Cataluna: AlphaEarth (UMAP conjunto, 64-dim)"
        ),
        "ndvi_ylabel": "NDVI medio zonal",
        "ndvi_xlabel": "Dia del anio (DOY)",
        "ndvi_suptitle": (
            "Desfase fenologico Francia vs Cataluna: NDVI Sentinel-2 por macro-clase (2019)"
        ),
    },
}


def _set_style() -> None:
    """Apply the scientific paper rcParams + fixed seed (CVPR/ISPRS, 300 DPI)."""
    np.random.seed(17)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def _save(fig: plt.Figure, stem: str, *, lang: str, out_dir: Path, dpi: int) -> dict[str, Path]:
    """Save a figure as PNG + SVG (language-suffixed) and close it.

    The English variant is the canonical base file (``<stem>.png``); every other
    language appends its suffix (e.g. ``<stem>_es.png``).

    Args:
        fig: Figure to export.
        stem: Language-neutral file stem (no extension, no suffix).
        lang: Figure language (``"en"`` base, ``"es"`` -> ``_es`` suffix).
        out_dir: Destination directory (created if missing).
        dpi: Raster resolution.

    Returns:
        Mapping ``{"png": path, "svg": path}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_lang = f"{stem}{_LANG_SUFFIX[lang]}"
    png = out_dir / f"{stem_lang}.png"
    svg = out_dir / f"{stem_lang}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure_saved", stem=stem_lang, lang=lang, png=str(png), svg=str(svg))
    return {"png": png, "svg": svg}


def _render_umap_figure(
    joint: pl.DataFrame, *, lang: str, out_dir: Path, dpi: int
) -> dict[str, Path]:
    """Render the two-panel joint-UMAP gap figure in one language.

    Args:
        joint: Joint FR+ES UMAP frame (columns ``region``, ``macro``, ``x``, ``y``).
        lang: Figure language (``"en"`` / ``"es"``).
        out_dir: Destination directory.
        dpi: Raster resolution.

    Returns:
        Mapping ``{"png": path, "svg": path}``.
    """
    txt = _STRINGS[lang]
    labels_map = MACRO_LABELS[lang]
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    macros = sorted(joint.get_column("macro").unique().to_list())
    for ax, region, title in (
        (axes[0], "FR", txt["umap_panel_fr"]),
        (axes[1], "ES", txt["umap_panel_es"]),
    ):
        sub = joint.filter(pl.col("region") == region)
        for macro in macros:
            pts = sub.filter(pl.col("macro") == macro)
            if pts.is_empty():
                continue
            ax.scatter(
                pts.get_column("x").to_numpy(),
                pts.get_column("y").to_numpy(),
                s=8,
                alpha=0.55,
                color=dg.MACRO_COLORS.get(macro, "#777777"),
                label=labels_map.get(macro, macro),
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel(txt["umap_xlabel"])
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel(txt["umap_ylabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.04),
        fontsize=7,
    )
    fig.suptitle(txt["umap_suptitle"], fontsize=11)
    fig.text(0.5, -0.10, _ATTRIB[lang], ha="center", fontsize=6, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, "domain_gap_umap", lang=lang, out_dir=out_dir, dpi=dpi)


def build_umap_figure(*, out_dir: Path, dpi: int, max_per_class: int) -> dict[str, dict[str, Path]]:
    """Materialize FR+ES AlphaEarth and render the joint-UMAP gap figure per language.

    The data is materialized once and rendered in every :data:`LANGS` language; the
    English variant is the canonical base file, Spanish gets the ``_es`` suffix.

    Args:
        out_dir: Destination directory.
        dpi: Raster resolution.
        max_per_class: Per-class centroid cap per patch.

    Returns:
        Mapping ``lang -> {"png": path, "svg": path}``.
    """
    es = dg.build_region_alphaearth(
        region="ES",
        patch_glob=dg.ES_PATCH_GLOB,
        out_parquet=DATA_OUT_DIR / "sen4agrinet_es_alphaearth.parquet",
        max_per_class=max_per_class,
    )
    fr = dg.build_region_alphaearth(
        region="FR",
        patch_glob=dg.FR_PATCH_GLOB,
        out_parquet=DATA_OUT_DIR / "sen4agrinet_fr_alphaearth.parquet",
        max_per_class=max_per_class,
    )
    joint = dg.compute_joint_umap(fr, es)
    return {lang: _render_umap_figure(joint, lang=lang, out_dir=out_dir, dpi=dpi) for lang in LANGS}


def _gather_region_ndvi(
    region: str, patch_glob: str, macros: list[str], year: int, *, max_per_class: int
) -> dict[str, pl.DataFrame]:
    """Pull the per-macro S2 NDVI series of one region from real GEE.

    Args:
        region: Region tag (``"ES"`` / ``"FR"``).
        patch_glob: Glob for that region's Sen4AgriNet patches.
        macros: Macro-classes to extract.
        year: Year of the NDVI series.
        max_per_class: Per-class centroid cap (for the AOI median centre).

    Returns:
        Mapping ``macro -> NDVI frame`` (only non-empty series are included).
    """
    points = dg.collect_centroids(patch_glob=patch_glob, region=region, max_per_class=max_per_class)
    out: dict[str, pl.DataFrame] = {}
    for macro in macros:
        bbox = dg.macro_aoi_bbox(points, macro)
        if bbox is None:
            logger.warning("ndvi_macro_absent", region=region, macro=macro)
            continue
        series = dg.extract_bbox_ndvi_series(bbox, year, cache_key=f"{region.lower()}_{macro}")
        if series.is_empty():
            logger.warning("ndvi_series_empty", region=region, macro=macro)
            continue
        out[macro] = series
    return out


def _render_ndvi_figure(
    present: list[str],
    fr_ndvi: dict[str, pl.DataFrame],
    es_ndvi: dict[str, pl.DataFrame],
    *,
    lang: str,
    out_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    """Render the FR vs ES per-macro NDVI phenology figure in one language.

    Args:
        present: Macro-classes with a paired FR+ES series, in plot order.
        fr_ndvi: Mapping ``macro -> FR NDVI frame`` (columns ``doy``, ``ndvi``).
        es_ndvi: Mapping ``macro -> ES NDVI frame`` (columns ``doy``, ``ndvi``).
        lang: Figure language (``"en"`` / ``"es"``).
        out_dir: Destination directory.
        dpi: Raster resolution.

    Returns:
        Mapping ``{"png": path, "svg": path}``.
    """
    txt = _STRINGS[lang]
    labels_map = MACRO_LABELS[lang]
    _set_style()
    n = len(present)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, 3.0 * nrows), sharex=True, squeeze=False)
    flat = axes.ravel()
    for i, macro in enumerate(present):
        ax = flat[i]
        for region, ndvi, color, ls in (
            ("FR", fr_ndvi[macro], "#2c6fbb", "-"),
            ("ES", es_ndvi[macro], "#e08214", "--"),
        ):
            sub = ndvi.sort("doy")
            ax.plot(
                sub.get_column("doy").to_numpy(),
                sub.get_column("ndvi").to_numpy(),
                ls,
                color=color,
                linewidth=1.4,
                marker=".",
                markersize=3,
                alpha=0.85,
                label=f"{region} (lat {'~44N' if region == 'FR' else '~41.7N'})",
            )
        ax.set_title(labels_map.get(macro, macro))
        ax.set_ylabel(txt["ndvi_ylabel"])
        ax.set_ylim(-0.1, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower center", fontsize=7)
    for j in range(len(present), len(flat)):
        flat[j].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(txt["ndvi_xlabel"])
    fig.suptitle(txt["ndvi_suptitle"], fontsize=11)
    fig.text(0.5, -0.02, _NDVI_ATTRIB[lang], ha="center", fontsize=6, color="0.4")
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    return _save(fig, "ndvi_phenology_offset", lang=lang, out_dir=out_dir, dpi=dpi)


def build_ndvi_figure(
    *, out_dir: Path, dpi: int, max_per_class: int, year: int = dg.ALPHAEARTH_YEAR
) -> dict[str, dict[str, Path]] | None:
    """Render the FR vs ES per-macro NDVI phenological-offset figure per language.

    The real GEE series are gathered once and rendered in every :data:`LANGS`
    language; English is the canonical base file, Spanish gets the ``_es`` suffix.

    Args:
        out_dir: Destination directory.
        dpi: Raster resolution.
        max_per_class: Per-class centroid cap (AOI centre).
        year: Year of the NDVI series.

    Returns:
        Mapping ``lang -> {"png": path, "svg": path}``, or ``None`` if no real
        series was retrieved for any class (degraded, never fabricated).
    """
    macros = ["cereals", "oilseed_industrial", "vineyard", "legumes_fodder"]
    fr_ndvi = _gather_region_ndvi("FR", dg.FR_PATCH_GLOB, macros, year, max_per_class=max_per_class)
    es_ndvi = _gather_region_ndvi("ES", dg.ES_PATCH_GLOB, macros, year, max_per_class=max_per_class)
    present = [m for m in macros if m in fr_ndvi and m in es_ndvi]
    if not present:
        logger.warning("ndvi_figure_no_paired_series")
        return None

    return {
        lang: _render_ndvi_figure(present, fr_ndvi, es_ndvi, lang=lang, out_dir=out_dir, dpi=dpi)
        for lang in LANGS
    }


def build_all(*, out_dir: Path = FIGURES_DIR, dpi: int = 200, max_per_class: int = 50) -> None:
    """Build both US-073 domain-gap figures from real GEE data.

    Args:
        out_dir: Destination directory.
        dpi: Raster resolution.
        max_per_class: Per-class centroid cap per patch.
    """
    umap_paths = build_umap_figure(out_dir=out_dir, dpi=dpi, max_per_class=max_per_class)
    ndvi_paths = build_ndvi_figure(out_dir=out_dir, dpi=dpi, max_per_class=max_per_class)
    logger.info(
        "us073_domain_gap_done",
        umap_en=str(umap_paths["en"]["png"]),
        umap_es=str(umap_paths["es"]["png"]),
        ndvi_en=str(ndvi_paths["en"]["png"]) if ndvi_paths else "PENDING (no paired S2 series)",
        ndvi_es=str(ndvi_paths["es"]["png"]) if ndvi_paths else "PENDING (no paired S2 series)",
    )


@app.command()
def run(
    out_dir: Annotated[Path, typer.Option("--out-dir")] = FIGURES_DIR,
    dpi: Annotated[int, typer.Option("--dpi")] = 200,
    max_per_class: Annotated[int, typer.Option("--max-per-class")] = 50,
) -> None:
    """CLI entry point: build both US-073 domain-gap figures (see :func:`build_all`).

    Args:
        out_dir: Destination directory.
        dpi: Raster resolution.
        max_per_class: Per-class centroid cap per patch.
    """
    build_all(out_dir=out_dir, dpi=dpi, max_per_class=max_per_class)


if __name__ == "__main__":
    app()
