"""Build the Italy AOI map figure over AlphaEarth for the paper (US-070, B-070-1).

Closes B-070-1 now that GEE is authenticated. Downloads a real agricultural AOI
of Italy (Po valley / Pianura Padana) on the AlphaEarth Satellite Embedding V1
Annual (v1.1) collection, exports the AOI footprint + sampled pixels to GeoJSON,
and renders it two ways:

- ``paper/figures/us-070/aoi_italy.{png,svg}`` -- a static map (English, canonical
  for the English paper): the AlphaEarth embedding visualised as a false-colour
  PCA-RGB scatter of the real sampled pixels over an OpenStreetMap basemap tile
  (via :mod:`xyzservices`, the same tile provider :mod:`contextily` uses; contextily
  itself is not a project dependency).
- ``paper/figures/us-070/aoi_italy_es.{png,svg}`` -- the same static map with every
  visible string translated to Spanish, for the Spanish paper.
- ``paper/figures/us-070/aoi_italy.html`` / ``aoi_italy_es.html`` -- interactive
  :mod:`folium` maps (English base, Spanish ``_es``) with the AOI rectangle + a heat
  of the sampled pixels (the project's canonical web map stack).
- ``data/aoi/italy_aois.geojson`` -- the AOI footprint as a real GeoJSON
  (EPSG:4326), the artefact the notebook B-070-1 cell consumes.

The AlphaEarth pull is cached under ``data/cache/gee/``. If the cache is present it
is reused; nothing is fabricated -- an empty GEE response aborts with an explicit
error rather than drawing a fake AOI.

Attributions (also in the figure footer / GeoJSON properties):
- AlphaEarth: Brown/Khanna et al., "AlphaEarth Foundations",
  GEE ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` (v1.1), CC-BY-4.0.
- Basemap tiles: OpenStreetMap contributors (ODbL).

Every figure is emitted in two languages: the English version is the canonical
base file (``aoi_italy.*``) for the English paper, and the Spanish version carries
the ``_es`` suffix (``aoi_italy_es.*``) for the Spanish paper. Only the visible
strings differ between the two; the plotted data and geometry are identical.

Project conventions: Polars, structlog, type hints, English docstrings, no emojis,
never fabricate a missing value.

Usage::

    python -m scripts.build_us070_italy_aoi
    python -m scripts.build_us070_italy_aoi --n-pixels 4000 --dpi 200
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")  # headless deterministic raster output

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
import typer

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

FIGURES_DIR = Path("paper/figures/us-070")
AOI_GEOJSON = Path("data/aoi/italy_aois.geojson")

#: Real agricultural AOI: a focused Po valley (Pianura Padana) rectangle in
#: EPSG:4326, the most intensive Italian crop region. Matches the already-cached
#: ``alphaearth_pianura_padana_2024`` extent (lon 10-11, lat 45-45.5).
PO_VALLEY_BBOX: dict[str, float] = {
    "min_lon": 10.0,
    "min_lat": 45.0,
    "max_lon": 11.0,
    "max_lat": 45.5,
}
AOI_NAME = "pianura_padana_po_valley"
AOI_YEAR = 2024
ALPHAEARTH_DIM_COLS: list[str] = [f"dim_{i:02d}" for i in range(64)]

#: Languages emitted by the builder. English is canonical (base filename); Spanish
#: carries the ``_es`` suffix.
Lang = Literal["en", "es"]
LANGS: tuple[Lang, ...] = ("en", "es")

#: Every visible string of the figure, per language. English is natural scientific
#: English; Spanish uses correct accents and enie. ``{n}`` is the real pixel count.
STRINGS: dict[Lang, dict[str, str]] = {
    "en": {
        "title_line1": (
            "Italian agricultural AOI (Pianura Padana, Po valley) over AlphaEarth v1.1"
        ),
        "title_line2": ("{n} real 2024 AlphaEarth pixels, coloured by their 64-dim embedding"),
        "attrib": (
            "AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0), real "
            "2024 pixels | basemap (c) OpenStreetMap contributors (ODbL)"
        ),
        "basemap_on": "OSM basemap",
        "basemap_off": "no basemap (tiles unavailable)",
        "folium_popup": "Pianura Padana AOI ({n} px AlphaEarth v1.1, 2024)",
    },
    "es": {
        "title_line1": ("AOI agricola Italia (Pianura Padana, valle del Po) sobre AlphaEarth v1.1"),
        "title_line2": (
            "{n} pixeles reales 2024 de AlphaEarth, coloreados por su embedding 64-dim"
        ),
        "attrib": (
            "AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL v1.1 (CC-BY-4.0), pixeles "
            "reales 2024 | basemap (c) OpenStreetMap contributors (ODbL)"
        ),
        "basemap_on": "basemap OSM",
        "basemap_off": "sin basemap (tiles no disponibles)",
        "folium_popup": "AOI Pianura Padana ({n} px AlphaEarth v1.1, 2024)",
    },
}


def _stem(base: str, lang: Lang) -> str:
    """Return the language-suffixed filename stem (English base, ``_es`` for Spanish).

    Args:
        base: Language-neutral stem (e.g. ``"aoi_italy"``).
        lang: Target language.

    Returns:
        ``base`` for English, ``f"{base}_es"`` for Spanish.
    """
    return base if lang == "en" else f"{base}_es"


def _load_alphaearth_aoi(*, n_pixels: int) -> pl.DataFrame:
    """Return the real AlphaEarth pixels over the Po-valley AOI (cached or GEE).

    Prefers the already-versioned ``alphaearth_pianura_padana_2024`` cache; if it
    is missing, pulls the AOI live via
    :func:`ml.ingest.gee_sampler.sample_alphaearth_roi`.

    Args:
        n_pixels: Number of pixels to request from GEE on a cache miss.

    Returns:
        Frame ``(px_id, lon, lat, roi, year, dim_00..dim_63)`` with real values.

    Raises:
        RuntimeError: if neither the cache nor GEE yields any pixel (never faked).
    """
    cached = Path("data/cache/gee/alphaearth_pianura_padana_2024_2000.parquet")
    if cached.exists():
        df = pl.read_parquet(cached)
        logger.info("aoi_alphaearth_cache_hit", path=str(cached), n=df.height)
        return df

    from ml.ingest.gee_sampler import init_ee, sample_alphaearth_roi

    init_ee(project="agrosat-copilot")
    import ee  # type: ignore[import-untyped]

    roi = ee.Geometry.Rectangle(
        [
            PO_VALLEY_BBOX["min_lon"],
            PO_VALLEY_BBOX["min_lat"],
            PO_VALLEY_BBOX["max_lon"],
            PO_VALLEY_BBOX["max_lat"],
        ]
    )
    df = sample_alphaearth_roi(roi, year=AOI_YEAR, n_pixels=n_pixels, roi_name="pianura_padana")
    if df.is_empty():
        raise RuntimeError(
            "AlphaEarth returned no pixel for the Po-valley AOI (check GEE auth / "
            "quota). No synthetic AOI is ever drawn."
        )
    return df


def _write_aoi_geojson(df: pl.DataFrame, *, out: Path) -> None:
    """Write the AOI footprint + sample bounds as a real GeoJSON FeatureCollection.

    Args:
        df: AlphaEarth AOI frame with ``lon`` / ``lat``.
        out: Destination GeoJSON path.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lon = df.get_column("lon")
    lat = df.get_column("lat")
    bbox = [
        float(lon.min()),  # type: ignore[arg-type]
        float(lat.min()),  # type: ignore[arg-type]
        float(lon.max()),  # type: ignore[arg-type]
        float(lat.max()),  # type: ignore[arg-type]
    ]
    ring = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]],
    ]
    fc = {
        "type": "FeatureCollection",
        "name": AOI_NAME,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "aoi": AOI_NAME,
                    "region": "Pianura Padana (Po valley), Italy",
                    "year": AOI_YEAR,
                    "n_pixels": int(df.height),
                    "source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (AlphaEarth v1.1)",
                    "license": "CC-BY-4.0",
                    "bbox": bbox,
                },
            }
        ],
    }
    out.write_text(json.dumps(fc, indent=2), encoding="utf-8", newline="\n")
    logger.info("aoi_geojson_written", path=str(out), bbox=bbox)


def _pca_rgb(df: pl.DataFrame) -> np.ndarray:
    """Project the 64-dim AlphaEarth pixels to a per-pixel false-colour RGB.

    Fits a 3-component PCA on the standardized embedding and min-max scales each
    component to ``[0, 1]`` so spatially-coherent crop/landcover structure shows as
    colour. Deterministic (fixed seed).

    Args:
        df: AlphaEarth AOI frame with the 64 ``dim_*`` columns.

    Returns:
        ``(n, 3)`` float array of RGB values in ``[0, 1]``.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = df.select(ALPHAEARTH_DIM_COLS).to_numpy()
    scaled = StandardScaler().fit_transform(x)
    comps = PCA(n_components=3, random_state=17).fit_transform(scaled)
    lo = comps.min(axis=0, keepdims=True)
    hi = comps.max(axis=0, keepdims=True)
    rgb: np.ndarray = np.clip((comps - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
    return rgb


def _add_osm_basemap(ax: plt.Axes, bbox: list[float]) -> bool:
    """Add an OpenStreetMap tile basemap to ``ax`` in Web Mercator, if possible.

    Uses :mod:`xyzservices` (the tile registry contextily wraps) to fetch and stitch
    OSM tiles for the AOI bbox, avoiding the missing ``contextily`` dependency.

    Args:
        ax: Target axes (already in Web Mercator / EPSG:3857 units).
        bbox: ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.

    Returns:
        ``True`` if a basemap was added, ``False`` if tiles could not be fetched
        (the scatter still renders, only without the basemap).
    """
    try:
        import io
        import urllib.request

        import xyzservices.providers as xyz
        from PIL import Image
        from pyproj import Transformer
    except ImportError:
        return False

    def _deg2num(lon: float, lat: float, z: int) -> tuple[int, int]:
        lat_rad = np.radians(lat)
        n = 2**z
        xt = int((lon + 180.0) / 360.0 * n)
        yt = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
        return xt, yt

    def _num2deg(xt: int, yt: int, z: int) -> tuple[float, float]:
        n = 2**z
        lon = xt / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * yt / n))))
        return lon, lat

    zoom = 9
    provider = xyz.OpenStreetMap.Mapnik
    x0, y1 = _deg2num(bbox[0], bbox[1], zoom)
    x1, y0 = _deg2num(bbox[2], bbox[3], zoom)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    try:
        tiles_x = range(min(x0, x1), max(x0, x1) + 1)
        tiles_y = range(min(y0, y1), max(y0, y1) + 1)
        for xt in tiles_x:
            for yt in tiles_y:
                url = provider.build_url(x=xt, y=yt, z=zoom)
                req = urllib.request.Request(  # noqa: S310 - fixed https OSM tile URL
                    url, headers={"User-Agent": "agrosatcopilot-paper/1.0"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                tl_lon, tl_lat = _num2deg(xt, yt, zoom)
                br_lon, br_lat = _num2deg(xt + 1, yt + 1, zoom)
                left, top = transformer.transform(tl_lon, tl_lat)
                right, bottom = transformer.transform(br_lon, br_lat)
                ax.imshow(
                    np.asarray(img),
                    extent=(left, right, bottom, top),
                    origin="upper",
                    zorder=0,
                    interpolation="bilinear",
                )
    except Exception as exc:  # noqa: BLE001 - basemap is optional
        logger.warning("osm_basemap_failed", error=str(exc))
        return False
    return True


def build_static_map(df: pl.DataFrame, *, out_dir: Path, dpi: int, lang: Lang) -> dict[str, Path]:
    """Render the static Italy AOI map (PCA-RGB pixels over OSM basemap) in one language.

    Only the visible strings (title, footer attribution) depend on ``lang``; the
    scatter, projection and data are identical across languages.

    Args:
        df: Real AlphaEarth AOI frame.
        out_dir: Destination directory.
        dpi: Raster resolution.
        lang: Language of the visible text; drives the ``_es`` filename suffix.

    Returns:
        Mapping ``{"png": path, "svg": path}``.
    """
    from pyproj import Transformer

    txt = STRINGS[lang]

    rgb = _pca_rgb(df)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx, my = transformer.transform(df.get_column("lon").to_numpy(), df.get_column("lat").to_numpy())
    bbox = [
        float(df.get_column("lon").min()),  # type: ignore[arg-type]
        float(df.get_column("lat").min()),  # type: ignore[arg-type]
        float(df.get_column("lon").max()),  # type: ignore[arg-type]
        float(df.get_column("lat").max()),  # type: ignore[arg-type]
    ]

    np.random.seed(17)
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 300, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    has_base = _add_osm_basemap(ax, bbox)
    ax.scatter(mx, my, c=rgb, s=10, alpha=0.9, edgecolors="none", zorder=2)
    ax.set_xlim(min(mx), max(mx))
    ax.set_ylim(min(my), max(my))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{txt['title_line1']}\n{txt['title_line2'].format(n=df.height)}")
    base_note = txt["basemap_on"] if has_base else txt["basemap_off"]
    fig.text(0.5, 0.005, f"{txt['attrib']} | {base_note}", ha="center", fontsize=6, color="0.35")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem("aoi_italy", lang)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    logger.info("aoi_static_map_saved", lang=lang, png=str(png), svg=str(svg), basemap=has_base)
    return {"png": png, "svg": svg}


def build_folium_map(df: pl.DataFrame, *, out_dir: Path, lang: Lang) -> Path:
    """Render the interactive folium AOI map (rectangle + sampled-pixel heat) per language.

    Args:
        df: Real AlphaEarth AOI frame.
        out_dir: Destination directory.
        lang: Language of the popup text; drives the ``_es`` filename suffix.

    Returns:
        Path to the saved ``aoi_italy.html`` (or ``aoi_italy_es.html`` for Spanish).
    """
    import folium
    from folium.plugins import HeatMap

    txt = STRINGS[lang]

    clat = float(df.get_column("lat").mean())  # type: ignore[arg-type]
    clon = float(df.get_column("lon").mean())  # type: ignore[arg-type]
    fmap = folium.Map(location=[clat, clon], zoom_start=9, tiles="OpenStreetMap")
    bounds = [
        [float(df.get_column("lat").min()), float(df.get_column("lon").min())],  # type: ignore[arg-type]
        [float(df.get_column("lat").max()), float(df.get_column("lon").max())],  # type: ignore[arg-type]
    ]
    folium.Rectangle(
        bounds=bounds,
        color="#d62728",
        weight=2,
        fill=False,
        popup=txt["folium_popup"].format(n=df.height),
    ).add_to(fmap)
    HeatMap(
        [[r["lat"], r["lon"]] for r in df.select("lat", "lon").iter_rows(named=True)],
        radius=6,
        blur=8,
    ).add_to(fmap)
    folium.map.Marker(
        [bounds[1][0], clon],
        icon=folium.DivIcon(
            html=(
                "<div style='font-size:9px;color:#333'>AlphaEarth v1.1 (CC-BY-4.0) | "
                "OSM (ODbL)</div>"
            )
        ),
    ).add_to(fmap)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{_stem('aoi_italy', lang)}.html"
    fmap.save(str(html))
    logger.info("aoi_folium_map_saved", lang=lang, html=str(html))
    return html


def build_all(*, out_dir: Path = FIGURES_DIR, dpi: int = 200, n_pixels: int = 2000) -> None:
    """Build the Italy AOI GeoJSON + static + interactive maps from real AlphaEarth.

    Emits every figure in both languages: the English base files (``aoi_italy.*``)
    and the Spanish ``_es`` variants (``aoi_italy_es.*``). The GeoJSON is data-only
    (no visible figure text) and is written once.

    Args:
        out_dir: Destination directory for the figures.
        dpi: Raster resolution for the static map.
        n_pixels: Pixels to request from GEE on a cache miss.
    """
    df = _load_alphaearth_aoi(n_pixels=n_pixels)
    _write_aoi_geojson(df, out=AOI_GEOJSON)
    outputs: dict[Lang, dict[str, Path]] = {}
    for lang in LANGS:
        static_paths = build_static_map(df, out_dir=out_dir, dpi=dpi, lang=lang)
        html = build_folium_map(df, out_dir=out_dir, lang=lang)
        outputs[lang] = {**static_paths, "html": html}
    logger.info(
        "us070_italy_aoi_done",
        geojson=str(AOI_GEOJSON),
        png_en=str(outputs["en"]["png"]),
        png_es=str(outputs["es"]["png"]),
        html_en=str(outputs["en"]["html"]),
        html_es=str(outputs["es"]["html"]),
    )


@app.command()
def run(
    out_dir: Annotated[Path, typer.Option("--out-dir")] = FIGURES_DIR,
    dpi: Annotated[int, typer.Option("--dpi")] = 200,
    n_pixels: Annotated[int, typer.Option("--n-pixels")] = 2000,
) -> None:
    """CLI entry point: build the Italy AOI figure set (see :func:`build_all`).

    Args:
        out_dir: Destination directory for the figures.
        dpi: Raster resolution for the static map.
        n_pixels: Pixels to request from GEE on a cache miss.
    """
    build_all(out_dir=out_dir, dpi=dpi, n_pixels=n_pixels)


if __name__ == "__main__":
    app()
