"""Generate the compact PASTIS-R subset for the dashboard folium map.

The original PASTIS-R ``metadata.geojson`` weighs 19 MB (2,433 patches in
EPSG:2154) and is not committed to the repo. For the dashboard to work on
Streamlit Cloud we need a compact version:

    1. Dissolve by ``TILE`` column (4 super-tiles).
    2. Reproject to EPSG:4326.
    3. Simplify geometries with 0.001 degree tolerance.
    4. Keep only ``tile`` + ``geometry``.

Result: ``data/reference/pastis_tiles_dissolved.geojson`` (~500 KB),
listed in the ``.gitignore`` whitelist (``!data/reference/**/*.geojson``).

Usage:
    python -m ml.report.generate_pastis_subset
    python -m ml.report.generate_pastis_subset --source data/PASTIS-R/metadata.geojson \\
        --output data/reference/pastis_tiles_dissolved.geojson
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(
    add_completion=False,
    help="Genera subset compacto PASTIS-R para el dashboard.",
)


@app.command()
def main(
    source: Path = typer.Option(
        REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson",
        "--source",
        "-s",
        help="GeoJSON original de PASTIS-R (2.433 patches en EPSG:2154).",
    ),
    output: Path = typer.Option(
        REPO_ROOT / "data" / "reference" / "pastis_tiles_dissolved.geojson",
        "--output",
        "-o",
        help="Destino del GeoJSON compacto.",
    ),
    tolerance: float = typer.Option(
        0.001,
        "--tolerance",
        "-t",
        help="Tolerancia de simplificación en grados (default 0,001 ≈ 100 m).",
    ),
) -> None:
    """Generate the compact subset from the full metadata.geojson."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        typer.echo(
            f"ERROR: geopandas no instalado ({exc}). Ejecutá `poetry install --with geo`.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    if not source.exists():
        typer.echo(
            f"ERROR: {source} no existe. Sincronizá PASTIS-R con DVC "
            "(`dvc pull data/PASTIS-R`) antes de regenerar el subset.",
            err=True,
        )
        raise typer.Exit(code=2)

    size_mb = source.stat().st_size / 1024 / 1024
    typer.echo(f"Leyendo {source} ({size_mb:.1f} MB)...")

    gdf = gpd.read_file(source)
    typer.echo(f"  Filas: {len(gdf)}, columnas: {list(gdf.columns)[:8]}")

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        typer.echo(f"  Reproyectando {gdf.crs} -> EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    tile_col = next((c for c in ("TILE", "tile", "Tile") if c in gdf.columns), None)
    typer.echo(f"  Columna TILE detectada: {tile_col}")

    if tile_col is not None:
        dissolved = gdf.dissolve(by=tile_col).reset_index()
        dissolved = dissolved[[tile_col, "geometry"]].rename(columns={tile_col: "tile"})
    else:
        dissolved = gdf[["geometry"]].copy()

    typer.echo(f"  Filas tras dissolve: {len(dissolved)}")

    if tolerance > 0:
        dissolved["geometry"] = dissolved["geometry"].simplify(
            tolerance=tolerance, preserve_topology=True
        )
        typer.echo(f"  Geometrías simplificadas con tolerance={tolerance}")

    output.parent.mkdir(parents=True, exist_ok=True)
    dissolved.to_file(output, driver="GeoJSON")
    out_kb = output.stat().st_size / 1024
    typer.echo(f"OK escrito {output} ({out_kb:.1f} KB)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
