"""Generate the demo fixture of a per-parcel multiband time series (US-015).

Permanent project operative tool: produces
``data/test_fixtures/parcel_demo_ts.nc`` (<100 KB) consumed by
``tests/ml/features/test_temporal_features.py::test_extract_demo_fixture_end_to_end``.

Note on the format: the US-015 §3.1 plan mentions ``.zarr`` as the intended
suffix. It is switched to NetCDF3 (scipy backend, no additional dependency)
to avoid introducing ``zarr`` into the stack. The logical content and the
attrs are identical; the change is purely about serialization.

The synthetic series has 30 timesteps spaced approximately every 12 days
during the year 2024 and 17 bands (the project's 17 canonical indices). The
NDVI band follows a gaussian centered at DOY 180 with peak 0.85 and σ=30 days
+ gaussian noise σ=0.02 (seed=42). The remaining bands use plausible curves
correlated with NDVI.

Usage:

    poetry run python scripts/generate_demo_parcel_ts.py \\
        --output data/test_fixtures/parcel_demo_ts.nc
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import structlog
import typer
import xarray as xr

logger = structlog.get_logger(__name__)

BANDS: tuple[str, ...] = (
    "NDVI",
    "NDWI",
    "EVI",
    "NDMI",
    "NBR",
    "MSAVI2",
    "NDRE",
    "MCARI",
    "CCCI",
    "GCVI",
    "PSRI",
    "NDCI",
    "FAPAR",
    "LAI",
    "RENDVI",
    "SAVI",
    "TSAVI",
)

app = typer.Typer(add_completion=False, help="Genera el fixture demo de serie temporal.")


def _build_dataarray(*, parcel_id: int, year: int, seed: int) -> xr.DataArray:
    """Build the deterministic synthetic DataArray."""
    rng = np.random.default_rng(seed)
    n_steps = 30
    # Timesteps regularly spaced (~12 days) during the year.
    start = np.datetime64(f"{year}-01-15", "ns")
    delta_days = 12
    times = np.array(
        [start + np.timedelta64(i * delta_days, "D") for i in range(n_steps)],
        dtype="datetime64[ns]",
    )

    doys = np.array(
        [(t - np.datetime64(f"{year}-01-01", "ns")) / np.timedelta64(1, "D") + 1 for t in times],
        dtype=np.float64,
    )

    # Canonical NDVI: gaussian DOY 180, peak 0.85, sigma 30 days + noise.
    ndvi = 0.85 * np.exp(-0.5 * ((doys - 180) / 30.0) ** 2)
    ndvi += rng.normal(0.0, 0.02, size=n_steps)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    values = np.empty((n_steps, len(BANDS)), dtype=np.float64)
    for j, band in enumerate(BANDS):
        if band == "NDVI":
            values[:, j] = ndvi
            continue
        # Plausible curves derived from NDVI with per-band noise.
        if band in {"EVI", "SAVI", "MSAVI2", "TSAVI", "RENDVI", "NDRE"}:
            base = 0.9 * ndvi + 0.05
        elif band in {"NDWI", "NDMI"}:
            base = 0.5 * ndvi - 0.1
        elif band == "NBR":
            base = 0.7 * ndvi + 0.05
        elif band in {"MCARI", "CCCI", "GCVI", "NDCI"}:
            base = 0.6 * ndvi + 0.1
        elif band == "PSRI":
            # Senescence: anti-correlated with vigor.
            base = -0.4 * ndvi + 0.2
        elif band == "FAPAR":
            base = 1.24 * ndvi - 0.168
        elif band == "LAI":
            arg = np.clip((ndvi - 0.05) / 0.95, 1e-3, 0.999)
            base = -np.log(1.0 - arg) / 0.5
        else:
            base = ndvi.copy()
        noise = rng.normal(0.0, 0.015, size=n_steps)
        values[:, j] = base + noise

    da = xr.DataArray(
        data=values,
        dims=("time", "band"),
        coords={"time": times, "band": list(BANDS)},
        attrs={"parcel_id": parcel_id, "year": year, "seed": seed},
        name="parcel_indices",
    )
    return da


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/test_fixtures/parcel_demo_ts.nc"),
        "--output",
        "-o",
        help="Ruta de salida del fixture NetCDF.",
    ),
    parcel_id: int = typer.Option(42, help="Identificador de la parcela sintética."),
    year: int = typer.Option(2024, help="Año del ciclo agrícola simulado."),
    seed: int = typer.Option(42, help="Semilla para reproducibilidad."),
) -> None:
    """Generate the demo fixture and write it to disk in Zarr format."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.is_dir():
            import shutil

            shutil.rmtree(output)
        else:
            output.unlink()

    da = _build_dataarray(parcel_id=parcel_id, year=year, seed=seed)
    ds = da.to_dataset(name="parcel_indices")
    # The scipy backend (NetCDF3) does not support variable-length strings in
    # "band"-type coords; we convert to a fixed object/string array via to_netcdf
    # with format NETCDF4 if available, fallback NETCDF3_64BIT.
    try:
        ds.to_netcdf(output, format="NETCDF4")
    except (ValueError, RuntimeError):
        # Convert the band coord to S1 fixed-length for NETCDF3.
        ds = ds.assign_coords(band=ds.coords["band"].astype("S16"))
        ds.to_netcdf(output, format="NETCDF3_64BIT", engine="scipy")

    logger.info(
        "demo fixture generated",
        output=str(output),
        timesteps=int(da.sizes["time"]),
        bands=int(da.sizes["band"]),
        parcel_id=parcel_id,
        year=year,
    )


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("demo fixture generation failed", error=str(exc))
        sys.exit(1)
