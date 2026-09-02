"""Operational Typer CLI to build the fused feature parquet (US-016).

Permanent project operational script. Chains:

1. :func:`ml.features.fusion.build_fused_features` — produces the fused
   frame (189 cols without FarSLIP / 701 with FarSLIP).
2. :func:`ml.features.spatial_split.build_spatial_kfold` — generates K spatial
   folds with H3 tessellation + KMeans + buffer.
3. :func:`ml.features.scaler.fit_scaler_on_train` — fits a StandardScaler
   on the Fold-0 train split and persists it with joblib.

Outputs (all versionable with DVC):

- ``data/features/features_fused_v1.parquet``
- ``data/splits/spatial_kfold_v1/fold_{0..k-1}/{train,val,test}_parcel_ids.parquet``
- ``artifacts/scaler_v1.pkl``

Determinism: two consecutive executions with the same input produce a
parquet with the same MD5 (as long as the parcels source and the GEE caches
are stable).

Typical execution:

::

    poetry run python scripts/build_parcel_features.py \\
        --year 2024 \\
        --regions pianura_padana,toscana_centrale,apulia \\
        --out data/features/features_fused_v1.parquet \\
        --scaler-out artifacts/scaler_v1.pkl \\
        --splits-out data/splits/spatial_kfold_v1/

Exit-0 output: structured JSON dump with ``{rows, cols, splits,
scaler_version, dvc_tracked}`` to stdout for CI consumption.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import polars as pl
import structlog
import typer

from ml.features.fusion import (
    EXPECTED_COL_COUNT_NO_FARSLIP,
    EXPECTED_COL_COUNT_WITH_FARSLIP,
    build_fused_features,
)
from ml.features.scaler import fit_scaler_on_train
from ml.features.spatial_split import build_spatial_kfold

logger = structlog.get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Construye el frame multisensor + spatial K-fold + scaler (US-016).",
)


@app.command()
def build(
    year: int = typer.Option(..., help="Año de referencia para AlphaEarth/S1/ERA5."),
    regions: str = typer.Option(
        "pianura_padana,toscana_centrale,apulia",
        help="Lista CSV de regiones a incluir (informativo, requiere parcels-path).",
    ),
    out: Path = typer.Option(
        Path("data/features/features_fused_v1.parquet"),
        help="Path destino del parquet fusionado.",
    ),
    scaler_out: Path = typer.Option(
        Path("artifacts/scaler_v1.pkl"),
        help="Path destino del scaler joblib (Fold-0 train).",
    ),
    splits_out: Path = typer.Option(
        Path("data/splits/spatial_kfold_v1/"),
        help="Carpeta destino de los splits spatial K-fold.",
    ),
    include_farslip: bool = typer.Option(
        False,
        "--include-farslip/--no-farslip",
        help="Si True intenta unir el bloque FarSLIP (512 cols) vía LEFT JOIN.",
    ),
    farslip_path: Path | None = typer.Option(
        None, help="Path al parquet de embeddings FarSLIP. Default heurístico."
    ),
    parcels_path: Path = typer.Option(
        Path("data/test_fixtures/parcels_demo_3regions.parquet"),
        help="Parquet/GeoParquet de parcelas con `parcel_id`, `year`, `geometry`.",
    ),
    k: int = typer.Option(5, help="Número de folds spatial K-fold."),
    h3_res: int = typer.Option(5, help="Resolución H3 para tessellation."),
    buffer_km: float = typer.Option(1.0, help="Buffer inter-fold en km."),
    val_fraction: float = typer.Option(0.2, help="Fracción de train usada como val."),
    random_state: int = typer.Option(42, help="Seed determinista."),
    scaler_version: str = typer.Option("v1", help="Tag de versión inyectado al scaler."),
) -> None:
    """Pipeline: multisensor fusion -> spatial K-fold -> fit scaler train Fold-0."""
    regions_list = [r.strip() for r in regions.split(",") if r.strip()]
    logger.info(
        "build_started",
        year=year,
        regions=regions_list,
        parcels_path=str(parcels_path),
        out=str(out),
        include_farslip=include_farslip,
    )

    parcels = _load_parcels(parcels_path)
    if "region" in parcels.columns and regions_list:
        before = len(parcels)
        parcels = parcels[parcels["region"].isin(regions_list)].copy()
        logger.info(
            "parcels_filtered_by_region",
            before=before,
            after=len(parcels),
            regions=regions_list,
        )

    # 1) Fusion frame.
    fused = build_fused_features(
        parcels,
        year=year,
        include_farslip=include_farslip,
        farslip_path=farslip_path,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fused.write_parquet(out)
    md5 = _md5_of_file(out)
    expected_cols = (
        EXPECTED_COL_COUNT_WITH_FARSLIP if include_farslip else EXPECTED_COL_COUNT_NO_FARSLIP
    )
    feature_count = fused.width - 2  # excluye parcel_id + year
    logger.info(
        "fusion_persisted",
        path=str(out),
        rows=fused.height,
        cols=fused.width,
        feature_cols=feature_count,
        expected_feature_cols=expected_cols,
        md5=md5,
    )

    # 2) Spatial K-fold.
    splits_out.mkdir(parents=True, exist_ok=True)
    folds = build_spatial_kfold(
        parcels,
        k=k,
        h3_res=h3_res,
        buffer_km=buffer_km,
        val_fraction=val_fraction,
        random_state=random_state,
    )
    for fold in folds:
        fold_dir = splits_out / f"fold_{fold.fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split_name, ids in (
            ("train", fold.train_ids),
            ("val", fold.val_ids),
            ("test", fold.test_ids),
        ):
            pl.DataFrame({"parcel_id": list(ids)}, schema={"parcel_id": pl.Int64}).write_parquet(
                fold_dir / f"{split_name}_parcel_ids.parquet"
            )
    logger.info(
        "splits_persisted",
        path=str(splits_out),
        k=k,
        h3_res=h3_res,
        buffer_km=buffer_km,
        fold_sizes=[
            {
                "fold": f.fold_id,
                "train": len(f.train_ids),
                "val": len(f.val_ids),
                "test": len(f.test_ids),
            }
            for f in folds
        ],
    )

    # 3) Scaler train Fold-0.
    fold0 = folds[0]
    feature_cols = tuple(c for c in fused.columns if c not in ("parcel_id", "year"))
    fit_scaler_on_train(
        fused,
        train_ids=fold0.train_ids,
        feature_cols=feature_cols,
        scaler_path=scaler_out,
        version=scaler_version,
        val_ids=fold0.val_ids,
        test_ids=fold0.test_ids,
    )

    summary = {
        "rows": int(fused.height),
        "cols": int(fused.width),
        "feature_cols": int(feature_count),
        "expected_feature_cols": int(expected_cols),
        "splits": int(k),
        "scaler_version": scaler_version,
        "scaler_path": str(scaler_out),
        "fusion_md5": md5,
        "dvc_tracked": False,
        "include_farslip": include_farslip,
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    raise typer.Exit(code=0)


def _load_parcels(path: Path) -> gpd.GeoDataFrame:
    """Loads parcels from GeoParquet, flat parquet with WKT, or GeoJSON.

    Supports three formats in order of attempt:

    1. **GeoParquet** (with `geo` metadata): via `gpd.read_parquet`.
    2. **Flat parquet** with a `geom` column (WKT string): via Polars + shapely
       (same loader as `dagster_project/assets/features.py` for coherence
       with the demo fixture).
    3. **GeoJSON/JSON**: via `gpd.read_file`.
    """
    if not path.exists():
        raise typer.BadParameter(f"Parcels path not found: {path}")
    if path.suffix in (".geoparquet", ".parquet"):
        try:
            gdf = gpd.read_parquet(path)
        except Exception:  # noqa: BLE001 - any read failure falls back to the WKT fixture
            # Fallback: parquet plano con `geom` WKT (fixture demo).
            from shapely import wkt

            frame = pl.read_parquet(path).to_pandas()
            geom_col = "geom" if "geom" in frame.columns else "geometry"
            if geom_col not in frame.columns:
                raise typer.BadParameter(
                    f"Parcels in {path} contains neither a `geom` nor a `geometry` column."
                ) from None
            geoms = [wkt.loads(g) if isinstance(g, str) else g for g in frame[geom_col].tolist()]
            gdf = gpd.GeoDataFrame(
                frame.drop(columns=[geom_col]),
                geometry=geoms,
                crs="EPSG:4326",
            )
    elif path.suffix in (".geojson", ".json"):
        gdf = gpd.read_file(path)
    else:
        raise typer.BadParameter(f"Unsupported format: {path.suffix}. Use .parquet or .geojson.")
    if "parcel_id" not in gdf.columns:
        raise typer.BadParameter(f"Parcels in {path} does not contain the `parcel_id` column.")
    return gdf


def _md5_of_file(path: Path) -> str:
    """Returns the hex MD5 of the file to verify determinism."""
    h = hashlib.md5(usedforsecurity=False)  # checksum, not a security primitive
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    sys.exit(app())
