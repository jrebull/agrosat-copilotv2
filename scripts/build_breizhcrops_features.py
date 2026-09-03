"""Materializa y sella las caracteristicas por parcela de BreizhCrops para la fase 4.

Submuestreo proporcional por region con semilla fija, como fija la enmienda 1 del
preregistro: preserva el reparto de clases, que es el objeto de estudio, y acota el coste
de extraer las 185 caracteristicas temporales.

Uso:
    poetry run python scripts/build_breizhcrops_features.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase4"
REGIONS = ("frh01", "frh04")


def main() -> None:
    """Build the per-parcel feature table for every region and seal its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ml.features.breizhcrops_features import build_breizhcrops_features
    from ml.ingest.breizhcrops_loader import breizhcrops_pixel_series

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for region in REGIONS:
        series = breizhcrops_pixel_series(region=region, sample_parcels=args.sample, seed=args.seed)
        logger.info("serie_cargada", region=region, filas=series.height)
        feats = build_breizhcrops_features(series).with_columns(pl.lit(region).alias("region"))
        logger.info("features_listas", region=region, parcelas=feats.height)
        frames.append(feats)

    table = pl.concat(frames, how="vertical")
    path = OUT_DIR / "breizhcrops_features.parquet"
    table.write_parquet(path, compression="zstd")

    support = (
        table.group_by("region", "class_id", "class_name")
        .len()
        .rename({"len": "n_parcelas"})
        .sort(["region", "n_parcelas"], descending=[False, True])
    )
    support.write_csv(OUT_DIR / "breizhcrops_soporte.csv")

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (OUT_DIR / "breizhcrops_procedencia.json").write_text(
        json.dumps(
            {
                "regiones": list(REGIONS),
                "submuestreo_por_region": args.sample,
                "semilla": args.seed,
                "n_parcelas": table.height,
                "n_features": table.width - 4,
                "anio": 2017,
                "nivel": "L2A",
                "code_version": head or "desconocido",
                "polars": pl.__version__,
                "generado": datetime.now(UTC).isoformat(timespec="seconds"),
                "nota": (
                    "Submuestreo proporcional por region con semilla fija, declarado en la "
                    "enmienda 1 del preregistro antes de ver ninguna cifra. Los bloques del "
                    "protocolo son las dos regiones, porque BreizhCrops no trae coordenadas "
                    "por parcela."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("sellado", parcelas=table.height, out=str(OUT_DIR))


if __name__ == "__main__":
    main()
