"""Seal the fold-5 parcel ground truth and centroids that every MICAI figure needs.

**La poblacion es la ELEGIBILIDAD DEL BANCO, no la interseccion de los miembros.** Es lo que
declara `docs/paper/estimando-v1.json`: `population = all_eligible_test_parcels` con
`include_non_delivery = true`. La version anterior intersecaba los miembros disponibles, y eso
tiene dos consecuencias que no se ven hasta que muerden: la poblacion cambiaba al cambiar el
estado de un fichero —de 28 532 a 16 640 con solo marcar un miembro como no canonico— y las
parcelas que un predictor no cubre desaparecian del denominador en vez de contar como no entrega,
que es exactamente el denominador movil que el articulo denuncia.

Ahora la poblacion sale del fold retenido de PASTIS-R con etiqueta semantic18 valida, cada
predictor se alinea con un LEFT JOIN, y **la ausencia es no entrega**. La cobertura de cada
miembro pasa a ser un dato que se reporta, no una definicion que se aplica.

Las etiquetas y los centroides se reconstruyen del PASTIS-R crudo, que son 68 GB y viven fuera del
repositorio. Este guion los deriva una vez y los escribe en ``reports/paper_micai/fase1/`` para que
un clon limpio pueda re-derivar cada figura impresa desde unos cientos de kilobytes.

Usage:
    poetry run python scripts/paper_micai_seal_fold5.py
    poetry run python scripts/paper_micai_seal_fold5.py --pastis-root data/PASTIS-R
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog

from ml.eval.oof.inventario import cargar_inventario

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
DEFAULT_PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase1"
KEY = "canonical_parcel_id"

#: Zenodo checksum of the PASTIS-R archive the labels come from, per docs/STATUS.md.
PASTIS_ARCHIVE_MD5 = "4887513d6c2d2b07fa935d325bd53e09"


def cobertura_por_miembro(oof_dir: Path, poblacion: list[str]) -> list[dict[str, object]]:
    """Coverage of every canonical member over the eligible population.

    Es un DATO que se reporta, no una definicion que se aplica. Antes, lo que un miembro no
    cubria desaparecia de la poblacion; ahora se cuenta y la parcela sigue ahi como no entrega.

    Args:
        oof_dir: Directory holding ``oof_parcel_*_fold5.parquet``.
        poblacion: The eligible parcel ids.

    Returns:
        One row per canonical member with how much of the population it covers.
    """
    inventario = cargar_inventario()
    elegibles = set(poblacion)
    filas: list[dict[str, object]] = []
    for path in sorted(oof_dir.glob("oof_parcel_*_fold5.parquet")):
        entrada = inventario["ficheros"].get(path.name, {})
        if entrada.get("estado") != "canonical":
            logger.info(
                "miembro_no_canonico_descartado", fichero=path.name, estado=entrada.get("estado")
            )
            continue
        ids = set(pl.read_parquet(path, columns=[KEY])[KEY].to_list())
        cubiertas = len(ids & elegibles)
        filas.append(
            {
                "miembro": path.stem.removeprefix("oof_parcel_").removesuffix("_fold5"),
                "parcelas_cubiertas": cubiertas,
                "parcelas_sin_entrega": len(elegibles) - cubiertas,
                "cobertura": round(cubiertas / len(elegibles), 6) if elegibles else 0.0,
                "fuera_de_la_poblacion": len(ids - elegibles),
            }
        )
    return filas


def git_head() -> str:
    """Return the short SHA of HEAD.

    Returns:
        The abbreviated commit hash, or ``"desconocido"`` outside a repository.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "desconocido"


def main() -> None:
    """Build and write the sealed fold-5 ground truth, centroids and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--pastis-root", type=Path, default=DEFAULT_PASTIS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from scripts.run_us040_ensembles import (
        _fold5_patch_ids,
        build_parcel_geometries,
        build_parcel_ground_truth,
    )

    patch_ids = _fold5_patch_ids(args.oof_dir)

    # La poblacion elegible: las parcelas del fold retenido con etiqueta semantic18 valida. No
    # depende de que miembros existan ni de su estado, que es el punto entero de esta correccion.
    gt = build_parcel_ground_truth(patch_ids, args.pastis_root)
    geoms = build_parcel_geometries(patch_ids, args.pastis_root)
    gt_shared = gt.sort(KEY)
    universe = gt_shared[KEY].to_list()
    logger.info("poblacion_elegible", n_parcels=len(universe), n_patches=len(patch_ids))

    keep = pl.DataFrame({KEY: universe})
    geoms_shared = geoms.join(keep, on=KEY, how="inner").sort(KEY)

    # La cobertura de cada miembro es un dato del informe, no un filtro de la poblacion.
    cobertura = cobertura_por_miembro(args.oof_dir, universe)
    for fila in cobertura:
        logger.info("cobertura_miembro", **fila)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.out_dir / "parcel_gt_fold5.parquet"
    geoms_path = args.out_dir / "parcel_centroids_fold5.parquet"
    gt_shared.write_parquet(gt_path, compression="zstd")
    geoms_shared.write_parquet(geoms_path, compression="zstd")

    support = (
        gt_shared.group_by("label")
        .len()
        .rename({"len": "n_parcels"})
        .sort("n_parcels", descending=True)
    )
    support_path = args.out_dir / "parcel_gt_fold5_support.csv"
    support.write_csv(support_path)

    provenance = {
        "descripcion": (
            "Etiquetas y centroides por parcela del fold 5 held-out de PASTIS-R, "
            "restringidos al universo compartido por los diez miembros OOF de Francia."
        ),
        "n_parcels": gt_shared.height,
        "n_patches": len(patch_ids),
        "n_classes": int(support.height),
        "pastis_archive_md5_zenodo": PASTIS_ARCHIVE_MD5,
        "oof_dir": str(args.oof_dir.relative_to(REPO_ROOT)),
        "code_version": git_head(),
        "polars": pl.__version__,
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.out_dir / "parcel_gt_fold5_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "sealed",
        gt=str(gt_path.relative_to(REPO_ROOT)),
        centroids=str(geoms_path.relative_to(REPO_ROOT)),
        support=str(support_path.relative_to(REPO_ROOT)),
        n_parcels=gt_shared.height,
    )


if __name__ == "__main__":
    main()
