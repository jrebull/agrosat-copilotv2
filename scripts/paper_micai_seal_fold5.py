"""Seal the fold-5 parcel ground truth and centroids that every MICAI figure needs.

The manuscript scores its ensembles on the 16 640 parcels shared by the ten
fold-5 out-of-fold members. Those posteriors are versioned in DVC, but the labels
and the centroids they are scored against are rebuilt from the raw PASTIS-R
dataset, which is 68 GB and lives outside the repository. This script derives both
frames once, restricted to that shared universe, and writes them under
``reports/paper_micai/fase1/`` so a clean clone can re-derive every printed figure
from a few hundred kilobytes.

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

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
DEFAULT_PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase1"
KEY = "canonical_parcel_id"

#: Zenodo checksum of the PASTIS-R archive the labels come from, per docs/STATUS.md.
PASTIS_ARCHIVE_MD5 = "4887513d6c2d2b07fa935d325bd53e09"


def shared_universe(oof_dir: Path) -> list[str]:
    """Intersect the parcel ids of every fold-5 member of the France universe.

    Args:
        oof_dir: Directory holding ``oof_parcel_*_fold5.parquet``.

    Returns:
        The sorted parcel ids present in every member.

    Raises:
        FileNotFoundError: if no parcel-level member is available.
    """
    members = sorted(
        p for p in oof_dir.glob("oof_parcel_*_fold5.parquet") if "italia" not in p.name
    )
    if not members:
        raise FileNotFoundError(f"no parcel OOF found in {oof_dir}; run `dvc pull {oof_dir}`.")
    shared: set[str] | None = None
    for path in members:
        ids = set(pl.read_parquet(path, columns=[KEY])[KEY].to_list())
        shared = ids if shared is None else (shared & ids)
        logger.info("member_read", member=path.stem, n_parcels=len(ids))
    assert shared is not None
    return sorted(shared)


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

    universe = shared_universe(args.oof_dir)
    patch_ids = _fold5_patch_ids(args.oof_dir)
    logger.info("universe", n_parcels=len(universe), n_patches=len(patch_ids))

    gt = build_parcel_ground_truth(patch_ids, args.pastis_root)
    geoms = build_parcel_geometries(patch_ids, args.pastis_root)

    keep = pl.DataFrame({KEY: universe})
    gt_shared = gt.join(keep, on=KEY, how="inner").sort(KEY)
    geoms_shared = geoms.join(keep, on=KEY, how="inner").sort(KEY)

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
