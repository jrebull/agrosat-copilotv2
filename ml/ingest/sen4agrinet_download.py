"""Selective download of a manageable Sen4AgriNet (S4A) subset (US-075).

The full S4A split is **281 GB**, so this tool NEVER downloads the ``complete``
config. It targets a few-GB subset of real netCDF patches for the EPIC 12
Franco-Iberian transfer experiment:

- **Catalonia** (Spain) tiles for the few-shot finetune / held-out evaluation.
- A handful of **France** tiles as the zero-shot reference domain.

Verified S4A repository layout (HF ``paren8esis/S4A`` / ``orion-ai-lab/S4A``,
inspected 20-jun-2026):

- Patches live at ``data/<year>/<TILE>/<year>_<TILE>_patch_<x>_<y>.nc`` (NOT under
  ``cat_2019/``: those are HuggingFace *dataset-builder config names* in
  ``S4A.py``, not directory prefixes, so ``snapshot_download`` ``allow_patterns``
  must use the real ``data/<year>/<TILE>/`` paths).
- Catalonia tiles (``patch_country_code='ES'``): ``31TBF 31TCF 31TCG 31TDF
  31TDG`` for years 2019 and 2020.
- France tiles (``patch_country_code='FR'``): ``31TCJ 31TDK 31TCL 31TDM 31UCP
  31UDR`` for year 2019.
- One patch is ~24 MB (366x366, 13 S2 bands with their own ``time`` dim ~26-50,
  plus ``labels`` and ``parcels`` 366x366 groups).

Most patches are dominated by background (label 0). To keep the subset small AND
useful, the downloader fetches candidate patches one at a time and KEEPS only
those whose ``parcels`` mask covers at least ``min_parcel_frac`` of the pixels,
deleting the rest before the caller runs ``dvc add``. This is the §3.1 / R1
mitigation in ``docs/us-planning/us-075.md``: "tras el snapshot, conservar solo
N patches con parcelas no vacias y borrar el resto antes de dvc add".

Run ON THE VM (F: has 3.4 TB), never on the dev laptop::

    F:/tools/micromamba.exe run -n agrosat python -m ml.ingest.sen4agrinet_download \\
        --out-dir F:/projects/agrosat-copilot/data/sen4agrinet \\
        --n-cat 30 --n-fr 10

Project convention: ``structlog`` logging, type hints, no pandas, no emojis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: HuggingFace dataset repo id (mirror ``paren8esis/S4A``).
S4A_REPO_ID: str = "paren8esis/S4A"
#: HuggingFace repo type.
S4A_REPO_TYPE: str = "dataset"

#: Catalonia (Spain) Sentinel-2 tiles, verified from ``S4A.py`` ``CAT_TILES``.
CAT_TILES: tuple[str, ...] = ("31TBF", "31TCF", "31TCG", "31TDF", "31TDG")
#: France Sentinel-2 tiles, verified from ``S4A.py`` ``FR_TILES``.
FR_TILES: tuple[str, ...] = ("31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR")

#: Years available per country in the S4A configs.
CAT_YEARS: tuple[str, ...] = ("2019", "2020")
FR_YEARS: tuple[str, ...] = ("2019",)

#: Preferred Catalonia tile (cited in plan v8 §US-075; present for 2019 AND 2020).
DEFAULT_CAT_TILE: str = "31TCG"
#: Preferred France reference tile (zero-shot domain).
DEFAULT_FR_TILE: str = "31TCJ"


@dataclass(frozen=True)
class SubsetManifest:
    """Summary of the downloaded subset, written next to the patches.

    Attributes:
        repo_id: Source HuggingFace dataset id.
        cat_patches: Relative repo paths of the kept Catalonia patches.
        fr_patches: Relative repo paths of the kept France patches.
        min_parcel_frac: Minimum parcel coverage required to keep a patch.
        total_size_mb: Total on-disk size of the kept ``.nc`` files in MB.
        n_cat: Number of Catalonia patches kept.
        n_fr: Number of France patches kept.
    """

    repo_id: str
    cat_patches: list[str]
    fr_patches: list[str]
    min_parcel_frac: float
    total_size_mb: float
    n_cat: int
    n_fr: int


def _list_tile_patches(tile: str, year: str) -> list[str]:
    """List the repo-relative ``.nc`` paths for one ``(tile, year)``.

    Args:
        tile: Sentinel-2 tile id (e.g. ``"31TCG"``).
        year: Acquisition year as a string (e.g. ``"2019"``).

    Returns:
        Sorted list of repo-relative paths ``data/<year>/<tile>/*.nc``.
    """
    from huggingface_hub import HfApi

    prefix = f"data/{year}/{tile}/"
    api = HfApi()
    files = api.list_repo_files(repo_id=S4A_REPO_ID, repo_type=S4A_REPO_TYPE)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".nc"))


def _parcel_fraction(nc_path: Path) -> float:
    """Return the fraction of pixels covered by a non-zero parcel id.

    Reads only the ``parcels`` group (cheap) to decide whether a patch carries
    enough agronomic signal to be worth keeping for the few-shot finetune.

    Args:
        nc_path: Local path to a downloaded ``.nc`` patch.

    Returns:
        Fraction in ``[0, 1]`` of pixels whose ``parcels`` value is non-zero.
        ``0.0`` if the group/variable is absent or unreadable.
    """
    import netCDF4

    try:
        ds = netCDF4.Dataset(str(nc_path))
    except OSError:
        return 0.0
    try:
        if "parcels" not in ds.groups:
            return 0.0
        grp = ds.groups["parcels"]
        if "parcels" not in grp.variables:
            return 0.0
        arr = np.asarray(grp.variables["parcels"][:])
        if arr.size == 0:
            return 0.0
        return float((arr != 0).sum()) / float(arr.size)
    finally:
        ds.close()


def _download_one(repo_path: str, out_dir: Path) -> Path:
    """Download a single patch into ``out_dir`` mirroring the repo layout.

    Args:
        repo_path: Repo-relative path (``data/<year>/<tile>/<patch>.nc``).
        out_dir: Local subset root.

    Returns:
        Local path to the downloaded ``.nc`` file.
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=S4A_REPO_ID,
        repo_type=S4A_REPO_TYPE,
        filename=repo_path,
        local_dir=str(out_dir),
    )
    return Path(local)


def _collect_patches(
    candidates: list[str],
    out_dir: Path,
    n_keep: int,
    min_parcel_frac: float,
) -> list[str]:
    """Download candidates one by one, keeping the first ``n_keep`` with parcels.

    Candidates that do not reach ``min_parcel_frac`` parcel coverage are deleted
    immediately so the subset stays small (R1 mitigation).

    Args:
        candidates: Repo-relative ``.nc`` paths to consider, in order.
        out_dir: Local subset root.
        n_keep: Number of patches to keep.
        min_parcel_frac: Minimum parcel coverage to keep a patch.

    Returns:
        Repo-relative paths of the kept patches (length ``<= n_keep``).
    """
    kept: list[str] = []
    for repo_path in candidates:
        if len(kept) >= n_keep:
            break
        local = _download_one(repo_path, out_dir)
        frac = _parcel_fraction(local)
        if frac >= min_parcel_frac:
            kept.append(repo_path)
            logger.info("s4a_patch_kept", patch=repo_path, parcel_frac=round(frac, 4))
        else:
            local.unlink(missing_ok=True)
            logger.info("s4a_patch_dropped_empty", patch=repo_path, parcel_frac=round(frac, 4))
    return kept


def _total_size_mb(out_dir: Path) -> float:
    """Sum the on-disk size of every ``.nc`` under ``out_dir`` in MB.

    Args:
        out_dir: Local subset root.

    Returns:
        Total size in megabytes (1e6 bytes), rounded to 2 decimals.
    """
    total = sum(p.stat().st_size for p in out_dir.rglob("*.nc"))
    return round(total / 1e6, 2)


def download_subset(
    out_dir: Path,
    *,
    cat_tile: str = DEFAULT_CAT_TILE,
    fr_tile: str = DEFAULT_FR_TILE,
    n_cat: int = 30,
    n_fr: int = 10,
    min_parcel_frac: float = 0.02,
) -> SubsetManifest:
    """Download a few-GB Catalonia + France subset of Sen4AgriNet.

    Catalonia patches come from ``cat_tile`` over 2019 and 2020 (interleaved so
    both years are represented); France patches from ``fr_tile`` over 2019. Only
    patches with parcel coverage ``>= min_parcel_frac`` are kept; the rest are
    deleted before the caller runs ``dvc add``.

    Args:
        out_dir: Local subset root (on the VM ``F:`` drive, never the laptop).
        cat_tile: Catalonia tile to sample (default ``31TCG``).
        fr_tile: France reference tile to sample (default ``31TCJ``).
        n_cat: Number of Catalonia patches to keep.
        n_fr: Number of France patches to keep.
        min_parcel_frac: Minimum parcel coverage to keep a patch.

    Returns:
        A :class:`SubsetManifest` describing what was kept and its total size.

    Raises:
        ValueError: if ``cat_tile`` / ``fr_tile`` are not valid S4A tiles.
    """
    if cat_tile not in CAT_TILES:
        raise ValueError(f"cat_tile {cat_tile!r} not in CAT_TILES {CAT_TILES}.")
    if fr_tile not in FR_TILES:
        raise ValueError(f"fr_tile {fr_tile!r} not in FR_TILES {FR_TILES}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Interleave Catalonia years so 2019 and 2020 are both represented.
    cat_by_year = {y: _list_tile_patches(cat_tile, y) for y in CAT_YEARS}
    cat_candidates: list[str] = []
    max_len = max((len(v) for v in cat_by_year.values()), default=0)
    for i in range(max_len):
        for y in CAT_YEARS:
            lst = cat_by_year[y]
            if i < len(lst):
                cat_candidates.append(lst[i])

    fr_candidates = _list_tile_patches(fr_tile, FR_YEARS[0])

    logger.info(
        "s4a_download_start",
        cat_tile=cat_tile,
        fr_tile=fr_tile,
        n_cat_candidates=len(cat_candidates),
        n_fr_candidates=len(fr_candidates),
        n_cat_target=n_cat,
        n_fr_target=n_fr,
    )

    cat_kept = _collect_patches(cat_candidates, out_dir, n_cat, min_parcel_frac)
    fr_kept = _collect_patches(fr_candidates, out_dir, n_fr, min_parcel_frac)

    manifest = SubsetManifest(
        repo_id=S4A_REPO_ID,
        cat_patches=cat_kept,
        fr_patches=fr_kept,
        min_parcel_frac=min_parcel_frac,
        total_size_mb=_total_size_mb(out_dir),
        n_cat=len(cat_kept),
        n_fr=len(fr_kept),
    )
    manifest_path = out_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    logger.info(
        "s4a_download_done",
        n_cat=manifest.n_cat,
        n_fr=manifest.n_fr,
        total_size_mb=manifest.total_size_mb,
        manifest=str(manifest_path),
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the downloader.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/sen4agrinet"),
        help="Local subset root (use F:/.../data/sen4agrinet on the VM).",
    )
    parser.add_argument("--cat-tile", default=DEFAULT_CAT_TILE, choices=CAT_TILES)
    parser.add_argument("--fr-tile", default=DEFAULT_FR_TILE, choices=FR_TILES)
    parser.add_argument("--n-cat", type=int, default=30)
    parser.add_argument("--n-fr", type=int, default=10)
    parser.add_argument("--min-parcel-frac", type=float, default=0.02)
    return parser


def main() -> None:
    """CLI entry point: download the subset and print the manifest."""
    args = _build_arg_parser().parse_args()
    manifest = download_subset(
        args.out_dir,
        cat_tile=args.cat_tile,
        fr_tile=args.fr_tile,
        n_cat=args.n_cat,
        n_fr=args.n_fr,
        min_parcel_frac=args.min_parcel_frac,
    )
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()
