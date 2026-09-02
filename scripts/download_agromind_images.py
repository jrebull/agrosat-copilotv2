"""Download ONLY the AgroMind images referenced by the 500-item subset (US-049).

The full AgroMind image set is ~9.4 GB across 10 category zips on HuggingFace
(``AgroMind/AgroMind``). The benchmark subset (``agromind_subset_500.json``)
references only a small slice of those images: each item may have a base
``image_path`` and, for the multi-image items, image-valued options. This script

1. reads the subset and collects every unique referenced relative path,
2. maps each path to its source zip by its first segment
   (``./Rural/piece_images/x.png`` -> ``Rural.zip``),
3. downloads from HuggingFace ONLY the zips that the subset actually needs
   (``hf_hub_download``), and
4. extracts ONLY the referenced members into ``data/agromind/images/``.

It is idempotent: already-extracted images are skipped, and a zip is only
fetched when at least one of its members is still missing. The HuggingFace token
is read from the environment (``HUGGINGFACE_TOKEN`` / ``HF_TOKEN``); the dataset
is public so the token is optional but honoured when present.

Usage:
    poetry run python scripts/download_agromind_images.py
    poetry run python scripts/download_agromind_images.py --subset path --dest dir

Project conventions: identifiers and docstrings in English (Google style),
visible CLI prose in Spanish; ``structlog`` for logging; full type hints; no
emojis; ``print`` only for the final operator-facing summary line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_DEST",
    "DEFAULT_REPO_ID",
    "DEFAULT_SUBSET_PATH",
    "collect_referenced_paths",
    "download_subset_images",
    "group_by_zip",
    "main",
]

#: HuggingFace dataset repo holding the AgroMind image zips.
DEFAULT_REPO_ID: str = "AgroMind/AgroMind"

#: Default location of the 500-item subset JSON.
DEFAULT_SUBSET_PATH: Path = Path("data/agromind/agromind_subset_500.json")

#: Default destination root where images are extracted (mirrors the relative
#: ``./Category/...`` layout the subset references).
DEFAULT_DEST: Path = Path("data/agromind/images")

#: Image-file suffixes recognised when collecting referenced paths.
_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _is_image_path(value: object) -> bool:
    """Return whether a value is an image path (vs. plain answer text).

    Args:
        value: A candidate option value or image-path string.

    Returns:
        ``True`` when ``value`` is a string ending in a known image suffix.
    """
    return isinstance(value, str) and value.lower().endswith(_IMAGE_SUFFIXES)


def _normalize(rel_path: str) -> str:
    """Normalise a subset relative path to ``Category/.../file`` form.

    Strips the leading ``./`` and back-slashes so the path can be used both as a
    zip-member suffix and as the on-disk destination under the image root.

    Args:
        rel_path: The raw relative path from the subset.

    Returns:
        The cleaned forward-slash relative path (no leading ``./``).
    """
    return rel_path.lstrip("./").replace("\\", "/")


def collect_referenced_paths(subset_path: Path) -> set[str]:
    """Collect every unique image path referenced by the subset.

    Gathers each item's base ``image_path`` (when present) and every
    image-valued option, normalised to ``Category/.../file`` form.

    Args:
        subset_path: Path to ``agromind_subset_500.json``.

    Returns:
        The set of unique normalised relative image paths.
    """
    raw = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    paths: set[str] = set()
    for record in raw:
        image_path = record.get("image_path")
        if isinstance(image_path, str) and image_path.strip():
            paths.add(_normalize(image_path))
        for value in (record.get("options") or {}).values():
            if _is_image_path(value):
                paths.add(_normalize(value))
    logger.info("agromind_referenced_paths", subset=str(subset_path), n_paths=len(paths))
    return paths


def group_by_zip(paths: set[str]) -> dict[str, set[str]]:
    """Group referenced paths by their source zip (first path segment).

    A path ``Rural/piece_images/x.png`` belongs to ``Rural.zip``. Paths without
    a category segment are skipped (logged) since they cannot be mapped.

    Args:
        paths: The normalised relative paths from :func:`collect_referenced_paths`.

    Returns:
        A mapping ``{zip_filename: {relative_path, ...}}``.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        segments = path.split("/", 1)
        if len(segments) < 2 or not segments[0]:
            logger.warning("agromind_unmappable_path", path=path)
            continue
        grouped[f"{segments[0]}.zip"].add(path)
    return dict(grouped)


def _missing_members(members: set[str], dest: Path) -> set[str]:
    """Return the subset of members not yet extracted under ``dest``.

    Args:
        members: Normalised relative paths expected under ``dest``.
        dest: The image destination root.

    Returns:
        The members whose destination file does not yet exist.
    """
    return {m for m in members if not (dest / m).exists()}


def _member_matches(name: str, wanted: str) -> bool:
    """Return whether a zip member name corresponds to a wanted relative path.

    Zip members may or may not carry the leading category directory and use
    either slash style, so matching is by normalised suffix.

    Args:
        name: A zip member name from ``namelist()``.
        wanted: The wanted normalised relative path (``Category/.../file``).

    Returns:
        ``True`` when ``name`` ends with ``wanted`` (slash-normalised).
    """
    normalized_name = name.replace("\\", "/").lstrip("./")
    return normalized_name == wanted or normalized_name.endswith("/" + wanted)


def _extract_members(zip_path: Path, wanted: set[str], dest: Path) -> int:
    """Extract only the wanted members from a downloaded zip into ``dest``.

    Each extracted member is written at ``dest / wanted`` (the subset-relative
    layout), regardless of whether the in-zip member carried the leading
    category directory.

    Args:
        zip_path: Local path to the downloaded category zip.
        wanted: The normalised relative paths to extract.
        dest: The image destination root.

    Returns:
        The number of members actually extracted.
    """
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for target in wanted:
            out_path = dest / target
            if out_path.exists():
                continue
            match = next((n for n in names if _member_matches(n, target)), None)
            if match is None:
                logger.warning("agromind_member_not_in_zip", zip=zip_path.name, member=target)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(match) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            extracted += 1
    return extracted


def download_subset_images(
    subset_path: Path,
    dest: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
) -> dict[str, int]:
    """Download and extract only the subset-referenced AgroMind images.

    For every source zip that still has missing members, downloads it from
    HuggingFace with ``hf_hub_download`` (cached) and extracts only the needed
    members. Idempotent: a zip whose members are all already on disk is not
    fetched, and existing files are never re-extracted.

    Args:
        subset_path: Path to the AgroMind subset JSON.
        dest: Destination image root.
        repo_id: HuggingFace dataset repo id holding the zips.
        token: Optional HuggingFace token (public dataset, so optional).

    Returns:
        A mapping ``{zip_filename: n_extracted}`` for the zips that were
        processed (zips fully satisfied from disk map to ``0``).
    """
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    paths = collect_referenced_paths(subset_path)
    grouped = group_by_zip(paths)

    per_zip_extracted: dict[str, int] = {}
    for zip_name, members in sorted(grouped.items()):
        missing = _missing_members(members, dest)
        if not missing:
            logger.info("agromind_zip_complete_on_disk", zip=zip_name, n_members=len(members))
            per_zip_extracted[zip_name] = 0
            continue
        logger.info(
            "agromind_zip_download_started",
            zip=zip_name,
            n_missing=len(missing),
            n_total=len(members),
        )
        local_zip = hf_hub_download(
            repo_id=repo_id,
            filename=zip_name,
            repo_type="dataset",
            token=token,
        )
        extracted = _extract_members(Path(local_zip), missing, dest)
        per_zip_extracted[zip_name] = extracted
        logger.info("agromind_zip_extracted", zip=zip_name, n_extracted=extracted)

    return per_zip_extracted


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success, ``1`` on failure).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Descarga SOLO las imagenes del subset 500 de AgroMind desde los "
            "zips de HuggingFace (idempotente; US-049)."
        )
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET_PATH,
        help="Ruta al subset JSON de AgroMind.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Carpeta de salida de las imagenes extraidas.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Repo de HuggingFace con los zips de AgroMind.",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        per_zip = download_subset_images(args.subset, args.dest, repo_id=args.repo_id, token=token)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        logger.error("agromind_image_download_failed", error=str(exc))
        return 1

    total = sum(per_zip.values())
    zips_used = sorted(z for z, n in per_zip.items() if n > 0)
    print(
        f"Extraidas {total} imagenes del subset a {args.dest} "
        f"desde {len(zips_used)} zips: {zips_used}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
