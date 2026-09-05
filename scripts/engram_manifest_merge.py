"""Repair or merge ``.engram/manifest.json``, the index of shared engram memory chunks.

``engram sync --import`` only applies the chunks listed in the manifest, so two branches that
both exported memories collide on that JSON and a naive "take ours" resolution silently drops
the other side's chunk. This script unions the entries by chunk id, keeps the richest metadata
for each id, adds chunk files present on disk but unlisted, and drops entries whose file is
gone. It also serves as a git merge driver (``make memory-setup`` registers it).

In driver mode the merge is a true three-way one: an entry that BASE listed and that both sides
dropped stays dropped, and so does one dropped by a side while the other left it untouched. Only
a chunk that at least one side still lists (or still modified) survives, so a deliberate purge is
not resurrected on the next merge.

Usage:
    python scripts/engram_manifest_merge.py            # repair in place (conflict markers ok)
    python scripts/engram_manifest_merge.py --check    # verify only, exit 1 on drift
    python scripts/engram_manifest_merge.py --driver BASE OURS THEIRS   # git merge driver

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: git runs merge drivers from the repo root, so the working directory wins when it has .engram/.
ENGRAM = (Path.cwd() / ".engram") if (Path.cwd() / ".engram").is_dir() else ROOT / ".engram"
MANIFEST = ENGRAM / "manifest.json"
CHUNKS = ENGRAM / "chunks"

#: A chunk id is the stem of its ``<id>.jsonl.gz`` file. Deliberately permissive: pinning it to
#: eight lowercase hex digits would make a longer or uppercase id from a future engram version
#: invisible to the parser, and an entry this script cannot see is an entry it silently drops.
_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

#: Keys that tell a manifest entry apart from any other object nested in the JSON.
_ENTRY_KEYS = frozenset({"created_at", "created_by", "memories", "sessions", "prompts"})

#: Git conflict markers, so a manifest can still be read after a failed merge.
_CONFLICT = re.compile(r"^(<{7}|\|{7}|={7}|>{7})", re.M)


def _conflict_sides(text: str) -> list[str]:
    """Split a conflicted manifest into the two texts git was trying to reconcile.

    Both sides are returned so the union keeps every chunk; when there are no markers the text
    is returned unchanged.
    """
    if not _CONFLICT.search(text):
        return [text]
    ours: list[str] = []
    theirs: list[str] = []
    target = "both"
    for line in text.splitlines(keepends=True):
        if line.startswith("<<<<<<<"):
            target = "ours"
        elif line.startswith("|||||||"):
            target = "base"
        elif line.startswith("======="):
            target = "theirs"
        elif line.startswith(">>>>>>>"):
            target = "both"
        elif target in ("ours", "both"):
            ours.append(line)
            if target == "both":
                theirs.append(line)
        elif target == "theirs":
            theirs.append(line)
    return ["".join(ours), "".join(theirs)]


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` span, nested ones included, ignoring braces inside strings.

    Used only when the text does not parse as JSON: a brace scanner survives a truncated or
    hand-edited manifest that a regex over flat objects would silently skip.
    """
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            yield text[stack.pop() : index + 1]


def _richness(entry: dict[str, object]) -> int:
    """Count the fields of an entry that actually carry information."""
    return sum(1 for value in entry.values() if value not in ("", 0, None))


def _absorb(entries: dict[str, dict[str, object]], entry: dict[str, object]) -> None:
    """Keep ``entry`` when it is at least as informative as the one already held for its id."""
    chunk_id = entry.get("id")
    if not isinstance(chunk_id, str) or not _ID.fullmatch(chunk_id):
        return
    if not _ENTRY_KEYS & set(entry):
        return
    current = entries.get(chunk_id)
    if current is None or _richness(entry) >= _richness(current):
        entries[chunk_id] = entry


def _entries_from_text(text: str) -> dict[str, dict[str, object]]:
    """Extract chunk entries even from a manifest that carries git conflict markers."""
    entries: dict[str, dict[str, object]] = {}
    for side in _conflict_sides(text):
        if not side.strip():
            continue
        try:
            parsed = json.loads(side)
        except json.JSONDecodeError:
            candidates = []
            for span in _balanced_objects(side):
                try:
                    candidates.append(json.loads(span))
                except json.JSONDecodeError:
                    continue
        else:
            candidates = parsed.get("chunks", []) if isinstance(parsed, dict) else []
        for candidate in candidates:
            if isinstance(candidate, dict):
                _absorb(entries, candidate)
    return entries


def _entry_from_chunk(path: Path) -> dict[str, object]:
    """Rebuild a manifest entry from the chunk file itself when it was never listed."""
    sessions = observations = prompts = 0
    latest = ""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sessions += len(row.get("sessions", []))
                prompts += len(row.get("prompts", []))
                obs = row.get("observations", [])
                observations += len(obs)
                latest = max([latest, *[str(o.get("created_at", "")) for o in obs]])
    except (OSError, ValueError):
        pass
    created = latest.replace(" ", "T") + "Z" if latest else ""
    if not created:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        created = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": path.name.split(".", 1)[0],
        "created_by": "team",
        "created_at": created,
        "sessions": sessions,
        "memories": observations,
        "prompts": prompts,
    }


def _reconcile_with_disk(
    entries: dict[str, dict[str, object]],
    prune_missing: bool,
    deleted: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Add chunks present on disk, drop entries whose file is gone, and order the result.

    Args:
        entries: Entries surviving the merge, keyed by chunk id.
        prune_missing: Drop entries whose chunk file is absent. Off in merge-driver mode, where
            git may run the driver before the other side's new chunk file reaches the worktree.
        deleted: Ids the merge resolved as deliberately removed; never re-added from disk.
    """
    on_disk = {p.name.split(".", 1)[0]: p for p in CHUNKS.glob("*.jsonl.gz")}
    for chunk_id, path in on_disk.items():
        if chunk_id not in entries and chunk_id not in deleted:
            entries[chunk_id] = _entry_from_chunk(path)
    for chunk_id in list(entries):
        if prune_missing and chunk_id not in on_disk:
            print(f"aviso: chunk {chunk_id} listado sin archivo; se retira del manifest")
            del entries[chunk_id]
    ordered = sorted(entries.values(), key=lambda e: (str(e.get("created_at", "")), e["id"]))
    return {"version": 1, "chunks": ordered}


def union(*texts: str, prune_missing: bool = True) -> dict[str, object]:
    """Merge manifest texts into one manifest, reconciled against the chunk files on disk.

    Every text contributes its entries and the richest wins per id; nothing is dropped for being
    listed only by one text. Use :func:`three_way` when a BASE is available, so that deletions
    are honoured instead of resurrected.

    Args:
        texts: Manifest contents to union (conflict markers tolerated).
        prune_missing: Drop entries whose chunk file is absent.
    """
    entries: dict[str, dict[str, object]] = {}
    for text in texts:
        for entry in _entries_from_text(text).values():
            _absorb(entries, entry)
    return _reconcile_with_disk(entries, prune_missing)


def three_way(base_text: str, ours_text: str, theirs_text: str) -> dict[str, object]:
    """Reconcile two manifests against their common ancestor, honouring deletions.

    An id that BASE listed survives only when a side still lists it *and* that side did not
    simply keep the ancestor's copy while the other side removed it. That is git's own
    delete/unchanged rule, and it is what stops a purged chunk from coming back at every merge.
    BASE never contributes an entry of its own: folding it in is exactly how deletions were
    resurrected before.
    """
    base = _entries_from_text(base_text)
    ours = _entries_from_text(ours_text)
    theirs = _entries_from_text(theirs_text)

    entries: dict[str, dict[str, object]] = {}
    for side in (ours, theirs):
        for entry in side.values():
            _absorb(entries, entry)

    deleted: set[str] = set()
    for chunk_id, ancestor in base.items():
        in_ours, in_theirs = chunk_id in ours, chunk_id in theirs
        if in_ours and in_theirs:
            continue
        if not in_ours and not in_theirs:
            deleted.add(chunk_id)
        elif not in_ours and theirs[chunk_id] == ancestor:
            deleted.add(chunk_id)
        elif not in_theirs and ours[chunk_id] == ancestor:
            deleted.add(chunk_id)
    for chunk_id in deleted:
        entries.pop(chunk_id, None)
        print(f"aviso: chunk {chunk_id} retirado en la fusion (borrado deliberado)")

    return _reconcile_with_disk(entries, prune_missing=False, deleted=frozenset(deleted))


def _write(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main() -> int:
    """Repair, verify or drive-merge the manifest.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="solo verifica, exit 1 si hay drift")
    parser.add_argument(
        "--driver", nargs=3, metavar=("BASE", "OURS", "THEIRS"), help="modo merge driver de git"
    )
    args = parser.parse_args()

    if args.driver:
        base, ours, theirs = (Path(p) for p in args.driver)
        merged = three_way(_read(base), _read(ours), _read(theirs))
        _write(ours, merged)
        print(f"manifest fusionado: {len(merged['chunks'])} chunk(s)")
        return 0

    if not ENGRAM.exists():
        print("sin .engram/: nada que reparar")
        return 0
    current = _read(MANIFEST)
    merged = union(current)
    canonical = json.dumps(merged, indent=2) + "\n"
    if args.check:
        try:
            parsed = json.loads(current)
        except json.JSONDecodeError:
            print(
                "manifest ilegible o con marcadores de conflicto: "
                "python scripts/engram_manifest_merge.py lo repara"
            )
            return 1
        listed = [str(e.get("id")) for e in parsed.get("chunks", [])]
        expected = [str(e["id"]) for e in merged["chunks"]]
        if sorted(listed) != sorted(expected):
            print(f"drift: listados={sorted(listed)} en_disco={sorted(expected)}")
            print("reparar con: python scripts/engram_manifest_merge.py")
            return 1
        print(f"manifest consistente: {len(expected)} chunk(s)")
        return 0
    if current != canonical:
        _write(MANIFEST, merged)
        print(f"manifest reparado: {len(merged['chunks'])} chunk(s)")
    else:
        print(f"manifest ya consistente: {len(merged['chunks'])} chunk(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
