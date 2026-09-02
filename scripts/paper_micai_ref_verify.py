"""Verify every reference of the MICAI 2027 related-work matrix against public APIs.

Reads a curated CSV whose rows carry an arXiv identifier, a DOI or a title to
resolve, queries arXiv, Crossref or OpenAlex accordingly, and writes back the
canonical title, authors and year exactly as the API returns them. No metadata
is ever taken from memory: a row without an API answer is marked NOT_FOUND and
must be dropped from the paper.

Usage:
    poetry run python scripts/paper_micai_ref_verify.py \
        --input reports/paper_micai/fase0/related_work_matrix.csv \
        --output reports/paper_micai/fase0/related_work_verified.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from defusedxml import ElementTree as DefusedET

logger = structlog.get_logger(__name__)

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MATCH_THRESHOLD = 0.72


def _normalise(text: str) -> str:
    """Lowercase a title and strip everything that is not a word character.

    Args:
        text: Raw title.

    Returns:
        Normalised title used for the similarity check.
    """
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _token_overlap(left: str, right: str) -> float:
    """Compute the Jaccard overlap between the token sets of two titles.

    Args:
        left: First title.
        right: Second title.

    Returns:
        Overlap in ``[0, 1]``; ``0.0`` when either title is empty.
    """
    a = set(_normalise(left).split())
    b = set(_normalise(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _arxiv_lookup(client: httpx.Client, arxiv_id: str) -> dict[str, Any] | None:
    """Resolve an arXiv identifier through the Atom API.

    Args:
        client: Shared HTTP client.
        arxiv_id: Identifier such as ``2505.12207``.

    Returns:
        Dictionary with title, authors, year and DOI, or ``None`` when absent.
    """
    response = client.get(ARXIV_ENDPOINT, params={"id_list": arxiv_id}, timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return None
    root = DefusedET.fromstring(response.text)
    entry = root.find("a:entry", ATOM_NS)
    if entry is None:
        return None
    title = " ".join(entry.findtext("a:title", "", ATOM_NS).split())
    if not title or title == "Error":
        return None
    doi_node = entry.find("arxiv:doi", ATOM_NS)
    return {
        "api": "arxiv",
        "title": title,
        "authors": "; ".join(
            a.findtext("a:name", "", ATOM_NS) for a in entry.findall("a:author", ATOM_NS)
        ),
        "year": entry.findtext("a:published", "", ATOM_NS)[:4],
        "doi": doi_node.text if doi_node is not None else "",
        "venue": "arXiv",
    }


def _crossref_by_doi(client: httpx.Client, doi: str) -> dict[str, Any] | None:
    """Resolve a DOI through the Crossref REST API.

    Args:
        client: Shared HTTP client.
        doi: Digital object identifier.

    Returns:
        Dictionary with the canonical metadata, or ``None`` when unresolved.
    """
    response = client.get(f"{CROSSREF_ENDPOINT}/{doi}", timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return None
    item = response.json().get("message", {})
    return _crossref_item(item)


def _crossref_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one Crossref work record.

    Args:
        item: Raw Crossref ``message`` (or one element of ``message.items``).

    Returns:
        Normalised dictionary, or ``None`` when the record has no title.
    """
    titles = item.get("title") or []
    if not titles:
        return None
    authors = "; ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (item.get("author") or [])
    )
    date_parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    return {
        "api": "crossref",
        "title": " ".join(titles[0].split()),
        "authors": authors,
        "year": str(date_parts[0][0] or ""),
        "doi": item.get("DOI", ""),
        "venue": (item.get("container-title") or [""])[0],
    }


def _crossref_by_title(client: httpx.Client, title: str) -> dict[str, Any] | None:
    """Search Crossref by bibliographic title and keep the best match.

    Args:
        client: Shared HTTP client.
        title: Title to search for.

    Returns:
        Normalised metadata when the overlap clears the threshold, else ``None``.
    """
    params = {"query.bibliographic": title, "rows": "5"}
    response = client.get(CROSSREF_ENDPOINT, params=params, timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for item in response.json().get("message", {}).get("items", []):
        record = _crossref_item(item)
        if record is None:
            continue
        score = _token_overlap(title, record["title"])
        if score > best_score:
            best, best_score = record, score
    if best is not None and best_score >= MATCH_THRESHOLD:
        best["match_score"] = round(best_score, 3)
        return best
    return None


def _openalex_by_title(client: httpx.Client, title: str) -> dict[str, Any] | None:
    """Search OpenAlex by title as a fallback for works Crossref does not index.

    Args:
        client: Shared HTTP client.
        title: Title to search for.

    Returns:
        Normalised metadata when the overlap clears the threshold, else ``None``.
    """
    params = {"search": title, "per-page": "5", "mailto": "rebull@outlook.com"}
    response = client.get(OPENALEX_ENDPOINT, params=params, timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for item in response.json().get("results", []):
        found = item.get("display_name") or ""
        score = _token_overlap(title, found)
        if score > best_score:
            authorships = item.get("authorships") or []
            best = {
                "api": "openalex",
                "title": found,
                "authors": "; ".join(
                    (a.get("author") or {}).get("display_name", "") for a in authorships
                ),
                "year": str(item.get("publication_year") or ""),
                "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                "venue": ((item.get("primary_location") or {}).get("source") or {}).get(
                    "display_name", ""
                )
                or "",
            }
            best_score = score
    if best is not None and best_score >= MATCH_THRESHOLD:
        best["match_score"] = round(best_score, 3)
        return best
    return None


def _resolve(client: httpx.Client, row: dict[str, str]) -> dict[str, Any]:
    """Resolve one matrix row through the API that matches its identifier type.

    Args:
        client: Shared HTTP client.
        row: Matrix row with ``id_type``, ``id`` and ``title_claimed``.

    Returns:
        The row enriched with the API answer and a verification status.
    """
    id_type = (row.get("id_type") or "").strip().lower()
    identifier = (row.get("id") or "").strip()
    claimed = (row.get("title_claimed") or "").strip()
    record: dict[str, Any] | None = None
    if id_type == "arxiv" and identifier:
        record = _arxiv_lookup(client, identifier)
    elif id_type == "doi" and identifier:
        record = _crossref_by_doi(client, identifier)
    if record is None and claimed:
        record = _crossref_by_title(client, claimed) or _openalex_by_title(client, claimed)
    out = dict(row)
    out["checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    if record is None:
        out.update(
            {
                "api": "",
                "title_verified": "",
                "authors_verified": "",
                "year_verified": "",
                "doi_verified": "",
                "venue_verified": "",
                "status": "NOT_FOUND",
            }
        )
        return out
    score = _token_overlap(claimed, record["title"]) if claimed else 1.0
    out.update(
        {
            "api": record["api"],
            "title_verified": record["title"],
            "authors_verified": record["authors"],
            "year_verified": record["year"],
            "doi_verified": record.get("doi", ""),
            "venue_verified": record.get("venue", ""),
            "status": "OK" if score >= MATCH_THRESHOLD else "TITLE_MISMATCH",
        }
    )
    return out


def main() -> None:
    """Verify every row of the input matrix and write the resolved CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    headers = {"User-Agent": "agrosat-micai-ref-verify/1.0 (mailto:rebull@outlook.com)"}
    resolved: list[dict[str, Any]] = []
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for row in rows:
            out = _resolve(client, row)
            resolved.append(out)
            logger.info("verified", key=row.get("key"), status=out["status"], api=out["api"])
            time.sleep(1.5)

    fieldnames = [
        *rows[0].keys(),
        "checked_at",
        "api",
        "title_verified",
        "authors_verified",
        "year_verified",
        "doi_verified",
        "venue_verified",
        "status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resolved)
    n_bad = sum(1 for r in resolved if r["status"] != "OK")
    logger.info("verify_done", total=len(resolved), not_ok=n_bad, out=str(args.output))


if __name__ == "__main__":
    main()
