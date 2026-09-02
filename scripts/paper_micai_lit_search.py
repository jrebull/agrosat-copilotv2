"""Systematic literature search for the MICAI 2027 novelty assessment (phase 0).

Runs a fixed, versioned set of queries against the arXiv Atom API, the
Semantic Scholar Graph API and the OpenAlex API across the four fronts defined in
``docs/plan-micai-2027.md`` (phase 0, step 2), and seals both the raw responses
and a reproducible search log under ``reports/paper_micai/fase0/``.

Google Scholar has no public API and is queried by hand; its queries and dates
are logged in ``docs/paper/novedad.md``, not here.

Usage:
    poetry run python scripts/paper_micai_lit_search.py
    poetry run python scripts/paper_micai_lit_search.py --source arxiv
    poetry run python scripts/paper_micai_lit_search.py --source openalex
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from defusedxml import ElementTree as DefusedET

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "fase0"
RAW_DIR = OUT_DIR / "raw"

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
S2_FIELDS = "title,year,venue,externalIds,citationCount,abstract,authors"
DATE_RANGE_ARXIV = "submittedDate:[201901010000 TO 202612312359]"
YEAR_RANGE_S2 = "2019-2026"
OPENALEX_FILTER = "from_publication_date:2019-01-01,to_publication_date:2026-12-31"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Front keys mirror the four fronts of phase 0 in docs/plan-micai-2027.md.
FRONT_STACKING = "F1-stacking-arbitraje"
FRONT_CARDINALITY = "F2-cardinalidad-selectiva"
FRONT_CONTEXT = "F3-contexto-espacial-fm"
FRONT_COPILOT = "F4-copiloto-llm-eo"

ARXIV_QUERIES: list[tuple[str, str]] = [
    (FRONT_STACKING, 'all:"crop type classification" AND all:"ensemble"'),
    (FRONT_STACKING, 'all:"stacking" AND all:"satellite image time series"'),
    (FRONT_STACKING, 'all:"heterogeneous ensemble" AND all:"remote sensing"'),
    (FRONT_STACKING, 'all:"meta-learner" AND all:"land cover classification"'),
    (FRONT_STACKING, 'all:"crop mapping" AND all:"class imbalance"'),
    (FRONT_STACKING, 'all:"ensemble" AND all:"per-class" AND all:"crop classification"'),
    (FRONT_STACKING, 'all:"model selection" AND all:"ensemble" AND all:"Sentinel-2"'),
    (FRONT_STACKING, 'all:"stacking" AND all:"crop"'),
    (FRONT_STACKING, 'all:"soft voting" AND all:"land cover"'),
    (FRONT_CARDINALITY, 'all:"selective classification" AND all:"reject option"'),
    (FRONT_CARDINALITY, 'all:"classification with rejection" AND all:"remote sensing"'),
    (FRONT_CARDINALITY, 'all:"long-tailed" AND all:"land cover"'),
    (FRONT_CARDINALITY, 'all:"crop type" AND all:"taxonomy" AND all:"hierarchy"'),
    (FRONT_CARDINALITY, 'all:"accuracy" AND all:"coverage" AND all:"abstention"'),
    (FRONT_CARDINALITY, 'all:"reject curves" AND all:"classification"'),
    (FRONT_CARDINALITY, 'all:"label space" AND all:"granularity" AND all:"classification"'),
    (FRONT_CARDINALITY, 'all:"crop type" AND all:"rare classes"'),
    (FRONT_CONTEXT, 'all:"AlphaEarth"'),
    (FRONT_CONTEXT, 'all:"geospatial foundation model" AND all:"embeddings"'),
    (FRONT_CONTEXT, 'all:"spatial context" AND all:"crop type mapping"'),
    (FRONT_CONTEXT, 'all:"spatial cross-validation" AND all:"remote sensing"'),
    (FRONT_CONTEXT, 'all:"self-supervised" AND all:"satellite image time series"'),
    (FRONT_CONTEXT, 'all:"satellite embedding" AND all:"crop"'),
    (FRONT_CONTEXT, 'all:"conditional random field" AND all:"crop classification"'),
    (FRONT_CONTEXT, 'all:"spatial autocorrelation" AND all:"model evaluation"'),
    (FRONT_COPILOT, 'all:"large language model" AND all:"remote sensing" AND all:"agent"'),
    (FRONT_COPILOT, 'all:"geospatial" AND all:"tool" AND all:"LLM agent"'),
    (FRONT_COPILOT, 'all:"vision language model" AND all:"agriculture" AND all:"benchmark"'),
    (FRONT_COPILOT, 'all:"retrieval augmented generation" AND all:"geospatial"'),
    (FRONT_COPILOT, 'all:"multi-agent" AND all:"perception" AND all:"reasoning"'),
    (FRONT_COPILOT, 'all:"GIS agent" AND all:"benchmark"'),
    (FRONT_COPILOT, 'all:"tool-augmented" AND all:"spatial analysis"'),
]

S2_QUERIES: list[tuple[str, str]] = [
    (FRONT_STACKING, "stacking ensemble crop type classification satellite time series"),
    (FRONT_STACKING, "heterogeneous ensemble per-class arbitration land cover"),
    (FRONT_STACKING, "ensemble deep learning crop mapping Sentinel-2 imbalance"),
    (FRONT_CARDINALITY, "selective classification coverage risk trade-off"),
    (FRONT_CARDINALITY, "reject option land cover classification confidence threshold"),
    (FRONT_CARDINALITY, "crop type class hierarchy nomenclature harmonisation EuroCrops"),
    (FRONT_CONTEXT, "AlphaEarth foundation embeddings agriculture benchmark"),
    (FRONT_CONTEXT, "foundation model embeddings spatial transferability crop"),
    (FRONT_CONTEXT, "spatial autocorrelation validation machine learning ecology"),
    (FRONT_CONTEXT, "neighbourhood context features parcel crop classification postprocessing"),
    (FRONT_COPILOT, "LLM agent geospatial analysis benchmark tools"),
    (FRONT_COPILOT, "agricultural remote sensing multimodal benchmark"),
    (FRONT_COPILOT, "grounded language model earth observation copilot"),
    (FRONT_COPILOT, "hallucination reduction retrieval grounding geospatial question answering"),
]


OPENALEX_QUERIES: list[tuple[str, str]] = [
    (FRONT_STACKING, "stacking ensemble crop type classification remote sensing"),
    (FRONT_STACKING, "heterogeneous ensemble learning land cover classification meta-learner"),
    (FRONT_STACKING, "ensemble crop mapping satellite image time series deep learning"),
    (FRONT_STACKING, "class imbalance macro F1 crop type classification"),
    (FRONT_CARDINALITY, "number of crop classes legend accuracy trade-off mapping"),
    (FRONT_CARDINALITY, "selective classification reject option remote sensing"),
    (FRONT_CARDINALITY, "crop type nomenclature hierarchy harmonisation EuroCrops HCAT"),
    (FRONT_CARDINALITY, "rare crop classes long tail parcel classification"),
    (FRONT_CONTEXT, "AlphaEarth Foundations embeddings evaluation"),
    (FRONT_CONTEXT, "spatial context object based post-classification smoothing crop map"),
    (FRONT_CONTEXT, "spatial cross-validation spatial autocorrelation model evaluation"),
    (FRONT_CONTEXT, "foundation model embeddings crop type classification transferability"),
    (FRONT_COPILOT, "large language model agent geospatial analysis benchmark"),
    (FRONT_COPILOT, "vision language model agriculture remote sensing benchmark"),
    (FRONT_COPILOT, "retrieval augmented generation grounding hallucination geospatial"),
]


def _slug(text: str) -> str:
    """Build a filesystem-safe slug from a query string.

    Args:
        text: Raw query string.

    Returns:
        Lowercase slug with only alphanumerics and hyphens, capped at 60 chars.
    """
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:60]


def _fetch_arxiv(client: httpx.Client, query: str, max_results: int) -> tuple[int, list[dict]]:
    """Query the arXiv Atom API and normalise the entries.

    Args:
        client: Shared HTTP client.
        query: arXiv ``search_query`` expression (date range is appended here).
        max_results: Maximum number of entries requested.

    Returns:
        Tuple of HTTP status code and the list of normalised records.
    """
    params = {
        "search_query": f"({query}) AND {DATE_RANGE_ARXIV}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = client.get(ARXIV_ENDPOINT, params=params, timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return response.status_code, []
    root = DefusedET.fromstring(response.text)
    records: list[dict] = []
    for entry in root.findall("a:entry", ATOM_NS):
        arxiv_url = entry.findtext("a:id", "", ATOM_NS)
        doi_node = entry.find("arxiv:doi", ATOM_NS)
        records.append(
            {
                "arxiv_id": arxiv_url.rsplit("/", 1)[-1],
                "title": " ".join(entry.findtext("a:title", "", ATOM_NS).split()),
                "authors": [
                    a.findtext("a:name", "", ATOM_NS) for a in entry.findall("a:author", ATOM_NS)
                ],
                "published": entry.findtext("a:published", "", ATOM_NS),
                "doi": doi_node.text if doi_node is not None else None,
                "summary": " ".join(entry.findtext("a:summary", "", ATOM_NS).split()),
            }
        )
    return response.status_code, records


def _fetch_s2(client: httpx.Client, query: str, limit: int) -> tuple[int, list[dict]]:
    """Query the Semantic Scholar Graph API with retries on rate limiting.

    Args:
        client: Shared HTTP client.
        query: Natural-language query.
        limit: Maximum number of papers requested.

    Returns:
        Tuple of the last HTTP status code and the list of returned papers.
    """
    params = {"query": query, "fields": S2_FIELDS, "limit": str(limit), "year": YEAR_RANGE_S2}
    status = 0
    for attempt in range(5):
        response = client.get(S2_ENDPOINT, params=params, timeout=60.0)
        status = response.status_code
        if status == httpx.codes.OK:
            return status, response.json().get("data", [])
        logger.warning("s2_retry", query=query, status=status, attempt=attempt)
        time.sleep(8 * (attempt + 1))
    return status, []


def _fetch_openalex(client: httpx.Client, query: str, limit: int) -> tuple[int, list[dict]]:
    """Query the OpenAlex works endpoint restricted to the 2019-2026 window.

    OpenAlex covers the journal literature (Remote Sensing, ISPRS, JAG) that
    arXiv does not index and answers without an API key, which makes it the
    practical substitute when Semantic Scholar throttles its shared pool.

    Args:
        client: Shared HTTP client.
        query: Natural-language query.
        limit: Maximum number of works requested.

    Returns:
        Tuple of HTTP status code and the list of normalised works.
    """
    params = {
        "search": query,
        "filter": OPENALEX_FILTER,
        "per-page": str(limit),
        "mailto": "rebull@outlook.com",
    }
    response = client.get(OPENALEX_ENDPOINT, params=params, timeout=60.0)
    if response.status_code != httpx.codes.OK:
        return response.status_code, []
    works: list[dict] = []
    for item in response.json().get("results", []):
        authorships = item.get("authorships") or []
        source = (item.get("primary_location") or {}).get("source") or {}
        works.append(
            {
                "openalex_id": (item.get("id") or "").rsplit("/", 1)[-1],
                "title": item.get("display_name") or "",
                "authors": [(a.get("author") or {}).get("display_name", "") for a in authorships],
                "year": item.get("publication_year"),
                "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                "venue": source.get("display_name") or "",
                "cited_by_count": item.get("cited_by_count"),
            }
        )
    return response.status_code, works


def _write_raw(path: Path, payload: dict[str, Any]) -> None:
    """Write a raw query payload unless a successful one is already stored.

    Args:
        path: Destination JSON path.
        payload: Payload of the current run.
    """
    if path.exists() and payload["status"] != int(httpx.codes.OK):
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("status") == int(httpx.codes.OK):
            return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_rows(
    path: Path, rows: list[dict[str, Any]], header: list[str], keys: tuple[str, ...]
) -> None:
    """Merge new rows into an existing CSV without ever losing a successful answer.

    Running the arXiv and Semantic Scholar passes separately must not truncate
    the log of the other source, so previous rows whose key is not re-executed
    in this run are preserved verbatim. A re-execution that got throttled must
    not overwrite the answer a previous run already obtained either: when the
    stored row carries HTTP 200 and the fresh one does not, the stored row wins.

    Args:
        path: Destination CSV path.
        rows: Rows produced by the current run.
        header: Column order.
        keys: Column names that identify a row.
    """
    fresh_by_key = {tuple(row[k] for k in keys): row for row in rows}
    kept: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for old_row in csv.DictReader(handle):
                key = tuple(old_row.get(k, "") for k in keys)
                fresh_row = fresh_by_key.get(key)
                if fresh_row is None:
                    kept.append(old_row)
                    continue
                old_ok = old_row.get("http_status") == str(int(httpx.codes.OK))
                new_ok = str(fresh_row.get("http_status")) == str(int(httpx.codes.OK))
                if old_ok and not new_ok:
                    kept.append(old_row)
                    fresh_by_key.pop(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(kept + list(fresh_by_key.values()))


def main() -> None:
    """Run every configured query and seal the raw payloads plus the search log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["arxiv", "s2", "openalex", "all"], default="all")
    parser.add_argument("--max-results", type=int, default=25)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    executed_at = datetime.now(UTC).isoformat(timespec="seconds")
    log_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    headers = {"User-Agent": "agrosat-micai-lit-search/1.0 (mailto:rebull@outlook.com)"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        if args.source in {"arxiv", "all"}:
            for front, query in ARXIV_QUERIES:
                status, records = _fetch_arxiv(client, query, args.max_results)
                name = f"arxiv_{front}_{_slug(query)}.json"
                (RAW_DIR / name).write_text(
                    json.dumps(
                        {
                            "front": front,
                            "source": "arxiv",
                            "query": query,
                            "executed_at": executed_at,
                            "status": status,
                            "records": records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                log_rows.append(
                    {
                        "front": front,
                        "source": "arxiv",
                        "query": query,
                        "executed_at": executed_at,
                        "http_status": status,
                        "n_results": len(records),
                        "raw_file": name,
                    }
                )
                for record in records:
                    candidate_rows.append(
                        {
                            "front": front,
                            "source": "arxiv",
                            "query": query,
                            "id": record["arxiv_id"],
                            "doi": record["doi"] or "",
                            "year": record["published"][:4],
                            "title": record["title"],
                            "first_author": (record["authors"] or [""])[0],
                        }
                    )
                logger.info("arxiv_query", front=front, query=query, n=len(records))
                time.sleep(3.5)

        if args.source in {"s2", "all"}:
            for front, query in S2_QUERIES:
                status, papers = _fetch_s2(client, query, args.max_results)
                name = f"s2_{front}_{_slug(query)}.json"
                (RAW_DIR / name).write_text(
                    json.dumps(
                        {
                            "front": front,
                            "source": "semanticscholar",
                            "query": query,
                            "executed_at": executed_at,
                            "status": status,
                            "records": papers,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                log_rows.append(
                    {
                        "front": front,
                        "source": "semanticscholar",
                        "query": query,
                        "executed_at": executed_at,
                        "http_status": status,
                        "n_results": len(papers),
                        "raw_file": name,
                    }
                )
                for paper in papers:
                    external = paper.get("externalIds") or {}
                    authors = paper.get("authors") or []
                    candidate_rows.append(
                        {
                            "front": front,
                            "source": "semanticscholar",
                            "query": query,
                            "id": external.get("ArXiv", "") or external.get("CorpusId", ""),
                            "doi": external.get("DOI", "") or "",
                            "year": str(paper.get("year") or ""),
                            "title": paper.get("title") or "",
                            "first_author": (authors[0].get("name") if authors else "") or "",
                        }
                    )
                logger.info("s2_query", front=front, query=query, n=len(papers))
                time.sleep(3.5)

        if args.source in {"openalex", "all"}:
            for front, query in OPENALEX_QUERIES:
                status, works = _fetch_openalex(client, query, args.max_results)
                name = f"openalex_{front}_{_slug(query)}.json"
                (RAW_DIR / name).write_text(
                    json.dumps(
                        {
                            "front": front,
                            "source": "openalex",
                            "query": query,
                            "executed_at": executed_at,
                            "status": status,
                            "records": works,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                log_rows.append(
                    {
                        "front": front,
                        "source": "openalex",
                        "query": query,
                        "executed_at": executed_at,
                        "http_status": status,
                        "n_results": len(works),
                        "raw_file": name,
                    }
                )
                for work in works:
                    candidate_rows.append(
                        {
                            "front": front,
                            "source": "openalex",
                            "query": query,
                            "id": work["openalex_id"],
                            "doi": work["doi"],
                            "year": str(work["year"] or ""),
                            "title": work["title"],
                            "first_author": (work["authors"] or [""])[0],
                        }
                    )
                logger.info("openalex_query", front=front, query=query, n=len(works))
                time.sleep(1.5)

    _merge_rows(
        OUT_DIR / "search_log.csv",
        log_rows,
        ["front", "source", "query", "executed_at", "http_status", "n_results", "raw_file"],
        ("front", "source", "query"),
    )
    _merge_rows(
        OUT_DIR / "search_candidates.csv",
        candidate_rows,
        ["front", "source", "query", "id", "doi", "year", "title", "first_author"],
        ("front", "source", "query", "id", "title"),
    )
    logger.info(
        "search_done", queries=len(log_rows), candidates=len(candidate_rows), out=str(OUT_DIR)
    )


if __name__ == "__main__":
    main()
