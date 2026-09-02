"""Paper-track multi-benchmark eval orchestrator for AgroSatCopilot (US-069).

This module is a **SIBLING** harness to :mod:`ml.eval.agent_bench` (it imports
that module's public pieces but never mutates them, exactly like
:mod:`ml.eval.agent_system_eval`). Where ``agent_bench`` scores the copilot on
AgroMind + GeoAnalystBench, ``paper_bench`` produces the paper's headline table:
the two reasoner variants -- **Gemini 2.5-pro** (cloud, GA, 1M ctx) and
**Qwen3-30B-A3B** (on-prem vLLM, GPTQ-Int4, single-GPU; data sovereignty) --
evaluated over **three** benchmarks:

- **GEO-Bench-2** -- the ServiceNow EO vision benchmark (successor of GEO-Bench
  2023), filtered to the agricultural tasks (crop classification / land-cover
  with crop classes). The VLM names the crop/land-cover class of a tile;
  exact-match + macro-F1 vs the gold label. NOT GeoAnalystBench (that is a
  plan-and-react code benchmark, already covered by ``agent_bench``).
- **AgroMind** -- the real eval-only subset (~100% visual, no train split; any
  fine-tune would be leakage). Scored by reusing ``agent_bench.eval_agromind``
  verbatim, so AgroMind never diverges between the two harnesses.
- **AgroMind-IT/ES** -- the project's own bilingual benchmark (US-068, 250 it +
  250 es, eval-only). Same QA contract as AgroMind, scored per language.

Statistical rigour (AC): each (variant, benchmark) is run over 3 seeds and
aggregated to ``mean +- std`` (reusing ``agent_bench._aggregate``); the paired
**Wilcoxon signed-rank** test compares Gemini vs Qwen on the per-item scores
(:func:`wilcoxon_paired`). The real p-value is reported -- with a small ``n`` it
may not reach significance, and that is reported honestly, never asserted as
"significant" without the test. Per-variant metrics: accuracy, F1, BERTScore,
tool-call accuracy, hallucination rate, latency p50/p95 and cost/query (all
reusing :mod:`ml.eval.agent_metrics` plus the local timing/cost helpers).

**Be My Eyes pattern** (Huang et al. 2025, arXiv:2511.19417): the reasoner is
FROZEN -- the perceiver is our segmentation models, the reasoner is the LLM that
communicates/explains. This benchmark measures communication/reasoning, NOT a
pixel classifier; the defensible reference targets are AgroMind >= 0.75 Gemini /
>= 0.70 Qwen (we do NOT claim a fine-tuned VLM beats Gemini). AlphaEarth is the
``SATELLITE_EMBEDDING/V1/ANNUAL`` data v1.1 64-dim GEE asset (CC-BY-4.0).

Degrade-clean contract (Arthur's zero-synthetic rule): when a benchmark dataset
is absent (GEO-Bench-2 not downloaded yet; AgroMind-IT/ES not delivered by
US-068) the loader returns ``[]`` and the run marks that (variant, benchmark)
cell ``pending`` -- it NEVER fabricates a score. When no model endpoint is
reachable the backend is injectable for tests (zero network); the real H100 run
of Qwen vLLM is the documented blocker (``docs/blockers/epic11-notas.md``).

Project conventions: identifiers and docstrings in English (Google style);
visible prose (CLI help, the LaTeX caption) in Spanish; ``structlog`` (never
``print`` in logic); full type hints; no emojis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from ml.eval import agent_metrics
from ml.eval.agent_bench import (
    DEFAULT_AGROMIND_PATH,
    DEFAULT_IMAGE_ROOT,
    ReasonerVariant,
    _aggregate,
    _is_nan,
    _resolve_backend,
    _run_backend_text,
    _save_checkpoint,
    eval_agromind,
    load_agromind_subset,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

    from ml.agent.backends import LLMBackend
    from ml.eval.agent_metrics import HallucinationJudge

logger = structlog.get_logger(__name__)

__all__ = [
    "BENCHMARKS",
    "DEFAULT_GEOBENCH_ROOT",
    "DEFAULT_ITSES_PATH",
    "PAPER_VARIANTS",
    "AgroMindItEsItem",
    "BenchmarkScores",
    "GeoBenchItem",
    "GeoBenchTask",
    "WilcoxonResult",
    "eval_agromind_itses",
    "eval_geobench2",
    "export_latex_table",
    "load_agromind_itses",
    "load_geobench2",
    "macro_f1",
    "main",
    "run_paper_benchmark",
    "wilcoxon_paired",
]

#: The three benchmark axes of the paper table (AC), in table order.
BENCHMARKS: tuple[str, ...] = ("GEO-Bench-2", "AgroMind", "AgroMind-IT/ES")

#: Default root where the GEO-Bench-2 agricultural subset is materialised (DVC).
#: Absent in a fresh checkout -> :func:`load_geobench2` returns ``[]`` (pending).
DEFAULT_GEOBENCH_ROOT: Path = Path("data/geobench2")

#: Default path of the AgroMind-IT/ES bilingual JSONL (US-068). Absent until
#: US-068 (native human review) delivers it -> the cell is marked pending. The
#: real delivered seed lives at ``data/benchmark/agromind_it_es/seed.jsonl`` and is
#: passed via ``--itses-path``.
DEFAULT_ITSES_PATH: Path = Path("data/agromind_itses/agromind_itses_500.jsonl")

#: Default base folder for the AgroMind-IT/ES Sentinel-2 tiles. The US-068 seed
#: references images by bare filename (``classification_0000.png``) that live in
#: ``data/s2_italia``, NOT under the AgroMind image root, so the IT/ES evaluator
#: resolves against this root independently of :data:`DEFAULT_IMAGE_ROOT`.
DEFAULT_ITSES_IMAGE_ROOT: Path = Path("data/s2_italia")

#: The two paper reasoner variants: the cloud Gemini reasoner and Qwen3-30B-A3B
#: (on-prem vLLM, GPTQ-Int4, single-GPU). ``multimodal`` gates whether a variant
#: may consume the GEO-Bench-2 / AgroMind tiles; the on-prem text Qwen is
#: text-only and skips image-only items (reported, never papered over). Gemma 4
#: 26B base-only is OUT of the headline pair (its LoRA is OUT, ADR-009) and can be
#: added via the CLI from ``agent_bench.DEFAULT_VARIANTS``.
#:
#: BENCHMARK MODEL CHOICE (US-069, documented deviation): the manuscript prose
#: cites Gemini **2.5-pro** (GA, 1M ctx) as the cloud reasoner, but the eval
#: BENCHMARK is run with **gemini-2.5-flash**. Flash is much faster and returns
#: far fewer ``504 DEADLINE_EXCEEDED`` on multimodal (image) items -- 2.5-pro over
#: images repeatedly deadlined and stalled the run (it spent ~$0 because the very
#: first multimodal call 504'd). The reported column is therefore flash; this is
#: stated in the LaTeX caption so the table never implies the pro model produced
#: the numbers. The pro model can still be forced via
#: ``--variants`` + a custom ReasonerVariant if a pro re-run is wanted later.
PAPER_VARIANTS: tuple[ReasonerVariant, ...] = (
    ReasonerVariant(name="gemini", model="gemini-2.5-flash", multimodal=True),
    ReasonerVariant(name="qwen", model="qwen35", multimodal=False),
)

#: Variant lookup by tag for the CLI.
_VARIANTS_BY_NAME: dict[str, ReasonerVariant] = {v.name: v for v in PAPER_VARIANTS}

#: MLflow experiment name for the paper benchmark (AC: lineage on :5010).
_EXPERIMENT_NAME: str = "us069_paper_bench"

#: Per-item timeout for a single model call (mirrors agent_bench hardening): a
#: stalled call (dropped tunnel) raises and is caught per item so the run
#: completes instead of hanging.
_ITEM_TIMEOUT_S: float = 200.0

#: Minimum number of GEO-Bench-2 items a variant must actually score for its
#: metrics to be a number rather than NaN (a text-only variant on an image-only
#: task evaluates a negligible, unrepresentative fraction).
_MIN_GEOBENCH_N: int = 5

#: Per-million-token USD price of each reasoner, used by :func:`_cost_per_query`.
#: Gemini 2.5-pro public pricing ($1.25 in / $10 out per M tokens). The on-prem
#: Qwen has NO per-token API price -- it is amortised GPU-hour cost, so its
#: per-query figure is derived from the measured wall-clock latency and an
#: H100 hourly rate (:data:`_QWEN_GPU_USD_PER_HOUR`), reported as "amortizado",
#: never $0. The dict carries ``(input_usd_per_m, output_usd_per_m)``; a missing
#: entry yields a NaN cost (reported as n/a), never a fabricated number.
_TOKEN_PRICE_USD_PER_M: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    # Gemini 2.5-flash public pricing ($0.30 in / $2.50 out per M tokens). This is
    # the model the US-069 benchmark actually runs (see PAPER_VARIANTS), so its
    # cost column is real rather than NaN.
    "gemini-2.5-flash": (0.30, 2.50),
}

#: Amortised H100 GPU-hour cost (USD) for the on-prem Qwen per-query estimate.
#: Azure Standard_NC40ads_H100_v5 spot reference; documented, overridable.
_QWEN_GPU_USD_PER_HOUR: float = 6.98


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeoBenchTask:
    """One GEO-Bench-2 agricultural task descriptor.

    Attributes:
        id: Stable task id (e.g. ``"m-eurocrops"``).
        name: Human-readable task name.
        modality: ``"classification"`` or ``"segmentation"``.
        label_space: The ordered list of class names the task uses (the label set
            the model must choose from / is scored against).
        split: The split tag the items come from (e.g. ``"test"``).
    """

    id: str
    name: str
    modality: str
    label_space: list[str]
    split: str


@dataclass(frozen=True)
class GeoBenchItem:
    """One GEO-Bench-2 example (a tile/chip + its gold label).

    Attributes:
        task_id: The owning :class:`GeoBenchTask` id.
        item_id: Stable example id within the task.
        image_path: Relative path to the tile/chip image (empty when the item is
            text-only metadata); resolved against the GEO-Bench-2 root.
        question: The prompt text describing what to identify (built by the loader
            from the task's label space when the raw record carries none).
        label_space: The class names the model picks from (copied from the task so
            the evaluator is self-contained per item).
        gold_label: The gold class name (one of ``label_space``).
        requires_image: ``True`` when answering needs the tile (a text-only
            reasoner skips it and reports ``n_skipped``).
    """

    task_id: str
    item_id: str
    image_path: str
    question: str
    label_space: list[str]
    gold_label: str
    requires_image: bool


@dataclass(frozen=True)
class AgroMindItEsItem:
    """One AgroMind-IT/ES bilingual QA item (US-068, eval-only).

    Schema compatible with the original AgroMind subset (AC US-068) plus the
    ``lang`` discriminator so the evaluator can break the metrics down per
    language. Open numeric/text items carry no ``options``.

    Attributes:
        item_id: Stable item id within the bilingual catalogue.
        lang: Language tag, ``"it"`` or ``"es"``.
        question: The question text (in ``lang``).
        options: Mapping of choice label to value (empty for open items).
        answer: The gold answer -- a choice letter for multiple-choice items, or
            a free number/short text for open ones.
        image_path: Relative path to the base image (empty when text-only).
        family: The AgroMind taxonomy family/catalogue tag (for slicing).
        is_multimodal: ``True`` when answering requires an image; a text-only
            reasoner skips these and reports ``n_skipped``.
    """

    item_id: str
    lang: str
    question: str
    options: dict[str, str]
    answer: str
    image_path: str
    family: str
    is_multimodal: bool


@dataclass
class BenchmarkScores:
    """Per-seed paired scores kept for the Wilcoxon test (one benchmark).

    The aggregator stores, per (variant, benchmark), the ordered per-item score
    vector of seed 0 so Gemini and Qwen can be paired item-by-item for
    :func:`wilcoxon_paired`. Only seed 0 is paired (the seeds re-sample model
    sampling, not the item order), which keeps the pairing well-defined.

    Attributes:
        item_ids: The ordered item ids scored (the pairing key).
        item_scores: The per-item exact-match scores aligned 1:1 with
            ``item_ids``.
    """

    item_ids: list[str] = field(default_factory=list)
    item_scores: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank test between two variants.

    Attributes:
        statistic: The Wilcoxon test statistic (NaN when undefined, e.g. all
            pairs are tied or fewer than the minimum pairs).
        p_value: The two-sided p-value (NaN when undefined).
        n_pairs: The number of paired, non-tied observations the test used.
        note: A short Spanish note when the test is not computable (e.g. too few
            pairs / all ties), empty otherwise.
    """

    statistic: float
    p_value: float
    n_pairs: int
    note: str = ""


# ---------------------------------------------------------------------------
# Loaders (degrade-clean: missing dataset -> [], NEVER fabricated)
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into records (blank lines skipped).

    Args:
        path: The JSONL dataset path.

    Returns:
        The parsed records (order preserved); ``[]`` when the file is absent.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_geobench2(
    root: Path = DEFAULT_GEOBENCH_ROOT,
    *,
    tasks: Sequence[str] | None = None,
    max_per_task: int = 0,
) -> list[GeoBenchItem]:
    """Load the GEO-Bench-2 agricultural subset from a materialised manifest.

    Reads ``<root>/manifest.json`` -- a small JSON the downloader writes after
    pulling the agricultural tasks via the official ``geobench`` loader -- with
    shape ``{"tasks": [{id, name, modality, label_space, split, items: [{item_id,
    image_path, question?, gold_label}]}]}``. This keeps the eval decoupled from
    the (GPU-heavy) ``geobench`` download: the manifest is the DVC-tracked
    artefact, and the raw tiles live alongside it.

    Degrades clean (Arthur's zero-synthetic rule): a missing root/manifest returns
    ``[]`` (the caller marks the GEO-Bench-2 cell ``pending``); it NEVER
    fabricates tasks or labels.

    Args:
        root: The GEO-Bench-2 subset root (holds ``manifest.json`` + tiles).
        tasks: Optional allow-list of task ids to keep (``None`` keeps all
            agricultural tasks in the manifest). The AC asks for >= 3 tasks.
        max_per_task: Cap the items per task for a cheaper run (``0`` = all).

    Returns:
        The flattened list of :class:`GeoBenchItem` (empty when the subset is
        absent).
    """
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        logger.warning(
            "geobench2_manifest_absent",
            path=str(manifest_path),
            reason="dataset_not_downloaded_pending_blocker",
        )
        return []
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    allow = frozenset(tasks) if tasks else None
    items: list[GeoBenchItem] = []
    for task_raw in raw.get("tasks", []):
        task_id = str(task_raw.get("id") or "")
        if allow is not None and task_id not in allow:
            continue
        label_space = [str(c) for c in task_raw.get("label_space") or []]
        modality = str(task_raw.get("modality") or "classification")
        records = task_raw.get("items") or []
        if max_per_task > 0:
            records = records[:max_per_task]
        for rec in records:
            image_path = str(rec.get("image_path") or "").strip()
            question = str(rec.get("question") or "") or _default_geobench_question(
                label_space, modality
            )
            items.append(
                GeoBenchItem(
                    task_id=task_id,
                    item_id=str(rec.get("item_id") or ""),
                    image_path=image_path,
                    question=question,
                    label_space=label_space,
                    gold_label=str(rec.get("gold_label") or "").strip(),
                    requires_image=bool(image_path),
                )
            )
    logger.info(
        "geobench2_loaded",
        root=str(root),
        n_tasks=len({it.task_id for it in items}),
        n_items=len(items),
    )
    return items


def _default_geobench_question(label_space: Sequence[str], modality: str) -> str:
    """Build the default Spanish GEO-Bench-2 prompt from a task's label space.

    Args:
        label_space: The ordered class names the model must choose from.
        modality: The task modality (only ``classification`` is prompted here;
            segmentation tasks needing a mask are skipped by the evaluator).

    Returns:
        A Spanish instruction asking for exactly one class from ``label_space``,
        ending with the ``Respuesta:`` marker the answer extractor reads.
    """
    options = ", ".join(label_space)
    return (
        "Eres un experto en teledeteccion agricola. Observa el mosaico satelital "
        "y elige UNA sola clase de la lista que mejor describe el cultivo o la "
        f"cobertura del suelo dominante.\nModalidad: {modality}.\n"
        f"Clases posibles: {options}.\n"
        "Razona brevemente y termina con una linea exactamente con el formato "
        "'Respuesta: <clase>'."
    )


def load_agromind_itses(path: Path = DEFAULT_ITSES_PATH) -> list[AgroMindItEsItem]:
    """Load the AgroMind-IT/ES bilingual JSONL (US-068), failing soft if absent.

    Each record carries ``item_id``, ``lang`` (``it``/``es``), ``question``,
    optional ``options``, ``answer``, optional ``image_path`` and ``family``. An
    item is multimodal when it has a base image; image-valued options also count
    (mirrors :func:`ml.eval.agent_bench.load_agromind_subset`).

    Degrades clean: a missing file returns ``[]`` (US-068 is FUTURE -- native
    human review pending); the cell is then marked ``pending``, never fabricated.

    Args:
        path: Path to ``agromind_itses_500.jsonl``.

    Returns:
        The parsed bilingual items (empty when the file is absent).
    """
    items: list[AgroMindItEsItem] = []
    for rec in _read_jsonl(Path(path)):
        options = {str(k): str(v) for k, v in (rec.get("options") or {}).items()}
        # The US-068 seed (seed.jsonl) carries the image as a bare filename under
        # the ``image`` key (e.g. ``classification_0000.png``, resolved against
        # data/s2_italia) and a ``category`` field rather than ``family``; older
        # fixtures used ``image_path``/``family``. Accept BOTH so the real seed is
        # not silently loaded as text-only (which made Qwen skip all 500 items and
        # Gemini answer blind -- the bug behind the empty IT/ES column).
        image_path = str(rec.get("image_path") or rec.get("image") or "").strip()
        has_option_image = any(_is_image_value(v) for v in options.values())
        # Honour an explicit ``is_multimodal`` flag when the record sets one;
        # otherwise infer it from the presence of an image (base or option).
        raw_multimodal = rec.get("is_multimodal")
        is_multimodal = (
            bool(raw_multimodal)
            if raw_multimodal is not None
            else (bool(image_path) or has_option_image)
        )
        items.append(
            AgroMindItEsItem(
                item_id=str(rec.get("item_id") or ""),
                lang=str(rec.get("lang") or "").lower(),
                question=str(rec.get("question") or ""),
                options=options,
                answer=str(rec.get("answer") or "").strip(),
                image_path=image_path,
                family=str(rec.get("family") or rec.get("category") or ""),
                is_multimodal=is_multimodal,
            )
        )
    logger.info(
        "agromind_itses_loaded",
        path=str(path),
        n_items=len(items),
        n_it=sum(1 for it in items if it.lang == "it"),
        n_es=sum(1 for it in items if it.lang == "es"),
    )
    return items


#: Image-file suffixes recognised when deciding whether an option is an image.
_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _is_image_value(value: str) -> bool:
    """Return whether an option value is an image path rather than answer text.

    Args:
        value: The option value (answer text or a relative image path).

    Returns:
        ``True`` when the lowercased value ends with a known image suffix.
    """
    return isinstance(value, str) and value.lower().endswith(_IMAGE_SUFFIXES)


# ---------------------------------------------------------------------------
# Local metric helpers (live here, NOT in agent_metrics, so its API is untouched)
# ---------------------------------------------------------------------------
def macro_f1(preds: Sequence[str], golds: Sequence[str], labels: Sequence[str]) -> float:
    """Unweighted macro-averaged F1 over a fixed multi-class label space.

    Each label's F1 is computed from its one-vs-rest confusion counts and the
    mean over ``labels`` is returned (a class never predicted nor present scores
    ``0.0`` by convention, included in the denominator -- the standard macro
    definition). Comparison is case-insensitive on the stripped string.

    Args:
        preds: Predicted class names, aligned 1:1 with ``golds``.
        golds: Gold class names.
        labels: The fixed label space to average over.

    Returns:
        Macro-F1 in ``[0.0, 1.0]``; ``0.0`` for empty inputs or an empty label
        space.
    """
    if not preds or not golds or not labels:
        return 0.0
    pairs = list(
        zip(
            [p.strip().lower() for p in preds],
            [g.strip().lower() for g in golds],
            strict=True,
        )
    )
    f1s: list[float] = []
    for label in labels:
        target = label.strip().lower()
        tp = sum(1 for p, g in pairs if p == target and g == target)
        fp = sum(1 for p, g in pairs if p == target and g != target)
        fn = sum(1 for p, g in pairs if p != target and g == target)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _percentiles(latencies_ms: Sequence[float]) -> tuple[float, float]:
    """Return the p50 and p95 of a latency sample (milliseconds).

    Args:
        latencies_ms: Per-item wall-clock latencies in milliseconds.

    Returns:
        ``(p50, p95)`` in milliseconds; ``(nan, nan)`` for an empty sample.
    """
    if not latencies_ms:
        return math.nan, math.nan
    ordered = sorted(latencies_ms)
    return _percentile(ordered, 50.0), _percentile(ordered, 95.0)


def _percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted sample.

    Args:
        ordered: The ascending-sorted latency sample.
        pct: The percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value (the single value for a 1-sample
        input).
    """
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _cost_per_query(
    variant: ReasonerVariant,
    *,
    mean_prompt_tokens: float,
    mean_completion_tokens: float,
    mean_latency_ms: float,
) -> float:
    """Estimate the USD cost of a single query for a variant.

    Cloud variants use their published per-million-token price applied to the
    measured mean token counts. The on-prem Qwen has NO API price, so its cost is
    the amortised GPU-hour rate scaled by the measured mean latency (reported as
    "amortizado", never $0). A variant with no price entry and no GPU rate yields
    NaN (rendered ``n/a``), never a fabricated number.

    Args:
        variant: The reasoner variant.
        mean_prompt_tokens: Mean prompt tokens per query (NaN when unmeasured).
        mean_completion_tokens: Mean completion tokens per query (NaN when
            unmeasured).
        mean_latency_ms: Mean wall-clock latency per query in milliseconds.

    Returns:
        The estimated USD cost per query, or NaN when it cannot be derived.
    """
    price = _TOKEN_PRICE_USD_PER_M.get(variant.model)
    if price is not None and not (_is_nan(mean_prompt_tokens) or _is_nan(mean_completion_tokens)):
        in_usd, out_usd = price
        return (mean_prompt_tokens * in_usd + mean_completion_tokens * out_usd) / 1_000_000.0
    # On-prem variant: amortised GPU-hour cost scaled by wall-clock latency.
    if not variant.multimodal and not _is_nan(mean_latency_ms) and variant.name == "qwen":
        return (_QWEN_GPU_USD_PER_HOUR / 3_600_000.0) * mean_latency_ms
    return math.nan


# ---------------------------------------------------------------------------
# Prompt + answer extraction (shared with the AgroMind contract)
# ---------------------------------------------------------------------------
def _extract_class_answer(answer: str, label_space: Sequence[str]) -> str:
    """Extract the chosen class name from a free-text model answer.

    Reads the text after the last ``Respuesta:`` / ``Answer:`` marker (the
    chain-of-thought before it is never scored), then matches it against the
    label space case-insensitively: an exact normalised match wins, else the
    longest label name that appears as a substring wins (so "es trigo de invierno"
    resolves to ``trigo de invierno``). Returns ``""`` when no label matches.

    Args:
        answer: The raw model answer (reasoning + a ``Respuesta:`` line).
        label_space: The task's class names.

    Returns:
        The matched class name (verbatim from ``label_space``), or ``""``.
    """
    tail = agent_metrics._normalize_text(_after_marker(answer))
    if not tail:
        return ""
    norm_to_label = {agent_metrics._normalize_text(c): c for c in label_space}
    if tail in norm_to_label:
        return norm_to_label[tail]
    best = ""
    for norm, label in sorted(norm_to_label.items(), key=lambda kv: -len(kv[0])):
        if norm and norm in tail:
            best = label
            break
    return best


def _after_marker(answer: str) -> str:
    """Return the text after the last ``Respuesta:`` / ``Answer:`` marker.

    Args:
        answer: The raw model answer.

    Returns:
        The trailing answer text (stripped), or the whole answer when no marker
        is present.
    """
    import re

    matches = list(re.finditer(r"(?:respuesta|answer)\s*:", answer, re.IGNORECASE))
    if not matches:
        return answer.strip()
    return answer[matches[-1].end() :].strip()


def _build_itses_prompt(item: AgroMindItEsItem) -> str:
    """Build the QA prompt for an AgroMind-IT/ES item (multiple-choice or open).

    Args:
        item: The bilingual QA item.

    Returns:
        A Spanish prompt ending with the ``Respuesta:`` marker; multiple-choice
        items list the real option labels, open items ask for a direct answer.
    """
    lines = [
        "Eres un evaluador experto en agricultura satelital. Responde la pregunta "
        "razonando brevemente y termina con una linea exactamente con el formato "
        "'Respuesta: <valor>'.",
        "",
        f"Pregunta ({item.lang}): {item.question}",
    ]
    if item.options:
        lines.append("Opciones:")
        for label in sorted(item.options):
            lines.append(f"  {label}. {item.options[label]}")
        lines.append("Elige UNA sola letra y termina con 'Respuesta: <letra>'.")
    else:
        lines.append("Da una respuesta directa y concisa en 'Respuesta: <valor>'.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
async def eval_geobench2(
    variant: ReasonerVariant,
    items: Sequence[GeoBenchItem],
    *,
    backend: LLMBackend | None = None,
    seed: int = 0,
    image_root: Path = DEFAULT_GEOBENCH_ROOT,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float | int]:
    """Evaluate one variant on the GEO-Bench-2 agricultural subset.

    For each classification item the VLM is asked to name the crop/land-cover
    class of the tile (image attached when the variant is multimodal and the file
    resolves), the chosen class is parsed (:func:`_extract_class_answer`) and
    scored by exact-match vs the gold label; macro-F1 over the union of label
    spaces and a BERTScore proxy are also reported. Text-only variants SKIP
    image-required items and report ``n_skipped`` (never scored as if they saw the
    tile). Per-item wall-clock latency feeds p50/p95.

    Args:
        variant: The reasoner variant under test.
        items: The GEO-Bench-2 items.
        backend: Injected backend (tests / wiring); built lazily when ``None``.
        seed: Seed tag (logging / reproducibility).
        image_root: Base folder for resolving the tile images.
        trace_sink: Optional side-effect-only per-item callback (not scored).

    Returns:
        A mapping with ``exact_match``, ``f1_macro``, ``bertscore``,
        ``latency_p50_ms``, ``latency_p95_ms``, ``n_evaluated``, ``n_skipped``,
        and the per-item paired vectors ``_item_ids`` / ``_item_scores`` for the
        Wilcoxon pairing.
    """
    resolved_backend = _resolve_backend(variant, backend)
    em_scores: list[float] = []
    pred_classes: list[str] = []
    gold_classes: list[str] = []
    item_ids: list[str] = []
    latencies_ms: list[float] = []
    label_union: list[str] = []
    seen_labels: set[str] = set()
    n_skipped = 0

    for item in items:
        if item.requires_image and not variant.multimodal:
            n_skipped += 1
            continue
        for label in item.label_space:
            if label.lower() not in seen_labels:
                seen_labels.add(label.lower())
                label_union.append(label)

        image_parts: list[Any] = []
        if variant.multimodal and item.image_path:
            resolved = _resolve_tile(item.image_path, image_root)
            if resolved is not None:
                image_parts.append(_image_part_for(resolved))

        start = time.perf_counter()
        try:
            # ``_run_backend_text`` (shared with agent_bench) applies a per-attempt
            # timeout and retries transient 504/503/429 errors with backoff
            # (US-069); wrapping it in an outer ``wait_for`` would truncate the
            # retry chain, so the per-item ``try`` alone guards the call.
            answer = await _run_backend_text(resolved_backend, item.question, image_parts)
        except Exception as exc:  # noqa: BLE001 - one item must not crash the run
            logger.warning(
                "geobench2_item_failed", variant=variant.name, item=item.item_id, error=str(exc)
            )
            answer = ""
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        pred = _extract_class_answer(answer, item.label_space)
        em = 1.0 if pred and pred.strip().lower() == item.gold_label.strip().lower() else 0.0
        em_scores.append(em)
        pred_classes.append(pred)
        gold_classes.append(item.gold_label)
        item_ids.append(f"{item.task_id}/{item.item_id}")
        if trace_sink is not None:
            trace_sink(
                {
                    "benchmark": "GEO-Bench-2",
                    "variant": variant.name,
                    "seed": seed,
                    "task_id": item.task_id,
                    "item_id": item.item_id,
                    "gold": item.gold_label,
                    "prediction": pred,
                    "exact_match": em,
                    "correct": em >= 0.5,
                }
            )

    n = len(em_scores)
    too_few = n < _MIN_GEOBENCH_N
    exact = (sum(em_scores) / n) if (n and not too_few) else math.nan
    f1 = macro_f1(pred_classes, gold_classes, label_union) if (n and not too_few) else math.nan
    bert = (
        agent_metrics.bertscore_f1(pred_classes, gold_classes) if (n and not too_few) else math.nan
    )
    p50, p95 = _percentiles(latencies_ms)
    logger.info(
        "geobench2_eval_done",
        variant=variant.name,
        seed=seed,
        exact_match=exact,
        n_evaluated=n,
        n_skipped=n_skipped,
    )
    return {
        "exact_match": exact,
        "f1_macro": f1,
        "bertscore": bert,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "n_evaluated": n,
        "n_skipped": n_skipped,
        "_item_ids": item_ids,  # type: ignore[dict-item]
        "_item_scores": em_scores,  # type: ignore[dict-item]
    }


async def eval_agromind_itses(
    variant: ReasonerVariant,
    items: Sequence[AgroMindItEsItem],
    *,
    backend: LLMBackend | None = None,
    judge: HallucinationJudge | None = None,
    seed: int = 0,
    image_root: Path = DEFAULT_ITSES_IMAGE_ROOT,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float | int]:
    """Evaluate one variant on AgroMind-IT/ES (bilingual QA, eval-only).

    Same contract as :func:`ml.eval.agent_bench.eval_agromind`: exact-match,
    SQuAD-F1, BERTScore proxy and the optional LLM-as-judge hallucination rate,
    with per-language (``it``/``es``) exact-match also reported. Text-only
    variants skip multimodal items and report ``n_skipped``; per-item latency
    feeds p50/p95.

    Args:
        variant: The reasoner variant under test.
        items: The bilingual QA items.
        backend: Injected backend; built lazily when ``None``.
        judge: Injectable hallucination judge (``None`` -> NaN, rendered n/a).
        seed: Seed tag.
        image_root: Base folder for resolving the item images.
        trace_sink: Optional side-effect-only per-item callback (not scored).

    Returns:
        A mapping with ``exact_match``, ``f1_squad``, ``bertscore``,
        ``hallucination``, ``exact_match_it``, ``exact_match_es``,
        ``latency_p50_ms``, ``latency_p95_ms``, ``n_evaluated``, ``n_skipped``,
        and the paired vectors ``_item_ids`` / ``_item_scores``.
    """
    resolved_backend = _resolve_backend(variant, backend)
    em_scores: list[float] = []
    f1_scores: list[float] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []
    item_ids: list[str] = []
    latencies_ms: list[float] = []
    judge_samples: list[dict[str, Any]] = []
    em_by_lang: dict[str, list[float]] = {"it": [], "es": []}
    n_skipped = 0

    for item in items:
        if item.is_multimodal and not variant.multimodal:
            n_skipped += 1
            continue
        image_parts: list[Any] = []
        if variant.multimodal and item.image_path:
            resolved = _resolve_tile(item.image_path, image_root)
            if resolved is not None:
                image_parts.append(_image_part_for(resolved))

        prompt = _build_itses_prompt(item)
        start = time.perf_counter()
        try:
            # Per-attempt timeout + transient-error retry (504/503/429) live inside
            # ``_run_backend_text`` (US-069); no outer ``wait_for`` so the retry
            # chain is not truncated.
            answer = await _run_backend_text(resolved_backend, prompt, image_parts)
        except Exception as exc:  # noqa: BLE001 - one item must not crash the run
            logger.warning(
                "itses_item_failed", variant=variant.name, item=item.item_id, error=str(exc)
            )
            answer = ""
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        final = _after_marker(answer)
        valid_letters = frozenset(item.options) if item.options else None
        em = agent_metrics.exact_match(final, item.answer, valid_letters)
        em_scores.append(em)
        gold_text = item.options.get(item.answer, item.answer)
        f1_scores.append(agent_metrics.f1_squad(final, gold_text))
        pred_texts.append(final)
        gold_texts.append(gold_text)
        item_ids.append(item.item_id)
        if item.lang in em_by_lang:
            em_by_lang[item.lang].append(em)
        judge_samples.append(
            {"input": item.question, "actual_output": answer, "context": [gold_text]}
        )
        if trace_sink is not None:
            trace_sink(
                {
                    "benchmark": "AgroMind-IT/ES",
                    "variant": variant.name,
                    "seed": seed,
                    "item_id": item.item_id,
                    "lang": item.lang,
                    "gold": item.answer,
                    "prediction": final,
                    "exact_match": em,
                    "correct": em >= 0.5,
                }
            )

    n = len(em_scores)
    exact = (sum(em_scores) / n) if n else math.nan
    f1 = (sum(f1_scores) / n) if n else math.nan
    bert = agent_metrics.bertscore_f1(pred_texts, gold_texts) if n else math.nan
    halluc = agent_metrics.hallucination_rate(judge_samples, judge) if n else math.nan
    p50, p95 = _percentiles(latencies_ms)
    logger.info(
        "itses_eval_done",
        variant=variant.name,
        seed=seed,
        exact_match=exact,
        n_evaluated=n,
        n_skipped=n_skipped,
    )
    return {
        "exact_match": exact,
        "f1_squad": f1,
        "bertscore": bert,
        "hallucination": halluc,
        "exact_match_it": (sum(em_by_lang["it"]) / len(em_by_lang["it"]))
        if em_by_lang["it"]
        else math.nan,
        "exact_match_es": (sum(em_by_lang["es"]) / len(em_by_lang["es"]))
        if em_by_lang["es"]
        else math.nan,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "n_evaluated": n,
        "n_skipped": n_skipped,
        "_item_ids": item_ids,  # type: ignore[dict-item]
        "_item_scores": em_scores,  # type: ignore[dict-item]
    }


def _resolve_tile(rel_path: str, root: Path) -> Path | None:
    """Resolve a relative tile/image path under a dataset root.

    Args:
        rel_path: The ``./...`` relative path from the manifest/record.
        root: The base folder where the tiles/images were materialised.

    Returns:
        The resolved path when the file exists locally, else ``None``.
    """
    if not rel_path:
        return None
    cleaned = rel_path.lstrip("./").replace("\\", "/")
    candidate = Path(root) / cleaned
    return candidate if candidate.exists() else None


def _image_part_for(path: Path) -> Any:
    """Build a ``google.genai`` image part from a local image file.

    Args:
        path: Local path to an existing image file.

    Returns:
        A ``types.Part`` carrying the image bytes (PNG/JPEG inferred).
    """
    from google.genai import types

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


# ---------------------------------------------------------------------------
# Statistical significance (paired Wilcoxon signed-rank)
# ---------------------------------------------------------------------------
def wilcoxon_paired(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    min_pairs: int = 6,
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test between two per-item score vectors.

    The vectors must be aligned 1:1 (same item order). Tied pairs (zero
    difference) are dropped by ``scipy.stats.wilcoxon`` itself; when fewer than
    ``min_pairs`` non-tied pairs remain -- or every pair is tied -- the test is
    not computable and a NaN result with a Spanish note is returned (the paper
    then reports "no significativo / n insuficiente" honestly, never a fabricated
    p-value).

    Args:
        scores_a: Per-item scores of variant A (e.g. Gemini exact-match).
        scores_b: Per-item scores of variant B (e.g. Qwen exact-match), aligned
            1:1 with ``scores_a``.
        min_pairs: Minimum non-tied pairs required to run the test.

    Returns:
        A :class:`WilcoxonResult` with the statistic, two-sided p-value and the
        number of non-tied pairs used (NaN + note when not computable).
    """
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return WilcoxonResult(math.nan, math.nan, 0, "sin pares")
    a = [float(x) for x in scores_a[:n]]
    b = [float(x) for x in scores_b[:n]]
    non_tied = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    if non_tied < min_pairs:
        return WilcoxonResult(
            math.nan,
            math.nan,
            non_tied,
            f"pares no empatados insuficientes ({non_tied} < {min_pairs})",
        )
    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return WilcoxonResult(float(stat), float(p_value), non_tied)
    except Exception as exc:  # noqa: BLE001 - never crash the table on a stats edge case
        logger.warning("wilcoxon_failed", error=str(exc), n_pairs=non_tied)
        return WilcoxonResult(math.nan, math.nan, non_tied, f"scipy fallo: {exc}")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _run_one(
    benchmark: str,
    variant: ReasonerVariant,
    *,
    geobench_items: Sequence[GeoBenchItem],
    agromind_items: Sequence[Any],
    itses_items: Sequence[AgroMindItEsItem],
    backend: LLMBackend | None,
    judge: HallucinationJudge | None,
    seed: int,
    image_root: Path,
    geobench_root: Path,
    itses_image_root: Path,
) -> dict[str, float | int]:
    """Run a single (benchmark, variant, seed) and return its raw metric mapping.

    Args:
        benchmark: One of :data:`BENCHMARKS`.
        variant: The reasoner variant.
        geobench_items: Loaded GEO-Bench-2 items (possibly empty).
        agromind_items: Loaded AgroMind items (possibly empty).
        itses_items: Loaded AgroMind-IT/ES items (possibly empty).
        backend: Injected backend for the variant (``None`` -> built lazily).
        judge: Injectable hallucination judge.
        seed: The seed tag.
        image_root: Base folder for AgroMind images.
        geobench_root: Base folder for GEO-Bench-2 tiles.
        itses_image_root: Base folder for AgroMind-IT/ES tiles (``data/s2_italia``;
            distinct from the AgroMind image root because the US-068 seed
            references bare filenames there).

    Returns:
        The raw metric mapping returned by the per-benchmark evaluator.
    """
    if benchmark == "GEO-Bench-2":
        return asyncio.run(
            eval_geobench2(
                variant, geobench_items, backend=backend, seed=seed, image_root=geobench_root
            )
        )
    if benchmark == "AgroMind":
        return asyncio.run(
            eval_agromind(
                variant,
                agromind_items,
                backend=backend,
                judge=judge,
                seed=seed,
                image_root=image_root,
            )
        )
    return asyncio.run(
        eval_agromind_itses(
            variant,
            itses_items,
            backend=backend,
            judge=judge,
            seed=seed,
            image_root=itses_image_root,
        )
    )


def _benchmark_available(benchmark: str, items: Sequence[Any]) -> bool:
    """Return whether a benchmark has data to score (else it is ``pending``).

    Args:
        benchmark: The benchmark name (for logging).
        items: The loaded items for that benchmark.

    Returns:
        ``True`` when there is at least one item; ``False`` marks the cell
        ``pending`` (no data downloaded / delivered yet).
    """
    if items:
        return True
    logger.warning("benchmark_pending", benchmark=benchmark, reason="no_items_loaded")
    return False


def run_paper_benchmark(
    variants: Sequence[ReasonerVariant] = PAPER_VARIANTS,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    geobench_root: Path = DEFAULT_GEOBENCH_ROOT,
    agromind_path: Path = DEFAULT_AGROMIND_PATH,
    itses_path: Path = DEFAULT_ITSES_PATH,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    itses_image_root: Path = DEFAULT_ITSES_IMAGE_ROOT,
    geobench_tasks: Sequence[str] | None = None,
    max_per_task: int = 0,
    max_items: int = 0,
    backends: dict[str, LLMBackend] | None = None,
    judge: HallucinationJudge | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    log_mlflow: bool = True,
    probe_server: bool = True,
    out_latex: Path | None = None,
) -> dict[str, Any]:
    """Run the three benchmarks for every variant over the seeds and report.

    For each (benchmark, variant) it runs every seed, aggregates ``mean +- std``
    (reusing :func:`ml.eval.agent_bench._aggregate`), keeps the seed-0 per-item
    score vector so Gemini vs Qwen can be paired for :func:`wilcoxon_paired`, then
    optionally logs to MLflow (``code_version`` + ``data_version`` tags) and
    exports the paper LaTeX table. Backends are injectable per variant so the run
    is deterministic and offline in tests; Gemini is evaluated first so the paid
    cloud variant is checkpointed earliest (never re-paid on a later failure).

    A benchmark with no loaded data is marked ``pending`` (never scored with a
    fabricated number), honouring Arthur's zero-synthetic rule.

    Args:
        variants: The reasoner variants (default the two paper variants).
        seeds: The evaluation seeds (3 by default for the error bars).
        geobench_root: GEO-Bench-2 subset root (holds ``manifest.json``).
        agromind_path: AgroMind subset JSON path.
        itses_path: AgroMind-IT/ES JSONL path (US-068).
        itses_image_root: Base folder for AgroMind-IT/ES tiles (``data/s2_italia``).
        image_root: Base folder for AgroMind / IT-ES images.
        geobench_tasks: Optional GEO-Bench-2 task allow-list (>= 3 agricultural).
        max_per_task: Cap items per GEO-Bench-2 task (``0`` = all).
        max_items: Cap the AgroMind / IT-ES item count for a bounded REAL subset
            run (``0`` = full corpus). The reported metrics stay real (computed
            over the first ``max_items`` items; ``n_evaluated`` reflects the cap).
        backends: Optional ``{variant_name: backend}`` injection map.
        judge: Optional hallucination judge (NaN when absent).
        checkpoint_path: Optional JSON checkpoint (persisted per variant).
        resume: Resume from the checkpoint, skipping done variants.
        log_mlflow: Whether to log the run to MLflow.
        probe_server: Forwarded to ``track_experiment`` (``False`` in tests).
        out_latex: Optional path for the exported LaTeX table.

    Returns:
        The nested results mapping ``{"results": {variant: {benchmark: {metric:
        {mean, std}}}}, "wilcoxon": {benchmark: WilcoxonResult-as-dict},
        "pending": [benchmark, ...], "targets": {...}}``.
    """
    backends = backends or {}
    geobench_items = load_geobench2(geobench_root, tasks=geobench_tasks, max_per_task=max_per_task)
    agromind_items = load_agromind_subset(agromind_path) if Path(agromind_path).exists() else []
    itses_items = load_agromind_itses(itses_path)

    # Optional REAL subset cap (US-069): when running all 3 seeds end-to-end the
    # full 500+500 multimodal corpus is multi-hour on the cloud reasoner. Capping
    # the AgroMind / IT-ES item lists keeps every reported number REAL (it is just
    # computed over the first ``max_items`` items, ``n_evaluated`` reflects it) and
    # lets the 3-seed error bars finish. ``0`` = no cap (full corpus). GEO-Bench-2
    # has its own ``max_per_task`` and is unaffected.
    if max_items and max_items > 0:
        agromind_items = list(agromind_items)[:max_items]
        itses_items = list(itses_items)[:max_items]
        logger.info(
            "paper_bench_item_cap_applied",
            max_items=max_items,
            n_agromind=len(agromind_items),
            n_itses=len(itses_items),
        )

    available: dict[str, bool] = {
        "GEO-Bench-2": _benchmark_available("GEO-Bench-2", geobench_items),
        "AgroMind": _benchmark_available("AgroMind", agromind_items),
        "AgroMind-IT/ES": _benchmark_available("AgroMind-IT/ES", itses_items),
    }
    pending = [b for b in BENCHMARKS if not available[b]]

    ordered = sorted(variants, key=lambda v: 0 if "gemini" in v.name.lower() else 1)
    results: dict[str, Any] = {}
    paired_seed0: dict[str, dict[str, dict[str, list[float] | list[str]]]] = {}
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = loaded.get("results", {})
        logger.info("paper_bench_checkpoint_loaded", done=sorted(results))

    for variant in ordered:
        if variant.name in results:
            logger.info("variant_skipped_resumed", variant=variant.name)
            continue
        backend = backends.get(variant.name)
        per_benchmark: dict[str, Any] = {}
        paired_seed0[variant.name] = {}
        for benchmark in BENCHMARKS:
            if not available[benchmark]:
                per_benchmark[benchmark] = {"status": "pending"}
                continue
            seed_metrics: list[dict[str, float]] = []
            for seed in seeds:
                if hasattr(backend, "reset"):
                    backend.reset()  # type: ignore[union-attr]
                raw = _run_one(
                    benchmark,
                    variant,
                    geobench_items=geobench_items,
                    agromind_items=agromind_items,
                    itses_items=itses_items,
                    backend=backend,
                    judge=judge,
                    seed=seed,
                    image_root=image_root,
                    geobench_root=geobench_root,
                    itses_image_root=itses_image_root,
                )
                if seed == seeds[0]:
                    paired_seed0[variant.name][benchmark] = {
                        "item_ids": list(raw.pop("_item_ids", [])),  # type: ignore[arg-type]
                        "item_scores": list(raw.pop("_item_scores", [])),  # type: ignore[arg-type]
                    }
                else:
                    raw.pop("_item_ids", None)
                    raw.pop("_item_scores", None)
                seed_metrics.append({k: float(v) for k, v in raw.items()})
            agg = _aggregate(seed_metrics)
            agg["status"] = "ok"  # type: ignore[assignment]
            agg["cost_per_query_usd"] = {  # type: ignore[assignment]
                "mean": _cost_per_query(
                    variant,
                    mean_prompt_tokens=math.nan,
                    mean_completion_tokens=math.nan,
                    mean_latency_ms=agg.get("latency_p50_ms", {}).get("mean", math.nan),
                ),
                "std": math.nan,
            }
            per_benchmark[benchmark] = agg
        results[variant.name] = per_benchmark
        if checkpoint_path is not None:
            _save_checkpoint({"results": results}, checkpoint_path)

    wilcoxon = _wilcoxon_table(paired_seed0, ordered)
    out: dict[str, Any] = {
        "results": results,
        "wilcoxon": {b: _wilcoxon_to_dict(w) for b, w in wilcoxon.items()},
        "pending": pending,
        "targets": {"AgroMind": {"gemini": 0.75, "qwen": 0.70}},
    }

    if log_mlflow:
        _log_to_mlflow(results, geobench_root, agromind_path, probe_server=probe_server)
    if out_latex is not None:
        export_latex_table(out, out_latex)
    logger.info(
        "paper_bench_done",
        variants=[v.name for v in ordered],
        pending=pending,
        seeds=list(seeds),
    )
    return out


def _wilcoxon_table(
    paired_seed0: dict[str, dict[str, dict[str, list[float] | list[str]]]],
    variants: Sequence[ReasonerVariant],
) -> dict[str, WilcoxonResult]:
    """Compute the Gemini-vs-Qwen paired Wilcoxon per benchmark.

    Pairs the seed-0 per-item exact-match vectors of ``gemini`` and ``qwen`` by
    their common item ids (so a text-only Qwen that skipped image-only items still
    pairs on the items both scored). A benchmark with no overlapping items yields
    a NaN result with a Spanish note.

    Args:
        paired_seed0: ``{variant: {benchmark: {item_ids, item_scores}}}``.
        variants: The evaluated variants (must include ``gemini`` + ``qwen`` for a
            test; otherwise every benchmark is marked not-computable).

    Returns:
        ``{benchmark: WilcoxonResult}`` over the benchmarks present for both.
    """
    names = {v.name for v in variants}
    table: dict[str, WilcoxonResult] = {}
    if not ({"gemini", "qwen"} <= names):
        for benchmark in BENCHMARKS:
            table[benchmark] = WilcoxonResult(
                math.nan, math.nan, 0, "se requieren las variantes gemini y qwen"
            )
        return table
    for benchmark in BENCHMARKS:
        g = paired_seed0.get("gemini", {}).get(benchmark)
        q = paired_seed0.get("qwen", {}).get(benchmark)
        if not g or not q:
            table[benchmark] = WilcoxonResult(math.nan, math.nan, 0, "benchmark pendiente")
            continue
        g_ids = cast("list[str]", g["item_ids"])
        q_ids = cast("list[str]", q["item_ids"])
        g_by_id = dict(zip(g_ids, cast("list[float]", g["item_scores"]), strict=True))
        q_by_id = dict(zip(q_ids, cast("list[float]", q["item_scores"]), strict=True))
        common = [i for i in g_ids if i in q_by_id]
        a = [float(g_by_id[i]) for i in common]
        b = [float(q_by_id[i]) for i in common]
        table[benchmark] = wilcoxon_paired(a, b)
    return table


def _wilcoxon_to_dict(result: WilcoxonResult) -> dict[str, Any]:
    """Serialise a :class:`WilcoxonResult` to a JSON-friendly mapping.

    Args:
        result: The Wilcoxon result.

    Returns:
        A mapping with ``statistic``, ``p_value``, ``n_pairs`` and ``note``.
    """
    return {
        "statistic": result.statistic,
        "p_value": result.p_value,
        "n_pairs": result.n_pairs,
        "note": result.note,
    }


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------
#: Column labels of the headline metrics shown per benchmark in the paper table.
#: The middle ``F1`` column resolves to a benchmark-specific key
#: (:data:`_F1_KEY_BY_BENCHMARK`): GEO-Bench-2 is multi-class classification so it
#: reports macro-F1; the two QA benchmarks report SQuAD token-F1.
_LATEX_METRIC_LABELS: tuple[str, ...] = ("Acc", "F1", "BERTSc")

#: The metric key for each headline column EXCEPT the F1 column (resolved per
#: benchmark). Aligned 1:1 with :data:`_LATEX_METRIC_LABELS` (``None`` = the F1
#: slot, filled by :data:`_F1_KEY_BY_BENCHMARK`).
_LATEX_METRIC_KEYS: tuple[str | None, ...] = ("exact_match", None, "bertscore")

#: The F1 metric key per benchmark: macro-F1 for the GEO-Bench-2 classification
#: axis, SQuAD token-F1 for the QA axes. So the F1 column populates for every
#: benchmark instead of reading a key the QA evaluators never emit (the IT/ES bug
#: caught in the smoke run).
_F1_KEY_BY_BENCHMARK: dict[str, str] = {
    "GEO-Bench-2": "f1_macro",
    "AgroMind": "f1_squad",
    "AgroMind-IT/ES": "f1_squad",
}


def _f1_key_for(benchmark: str) -> str:
    """Return the F1 metric key for a benchmark's headline column.

    Args:
        benchmark: One of :data:`BENCHMARKS`.

    Returns:
        ``"f1_macro"`` for the GEO-Bench-2 classification axis, ``"f1_squad"``
        for the QA axes (default).
    """
    return _F1_KEY_BY_BENCHMARK.get(benchmark, "f1_squad")


def export_latex_table(out: dict[str, Any], path: Path) -> Path:
    """Export the paper benchmark table as a ``booktabs`` LaTeX fragment.

    The table has one row block per variant and, per benchmark, the headline
    ``mean ± std`` cells; pending benchmarks render ``\\textit{pendiente}`` (never
    a fabricated number) and the Wilcoxon p-value per benchmark is reported in the
    caption. The caption carries the correct attributions (AlphaEarth
    ``SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1 CC-BY-4.0; Gemini 2.5-pro 1M ctx;
    AgroMind eval-only; Qwen3-30B-A3B GPTQ-Int4 single-GPU) and the reference
    targets (AgroMind >= 0.75 Gemini / >= 0.70 Qwen, not over-claimed).

    Args:
        out: The mapping returned by :func:`run_paper_benchmark`.
        path: Destination ``.tex`` path (parent dirs created).

    Returns:
        The :class:`~pathlib.Path` of the written ``.tex``.
    """
    results = out.get("results", {})
    wilcoxon = out.get("wilcoxon", {})
    pending = out.get("pending", [])
    n_metrics = len(_LATEX_METRIC_LABELS)
    col_fmt = "l" + "".join("r" * n_metrics for _ in BENCHMARKS)

    header_top = " & ".join(
        f"\\multicolumn{{{n_metrics}}}{{c}}{{{_latex_escape(b)}}}" for b in BENCHMARKS
    )
    metric_labels = " & ".join(_LATEX_METRIC_LABELS)
    header_sub = " & ".join(metric_labels for _ in BENCHMARKS)

    rows: list[str] = []
    for variant in sorted(results):
        cells: list[str] = [_latex_escape(variant)]
        for benchmark in BENCHMARKS:
            block = results[variant].get(benchmark, {})
            if block.get("status") != "ok":
                cells.extend(["\\textit{pend.}"] * n_metrics)
                continue
            # Resolve the F1 column to the benchmark-specific key (macro vs SQuAD).
            keys = [k if k is not None else _f1_key_for(benchmark) for k in _LATEX_METRIC_KEYS]
            for key in keys:
                stats = block.get(key, {})
                cells.append(_fmt_cell(stats.get("mean"), stats.get("std")))
        rows.append(" & ".join(cells) + " \\\\")

    wilcoxon_clause = "; ".join(
        f"{_latex_escape(b)}: p={_fmt_p(wilcoxon.get(b, {}).get('p_value'))} "
        f"(n={wilcoxon.get(b, {}).get('n_pairs', 0)})"
        for b in BENCHMARKS
    )
    pending_clause = (
        f" Benchmarks pendientes (sin datos a la fecha): {', '.join(pending)}." if pending else ""
    )
    caption = (
        "Comparativa multi-benchmark de los dos reasoners frozen del copiloto "
        "(patron Be My Eyes, arXiv:2511.19417): Gemini 2.5-flash (nube; variante "
        "ejecutada en el benchmark por rapidez y menor tasa de 504 en items "
        "multimodales -- el manuscrito cita 2.5-pro como reasoner de produccion) "
        "y Qwen3-30B-A3B (on-prem vLLM, GPTQ-Int4, single-GPU). Embeddings de "
        "AlphaEarth Foundations (SATELLITE\\_EMBEDDING/V1/ANNUAL, data v1.1, "
        "64-dim, CC-BY-4.0). AgroMind y AgroMind-IT/ES son eval-only (sin "
        "re-entrenamiento). Celdas mean $\\pm$ std sobre 3 corridas. Wilcoxon "
        f"signed-rank pareado Gemini vs Qwen: {wilcoxon_clause}. Objetivos de "
        "referencia (no sobre-afirmados): AgroMind $\\geq$ 0.75 Gemini / $\\geq$ "
        f"0.70 Qwen.{pending_clause}"
    )

    lines = [
        "% Generado por ml/eval/paper_bench.py (US-069). NO editar a mano.",
        "% Las celdas se pueblan SOLO tras una corrida real (datos reales, cero placeholders).",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{tab:paper-benchmark-comparison}",
        "\\begin{tabular}{" + col_fmt + "}",
        "\\toprule",
        "Variante & " + header_top + " \\\\",
        "\\cmidrule(lr){2-" + str(1 + n_metrics * len(BENCHMARKS)) + "}",
        " & " + header_sub + " \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("paper_bench_latex_written", path=str(out_path))
    return out_path


def _latex_escape(text: str) -> str:
    """Escape the LaTeX-special characters in a short label.

    Args:
        text: The raw label (variant or benchmark name).

    Returns:
        The label with ``&``, ``%``, ``_``, ``#`` escaped.
    """
    for char in ("\\", "&", "%", "#"):
        text = text.replace(char, "\\" + char)
    return text.replace("_", "\\_")


def _fmt_cell(mean: Any, std: Any) -> str:
    """Format a ``mean ± std`` LaTeX cell, or ``n/a`` for a NaN/absent mean.

    Args:
        mean: The metric mean (may be NaN / ``None``).
        std: The metric std (may be NaN / ``None``).

    Returns:
        A ``$0.123 \\pm 0.004$`` cell, or ``n/a`` when the mean is unavailable.
    """
    if mean is None or _is_nan(float(mean)):
        return "n/a"
    std_val = 0.0 if std is None or _is_nan(float(std)) else float(std)
    return f"${float(mean):.3f} \\pm {std_val:.3f}$"


def _fmt_p(p_value: Any) -> str:
    """Format a Wilcoxon p-value for the caption, or ``n/a`` when undefined.

    Args:
        p_value: The two-sided p-value (may be NaN / ``None``).

    Returns:
        The p-value to three decimals, or ``n/a``.
    """
    if p_value is None or _is_nan(float(p_value)):
        return "n/a"
    return f"{float(p_value):.3f}"


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def _log_to_mlflow(
    results: dict[str, Any],
    geobench_root: Path,
    agromind_path: Path,
    *,
    probe_server: bool,
) -> None:
    """Log the aggregated results to MLflow with versioning tags.

    Opens one run via ``track_experiment`` (sets ``code_version`` +
    ``data_version``) and logs every ``{variant}/{benchmark}/{metric}`` mean and
    std. Logging failures are caught and logged so the benchmark + table still
    complete (eval-only, no training side effects).

    Args:
        results: The nested per-variant results mapping.
        geobench_root: GEO-Bench-2 root (its ``.dvc`` drives the data_version).
        agromind_path: AgroMind subset path (fallback data_version source).
        probe_server: Forwarded to ``track_experiment``.
    """
    try:
        import mlflow

        from ml.utils.mlflow_utils import track_experiment

        dvc_target = str(geobench_root) if Path(geobench_root).exists() else str(agromind_path)
        with track_experiment(_EXPERIMENT_NAME, dvc_path=dvc_target, probe_server=probe_server):
            for variant, benchmarks in results.items():
                for benchmark, metrics in benchmarks.items():
                    if metrics.get("status") != "ok":
                        continue
                    for metric, stats in metrics.items():
                        if not isinstance(stats, dict):
                            continue
                        mean = stats.get("mean", math.nan)
                        std = stats.get("std", math.nan)
                        tag = f"{variant}/{benchmark}/{metric}".replace(" ", "_")
                        if not _is_nan(mean):
                            mlflow.log_metric(f"{tag}/mean", mean)
                        if not _is_nan(std):
                            mlflow.log_metric(f"{tag}/std", std)
        logger.info("paper_bench_mlflow_logged", experiment=_EXPERIMENT_NAME)
    except Exception as exc:  # noqa: BLE001 - tracking must not break the eval run
        logger.warning("paper_bench_mlflow_failed", error=str(exc))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_variants(names: Sequence[str] | None) -> list[ReasonerVariant]:
    """Resolve CLI variant tags to :class:`ReasonerVariant` objects.

    Falls back to ``agent_bench.DEFAULT_VARIANTS`` for tags outside the paper pair
    (e.g. ``gemma-base``), so the optional Gemma-base column can be added without
    duplicating the variant registry.

    Args:
        names: The variant tags from the CLI, or ``None`` for the paper pair.

    Returns:
        The resolved variants (defaults to the two paper variants).
    """
    if not names:
        return list(PAPER_VARIANTS)
    from ml.eval.agent_bench import DEFAULT_VARIANTS

    # PAPER_VARIANTS take precedence over agent_bench.DEFAULT_VARIANTS: the paper
    # pair pins gemini-2.5-pro (US-069 AC), while DEFAULT_VARIANTS carries a
    # different default model for the same "gemini" tag. DEFAULT_VARIANTS only
    # fills tags the paper pair does not define (e.g. "gemma-base").
    registry = {v.name: v for v in (*DEFAULT_VARIANTS, *PAPER_VARIANTS)}
    resolved: list[ReasonerVariant] = []
    for name in names:
        variant = registry.get(name)
        if variant is None:
            raise SystemExit(f"Variante desconocida: {name!r}. Validas: {sorted(registry)}")
        resolved.append(variant)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the paper multi-benchmark eval (US-069).

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evalua los dos reasoners frozen (Gemini 2.5-pro nube y Qwen3-30B-A3B "
            "on-prem) en GEO-Bench-2, AgroMind y AgroMind-IT/ES con barras de error "
            "y Wilcoxon pareado (eval-only, sin entrenamiento; US-069 paper track)."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variantes a evaluar (por defecto gemini y qwen).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds de evaluacion para las barras de error (por defecto 0 1 2).",
    )
    parser.add_argument(
        "--geobench-root",
        type=Path,
        default=DEFAULT_GEOBENCH_ROOT,
        help="Raiz del subset agricola de GEO-Bench-2 (con manifest.json).",
    )
    parser.add_argument(
        "--agromind-path",
        type=Path,
        default=DEFAULT_AGROMIND_PATH,
        help="Ruta al subset JSON de AgroMind.",
    )
    parser.add_argument(
        "--itses-path",
        type=Path,
        default=DEFAULT_ITSES_PATH,
        help="Ruta al JSONL de AgroMind-IT/ES (US-068).",
    )
    parser.add_argument(
        "--itses-image-root",
        type=Path,
        default=DEFAULT_ITSES_IMAGE_ROOT,
        help="Carpeta base de los tiles de AgroMind-IT/ES (por defecto data/s2_italia).",
    )
    parser.add_argument(
        "--geobench-tasks",
        nargs="+",
        default=None,
        help="Allow-list de tasks agricolas de GEO-Bench-2 (>=3).",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="Limite de items por task de GEO-Bench-2 (0 = todos).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Limite de items de AgroMind / AgroMind-IT/ES para una corrida subset "
            "REAL acotada (0 = corpus completo). Las metricas siguen siendo reales "
            "(calculadas sobre los primeros N items; n_evaluated lo refleja)."
        ),
    )
    parser.add_argument(
        "--out-latex",
        type=Path,
        default=Path("paper/tables/us-069/benchmark_comparison.tex"),
        help="Ruta de salida de la tabla LaTeX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta del checkpoint JSON; cada variante se persiste al terminar.",
    )
    parser.add_argument("--resume", action="store_true", help="Reanudar desde el checkpoint.")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="No registrar la corrida en MLflow."
    )
    args = parser.parse_args(argv)

    variants = _resolve_variants(args.variants)
    out = run_paper_benchmark(
        variants,
        seeds=tuple(args.seeds),
        geobench_root=args.geobench_root,
        agromind_path=args.agromind_path,
        itses_path=args.itses_path,
        itses_image_root=args.itses_image_root,
        geobench_tasks=args.geobench_tasks,
        max_per_task=args.max_per_task,
        max_items=args.max_items,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        log_mlflow=not args.no_mlflow,
        out_latex=args.out_latex,
    )
    _print_summary(out)
    return 0


def _print_summary(out: dict[str, Any]) -> None:
    """Write a compact JSON summary of the headline cells to stdout.

    Args:
        out: The mapping returned by :func:`run_paper_benchmark`.
    """
    summary: dict[str, Any] = {"pending": out.get("pending", []), "variants": {}}
    for variant, benchmarks in out.get("results", {}).items():
        per: dict[str, Any] = {}
        for benchmark, block in benchmarks.items():
            if block.get("status") == "ok":
                per[benchmark] = block.get("exact_match", {}).get("mean")
            else:
                per[benchmark] = "pending"
        summary["variants"][variant] = per
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
