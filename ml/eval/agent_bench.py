"""Agent benchmark orchestrator for AgroSatCopilot (US-049).

This module is the harness that evaluates the conversational copilot variants
against two public benchmarks, **EVAL-ONLY** (it never trains: AgroMind ships
no train split, so any fine-tune would be leakage, see AC-3 / ADR-009). It only
runs inference, parses the model output and scores it with the pure metrics in
:mod:`ml.eval.agent_metrics`, then renders the comparison report via
:mod:`ml.eval.agent_report`.

Pieces:

- Data models: :class:`AgroMindItem`, :class:`GeoTask`, :class:`ReasonerVariant`.
- Loaders: :func:`load_agromind_subset` (the real 500-item JSON subset) and
  :func:`load_geoanalystbench` (the real 50-task CSV, read with Polars).
- Runners: :func:`eval_agromind` (multiple-choice QA -> letter -> exact match,
  plus the textual proxies and the optional LLM-as-judge hallucination rate) and
  :func:`eval_geoanalyst` (plan-and-react -> workflow + code -> semantic
  similarity vs the human workflow and canonical CodeBLEU vs the reference).
- Aggregator + entry point: :func:`run_benchmark` runs every variant over the
  seeds, aggregates ``mean +- std``, optionally logs to MLflow with the
  ``code_version`` + ``data_version`` tags (lineage on ``:5010``) and builds the
  HTML report; :func:`main` is the argparse CLI.

Multimodality tension (documented, AC-3 / plan Section 3): AgroMind is
multimodal. ``gemini`` and ``gemma-base`` are multimodal and evaluate the full
subset; ``qwen`` is **text-only** (it is the on-prem reasoner, not a VLM), so it
SKIPS the multimodal items (those with a base image or image options) and is
scored only on the purely-textual subset, with ``n_skipped`` reported so the
limitation is explicit and never papered over. GeoAnalystBench is 100% text, so
every variant runs it in full.

Backends are **injectable** (``backends`` / ``backend`` parameters) so the whole
harness runs in tests with mocks and zero network. When no backend is injected
one is built with :func:`ml.agent.backends.make_backend`. The real Qwen run
depends on the on-prem vLLM endpoint of US-048 (currently blocked): when that
endpoint exists the same harness scores Qwen unchanged; Gemini / Gemma run via
their cloud API.

Project conventions: identifiers and docstrings in English (Google style),
visible prose (CLI help, the report) in Spanish; ``structlog`` (never
``print`` in logic); Polars for the tabular load; full type hints; no emojis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import structlog

from ml.eval import agent_metrics
from ml.eval.agent_report import DEFAULT_REPORT_DIR, build_report_html

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

    from ml.agent.backends import LLMBackend
    from ml.eval.agent_metrics import HallucinationJudge

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_AGROMIND_PATH",
    "DEFAULT_GEO_PATH",
    "AgroMindItem",
    "AgroMindResult",
    "GeoResult",
    "GeoTask",
    "ReasonerVariant",
    "eval_agromind",
    "eval_geoanalyst",
    "load_agromind_subset",
    "load_geoanalystbench",
    "main",
    "run_benchmark",
]

#: Default location of the real AgroMind 500-item stratified subset.
DEFAULT_AGROMIND_PATH: Path = Path("data/agromind/agromind_subset_500.json")

#: Default location of the real GeoAnalystBench 50-task CSV.
DEFAULT_GEO_PATH: Path = Path("data/geoanalystbench/GeoAnalystBench.csv")

#: Default base folder where the subset images are extracted (see
#: ``scripts/download_agromind_images.py``). Used to resolve ``image_path`` for
#: the multimodal variants; absent files degrade to a text-only prompt.
DEFAULT_IMAGE_ROOT: Path = Path("data/agromind/images")

#: Pass threshold for a GeoAnalystBench task: a task counts as passed when its
#: workflow semantic similarity to the human-designed workflow EXCEEDS this.
#:
#: CALIBRATION (documented, defensible): the similarity is the cosine of two
#: ``all-MiniLM-L6-v2`` embeddings of short, ~5-step workflows. Two DIFFERENT but
#: equally-correct phrasings of the SAME workflow typically land around
#: ``0.40-0.60`` cosine on this encoder (MiniLM compresses paraphrase distance,
#: it does not push synonyms to ~1.0). A ``0.50`` cut therefore rejects many
#: valid paraphrases as failures and makes the pass-rate artificially harsh, so
#: the default is lowered to ``0.35`` -- below the paraphrase band's floor, above
#: the ``< 0.30`` noise of genuinely unrelated workflows. The raw
#: ``mean_semantic_sim`` and ``mean_codebleu`` are also reported (see
#: :meth:`GeoResult.as_metrics`) so the threshold choice is transparent and the
#: continuous signal is never hidden behind the binary pass decision. The rubric
#: pass-rate target (>= 0.65) is then applied on top of this per-task decision,
#: and the threshold stays overridable via the ``pass_threshold`` parameter.
GEO_PASS_THRESHOLD: float = 0.35

#: Per-item timeout for a single model call. A stalled call (dropped tunnel,
#: wedged socket) raises ``asyncio.TimeoutError`` -- caught by the existing
#: per-item ``except`` -- so the run keeps going and COMPLETES instead of hanging
#: forever (US-049 hardening). Generous on purpose for slow on-prem multimodal.
_ITEM_TIMEOUT_S: float = 200.0

#: Number of attempts for a single model call before giving up (US-069 fix). A
#: multimodal Gemini call over images intermittently returns ``504
#: DEADLINE_EXCEEDED`` / ``503 UNAVAILABLE``; a single 504 must NOT abandon the
#: item (the previous run stalled on item 182's 504 with ~$0 spent). The call is
#: retried with exponential backoff on transient errors only; a permanent error
#: (bad request, auth) fails fast on the first attempt.
_CALL_MAX_ATTEMPTS: int = 3

#: Base backoff seconds between retries; attempt ``i`` (0-indexed) sleeps
#: ``_CALL_BACKOFF_BASE_S * 2**i`` -> 2s, 4s, 8s ... so transient server-side
#: pressure is given time to clear before the next attempt.
_CALL_BACKOFF_BASE_S: float = 2.0

#: Substrings (case-insensitive) that mark a model-call error as TRANSIENT and
#: therefore worth retrying. Covers Gemini ``504 DEADLINE_EXCEEDED`` /
#: ``503 UNAVAILABLE`` / ``500 INTERNAL`` / ``429 RESOURCE_EXHAUSTED`` and the
#: socket-level timeouts (``asyncio.TimeoutError`` is matched by type below). A
#: 4xx that is none of these (e.g. ``400 INVALID_ARGUMENT``, ``401``) is permanent
#: and is NOT retried -- it would just waste three attempts and money.
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "deadline_exceeded",
    "deadline exceeded",
    "504",
    "503",
    "unavailable",
    "500 internal",
    "internalservererror",
    "429",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Return whether a model-call exception is worth retrying.

    Treats ``asyncio.TimeoutError`` (the per-attempt timeout) and any error whose
    string carries one of :data:`_TRANSIENT_ERROR_MARKERS` (Gemini
    ``504 DEADLINE_EXCEEDED``, ``503``, ``429`` ...) as transient. A permanent
    error (``400``/``401``/schema) returns ``False`` so it fails fast instead of
    burning three attempts.

    Args:
        exc: The exception raised by the backend call.

    Returns:
        ``True`` when the call should be retried, ``False`` for a permanent error.
    """
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


#: How often to emit a per-item progress log so a long phase is observable
#: instead of a silent black box (US-049 hardening).
_PROGRESS_EVERY: int = 20

#: Minimum number of AgroMind items a variant must actually score for its
#: AgroMind metrics to be reported as a number rather than ``n/a``. AgroMind is
#: ~100% visual (a full-corpus scan found 0 purely-textual items), so a text-only
#: variant evaluates a negligible, unrepresentative fraction; below this floor the
#: metrics are NaN to avoid presenting noise as a comparable score.
_MIN_AGROMIND_N: int = 30

#: The three default reasoner variants (AC-1). ``multimodal`` gates whether a
#: variant may consume AgroMind images; ``qwen`` is text-only on purpose.
DEFAULT_VARIANTS: tuple[ReasonerVariant, ...]  # populated after the dataclass.

#: Image-file suffixes recognised when deciding whether an option is an image.
_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#: MLflow experiment name for this benchmark (AC-6).
_EXPERIMENT_NAME: str = "us049_agent_bench"

#: Schema version stamped on every JSONL trace record, so a future report reader
#: can reject or migrate older dumps (e.g. if the answer-type buckets change).
_TRACE_SCHEMA_VERSION: int = 1

#: Char caps for the prompt / prediction stored in the per-item trace. Without
#: them the geo prompts (instruction + domain + dataset) and code predictions
#: would bloat the dump to tens of KB per record; the caps are load-bearing.
PROMPT_TRACE_CAP: int = 2000
PRED_TRACE_CAP: int = 2000

#: A normalised numeric bounding box ``[a, b, ...]`` with >= 2 comma-separated
#: numbers (floats or ints), used to bucket an open gold as ``open_numeric_bbox``.
_BBOX_RE: re.Pattern[str] = re.compile(
    r"^\s*\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,}\s*\]\s*$"
)

#: A pure number (int or float, optional sign), bucketed as ``open_number``.
_NUMBER_RE: re.Pattern[str] = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

#: Gold values (case-insensitive) treated as a yes/no answer.
_YES_NO_VALUES: frozenset[str] = frozenset({"yes", "no", "si", "sí", "true", "false"})


def _classify_answer_type(item: AgroMindItem) -> str:
    """Classify an item by the shape of its gold answer (seed-independent).

    The classification depends only on ``item.options`` and the gold string,
    never on the model prediction, so it is deterministic and stable across
    seeds. It lets the report slice exact-match by answer type and explain why
    the AgroMind headline is depressed (the numeric buckets are exact-match
    hostile by construction).

    Args:
        item: The AgroMind item to classify.

    Returns:
        One of ``"multiple_choice"`` (the item has options),
        ``"open_numeric_bbox"`` (gold is a ``[a, b, ...]`` numeric box),
        ``"open_number"`` (gold is a pure int/float),
        ``"yes_no"`` (gold is a yes/no token, any of es/it/en) or
        ``"open_text"`` (anything else).
    """
    if item.options:
        return "multiple_choice"
    gold = item.answer.strip()
    if _BBOX_RE.match(gold):
        return "open_numeric_bbox"
    if _NUMBER_RE.match(gold):
        return "open_number"
    if gold.lower() in _YES_NO_VALUES:
        return "yes_no"
    return "open_text"


def _truncate(text: str, cap: int) -> str:
    """Truncate ``text`` to ``cap`` chars with a Spanish elision suffix.

    Args:
        text: The full string (prompt or prediction).
        cap: Maximum number of leading characters to keep.

    Returns:
        ``text`` unchanged when within ``cap``, otherwise the first ``cap``
        characters followed by ``"... [truncado N chars]"`` where ``N`` is the
        number of dropped characters.
    """
    if len(text) <= cap:
        return text
    dropped = len(text) - cap
    return f"{text[:cap]}... [truncado {dropped} chars]"


class _JsonlTraceWriter:
    """File-backed per-variant JSONL sink for the per-item inference trace.

    Opens one UTF-8 file in truncate mode and appends one
    ``json.dumps(record, ensure_ascii=False)`` line per scored item, flushing
    after each record so a crashed run (dropped H100 tunnel) still leaves a
    readable partial dump. Used as a context manager so the file is closed even
    when the eval raises; the ``sink`` method matches the
    ``Callable[[dict[str, Any]], None]`` contract threaded into the eval
    runners.

    Attributes:
        path: Destination JSONL path (parent dirs created on open).
        variant: Variant tag (for the close log only; records carry their own).
        benchmark: Benchmark tag (for the close log only).
    """

    def __init__(self, path: Path, *, variant: str, benchmark: str) -> None:
        """Open the per-variant trace file in UTF-8 truncate mode.

        Args:
            path: Destination JSONL path.
            variant: Variant tag for the open/close logs.
            benchmark: Benchmark tag for the open/close logs.
        """
        self.path = path
        self.variant = variant
        self.benchmark = benchmark
        self._n_records = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        logger.info(
            "trace_dump_opened",
            path=str(path),
            variant=variant,
            benchmark=benchmark,
        )

    def sink(self, record: dict[str, Any]) -> None:
        """Write one trace record as a UTF-8 JSONL line and flush.

        Args:
            record: The per-item trace record to serialise.
        """
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._n_records += 1

    def close(self) -> None:
        """Close the underlying file and log the record count."""
        if self._fh.closed:
            return
        self._fh.close()
        logger.info(
            "trace_dump_closed",
            path=str(self.path),
            variant=self.variant,
            benchmark=self.benchmark,
            n_records=self._n_records,
        )

    def __enter__(self) -> _JsonlTraceWriter:
        """Return ``self`` for use as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the file on context exit (also on exception)."""
        self.close()


@dataclass(frozen=True)
class AgroMindItem:
    """One AgroMind multiple-choice QA item (the real subset schema).

    Attributes:
        image_path: Relative path to the question's base image (e.g.
            ``./Rural/piece_images/x.png``); empty when the item has no base
            image. Resolved against :data:`DEFAULT_IMAGE_ROOT` at eval time.
        question: The question text.
        options: Mapping of choice label to its value (the real subset uses up
            to ten labels ``A``-``J``; open numeric/text items have NO options).
            Values are either answer text or, for multi-image items, a relative
            image path.
        answer: The gold answer -- a choice letter (``A``-``J``) for
            multiple-choice items, or a free number/short text for open items.
        type_id: AgroMind question-type id.
        item_id: AgroMind item id within its task file.
        level1_id: Top-level taxonomy id (used for the stratified subset).
        level2_id: Second-level taxonomy id.
        level3_id: Third-level taxonomy id.
        task_file: The source QA task file tag (e.g. ``"BD"``).
        is_multimodal: ``True`` when answering requires an image -- either the
            item has a base ``image_path`` or any option value is an image path.
            Text-only reasoners (Qwen) skip these.
    """

    image_path: str
    question: str
    options: dict[str, str]
    answer: str
    type_id: int
    item_id: int
    level1_id: int
    level2_id: int
    level3_id: int
    task_file: str
    is_multimodal: bool

    @property
    def option_image_paths(self) -> dict[str, str]:
        """Return the subset of options whose value is an image path.

        Returns:
            A mapping ``{label: relative_image_path}`` for image-valued options
            (empty when the options are plain text).
        """
        return {label: value for label, value in self.options.items() if _is_image_path(value)}


@dataclass(frozen=True)
class GeoTask:
    """One GeoAnalystBench plan-and-react task (the real CSV schema).

    Attributes:
        id: Task id (``"1"`` .. ``"50"``).
        task: Short task title.
        instruction: The full instruction handed to the reasoner.
        domain_knowledge: Background domain knowledge for the task.
        dataset_description: Description of the available datasets.
        human_workflow: The gold human-designed workflow (numbered steps),
            used as the reference for :func:`workflow_semantic_similarity`.
        code_string: The reference Python code, used for the simplified
            CodeBLEU.
        task_length: The task length (number of expected steps), as a string.
    """

    id: str
    task: str
    instruction: str
    domain_knowledge: str
    dataset_description: str
    human_workflow: str
    code_string: str
    task_length: str


@dataclass(frozen=True)
class ReasonerVariant:
    """A reasoner under evaluation.

    Attributes:
        name: Variant tag, one of ``"gemini"``, ``"qwen"``, ``"gemma-base"``.
        model: The concrete model id passed to :func:`make_backend` (or the
            multimodal API).
        multimodal: Whether the variant can consume images. ``qwen`` is
            text-only, so it is ``False`` and skips multimodal AgroMind items.
    """

    name: str
    model: str
    multimodal: bool


# Populated here (after the dataclass is defined) so the module exposes the
# canonical three variants used by the CLI and the rubric targets.
DEFAULT_VARIANTS = (
    ReasonerVariant(name="gemini", model="gemini-3.5-flash", multimodal=True),
    ReasonerVariant(name="qwen", model="qwen35", multimodal=False),
    ReasonerVariant(name="gemma-base", model="gemma4:26b-a4b-it-q4_K_M", multimodal=True),
    # On-prem multimodal comparative VLM (Qwen3.6-35B-A3B via llama.cpp + mmproj).
    # multimodal=True so it sees AgroMind images like Gemini/Gemma, making the
    # AgroMind cell comparable across all three vision reasoners (the text-only
    # ``qwen`` above stays for the text GeoAnalystBench, its real job).
    ReasonerVariant(name="qwen36-vl", model="qwen36-vl", multimodal=True),
)

#: Variant lookup by tag for the CLI.
_VARIANTS_BY_NAME: dict[str, ReasonerVariant] = {v.name: v for v in DEFAULT_VARIANTS}


@dataclass
class AgroMindResult:
    """Per-seed AgroMind scores for one variant.

    Attributes:
        exact_match: Mean exact-match over the evaluated items.
        f1_squad: Mean SQuAD-style token F1 over the textual answers.
        bertscore: Semantic-proxy BERTScore F1 over the textual answers.
        hallucination: Mean hallucination rate (NaN when no judge is given).
        n_evaluated: Number of items actually scored.
        n_skipped: Number of items skipped (text-only variant on multimodal).
    """

    exact_match: float
    f1_squad: float
    bertscore: float
    hallucination: float
    n_evaluated: int
    n_skipped: int

    def as_metrics(self) -> dict[str, float]:
        """Return the per-metric mapping consumed by the aggregator.

        Returns:
            A mapping of metric name to its scalar value for this seed.
        """
        return {
            "exact_match": self.exact_match,
            "f1_squad": self.f1_squad,
            "bertscore": self.bertscore,
            "hallucination": self.hallucination,
            "n_evaluated": float(self.n_evaluated),
            "n_skipped": float(self.n_skipped),
        }


@dataclass
class GeoResult:
    """Per-seed GeoAnalystBench scores for one variant.

    Attributes:
        pass_rate: Fraction of tasks whose workflow similarity passed the
            threshold (the rubric headline metric for this benchmark).
        mean_semantic_sim: Mean workflow semantic similarity over tasks
            (reported prominently alongside ``pass_rate`` so the continuous
            signal is visible, not just the thresholded pass decision).
        mean_codebleu: Mean canonical CodeBLEU over tasks (AST + data-flow),
            likewise reported as a raw headline metric.
        n: Number of tasks evaluated.
    """

    pass_rate: float
    mean_semantic_sim: float
    mean_codebleu: float
    n: int

    def as_metrics(self) -> dict[str, float]:
        """Return the per-metric mapping consumed by the aggregator.

        Returns:
            A mapping of metric name to its scalar value for this seed.
        """
        return {
            "pass_rate": self.pass_rate,
            "mean_semantic_sim": self.mean_semantic_sim,
            "mean_codebleu": self.mean_codebleu,
            "n": float(self.n),
        }


def _is_image_path(value: str) -> bool:
    """Return whether an option value is an image path rather than answer text.

    Args:
        value: The option value (answer text or a relative image path).

    Returns:
        ``True`` when the lowercased value ends with a known image suffix.
    """
    return isinstance(value, str) and value.lower().endswith(_IMAGE_SUFFIXES)


def _coerce_int(value: Any, default: int = -1) -> int:
    """Best-effort integer coercion for the (mostly-int) AgroMind id fields.

    Args:
        value: The raw value from the JSON (int, str or ``None``).
        default: Value returned when coercion is not possible.

    Returns:
        The integer value, or ``default`` when ``value`` is missing/invalid.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_agromind_subset(path: Path) -> list[AgroMindItem]:
    """Load and parse the real AgroMind 500-item subset JSON.

    Each record is mapped to an :class:`AgroMindItem`. An item is marked
    :attr:`~AgroMindItem.is_multimodal` when it carries a base ``image_path`` or
    any of its options is an image path, so the text-only variant can skip it.

    Args:
        path: Path to ``agromind_subset_500.json``.

    Returns:
        The list of parsed items (one per record, order preserved).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[AgroMindItem] = []
    for record in raw:
        options = {str(label): str(value) for label, value in (record.get("options") or {}).items()}
        image_path = str(record.get("image_path") or "").strip()
        has_option_image = any(_is_image_path(v) for v in options.values())
        is_multimodal = bool(image_path) or has_option_image
        items.append(
            AgroMindItem(
                image_path=image_path,
                question=str(record.get("question") or ""),
                options=options,
                answer=str(record.get("answer") or "").strip(),
                type_id=_coerce_int(record.get("type_id")),
                item_id=_coerce_int(record.get("item_id")),
                level1_id=_coerce_int(record.get("level1_id")),
                level2_id=_coerce_int(record.get("level2_id")),
                level3_id=_coerce_int(record.get("level3_id")),
                task_file=str(record.get("_task_file") or ""),
                is_multimodal=is_multimodal,
            )
        )
    n_multimodal = sum(1 for it in items if it.is_multimodal)
    logger.info(
        "agromind_subset_loaded",
        path=str(path),
        n_items=len(items),
        n_multimodal=n_multimodal,
        n_textual=len(items) - n_multimodal,
    )
    return items


def load_geoanalystbench(csv: Path) -> list[GeoTask]:
    """Load and parse the real GeoAnalystBench 50-task CSV with Polars.

    Read with ``infer_schema_length=0`` so every column stays ``Utf8`` (the CSV
    mixes numbered workflows, multiline code and ids). Rows with an empty ``id``
    are dropped (the file carries one trailing blank row), yielding the 50 tasks.

    Args:
        csv: Path to ``GeoAnalystBench.csv``.

    Returns:
        The list of parsed :class:`GeoTask` (one per non-empty row).
    """
    frame = pl.read_csv(csv, infer_schema_length=0)
    tasks: list[GeoTask] = []
    for row in frame.iter_rows(named=True):
        task_id = (row.get("id") or "").strip()
        if not task_id:
            continue
        tasks.append(
            GeoTask(
                id=task_id,
                task=(row.get("Task") or "").strip(),
                instruction=(row.get("Instruction") or "").strip(),
                domain_knowledge=(row.get("Domain Knowledge") or "").strip(),
                dataset_description=(row.get("Dataset Description") or "").strip(),
                human_workflow=(row.get("Human Designed Workflow") or "").strip(),
                code_string=(row.get("CodeString") or "").strip(),
                task_length=(row.get("Task Length") or "").strip(),
            )
        )
    logger.info("geoanalystbench_loaded", path=str(csv), n_tasks=len(tasks))
    return tasks


def _build_agromind_prompt(item: AgroMindItem, *, with_images: bool) -> str:
    """Build the few-shot + chain-of-thought prompt for an AgroMind item.

    The prompt is **adaptive** to the item shape (B-6) AND now few-shot +
    chain-of-thought, branched by :func:`_classify_answer_type`, so the reasoner
    can think before committing and scores closer to its true ability. Every
    branch ships exactly ONE compact worked example (token cost stays bounded)
    and ends the contract with a single ``Respuesta:`` line that the final-answer
    extractor (:func:`_extract_final_answer`) reads back before scoring:

    - ``multiple_choice``: the instruction lists the REAL label set present
      (derived from ``sorted(item.options)``, e.g. ``"A, B o C"``) so the model
      is never told to pick a non-existent ``D`` nor steered away from a valid
      ``E``-``J``. For multi-image options the value is shown as a reference path
      when ``with_images`` is ``False`` (the text-only variant) and as an image
      marker otherwise. The model reasons briefly, then ends with
      ``Respuesta: <letra>``.
    - ``open_numeric_bbox`` / ``open_number``: the model reasons, then ends with
      ``Respuesta: <numero o [x,y,x,y]>``.
    - ``yes_no``: the model reasons, then ends with ``Respuesta: Si`` / ``No``.
    - ``open_text``: the model gives a concise direct answer (no letter) on the
      ``Respuesta:`` line.

    Args:
        item: The AgroMind item.
        with_images: Whether the caller will also attach the images.

    Returns:
        The composed prompt string.
    """
    answer_type = _classify_answer_type(item)
    if answer_type == "multiple_choice":
        return _build_mc_prompt(item, with_images=with_images)
    if answer_type in ("open_numeric_bbox", "open_number"):
        return _build_open_numeric_prompt(item)
    if answer_type == "yes_no":
        return _build_yes_no_prompt(item)
    return _build_open_text_prompt(item)


def _build_mc_prompt(item: AgroMindItem, *, with_images: bool) -> str:
    """Build the multiple-choice few-shot + CoT prompt for an AgroMind item.

    Args:
        item: The multiple-choice AgroMind item (has options).
        with_images: Whether the caller will also attach the images.

    Returns:
        The composed prompt string ending with the ``Respuesta: <letra>`` contract.
    """
    labels = sorted(item.options)
    letters_clause = _format_letter_set(labels)
    lines = [
        "Eres un evaluador experto en agricultura satelital. Responde la pregunta "
        f"de opcion multiple eligiendo UNA sola letra ({letters_clause}).",
        "La pregunta puede estar en ingles; tu respuesta final debe ser UNICAMENTE "
        "la letra de la opcion correcta, sin prosa, explicaciones ni el texto de la "
        "opcion.",
        "Razona brevemente paso a paso y termina SIEMPRE con una linea final con "
        "exactamente el formato 'Respuesta: <letra>' (por ejemplo 'Respuesta: A'). "
        "Es OBLIGATORIO cerrar con esa linea aunque la pregunta este en ingles; no "
        "respondas solo con prosa.",
        "",
        "Ejemplo:",
        "Pregunta: Que indice resalta la vegetacion sana?",
        "Opciones:",
        "  A. NDWI",
        "  B. NDVI",
        "  C. NDBI",
        "Razonemos: el NDVI usa el rojo y el infrarrojo cercano y crece con la "
        "clorofila, por lo que mide vegetacion sana.",
        "Respuesta: B",
        "",
        f"Pregunta: {item.question}",
        "",
        "Opciones:",
    ]
    for label in labels:
        value = item.options[label]
        if _is_image_path(value):
            shown = f"[imagen {Path(value).name}]" if with_images else f"[imagen: {value}]"
        else:
            shown = value
        lines.append(f"  {label}. {shown}")
    lines.extend(
        [
            "",
            "Razona paso a paso de forma breve y termina OBLIGATORIAMENTE con la "
            f"linea 'Respuesta: <letra>' usando solo una de estas letras: {letters_clause}.",
        ]
    )
    return "\n".join(lines)


def _build_open_numeric_prompt(item: AgroMindItem) -> str:
    """Build the open numeric / bbox few-shot + CoT prompt for an AgroMind item.

    Args:
        item: The open numeric or bounding-box AgroMind item (no options).

    Returns:
        The composed prompt string ending with the
        ``Respuesta: <numero o [x,y,x,y]>`` contract.
    """
    return "\n".join(
        [
            "Eres un evaluador experto en agricultura satelital. Responde la "
            "pregunta con un numero o una caja delimitadora normalizada.",
            "Razona brevemente paso a paso y termina con una linea exactamente con "
            "el formato 'Respuesta: <numero o [x,y,x,y]>'.",
            "",
            "Ejemplo:",
            "Pregunta: Cuantas parcelas de maiz hay en la imagen?",
            "Razonemos: cuento tres lotes contiguos de maiz claramente separados.",
            "Respuesta: 3",
            "",
            f"Pregunta: {item.question}",
            "",
            "Razona paso a paso de forma breve y termina con 'Respuesta: <numero o [x,y,x,y]>'.",
        ]
    )


def _build_yes_no_prompt(item: AgroMindItem) -> str:
    """Build the yes/no few-shot + CoT prompt for an AgroMind item.

    Args:
        item: The yes/no AgroMind item (no options, boolean gold).

    Returns:
        The composed prompt string ending with the ``Respuesta: Si`` / ``No``
        contract.
    """
    return "\n".join(
        [
            "Eres un evaluador experto en agricultura satelital. Responde la "
            "pregunta de forma binaria.",
            "Razona brevemente paso a paso y termina con una linea exactamente con "
            "el formato 'Respuesta: Si' o 'Respuesta: No'.",
            "",
            "Ejemplo:",
            "Pregunta: Hay presencia de agua en la parcela?",
            "Razonemos: la zona central muestra valores altos de NDWI, indicando "
            "una lamina de agua.",
            "Respuesta: Si",
            "",
            f"Pregunta: {item.question}",
            "",
            "Razona paso a paso de forma breve y termina con 'Respuesta: Si' o 'Respuesta: No'.",
        ]
    )


def _build_open_text_prompt(item: AgroMindItem) -> str:
    """Build the open-text few-shot + CoT prompt for an AgroMind item.

    Args:
        item: The open free-text AgroMind item (no options, free-text gold).

    Returns:
        The composed prompt string ending with a concise ``Respuesta:`` line
        (no letter).
    """
    return "\n".join(
        [
            "Eres un evaluador experto en agricultura satelital. Responde la "
            "pregunta con una respuesta directa y concisa (una frase corta, sin "
            "una letra de opcion).",
            "Razona brevemente paso a paso y termina con una linea exactamente con "
            "el formato 'Respuesta: <respuesta concisa>'.",
            "",
            "Ejemplo:",
            "Pregunta: Que cultivo predomina en la parcela?",
            "Razonemos: el patron temporal de NDVI con un pico unico en verano es "
            "tipico del trigo de invierno.",
            "Respuesta: trigo de invierno",
            "",
            f"Pregunta: {item.question}",
            "",
            "Razona paso a paso de forma breve y termina con 'Respuesta: <respuesta concisa>'.",
        ]
    )


def _format_letter_set(labels: Sequence[str]) -> str:
    """Render option labels as a Spanish ``"A, B o C"`` clause for the prompt.

    Args:
        labels: The sorted option labels actually present on the item.

    Returns:
        A comma-separated list with a trailing ``" o "`` before the last label
        (e.g. ``"A, B, C o D"``); a single label is returned verbatim. The
        caller guarantees a non-empty ``labels`` (open items take another path).
    """
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} o {labels[-1]}"


#: Matches the final-answer marker line emitted by the few-shot + CoT prompts,
#: case-insensitive, in Spanish (``Respuesta:``) or English (``Answer:``). The
#: extractor reads the text after the LAST such marker so the chain-of-thought
#: that precedes it is never scored (it would defeat the exact-match parser).
_FINAL_ANSWER_RE: re.Pattern[str] = re.compile(r"(?:respuesta|answer)\s*:", re.IGNORECASE)


def _extract_final_answer(answer: str) -> str:
    """Extract the model's final answer after the LAST ``Respuesta:`` marker.

    Because the prompts now ask for chain-of-thought, the model emits reasoning
    THEN a final ``Respuesta: X`` (or ``Answer: X``) line. The exact-match parser
    must score only that final answer, not the whole CoT (which would leak stray
    letters / numbers and depress the score). This splits on the LAST marker
    (case-insensitive) and returns the trailing text, stripped. When no marker is
    present it falls back to the full answer unchanged -- preserving today's
    behaviour for un-marked responses and never breaking the ``valid_letters``
    letter path.

    Args:
        answer: The raw model answer (reasoning followed by a marker line, or an
            un-marked direct answer).

    Returns:
        The text after the last ``Respuesta:`` / ``Answer:`` marker (stripped),
        or the original ``answer`` (stripped) when no marker is found.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(answer))
    if not matches:
        return answer.strip()
    return answer[matches[-1].end() :].strip()


def _resolve_choice_to_letter(final_answer: str, options: dict[str, str]) -> str:
    """Resolve a multiple-choice prediction back to its option letter.

    Defense-in-depth for the AgroMind multiple-choice contract: the prompt asks
    the model to end with ``Respuesta: <letra>`` (only the letter), but a frozen
    reasoner sometimes answers with the option *text* instead of its letter (e.g.
    gold ``"A"`` whose option A is ``"maize field"`` and the model replies ``"It
    is a maize field"``). When the extracted answer carries no recoverable letter
    (:func:`ml.eval.agent_metrics._extract_choice_letter` returns ``None``) this
    maps the answer text to a letter by matching the option *values*: an exact
    normalised match wins, else the longest option value that appears as a
    substring of the answer wins. The original ``final_answer`` is returned
    unchanged when no option text matches, so the downstream letter parser keeps
    today's behaviour and open (no-option) items are untouched.

    Args:
        final_answer: The extracted final answer (after the ``Respuesta:`` marker).
        options: The item's ``{letter: value}`` option map (empty for open items).

    Returns:
        The matched option letter when the answer text maps to exactly one option
        value; otherwise the original ``final_answer`` unchanged.
    """
    if not options or not final_answer:
        return final_answer
    valid_letters = frozenset(options)
    # If a choice letter is already recoverable, the existing parser handles it.
    if agent_metrics._extract_choice_letter(final_answer, valid_letters) is not None:
        return final_answer
    norm_answer = agent_metrics._normalize_text(final_answer)
    if not norm_answer:
        return final_answer
    norm_to_letter: dict[str, str] = {}
    for letter, value in options.items():
        if _is_image_path(value):
            continue
        norm_value = agent_metrics._normalize_text(value)
        if norm_value:
            norm_to_letter[norm_value] = letter
    if norm_answer in norm_to_letter:
        return norm_to_letter[norm_answer]
    for norm_value, letter in sorted(norm_to_letter.items(), key=lambda kv: -len(kv[0])):
        if norm_value in norm_answer:
            return letter
    return final_answer


def _resolve_image(rel_path: str, image_root: Path) -> Path | None:
    """Resolve an AgroMind relative image path under the local image root.

    Args:
        rel_path: The ``./Category/...`` relative path from the subset.
        image_root: The base folder where subset images were extracted.

    Returns:
        The resolved path when the file exists locally, else ``None``.
    """
    if not rel_path:
        return None
    cleaned = rel_path.lstrip("./").replace("\\", "/")
    candidate = image_root / cleaned
    return candidate if candidate.exists() else None


def _image_part(path: Path) -> Any:
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


def _build_contents(prompt: str, image_parts: Sequence[Any]) -> list[Any]:
    """Build a single-user-turn ``contents`` list for the backend.

    Args:
        prompt: The textual prompt.
        image_parts: Zero or more image parts to attach before the text.

    Returns:
        A one-element list with a ``types.Content`` user turn.
    """
    from google.genai import types

    parts = [*image_parts, types.Part.from_text(text=prompt)]
    return [types.Content(role="user", parts=parts)]


async def _run_backend_text(
    backend: LLMBackend,
    prompt: str,
    image_parts: Sequence[Any],
    *,
    max_attempts: int = _CALL_MAX_ATTEMPTS,
    per_attempt_timeout_s: float = _ITEM_TIMEOUT_S,
) -> str:
    """Drive a backend for one non-streaming text answer (no tools), with retry.

    Consumes the backend's chunk stream and concatenates the text deltas. Tool
    calls are not requested here (the benchmark asks for a direct answer), so any
    function-call chunk is ignored.

    Hardening (US-069): each attempt is bounded by ``per_attempt_timeout_s`` and a
    TRANSIENT failure (Gemini ``504 DEADLINE_EXCEEDED`` / ``503`` / ``429`` or a
    socket/per-attempt timeout, classified by :func:`_is_transient_error`) is
    retried up to ``max_attempts`` with exponential backoff
    (:data:`_CALL_BACKOFF_BASE_S` x ``2**i`` -> 2s, 4s, 8s). A PERMANENT error
    (``400``/``401``/schema) is re-raised immediately so it fails fast. When every
    attempt is exhausted the last transient exception is re-raised so the caller's
    per-item ``except`` records the item as ``""`` (one item never crashes the
    run) -- but a single 504 no longer abandons the item, which is the bug that
    cost the previous run ~$0 of Gemini spend.

    Args:
        backend: The injected or constructed :class:`LLMBackend`.
        prompt: The user prompt.
        image_parts: Image parts to attach (empty for text-only).
        max_attempts: Maximum attempts before giving up on a transient error.
        per_attempt_timeout_s: Wall-clock timeout for each individual attempt.

    Returns:
        The concatenated answer text (stripped).

    Raises:
        Exception: The last exception when all attempts fail, or a permanent
            (non-transient) error on the attempt it occurred.
    """
    contents = _build_contents(prompt, image_parts)
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await asyncio.wait_for(
                _consume_text_stream(backend, contents),
                timeout=per_attempt_timeout_s,
            )
        except Exception as exc:
            last_exc = exc
            transient = _is_transient_error(exc)
            is_last = attempt == max_attempts - 1
            if not transient or is_last:
                if transient and is_last:
                    logger.warning(
                        "backend_call_retries_exhausted",
                        attempts=max_attempts,
                        error=str(exc),
                    )
                raise
            backoff = _CALL_BACKOFF_BASE_S * (2.0**attempt)
            logger.warning(
                "backend_call_retrying",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                backoff_s=backoff,
                error=str(exc),
            )
            await asyncio.sleep(backoff)
    # Unreachable (the loop either returns or raises) but keeps mypy happy.
    raise last_exc if last_exc is not None else RuntimeError("backend call failed")


async def _consume_text_stream(backend: LLMBackend, contents: list[Any]) -> str:
    """Consume a backend's chunk stream into one stripped text answer.

    Args:
        backend: The backend to drive (no tools advertised).
        contents: The ``google.genai`` contents built for this prompt.

    Returns:
        The concatenated, stripped text of the streamed response.
    """
    buffer: list[str] = []
    async for chunk in backend.generate_stream(contents=contents, tools=[], system_instruction=""):
        text = getattr(chunk, "text", None)
        if text:
            buffer.append(text)
    return "".join(buffer).strip()


def _resolve_backend(variant: ReasonerVariant, backend: LLMBackend | None) -> LLMBackend:
    """Return the backend to use for a variant, building one if not injected.

    Args:
        variant: The reasoner variant.
        backend: An injected backend (tests / explicit wiring) or ``None``.

    Returns:
        The injected backend, or one built with :func:`make_backend`.
    """
    if backend is not None:
        return backend
    from ml.agent.backends import make_backend

    # Pass Settings so the Gemini/vLLM credentials from .env.local reach the
    # backend (they are NOT exported to os.environ for the SDK to auto-discover).
    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional (tests inject the backend)
        settings = None
    return make_backend(variant.model, settings)


async def eval_agromind(
    variant: ReasonerVariant,
    items: Sequence[AgroMindItem],
    *,
    backend: LLMBackend | None = None,
    judge: HallucinationJudge | None = None,
    seed: int = 0,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float | int]:
    """Evaluate one variant on AgroMind (multiple-choice QA).

    For each item it builds the prompt (question + options), attaches the images
    when the variant is multimodal and the files are present, runs the backend,
    parses the chosen letter and scores exact-match vs the gold answer. The
    textual proxies (:func:`f1_squad`, :func:`bertscore_f1`) are computed over
    the rendered prediction vs the gold option text, and the optional
    LLM-as-judge :func:`hallucination_rate` over the answer samples.

    Text-only variants (``variant.multimodal`` is ``False``) SKIP every
    multimodal item (base image or image options) and report ``n_skipped`` so
    the limitation is explicit (AC-3 / plan Section 3); they are never scored as
    if they had seen the image.

    Args:
        variant: The reasoner variant under test.
        items: The AgroMind items to evaluate.
        backend: Injected backend (tests / explicit wiring); built lazily when
            ``None`` via :func:`make_backend`.
        judge: Injectable hallucination judge; ``None`` reports hallucination as
            NaN (rendered ``n/a``).
        seed: Seed tag (carried for logging / reproducibility of the run id).
        image_root: Base folder for resolving the subset images.
        trace_sink: Optional side-effect-only callback invoked once per scored
            (non-skipped) item with a per-item trace record (see the US-049
            JSONL dump). It MUST NOT mutate the returned metrics; skipped items
            are never emitted (they have no prediction to score).

    Returns:
        A mapping ``{exact_match, f1_squad, bertscore, hallucination,
        n_evaluated, n_skipped}``.
    """
    resolved_backend = _resolve_backend(variant, backend)
    em_scores: list[float] = []
    f1_scores: list[float] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []
    judge_samples: list[dict[str, Any]] = []
    n_skipped = 0

    for item in items:
        if item.is_multimodal and not variant.multimodal:
            n_skipped += 1
            continue

        image_parts: list[Any] = []
        if variant.multimodal:
            base = _resolve_image(item.image_path, image_root)
            if base is not None:
                image_parts.append(_image_part(base))
            for opt_path in item.option_image_paths.values():
                resolved = _resolve_image(opt_path, image_root)
                if resolved is not None:
                    image_parts.append(_image_part(resolved))

        prompt = _build_agromind_prompt(item, with_images=bool(image_parts))
        errored = False
        try:
            # ``_run_backend_text`` bounds each attempt by ``_ITEM_TIMEOUT_S`` and
            # retries transient 504/503/429 errors internally (US-069), so no outer
            # ``wait_for`` is needed -- wrapping it would cut the retry chain short.
            answer = await _run_backend_text(resolved_backend, prompt, image_parts)
        except Exception as exc:  # noqa: BLE001 - one item must not crash the run
            logger.warning(
                "agromind_item_failed",
                variant=variant.name,
                item_id=item.item_id,
                error=str(exc),
            )
            answer = ""
            errored = True

        # Strip the chain-of-thought: the few-shot prompts ask the model to reason
        # then end with a ``Respuesta: X`` line, so score only that final answer.
        # Falls back to the full answer when no marker is present (un-marked
        # responses keep today's behaviour).
        final_answer = _extract_final_answer(answer)
        # Defense-in-depth for the multiple-choice contract: when the reasoner
        # answers with the option TEXT instead of its letter (no recoverable
        # letter), resolve the text back to the option letter so the exact-match
        # still scores. This is a no-op for open items and for answers that
        # already carry a letter, so it never alters today's behaviour there.
        final_answer = _resolve_choice_to_letter(final_answer, item.options)
        # Constrain the letter parser to the labels that actually exist for the
        # item (B-5): open items (no options) score via the text fallback, and a
        # choice item with options E-J scores its real letter, not a capped A-D.
        valid_letters = frozenset(item.options) if item.options else None
        em_scores.append(agent_metrics.exact_match(final_answer, item.answer, valid_letters))
        if len(em_scores) % _PROGRESS_EVERY == 0:
            logger.info(
                "agromind_progress",
                variant=variant.name,
                evaluated=len(em_scores),
                skipped=n_skipped,
            )
        gold_text = item.options.get(item.answer, item.answer)
        # Score the textual proxies on the extracted final answer too (not the
        # chain-of-thought), so the CoT prose does not dilute token-overlap F1 and
        # the semantic proxy. Compute f1 once per item so the trace record and the
        # post-loop mean share a single source of truth (no double-computation).
        f1_i = agent_metrics.f1_squad(final_answer, gold_text)
        f1_scores.append(f1_i)
        pred_texts.append(final_answer)
        gold_texts.append(gold_text)
        judge_samples.append(
            {
                "input": item.question,
                "actual_output": answer,
                "context": [gold_text],
            }
        )

        if trace_sink is not None:
            trace_sink(
                {
                    "schema_version": _TRACE_SCHEMA_VERSION,
                    "benchmark": "AgroMind",
                    "variant": variant.name,
                    "model": variant.model,
                    "seed": seed,
                    "item_id": item.item_id,
                    "task_file": item.task_file,
                    "type_id": item.type_id,
                    "level1_id": item.level1_id,
                    "level2_id": item.level2_id,
                    "level3_id": item.level3_id,
                    "is_multimodal": item.is_multimodal,
                    "n_options": len(item.options),
                    "answer_type": _classify_answer_type(item),
                    "prompt": _truncate(prompt, PROMPT_TRACE_CAP),
                    "prompt_chars": len(prompt),
                    "n_image_parts": len(image_parts),
                    "gold": item.answer,
                    "gold_text": gold_text,
                    "prediction": _truncate(answer, PRED_TRACE_CAP),
                    "prediction_chars": len(answer),
                    "final_answer": _truncate(final_answer, PRED_TRACE_CAP),
                    "errored": errored,
                    "exact_match": em_scores[-1],
                    "f1_squad": f1_i,
                    "correct": em_scores[-1] >= 0.5,
                }
            )

    n_evaluated = len(em_scores)
    # AgroMind is ~100% visual: a verified scan of the full 28k-item corpus found
    # ZERO purely-textual items (the "textual" ones still reference an image). A
    # text-only variant therefore skips almost everything and is left with a tiny,
    # unrepresentative n. Reporting a score over n<=_MIN_AGROMIND_N would read as a
    # comparable number when it is noise, so the metrics are reported as NaN ->
    # rendered "n/a" in the report and EXCLUDED from the aggregate. This is the
    # honest verdict: a text-only reasoner is NOT evaluable on AgroMind (use the
    # multimodal ``qwen36-vl`` for the on-prem comparison instead). ``n_evaluated``
    # / ``n_skipped`` are still returned so the coverage is explicit.
    too_few = n_evaluated < _MIN_AGROMIND_N
    exact = (sum(em_scores) / n_evaluated) if (n_evaluated and not too_few) else math.nan
    f1 = (sum(f1_scores) / n_evaluated) if (n_evaluated and not too_few) else math.nan
    bert = (
        agent_metrics.bertscore_f1(pred_texts, gold_texts)
        if (n_evaluated and not too_few)
        else math.nan
    )
    halluc = agent_metrics.hallucination_rate(judge_samples, judge) if not too_few else math.nan
    if too_few:
        logger.warning(
            "agromind_insufficient_coverage",
            variant=variant.name,
            n_evaluated=n_evaluated,
            n_skipped=n_skipped,
            min_required=_MIN_AGROMIND_N,
            reason="agromind_is_visual_only_text_variant_not_evaluable",
        )

    logger.info(
        "agromind_eval_done",
        variant=variant.name,
        seed=seed,
        exact_match=exact,
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
    )
    return {
        "exact_match": exact,
        "f1_squad": f1,
        "bertscore": bert,
        "hallucination": halluc,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
    }


def _build_geo_prompt(task: GeoTask) -> str:
    """Build the plan-and-react prompt for a GeoAnalystBench task.

    Asks the reasoner for a numbered workflow followed by a Python code block,
    in two clearly delimited sections so the response can be split for scoring.

    Args:
        task: The GeoAnalystBench task.

    Returns:
        The composed prompt string.
    """
    return "\n".join(
        [
            "Eres un analista geoespacial. Resuelve la siguiente tarea en dos secciones.",
            "Primero piensa paso a paso en el flujo de trabajo antes de escribirlo.",
            "Primero, un flujo de trabajo numerado paso a paso bajo el encabezado 'WORKFLOW:'.",
            "Despues, el codigo Python completo bajo el encabezado 'CODE:' "
            "dentro de un bloque ```python```.",
            "",
            "Ejemplo de formato:",
            "WORKFLOW:",
            "1. Cargar el raster de entrada.",
            "2. Calcular el NDVI y exportar el resultado.",
            "CODE:",
            "```python",
            "ndvi = (nir - red) / (nir + red)",
            "```",
            "",
            f"Tarea: {task.task}",
            "",
            f"Instruccion: {task.instruction}",
            "",
            f"Conocimiento de dominio: {task.domain_knowledge}",
            "",
            f"Descripcion de datos: {task.dataset_description}",
        ]
    )


def _split_workflow_and_code(answer: str) -> tuple[str, str]:
    """Split a plan-and-react answer into its workflow and code sections.

    Recognises a fenced ```python``` (or bare ```) code block as the code, and
    treats ALL the remaining text -- both before AND after the code block, with
    any ``WORKFLOW:`` / ``CODE:`` headers stripped -- as the workflow. Earlier
    this took only ``answer[:first_fence]``, silently dropping workflow prose
    that the model placed after the code block, which zeroed the similarity and
    caused a false pass-rate failure (B-10). Falls back gracefully when the
    response is not perfectly formatted.

    Args:
        answer: The raw model answer.

    Returns:
        A ``(workflow_text, code_text)`` tuple.
    """
    code = ""
    workflow = answer
    fence = "```"
    if fence in answer:
        first = answer.find(fence)
        rest = answer[first + len(fence) :]
        end = rest.find(fence)
        if end == -1:
            # Unterminated fence: everything after it is the code block.
            block = rest
            post = ""
        else:
            block = rest[:end]
            # Skip the closing fence so its delimiter is not kept as workflow.
            post = rest[end + len(fence) :]
        if block.lower().startswith("python"):
            block = block[len("python") :]
        code = block.strip()
        # Keep both the pre-fence and post-fence prose for the workflow so any
        # workflow text written after the code block is not discarded (B-10).
        workflow = f"{answer[:first]}\n{post}"
    for header in ("WORKFLOW:", "CODE:", "Workflow:", "Code:"):
        workflow = workflow.replace(header, " ")
    return workflow.strip(), code.strip()


async def eval_geoanalyst(
    variant: ReasonerVariant,
    tasks: Sequence[GeoTask],
    *,
    backend: LLMBackend | None = None,
    seed: int = 0,
    pass_threshold: float = GEO_PASS_THRESHOLD,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float | int]:
    """Evaluate one variant on GeoAnalystBench (plan-and-react).

    For each task the reasoner receives the instruction and returns a workflow +
    Python code. The workflow is scored against the human-designed workflow with
    :func:`workflow_semantic_similarity` and the code against the reference with
    the canonical :func:`codebleu_score` (AST + data-flow). A task passes when
    its workflow similarity exceeds ``pass_threshold`` (calibrated to ``0.35``,
    see :data:`GEO_PASS_THRESHOLD`); the pass-rate is the headline metric, with
    the raw ``mean_semantic_sim`` and ``mean_codebleu`` reported alongside.

    GeoAnalystBench is 100% text, so every variant (including text-only Qwen)
    runs the full task set.

    Args:
        variant: The reasoner variant under test.
        tasks: The GeoAnalystBench tasks.
        backend: Injected backend; built lazily when ``None``.
        seed: Seed tag (logging / reproducibility).
        pass_threshold: Workflow-similarity threshold for the per-task pass.
        trace_sink: Optional side-effect-only callback invoked once per task
            with a per-task trace record (see the US-049 JSONL dump). It MUST
            NOT mutate the returned metrics.

    Returns:
        A mapping ``{pass_rate, mean_semantic_sim, mean_codebleu, n}``.
    """
    resolved_backend = _resolve_backend(variant, backend)
    logger.info(
        "geoanalyst_pass_threshold",
        variant=variant.name,
        seed=seed,
        pass_threshold=pass_threshold,
    )
    sims: list[float] = []
    bleus: list[float] = []
    passes: list[float] = []

    for task in tasks:
        prompt = _build_geo_prompt(task)
        errored = False
        try:
            # Per-attempt timeout + transient-error retry live inside
            # ``_run_backend_text`` (US-069); no outer ``wait_for`` (it would abort
            # the retry chain).
            answer = await _run_backend_text(resolved_backend, prompt, [])
        except Exception as exc:  # noqa: BLE001 - one task must not crash the run
            logger.warning(
                "geoanalyst_task_failed",
                variant=variant.name,
                task_id=task.id,
                error=str(exc),
            )
            answer = ""
            errored = True
        workflow, code = _split_workflow_and_code(answer)
        sim = agent_metrics.workflow_semantic_similarity(workflow, task.human_workflow)
        bleu = agent_metrics.codebleu_score(code, task.code_string)
        sims.append(sim)
        bleus.append(bleu)
        passed = sim > pass_threshold
        passes.append(1.0 if passed else 0.0)
        if len(passes) % _PROGRESS_EVERY == 0:
            logger.info("geoanalyst_progress", variant=variant.name, evaluated=len(passes))

        if trace_sink is not None:
            trace_sink(
                {
                    "schema_version": _TRACE_SCHEMA_VERSION,
                    "benchmark": "GeoAnalystBench",
                    "variant": variant.name,
                    "model": variant.model,
                    "seed": seed,
                    "task_id": task.id,
                    "task": task.task,
                    "prompt": _truncate(prompt, PROMPT_TRACE_CAP),
                    "prompt_chars": len(prompt),
                    "gold": _truncate(task.human_workflow, PROMPT_TRACE_CAP),
                    "prediction": _truncate(answer, PRED_TRACE_CAP),
                    "prediction_chars": len(answer),
                    "workflow_sim": sim,
                    "codebleu": bleu,
                    "passed": passed,
                    "errored": errored,
                    "correct": passed,
                }
            )

    n = len(tasks)
    pass_rate = sum(passes) / n if n else 0.0
    mean_sim = sum(sims) / n if n else 0.0
    mean_bleu = sum(bleus) / n if n else 0.0
    logger.info(
        "geoanalyst_eval_done",
        variant=variant.name,
        seed=seed,
        pass_rate=pass_rate,
        n=n,
    )
    return {
        "pass_rate": pass_rate,
        "mean_semantic_sim": mean_sim,
        "mean_codebleu": mean_bleu,
        "n": n,
    }


def _aggregate(per_seed: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metric dicts into ``{metric: {mean, std}}``.

    NaN values (e.g. hallucination with no judge) are excluded from the mean/std
    so a missing metric stays NaN instead of poisoning the aggregate.

    Args:
        per_seed: One metric mapping per seed.

    Returns:
        ``{metric: {"mean": float, "std": float}}`` over the seeds.
    """
    metric_names: set[str] = set()
    for seed_metrics in per_seed:
        metric_names.update(seed_metrics.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [
            float(seed_metrics[metric])
            for seed_metrics in per_seed
            if metric in seed_metrics and not _is_nan(seed_metrics[metric])
        ]
        if not values:
            aggregated[metric] = {"mean": math.nan, "std": math.nan}
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
        else:
            std = 0.0
        aggregated[metric] = {"mean": mean, "std": std}
    return aggregated


def _is_nan(value: Any) -> bool:
    """Return whether ``value`` is a float NaN.

    Args:
        value: Any candidate value.

    Returns:
        ``True`` when ``value`` is a float NaN.
    """
    return isinstance(value, float) and math.isnan(value)


def run_benchmark(
    variants: Sequence[ReasonerVariant],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    agromind_path: Path = DEFAULT_AGROMIND_PATH,
    geo_path: Path = DEFAULT_GEO_PATH,
    backends: dict[str, LLMBackend] | None = None,
    judge: HallucinationJudge | None = None,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    report_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    log_mlflow: bool = True,
    probe_server: bool = True,
    dump_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Run both benchmarks for every variant over the seeds and report.

    Loads the real datasets once, then for each variant and seed evaluates
    AgroMind and GeoAnalystBench, aggregates ``mean +- std`` across seeds
    (error bars, AC-4), optionally logs every metric to MLflow with the
    ``code_version`` + ``data_version`` tags (AC-6, lineage on ``:5010``) and
    finally builds the HTML comparison report (AC-4/AC-5).

    Backends are injectable per variant via ``backends[variant.name]`` so the
    whole run is deterministic and offline in tests; when a variant has no
    injected backend one is built with :func:`make_backend` (real API / vLLM).

    Args:
        variants: The reasoner variants to evaluate.
        seeds: The evaluation seeds (3 by default, AC-4).
        agromind_path: Path to the AgroMind subset JSON.
        geo_path: Path to the GeoAnalystBench CSV.
        backends: Optional ``{variant_name: backend}`` injection map.
        judge: Optional hallucination judge (NaN when absent).
        image_root: Base folder for the AgroMind subset images.
        report_path: Output HTML path; defaults to
            ``reports/agent_bench/agent_bench.html``.
        log_mlflow: Whether to log the run to MLflow (AC-6).
        probe_server: Forwarded to ``track_experiment`` (set ``False`` in tests).
        dump_jsonl: Optional folder where the per-item inference trace is dumped
            as one JSONL file per (variant, benchmark). When set, the trace is
            also read back to feed the report's answer-type breakdown and the
            example rows; when ``None`` no trace is written and the report is
            unchanged.

    Returns:
        The nested results mapping
        ``{variant: {benchmark: {metric: {"mean", "std"}}}}`` (also passed to
        :func:`build_report_html`).
    """
    backends = backends or {}
    items = load_agromind_subset(agromind_path)
    tasks = load_geoanalystbench(geo_path)

    # Evaluate the paid cloud variant (Gemini) FIRST so it is checkpointed
    # earliest and never recomputed if a later on-prem variant fails -- do not
    # re-pay the Gemini API (US-049 hardening).
    variants = sorted(variants, key=lambda v: 0 if "gemini" in v.name.lower() else 1)

    results: dict[str, Any] = {}
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        logger.info(
            "benchmark_checkpoint_loaded",
            path=str(checkpoint_path),
            done=sorted(results),
        )

    for variant in variants:
        if variant.name in results:
            logger.info("variant_skipped_resumed", variant=variant.name)
            continue
        backend = backends.get(variant.name)
        agromind_seeds: list[dict[str, float]] = []
        geo_seeds: list[dict[str, float]] = []

        # Open one writer per (variant, benchmark) BEFORE the seed loop (truncate
        # mode) so all seeds append into a single file; records carry ``seed`` so
        # they stay distinguishable. try/finally so a sink write failure never
        # aborts the MLflow logging + report (eval-only no-crash contract).
        agromind_writer: _JsonlTraceWriter | None = None
        geo_writer: _JsonlTraceWriter | None = None
        if dump_jsonl is not None:
            agromind_writer = _JsonlTraceWriter(
                dump_jsonl / f"trace_{variant.name}_AgroMind.jsonl",
                variant=variant.name,
                benchmark="AgroMind",
            )
            geo_writer = _JsonlTraceWriter(
                dump_jsonl / f"trace_{variant.name}_GeoAnalystBench.jsonl",
                variant=variant.name,
                benchmark="GeoAnalystBench",
            )
        agromind_sink = agromind_writer.sink if agromind_writer is not None else None
        geo_sink = geo_writer.sink if geo_writer is not None else None
        try:
            for seed in seeds:
                agromind_seeds.append(
                    asyncio.run(
                        eval_agromind(
                            variant,
                            items,
                            backend=backend,
                            judge=judge,
                            seed=seed,
                            image_root=image_root,
                            trace_sink=agromind_sink,
                        )
                    )
                )
                geo_seeds.append(
                    asyncio.run(
                        eval_geoanalyst(
                            variant,
                            tasks,
                            backend=backend,
                            seed=seed,
                            trace_sink=geo_sink,
                        )
                    )
                )
        finally:
            if agromind_writer is not None:
                agromind_writer.close()
            if geo_writer is not None:
                geo_writer.close()
        results[variant.name] = {
            "AgroMind": _aggregate(agromind_seeds),
            "GeoAnalystBench": _aggregate(geo_seeds),
        }
        if checkpoint_path is not None:
            _save_checkpoint(results, checkpoint_path)
            logger.info(
                "variant_checkpointed",
                variant=variant.name,
                path=str(checkpoint_path),
            )

    if log_mlflow:
        _log_to_mlflow(results, agromind_path, probe_server=probe_server)

    out_path = report_path or (DEFAULT_REPORT_DIR / "agent_bench.html")
    if dump_jsonl is not None:
        variant_names = [v.name for v in variants]
        breakdown = _answer_type_breakdown(dump_jsonl, variant_names)
        examples = _example_records(dump_jsonl, variant_names)
        build_report_html(
            results,
            out_path,
            examples=examples,
            answer_type_breakdown=breakdown,
        )
    else:
        build_report_html(results, out_path)
    logger.info("agent_bench_done", variants=[v.name for v in variants], report=str(out_path))
    return results


def _read_trace_file(path: Path) -> list[dict[str, Any]]:
    """Read a per-variant JSONL trace file back into a list of records.

    Tolerant of a missing file (returns ``[]``) and of a truncated last line
    from a crashed run (the undecodable line is skipped). Always opened with
    ``encoding="utf-8"`` (the dump is written the same way; a bare ``open`` would
    raise on the accented Spanish prose under Windows cp1252).

    Args:
        path: The JSONL file path.

    Returns:
        The parsed records (order preserved), or ``[]`` when the file is absent.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    skipped = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped = True
    if skipped:
        logger.warning("trace_dump_line_skipped", path=str(path))
    return records


def _answer_type_breakdown(
    dump_dir: Path, variants: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the AgroMind answer-type breakdown from the dumped traces.

    Args:
        dump_dir: Folder holding the per-variant JSONL trace files.
        variants: Variant names to read back.

    Returns:
        ``{variant: {answer_type: {"n": float, "exact_match_mean": float}}}``
        over the AgroMind records (empty inner dict when a variant has no dump).
    """
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variants:
        path = dump_dir / f"trace_{variant}_AgroMind.jsonl"
        records = _read_trace_file(path)
        buckets: dict[str, list[float]] = {}
        for record in records:
            answer_type = str(record.get("answer_type", "open_text"))
            buckets.setdefault(answer_type, []).append(float(record.get("exact_match", 0.0)))
        breakdown[variant] = {
            answer_type: {
                "n": float(len(scores)),
                "exact_match_mean": sum(scores) / len(scores) if scores else 0.0,
            }
            for answer_type, scores in buckets.items()
        }
    return breakdown


def _example_records(
    dump_dir: Path, variants: Sequence[str], *, per_bucket: int = 3
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Pick a few correct/wrong example records per (variant, benchmark).

    Selection is order-deterministic (the dump preserves the fixed-seed item
    iteration order), so re-running with different seeds changes predictions but
    not which records surface first.

    Args:
        dump_dir: Folder holding the per-variant JSONL trace files.
        variants: Variant names to read back.
        per_bucket: How many correct and how many wrong examples to keep.

    Returns:
        ``{variant: {benchmark: [record, ...]}}`` with up to ``per_bucket``
        correct followed by up to ``per_bucket`` wrong records per benchmark.
    """
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in variants:
        per_benchmark: dict[str, list[dict[str, Any]]] = {}
        for benchmark in ("AgroMind", "GeoAnalystBench"):
            path = dump_dir / f"trace_{variant}_{benchmark}.jsonl"
            records = _read_trace_file(path)
            correct = [r for r in records if r.get("correct")][:per_bucket]
            wrong = [r for r in records if not r.get("correct")][:per_bucket]
            per_benchmark[benchmark] = correct + wrong
        examples[variant] = per_benchmark
    return examples


def _save_checkpoint(results: dict[str, Any], path: Path) -> None:
    """Persist the per-variant results to disk atomically (US-049 hardening).

    Written after each variant completes so a later failure (dropped tunnel,
    wedged socket) never forces recomputing the variants already done -- notably
    the paid Gemini variant. The write goes through a temp file + atomic replace
    so a crash mid-write cannot corrupt the checkpoint.

    Args:
        results: The accumulated ``{variant: {benchmark: ...}}`` mapping.
        path: Destination JSON path (parent dirs created as needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _log_to_mlflow(results: dict[str, Any], agromind_path: Path, *, probe_server: bool) -> None:
    """Log the aggregated results to MLflow with versioning tags (AC-6).

    Opens one run via :func:`track_experiment` (which sets ``code_version`` and
    ``data_version``) and logs every ``{variant}/{benchmark}/{metric}`` mean and
    std as a metric. Logging failures are caught and logged: the benchmark and
    its report must still complete (eval-only, no training side effects).

    Args:
        results: The nested results mapping.
        agromind_path: DVC-tracked subset path used for the ``data_version`` tag.
        probe_server: Forwarded to ``track_experiment``.
    """
    try:
        import mlflow

        from ml.utils.mlflow_utils import track_experiment

        with track_experiment(
            _EXPERIMENT_NAME, dvc_path=str(agromind_path), probe_server=probe_server
        ):
            for variant, benchmarks in results.items():
                for benchmark, metrics in benchmarks.items():
                    for metric, stats in metrics.items():
                        mean = stats.get("mean", math.nan)
                        std = stats.get("std", math.nan)
                        if not _is_nan(mean):
                            mlflow.log_metric(f"{variant}/{benchmark}/{metric}/mean", mean)
                        if not _is_nan(std):
                            mlflow.log_metric(f"{variant}/{benchmark}/{metric}/std", std)
        logger.info("agent_bench_mlflow_logged", experiment=_EXPERIMENT_NAME)
    except Exception as exc:  # noqa: BLE001 - tracking must not break the eval run
        logger.warning("agent_bench_mlflow_failed", error=str(exc))


def _resolve_variants(names: Sequence[str] | None) -> list[ReasonerVariant]:
    """Resolve CLI variant tags to :class:`ReasonerVariant` objects.

    Args:
        names: The variant tags from the CLI, or ``None`` for all three.

    Returns:
        The resolved variants (defaults to all three, order preserved).
    """
    if not names:
        return list(DEFAULT_VARIANTS)
    resolved: list[ReasonerVariant] = []
    for name in names:
        variant = _VARIANTS_BY_NAME.get(name)
        if variant is None:
            valid = sorted(_VARIANTS_BY_NAME)
            raise SystemExit(f"Variante desconocida: {name!r}. Validas: {valid}")
        resolved.append(variant)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the agent benchmark.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evalua el copiloto AgroSat en AgroMind y GeoAnalystBench "
            "(eval-only, sin entrenamiento; US-049)."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(_VARIANTS_BY_NAME),
        default=None,
        help="Variantes a evaluar (por defecto las tres).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds de evaluacion para las barras de error (por defecto 0 1 2).",
    )
    parser.add_argument(
        "--agromind",
        type=Path,
        default=DEFAULT_AGROMIND_PATH,
        help="Ruta al subset JSON de AgroMind.",
    )
    parser.add_argument(
        "--geo",
        type=Path,
        default=DEFAULT_GEO_PATH,
        help="Ruta al CSV de GeoAnalystBench.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Carpeta base de las imagenes del subset de AgroMind.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Ruta de salida del reporte HTML.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="No registrar la corrida en MLflow.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta del checkpoint JSON; cada variante se persiste al terminar.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar desde el checkpoint: salta variantes ya evaluadas.",
    )
    parser.add_argument(
        "--dump-jsonl",
        type=Path,
        default=None,
        help="Carpeta donde volcar la traza por item (JSONL, una por variante y benchmark).",
    )
    args = parser.parse_args(argv)

    variants = _resolve_variants(args.variants)
    results = run_benchmark(
        variants,
        seeds=tuple(args.seeds),
        agromind_path=args.agromind,
        geo_path=args.geo,
        image_root=args.image_root,
        report_path=args.report,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        log_mlflow=not args.no_mlflow,
        dump_jsonl=args.dump_jsonl,
    )
    logger.info("agent_bench_cli_done", n_variants=len(variants), n_seeds=len(args.seeds))
    # Emit a compact JSON summary to stdout for the calling script / operator.
    _print_summary(results)
    return 0


def _print_summary(results: dict[str, Any]) -> None:
    """Write a compact JSON summary of the headline metrics to stdout.

    Args:
        results: The nested results mapping returned by :func:`run_benchmark`.
    """
    summary: dict[str, Any] = {}
    for variant, benchmarks in results.items():
        em = benchmarks.get("AgroMind", {}).get("exact_match", {})
        pr = benchmarks.get("GeoAnalystBench", {}).get("pass_rate", {})
        summary[variant] = {
            "AgroMind/exact_match": em.get("mean"),
            "GeoAnalystBench/pass_rate": pr.get("mean"),
        }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
