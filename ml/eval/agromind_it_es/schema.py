"""Schema of the bilingual AgroMind-IT/ES benchmark (US-068).

This module is the single source of truth for one AgroMind-IT/ES Q&A record.
It is designed so that compatibility with the original AgroMind format is
*verifiable*, not declared: :func:`to_agromind_item` IMPORTS the real
:class:`ml.eval.agent_bench.AgroMindItem` and constructs it from a
:class:`QAItem`. If that construction succeeds the record is, by definition,
consumable by ``ml.eval.agent_bench.load_agromind_subset`` (or a twin loader)
without changing the harness -- this is exactly what the schema test exercises.

The record carries the fields requested by the AgroMind-IT/ES spec
(``image``, ``question``, ``answer``, ``category``, ``lang``) plus the
multiple-choice ``options`` and the review provenance (``reviewer``,
``source``, ``reviewed``) needed by the human-review app. ``category`` is one of
the ten copilot question families (:class:`QuestionFamily`); ``lang`` is the
ISO-639-1 code of the pair (``it`` or ``es``).

EVAL-ONLY GUARD: there is intentionally NO ``split`` field. Any record that
smuggles a train mark (a ``split`` key valued ``train`` / ``training``, or a
truthy ``is_train``) is rejected by :func:`validate_record` and therefore by
:func:`load_jsonl`. AgroMind has no train split, so a train-marked pair would be
leakage by construction.

Project conventions: identifiers and docstrings in English (Google style),
visible benchmark prose in Italian/Spanish; full type hints; ``structlog``
(never ``print``); no emojis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

    from ml.eval.agent_bench import AgroMindItem

logger = structlog.get_logger(__name__)

__all__ = [
    "ALLOWED_LANGUAGES",
    "ALLOWED_SOURCES",
    "QAItem",
    "QuestionFamily",
    "SchemaValidationError",
    "dump_jsonl",
    "load_jsonl",
    "to_agromind_item",
    "validate_record",
]


class QuestionFamily(StrEnum):
    """The ten copilot question families (US-068 AC, the catalog's intents).

    Each family anchors to a tool / intent of the conversational copilot (the
    nine ``ml/agent/tools/`` FunctionTools plus the phenology / explainability
    descriptors). The generator parametrises ``category`` by this enum so the
    250 ``it`` + 250 ``es`` target covers all ten families (~25 pairs per family
    per language). Identifiers are in English; the human-facing labels (Italian
    / Spanish) live in the prompt templates, not here.

    Members:
        CLASSIFICATION: Crop-type classification (``classify_new_parcel``).
        QUANTIFICATION: Area / count per class.
        VIGOR: Vegetation vigour via NDVI / EVI.
        WATER_STRESS: Water stress via NDWI / moisture.
        PHENOLOGY: Phenological stage (``phenology_descriptor``).
        COMPARISON: Model-vs-model or parcel-vs-parcel comparison.
        ANOMALY: Temporal outlier / anomaly detection.
        METADATA: Parcel attributes / acquisition date.
        INTERSECTION: Spatial join / neighbourhood (PostGIS).
        EXPLAINABILITY: Prediction explanation (``explain_prediction``).
    """

    CLASSIFICATION = "classification"
    QUANTIFICATION = "quantification"
    VIGOR = "vigor"
    WATER_STRESS = "water_stress"
    PHENOLOGY = "phenology"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    METADATA = "metadata"
    INTERSECTION = "intersection"
    EXPLAINABILITY = "explainability"


#: The two benchmark languages (ISO-639-1). Italian is reviewed by a Scuola
#: Sant'Anna native speaker, Spanish by a team member (US-068 AC).
ALLOWED_LANGUAGES: frozenset[str] = frozenset({"it", "es"})

#: Allowed provenance markers for a record. ``gemini-seed`` is a raw Gemini
#: 2.5-pro draft, ``dry-run`` is a scaffold placeholder emitted with no API
#: call, ``human-edited`` is the post-review benchmark record (the only one that
#: belongs in the published 500-pair set), and ``fixture`` tags the tiny
#: repo-committed example set used by tests (never the benchmark).
ALLOWED_SOURCES: frozenset[str] = frozenset({"gemini-seed", "dry-run", "human-edited", "fixture"})

#: JSON keys that, if present with a train value, mark a record as a training
#: example. The benchmark is eval-only, so :func:`validate_record` rejects them.
_TRAIN_SPLIT_VALUES: frozenset[str] = frozenset({"train", "training"})


class SchemaValidationError(ValueError):
    """Raised when a record violates the AgroMind-IT/ES schema or eval-only rule."""


@dataclass
class QAItem:
    """One bilingual AgroMind-IT/ES Q&A item.

    Mirrors the AgroMind multiple-choice shape (question + lettered options +
    gold letter) and adds the dataset's bilingual / review provenance. The
    ``category`` is one of the ten :class:`QuestionFamily` values; ``lang`` is
    ``it`` or ``es``.

    Attributes:
        item_id: Stable id within the dataset (e.g. ``it-vigor-0007``).
        category: The copilot question family this pair belongs to.
        lang: ISO-639-1 language of the pair (``it`` or ``es``).
        question: The question text in ``lang``.
        options: Mapping of choice label (``A``-``J``) to its text value. Empty
            for open-ended numeric / text items (kept for AgroMind parity).
        answer: Gold answer -- a choice letter for multiple-choice items, or a
            free number / short text for open items.
        image: Relative path to the question's Sentinel-2 image of Italy, or
            ``None`` for a text-only item. Mirrors AgroMind ``image_path``.
        is_multimodal: ``True`` when answering requires the image. Derived from
            ``image`` when not set explicitly (see :meth:`with_derived_flags`).
        reviewed: ``True`` once a native reviewer accepted / edited the pair.
        reviewer: Identifier of the human reviewer (``None`` until reviewed).
        source: Provenance, one of :data:`ALLOWED_SOURCES`.
        type_id: Optional AgroMind-style numeric type id (defaults to the
            family ordinal so the bridge to ``AgroMindItem`` is total).
    """

    item_id: str
    category: QuestionFamily
    lang: Literal["it", "es"]
    question: str
    options: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    image: str | None = None
    is_multimodal: bool = False
    reviewed: bool = False
    reviewer: str | None = None
    source: str = "dry-run"
    type_id: int | None = None

    def with_derived_flags(self) -> QAItem:
        """Return a copy with ``is_multimodal`` / ``type_id`` filled in.

        ``is_multimodal`` is derived from the presence of ``image``; ``type_id``
        defaults to the 1-based family ordinal when unset, so the bridge to
        :class:`~ml.eval.agent_bench.AgroMindItem` (which requires an int
        ``type_id``) is always well-defined.

        Returns:
            A new :class:`QAItem` with the derived fields populated.
        """
        derived_type = self.type_id
        if derived_type is None:
            derived_type = list(QuestionFamily).index(self.category) + 1
        return QAItem(
            item_id=self.item_id,
            category=self.category,
            lang=self.lang,
            question=self.question,
            options=dict(self.options),
            answer=self.answer,
            image=self.image,
            is_multimodal=bool(self.image),
            reviewed=self.reviewed,
            reviewer=self.reviewer,
            source=self.source,
            type_id=derived_type,
        )

    def to_record(self) -> dict[str, Any]:
        """Serialise to a JSON-ready record dict.

        Returns:
            A plain dict with ``category`` rendered as its string value and the
            optional ``image`` / ``reviewer`` kept as ``None`` when absent.
        """
        record = asdict(self)
        record["category"] = str(self.category)
        return record


def _coerce_family(value: Any) -> QuestionFamily:
    """Coerce a raw category value into a :class:`QuestionFamily`.

    Args:
        value: A :class:`QuestionFamily` or its string value.

    Returns:
        The matching :class:`QuestionFamily`.

    Raises:
        SchemaValidationError: When ``value`` is not one of the ten families.
    """
    if isinstance(value, QuestionFamily):
        return value
    try:
        return QuestionFamily(str(value))
    except ValueError as exc:
        allowed = ", ".join(f.value for f in QuestionFamily)
        raise SchemaValidationError(
            f"unknown category {value!r}; expected one of: {allowed}"
        ) from exc


def validate_record(record: dict[str, Any]) -> QAItem:
    """Validate a raw JSONL record and build a :class:`QAItem`.

    Enforces the schema (required fields, known family, known language, known
    source) and the eval-only rule: any train mark raises. This is the single
    gate every loaded record passes through.

    Args:
        record: A raw dict parsed from one JSONL line.

    Returns:
        The validated :class:`QAItem` (with derived flags filled in).

    Raises:
        SchemaValidationError: On a missing required field, an unknown
            family / language / source, or a smuggled train split mark.
    """
    # Eval-only guard FIRST: a train-marked record never becomes a QAItem.
    split = record.get("split")
    if isinstance(split, str) and split.strip().lower() in _TRAIN_SPLIT_VALUES:
        raise SchemaValidationError(
            "eval-only benchmark: a record with split="
            f"{split!r} is rejected (fine-tuning on AgroMind-IT/ES is leakage)"
        )
    if bool(record.get("is_train")):
        raise SchemaValidationError(
            "eval-only benchmark: a record with is_train=true is rejected "
            "(fine-tuning on AgroMind-IT/ES is leakage)"
        )

    for required in ("item_id", "category", "lang", "question"):
        if required not in record or record[required] in (None, ""):
            raise SchemaValidationError(f"missing required field {required!r}")

    lang = str(record["lang"])
    if lang not in ALLOWED_LANGUAGES:
        raise SchemaValidationError(
            f"unknown lang {lang!r}; expected one of {sorted(ALLOWED_LANGUAGES)}"
        )

    source = str(record.get("source", "dry-run"))
    if source not in ALLOWED_SOURCES:
        raise SchemaValidationError(
            f"unknown source {source!r}; expected one of {sorted(ALLOWED_SOURCES)}"
        )

    options_raw = record.get("options") or {}
    options = {str(label): str(value) for label, value in options_raw.items()}

    item = QAItem(
        item_id=str(record["item_id"]),
        category=_coerce_family(record["category"]),
        lang=lang,  # type: ignore[arg-type]  # checked against ALLOWED_LANGUAGES
        question=str(record["question"]),
        options=options,
        answer=str(record.get("answer", "")),
        image=(str(record["image"]) if record.get("image") else None),
        is_multimodal=bool(record.get("is_multimodal", False)),
        reviewed=bool(record.get("reviewed", False)),
        reviewer=(str(record["reviewer"]) if record.get("reviewer") else None),
        source=source,
        type_id=(int(record["type_id"]) if record.get("type_id") is not None else None),
    )
    return item.with_derived_flags()


def to_agromind_item(item: QAItem) -> AgroMindItem:
    """Bridge a :class:`QAItem` to the real :class:`AgroMindItem`.

    This is the *verifiable* compatibility proof for the US-068 acceptance
    criterion "JSONL schema compatible with the original AgroMind": it imports
    the canonical :class:`ml.eval.agent_bench.AgroMindItem` and constructs it
    from the bilingual record. If construction succeeds the record is consumable
    by the existing harness unchanged. The bridge never mutates the harness API.

    Args:
        item: The bilingual Q&A item.

    Returns:
        The equivalent :class:`AgroMindItem` (``task_file`` tags the language so
        the provenance survives the bridge).
    """
    from ml.eval.agent_bench import AgroMindItem

    filled = item.with_derived_flags()
    type_id = filled.type_id if filled.type_id is not None else 0
    return AgroMindItem(
        image_path=filled.image or "",
        question=filled.question,
        options=dict(filled.options),
        answer=filled.answer,
        type_id=type_id,
        item_id=_stable_int_id(filled.item_id),
        level1_id=type_id,
        level2_id=0,
        level3_id=0,
        task_file=f"AGROMIND-IT-ES/{filled.lang}",
        is_multimodal=bool(filled.image),
    )


def _stable_int_id(item_id: str) -> int:
    """Derive a stable non-negative int id from a string item id.

    ``AgroMindItem.item_id`` is an int; the bilingual ids are strings
    (``it-vigor-0007``). A trailing numeric suffix is used when present,
    otherwise a stable hash, so the bridge is total and deterministic.

    Args:
        item_id: The string item id.

    Returns:
        A non-negative integer id.
    """
    tail = item_id.rsplit("-", 1)[-1]
    if tail.isdigit():
        return int(tail)
    # Deterministic, process-independent fallback (no salt): zlib.crc32.
    import zlib

    return zlib.crc32(item_id.encode("utf-8"))


def dump_jsonl(items: Iterable[QAItem], path: Path) -> int:
    """Write items as one UTF-8 JSON object per line.

    Args:
        items: The Q&A items to serialise (each is validated by construction).
        path: Destination JSONL path (parent dirs are created).

    Returns:
        The number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")
            n += 1
    logger.info("agromind_it_es_jsonl_written", path=str(path), n_records=n)
    return n


def load_jsonl(path: Path) -> list[QAItem]:
    """Load and validate a JSONL benchmark file.

    Every line passes through :func:`validate_record`, so a malformed record or
    a smuggled train mark fails the load loudly (eval-only guarantee).

    Args:
        path: Path to the JSONL file.

    Returns:
        The list of validated :class:`QAItem` (order preserved).

    Raises:
        SchemaValidationError: On the first invalid / train-marked record.
    """
    path = Path(path)
    items: list[QAItem] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"line {line_no}: invalid JSON ({exc})") from exc
        try:
            items.append(validate_record(record))
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"line {line_no}: {exc}") from exc
    logger.info("agromind_it_es_jsonl_loaded", path=str(path), n_records=len(items))
    return items


def family_coverage(items: Sequence[QAItem]) -> dict[str, dict[str, int]]:
    """Count items per family per language (coverage report helper).

    Args:
        items: The loaded benchmark items.

    Returns:
        A mapping ``{family_value: {"it": n_it, "es": n_es}}`` over all ten
        families (zero-filled for families with no items yet).
    """
    coverage: dict[str, dict[str, int]] = {
        family.value: {"it": 0, "es": 0} for family in QuestionFamily
    }
    for item in items:
        coverage[item.category.value][item.lang] += 1
    return coverage
