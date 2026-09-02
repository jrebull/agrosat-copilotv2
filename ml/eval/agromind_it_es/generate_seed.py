"""Seed generator scaffold for AgroMind-IT/ES (US-068).

Builds the *seed* of the bilingual benchmark with Gemini 2.5-pro anchored on
REAL Sentinel-2 images of Italy, one multiple-choice agricultural Q&A pair per
copilot question family (:class:`~ml.eval.agromind_it_es.schema.QuestionFamily`)
per language (``it`` / ``es``). The seed is a *draft*: it only becomes the
benchmark after native human review (the Streamlit review app); nothing here is
ever the final 500-pair set (Arthur's rule: no synthetic / placeholder pairs in
the published dataset).

Two modes:

- ``dry_run=True`` (the AUTONOMOUS mode of US-068, and the default when no
  Gemini key is configured): emits the PLAN -- the rendered prompt and the
  target pair count per family x language -- WITHOUT calling the API. Each
  emitted :class:`QAItem` is tagged ``source="dry-run"`` and is NOT a real pair.
- ``dry_run=False`` with a configured key and image root: calls Gemini 2.5-pro
  (model id read from ``get_settings().gemini_model``) per family x language,
  parses the structured JSON answer into a :class:`QAItem` tagged
  ``source="gemini-seed"``, ready for human review. This path is BLOCKED here
  (needs a live Gemini key + GEE-downloaded S2 images of Italy, see
  ``docs/blockers/epic11-notas.md`` B1); the code is ready to run when they
  exist.

The Gemini model id is read from settings (NEVER hardcoded, NEVER ``os.environ``)
and the spec id is ``gemini-2.5-pro`` (GA, 1M ctx). The credential is read via
``get_settings()``; when it is missing the generator degrades to ``dry_run``.

Project conventions: identifiers / docstrings in English; the prompt prose is in
Italian / Spanish per the target language; ``structlog`` (never ``print``); full
type hints; no emojis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ml.eval.agromind_it_es.schema import (
    QAItem,
    QuestionFamily,
    dump_jsonl,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_SEED_PATH",
    "FAMILY_INTENTS",
    "GEMINI_SPEC_MODEL",
    "SeedGenerator",
    "build_generation_prompt",
    "main",
]

#: The spec-mandated Gemini model id (GA, 1M ctx). The ACTUAL id is read from
#: ``settings.gemini_model``; this constant documents the US-068 default and is
#: used when settings are unavailable (tests inject a client / use dry-run).
GEMINI_SPEC_MODEL: str = "gemini-2.5-pro"

#: Default destination of the generated seed JSONL (gitignored when large; only
#: the tiny ``seed.fixture.jsonl`` is committed).
DEFAULT_SEED_PATH: Path = Path("data/benchmark/agromind_it_es/seed.jsonl")

#: One-line description of the copilot intent each family probes, woven into the
#: generation prompt so Gemini grounds the pair in a real copilot capability.
#: Keyed by family; values bilingual-neutral (English here; the prompt renders
#: the language-specific instruction around them).
FAMILY_INTENTS: dict[QuestionFamily, str] = {
    QuestionFamily.CLASSIFICATION: (
        "identify the dominant crop type of a parcel from its Sentinel-2 imagery"
    ),
    QuestionFamily.QUANTIFICATION: (
        "estimate an area or a count of parcels / a crop class in the scene"
    ),
    QuestionFamily.VIGOR: ("assess vegetation vigour using NDVI / EVI from the imagery"),
    QuestionFamily.WATER_STRESS: ("assess water stress / moisture using NDWI from the imagery"),
    QuestionFamily.PHENOLOGY: ("reason about the phenological stage of the crop in the scene"),
    QuestionFamily.COMPARISON: ("compare two parcels (or two model outputs) shown in the scene"),
    QuestionFamily.ANOMALY: ("spot a temporal or spatial anomaly / outlier in the parcel"),
    QuestionFamily.METADATA: (
        "read a parcel attribute or acquisition metadata from the scene context"
    ),
    QuestionFamily.INTERSECTION: ("reason about spatial neighbourhood / intersection of parcels"),
    QuestionFamily.EXPLAINABILITY: ("explain why a crop prediction was made for the parcel"),
}

#: Per-language framing of the generation instruction. Visible prose is in the
#: target language (Italian / Spanish); the JSON contract keys stay English so
#: the parser is language-agnostic.
_LANG_FRAMING: dict[str, dict[str, str]] = {
    "it": {
        "role": (
            "Sei un esperto di agricoltura satellitare. Genera UNA domanda a "
            "scelta multipla in italiano su un'immagine Sentinel-2 reale "
            "dell'Italia."
        ),
        "intent_lead": "La domanda deve riguardare:",
        "contract": (
            "Rispondi SOLO con un oggetto JSON con le chiavi: question (stringa "
            "in italiano), options (oggetto con chiavi A, B, C, D), answer (la "
            "lettera corretta). Non aggiungere testo fuori dal JSON."
        ),
    },
    "es": {
        "role": (
            "Eres un experto en agricultura satelital. Genera UNA pregunta de "
            "opcion multiple en espanol sobre una imagen Sentinel-2 real de "
            "Italia."
        ),
        "intent_lead": "La pregunta debe tratar sobre:",
        "contract": (
            "Responde SOLO con un objeto JSON con las claves: question (cadena "
            "en espanol), options (objeto con claves A, B, C, D), answer (la "
            "letra correcta). No agregues texto fuera del JSON."
        ),
    },
}


def build_generation_prompt(family: QuestionFamily, language: str) -> str:
    """Build the Gemini 2.5-pro generation prompt for a family x language.

    The prompt instructs the model to produce ONE multiple-choice agricultural
    Q&A pair, in ``language``, grounded on the attached real Sentinel-2 image of
    Italy, probing the given copilot ``family`` intent, and to answer ONLY with a
    strict JSON object (``question`` / ``options`` / ``answer``) so the response
    is machine-parseable.

    Args:
        family: The copilot question family to probe.
        language: Target language (``it`` or ``es``).

    Returns:
        The composed prompt string.

    Raises:
        ValueError: When ``language`` is not ``it`` / ``es``.
    """
    framing = _LANG_FRAMING.get(language)
    if framing is None:
        raise ValueError(f"unsupported language {language!r}; expected 'it' or 'es'")
    intent = FAMILY_INTENTS[family]
    return "\n".join(
        [
            framing["role"],
            "",
            f"{framing['intent_lead']} {intent} (family: {family.value}).",
            "",
            framing["contract"],
        ]
    )


@dataclass
class SeedGenerator:
    """Generate the AgroMind-IT/ES seed via Gemini 2.5-pro (or dry-run).

    The model id is read from ``get_settings().gemini_model`` (single source of
    truth, never ``os.environ``); the credential likewise. When neither a key
    nor an injected client is available the generator forces ``dry_run`` and
    emits the plan without any network call -- the autonomous mode of US-068.

    Attributes:
        model: The resolved Gemini model id (defaults to
            :data:`GEMINI_SPEC_MODEL` when settings are unavailable).
        client: An injected ``google.genai.Client`` (tests / explicit wiring),
            or ``None`` to build lazily from settings on the real path.
        image_root: Base folder holding the real Sentinel-2 images of Italy.
            Absent images degrade an item to text-only.
    """

    model: str = GEMINI_SPEC_MODEL
    client: Any | None = None
    image_root: Path | None = None

    @classmethod
    def from_settings(cls, *, image_root: Path | None = None) -> SeedGenerator:
        """Build a generator resolving the Gemini model id from settings.

        The benchmark seed is mandated to use ``gemini-2.5-pro`` (US-068 AC, GA,
        1M ctx) -- a role DISTINCT from the copilot reasoner. ``settings`` is
        still the single source of truth, but only a ``gemini-2.5`` family id is
        adopted from it; any other configured id (e.g. the copilot's
        ``gemini-3.5-flash`` or a stale ``gemini-3.1-pro`` override in
        ``.env.local``) is NOT propagated to the benchmark, so the generator is
        never silently pinned to the wrong model. The id is never hardcoded past
        the spec default and never read from ``os.environ``.

        Args:
            image_root: Base folder for the real S2 images of Italy.

        Returns:
            A :class:`SeedGenerator`. The credential is not eagerly fetched; the
            real path builds the ``google.genai`` client lazily and falls back
            to ``dry_run`` when the key is missing.
        """
        model = GEMINI_SPEC_MODEL
        try:
            from backend.app.core.config import get_settings

            configured = get_settings().gemini_model
            if configured and configured.startswith("gemini-2.5"):
                model = configured
            elif configured:
                logger.info(
                    "agromind_it_es_settings_model_ignored",
                    configured=configured,
                    using=model,
                    reason="benchmark_seed_requires_gemini_2_5_pro",
                )
        except Exception:  # noqa: BLE001 - settings optional (tests / no .env.local)
            logger.info("agromind_it_es_settings_unavailable", fallback_model=model)
        return cls(model=model, image_root=image_root)

    def _has_credential(self) -> bool:
        """Return whether a Gemini credential is configured.

        Returns:
            ``True`` when a client is injected or a key / Vertex project is set
            in settings; ``False`` otherwise (forces dry-run).
        """
        if self.client is not None:
            return True
        try:
            from backend.app.core.config import get_settings

            settings = get_settings()
        except Exception:  # noqa: BLE001 - no settings -> no credential
            return False
        return bool(settings.gemini_api_key or settings.google_cloud_project)

    def generate(
        self,
        family: QuestionFamily,
        language: str,
        n: int,
        *,
        dry_run: bool | None = None,
    ) -> list[QAItem]:
        """Generate ``n`` seed items for one family x language.

        In dry-run mode (explicit, or implied when no credential is configured)
        it emits ``n`` plan placeholders (``source="dry-run"``) carrying the
        rendered prompt as the question, WITHOUT calling the API. On the real
        path it calls Gemini ``n`` times and parses each structured JSON answer
        into a ``source="gemini-seed"`` item ready for human review.

        Args:
            family: The copilot question family.
            language: Target language (``it`` or ``es``).
            n: Number of pairs to generate for this cell.
            dry_run: Force dry-run; when ``None`` it is inferred from the
                credential availability.

        Returns:
            The list of generated :class:`QAItem` (length ``n``).
        """
        effective_dry_run = dry_run if dry_run is not None else not self._has_credential()
        prompt = build_generation_prompt(family, language)
        if effective_dry_run:
            logger.info(
                "agromind_it_es_dry_run",
                family=family.value,
                language=language,
                n=n,
                model=self.model,
            )
            return [self._dry_run_item(family, language, i, prompt) for i in range(n)]

        logger.info(
            "agromind_it_es_generate_real",
            family=family.value,
            language=language,
            n=n,
            model=self.model,
        )
        return [self._generate_one(family, language, i, prompt) for i in range(n)]

    def generate_plan(
        self,
        families: Sequence[QuestionFamily],
        languages: Sequence[str],
        n_per_family: int,
        *,
        dry_run: bool | None = None,
    ) -> list[QAItem]:
        """Generate the full plan over the family x language grid.

        Args:
            families: The families to cover (defaults to all ten upstream).
            languages: The languages to cover (``it`` / ``es``).
            n_per_family: Target pairs per family per language.
            dry_run: Force dry-run; ``None`` infers from credential availability.

        Returns:
            The flattened list of all generated items.
        """
        items: list[QAItem] = []
        for language in languages:
            for family in families:
                items.extend(self.generate(family, language, n_per_family, dry_run=dry_run))
        logger.info(
            "agromind_it_es_plan_done",
            n_families=len(families),
            n_languages=len(languages),
            n_per_family=n_per_family,
            n_total=len(items),
        )
        return items

    def _dry_run_item(
        self, family: QuestionFamily, language: str, index: int, prompt: str
    ) -> QAItem:
        """Build one dry-run placeholder item (no API call).

        Args:
            family: The copilot question family.
            language: Target language.
            index: 0-based index within the family x language cell.
            prompt: The rendered generation prompt (stored as the question so the
                plan is inspectable; this is NOT a real benchmark pair).

        Returns:
            A ``source="dry-run"`` :class:`QAItem`.
        """
        return QAItem(
            item_id=f"{language}-{family.value}-{index:04d}",
            category=family,
            lang=language,  # type: ignore[arg-type]  # caller passes it/es
            question=f"[DRY-RUN PROMPT] {prompt}",
            options={},
            answer="",
            image=None,
            is_multimodal=False,
            reviewed=False,
            reviewer=None,
            source="dry-run",
        ).with_derived_flags()

    def _generate_one(
        self, family: QuestionFamily, language: str, index: int, prompt: str
    ) -> QAItem:
        """Call Gemini once and parse one Q&A pair (the real, blocked path).

        Args:
            family: The copilot question family.
            language: Target language.
            index: 0-based index within the cell.
            prompt: The rendered generation prompt.

        Returns:
            A ``source="gemini-seed"`` :class:`QAItem` ready for human review.
        """
        client = self._resolve_client()
        image_part = self._resolve_image_part(family, index)
        contents = self._build_contents(prompt, image_part)
        response = client.models.generate_content(model=self.model, contents=contents)
        raw_text = getattr(response, "text", "") or ""
        parsed = _parse_qa_json(raw_text)
        # Record the real, deterministic image path that was grounded into Gemini
        # (``image_part`` is built from ``{family}_{index:04d}.png``). The model
        # never echoes the image back, so reading ``parsed["_image"]`` always
        # yielded ``None`` and silently downgraded the pair to text-only -- the
        # exact failure that made the AgroMind subset text-only in US-049. Anchor
        # it to the source PNG so ``is_multimodal`` stays True for the VLM eval.
        image_ref = f"{family.value}_{index:04d}.png" if image_part is not None else None
        return QAItem(
            item_id=f"{language}-{family.value}-{index:04d}",
            category=family,
            lang=language,  # type: ignore[arg-type]  # caller passes it/es
            question=str(parsed.get("question", "")),
            options={str(k): str(v) for k, v in (parsed.get("options") or {}).items()},
            answer=str(parsed.get("answer", "")),
            image=image_ref,
            is_multimodal=image_part is not None,
            reviewed=False,
            reviewer=None,
            source="gemini-seed",
        ).with_derived_flags()

    def _resolve_client(self) -> Any:
        """Return the injected client or build a ``google.genai`` client lazily.

        Returns:
            The ``google.genai.Client`` to call (or the injected test double).
        """
        if self.client is not None:
            return self.client
        from google import genai

        from backend.app.core.config import get_settings

        settings = get_settings()
        if settings.gemini_api_key:
            return genai.Client(api_key=settings.gemini_api_key)
        return genai.Client()

    def _resolve_image_part(self, family: QuestionFamily, index: int) -> Any | None:
        """Resolve the real Sentinel-2 image part for an item, if present.

        Args:
            family: The copilot question family (used to look up an image).
            index: 0-based index within the cell.

        Returns:
            A ``google.genai`` image part, or ``None`` when no image is found
            (the item degrades to text-only).
        """
        if self.image_root is None:
            return None
        candidate = Path(self.image_root) / f"{family.value}_{index:04d}.png"
        if not candidate.exists():
            return None
        from google.genai import types

        return types.Part.from_bytes(data=candidate.read_bytes(), mime_type="image/png")

    @staticmethod
    def _build_contents(prompt: str, image_part: Any | None) -> list[Any]:
        """Build the ``contents`` list for a single generation turn.

        Args:
            prompt: The textual generation prompt.
            image_part: An optional image part to attach before the text.

        Returns:
            A one-element list with a ``types.Content`` user turn.
        """
        from google.genai import types

        parts = []
        if image_part is not None:
            parts.append(image_part)
        parts.append(types.Part.from_text(text=prompt))
        return [types.Content(role="user", parts=parts)]


def _parse_qa_json(raw_text: str) -> dict[str, Any]:
    """Parse the strict JSON object emitted by the generation prompt.

    Tolerates a fenced ```json``` block around the object. Returns an empty dict
    on a malformed answer so the caller still produces a (low-quality) item that
    the human review step can reject -- no silent crash mid-batch.

    Args:
        raw_text: The raw model answer.

    Returns:
        The parsed dict (``{}`` when no JSON object can be extracted).
    """
    text = raw_text.strip()
    if "```" in text:
        first = text.find("```")
        rest = text[first + 3 :]
        if rest.lower().startswith("json"):
            rest = rest[4:]
        end = rest.find("```")
        text = rest[:end] if end != -1 else rest
    text = text.strip()
    start, stop = text.find("{"), text.rfind("}")
    if start == -1 or stop == -1 or stop <= start:
        return {}
    try:
        parsed = json.loads(text[start : stop + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the seed generator.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genera el seed del benchmark AgroMind-IT/ES con Gemini 2.5-pro "
            "sobre imagenes Sentinel-2 reales de Italia. Sin key / imagenes "
            "corre en dry-run (emite el plan sin llamar a la API)."
        )
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Carpeta con imagenes Sentinel-2 reales de Italia (BLOCKER B1 sin ella).",
    )
    parser.add_argument(
        "--n-per-family",
        type=int,
        default=25,
        help="Pares objetivo por familia por idioma (default 25 -> 250 por idioma).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["it", "es"],
        choices=["it", "es"],
        help="Idiomas a generar (default: it es).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="Ruta JSONL de salida del seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forzar dry-run (sin API). Por defecto se infiere de la credencial.",
    )
    args = parser.parse_args(argv)

    generator = SeedGenerator.from_settings(image_root=args.image_root)
    families = list(QuestionFamily)
    items = generator.generate_plan(
        families,
        args.languages,
        args.n_per_family,
        dry_run=(True if args.dry_run else None),
    )
    n = dump_jsonl(items, args.out)
    logger.info("agromind_it_es_seed_written", out=str(args.out), n_records=n)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
