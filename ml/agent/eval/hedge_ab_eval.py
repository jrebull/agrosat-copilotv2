"""Out-of-vocabulary hedge A/B harness for the copilot (US-081 AC8).

The copilot serves the Voting-3 v2 champion restricted to ``france-12`` (twelve
resolved crops). For the SIX dropped crops the v2 cannot resolve -- Winter
triticale, Fruits/vegetables/flowers, Potatoes, Leguminous fodder, Mixed cereal,
Sorghum -- the perceiver does NOT force an in-vocabulary label: it surfaces an
``unresolved_candidate`` and the reasoner is supposed to produce an HONEST HEDGE
grounded in neighbouring-parcel evidence (RAG) and phenology, not a confident
wrong call (the "handoff, not a wall" design of ``ml/agent/prompts.py``).

That handoff is, until now, a HYPOTHESIS WITH CABLING. This module turns it into
a MEASUREMENT: for each out-of-vocabulary parcel it runs the reasoner TWICE --

- **B (ungrounded)**: no ``retrieve_context`` evidence; the reasoner sees only the
  classifier output (the ``unresolved_candidate`` and the resolved-class
  posterior).
- **A (grounded)**: the SAME parcel, plus the citation-tagged grounding block the
  real ``retrieve_context`` tool emits for the neighbourhood
  (``_build_grounding_text`` shape).

and scores both analyses with an INJECTABLE judge on two axes:

- ``hedge_quality`` -- does the analysis (1) acknowledge the crop is outside the
  calibrated vocabulary instead of asserting a confident wrong label, and (2) make
  a USEFUL grounded conjecture toward the TRUE out-of-vocabulary crop? The
  grounded run is expected to score higher BECAUSE the true crop is only knowable
  from the neighbouring evidence.
- ``forced_label_rate`` -- the failure mode: does the analysis confidently report
  one of the twelve RESOLVED crops as THE answer (a forced in-vocabulary label)?
  Expected to be LOWER on the grounded run.

The headline is the delta ``hedge_quality_grounded - hedge_quality_ungrounded``.
HONEST FRAMING (mirrors :func:`ml.eval.agent_system_eval.eval_rag_ab`): the delta
is the improvement RELATIVE TO the ungrounded reasoner. A reasoner that already
hedges well ungrounded yields a small delta even though RAG is doing its job, so
BOTH raw rates are reported separately and the delta is read together with them,
never alone. If grounding does NOT help, that is reported as-is -- the value then
is the honesty (the system does not invent a label), not an accuracy gain.

This is a SIBLING harness to :mod:`ml.eval.agent_system_eval` (it reuses its
backend-driving helpers and the :class:`ml.eval.agent_metrics.HallucinationJudge`
contract via a dedicated hedge judge) and is fully INJECTABLE: the reasoner
backend and the judge are passed in, so tests drive it with a scripted fake
reasoner and a deterministic fake judge -- zero network, no Vertex AI, no vLLM, no
pgvector. The REAL number (AC8) is produced by injecting a live Gemini/Qwen
backend and a real LLM-as-judge; when those credentials are absent the harness is
LISTED + TESTED and the real delta is recorded as a documented blocker.

Project conventions: identifiers and docstrings in English (Google style); the
curated prompts/report in Spanish; ``structlog`` (never ``print``); full type
hints; no emojis.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
from google.genai import types

from ml.eval.class_remap import get_label_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ml.agent.backends import LLMBackend

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_HEDGE_PATH",
    "HedgeABResult",
    "HedgeCase",
    "HedgeJudge",
    "KeywordHedgeJudge",
    "LLMHedgeJudge",
    "build_hedge_judge",
    "load_hedge_cases",
    "run_hedge_ab",
]

#: Default dataset location (small curated JSONL of REAL out-of-vocabulary PASTIS
#: parcels; committed, not DVC).
DEFAULT_HEDGE_PATH: Path = Path("data/agent_eval/hedge_oov_cases.jsonl")

#: Per-item timeout for a single reasoner call (mirrors agent_system_eval).
_ITEM_TIMEOUT_S: float = 200.0

#: Label-space the hedge is measured under (the v2 champion's twelve resolved
#: crops); its six dropped classes are the out-of-vocabulary universe.
_LABEL_SPACE_NAME: str = "france-12"


# ---------------------------------------------------------------------------
# Judge contract
# ---------------------------------------------------------------------------
@runtime_checkable
class HedgeJudge(Protocol):
    """Injectable judge scoring one out-of-vocabulary hedge analysis (AC8).

    A judge inspects a single analysis (the reasoner's prose) against the case's
    true out-of-vocabulary crop and the resolved vocabulary, and returns two
    scores in ``[0, 1]``: ``hedge_quality`` (acknowledges the limit AND conjectures
    toward the true crop) and ``forced_label`` (1.0 when it confidently asserts a
    resolved crop as the answer). Implemented over an LLM-as-judge in production
    and mocked deterministically in tests.
    """

    def score(self, sample: dict[str, Any]) -> dict[str, float]:
        """Return ``{"hedge_quality": float, "forced_label": float}`` for a sample."""
        ...


class KeywordHedgeJudge:
    """Deterministic, network-free :class:`HedgeJudge` for offline runs / CI.

    A transparent rubric (NOT an LLM): it credits an analysis for naming the TRUE
    out-of-vocabulary crop and for hedge language ("fuera del vocabulario", "no
    puedo confirmar", "posible", "vecinas"), and penalises it for confidently
    asserting a RESOLVED crop. It is the honest default when no LLM judge is wired
    -- documented as a keyword proxy, never presented as a calibrated judge. The
    real AC8 number uses an LLM-as-judge (see :func:`run_hedge_ab`).

    Attributes:
        hedge_markers: Spanish phrases that signal an honest hedge.
    """

    #: Spanish phrases that mark an honest acknowledgement of the vocabulary limit.
    hedge_markers: tuple[str, ...] = (
        "fuera del vocabulario",
        "fuera de vocabulario",
        "no puedo confirmar",
        "no esta en el conjunto",
        "no se resuelve",
        "no resuelta",
        "posible",
        "podria ser",
        "sugiere",
        "parcelas vecinas",
        "evidencia vecina",
        "fenolog",
    )

    def score(self, sample: dict[str, Any]) -> dict[str, float]:
        """Score one analysis with the keyword rubric.

        Args:
            sample: ``{analysis, true_crop, resolved_crops, unresolved_candidate}``.

        Returns:
            ``{"hedge_quality", "forced_label"}`` in ``[0, 1]``.
        """
        analysis = str(sample.get("analysis", "")).lower()
        true_crop = str(sample.get("true_crop", "")).lower()
        resolved = [str(c).lower() for c in sample.get("resolved_crops", [])]

        names_true = bool(true_crop) and true_crop in analysis
        hedges = any(marker in analysis for marker in self.hedge_markers)
        # A forced label = a resolved crop named WITHOUT any hedge language.
        asserts_resolved = any(crop and crop in analysis for crop in resolved)
        forced = 1.0 if (asserts_resolved and not hedges) else 0.0

        # Hedge quality: half-credit for honest hedging, half for steering toward
        # the true out-of-vocabulary crop; a forced resolved label zeroes it.
        quality = 0.0
        if forced == 0.0:
            quality += 0.5 if hedges else 0.0
            quality += 0.5 if names_true else 0.0
        return {"hedge_quality": float(quality), "forced_label": float(forced)}


class LLMHedgeJudge:
    """LLM-as-judge :class:`HedgeJudge` driven by an injected backend (real AC8).

    Wraps an :class:`~ml.agent.backends.LLMBackend` (e.g. a Gemini backend) as the
    evaluation LLM: it sends the analysis plus the gold (the true out-of-vocabulary
    crop and the resolved vocabulary) and asks for a STRICT JSON verdict
    ``{"hedge_quality": float, "forced_label": float}``. This is the judge that
    produces the REAL AC8 number; it is constructed from the project settings via
    :func:`build_hedge_judge` so credentials stay out of the harness, and it
    degrades to ``{0.0, 0.0}`` (logged) on any parse/LLM failure rather than
    crashing the run.

    The backend is driven SYNCHRONOUSLY here (the judge contract is sync) by
    running its async stream to completion on a fresh event loop; the harness
    itself is async, so the judge call happens between awaited reasoner turns. The
    backend is injectable so tests substitute a scripted fake.

    Attributes:
        backend: The injected evaluation-LLM backend.
    """

    def __init__(self, backend: LLMBackend) -> None:
        """Initialise the judge over an evaluation-LLM backend.

        Args:
            backend: The :class:`~ml.agent.backends.LLMBackend` used as judge.
        """
        self.backend = backend

    def score(self, sample: dict[str, Any]) -> dict[str, float]:
        """Score one analysis by asking the evaluation LLM for a JSON verdict.

        The :class:`HedgeJudge` contract is SYNCHRONOUS but the backend stream is
        async, and this is called from INSIDE the running event loop of
        :func:`run_hedge_ab`. So the async drive is executed on a dedicated thread
        with its own loop (:func:`_run_coro_blocking`), which works whether or not
        an outer loop is already running (a plain ``asyncio.run`` would raise
        "cannot be called from a running event loop").

        Args:
            sample: ``{analysis, true_crop, resolved_crops, unresolved_candidate}``.

        Returns:
            ``{"hedge_quality", "forced_label"}`` in ``[0, 1]``; ``{0.0, 0.0}`` on
            any failure (logged ``hedge_llm_judge_failed``), never raising.
        """
        prompt = self._judge_prompt(sample)
        try:
            text = _run_coro_blocking(_drive_for_text(self.backend, prompt))
            verdict = _parse_judge_json(text)
        except Exception as exc:  # noqa: BLE001 - judge errors must not crash eval
            logger.warning("hedge_llm_judge_failed", error=str(exc))
            return {"hedge_quality": 0.0, "forced_label": 0.0}
        return verdict

    @staticmethod
    def _judge_prompt(sample: dict[str, Any]) -> str:
        """Compose the Spanish judging prompt asking for a strict JSON verdict.

        Args:
            sample: The analysis + gold fields.

        Returns:
            The composed prompt string.
        """
        resolved = ", ".join(str(c) for c in sample.get("resolved_crops", []))
        return "\n".join(
            [
                "Eres un evaluador estricto. Una parcela tiene un cultivo FUERA del "
                "vocabulario de doce clases del clasificador. Evalua el siguiente "
                "analisis del copiloto en dos ejes (0.0 a 1.0):",
                "- hedge_quality: 1.0 si reconoce que el cultivo esta fuera del "
                "vocabulario Y conjetura razonadamente hacia el cultivo verdadero "
                "apoyandose en evidencia; 0.0 si fuerza una de las doce clases.",
                "- forced_label: 1.0 si afirma con confianza una de las doce clases "
                "resueltas como respuesta; 0.0 si no.",
                "",
                f"Cultivo verdadero (fuera de vocabulario): {sample.get('true_crop', '')}",
                f"Clases resueltas del clasificador: {resolved}",
                f"Analisis a evaluar: {sample.get('analysis', '')}",
                "",
                'Responde UNICAMENTE con el JSON {"hedge_quality": <float>, '
                '"forced_label": <float>} sin texto adicional.',
            ]
        )


def build_hedge_judge(settings: Any = None, model: str = "gemini-2.5-pro") -> HedgeJudge:
    """Build the real LLM-as-judge from settings, or degrade to the keyword proxy.

    Constructs an :class:`LLMHedgeJudge` over a Gemini backend wired with the
    project's credentials (via :func:`ml.agent.backends.make_backend`). When the
    settings cannot be loaded or carry no usable Gemini credentials, it returns the
    deterministic :class:`KeywordHedgeJudge` (logged) so the harness still produces
    a number -- but the REAL AC8 number requires the LLM judge.

    Args:
        settings: Project ``Settings`` (``None`` loads them lazily).
        model: The judge model id (default the Gemini reasoner).

    Returns:
        A ready :class:`HedgeJudge` (LLM-backed when possible, keyword proxy else).
    """
    if settings is None:
        try:
            from backend.app.core.config import get_settings

            settings = get_settings()
        except Exception as exc:  # noqa: BLE001 - settings optional outside the app
            logger.warning("hedge_judge_no_settings", error=str(exc))
            return KeywordHedgeJudge()
    api_key = getattr(settings, "gemini_api_key", "") or getattr(settings, "google_api_key", "")
    use_vertexai = str(getattr(settings, "google_genai_use_vertexai", "")).lower() in (
        "1",
        "true",
        "yes",
    )
    if not api_key and not use_vertexai:
        logger.warning("hedge_judge_no_credentials", reason="no Gemini key/Vertex configured")
        return KeywordHedgeJudge()
    from ml.agent.backends import make_backend

    return LLMHedgeJudge(make_backend(model, settings))


def _parse_judge_json(text: str) -> dict[str, float]:
    """Parse the ``{"hedge_quality", "forced_label"}`` JSON from a judge answer.

    Extracts the first balanced ``{...}`` block (the LLM may wrap it in prose),
    parses it and clamps both scores to ``[0, 1]``. Missing keys default to ``0.0``.

    Args:
        text: The raw judge answer.

    Returns:
        ``{"hedge_quality", "forced_label"}`` in ``[0, 1]``.

    Raises:
        ValueError: when no JSON object is parseable (the caller logs + degrades).
    """
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in judge answer")
    obj = json.loads(match.group(0))

    def _clamp(value: Any) -> float:
        return max(0.0, min(1.0, float(value)))

    return {
        "hedge_quality": _clamp(obj.get("hedge_quality", 0.0)),
        "forced_label": _clamp(obj.get("forced_label", 0.0)),
    }


# ---------------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HedgeCase:
    """One out-of-vocabulary hedge A/B case (AC8).

    Attributes:
        id: Stable case id (e.g. ``"hedge-001"``).
        parcel_id: The persisted parcel referenced (trace/logging).
        true_crop: The REAL out-of-vocabulary crop (a ``SEMANTIC18_CLASS_NAMES``
            value that is in the label-space's DROPPED set), e.g. ``"Sorghum"``.
        unresolved_candidate: The out-of-vocabulary crop the classifier's RAW
            argmax leaned toward (mirrors
            :attr:`ml.agent.schemas.ClassificationResult.unresolved_candidate`);
            normally equals ``true_crop`` for a clean out-of-vocabulary case.
        restricted_probabilities: ``{resolved_crop: prob}`` the perceiver reports
            (the renormalized posterior over the twelve resolved crops). Its argmax
            is the FORCED label the ungrounded reasoner is tempted to report.
        grounding_text: The citation-tagged neighbouring-parcel evidence the REAL
            ``retrieve_context`` emits (``_build_grounding_text`` shape). Injected
            only on the A (grounded) run; it carries the signal that points at the
            TRUE out-of-vocabulary crop.
        user_query: The Spanish user turn.
    """

    id: str
    parcel_id: int
    true_crop: str
    unresolved_candidate: str
    restricted_probabilities: dict[str, float]
    grounding_text: str
    user_query: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of records (blank lines skipped).

    Args:
        path: The JSONL dataset path.

    Returns:
        The parsed records, order preserved.
    """
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_hedge_cases(path: Path = DEFAULT_HEDGE_PATH) -> list[HedgeCase]:
    """Load and validate the out-of-vocabulary hedge cases.

    Every case's ``true_crop`` MUST be in the dropped set of the ``france-12``
    label-space (otherwise it is not an out-of-vocabulary case and the A/B is
    meaningless); a misfiled case fails fast.

    Args:
        path: Path to ``hedge_oov_cases.jsonl``.

    Returns:
        The parsed :class:`HedgeCase` list.

    Raises:
        ValueError: if a case's ``true_crop`` is not in the label-space's dropped
            (out-of-vocabulary) set.
    """
    space = get_label_space(_LABEL_SPACE_NAME)
    dropped = set(space.dropped_class_names.values())
    cases: list[HedgeCase] = []
    for record in _read_jsonl(path):
        true_crop = str(record["true_crop"])
        if true_crop not in dropped:
            raise ValueError(
                f"hedge case {record.get('id')!r} true_crop {true_crop!r} is not "
                f"out-of-vocabulary for {_LABEL_SPACE_NAME}; dropped set: {sorted(dropped)}"
            )
        cases.append(
            HedgeCase(
                id=str(record["id"]),
                parcel_id=int(record["parcel_id"]),
                true_crop=true_crop,
                unresolved_candidate=str(record.get("unresolved_candidate") or true_crop),
                restricted_probabilities=dict(record["restricted_probabilities"]),
                grounding_text=str(record["grounding_text"]),
                user_query=str(record["user_query"]),
            )
        )
    logger.info("hedge_cases_loaded", path=str(path), n=len(cases))
    return cases


# ---------------------------------------------------------------------------
# Reasoner driving
# ---------------------------------------------------------------------------
def _user_contents(text: str) -> list[types.Content]:
    """Build a single-user-turn ``contents`` list for a backend call.

    Args:
        text: The user prompt text.

    Returns:
        A one-element ``role="user"`` contents list.
    """
    return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]


def _run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion, working even inside a running event loop.

    The synchronous :class:`HedgeJudge.score` is invoked from within the async
    :func:`run_hedge_ab` loop, so a plain ``asyncio.run`` would raise. This runs
    the coroutine on a dedicated worker thread with its own fresh event loop when
    an outer loop is already running, and falls back to ``asyncio.run`` when called
    from a purely synchronous context.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: safe to drive directly.
        return asyncio.run(coro)

    # A loop is running on this thread; execute the coroutine on a separate thread
    # with its own loop so it does not collide with the outer loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _drive_for_text(backend: LLMBackend, prompt: str) -> str:
    """Drive a backend for one text answer (no tools).

    Args:
        backend: The injected reasoner backend.
        prompt: The composed prompt.

    Returns:
        The concatenated answer text (stripped).
    """
    buffer: list[str] = []
    async for chunk in backend.generate_stream(
        contents=_user_contents(prompt),
        tools=[],
        system_instruction=_HEDGE_SYSTEM_INSTRUCTION,
    ):
        text = getattr(chunk, "text", None)
        if text:
            buffer.append(text)
    return "".join(buffer).strip()


#: System instruction that states the honest-hedge contract (mirrors the rule in
#: ``ml/agent/prompts.py``): the reasoner must NOT force an in-vocabulary label for
#: an out-of-vocabulary crop, and must lean on the retrieved evidence + phenology.
_HEDGE_SYSTEM_INSTRUCTION: str = (
    "Eres el copiloto agricola satelital. El clasificador resuelve solo doce "
    "cultivos. Cuando una parcela queda FUERA de ese vocabulario (campo "
    "'unresolved_candidate' no nulo), NO fuerces una de las doce clases como "
    "respuesta: reconoce honestamente que el cultivo esta fuera del vocabulario "
    "calibrado y, si tienes contexto recuperado de parcelas vecinas o fenologia, "
    "usalo para conjeturar el cultivo mas probable citando esa evidencia. Si no "
    "tienes evidencia, dilo sin inventar."
)


def _build_prompt(case: HedgeCase, *, grounded: bool) -> str:
    """Compose the reasoner prompt for one A/B side.

    Both sides see the SAME classifier output (the resolved-class posterior and the
    ``unresolved_candidate``). Only the A (grounded) side additionally receives the
    neighbouring-parcel grounding block, so any improvement is attributable to the
    grounding, not to a different prompt.

    Args:
        case: The hedge case.
        grounded: ``True`` for the A run (inject ``grounding_text``), ``False`` for
            the B (ungrounded) run.

    Returns:
        The composed Spanish prompt string.
    """
    classifier_block = json.dumps(
        {
            "unresolved_candidate": case.unresolved_candidate,
            "class_probabilities_resolved": case.restricted_probabilities,
        },
        ensure_ascii=False,
    )
    lines = [
        "Resultado del clasificador (posterior renormalizado SOLO sobre las clases "
        "resueltas; 'unresolved_candidate' indica que el argmax crudo cayo fuera "
        "del vocabulario):",
        classifier_block,
        "",
    ]
    if grounded and case.grounding_text:
        lines.extend(["Contexto recuperado (retrieve_context):", case.grounding_text, ""])
    else:
        lines.extend(["Contexto recuperado: (no disponible)", ""])
    lines.append(f"Pregunta del usuario: {case.user_query}")
    lines.append(
        "Da un analisis honesto del cultivo de la parcela siguiendo la regla del "
        "sistema (no fuerces una de las doce clases si esta fuera de vocabulario)."
    )
    return "\n".join(lines)


async def _run_side(
    backend: LLMBackend,
    cases: Sequence[HedgeCase],
    *,
    grounded: bool,
    judge: HedgeJudge,
    resolved_crops: list[str],
) -> tuple[float, float]:
    """Run one A/B side and return its ``(hedge_quality, forced_label_rate)`` means.

    Args:
        backend: The injected reasoner backend.
        cases: The hedge cases.
        grounded: ``True`` for the grounded (A) run, ``False`` for ungrounded (B).
        judge: The injectable hedge judge.
        resolved_crops: The twelve resolved crop names (passed to the judge so it
            can detect a forced in-vocabulary label).

    Returns:
        ``(mean_hedge_quality, mean_forced_label_rate)`` over the cases (NaN when
        no case scored).
    """
    quality_scores: list[float] = []
    forced_scores: list[float] = []
    for case in cases:
        prompt = _build_prompt(case, grounded=grounded)
        if hasattr(backend, "reset"):
            backend.reset()
        try:
            analysis = await asyncio.wait_for(
                _drive_for_text(backend, prompt), timeout=_ITEM_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001 - one case must not crash the run
            logger.warning("hedge_case_failed", case=case.id, grounded=grounded, error=str(exc))
            analysis = ""
        sample = {
            "analysis": analysis,
            "true_crop": case.true_crop,
            "resolved_crops": resolved_crops,
            "unresolved_candidate": case.unresolved_candidate,
        }
        try:
            # Score on a worker thread: an LLM-as-judge drives its OWN async
            # backend, and calling it directly here would nest that drive inside
            # this running loop. Offloading to a thread gives the judge a clean
            # (loop-free) thread so ``asyncio.run`` works without the nested-loop
            # fragility; a pure keyword judge is unaffected.
            scored = await asyncio.to_thread(judge.score, sample)
        except Exception as exc:  # noqa: BLE001 - judge errors must not crash eval
            logger.warning("hedge_judge_failed", case=case.id, error=str(exc))
            continue
        quality_scores.append(float(scored.get("hedge_quality", 0.0)))
        forced_scores.append(float(scored.get("forced_label", 0.0)))
    quality = sum(quality_scores) / len(quality_scores) if quality_scores else math.nan
    forced = sum(forced_scores) / len(forced_scores) if forced_scores else math.nan
    return quality, forced


# ---------------------------------------------------------------------------
# A/B result + entry point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HedgeABResult:
    """Outcome of the out-of-vocabulary hedge A/B for one reasoner (AC8).

    Attributes:
        hedge_quality_ungrounded: Mean hedge quality on the B (ungrounded) run.
        hedge_quality_grounded: Mean hedge quality on the A (grounded) run.
        hedge_quality_delta: ``grounded - ungrounded`` (the headline improvement
            from RAG, read TOGETHER with the two raw rates, never alone).
        forced_label_rate_ungrounded: Forced-in-vocabulary-label rate, B run.
        forced_label_rate_grounded: Forced-in-vocabulary-label rate, A run.
        n: Number of cases evaluated.
    """

    hedge_quality_ungrounded: float
    hedge_quality_grounded: float
    hedge_quality_delta: float
    forced_label_rate_ungrounded: float
    forced_label_rate_grounded: float
    n: int

    def as_metrics(self) -> dict[str, float | int]:
        """Return the result as a flat metric mapping (report/aggregation).

        Returns:
            A ``{metric: value}`` dict mirroring the dataclass fields.
        """
        return {
            "hedge_quality_ungrounded": self.hedge_quality_ungrounded,
            "hedge_quality_grounded": self.hedge_quality_grounded,
            "hedge_quality_delta": self.hedge_quality_delta,
            "forced_label_rate_ungrounded": self.forced_label_rate_ungrounded,
            "forced_label_rate_grounded": self.forced_label_rate_grounded,
            "n": self.n,
        }


async def run_hedge_ab(
    cases: Sequence[HedgeCase],
    *,
    backend: LLMBackend,
    judge: HedgeJudge | None = None,
    label_space: str = _LABEL_SPACE_NAME,
) -> HedgeABResult:
    """Run the out-of-vocabulary hedge A/B for one reasoner backend (AC8).

    Runs every case twice (B = ungrounded, A = grounded with the injected
    ``grounding_text``), scores both with the injectable judge and reports the
    delta. Fully injectable: the ``backend`` and ``judge`` are passed in, so tests
    drive a scripted fake reasoner + a deterministic judge with zero network. For
    the REAL AC8 number, inject a live Gemini/Qwen backend and an LLM-as-judge.

    Honest framing: the delta is the improvement RELATIVE TO the ungrounded
    reasoner. A reasoner that already hedges well ungrounded yields a small delta
    even though grounding helps, so the two raw rates are reported separately. If
    grounding does not help, that is reported as-is (the value is then the honesty,
    not an accuracy gain).

    Args:
        cases: The out-of-vocabulary hedge cases.
        backend: The injected reasoner backend (scripted fake in CI, live for the
            real number).
        judge: The injectable hedge judge. ``None`` falls back to the deterministic
            :class:`KeywordHedgeJudge` (a documented keyword proxy, not an
            LLM-as-judge) so the harness always produces a number; the real AC8
            number passes an LLM-as-judge.
        label_space: The label-space whose RESOLVED crops define the forced-label
            universe (default ``france-12``).

    Returns:
        The :class:`HedgeABResult` for this reasoner.
    """
    active_judge: HedgeJudge = judge or KeywordHedgeJudge()
    space = get_label_space(label_space)
    resolved_crops = list(space.class_names.values())

    quality_b, forced_b = await _run_side(
        backend, cases, grounded=False, judge=active_judge, resolved_crops=resolved_crops
    )
    quality_a, forced_a = await _run_side(
        backend, cases, grounded=True, judge=active_judge, resolved_crops=resolved_crops
    )
    delta = (
        quality_a - quality_b if not (math.isnan(quality_a) or math.isnan(quality_b)) else math.nan
    )
    result = HedgeABResult(
        hedge_quality_ungrounded=quality_b,
        hedge_quality_grounded=quality_a,
        hedge_quality_delta=delta,
        forced_label_rate_ungrounded=forced_b,
        forced_label_rate_grounded=forced_a,
        n=len(cases),
    )
    logger.info(
        "hedge_ab_done",
        hedge_quality_ungrounded=quality_b,
        hedge_quality_grounded=quality_a,
        delta=delta,
        forced_ungrounded=forced_b,
        forced_grounded=forced_a,
        n=len(cases),
    )
    return result
