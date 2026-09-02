"""Tests for the out-of-vocabulary hedge A/B harness (US-081 AC8).

Fully network-free: the reasoner backend and the judge are injected, so a scripted
fake reasoner returns different prose on the grounded vs ungrounded run and a
deterministic judge scores it. No Vertex AI / vLLM / pgvector is touched. The real
AC8 number is produced by injecting a live backend + an LLM-as-judge (documented
as a blocker when creds are absent).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ml.agent.eval.hedge_ab_eval import (
    DEFAULT_HEDGE_PATH,
    HedgeABResult,
    HedgeCase,
    KeywordHedgeJudge,
    LLMHedgeJudge,
    build_hedge_judge,
    load_hedge_cases,
    run_hedge_ab,
)


@dataclass
class _Chunk:
    """Duck-typed ``BackendChunk`` stand-in (text only for this text-eval)."""

    text: str | None = None
    function_call: None = None


@dataclass
class _GroundingAwareBackend:
    """Fake reasoner that answers differently with vs without grounding.

    Detects the grounding marker in the prompt: when present (the A run) it
    produces an honest hedge that NAMES the true crop from the evidence; otherwise
    (the B run) it forces the resolved-class argmax with no hedge. The true crop is
    read from the prompt's injected grounding text so the same instance serves any
    case.

    Attributes:
        model: Backend id (for parity with real backends).
        prompts_seen: Every prompt passed to ``generate_stream`` (for assertions).
    """

    model: str = "fake-reasoner"
    prompts_seen: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """No-op reset (stateless aside from the prompt log)."""

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield a grounded hedge or an ungrounded forced label by prompt content."""
        del tools, system_instruction
        prompt = contents[0].parts[0].text
        self.prompts_seen.append(prompt)
        grounded = "retrieve_context" in prompt
        if grounded:
            # Extract the true crop the grounding evidence labels.
            crop = _crop_from_grounding(prompt)
            yield _Chunk(
                text=(
                    f"El cultivo esta fuera del vocabulario calibrado; las parcelas "
                    f"vecinas y la fenologia sugieren que posiblemente sea {crop}."
                )
            )
        else:
            yield _Chunk(text="Es Corn con alta confianza.")


def _crop_from_grounding(prompt: str) -> str:
    """Pull the field-labelled crop out of the grounding block (test helper)."""
    marker = "etiqueta de campo "
    idx = prompt.find(marker)
    if idx < 0:
        return "desconocido"
    tail = prompt[idx + len(marker) :]
    return tail.split(".")[0].strip()


def _case(true_crop: str, resolved_argmax: str = "Corn") -> HedgeCase:
    """Build a minimal hedge case for a given out-of-vocabulary crop."""
    return HedgeCase(
        id=f"hedge-{true_crop[:4]}",
        parcel_id=1,
        true_crop=true_crop,
        unresolved_candidate=true_crop,
        restricted_probabilities={resolved_argmax: 0.5, "Sunflower": 0.3, "Soybeans": 0.2},
        grounding_text=(
            f"Contexto recuperado de parcelas vecinas (corpus PASTIS-R):\n"
            f"[pastis:31_4] Parcela vecina: etiqueta de campo {true_crop}."
        ),
        user_query="Que cultivo hay en esta parcela?",
    )


def test_grounding_improves_hedge_quality_and_cuts_forced_labels() -> None:
    """Grounded run hedges + names the true crop; ungrounded forces a label."""
    cases = [_case("Sorghum"), _case("Potatoes")]
    backend = _GroundingAwareBackend()

    result = asyncio.run(run_hedge_ab(cases, backend=backend, judge=KeywordHedgeJudge()))

    assert isinstance(result, HedgeABResult)
    assert result.n == 2
    # Grounded analysis hedges and names the true crop -> high quality.
    assert result.hedge_quality_grounded == pytest.approx(1.0)
    # Ungrounded forces "Corn" (a resolved crop) with no hedge -> zero quality.
    assert result.hedge_quality_ungrounded == pytest.approx(0.0)
    assert result.hedge_quality_delta == pytest.approx(1.0)
    # Forced-label rate is high ungrounded, zero grounded.
    assert result.forced_label_rate_ungrounded == pytest.approx(1.0)
    assert result.forced_label_rate_grounded == pytest.approx(0.0)


def test_both_runs_see_same_classifier_output_only_grounding_differs() -> None:
    """The A and B prompts differ ONLY by the grounding block."""
    backend = _GroundingAwareBackend()
    asyncio.run(run_hedge_ab([_case("Mixed cereal")], backend=backend))

    # Two prompts per case (B then A): ungrounded first, grounded second.
    ungrounded, grounded = backend.prompts_seen
    assert "retrieve_context" not in ungrounded
    assert "retrieve_context" in grounded
    # Both carry the classifier output (the unresolved_candidate cue).
    assert "unresolved_candidate" in ungrounded
    assert "unresolved_candidate" in grounded


def test_injected_judge_is_used() -> None:
    """A custom judge fully drives the scores (injectable contract)."""

    class _ConstJudge:
        def score(self, sample: dict[str, Any]) -> dict[str, float]:
            return {"hedge_quality": 0.42, "forced_label": 0.1}

    result = asyncio.run(
        run_hedge_ab([_case("Sorghum")], backend=_GroundingAwareBackend(), judge=_ConstJudge())
    )
    assert result.hedge_quality_grounded == pytest.approx(0.42)
    assert result.hedge_quality_ungrounded == pytest.approx(0.42)
    assert result.hedge_quality_delta == pytest.approx(0.0)
    assert result.forced_label_rate_grounded == pytest.approx(0.1)


def test_keyword_judge_zeroes_quality_on_forced_label() -> None:
    """The keyword judge gives zero quality when a resolved crop is forced."""
    judge = KeywordHedgeJudge()
    scored = judge.score(
        {
            "analysis": "Es Corn con alta confianza.",
            "true_crop": "Sorghum",
            "resolved_crops": ["Corn", "Sunflower"],
            "unresolved_candidate": "Sorghum",
        }
    )
    assert scored["forced_label"] == pytest.approx(1.0)
    assert scored["hedge_quality"] == pytest.approx(0.0)


def test_keyword_judge_rewards_hedge_and_true_crop() -> None:
    """Hedge language + naming the true crop earns full quality, no forced label."""
    judge = KeywordHedgeJudge()
    scored = judge.score(
        {
            "analysis": (
                "El cultivo esta fuera del vocabulario; las parcelas vecinas sugieren Sorghum."
            ),
            "true_crop": "Sorghum",
            "resolved_crops": ["Corn", "Sunflower"],
            "unresolved_candidate": "Sorghum",
        }
    )
    assert scored["forced_label"] == pytest.approx(0.0)
    assert scored["hedge_quality"] == pytest.approx(1.0)


def test_empty_cases_yield_nan_metrics() -> None:
    """No cases -> NaN metrics (honest 'nothing measured'), no crash."""
    result = asyncio.run(run_hedge_ab([], backend=_GroundingAwareBackend()))
    assert result.n == 0
    assert math.isnan(result.hedge_quality_grounded)
    assert math.isnan(result.hedge_quality_delta)


def test_judge_failure_does_not_crash_run() -> None:
    """A judge that raises on a case is skipped, the run still completes."""

    class _FlakyJudge:
        def __init__(self) -> None:
            self.n = 0

        def score(self, sample: dict[str, Any]) -> dict[str, float]:
            self.n += 1
            raise RuntimeError("judge boom")

    result = asyncio.run(
        run_hedge_ab([_case("Sorghum")], backend=_GroundingAwareBackend(), judge=_FlakyJudge())
    )
    # All cases skipped -> NaN, but no exception escaped.
    assert math.isnan(result.hedge_quality_grounded)


def test_load_real_dataset_validates_out_of_vocabulary() -> None:
    """The committed dataset loads and every true_crop is out-of-vocabulary."""
    cases = load_hedge_cases(DEFAULT_HEDGE_PATH)
    assert len(cases) >= 6
    dropped = {
        "Winter triticale",
        "Fruits, vegetables, flowers",
        "Potatoes",
        "Leguminous fodder",
        "Mixed cereal",
        "Sorghum",
    }
    for case in cases:
        assert case.true_crop in dropped
        assert "retrieve_context" not in case.grounding_text or case.grounding_text
        # The forced-label trap is over RESOLVED crops only.
        assert case.restricted_probabilities


def test_llm_hedge_judge_parses_json_verdict() -> None:
    """The LLM judge drives an injected backend and parses its JSON verdict."""

    @dataclass
    class _JudgeBackend:
        model: str = "judge"

        def reset(self) -> None: ...

        async def generate_stream(
            self, *, contents: list, tools: list, system_instruction: str
        ) -> AsyncIterator[_Chunk]:
            del contents, tools, system_instruction
            yield _Chunk(text='Mi veredicto: {"hedge_quality": 0.8, "forced_label": 0.0}')

    judge = LLMHedgeJudge(_JudgeBackend())
    scored = judge.score(
        {
            "analysis": "fuera del vocabulario, posiblemente Sorghum",
            "true_crop": "Sorghum",
            "resolved_crops": ["Corn"],
            "unresolved_candidate": "Sorghum",
        }
    )
    assert scored["hedge_quality"] == pytest.approx(0.8)
    assert scored["forced_label"] == pytest.approx(0.0)


def test_llm_hedge_judge_clamps_and_degrades_on_bad_output() -> None:
    """Out-of-range scores clamp; unparseable output degrades to zeros (no crash)."""

    @dataclass
    class _BadBackend:
        text: str
        model: str = "judge"

        def reset(self) -> None: ...

        async def generate_stream(
            self, *, contents: list, tools: list, system_instruction: str
        ) -> AsyncIterator[_Chunk]:
            del contents, tools, system_instruction
            yield _Chunk(text=self.text)

    sample = {
        "analysis": "x",
        "true_crop": "Sorghum",
        "resolved_crops": [],
        "unresolved_candidate": "Sorghum",
    }
    clamped = LLMHedgeJudge(_BadBackend('{"hedge_quality": 1.7, "forced_label": -0.2}')).score(
        sample
    )
    assert clamped["hedge_quality"] == pytest.approx(1.0)
    assert clamped["forced_label"] == pytest.approx(0.0)

    degraded = LLMHedgeJudge(_BadBackend("no json here")).score(sample)
    assert degraded == {"hedge_quality": 0.0, "forced_label": 0.0}


def test_build_hedge_judge_degrades_to_keyword_without_credentials() -> None:
    """With no Gemini credentials the factory returns the keyword proxy, not a crash."""

    class _NoCredsSettings:
        gemini_api_key = ""
        google_api_key = ""
        google_genai_use_vertexai = ""

    judge = build_hedge_judge(settings=_NoCredsSettings())
    assert isinstance(judge, KeywordHedgeJudge)


def test_load_rejects_in_vocabulary_case(tmp_path) -> None:
    """A case whose true_crop is resolved (in-vocabulary) fails fast on load."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "x", "parcel_id": 1, "true_crop": "Corn", "unresolved_candidate": "Corn", '
        '"restricted_probabilities": {"Corn": 1.0}, "grounding_text": "g", '
        '"user_query": "q"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        load_hedge_cases(bad)
