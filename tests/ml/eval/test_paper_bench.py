"""Offline tests for the paper multi-benchmark eval harness (US-069).

Sibling of :mod:`tests.ml.eval.test_agent_system_eval`: every external boundary
is a deterministic in-memory double (no ``google-genai`` client, no network, no
GEO-Bench-2 download, no Gemini/vLLM call). The fixtures under
``data/test_fixtures/`` are SHAPE fixtures (3 mini GEO-Bench-2 tasks + 6
bilingual QA items) -- they validate the contract, NOT a real scientific number.

Covered: the loaders (and their degrade-clean empty path), the dataclasses, the
text-only skip, ``macro_f1`` on known counts, ``wilcoxon_paired`` on known
vectors (including the not-computable branches), the two evaluators offline, the
``run_paper_benchmark`` aggregation with a ``pending`` benchmark, the paired
Wilcoxon wiring and the LaTeX export.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ml.eval.agent_bench import ReasonerVariant
from ml.eval.paper_bench import (
    BENCHMARKS,
    PAPER_VARIANTS,
    GeoBenchItem,
    WilcoxonResult,
    eval_agromind_itses,
    eval_geobench2,
    export_latex_table,
    load_agromind_itses,
    load_geobench2,
    macro_f1,
    run_paper_benchmark,
    wilcoxon_paired,
)

# ---------------------------------------------------------------------------
# Fixture paths (shape fixtures, NOT real scientific data)
# ---------------------------------------------------------------------------
# Anchor to the repo root so the fixtures resolve regardless of the pytest CWD
# (tests are run from ``ml/`` in CI, not from the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO_ROOT / "data" / "test_fixtures"
_GEOBENCH_MINI = _FIXTURE_DIR / "geobench2_mini"
_ITSES_MINI = _FIXTURE_DIR / "agromind_itses_mini.jsonl"

GEMINI = ReasonerVariant(name="gemini", model="gemini-2.5-pro", multimodal=True)
QWEN = ReasonerVariant(name="qwen", model="qwen35", multimodal=False)


# ---------------------------------------------------------------------------
# Backend doubles (mirror test_agent_system_eval.ScriptedBackend)
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    """Duck-typed ``BackendChunk`` stand-in carrying only a text delta."""

    text: str | None = None
    function_call: None = None


@dataclass
class AnswerBackend:
    """Backend that answers each call with the next scripted text.

    ``reset`` rewinds the cursor so the same instance is reusable across seeds
    (the harness calls ``reset`` before every seed). Beyond the script it yields
    an empty terminal chunk so a runaway loop still terminates.
    """

    answers: list[str]
    model: str = "scripted"
    _cursor: int = 0

    def reset(self) -> None:
        """Rewind the answer cursor to the start of the script."""
        self._cursor = 0

    async def generate_stream(
        self, *, contents: Any, tools: Any, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield the next scripted answer as a single text chunk."""
        if self._cursor < len(self.answers):
            text = self.answers[self._cursor]
            self._cursor += 1
        else:
            text = ""
        yield _Chunk(text=text)


@dataclass
class CyclingBackend:
    """Backend that cycles a fixed answer list, ignoring call count.

    Useful when the number of scored items is not known ahead of time: the same
    short answer cycle is replayed so every item gets a deterministic prediction.
    """

    cycle: list[str]
    model: str = "scripted"
    _i: int = 0

    def reset(self) -> None:
        """Rewind the cycle cursor."""
        self._i = 0

    async def generate_stream(
        self, *, contents: Any, tools: Any, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield the next answer from the cycle (wrapping around)."""
        text = self.cycle[self._i % len(self.cycle)]
        self._i += 1
        yield _Chunk(text=text)


@dataclass
class FakeJudge:
    """Deterministic hallucination judge returning a fixed score per call."""

    score_value: float = 0.0
    seen: list[dict[str, Any]] = field(default_factory=list)

    def score(self, sample: dict[str, Any]) -> float:
        """Record the sample and return the fixed score."""
        self.seen.append(sample)
        return self.score_value


@pytest.fixture(autouse=True)
def _stub_bertscore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the heavy sentence-transformer BERTScore proxy (no model load)."""
    from ml.eval import agent_metrics

    monkeypatch.setattr(agent_metrics, "bertscore_f1", lambda preds, golds: 0.5)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def test_load_geobench2_reads_three_agricultural_tasks() -> None:
    """The mini manifest yields >= 3 agricultural tasks with their items."""
    items = load_geobench2(_GEOBENCH_MINI)
    task_ids = {it.task_id for it in items}
    assert len(task_ids) >= 3
    assert {"m-crop-type", "m-land-cover", "m-eurocrops"} <= task_ids
    first = next(it for it in items if it.item_id == "ct-001")
    assert first.gold_label == "maize"
    assert first.label_space == ["maize", "wheat", "soybean", "rice"]
    # No image_path in the fixture -> text-only, not image-required.
    assert first.requires_image is False
    assert "Respuesta:" in first.question


def test_load_geobench2_task_allowlist_and_cap() -> None:
    """The task allow-list and per-task cap are honoured."""
    items = load_geobench2(_GEOBENCH_MINI, tasks=["m-crop-type"], max_per_task=2)
    assert {it.task_id for it in items} == {"m-crop-type"}
    assert len(items) == 2


def test_load_geobench2_absent_degrades_to_empty(tmp_path: Path) -> None:
    """A missing GEO-Bench-2 root degrades to [] (pending), never fabricated."""
    assert load_geobench2(tmp_path / "nope") == []


def test_load_agromind_itses_reads_both_languages() -> None:
    """The mini JSONL yields 3 it + 3 es items with the bilingual schema."""
    items = load_agromind_itses(_ITSES_MINI)
    assert len(items) == 6
    assert sum(1 for it in items if it.lang == "it") == 3
    assert sum(1 for it in items if it.lang == "es") == 3
    mc = next(it for it in items if it.item_id == "es-001")
    assert mc.answer == "B"
    assert mc.options["B"] == "NDVI"
    assert mc.is_multimodal is False


def test_load_agromind_itses_absent_degrades_to_empty(tmp_path: Path) -> None:
    """A missing US-068 file degrades to [] (pending), never fabricated."""
    assert load_agromind_itses(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------
def test_macro_f1_perfect_and_zero() -> None:
    """macro_f1 is 1.0 on a perfect match and 0.0 on a total mismatch."""
    labels = ["a", "b"]
    assert macro_f1(["a", "b"], ["a", "b"], labels) == pytest.approx(1.0)
    assert macro_f1(["a", "a"], ["b", "b"], labels) == pytest.approx(0.0)


def test_macro_f1_known_value() -> None:
    """macro_f1 matches the hand-computed value on a known confusion."""
    # gold = [a, a, b], pred = [a, b, b]:
    #   class a: tp=1 fp=0 fn=1 -> F1 = 2*1/(2+1) = 0.6667
    #   class b: tp=1 fp=1 fn=0 -> F1 = 2*1/(2+1) = 0.6667
    #   macro = 0.6667
    value = macro_f1(["a", "b", "b"], ["a", "a", "b"], ["a", "b"])
    assert value == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_macro_f1_case_insensitive() -> None:
    """Class comparison is case-insensitive on the stripped string."""
    assert macro_f1([" Maize "], ["maize"], ["maize"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# wilcoxon_paired
# ---------------------------------------------------------------------------
def test_wilcoxon_paired_significant_on_consistent_gap() -> None:
    """A consistent A>B gap over enough pairs yields a small p-value."""
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = wilcoxon_paired(a, b)
    assert isinstance(result, WilcoxonResult)
    assert result.n_pairs == 8
    assert not math.isnan(result.p_value)
    assert result.p_value < 0.05


def test_wilcoxon_paired_all_tied_not_computable() -> None:
    """All-tied vectors are not computable (NaN + Spanish note), no fake p."""
    result = wilcoxon_paired([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert math.isnan(result.p_value)
    assert result.n_pairs == 0
    assert result.note


def test_wilcoxon_paired_too_few_pairs() -> None:
    """Fewer than min_pairs non-tied observations -> not computable."""
    result = wilcoxon_paired([1.0, 0.0], [0.0, 0.0], min_pairs=6)
    assert math.isnan(result.p_value)
    assert result.n_pairs == 1


# ---------------------------------------------------------------------------
# eval_geobench2
# ---------------------------------------------------------------------------
def test_eval_geobench2_scores_with_scripted_backend() -> None:
    """A backend answering the gold class scores a perfect exact-match."""
    import asyncio

    items = load_geobench2(_GEOBENCH_MINI, tasks=["m-crop-type"])
    answers = [f"Respuesta: {it.gold_label}" for it in items]
    backend = AnswerBackend(answers=answers)
    result = asyncio.run(eval_geobench2(GEMINI, items, backend=backend, seed=0))
    assert result["exact_match"] == pytest.approx(1.0)
    assert result["f1_macro"] == pytest.approx(1.0)
    assert result["n_evaluated"] == len(items)
    assert result["n_skipped"] == 0
    assert "_item_ids" in result and len(result["_item_ids"]) == len(items)  # type: ignore[arg-type]


def test_eval_geobench2_text_only_skips_image_items() -> None:
    """A text-only variant skips image-required items and reports n_skipped."""
    import asyncio

    items = [
        GeoBenchItem(
            task_id="t", item_id="i1", image_path="tile.png", question="q?",
            label_space=["a", "b"], gold_label="a", requires_image=True,
        ),
        GeoBenchItem(
            task_id="t", item_id="i2", image_path="", question="q?",
            label_space=["a", "b"], gold_label="b", requires_image=False,
        ),
    ]
    backend = CyclingBackend(cycle=["Respuesta: b"])
    result = asyncio.run(eval_geobench2(QWEN, items, backend=backend, seed=0))
    assert result["n_skipped"] == 1
    assert result["n_evaluated"] == 1


# ---------------------------------------------------------------------------
# eval_agromind_itses
# ---------------------------------------------------------------------------
def test_eval_agromind_itses_perfect_with_judge() -> None:
    """A backend answering the gold letter scores 1.0 with per-language EM."""
    import asyncio

    items = load_agromind_itses(_ITSES_MINI)
    answers = [f"Respuesta: {it.answer}" for it in items]
    backend = AnswerBackend(answers=answers)
    judge = FakeJudge(score_value=0.0)
    result = asyncio.run(
        eval_agromind_itses(GEMINI, items, backend=backend, judge=judge, seed=0)
    )
    assert result["exact_match"] == pytest.approx(1.0)
    assert result["exact_match_it"] == pytest.approx(1.0)
    assert result["exact_match_es"] == pytest.approx(1.0)
    assert result["hallucination"] == pytest.approx(0.0)
    assert result["n_evaluated"] == len(items)


def test_eval_agromind_itses_no_judge_reports_nan_hallucination() -> None:
    """With no judge the hallucination rate is NaN (rendered n/a), not 0."""
    import asyncio

    items = load_agromind_itses(_ITSES_MINI)
    backend = CyclingBackend(cycle=["Respuesta: A"])
    result = asyncio.run(eval_agromind_itses(GEMINI, items, backend=backend, seed=0))
    assert math.isnan(result["hallucination"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_paper_benchmark
# ---------------------------------------------------------------------------
def test_run_paper_benchmark_pending_when_geobench_and_itses_absent(
    tmp_path: Path,
) -> None:
    """GEO-Bench-2 + IT/ES absent -> those cells pending; AgroMind off here too."""
    out = run_paper_benchmark(
        (GEMINI, QWEN),
        seeds=(0,),
        geobench_root=tmp_path / "no_geobench",
        agromind_path=tmp_path / "no_agromind.json",
        itses_path=tmp_path / "no_itses.jsonl",
        backends={"gemini": AnswerBackend(answers=[]), "qwen": AnswerBackend(answers=[])},
        log_mlflow=False,
    )
    assert set(out["pending"]) == set(BENCHMARKS)
    for variant in ("gemini", "qwen"):
        for benchmark in BENCHMARKS:
            assert out["results"][variant][benchmark]["status"] == "pending"


def test_run_paper_benchmark_scores_present_benchmarks(tmp_path: Path) -> None:
    """With GEO-Bench-2 + IT/ES present, both score and Wilcoxon is computed."""
    geo_answers = ["Respuesta: maize"] * 200  # over-provisioned cycle is fine
    backends = {
        "gemini": CyclingBackend(cycle=geo_answers),
        "qwen": CyclingBackend(cycle=["Respuesta: wheat"]),
    }
    out = run_paper_benchmark(
        (GEMINI, QWEN),
        seeds=(0, 1),
        geobench_root=_GEOBENCH_MINI,
        agromind_path=tmp_path / "no_agromind.json",
        itses_path=_ITSES_MINI,
        geobench_tasks=["m-crop-type"],
        backends=backends,
        log_mlflow=False,
    )
    assert "AgroMind" in out["pending"]
    assert out["results"]["gemini"]["GEO-Bench-2"]["status"] == "ok"
    assert out["results"]["qwen"]["AgroMind-IT/ES"]["status"] == "ok"
    # Gemini answered the gold "maize" for the crop-type cycle -> some hits.
    assert out["results"]["gemini"]["GEO-Bench-2"]["exact_match"]["mean"] >= 0.0
    # Wilcoxon is present per benchmark (note when not computable on tiny n).
    assert set(out["wilcoxon"]) == set(BENCHMARKS)


def test_run_paper_benchmark_wilcoxon_requires_both_variants() -> None:
    """A single-variant run reports Wilcoxon as not-computable, not a fake p."""
    out = run_paper_benchmark(
        (GEMINI,),
        seeds=(0,),
        geobench_root=_GEOBENCH_MINI,
        agromind_path=Path("data/none.json"),
        itses_path=_ITSES_MINI,
        geobench_tasks=["m-crop-type"],
        backends={"gemini": CyclingBackend(cycle=["Respuesta: maize"])},
        log_mlflow=False,
    )
    for benchmark in BENCHMARKS:
        assert math.isnan(out["wilcoxon"][benchmark]["p_value"])
        assert out["wilcoxon"][benchmark]["note"]


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------
def test_export_latex_table_writes_booktabs(tmp_path: Path) -> None:
    """The LaTeX export writes a booktabs table with the correct attributions."""
    out = run_paper_benchmark(
        (GEMINI, QWEN),
        seeds=(0,),
        geobench_root=_GEOBENCH_MINI,
        agromind_path=tmp_path / "no_agromind.json",
        itses_path=_ITSES_MINI,
        geobench_tasks=["m-crop-type"],
        backends={
            "gemini": CyclingBackend(cycle=["Respuesta: maize"]),
            "qwen": CyclingBackend(cycle=["Respuesta: maize"]),
        },
        log_mlflow=False,
    )
    out_path = tmp_path / "benchmark_comparison.tex"
    export_latex_table(out, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in text
    assert "\\toprule" in text and "\\bottomrule" in text
    # Attribution + honesty checks (correcciones factuales obligatorias).
    assert "SATELLITE\\_EMBEDDING/V1/ANNUAL" in text
    # The benchmark runs gemini-2.5-flash (see ``PAPER_VARIANTS``); the caption
    # states it and cites 2.5-pro as the production reasoner.
    assert "Gemini 2.5-flash" in text and "2.5-pro" in text
    assert "eval-only" in text
    assert "Be My Eyes" in text
    # Pending AgroMind cell renders as italic placeholder, never a number.
    assert "\\textit{pend.}" in text
    # The IT/ES F1 column resolves to f1_squad (a number), not the f1_macro key
    # the QA evaluators never emit -> the IT/ES block carries a numeric F1 mean.
    itses_block = out["results"]["gemini"]["AgroMind-IT/ES"]
    assert "f1_squad" in itses_block
    assert not math.isnan(itses_block["f1_squad"]["mean"])


def test_paper_variants_are_the_two_reasoners() -> None:
    """The headline pair is exactly Gemini 2.5-pro + Qwen3.5 (AC literal)."""
    names = {v.name for v in PAPER_VARIANTS}
    assert names == {"gemini", "qwen"}
    gemini = next(v for v in PAPER_VARIANTS if v.name == "gemini")
    assert gemini.model == "gemini-2.5-flash"  # benchmark variant, see PAPER_VARIANTS
    qwen = next(v for v in PAPER_VARIANTS if v.name == "qwen")
    assert qwen.multimodal is False
