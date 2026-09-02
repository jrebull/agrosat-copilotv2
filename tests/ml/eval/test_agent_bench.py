"""Integration tests for the agent-benchmark harness (US-049).

These tests drive :mod:`ml.eval.agent_bench` with **mock backends only** (no
network, no real LLM) and the **real** datasets shipped under ``data/`` so the
loaders are exercised against the genuine 500-item AgroMind subset and the
50-task GeoAnalystBench CSV.

Key doubles:

- :class:`_FixedBackend` implements the :class:`~ml.agent.backends.LLMBackend`
  contract by exposing an async ``generate_stream`` that yields duck-typed
  chunks with a ``text`` attribute (the only thing the harness reads). It returns
  a canned answer so the parsed letter / workflow / code are deterministic.
- The sentence encoder behind the semantic proxies is monkeypatched with a
  deterministic fake (``sentence_transformers.SentenceTransformer``) and the
  module cache ``agent_metrics._sentence_model`` is reset, so no real model is
  loaded and ``mean_semantic_sim`` / ``bertscore`` stay finite and offline.

The Qwen text-only tension (AC-3) is verified explicitly: a non-multimodal
variant skips every multimodal AgroMind item and reports ``n_skipped`` > 0.

Conventions: identifiers and docstrings in English; visible prose elsewhere in
Spanish; no emojis; full type hints; ``pytest-asyncio`` in auto mode (no
decorator needed for the ``async def`` tests).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.eval import agent_bench, agent_metrics, agent_report

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGROMIND_PATH = _REPO_ROOT / "data" / "agromind" / "agromind_subset_500.json"
_GEO_PATH = _REPO_ROOT / "data" / "geoanalystbench" / "GeoAnalystBench.csv"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _Chunk:
    """Minimal backend chunk: only the ``text`` attribute is read by the harness."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FixedBackend:
    """Backend double returning a canned answer for every turn (no network).

    Implements the slice of :class:`~ml.agent.backends.LLMBackend` the harness
    uses: an async ``generate_stream`` yielding chunks with a ``text`` delta.
    """

    def __init__(self, answer: str) -> None:
        self.model = "mock"
        self._answer = answer
        self.calls = 0

    async def generate_stream(
        self,
        *,
        contents: Any,
        tools: Any,
        system_instruction: str,
    ) -> Any:
        """Yield the canned answer as a single text chunk."""
        self.calls += 1
        yield _Chunk(self._answer)


class _FakeEncoder:
    """Deterministic ``SentenceTransformer`` stand-in (see metrics test)."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._dim = 16

    def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
        """Embed strings to stable per-string vectors (identical -> identical)."""
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.standard_normal(self._dim))
        return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def fake_sentence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the sentence encoder and reset the module cache (offline proxies)."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeEncoder)
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)
    yield
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)


def _require_data() -> None:
    """Skip the test when the real datasets are not present locally."""
    if not _AGROMIND_PATH.exists():
        pytest.skip(f"AgroMind subset missing: {_AGROMIND_PATH}")
    if not _GEO_PATH.exists():
        pytest.skip(f"GeoAnalystBench CSV missing: {_GEO_PATH}")


# --------------------------------------------------------------------------- #
# Loaders against the real data
# --------------------------------------------------------------------------- #


class TestLoaders:
    """The loaders parse the genuine shipped datasets."""

    def test_load_agromind_subset_real_500(self) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        assert len(items) == 500
        n_multimodal = sum(1 for it in items if it.is_multimodal)
        # Documented split for this subset: 494 multimodal, 6 purely textual.
        assert n_multimodal == 494
        assert len(items) - n_multimodal == 6
        first = items[0]
        assert first.answer in {"A", "B", "C", "D"}
        assert set("ABCD").issuperset(first.options.keys())

    def test_option_image_paths_property(self) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        # Items whose options are image paths expose them via the property; the
        # property is a subset of options keyed by label.
        multi_opt = next((it for it in items if it.option_image_paths), None)
        if multi_opt is not None:
            assert set(multi_opt.option_image_paths).issubset(multi_opt.options)

    def test_load_geoanalystbench_real_50(self) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)
        assert len(tasks) == 50
        # The trailing blank row (empty id) must have been dropped.
        assert all(t.id for t in tasks)
        first = tasks[0]
        assert first.instruction
        assert first.human_workflow


# --------------------------------------------------------------------------- #
# _build_agromind_prompt (B-6: adaptive to the item shape)
# --------------------------------------------------------------------------- #


def _make_item(
    *, options: dict[str, str], answer: str, question: str = "Que cultivo es?"
) -> agent_bench.AgroMindItem:
    """Build a minimal :class:`AgroMindItem` for prompt/scoring unit tests."""
    return agent_bench.AgroMindItem(
        image_path="",
        question=question,
        options=options,
        answer=answer,
        type_id=0,
        item_id=0,
        level1_id=0,
        level2_id=0,
        level3_id=0,
        task_file="T",
        is_multimodal=False,
    )


class TestBuildAgromindPrompt:
    """The prompt adapts to the real label set and to open (no-option) items."""

    def test_open_item_asks_for_direct_answer_not_a_letter(self) -> None:
        # B-6: an item with no options must NOT be told to pick a letter; it asks
        # for the direct numeric/text answer (few-shot + CoT) instead, ending with
        # the ``Respuesta:`` contract.
        item = _make_item(options={}, answer="10", question="Cuantas parcelas?")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "letra" not in prompt.lower()
        assert "Opciones:" not in prompt
        assert "Respuesta:" in prompt
        assert "Cuantas parcelas?" in prompt

    def test_multiple_choice_lists_the_real_letter_set(self) -> None:
        # B-6: the instruction must reflect the labels actually present, not a
        # hardcoded "A, B, C o D".
        item = _make_item(options={"A": "maiz", "B": "trigo", "C": "arroz"}, answer="B")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "A, B o C" in prompt
        assert " o D" not in prompt
        assert "  A. maiz" in prompt

    def test_six_options_advertise_letters_up_to_f(self) -> None:
        # B-6: items with E/F options must surface those letters in the prompt,
        # not stop at D.
        item = _make_item(
            options={
                "A": "uno",
                "B": "dos",
                "C": "tres",
                "D": "cuatro",
                "E": "cinco",
                "F": "seis",
            },
            answer="F",
        )
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "A, B, C, D, E o F" in prompt
        assert "  F. seis" in prompt


class TestBuildAgromindPromptCoT:
    """The few-shot + CoT prompt branches by answer type with one example each."""

    def test_multiple_choice_prompt_is_few_shot_and_cot(self) -> None:
        # A multiple-choice item ships exactly one worked example and the
        # ``Respuesta: <letra>`` contract so the model reasons before committing.
        item = _make_item(options={"A": "maiz", "B": "trigo", "C": "arroz"}, answer="B")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "Ejemplo:" in prompt
        assert "Razonemos:" in prompt
        assert "Respuesta: B" in prompt  # the worked example's final line
        assert "paso a paso" in prompt

    def test_open_numeric_prompt_states_the_number_or_bbox_format(self) -> None:
        item = _make_item(options={}, answer="3", question="Cuantas parcelas?")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "Ejemplo:" in prompt
        assert "[x,y,x,y]" in prompt
        assert "Respuesta: 3" in prompt  # the worked example

    def test_yes_no_prompt_ends_with_si_no_contract(self) -> None:
        item = _make_item(options={}, answer="si", question="Hay agua?")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "Respuesta: Si" in prompt
        assert "Respuesta: No" in prompt

    def test_open_text_prompt_asks_for_concise_no_letter(self) -> None:
        item = _make_item(options={}, answer="trigo de invierno", question="Cultivo?")
        prompt = agent_bench._build_agromind_prompt(item, with_images=False)
        assert "letra" in prompt.lower()  # only as a negative instruction
        assert "sin una letra" in prompt.lower()
        assert "Respuesta:" in prompt


# --------------------------------------------------------------------------- #
# _extract_final_answer (CoT final-answer extraction, US-049 prompting upgrade)
# --------------------------------------------------------------------------- #


class TestExtractFinalAnswer:
    """The extractor reads the text after the LAST ``Respuesta:`` marker."""

    def test_reads_text_after_last_respuesta_marker(self) -> None:
        answer = "Razonemos: el NDVI mide vegetacion.\nRespuesta: C"
        assert agent_bench._extract_final_answer(answer) == "C"

    def test_reads_open_numeric_final_answer(self) -> None:
        answer = "Razonemos: cuento cero lotes de maiz.\nRespuesta: 0"
        assert agent_bench._extract_final_answer(answer) == "0"

    def test_last_marker_wins_when_several_present(self) -> None:
        # A model may echo the example's ``Respuesta:`` before its own; the LAST
        # one is the real final answer.
        answer = "Respuesta: B\nrevisando...\nRespuesta: D"
        assert agent_bench._extract_final_answer(answer) == "D"

    def test_answer_marker_in_english_is_also_recognised(self) -> None:
        answer = "Reasoning...\nAnswer: A"
        assert agent_bench._extract_final_answer(answer) == "A"

    def test_no_marker_falls_back_to_full_answer(self) -> None:
        # Un-marked responses keep today's behaviour (the whole answer is scored).
        assert agent_bench._extract_final_answer("B") == "B"
        assert agent_bench._extract_final_answer("The answer is F") == "The answer is F"

    def test_cot_answer_scores_exact_match_against_gold_letter(self) -> None:
        # End to end: a CoT answer must parse to exact_match 1.0 against gold "C".
        answer = "Razonemos: la opcion C describe trigo.\nRespuesta: C"
        final = agent_bench._extract_final_answer(answer)
        assert agent_metrics.exact_match(final, "C", frozenset("ABC")) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_cot_open_answer_extracts_zero(self) -> None:
        answer = "Razonemos: no hay parcelas.\nRespuesta: 0"
        final = agent_bench._extract_final_answer(answer)
        assert final == "0"
        assert agent_metrics.exact_match(final, "0", None) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# _split_workflow_and_code (B-10: keep prose written after the code block)
# --------------------------------------------------------------------------- #


class TestSplitWorkflowAndCode:
    """The splitter keeps workflow prose on BOTH sides of the code block."""

    def test_workflow_before_and_after_code_is_kept(self) -> None:
        # B-10: previously only the pre-fence text survived; workflow steps after
        # the code block were dropped, zeroing the similarity. Both must remain.
        answer = (
            "WORKFLOW:\n1. cargar raster\n"
            "CODE:\n```python\nx = load()\n```\n"
            "2. recortar al AOI\n3. calcular NDVI"
        )
        workflow, code = agent_bench._split_workflow_and_code(answer)
        assert code == "x = load()"
        assert "1. cargar raster" in workflow
        assert "2. recortar al AOI" in workflow
        assert "3. calcular NDVI" in workflow
        # The code body must not leak into the workflow text.
        assert "x = load()" not in workflow
        assert "```" not in workflow

    def test_pre_fence_only_still_works(self) -> None:
        answer = "WORKFLOW:\n1. paso uno\nCODE:\n```python\npass\n```"
        workflow, code = agent_bench._split_workflow_and_code(answer)
        assert code == "pass"
        assert "1. paso uno" in workflow
        assert "```" not in workflow

    def test_no_fence_returns_full_text_as_workflow(self) -> None:
        answer = "WORKFLOW:\n1. solo prosa, sin codigo"
        workflow, code = agent_bench._split_workflow_and_code(answer)
        assert code == ""
        assert "1. solo prosa, sin codigo" in workflow


# --------------------------------------------------------------------------- #
# eval_agromind
# --------------------------------------------------------------------------- #


class TestEvalAgromind:
    """AgroMind evaluation with a mock backend."""

    async def test_multimodal_variant_scores_all_items(self, fake_sentence_model: None) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        gold = items[0].answer
        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        backend = _FixedBackend(answer=gold)
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=backend,
            judge=None,
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        # Multimodal variant evaluates the full subset, skips nothing.
        assert result["n_evaluated"] == 500
        assert result["n_skipped"] == 0
        assert 0.0 <= result["exact_match"] <= 1.0
        # The backend always answers the single letter ``gold``; every item whose
        # gold answer is that same letter is a hit, so exact_match is at least the
        # fraction of items carrying that letter (free-text golds that embed an
        # A-D token may also match, so this is a lower bound, never above 1).
        n_gold = sum(1 for it in items if it.answer == gold)
        assert result["exact_match"] >= n_gold / 500 - 1e-9

    async def test_text_only_variant_skips_multimodal(self, fake_sentence_model: None) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        backend = _FixedBackend(answer="A")
        result = await agent_bench.eval_agromind(
            variant, items, backend=backend, judge=None, seed=0
        )
        # Text-only Qwen skips the 494 multimodal items and scores only the 6
        # purely-textual ones; the limitation is reported, never papered over.
        assert result["n_skipped"] == 494
        assert result["n_evaluated"] == 6
        assert backend.calls == 6

    async def test_open_numeric_item_scores_via_text_fallback(
        self, fake_sentence_model: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-5/B-6: an item with no options carries a numeric gold; the backend
        # returns the number directly and it must score 1.0 through the text
        # fallback (no spurious letter parsing). The coverage floor is patched to
        # 1 so this single-item scoring unit test is not NaN'd by it.
        monkeypatch.setattr(agent_bench, "_MIN_AGROMIND_N", 1)
        item = _make_item(options={}, answer="10", question="Cuantas?")
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        result = await agent_bench.eval_agromind(
            variant, [item], backend=_FixedBackend(answer="10"), judge=None, seed=0
        )
        assert result["n_evaluated"] == 1
        assert result["exact_match"] == pytest.approx(1.0, abs=1e-9)

    async def test_cot_answer_is_scored_on_final_respuesta_line(
        self, fake_sentence_model: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The model returns reasoning THEN ``Respuesta: B``; the harness must
        # extract the final line and score it, not the chain-of-thought. Floor
        # patched to 1 so the single-item score is not NaN'd by the coverage gate.
        monkeypatch.setattr(agent_bench, "_MIN_AGROMIND_N", 1)
        item = _make_item(options={"A": "maiz", "B": "trigo", "C": "arroz"}, answer="B")
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        backend = _FixedBackend(
            answer="Razonemos: el patron de NDVI corresponde a trigo.\nRespuesta: B"
        )
        records: list[dict[str, Any]] = []
        result = await agent_bench.eval_agromind(
            variant,
            [item],
            backend=backend,
            judge=None,
            seed=0,
            trace_sink=records.append,
        )
        assert result["exact_match"] == pytest.approx(1.0, abs=1e-9)
        # The trace keeps the full CoT prediction and the extracted final answer.
        assert records[0]["final_answer"] == "B"
        assert "Razonemos" in records[0]["prediction"]

    async def test_high_letter_item_scores_when_answer_is_f(
        self, fake_sentence_model: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # B-5: a six-option item whose gold is F must score 1.0 when the backend
        # answers "F" wrapped in prose (the old [A-D] parser scored 0). Floor
        # patched to 1 so this single-item unit test is not NaN'd by it.
        monkeypatch.setattr(agent_bench, "_MIN_AGROMIND_N", 1)
        item = _make_item(
            options={
                "A": "uno",
                "B": "dos",
                "C": "tres",
                "D": "cuatro",
                "E": "cinco",
                "F": "seis",
            },
            answer="F",
        )
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        result = await agent_bench.eval_agromind(
            variant,
            [item],
            backend=_FixedBackend(answer="The answer is F"),
            judge=None,
            seed=0,
        )
        assert result["exact_match"] == pytest.approx(1.0, abs=1e-9)

    async def test_hallucination_is_nan_without_judge(self, fake_sentence_model: None) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)[:5]
        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="A"),
            judge=None,
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        assert math.isnan(float(result["hallucination"]))

    async def test_judge_injected_yields_finite_hallucination(
        self, fake_sentence_model: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_data()
        # Patch the coverage floor so the 4-item judge sample is scored, not NaN'd.
        monkeypatch.setattr(agent_bench, "_MIN_AGROMIND_N", 1)
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)[:4]

        class _Judge:
            def score(self, sample: dict[str, Any]) -> float:
                return 0.1

        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="A"),
            judge=_Judge(),
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        assert result["hallucination"] == pytest.approx(0.1, abs=1e-9)


# --------------------------------------------------------------------------- #
# eval_geoanalyst
# --------------------------------------------------------------------------- #


class TestEvalGeoanalyst:
    """GeoAnalystBench evaluation with a mock backend."""

    async def test_pass_rate_high_when_workflow_matches(self, fake_sentence_model: None) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)[:5]
        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        # The mock echoes each task's own gold workflow + code, so the fake
        # encoder embeds identical text (cosine 1.0) -> every task passes.
        first = tasks[0]
        answer = f"WORKFLOW:\n{first.human_workflow}\nCODE:\n```python\n{first.code_string}\n```"
        backend = _FixedBackend(answer=answer)
        result = await agent_bench.eval_geoanalyst(variant, [first], backend=backend, seed=0)
        assert result["n"] == 1
        assert result["pass_rate"] == pytest.approx(1.0, abs=1e-9)
        assert result["mean_semantic_sim"] == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= result["mean_codebleu"] <= 1.0

    async def test_full_taskset_runs_for_text_only_variant(self, fake_sentence_model: None) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        backend = _FixedBackend(answer="WORKFLOW:\n1. do nothing\nCODE:\n```python\npass\n```")
        result = await agent_bench.eval_geoanalyst(variant, tasks, backend=backend, seed=0)
        # GeoAnalystBench is 100% text: every variant runs the full task set.
        assert result["n"] == 50
        assert backend.calls == 50
        assert 0.0 <= result["pass_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# run_benchmark (aggregation + report)
# --------------------------------------------------------------------------- #


class TestRunBenchmark:
    """End-to-end aggregation over seeds and HTML report generation."""

    def test_run_benchmark_aggregates_and_writes_report(
        self, fake_sentence_model: None, tmp_path: Path
    ) -> None:
        _require_data()
        variants = [
            agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True),
            agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False),
        ]
        backends = {
            "gemini": _FixedBackend(answer="A"),
            "qwen": _FixedBackend(answer="A"),
        }
        report_path = tmp_path / "agent_bench.html"
        results = agent_bench.run_benchmark(
            variants,
            seeds=(0, 1),
            agromind_path=_AGROMIND_PATH,
            geo_path=_GEO_PATH,
            backends=backends,
            judge=None,
            image_root=tmp_path / "no_images",
            report_path=report_path,
            log_mlflow=False,
            probe_server=False,
        )

        # Nested shape: {variant: {benchmark: {metric: {"mean", "std"}}}}.
        assert set(results) == {"gemini", "qwen"}
        for variant in ("gemini", "qwen"):
            assert set(results[variant]) == {"AgroMind", "GeoAnalystBench"}
            agro = results[variant]["AgroMind"]
            geo = results[variant]["GeoAnalystBench"]
            # Headline keys are populated with the exact expected names.
            assert "exact_match" in agro
            assert "pass_rate" in geo
            for metric_stats in (agro["exact_match"], geo["pass_rate"]):
                assert set(metric_stats) == {"mean", "std"}
                assert isinstance(metric_stats["mean"], float)
                assert isinstance(metric_stats["std"], float)
            # The deterministic GeoAnalystBench pass_rate has std 0.0 across seeds.
            assert geo["pass_rate"]["std"] == pytest.approx(0.0, abs=1e-9)
        # Gemini scores the full 500-item subset: its AgroMind exact_match is a
        # finite number with std 0.0 (deterministic mock). Qwen is text-only and
        # evaluates only the 6 textual items (< the coverage floor), so its
        # AgroMind exact_match is NaN by design -- the honest "not evaluable"
        # verdict, not a fabricated comparable score.
        assert results["gemini"]["AgroMind"]["exact_match"]["std"] == pytest.approx(0.0, abs=1e-9)
        assert not math.isnan(results["gemini"]["AgroMind"]["exact_match"]["mean"])
        assert math.isnan(results["qwen"]["AgroMind"]["exact_match"]["mean"])

        # The text-only Qwen carries the n_skipped signal in its AgroMind table.
        assert results["qwen"]["AgroMind"]["n_skipped"]["mean"] == pytest.approx(494.0, abs=1e-9)

        # The HTML report was written.
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "gemini" in content
        assert "qwen" in content

    def test_run_benchmark_checkpoints_each_variant(
        self, fake_sentence_model: None, tmp_path: Path
    ) -> None:
        """Each variant's metrics are persisted to the checkpoint as it finishes."""
        _require_data()
        variants = [
            agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True),
            agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False),
        ]
        backends = {"gemini": _FixedBackend("A"), "qwen": _FixedBackend("A")}
        ckpt = tmp_path / "ck.json"
        agent_bench.run_benchmark(
            variants,
            seeds=(0,),
            agromind_path=_AGROMIND_PATH,
            geo_path=_GEO_PATH,
            backends=backends,
            image_root=tmp_path / "none",
            report_path=tmp_path / "r.html",
            checkpoint_path=ckpt,
            log_mlflow=False,
            probe_server=False,
        )
        assert ckpt.exists()
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        assert set(saved) == {"gemini", "qwen"}
        assert "AgroMind" in saved["gemini"]

    def test_run_benchmark_resume_skips_done_variants(
        self, fake_sentence_model: None, tmp_path: Path
    ) -> None:
        """``resume`` loads the checkpoint and never recomputes a done variant."""
        _require_data()
        ckpt = tmp_path / "ck.json"
        # Pre-seed the checkpoint as if 'gemini' had already been evaluated.
        sentinel = {
            "gemini": {
                "AgroMind": {"exact_match": {"mean": 0.42, "std": 0.0}},
                "GeoAnalystBench": {"pass_rate": {"mean": 0.5, "std": 0.0}},
            }
        }
        ckpt.write_text(json.dumps(sentinel), encoding="utf-8")
        gem = _FixedBackend("A")
        qwen = _FixedBackend("A")
        variants = [
            agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True),
            agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False),
        ]
        results = agent_bench.run_benchmark(
            variants,
            seeds=(0,),
            agromind_path=_AGROMIND_PATH,
            geo_path=_GEO_PATH,
            backends={"gemini": gem, "qwen": qwen},
            image_root=tmp_path / "none",
            report_path=tmp_path / "r.html",
            checkpoint_path=ckpt,
            resume=True,
            log_mlflow=False,
            probe_server=False,
        )
        # Gemini came from the checkpoint -> its backend never ran and its metrics
        # are the pre-seeded ones (not recomputed). Qwen still ran.
        assert gem.calls == 0
        assert results["gemini"]["AgroMind"]["exact_match"]["mean"] == pytest.approx(0.42)
        assert qwen.calls > 0

    def test_eval_agromind_item_timeout_does_not_hang(
        self,
        fake_sentence_model: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wedged model call is bounded by the per-item timeout, not infinite."""
        _require_data()
        monkeypatch.setattr(agent_bench, "_ITEM_TIMEOUT_S", 0.05)

        class _HangBackend:
            """generate_stream that never yields (simulates a wedged socket)."""

            model = "mock"

            async def generate_stream(self, *, contents, tools, system_instruction):
                await asyncio.Event().wait()  # never resolves
                yield _Chunk("A")  # pragma: no cover - unreachable

        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)[:3]
        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        # Without the wait_for this hangs forever; the test completing at all proves
        # the timeout fires and each item is recorded as a failure (empty answer).
        result = asyncio.run(
            agent_bench.eval_agromind(
                variant, items, backend=_HangBackend(), image_root=tmp_path / "none"
            )
        )
        assert result["n_evaluated"] >= 1


# --------------------------------------------------------------------------- #
# _classify_answer_type (per-item trace bucketing, US-049)
# --------------------------------------------------------------------------- #


class TestClassifyAnswerType:
    """The answer-type classifier covers all five documented buckets."""

    def test_multiple_choice_when_item_has_options(self) -> None:
        # Any item carrying options is multiple_choice, regardless of the gold.
        item = _make_item(options={"A": "maiz", "B": "trigo"}, answer="A")
        assert agent_bench._classify_answer_type(item) == "multiple_choice"

    def test_open_numeric_bbox_when_gold_is_a_box(self) -> None:
        # An open item whose gold is a [a, b, ...] numeric box buckets as bbox.
        item = _make_item(options={}, answer="[0.1, 0.2, 0.3, 0.4]")
        assert agent_bench._classify_answer_type(item) == "open_numeric_bbox"

    def test_open_number_when_gold_is_a_pure_number(self) -> None:
        # A bare int/float gold (no options, not a box) buckets as open_number.
        item = _make_item(options={}, answer="42")
        assert agent_bench._classify_answer_type(item) == "open_number"
        item_float = _make_item(options={}, answer="-3.14")
        assert agent_bench._classify_answer_type(item_float) == "open_number"

    def test_yes_no_when_gold_is_a_boolean_token(self) -> None:
        # Yes/no tokens across es/it/en bucket as yes_no (case-insensitive).
        for gold in ("yes", "No", "si", "TRUE"):
            item = _make_item(options={}, answer=gold)
            assert agent_bench._classify_answer_type(item) == "yes_no"

    def test_open_text_when_gold_is_free_text(self) -> None:
        # Anything else (a phrase, no options) falls through to open_text.
        item = _make_item(options={}, answer="trigo de invierno")
        assert agent_bench._classify_answer_type(item) == "open_text"


# --------------------------------------------------------------------------- #
# trace_sink wiring (US-049 per-item trace)
# --------------------------------------------------------------------------- #


class TestTraceSink:
    """The eval runners emit one trace record per scored item to the sink."""

    async def test_eval_agromind_invokes_trace_sink(self, fake_sentence_model: None) -> None:
        # Two purely-textual items (no options) so a text-only variant scores both
        # and the sink receives exactly one record each, with the documented keys.
        items = [
            _make_item(options={}, answer="10", question="Cuantas parcelas?"),
            _make_item(options={}, answer="20", question="Cuantos lotes?"),
        ]
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        records: list[dict[str, Any]] = []
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="10"),
            judge=None,
            seed=0,
            trace_sink=records.append,
        )
        # One record per evaluated (non-skipped) item.
        assert result["n_evaluated"] == 2
        assert len(records) == 2
        for record in records:
            # Documented fields present on every AgroMind trace record.
            for key in ("variant", "gold", "prediction", "correct", "answer_type"):
                assert key in record
            assert record["variant"] == "qwen"
            assert record["benchmark"] == "AgroMind"
            assert record["answer_type"] == "open_number"
            assert isinstance(record["correct"], bool)
        # The backend always answers "10": item one is correct, item two is not.
        assert records[0]["gold"] == "10"
        assert records[0]["correct"] is True
        assert records[1]["gold"] == "20"
        assert records[1]["correct"] is False

    async def test_eval_agromind_trace_sink_skips_multimodal_items(
        self, fake_sentence_model: None
    ) -> None:
        # A text-only variant skips multimodal items: the sink must NOT see them
        # (a skipped item has no prediction to score).
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        variant = agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False)
        records: list[dict[str, Any]] = []
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="A"),
            judge=None,
            seed=0,
            trace_sink=records.append,
        )
        # Exactly the evaluated items reach the sink; the 494 skipped do not.
        assert len(records) == result["n_evaluated"] == 6

    async def test_eval_geoanalyst_invokes_trace_sink(self, fake_sentence_model: None) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)[:3]
        variant = agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True)
        records: list[dict[str, Any]] = []
        result = await agent_bench.eval_geoanalyst(
            variant,
            tasks,
            backend=_FixedBackend(answer="WORKFLOW:\n1. nada\nCODE:\n```python\npass\n```"),
            seed=0,
            trace_sink=records.append,
        )
        # One record per task, carrying the geo-specific documented fields.
        assert result["n"] == 3
        assert len(records) == 3
        for record in records:
            for key in ("variant", "gold", "prediction", "workflow_sim", "passed"):
                assert key in record
            assert record["variant"] == "gemini"
            assert record["benchmark"] == "GeoAnalystBench"
            assert isinstance(record["passed"], bool)
            assert record["passed"] == record["correct"]


# --------------------------------------------------------------------------- #
# run_benchmark dump_jsonl (per-variant JSONL trace files)
# --------------------------------------------------------------------------- #


class TestRunBenchmarkDumpJsonl:
    """``dump_jsonl`` writes one valid JSONL file per (variant, benchmark)."""

    def test_run_benchmark_dump_jsonl_writes_per_variant_files(
        self, fake_sentence_model: None, tmp_path: Path
    ) -> None:
        _require_data()
        variants = [
            agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True),
            agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False),
        ]
        backends = {
            "gemini": _FixedBackend(answer="A"),
            "qwen": _FixedBackend(answer="A"),
        }
        dump_dir = tmp_path / "traces"
        agent_bench.run_benchmark(
            variants,
            seeds=(0,),
            agromind_path=_AGROMIND_PATH,
            geo_path=_GEO_PATH,
            backends=backends,
            judge=None,
            image_root=tmp_path / "no_images",
            report_path=tmp_path / "agent_bench.html",
            dump_jsonl=dump_dir,
            log_mlflow=False,
            probe_server=False,
        )
        # One JSONL per (variant, benchmark) exists.
        expected = [
            dump_dir / "trace_gemini_AgroMind.jsonl",
            dump_dir / "trace_gemini_GeoAnalystBench.jsonl",
            dump_dir / "trace_qwen_AgroMind.jsonl",
            dump_dir / "trace_qwen_GeoAnalystBench.jsonl",
        ]
        for path in expected:
            assert path.exists(), f"missing trace file: {path}"
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert lines, f"empty trace file: {path}"
            # Every line is a valid JSON object carrying its variant tag.
            for line in lines:
                record = json.loads(line)
                assert isinstance(record, dict)
                assert "variant" in record
                assert "benchmark" in record

        # Gemini (multimodal) scores the full subset; Qwen only the 6 textual.
        gemini_agro = (dump_dir / "trace_gemini_AgroMind.jsonl").read_text(encoding="utf-8")
        assert len([ln for ln in gemini_agro.splitlines() if ln.strip()]) == 500
        qwen_agro = (dump_dir / "trace_qwen_AgroMind.jsonl").read_text(encoding="utf-8")
        assert len([ln for ln in qwen_agro.splitlines() if ln.strip()]) == 6


# --------------------------------------------------------------------------- #
# build_report_html (examples / answer_type_breakdown sections, US-049)
# --------------------------------------------------------------------------- #


class TestBuildReportHtmlExamples:
    """The two trace-backed sections render and are strictly opt-in (back-compat)."""

    @staticmethod
    def _results() -> dict[str, Any]:
        """A minimal nested results mapping for the report."""
        return {
            "gemini": {
                "AgroMind": {
                    "exact_match": {"mean": 0.8, "std": 0.01},
                    "n_skipped": {"mean": 0.0, "std": 0.0},
                },
                "GeoAnalystBench": {"pass_rate": {"mean": 0.7, "std": 0.0}},
            }
        }

    def test_build_report_html_with_examples_renders_sections(self, tmp_path: Path) -> None:
        results = self._results()
        breakdown = {
            "gemini": {
                "open_number": {"n": 4.0, "exact_match_mean": 0.25},
                "multiple_choice": {"n": 10.0, "exact_match_mean": 0.9},
            }
        }
        examples = {
            "gemini": {
                "AgroMind": [
                    {
                        "prompt": "Cuantas parcelas?",
                        "gold": "10",
                        "prediction": "10",
                        "answer_type": "open_number",
                        "exact_match": 1.0,
                        "correct": True,
                    }
                ],
                "GeoAnalystBench": [
                    {
                        "prompt": "Calcula NDVI",
                        "gold": "1. cargar raster",
                        "prediction": "1. cargar raster",
                        "workflow_sim": 0.9,
                        "codebleu": 0.5,
                        "passed": True,
                    }
                ],
            }
        }
        out = tmp_path / "with_sections.html"
        agent_report.build_report_html(
            results,
            out,
            examples=examples,
            answer_type_breakdown=breakdown,
        )
        content = out.read_text(encoding="utf-8")
        # The two new Spanish <h2> sections appear.
        assert "<h2>Desglose por tipo de respuesta (AgroMind)</h2>" in content
        assert "<h2>Ejemplos de inferencia (aciertos y fallos)</h2>" in content
        # Section content is rendered, not just the headings.
        assert "open_number" in content
        assert "Calcula NDVI" in content

    def test_build_report_html_without_sections_is_byte_identical(self, tmp_path: Path) -> None:
        # Back-compat: calling WITHOUT the new kwargs must produce the exact same
        # bytes as the previous two-argument behaviour (no trace sections, not
        # even empty placeholders).
        results = self._results()
        out_plain = tmp_path / "plain.html"
        out_explicit_none = tmp_path / "explicit_none.html"
        agent_report.build_report_html(results, out_plain)
        agent_report.build_report_html(
            results,
            out_explicit_none,
            examples=None,
            answer_type_breakdown=None,
        )
        plain_bytes = out_plain.read_bytes()
        none_bytes = out_explicit_none.read_bytes()
        # Passing the kwargs as None is identical to omitting them.
        assert plain_bytes == none_bytes
        # And neither carries the trace section headings.
        text = plain_bytes.decode("utf-8")
        assert "Desglose por tipo de respuesta" not in text
        assert "Ejemplos de inferencia" not in text
