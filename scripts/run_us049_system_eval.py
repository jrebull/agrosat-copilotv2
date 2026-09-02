"""US-049: run the PROJECT-GROUNDED agent eval live over the four variants.

This is the eval that measures OUR system on OUR tools/data (tool-calling
correctness, grounded-crop orchestration, RAG A/B hallucination), as opposed to
the external public benchmarks (AgroMind/GeoAnalystBench) which probe base-VLM
perception. It wires the live reasoner backends (Gemini cloud + the three on-prem
endpoints) and the real tool registry, and writes a JSON summary.

It complements ``scripts/run_us049_eval.py`` (the public benchmark). Run with the
H100 endpoints up and the local forwards active (:8002 Qwen text, :8003 Qwen3.6
VL, :11435 Gemma); Gemini reads its key from ``.env.local`` via ``Settings``.

Grounded-crop stubs the classifier seams (``_fetch_parcel_embedding`` +
``_load_classifier``) with a standalone ``pytest.MonkeyPatch`` so the REAL
``classify.run`` plumbing runs while the ensemble output is the injected,
deterministic per-case result -- the eval scores AGENT ORCHESTRATION + faithful
reporting, never the classifier's own accuracy.

US-081 adds a ``--label-space france-12`` grounded-crop variant (the
``--france12-offline`` mode). It re-runs ONLY the grounded-crop eval under the
twelve-class champion vocabulary with a deterministic per-case ORACLE backend
(cero red, cero creds): the oracle always routes to ``classify_new_parcel`` and
faithfully names the crop the REAL ``classify.run`` returns (the in-vocab argmax),
or, for an out-of-vocabulary case, hedges toward the ``unresolved_candidate`` the
REAL restriction surfaces. This isolates and measures the pieces US-081 actually
changed -- the france-12 restriction, the three new resolved crops, and the
out-of-vocabulary handoff -- without an LLM (the live-LLM scorecard is the
blocked path, run with the four variants once endpoints/creds are available).

Usage:
    poetry run python scripts/run_us049_system_eval.py --seeds 0
    poetry run python scripts/run_us049_system_eval.py --variants gemini qwen36-vl
    # US-081 AC2/AC3 (offline, real posteriors, cero red):
    poetry run python scripts/run_us049_system_eval.py --france12-offline \
        --out reports/agent_bench/us081_grounded_crop_france12.json
    # US-081 live scorecard under france-12 (needs endpoints/creds):
    poetry run python scripts/run_us049_system_eval.py --label-space france-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

# Force UTF-8 on the standard streams BEFORE any logging is configured. On a
# Windows cp1252 console structlog otherwise raises ``UnicodeEncodeError`` while
# emitting accented Spanish answer prose, which the per-case ``except`` records as
# a failed (charmap) case and silently DEPRESSES crop_match -- a measurement
# artifact, not a model error. ``reconfigure`` works even after interpreter start,
# so it fixes runs launched without ``PYTHONUTF8=1``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _build_live_backends(variant_names: Sequence[str]) -> dict[str, object]:
    """Build one live :class:`LLMBackend` per variant name via ``make_backend``.

    Args:
        variant_names: Variant tags to build backends for.

    Returns:
        A ``{variant_name: backend}`` mapping reused for all three evals.
    """
    from ml.agent.backends import make_backend
    from ml.eval.agent_bench import _VARIANTS_BY_NAME

    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional outside the app
        settings = None

    backends: dict[str, object] = {}
    for name in variant_names:
        variant = _VARIANTS_BY_NAME[name]
        backends[name] = make_backend(variant.model, settings)
    return backends


def run_france12_offline(crop_path: Path, out_path: Path, *, seed: int = 0) -> dict[str, object]:
    """Run the grounded-crop eval under ``france-12`` with a deterministic oracle.

    US-081 AC2/AC3. Re-runs ONLY :func:`ml.eval.agent_system_eval.eval_grounded_crop`
    with ``LABEL_SPACE=france-12`` so the REAL ``classify_new_parcel.run`` restricts
    its posterior to the twelve-class champion vocabulary, over the curated cases
    (including the three new resolved crops Spring barley / Winter durum wheat /
    Orchard and the out-of-vocabulary Winter triticale handoff, all carrying REAL
    Voting-3 v2 fold-5 posteriors). The LLM is replaced by a per-case ORACLE backend
    (cero red, cero creds) that always routes to the tool and faithfully names the
    crop the REAL restriction returns, or hedges toward the surfaced
    ``unresolved_candidate`` for the out-of-vocabulary case.

    Args:
        crop_path: Path to ``grounded_crop_cases.jsonl``.
        out_path: Where to dump the JSON scorecard.
        seed: Seed tag for the run.

    Returns:
        The grounded-crop metric mapping (routing, crop_match, faithfulness,
        oov_handoff, n, n_oov) under ``france-12``.
    """
    import asyncio
    from collections.abc import AsyncIterator
    from dataclasses import dataclass, field

    import structlog
    from pytest import MonkeyPatch

    from ml.agent.context import ToolContext
    from ml.eval.agent_bench import ReasonerVariant
    from ml.eval.agent_system_eval import CropCase, eval_grounded_crop, load_crop_cases

    logger = structlog.get_logger("run_france12_offline")

    os.environ["LABEL_SPACE"] = "france-12"

    @dataclass
    class _FC:
        """Duck-typed ``BackendFunctionCall`` (name/args/id)."""

        name: str
        args: dict[str, object]
        id: str | None = None

    @dataclass
    class _Chunk:
        """Duck-typed ``BackendChunk`` (text and/or function_call)."""

        text: str | None = None
        function_call: _FC | None = None

    @dataclass
    class _OracleBackend:
        """Per-case oracle: route to classify, then faithfully name the gold crop.

        Turn 0 emits the ``classify_new_parcel`` call; turn 1 names ``answer_crop``
        (the in-vocab crop the restriction returns, or the unresolved candidate for
        an out-of-vocabulary case). It never inspects the tool result -- the gold
        crop is derived from the case up front -- so it is a deterministic stand-in
        for a perfectly-faithful reasoner, isolating the orchestration + restriction
        + handoff pipeline US-081 changed.
        """

        answer_crop: str
        aoi: dict[str, object]
        model: str = "oracle-france12"
        _turn: int = field(default=0)

        def reset(self) -> None:
            """Rewind the turn cursor (the harness resets before each case)."""
            self._turn = 0

        async def generate_stream(
            self, *, contents: list, tools: list, system_instruction: str
        ) -> AsyncIterator[_Chunk]:
            """Yield the routing call (turn 0) then the faithful answer (turn 1)."""
            index = self._turn
            self._turn += 1
            if index == 0:
                yield _Chunk(function_call=_FC(name="classify_new_parcel", args={"aoi": self.aoi}))
            else:
                yield _Chunk(
                    text=(
                        f"Segun la clasificacion, el cultivo de la parcela es {self.answer_crop}."
                    )
                )

    # Cero red: stub the live GEE sampler to None so a needs-GEE case (whose
    # embedding fetch is also stubbed to None inside eval_grounded_crop) returns the
    # needs_gee sentinel deterministically instead of attempting a real GEE call.
    import ml.agent.tools.classify as classify_mod

    async def _no_gee(_ctx: object, _year: int, _aoi: object) -> None:
        return None

    classify_mod._sample_embedding_via_gee = _no_gee  # type: ignore[assignment]

    cases = load_crop_cases(crop_path)
    variant = ReasonerVariant(name="oracle-france12", model="oracle", multimodal=False)

    def _gold_answer_crop(case: CropCase) -> str:
        """The crop a perfectly faithful reasoner names for this case.

        For an out-of-vocabulary case the gold answer is the unresolved candidate
        (the real dropped crop the model leans to); otherwise it is ``true_crop``
        (the in-vocab argmax the france-12 restriction returns unchanged).
        """
        if case.expects_out_of_vocab:
            return case.expected_unresolved_candidate or case.true_crop
        if case.expects_needs_gee:
            return "muestreo GEE"  # named so the needs-GEE refusal prose has content
        return case.true_crop

    settings_label_space = "france-12"

    def make_ctx(session_id=None, defer=None):
        """Build a ToolContext whose settings pin the france-12 label-space."""
        from uuid import uuid4

        class _S:
            label_space = settings_label_space
            rag_enabled = False

        return ToolContext(
            pool=None,  # type: ignore[arg-type]
            settings=_S(),  # type: ignore[arg-type]
            session_id=session_id or uuid4(),
            defer=defer,
        )

    async def _score_all() -> list[dict[str, float | int]]:
        results: list[dict[str, float | int]] = []
        for case in cases:
            backend = _OracleBackend(answer_crop=_gold_answer_crop(case), aoi=case.aoi_geometry)
            mp = MonkeyPatch()
            try:
                metrics = await eval_grounded_crop(
                    variant,
                    [case],
                    backend=backend,  # type: ignore[arg-type]
                    make_ctx=make_ctx,
                    monkeypatch_target=mp,
                    seed=seed,
                )
            finally:
                mp.undo()
            metrics["case_id"] = case.id  # type: ignore[assignment]
            results.append(metrics)
        return results

    per_case = asyncio.run(_score_all())

    # Aggregate the per-case single-item runs into one scorecard. Each per-case
    # metric is 1.0/0.0 (single item) or NaN (metric not applicable to the case).
    def _mean(key: str) -> float:
        vals = [
            float(m[key])
            for m in per_case
            if key in m and not (isinstance(m[key], float) and m[key] != m[key])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    n_total = len(cases)
    n_oov = sum(1 for c in cases if c.expects_out_of_vocab)
    n_new = sum(
        1 for c in cases if c.true_crop in ("Spring barley", "Winter durum wheat", "Orchard")
    )
    scorecard = {
        "label_space": "france-12",
        "mode": "offline-oracle",
        "n": n_total,
        "n_out_of_vocab": n_oov,
        "n_new_france12_crops": n_new,
        "routing_accuracy": round(_mean("routing_accuracy"), 4),
        "crop_match_accuracy": round(_mean("crop_match_accuracy"), 4),
        "faithfulness_crop": round(_mean("faithfulness_crop"), 4),
        "oov_handoff_accuracy": round(_mean("oov_handoff_accuracy"), 4),
        "per_case": per_case,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "france12_offline_scorecard_written",
        path=str(out_path),
        routing=scorecard["routing_accuracy"],
        crop_match=scorecard["crop_match_accuracy"],
        oov_handoff=scorecard["oov_handoff_accuracy"],
    )
    return scorecard


def _build_judge() -> object | None:
    """Build a live hallucination judge backed by Gemini, or ``None`` on failure.

    Returns:
        A judge object exposing ``score(sample) -> float`` in ``[0, 1]``, or
        ``None`` when no judge can be built (then RAG hallucination is NaN).
    """
    try:
        from ml.eval.agent_metrics import build_gemini_judge

        return build_gemini_judge()
    except Exception:  # noqa: BLE001 - judge optional; NaN is the honest fallback
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the project-grounded agent eval.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the live US-049 system eval.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["gemini", "qwen", "gemma-base", "qwen36-vl"],
        help="Variantes a evaluar (por defecto las cuatro).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--qwen-vl-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/agent_bench/us049_system_eval.json"),
        help="Ruta del JSON de resultados.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="No usar juez de alucinacion (RAG hallucination queda NaN).",
    )
    parser.add_argument(
        "--label-space",
        default=None,
        help=(
            "Label-space para la eval live (france-9 / france-12). Fija LABEL_SPACE "
            "para que classify.run restrinja a ese vocabulario."
        ),
    )
    parser.add_argument(
        "--france12-offline",
        action="store_true",
        help=(
            "US-081 AC2/AC3: corre SOLO grounded_crop bajo france-12 con un oraculo "
            "deterministico (cero red, cero creds) sobre los casos con posteriores "
            "reales del campeon v2."
        ),
    )
    parser.add_argument(
        "--crop-cases",
        type=Path,
        default=Path("data/agent_eval/grounded_crop_cases.jsonl"),
        help="Ruta del JSONL de casos grounded_crop.",
    )
    args = parser.parse_args(argv)

    if args.france12_offline:
        import structlog

        out = args.out
        if out == Path("reports/agent_bench/us049_system_eval.json"):
            out = Path("reports/agent_bench/us081_grounded_crop_france12.json")
        scorecard = run_france12_offline(args.crop_cases, out, seed=args.seeds[0])
        structlog.get_logger("run_us049_system_eval").info("france12_offline_done", out=str(out))
        print(json.dumps(scorecard, ensure_ascii=False, indent=2))
        return 0

    if args.label_space:
        os.environ["LABEL_SPACE"] = args.label_space

    os.environ["VLLM_QWEN35_URL"] = args.qwen_url
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    os.environ["QWEN36_VL_URL"] = args.qwen_vl_url
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    import structlog
    from pytest import MonkeyPatch

    from ml.agent.context import ToolContext
    from ml.eval.agent_bench import _VARIANTS_BY_NAME
    from ml.eval.agent_system_eval import run_system_eval

    logger = structlog.get_logger("run_us049_system_eval")

    variants = [_VARIANTS_BY_NAME[name] for name in args.variants]
    backends = _build_live_backends(args.variants)

    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional outside the app
        settings = None

    def make_ctx(session_id=None, defer=None):
        """Build a live ToolContext (DB pool unused for the stubbed crop eval)."""
        from uuid import uuid4

        return ToolContext(
            pool=None,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            session_id=session_id or uuid4(),
            defer=defer,
        )

    judge = None if args.no_judge else _build_judge()
    monkeypatch = MonkeyPatch()
    try:
        results = run_system_eval(
            variants,
            seeds=tuple(args.seeds),
            toolcall_backends=backends,
            crop_backends=backends,
            rag_backends=backends,
            make_ctx=make_ctx,
            monkeypatch_target=monkeypatch,
            judge=judge,
        )
    finally:
        monkeypatch.undo()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("system_eval_written", path=str(args.out), variants=args.variants)
    # Compact summary to stdout.
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
