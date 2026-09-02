"""Tests for the AgroMind-IT/ES benchmark scaffold (US-068).

Exercises the schema (round-trip, the ten copilot families, the verifiable
AgroMind compatibility bridge, the eval-only train guard), the seed generator in
dry-run AND with a mocked Gemini client (no network), and the Zenodo metadata
builder -- all over the tiny repo-committed fixture, never a fabricated 500-pair
set (Arthur's rule). Every external boundary (Gemini, GEE, Streamlit) is mocked
or import-only.

Conventions: identifiers / docstrings in English; no network; full type hints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.eval.agent_bench import AgroMindItem
from ml.eval.agromind_it_es.generate_seed import (
    SeedGenerator,
    build_generation_prompt,
)
from ml.eval.agromind_it_es.schema import (
    ALLOWED_LANGUAGES,
    QAItem,
    QuestionFamily,
    SchemaValidationError,
    dump_jsonl,
    load_jsonl,
    to_agromind_item,
    validate_record,
)
from ml.eval.agromind_it_es.zenodo_metadata import (
    DATASET_LICENSE,
    build_zenodo_metadata,
)

#: The committed fixture (3 structure-example pairs, source=fixture). NOT the
#: benchmark; just enough to exercise the loader and the bridge.
_FIXTURE: Path = Path("data/benchmark/agromind_it_es/seed.fixture.jsonl")


def _sample_item() -> QAItem:
    """Return one well-formed multiple-choice QAItem for tests.

    Returns:
        A multiple-choice :class:`QAItem` with derived flags filled in.
    """
    return QAItem(
        item_id="it-classification-0001",
        category=QuestionFamily.CLASSIFICATION,
        lang="it",
        question="Quale coltura predomina nella parcella centrale?",
        options={"A": "Mais", "B": "Riso", "C": "Vigneto", "D": "Uliveto"},
        answer="B",
        image="it/classification/po_valley_0001.png",
        source="fixture",
    ).with_derived_flags()


def test_ten_families_enum() -> None:
    """The enum has exactly the ten copilot question families of the AC."""
    expected = {
        "classification",
        "quantification",
        "vigor",
        "water_stress",
        "phenology",
        "comparison",
        "anomaly",
        "metadata",
        "intersection",
        "explainability",
    }
    assert {f.value for f in QuestionFamily} == expected
    assert len(list(QuestionFamily)) == 10


def test_schema_roundtrip() -> None:
    """QAItem -> JSONL -> QAItem is idempotent over the record fields."""
    item = _sample_item()
    record = item.to_record()
    rebuilt = validate_record(record)
    assert rebuilt.item_id == item.item_id
    assert rebuilt.category is QuestionFamily.CLASSIFICATION
    assert rebuilt.lang == "it"
    assert rebuilt.options == item.options
    assert rebuilt.answer == "B"
    assert rebuilt.is_multimodal is True


def test_dump_load_roundtrip(tmp_path: Path) -> None:
    """dump_jsonl then load_jsonl preserves the items in order."""
    items = [_sample_item()]
    out = tmp_path / "seed.jsonl"
    assert dump_jsonl(items, out) == 1
    loaded = load_jsonl(out)
    assert len(loaded) == 1
    assert loaded[0].item_id == items[0].item_id


def test_fixture_loads_and_is_eval_only() -> None:
    """The committed fixture loads cleanly and carries no train mark."""
    if not _FIXTURE.exists():
        pytest.skip("fixture not materialised in this checkout")
    items = load_jsonl(_FIXTURE)
    assert len(items) >= 1
    assert all(item.source == "fixture" for item in items)
    assert all(item.lang in ALLOWED_LANGUAGES for item in items)


def test_agromind_compat() -> None:
    """to_agromind_item builds a valid AgroMindItem (compat verified, not declared)."""
    item = _sample_item()
    bridged = to_agromind_item(item)
    assert isinstance(bridged, AgroMindItem)
    assert bridged.question == item.question
    assert bridged.options == item.options
    assert bridged.answer == "B"
    assert bridged.is_multimodal is True
    assert bridged.task_file.endswith("/it")


def test_agromind_compat_open_item() -> None:
    """An open (no-options) item also bridges to a valid AgroMindItem."""
    item = QAItem(
        item_id="es-quantification-0001",
        category=QuestionFamily.QUANTIFICATION,
        lang="es",
        question="Cuantas parcelas hay?",
        options={},
        answer="4",
        source="fixture",
    ).with_derived_flags()
    bridged = to_agromind_item(item)
    assert isinstance(bridged, AgroMindItem)
    assert bridged.options == {}
    assert bridged.answer == "4"


def test_eval_only_guard_split_train() -> None:
    """A record with split=train is rejected by the validator."""
    record = _sample_item().to_record()
    record["split"] = "train"
    with pytest.raises(SchemaValidationError, match="eval-only"):
        validate_record(record)


def test_eval_only_guard_is_train_flag() -> None:
    """A record with is_train=true is rejected by the validator."""
    record = _sample_item().to_record()
    record["is_train"] = True
    with pytest.raises(SchemaValidationError, match="eval-only"):
        validate_record(record)


def test_eval_only_guard_loader(tmp_path: Path) -> None:
    """load_jsonl rejects a file that smuggles a train-marked record."""
    record = _sample_item().to_record()
    record["split"] = "training"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="eval-only"):
        load_jsonl(bad)


def test_validate_rejects_unknown_family() -> None:
    """An unknown category value is rejected."""
    record = _sample_item().to_record()
    record["category"] = "not_a_family"
    with pytest.raises(SchemaValidationError, match="unknown category"):
        validate_record(record)


def test_validate_rejects_unknown_lang() -> None:
    """An unsupported language is rejected (benchmark is it/es only)."""
    record = _sample_item().to_record()
    record["lang"] = "fr"
    with pytest.raises(SchemaValidationError, match="unknown lang"):
        validate_record(record)


def test_generation_prompt_covers_family_and_language() -> None:
    """The generation prompt mentions the family and is built per language."""
    prompt_it = build_generation_prompt(QuestionFamily.VIGOR, "it")
    prompt_es = build_generation_prompt(QuestionFamily.VIGOR, "es")
    assert "vigor" in prompt_it
    assert "italiano" in prompt_it.lower()
    assert "espanol" in prompt_es.lower()
    assert prompt_it != prompt_es


def test_generation_prompt_rejects_bad_language() -> None:
    """An unsupported language raises in the prompt builder."""
    with pytest.raises(ValueError, match="unsupported language"):
        build_generation_prompt(QuestionFamily.VIGOR, "de")


def test_generator_dry_run_no_api() -> None:
    """Dry-run emits N items per family x language, never calling the API."""
    generator = SeedGenerator(client=None)  # no client -> dry-run forced
    items = generator.generate(QuestionFamily.PHENOLOGY, "es", 3, dry_run=True)
    assert len(items) == 3
    assert all(it.source == "dry-run" for it in items)
    assert all(it.category is QuestionFamily.PHENOLOGY for it in items)
    assert all(it.lang == "es" for it in items)
    assert all(it.item_id.startswith("es-phenology-") for it in items)


def test_generator_full_plan_covers_grid() -> None:
    """The plan covers the full family x language grid with the target count."""
    generator = SeedGenerator(client=None)
    items = generator.generate_plan(list(QuestionFamily), ["it", "es"], 2, dry_run=True)
    # 10 families * 2 languages * 2 per cell.
    assert len(items) == 40
    families_seen = {it.category for it in items}
    assert families_seen == set(QuestionFamily)
    assert {it.lang for it in items} == {"it", "es"}


class _FakeModels:
    """Stub of ``client.models`` returning a fixed structured JSON answer."""

    def generate_content(self, *, model: str, contents: Any) -> Any:
        """Return an object with a ``.text`` JSON payload (mock Gemini)."""

        class _Resp:
            text = json.dumps(
                {
                    "question": "Quale coltura?",
                    "options": {"A": "Mais", "B": "Riso"},
                    "answer": "B",
                }
            )

        return _Resp()


class _FakeGeminiClient:
    """Minimal ``google.genai.Client`` double exposing ``.models``."""

    def __init__(self) -> None:
        self.models = _FakeModels()


def test_generator_real_path_with_mocked_client() -> None:
    """The real path parses a mocked Gemini JSON answer into a QAItem."""
    generator = SeedGenerator(client=_FakeGeminiClient())
    items = generator.generate(QuestionFamily.CLASSIFICATION, "it", 1, dry_run=False)
    assert len(items) == 1
    item = items[0]
    assert item.source == "gemini-seed"
    assert item.question == "Quale coltura?"
    assert item.answer == "B"
    assert item.options == {"A": "Mais", "B": "Riso"}
    # And the produced item bridges to a valid AgroMindItem.
    assert isinstance(to_agromind_item(item), AgroMindItem)


def test_zenodo_metadata_valid() -> None:
    """build_zenodo_metadata emits CC-BY-4.0 + an eval-only description."""
    payload = build_zenodo_metadata()
    meta = payload["metadata"]
    assert meta["license"] == DATASET_LICENSE == "cc-by-4.0"
    assert meta["upload_type"] == "dataset"
    assert "eval-only" in meta["description"].lower()
    assert meta["creators"]
    assert "Sentinel-2" in " ".join(meta["keywords"]) or "Sentinel-2" in meta["description"]


def test_zenodo_metadata_write(tmp_path: Path) -> None:
    """write_zenodo_metadata produces a parseable .zenodo.json file."""
    from ml.eval.agromind_it_es.zenodo_metadata import write_zenodo_metadata

    out = tmp_path / ".zenodo.json"
    write_zenodo_metadata(out, version="1.2.3")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["version"] == "1.2.3"


def test_review_app_imports() -> None:
    """The review app module imports without streamlit installed (smoke)."""
    import ml.eval.agromind_it_es.review_app as review_app

    assert hasattr(review_app, "run_app")
    assert hasattr(review_app, "accept_item")
    # accept_item is pure (no streamlit): it marks the item human-edited.
    accepted = review_app.accept_item(_sample_item(), reviewer="tester")
    assert accepted.reviewed is True
    assert accepted.reviewer == "tester"
    assert accepted.source == "human-edited"
