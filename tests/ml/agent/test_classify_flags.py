"""``classify_new_parcel`` flag-behaviour tests (US-053).

The honest classifier exposes two independent flags on
:class:`ml.agent.schemas.ClassifyParcelInput`. This module pins their *behaviour*
end to end through ``classify.run`` (the DB / OOF / PASTIS-R boundaries are
mocked, never hit):

- ``restrict_to_resolved_classes`` (default ON): the 18-class posterior is masked
  down to the nine ``france-9`` classes and renormalized so the surfaced posterior
  sums to 1 over exactly nine classes; turning it OFF surfaces the full 18.
- ``use_stacking`` (default OFF): when a Stacking-5 posterior is available the
  meta-learner output is served (the ``xgb-alphaearth`` fallback is NOT invoked);
  when no OOF row matches OR the OOF artifacts are missing the tool degrades
  cleanly to ``xgb-alphaearth`` with a structured warning and never crashes.
- the label-space is a *registry* lookup: a custom space registered at runtime
  (the EPIC 12 US-074 seam) is honoured by ``run`` without editing ``classify``.

Determinism: the meta / xgb posteriors are fixed synthetic vectors (or a real
OOF row when the parquet is present), never random; no GPU, GEE, DVC or live DB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ml.agent.tools.classify as classify_mod
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput
from ml.eval.class_remap import (
    HARNESS_NUM_CLASSES,
    LabelSpace,
    get_label_space,
    restrict_posterior,
)

from .conftest import SESSION_A

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"
_REAL_PARCEL = "10003_1103071"
_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
}


def _deterministic_posterior() -> np.ndarray:
    """A fixed 18-class post-softmax row with mass on kept AND dropped classes.

    Prefers a real ``xgb-alphaearth`` OOF row when the parquet is present (keeps
    the test grounded in genuine probabilities); otherwise falls back to a hand-
    built vector that still spans kept (france-9) and dropped ids so the
    mask/renorm assertions remain meaningful without any DVC pull.
    """
    path = _OOF_DIR / "oof_parcel_xgb-alphaearth_fold5.parquet"
    if path.exists():
        import polars as pl

        from ml.utils.parcel_id import canonical_parcel_id
        from ml.utils.parcel_reconcile import PROB_COLUMNS

        frame = canonical_parcel_id(pl.read_parquet(path), col="canonical_parcel_id")
        row = frame.filter(pl.col("canonical_parcel_id") == _REAL_PARCEL)
        if row.height:
            return row.select(PROB_COLUMNS).to_numpy().astype(np.float64)[0]

    # Deterministic synthetic fallback (sums to 1, spans kept + dropped ids).
    proba = np.full(HARNESS_NUM_CLASSES, 0.01, dtype=np.float64)
    proba[2] = 0.40  # Corn (kept)
    proba[1] = 0.20  # Soft winter wheat (kept)
    proba[0] = 0.10  # Meadow (kept)
    proba[5] = 0.13  # Spring barley (dropped)
    proba[9] = 0.01  # Winter triticale (dropped)
    return proba / proba.sum()


class _FakeClassifier:
    """``_XgbAlphaEarthClassifier`` stand-in returning a fixed posterior."""

    def __init__(self, proba: np.ndarray, class_names: dict[int, str]) -> None:
        self._proba = proba
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        return self._proba


def _patch_embedding(monkeypatch) -> None:
    """Make the embedding fetch succeed with a deterministic 64-dim vector."""

    async def _fake_fetch(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch)


# ---------------------------------------------------------------------------
# restrict_to_resolved_classes
# ---------------------------------------------------------------------------
async def test_restrict_on_reduces_to_nine_and_renormalizes(monkeypatch, make_ctx) -> None:
    """Explicit france-9 (restrict ON) collapses 18 -> 9 classes summing to ~1."""
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            label_space="france-9",
            model="xgb",
        ),
        make_ctx(),
    )

    space = get_label_space("france-9")
    assert isinstance(out, ClassificationResult)
    # Exactly the nine resolved classes, renormalized to ~1.
    assert len(out.class_probabilities) == 9
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # The surfaced labels are the real france-9 crop names (not class_xx).
    assert set(out.class_probabilities) == set(space.class_names.values())
    # The headline class matches the restricted argmax + its confidence.
    expected = restrict_posterior(proba, space)
    top_cid = max(expected, key=lambda c: expected[c])
    assert out.crop_class == space.class_names[top_cid]
    assert out.confidence == pytest.approx(float(expected[top_cid]))


async def test_restrict_off_surfaces_full_eighteen(monkeypatch, make_ctx) -> None:
    """``restrict_to_resolved_classes=False`` surfaces the full 18-class posterior."""
    proba = _deterministic_posterior()
    expected_idx = int(np.argmax(proba))
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            restrict_to_resolved_classes=False,
            model="xgb",
        ),
        make_ctx(),
    )

    assert len(out.class_probabilities) == 18
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert out.crop_class == f"class_{expected_idx}"
    assert out.confidence == pytest.approx(float(proba[expected_idx]))


async def test_restrict_on_zero_kept_mass_is_unresolved(monkeypatch, make_ctx) -> None:
    """All mass on dropped classes -> ``crop_class='unresolved'`` confidence 0.0."""
    # Put every unit of mass on dropped ids (5, 9, 10 are all dropped by france-9).
    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[5] = 0.5
    proba[9] = 0.5
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            label_space="france-9",
            model="xgb",
        ),
        make_ctx(),
    )

    assert out.crop_class == "unresolved"
    assert out.confidence == 0.0
    # The nine kept ids are still surfaced, all at zero (honest "no signal").
    assert all(v == 0.0 for v in out.class_probabilities.values())


async def test_out_of_vocabulary_handoff_is_flagged(monkeypatch, make_ctx) -> None:
    """A raw argmax OUTSIDE the vocabulary flags the out-of-vocab handoff (not a wall).

    france-12 drops Potatoes (id 12). A posterior whose RAW top class is Potatoes
    still reports a restricted in-vocabulary ``crop_class`` (the renormalized
    argmax) BUT also carries ``unresolved_candidate="Potatoes"`` and the dropped
    crop names in ``out_of_vocabulary_classes`` -- the cue the reasoner uses to
    hedge with RAG/phenology instead of trusting the headline. A raw top class IN
    vocabulary leaves ``unresolved_candidate`` ``None``.
    """
    # Raw argmax on Potatoes (12, out-of-vocab); the only kept mass is Corn (2).
    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[12] = 0.6
    proba[2] = 0.4
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019, model="xgb"),
        make_ctx(),
    )
    assert out.unresolved_candidate == "Potatoes"
    assert "Potatoes" in out.out_of_vocabulary_classes
    assert len(out.out_of_vocabulary_classes) == 6
    # The in-vocab headline is the restricted argmax (Corn), not the raw Potatoes.
    assert out.crop_class == "Corn"

    # A raw top class IN vocabulary -> no handoff flag.
    proba_ok = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba_ok[2] = 0.9
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba_ok, class_names)
    )
    out_ok = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019, model="xgb"),
        make_ctx(),
    )
    assert out_ok.unresolved_candidate is None
    assert out_ok.crop_class == "Corn"


# ---------------------------------------------------------------------------
# use_stacking
# ---------------------------------------------------------------------------
async def test_use_stacking_serves_meta_posterior(monkeypatch, make_ctx) -> None:
    """Legacy ``model="xgb" + use_stacking=True`` serves the Stacking-5 posterior.

    Since US-081 flipped the default ``model`` to ``"voting3"``, the legacy
    ``use_stacking`` promotion to Stacking-5 fires ONLY when ``model`` is set
    explicitly to ``"xgb"`` (see :attr:`ClassifyParcelInput.resolved_model`); this
    test pins that legacy path. The Stacking-5 branch is stubbed to return a
    deterministic meta posterior (no PASTIS-R / OOF I/O). The xgb fallback is wired
    to fail loudly if called, so a passing test proves the stacking branch
    produced the result.
    """
    meta_proba = _deterministic_posterior()
    _patch_embedding(monkeypatch)

    async def _fake_stacking(ctx, inp):
        return meta_proba

    monkeypatch.setattr(classify_mod, "_stacking_posterior", _fake_stacking)
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: _FakeClassifier(np.full(18, np.nan), {i: f"class_{i}" for i in range(18)}),
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            model="xgb",
            use_stacking=True,
            label_space="france-9",
        ),
        make_ctx(),
    )

    space = get_label_space("france-9")
    expected = restrict_posterior(meta_proba, space)
    top_cid = max(expected, key=lambda c: expected[c])
    # The reported class/confidence come from the META posterior, not the NaN xgb.
    assert out.crop_class == space.class_names[top_cid]
    assert out.confidence == pytest.approx(float(expected[top_cid]))
    assert not np.isnan(out.confidence)


async def test_use_stacking_degrades_to_xgb_without_oof(monkeypatch, make_ctx, capsys) -> None:
    """No OOF artifacts -> clean fallback to xgb-alphaearth, no crash, warned.

    ``_resolve_canonical_parcel_id`` resolves an id but ``_load_stacking_five``
    raises ``FileNotFoundError`` (DVC not pulled). ``run`` must serve the xgb
    posterior and emit ``classify_stacking_unavailable`` rather than propagating.
    The warning is structlog-rendered to stdout, captured here via ``capsys``.
    """
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)

    async def _fake_resolve(ctx, aoi):
        return _REAL_PARCEL

    def _raise_missing_oof():
        raise FileNotFoundError("Stacking-5 OOF parquet missing (dvc not pulled).")

    monkeypatch.setattr(classify_mod, "_resolve_canonical_parcel_id", _fake_resolve)
    monkeypatch.setattr(classify_mod, "_load_stacking_five", _raise_missing_oof)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            model="xgb",
            use_stacking=True,
            restrict_to_resolved_classes=False,
        ),
        make_ctx(),
    )

    # Fell back to xgb (full 18 since restrict is off), did not crash.
    assert isinstance(out, ClassificationResult)
    assert len(out.class_probabilities) == 18
    assert out.crop_class == f"class_{int(np.argmax(proba))}"
    # The structured degradation warning was emitted (AC-8).
    assert "classify_stacking_unavailable" in capsys.readouterr().out


async def test_use_stacking_degrades_when_parcel_not_in_oof(monkeypatch, make_ctx) -> None:
    """A new polygon (no OOF row) degrades to xgb without raising.

    ``_resolve_canonical_parcel_id`` returns ``None`` (the AOI maps to no persisted
    parcel), so ``_stacking_posterior`` returns ``None`` and ``run`` uses xgb.
    """
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)

    async def _resolve_none(ctx, aoi):
        return None

    def _stacking_must_not_load():
        pytest.fail("Stacking-5 must not load when no parcel resolves to the OOF.")

    monkeypatch.setattr(classify_mod, "_resolve_canonical_parcel_id", _resolve_none)
    monkeypatch.setattr(classify_mod, "_load_stacking_five", _stacking_must_not_load)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            model="xgb",
            use_stacking=True,
            restrict_to_resolved_classes=False,
        ),
        make_ctx(),
    )

    assert isinstance(out, ClassificationResult)
    assert out.crop_class == f"class_{int(np.argmax(proba))}"


async def test_default_model_is_voting3_degrading_to_xgb(monkeypatch, make_ctx) -> None:
    """Default ``model`` is ``"voting3"`` (US-081 AC4a), degrading to xgb on a fresh AOI.

    With every flag at its default, ``run`` dispatches to the Voting-3 champion
    branch. For an AOI that resolves to no fold-5 OOF parcel (``_voting_posterior``
    returns ``None``) it degrades CLEANLY to the ``xgb-alphaearth`` posterior --
    the exact safe behaviour the historical ``xgb`` default had -- restricted to
    the configured ``DEFAULT_LABEL_SPACE`` (``france-12``, twelve classes). The
    legacy stacking branch must NOT be consulted (``use_stacking`` defaults OFF).
    """
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)

    async def _stacking_must_not_run(ctx, inp):
        pytest.fail("stacking branch must not run when use_stacking defaults OFF")

    # The default model is voting3; the AOI maps to no persisted fold-5 parcel, so
    # the vote degrades to xgb (returns None) without touching the DB pool.
    async def _voting_degrades(ctx, inp):
        return None

    monkeypatch.setattr(classify_mod, "_stacking_posterior", _stacking_must_not_run)
    monkeypatch.setattr(classify_mod, "_voting_posterior", _voting_degrades)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    inp = ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019)
    assert inp.model == "voting3"
    assert inp.resolved_model == "voting3"

    out = await classify_mod.run(inp, make_ctx())

    assert len(out.class_probabilities) == 12
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# label-space registry is honoured by run() (EPIC 12 US-074 extensibility)
# ---------------------------------------------------------------------------
async def test_run_honours_registered_custom_label_space(monkeypatch, make_ctx) -> None:
    """A custom space registered at runtime gates ``run`` without editing classify.

    Registers a two-class space and asks ``run`` to use it: the surfaced posterior
    must carry exactly those two classes, proving the classifier resolves the
    active space purely by name (the EPIC 12 US-074 seam).
    """
    from ml.agent.schemas import ClassifyParcelInput as _Input  # noqa: F401  (clarity)
    from ml.eval import class_remap as cr

    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[2] = 0.6  # Corn
    proba[7] = 0.2  # Grapevine
    proba[5] = 0.2  # Spring barley (outside the custom space -> dropped)
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    custom = LabelSpace(
        name="corn-grape-2",
        kept_class_ids=(2, 7),
        dropped_class_ids=tuple(c for c in range(HARNESS_NUM_CLASSES) if c not in (2, 7)),
        class_names={2: "Corn", 7: "Grapevine"},
        source="unit-test custom space (EPIC 12 US-074 seam)",
    )
    cr.register_label_space(custom)
    try:
        out = await classify_mod.run(
            ClassifyParcelInput(
                session_id=SESSION_A,
                aoi=_POLYGON,
                year=2019,
                label_space="corn-grape-2",
                model="xgb",
            ),
            make_ctx(),
        )
    finally:
        cr._REGISTRY.pop("corn-grape-2", None)

    assert set(out.class_probabilities) == {"Corn", "Grapevine"}
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # Corn dominates (0.6 vs 0.2) -> it is the headline class.
    assert out.crop_class == "Corn"


async def test_run_unknown_label_space_raises_keyerror(monkeypatch, make_ctx) -> None:
    """An unregistered ``label_space`` name fails fast with ``KeyError``."""
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier must not load for an unknown label-space"),
    )

    with pytest.raises(KeyError):
        await classify_mod.run(
            ClassifyParcelInput(
                session_id=SESSION_A,
                aoi=_POLYGON,
                year=2019,
                label_space="does-not-exist",
            ),
            make_ctx(),
        )


# ---------------------------------------------------------------------------
# ctx.crop_model: the user's UI pin is a CONTRACT, not a hint
# ---------------------------------------------------------------------------
async def test_user_pin_overrides_the_model_the_reasoner_asked_for(monkeypatch, make_ctx) -> None:
    """``ctx.crop_model`` wins over ``inp.model``: the UI switch is a hard choice.

    The crop-model selector promises the user a specific model. Before this was
    enforced at the tool boundary it was merely *requested* from the reasoner via a
    system instruction, so an LLM that ignored the instruction silently served a
    different model than the one the user picked. Here the reasoner asks for
    ``voting3`` while the user pinned ``xgb``: the tabular member must serve.
    """
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    # Voting-3 WOULD serve happily here: if the pin were ignored, `member` becomes
    # "voting-3". This is what makes the assertion below non-vacuous (a parcel that
    # simply failed to resolve would degrade to xgb-alphaearth for unrelated
    # reasons and the test would pass without proving anything).
    calls: list[str] = []

    async def _voting_would_serve(ctx, inp):
        calls.append("voting")
        return _deterministic_posterior()

    monkeypatch.setattr(classify_mod, "_voting_posterior", _voting_would_serve)

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            model="voting3",  # what the REASONER asked for
        ),
        make_ctx(crop_model="xgb"),  # what the USER pinned -> must win
    )

    assert out.served_model == "xgb-alphaearth"
    assert calls == [], "voting-3 was consulted despite the user pinning xgb"


async def test_no_user_pin_leaves_the_reasoner_argument_alone(monkeypatch, make_ctx) -> None:
    """With no pin (``ctx.crop_model=None``) the reasoner's ``model`` still stands.

    The override is scoped to an explicit user choice; it must not hijack the
    normal path where the reasoner (or the ``voting3`` default) selects the model.
    """
    proba = _deterministic_posterior()
    class_names = {i: f"class_{i}" for i in range(HARNESS_NUM_CLASSES)}
    _patch_embedding(monkeypatch)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    async def _voting_would_serve(ctx, inp):
        return _deterministic_posterior()

    monkeypatch.setattr(classify_mod, "_voting_posterior", _voting_would_serve)

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            model="voting3",  # the reasoner's choice, with no user pin to override it
        ),
        make_ctx(),  # no pin
    )

    assert out.served_model == "voting-3"
