"""Label-space registry tests living in ``ml.eval.class_remap`` (US-053).

The honest classifier (``ml.agent.tools.classify``) does not hardcode WHICH of
the 18 semantic classes it trusts: it asks a *label-space* registry. This module
proves that registry, which lives in :mod:`ml.eval.class_remap`, is correct and
EPIC 12 US-074-extensible:

- ``france-9`` keeps exactly the nine well-resolved semantic18 ids (F1 OOF
  fold-5) and resolves their crop names against ``SEMANTIC18_CLASS_NAMES`` (by
  id, never by free-form string) so a future rename of the name table can never
  silently drift the ids.
- ``restrict_posterior`` masks the dropped-class mass and renormalizes over the
  kept ids: the result sums to 1 (or is an honest all-zero when no mass lands on
  the resolved classes), and the kept ids keep their relative proportions.
- the registry is extensible: registering a brand-new ``LabelSpace`` (the EPIC 12
  US-074 HCAT crosswalk seam) does NOT mutate ``france-9`` and requires no change
  to the classifier; duplicate / empty / out-of-range registrations are rejected.

Pure computation: no checkpoints, no DVC, no DB -- deterministic synthetic
posteriors only. These tests are intentionally focused on the *registry* contract
(``tests/ml/agent/test_class_remap.py`` covers the same module from the agent
side; here the lens is the label-space abstraction the classifier depends on).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
from ml.eval.class_remap import (
    DEFAULT_LABEL_SPACE,
    FRANCE_9,
    FRANCE_12,
    HARNESS_NUM_CLASSES,
    LabelSpace,
    get_label_space,
    list_label_spaces,
    register_label_space,
    restrict_posterior,
)

#: The nine ``france-9`` crop names (published F1 OOF fold-5, notebook 06c).
_FRANCE_9_NAMES: frozenset[str] = frozenset(
    {
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Soybeans",
    }
)

#: The nine semantic18 ids ``france-9`` keeps (verified against the F1 OOF curve).
_FRANCE_9_KEPT: frozenset[int] = frozenset({0, 1, 2, 3, 4, 6, 7, 8, 14})


# ---------------------------------------------------------------------------
# france-9: the canonical 9-class space the classifier defaults to
# ---------------------------------------------------------------------------
def test_france9_has_nine_kept_ids() -> None:
    """``france-9`` keeps exactly the nine well-resolved semantic18 ids."""
    space = get_label_space("france-9")
    assert space is FRANCE_9
    assert space.name == "france-9"
    assert len(space.kept_class_ids) == 9
    assert frozenset(space.kept_class_ids) == _FRANCE_9_KEPT
    assert all(0 <= cid < HARNESS_NUM_CLASSES for cid in space.kept_class_ids)


def test_france9_kept_and_dropped_partition_the_18_space() -> None:
    """Kept + dropped ids partition ``[0, 18)`` with no overlap or gap."""
    space = get_label_space("france-9")
    kept = set(space.kept_class_ids)
    dropped = set(space.dropped_class_ids)
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(range(HARNESS_NUM_CLASSES))
    assert len(dropped) == 9


def test_france9_names_resolved_by_id_not_hardcoded() -> None:
    """The nine kept names come from ``SEMANTIC18_CLASS_NAMES`` by id (no drift)."""
    space = get_label_space("france-9")
    assert len(space.class_names) == 9
    for cid, name in space.class_names.items():
        assert SEMANTIC18_CLASS_NAMES[cid] == name
    assert frozenset(space.class_names.values()) == _FRANCE_9_NAMES


def test_get_label_space_default_is_france12() -> None:
    """Calling ``get_label_space()`` with no name returns the champion-v2 default.

    The default label space was promoted to ``france-12`` in commit ``8dd4335``
    when the copilot was re-pointed at the Voting-3 v2 champion (12 classes,
    macro-F1 0.9001 at 12 per ``reports/voting_new/cardinalidad.json``). US-081
    consolidates that default; ``DEFAULT_LABEL_SPACE`` is the single source of
    truth (no hardcoded ``france-9``/``france-12`` at call sites).
    """
    assert DEFAULT_LABEL_SPACE == FRANCE_12.name
    assert get_label_space() is FRANCE_12


def test_get_label_space_unknown_raises_keyerror() -> None:
    """An unregistered name raises ``KeyError`` (never a silent fallback)."""
    with pytest.raises(KeyError):
        get_label_space("hcat-global-20")


# ---------------------------------------------------------------------------
# restrict_posterior: mask + renormalization correctness
# ---------------------------------------------------------------------------
def test_restrict_posterior_masks_and_sums_to_one() -> None:
    """Mass on dropped classes is discarded; kept ids renormalize to ~1."""
    space = get_label_space("france-9")
    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[0] = 0.30  # Meadow (kept)
    proba[2] = 0.20  # Corn (kept)
    proba[5] = 0.50  # Spring barley (dropped)

    restricted = restrict_posterior(proba, space)

    assert set(restricted) == set(space.kept_class_ids)
    assert sum(restricted.values()) == pytest.approx(1.0, abs=1e-9)
    # Only the kept ids carry mass; the dropped half is absent and gone.
    assert restricted[0] == pytest.approx(0.30 / 0.50)
    assert restricted[2] == pytest.approx(0.20 / 0.50)
    assert all(restricted[cid] == 0.0 for cid in space.kept_class_ids if cid not in (0, 2))


def test_restrict_posterior_preserves_relative_proportions() -> None:
    """Renormalization rescales but never reorders the kept-class probabilities."""
    space = get_label_space("france-9")
    rng = np.random.default_rng(53)
    proba = rng.random(HARNESS_NUM_CLASSES)
    proba = proba / proba.sum()  # a genuine post-softmax row

    restricted = restrict_posterior(proba, space)

    assert sum(restricted.values()) == pytest.approx(1.0, abs=1e-9)
    # Ratio between any two kept ids is unchanged by masking + renormalizing.
    a, b = space.kept_class_ids[0], space.kept_class_ids[1]
    assert restricted[a] / restricted[b] == pytest.approx(proba[a] / proba[b])


def test_restrict_posterior_zero_mass_is_honest_not_uniform() -> None:
    """No mass on the resolved classes -> every kept id is 0.0 (no fake prior)."""
    space = get_label_space("france-9")
    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[5] = 1.0  # Spring barley, a dropped class

    restricted = restrict_posterior(proba, space)

    assert set(restricted) == set(space.kept_class_ids)
    assert all(v == 0.0 for v in restricted.values())
    # Crucially NOT a uniform 1/9 prior -- the model gets to say "no signal".
    assert sum(restricted.values()) == pytest.approx(0.0)


def test_restrict_posterior_rejects_non_18_vector() -> None:
    """A non-(18,) posterior (e.g. a 20-class softmax) is rejected loudly."""
    space = get_label_space("france-9")
    with pytest.raises(ValueError, match="semantic18 posterior"):
        restrict_posterior(np.zeros(20, dtype=np.float64), space)


def test_restrict_posterior_over_an_arbitrary_space() -> None:
    """``restrict_posterior`` honours WHATEVER space it is handed (parametric)."""
    space = LabelSpace(
        name="pair-test",
        kept_class_ids=(2, 7),
        dropped_class_ids=tuple(c for c in range(HARNESS_NUM_CLASSES) if c not in (2, 7)),
        class_names={2: "Corn", 7: "Grapevine"},
        source="unit-test fixture",
    )
    proba = np.zeros(HARNESS_NUM_CLASSES, dtype=np.float64)
    proba[2] = 0.1
    proba[7] = 0.3
    proba[0] = 0.6  # not in the space -> discarded

    restricted = restrict_posterior(proba, space)

    assert set(restricted) == {2, 7}
    assert sum(restricted.values()) == pytest.approx(1.0, abs=1e-9)
    assert restricted[7] == pytest.approx(0.3 / 0.4)


# ---------------------------------------------------------------------------
# EPIC 12 US-074 extensibility: register a new space without touching classify
# ---------------------------------------------------------------------------
def test_register_new_space_is_listed_and_retrievable() -> None:
    """A registered space appears in ``list_label_spaces`` and round-trips."""
    space = LabelSpace(
        name="iberia-stub-5",
        kept_class_ids=(0, 1, 2, 3, 4),
        dropped_class_ids=tuple(c for c in range(HARNESS_NUM_CLASSES) if c >= 5),
        class_names={cid: SEMANTIC18_CLASS_NAMES[cid] for cid in (0, 1, 2, 3, 4)},
        source="EPIC 12 US-074 HCAT crosswalk (stub)",
    )
    try:
        register_label_space(space)
        assert "iberia-stub-5" in list_label_spaces()
        got = get_label_space("iberia-stub-5")
        assert got.kept_class_ids == (0, 1, 2, 3, 4)
        assert got is space
    finally:
        from ml.eval import class_remap as cr

        cr._REGISTRY.pop("iberia-stub-5", None)


def test_register_new_space_does_not_mutate_france9() -> None:
    """The EPIC 12 seam: adding a space leaves ``france-9`` byte-identical."""
    before = get_label_space("france-9")
    before_kept = before.kept_class_ids
    space = LabelSpace(
        name="hcat-stub-20",
        kept_class_ids=tuple(range(HARNESS_NUM_CLASSES)),
        dropped_class_ids=(),
        class_names=dict(SEMANTIC18_CLASS_NAMES),
        source="EPIC 12 US-074 global HCAT (stub)",
    )
    try:
        register_label_space(space)
        # france-9 is the SAME frozen object, with the SAME kept ids.
        assert get_label_space("france-9") is before
        assert get_label_space("france-9").kept_class_ids == before_kept
    finally:
        from ml.eval import class_remap as cr

        cr._REGISTRY.pop("hcat-stub-20", None)


def test_register_duplicate_requires_overwrite() -> None:
    """Re-registering an existing name without ``overwrite`` is rejected."""
    with pytest.raises(ValueError, match="already registered"):
        register_label_space(FRANCE_9)


def test_register_overwrite_replaces_deliberately() -> None:
    """``overwrite=True`` replaces a space on purpose, then can be restored."""
    original = get_label_space("france-9")
    replacement = LabelSpace(
        name="france-9",
        kept_class_ids=(0,),
        dropped_class_ids=tuple(range(1, HARNESS_NUM_CLASSES)),
        class_names={0: "Meadow"},
        source="overwrite test",
    )
    try:
        register_label_space(replacement, overwrite=True)
        assert get_label_space("france-9") is replacement
    finally:
        from ml.eval import class_remap as cr

        cr._REGISTRY["france-9"] = original
    assert get_label_space("france-9") is original


def test_register_rejects_empty_kept_ids() -> None:
    """A space that keeps no class id is rejected (it would mask everything)."""
    with pytest.raises(ValueError, match="at least one class id"):
        register_label_space(
            LabelSpace("empty-9", (), tuple(range(HARNESS_NUM_CLASSES)), {}, "x"),
            overwrite=True,
        )


def test_register_rejects_out_of_range_id() -> None:
    """A kept id outside the semantic18 range ``[0, 18)`` is rejected."""
    with pytest.raises(ValueError, match="outside the semantic18 range"):
        register_label_space(LabelSpace("bad-id", (18,), (), {18: "x"}, "x"), overwrite=True)
