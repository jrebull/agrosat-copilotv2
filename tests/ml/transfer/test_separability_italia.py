"""Tests for ml.transfer.separability_italia (US-082 scoping diagnostic)."""

from __future__ import annotations

import numpy as np

from ml.transfer.separability_italia import (
    _JM_SEPARABLE,
    _MIN_PARCELS_FOR_TARGET,
    ClassSeparability,
    _bhattacharyya_gaussian,
    _jeffries_matusita,
)


def test_jm_is_bounded_in_zero_two() -> None:
    """The Jeffries-Matusita transform maps any B >= 0 into [0, 2)."""
    assert _jeffries_matusita(0.0) == 0.0
    assert 0.0 < _jeffries_matusita(0.5) < 2.0
    assert 1.9 < _jeffries_matusita(10.0) < 2.0


def test_bhattacharyya_zero_for_identical_gaussians() -> None:
    """Two identical Gaussians have ~zero Bhattacharyya distance."""
    rng = np.random.default_rng(0)
    mu = rng.normal(size=8)
    cov = np.eye(8)
    b = _bhattacharyya_gaussian(mu, cov, mu, cov)
    assert abs(b) < 1e-6


def test_bhattacharyya_grows_with_mean_separation() -> None:
    """Pulling the means apart strictly increases the distance."""
    dim = 8
    cov = np.eye(dim)
    mu_a = np.zeros(dim)
    near = _bhattacharyya_gaussian(mu_a, cov, mu_a + 0.5, cov)
    far = _bhattacharyya_gaussian(mu_a, cov, mu_a + 5.0, cov)
    assert far > near > 0.0


def test_class_separability_gates() -> None:
    """has_support, is_separable and is_rescuable apply the documented gates."""
    starved = ClassSeparability(
        class_id=1,
        name="sunflower",
        n_parcels=_MIN_PARCELS_FOR_TARGET - 1,
        mean_jm=2.0,
        nearest_name="olive",
        nearest_jm=2.0,
    )
    assert not starved.has_support
    assert starved.is_separable
    assert not starved.is_rescuable  # support gate fails

    overlapping = ClassSeparability(
        class_id=2,
        name="barley",
        n_parcels=_MIN_PARCELS_FOR_TARGET + 10,
        mean_jm=0.5,
        nearest_name="oats",
        nearest_jm=_JM_SEPARABLE - 0.1,
    )
    assert overlapping.has_support
    assert not overlapping.is_separable
    assert not overlapping.is_rescuable  # separability gate fails

    rescuable = ClassSeparability(
        class_id=3,
        name="olive",
        n_parcels=_MIN_PARCELS_FOR_TARGET + 100,
        mean_jm=1.8,
        nearest_name="vineyards",
        nearest_jm=_JM_SEPARABLE + 0.5,
    )
    assert rescuable.is_rescuable  # both gates clear
