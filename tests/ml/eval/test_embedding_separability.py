"""Tests for ml.eval.embedding_separability.

Cover the pure-logic helpers (balanced sampling, space alignment, multi-year
combination, column selection) plus a small eval_space run on synthetic,
separable data so the metric wiring is exercised without heavy dependencies.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.eval.embedding_separability import (
    align_spaces_on_parcels,
    build_balanced_eval_set,
    combine_year_embeddings,
    embedding_columns,
    eval_space,
    load_alphaearth_embeddings,
    space_matrix,
)


def _toy_universe(n_per_class: dict[int, int]) -> pl.DataFrame:
    rows = []
    for cls, n in n_per_class.items():
        for i in range(n):
            rows.append({"parcel_id": f"{cls}000_{i}", "class_id": cls})
    return pl.DataFrame(rows)


def test_build_balanced_eval_set_caps_and_drops() -> None:
    universe = _toy_universe({1: 100, 2: 100, 3: 10})  # class 3 is rare
    balanced, dropped = build_balanced_eval_set(universe, per_class_cap=50, min_class_samples=20)
    counts = dict(balanced.group_by("class_id").len().iter_rows())
    assert dropped == [3]
    assert counts == {1: 50, 2: 50}
    assert 3 not in counts


def test_build_balanced_eval_set_is_deterministic() -> None:
    universe = _toy_universe({1: 200, 2: 200})
    a, _ = build_balanced_eval_set(universe, per_class_cap=30, min_class_samples=10)
    b, _ = build_balanced_eval_set(universe, per_class_cap=30, min_class_samples=10)
    assert a.sort("parcel_id")["parcel_id"].to_list() == b.sort("parcel_id")["parcel_id"].to_list()


def test_build_balanced_eval_set_adds_class_names() -> None:
    universe = _toy_universe({1: 30, 2: 30})
    balanced, _ = build_balanced_eval_set(
        universe, per_class_cap=10, min_class_samples=5, class_names={1: "Meadow", 2: "Corn"}
    )
    assert "class_name" in balanced.columns
    assert set(balanced["class_name"].unique().to_list()) == {"Meadow", "Corn"}


def test_embedding_columns_sorted_prefix() -> None:
    df = pl.DataFrame({"parcel_id": ["a"], "emb_002": [1.0], "emb_000": [2.0], "x": [3.0]})
    assert embedding_columns(df, "emb_") == ["emb_000", "emb_002"]


def test_align_spaces_inner_join_and_prefixes() -> None:
    labels = pl.DataFrame({"parcel_id": ["p1", "p2", "p3"], "class_id": [1, 2, 1]})
    space_a = pl.DataFrame(
        {"parcel_id": ["p1", "p2", "p3"], "emb_000": [0.1, 0.2, 0.3], "emb_001": [1.0, 1.1, 1.2]}
    )
    # space_b is missing p3 -> inner join must drop it everywhere.
    space_b = pl.DataFrame({"parcel_id": ["p1", "p2"], "dim_00": [5.0, 6.0]})
    merged, cols = align_spaces_on_parcels(
        labels, {"far": (space_a, "emb_"), "ae": (space_b, "dim_")}
    )
    assert merged.height == 2  # p3 dropped
    assert cols["far"] == ["far__emb_000", "far__emb_001"]
    assert cols["ae"] == ["ae__dim_00"]
    assert space_matrix(merged, cols["far"]).shape == (2, 2)


def test_combine_year_embeddings_concatenates_dims() -> None:
    e2018 = pl.DataFrame({"parcel_id": ["p1", "p2"], "class_id": [1, 2], "dim_00_2018": [1.0, 2.0]})
    e2019 = pl.DataFrame({"parcel_id": ["p1", "p2"], "dim_00_2019": [3.0, 4.0]})
    combined = combine_year_embeddings(e2018, e2019)
    assert combined.height == 2
    assert "dim_00_2018" in combined.columns
    assert "dim_00_2019" in combined.columns
    assert "class_id" in combined.columns


def test_load_alphaearth_embeddings_suffix(tmp_path) -> None:
    p = tmp_path / "ae.parquet"
    pl.DataFrame(
        {"parcel_id": ["p1"], "class_id": [1], "year": [2019], "dim_00": [0.5], "dim_01": [0.6]}
    ).write_parquet(p)
    df = load_alphaearth_embeddings(p, year_suffix="_2019")
    assert set(df.columns) == {"parcel_id", "class_id", "dim_00_2019", "dim_01_2019"}


def test_eval_space_separable_data_high_f1() -> None:
    rng = np.random.default_rng(0)
    # Two well-separated Gaussian blobs in 8D -> F1-macro should be ~1.0.
    a = rng.normal(loc=0.0, scale=0.1, size=(60, 8))
    b = rng.normal(loc=5.0, scale=0.1, size=(60, 8))
    matrix = np.vstack([a, b])
    labels = np.array([0] * 60 + [1] * 60)
    result = eval_space(matrix, labels, label="toy", n_splits=5)
    assert result.n_samples == 120
    assert result.n_dims == 8
    assert result.n_classes == 2
    assert result.f1_macro_mean > 0.95
    assert result.silhouette > 0.5


def test_eval_space_random_data_low_f1() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(120, 8))
    labels = np.array([0, 1, 2] * 40)  # 3 classes, no signal
    result = eval_space(matrix, labels, label="noise", n_splits=5)
    # Random 3-class -> F1-macro near chance, well below a separable space.
    assert result.f1_macro_mean < 0.6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
