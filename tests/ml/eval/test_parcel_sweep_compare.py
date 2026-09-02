"""Tests for the parcel-sweep 3:1-filter A/B comparison (US-036-b).

Covers ``ml/eval/parcel_sweep_compare.py``: the inner join of the two sweep
curves on ``n_classes``, the per-N delta, the guards (missing column, empty
join, missing file), and that the overlay figure builds. No real PASTIS, no
training: tiny CSVs written to ``tmp_path``.
"""

from __future__ import annotations

import polars as pl
import pytest

from ml.eval.parcel_sweep_compare import (
    build_dominance_comparison,
    dominance_curves_figure,
)


def _write_csv(path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_build_comparison_joins_and_deltas(tmp_path) -> None:
    """The join keeps shared N and computes ``dom3 - no_filter`` per row."""
    base = tmp_path / "sweep.csv"
    dom = tmp_path / "sweep_dom3.csv"
    _write_csv(
        base,
        [
            {"n_classes": 4, "macro_f1": 0.70, "n_eval_parcels": 16640, "dominance_ratio": -1.0},
            {"n_classes": 6, "macro_f1": 0.46, "n_eval_parcels": 17000, "dominance_ratio": -1.0},
        ],
    )
    _write_csv(
        dom,
        [
            {"n_classes": 4, "macro_f1": 0.66, "n_eval_parcels": 9000, "dominance_ratio": 3.0},
            {"n_classes": 6, "macro_f1": 0.50, "n_eval_parcels": 9500, "dominance_ratio": 3.0},
        ],
    )

    out = build_dominance_comparison(base, dom)
    assert out["n_classes"].to_list() == [4, 6]
    assert out["macro_f1_no_filter"].to_list() == [0.70, 0.46]
    assert out["macro_f1_dom3"].to_list() == [0.66, 0.50]
    # delta = dom3 - no_filter (negative where the filter hurts, positive where it helps)
    deltas = out["delta_macro_f1"].to_list()
    assert deltas[0] == pytest.approx(-0.04, abs=1e-6)
    assert deltas[1] == pytest.approx(0.04, abs=1e-6)
    assert "n_eval_no_filter" in out.columns and "n_eval_dom3" in out.columns


def test_build_comparison_inner_join_only_shared_n(tmp_path) -> None:
    """N present in only one curve is dropped (inner join)."""
    base = tmp_path / "sweep.csv"
    dom = tmp_path / "sweep_dom3.csv"
    _write_csv(
        base,
        [
            {"n_classes": 4, "macro_f1": 0.70},
            {"n_classes": 8, "macro_f1": 0.40},
        ],
    )
    _write_csv(
        dom,
        [
            {"n_classes": 4, "macro_f1": 0.66},
            {"n_classes": 12, "macro_f1": 0.33},
        ],
    )
    out = build_dominance_comparison(base, dom)
    assert out["n_classes"].to_list() == [4]  # only N=4 shared


def test_build_comparison_no_eval_cols_ok(tmp_path) -> None:
    """The eval-parcel columns are optional; absence does not break the join."""
    base = tmp_path / "sweep.csv"
    dom = tmp_path / "sweep_dom3.csv"
    _write_csv(base, [{"n_classes": 4, "macro_f1": 0.70}])
    _write_csv(dom, [{"n_classes": 4, "macro_f1": 0.66}])
    out = build_dominance_comparison(base, dom)
    assert "n_eval_no_filter" not in out.columns
    assert out["delta_macro_f1"].to_list() == [pytest.approx(-0.04, abs=1e-6)]


def test_missing_required_column_raises(tmp_path) -> None:
    """A CSV lacking ``macro_f1`` is rejected with a clear error."""
    bad = tmp_path / "bad.csv"
    good = tmp_path / "good.csv"
    _write_csv(bad, [{"n_classes": 4, "iou": 0.5}])
    _write_csv(good, [{"n_classes": 4, "macro_f1": 0.66}])
    with pytest.raises(ValueError, match="missing required columns"):
        build_dominance_comparison(bad, good)


def test_empty_join_raises(tmp_path) -> None:
    """No shared N between curves is an explicit error, not a silent empty table."""
    base = tmp_path / "sweep.csv"
    dom = tmp_path / "sweep_dom3.csv"
    _write_csv(base, [{"n_classes": 4, "macro_f1": 0.70}])
    _write_csv(dom, [{"n_classes": 6, "macro_f1": 0.50}])
    with pytest.raises(ValueError, match="no shared n_classes"):
        build_dominance_comparison(base, dom)


def test_missing_file_raises(tmp_path) -> None:
    """A non-existent CSV path raises FileNotFoundError."""
    good = tmp_path / "good.csv"
    _write_csv(good, [{"n_classes": 4, "macro_f1": 0.66}])
    with pytest.raises(FileNotFoundError):
        build_dominance_comparison(tmp_path / "nope.csv", good)


def test_figure_builds(tmp_path) -> None:
    """The overlay figure builds with two lines over the shared N axis."""
    base = tmp_path / "sweep.csv"
    dom = tmp_path / "sweep_dom3.csv"
    _write_csv(
        base,
        [
            {"n_classes": 4, "macro_f1": 0.70},
            {"n_classes": 6, "macro_f1": 0.46},
        ],
    )
    _write_csv(
        dom,
        [
            {"n_classes": 4, "macro_f1": 0.66},
            {"n_classes": 6, "macro_f1": 0.50},
        ],
    )
    out = build_dominance_comparison(base, dom)
    fig = dominance_curves_figure(out)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Numero de clases (N)"
    import matplotlib.pyplot as plt

    plt.close(fig)
