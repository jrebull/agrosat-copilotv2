"""Tests for the Stacking 'in action' inference cards (06c).

Covers ``ml/report/ensemble_inference_cards.py``: the rule-based phenology
description, the per-parcel NDVI curve, the Stacking argmax reconstruction from
the three base-learner OOF (mocked parquets), and an end-to-end card on real
PASTIS (gated by a skip). No network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.report.ensemble_inference_cards import (
    _describe_phenology,
    _parcel_ndvi,
    _stacking_pred_by_parcel,
)

_PASTIS_ROOT = Path("data/PASTIS-R")


def test_describe_phenology_high_late_vs_low_flat() -> None:
    """The description reflects peak height, timing and amplitude."""
    t = np.linspace(0, 1, 20)
    high_late = 0.1 + 0.8 * t  # rises to a high late peak, big amplitude
    txt = _describe_phenology(high_late)
    assert "alto" in txt and "tardío" in txt and "marcada" in txt

    flat_low = np.full(20, 0.15)  # no dynamics, low vigor
    txt2 = _describe_phenology(flat_low)
    assert "bajo" in txt2 and "plana" in txt2


def test_describe_phenology_empty() -> None:
    """An empty NDVI curve yields a graceful message, not a crash."""
    assert "sin observaciones" in _describe_phenology(np.array([]))


def test_parcel_ndvi_matches_manual() -> None:
    """The parcel NDVI equals (B08-B04)/(B08+B04) over the masked pixels."""
    t, h, w = 3, 4, 4
    s2 = np.zeros((t, 10, h, w), dtype=np.int16)
    s2[:, 2, :, :] = 1000  # B04
    s2[:, 6, :, :] = 3000  # B08
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    ndvi = _parcel_ndvi(s2.astype(float), mask)
    # (3000-1000)/(3000+1000) = 0.5
    assert ndvi.shape == (t,)
    assert np.allclose(ndvi, 0.5, atol=1e-3)


def test_stacking_pred_argmax_from_three_oof(tmp_path) -> None:
    """The Stacking prediction is the argmax of the 3-member average; only
    parcels present in ALL three members are returned (+1 class offset)."""
    prob_cols = [f"prob_{i:03d}" for i in range(18)]

    def _frame(ids: list[str], peak_class: int) -> pl.DataFrame:
        data = {"canonical_parcel_id": ids}
        for i, c in enumerate(prob_cols):
            data[c] = [0.9 if i == peak_class else 0.1 / 17 for _ in ids]
        return pl.DataFrame(data)

    # p1 present in all 3 (class 5 wins); p2 only in 2 -> dropped.
    _frame(["10_1", "10_2"], peak_class=4).write_parquet(
        tmp_path / "oof_parcel_tsvit-pheno_fold5.parquet"
    )
    _frame(["10_1", "10_2"], peak_class=4).write_parquet(tmp_path / "oof_parcel_utae_fold5.parquet")
    _frame(["10_1"], peak_class=4).write_parquet(
        tmp_path / "oof_parcel_xgb-alphaearth_fold5.parquet"
    )

    preds = _stacking_pred_by_parcel(["10_1", "10_2"], tmp_path)
    assert preds == {"10_1": 5}  # prob_004 -> class 5; p2 dropped (not in all 3)


def test_consensus_respects_custom_members(tmp_path) -> None:
    """A custom ``members`` tuple only requires presence in THOSE members."""
    prob_cols = [f"prob_{i:03d}" for i in range(18)]

    def _frame(ids: list[str], peak_class: int) -> pl.DataFrame:
        data = {"canonical_parcel_id": ids}
        for i, c in enumerate(prob_cols):
            data[c] = [0.9 if i == peak_class else 0.1 / 17 for _ in ids]
        return pl.DataFrame(data)

    # Only two members exist; p1 in both -> kept when members=(a, b).
    _frame(["10_1"], peak_class=6).write_parquet(tmp_path / "oof_parcel_farslip-ft18_fold5.parquet")
    _frame(["10_1"], peak_class=6).write_parquet(
        tmp_path / "oof_parcel_farslip-zeroshot_fold5.parquet"
    )

    preds = _stacking_pred_by_parcel(
        ["10_1"], tmp_path, members=("farslip-ft18", "farslip-zeroshot")
    )
    assert preds == {"10_1": 7}  # prob_006 -> class 7


@pytest.mark.skipif(
    not (_PASTIS_ROOT / "DATA_S2").exists(),
    reason="PASTIS-R not present on disk",
)
def test_build_cards_end_to_end_on_real_pastis(tmp_path) -> None:
    """A real fold-5 patch yields a card with a 3-panel PNG and a parcel table."""
    from ml.report.ensemble_inference_cards import build_stacking_inference_cards

    oof_dir = Path("ml/eval/oof")
    if not (oof_dir / "oof_parcel_tsvit-pheno_fold5.parquet").exists():
        pytest.skip("parcel OOF not pulled from DVC")
    oof = pl.read_parquet(oof_dir / "oof_parcel_tsvit-pheno_fold5.parquet").filter(
        pl.col("held_out")
    )
    pid = oof.group_by("patch_id").len().sort("len", descending=True)["patch_id"][0]

    cards = build_stacking_inference_cards(
        [pid],
        pastis_root=_PASTIS_ROOT,
        oof_dir=oof_dir,
        features_path=Path("data/features/features_fused_pastis.parquet"),
        out_dir=tmp_path / "cards",
    )
    assert len(cards) == 1
    card = cards[0]
    assert card.figure_path.is_file()
    assert card.n_parcels > 0
    assert 0 <= card.n_correct <= card.n_parcels
    assert set(card.table.columns) >= {
        "parcela",
        "clase_real",
        "clase_predicha",
        "acierto",
        "fenologia",
    }
