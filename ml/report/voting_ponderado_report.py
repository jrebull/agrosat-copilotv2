"""Report helpers for the Voting-ponderado vs Stacking finding (engram #340/#342).

Computes, on the real PASTIS-R fold-5 OOF, the head-to-head between the weighted
parcel vote (``WeightedVotingEnsemble``, N convex weights) and the Stacking
champions (meta-LogReg, 3 and 5 members), under THREE label-spaces measured with
the deployed ``restrict`` protocol:

- ``semantic18``  -- all 18 PASTIS classes (every parcel scored, argmax over 18);
- ``france-9``    -- the 9 best-resolved classes (the deployed champion space);
- ``france-10``   -- france-9 + Winter durum wheat (id 10).

It also exposes the weighted vote's per-parcel prediction (mapped to the PASTIS
class id space ``1..18``) so :func:`ml.report.ensemble_inference_cards.
build_stacking_inference_cards` can render the "Voting ponderado en accion" cards.

Nothing is trained: the members' posteriors are the pre-materialised OOF; the vote
learns N weights with ``scipy.optimize`` (CPU, seconds). PASTIS-R real data only.

Visible prose Spanish elsewhere; this is code, so identifiers/docstrings English.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "FRANCE_9_IDS",
    "FRANCE_10_IDS",
    "SEMANTIC18_NAMES",
    "VotingReport",
    "build_voting_report",
]

#: semantic18 id ``c`` corresponds to PASTIS class ``c + 1`` (Background dropped).
SEMANTIC18_NAMES: tuple[str, ...] = (
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits/veg/flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
)

#: The nine best-resolved semantic18 ids (the deployed france-9 label-space).
FRANCE_9_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 8, 14)
#: france-9 + Winter durum wheat (id 10), the 10th by the official discard order.
FRANCE_10_IDS: tuple[int, ...] = tuple(sorted((*FRANCE_9_IDS, 10)))

_VOTE3_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")
_STACK5_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth",
    "farslip-ft18",
    "farslip-zeroshot",
)


@dataclass
class VotingReport:
    """Materialised tables + predictions for the Voting-ponderado notebook."""

    headline: pl.DataFrame
    perclass_fr9: pl.DataFrame
    perclass_fr10: pl.DataFrame
    weights: dict[str, float]
    pred_by_parcel: dict[str, int]
    fold5_patch_ids: list[str]
    n_parcels: int
    meta: dict[str, object] = field(default_factory=dict)


def _restrict(
    labels: np.ndarray, proba18: np.ndarray, kept: tuple[int, ...]
) -> tuple[dict[str, float], dict[int, float]]:
    """Deployed ``restrict`` metrics over ``kept``: macro-F1, accuracy, per-class.

    Scores only parcels whose true class is in ``kept`` and masks the posterior to
    ``kept`` (argmax over the kept columns), mirroring ``restrict_posterior`` of the
    deployed perceiver. Returns ``(summary, per_class_f1)``.
    """
    k = np.asarray(kept)
    preds = k[proba18[:, k].argmax(axis=1)]
    mask = np.isin(labels, k)
    yt, yp = labels[mask], preds[mask]
    f1: dict[int, float] = {}
    for c in k:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        den = 2 * tp + fp + fn
        f1[int(c)] = (2 * tp / den) if den else 0.0
    summary: dict[str, float] = {
        "f1_macro": float(np.mean(list(f1.values()))),
        "accuracy": float((yt == yp).mean()) if yt.size else 0.0,
        "n": int(yt.size),
    }
    return summary, f1


def build_voting_report(
    oof_dir: Path,
    pastis_root: Path,
    *,
    random_state: int = 42,
) -> VotingReport:
    """Fit the vote + stacking champions and build the comparison tables.

    Args:
        oof_dir: directory with the per-parcel OOF parquets (US-031).
        pastis_root: PASTIS-R root (ground truth + parcel geometries).
        random_state: deterministic seed.

    Returns:
        A :class:`VotingReport` with the headline / per-class tables, the learned
        weights and the weighted-vote per-parcel prediction (PASTIS class ids).
    """
    from ml.ensemble.base import EnsembleModel
    from ml.ensemble.stacking import StackingEnsemble
    from ml.ensemble.voting_weighted import WeightedVotingEnsemble

    # The US-040 closing-run helpers are the single source of truth for the GT,
    # geometry and label alignment that produced the 0.7470 Stacking number.
    from scripts.run_us040_ensembles import (
        _aligned_labels,
        _fold5_patch_ids,
        _geoms_for_blending,
        build_parcel_geometries,
        build_parcel_ground_truth,
    )

    oof_dir = Path(oof_dir)
    pastis_root = Path(pastis_root)

    patch_ids = _fold5_patch_ids(oof_dir)
    gt = build_parcel_ground_truth(patch_ids, pastis_root)
    geoms_pl = build_parcel_geometries(patch_ids, pastis_root)
    geoms_gdf = _geoms_for_blending(geoms_pl)

    def macro18(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
        return EnsembleModel.compute_metrics(labels, preds, ignore_index=None)

    # --- Weighted vote (3 members) -------------------------------------------
    vote3 = WeightedVotingEnsemble(_VOTE3_MEMBERS, oof_dir=oof_dir, random_state=random_state).fit(
        geoms_gdf, y_true=gt
    )
    v_ids = vote3._member_ids
    v_labels = _aligned_labels(v_ids, gt)
    v_proba = vote3.predict_proba()

    # --- Simple vote (1/N) over the same cached tensor -----------------------
    member_probs = vote3._member_probs
    assert member_probs is not None  # populated by ``fit`` above
    n_m = member_probs.shape[0]
    simple_proba = vote3._blend(member_probs, np.full(n_m, 1.0 / n_m))

    # --- Stacking (3 and 5 members) ------------------------------------------
    def _stack(members: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        st = StackingEnsemble(
            members, meta="logreg", oof_dir=oof_dir, random_state=random_state
        ).fit(geoms_pl, gt_labels=gt)
        proba = st.predict_proba()
        keys, _, _ = st.build_meta_features(gt_labels=None)
        labels = _aligned_labels(keys["canonical_parcel_id"].to_list(), gt)
        return proba, labels

    stack3_proba, stack3_labels = _stack(_VOTE3_MEMBERS)
    stack5_proba, stack5_labels = _stack(_STACK5_MEMBERS)

    # --- Headline table: three label-spaces, deployed restrict protocol -------
    spaces = (
        ("semantic18", tuple(range(18))),
        ("france-9", FRANCE_9_IDS),
        ("france-10", FRANCE_10_IDS),
    )
    models = (
        ("Voting ponderado (3m)", v_proba, v_labels, 3),
        ("Stacking-5 (campeon)", stack5_proba, stack5_labels, 90),
        ("Stacking-3", stack3_proba, stack3_labels, 54),
        ("Voting simple (1/N)", simple_proba, v_labels, 3),
    )
    rows: list[dict[str, object]] = []
    for name, proba, labels, nw in models:
        row: dict[str, object] = {"modelo": name, "n_pesos": nw}
        for space, kept in spaces:
            if space == "semantic18":
                m = macro18(labels, proba.argmax(-1))
                row["f1_semantic18"] = round(m["f1_macro"], 4)
                row["acc_semantic18"] = round(m["accuracy"], 4)
            else:
                s, _ = _restrict(labels, proba, kept)
                tag = space.replace("france-", "fr")
                row[f"f1_{tag}"] = round(s["f1_macro"], 4)
                row[f"acc_{tag}"] = round(s["accuracy"], 4)
        rows.append(row)
    headline = pl.DataFrame(rows)

    # --- Per-class tables (france-9 and france-10) ---------------------------
    def _perclass(kept: tuple[int, ...]) -> pl.DataFrame:
        _, vote_f1 = _restrict(v_labels, v_proba, kept)
        _, stack_f1 = _restrict(stack5_labels, stack5_proba, kept)
        support = np.bincount(v_labels, minlength=18)
        return pl.DataFrame(
            {
                "clase": [SEMANTIC18_NAMES[c] for c in kept],
                "support": [int(support[c]) for c in kept],
                "Voting-3": [round(vote_f1[c], 3) for c in kept],
                "Stacking-5": [round(stack_f1[c], 3) for c in kept],
                "delta": [round(vote_f1[c] - stack_f1[c], 3) for c in kept],
            }
        )

    perclass_fr9 = _perclass(FRANCE_9_IDS)
    perclass_fr10 = _perclass(FRANCE_10_IDS)

    # --- Weighted-vote per-parcel prediction for the inference cards ----------
    # semantic18 id c -> PASTIS class id c+1 (the cards' class space).
    vote_pred18 = v_proba.argmax(axis=-1)
    pred_by_parcel = {
        cid: int(pred) + 1 for cid, pred in zip(v_ids, vote_pred18.tolist(), strict=True)
    }
    weights = {m: round(float(w), 4) for m, w in zip(_VOTE3_MEMBERS, vote3.weights, strict=True)}

    logger.info(
        "voting_report_built",
        n_parcels=len(v_ids),
        vote3_f1_18=headline.filter(pl.col("modelo").str.contains("ponderado"))["f1_semantic18"][0],
        weights=weights,
    )
    return VotingReport(
        headline=headline,
        perclass_fr9=perclass_fr9,
        perclass_fr10=perclass_fr10,
        weights=weights,
        pred_by_parcel=pred_by_parcel,
        fold5_patch_ids=list(patch_ids),
        n_parcels=len(v_ids),
        meta={"oof_cv_f1_vote3": round(vote3.oof_cv_metrics_.get("f1_macro", float("nan")), 4)},
    )
