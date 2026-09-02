"""Perceiver champion-vs-baseline evaluation (US-046 / US-049 / US-081 re-wiring check).

Quantifies the impact of re-wiring the agent perceiver from the ``xgb-alphaearth``
baseline to the EPIC champion, restricted to a configurable label-space. Over the
real fold-5 OOF universe (the parcels held out from every base member, leak-free),
it compares, per parcel:

* ``xgb-alphaearth`` argmax (the OLD perceiver path),
* the CHAMPION argmax (the NEW perceiver path), both restricted to the active
  label-space,

against the per-parcel semantic18 ground truth reconstructed from PASTIS-R. It
reports the label-space accuracy and macro-F1 of each, plus the agreement and the
net parcels the champion fixes vs breaks. This is the project-grounded evidence
that the re-wiring improves the agent's perception (the US-049 system-eval uses a
stub classifier by design, so it cannot show this difference).

Two champions are selectable so the SAME harness measures both EPIC milestones:

* ``stacking5`` -- the EPIC 6 / US-043 Stacking-5 meta (the v1 champion the
  perceiver was first re-wired to); restricted to ``france-9`` it is the
  comparable "previous champion" baseline.
* ``voting3`` -- the EPIC 12 / US-079 Voting-3 v2 champion
  (``tsvit-pheno-fullm-v2`` @ ``n_timesteps=32`` + ``utae`` + ``xgb-alphaearth``,
  pinned deployment weights ``0.902 / 0.0 / 0.098``) -- the model the perceiver
  serves today, evaluated over ``france-9`` (comparable) AND ``france-12`` (the
  three new resolved crops Spring barley / Winter durum wheat / Orchard).

The evaluation reuses the cached loaders of :mod:`ml.agent.tools.classify`
(``_load_stacking_five``, ``_load_voting_three``) and the ground-truth
reconstructor (``_build_parcel_ground_truth``) -- it never re-implements the model
logic. It is CPU-only (no GPU, no raster): the dense members are consumed through
their pre-materialised OOF, exactly like the deployed perceiver.

Run:
    # Single champion + label-space:
    poetry run python -m ml.eval.perceiver_champion_eval \
        --champion voting3 --label-space france-12 \
        --out reports/agent_bench/perceiver_champion_eval_v2.json

    # The full US-081 v2 report (france-9 + france-12, both champions):
    poetry run python -m ml.eval.perceiver_champion_eval --v2-report \
        --out reports/agent_bench/perceiver_champion_eval_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

from ml.eval.class_remap import DEFAULT_LABEL_SPACE, get_label_space, restrict_posterior

logger = structlog.get_logger(__name__)

#: Supported champion identifiers (the model the NEW perceiver path serves).
CHAMPIONS: tuple[str, ...] = ("stacking5", "voting3")

#: Default JSON output path for a single-champion run.
DEFAULT_OUT: Path = Path("reports/agent_bench/perceiver_champion_eval.json")

#: Default output path for the full US-081 v2 report (both spaces, both champions).
DEFAULT_V2_OUT: Path = Path("reports/agent_bench/perceiver_champion_eval_v2.json")


class _ChampionModel(Protocol):
    """Minimal protocol the champion loaders satisfy for this eval.

    Both :class:`ml.agent.tools.classify._StackingFive` and
    :class:`ml.agent.tools.classify._VotingThree` expose ``posterior_for_parcel``;
    the runner only needs that plus the per-parcel id universe.
    """

    def posterior_for_parcel(self, canonical_id: str) -> np.ndarray | None:
        """Return the ``(18,)`` posterior for ``canonical_id`` (or ``None``)."""
        ...


def _restricted_argmax(proba18: np.ndarray, label_space: Any) -> int | None:
    """Argmax semantic18 class id after restricting an 18-vector to a label-space.

    Args:
        proba18: A ``(18,)`` post-softmax posterior over the semantic18 space.
        label_space: The active :class:`~ml.eval.class_remap.LabelSpace`.

    Returns:
        The semantic18 id of the top kept class, or ``None`` when no mass landed on
        the resolved classes (an honest abstention).
    """
    restricted = restrict_posterior(proba18, label_space)
    if not restricted or max(restricted.values()) <= 0.0:
        return None
    return max(restricted, key=lambda cid: restricted[cid])


def _macro_f1(y_true: list[int], y_pred: list[int | None]) -> float:
    """Macro-F1 over the union of observed classes (None preds count as wrong).

    Args:
        y_true: Ground-truth semantic18 ids.
        y_pred: Predicted semantic18 ids (``None`` for abstentions, scored as miss).

    Returns:
        Unweighted mean per-class F1 over the classes present in ``y_true``.
    """
    classes = sorted(set(y_true))
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _load_champion(champion: str) -> tuple[_ChampionModel, list[str]]:
    """Load a champion model and the canonical-id universe it can score.

    Args:
        champion: One of :data:`CHAMPIONS`.

    Returns:
        A ``(model, canonical_ids)`` pair where ``model.posterior_for_parcel`` is
        the NEW perceiver path and ``canonical_ids`` is every parcel in the
        champion's fold-5 OOF universe.

    Raises:
        ValueError: if ``champion`` is not a supported identifier.
        FileNotFoundError: if the fold-5 OOF parquets are unavailable (run
            ``dvc pull ml/eval/oof ml/eval/oof_new32``).
    """
    from ml.agent.tools import classify as cls

    if champion == "stacking5":
        model = cls._load_stacking_five()
        return model, list(model.meta_features_by_id.keys())
    if champion == "voting3":
        voting = cls._load_voting_three()
        return voting, list(voting.member_probs_by_id.keys())
    raise ValueError(f"unknown champion {champion!r}; supported: {CHAMPIONS}.")


def _baseline_xgb_posterior(champion: str, model: _ChampionModel, cid: str) -> np.ndarray | None:
    """Return the ``xgb-alphaearth`` fold-5 OOF posterior for one parcel.

    The baseline is the SAME single-member posterior the perceiver's degraded path
    would emit. It is read out of the champion's cached member tensor (no model
    re-run, no embedding fetch), so the comparison is leak-free and CPU-only:

    * ``stacking5``: the 90-dim meta-feature row is the 5 members x 18 probs
      concatenated; the ``xgb-alphaearth`` block is sliced by its position in
      ``_STACKING_MEMBERS``.
    * ``voting3``: the cached ``(3, 18)`` member tensor carries the xgb row at the
      ``_VOTING_MEMBERS`` index of ``xgb-alphaearth``.

    Args:
        champion: The active champion identifier.
        model: The loaded champion model holding the cached member rows.
        cid: Canonical parcel id to look up.

    Returns:
        The ``(18,)`` xgb-alphaearth posterior, or ``None`` when the parcel is
        absent from the cached universe.
    """
    from ml.agent.tools import classify as cls

    if champion == "stacking5":
        row = model.meta_features_by_id.get(cid)  # type: ignore[attr-defined]
        if row is None:
            return None
        idx = cls._STACKING_MEMBERS.index("xgb-alphaearth")
        return np.asarray(row[idx * 18 : (idx + 1) * 18])
    # voting3: (3, 18) member tensor; pick the xgb-alphaearth member row.
    member_rows = model.member_probs_by_id.get(cid)  # type: ignore[attr-defined]
    if member_rows is None:
        return None
    idx = cls._VOTING_MEMBERS.index("xgb-alphaearth")
    return np.asarray(member_rows[idx])


def evaluate(
    *,
    champion: str = "voting3",
    label_space_name: str | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Compare a champion vs the xgb baseline over the fold-5 OOF universe.

    Loads the requested champion (cached), scores every fold-5 parcel both ways
    restricted to ``label_space_name``, and contrasts them against the PASTIS-R
    ground truth.

    Args:
        champion: Which champion to serve as the NEW perceiver path (one of
            :data:`CHAMPIONS`). Defaults to the deployed ``voting3`` v2 champion.
        label_space_name: Registered label-space name (``"france-9"`` /
            ``"france-12"``). ``None`` resolves to
            :data:`~ml.eval.class_remap.DEFAULT_LABEL_SPACE`.
        out_path: Optional path to dump the JSON summary.

    Returns:
        A summary dict with per-model accuracy/macro-F1 over the label-space,
        agreement, and the net parcels the champion fixes vs breaks vs the baseline.

    Raises:
        ValueError: if ``champion`` is unknown or no scorable parcel remains.
        FileNotFoundError: if the fold-5 OOF parquets or PASTIS-R ground truth are
            unavailable (run ``dvc pull ml/eval/oof ml/eval/oof_new32`` /
            ``dvc pull data/PASTIS-R``).
    """
    from ml.agent.tools import classify as cls

    label_space = get_label_space(label_space_name)
    model, canonical_ids = _load_champion(champion)

    # Ground truth for every parcel the champion can score (its joined OOF).
    gt_frame = cls._build_parcel_ground_truth(canonical_ids)
    gt_by_id = dict(
        zip(
            gt_frame.get_column("canonical_parcel_id").to_list(),
            gt_frame.get_column("label").to_list(),
            strict=True,
        )
    )

    kept = set(label_space.kept_class_ids)
    y_true: list[int] = []
    pred_champion: list[int | None] = []
    pred_baseline: list[int | None] = []

    for cid, label in gt_by_id.items():
        if int(label) not in kept:
            continue  # GT outside the label-space: not scorable here.
        champ_proba = model.posterior_for_parcel(cid)
        if champ_proba is None:
            continue
        base_proba = _baseline_xgb_posterior(champion, model, cid)
        if base_proba is None:
            continue
        y_true.append(int(label))
        pred_champion.append(_restricted_argmax(champ_proba, label_space))
        pred_baseline.append(_restricted_argmax(base_proba, label_space))

    n = len(y_true)
    if n == 0:
        raise ValueError(f"no {label_space.name} parcels with both GT and a {champion} posterior.")

    champ_acc = sum(1 for t, p in zip(y_true, pred_champion, strict=True) if t == p) / n
    base_acc = sum(1 for t, p in zip(y_true, pred_baseline, strict=True) if t == p) / n
    agreement = sum(1 for a, b in zip(pred_champion, pred_baseline, strict=True) if a == b) / n
    champion_fixes = sum(
        1 for t, c, b in zip(y_true, pred_champion, pred_baseline, strict=True) if c == t and b != t
    )
    champion_breaks = sum(
        1 for t, c, b in zip(y_true, pred_champion, pred_baseline, strict=True) if c != t and b == t
    )

    summary = {
        "label_space": label_space.name,
        "champion": champion,
        "n_parcels": n,
        "n_classes": len(label_space.kept_class_ids),
        "baseline_xgb": {
            "accuracy": round(base_acc, 4),
            "macro_f1": round(_macro_f1(y_true, pred_baseline), 4),
        },
        f"champion_{champion}": {
            "accuracy": round(champ_acc, 4),
            "macro_f1": round(_macro_f1(y_true, pred_champion), 4),
        },
        "delta_accuracy": round(champ_acc - base_acc, 4),
        "delta_macro_f1": round(
            _macro_f1(y_true, pred_champion) - _macro_f1(y_true, pred_baseline), 4
        ),
        "agreement": round(agreement, 4),
        "champion_fixes": champion_fixes,
        "champion_breaks": champion_breaks,
        "net_fixed": champion_fixes - champion_breaks,
    }
    logger.info(
        "perceiver_champion_eval_done",
        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("perceiver_champion_eval_written", path=str(out_path))
    return summary


def evaluate_v2_report(out_path: Path | None = None) -> dict[str, Any]:
    """Produce the full US-081 v2 perception report (AC1).

    Bundles, over the real fold-5 OOF, three sub-evaluations that together answer
    "did the COPILOT's perception improve with the v2 champion?":

    * ``previous_champion_france9`` -- xgb baseline vs the Stacking-5 v1 champion,
      restricted to ``france-9`` (the comparable "previous champion" number).
    * ``v2_champion_france9`` -- xgb baseline vs the Voting-3 v2 champion over the
      SAME ``france-9`` space (apples-to-apples uplift of the model swap).
    * ``v2_champion_france12`` -- xgb baseline vs the Voting-3 v2 champion over the
      expanded ``france-12`` space (the three new resolved crops in scope).

    Args:
        out_path: Optional path to dump the combined JSON.

    Returns:
        A dict with the three sub-summaries plus a ``deltas`` block contrasting the
        v2 champion against the v1 champion over ``france-9`` (the head-to-head the
        US-081 narrative reports).
    """
    previous_f9 = evaluate(champion="stacking5", label_space_name="france-9")
    v2_f9 = evaluate(champion="voting3", label_space_name="france-9")
    v2_f12 = evaluate(champion="voting3", label_space_name="france-12")

    report: dict[str, Any] = {
        "description": (
            "US-081 AC1: perceiver champion-vs-baseline over the real fold-5 OOF. "
            "Previous champion (Stacking-5 v1) vs v2 champion (Voting-3 v2) over "
            "france-9 (comparable) and france-12 (expanded scope)."
        ),
        "previous_champion_france9": previous_f9,
        "v2_champion_france9": v2_f9,
        "v2_champion_france12": v2_f12,
        "deltas": {
            "v2_vs_v1_france9_accuracy": round(
                v2_f9["champion_voting3"]["accuracy"]
                - previous_f9["champion_stacking5"]["accuracy"],
                4,
            ),
            "v2_vs_v1_france9_macro_f1": round(
                v2_f9["champion_voting3"]["macro_f1"]
                - previous_f9["champion_stacking5"]["macro_f1"],
                4,
            ),
            "v2_france12_minus_france9_accuracy": round(
                v2_f12["champion_voting3"]["accuracy"] - v2_f9["champion_voting3"]["accuracy"],
                4,
            ),
        },
    }
    logger.info(
        "perceiver_champion_eval_v2_report_done",
        v2_vs_v1_france9_accuracy=report["deltas"]["v2_vs_v1_france9_accuracy"],
        v2_france9_macro_f1=v2_f9["champion_voting3"]["macro_f1"],
        v2_france12_macro_f1=v2_f12["champion_voting3"]["macro_f1"],
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("perceiver_champion_eval_v2_report_written", path=str(out_path))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the perceiver champion-vs-baseline evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion",
        choices=CHAMPIONS,
        default="voting3",
        help="Champion model to serve as the NEW perceiver path.",
    )
    parser.add_argument(
        "--label-space",
        default=None,
        help=(
            "Registered label-space name (france-9 / france-12). "
            f"Default resolves to {DEFAULT_LABEL_SPACE!r}."
        ),
    )
    parser.add_argument(
        "--v2-report",
        action="store_true",
        help=(
            "Run the full US-081 v2 report (france-9 + france-12, v1 vs v2 "
            "champion). Ignores --champion / --label-space."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to dump the JSON summary (defaults depend on --v2-report).",
    )
    args = parser.parse_args(argv)

    if args.v2_report:
        out = args.out or DEFAULT_V2_OUT
        summary = evaluate_v2_report(out_path=out)
    else:
        out = args.out or DEFAULT_OUT
        summary = evaluate(
            champion=args.champion,
            label_space_name=args.label_space,
            out_path=out,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
