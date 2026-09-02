"""US-079 A/B ablation runner: warm-start vs no-warm-start (+ original).

Executes the per-class comparison of the three Italian dense fine-tunes over their
held-out ``test_softmax.npz`` dumps and emits (a) a JSON report with the per-class
tables (fine + coarse), the warm-start verdict and the discard curves, and (b)
three figures. It is the runner around :mod:`ml.eval.us079_ablation_compare`.

The three arms (the canonical aliases the module expects):

- ``--run-a``: ``ablA_warmstart-tsvit-pheno`` -- conserved head warm-started from
  PASTIS (the kept-class flag).
- ``--run-b``: ``ablB_nowarmstart-tsvit-pheno`` -- every head row random.
- ``--run-original``: ``us079_v2-tsvit-pheno`` -- the reference run.

Each ``--run-*`` accepts either the ``test_softmax.npz`` file or the directory
containing it. A run that is still training (its dump absent) is reported as
``pending`` and OMITTED from the tables / figures / verdict -- never fabricated, so
the runner is safe to launch with arm B missing (it then validates A + original
end-to-end and reports B as pending). Re-running later with ``--run-b`` completes
the full A/B.

Usage
-----
    poetry run python scripts/run_us079_ablation_analysis.py \
        --run-a checkpoints/transfer/tsvit-pheno-italia/ablA_warmstart-tsvit-pheno \
        --run-original checkpoints/transfer/tsvit-pheno-italia/us079_v2-tsvit-pheno \
        --dataset-root data/pastis_italia_2018 \
        --out-json reports/us079_figs/ablation_compare.json \
        --fig-dir reports/us079_figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

from ml.eval.us079_ablation_compare import (
    RunPerClass,
    compare_runs,
    discard_curve_compare,
    load_run_masks,
    warm_start_ablation_verdict,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET_ROOT = _REPO_ROOT / "data" / "pastis_italia_2018"
_DEFAULT_FIG_DIR = _REPO_ROOT / "reports" / "us079_figs"

#: Canonical arm aliases -> the CLI flag that feeds them. The module keys the A/B
#: verdict / delta columns off these exact aliases.
_ARM_ALIASES: dict[str, str] = {
    "A_warmstart": "run_a",
    "B_nowarmstart": "run_b",
    "original": "run_original",
}

#: Bar colours (colour-blind friendly): A vs B vs original.
_ARM_COLORS: dict[str, str] = {
    "A_warmstart": "#1b9e77",
    "B_nowarmstart": "#d95f02",
    "original": "#7570b3",
}


def _collect_runs(args: argparse.Namespace) -> dict[str, Path]:
    """Map the present ``--run-*`` flags to their canonical aliases.

    A flag left unset (``None``) is dropped here so the module never sees a None
    path; it is reported as pending later via the missing-arm log in ``compare_runs``.

    Args:
        args: The parsed CLI namespace.

    Returns:
        ``{alias: path}`` for the arms the operator supplied.
    """
    runs: dict[str, Path] = {}
    for alias, attr in _ARM_ALIASES.items():
        value = getattr(args, attr)
        if value is not None:
            runs[alias] = Path(value)
    return runs


def _arm_status(runs: Mapping[str, Path], scored: Mapping[str, RunPerClass]) -> dict[str, str]:
    """Report each arm as scored / pending / not-requested (honest coverage).

    Args:
        runs: The requested ``{alias: path}``.
        scored: The aliases that produced real scores.

    Returns:
        ``{alias: status}`` for the three canonical arms.
    """
    status: dict[str, str] = {}
    for alias in _ARM_ALIASES:
        if alias in scored:
            status[alias] = "scored"
        elif alias in runs:
            status[alias] = "pending"  # requested but the .npz is absent yet.
        else:
            status[alias] = "not_requested"
    return status


def fig_ab_per_class(
    table: pl.DataFrame,
    scored: Mapping[str, RunPerClass],
    out_path: Path,
) -> Path:
    """Grouped bars of per-class F1 (A vs B vs original), conserved highlighted.

    Args:
        table: The per-class comparison table (``compare_runs``).
        scored: The scored runs (one bar group per arm present).
        out_path: Destination PNG path.

    Returns:
        The written figure path.
    """
    aliases = [a for a in _ARM_ALIASES if a in scored]
    classes = table.sort("support", descending=True)
    names = classes["class_name"].to_list()
    conserved_flags = classes["is_conserved"].to_list()
    n = len(names)
    x = np.arange(n)
    width = 0.8 / max(len(aliases), 1)

    fig, ax = plt.subplots(figsize=(max(10.0, 0.42 * n), 6.0))
    for k, alias in enumerate(aliases):
        vals = classes[f"f1_{alias}"].to_numpy()
        ax.bar(
            x + k * width,
            vals,
            width,
            label=alias,
            color=_ARM_COLORS.get(alias, "#999999"),
        )
    ax.set_xticks(x + width * (len(aliases) - 1) / 2)
    labels = [f"* {nm}" if cons else nm for nm, cons in zip(names, conserved_flags, strict=True)]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    for tick, cons in zip(ax.get_xticklabels(), conserved_flags, strict=True):
        if cons:
            tick.set_color("#b30000")
            tick.set_fontweight("bold")
    ax.set_ylabel("F1 por clase")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("US-079 ablacion A/B: F1 por clase (conservadas en rojo, prefijo *)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("us079_fig_written", fig="ab_per_class", path=str(out_path))
    return out_path


def fig_ab_conserved_delta(table: pl.DataFrame, out_path: Path) -> Path | None:
    """Bar plot of the B-A F1 delta on the CONSERVED classes (positive = B better).

    Only drawn when both arms are present (the ``delta_b_minus_a`` column exists);
    returns ``None`` otherwise (the A/B is not complete yet).

    Args:
        table: The per-class comparison table.
        out_path: Destination PNG path.

    Returns:
        The written figure path, or ``None`` if the A/B is incomplete.
    """
    if "delta_b_minus_a" not in table.columns:
        logger.info("us079_fig_skipped", fig="ab_conserved_delta", reason="no B arm yet")
        return None
    conserved = table.filter(pl.col("is_conserved")).sort("delta_b_minus_a", descending=True)
    if conserved.height == 0:
        logger.info("us079_fig_skipped", fig="ab_conserved_delta", reason="no conserved class")
        return None
    names = conserved["class_name"].to_list()
    deltas = conserved["delta_b_minus_a"].to_numpy()
    colors = ["#d95f02" if d > 0 else "#1b9e77" for d in deltas]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(names)), 5.5))
    ax.bar(np.arange(len(names)), deltas, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("delta F1 (B sin warm-start  -  A con warm-start)")
    ax.set_title(
        "US-079: delta B-A en clases conservadas\n"
        "(positivo = SIN warm-start es mejor -> el prior PASTIS perjudica)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("us079_fig_written", fig="ab_conserved_delta", path=str(out_path))
    return out_path


def fig_discard_compare(
    curves: Mapping[str, list[dict[str, object]]],
    out_path: Path,
) -> Path | None:
    """Overlay the discard curves (macro F1 vs top-N classes) of every arm.

    Args:
        curves: ``{alias: [{"n_classes", "macro_f1", ...}]}`` from
            :func:`ml.eval.us079_ablation_compare.discard_curve_compare`.
        out_path: Destination PNG path.

    Returns:
        The written figure path, or ``None`` if no arm has a curve.
    """
    drawable = {a: c for a, c in curves.items() if c}
    if not drawable:
        logger.info("us079_fig_skipped", fig="discard_compare", reason="no curve")
        return None
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for alias, curve in drawable.items():
        ns = [int(row["n_classes"]) for row in curve]
        f1s = [float(row["macro_f1"]) for row in curve]
        ax.plot(
            ns,
            f1s,
            marker="o",
            markersize=3,
            label=alias,
            color=_ARM_COLORS.get(alias, "#999999"),
        )
    ax.axhline(0.9, color="#b30000", linestyle="--", linewidth=0.9, label="umbral 0.9")
    ax.set_xlabel("Numero de mejores clases retenidas (top-N por F1)")
    ax.set_ylabel("F1-macro sobre las top-N")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("US-079: curva de descarte A vs B vs original")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("us079_fig_written", fig="discard_compare", path=str(out_path))
    return out_path


def _run_level(
    runs: Mapping[str, Path],
    dataset_root: Path,
    *,
    level: str,
    fig_dir: Path,
    n_timesteps: int,
    masks_by_patch: Mapping[int, np.ndarray],
) -> dict[str, object]:
    """Score every arm at one granularity and emit its tables / verdict / figures.

    Args:
        runs: The requested ``{alias: path}``.
        dataset_root: The homologue dataset root.
        level: ``"fine"`` or ``"coarse"``.
        fig_dir: Output directory for the figures.
        n_timesteps: Forwarded to the mask loader / compare.
        masks_by_patch: Pre-loaded ground-truth masks.

    Returns:
        The JSON block for this level (tables, verdict, discard curves, fig paths,
        arm status).
    """
    table, scored = compare_runs(
        runs,
        dataset_root,
        level=level,
        n_timesteps=n_timesteps,
        masks_by_patch=masks_by_patch,
    )
    verdict = warm_start_ablation_verdict(table)
    curves = discard_curve_compare(scored)
    status = _arm_status(runs, scored)

    suffix = f"_{level}"
    figs: dict[str, str | None] = {}
    if scored:
        figs["fig_ab_per_class"] = str(
            fig_ab_per_class(table, scored, fig_dir / f"fig_ab_per_class{suffix}.png")
        )
    delta_fig = fig_ab_conserved_delta(table, fig_dir / f"fig_ab_conserved_delta{suffix}.png")
    figs["fig_ab_conserved_delta"] = str(delta_fig) if delta_fig else None
    discard_fig = fig_discard_compare(curves, fig_dir / f"fig_discard_compare{suffix}.png")
    figs["fig_discard_compare"] = str(discard_fig) if discard_fig else None

    return {
        "level": level,
        "arm_status": status,
        "n_patches_per_arm": {a: r.n_patches for a, r in scored.items()},
        "per_class_table": table.to_dicts(),
        "warm_start_verdict": verdict,
        "discard_curves": curves,
        "figures": figs,
    }


def main() -> None:
    """Parse the CLI, run the A/B (and original) comparison and write the report."""
    parser = argparse.ArgumentParser(
        description="US-079 A/B ablation analysis (warm-start vs no-warm-start)."
    )
    parser.add_argument(
        "--run-a",
        default=None,
        help="ablA_warmstart test_softmax.npz or its dir (conserved head warm-started).",
    )
    parser.add_argument(
        "--run-b",
        default=None,
        help="ablB_nowarmstart test_softmax.npz or its dir (random head). Omit while "
        "the arm trains; re-run with it to complete the A/B.",
    )
    parser.add_argument(
        "--run-original",
        default=None,
        help="us079_v2 test_softmax.npz or its dir (reference run).",
    )
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_DEFAULT_FIG_DIR / "ablation_compare.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=_DEFAULT_FIG_DIR)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["fine", "coarse"],
        choices=("fine", "coarse"),
        help="Granularities to score (default both).",
    )
    args = parser.parse_args()

    runs = _collect_runs(args)
    if not runs:
        parser.error("pass at least one of --run-a / --run-b / --run-original.")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    # Load the masks ONCE and reuse them across levels (they are split-agnostic).
    masks_by_patch = load_run_masks(args.dataset_root, n_timesteps=args.n_timesteps)

    levels: dict[str, object] = {}
    for level in args.levels:
        levels[level] = _run_level(
            runs,
            args.dataset_root,
            level=level,
            fig_dir=args.fig_dir,
            n_timesteps=args.n_timesteps,
            masks_by_patch=masks_by_patch,
        )

    fine_verdict = (
        levels.get("fine", {}).get("warm_start_verdict")  # type: ignore[union-attr]
        if "fine" in levels
        else None
    )
    report = {
        "us": "US-079",
        "analysis": "warm_start_ablation_compare",
        "dataset_root": str(args.dataset_root),
        "requested_arms": {a: str(p) for a, p in runs.items()},
        "fine_verdict_summary": _verdict_summary(fine_verdict),
        "levels": levels,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "us079_ablation_report_written",
        path=str(args.out_json),
        arms=list(runs),
        fine_verdict=report["fine_verdict_summary"],
    )


def _verdict_summary(verdict: object) -> dict[str, object] | None:
    """Flatten the fine-level verdict to a compact top-level summary.

    Args:
        verdict: The ``warm_start_verdict`` block (or ``None``).

    Returns:
        A compact ``{available, warm_start_hurts_conserved, ...}`` dict, or
        ``None`` when no fine verdict was computed.
    """
    if not isinstance(verdict, dict):
        return None
    return {
        "available": verdict.get("available"),
        "reason": verdict.get("reason"),
        "n_conserved_compared": verdict.get("n_conserved_compared"),
        "mean_f1_warmstart": verdict.get("mean_f1_warmstart"),
        "mean_f1_nowarmstart": verdict.get("mean_f1_nowarmstart"),
        "warm_start_hurts_conserved": verdict.get("warm_start_hurts_conserved"),
    }


if __name__ == "__main__":
    main()
