"""Figures for the per-parcel FarSLIP N-class sweep (US-036-b, notebook 06b).

Renders, from the real sweep CSV (``reports/farslip/metrics/parcel_sweep.csv``),
the two figures that tell the per-parcel story:

1. ``parcel_sweep_curve.png`` -- macro-F1 and macro-IoU vs N (the honest ceiling
   by cardinality), annotated with the per-N values.
2. ``parcel_vs_patch.png`` -- the headline bar: the per-parcel model at its
   sweet spot (N=4) against the old per-patch model, showing the jump that broke
   the ~4-class ceiling.

Eval/presentation only: reads metrics already computed on the H100, never trains.
Conventions: Polars I/O, matplotlib Agg, structlog, type hints, English
docstrings, Spanish prose in user-facing labels, no emojis.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["render_parcel_sweep_figures"]

#: macro-F1 of the OLD per-patch FarSLIP (US-037, 18-class eval) -- the baseline
#: the per-parcel model is compared against. Sourced from
#: ``reports/farslip/metrics/faithful_v2_per_class.csv`` (macro 0.164).
_PATCH_LEVEL_MACRO_F1: float = 0.164


def render_parcel_sweep_figures(
    sweep_csv: Path,
    out_dir: Path,
    *,
    patch_level_macro_f1: float = _PATCH_LEVEL_MACRO_F1,
) -> dict[str, Path]:
    """Render the two per-parcel sweep figures from the real CSV.

    Args:
        sweep_csv: path to the sweep CSV (``n_classes, macro_f1, macro_iou, ...``).
        out_dir: directory where the PNGs are written (created if missing).
        patch_level_macro_f1: macro-F1 of the old per-patch model for the
            headline comparison bar.

    Returns:
        Mapping ``{"curve": path, "parcel_vs_patch": path}`` of written PNGs.

    Raises:
        FileNotFoundError: if ``sweep_csv`` does not exist.
        ValueError: if the CSV lacks the required columns.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep CSV not found: {sweep_csv}")
    df = pl.read_csv(sweep_csv).sort("n_classes")
    for col in ("n_classes", "macro_f1", "macro_iou"):
        if col not in df.columns:
            raise ValueError(f"sweep CSV missing column {col!r}; got {df.columns}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = df["n_classes"].to_list()
    f1 = df["macro_f1"].to_list()
    iou = df["macro_iou"].to_list()

    # --- Figure 1: curve N vs macro-F1 / macro-IoU ---
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(n, f1, marker="o", color="#1f77b4", label="macro-F1")
    ax1.plot(n, iou, marker="s", linestyle="--", color="#ff7f0e", label="macro-IoU")
    for xi, yi in zip(n, f1, strict=True):
        ax1.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax1.set_xlabel("Numero de clases (N, ordenadas por frecuencia)")
    ax1.set_ylabel("Metrica (por parcela, fold de validacion)")
    ax1.set_title("FarSLIP por-parcela: techo real por cardinalidad")
    ax1.set_xticks(n)
    ax1.set_ylim(0.0, max(0.8, max(f1) + 0.1))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    curve_path = out_dir / "parcel_sweep_curve.png"
    fig1.savefig(curve_path, dpi=130)
    plt.close(fig1)

    # --- Figure 2: headline per-parcel (N=4) vs per-patch ---
    best_f1 = max(f1)
    best_n = n[f1.index(best_f1)]
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.4))
    bars = ax2.bar(
        [
            "FarSLIP por-patch\n(US-037, 18 clases)",
            f"FarSLIP por-parcela\n(N={best_n}, sweet spot)",
        ],
        [patch_level_macro_f1, best_f1],
        color=["#d62728", "#2ca02c"],
    )
    for b, v in zip(bars, [patch_level_macro_f1, best_f1], strict=True):
        ax2.annotate(
            f"{v:.3f}",
            (b.get_x() + b.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    factor = best_f1 / patch_level_macro_f1 if patch_level_macro_f1 else float("nan")
    ax2.set_ylabel("macro-F1")
    ax2.set_title(f"El grano-parcela rompe el techo: x{factor:.1f} en macro-F1")
    ax2.set_ylim(0.0, max(0.8, best_f1 + 0.12))
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    pvp_path = out_dir / "parcel_vs_patch.png"
    fig2.savefig(pvp_path, dpi=130)
    plt.close(fig2)

    logger.info(
        "parcel_sweep_figures_rendered",
        curve=str(curve_path),
        parcel_vs_patch=str(pvp_path),
        best_n=best_n,
        best_f1=round(best_f1, 4),
        factor=round(factor, 2),
    )
    return {"curve": curve_path, "parcel_vs_patch": pvp_path}
