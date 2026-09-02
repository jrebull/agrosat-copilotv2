"""Build the WorldCereal tropical transfer figure (Experimento 3, EPIC 12).

Reads ONLY the real artefacts produced by ``ml.transfer.worldcereal_tropical``
(no hand-typed numbers) and emits a two-panel figure under
``paper/figures/us-073-transfer/``:

- Left panel: the few-shot F1-macro-vs-k curve over the LOCAL tropical
  WorldCereal classes for Brazil (Cerrado) and India (Karnataka), with per-seed
  error bars, plus a dashed line at the fully-supervised in-domain ceiling.
- Right panel: the honest zero-shot bar -- the European PASTIS-18 AlphaEarth
  classifier applied to each tropical region, scored only on the SINGLE shared
  concept (maize / Corn), versus the maize base-rate (the trivial reference).

Inputs (all REAL, versioned under ``data/transfer/``):
- ``worldcereal_brazil_cerrado.parquet`` / ``worldcereal_india_karnataka.parquet``
  (joined WorldCereal label + AlphaEarth 64-dim).
- ``worldcereal_fewshot_results.parquet`` (Brazil curve, 3 seeds).
- ``worldcereal_fewshot_india.parquet`` (India curve, 3 seeds).

The zero-shot and in-domain numbers are recomputed from the datasets at build
time so the figure can never drift from a stale cached scalar. Deterministic
(Agg backend, fixed seeds), idempotent.

The figure is emitted in BOTH languages: the English version is the canonical
base file (``<name>.png`` / ``<name>.svg`` for the English paper) and the
Spanish version carries the ``_es`` suffix (``<name>_es.png`` / ``<name>_es.svg``
for the Spanish paper). Only visible strings change between languages; the
plotted numbers and drawing logic are identical. English identifiers, no emojis.
If an input is missing the script raises explicitly and fabricates nothing.

Usage::

    python -m scripts.build_worldcereal_tropical_figure --repo-root .
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import structlog

from ml.transfer.worldcereal_tropical import (
    summarize_curve,
    zero_shot_europe_to_tropics,
    zero_shot_separability,
)

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The two tropical regions plotted, with a language-keyed display label.
#: (english_label, spanish_label, dataset parquet, curve parquet, colour).
_REGIONS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Brazil (Cerrado)",
        "Brasil (Cerrado)",
        "worldcereal_brazil_cerrado.parquet",
        "worldcereal_fewshot_results.parquet",
        "#1b7837",
    ),
    (
        "India (Karnataka)",
        "India (Karnataka)",
        "worldcereal_india_karnataka.parquet",
        "worldcereal_fewshot_india.parquet",
        "#762a83",
    ),
)


@dataclass(frozen=True)
class FigureStrings:
    """Language-specific visible strings for the tropical transfer figure.

    All plotted numbers are language-invariant; only these strings differ
    between the English (canonical) and Spanish variants of the figure.

    Attributes:
        curve_xlabel: X-axis label of the few-shot curve panel.
        curve_ylabel: Y-axis label of the few-shot curve panel.
        curve_title: Title of the few-shot curve panel.
        zs_zero_shot_label: Legend label of the zero-shot F1 bars.
        zs_base_rate_label: Legend label of the base-rate reference bars.
        zs_ylabel: Y-axis label of the zero-shot panel.
        zs_title: Title of the zero-shot panel.
        suptitle: Overall figure title.
    """

    curve_xlabel: str
    curve_ylabel: str
    curve_title: str
    zs_zero_shot_label: str
    zs_base_rate_label: str
    zs_ylabel: str
    zs_title: str
    suptitle: str


#: Canonical English strings (base file) and Spanish translation (``_es``).
_STRINGS: dict[str, FigureStrings] = {
    "en": FigureStrings(
        curve_xlabel="Local samples per class (k, log scale)",
        curve_ylabel="F1-macro (local tropical classes)",
        curve_title=(
            "Few-shot over tropical WorldCereal classes\n"
            "(dashed line = in-domain supervised ceiling)"
        ),
        zs_zero_shot_label="Zero-shot F1 (Europe -> tropics, maize only)",
        zs_base_rate_label="Maize base rate (trivial reference)",
        zs_ylabel="Binary F1 for maize detection",
        zs_title=(
            "Zero-shot: European PASTIS-18 classifier\n"
            "applied to the tropics (only shared class: maize)"
        ),
        suptitle=(
            "Experiment 3 -- Multi-region transfer to the tropics (ESA WorldCereal, CC-BY-4.0)"
        ),
    ),
    "es": FigureStrings(
        curve_xlabel="Muestras locales por clase (k, escala log)",
        curve_ylabel="F1-macro (clases tropicales locales)",
        curve_title=(
            "Few-shot sobre clases WorldCereal tropicales\n"
            "(linea discontinua = techo supervisado in-domain)"
        ),
        zs_zero_shot_label="F1 zero-shot (Europa -> tropico, solo maiz)",
        zs_base_rate_label="Tasa base de maiz (referencia trivial)",
        zs_ylabel="F1 binario de deteccion de maiz",
        zs_title=(
            "Zero-shot: clasificador europeo PASTIS-18\n"
            "aplicado al tropico (unica clase compartida: maiz)"
        ),
        suptitle=(
            "Experimento 3 -- Transfer multi-region a zona tropical (ESA WorldCereal, CC-BY-4.0)"
        ),
    ),
}


def _require(path: Path) -> Path:
    """Return ``path`` or raise if it does not exist (no fabricated inputs).

    Args:
        path: Expected input file.

    Returns:
        The same path when it exists.

    Raises:
        FileNotFoundError: if the input is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required real artefact missing: {path}. Run "
            "ml.transfer.worldcereal_tropical.build_dataset first."
        )
    return path


def _region_label(region: tuple[str, str, str, str, str], lang: str) -> str:
    """Return the display label of a region for the given language.

    Args:
        region: A row of ``_REGIONS`` (en label, es label, ds, curve, colour).
        lang: Language code, ``"en"`` or ``"es"``.

    Returns:
        The English label when ``lang == "en"``, else the Spanish label.
    """
    return region[0] if lang == "en" else region[1]


def _render_figure(
    lang: str,
    strings: FigureStrings,
    transfer_dir: Path,
    out_dir: Path,
    *,
    dpi: int,
) -> tuple[Path, Path]:
    """Render one language variant of the two-panel figure.

    Args:
        lang: Language code, ``"en"`` (canonical base) or ``"es"``.
        strings: Visible strings for this language.
        transfer_dir: Directory holding the real input parquets.
        out_dir: Directory to write the figure files into.
        dpi: Raster resolution for the PNG output.

    Returns:
        Tuple ``(png_path, svg_path)`` of the written figure files.

    Raises:
        FileNotFoundError: if any required input parquet is absent.
    """
    fig, (ax_curve, ax_zs) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    zs_labels: list[str] = []
    zs_f1: list[float] = []
    zs_base: list[float] = []
    zs_colours: list[str] = []

    for region in _REGIONS:
        _, _, ds_name, curve_name, colour = region
        label = _region_label(region, lang)
        dataset = pl.read_parquet(_require(transfer_dir / ds_name))
        curve = pl.read_parquet(_require(transfer_dir / curve_name))
        summary = summarize_curve(curve).sort("k")

        ks = summary.get_column("k").to_list()
        means = summary.get_column("f1_mean").to_list()
        stds = summary.get_column("f1_std").to_list()
        ax_curve.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=colour,
            label=label,
        )

        ceiling = zero_shot_separability(dataset)["f1_macro_cv"]
        ax_curve.axhline(ceiling, ls="--", lw=1.0, color=colour, alpha=0.6)

        zs = zero_shot_europe_to_tropics(dataset)
        zs_labels.append(label)
        zs_f1.append(zs["maize_f1_zero_shot"])
        zs_base.append(zs["base_rate"])
        zs_colours.append(colour)
        logger.info(
            "worldcereal_figure_region",
            lang=lang,
            region=label,
            ceiling=round(ceiling, 4),
            zero_shot_maize_f1=round(zs["maize_f1_zero_shot"], 4),
        )

    ax_curve.set_xscale("log")
    ax_curve.set_xlabel(strings.curve_xlabel)
    ax_curve.set_ylabel(strings.curve_ylabel)
    ax_curve.set_title(strings.curve_title)
    ax_curve.set_ylim(0.0, 0.85)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="lower right")

    # Right panel: zero-shot maize-detection bars vs base rate.
    x = range(len(zs_labels))
    width = 0.38
    ax_zs.bar(
        [i - width / 2 for i in x],
        zs_f1,
        width=width,
        color=zs_colours,
        label=strings.zs_zero_shot_label,
    )
    ax_zs.bar(
        [i + width / 2 for i in x],
        zs_base,
        width=width,
        color="#bbbbbb",
        label=strings.zs_base_rate_label,
    )
    ax_zs.set_xticks(list(x))
    ax_zs.set_xticklabels(zs_labels)
    ax_zs.set_ylabel(strings.zs_ylabel)
    ax_zs.set_ylim(0.0, 0.6)
    ax_zs.set_title(strings.zs_title)
    ax_zs.grid(True, axis="y", alpha=0.3)
    ax_zs.legend(loc="upper left", fontsize=8)
    for i, v in enumerate(zs_f1):
        ax_zs.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(strings.suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    suffix = "" if lang == "en" else "_es"
    png = out_dir / f"worldcereal_tropical_transfer{suffix}.png"
    svg = out_dir / f"worldcereal_tropical_transfer{suffix}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    logger.info("worldcereal_figure_saved", lang=lang, png=str(png), svg=str(svg))
    return png, svg


def build_figure(*, repo_root: Path | None = None, dpi: int = 150) -> dict[str, tuple[Path, Path]]:
    """Render the two-panel figure in English (base) and Spanish (``_es``).

    Args:
        repo_root: Repository root. Defaults to the repo the script lives in.
        dpi: Raster resolution for the PNG output.

    Returns:
        Mapping ``{lang: (png_path, svg_path)}`` for ``"en"`` and ``"es"``.

    Raises:
        FileNotFoundError: if any required input parquet is absent.
    """
    root = (repo_root or _REPO_ROOT).resolve()
    transfer_dir = root / "data" / "transfer"
    out_dir = root / "paper" / "figures" / "us-073-transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[Path, Path]] = {}
    for lang in ("en", "es"):
        outputs[lang] = _render_figure(lang, _STRINGS[lang], transfer_dir, out_dir, dpi=dpi)
    return outputs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with ``repo_root`` and ``dpi`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the WorldCereal tropical transfer figure in English (base) and Spanish (_es)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (defaults to the repo the script lives in).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Raster resolution for the PNG output (default: 150).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_figure(repo_root=args.repo_root, dpi=args.dpi)
