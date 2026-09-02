"""Bilingual builders for the two paper figures the main localization pass did
not cover: the winner cardinality curve (appendix) and the EuroCropsML
AlphaEarth-vs-Sentinel-2 few-shot comparison (multi-region).

Each figure is emitted twice: English as the canonical base name
``<stem>.png/.svg`` (for the EN manuscript) and Spanish as ``<stem>_es.png/.svg``
(for the ES mirror). Only visible strings change per language; every number is
read from the real repo artefacts, nothing is hand-typed.

Usage::

    poetry run python -m scripts.build_paper_straggler_figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

Lang = Literal["en", "es"]
LANGS: tuple[Lang, ...] = ("en", "es")

_ROOT = Path(__file__).resolve().parents[1]
_FIG_DIR = _ROOT / "paper" / "figures" / "us-070"

# Paper style (serif, 300 dpi) consistent with ml/report/paper_figures.py.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

_GREEN = "#2f7d52"
_BLUE = "#4c78a8"
_AMBER = "#e0a53f"
_RED = "#c44e52"


def _stem(stem: str, lang: Lang) -> Path:
    return _FIG_DIR / (stem if lang == "en" else f"{stem}_es")


def _save(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in (".png", ".svg"):
        p = stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        out.append(str(p.relative_to(_ROOT)))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 1. Winner cardinality curve (macro-F1 vs coverage as a function of top-k crops)
# ---------------------------------------------------------------------------
_CARD_STRINGS: dict[Lang, dict[str, str]] = {
    "en": {
        "title": "Deployed Voting-3 cardinality: cumulative macro-F1 by classes kept",
        "xlabel": "k (number of best-resolved classes kept)",
        "ylabel_l": "cumulative macro-F1 over the kept k classes",
        "annot": "k=12 (production): F1 {f1:.3f}",
        "attrib": "Deployed Voting-3 ensemble; held-out fold. AlphaEarth CC-BY-4.0.",
    },
    "es": {
        "title": "Cardinalidad de Voting-3 desplegado: F1 macro acumulado por clases retenidas",
        "xlabel": "k (número de clases mejor resueltas retenidas)",
        "ylabel_l": "F1 macro acumulado sobre las k clases retenidas",
        "annot": "k=12 (producción): F1 {f1:.3f}",
        "attrib": "Ensamble Voting-3 desplegado; fold de prueba retenido. AlphaEarth CC-BY-4.0.",
    },
}


def build_cardinality(lang: Lang) -> list[str]:
    # Deployed Voting-3 cardinality curve. reports/voting_new/cardinalidad.json
    # holds ["new"] rows [k, cumulative_macro_f1, cumulative_secondary, class_name].
    import json

    data = json.loads(
        (_ROOT / "reports" / "voting_new" / "cardinalidad.json").read_text(encoding="utf-8")
    )
    rows = data["new"]
    k = [int(r[0]) for r in rows]
    f1 = [float(r[1]) for r in rows]
    s = _CARD_STRINGS[lang]
    f1_12 = next(float(r[1]) for r in rows if int(r[0]) == 12)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(k, f1, "-o", color=_GREEN)
    ax.set_xlabel(s["xlabel"])
    ax.set_ylabel(s["ylabel_l"], color=_GREEN)
    ax.tick_params(axis="y", labelcolor=_GREEN)
    ax.set_xticks(k)
    ax.axhline(0.90, color="#999", ls=":", lw=1.0)
    ax.axvline(12, color=_AMBER, ls="--", lw=1.2)
    ax.annotate(
        s["annot"].format(f1=f1_12),
        xy=(12, f1_12),
        xytext=(9.3, f1_12 - 0.045),
        color=_AMBER,
        fontsize=9,
    )
    ax.set_title(s["title"])
    ax.annotate(s["attrib"], xy=(0.0, -0.16), xycoords="axes fraction", fontsize=7, color="#888")
    fig.tight_layout()
    return _save(fig, _stem("cardinality_curve", lang))


# ---------------------------------------------------------------------------
# 2. EuroCropsML AlphaEarth vs raw Sentinel-2 few-shot (LV -> EE)
# ---------------------------------------------------------------------------
_AE_STRINGS: dict[Lang, dict[str, str]] = {
    "en": {
        "title": "EuroCropsML LV -> EE few-shot: AlphaEarth vs. raw Sentinel-2",
        "xlabel": "k (shots per class, target = Estonia)",
        "ylabel": "F1-macro (Estonia held-out query set)",
        "ae": "AlphaEarth (LV -> EE)",
        "s2": "XGBoost raw S2 (LV -> EE)",
        "attrib": "EuroCropsML (Reuss et al. 2025, CC-BY-SA-4.0). Mean of 3 seeds.",
    },
    "es": {
        "title": "EuroCropsML LV -> EE few-shot: AlphaEarth vs. Sentinel-2 crudo",
        "xlabel": "k (ejemplos por clase, objetivo = Estonia)",
        "ylabel": "F1-macro (Estonia, conjunto de consulta retenido)",
        "ae": "AlphaEarth (LV -> EE)",
        "s2": "XGBoost S2 crudo (LV -> EE)",
        "attrib": "EuroCropsML (Reuss et al. 2025, CC-BY-SA-4.0). Promedio de 3 semillas.",
    },
}


def build_ae_vs_s2(lang: Lang) -> list[str]:
    # Reuse the same delta parquet that documents the AlphaEarth-vs-S2 curve.
    df = pl.read_parquet(_ROOT / "data" / "transfer" / "eurocropsml_alphaearth_vs_s2_delta.parquet")
    # expected columns: k, f1_alphaearth, f1_s2 (fallback to any close names)
    cols = {c.lower(): c for c in df.columns}
    kc = cols.get("k")
    aec = next((cols[c] for c in cols if "ae_f1" in c or "alphaearth" in c), None)
    s2c = next((cols[c] for c in cols if "s2_f1" in c or "raw" in c or "sentinel" in c), None)
    if not (kc and aec and s2c):
        logger.warning("ae_vs_s2_cols_unexpected", cols=df.columns)
        # nothing to draw honestly; skip
        return []
    df = df.sort(kc)
    k = df[kc].to_list()
    ae = df[aec].to_list()
    s2 = df[s2c].to_list()
    s = _AE_STRINGS[lang]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(k, ae, "-o", color=_GREEN, label=s["ae"])
    ax.plot(k, s2, "-s", color=_RED, label=s["s2"])
    ax.set_xscale("log")
    ax.set_xlabel(s["xlabel"])
    ax.set_ylabel(s["ylabel"])
    ax.set_title(s["title"])
    ax.legend(loc="lower right", fontsize=9)
    ax.annotate(s["attrib"], xy=(0.0, -0.16), xycoords="axes fraction", fontsize=7, color="#888")
    fig.tight_layout()
    return _save(fig, _stem("eurocropsml_ae_vs_s2", lang))


def main() -> int:
    written: list[str] = []
    for lang in LANGS:
        written += build_cardinality(lang)
        written += build_ae_vs_s2(lang)
    logger.info("straggler_figures_done", n=len(written), files=written)
    for w in written:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
