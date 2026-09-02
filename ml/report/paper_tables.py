"""Data-driven LaTeX tables for the AgroSatCopilot paper (US-070, EPIC 11).

Every table is materialized from a **real artifact on disk** under ``reports/**``
(CSV/JSON produced by the training and evaluation pipelines). There is **no
hardcoded numeric literal**: each cell is read from its source file and only
formatted. This is the anti-hardcode contract verified by the unit tests
(``tests/ml/report/test_paper_tables.py``): the number written to the ``.tex``
must equal the number read from the source.

Covered tables (US-070 acceptance criteria):

- ``T1`` Foundation-model comparison (AlphaEarth ``SATELLITE_EMBEDDING/V1/ANNUAL``
  data v1.1, 64-dim, CC-BY-4.0 vs raw Sentinel-2 vs FarSLIP faithful).
- ``T2`` EPIC 5 individual segmentation models **re-scored with the US-030
  harness on fold-5** (18-class contiguous, 128px NEAREST; SegFormer is B0 RGB
  3-band, AnySat substitutes Swin-UNETR which was never trained).
- ``T3`` EPIC 6 ensembles (4 rubric strategies + incremental E-a/E-b + FarSLIP).
- ``T4`` LLM benchmark (Gemini 2.5 Pro frozen reasoner vs Qwen3-30B-A3B vLLM
  on-prem). AgroMind-IT/ES + error bars columns are blocked (US-068/US-069, H100).
- ``T5`` Tool ablation (tool selection / arg match / crop grounding / routing).
- ``Tx`` FarSLIP band ablation (faithful FarSLIP vs AlphaEarth separability).

The LaTeX style mirrors the hand-written
``paper/tables/us-023-preview/baseline_v2_comparison.tex`` (``booktabs`` rules).
Prose visible to the reader (captions) is in Spanish; identifiers and docstrings
are in English per the project language policy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "REPORTS_DIR",
    "build_ensemble_table",
    "build_farslip_band_ablation_table",
    "build_fm_comparison_table",
    "build_llm_benchmark_table",
    "build_segmentation_table",
    "build_tool_ablation_table",
    "render_latex_table",
    "write_all_tables",
]

#: Repository ``reports/`` root. All table sources live below this directory.
REPORTS_DIR = Path("reports")

#: Pending marker used for blocked cells (H100 / GEE / human review). The tests
#: assert it never appears where a real artifact value is available.
PENDING = r"\textit{pendiente}"


def _fmt(value: Any, *, decimals: int = 4) -> str:
    """Format a scalar cell for LaTeX without inventing values.

    Args:
        value: Raw cell value (number, string, ``None`` or NaN).
        decimals: Decimal places for floats.

    Returns:
        LaTeX-escaped string; ``NaN``/``None`` render as ``NaN`` (real absence,
        not a fabricated zero).
    """
    if value is None:
        return "NaN"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.{decimals}f}"
    if isinstance(value, int):
        return str(value)
    return _escape_latex(str(value))


def _escape_latex(text: str) -> str:
    """Escape the LaTeX special characters that appear in our labels.

    Args:
        text: Raw string (model name, scenario, note).

    Returns:
        String safe to embed in a ``tabular`` cell.
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_latex_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str,
    label: str,
    column_spec: str | None = None,
) -> str:
    """Render a ``booktabs`` table mirroring the project's reference style.

    Args:
        header: Column titles (already LaTeX-safe).
        rows: Pre-formatted cells (already LaTeX-safe).
        caption: Spanish caption shown to the reader.
        label: LaTeX ``\\label`` (e.g. ``tab:fm_comparison``).
        column_spec: ``tabular`` column spec; defaults to first column left,
            the rest right-aligned.

    Returns:
        Full ``table`` environment as a string ending with a newline.
    """
    n_cols = len(header)
    if column_spec is None:
        column_spec = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a CSV artifact as a Polars DataFrame.

    Args:
        path: CSV path under ``reports/``.

    Returns:
        Eagerly loaded DataFrame.

    Raises:
        FileNotFoundError: if the artifact is missing (a blocked source).
    """
    if not path.exists():
        raise FileNotFoundError(f"missing real artifact: {path}")
    return pl.read_csv(path)


# --------------------------------------------------------------------------- #
# T1 -- Foundation-model comparison
# --------------------------------------------------------------------------- #
def build_fm_comparison_table(
    *,
    s2_csv: Path = REPORTS_DIR / "baseline" / "comparison_alphaearth_vs_s2.csv",
    farslip_csv: Path = REPORTS_DIR
    / "farslip"
    / "metrics"
    / "us037_farslip_fiel_vs_alphaearth.csv",
) -> str:
    """Build T1: AlphaEarth v1.1 vs raw Sentinel-2 vs FarSLIP faithful.

    Reads the tabular FM comparison (XGB/RF over AlphaEarth 64-dim, the combined
    187-feature vector and raw Sentinel-2) and the FarSLIP-faithful separability
    row, so the table contrasts the foundation models actually used. No literal
    is hardcoded.

    Args:
        s2_csv: ``comparison_alphaearth_vs_s2.csv`` source.
        farslip_csv: FarSLIP-faithful vs AlphaEarth separability source.

    Returns:
        LaTeX ``table`` string.
    """
    df = _read_csv(s2_csv)
    header = ["Representacion", "Modelo", "n\\_feat", "F1-macro", "F1-w", "mIoU", "t (s)"]
    rows: list[list[str]] = []
    for r in df.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["scenario"])),
                _escape_latex(str(r["model"])),
                _fmt(r["n_features"]),
                _fmt(r["f1_macro"]),
                _fmt(r["f1_weighted"]),
                _fmt(r["miou"]),
                _fmt(r["train_time_s"], decimals=1),
            ]
        )
    # FarSLIP-faithful separability rows (silhouette + clustered F1-macro).
    fdf = _read_csv(farslip_csv)
    for r in fdf.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["space"])),
                "KMeans",
                _fmt(r["n_dims"]),
                _fmt(r["f1_macro_mean"]),
                "NaN",
                "NaN",
                "NaN",
            ]
        )
    caption = (
        "Comparativa de modelos fundacionales. AlphaEarth = "
        "SATELLITE\\_EMBEDDING/V1/ANNUAL, data v1.1, 64-dim, CC-BY-4.0. "
        "Sentinel-2 crudo 10 bandas y FarSLIP fiel (separabilidad por KMeans) "
        "como referencias. Cifras leidas de reports/ (sin hardcode)."
    )
    return render_latex_table(header, rows, caption=caption, label="tab:fm_comparison")


# --------------------------------------------------------------------------- #
# T2 -- EPIC 5 individual models re-scored fold-5
# --------------------------------------------------------------------------- #
def build_segmentation_table(
    metrics_csv: Path = REPORTS_DIR / "segmentation" / "metrics" / "model_comparison_fold5.csv",
) -> str:
    """Build T2: EPIC 5 individual models re-scored with the US-030 harness.

    Apples-to-apples fold-5 re-score (18-class contiguous, 128px NEAREST). The
    ``in_channels`` / ``needs_resize`` columns make explicit that SegFormer ran
    as B0 RGB 3-band and the rest at 10 bands. AnySat substitutes Swin-UNETR
    (never trained). Rows are sorted by mIoU descending.

    Args:
        metrics_csv: ``model_comparison_fold5.csv`` source.

    Returns:
        LaTeX ``table`` string.
    """
    df = _read_csv(metrics_csv).sort("miou", descending=True)
    header = [
        "Modelo",
        "mIoU",
        "F1-macro",
        "Pix-acc",
        "Kappa",
        "Bal-acc",
        "Bandas",
        "Resize",
    ]
    rows: list[list[str]] = []
    for r in df.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["model"])),
                _fmt(r["miou"]),
                _fmt(r["f1_macro"]),
                _fmt(r["pixel_accuracy"]),
                _fmt(r["cohen_kappa"]),
                _fmt(r["balanced_acc"]),
                _fmt(int(r["in_channels"])),
                "si" if bool(r["needs_resize"]) else "no",
            ]
        )
    caption = (
        "Modelos individuales EPIC 5 re-scoreados con el harness unico de "
        "US-030 (fold-5, 18 clases contiguas, 128px NEAREST). SegFormer = B0 "
        "RGB 3-banda; AnySat sustituye a Swin-UNETR (nunca entrenado). "
        "Cifras leidas de model\\_comparison\\_fold5.csv."
    )
    return render_latex_table(
        header, rows, caption=caption, label="tab:segmentation_individual_fold5"
    )


# --------------------------------------------------------------------------- #
# T3 -- EPIC 6 ensembles
# --------------------------------------------------------------------------- #
def build_ensemble_table(
    *,
    us040_csv: Path = REPORTS_DIR / "ensemble" / "metrics" / "comparison_us040.csv",
    ea_eb_csv: Path = REPORTS_DIR / "ensemble" / "metrics" / "us041_042_ea_eb_results.csv",
    farslip_grid_csv: Path = REPORTS_DIR / "ensemble" / "metrics" / "us043_farslip_grid.csv",
) -> str:
    """Build T3: EPIC 6 ensembles (rubric 4 + E-a/E-b + FarSLIP grid).

    Concatenates the three real ensemble artifacts into a single comparison.
    The FarSLIP grid contributes the champion Stacking-5 row and the Blending-5
    regression. No literal is hardcoded.

    Args:
        us040_csv: 4 rubric strategies (Voting/Bagging/Stacking/Blending).
        ea_eb_csv: incremental E-a / E-b fusion experiments.
        farslip_grid_csv: FarSLIP grid (Stacking/Blending 3 vs 5 members).

    Returns:
        LaTeX ``table`` string.
    """
    header = ["Estrategia / fuente", "F1-macro", "Accuracy", "Nota"]
    rows: list[list[str]] = []

    df40 = _read_csv(us040_csv)
    for r in df40.iter_rows(named=True):
        flag = " (elegido)" if bool(r["chosen"]) else ""
        rows.append(
            [
                _escape_latex(str(r["model"]) + flag),
                _fmt(r["f1_macro"]),
                _fmt(r["accuracy"]),
                "US-040",
            ]
        )

    dfea = _read_csv(ea_eb_csv)
    for r in dfea.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["modelo"])),
                _fmt(r["f1_macro"]),
                _fmt(r["accuracy"]),
                _escape_latex(_short(str(r["nota"]))),
            ]
        )

    dfg = _read_csv(farslip_grid_csv)
    for r in dfg.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["modelo"])),
                _fmt(r["f1_macro"]),
                _fmt(r["accuracy"]),
                "delta " + _fmt(r["delta_farslip"]),
            ]
        )
    caption = (
        "Ensambles EPIC 6: 4 estrategias de rubrica (US-040), incrementales "
        "E-a/E-b (US-041/042) y grid FarSLIP (US-043). Campeon: Stacking-5 "
        "(+FarSLIP). Cifras leidas de reports/ensemble/metrics/."
    )
    return render_latex_table(header, rows, caption=caption, label="tab:ensembles_e6")


def _short(note: str, *, limit: int = 60) -> str:
    """Truncate a free-text note so the LaTeX cell stays readable.

    Args:
        note: Source note string.
        limit: Maximum characters before ellipsis.

    Returns:
        Possibly truncated note.
    """
    note = note.strip()
    return note if len(note) <= limit else note[: limit - 1] + "..."


# --------------------------------------------------------------------------- #
# T4 -- LLM benchmark
# --------------------------------------------------------------------------- #
def build_llm_benchmark_table(
    eval_json: Path = REPORTS_DIR / "agent_bench" / "us049_system_eval.json",
    *,
    models: Sequence[str] = ("gemini", "qwen"),
) -> str:
    """Build T4: LLM benchmark (Gemini frozen reasoner vs Qwen on-prem).

    Reads the per-model evaluation block from ``us049_system_eval.json``. The
    AgroMind-IT/ES columns and 3-run error bars (Wilcoxon) are blocked
    (US-068/US-069, H100) and rendered as ``pendiente`` -- never fabricated.

    Args:
        eval_json: ``us049_system_eval.json`` source.
        models: Model keys to include (default Gemini + Qwen, both real today).

    Returns:
        LaTeX ``table`` string.
    """
    if not eval_json.exists():
        raise FileNotFoundError(f"missing real artifact: {eval_json}")
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    header = [
        "Modelo",
        "Tool-sel",
        "Arg-match",
        "Crop-match",
        "Routing",
        "Halluc-RAG",
        "AgroMind IT/ES",
    ]
    label_map = {
        "gemini": "Gemini 2.5 Pro (frozen)",
        "qwen": "Qwen3-30B-A3B (on-prem)",
    }
    rows: list[list[str]] = []
    for key in models:
        block = data.get(key)
        if block is None:
            continue
        tc = block.get("tool_calling", {})
        gc = block.get("grounded_crop", {})
        rag = block.get("rag_ab", {})
        rows.append(
            [
                _escape_latex(label_map.get(key, key)),
                _fmt(_mean(tc, "tool_selection_accuracy")),
                _fmt(_mean(tc, "arg_match_accuracy")),
                _fmt(_mean(gc, "crop_match_accuracy")),
                _fmt(_mean(gc, "routing_accuracy")),
                _fmt(_mean(rag, "hallucination_rate_grounded")),
                PENDING,
            ]
        )
    caption = (
        "Benchmark LLMs del copiloto: Gemini 2.5 Pro (reasoner frozen, patron "
        "Be My Eyes) vs Qwen3-30B-A3B vLLM GPTQ-Int4 on-prem (soberania de "
        "datos). Columna AgroMind-IT/ES pendiente (US-068/US-069, H100). "
        "Cifras leidas de us049\\_system\\_eval.json."
    )
    return render_latex_table(header, rows, caption=caption, label="tab:llm_benchmark")


def _mean(block: dict[str, Any], metric: str) -> float | None:
    """Extract the ``mean`` of a nested ``{metric: {mean, std}}`` entry.

    Args:
        block: One sub-dict of the eval JSON (``tool_calling`` etc.).
        metric: Metric key.

    Returns:
        The mean value, or ``None`` if absent.
    """
    entry = block.get(metric)
    if isinstance(entry, dict):
        return entry.get("mean")
    return None


# --------------------------------------------------------------------------- #
# T5 -- Tool ablation
# --------------------------------------------------------------------------- #
def build_tool_ablation_table(
    eval_json: Path = REPORTS_DIR / "agent_bench" / "us049_system_eval.json",
    *,
    model: str = "gemini",
) -> str:
    """Build T5: tool-use ablation for the agent (Gemini real run).

    One row per measured tool-use dimension (native calling, selection accuracy,
    argument match, no-call rate, crop grounding, routing, faithfulness) read
    from the real Gemini block. The on-prem variant column is blocked.

    Args:
        eval_json: ``us049_system_eval.json`` source.
        model: Model key whose tool metrics are tabulated.

    Returns:
        LaTeX ``table`` string.
    """
    if not eval_json.exists():
        raise FileNotFoundError(f"missing real artifact: {eval_json}")
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    block = data[model]
    tc = block.get("tool_calling", {})
    gc = block.get("grounded_crop", {})
    dimensions: list[tuple[str, float | None]] = [
        ("Llamada nativa a tool", _mean(tc, "tool_calling_native")),
        ("Tasa de fallo de parseo", _mean(tc, "parse_failure_rate")),
        ("Exactitud de seleccion de tool", _mean(tc, "tool_selection_accuracy")),
        ("Coincidencia de argumentos", _mean(tc, "arg_match_accuracy")),
        ("Tasa sin llamada", _mean(tc, "no_call_rate")),
        ("Crop grounding (match)", _mean(gc, "crop_match_accuracy")),
        ("Routing accuracy", _mean(gc, "routing_accuracy")),
        ("Faithfulness de crop", _mean(gc, "faithfulness_crop")),
    ]
    header = ["Dimension de uso de tools", "Gemini", "Qwen (on-prem)"]
    rows = [[_escape_latex(name), _fmt(val), PENDING] for name, val in dimensions]
    caption = (
        "Ablacion de uso de herramientas del agente ADK (corrida real Gemini, "
        "20 escenarios). Variante on-prem Qwen pendiente (US-069, H100). "
        "Cifras leidas de us049\\_system\\_eval.json."
    )
    return render_latex_table(header, rows, caption=caption, label="tab:tool_ablation")


# --------------------------------------------------------------------------- #
# Tx -- FarSLIP band ablation
# --------------------------------------------------------------------------- #
def build_farslip_band_ablation_table(
    *,
    sweep_csv: Path = REPORTS_DIR / "farslip" / "metrics" / "parcel_sweep.csv",
    faithful_csv: Path = REPORTS_DIR
    / "farslip"
    / "metrics"
    / "us037_farslip_fiel_vs_alphaearth.csv",
) -> str:
    """Build Tx: FarSLIP band / cardinality ablation.

    The full 3-band-space ablation (rgb vs nir-rgb false-color vs 4band-pheno)
    requires per-variant embedding re-extraction on H100 (blocked, B-070-5). We
    materialize the available real evidence: the parcel-level class-cardinality
    sweep and the FarSLIP-faithful vs AlphaEarth separability, which already
    quantify the FarSLIP signal vs AlphaEarth.

    Args:
        sweep_csv: parcel cardinality sweep source.
        faithful_csv: FarSLIP-faithful vs AlphaEarth separability source.

    Returns:
        LaTeX ``table`` string.
    """
    sdf = _read_csv(sweep_csv)
    header = ["n\\_clases", "macro-F1", "macro-IoU", "bien resueltas", "parcelas eval"]
    rows: list[list[str]] = []
    for r in sdf.iter_rows(named=True):
        rows.append(
            [
                _fmt(int(r["n_classes"])),
                _fmt(r["macro_f1"]),
                _fmt(r["macro_iou"]),
                _fmt(int(r["n_well_resolved"])),
                _fmt(int(r["n_eval_parcels"])),
            ]
        )
    # Separability contrast (FarSLIP faithful vs AlphaEarth) as a trailing block.
    fdf = _read_csv(faithful_csv)
    for r in fdf.iter_rows(named=True):
        rows.append(
            [
                _escape_latex(str(r["space"])),
                _fmt(r["f1_macro_mean"]),
                _fmt(r["silhouette"]),
                "-",
                _fmt(int(r["n_samples"])),
            ]
        )
    caption = (
        "Ablacion FarSLIP: barrido de cardinalidad de clases a nivel parcela y "
        "separabilidad FarSLIP fiel vs AlphaEarth (silhouette/F1-macro KMeans). "
        "Las 3 variantes de banda completas (rgb/nir-rgb/4band-pheno) requieren "
        "re-extraccion en H100 (B-070-5). Cifras leidas de reports/farslip/."
    )
    return render_latex_table(header, rows, caption=caption, label="tab:farslip_band_ablation")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def write_all_tables(out_dir: Path = Path("paper/tables/us-070")) -> dict[str, Path]:
    """Generate every paper table whose real source exists.

    Tables backed by a missing artifact (blocked) are skipped and logged; they
    are documented in ``docs/blockers/epic11-notas.md``. The function never
    fabricates a table.

    Args:
        out_dir: Output directory for the ``.tex`` files.

    Returns:
        Mapping of table stem -> written path (only the ones produced).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    builders: dict[str, Callable[[], str]] = {
        "fm_comparison": build_fm_comparison_table,
        "segmentation_individual_fold5": build_segmentation_table,
        "ensembles_e6": build_ensemble_table,
        "llm_benchmark": build_llm_benchmark_table,
        "tool_ablation": build_tool_ablation_table,
        "farslip_band_ablation": build_farslip_band_ablation_table,
    }
    written: dict[str, Path] = {}
    for stem, builder in builders.items():
        try:
            tex = builder()
        except FileNotFoundError as exc:
            logger.warning("paper_table_skipped", table=stem, reason=str(exc))
            continue
        path = out_dir / f"{stem}.tex"
        path.write_text(tex, encoding="utf-8")
        written[stem] = path
        logger.info("paper_table_written", table=stem, path=str(path))
    return written


if __name__ == "__main__":  # pragma: no cover - manual entry point
    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    produced = write_all_tables()
    for name, p in produced.items():
        logger.info("table", name=name, path=str(p))
