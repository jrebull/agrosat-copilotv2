"""HTML comparison report for the agent benchmark (US-049, AC-4/AC-5).

Renders a self-contained HTML report comparing the reasoner variants
(``gemini`` / ``qwen`` / ``gemma-base``) across the two benchmarks
(``AgroMind`` / ``GeoAnalystBench``). For every (variant, benchmark, metric)
cell it shows the mean and standard deviation aggregated over the evaluation
seeds (error bars, AC-4), and it flags the rubric targets (AC-5) with a textual
check / cross marker:

- AgroMind exact-match >= 0.75 for ``gemini`` (multimodal full subset).
- AgroMind exact-match >= 0.70 for ``qwen`` (text-only subset, documented).
- GeoAnalystBench pass-rate >= 0.65 for every variant.

A matplotlib grouped bar chart with error bars is embedded inline as a base64
PNG so the report is a single portable file (no external image assets).

Project conventions: ``matplotlib`` with the ``Agg`` backend (no display);
identifiers and docstrings in English; visible prose (titles, table headers,
captions) in Spanish; ``structlog`` for logging (never ``print``); no emojis.
"""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import structlog

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Mapping, Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_REPORT_DIR",
    "RUBRIC_TARGETS",
    "build_report_html",
]

#: Default output folder for the agent benchmark report.
DEFAULT_REPORT_DIR: Path = Path("reports/agent_bench")

#: Rubric targets (AC-5). Keyed by (variant, benchmark, metric) -> minimum.
#: Only the cells listed here get a pass/fail marker; everything else is shown
#: without a target.
RUBRIC_TARGETS: dict[tuple[str, str, str], float] = {
    ("gemini", "AgroMind", "exact_match"): 0.75,
    ("qwen", "AgroMind", "exact_match"): 0.70,
    # Multimodal on-prem comparative (Qwen3.6-VL): it sees AgroMind images, so it
    # gets the same multimodal AgroMind target. The text-only ``qwen`` keeps its
    # separate lower target because it is scored on the textual subset only.
    ("qwen36-vl", "AgroMind", "exact_match"): 0.70,
    ("gemini", "GeoAnalystBench", "pass_rate"): 0.65,
    ("qwen", "GeoAnalystBench", "pass_rate"): 0.65,
    ("gemma-base", "GeoAnalystBench", "pass_rate"): 0.65,
    ("qwen36-vl", "GeoAnalystBench", "pass_rate"): 0.65,
}

#: Textual markers (no emojis, per project rules).
_OK_MARKER = "[OK]"
_FAIL_MARKER = "[X]"
_NA_MARKER = "n/a"

#: Benchmark display order in the report.
_BENCHMARK_ORDER: tuple[str, ...] = ("AgroMind", "GeoAnalystBench")

#: Metric used for the headline bar chart per benchmark.
_HEADLINE_METRIC: dict[str, str] = {
    "AgroMind": "exact_match",
    "GeoAnalystBench": "pass_rate",
}


def _fmt_mean_std(mean: float, std: float) -> str:
    """Format a ``mean +- std`` cell, rendering NaN as ``n/a``.

    Args:
        mean: Aggregated mean over seeds.
        std: Aggregated standard deviation over seeds.

    Returns:
        A display string such as ``"0.812 +- 0.014"`` or ``"n/a"``.
    """
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return _NA_MARKER
    std_val = 0.0 if std is None or math.isnan(std) else std
    return f"{mean:.3f} +- {std_val:.3f}"


def _target_marker(variant: str, benchmark: str, metric: str, mean: float) -> str:
    """Return the rubric pass/fail marker for a metric cell, if any.

    Args:
        variant: Reasoner variant name.
        benchmark: Benchmark name.
        metric: Metric name.
        mean: Aggregated mean value for the cell.

    Returns:
        ``""`` when the cell has no rubric target, ``n/a`` when the value is
        NaN, otherwise the pass/fail marker plus the target threshold.
    """
    target = RUBRIC_TARGETS.get((variant, benchmark, metric))
    if target is None:
        return ""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return f"{_NA_MARKER} (>= {target:.2f})"
    mark = _OK_MARKER if mean >= target else _FAIL_MARKER
    return f"{mark} (>= {target:.2f})"


def _collect_metric_names(results: Mapping[str, Any]) -> dict[str, list[str]]:
    """Collect the union of metric names present per benchmark.

    Args:
        results: The nested results mapping
            ``{variant: {benchmark: {metric: {"mean", "std"}}}}``.

    Returns:
        A mapping ``{benchmark: [metric_name, ...]}`` with a stable order
        (rubric headline metric first, then the rest alphabetically).
    """
    per_benchmark: dict[str, set[str]] = {}
    for benchmarks in results.values():
        for benchmark, metrics in benchmarks.items():
            per_benchmark.setdefault(benchmark, set()).update(metrics.keys())

    ordered: dict[str, list[str]] = {}
    for benchmark, names in per_benchmark.items():
        headline = _HEADLINE_METRIC.get(benchmark)
        rest = sorted(n for n in names if n != headline)
        ordered[benchmark] = ([headline] if headline in names else []) + rest
    return ordered


def _get_cell(
    results: Mapping[str, Any], variant: str, benchmark: str, metric: str
) -> tuple[float, float]:
    """Extract the ``(mean, std)`` for one (variant, benchmark, metric) cell.

    Args:
        results: The nested results mapping.
        variant: Reasoner variant name.
        benchmark: Benchmark name.
        metric: Metric name.

    Returns:
        A ``(mean, std)`` tuple; missing cells return ``(nan, nan)``.
    """
    cell = results.get(variant, {}).get(benchmark, {}).get(metric)
    if cell is None:
        return math.nan, math.nan
    mean = float(cell.get("mean", math.nan))
    std = float(cell.get("std", math.nan))
    return mean, std


def _render_headline_chart(results: Mapping[str, Any], variants: Sequence[str]) -> str:
    """Render the grouped headline bar chart with error bars as a base64 PNG.

    One bar group per benchmark, one bar per variant, using the rubric headline
    metric of each benchmark (exact-match for AgroMind, pass-rate for
    GeoAnalystBench) with the seed standard deviation as the error bar.

    Args:
        results: The nested results mapping.
        variants: Ordered variant names to plot.

    Returns:
        An ``data:image/png;base64,...`` URI string ready for an ``<img>`` tag.
    """
    benchmarks = [b for b in _BENCHMARK_ORDER if b in _HEADLINE_METRIC]
    n_groups = len(benchmarks)
    n_variants = len(variants)
    width = 0.8 / max(n_variants, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_positions = list(range(n_groups))
    for v_idx, variant in enumerate(variants):
        means: list[float] = []
        errs: list[float] = []
        for benchmark in benchmarks:
            metric = _HEADLINE_METRIC[benchmark]
            mean, std = _get_cell(results, variant, benchmark, metric)
            means.append(0.0 if math.isnan(mean) else mean)
            errs.append(0.0 if math.isnan(std) else std)
        offsets = [x + (v_idx - (n_variants - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, means, width=width, yerr=errs, capsize=4, label=variant)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{b}\n({_HEADLINE_METRIC[b]})" for b in benchmarks])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Puntuacion (media sobre seeds)")
    ax.set_title("Comparativa por variante con barras de error")
    ax.legend(title="Variante")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_table(
    results: Mapping[str, Any],
    variants: Sequence[str],
    metric_names: Mapping[str, list[str]],
) -> str:
    """Render the comparison HTML table (variant x benchmark x metric).

    Args:
        results: The nested results mapping.
        variants: Ordered variant names.
        metric_names: ``{benchmark: [metric, ...]}`` from
            :func:`_collect_metric_names`.

    Returns:
        The ``<table>`` HTML fragment as a string.
    """
    rows: list[str] = []
    header = (
        "<tr><th>Variante</th><th>Benchmark</th><th>Metrica</th>"
        "<th>Media +- Desv.</th><th>Objetivo rubrica</th></tr>"
    )
    for variant in variants:
        for benchmark in _BENCHMARK_ORDER:
            metrics = metric_names.get(benchmark, [])
            for metric in metrics:
                mean, std = _get_cell(results, variant, benchmark, metric)
                marker = _target_marker(variant, benchmark, metric, mean)
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(variant)}</td>"
                    f"<td>{html.escape(benchmark)}</td>"
                    f"<td>{html.escape(metric)}</td>"
                    f"<td>{html.escape(_fmt_mean_std(mean, std))}</td>"
                    f"<td>{html.escape(marker)}</td>"
                    "</tr>"
                )
    return f"<table>{header}{''.join(rows)}</table>"


def _render_answer_type_table(
    breakdown: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    """Render the AgroMind answer-type exact-match breakdown table.

    Shows, per variant and answer type, the item count and the mean exact-match,
    so the report explains WHY the AgroMind headline is depressed (the
    ``open_numeric_bbox`` / ``open_number`` buckets are exact-match hostile by
    construction).

    Args:
        breakdown: ``{variant: {answer_type: {"n", "exact_match_mean"}}}`` from
            the dumped trace read-back.

    Returns:
        The ``<table>`` HTML fragment as a string (empty paragraph when there is
        nothing to show).
    """
    rows: list[str] = []
    for variant in sorted(breakdown):
        per_type = breakdown[variant]
        for answer_type in sorted(per_type):
            stats = per_type[answer_type]
            n = int(stats.get("n", 0.0))
            mean = float(stats.get("exact_match_mean", 0.0))
            rows.append(
                "<tr>"
                f"<td>{html.escape(variant)}</td>"
                f"<td>{html.escape(answer_type)}</td>"
                f"<td>{n}</td>"
                f"<td>{mean:.3f}</td>"
                "</tr>"
            )
    if not rows:
        return "<p>Sin traza por item para desglosar.</p>"
    header = (
        "<tr><th>Variante</th><th>Tipo de respuesta</th><th>N</th><th>Exact-match (media)</th></tr>"
    )
    return f"<table>{header}{''.join(rows)}</table>"


def _render_examples(
    examples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> str:
    """Render the per-variant inference examples section (hits and misses).

    For AgroMind shows question/gold/prediction/score; for GeoAnalystBench shows
    the workflow similarity, CodeBLEU and pass flag. Selection is
    order-deterministic (the dump preserves the fixed-seed iteration order).

    Args:
        examples: ``{variant: {benchmark: [record, ...]}}`` from the read-back.

    Returns:
        The HTML fragment (heading-per-variant + tables) as a string.
    """
    blocks: list[str] = []
    for variant in sorted(examples):
        per_benchmark = examples[variant]
        for benchmark in ("AgroMind", "GeoAnalystBench"):
            records = per_benchmark.get(benchmark, [])
            if not records:
                continue
            blocks.append(f"<h3>{html.escape(variant)} - {html.escape(benchmark)}</h3>")
            if benchmark == "GeoAnalystBench":
                header = (
                    "<tr><th>Pregunta</th><th>Gold</th><th>Prediccion</th>"
                    "<th>Sim. flujo</th><th>CodeBLEU</th><th>Pasa</th></tr>"
                )
            else:
                header = (
                    "<tr><th>Pregunta</th><th>Gold</th><th>Prediccion</th>"
                    "<th>Tipo</th><th>Exact-match</th><th>Acierto</th></tr>"
                )
            rows: list[str] = []
            for record in records:
                prompt = html.escape(str(record.get("prompt", "")))
                gold = html.escape(str(record.get("gold", "")))
                prediction = html.escape(str(record.get("prediction", "")))
                if benchmark == "GeoAnalystBench":
                    sim = float(record.get("workflow_sim", 0.0))
                    bleu = float(record.get("codebleu", 0.0))
                    passed = "Si" if record.get("passed") else "No"
                    rows.append(
                        "<tr>"
                        f"<td><pre>{prompt}</pre></td>"
                        f"<td><pre>{gold}</pre></td>"
                        f"<td><pre>{prediction}</pre></td>"
                        f"<td>{sim:.3f}</td><td>{bleu:.3f}</td>"
                        f"<td>{html.escape(passed)}</td>"
                        "</tr>"
                    )
                else:
                    answer_type = html.escape(str(record.get("answer_type", "")))
                    em = float(record.get("exact_match", 0.0))
                    acierto = "Si" if record.get("correct") else "No"
                    rows.append(
                        "<tr>"
                        f"<td><pre>{prompt}</pre></td>"
                        f"<td><pre>{gold}</pre></td>"
                        f"<td><pre>{prediction}</pre></td>"
                        f"<td>{answer_type}</td><td>{em:.3f}</td>"
                        f"<td>{html.escape(acierto)}</td>"
                        "</tr>"
                    )
            blocks.append(f"<table>{header}{''.join(rows)}</table>")
    if not blocks:
        return "<p>Sin ejemplos de inferencia disponibles.</p>"
    return "".join(blocks)


def build_report_html(
    results: dict[str, Any],
    out_path: Path,
    *,
    examples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    answer_type_breakdown: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
) -> Path:
    """Build the agent-benchmark comparison HTML report.

    Writes a single self-contained HTML file with (a) a comparison table of
    every (variant, benchmark, metric) cell showing ``mean +- std`` over seeds
    and the rubric pass/fail marker (AC-4/AC-5), and (b) an embedded base64 PNG
    grouped bar chart with error bars. Visible prose is in Spanish; the output
    folder is created if it does not exist.

    When the optional ``examples`` and ``answer_type_breakdown`` are provided
    (from the US-049 per-item JSONL trace read-back), two extra Spanish sections
    are appended after the metrics table. When both are ``None`` the output is
    byte-identical to the previous two-argument behaviour.

    Args:
        results: Nested results mapping
            ``{variant: {benchmark: {metric: {"mean": float, "std": float}}}}``.
            Missing cells are rendered as ``n/a``. Metric names are discovered
            from the data, so the consumer (``ml/eval/agent_bench.py``) only has
            to populate this structure.
        out_path: Destination ``.html`` path. Parent directories are created.
        examples: Optional ``{variant: {benchmark: [record, ...]}}`` example
            records for the "Ejemplos de inferencia" section.
        answer_type_breakdown: Optional
            ``{variant: {answer_type: {"n", "exact_match_mean"}}}`` for the
            AgroMind answer-type breakdown section.

    Returns:
        The ``out_path`` that was written (as a :class:`~pathlib.Path`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants = list(results.keys())
    metric_names = _collect_metric_names(results)

    if variants:
        chart_uri = _render_headline_chart(results, variants)
        chart_html = (
            f'<img src="{chart_uri}" alt="Comparativa por variante" '
            'style="max-width:100%;height:auto;" />'
        )
    else:
        chart_html = "<p>Sin resultados para graficar.</p>"

    table_html = _render_table(results, variants, metric_names)

    # Optional trace-backed sections. Empty string when no trace is supplied so
    # the document is byte-identical to the previous two-argument behaviour.
    trace_sections = ""
    if answer_type_breakdown is not None:
        trace_sections += (
            "<h2>Desglose por tipo de respuesta (AgroMind)</h2>"
            f"{_render_answer_type_table(answer_type_breakdown)}"
        )
        logger.info(
            "agent_report_breakdown_rendered",
            variants=sorted(answer_type_breakdown),
        )
    if examples is not None:
        trace_sections += (
            f"<h2>Ejemplos de inferencia (aciertos y fallos)</h2>{_render_examples(examples)}"
        )
        logger.info("agent_report_examples_rendered", variants=sorted(examples))

    document = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Benchmark del copiloto AgroSat (US-049)</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.6rem; }}
  table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  caption {{ text-align: left; font-weight: 600; margin-bottom: 0.5rem; }}
  .nota {{ color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Evaluacion comparativa del copiloto (AgroMind y GeoAnalystBench)</h1>
<p class="nota">
  Metricas agregadas como media +- desviacion estandar sobre los seeds de
  evaluacion (barras de error). Los marcadores {_OK_MARKER} / {_FAIL_MARKER}
  indican si la celda cumple el objetivo de rubrica (AgroMind exact-match
  &gt;= 0.75 Gemini y &gt;= 0.70 Qwen; GeoAnalystBench pass-rate &gt;= 0.65).
  Qwen es solo texto: su numero de AgroMind corresponde al subconjunto textual,
  no a la evaluacion multimodal completa. Ademas del exact-match destacado, la
  tabla muestra f1_squad (F1 de solape de tokens estilo SQuAD) y bertscore por
  celda. BERTScore es un proxy semantico (all-MiniLM-L6-v2), pero CodeBLEU ahora
  es la metrica canonica real (BLEU n-gram + AST + flujo de datos), no una
  aproximacion. La similitud de flujo usa un umbral calibrado de 0.35 y se
  reporta tambien su media cruda (mean_semantic_sim) junto al pass-rate.
</p>
<h2>Grafico comparativo</h2>
{chart_html}
<h2>Tabla de metricas</h2>
{table_html}
{trace_sections}</body>
</html>
"""

    out_path.write_text(document, encoding="utf-8")
    logger.info(
        "agent_report_written",
        path=str(out_path),
        variants=variants,
        benchmarks=list(metric_names.keys()),
    )
    return out_path
