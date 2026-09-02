"""CLI to export the Avance 1 (EDA) PDF report.

Renders a Jinja2 template with five cards (univariate Sentinel-2, AlphaEarth
Foundations, bivariate/temporal, consolidated PASTIS-R and global conclusions)
and converts it to PDF via WeasyPrint. Satisfies AC-8 and AC-9 of US-013: PDF
generated + structure consistent with the actual Avance 1 notebooks.

Usage:
    python -m ml.report.export_pdf
    python -m ml.report.export_pdf --output paper/avance1_eda_report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ml.report.figure_narratives import get_narrative
from ml.report.notebook_content import EDA_DISPLAY_CARDS, NotebookCard, list_figures

try:
    import structlog

    logger: Any = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover - fallback simple
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "avance1_eda_report.pdf"
DEFAULT_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
DEFAULT_TEMPLATE = REPO_ROOT / "ml" / "report" / "templates" / "avance1_eda.html.j2"
DEFAULT_TITLE = "Avance 1 - Analisis Exploratorio de Datos - AgroSatCopilot"


app = typer.Typer(add_completion=False, help="Exporta el reporte PDF del Avance 1 EDA.")


def _collect_card_figures(
    cards: tuple[NotebookCard, ...],
    figures_dir: Path,
) -> dict[str, list[Path]]:
    """Group the figures by ``notebook_id`` according to each card's configuration.

    Args:
        cards: Tuple of cards to process (presentation order of the report).
        figures_dir: Root directory containing the per-card subdirectories
            (``us-010``, ``us-011``, ``us-012``, ``avance1``, ...).

    Returns:
        Dictionary ``{notebook_id: [paths_png]}`` with sorted lists. The list
        is empty for cards without ``figures_dir`` or without available PNGs.
    """
    return {card.notebook_id: list_figures(card, figures_dir) for card in cards}


def _render_html(template_path: Path, context: dict[str, Any]) -> str:
    """Render the Jinja2 template with autoescape active.

    Args:
        template_path: Path to the .html.j2 template.
        context: Dictionary of variables injected into the template.

    Returns:
        HTML as a string ready for WeasyPrint.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "j2", "html.j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    return template.render(**context)


def _html_to_pdf(html_str: str, output: Path, css_path: Path, base_url: Path) -> None:
    """Convert HTML to PDF via WeasyPrint.

    Args:
        html_str: Full HTML to render.
        output: Destination path of the PDF.
        css_path: Path to the CSS stylesheet.
        base_url: Base path to resolve images referenced in the HTML.

    Raises:
        RuntimeError: If WeasyPrint is not installed or if native GTK/cairo/pango
            dependencies are missing on Windows.
    """
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise RuntimeError(
            "WeasyPrint is not installed. Run 'poetry install --with paper' "
            "to add the paper group that includes weasyprint and jinja2."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        html_doc = HTML(string=html_str, base_url=str(base_url))
        stylesheets = [CSS(filename=str(css_path))] if css_path.exists() else []
        html_doc.write_pdf(str(output), stylesheets=stylesheets)
    except OSError as exc:
        gtk_url = "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer"
        raise RuntimeError(
            "WeasyPrint failed to render the PDF. On Windows it requires the GTK runtime "
            f"(libpango/libcairo). Install GTK3 ({gtk_url}) or run inside WSL2/Linux "
            f"where the native dependencies are available. Original error: {exc}"
        ) from exc


@app.command()
def main(
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Path destino del PDF generado.",
    ),
    figures_dir: Path = typer.Option(
        DEFAULT_FIGURES_DIR,
        "--figures-dir",
        "-f",
        help="Directorio raiz con subdirectorios us-010/us-011/us-012/avance1.",
    ),
    template: Path = typer.Option(
        DEFAULT_TEMPLATE,
        "--template",
        "-t",
        help="Path al template Jinja2 .html.j2.",
    ),
    title: str = typer.Option(
        DEFAULT_TITLE,
        "--title",
        help="Titulo de portada del reporte.",
    ),
) -> None:
    """Generate the Avance 1 PDF report from the five configured cards."""
    figures_dir = figures_dir.resolve()
    template = template.resolve()
    output = output.resolve()

    if not template.exists():
        typer.echo(f"ERROR: template no existe: {template}", err=True)
        raise typer.Exit(code=2)

    card_figures = _collect_card_figures(EDA_DISPLAY_CARDS, figures_dir)
    total = sum(len(v) for v in card_figures.values())
    logger.info(
        "report_collect_figures",
        figures_dir=str(figures_dir),
        per_card={k: len(v) for k, v in card_figures.items()},
        total=total,
    )

    context = {
        "title": title,
        "report_date": "2026-05-13",
        "team": [
            "Arthur Zizumbo (MLOps / Platform Lead)",
            "Aaron Bocanegra (Full-Stack / Backend Lead)",
            "Isaac Ávila (ML / Data Scientist)",
        ],
        "sponsor": "Dr. Gerardo Camacho (gjcamacho@tec.mx)",
        "course": "MNA — Tec de Monterrey",
        "cards": EDA_DISPLAY_CARDS,
        "card_figures": card_figures,
        "figures_base_url": str(figures_dir.parent) + "/",
        "get_narrative": get_narrative,
    }

    html_str = _render_html(template, context)

    css_path = template.parent / "styles.css"
    base_url = figures_dir.parent.parent

    try:
        _html_to_pdf(html_str, output, css_path, base_url)
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    size_bytes = output.stat().st_size if output.exists() else 0
    logger.info(
        "report_pdf_generated",
        output=str(output),
        size_bytes=size_bytes,
        size_mb=round(size_bytes / 1024 / 1024, 2),
    )
    typer.echo(f"OK PDF generado: {output} ({size_bytes / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
