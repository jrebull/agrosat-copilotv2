"""Tests para ml.report.export_pdf - cobertura objetivo >= 70 %.

Verifica:
    - Agrupacion de figuras por notebook_id usando ``_collect_card_figures`` (AC-8).
    - Renderizado Jinja2 con autoescape activo (seguridad XSS).
    - Renderizado de la plantilla canonica con las 5 fichas y conclusiones.
    - Generacion de PDF valido si WeasyPrint esta disponible (AC-8).
    - Manejo limpio de error cuando GTK falta en Windows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ml.report import export_pdf
from ml.report.export_pdf import (
    _collect_card_figures,
    _render_html,
    app,
)
from ml.report.figure_narratives import get_narrative
from ml.report.notebook_content import CARDS, SENTINEL2_CARD

WEASYPRINT_AVAILABLE = importlib.util.find_spec("weasyprint") is not None
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa3R\xafG\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def figures_dir(tmp_path: Path) -> Path:
    """Crea un figures_dir con subdirs para cada ficha y un PNG en cada uno."""
    base = tmp_path / "figures"
    for card in CARDS:
        if not card.figures_dir:
            continue
        sub = base / card.figures_dir
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"{card.figures_dir}_plot.png").write_bytes(PNG_BYTES)
    return base


def test_collect_card_figures_groups_by_notebook_id(figures_dir: Path) -> None:
    """``_collect_card_figures`` agrupa por notebook_id segun la ficha."""
    figures = _collect_card_figures(CARDS, figures_dir)

    expected_ids = {card.notebook_id for card in CARDS}
    assert set(figures.keys()) == expected_ids

    for card in CARDS:
        if card.figures_dir:
            assert len(figures[card.notebook_id]) == 1, f"Ficha {card.notebook_id} sin figura"
            assert figures[card.notebook_id][0].name.endswith(".png")
        else:
            assert figures[card.notebook_id] == []


def test_collect_card_figures_sorted_alphabetically(tmp_path: Path) -> None:
    """Verifica orden alfabetico para reproducibilidad del PDF."""
    base = tmp_path / "figures"
    sub = base / SENTINEL2_CARD.figures_dir
    sub.mkdir(parents=True)
    for name in ("zeta.png", "alpha.png", "mango.png"):
        (sub / name).write_bytes(PNG_BYTES)

    figures = _collect_card_figures(CARDS, base)
    names = [p.name for p in figures[SENTINEL2_CARD.notebook_id]]
    assert names == ["alpha.png", "mango.png", "zeta.png"]


def test_collect_card_figures_missing_subdir_returns_empty(tmp_path: Path) -> None:
    """Subdirs ausentes producen listas vacias, no errores."""
    base = tmp_path / "figures"
    base.mkdir()
    (base / SENTINEL2_CARD.figures_dir).mkdir()
    (base / SENTINEL2_CARD.figures_dir / "x.png").write_bytes(PNG_BYTES)

    figures = _collect_card_figures(CARDS, base)
    assert len(figures[SENTINEL2_CARD.notebook_id]) == 1
    # Las otras fichas con figures_dir pero sin subdir deben devolver lista vacia.
    other_cards = [
        c for c in CARDS if c.notebook_id != SENTINEL2_CARD.notebook_id and c.figures_dir
    ]
    for card in other_cards:
        assert figures[card.notebook_id] == []


def test_render_html_uses_template(tmp_path: Path) -> None:
    """Verifica renderizado Jinja2 con un template inline minimo."""
    template = tmp_path / "mini.html.j2"
    template.write_text(
        "<html><body><h1>{{ title }}</h1>"
        "{% for card in cards %}<h2>{{ card.title }}</h2>{% endfor %}"
        "</body></html>",
        encoding="utf-8",
    )
    context = {"title": "Test Report", "cards": CARDS}
    html = _render_html(template, context)

    assert "Test Report" in html
    for card in CARDS:
        assert card.title in html


def test_render_html_autoescape_active(tmp_path: Path) -> None:
    """Confirma que autoescape Jinja2 escapa HTML en variables (seguridad XSS)."""
    template = tmp_path / "x.html.j2"
    template.write_text("<p>{{ title }}</p>", encoding="utf-8")
    html = _render_html(template, {"title": "<script>alert(1)</script>"})

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_canonical_template_renders_all_cards(tmp_path: Path, figures_dir: Path) -> None:
    """La plantilla canonica del Avance 1 renderiza las 5 fichas + conclusiones."""
    repo_root = Path(__file__).resolve().parents[3]
    template = repo_root / "ml" / "report" / "templates" / "avance1_eda.html.j2"

    card_figures = _collect_card_figures(CARDS, figures_dir)
    context = {
        "title": "Avance 1 EDA",
        "report_date": "2026-05-13",
        "team": ["Arthur", "Aaron", "Isaac"],
        "sponsor": "Dr. Camacho",
        "course": "MNA",
        "cards": CARDS,
        "card_figures": card_figures,
        "figures_base_url": str(figures_dir.parent) + "/",
        "get_narrative": get_narrative,
    }
    html = _render_html(template, context)

    for card in CARDS:
        assert card.title in html, f"Falta titulo de ficha: {card.title}"
    # Al menos una conclusion por ficha que tenga conclusions.
    for card in CARDS:
        if card.conclusions:
            first_heading, _ = card.conclusions[0]
            assert first_heading in html, f"Falta conclusion: {first_heading}"


def test_cli_template_not_found(tmp_path: Path) -> None:
    """CLI debe fallar con exit code 2 si el template no existe."""
    runner = CliRunner()
    output = tmp_path / "out.pdf"
    missing_template = tmp_path / "missing.html.j2"

    result = runner.invoke(
        app,
        [
            "--output",
            str(output),
            "--figures-dir",
            str(tmp_path),
            "--template",
            str(missing_template),
        ],
    )
    assert result.exit_code == 2


@pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="weasyprint no instalado")
def test_export_pdf_generates_valid_pdf(tmp_path: Path, figures_dir: Path) -> None:
    """Genera un PDF real con WeasyPrint y verifica tamano minimo (AC-8).

    Skipea si GTK runtime falta en Windows (requiere instalacion adicional;
    ver MT-4 en docs/manual-test/us-013.md).
    """
    repo_root = Path(__file__).resolve().parents[3]
    template = repo_root / "ml" / "report" / "templates" / "avance1_eda.html.j2"
    output = tmp_path / "out.pdf"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--output",
            str(output),
            "--figures-dir",
            str(figures_dir),
            "--template",
            str(template),
        ],
    )

    if result.exit_code != 0 and (
        "GTK" in result.output
        or "libgobject" in str(result.exception)
        or "WeasyPrint could not import" in result.output
    ):
        pytest.skip("GTK runtime no disponible (Windows). Ver MT-4.")

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.stat().st_size > 1000


def test_html_to_pdf_handles_missing_gtk(tmp_path: Path, figures_dir: Path) -> None:
    """Mockea OSError de WeasyPrint (GTK ausente Windows) y verifica mensaje claro."""
    repo_root = Path(__file__).resolve().parents[3]
    template = repo_root / "ml" / "report" / "templates" / "avance1_eda.html.j2"
    output = tmp_path / "out.pdf"

    fake_weasyprint = MagicMock()
    fake_html_instance = MagicMock()
    fake_html_instance.write_pdf.side_effect = OSError("cannot load library 'libgobject-2.0-0'")
    fake_weasyprint.HTML.return_value = fake_html_instance
    fake_weasyprint.CSS.return_value = MagicMock()

    runner = CliRunner()
    with patch.dict("sys.modules", {"weasyprint": fake_weasyprint}):
        result = runner.invoke(
            app,
            [
                "--output",
                str(output),
                "--figures-dir",
                str(figures_dir),
                "--template",
                str(template),
            ],
        )

    assert result.exit_code != 0
    assert "GTK" in result.output or "WeasyPrint" in result.output


def test_html_to_pdf_runtime_when_weasyprint_missing(tmp_path: Path) -> None:
    """_html_to_pdf debe lanzar RuntimeError si weasyprint no es importable."""
    with patch.dict("sys.modules", {"weasyprint": None}):
        with pytest.raises(RuntimeError, match="WeasyPrint is not installed"):
            export_pdf._html_to_pdf(
                html_str="<html></html>",
                output=tmp_path / "x.pdf",
                css_path=tmp_path / "styles.css",
                base_url=tmp_path,
            )
