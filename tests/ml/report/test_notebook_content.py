"""Tests para ml.report.notebook_content y ml.report.extract_notebook_figures."""

from __future__ import annotations

import base64
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ml.report.extract_notebook_figures import (
    _nearest_heading,
    _slugify,
    extract_png_outputs,
)
from ml.report.notebook_content import (
    ALPHAEARTH_CARD,
    BIVARIATE_CARD,
    CARDS,
    GLOBAL_CARD,
    KPI,
    PASTIS_CARD,
    SENTINEL2_CARD,
    list_figures,
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa3R\xafG\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# notebook_content.CARDS shape
# ---------------------------------------------------------------------------


def test_cards_tuple_has_five_entries() -> None:
    """Hay 5 fichas: sentinel2, alphaearth, bivariate-temporal, pastis, globales."""
    assert len(CARDS) == 5
    notebook_ids = [card.notebook_id for card in CARDS]
    assert notebook_ids == [
        "sentinel2",
        "alphaearth",
        "bivariate-temporal",
        "pastis-consolidado",
        "globales",
    ]


@pytest.mark.parametrize(
    "card",
    [SENTINEL2_CARD, ALPHAEARTH_CARD, BIVARIATE_CARD, PASTIS_CARD, GLOBAL_CARD],
    ids=lambda c: c.notebook_id,
)
def test_each_card_has_required_fields(card: object) -> None:
    """Cada ficha define title, subtitle, sections y conclusions no triviales."""
    assert card.title  # type: ignore[attr-defined]
    assert card.subtitle  # type: ignore[attr-defined]
    assert card.sections  # type: ignore[attr-defined]
    assert card.conclusions  # type: ignore[attr-defined]


def test_card_conclusions_are_pairs() -> None:
    """Cada item de conclusions es una tupla (heading, body) no vacia."""
    for card in CARDS:
        for entry in card.conclusions:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            heading, body = entry
            assert heading.strip()
            assert body.strip()


def test_global_card_has_no_figures_dir() -> None:
    """La ficha de conclusiones globales no tiene figures_dir asociado."""
    assert GLOBAL_CARD.figures_dir == ""


@pytest.mark.parametrize(
    "card",
    [SENTINEL2_CARD, ALPHAEARTH_CARD, BIVARIATE_CARD, PASTIS_CARD, GLOBAL_CARD],
    ids=lambda c: c.notebook_id,
)
def test_each_card_has_four_non_empty_kpis(card: object) -> None:
    """Cada ficha define exactamente 4 KPIs con label/value/delta no vacios."""
    kpis = card.kpis  # type: ignore[attr-defined]
    assert len(kpis) == 4
    for kpi in kpis:
        assert isinstance(kpi, KPI)
        assert kpi.label.strip()
        assert kpi.value.strip()
        assert kpi.delta.strip()


def test_kpi_dataclass_is_frozen() -> None:
    """KPI es frozen — no se puede mutar accidentalmente desde un canal."""
    kpi = KPI("Bandas", "10", "Sentinel-2")
    with pytest.raises(FrozenInstanceError):
        kpi.value = "11"  # type: ignore[misc]


def test_list_figures_returns_sorted_pngs(tmp_path: Path) -> None:
    """``list_figures`` devuelve PNGs ordenados alfabeticamente."""
    figures_root = tmp_path
    subdir = figures_root / SENTINEL2_CARD.figures_dir
    subdir.mkdir(parents=True)
    for name in ("zeta.png", "alpha.png", "mango.png"):
        (subdir / name).write_bytes(PNG_BYTES)

    result = list_figures(SENTINEL2_CARD, figures_root)
    assert [p.name for p in result] == ["alpha.png", "mango.png", "zeta.png"]


def test_list_figures_empty_when_no_figures_dir(tmp_path: Path) -> None:
    """``list_figures`` devuelve [] cuando la ficha no tiene figures_dir."""
    assert list_figures(GLOBAL_CARD, tmp_path) == []


def test_list_figures_empty_when_subdir_missing(tmp_path: Path) -> None:
    """``list_figures`` devuelve [] cuando el subdir esperado no existe."""
    assert list_figures(SENTINEL2_CARD, tmp_path) == []


# ---------------------------------------------------------------------------
# extract_notebook_figures
# ---------------------------------------------------------------------------


def test_slugify_handles_accents_and_punct() -> None:
    """``_slugify`` normaliza acentos, espacios y caracteres especiales."""
    assert _slugify("# 1.2 Anaisis de Sesgo!") == "1_2_anaisis_de_sesgo"
    assert _slugify("Conclusion - Que aprendimos") == "conclusion_que_aprendimos"
    assert _slugify("") == ""


def test_slugify_truncates_at_max_len() -> None:
    """``_slugify`` corta a ``max_len`` caracteres."""
    assert len(_slugify("a" * 200, max_len=20)) == 20


def test_nearest_heading_returns_previous_markdown() -> None:
    """``_nearest_heading`` retorna el markdown previo no vacio."""
    cells = [
        {"cell_type": "markdown", "source": ["# Heading A\n"]},
        {"cell_type": "code", "source": ["1+1"]},
        {"cell_type": "markdown", "source": ["## Heading B"]},
        {"cell_type": "code", "source": ["plt.show()"]},
    ]
    assert _nearest_heading(cells, 3) == "## Heading B"
    assert _nearest_heading(cells, 1) == "# Heading A"
    assert _nearest_heading(cells, 0) == "# Heading A"


def test_nearest_heading_returns_empty_when_no_markdown() -> None:
    """Sin celdas markdown previas, devuelve cadena vacia."""
    cells = [{"cell_type": "code", "source": ["x"]}]
    assert _nearest_heading(cells, 0) == ""


def _make_notebook(tmp_path: Path) -> Path:
    """Construye un notebook minimo con una imagen PNG inline en output."""
    nb_path = tmp_path / "demo.ipynb"
    png_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["## Seccion 1"]},
            {
                "cell_type": "code",
                "source": ["plt.show()"],
                "outputs": [{"output_type": "display_data", "data": {"image/png": png_b64}}],
            },
            {"cell_type": "markdown", "source": ["## Seccion 2 - Otra"]},
            {
                "cell_type": "code",
                "source": ["plt.show()"],
                "outputs": [{"output_type": "display_data", "data": {"image/png": png_b64}}],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    return nb_path


def test_extract_png_outputs_creates_files(tmp_path: Path) -> None:
    """``extract_png_outputs`` decodifica las imagenes inline y las escribe en disco."""
    notebook = _make_notebook(tmp_path)
    output_dir = tmp_path / "out"
    created = extract_png_outputs(notebook, output_dir)

    assert len(created) == 2
    for path in created:
        assert path.exists()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    names = [p.name for p in created]
    assert any("seccion_1" in n for n in names)
    assert any("seccion_2" in n for n in names)


def test_extract_png_outputs_raises_on_missing_notebook(tmp_path: Path) -> None:
    """Notebook inexistente debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_png_outputs(tmp_path / "missing.ipynb", tmp_path / "out")


def test_extract_png_outputs_raises_on_invalid_json(tmp_path: Path) -> None:
    """Notebook con JSON invalido debe lanzar ValueError."""
    bad = tmp_path / "bad.ipynb"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid notebook JSON"):
        extract_png_outputs(bad, tmp_path / "out")


def test_extract_png_outputs_skips_cells_without_images(tmp_path: Path) -> None:
    """Celdas sin output PNG no producen archivos."""
    nb_path = tmp_path / "empty.ipynb"
    nb = {
        "cells": [
            {"cell_type": "code", "source": ["1+1"], "outputs": []},
            {"cell_type": "markdown", "source": ["text"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    created = extract_png_outputs(nb_path, tmp_path / "out")
    assert created == []
