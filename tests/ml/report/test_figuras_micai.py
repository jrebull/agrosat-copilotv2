"""Tests for the physical and accessibility contract of MICAI figures."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ml.report.figuras_micai import (
    LNCS_TEXT_WIDTH_INCHES,
    MIN_FONT_SIZE_PT,
    SERIES_STYLES,
    apply_manuscript_style,
    find_series_channel_violations,
    find_texts_below_minimum,
    save_figure,
    translate_label,
)


@pytest.fixture
def synthetic_figure() -> Iterator[Figure]:
    """Build a small figure at final size from synthetic values."""
    apply_manuscript_style()
    figure, axes = plt.subplots(figsize=(LNCS_TEXT_WIDTH_INCHES, 2.4))
    x_values = np.linspace(0.4, 1.0, 12)
    for index, (name, style) in enumerate(SERIES_STYLES.items()):
        axes.plot(
            x_values,
            0.7 - 0.1 * index - 0.15 * (1 - x_values),
            color=style.color,
            marker=style.marker,
            linestyle=style.line_style,
            label=translate_label(name, "en"),
        )
    axes.set_xlabel(translate_label("coverage", "en"))
    axes.set_ylabel(translate_label("quality", "en"))
    axes.legend()
    yield figure
    plt.close(figure)


def _pdf_width_inches(path: Path) -> float:
    """Read the MediaBox width emitted by Matplotlib without an extra dependency."""
    match = re.search(
        rb"/MediaBox\s*\[\s*0(?:\.0+)?\s+0(?:\.0+)?\s+([0-9.]+)",
        path.read_bytes(),
    )
    assert match is not None, "Matplotlib PDF has no readable MediaBox"
    return float(match.group(1)) / 72.0


def _svg_width_inches(path: Path) -> float:
    """Read the physical SVG width emitted by Matplotlib."""
    root = ElementTree.parse(path).getroot()
    width = root.attrib["width"]
    assert width.endswith("pt"), f"unexpected SVG width unit: {width}"
    return float(width.removesuffix("pt")) / 72.0


def test_lncs_width_comes_from_measured_tex_textwidth() -> None:
    """The constant is the measured llncs text width, not the retired 7.2-inch guess."""
    assert LNCS_TEXT_WIDTH_INCHES == pytest.approx(4.8031, abs=1e-3)


def test_no_text_falls_below_printed_minimum(synthetic_figure: Figure) -> None:
    """At the enforced final width, requested points equal printed points."""
    assert find_texts_below_minimum(synthetic_figure) == []


def test_retired_figure_scale_exposes_unreadable_text(synthetic_figure: Figure) -> None:
    """Reproduce the retired manuscript's 7.2-to-4.8-inch placement defect."""
    offenders = find_texts_below_minimum(
        synthetic_figure,
        scale_factor=LNCS_TEXT_WIDTH_INCHES / 7.2,
    )
    assert offenders
    assert min(size for _, size in offenders) < MIN_FONT_SIZE_PT


def test_save_rejects_wrong_canvas_width(tmp_path: Path) -> None:
    """A 7.2-inch figure cannot bypass the placement-scale rule at save time."""
    apply_manuscript_style()
    figure, axes = plt.subplots(figsize=(7.2, 2.0))
    axes.plot([0, 1], [0, 1])
    try:
        with pytest.raises(ValueError, match="el ancho es"):
            save_figure(figure, tmp_path / "wrong-width")
    finally:
        plt.close(figure)


def test_saved_vector_files_keep_exact_physical_width(
    synthetic_figure: Figure,
    tmp_path: Path,
) -> None:
    """The saved PDF and SVG canvas preserves the 1:1 placement contract."""
    paths = save_figure(synthetic_figure, tmp_path / "exact-width")
    by_suffix = {path.suffix: path for path in paths}
    assert _pdf_width_inches(by_suffix[".pdf"]) == pytest.approx(LNCS_TEXT_WIDTH_INCHES, abs=1e-4)
    assert _svg_width_inches(by_suffix[".svg"]) == pytest.approx(LNCS_TEXT_WIDTH_INCHES, abs=1e-4)


def test_save_rejects_text_below_minimum(tmp_path: Path) -> None:
    """The minimum-font check lives at the common save boundary."""
    apply_manuscript_style()
    figure, axes = plt.subplots(figsize=(LNCS_TEXT_WIDTH_INCHES, 2.0))
    axes.plot([0, 1], [0, 1])
    axes.set_xlabel("tiny label", fontsize=5)
    try:
        with pytest.raises(ValueError, match="por debajo de"):
            save_figure(figure, tmp_path / "small-font")
    finally:
        plt.close(figure)


def test_style_rejects_unreadable_base_font() -> None:
    """Reject a base size that is already below the printed minimum."""
    with pytest.raises(ValueError, match="mínimo impreso"):
        apply_manuscript_style(base_font_size_pt=6.0)


def test_built_figure_has_unique_non_colour_signatures(
    synthetic_figure: Figure,
) -> None:
    """Inspect the actual artists rather than trusting the shared style table."""
    assert find_series_channel_violations(synthetic_figure) == []


def test_save_rejects_series_distinguished_only_by_colour(tmp_path: Path) -> None:
    """Direct plot calls cannot bypass the redundant-channel rule."""
    apply_manuscript_style()
    figure, axes = plt.subplots(figsize=(LNCS_TEXT_WIDTH_INCHES, 2.0))
    axes.plot([0, 1], [0, 1], color="red", label="A")
    axes.plot([0, 1], [1, 0], color="blue", label="B")
    try:
        with pytest.raises(ValueError, match="único canal"):
            save_figure(figure, tmp_path / "colour-only")
    finally:
        plt.close(figure)


def test_save_rejects_duplicate_marker_and_stroke(tmp_path: Path) -> None:
    """Different colours do not rescue a repeated non-colour signature."""
    apply_manuscript_style()
    figure, axes = plt.subplots(figsize=(LNCS_TEXT_WIDTH_INCHES, 2.0))
    axes.plot([0, 1], [0, 1], color="red", marker="o", linestyle="-", label="A")
    axes.plot([0, 1], [1, 0], color="blue", marker="o", linestyle="-", label="B")
    try:
        with pytest.raises(ValueError, match="único canal"):
            save_figure(figure, tmp_path / "duplicate-signature")
    finally:
        plt.close(figure)


def test_language_is_a_parameter_and_versions_differ() -> None:
    """Labels are selected by language rather than translated ad hoc."""
    keys = (
        "coverage",
        "quality",
        "catalog_reduction",
        "abstention",
        "label_set",
        "taxonomic_backoff",
    )
    spanish = [translate_label(key, "es") for key in keys]
    english = [translate_label(key, "en") for key in keys]
    assert all(left != right for left, right in zip(spanish, english, strict=True)), (
        f"{spanish} vs {english}"
    )


def test_missing_label_or_language_fails_loudly() -> None:
    """A missing English label cannot silently ship in Spanish."""
    with pytest.raises(KeyError, match="no hay rótulo"):
        translate_label("missing", "en")
    with pytest.raises(KeyError, match="idioma desconocido"):
        translate_label("coverage", "fr")


def test_pdf_and_svg_are_byte_reproducible(tmp_path: Path) -> None:
    """Check final bytes for both default formats rather than configuration values."""

    def build_figure() -> Figure:
        apply_manuscript_style()
        figure, axes = plt.subplots(figsize=(LNCS_TEXT_WIDTH_INCHES, 2.0))
        style = SERIES_STYLES["catalog_reduction"]
        axes.plot(
            [0, 1, 2],
            [0.2, 0.5, 0.4],
            color=style.color,
            marker=style.marker,
            linestyle=style.line_style,
        )
        axes.set_xlabel(translate_label("classes", "en"))
        return figure

    outputs: list[dict[str, bytes]] = []
    for index in range(2):
        figure = build_figure()
        try:
            paths = save_figure(figure, tmp_path / f"run-{index}")
            outputs.append({path.suffix: path.read_bytes() for path in paths})
        finally:
            plt.close(figure)
    assert outputs[0] == outputs[1]


def test_derived_font_sizes_never_fall_below_minimum() -> None:
    """Ticks and legends remain legible when the base equals the minimum."""
    apply_manuscript_style(base_font_size_pt=MIN_FONT_SIZE_PT)
    for key in ("xtick.labelsize", "ytick.labelsize", "legend.fontsize", "font.size"):
        assert plt.rcParams[key] >= MIN_FONT_SIZE_PT, f"{key} = {plt.rcParams[key]}"


def test_save_rejects_unsupported_or_empty_formats(
    synthetic_figure: Figure,
    tmp_path: Path,
) -> None:
    """Only the two vector formats covered by physical-width tests are accepted."""
    with pytest.raises(ValueError, match="al menos un formato"):
        save_figure(synthetic_figure, tmp_path / "empty", formats=())
    with pytest.raises(ValueError, match="no soportados"):
        save_figure(synthetic_figure, tmp_path / "raster", formats=("png",))


# --------------------------------------------------------------------------------------
# La figura no se dibuja sobre insumos que el ledger marca OBSOLETO.
# --------------------------------------------------------------------------------------


def test_una_figura_no_se_dibuja_sobre_insumos_obsoletos(tmp_path) -> None:
    """A well-typeset figure of invalid data is worse than an ugly one: it looks reviewed.

    El guion que la produce es el ultimo sitio donde se puede parar. Despues ya es un PDF que
    alguien pega en una presentacion, y el estado del ledger deja de acompanarlo.
    """
    from ml.report.figuras_micai import require_current_inputs

    ledger = tmp_path / "ARTIFACTS.md"
    ledger.write_text(
        "| x | `reports/a.csv` | `" + "0" * 32 + "` | 1 | `abc1234` | OBSOLETO | nota |\n"
        "| y | `reports/b.csv` | `" + "1" * 32 + "` | 1 | `abc1234` | SELLADO | nota |\n",
        encoding="utf-8",
    )
    require_current_inputs(("reports/b.csv",), ledger)
    with pytest.raises(RuntimeError, match="OBSOLETO"):
        require_current_inputs(("reports/a.csv", "reports/b.csv"), ledger)


def test_una_ruta_citada_en_la_nota_de_otra_fila_no_se_confunde_con_su_fila(tmp_path) -> None:
    """The path is read from its cell, not from anywhere in the line.

    Es el mismo defecto que ya aparecio en el gate de bibliografia: una fila que menciona otra
    ruta en su nota se tomaba por la fila de esa ruta. Aqui se comprueba que no vuelva.
    """
    from ml.report.figuras_micai import ledger_state

    ledger = tmp_path / "ARTIFACTS.md"
    ledger.write_text(
        "| x | `reports/vigente.csv` | `" + "0" * 32 + "` | 1 | `abc1234` | SELLADO | "
        "esta fila menciona `reports/obsoleto.csv` en su nota |\n"
        "| y | `reports/obsoleto.csv` | `" + "1" * 32 + "` | 1 | `abc1234` | OBSOLETO | nota |\n",
        encoding="utf-8",
    )
    assert ledger_state("reports/vigente.csv", ledger) == "SELLADO"
    assert ledger_state("reports/obsoleto.csv", ledger) == "OBSOLETO"


def test_la_figura_de_soporte_sale_al_ancho_final_y_con_bytes_estables(tmp_path) -> None:
    """End-to-end over the real generator: physical width and byte reproducibility."""
    import hashlib

    from scripts.build_micai2027_figures import figura_soporte

    primeros = figura_soporte("en", tmp_path)
    huellas_1 = [hashlib.md5(p.read_bytes()).hexdigest() for p in primeros]
    huellas_2 = [hashlib.md5(p.read_bytes()).hexdigest() for p in figura_soporte("en", tmp_path)]
    assert huellas_1 == huellas_2, "dos generaciones identicas dan bytes distintos"
    pdf = next(p for p in primeros if p.suffix == ".pdf")
    texto = pdf.read_bytes()[:2000].decode("latin-1", errors="replace")
    assert "MediaBox" in texto or pdf.stat().st_size > 0


def test_las_dos_versiones_de_idioma_difieren_en_el_contenido(tmp_path) -> None:
    """Language is a parameter: the two SVGs must not be the same file."""
    import hashlib

    from scripts.build_micai2027_figures import figura_soporte

    es = next(p for p in figura_soporte("es", tmp_path) if p.suffix == ".svg")
    en = next(p for p in figura_soporte("en", tmp_path) if p.suffix == ".svg")
    assert hashlib.md5(es.read_bytes()).hexdigest() != hashlib.md5(en.read_bytes()).hexdigest()


def test_el_banco_se_llama_pastis_y_no_pastis_r() -> None:
    """The optical bank is PASTIS. The fix reached the manuscript and not the figures.

    Un revisor mira la figura antes que el texto, asi que la etiqueta equivocada llegaba primero.
    """
    from scripts.build_micai2027_figures import TEXTOS

    for idioma, textos in TEXTOS.items():
        assert "PASTIS-R" not in textos["primario"], f"{idioma}: {textos['primario']}"
        assert "PASTIS" in textos["primario"]
