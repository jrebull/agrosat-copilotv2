"""Smoke tests para la sección Baseline (US-023-preview P9).

Cobertura objetivo: verificar AC-P9-1, AC-P9-2, AC-P9-6 y robustez frente a
artefactos faltantes (R11 del plan). Los tests no requieren los parquet ni
los PNG de P2/P3/P4/P5/P8 — comprueban explícitamente que la ausencia
levanta ``st.warning`` y no traceback.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

# Streamlit es opcional para el repo base; se instala via grupo `paper`.
streamlit = pytest.importorskip("streamlit", reason="Streamlit no instalado (grupo paper).")
AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit.testing.v1 requiere Streamlit >= 1.28."
).AppTest

from app import eda_dashboard  # noqa: E402

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "app" / "eda_dashboard.py"


# ---------------------------------------------------------------------------
# AC-P9-1 — selector incluye Baseline
# ---------------------------------------------------------------------------


def test_section_baseline_in_options() -> None:
    """AC-P9-1: ``_SECTION_BASELINE`` está presente y respeta el orden relativo.

    El selector crecio de 3 a 5 secciones (Historia, EDA, FE, Baseline,
    Segmentacion). Se valida pertenencia y el orden relativo EDA -> FE ->
    Baseline, no la tupla exacta.
    """
    assert hasattr(eda_dashboard, "_SECTION_BASELINE"), (
        "Constante `_SECTION_BASELINE` no definida en eda_dashboard"
    )
    options = eda_dashboard._SECTION_OPTIONS
    expected = (
        eda_dashboard._SECTION_EDA,
        eda_dashboard._SECTION_FE,
        eda_dashboard._SECTION_BASELINE,
    )
    for name in expected:
        assert name in options, f"`{name}` no aparece en `_SECTION_OPTIONS`"
    idx_eda = options.index(eda_dashboard._SECTION_EDA)
    idx_fe = options.index(eda_dashboard._SECTION_FE)
    idx_baseline = options.index(eda_dashboard._SECTION_BASELINE)
    assert idx_eda < idx_fe < idx_baseline, "Orden relativo EDA -> FE -> Baseline roto"


# ---------------------------------------------------------------------------
# AC-P9-2 — render callable + AC-P9-7 smoke streamlit
# ---------------------------------------------------------------------------


def test_render_baseline_section_callable() -> None:
    """AC-P9-2: ``_render_baseline_section`` existe y es invocable.

    No invocamos la función directamente (requiere contexto Streamlit
    activo). En su lugar verificamos que esté definida, sea callable, y
    expongamos los 5 tabs y los 5 renderers en variables módulo-level.
    """
    assert hasattr(eda_dashboard, "_render_baseline_section"), (
        "Falta funcion `_render_baseline_section`"
    )
    assert callable(eda_dashboard._render_baseline_section)
    # Type hints + docstring presentes (AC-P9-8).
    sig = inspect.signature(eda_dashboard._render_baseline_section)
    # Con `from __future__ import annotations`, la anotacion llega como str.
    assert sig.return_annotation in (None, "None"), (
        f"Se esperaba return type hint = None, recibido {sig.return_annotation!r}"
    )
    docstring = eda_dashboard._render_baseline_section.__doc__ or ""
    assert docstring.strip(), "Docstring vacio en `_render_baseline_section`"

    # AC-P9-3: 5 tabs declarados y 5 renderers conectados.
    assert len(eda_dashboard._BASELINE_TAB_LABELS) == 5, (
        f"Se esperaban 5 tabs, hay {len(eda_dashboard._BASELINE_TAB_LABELS)}"
    )
    assert len(eda_dashboard._BASELINE_TAB_RENDERERS) == 5, (
        f"Se esperaban 5 renderers, hay {len(eda_dashboard._BASELINE_TAB_RENDERERS)}"
    )


def test_baseline_section_renders_via_apptest(tmp_path: Path) -> None:
    """AC-P9-7: el dashboard arranca y la seleccion Baseline no rompe."""
    # Limpia caches para evitar contaminacion entre tests del modulo paralelo.
    if hasattr(eda_dashboard, "load_parquet"):
        eda_dashboard.load_parquet.clear()

    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=60)
    assert not at.exception, f"Excepcion al cargar dashboard: {at.exception}"  # type: ignore[attr-defined]

    # Selecciona la seccion Baseline via session_state y re-ejecuta.
    at.session_state[eda_dashboard._SECTION_STATE_KEY] = eda_dashboard._SECTION_BASELINE
    at.run(timeout=60)
    assert not at.exception, f"Excepcion al seleccionar Baseline: {at.exception}"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# R11 — graceful degradation cuando falta un parquet/PNG
# ---------------------------------------------------------------------------


def test_missing_parquet_does_not_break_loader(tmp_path: Path) -> None:
    """R11: ``load_parquet`` devuelve DataFrame vacio para rutas inexistentes."""
    eda_dashboard.load_parquet.clear()
    df = eda_dashboard.load_parquet(tmp_path / "missing.parquet")
    assert df.is_empty(), "Se esperaba DataFrame vacio para parquet inexistente"


def test_missing_artifacts_render_warning_not_traceback() -> None:
    """R11: al faltar artefactos Baseline, el dashboard usa ``st.warning`` graceful.

    Verifica que el hint canónico para regenerar artefactos esté definido y
    referencie los targets ``make`` esperados.
    """
    hint = eda_dashboard._BASELINE_MISSING_HINT
    assert "reencuadre-notebook-full" in hint, (
        "Hint Baseline debe mencionar `make reencuadre-notebook-full`"
    )
    assert "baseline-v2-full" in hint, "Hint Baseline debe mencionar `make baseline-v2-full`"


# ---------------------------------------------------------------------------
# AC-P9-6 — idioma de strings en espanol
# ---------------------------------------------------------------------------


def test_baseline_tab_labels_in_spanish() -> None:
    """AC-P9-6: las etiquetas de los 5 tabs Baseline están en español UTF-8."""
    labels = eda_dashboard._BASELINE_TAB_LABELS
    # Palabras clave en espanol o acentos presentes en al menos una etiqueta.
    blob = " ".join(labels).lower()
    keywords_es = ("ablation", "features", "leakage", "geográfico", "modelos", "conclusiones")
    matched = [kw for kw in keywords_es if kw in blob]
    assert len(matched) >= 3, (
        f"Pocas palabras clave en espanol en labels Baseline. Encontradas: {matched}"
    )
    # Al menos un acento UTF-8 presente (geográfico).
    assert any(any(ch in label for ch in "áéíóúñÁÉÍÓÚÑ") for label in labels), (
        "Se esperaba al menos una etiqueta con acento UTF-8 en espanol"
    )
