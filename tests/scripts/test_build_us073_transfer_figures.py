"""Tests for the US-073 multi-region transfer figure/table builder.

Scope: ONLY the artefacts this US touches (``scripts/build_us073_transfer_figures.py``).
The tests verify that every generated number traces back to the real EPIC 12 artefacts
(no hand-typed values, no synthetic data) and that the factual corrections of the paper
are not regressed. They are skipped if a real input artefact is absent (e.g. DVC not
pulled), never run against fabricated data.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from scripts import build_us073_transfer_figures as b

REPO_ROOT = Path(__file__).resolve().parents[2]
DENSE_JSON = REPO_ROOT / b.DENSE_RESULT_JSON
KSHOT_PARQUET = REPO_ROOT / b.KSHOT_PARQUET

requires_artefacts = pytest.mark.skipif(
    not (DENSE_JSON.is_file() and KSHOT_PARQUET.is_file()),
    reason="Real EPIC 12 artefacts absent (run `dvc pull` / the EPIC 12 pipeline).",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the builder once into an isolated tree mirroring the repo inputs.

    Args:
        tmp_path_factory: Pytest temp-dir factory.

    Returns:
        The temporary repo root where the artefacts were written.
    """
    root = tmp_path_factory.mktemp("us073")
    # Mirror the two real inputs into the temp root so the builder reads them there
    # (the Mexico NDVI parquet is no longer consumed by ``build_all``).
    for rel in (b.DENSE_RESULT_JSON, b.KSHOT_PARQUET):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    b.build_all(root, dpi=80)
    return root


@requires_artefacts
def test_domain_gap_table_matches_json(built: Path) -> None:
    """The dense table reproduces the exact mIoU/F1/pixel-acc/Delta of the JSON."""
    result = json.loads(DENSE_JSON.read_text(encoding="utf-8"))
    tex = (built / b.DENSE_TABLE).read_text(encoding="utf-8")
    assert f"{result['few_shot_metrics']['miou']:.4f}" == "0.2468"
    assert "0.0000 & 0.0000 & 0.0000" in tex  # zero-shot row
    assert "0.2468 & 0.3005 & 0.9179" in tex  # few-shot row
    assert f"+{float(result['delta_miou']):.4f}" in tex  # Delta mIoU
    assert "$+0.2468$" in tex


@requires_artefacts
def test_kshot_table_matches_parquet(built: Path) -> None:
    """The k-shot table reproduces the per-(scenario, k) mean/std of the parquet."""
    curve = pl.read_parquet(KSHOT_PARQUET)
    summary = b._kshot_summary(curve)
    tex = (built / b.KSHOT_TABLE).read_text(encoding="utf-8")
    # Spot-check three real cells against the recomputed summary (no hardcoding).
    for scenario, k in (("LV->EE", 500), ("sin-pretrain->EE", 1), ("LV+PT->EE", 100)):
        row = summary.filter((pl.col("scenario") == scenario) & (pl.col("k") == k))
        mean = float(row.get_column("f1_mean")[0])
        std = float(row.get_column("f1_std")[0])
        assert f"${mean:.3f} \\pm {std:.3f}$" in tex


@requires_artefacts
def test_no_alphaearth_v21_string(built: Path) -> None:
    """No generated .tex contains the non-existent AlphaEarth 'v2.1' (factual regression)."""
    for rel in (b.DENSE_TABLE, b.KSHOT_TABLE):
        tex = (built / rel).read_text(encoding="utf-8")
        assert "v2.1" not in tex
        assert "v2,1" not in tex


@requires_artefacts
def test_eurocropsml_scenario_is_real(built: Path) -> None:
    """The k-shot table uses the real LV[+PT]->EE protocol, never 'France->Estonia'."""
    tex = (built / b.KSHOT_TABLE).read_text(encoding="utf-8")
    assert "LV+PT" in tex and "LV $\\rightarrow$ EE" in tex
    lowered = tex.lower()
    # The real protocol is LV[+PT] -> EE; France appears only in the corrective
    # disclaimer ("France is NOT in EuroCropsML" / "not France -> Estonia"), and the
    # France -> Estonia arrow is always negated by a preceding "not ".
    assert "lv[+pt] $\\rightarrow$ ee" in lowered
    assert "france is not in eurocropsml" in lowered
    assert "not france $\\rightarrow$ estonia" in lowered
    positive = lowered.replace("not france $\\rightarrow$ estonia", "")
    assert "france $\\rightarrow$ estonia" not in positive


@requires_artefacts
def test_script_is_idempotent(built: Path, tmp_path: Path) -> None:
    """Building twice yields byte-identical .tex tables (deterministic, fixed seed)."""
    first_dense = (built / b.DENSE_TABLE).read_bytes()
    first_kshot = (built / b.KSHOT_TABLE).read_bytes()
    # Re-run into a fresh tree from the same inputs.
    for rel in (b.DENSE_RESULT_JSON, b.KSHOT_PARQUET):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    b.build_all(tmp_path, dpi=80)
    assert (tmp_path / b.DENSE_TABLE).read_bytes() == first_dense
    assert (tmp_path / b.KSHOT_TABLE).read_bytes() == first_kshot


@requires_artefacts
def test_figures_emitted(built: Path) -> None:
    """The builder writes the expected PNG+SVG figures (smoke, no pixel diff).

    Every figure is emitted in both languages: English keeps the canonical base name
    and Spanish gets an ``_es`` suffix on the stem.
    """
    for stem in (b.KSHOT_FIGURE,):
        for lang in b.LANGS:
            lang_stem = b._lang_stem(built / stem, lang)
            png = lang_stem.with_suffix(".png")
            svg = lang_stem.with_suffix(".svg")
            assert png.is_file() and png.stat().st_size > 0
            assert svg.is_file() and svg.stat().st_size > 0
    # The English base name must exist bare (no language suffix).
    assert (built / Path(f"{b.KSHOT_FIGURE}.png")).is_file()
    # The Spanish variant must exist with the ``_es`` suffix.
    assert (built / b.FIGURES_DIR / "kshot_curve_es.png").is_file()
    # The Mexico avocado/guava figure is intentionally no longer emitted (the
    # manuscript keeps real-metric experiments only; see ``build_all``).
    assert not (built / Path(f"{b.MEXICO_FIGURE}.png")).exists()


@requires_artefacts
def test_figure_text_is_translated(built: Path) -> None:
    """The English base and Spanish ``_es`` SVGs carry the correct visible strings.

    Matplotlib's SVG backend records each rendered string in an XML comment, so the
    presence of a phrase in the SVG proves it was drawn. English words must be absent
    from the Spanish variant and vice-versa (only strings differ, never the data).
    """
    en_svg = (built / Path(f"{b.KSHOT_FIGURE}.svg")).read_text(encoding="utf-8")
    es_svg = (built / b.FIGURES_DIR / "kshot_curve_es.svg").read_text(encoding="utf-8")
    assert "labelled samples" in en_svg and "query set" in en_svg
    assert "labelled samples" not in es_svg
    # Spanish accented strings round-trip through the UTF-8 SVG.
    assert "muestras etiquetadas" in es_svg
    assert "país objetivo" in es_svg  # "país objetivo"
    assert "conjunto de consulta" in es_svg
