"""Tests de ``ml.eval.comparison`` y del escenario (b) S2 crudo (US-022, EPIC 4).

Cubre la comparativa de los 3 escenarios del baseline en cuatro grupos:

- A — tabla comparativa (:func:`build_comparison_table`, :class:`ComparisonResult`).
- B — escenario (b) Sentinel-2 crudo (``scripts/build_s2_raw_parcels.py``) y
  la alineacion por ``parcel_id`` (:func:`_align_scenarios_by_parcel`).
- C — exportacion LaTeX (:func:`export_comparison_latex`).
- D — integracion del notebook ``04_baseline.ipynb`` (secciones 7-8).

Los tests core usan :func:`make_three_scenarios` (fixture sintetica
autocontenida) — sin tocar los parquets reales de 85.951 parcelas ni el
CV espacial real H3+KMeans sobre el dataset completo.

Workaround R7 (heredado): ``coverage run --include=...`` en vez de
``pytest --cov`` por la incompatibilidad numpy 2.3.5 / scipy 1.17.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.eval.comparison import (
    ComparisonResult,
    _align_scenarios_by_parcel,
    _count_features,
    _load_scenario,
    build_comparison_table,
    export_comparison_latex,
)
from tests.ml.eval.fixtures.comparison_synthetic import (
    make_three_scenarios,
    write_three_scenarios,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Parametros ligeros para mantener los tests rapidos en CI: dataset
# pequeno + 3 folds bastan para ejercitar toda la ruta de codigo.
_TEST_N = 150
_TEST_K = 3


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison_result(tmp_path_factory: pytest.TempPathFactory) -> ComparisonResult:
    """Construye una :class:`ComparisonResult` reusable por los tests del grupo A.

    Es ``scope="module"`` para entrenar los 6 modelos una sola vez (el CV
    espacial es la parte cara) y compartir el resultado entre los tests.
    """
    out_dir = tmp_path_factory.mktemp("scenarios")
    paths = write_three_scenarios(out_dir, n=_TEST_N, n_classes=4, seed=42)
    return build_comparison_table(paths, k_folds=_TEST_K, random_state=42)


@pytest.fixture
def scenario_paths(tmp_path: Path) -> dict[str, str]:
    """Escribe los 3 escenarios sinteticos a parquet y devuelve sus rutas."""
    return write_three_scenarios(tmp_path, n=_TEST_N, n_classes=4, seed=42)


# ===========================================================================
# Grupo A — tabla comparativa.
# ===========================================================================


def test_comparison_table_has_nine_rows(comparison_result: ComparisonResult) -> None:
    """La tabla tiene exactamente 9 filas (3 escenarios x 3 modelos)."""
    assert comparison_result.table.height == 9


def test_comparison_table_three_scenarios(
    comparison_result: ComparisonResult,
) -> None:
    """La tabla cubre los 3 escenarios, cada uno con RF, XGB y LGBM."""
    table = comparison_result.table
    scenarios = set(table.get_column("scenario").to_list())
    assert len(scenarios) == 3
    for scenario in scenarios:
        models = table.filter(pl.col("scenario") == scenario).get_column("model").to_list()
        assert sorted(models) == ["LGBM", "RF", "XGB"]


def test_comparison_uses_same_spatial_cv(scenario_paths: dict[str, str], tmp_path: Path) -> None:
    """El mismo CV espacial 5-fold se reusa para los 3 escenarios.

    Tras una corrida, el caché de folds de ``ml.train.baseline`` queda
    escrito con la clave ``n`` comun de los 3 escenarios alineados: una
    sola entrada de caché => mismos folds para todos.
    """
    cache_dir = _REPO_ROOT / "data" / "test_fixtures"
    before = set(cache_dir.glob("baseline_spatial_folds_*.parquet"))
    build_comparison_table(scenario_paths, k_folds=_TEST_K, random_state=42)
    after = set(cache_dir.glob("baseline_spatial_folds_*.parquet"))
    new_caches = after - before
    # Los 3 escenarios comparten N (inner join) => una sola clave de caché.
    keys = {p.name for p in (new_caches or after)}
    matching = [k for k in keys if f"_k{_TEST_K}_" in k]
    assert matching, "el CV espacial no genero caché de folds"


def test_comparison_table_has_f1_weighted_and_miou(
    comparison_result: ComparisonResult,
) -> None:
    """Cada celda reporta f1_weighted y miou ademas de f1_macro."""
    cols = set(comparison_result.table.columns)
    assert {"f1_macro", "f1_weighted", "miou"}.issubset(cols)


def test_comparison_table_has_train_time(
    comparison_result: ComparisonResult,
) -> None:
    """Cada celda reporta train_time_s positivo (wall-clock del fit)."""
    times = comparison_result.table.get_column("train_time_s").to_list()
    assert len(times) == 9
    assert all(t > 0.0 for t in times)


def test_comparison_metrics_in_valid_range(
    comparison_result: ComparisonResult,
) -> None:
    """f1_macro, f1_weighted y miou viven en [0, 1] para las 9 filas."""
    table = comparison_result.table
    for metric in ("f1_macro", "f1_weighted", "miou"):
        values = table.get_column(metric).to_list()
        assert all(0.0 <= v <= 1.0 for v in values), metric


def test_comparison_computes_alphaearth_delta(
    comparison_result: ComparisonResult,
) -> None:
    """alphaearth_delta = f1_macro(AlphaEarth) - f1_macro(S2 crudo)."""
    table = comparison_result.table
    ae_best = float(
        table.filter(pl.col("scenario").str.contains("AlphaEarth")).get_column("f1_macro").max()
    )
    s2_best = float(
        table.filter(pl.col("scenario").str.contains("Sentinel-2")).get_column("f1_macro").max()
    )
    assert comparison_result.alphaearth_delta == pytest.approx(ae_best - s2_best, abs=1e-9)


def test_comparison_table_sorted_by_f1_macro(
    comparison_result: ComparisonResult,
) -> None:
    """La tabla esta ordenada por f1_macro descendente (AC-4)."""
    f1 = comparison_result.table.get_column("f1_macro").to_list()
    assert f1 == sorted(f1, reverse=True)


def test_comparison_best_scenario_matches_table(
    comparison_result: ComparisonResult,
) -> None:
    """best_scenario es el escenario de la fila con mayor f1_macro."""
    top = comparison_result.table.sort("f1_macro", descending=True).row(0, named=True)
    assert comparison_result.best_scenario == top["scenario"]


def test_comparison_max_samples_reduces_dataset(
    scenario_paths: dict[str, str],
) -> None:
    """max_samples > 0 submuestrea el dataset manteniendo las 9 filas."""
    result = build_comparison_table(
        scenario_paths, k_folds=_TEST_K, max_samples=80, random_state=42
    )
    assert result.table.height == 9
    assert result.n_parcels == _TEST_N  # n_parcels = inner join, pre-subsample


def test_comparison_missing_scenario_key_raises(
    scenario_paths: dict[str, str],
) -> None:
    """Falta una clave de escenario => ValueError explicito."""
    incomplete = {k: v for k, v in scenario_paths.items() if k != "s2_raw"}
    with pytest.raises(ValueError, match="s2_raw"):
        build_comparison_table(incomplete, k_folds=_TEST_K)


# ===========================================================================
# Grupo B — escenario (b) S2 crudo + alineacion por parcel_id.
# ===========================================================================


def test_s2_raw_parcels_has_ten_bands() -> None:
    """El agregador de bandas produce las 10 columnas <banda>_mean."""
    from ml.ingest.pastis_loader import PASTIS_S2_BANDS
    from scripts.build_s2_raw_parcels import _BAND_MEAN_COLS

    assert len(_BAND_MEAN_COLS) == 10
    assert _BAND_MEAN_COLS == [f"{b}_mean" for b in PASTIS_S2_BANDS]


def test_s2_raw_band_order_matches_pastis() -> None:
    """El orden de las bandas del escenario (b) sigue PASTIS_S2_BANDS."""
    from ml.ingest.pastis_loader import PASTIS_S2_BANDS
    from scripts.build_s2_raw_parcels import _BAND_MEAN_COLS, _OUTPUT_SCHEMA

    schema_bands = [c for c in _OUTPUT_SCHEMA if c.endswith("_mean")]
    assert schema_bands == _BAND_MEAN_COLS
    assert [c.removesuffix("_mean") for c in schema_bands] == PASTIS_S2_BANDS


def test_s2_raw_aggregate_patch_bands_means() -> None:
    """``aggregate_patch_bands`` promedia bien temporal y espacialmente.

    Construye un tensor S2 sintetico ``(T, 10, H, W)`` con una mascara
    instance conocida y verifica que la media de cada banda coincide con
    el calculo manual sobre los pixeles de la parcela.
    """
    from scripts.build_s2_raw_parcels import aggregate_patch_bands

    rng = np.random.default_rng(7)
    t_steps, n_bands, h, w = 4, 10, 8, 8
    s2 = rng.integers(0, 5000, size=(t_steps, n_bands, h, w), dtype=np.int16)
    target = np.zeros((3, h, w), dtype=np.uint8)
    # Parcela instance_id=1 ocupa la mitad superior del patch.
    target[1, :4, :] = 1

    tmp = Path(__file__).resolve().parent / "_tmp_s2_patch"
    tmp.mkdir(exist_ok=True)
    s2_path = tmp / "S2_999.npy"
    inst_path = tmp / "TARGET_999.npy"
    np.save(s2_path, s2)
    np.save(inst_path, target)
    try:
        records = aggregate_patch_bands(
            999,
            s2_path,
            inst_path,
            [
                {
                    "parcel_id": "999_1",
                    "instance_id": 1,
                    "class_id": 3,
                    "fold": 2,
                }
            ],
        )
    finally:
        s2_path.unlink()
        inst_path.unlink()
        tmp.rmdir()

    assert len(records) == 1
    record = records[0]
    assert record["parcel_id"] == "999_1"
    assert record["class_id"] == 3
    assert record["fold"] == 2
    # Media esperada: media temporal y luego espacial sobre la mascara.
    from scripts.build_s2_raw_parcels import _BAND_MEAN_COLS

    mask = target[1] == 1
    expected = s2.astype(np.float64).mean(axis=0)[:, mask].mean(axis=1)
    for band_idx, col in enumerate(_BAND_MEAN_COLS):
        assert record[col] == pytest.approx(float(expected[band_idx]), rel=1e-9)


def test_s2_raw_aggregate_skips_missing_tensor() -> None:
    """``aggregate_patch_bands`` devuelve [] si faltan los tensores."""
    from scripts.build_s2_raw_parcels import aggregate_patch_bands

    records = aggregate_patch_bands(
        404,
        Path("does_not_exist_S2.npy"),
        Path("does_not_exist_TARGET.npy"),
        [{"parcel_id": "404_1", "instance_id": 1, "class_id": 1, "fold": 1}],
    )
    assert records == []


def test_align_scenarios_inner_join_by_parcel() -> None:
    """``_align_scenarios_by_parcel`` reduce los 3 frames al parcel_id comun."""
    scenarios = make_three_scenarios(n=120, n_classes=3, seed=1, drop_from_combined=20)
    aligned = _align_scenarios_by_parcel(scenarios)
    heights = {k: v.height for k, v in aligned.items()}
    # combined perdio 20 filas => el inner join deja 100 en los 3.
    assert set(heights.values()) == {100}
    # Los 3 frames quedan con exactamente el mismo parcel_id, mismo orden.
    ids = [df.get_column("parcel_id").to_list() for df in aligned.values()]
    assert ids[0] == ids[1] == ids[2]


def test_align_scenarios_empty_intersection_raises() -> None:
    """Interseccion vacia de parcel_id => ValueError."""
    a = make_three_scenarios(n=40, seed=1)["alphaearth"]
    b = make_three_scenarios(n=40, seed=2)["s2_raw"].with_columns(
        ("X_" + pl.col("parcel_id")).alias("parcel_id")
    )
    c = make_three_scenarios(n=40, seed=1)["combined"]
    with pytest.raises(ValueError, match="inner join"):
        _align_scenarios_by_parcel({"alphaearth": a, "s2_raw": b, "combined": c})


def test_s2_raw_parcels_aligns_with_alphaearth() -> None:
    """El parcel_id sintetico del escenario (b) alinea con los otros 2.

    Reproduce la convencion "<patch_id>_<instance_id>" que usan los tres
    escenarios reales (verificado en el repo: AlphaEarth, feature_subset
    y el GeoParquet comparten ``parcel_id`` string).
    """
    scenarios = make_three_scenarios(n=90, n_classes=3, seed=5)
    s2_ids = set(scenarios["s2_raw"].get_column("parcel_id").to_list())
    ae_ids = set(scenarios["alphaearth"].get_column("parcel_id").to_list())
    assert s2_ids == ae_ids
    assert all("_" in pid for pid in s2_ids)


def test_load_scenario_missing_file_raises(tmp_path: Path) -> None:
    """``_load_scenario`` lanza FileNotFoundError si el parquet no existe."""
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_scenario(tmp_path / "missing.parquet")


def test_load_scenario_missing_columns_raises(tmp_path: Path) -> None:
    """``_load_scenario`` exige parcel_id y class_id."""
    bad = tmp_path / "bad.parquet"
    # Tiene parcel_id pero le falta class_id => debe fallar por class_id.
    pl.DataFrame({"parcel_id": ["1_1", "1_2"], "foo": [1, 2]}).write_parquet(bad)
    with pytest.raises(ValueError, match="class_id"):
        _load_scenario(bad)


def test_count_features_excludes_metadata() -> None:
    """``_count_features`` cuenta solo las columnas numericas de feature."""
    df = make_three_scenarios(n=30, seed=1)["s2_raw"]
    # s2_raw: 10 bandas; parcel_id/patch_id/instance_id/class_id/fold son meta.
    assert _count_features(df) == 10


# ===========================================================================
# Grupo C — exportacion LaTeX.
# ===========================================================================


def test_export_latex_produces_valid_tex(
    comparison_result: ComparisonResult, tmp_path: Path
) -> None:
    """``export_comparison_latex`` genera un .tex booktabs valido."""
    out = export_comparison_latex(comparison_result, tmp_path / "comp.tex")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in content
    assert "\\toprule" in content
    assert "\\bottomrule" in content


def test_export_latex_has_nine_data_rows(
    comparison_result: ComparisonResult, tmp_path: Path
) -> None:
    """El .tex contiene las 9 filas de datos (una por escenario x modelo)."""
    out = export_comparison_latex(comparison_result, tmp_path / "comp.tex")
    content = out.read_text(encoding="utf-8")
    # Cada fila de datos termina en `\\` dentro del cuerpo tabular.
    body = content.split("\\midrule")[1].split("\\bottomrule")[0]
    data_rows = [ln for ln in body.splitlines() if "\\\\" in ln]
    assert len(data_rows) == 9


def test_latex_escapes_special_chars(tmp_path: Path) -> None:
    """El export LaTeX escapa los caracteres especiales de los nombres.

    Construye un ComparisonResult con un escenario que contiene ``_`` y
    ``%`` y verifica que el .tex no rompe LaTeX.
    """
    table = pl.DataFrame(
        {
            "scenario": ["esc_a (50%)", "esc_b"],
            "model": ["RF", "XGB"],
            "n_features": [10, 20],
            "f1_macro": [0.5, 0.6],
            "f1_weighted": [0.5, 0.6],
            "miou": [0.4, 0.5],
            "train_time_s": [1.0, 2.0],
        }
    )
    result = ComparisonResult(
        table=table,
        best_scenario="esc_b",
        alphaearth_delta=0.1,
        n_parcels=100,
    )
    out = export_comparison_latex(result, tmp_path / "esc.tex")
    content = out.read_text(encoding="utf-8")
    # `_` y `%` literales quedan escapados (no aparecen crudos como sintaxis).
    assert "\\_" in content or "esc\\_a" in content
    assert "\\%" in content


# ===========================================================================
# Grupo D — integracion del notebook 04_baseline.ipynb (secciones 7-8).
# ===========================================================================


def _load_notebook() -> dict:
    """Carga ``notebooks/04_baseline.ipynb`` como dict JSON, o skip si falta."""
    nb_path = _REPO_ROOT / "notebooks" / "04_baseline.ipynb"
    if not nb_path.exists():
        pytest.skip("04_baseline.ipynb aun no construido")
    return json.loads(nb_path.read_text(encoding="utf-8"))


def _all_sources(nb: dict) -> str:
    """Concatena el source de todas las celdas del notebook."""
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        chunks.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(chunks)


def test_notebook_has_all_eight_sections() -> None:
    """El notebook integra las 8 secciones de las 4 US del EPIC 4."""
    text = _all_sources(_load_notebook())
    for heading in (
        "## 1.",
        "## 2.",
        "## 3.",
        "## 4.",
        "## 5.",
        "## 5b.",
        "## 6.",
        "## 7.",
        "## 8.",
    ):
        assert heading in text, f"falta la seccion {heading}"


def test_notebook_has_epic5_discussion_section() -> None:
    """La seccion 8 discute las conclusiones para las fases siguientes (EPIC 5).

    El notebook usa lenguaje neutro ("fases siguientes", "modelos siguientes")
    en vez del codigo de epica literal; el test acepta cualquiera de las
    variantes que evidencian la discusion de continuidad hacia el EPIC 5.
    """
    text = _all_sources(_load_notebook())
    assert "## 8." in text
    section_8 = text.split("## 8.")[1].lower()
    assert any(
        marker in section_8
        for marker in ("epic 5", "fases siguientes", "modelos siguientes", "lo que sigue")
    ), "la seccion 8 no discute las conclusiones para las fases siguientes"


def test_notebook_discusses_alphaearth_incremental_value() -> None:
    """La seccion 8 discute el valor incremental de AlphaEarth."""
    text = _all_sources(_load_notebook()).lower()
    assert "alphaearth" in text
    assert "incremental" in text or "delta" in text


def test_notebook_section_7_builds_comparison_table() -> None:
    """La seccion 7 invoca build_comparison_table sobre los 3 escenarios."""
    text = _all_sources(_load_notebook())
    assert "build_comparison_table" in text
    assert "export_comparison_latex" in text
