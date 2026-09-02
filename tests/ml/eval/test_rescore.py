"""Consolidation / structure tests of the fold-5 re-score harness (US-030).

These tests cover the **pure consolidation layer** of the harness (registry +
fold-5 table/figure) without ever running real inference: the 6 ``best.pt`` are
DVC/gitignored (~2 GB) and AnySat pulls its encoder over ``torch.hub``. The real
metrics are produced at closure; here we assert the contract:

- ``CHECKPOINT_REGISTRY`` maps the canonical models (no HCAT/alt-* leaks): the
  6 US-030 originals + ``tsvit`` (US-038 Full-M base) + ``tsvit-pheno-fullm``
  (US-039 Full-M pheno) = 8 entries.
- ``resolve_state_dict`` tolerates the 3 checkpoint conventions.
- ``build_fold5_comparison_table`` consolidates a re-score DataFrame into the
  ``model_comparison_fold5.csv`` (6 rows, contract columns, floats in ``[0, 1]``,
  sorted by ``miou`` descending).
- ``fold5_barplot_figure`` returns a matplotlib ``Figure``.
- ``rescore_all_checkpoints`` (harness core, ml/A) yields ``status="missing"``
  for absent checkpoints instead of raising, when reachable via a stubbed
  registry — exercised only if the harness symbol is present.

Pattern mirrors ``tests/ml/eval/test_comparison.py`` (synthetic frames +
``tmp_path``). Workaround R7 (heredado): ``coverage run --include=...`` en vez de
``pytest --cov`` por la incompatibilidad numpy 2.3.5 / scipy 1.17.1.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.eval.checkpoint_registry import (
    CHECKPOINT_REGISTRY,
    CheckpointSpec,
    resolve_state_dict,
)
from ml.eval.comparison import (
    _FOLD5_TABLE_COLUMNS,
    build_fold5_comparison_table,
    fold5_barplot_figure,
)

# Canonical model set of the harness (AnySat, NOT Swin-UNETR). US-038 added the
# base ``tsvit`` (Full-M retrain) alongside the original L4 ``tsvit-pheno``;
# US-039 adds ``tsvit-pheno-fullm`` (Full-M + phenology contrastive branch,
# coexisting with the L4 ``tsvit-pheno`` so the published US-030 fold-5 table
# stays comparable, R10). The harness now re-scores 8 entries apples-to-apples.
_EXPECTED_MODELS: frozenset[str] = frozenset(
    {
        "unet",
        "deeplabv3plus",
        "segformer",
        "utae",
        "tsvit",
        "tsvit-pheno",
        "tsvit-pheno-fullm",
        "anysat",
    }
)

# Variants that must NEVER leak into the registry (duplicated / HCAT / stray
# mlruns weights). ``alt-tsvit-fullm-v1`` (US-038) and ``tsvit-pheno-fullm-v1``
# (US-039) are LEGITIMATE Full-M checkpoints; the forbidden fragments are the
# historical L4 / stray runs (``alt-tsvit-v1``, ``alt-tsvit-pheno``, the bare
# ``tsvit-v1`` stray). NOTE: ``tsvit-v1`` is NOT a substring of the legitimate
# ``tsvit-pheno-v1`` (L4 ``tsvit-pheno`` entry, R10) nor of
# ``tsvit-pheno-fullm-v1`` (US-039), so the guard stays sharp.
_FORBIDDEN_FRAGMENTS: tuple[str, ...] = (
    "hcat",
    "alt-tsvit-v1",
    "alt-tsvit-pheno",
    "tsvit-v1",
    "mlruns",
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic fold-5 re-score frame (no inference).
# ---------------------------------------------------------------------------


def _make_fold5_frame(*, with_missing: bool = False) -> pl.DataFrame:
    """Build a synthetic re-score DataFrame with the harness contract schema.

    Reproduces the columns of
    :func:`ml.eval.dense_metrics.rescore_all_checkpoints` for the 6 models with
    deterministic, plausible metrics. When ``with_missing`` is True, one model
    carries ``status="missing"`` with null metrics (skip-graceful contract).

    Args:
        with_missing: If True, mark ``anysat`` as a missing checkpoint.

    Returns:
        A 6-row Polars DataFrame with the contract columns.
    """
    rows: list[dict[str, object]] = [
        {
            "model": "tsvit-pheno",
            "miou": 0.6253,
            "f1_macro": 0.7500,
            "pixel_accuracy": 0.8759,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
        {
            "model": "utae",
            "miou": 0.5101,
            "f1_macro": 0.6402,
            "pixel_accuracy": 0.8210,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
        {
            "model": "unet",
            "miou": 0.3502,
            "f1_macro": 0.4803,
            "pixel_accuracy": 0.7204,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
        {
            "model": "deeplabv3plus",
            "miou": 0.2709,
            "f1_macro": 0.3864,
            "pixel_accuracy": 0.6743,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
        {
            "model": "segformer",
            "miou": 0.1805,
            "f1_macro": 0.2906,
            "pixel_accuracy": 0.5907,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
        {
            "model": "anysat",
            "miou": 0.2204,
            "f1_macro": 0.3305,
            "pixel_accuracy": 0.6108,
            "fold": 5,
            "n_patches": 480,
            "status": "ok",
        },
    ]
    if with_missing:
        rows[-1] = {
            "model": "anysat",
            "miou": None,
            "f1_macro": None,
            "pixel_accuracy": None,
            "fold": 5,
            "n_patches": 0,
            "status": "missing",
        }

    schema = {
        "model": pl.Utf8,
        "miou": pl.Float64,
        "f1_macro": pl.Float64,
        "pixel_accuracy": pl.Float64,
        "fold": pl.Int64,
        "n_patches": pl.Int64,
        "status": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


# ===========================================================================
# Grupo A — registry (mapeo explicito de los 6 checkpoints).
# ===========================================================================


def test_registry_has_exactly_six_models() -> None:
    """El registry mapea los canonicos (US-030 6 + US-038 tsvit + US-039 pheno = 8)."""
    assert set(CHECKPOINT_REGISTRY) == _EXPECTED_MODELS
    assert len(CHECKPOINT_REGISTRY) == 8


def test_registry_excludes_duplicated_variants() -> None:
    """Ninguna ruta apunta a variantes HCAT / alt-* / tsvit-v1 / mlruns (AC-8)."""
    for spec in CHECKPOINT_REGISTRY.values():
        path_str = str(spec.path).lower().replace("\\", "/")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in path_str, (
                f"{spec.name} apunta a una variante prohibida: {spec.path}"
            )


def test_registry_paths_match_contract() -> None:
    """Cada modelo apunta a su ruta real verificada (CONTRATO US-030).

    Las rutas se anclan al repo root via ``__file__`` (absolutas, robustas al cwd
    del que invoca el harness), asi que el contrato se valida sobre el sufijo
    relativo y que la ruta sea absoluta.
    """
    expected = {
        "unet": "checkpoints/segmentation/unet-aaron/unet_pastis.pt",
        "deeplabv3plus": "checkpoints/segmentation/deeplab-18/best.pt",
        "segformer": "checkpoints/segmentation/segformer-isaac/hf_model",
        "utae": "checkpoints/segmentation/utae-isaac/best_model.pt",
        "tsvit": "checkpoints/segmentation/alt-tsvit-fullm-v1/best.pt",
        "tsvit-pheno": "checkpoints/segmentation/tsvit-pheno-v1/best.pt",
        "tsvit-pheno-fullm": "checkpoints/segmentation/tsvit-pheno-fullm-v1/best.pt",
        "anysat": "checkpoints/segmentation/anysat-aaron/anysat_pastis.pt",
    }
    for name, rel in expected.items():
        path = CHECKPOINT_REGISTRY[name].path
        actual = str(path).replace("\\", "/")
        assert path.is_absolute(), f"{name} debe anclarse al repo root (absoluta)"
        assert actual.endswith(rel), f"{name}: {actual} no termina en {rel}"


def test_registry_segformer_is_three_rgb_and_resize() -> None:
    """SegFormer va con 3 bandas RGB y needs_resize=True (256->128) (R2)."""
    spec = CHECKPOINT_REGISTRY["segformer"]
    assert spec.in_channels == 3
    assert spec.needs_resize is True
    assert spec.native_num_classes == 20


def test_registry_native_class_conventions() -> None:
    """18-clase nativos ignore=255; 20-clase ignore=19 (R6)."""
    for name in ("deeplabv3plus", "tsvit", "tsvit-pheno", "tsvit-pheno-fullm"):
        spec = CHECKPOINT_REGISTRY[name]
        assert spec.native_num_classes == 18
        assert spec.native_ignore_index == 255
    for name in ("unet", "segformer", "utae", "anysat"):
        spec = CHECKPOINT_REGISTRY[name]
        assert spec.native_num_classes == 20
        assert spec.native_ignore_index == 19


def test_checkpointspec_is_frozen() -> None:
    """CheckpointSpec es un dataclass inmutable (frozen)."""
    spec = CHECKPOINT_REGISTRY["unet"]
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "tampered"  # type: ignore[misc]


# ===========================================================================
# Grupo B — resolve_state_dict (3 convenciones de clave, R5).
# ===========================================================================


def test_resolve_state_dict_model_state_key() -> None:
    """Convencion 1: dict bajo `model_state` (DeepLab / TSViT-pheno)."""
    spec = CHECKPOINT_REGISTRY["deeplabv3plus"]
    inner = {"layer.weight": [1, 2, 3]}
    loaded = {"model_state": inner, "epoch": 7}
    assert resolve_state_dict(loaded, spec) is inner


def test_resolve_state_dict_model_state_dict_key() -> None:
    """Convencion 2: dict bajo `model_state_dict` + val_miou (U-TAE)."""
    spec = CHECKPOINT_REGISTRY["utae"]
    inner = {"in_conv.weight": [0.1]}
    loaded = {"model_state_dict": inner, "val_miou": 0.51}
    assert resolve_state_dict(loaded, spec) is inner


def test_resolve_state_dict_pure_state_dict() -> None:
    """Convencion 3: state_dict puro sin wrapper (U-Net / AnySat)."""
    spec = CHECKPOINT_REGISTRY["unet"]
    pure = {"encoder.conv1.weight": [0.0], "decoder.out.bias": [0.0]}
    # Ninguna clave candidata mapea a un dict -> se usa tal cual.
    assert resolve_state_dict(pure, spec) is pure


def test_resolve_state_dict_rejects_non_dict() -> None:
    """Un objeto que no es dict (ni state_dict) lanza TypeError."""
    spec = CHECKPOINT_REGISTRY["unet"]
    with pytest.raises(TypeError):
        resolve_state_dict([1, 2, 3], spec)  # type: ignore[arg-type]


# ===========================================================================
# Grupo C — consolidacion tabla/figura fold-5.
# ===========================================================================


def test_rescore_table_schema(tmp_path: Path) -> None:
    """La tabla consolidada: 6 filas, columnas contrato, floats [0,1], orden desc."""
    df = _make_fold5_frame()
    table = build_fold5_comparison_table(df, out_dir=tmp_path)

    assert table.height == 6
    for col in _FOLD5_TABLE_COLUMNS:
        assert col in table.columns

    # Floats de metrica en [0, 1].
    for metric in ("miou", "f1_macro", "pixel_accuracy"):
        values = [v for v in table.get_column(metric).to_list() if v is not None]
        assert all(0.0 <= v <= 1.0 for v in values), metric

    # Ordenado por miou descendente (nulls al final).
    miou = [v for v in table.get_column("miou").to_list() if v is not None]
    assert miou == sorted(miou, reverse=True)
    # El mejor modelo (tsvit-pheno) queda primero.
    assert table.get_column("model").to_list()[0] == "tsvit-pheno"


def test_rescore_skip_missing(tmp_path: Path) -> None:
    """Un checkpoint faltante -> fila status='missing' sin excepcion (AC-7)."""
    df = _make_fold5_frame(with_missing=True)
    table = build_fold5_comparison_table(df, out_dir=tmp_path)

    assert table.height == 6
    statuses = table.get_column("status").to_list()
    assert "missing" in statuses

    missing_row = table.filter(pl.col("status") == "missing")
    assert missing_row.height == 1
    assert missing_row.get_column("model").to_list() == ["anysat"]
    # La fila missing va al final por miou null (nulls_last) y no rompe el orden.
    assert table.get_column("model").to_list()[-1] == "anysat"


def test_csv_figure_written(tmp_path: Path) -> None:
    """build_fold5_comparison_table escribe el CSV; fold5_barplot_figure -> Figure."""
    from matplotlib.figure import Figure

    df = _make_fold5_frame()
    table = build_fold5_comparison_table(df, out_dir=tmp_path)

    csv_path = tmp_path / "model_comparison_fold5.csv"
    assert csv_path.exists()
    reloaded = pl.read_csv(csv_path)
    assert reloaded.height == 6
    assert set(_FOLD5_TABLE_COLUMNS).issubset(reloaded.columns)

    fig = fold5_barplot_figure(table)
    assert isinstance(fig, Figure)
    # Un eje con barras de mIoU + F1-macro.
    assert len(fig.axes) == 1


def test_build_fold5_table_missing_column_raises(tmp_path: Path) -> None:
    """Falta una columna del contrato -> ValueError explicito."""
    df = _make_fold5_frame().drop("pixel_accuracy")
    with pytest.raises(ValueError, match="pixel_accuracy"):
        build_fold5_comparison_table(df, out_dir=tmp_path)


def test_build_fold5_table_from_parquet_path(tmp_path: Path) -> None:
    """Acepta una ruta de parquet ademas de un DataFrame en memoria."""
    df = _make_fold5_frame()
    src = tmp_path / "rescore.parquet"
    df.write_parquet(src)
    table = build_fold5_comparison_table(src, out_dir=tmp_path)
    assert table.height == 6


def test_build_fold5_table_missing_path_raises(tmp_path: Path) -> None:
    """Ruta inexistente -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_fold5_comparison_table(tmp_path / "nope.parquet", out_dir=tmp_path)


def test_fold5_barplot_skips_missing_status() -> None:
    """El barplot omite filas status!='ok' pero no falla."""
    from matplotlib.figure import Figure

    df = _make_fold5_frame(with_missing=True)
    fig = fold5_barplot_figure(df)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # 5 modelos 'ok' x 2 barras (mIoU + F1) = 10 barras; anysat (missing) fuera.
    bar_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "anysat" not in bar_labels
    assert len(bar_labels) == 5


# ===========================================================================
# Grupo D — harness core (rescore_all_checkpoints), solo si esta presente.
# ===========================================================================


def test_rescore_all_checkpoints_skip_missing_via_stub_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rescore_all_checkpoints con registry de rutas inexistentes no crashea.

    Verifica la garantia skip-graceful (AC-7) a nivel del harness core (ml/A):
    un registry de stubs apuntando a rutas inexistentes debe producir filas
    ``status="missing"`` sin excepcion. Se omite si el harness aun no expone
    ``rescore_all_checkpoints`` (orden de merge A/B/C).
    """
    dense_metrics = pytest.importorskip("ml.eval.dense_metrics")
    rescore = getattr(dense_metrics, "rescore_all_checkpoints", None)
    if rescore is None:
        pytest.skip("rescore_all_checkpoints aun no integrado (ml/A pendiente)")

    stub_registry: dict[str, CheckpointSpec] = {
        "unet": CheckpointSpec(
            name="unet",
            model_kind="unet",
            path=Path("does/not/exist/unet.pt"),
            native_num_classes=20,
            native_ignore_index=19,
        ),
        "deeplabv3plus": CheckpointSpec(
            name="deeplabv3plus",
            model_kind="deeplabv3plus",
            path=Path("does/not/exist/deeplab.pt"),
            native_num_classes=18,
            native_ignore_index=255,
        ),
    }

    result = rescore(stub_registry, fold=5, skip_missing=True, max_patches=1)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 2
    assert set(result.get_column("status").to_list()) == {"missing"}
