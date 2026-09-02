"""Suite de tests US-018 extension — ``ml.features.encoding``.

Cubre los 30 pts del criterio "Construccion de features" del Avance 2
(rubrica) que el modulo ``selection.py`` no cubria:

- Grupo K — ordinal encoding (mapping explicito, valores desconocidos).
- Grupo L — one-hot Polars-nativo (sin pandas).
- Grupo M — target mean encoding con smoothing bayesiano (Galli 2022).
- Grupo N — derivacion de atributos (season desde DOY, crop_group desde
  class_id PASTIS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.features.encoding import (
    derive_crop_group_from_class_id,
    derive_season_from_doy,
    encode_onehot,
    encode_ordinal,
    encode_target_mean,
)
from tests.ml.features.fixtures.selection_synthetic import (
    make_categorical_fixture,
)

MODULE_PATH = Path(__file__).resolve().parents[3] / "ml" / "features" / "encoding.py"


# ---------------------------------------------------------------------------
# Grupo K — ordinal encoding
# ---------------------------------------------------------------------------


def test_encode_ordinal_basic() -> None:
    df = make_categorical_fixture(n=80, n_classes=4, seed=42)
    season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    encoded, report = encode_ordinal(df, {"season": season_map})
    assert encoded.get_column("season").dtype == pl.Int64
    values = set(encoded.get_column("season").to_list())
    assert values <= {0, 1, 2, 3}
    assert "season" in report
    assert report["season"]["mapping"] == season_map
    assert report["season"]["unknown_count"] == 0


def test_encode_ordinal_unknown_to_negative_one() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3],
            "year": [2024, 2024, 2024],
            "color": ["red", "blue", "purple"],
        }
    )
    mapping = {"color": {"red": 0, "blue": 1}}
    encoded, report = encode_ordinal(df, mapping)
    assert encoded.get_column("color").to_list() == [0, 1, -1]
    assert report["color"]["unknown_count"] == 1


def test_encode_ordinal_preserves_exclude_cols() -> None:
    df = make_categorical_fixture(n=40, n_classes=3, seed=1)
    # parcel_id no debe codificarse aunque entre en mapping (defensa).
    mapping = {"parcel_id": {1: 99}, "season": {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}}
    encoded, report = encode_ordinal(df, mapping)
    assert "parcel_id" not in report
    # parcel_id intacto (Int64 con valores originales).
    assert encoded.get_column("parcel_id").to_list() == list(range(1, 41))


def test_encode_ordinal_missing_column_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1, 2], "year": [2024, 2024], "a": ["x", "y"]})
    with pytest.raises(ValueError):
        encode_ordinal(df, {"nonexistent": {"x": 0}})


# ---------------------------------------------------------------------------
# Grupo L — one-hot encoding (Polars nativo, sin pandas)
# ---------------------------------------------------------------------------


def test_encode_onehot_polars_no_pandas() -> None:
    """Verifica que el modulo no importa pandas (regla CLAUDE.md)."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert "from pandas" not in source
    assert "pd.get_dummies" not in source


def test_encode_onehot_creates_expected_cols() -> None:
    df = make_categorical_fixture(n=60, n_classes=3, seed=42)
    encoded, report = encode_onehot(df, ["crop_group", "season"])
    # Cada categoria genera una columna `{col}__{categoria}`.
    assert "crop_group" not in encoded.columns
    assert all(c.startswith("crop_group__") for c in report["crop_group"])
    assert all(c.startswith("season__") for c in report["season"])
    # Cardinalidad observada: 3 grupos + 4 estaciones (max), pueden ser <=.
    assert 1 <= len(report["crop_group"]) <= 3
    assert 1 <= len(report["season"]) <= 4


def test_encode_onehot_drop_first() -> None:
    df = make_categorical_fixture(n=80, n_classes=4, seed=7)
    encoded_full, report_full = encode_onehot(df, ["crop_group"], drop_first=False)
    encoded_drop, report_drop = encode_onehot(df, ["crop_group"], drop_first=True)
    n_full = len(report_full["crop_group"])
    n_drop = len(report_drop["crop_group"])
    assert n_full == n_drop + 1
    assert encoded_drop.width == encoded_full.width - 1


def test_encode_onehot_missing_column_raises() -> None:
    df = pl.DataFrame({"parcel_id": [1], "year": [2024], "a": ["x"]})
    with pytest.raises(ValueError):
        encode_onehot(df, ["nonexistent"])


def test_encode_onehot_excludes_index_cols() -> None:
    df = make_categorical_fixture(n=30, n_classes=2, seed=0)
    # parcel_id en columns -> debe filtrarse defensivamente.
    _encoded, report = encode_onehot(df, ["parcel_id", "season"])
    assert "parcel_id" not in report
    assert "season" in report


# ---------------------------------------------------------------------------
# Grupo M — target mean encoding (bayesiano con smoothing)
# ---------------------------------------------------------------------------


def test_encode_target_mean_shape() -> None:
    df = make_categorical_fixture(n=120, n_classes=4, seed=42)
    encoded, report = encode_target_mean(
        df, target_col="yield_proxy", columns=["crop_group"], smoothing=5.0
    )
    assert "crop_group_target_enc" in encoded.columns
    assert encoded.height == df.height
    assert encoded.get_column("crop_group_target_enc").dtype == pl.Float64
    assert "global_mean" in report
    assert "per_column" in report
    assert "crop_group" in report["per_column"]


def test_encode_target_mean_smoothing_pulls_to_global() -> None:
    """smoothing alto -> encoded values cerca de global_mean.

    Construye un fixture donde el target varia fuertemente por categoria
    y compara smoothing=0 (puro mean por cat) vs smoothing=1e6 (debe
    aplanar a global_mean).
    """
    df = pl.DataFrame(
        {
            "parcel_id": list(range(1, 21)),
            "year": [2024] * 20,
            "cat": (["a"] * 10) + (["b"] * 10),
            "target": ([0.0] * 10) + ([10.0] * 10),
        }
    )
    enc_low, _ = encode_target_mean(df, "target", ["cat"], smoothing=0.0)
    enc_high, report_high = encode_target_mean(df, "target", ["cat"], smoothing=1e6)
    global_mean = report_high["global_mean"]
    low_vals = enc_low.get_column("cat_target_enc").to_list()
    high_vals = enc_high.get_column("cat_target_enc").to_list()
    # smoothing=0 -> los dos grupos quedan en 0 y 10.
    assert min(low_vals) == pytest.approx(0.0)
    assert max(low_vals) == pytest.approx(10.0)
    # smoothing alto -> todos cerca de global_mean.
    for v in high_vals:
        assert abs(v - global_mean) < 0.01


def test_encode_target_mean_perfect_separation() -> None:
    """Sin smoothing, la separacion debe ser perfecta."""
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3, 4],
            "year": [2024] * 4,
            "cat": ["a", "a", "b", "b"],
            "target": [1.0, 1.0, 5.0, 5.0],
        }
    )
    _enc, report = encode_target_mean(df, "target", ["cat"], smoothing=0.0)
    per_col = report["per_column"]["cat"]
    assert per_col["a"] == pytest.approx(1.0)
    assert per_col["b"] == pytest.approx(5.0)


def test_encode_target_mean_missing_target_raises() -> None:
    df = make_categorical_fixture(n=20, n_classes=2, seed=0)
    with pytest.raises(ValueError):
        encode_target_mean(df, "nonexistent_target", ["crop_group"])


def test_encode_target_mean_non_numeric_target_raises() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2],
            "year": [2024, 2024],
            "cat": ["a", "b"],
            "target": ["x", "y"],
        }
    )
    with pytest.raises(ValueError):
        encode_target_mean(df, "target", ["cat"])


# ---------------------------------------------------------------------------
# Grupo N — derivacion de atributos (season, crop_group)
# ---------------------------------------------------------------------------


def test_derive_season_from_doy_north_hemisphere() -> None:
    # 15-ene (DOY ~15), 15-abr (DOY ~105), 15-jul (DOY ~196), 15-oct (DOY ~288).
    series = pl.Series("peak_doy", [15, 105, 196, 288])
    seasons = derive_season_from_doy(series, hemisphere="north")
    out = seasons.to_list()
    assert out == ["winter", "spring", "summer", "autumn"]
    assert seasons.name == "peak_doy__season"


def test_derive_season_from_doy_south_hemisphere() -> None:
    series = pl.Series("peak_doy", [15, 105, 196, 288])
    seasons = derive_season_from_doy(series, hemisphere="south")
    out = seasons.to_list()
    # Invertidas vs norte.
    assert out == ["summer", "autumn", "winter", "spring"]


def test_derive_season_handles_nan() -> None:
    series = pl.Series("peak_doy", [15.0, float("nan"), 196.0])
    seasons = derive_season_from_doy(series)
    out = seasons.to_list()
    assert out[0] == "winter"
    assert out[1] == "unknown"
    assert out[2] == "summer"


def test_derive_crop_group_from_class_id_uses_hcat_taxonomy() -> None:
    """Debe usar la taxonomia HCAT oficial de EuroCrops (cereal, etc.)."""
    # class_id 2 = "Soft winter wheat" -> cereal
    # class_id 5 = "Winter rapeseed" -> industrial_nonfood
    # class_id 8 = "Grapevine" -> vineyard
    series = pl.Series("class_id", [2, 5, 8, 0, 19])
    groups = derive_crop_group_from_class_id(series)
    out = groups.to_list()
    assert out[0] == "cereal"
    assert out[1] == "industrial_nonfood"
    assert out[2] == "vineyard"
    # 0=background, 19=void; ambos en el mapeo HCAT inline.
    assert out[3] in {"background", "unknown"}
    assert out[4] in {"void", "unknown"}


def test_derive_crop_group_hcat_codes_are_valid() -> None:
    """Cada grupo HCAT productivo debe tener un codigo HCAT3 trazable."""
    from ml.features.encoding import _HCAT_GROUP_CODES

    expected_groups = {
        "cereal",
        "legume",
        "root_tuber",
        "industrial_nonfood",
        "vegetable",
        "grassland",
        "orchard",
        "vineyard",
    }
    assert expected_groups.issubset(_HCAT_GROUP_CODES.keys())
    # Los codigos productivos HCAT3 arrancan con 33 (rama crop_type).
    for grp in expected_groups:
        assert _HCAT_GROUP_CODES[grp].startswith("33")


def test_derive_crop_group_with_custom_mapping() -> None:
    series = pl.Series("class_id", [1, 2, 99])
    custom = {1: "alpha", 2: "beta"}
    groups = derive_crop_group_from_class_id(series, mapping=custom)
    out = groups.to_list()
    assert out == ["alpha", "beta", "unknown"]


def test_derive_crop_group_handles_none_name() -> None:
    series = pl.Series([1, 2])
    groups = derive_crop_group_from_class_id(series)
    # Default name path activado.
    assert groups.name in {"crop_group", "__group"}


# ---------------------------------------------------------------------------
# Smoke integration: encoding pipeline completo sobre fixture
# ---------------------------------------------------------------------------


def test_encoding_pipeline_end_to_end() -> None:
    """Simula el flujo del notebook Avance2: deriva, ordinal, onehot, target."""
    df = make_categorical_fixture(n=200, n_classes=5, seed=42)
    # 1. Derivacion: peak_doy -> season.
    season_series = derive_season_from_doy(df.get_column("peak_doy"))
    df2 = df.with_columns(season_series.rename("season_derived"))
    # 2. Ordinal sobre estacion derivada.
    season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3, "unknown": -1}
    df3, _ = encode_ordinal(df2, {"season_derived": season_map})
    # 3. One-hot sobre crop_group.
    df4, oh_report = encode_onehot(df3, ["crop_group"])
    # 4. Target mean sobre season original (Utf8 todavia).
    df5, te_report = encode_target_mean(
        df4, target_col="yield_proxy", columns=["season"], smoothing=10.0
    )
    assert df5.height == 200
    assert df5.get_column("season_derived").dtype == pl.Int64
    assert "season_target_enc" in df5.columns
    assert len(oh_report["crop_group"]) >= 1
    assert "season" in te_report["per_column"]
    # Sin NaN en columnas codificadas (proxy de integridad numerica).
    assert np.isfinite(np.asarray(df5.get_column("season_target_enc").to_list())).all()
