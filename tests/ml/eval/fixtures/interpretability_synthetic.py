"""Modelos y datos sinteticos para los tests de ``ml.eval.interpretability``.

Provee ``make_trained_tree_model``: un Random Forest o XGBoost ya ajustado
sobre un dataset sintetico separable, autocontenido — no depende de los
artefactos joblib de US-019 ni del parquet de US-018.

El dataset reproduce la convencion de nombres del baseline real: un bloque de
dimensiones AlphaEarth (``dim_00``..``dim_NN``) y un bloque de indices
espectrales (``NDVI_mean``, ``EVI_p95``, ...), de modo que los tests de
``is_alphaearth_dim`` y ``alphaearth_dominance_table`` operen sobre nombres
realistas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

ModelKind = Literal["rf", "xgb"]

# Indices espectrales usados para nombrar el bloque no-AlphaEarth del dataset
# sintetico (subconjunto de los reales del parquet de US-018).
_SPECTRAL_NAMES: tuple[str, ...] = (
    "NDVI_mean",
    "EVI_p95",
    "NDWI_std",
    "NDMI_p50",
    "SAVI_max",
    "NBR_min",
)


@dataclass(frozen=True)
class SyntheticModel:
    """Bundle de un modelo de arboles sintetico ya ajustado.

    Attributes:
        model: Estimador ``RandomForestClassifier`` o ``XGBClassifier``
            ajustado sobre ``X``/``y``.
        model_kind: ``"rf"`` o ``"xgb"``.
        X: ``pl.DataFrame`` de features (mezcla de dims AlphaEarth e indices).
        y: ``pl.Series`` con la clase codificada en ``0..n_classes-1``.
        feature_cols: Nombres de las features en el orden de las columnas.
        n_classes: Numero de clases del problema.
    """

    model: ClassifierMixin
    model_kind: ModelKind
    X: pl.DataFrame
    y: pl.Series
    feature_cols: tuple[str, ...]
    n_classes: int


def _build_feature_names(n_features: int) -> tuple[str, ...]:
    """Construye nombres de feature mezclando dims AlphaEarth e indices.

    Las primeras columnas son dimensiones AlphaEarth (``dim_00``, ``dim_01``,
    ...); el resto recibe nombres de indices espectrales reciclando
    ``_SPECTRAL_NAMES``.

    Args:
        n_features: Numero total de features a nombrar.

    Returns:
        Tupla de ``n_features`` nombres unicos.
    """
    n_alphaearth = max(2, n_features // 2)
    names: list[str] = [f"dim_{i:02d}" for i in range(n_alphaearth)]
    for i in range(n_features - n_alphaearth):
        base = _SPECTRAL_NAMES[i % len(_SPECTRAL_NAMES)]
        # Sufijo numerico para garantizar unicidad si se recicla un nombre.
        suffix = i // len(_SPECTRAL_NAMES)
        names.append(base if suffix == 0 else f"{base}_{suffix}")
    return tuple(names)


def make_trained_tree_model(
    kind: ModelKind = "rf",
    *,
    n_classes: int = 4,
    n_features: int = 12,
    n_samples: int = 240,
    seed: int = 42,
) -> SyntheticModel:
    """Entrena un RF/XGB sintetico sobre un dataset separable y reproducible.

    El dataset se genera con :func:`sklearn.datasets.make_classification` con
    clases bien separadas para que el modelo aprenda senal real (importancias y
    valores SHAP no triviales) en pocos arboles y de forma rapida.

    Args:
        kind: ``"rf"`` para :class:`RandomForestClassifier` o ``"xgb"`` para
            :class:`xgboost.XGBClassifier`.
        n_classes: Numero de clases del problema sintetico.
        n_features: Numero de features (la mitad nombradas como dims
            AlphaEarth, la otra mitad como indices espectrales).
        n_samples: Numero de filas del dataset sintetico.
        seed: Semilla del generador y del estimador.

    Returns:
        Un :class:`SyntheticModel` con el modelo ajustado y sus datos.

    Raises:
        ValueError: si ``kind`` no es ``"rf"`` ni ``"xgb"``.
    """
    if kind not in ("rf", "xgb"):
        raise ValueError(f"`kind` debe ser 'rf' o 'xgb'; recibido {kind!r}.")

    n_informative = max(n_classes, n_features // 2)
    matrix, labels = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    feature_cols = _build_feature_names(n_features)

    if kind == "rf":
        model: ClassifierMixin = RandomForestClassifier(
            n_estimators=25,
            max_depth=6,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(matrix, labels)
    else:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=25,
            max_depth=4,
            learning_rate=0.2,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(matrix, labels)

    X = pl.DataFrame({name: matrix[:, idx] for idx, name in enumerate(feature_cols)})
    y = pl.Series("class_id", labels.astype(np.int64))
    return SyntheticModel(
        model=model,
        model_kind=kind,
        X=X,
        y=y,
        feature_cols=feature_cols,
        n_classes=n_classes,
    )
