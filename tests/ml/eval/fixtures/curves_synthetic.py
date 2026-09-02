"""Datasets sinteticos para los tests de ``ml.eval.learning_curves`` (US-021).

Provee :func:`make_curve_dataset`: un DataFrame Polars con la convencion de
columnas del baseline real (``parcel_id``, ``patch_id``, ``class_id`` y un
bloque de features) y una **separabilidad ajustable** que permite forzar de
forma determinista cada veredicto de :func:`ml.eval.learning_curves.diagnose_fit`:

- ``separability="clean"``    -> clases bien separadas  -> ``good_fit``.
- ``separability="low"``      -> features casi ruido    -> ``underfit``.
- ``separability="memorizable"`` -> muchas features, pocas muestras -> ``overfit``.

Tambien expone :func:`make_cv_splits`, que produce una lista materializada de
splits ``(train_idx, test_idx)`` analoga a la salida de
``ml.train.baseline._build_cv_splits`` — autocontenida, sin tocar el parquet de
US-018 ni el CV espacial real (que es O(N^2) y lento para tests unitarios).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.datasets import make_classification

Separability = Literal["clean", "low", "memorizable"]

# Indices espectrales para nombrar el bloque no-AlphaEarth (subconjunto real).
_SPECTRAL_NAMES: tuple[str, ...] = (
    "NDVI_mean",
    "EVI_p95",
    "NDWI_std",
    "NDMI_p50",
    "SAVI_max",
    "NBR_min",
)


def _feature_names(n_features: int) -> list[str]:
    """Construye nombres de feature mezclando dims AlphaEarth e indices.

    Args:
        n_features: Numero total de features a nombrar.

    Returns:
        Lista de ``n_features`` nombres unicos.
    """
    n_alphaearth = max(2, n_features // 2)
    names: list[str] = [f"dim_{i:02d}" for i in range(n_alphaearth)]
    for i in range(n_features - n_alphaearth):
        base = _SPECTRAL_NAMES[i % len(_SPECTRAL_NAMES)]
        suffix = i // len(_SPECTRAL_NAMES)
        names.append(base if suffix == 0 else f"{base}_{suffix}")
    return names


def make_curve_dataset(
    *,
    n: int = 300,
    n_classes: int = 4,
    n_features: int = 12,
    separability: Separability = "clean",
    seed: int = 42,
) -> pl.DataFrame:
    """Genera un DataFrame Polars sintetico con separabilidad ajustable.

    El dataset reproduce las columnas obligatorias del baseline (``parcel_id``,
    ``patch_id``, ``class_id``) mas un bloque de features. La separabilidad
    controla el veredicto esperado de ``diagnose_fit``:

    - ``"clean"``: ``class_sep`` alto y suficientes muestras -> ``good_fit``.
    - ``"low"``: ``class_sep`` casi nulo, features informativas escasas ->
      tanto train como validacion bajos -> ``underfit``.
    - ``"memorizable"``: muchas features y pocas muestras por clase -> el modelo
      memoriza el train pero no generaliza -> ``overfit``.

    Args:
        n: Numero de filas (parcelas) del dataset.
        n_classes: Numero de clases del problema.
        n_features: Numero de features (la mitad como dims AlphaEarth).
        separability: Regimen de separabilidad (``"clean"``, ``"low"`` o
            ``"memorizable"``).
        seed: Semilla determinista del generador.

    Returns:
        DataFrame Polars con ``parcel_id``, ``patch_id``, ``class_id`` y las
        columnas de feature.

    Raises:
        ValueError: si ``separability`` no es un valor soportado.
    """
    if separability not in ("clean", "low", "memorizable"):
        raise ValueError(
            f"`separability` debe ser 'clean', 'low' o 'memorizable'; recibido {separability!r}."
        )

    if separability == "clean":
        class_sep = 3.0
        n_informative = max(n_classes, n_features // 2)
    elif separability == "low":
        # Casi ruido: una sola feature debilmente informativa, sin separacion.
        class_sep = 0.05
        n_informative = max(n_classes, 2)
    else:  # memorizable
        # Todas las features informativas pero ruidosas: con pocas muestras por
        # clase el modelo las memoriza sin generalizar.
        class_sep = 0.6
        n_informative = n_features

    n_redundant = max(0, n_features - n_informative)
    matrix, labels = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0.25 if separability == "low" else 0.0,
        random_state=seed,
    )
    feature_cols = _feature_names(n_features)

    rng = np.random.default_rng(seed)
    # `patch_id` agrupa parcelas: varias parcelas comparten patch (como PASTIS-R).
    n_patches = max(2, n // 10)
    patch_ids = rng.integers(0, n_patches, size=n, dtype=np.int64)

    data: dict[str, object] = {
        "parcel_id": np.arange(n, dtype=np.int64),
        "patch_id": patch_ids,
        "class_id": labels.astype(np.int64),
    }
    for idx, name in enumerate(feature_cols):
        data[name] = matrix[:, idx].astype(np.float64)
    return pl.DataFrame(data)


def make_cv_splits(
    n: int,
    *,
    k: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye una lista materializada de splits ``(train_idx, test_idx)``.

    Particiona ``[0, n)`` en ``k`` bloques contiguos disjuntos: cada bloque es
    el test de un fold y el resto es su train. Es analoga a la salida de
    ``ml.train.baseline._build_cv_splits`` (lista de tuplas de indices
    posicionales) pero autocontenida y rapida — los tests unitarios no necesitan
    el CV espacial real H3+KMeans, que es O(N^2).

    Args:
        n: Numero total de muestras.
        k: Numero de folds.
        seed: Semilla del shuffle previo a la particion (folds no contiguos).

    Returns:
        Lista de ``k`` tuplas ``(train_idx, test_idx)`` de arrays ``np.int64``.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).astype(np.int64)
    blocks = np.array_split(order, k)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(k):
        test_idx = np.sort(blocks[fold_idx]).astype(np.int64)
        train_idx = np.sort(np.concatenate([blocks[j] for j in range(k) if j != fold_idx])).astype(
            np.int64
        )
        splits.append((train_idx, test_idx))
    return splits
