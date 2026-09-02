"""Datasets sinteticos para los tests de ``ml.eval.comparison`` (US-022).

Provee :func:`make_three_scenarios`: tres DataFrames Polars alineados por
``parcel_id`` que reproducen la convencion de columnas de los tres
escenarios reales de la comparativa del baseline:

- **alphaearth** — bloque ``dim_00..dim_NN`` (el embedding de 64-dim,
  reducido a pocas dims en el sintetico).
- **s2_raw** — las 10 bandas ``B02_mean..B12_mean`` de Sentinel-2.
- **combined** — un bloque de indices espectrales con nombres realistas.

Los tres frames comparten ``parcel_id``, ``patch_id``, ``class_id`` y
``fold``, de modo que el *inner join* de
:func:`ml.eval.comparison._align_scenarios_by_parcel` los alinea sin
perdida. La separabilidad es alta para que RF/XGB converjan rapido en CI.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.datasets import make_classification

from ml.ingest.pastis_loader import PASTIS_S2_BANDS

# Indices espectrales realistas para el escenario combinado.
_SPECTRAL_NAMES: tuple[str, ...] = (
    "NDVI_mean",
    "NDVI_std",
    "EVI_p95",
    "NDWI_mean",
    "NDMI_p50",
    "SAVI_max",
    "NBR_min",
    "GNDVI_mean",
)


def make_three_scenarios(
    *,
    n: int = 240,
    n_classes: int = 4,
    seed: int = 42,
    drop_from_combined: int = 0,
) -> dict[str, pl.DataFrame]:
    """Genera los 3 escenarios sinteticos alineados por ``parcel_id``.

    Construye un problema de clasificacion base con
    :func:`sklearn.datasets.make_classification` y deriva tres vistas de
    las mismas parcelas, una por escenario, cada una con su propio bloque
    de features y la convencion de nombres del dataset real.

    Args:
        n: Numero de parcelas (filas) de cada escenario.
        n_classes: Numero de clases del problema.
        seed: Semilla determinista.
        drop_from_combined: Si ``> 0``, elimina las ultimas N filas del
            escenario ``combined`` — util para probar que el *inner join*
            de :func:`ml.eval.comparison._align_scenarios_by_parcel`
            reduce el conjunto al comun.

    Returns:
        Mapa ``{"alphaearth": df, "s2_raw": df, "combined": df}`` con los
        tres DataFrames Polars. Cada uno contiene ``parcel_id``,
        ``patch_id``, ``class_id``, ``fold`` y su bloque de features.
    """
    rng = np.random.default_rng(seed)

    # `parcel_id` con la convencion real "<patch_id>_<instance_id>".
    n_patches = max(2, n // 8)
    patch_ids = rng.integers(0, n_patches, size=n, dtype=np.int64)
    instance_ids = np.arange(1, n + 1, dtype=np.int64)
    parcel_ids = [f"{int(p)}_{int(i)}" for p, i in zip(patch_ids, instance_ids, strict=True)]
    folds = (instance_ids % 5 + 1).astype(np.int64)

    # Problema base compartido: las tres vistas describen las mismas
    # parcelas con feature blocks distintos pero la misma etiqueta.
    n_alphaearth = 8
    n_combined = len(_SPECTRAL_NAMES)
    n_s2 = len(PASTIS_S2_BANDS)
    total_features = n_alphaearth + n_combined + n_s2
    matrix, labels = make_classification(
        n_samples=n,
        n_features=total_features,
        n_informative=max(n_classes, total_features // 2),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.5,
        random_state=seed,
    )
    class_ids = labels.astype(np.int64)

    meta = {
        "parcel_id": parcel_ids,
        "patch_id": patch_ids,
        "class_id": class_ids,
        "fold": folds,
    }

    # Escenario (a) AlphaEarth: bloque dim_NN.
    ae_data: dict[str, object] = dict(meta)
    for i in range(n_alphaearth):
        ae_data[f"dim_{i:02d}"] = matrix[:, i].astype(np.float64)
    df_alphaearth = pl.DataFrame(ae_data)

    # Escenario (b) S2 crudo: 10 bandas B02_mean..B12_mean.
    s2_data: dict[str, object] = dict(meta)
    for i, band in enumerate(PASTIS_S2_BANDS):
        col = matrix[:, n_alphaearth + n_combined + i].astype(np.float64)
        # Reescala a un rango tipo reflectancia int16 promediada.
        s2_data[f"{band}_mean"] = (col * 200.0 + 2000.0).astype(np.float64)
    df_s2_raw = pl.DataFrame(s2_data)

    # Escenario (c) combinado: bloque de indices espectrales.
    combined_data: dict[str, object] = dict(meta)
    for i, name in enumerate(_SPECTRAL_NAMES):
        combined_data[name] = matrix[:, n_alphaearth + i].astype(np.float64)
    df_combined = pl.DataFrame(combined_data)

    if drop_from_combined > 0:
        df_combined = df_combined.head(max(0, n - drop_from_combined))

    return {
        "alphaearth": df_alphaearth,
        "s2_raw": df_s2_raw,
        "combined": df_combined,
    }


def write_three_scenarios(
    out_dir,  # type: ignore[no-untyped-def]
    **kwargs: object,
) -> dict[str, str]:
    """Escribe los 3 escenarios sinteticos a parquet y devuelve sus rutas.

    Util para los tests que ejercitan :func:`ml.eval.comparison.build_comparison_table`,
    cuya API recibe rutas de parquet en lugar de DataFrames en memoria.

    Args:
        out_dir: Directorio destino (``pathlib.Path`` o ``str``).
        **kwargs: Argumentos reenviados a :func:`make_three_scenarios`.

    Returns:
        Mapa ``{escenario: ruta_parquet}`` con las tres rutas escritas.
    """
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scenarios = make_three_scenarios(**kwargs)  # type: ignore[arg-type]
    paths: dict[str, str] = {}
    for key, df in scenarios.items():
        parquet_path = out_path / f"scenario_{key}.parquet"
        df.write_parquet(parquet_path)
        paths[key] = str(parquet_path)
    return paths
