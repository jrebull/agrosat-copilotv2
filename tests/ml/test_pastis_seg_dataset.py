"""Tests para ``ml.data.pastis_seg_dataset.PASTISSegmentationDataset`` (US-025).

La mayor parte de los tests corre sobre un **mini-fixture sintetico** escrito
en ``tmp_path`` (3-4 patches de ``(T,10,16,16)`` con folds y clases conocidas),
de modo que no dependen de ``data/PASTIS-R/`` ni del disco lento. Esto permite
asertar de forma exacta shapes, rangos de etiquetas, determinismo del
submuestreo temporal, fallback de normalizacion y disjuntez del split por fold.

Un unico smoke ``@pytest.mark.slow`` toca el dataset real si esta descargado
(folds=(1,), 2 patches) para verificar el contrato end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_REAL_S2_DIR = _REAL_PASTIS_ROOT / "DATA_S2"

_pastis_present = _REAL_S2_DIR.exists() and any(_REAL_S2_DIR.glob("S2_*.npy"))

# H/W reducido para que los .npy sinteticos sean diminutos (128px nativo no es
# necesario: el dataset no asume tamano fijo, solo (T,10,H,W) / (H,W)).
_H = _W = 16
_N_BANDS = 10


def _write_synthetic_patch(
    root: Path,
    patch_id: str,
    *,
    n_timesteps: int,
    fold: int,
    semantic_fill: np.ndarray,
    s2_value: int = 5000,
) -> None:
    """Escribe un patch PASTIS-R sintetico (S2 + TARGET) a disco.

    Args:
        root: Raiz del dataset sintetico (se crean ``DATA_S2/`` y
            ``ANNOTATIONS/``).
        patch_id: Identificador del patch.
        n_timesteps: Numero de fechas ``T`` del tensor S2.
        fold: Fold oficial del patch (se anota en ``metadata.geojson`` aparte).
        semantic_fill: Mascara semantica ``(H, W)`` uint8 (canal 0 de TARGET).
        s2_value: Valor int16 constante de la reflectancia (escala 0..10000).
    """
    s2_dir = root / "DATA_S2"
    ann_dir = root / "ANNOTATIONS"
    s2_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # S2 con un gradiente temporal para distinguir submuestreos (cada frame t
    # tiene un valor base distinto -> el submuestreo es observable).
    s2 = np.zeros((n_timesteps, _N_BANDS, _H, _W), dtype=np.int16)
    for t in range(n_timesteps):
        s2[t] = np.int16(s2_value + t * 100)
    np.save(s2_dir / f"S2_{patch_id}.npy", s2)

    target = np.stack(
        [semantic_fill, np.zeros_like(semantic_fill), np.zeros_like(semantic_fill)],
        axis=0,
    ).astype(np.uint8)
    np.save(ann_dir / f"TARGET_{patch_id}.npy", target)


def _write_metadata(root: Path, fold_by_pid: dict[str, int]) -> None:
    """Escribe un ``metadata.geojson`` minimo con ``ID_PATCH`` + ``Fold``.

    Args:
        root: Raiz del dataset sintetico.
        fold_by_pid: Mapa ``{patch_id: fold}``.
    """
    features = [
        {
            "id": pid,
            "type": "Feature",
            "properties": {"ID_PATCH": int(pid), "Fold": int(fold), "TILE": "T30UXV"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]]],
            },
        }
        for pid, fold in fold_by_pid.items()
    ]
    (root / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_pastis(tmp_path: Path) -> Path:
    """Construye un mini PASTIS-R sintetico de 4 patches en folds 1, 2 y 4.

    Layout:
        - patch 10000 (fold 1): T=12, clases {1, 2} (mapean a 0, 1 en semantic18)
        - patch 10001 (fold 1): T=8,  clase 3
        - patch 20000 (fold 2): T=15, clases {0 (bg), 19 (void), 5}
        - patch 40000 (fold 4): T=10, clase 2

    Returns:
        ``Path`` a la raiz del dataset sintetico (sin ``NORM_S2_patch.json``,
        para forzar el fallback ``/10000``).
    """
    root = tmp_path / "PASTIS-R"

    sem_a = np.zeros((_H, _W), dtype=np.uint8)
    sem_a[:, : _W // 2] = 1  # clase 1 -> label 0
    sem_a[:, _W // 2 :] = 2  # clase 2 -> label 1
    _write_synthetic_patch(root, "10000", n_timesteps=12, fold=1, semantic_fill=sem_a)

    sem_b = np.full((_H, _W), 3, dtype=np.uint8)  # clase 3 -> label 2
    _write_synthetic_patch(root, "10001", n_timesteps=8, fold=1, semantic_fill=sem_b)

    sem_c = np.zeros((_H, _W), dtype=np.uint8)
    sem_c[: _H // 3, :] = 0  # Background -> ignore
    sem_c[_H // 3 : 2 * _H // 3, :] = 19  # Void -> ignore
    sem_c[2 * _H // 3 :, :] = 5  # clase 5 -> label 4
    _write_synthetic_patch(root, "20000", n_timesteps=15, fold=2, semantic_fill=sem_c)

    sem_d = np.full((_H, _W), 2, dtype=np.uint8)
    _write_synthetic_patch(root, "40000", n_timesteps=10, fold=4, semantic_fill=sem_d)

    _write_metadata(root, {"10000": 1, "10001": 1, "20000": 2, "40000": 4})
    return root


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_2d_mode_shape(synthetic_pastis: Path) -> None:
    """Modo 2D (``collapse_time='median'``) entrega ``x (10,H,W)`` + ``y (H,W)``."""
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), collapse_time="median")
    x, y = ds[0]
    assert x.shape == (_N_BANDS, _H, _W)
    assert y.shape == (_H, _W)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_temporal_mode_shape(synthetic_pastis: Path) -> None:
    """Modo temporal (``collapse_time=None``) entrega ``x (T_sub,10,H,W)``.

    El patch 10000 tiene T=12; con ``n_timesteps=10`` el submuestreo
    equiespaciado deja ``T_sub = 10`` fechas.
    """
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1,), n_timesteps=10, collapse_time=None
    )
    # patch 10000 (idx 0 tras orden numerico) tiene T=12.
    x, y = ds[0]
    assert x.ndim == 4
    t_sub = x.shape[0]
    assert t_sub <= 10
    assert x.shape[1:] == (_N_BANDS, _H, _W)
    assert y.shape == (_H, _W)


def test_temporal_keeps_all_when_t_below_n(synthetic_pastis: Path) -> None:
    """Si ``T < n_timesteps`` el modo temporal conserva todas las fechas."""
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1,), n_timesteps=10, collapse_time=None
    )
    # patch 10001 tiene T=8 < 10 -> conserva las 8.
    idx = ds.patch_ids.index("10001")
    x, _ = ds[idx]
    assert x.shape[0] == 8


def test_pick_mode_shape(synthetic_pastis: Path) -> None:
    """Modo ``pick`` colapsa a un frame central ``(10,H,W)``."""
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), collapse_time="pick")
    x, _ = ds[0]
    assert x.shape == (_N_BANDS, _H, _W)


# ---------------------------------------------------------------------------
# Rango de etiquetas
# ---------------------------------------------------------------------------


def test_semantic18_label_range(synthetic_pastis: Path) -> None:
    """Target ``semantic18`` vive en ``[0..17] u {ignore_index}``."""
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1, 2), target="semantic18", ignore_index=255
    )
    seen: set[int] = set()
    for i in range(len(ds)):
        _, y = ds[i]
        seen.update(int(v) for v in torch.unique(y).tolist())
    valid = {v for v in seen if v != 255}
    assert valid, "deberia haber al menos una clase valida"
    assert all(0 <= v <= 17 for v in valid)
    # El patch 20000 tiene Background(0) y Void(19) -> deben caer en ignore.
    assert 255 in seen
    # Clase 1 -> 0, clase 2 -> 1, clase 3 -> 2, clase 5 -> 4.
    assert {0, 1, 2, 4}.issubset(valid)


def test_hcat6_label_range(synthetic_pastis: Path) -> None:
    """Target ``hcat6`` vive en ``[0..5] u {ignore_index}`` y ``num_classes==6``."""
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1, 2), target="hcat6", ignore_index=255
    )
    assert ds.num_classes == 6
    seen: set[int] = set()
    for i in range(len(ds)):
        _, y = ds[i]
        seen.update(int(v) for v in torch.unique(y).tolist())
    valid = {v for v in seen if v != 255}
    assert valid
    assert all(0 <= v <= 5 for v in valid)
    assert 255 in seen  # Background/Void del patch 20000.


def test_ignore_index_custom_value(synthetic_pastis: Path) -> None:
    """Un ``ignore_index`` personalizado se respeta en la mascara de salida."""
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(2,), target="semantic18", ignore_index=99
    )
    # patch 20000 tiene Background y Void -> deben mapear a 99, no 255.
    _, y = ds[0]
    uniques = set(int(v) for v in torch.unique(y).tolist())
    assert 99 in uniques
    assert 255 not in uniques


# ---------------------------------------------------------------------------
# Determinismo del submuestreo temporal
# ---------------------------------------------------------------------------


def test_temporal_subsample_deterministic(synthetic_pastis: Path) -> None:
    """Dos instancias con la misma config dan EXACTAMENTE el mismo submuestreo.

    Cada frame t del fixture tiene reflectancia ``5000 + t*100``; si el
    submuestreo fuera estocastico las dos instancias diferirian. Comparamos los
    tensores completos byte-a-byte.
    """
    ds_a = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1,), n_timesteps=6, collapse_time=None, seed=42
    )
    ds_b = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1,), n_timesteps=6, collapse_time=None, seed=123
    )
    x_a, _ = ds_a[0]
    x_b, _ = ds_b[0]
    assert x_a.shape == x_b.shape
    assert torch.equal(x_a, x_b)


def test_temporal_subsample_includes_first_and_last(synthetic_pastis: Path) -> None:
    """El submuestreo equiespaciado incluye la primera y la ultima fecha.

    El frame 0 tiene base 5000/10000=0.5 y el frame T-1 (T=12) tiene base
    (5000+11*100)/10000=0.61. Ambos deben aparecer en el tensor submuestreado.
    """
    ds = PASTISSegmentationDataset(
        root=synthetic_pastis, folds=(1,), n_timesteps=4, collapse_time=None
    )
    x, _ = ds[0]  # patch 10000, T=12 -> 4 fechas
    first_base = x[0].mean().item()
    last_base = x[-1].mean().item()
    assert abs(first_base - 0.5) < 1e-4
    assert abs(last_base - (5000 + 11 * 100) / 10000.0) < 1e-4


# ---------------------------------------------------------------------------
# Normalizacion: fallback /10000 vs NORM stats
# ---------------------------------------------------------------------------


def test_fallback_scale_when_no_norm_stats(synthetic_pastis: Path) -> None:
    """Sin ``NORM_S2_patch.json`` se aplica la escala simple ``/10000``.

    El frame central del patch 10000 (modo median) vale 5000+t*100; la mediana
    de las 12 fechas (bases 5000..6100) es ~5550/10000 = 0.555.
    """
    assert not (synthetic_pastis / "NORM_S2_patch.json").exists()
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), collapse_time="median")
    assert not ds._norm_stats
    x, _ = ds[0]
    # Mediana temporal de bases [5000,5100,...,6100] = 5550 -> /10000 = 0.555.
    assert torch.all(x >= 0.0)
    assert torch.all(x <= 1.0)
    assert abs(x.mean().item() - 0.555) < 1e-3


def test_norm_stats_applied_when_present(tmp_path: Path) -> None:
    """Con ``NORM_S2_patch.json`` se estandariza ``(x/scale - mean)/std`` por banda."""
    root = tmp_path / "PASTIS-R"
    sem = np.full((_H, _W), 1, dtype=np.uint8)
    _write_synthetic_patch(root, "10000", n_timesteps=4, fold=1, semantic_fill=sem)
    _write_metadata(root, {"10000": 1})

    # mean en escala reflectancia (0..10000); std=10000 -> tras /scale std=1.
    norm = {
        "Fold_1": {
            "mean": [5000.0] * _N_BANDS,
            "std": [10000.0] * _N_BANDS,
        }
    }
    (root / "NORM_S2_patch.json").write_text(json.dumps(norm), encoding="utf-8")

    ds = PASTISSegmentationDataset(root=root, folds=(1,), collapse_time="median")
    assert ds._norm_stats
    x, _ = ds[0]
    # Bases temporales 5000..5300 -> mediana 5150; (5150-5000)/10000 = 0.015.
    assert abs(x.mean().item() - 0.015) < 1e-3


# ---------------------------------------------------------------------------
# Split por fold disjunto
# ---------------------------------------------------------------------------


def test_fold_split_disjoint(synthetic_pastis: Path) -> None:
    """Splits por fold distintos no comparten ningun patch_id."""
    train = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,))
    val = PASTISSegmentationDataset(root=synthetic_pastis, folds=(2,))
    test = PASTISSegmentationDataset(root=synthetic_pastis, folds=(4,))

    train_ids = set(train.patch_ids)
    val_ids = set(val.patch_ids)
    test_ids = set(test.patch_ids)

    assert train_ids == {"10000", "10001"}
    assert val_ids == {"20000"}
    assert test_ids == {"40000"}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_fold_union_matches_individual(synthetic_pastis: Path) -> None:
    """Un split multi-fold es la union de los splits individuales."""
    multi = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1, 2))
    assert set(multi.patch_ids) == {"10000", "10001", "20000"}
    assert len(multi) == 3


def test_unknown_fold_yields_empty(synthetic_pastis: Path) -> None:
    """Un fold sin patches en disco produce un dataset vacio (no error)."""
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(3,))
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# Validacion de argumentos
# ---------------------------------------------------------------------------


def test_invalid_collapse_time_raises(synthetic_pastis: Path) -> None:
    """``collapse_time`` invalido lanza ``ValueError``."""
    with pytest.raises(ValueError, match="collapse_time"):
        PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), collapse_time="mean")  # type: ignore[arg-type]


def test_invalid_target_raises(synthetic_pastis: Path) -> None:
    """``target`` invalido lanza ``ValueError``."""
    with pytest.raises(ValueError, match="target"):
        PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), target="semantic20")  # type: ignore[arg-type]


def test_nonpositive_timesteps_raises(synthetic_pastis: Path) -> None:
    """``n_timesteps <= 0`` lanza ``ValueError``."""
    with pytest.raises(ValueError, match="n_timesteps"):
        PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,), n_timesteps=0)


def test_missing_s2_dir_raises(tmp_path: Path) -> None:
    """Una raiz sin ``DATA_S2/`` lanza ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        PASTISSegmentationDataset(root=tmp_path / "nope", folds=(1,))


def test_index_out_of_range_raises(synthetic_pastis: Path) -> None:
    """``__getitem__`` fuera de rango lanza ``IndexError``."""
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,))
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_negative_index_supported(synthetic_pastis: Path) -> None:
    """Indice negativo indexa desde el final (consistencia Python)."""
    ds = PASTISSegmentationDataset(root=synthetic_pastis, folds=(1,))
    x_last, _ = ds[-1]
    x_pos, _ = ds[len(ds) - 1]
    assert torch.equal(x_last, x_pos)


# ---------------------------------------------------------------------------
# Smoke sobre el dataset real (lento, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _pastis_present, reason="PASTIS-R no descargado en data/PASTIS-R/DATA_S2/.")
def test_real_dataset_contract_2d() -> None:
    """Smoke end-to-end sobre PASTIS-R real (fold 1, 1 patch, modo 2D)."""
    ds = PASTISSegmentationDataset(root=_REAL_PASTIS_ROOT, folds=(1,), collapse_time="median")
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (10, 128, 128)
    assert y.shape == (128, 128)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    valid = y[y != ds.ignore_index]
    if valid.numel() > 0:
        assert int(valid.min()) >= 0
        assert int(valid.max()) <= 17


@pytest.mark.slow
@pytest.mark.skipif(not _pastis_present, reason="PASTIS-R no descargado en data/PASTIS-R/DATA_S2/.")
def test_real_dataset_contract_temporal() -> None:
    """Smoke end-to-end sobre PASTIS-R real (fold 1, modo temporal T=10)."""
    ds = PASTISSegmentationDataset(
        root=_REAL_PASTIS_ROOT, folds=(1,), n_timesteps=10, collapse_time=None
    )
    x, y = ds[0]
    assert x.ndim == 4
    assert x.shape[0] <= 10
    assert x.shape[1:] == (10, 128, 128)
    assert y.shape == (128, 128)
