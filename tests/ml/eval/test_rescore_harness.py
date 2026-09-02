"""Harness-core tests of the fold-5 re-score (US-030, agente ml/A).

Complementan, sin duplicar, los tests de consolidacion/registry de
``tests/ml/eval/test_rescore.py`` (ml/C) y los golden-value de
``tests/ml/eval/test_class_remap.py`` (ml/B). Aqui se ejercita la **logica
interna del harness** (``ml.eval.dense_metrics`` + ``ml.eval.segmentation_inference``)
con dataset y modelos MOCKEADOS: nunca se descarga un checkpoint, nunca se contacta
``torch.hub`` (encoder AnySat) ni el server MLflow, y NUNCA se corre inferencia
real sobre PASTIS.

Cobertura especifica (gaps no cubiertos por ml/B ni ml/C):

- ``test_rescore_fold_is_5``: el dataset se instancia con ``folds=(5,)`` (AC-1) —
  un dataset fake registra el ``folds`` recibido.
- ``test_rescore_norm_uses_train_only``: las stats de normalizacion se promedian
  con ``folds=(1, 2, 3)``, nunca incluyendo el fold-5 held-out (AC-6 / R4).
- ``build_model_for_kind`` para U-TAE (construye con ``num_classes=20`` y las keys
  ``out_conv`` quedan intactas, R1) y AnySat (forward dummy materializa la
  ``LazyConv2d`` ANTES de cargar pesos, R7) con ``torch.hub`` mockeado.

Patron: ``tests/ml/eval/test_dense_metrics.py`` + ``test_comparison.py``
(monkeypatch + arrays deterministas seed fija). Workaround R7 heredado:
``coverage run --include=...`` en vez de ``pytest --cov`` (incompat
numpy 2.3.5 / scipy 1.17.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

import ml.eval.dense_metrics as dense_metrics
import ml.eval.segmentation_inference as seg_inf
from ml.eval.checkpoint_registry import CheckpointSpec
from ml.eval.class_remap import HARNESS_IGNORE_INDEX

# ===========================================================================
# Mocks deterministas del dataset y del modelo (sin checkpoints / sin PASTIS).
# ===========================================================================


class _FakeSegDataset:
    """Dataset fake que registra el ``folds`` recibido y devuelve pares (x, y).

    Reproduce el contrato minimo que ``_rescore_one`` consume de
    :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`: ``folds``,
    ``__len__``, ``__getitem__`` (devuelve ``(x, y)``), ``patch_ids``, ``root``,
    y los atributos internos de normalizacion (``_norm_stats`` / ``_fold_of``)
    que muta :func:`ml.eval.dense_metrics._apply_train_norm`.

    El target ``y`` ya esta en el espacio contiguo 18-clase (como
    ``target="semantic18"``); ``x`` es un patch dummy de la forma temporal o 2D
    segun ``collapse_time``.
    """

    #: Captura el ultimo conjunto de kwargs con que se instancio (introspeccion).
    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.folds = tuple(kwargs.get("folds", ()))  # type: ignore[arg-type]
        self.collapse_time = kwargs.get("collapse_time", "median")
        self.target = kwargs.get("target", "semantic18")
        self.ignore_index = int(kwargs.get("ignore_index", HARNESS_IGNORE_INDEX))  # type: ignore[arg-type]
        self.root = Path(str(kwargs.get("root", "data/PASTIS-R")))
        self.patch_ids = ["10000", "10001"]
        # Per-fold normalization stats covering ALL folds 1..5; the harness must
        # average ONLY the train folds (1, 2, 3) and never touch the fold-5 stats.
        self._norm_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {
            f: (
                np.full(10, float(f), dtype=np.float64),
                np.full(10, float(f), dtype=np.float64),
            )
            for f in (1, 2, 3, 4, 5)
        }
        self._fold_of: dict[str, int] = {pid: 5 for pid in self.patch_ids}
        rng = np.random.default_rng(0)
        n = len(self.patch_ids)
        # Targets in the contiguous 18-class space with a few ignore pixels.
        self._targets = [
            torch.from_numpy(rng.integers(0, 18, size=(8, 8)).astype(np.int64)) for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 2D patch (10, H, W); the harness only forwards it to the (mocked)
        # predict, which ignores the content and returns a deterministic map.
        x = torch.zeros(10, 8, 8, dtype=torch.float32)
        return x, self._targets[idx]


class _DummyModel(nn.Module):
    """Modelo fake con un parametro (para que ``next(model.parameters())`` exista)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))


def _install_harness_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    native_num_classes: int,
) -> type[_FakeSegDataset]:
    """Wire dataset/model/predict mocks into the harness import sites.

    ``_rescore_one`` importa ``PASTISSegmentationDataset`` desde
    ``ml.data.pastis_seg_dataset`` y ``load_checkpoint_model`` /
    ``predict_patch_for_kind`` desde ``ml.eval.segmentation_inference`` DENTRO de
    la funcion; por eso se parchea en el modulo origen de cada simbolo.

    Args:
        monkeypatch: Fixture de pytest.
        native_num_classes: Clases nativas del checkpoint simulado; decide si la
            prediccion fake esta en el espacio 20 (se remapea) o 18 (sin remap).

    Returns:
        La clase ``_FakeSegDataset`` (para introspeccionar ``last_kwargs``).
    """
    import ml.data.pastis_seg_dataset as ds_mod

    _FakeSegDataset.last_kwargs = {}
    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeSegDataset)

    monkeypatch.setattr(
        seg_inf,
        "load_checkpoint_model",
        lambda spec, **_kw: _DummyModel(),
    )

    def _fake_predict(model: nn.Module, x: torch.Tensor, *, model_kind: str) -> np.ndarray:
        # Deterministic native-space prediction (8x8). For 20-class models the
        # values live in [1..18] so the 20->18 remap produces valid agronomic
        # classes; for 18-class models they live in [0..17].
        rng = np.random.default_rng(1)
        if native_num_classes >= 20:
            return rng.integers(1, 19, size=(8, 8)).astype(np.int64)
        return rng.integers(0, 18, size=(8, 8)).astype(np.int64)

    monkeypatch.setattr(seg_inf, "predict_patch_for_kind", _fake_predict)
    return _FakeSegDataset


def _spec(model_kind: str, native_num_classes: int) -> CheckpointSpec:
    """Build a CheckpointSpec pointing at an existing path (so it is not 'missing').

    The path itself is never read: ``load_checkpoint_model`` is mocked. We point
    it at this test file (guaranteed to exist) so ``spec.path.exists()`` is True.
    """
    return CheckpointSpec(
        name=model_kind,
        model_kind=model_kind,  # type: ignore[arg-type]
        path=Path(__file__).resolve(),
        native_num_classes=native_num_classes,
        native_ignore_index=19 if native_num_classes >= 20 else 255,
    )


# ===========================================================================
# Grupo A — fold-5 held-out (AC-1) y anti-fuga de normalizacion (AC-6 / R4).
# ===========================================================================


def test_rescore_fold_is_5(monkeypatch: pytest.MonkeyPatch) -> None:
    """El harness instancia el dataset con ``folds=(5,)`` (held-out), no fold-4.

    Mockea el dataset (registra ``last_kwargs``), el loader y la prediccion: la
    unica asercion es que el split de scoring se pide con ``folds=(5,)``.
    """
    fake_ds = _install_harness_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)

    df = dense_metrics.rescore_all_checkpoints(
        {"deeplabv3plus": spec},
        fold=5,
        device="cpu",
        skip_missing=False,
    )

    assert fake_ds.last_kwargs.get("folds") == (5,)
    assert fake_ds.last_kwargs.get("target") == "semantic18"
    assert fake_ds.last_kwargs.get("ignore_index") == HARNESS_IGNORE_INDEX
    # Una fila ok con la metrica calculada sobre el patch fake.
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["status"] == "ok"
    assert row["fold"] == 5
    assert row["n_patches"] == 2


def test_rescore_fold_param_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """El parametro ``fold`` llega al dataset (no esta hardcodeado a 5)."""
    fake_ds = _install_harness_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("tsvit-pheno", 18)

    dense_metrics.rescore_all_checkpoints(
        {"tsvit-pheno": spec},
        fold=4,
        device="cpu",
        skip_missing=False,
    )
    assert fake_ds.last_kwargs.get("folds") == (4,)


def test_rescore_norm_uses_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Las stats de normalizacion se promedian SOLO con folds (1, 2, 3) (R4/AC-6).

    ``_apply_train_norm`` reescribe ``dataset._norm_stats`` para que TODO patch
    use el promedio de los folds de entrenamiento. Con stats por fold = (f, f),
    el promedio de (1, 2, 3) es exactamente 2.0 en cada banda; el fold-5 (=5.0)
    no debe filtrarse.
    """
    captured: dict[str, _FakeSegDataset] = {}

    fake_ds_cls = _install_harness_mocks(monkeypatch, native_num_classes=18)

    real_apply = dense_metrics._apply_train_norm

    def _spy_apply(dataset: object) -> None:
        captured["ds"] = dataset  # type: ignore[assignment]
        real_apply(dataset)

    monkeypatch.setattr(dense_metrics, "_apply_train_norm", _spy_apply)

    spec = _spec("unet", 20)
    dense_metrics.rescore_all_checkpoints({"unet": spec}, fold=5, device="cpu", skip_missing=False)

    ds = captured["ds"]
    assert isinstance(ds, fake_ds_cls)
    # Every fold present now maps to the SAME averaged train stats (mean of 1,2,3).
    overwritten = list(ds._norm_stats.values())  # type: ignore[attr-defined]
    assert overwritten, "norm stats were not overwritten"
    first_mean, first_std = overwritten[0]
    np.testing.assert_allclose(first_mean, np.full(10, 2.0))
    np.testing.assert_allclose(first_std, np.full(10, 2.0))
    for m, s in overwritten:
        # All folds share the SAME averaged train stats (mean of 1,2,3 == 2.0).
        np.testing.assert_allclose(m, np.full(10, 2.0))
        np.testing.assert_allclose(s, np.full(10, 2.0))
        # The held-out fold-5 stats (5.0) must NOT survive anywhere.
        assert not np.allclose(m, np.full(10, 5.0))
        assert not np.allclose(s, np.full(10, 5.0))


def test_train_norm_stats_default_folds_excludes_5() -> None:
    """``_train_norm_stats`` por defecto promedia (1, 2, 3) y nunca el fold-5."""
    assert dense_metrics._TRAIN_NORM_FOLDS == (1, 2, 3)
    raw = {f: (np.full(10, float(f)), np.full(10, float(f))) for f in (1, 2, 3, 4, 5)}
    mean, std = dense_metrics._train_norm_stats(raw)  # type: ignore[misc]
    # (1 + 2 + 3) / 3 == 2.0; fold-4 and fold-5 are excluded.
    np.testing.assert_allclose(mean, np.full(10, 2.0))
    np.testing.assert_allclose(std, np.full(10, 2.0))


def test_train_norm_stats_returns_none_without_train_folds() -> None:
    """Sin ninguno de los folds (1, 2, 3) presentes -> None (fallback /10000)."""
    raw = {5: (np.full(10, 5.0), np.full(10, 5.0))}
    assert dense_metrics._train_norm_stats(raw) is None  # type: ignore[misc]


# ===========================================================================
# Grupo B — remap 20->18 vivo en el harness (predicciones de 20-clase).
# ===========================================================================


def test_rescore_20class_model_remaps_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un modelo de 20 clases produce una fila ``ok`` con metrica en [0, 1].

    Verifica de extremo a extremo (mockeado) que la rama de remap 20->18 del
    harness se ejerce sin romper la acumulacion en el espacio 18-clase.
    """
    _install_harness_mocks(monkeypatch, native_num_classes=20)
    spec = _spec("utae", 20)
    df = dense_metrics.rescore_all_checkpoints(
        {"utae": spec}, fold=5, device="cpu", skip_missing=False
    )
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["status"] == "ok"
    assert row["model_kind"] == "utae"
    for metric in ("miou", "f1_macro", "pixel_accuracy"):
        assert 0.0 <= float(row[metric]) <= 1.0
    # per_class_iou is a list of 18 (per-class derivatives populated).
    assert isinstance(row["per_class_iou"], list)
    assert len(row["per_class_iou"]) == 18


def test_rescore_returns_polars_with_contract_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El DataFrame devuelto trae las columnas del contrato US-030."""
    _install_harness_mocks(monkeypatch, native_num_classes=18)
    spec = _spec("deeplabv3plus", 18)
    df = dense_metrics.rescore_all_checkpoints(
        {"deeplabv3plus": spec}, fold=5, device="cpu", skip_missing=False
    )
    assert isinstance(df, pl.DataFrame)
    for col in (
        "model",
        "model_kind",
        "miou",
        "f1_macro",
        "pixel_accuracy",
        "fold",
        "n_patches",
        "status",
        "per_class_iou",
    ):
        assert col in df.columns, col


# ===========================================================================
# Grupo C — build_model_for_kind: U-TAE (keys intactas) y AnySat (lazy head).
# ===========================================================================


def test_build_model_for_kind_utae_20_classes_keys_intact() -> None:
    """U-TAE se construye con ``num_classes=20`` y las keys del head quedan intactas.

    R1: PROHIBIDO renombrar ``out_conv``; el head debe tener 20 canales de salida
    para que el checkpoint cargue, y las claves del ``state_dict`` deben incluir
    ``out_conv`` (la convencion del checkpoint 04j).
    """
    spec = _spec("utae", 20)
    model = seg_inf.build_model_for_kind(spec, n_timesteps=4, device="cpu")

    keys = list(model.state_dict().keys())
    assert any(k.startswith("out_conv") for k in keys), keys[:8]
    # The final out_conv layer must output exactly the native 20 classes.
    out_weights = [v for k, v in model.state_dict().items() if k.startswith("out_conv")]
    final = out_weights[-1]
    assert final.shape[0] == 20, final.shape
    # The canonical encoder/decoder keys are present (load_state_dict won't break).
    assert any(k.startswith("in_conv") for k in keys)
    assert any(k.startswith("temporal_encoder") for k in keys)


def test_build_model_for_kind_segformer_rejected() -> None:
    """SegFormer NO se construye aqui (se carga via from_pretrained)."""
    spec = _spec("segformer", 20)
    with pytest.raises(ValueError, match="HuggingFace"):
        seg_inf.build_model_for_kind(spec, device="cpu")


def test_build_model_for_kind_unknown_kind_raises() -> None:
    """Un ``model_kind`` no soportado dispara ValueError."""
    spec = CheckpointSpec(
        name="bogus",
        model_kind="bogus",  # type: ignore[arg-type]
        path=Path(__file__).resolve(),
        native_num_classes=18,
        native_ignore_index=255,
    )
    with pytest.raises(ValueError, match="Unsupported model_kind"):
        seg_inf.build_model_for_kind(spec, device="cpu")


class _SyntheticAnySatEncoder(nn.Module):
    """Encoder sintetico que sustituye al de ``torch.hub`` (sin red).

    Acepta la imagen ``(B, T, C, H, W)`` y devuelve un mapa denso
    ``(B, D, h, w)``; ``AnySatSegmenter._encode`` cae a esta firma simple cuando
    la llamada con ``patch_size=...`` lanza ``TypeError``.
    """

    feature_dim: int = 6

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        h = w = 8
        return torch.zeros(b, self.feature_dim, h, w)


def test_build_model_for_kind_anysat_lazy_head_materialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnySat: el forward dummy materializa la LazyConv2d ANTES de cargar pesos (R7).

    Mockea ``torch.hub`` (encoder via ``load_anysat_encoder``) con un encoder
    sintetico. Tras ``build_model_for_kind`` el head debe ser un ``Conv2d`` real
    (lazy ya materializado) con 20 canales de salida y el feature_dim del encoder.
    """
    import ml.models.anysat_wrapper as anysat_mod

    monkeypatch.setattr(
        anysat_mod, "load_anysat_encoder", lambda *a, **k: _SyntheticAnySatEncoder()
    )

    spec = _spec("anysat", 20)
    model = seg_inf.build_model_for_kind(spec, n_timesteps=4, device="cpu")

    head = model.head  # type: ignore[attr-defined]
    # After the dummy forward the lazy head is a concrete Conv2d (not LazyConv2d).
    assert isinstance(head, nn.Conv2d)
    assert not isinstance(head, nn.LazyConv2d)
    assert head.out_channels == 20
    assert head.in_channels == _SyntheticAnySatEncoder.feature_dim
    # A subsequent forward succeeds (state_dict could now bind to a real head).
    dummy = torch.zeros(1, 4, 10, 16, 16)
    with torch.no_grad():
        logits = model(dummy, torch.zeros(1, 4, dtype=torch.long))
    assert logits.shape[1] == 20


def test_build_model_for_kind_anysat_forward_before_load_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tras materializar, ``load_state_dict`` del propio modelo no rompe (R7).

    Captura el contrato del loader: el dummy forward debe ocurrir ANTES de
    ``load_state_dict``. Aqui re-cargamos el state_dict del propio modelo (round
    trip) para confirmar que las keys del head ya existen y bindean.
    """
    import ml.models.anysat_wrapper as anysat_mod

    monkeypatch.setattr(
        anysat_mod, "load_anysat_encoder", lambda *a, **k: _SyntheticAnySatEncoder()
    )

    spec = _spec("anysat", 20)
    model = seg_inf.build_model_for_kind(spec, n_timesteps=4, device="cpu")

    state = model.state_dict()
    assert any(k.startswith("head") for k in state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The head keys are present, so no head key is missing after materialization.
    assert not [k for k in missing if k.startswith("head")]
    assert not [k for k in unexpected if k.startswith("head")]
