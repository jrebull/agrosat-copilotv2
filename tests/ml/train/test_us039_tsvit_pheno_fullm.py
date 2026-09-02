"""Tests del orquestador TSViT-pheno Full-M (US-039).

Cubren el **contrato del orquestador** (config full == US-038, lambda=0.3,
prototipos no-regenerados, folds oficiales, tabla delta honesta, rechazo de root
sintetico) con el trainer y el harness de re-score MOCKEADOS: cero GPU, cero
PASTIS real, cero Gemini. La logica del loop y de la loss contrastiva ya esta
cubierta por ``test_pheno_semantic_branch.py`` y ``test_segmentation_models.py``
(US-025) -- NO se re-testea aqui.

Workaround R7 (heredado US-030): ``coverage run --include=...`` en vez de
``pytest --cov`` por la incompatibilidad numpy 2.3.5 / scipy 1.17.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ml.models.tsvit_wrapper import TSVIT_FULLM_CONFIG
from scripts.run_us039_tsvit_pheno_fullm import (
    CFG_FULL_TSVIT,
    build_pheno_vs_base_table,
    run_tsvit_pheno_full,
)

# ---------------------------------------------------------------------------
# Dobles de prueba: trainer y harness de re-score (cero GPU/PASTIS/Gemini).
# ---------------------------------------------------------------------------


class _TrainSpy:
    """Captura los argumentos con que el orquestador invoca ``build_and_train``."""

    def __init__(self, ret: dict[str, float] | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._ret = ret or {"miou": 0.66, "f1_macro": 0.75, "pixel_acc": 0.88}

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.calls.append((args, kwargs))
        return dict(self._ret)

    @property
    def last_kwargs(self) -> dict[str, Any]:
        return self.calls[-1][1]

    @property
    def last_args(self) -> tuple[Any, ...]:
        return self.calls[-1][0]


def _fake_rescore(*, base_miou: float = 0.659, pheno_miou: float = 0.661) -> Any:
    """Factory of a stub ``rescore_all_checkpoints`` (returns a Polars frame)."""

    def _stub(*, fold: int, device: str, skip_missing: bool) -> pl.DataFrame:
        _stub.last_fold = fold  # type: ignore[attr-defined]
        return pl.DataFrame(
            [
                {
                    "model": "tsvit",
                    "miou": base_miou,
                    "f1_macro": 0.748,
                    "fold": fold,
                    "status": "ok",
                },
                {
                    "model": "tsvit-pheno-fullm",
                    "miou": pheno_miou,
                    "f1_macro": 0.749,
                    "fold": fold,
                    "status": "ok",
                },
            ]
        )

    return _stub


# ---------------------------------------------------------------------------
# Contrato del orquestador.
# ---------------------------------------------------------------------------


def test_invokes_pheno_with_lambda_03() -> None:
    """``build_and_train`` se invoca con ``tsvit-pheno`` y ``lambda_contrast==0.3``."""
    spy = _TrainSpy()
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    assert spy.last_args[0] == "tsvit-pheno"
    assert spy.last_kwargs["lambda_contrast"] == 0.3


def test_uses_full_config_from_us038() -> None:
    """La capacidad pasada == ``TSVIT_FULLM_CONFIG`` (un solo lugar de verdad, R4)."""
    spy = _TrainSpy()
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    kw = spy.last_kwargs
    # Capacidad Full-M byte-identica a la base US-038 (apples-to-apples).
    # n_timesteps == PASTIS T_MIN (37): el valor con que se entrenaron los
    # checkpoints (PE ordinal (1, 37, dim)); single source TSVIT_FULLM_CONFIG.
    assert kw["n_timesteps"] == TSVIT_FULLM_CONFIG["n_timesteps"] == 37
    assert kw["dim"] == TSVIT_FULLM_CONFIG["dim"] == 192
    assert kw["depth_temporal"] == TSVIT_FULLM_CONFIG["depth_temporal"] == 6
    assert kw["depth_spatial"] == TSVIT_FULLM_CONFIG["depth_spatial"] == 6
    assert kw["heads"] == TSVIT_FULLM_CONFIG["heads"] == 6
    assert kw["dim_head"] == TSVIT_FULLM_CONFIG["dim_head"] == 64


def test_cfg_full_capacity_tracks_us038_single_source() -> None:
    """``CFG_FULL_TSVIT`` deriva de ``TSVIT_FULLM_CONFIG`` (no se redefine, R4)."""
    for key in ("n_timesteps", "dim", "depth_temporal", "depth_spatial", "heads", "dim_head"):
        assert CFG_FULL_TSVIT[key] == TSVIT_FULLM_CONFIG[key]


def test_folds_train_val_official() -> None:
    """Folds train=(1,2,3), val=(4); fold-5 NO entra en train/val (held-out, R7)."""
    spy = _TrainSpy()
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    kw = spy.last_kwargs
    assert kw["train_folds"] == (1, 2, 3)
    assert kw["val_folds"] == (4,)
    assert 5 not in kw["train_folds"]
    assert 5 not in kw["val_folds"]


def test_run_name_is_fullm() -> None:
    """El run MLflow es ``alt-tsvit-pheno-fullm-v1`` (AC-4)."""
    spy = _TrainSpy()
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    assert spy.last_kwargs["mlflow_run_name"] == "alt-tsvit-pheno-fullm-v1"


def test_default_ckpt_dir_is_fullm_slug() -> None:
    """El checkpoint default vive en ``tsvit-pheno-fullm-v1`` (NO pisa el L4, R10)."""
    spy = _TrainSpy()
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    ckpt = Path(spy.last_kwargs["ckpt_dir"])
    assert ckpt.name == "tsvit-pheno-fullm-v1"


def test_prototypes_not_regenerated_no_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """El orquestador NO regenera el parquet US-033 ni llama a Gemini (AC-3, R5).

    El orquestador delega la carga de prototipos a ``build_and_train`` (que usa
    ``PhenoSemanticBranch`` -> ``load_class_prototype_embeddings``, solo LEE). Aqui
    verificamos que el orquestador NUNCA toca el generador ni el cliente Gemini:
    si lo hiciera, estos stubs explotarian.
    """
    import ml.features.phenology_class_prototypes as proto_mod

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("US-039 NO debe regenerar prototipos (AC-3).")

    monkeypatch.setattr(proto_mod, "generate_class_prototypes", _boom, raising=False)

    spy = _TrainSpy()
    # No explota -> el orquestador no toco el generador.
    run_tsvit_pheno_full(train_fn=spy, rescore_fold5=False, mlflow_uri=None)
    assert spy.calls  # se entreno via el spy, sin pasar por Gemini


def test_rejects_synthetic_root() -> None:
    """Un root tipo ``data/farslip_pairs`` -> ``ValueError`` (anti-sintetico, R6)."""
    spy = _TrainSpy()
    with pytest.raises(ValueError, match="farslip_pairs"):
        run_tsvit_pheno_full(
            train_fn=spy,
            data_root=Path("data/farslip_pairs"),
            rescore_fold5=False,
            mlflow_uri=None,
        )
    assert not spy.calls  # falla ANTES de entrenar


def test_rejects_cfg_smuggling_lambda() -> None:
    """``cfg_full`` no puede colar ``lambda_contrast`` (contrato single-source)."""
    spy = _TrainSpy()
    bad_cfg = dict(CFG_FULL_TSVIT)
    bad_cfg["lambda_contrast"] = 0.9
    with pytest.raises(ValueError, match="lambda_contrast"):
        run_tsvit_pheno_full(train_fn=spy, cfg_full=bad_cfg, rescore_fold5=False, mlflow_uri=None)


# ---------------------------------------------------------------------------
# Tabla delta honesta + re-score fold-5.
# ---------------------------------------------------------------------------


def test_delta_table_schema(tmp_path: Path) -> None:
    """Tabla delta: filas base/pheno, columnas contrato, delta = pheno - base."""
    out = tmp_path / "tsvit_pheno_vs_base_fold5.csv"
    table = build_pheno_vs_base_table(
        base_miou=0.659,
        base_f1_macro=0.748,
        pheno_miou=0.661,
        pheno_f1_macro=0.749,
        out_path=out,
    )
    assert table.height == 2
    for col in ("model", "fold", "miou", "f1_macro", "delta_miou", "delta_f1_macro", "note"):
        assert col in table.columns

    models = table.get_column("model").to_list()
    assert "tsvit-fullm-base" in models
    assert "tsvit-pheno-fullm" in models

    pheno = table.filter(pl.col("model") == "tsvit-pheno-fullm")
    delta_miou = pheno.get_column("delta_miou").to_list()[0]
    delta_f1 = pheno.get_column("delta_f1_macro").to_list()[0]
    assert delta_miou == pytest.approx(0.661 - 0.659, abs=1e-9)
    assert delta_f1 == pytest.approx(0.749 - 0.748, abs=1e-9)

    # Nota de honestidad presente (R1).
    note = pheno.get_column("note").to_list()[0]
    assert "NO 5%" in note
    assert "FarSLIP" in note

    assert out.exists()
    reloaded = pl.read_csv(out)
    assert reloaded.height == 2


def test_delta_table_accepts_negative_delta(tmp_path: Path) -> None:
    """Delta < 0 es resultado VALIDO y se reporta tal cual (R2, saturacion)."""
    table = build_pheno_vs_base_table(
        base_miou=0.665,
        base_f1_macro=0.752,
        pheno_miou=0.661,
        pheno_f1_macro=0.749,
        out_path=tmp_path / "neg.csv",
    )
    pheno = table.filter(pl.col("model") == "tsvit-pheno-fullm")
    assert pheno.get_column("delta_miou").to_list()[0] < 0.0


def test_rescore_uses_fold5(tmp_path: Path) -> None:
    """El re-score invoca el harness US-030 con ``fold=5`` (R7) y construye la tabla."""
    spy = _TrainSpy()
    stub = _fake_rescore(base_miou=0.659, pheno_miou=0.6605)
    out = tmp_path / "delta.csv"
    result = run_tsvit_pheno_full(
        train_fn=spy,
        rescore_fn=stub,
        rescore_fold5=True,
        delta_csv=out,
        mlflow_uri=None,
    )
    assert stub.last_fold == 5  # type: ignore[attr-defined]
    assert out.exists()
    # El delta del resultado coincide con (pheno - base) del harness mock.
    assert result["fold5_base_miou"] == pytest.approx(0.659, abs=1e-9)
    assert result["fold5_pheno_miou"] == pytest.approx(0.6605, abs=1e-9)
    assert result["delta_miou"] == pytest.approx(0.6605 - 0.659, abs=1e-9)


def test_rescore_extracts_base_and_pheno_rows(tmp_path: Path) -> None:
    """El re-score lee la fila base (``tsvit``) y la pheno (``tsvit-pheno-fullm``)."""
    spy = _TrainSpy()
    stub = _fake_rescore(base_miou=0.640, pheno_miou=0.642)
    table_path = tmp_path / "delta.csv"
    run_tsvit_pheno_full(
        train_fn=spy,
        rescore_fn=stub,
        rescore_fold5=True,
        delta_csv=table_path,
        mlflow_uri=None,
    )
    table = pl.read_csv(table_path)
    base = table.filter(pl.col("model") == "tsvit-fullm-base")
    pheno = table.filter(pl.col("model") == "tsvit-pheno-fullm")
    assert base.get_column("miou").to_list()[0] == pytest.approx(0.640, abs=1e-9)
    assert pheno.get_column("miou").to_list()[0] == pytest.approx(0.642, abs=1e-9)
