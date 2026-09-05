"""Pruebas del verificador de re-volcados, que es el control que dejo pasar el defecto de US-118.

El guion comparaba el re-volcado con el fichero anterior y los dos salian del mismo bug: el
volcado reconstruia el modelo con ``n_timesteps=37`` y alimentaba al dataset con 10. Coincidian, y
la coincidencia se leyo como confirmacion. Un control sin pruebas es una intencion: aqui se le ve
fallar en cada cosa que dice comprobar, y en particular en esa.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.utils.parcel_reconcile import PROB_COLUMNS  # noqa: E402
from scripts import verificar_redump_oof as verificador  # noqa: E402

N_PARCELAS = 12


def _posteriores(n: int, semilla: int, *, suman_uno: bool = True) -> dict[str, list[float]]:
    """Posteriores sinteticas normalizadas por fila.

    Args:
        n: Numero de parcelas.
        semilla: Semilla del generador.
        suman_uno: Si ``False``, deja filas que no son distribuciones.

    Returns:
        Columnas ``prob_XXX``.
    """
    rng = np.random.default_rng(semilla)
    bruto = rng.random((n, len(PROB_COLUMNS)))
    matriz = bruto / bruto.sum(axis=1, keepdims=True) if suman_uno else bruto
    return {c: matriz[:, i].tolist() for i, c in enumerate(PROB_COLUMNS)}


def _tabla(ids: list[str], semilla: int, *, suman_uno: bool = True) -> pl.DataFrame:
    """Tabla OOF por parcela con la forma que el verificador espera."""
    return pl.DataFrame(
        {"canonical_parcel_id": ids, **_posteriores(len(ids), semilla, suman_uno=suman_uno)}
    )


@pytest.fixture
def escenario(tmp_path: Path) -> dict[str, Any]:
    """Un directorio temporal, uno de OOF y un miembro tabular que pasa todas las comprobaciones."""
    ids = [f"p{i:04d}" for i in range(N_PARCELAS)]
    temporal = tmp_path / "tmp"
    oof = tmp_path / "oof"
    temporal.mkdir()
    oof.mkdir()
    _tabla(ids, 0).write_parquet(temporal / "oof_parcel_xgb-remat_fold5.parquet")
    _tabla(ids, 1).write_parquet(oof / "oof_parcel_tsvit-pheno_fold5.parquet")
    _tabla(ids, 2).write_parquet(oof / "oof_parcel_xgb-remat_fold5.parquet")
    return {"ids": ids, "temporal": temporal, "oof": oof, "informe": tmp_path / "informe.json"}


def _correr(escenario: dict[str, Any], miembro: str, monkeypatch: pytest.MonkeyPatch) -> int:
    """Invocar el verificador con los argumentos del escenario."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verificar_redump_oof.py",
            "--temporal",
            str(escenario["temporal"]),
            "--miembro",
            miembro,
            "--oof",
            str(escenario["oof"]),
            "--informe",
            str(escenario["informe"]),
        ],
    )
    return verificador.main()


def test_un_revolcado_tabular_correcto_se_puede_promover(
    escenario: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """El camino feliz, que hace falta para que los fallos de abajo signifiquen algo."""
    assert _correr(escenario, "xgb-remat", monkeypatch) == 0
    informe = json.loads(escenario["informe"].read_text(encoding="utf-8"))
    assert informe["veredicto"] == "se_puede_promover"
    assert informe["tipo"] == "tabular"
    assert informe["fallos"] == []


def test_el_informe_sella_los_estratos_de_confianza(
    escenario: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """La comparacion no puede vivir solo en la prosa de un inventario.

    Es el defecto que el proyecto lleva ocho rondas persiguiendo: una verificacion cuyo resultado
    no se puede re-derivar no es una verificacion, es un recuerdo.
    """
    _correr(escenario, "xgb-remat", monkeypatch)
    comparacion = json.loads(escenario["informe"].read_text(encoding="utf-8"))[
        "comparacion_con_el_anterior"
    ]
    assert comparacion["n_parcelas_comparables"] == N_PARCELAS
    assert 0.0 <= comparacion["coincidencia_argmax"] <= 1.0
    assert comparacion["por_estrato_de_confianza"], comparacion


def test_unas_posteriores_que_no_suman_uno_no_se_promueven(
    escenario: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lo que no es una distribucion no puede entrar como si lo fuera."""
    _tabla(escenario["ids"], 7, suman_uno=False).write_parquet(
        escenario["temporal"] / "oof_parcel_xgb-remat_fold5.parquet"
    )
    assert _correr(escenario, "xgb-remat", monkeypatch) == 1
    assert "no suman uno" in capsys.readouterr().out


def test_parcelas_de_mas_no_se_promueven(
    escenario: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sobrar siempre es un fallo: son parcelas que el universo del banco no contiene."""
    _tabla([*escenario["ids"], "intrusa"], 3).write_parquet(
        escenario["temporal"] / "oof_parcel_xgb-remat_fold5.parquet"
    )
    assert _correr(escenario, "xgb-remat", monkeypatch) == 1
    assert "no estan en tsvit-pheno" in capsys.readouterr().out


def _preparar_denso(
    escenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    n_timesteps_dataset: int | None,
    n_timesteps_model_spec: int | None = None,
) -> None:
    """Montar un miembro denso con el manifiesto que se quiera probar."""
    from ml.eval import checkpoint_registry

    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"pesos de mentira")
    spec = replace(checkpoint_registry.CHECKPOINT_REGISTRY["tsvit-pheno-fullm"], path=pesos)
    monkeypatch.setitem(checkpoint_registry.CHECKPOINT_REGISTRY, "denso-de-prueba", spec)

    entrada: dict[str, Any] = {"status": "ok", "n_patches": verificador.PATCHES_ESPERADOS}
    if n_timesteps_dataset is not None:
        entrada["n_timesteps_dataset"] = n_timesteps_dataset
    entrada["n_timesteps_model_spec"] = (
        n_timesteps_model_spec
        if n_timesteps_model_spec is not None
        else int(spec.model_kwargs["n_timesteps"])
    )
    (escenario["temporal"] / "manifest.json").write_text(
        json.dumps({"models": {"denso-de-prueba": entrada}}), encoding="utf-8"
    )
    _tabla(escenario["ids"], 4).write_parquet(
        escenario["temporal"] / "oof_parcel_denso-de-prueba_fold5.parquet"
    )


def test_un_volcado_denso_sin_el_n_timesteps_del_dataset_no_se_promueve(
    escenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sin ese campo, un volcado con la T equivocada es indistinguible de uno correcto.

    Es exactamente la situacion en la que este guion dio por bueno un fichero producido por un bug.
    """
    _preparar_denso(escenario, monkeypatch, tmp_path, n_timesteps_dataset=None)
    assert _correr(escenario, "denso-de-prueba", monkeypatch) == 1
    assert "no registra el n_timesteps del dataset" in capsys.readouterr().out


def test_un_volcado_denso_con_la_t_equivocada_no_se_promueve(
    escenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El defecto real: modelo reconstruido con T=37 y dataset alimentado con 10."""
    _preparar_denso(escenario, monkeypatch, tmp_path, n_timesteps_dataset=10)
    assert _correr(escenario, "denso-de-prueba", monkeypatch) == 1
    salida = capsys.readouterr().out
    assert "el dataset uso n_timesteps=10" in salida
    assert "37" in salida


def test_un_volcado_denso_acoplado_pasa_esa_comprobacion(
    escenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con la T correcta, la comprobacion del acoplamiento no protesta."""
    _preparar_denso(escenario, monkeypatch, tmp_path, n_timesteps_dataset=37)
    _correr(escenario, "denso-de-prueba", monkeypatch)
    salida = capsys.readouterr().out
    assert "n_timesteps: model_spec=37  dataset=37" in salida
    assert "el dataset uso n_timesteps" not in salida


def test_un_manifiesto_que_declara_otra_configuracion_no_se_promueve(
    escenario: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El manifiesto tiene que declarar la MISMA capacidad que el registro activo."""
    _preparar_denso(
        escenario, monkeypatch, tmp_path, n_timesteps_dataset=37, n_timesteps_model_spec=10
    )
    assert _correr(escenario, "denso-de-prueba", monkeypatch) == 1
    assert "n_timesteps_model_spec=10" in capsys.readouterr().out
