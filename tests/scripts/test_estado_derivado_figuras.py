"""Una figura no puede estar mas viva que los datos de los que sale.

El ledger marcaba ``leyendas.svg`` y ``cobertura.svg`` como SELLADO mientras sus insumos de fase
3 y fase 4 estaban OBSOLETO, y ``replica.svg`` -exactamente el mismo caso- si estaba marcada. El
estado se habia puesto artefacto por artefacto, a mano, y por eso salio distinto para casos
iguales.

SELLADO significa "los bytes son los que dice el ledger", no "se puede citar". Pero nadie lee esa
distincion en una tabla de noventa filas: lo que se lee es que la figura esta bien. Aqui el estado
derivado se comprueba, en vez de recordarse.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "paper" / "ARTIFACTS.md"

#: Guiones que declaran de que datos sale cada figura que producen.
PRODUCTORES: tuple[str, ...] = (
    "scripts.build_paper_micai_extra_figures",
    "scripts.build_paper_micai_fase3_figure",
    "scripts.build_paper_micai_fase4_figure",
)


def _estados() -> dict[str, str]:
    """Estado de custodia de cada ruta del ledger.

    Returns:
        Ruta relativa -> estado.
    """
    estados: dict[str, str] = {}
    minimo_celdas = 6
    for linea in LEDGER.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|"):
            continue
        celdas = linea.strip().strip("|").split("|")
        if len(celdas) < minimo_celdas:
            continue
        ruta = re.search(r"`([^`]+)`", celdas[1])
        if ruta is not None:
            estados[ruta.group(1)] = celdas[5].strip()
    return estados


def _mapas() -> dict[str, tuple[str, ...]]:
    """Mapa salida -> insumos de todos los productores declarados."""
    mapa: dict[str, tuple[str, ...]] = {}
    for modulo in PRODUCTORES:
        mapa.update(importlib.import_module(modulo).INSUMOS_POR_FIGURA)
    return mapa


def test_los_productores_declaran_de_que_datos_sale_cada_figura() -> None:
    """Sin la declaracion no hay nada que contrastar, asi que se exige."""
    mapa = _mapas()
    assert len(mapa) >= len(PRODUCTORES), mapa
    for salida, insumos in mapa.items():
        assert salida.endswith(".svg"), salida
        assert insumos, f"{salida} no declara insumos"


@pytest.mark.parametrize("salida", sorted(_mapas()))
def test_una_figura_no_puede_estar_mas_viva_que_sus_insumos(salida: str) -> None:
    """Si un insumo esta OBSOLETO, la figura tambien lo esta."""
    estados = _estados()
    insumos = _mapas()[salida]
    obsoletos = [i for i in insumos if estados.get(i) == "OBSOLETO"]
    if not obsoletos:
        return
    assert estados.get(salida) == "OBSOLETO", (
        f"{salida} figura como {estados.get(salida)!r} en el ledger y sale de datos OBSOLETO: "
        f"{', '.join(obsoletos)}"
    )


def test_toda_figura_declarada_tiene_fila_en_el_ledger() -> None:
    """Una figura sin fila no tiene estado que comprobar, y eso es el agujero anterior."""
    estados = _estados()
    sin_fila = [s for s in _mapas() if s not in estados]
    assert not sin_fila, f"figuras sin fila en el ledger: {sin_fila}"
