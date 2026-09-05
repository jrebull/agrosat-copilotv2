"""Pruebas del gate de doble ciego, que hasta ahora no tenia ninguna.

Trae su propia autoprueba -``--autoprueba`` inyecta cada token y exige que el detector lo
encuentre-, pero nadie la corria: un control que no se ejecuta no esta verde, esta sin mirar. Aqui
se ejecuta, y ademas se comprueba la mitad del control que se saltaba en silencio.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import paper_micai_anon_check as gate  # noqa: E402


def test_la_autoprueba_del_gate_pasa() -> None:
    """Cada token se inyecta en un texto inocuo y el detector tiene que encontrarlo."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "paper_micai_anon_check.py"), "--autoprueba"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_los_apellidos_de_los_dos_autores_estan_entre_los_tokens() -> None:
    """La lista no puede envejecer respecto de quien firma.

    Autores fijados: Arthur Zizumbo primero, Javier A. Rebull-Saucedo segundo.
    """
    for apellido in ("zizumbo", "rebull", "saucedo"):
        assert apellido in gate.TOKENS


def test_sin_pdfinfo_el_gate_falla_en_vez_de_aprobar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Devolver cadena vacia cuando la herramienta no esta era aprobar sin haber mirado.

    ``pdftotext`` no extrae el titulo ni el autor del diccionario del PDF, que es justo por donde
    una identidad se cuela sin aparecer en ninguna fuente que uno recuerde revisar.
    """

    def _sin_binario(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("pdfinfo")

    monkeypatch.setattr(gate.subprocess, "run", _sin_binario)
    with pytest.raises(RuntimeError, match="pdfinfo no esta disponible"):
        gate._pdf_metadata(Path("cualquiera.pdf"))


def test_el_detector_no_acusa_a_un_texto_limpio() -> None:
    """Un detector que salta con todo es tan inutil como uno que no salta con nada."""
    assert gate._hits("un texto sin ninguna identidad dentro") == []
