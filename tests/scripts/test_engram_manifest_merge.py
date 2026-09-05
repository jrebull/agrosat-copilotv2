"""Tests para scripts/engram_manifest_merge.py.

El driver decide que memorias del equipo sobreviven a un merge. Cada test de aqui cierra un
hueco concreto que la version anterior tenia y que git resolvia en verde: entradas descartadas
en silencio por la forma del id, borrados deliberados que volvian desde BASE, y una entrada
pobre pisando a una completa. Todos fallan sobre esa version.

Los datos son minimos y sinteticos a proposito: esto es mecanica de fusion de un JSON, no una
metrica del articulo.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "engram_manifest_merge.py"


@pytest.fixture
def merge(tmp_path: Path):
    """Carga el script con CHUNKS apuntando a un directorio vacio.

    Aisla la logica de fusion de los chunks reales del repo: sin archivos en disco, lo que
    sobrevive es exactamente lo que la fusion decidio.
    """
    spec = importlib.util.spec_from_file_location("engram_manifest_merge", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["engram_manifest_merge"] = module
    spec.loader.exec_module(module)
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    module.CHUNKS = chunks
    return module


def _entry(chunk_id: str, memories: int = 5, **extra: object) -> dict[str, object]:
    """Una entrada de manifest con la forma que exporta engram."""
    return {
        "id": chunk_id,
        "created_by": "team",
        "created_at": "2026-09-04T10:00:00Z",
        "sessions": 1,
        "memories": memories,
        "prompts": 0,
        **extra,
    }


def _manifest(*entries: dict[str, object]) -> str:
    return json.dumps({"version": 1, "chunks": list(entries)}, indent=2) + "\n"


def _ids(manifest: dict[str, object]) -> list[str]:
    return sorted(str(entry["id"]) for entry in manifest["chunks"])


def test_id_fuera_del_formato_de_ocho_hex_sobrevive(merge) -> None:
    """Un id mas largo o en mayusculas no puede desaparecer sin aviso."""
    largo = _entry("ABCDEF123456")
    resultado = merge.union(_manifest(_entry("aaaa1111"), largo), prune_missing=False)
    assert "ABCDEF123456" in _ids(resultado)


def test_entrada_con_objeto_anidado_sobrevive(merge) -> None:
    """Un campo nuevo con forma de objeto no puede tirar la entrada entera."""
    anidada = _entry("bbbb2222", meta={"host": "laptop", "engram": "0.9"})
    resultado = merge.union(_manifest(anidada), prune_missing=False)
    assert "bbbb2222" in _ids(resultado)


def test_borrado_en_ambos_lados_no_resucita(merge) -> None:
    """Una purga hecha en las dos ramas se queda hecha."""
    base = _manifest(_entry("aaaa1111"), _entry("bbbb2222"))
    lado = _manifest(_entry("aaaa1111"))
    resultado = merge.three_way(base, lado, lado)
    assert _ids(resultado) == ["aaaa1111"]


def test_borrado_en_un_lado_con_el_otro_intacto_se_respeta(merge) -> None:
    """Regla de git: borrado contra no-modificado es borrado."""
    base = _manifest(_entry("aaaa1111"), _entry("bbbb2222"))
    ours = _manifest(_entry("aaaa1111"))
    theirs = _manifest(_entry("aaaa1111"), _entry("bbbb2222"))
    resultado = merge.three_way(base, ours, theirs)
    assert _ids(resultado) == ["aaaa1111"]


def test_chunk_modificado_por_el_otro_lado_gana_al_borrado(merge) -> None:
    """Si el companero siguio escribiendo en ese chunk, no se tira."""
    base = _manifest(_entry("aaaa1111"), _entry("bbbb2222", memories=5))
    ours = _manifest(_entry("aaaa1111"))
    theirs = _manifest(_entry("aaaa1111"), _entry("bbbb2222", memories=9))
    resultado = merge.three_way(base, ours, theirs)
    assert _ids(resultado) == ["aaaa1111", "bbbb2222"]


def test_chunk_nuevo_del_companero_sobrevive(merge) -> None:
    """Lo que solo trae un lado es exactamente lo que el driver existe para conservar."""
    base = _manifest(_entry("aaaa1111"))
    resultado = merge.three_way(
        base, _manifest(_entry("aaaa1111")), _manifest(_entry("aaaa1111"), _entry("cccc3333"))
    )
    assert _ids(resultado) == ["aaaa1111", "cccc3333"]


def test_entrada_pobre_no_pisa_a_la_completa(merge) -> None:
    """El ultimo en llegar no gana: gana el que trae mas metadatos."""
    completa = _entry("aaaa1111", memories=9)
    pobre = {"id": "aaaa1111", "created_by": "", "created_at": "", "sessions": 0, "memories": 0}
    resultado = merge.three_way(_manifest(), _manifest(completa), _manifest(pobre))
    assert resultado["chunks"][0]["memories"] == 9


def test_manifest_con_marcadores_de_conflicto_recupera_los_dos_lados(merge) -> None:
    """Tras un merge fallido, ningun chunk se pierde por los marcadores."""
    conflicto = (
        '{\n "version": 1,\n "chunks": [\n'
        "<<<<<<< HEAD\n"
        f"{json.dumps(_entry('aaaa1111'))}\n"
        "=======\n"
        f"{json.dumps(_entry('dddd4444'))}\n"
        ">>>>>>> theirs\n ]\n}\n"
    )
    resultado = merge.union(conflicto, prune_missing=False)
    assert _ids(resultado) == ["aaaa1111", "dddd4444"]


def test_check_apunta_al_comando_que_repara(merge, tmp_path, capsys) -> None:
    """El mensaje de fallo no puede mandar al comando que solo vuelve a fallar."""
    engram = tmp_path / ".engram"
    (engram / "chunks").mkdir(parents=True)
    (engram / "manifest.json").write_text("<<<<<<< HEAD\n{}\n", encoding="utf-8")
    merge.ENGRAM = engram
    merge.MANIFEST = engram / "manifest.json"
    merge.CHUNKS = engram / "chunks"
    sys.argv = ["engram_manifest_merge.py", "--check"]
    assert merge.main() == 1
    salida = capsys.readouterr().out
    assert "python scripts/engram_manifest_merge.py" in salida
    assert "make memory-import lo repara" not in salida
