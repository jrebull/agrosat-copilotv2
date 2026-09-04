"""Construye los cuatro PDF que acompanan la consulta al comite de etica de US-172.

Se generan desde `docs/paper/perdidas-protocolo.md`, que es la fuente unica: extraer los anexos a
mano produciria cuatro documentos que dicen cosas ligeramente distintas del protocolo que dicen
acompanar, y eso es lo que un comite encuentra primero.

Cada PDF lleva en portada el **commit exacto** del protocolo del que sale, para que sea inmutable en
el sentido que importa: cualquiera puede recuperar la fuente byte a byte.

Uso:
    poetry run python scripts/build_us172_adjuntos.py --salida <directorio>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOLO = REPO_ROOT / "docs" / "paper" / "perdidas-protocolo.md"

#: (nombre del PDF, titulo, encabezados del protocolo que lo componen).
ADJUNTOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "consentimiento-US172-v0.2",
        "Consentimiento informado — elicitacion de costes (US-172)",
        ("## Anexo A",),
    ),
    (
        "filtro-elegibilidad-US172-v0.2",
        "Filtro de elegibilidad — elicitacion de costes (US-172)",
        ("## Anexo D",),
    ),
    (
        "plan-custodia-US172-v0.2",
        "Plan de custodia de datos — elicitacion de costes (US-172)",
        ("## 6.", "## 9.", "## 10 bis."),
    ),
)


def _secciones(texto: str) -> dict[str, str]:
    """Split the protocol into sections keyed by their heading line."""
    salida: dict[str, str] = {}
    actual: str | None = None
    buffer: list[str] = []
    for linea in texto.splitlines():
        if linea.startswith("## "):
            if actual is not None:
                salida[actual] = "\n".join(buffer).strip()
            actual = linea
            buffer = [linea]
        elif actual is not None:
            buffer.append(linea)
    if actual is not None:
        salida[actual] = "\n".join(buffer).strip()
    return salida


#: Fecha fija para que el PDF sea reproducible byte a byte. Un adjunto que cambia de MD5 en cada
#: compilacion no se puede sellar, y sin sello el «PDF inmutable» es una palabra.
EPOCH_FIJO = "1757030400"  # 2025-09-04T00:00:00Z


def _pdf(fuente: str, destino: Path, titulo: str) -> None:
    """Render one Markdown string to a byte-reproducible PDF with xelatex."""
    entorno = {**os.environ, "SOURCE_DATE_EPOCH": EPOCH_FIJO, "FORCE_SOURCE_DATE": "1"}
    subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown",
            "-o",
            str(destino),
            "--pdf-engine=xelatex",
            "-V",
            "lang=es",
            "-V",
            "geometry:margin=2.5cm",
            "-V",
            f"title={titulo}",
            "--metadata",
            "author=",
        ],
        input=fuente,
        text=True,
        check=True,
        capture_output=True,
        env=entorno,
    )


def main() -> int:
    """Build the four attachments and print their names and digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    sha = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", str(PROTOCOLO)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    sucio = subprocess.run(
        ["git", "status", "--porcelain", "--", str(PROTOCOLO)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if sucio:
        print("ERROR: el protocolo tiene cambios sin commitear; el PDF no seria trazable")
        return 1

    texto = PROTOCOLO.read_text(encoding="utf-8")
    pie = (
        f"\n\n---\n\nFuente: `docs/paper/perdidas-protocolo.md`, commit `{sha}` del repositorio "
        "`agrosat-copilotv2`. Este PDF se genera desde esa fuente y no se edita a mano.\n"
    )

    generados: list[Path] = []
    completo = args.salida / f"protocolo-US172-v0.2-{sha}.pdf"
    _pdf(texto + pie, completo, "Protocolo de elicitacion de la tabla de perdidas (US-172) v0.2")
    generados.append(completo)

    secciones = _secciones(texto)
    for nombre, titulo, prefijos in ADJUNTOS:
        partes = [v for k, v in secciones.items() if k.startswith(prefijos)]
        if not partes:
            print(f"ERROR: no se encontro ninguna seccion para {nombre}")
            return 1
        destino = args.salida / f"{nombre}.pdf"
        _pdf("\n\n".join(partes) + pie, destino, titulo)
        generados.append(destino)

    for ruta in generados:
        digest = hashlib.md5(ruta.read_bytes()).hexdigest()  # noqa: S324 - sello de custodia
        print(f"{ruta.name}  {digest}  {ruta.stat().st_size} bytes")
    print(f"protocolo en el commit {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
