"""Gate de doble ciego: ningun token identificatorio puede sobrevivir en el PDF de envio.

Extrae el texto del PDF **ensamblado**, no del `.tex`, porque es el PDF lo que ve el revisor y
porque una identidad puede entrar por la bibliografia, por un pie de figura o por los metadatos
sin aparecer nunca en una fuente que uno recuerde revisar. Comprueba tambien el titulo y el autor
del diccionario del PDF, que `pdftotext` no extrae.

El gate se prueba en negativo con `--autoprueba`: inyecta cada token en un texto de mentira y
exige que el detector lo encuentre. Un control que nunca se ha visto fallar no se sabe si funciona.

Uso:
    poetry run python scripts/paper_micai_anon_check.py paper/micai/main.pdf
    poetry run python scripts/paper_micai_anon_check.py --autoprueba
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Tokens que delatarian a los autores o a la institucion. En minusculas; la busqueda ignora
#: mayusculas y acentos no se normalizan porque los tokens se escriben como aparecerian.
TOKENS: tuple[str, ...] = (
    "zizumbo",
    "rebull",
    "saucedo",
    "javier",
    "arthur",
    "agrosat",
    "agrosatcopilot",
    "jrebull",
    "haowei",
    "tecnologico de monterrey",
    "tec de monterrey",
    "itesm",
    "0009-0002-1603-8946",
    "github.com/jrebull",
    "agrosat-micai-site",
    "rebull@outlook.com",
)


def _pdf_text(pdf: Path) -> str:
    """Extract the full text of the assembled PDF.

    Args:
        pdf: Path to the PDF.

    Returns:
        The extracted text, lowercased.

    Raises:
        RuntimeError: if ``pdftotext`` is unavailable or fails.
    """
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext fallo sobre {pdf}: {result.stderr.strip()}")
    return result.stdout.lower()


def _pdf_metadata(pdf: Path) -> str:
    """Extract the PDF dictionary metadata, which the text layer does not carry.

    Args:
        pdf: Path to the PDF.

    Returns:
        The metadata block, lowercased; empty when ``pdfinfo`` is unavailable.
    """
    result = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False)
    return result.stdout.lower() if result.returncode == 0 else ""


def _hits(haystack: str) -> list[str]:
    """Find every identity token present in a text.

    Args:
        haystack: Lowercased text to scan.

    Returns:
        The tokens found, in declaration order.
    """
    return [t for t in TOKENS if re.search(re.escape(t), haystack)]


def _self_test() -> int:
    """Prove the detector fires, one token at a time.

    Returns:
        Process exit code.
    """
    # Se comprueba pertenencia y no igualdad: varios tokens se contienen entre si
    # ("agrosat" dentro de "agrosatcopilot"), asi que inyectar uno dispara legitimamente
    # mas de una coincidencia. Exigir la lista exacta rompia la autoprueba, no el detector.
    failures = [t for t in TOKENS if t not in _hits(f"texto inocuo {t} mas texto inocuo")]
    if failures:
        logger.error("autoprueba_fallida", tokens=failures)
        return 1
    if _hits("un texto sin ninguna identidad dentro"):
        logger.error("autoprueba_fallida", motivo="falso positivo sobre texto limpio")
        return 1
    logger.info("autoprueba_ok", tokens=len(TOKENS))
    return 0


def main() -> int:
    """Run the gate over one PDF, or run the negative self-test.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, help="PDF de envio a comprobar.")
    parser.add_argument(
        "--autoprueba",
        action="store_true",
        help="Prueba el detector en negativo y sale, sin mirar ningun PDF.",
    )
    args = parser.parse_args()

    if args.autoprueba:
        return _self_test()
    if args.pdf is None:
        parser.error("hace falta la ruta del PDF, o bien --autoprueba")

    if _self_test() != 0:
        return 1

    body = _hits(_pdf_text(args.pdf))
    meta = _hits(_pdf_metadata(args.pdf))
    if body or meta:
        logger.error("identidad_filtrada", cuerpo=body, metadatos=meta, pdf=str(args.pdf))
        print("paper-anon-check: FALLO, el PDF de envio revela identidad")
        return 1
    logger.info("anonimo_ok", pdf=str(args.pdf), tokens=len(TOKENS))
    print("paper-anon-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
