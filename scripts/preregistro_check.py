"""Comprueba que el contrato del estimando y la prosa del preregistro dicen lo mismo.

`docs/paper/estimando-v1.json` es la fuente normativa del estimando de US-173. La prosa de la
seccion 4.5 del preregistro dice lo mismo en castellano. **Dos fuentes que dicen lo mismo se
separan**: es lo que ha pasado en este proyecto cada vez que un numero vivia en dos sitios, y no
hay motivo para que un contrato se comporte mejor que un numero.

Qué comprueba:

1. Las claves obligatorias existen y valen exactamente lo decidido. Tres de ellas —punto de
   operacion, reigualado en prueba y afirmacion de transporte— son las que convertirian el estudio
   en otro estudio, y por eso tienen valor fijo y no solo tipo.
2. La prosa de la seccion 4.5 afirma lo mismo: cada valor del contrato tiene una marca textual que
   tiene que estar presente, y su contraria no puede estarlo.

Uso:
    poetry run python scripts/preregistro_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRATO = REPO_ROOT / "docs" / "paper" / "estimando-v1.json"
DEFAULT_PREREGISTRO = REPO_ROOT / "docs" / "paper" / "preregistro-v2-borrador.md"

#: Valor exigido para cada clave del contrato. No es un esquema de tipos: son las decisiones.
EXIGIDO: dict[str, Any] = {
    "scope": "dataset_conditional",
    "population": "all_eligible_test_parcels",
    "analysis_unit": "parcel",
    "dependence_cluster": "patch_id",
    "include_non_delivery": True,
    "class_universe_source": "training_only",
    "common_class_universe": True,
    "operating_point_source": "training_validation_only",
    "rematch_on_test": False,
    "pool_across_datasets": False,
    "transport_claim": False,
    "minimum_unique_paired_clusters": 3,
    "k_role": "spatial_sensitivity_not_replication",
}

#: Marcas textuales que la prosa tiene que afirmar, y las que no puede afirmar.
#: (clave del contrato, fragmento que DEBE estar, fragmentos que NO pueden estar)
COHERENCIA: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("scope", "Condicional a cada banco", ("entre regiones y campañas, con sitio-año",)),
    ("population", "incluidas las que no reciben entrega", ()),
    ("analysis_unit", "La parcela", ()),
    ("dependence_cluster", "`patch_id`", ()),
    ("operating_point_source", "ENTERAMENTE de entrenamiento y validación", ()),
    ("rematch_on_test", "Queda prohibido volver a igualar", ()),
    ("pool_across_datasets", "Ningún promedio inferencial", ()),
    ("transport_claim", "**No se afirma.**", ()),
    ("minimum_unique_paired_clusters", "tres clústeres únicos pareados", ()),
    ("k_role", "sensibilidad espacial", ()),
    ("class_universe_source", "fijado exclusivamente desde ENTRENAMIENTO", ()),
)

HEADER_COHERENCE_CHECKS = 3

#: Sello de la prosa normativa. Comprobar que una FRASE esta presente no dice nada de lo que el
#: parrafo afirma: el gate del panel pasaba mientras la seccion 4.6 decia que un miembro del panel
#: "se excluye". Aqui no hay forma de entender la prosa con un patron, asi que se hace lo unico
#: honesto: si el texto normativo cambia, el gate se pone en rojo hasta que alguien lo relea contra
#: el contrato y actualice el sello EN EL MISMO COMMIT. No afirma que la prosa sea correcta; afirma
#: que nadie la ha movido desde que se leyo.
SELLOS: dict[str, str] = {
    "cabecera": "4f1d07c6b9b192a4c385ceeffe590eae",
    "seccion_4_5": "19851f95be1449f7098674e3dad49848",
}


def _seccion_45(texto: str) -> str:
    """The prose of section 4.5, which is what the contract has to agree with."""
    inicio = texto.find("### 4.5")
    if inicio < 0:
        return ""
    fin = texto.find("\n## ", inicio)
    return texto[inicio : fin if fin > 0 else len(texto)]


def _cabecera(texto: str) -> str:
    """Return the pre-registration header before the first numbered section."""
    fin = texto.find("\n## 1.")
    return texto[: fin if fin > 0 else len(texto)]


def main() -> int:
    """Check the estimand contract against the pre-registration prose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrato", type=Path, default=DEFAULT_CONTRATO)
    parser.add_argument("--preregistro", type=Path, default=DEFAULT_PREREGISTRO)
    parser.add_argument(
        "--resellar",
        action="store_true",
        help=(
            "Actualiza el sello de la prosa normativa. Usalo SOLO despues de releer la cabecera y "
            "la seccion 4.5 contra el contrato: el sello no dice que el texto sea correcto, dice "
            "que alguien lo leyo."
        ),
    )
    args = parser.parse_args()

    if not args.contrato.exists():
        print(f"ERROR: no existe el contrato {args.contrato}")
        return 2
    try:
        contrato = json.loads(args.contrato.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: el contrato no es JSON valido: {exc}")
        return 2

    fallos: list[str] = []
    for clave, valor in EXIGIDO.items():
        if clave not in contrato:
            fallos.append(f"falta la clave `{clave}` en el contrato")
        elif contrato[clave] != valor:
            fallos.append(
                f"`{clave}` vale {contrato[clave]!r} y la decision de US-173 es {valor!r}"
            )

    texto_preregistro = args.preregistro.read_text(encoding="utf-8")
    cabecera = _cabecera(texto_preregistro)
    prosa = _seccion_45(texto_preregistro)
    if "**Tres** parámetros están abiertos" not in cabecera:
        fallos.append("la cabecera no declara exactamente tres parámetros abiertos")
    if "No se firma hasta cerrar esos tres" not in cabecera:
        fallos.append("la regla de firma no exige cerrar los tres parámetros que siguen abiertos")
    if "cerrarlos los cuatro" in cabecera:
        fallos.append("la cabecera cierra el estimando pero todavía exige cerrar cuatro parámetros")
    if not prosa:
        fallos.append("el preregistro no tiene seccion 4.5: el contrato no tiene con que cuadrar")
    else:
        # La clave del universo de clases se decide en 4.2, no en 4.5.
        completo = texto_preregistro
        for clave, debe, no_puede in COHERENCIA:
            ambito = completo if clave == "class_universe_source" else prosa
            if debe not in ambito:
                fallos.append(f"el contrato fija `{clave}` y la prosa no lo dice: falta «{debe}»")
            for prohibido in no_puede:
                if prohibido in ambito:
                    fallos.append(
                        f"la prosa conserva «{prohibido}», que contradice `{clave}` del contrato"
                    )

    # Sello de la prosa normativa: lo unico que un patron puede garantizar de un texto es que no
    # ha cambiado. Si cambia, se relee y se vuelve a sellar en el mismo commit.
    for nombre, texto_sellado in (("cabecera", cabecera), ("seccion_4_5", prosa)):
        digest = hashlib.md5(texto_sellado.encode("utf-8")).hexdigest()  # noqa: S324 - sello
        if args.resellar and digest != SELLOS[nombre]:
            fuente = Path(__file__)
            fuente.write_text(
                fuente.read_text(encoding="utf-8").replace(
                    f'"{nombre}": "{SELLOS[nombre]}"', f'"{nombre}": "{digest}"', 1
                ),
                encoding="utf-8",
            )
            print(f"resellado `{nombre}`: {SELLOS[nombre]} -> {digest}")
            continue
        if digest != SELLOS[nombre]:
            fallos.append(
                f"la prosa normativa `{nombre}` cambio ({digest} frente a {SELLOS[nombre]}): "
                "reelela contra el contrato y actualiza el sello en scripts/preregistro_check.py "
                "en el mismo commit. El gate no puede leer un parrafo; solo puede saber si se movio"
            )

    # La formula del estimando tiene que estar, y no puede traer una perdida inventada.
    if not re.search(r"R_\{d,a,m\}", texto_preregistro):
        fallos.append("el preregistro no declara el estimando simbolicamente")

    print(f"claves del contrato verificadas: {len(EXIGIDO)}")
    print(f"comprobaciones de coherencia con la prosa: {len(COHERENCIA) + HEADER_COHERENCE_CHECKS}")
    for fallo in fallos:
        print(f"FALLO: {fallo}")
    if fallos:
        print(f"preregistro-check: {len(fallos)} fallo(s)")
        return 1
    print("preregistro-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
