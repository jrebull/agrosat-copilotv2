# Metadatos de autoría — fuente única de verdad

**Para qué**: Springer exige que el nombre sea idéntico carácter a carácter en el artículo,
el sistema del congreso, la licencia y el repositorio citable, y **no permite cambiar el
autor de correspondencia después del camera-ready**. Este archivo fija esas cadenas en un
solo sitio para que nadie las teclee de memoria.

**Última verificación**: 3 de septiembre de 2026.

---

## ORCID, verificados contra el registro público

Consultados en `pub.orcid.org/v3.0/{id}/person` el 2 de septiembre de 2026, no copiados de
un correo ni de un documento anterior.

| Autor | ORCID | Nombre que devuelve el registro | Visibilidad |
|---|---|---|---|
| Primer autor y **autor de correspondencia** | [`0009-0002-1603-8946`](https://orcid.org/0009-0002-1603-8946) | Arthur Jafed · Zizumbo Velasco | pública |
| Segundo autor | [`0009-0008-2089-5274`](https://orcid.org/0009-0008-2089-5274) | Javier Augusto · Rebull Saucedo | pública |

Los dos coinciden con las personas que firman. El de Arthur queda registrado y su punto
sale de la lista de pendientes.

## Orden de firma

1. Arthur Jafed Zizumbo-Velasco
2. Javier A. Rebull-Saucedo

**Con guion medio los dos**, decidido el 3 de septiembre de 2026. El registro ORCID de Arthur
devuelve «Zizumbo Velasco» sin guion; la forma que firma el artículo lleva guion y es la que hay
que usar también en el sistema del congreso y en la licencia.

**Isaac Ávila y Aaron Bocanegra van en los agradecimientos, no como autores del código.**
Corregido el 3 de septiembre: el camera-ready los presentaba como autores del software, y eso
**no era cierto**. Ahora se les agradece sin atribuirles un papel que no se ha verificado.

## Autor de correspondencia

**Arthur Jafed Zizumbo-Velasco**, decidido el 3 de septiembre de 2026. Ya está impreso en el
camera-ready con `\thanks{Corresponding author.}`, y la nota provisional que advertía de que
faltaba decidirlo ha desaparecido del fuente.

Firma la licencia de Springer **a mano** —las digitales no se aceptan— y Springer no permite
cambiarlo después del camera-ready. Es la única decisión de este archivo que no se puede posponer: esa persona
firma la licencia de Springer **a mano** —las firmas digitales no se aceptan— y Springer no
permite cambiarla después del camera-ready.

## Afiliación y financiamiento

Tecnológico de Monterrey. Una sola afiliación para ambos. El camera-ready **agradece** a la
institución y a la vez **declara explícitamente que el trabajo no recibió financiamiento, beca
ni apoyo institucional**, ni del Tec ni de ningún otro organismo. Las dos cosas conviven sin
contradicción: una reconoce dónde se hizo el trabajo, la otra dice quién no lo pagó. En el artículo se escribe
«Tecnológico de Monterrey», no «Tec de Monterrey» ni «ITESM».

## Discrepancias detectadas en el repositorio

Hay que unificarlas antes del camera-ready. No se corrigen aquí de oficio porque la forma
canónica del nombre la eligen sus dueños, no yo.

| Cadena | Dónde aparece | Nota |
|---|---|---|
| `Arthur Jafed Zizumbo Velasco` | `paper/main.tex`, `paper/main_es.tex`, `README.md`, `LICENSE` | Forma completa; coincide con el ORCID |
| `Arthur Zizumbo` | `pyproject.toml`, varios documentos de contexto | Forma corta; **no** coincide con la anterior |
| `Javier A. Rebull-Saucedo` | `paper/main.tex`, `README.md`, `pyproject.toml` | Con guion |
| `Javier Augusto Rebull Saucedo` | registro ORCID | **Sin guion**. Decidir cuál es la forma canónica y usarla en los dos sitios |

## Correos

- Arthur: `A01796363@tec.mx` (el que hoy figura en `paper/main.tex`; confirmar que es el
  que quiere publicar).
- Javier: `rebull@exatec.tec.mx`.

## Antes del envío, por el doble ciego

El cuaderno del proyecto vive en [`jrebull/agrosat-micai-site`](https://github.com/jrebull/agrosat-micai-site),
**público por decisión del equipo el 2 de septiembre de 2026**. El sitio desplegado lleva
`noindex`, pero el repositorio sí es indexable y contiene los dos nombres, los dos ORCID y
la afiliación.

- [ ] **Antes de enviar**, hacer una de estas dos y dejar constancia de cuál: pasar el
      repositorio a privado (`gh repo edit --visibility private`), o retirar del sitio los
      nombres, los ORCID y la afiliación hasta después de la notificación.
- [ ] No enlazar el sitio desde el artículo ni desde el repositorio citado mientras dure la
      revisión.

## Estado del camera-ready

Existe y compila: `make micai-pdf-cr` produce `paper/micai/main_cr.pdf` y su versión en español,
quince páginas cada una, con ORCID, afiliación, agradecimientos a Isaac Ávila y Aaron Bocanegra
como autores del software, y «Disclosure of Interests».

Sale del **mismo** `main.tex` que la versión anónima, mediante un testigo en disco que el objetivo
de Make crea y borra. Comprobado: el PDF anónimo conserva el mismo MD5 antes y después de escribir
todo el bloque de firmas, así que **el cuerpo del artículo es byte a byte el mismo** en las dos
salidas, que es lo que exige la fase 8.

El objetivo de Make además **exige que el gate de anonimato falle** sobre el camera-ready. Un gate
que solo se comprueba en verde no distingue entre «está anónimo» y «no lo he mirado».

## Qué falta

- [x] ~~Decidir el autor de correspondencia.~~ Arthur, 3 de septiembre de 2026.
- [x] ~~Elegir la forma canónica del apellido.~~ Con guion medio los dos, 3 de septiembre de 2026.
- [ ] **Discrepancia abierta, y no la resuelvo de oficio**: `README.md`, `LICENSE` y
      `pyproject.toml` siguen presentando a Isaac Ávila y Aaron Bocanegra como autores del
      código, heredado del proyecto original. El artículo ya no lo dice. Si eso tampoco es
      cierto en el repositorio, hay que corregirlo ahí, pero la autoría del software y su
      licencia las deciden sus dueños.
- [ ] Propagar la forma con guion a `pyproject.toml`, `README.md` y `LICENSE`, hoy sin él.
- [ ] Unificar `Arthur Zizumbo` a la forma completa en `pyproject.toml`.
- [ ] Confirmar el correo de publicación de Arthur.
- [ ] `CITATION.cff`, que hoy no existe.
