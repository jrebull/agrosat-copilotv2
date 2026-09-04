# Estado del manuscrito de la fase 6

**Este manuscrito NO se envía.** Está sellado como artefacto de la fase 6 y se conserva sin tocar
—cambiarlo rompería el ledger de custodia— pero sus afirmaciones **no son las vigentes**.

## Por qué

Cuatro revisiones a ciegas con criterios de MICAI recomendaron rechazo, la revisión del coautor
encontró falsa una de las premisas, y dos auditorías externas encontraron defectos que las nueve
internas no vieron. El resultado central no sobrevive: al componer las dos correcciones de
protocolo, el intervalo pasa de (−0,0430, −0,0147) con p = 0,005 a (−0,0410, +0,0077) con p = 0,130.

## Qué dice este PDF que ya no se sostiene

| Dónde | Qué afirma | Estado |
|---|---|---|
| `sections/04-results.tex:85` | evidencia significativa en el banco mayor | El inferencial que la produce conserva los tres defectos de `ml/eval/paper_micai_coverage.py`, sin reparar |
| `sections/06-conclusion.tex:24` | «no encontramos evidencia de que recortar la leyenda supere a abstenerse, y sí evidencia significativa de lo contrario» | La segunda mitad depende del mismo inferencial |
| Cualquier «a igual cobertura» | El estimando alineado | No está definido para dos de los cuatro mecanismos; se rehace con el marco de conjuntos |

## Qué lo sustituye

El plan por épicas, con la **EPIC 27** a la cabeza: la tabla de pérdidas, el estimando y su
población, y el margen práctico. Nada se computa antes de que el preregistro esté firmado
(`docs/paper/preregistro-v2-borrador.md`, hoy en BORRADOR con cuatro parámetros abiertos).

El manuscrito nuevo es US-144 y siguientes, y **empieza cuando la EPIC 27 cierre**, no antes.
