# Estado del manuscrito de la fase 6

**Este manuscrito NO se envía.** Está sellado como artefacto de la fase 6 y se conserva sin tocar
—cambiarlo rompería el ledger de custodia— pero sus afirmaciones **no son las vigentes**.

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`../ARTIFACTS.md`](../ARTIFACTS.md), y se conservan **solo como registro histórico**. No
> sostienen ninguna afirmación vigente.

## Por qué

Cuatro revisiones a ciegas con criterios de MICAI recomendaron rechazo, la revisión del coautor
encontró falsa una de las premisas, y varias auditorías externas encontraron defectos que las
internas no vieron.

**El manuscrito se retira porque su cadena inferencial no es válida**, no porque se conozca ya la
dirección del resultado. Son dos cosas distintas y este apartado las confundía: decía «el resultado
central no sobrevive», que es un veredicto, y lo apoyaba en cifras que el propio ledger marca como
inválidas. **El veredicto sobre `H1-2026` está PENDIENTE** y puede salir en cualquiera de los dos
sentidos cuando se recalcule.

Lo que sí se sostiene sin depender de ninguna cifra: el estimando no estaba alineado, la regla de
entrega leía la etiqueta verdadera, y el intervalo remuestreaba la unidad equivocada. Los tres
defectos están reparados en el código y **sus artefactos siguen sin regenerar**.

**Registro histórico, en cuarentena y sin valor probatorio**: con el aparato defectuoso, al componer
las dos correcciones de protocolo el intervalo pasaba de (−0,0430, −0,0147) con p = 0,005 a
(−0,0410, +0,0077) con p = 0,130. Las cuatro cifras salieron del módulo con los tres defectos.

## Qué dice este PDF que ya no se sostiene

| Dónde | Qué afirma | Estado |
|---|---|---|
| `sections/04-results.tex:85` | evidencia significativa en el banco mayor | El inferencial que la produce conserva los tres defectos de `ml/eval/paper_micai_coverage.py`, sin reparar |
| `sections/06-conclusion.tex:24` | «no encontramos evidencia de que recortar la leyenda supere a abstenerse, y sí evidencia significativa de lo contrario» | La segunda mitad depende del mismo inferencial |
| Cualquier «a igual cobertura» | El estimando alineado | No está definido para dos de los cuatro mecanismos; se rehace con el marco de conjuntos |

## Qué lo sustituye

El plan por épicas, con la **EPIC 27** a la cabeza: la tabla de pérdidas, el estimando y su
población, y el margen práctico. Nada se computa antes de que el preregistro esté firmado
(`docs/paper/preregistro-v2-borrador.md`, hoy en BORRADOR con tres parámetros abiertos: función de
pérdida, margen práctico y criterio principal; el estimando y su población ya están cerrados).

El manuscrito nuevo es US-144 y siguientes, y **empieza cuando la EPIC 27 cierre**, no antes.
