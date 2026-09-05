# Sanidad de los miembros heredados (US-119)

**Artefacto**: [`reports/paper_micai/us119/sanidad_miembros.csv`](../../reports/paper_micai/us119/sanidad_miembros.csv)
y su JSON con procedencia · **Productor**: `scripts/run_us119_sanidad_miembros.py` · **Umbral**: 0,15

Un modelo puede caer entre su validación y el fold retenido por dos motivos muy distintos: porque el
fold 5 es más difícil, o porque el arnés lo está cargando o alimentando mal. Lo segundo ya pasó
—`tsvit-pheno-fullm` marcaba 0,2552 en vez de 0,7883 porque el dataset le daba T=10 y el modelo
esperaba T=37— y se descubrió por casualidad. Esto lo convierte en una comprobación.

## La tabla, y las dos columnas son por píxel

**Las dos métricas comparadas son por píxel.** El F1 por parcela se calcula, se publica en su propia
columna y **no se resta de nada**: mezclarlo con mIoU o F1 por píxel es lo que el criterio de
aceptación prohíbe expresamente, y es fácil de hacer sin darse cuenta porque los dos números existen
para cada miembro y se parecen.

| miembro | declarado (píxel) | fold 5 (píxel) | Δ | >0,15 | F1 parcela (aparte) |
|---|---:|---:|---:|:--:|---:|
| `tsvit-pheno` | 0,7500 | 0,7401 | 0,0099 | no | 0,7367 |
| `tsvit-pheno-fullm` | 0,8078 | 0,7918 | 0,0160 | no | 0,7883 |
| `deeplabv3plus` | 0,3864 | 0,3614 | 0,0250 | no | 0,3546 |
| `segformer` | 0,3382 | 0,3617 | **−0,0235** | no | 0,3382 |
| `unet` | 0,3463 | 0,2260 | 0,1204 | no | 0,2055 |
| **`anysat`** | 0,5716 | 0,2119 | **0,3597** | **sí** | 0,1873 |
| **`utae`** | 0,6163 | 0,2152 | **0,4010** | **sí** | 0,1880 |

`segformer` **sube** en fold 5, que es lo contrario de lo que sugería la premisa de esta historia:
la sospecha era que «utae, anysat y segformer caen tanto», y segformer no cae.

## Lo que se comprobó, con evidencia

**1. Que la métrica declarada describe los pesos que el arnés carga.** `unet`, `anysat` y
`segformer` guardan su métrica en un fichero **distinto** del que carga el registro. Comparar la
métrica de un fichero con el volcado de otro solo vale si los pesos son los mismos, así que el
productor los compara tensor a tensor: **380 de 380 idénticos en `unet`, 501 de 501 en `anysat` y
208 de 208 en `segformer`**. El JSON conserva ambas rutas, sus SHA-256, los tres conteos y el booleano
de identidad; el productor aborta si una sola fila no coincide. Ya no son cifras escritas a mano en
esta página.

**2. Que ningún checkpoint registra sus folds de entrenamiento.** La búsqueda recorre de forma
recursiva todos los diccionarios y listas del checkpoint y el resultado vacío queda en el JSON. Se
declara que **el registro no existe**, en vez de deducirlo. El volcado usa `(1,2,3)` para las
estadísticas de normalización, pero eso es una elección del arnés y no una constancia de cómo se
entrenó cada modelo.

**3. Que la exposición al defecto de `n_timesteps` es la misma en los dos que fallan.** `anysat` y
`utae` son temporales y tienen `model_kwargs` **vacío**, así que el arnés les da la T por defecto de
10 sin ningún registro de con cuál se entrenaron. Es la misma forma de exposición que hundió a
`tsvit-pheno-fullm`. Los cinco que pasan el umbral son los no temporales, o los temporales cuya T sí
está fijada.

## Lo que NO se pudo comprobar, dicho como tal

**La causa de la caída de `anysat` y `utae` no está identificada.** Se intentó y no salió:

- Una sonda directa —cargar el modelo y evaluarlo variando T— **no vale**: `utae` necesita
  `positions` en su `forward` y `anysat` falla con `KeyError: 's2_dates'`. Mi sonda no reproducía la
  convención de llamada del arnés, así que sus números no dicen nada y **no se reportan**.
- El experimento correcto —variar `n_timesteps` en la especificación y volcar con el propio arnés—
  **excedió el presupuesto de tiempo en CPU** y quedó sin terminar.

Así que la exposición está documentada y **la causa no**. Lo que no se hace es rellenar ese hueco
con una cláusula de escape.

Los posteriores contienen 28 532 identificadores, pero 11 892 no tienen etiqueta `semantic18`
válida. El estimando conserva las 16 640 parcelas elegibles; el productor exige que cada miembro las
cubra todas y registra los 11 892 posteriores fuera de esa población, en vez de confundir presencia
de una posterior con elegibilidad para evaluación.

## Decisión por miembro

| miembro | decisión | motivo |
|---|---|---|
| `tsvit-pheno` | **incluir** | Δ 0,0099 |
| `tsvit-pheno-fullm` | **incluir** | Δ 0,0160, tras el fix de `n_timesteps` |
| `deeplabv3plus` | **incluir** | Δ 0,0250 |
| `segformer` | **incluir** | Δ −0,0235: mejora en fold 5 |
| `unet` | **incluir** | Δ 0,1204, por debajo del umbral. Se reporta el valor, que es alto |
| `anysat` | **excluir del panel inferencial** | Δ 0,3597 con causa sin identificar. Se conserva como descriptivo |
| `utae` | **excluir del panel inferencial** | Δ 0,4010 con causa sin identificar. Se conserva como descriptivo |

**«Excluir» aquí es del panel inferencial, no del repositorio.** Aplicarlo al inventario es el paso
de US-139, que es donde el panel se congela; separar las dos cosas evita que un mismo campo cargue
con dos significados.

La decisión tampoco vive solo en esta tabla: cada fila del CSV y del JSON contiene
`decision_panel`, y el resumen JSON enumera los cinco incluidos, los dos excluidos y el miembro que
desmiente la premisa.

**Y «corregir» sigue disponible**: si se identifica la causa —lo más probable, la T con la que se
entrenaron— los dos vuelven al panel con su procedencia. La exclusión es por no saber, no por saber
que están mal.

## Lo que esto le cambia al artículo

El panel elegible se queda en **cinco miembros**, y dos de ellos —`tsvit-pheno` y
`tsvit-pheno-fullm`— son de la misma familia. US-139 pide **al menos tres predictores de familias
distintas por banco**: con `deeplabv3plus`, `segformer` y la familia TSViT se cumple, pero por poco,
y perder uno más dejaría el panel por debajo del mínimo. Merece decirse antes de congelarlo.
