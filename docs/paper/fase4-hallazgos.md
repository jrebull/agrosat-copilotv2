# Fase 4 · La descomposición se transporta, y más limpia que en el conjunto primario

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md): las produjo `ml/eval/paper_micai_coverage.py`
> cuando aun tenia los tres defectos —denominador movil, punto de operacion elegido dentro del
> bloque evaluado, y remuestreo a nivel de parcela—. **Ninguna de esas cifras entra en el
> articulo** hasta regenerarlas (US-124, US-125). Se conservan sin retocar porque el registro de
> lo que creimos importa tanto como lo que resulte.

**Fecha**: 3 de septiembre de 2026 · **Artefactos**: `reports/paper_micai/fase4/` ·
**Preregistro**: [`preregistro-fases-3-4.md`](preregistro-fases-3-4.md), enmienda 1 y
correcciones 1 y 2 · **Fase previa**: [`fase3-hallazgos.md`](fase3-hallazgos.md)

## Qué se corrió

El **mismo** protocolo de la fase 3, con el mismo módulo (`ml/eval/paper_micai_coverage.py`) y
sin tocar una línea de su método, sobre **BreizhCrops 2017 L2A**: otra región de Francia, otro
año, otro reparto de clases y otro clasificador. Cambian solo las adaptaciones que el
preregistro declaró **antes** de entrenar —bloques por región en vez de hexágonos H3, dos
universos de clases, submuestreo de 30 000 parcelas por región— y las dos correcciones que se
escribieron cuando aparecieron los problemas, no después de ver el resultado.

El clasificador es XGBoost sobre las **mismas 185 características** que el conjunto primario,
entrenado **dejando una región fuera**: la posterior de cada parcela viene de un modelo que no
vio su región. Da F1-macro 0,4979 sobre frh01 y 0,4811 sobre frh04, con accuracy 0,7421 y
0,7257. Es transferencia entre regiones, no validación interna.

## El resultado, en una línea

**Se transporta, y más fuerte.** En el conjunto primario, el 87,1 % de la mejora aparente al
recortar la leyenda era el denominador. Aquí es el **95,1 %** en el criterio principal, y el
**100,0 %** en todo el tramo donde la cobertura sigue completa.

## El pasillo donde los cuatro mecanismos son idénticos

Es el hallazgo más limpio de las dos fases, y no estaba previsto.

Al pasar de nueve clases a seis, el F1-macro sube de **0,4895 a 0,7342**, una mejora aparente de
**+0,2447**. Y las cuatro series —retirada por F1, retirada por soporte, rechazo por confianza y
el control sin mecanismo— dan **exactamente el mismo número** en los cuatro puntos, con
cobertura **1,0000**.

La razón es aritmética y por eso es concluyente: las tres clases que se retiran primero —girasol
con 2 parcelas, nueces con 7, huertos con 289 de 60 000— son tan raras que el argmax libre no las
predice nunca. Retirarlas no retira ni una sola parcela de la entrega, así que **ningún mecanismo
hace nada** y la métrica sube sola casi un cuarto de punto. Es el denominador desnudo, sin
mecanismo que lo acompañe ni interpretación que lo disimule.

## Descomposición, universo de nueve clases

| K | F1 aparente | mejora aparente | denominador | mecanismo | cobertura |
|---:|---:|---:|---:|---:|---:|
| 9 | 0,4895 | — | — | — | 1,000 |
| 8 | 0,5507 | +0,0612 | +0,0612 (**100,0 %**) | +0,0000 (0,0 %) | 1,000 |
| 7 | 0,6293 | +0,1398 | +0,1398 (**100,0 %**) | +0,0000 (0,0 %) | 1,000 |
| 6 | 0,7342 | +0,2447 | +0,2447 (**100,0 %**) | +0,0000 (0,0 %) | 1,000 |
| **5** | **0,8055** | **+0,3160** | **+0,3004 (95,1 %)** | **+0,0156 (4,9 %)** | **0,859** |
| 4 | 0,8579 | +0,3684 | +0,3352 (91,0 %) | +0,0333 (9,0 %) | 0,806 |
| 3 | 0,9126 | +0,4232 | +0,3826 (90,4 %) | +0,0406 (9,6 %) | 0,444 |

K = 5 es el criterio principal preregistrado. «Denominador» es el control sin mecanismo, que
puntúa al predictor **intacto** sobre la misma leyenda entregándolo todo.

## H1 vuelve a caer, y esta vez cae hacia el otro lado

En la fase 3, el rechazo por confianza salía por delante de la retirada de clases, pero el
intervalo incluía el cero y ningún contraste sobrevivía a Holm: **no se podía distinguir de
cero**. Aquí sí se distingue, y el signo es el contrario al que H1 predecía.

| | conjunto primario (K = 9 de 18) | BreizhCrops (K = 5 de 9) |
|---|---|---|
| retirada por F1 | 0,7645 | 0,8055 |
| rechazo por confianza | 0,7933 | 0,8303 |
| delta | −0,0288 | −0,0248 |
| IC 95 % pareado | (−0,0414, **+0,0080**) | (−0,0272, **−0,0225**) |
| Holm | 0,3360 | **< 0,0001** |
| las dos regiones/bloques coinciden en signo | — | sí: −0,0248 y −0,0249 |

Es decir: **igualando la cobertura, abstenerse en las parcelas de baja confianza da mejor calidad
que retirar clases del catálogo**, y en el segundo conjunto la diferencia es estadísticamente
distinguible de cero con corrección por multiplicidad. La hipótesis de la que partió el equipo no
solo no se sostiene: su contraria sí.

## Retirar por soporte es el peor criterio, y no es monótono

Es el criterio que el equipo declaró haber usado de verdad para pasar de dieciocho clases a
`france-12`. Aquí queda **último de los tres en todos los puntos**, por **0,0926** en el criterio
principal y por **0,2085** en K = 3.

Y hace algo que ninguno de los otros hace: **empeora al retirar más**. De K = 5 a K = 3 baja de
0,7129 a 0,7041. Retirar clases por soporte bajo saca del catálogo clases que el modelo predecía
bien —colza, con 1 718 parcelas y separable— y deja dentro las dos praderas, que el modelo confunde
entre sí. El criterio optimiza el tamaño de la muestra, no la calidad de lo que se entrega.

## Los dos universos dicen lo mismo

Los contrastes del universo de siete clases coinciden con los del de nueve **hasta el cuarto
decimal** en todos los K compartidos. Tiene la misma causa que el pasillo: las clases por debajo
de cien parcelas nunca se predicen, así que quitarlas del universo no mueve nada. La robustez
frente a la elección del umbral no hay que argumentarla, se ve.

## Limitaciones, dichas por su nombre

1. **Dos bloques, no cinco.** BreizhCrops no trae coordenadas por parcela, así que la única
   estructura espacial disponible es la región. El remuestreo captura la variación **dentro** de
   cada región, no **entre** regiones, y por eso los intervalos son estrechos. Se publican los dos
   deltas por bloque para que se vea que las dos regiones coinciden en signo y en magnitud.
2. **Un 1,62 % de las celdas está imputado**, cosa que no ocurría en el conjunto primario. Ver
   corrección 2 del preregistro. No afecta a la comparación entre mecanismos, porque todos leen la
   misma posterior.
3. **No es una réplica ciega.** BreizhCrops estaba versionado en el repositorio desde mayo. Es una
   réplica **preespecificada sobre un conjunto conocido**, que es menos, y así se declara.
4. **Submuestreo de 30 000 parcelas por región**, proporcional y con semilla fija, decidido antes
   de ver resultados por un coste de extracción que resultó ser de doce horas y media.

## Qué le da esto al artículo

La contribución deja de depender de un conjunto de datos. La afirmación que el artículo puede
sostener ahora, con dos bancos, dos años, dos regiones, dos familias de modelo y el mismo
protocolo, es ésta: **cuando se reporta F1-macro sobre un catálogo de clases recortado, casi toda
la mejora es el cambio de denominador, no el método**; y **el control que lo revela cuesta una
línea**: puntuar el predictor intacto sobre la misma leyenda.

En el conjunto primario ese control se llevaba el 87 % de la mejora. En el segundo, el 95 %, y en
el tramo donde nada se retira de la entrega, el 100 %.
