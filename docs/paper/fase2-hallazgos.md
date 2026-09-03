# Fase 2 — hallazgos de los experimentos en CPU

**Fase**: 2 de [`docs/plan-micai-2027.md`](../plan-micai-2027.md), bajo el ángulo reencuadrado de [`ADR-013`](../decisions/ADR-013-angulo-micai.md).
**Fecha**: 2 de septiembre de 2026. **Responsable**: Javier A. Rebull-Saucedo.
**Universo**: las 16 640 parcelas del fold 5 held-out de PASTIS-R, contra el ground truth sellado en `reports/paper_micai/fase1/parcel_gt_fold5.parquet`.
**Artefactos**: `reports/paper_micai/fase2/`, sellados en [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md).
**Código**: [`ml/eval/paper_micai_arbitration.py`](../../ml/eval/paper_micai_arbitration.py) y [`scripts/run_paper_micai_fase2.py`](../../scripts/run_paper_micai_fase2.py).

> Todas las cifras de este documento salen de esos artefactos. Ninguna procede del manuscrito heredado ni de los CSV cuyo generador no está versionado.

---

## 1. El número campeón del manuscrito es in-sample para el meta-modelo

Es el hallazgo que condiciona todo lo demás y está reproducido al sexto decimal.

`StackingEnsemble.fit()` hace dos cosas: estima la calidad del meta-modelo con validación cruzada espacial y después **lo reentrena sobre todas las parcelas del fold 5**. `predict_proba()` usa ese segundo modelo. Es decir, el meta-modelo predice las mismas parcelas cuyas etiquetas acaba de ver. Los miembros base sí son honestos —entrenados en los folds 1 a 4—, pero el árbitro no.

| Universo | Refit sobre todo, medido en las mismas parcelas | Bloques espaciales, libre de fuga |
|---|---|---|
| tsvit-pheno + 4 (campeón) | **0.748614** | 0.529596 |
| tsvit-pheno + 2 | **0.747045** | 0.536002 |
| tsvit-pheno-fullm + 4 | **0.647707** | 0.428674 |
| tsvit-pheno-fullm + 2 | **0.635876** | 0.430341 |

La columna de la izquierda reproduce exactamente las cuatro cifras selladas del proyecto: 0,7486 y 0,8495 del `us043_farslip_grid.csv`, 0,7470 del Stacking-3, y 0,6477 y 0,6359 del `us043_farslip_summary.json`.

Dos consecuencias:

1. **Las cuatro cifras pertenecen al mismo régimen**, el in-sample del meta-modelo. La auditoría heredada acertaba al decir que 0,7470 y 0,6477 no se diferencian por régimen sino por miembro, y se equivocaba al llamar *held-out* a ese régimen.
2. **La clave `stacking_5_oof_cv` del JSON está mal nombrada.** 0,6477 no es la validación cruzada del meta-modelo, que en ese universo da 0,4287, sino su reentrenamiento sobre todo.

## 1 bis. El proyecto ya había calculado la cifra honesta y publicó la otra

Corroboración encontrada después de medir, en un artefacto que llevaba meses en el
repositorio: `reports/ensemble/metrics/weighted_voting_pastis.csv` trae las dos columnas
juntas, `f1_macro` y `f1_macro_spatialcv`.

| Modelo | `f1_macro` | `f1_macro_spatialcv` | Combinador |
|---|---|---|---|
| Stacking (meta-LogReg, 54 pesos) | 0,747 | 0,536 | `meta_logreg` |
| Voting ponderado (3 pesos) | 0,7444 | 0,5581 | `weighted_vote_f1max` |
| Blending (Optuna, 3 pesos) | 0,7436 | — | `optuna_simplex` |
| Voting simple (1/N) | 0,673 | 0,4877 | `mean_1_over_N` |

Mi medición independiente del stacking de tres miembros dio 0,747045 y 0,536002. Coincide
con esa tabla al tercer decimal. **La cifra libre de fuga no se descubrió aquí: existía y
estaba guardada al lado de la que se publicó.**

### Y la fila del voto simple permite separar dos efectos que se confundían

El voto simple `1/N` **no ajusta nada**, así que no puede filtrar. Aun así cae de 0,673 a
0,4877 en la columna de validación espacial. Esa caída no es fuga: es el efecto de promediar
macros por bloque, donde un bloque de dos a cuatro mil parcelas rara vez contiene las
dieciocho clases y las ausentes entran como ceros. Es exactamente la razón por la que la
fase 2 agrupa las predicciones y puntúa una sola vez en lugar de promediar macros.

Con eso, el hueco del stacking de tres miembros se descompone:

| Estimador | F1-macro | Qué mide |
|---|---|---|
| Reajuste sobre todas las parcelas | 0,7470 | In-sample para el meta-modelo |
| Agrupado espacial (fase 2) | 0,6789 | Libre de fuga, una sola puntuación |
| Media de macros por bloque | 0,5360 | Libre de fuga, penalizado por la agregación |

La fuga cuesta **0,068** y la forma de agregar cuesta otros **0,143**. Confundir las dos
lleva a decir que el número honesto es 0,536, y no lo es: es 0,679. El primero castiga al
modelo por un artefacto de medición encima de su error real.

Para el artículo esto tiene dos consecuencias. La primera es que el estimador libre de fuga
que se reporta debe ser el agrupado, no la media de macros por bloque. La segunda es que la
corroboración se cita tal cual: el proyecto tenía la columna correcta y arrastró la otra,
que es un modo de fallo mucho más común e instructivo que un error de cálculo.

### El campeón desplegado tiene la misma estructura

El modelo que quedó en producción no es el Stacking-5 sino un **Voting-3 v2** sobre
`france-12` —doce clases, 14 688 parcelas, F1-macro 0,8992 según
`reports/agent_bench/perceiver_champion_eval_v2.json`—. No escapa al problema:
`ml.agent.tools.classify._load_voting_three` aprende sus tres pesos convexos maximizando
F1-macro y los reajusta sobre todas las parcelas del fold 5, igual que el stacking. Su
propia fila en el CSV lo dice: 0,7444 reajustado frente a 0,5581 en validación espacial.

Y su 0,8992 se mide sobre **doce** clases, no dieciocho, así que no es comparable con
ninguna cifra de este documento sin decir eso primero.

### Lo que este hallazgo le regala al artículo

El equipo **retiró seis clases a mano para poder desplegar**, y construyó dos espacios de
etiquetas, `france-9` y `france-12`, para hacerlo. Eso convierte la contribución de
hipotética en documentada: la decisión que el artículo formaliza y mide ya se toma en la
práctica, sin medirla y sin declarar el compromiso. Queda pendiente preguntar al equipo
**qué seis clases y con qué criterio**, porque ese criterio real es el tercer mecanismo que
la fase 3 iba a comparar contra una regla inventada.

## 2. Bajo un protocolo libre de fuga ninguna combinación mejora al mejor miembro

Los diez miembros medidos con el mismo arnés, las mismas parcelas y la misma métrica:

| Miembro | F1-macro | Exactitud |
|---|---|---|
| tsvit-pheno | **0.7367** | 0.8579 |
| xgb-alphaearth | 0.5913 | 0.7811 |
| deeplabv3plus | 0.3546 | 0.6490 |
| segformer | 0.3382 | 0.6474 |
| tsvit-pheno-fullm | 0.2552 | 0.5334 |
| unet | 0.2055 | 0.5541 |
| utae | 0.1880 | 0.5407 |
| anysat | 0.1873 | 0.5710 |
| farslip-ft18 | 0.1730 | 0.3142 |
| farslip-zeroshot | 0.0306 | 0.0948 |

Y las cuatro reglas de combinación sobre los cinco miembros del campeón:

| Regla | Régimen | F1-macro | Exactitud |
|---|---|---|---|
| Árbitro entrenado, refit sobre todo | in-sample | 0.7486 | 0.8494 |
| Voto ponderado, pesos del propio conjunto | in-sample | 0.7235 | 0.8541 |
| Voto ponderado, pesos de otros bloques | held-out | 0.7231 | 0.8539 |
| Árbitro entrenado, bloques espaciales | held-out | 0.6794 | 0.8466 |
| Promedio simple | held-out | 0.6790 | 0.8365 |

Con bootstrap pareado por parcela (B = 1000, semilla 42) y McNemar exacto:

| Comparación | Delta F1-macro | IC 95 % | ¿Excluye el cero? | McNemar |
|---|---|---|---|---|
| Árbitro agrupado − promedio simple | +0.0005 | [−0.0086, +0.0097] | no | p = 3.4e−06 |
| Árbitro agrupado − voto ponderado | −0.0437 | [−0.0506, −0.0365] | sí | p = 4.3e−05 |
| Árbitro agrupado − mejor individual | −0.0572 | [−0.0644, −0.0499] | sí | p = 1.4e−09 |
| Promedio simple − mejor individual | −0.0577 | [−0.0680, −0.0484] | sí | p = 7.6e−24 |

Lectura honesta, aplicando la **regla R3** de ADR-013: el árbitro entrenado **no** gana al promedio homogéneo —la diferencia es de cinco diezmilésimas y su intervalo cruza el cero— y **pierde** frente a un voto ponderado global y frente a un solo TSViT-pheno. La pata del arbitraje baja a discusión, tal como la regla preveía.

Dos matices que no se pueden omitir:

- **El meta-modelo se entrena con muy poco.** Los volcados OOF solo existen para el fold 5, así que el árbitro solo puede entrenarse con bloques del propio fold 5 y generalizar a otro bloque geográfico. Un stacking bien alimentado usaría predicciones OOF de los folds 1 a 4, que no existen como artefacto. Esto mide *el stacking que se puede validar con los artefactos que hay*, no que el stacking sea imposible.
- **La fuga que introduje y corregí.** La primera versión del voto ponderado tomaba los pesos del F1 medido sobre las mismas parcelas que puntuaba. Corregido a pesos estimados en los otros bloques, la diferencia resultó mínima (0,7235 frente a 0,7231), pero la corrección era obligatoria y queda medida.

## 3. El árbitro sí arbitra por clase, y además retira una clase en silencio

`arbitraje_por_clase.csv` explica el agregado. El árbitro mejora al mejor individual en las clases 0, 11, 13, 15, 4, 9 y 12, varias de ellas minoritarias. Pero:

| Clase | Soporte | F1 mejor individual | F1 árbitro agrupado |
|---|---|---|---|
| 10 | 355 | 0.7790 | **0.0000** |
| 16 | 193 | 0.4949 | 0.2709 |
| 5 | 198 | 0.6667 | 0.5228 |

En la clase 10 el árbitro **no emite ni una sola predicción** en los bloques que no vio, y el F1-macro castiga eso con toda su fuerza. El mecanismo es el que da sentido al artículo: **un árbitro entrenado acaba retirando clases por su cuenta, sin declararlo y sin que nadie elija el punto de operación.** La contribución del artículo es hacer esa decisión explícita y medible en lugar de dejarla implícita.

## 4. Contribución central: retirar clases domina al rechazo por confianza

Aplicando la **regla R2** de ADR-013, la curva se compara contra el baseline obligatorio. Cada bloque espacial se evalúa contra **una** leyenda, elegida con los otros bloques; el umbral de confianza se fija igual, en los otros bloques, para igualar la cobertura.

| K | Cobertura, idéntica en ambos | F1-macro retirada | F1-macro confianza | Delta | IC 95 % | ¿Excluye el cero? |
|---|---|---|---|---|---|---|
| 18 | 1.000 | 0.5612 | 0.5612 | +0.0000 | [−0.0294, +0.0308] | no |
| 16 | 0.942 | 0.5866 | 0.5863 | +0.0003 | [−0.0364, +0.0287] | no |
| 14 | 0.890 | 0.6236 | 0.6160 | +0.0076 | [−0.0265, +0.0437] | no |
| 12 | 0.865 | 0.6715 | 0.6210 | **+0.0504** | [+0.0092, +0.0909] | sí |
| 10 | 0.839 | 0.7694 | 0.6271 | **+0.1423** | [+0.0934, +0.1790] | sí |
| 9 | 0.829 | 0.7783 | 0.6282 | **+0.1501** | [+0.1081, +0.1907] | sí |
| 8 | 0.810 | 0.8243 | 0.6298 | **+0.1944** | [+0.1428, +0.2304] | sí |

La fila de K = 18 es la comprobación de cordura: a cobertura completa los dos mecanismos
son el mismo objeto y el delta sale exactamente cero. Estas cifras son las de la corrección
descrita en la auditoría del 2 de septiembre; la versión anterior comparaba a coberturas que
no coincidían.

**La contribución central se sostiene.** A igual cobertura, recortar la leyenda compra más calidad macro que rechazar parcelas por confianza, y la ventaja crece según se acorta la leyenda: por debajo de doce clases el intervalo excluye el cero.

Tres advertencias que van al artículo con el resultado:

1. **Los dos mecanismos entregan productos distintos.** Uno promete una leyenda más corta en todas sus parcelas; el otro, la leyenda completa en menos parcelas. La comparación es de la frontera calidad-cobertura, no una prueba de equivalencia.
2. **La leyenda no es estable entre regiones.** A K = 12 solo nueve de las catorce clases retenidas aparecen en las cinco leyendas; a K = 16, once de dieciocho. Qué clases retirar depende de la región, y eso es parte del resultado, no una nota al pie.
3. **El F1-macro por bloque es más bajo que el agregado** (0,5612 frente a 0,7367 a K = 18) porque un bloque de dos a cuatro mil parcelas castiga más a las clases raras. Las dos cifras responden preguntas distintas: calidad media por región frente a calidad sobre el mapa completo. La comparación entre mecanismos se hace *dentro* de cada bloque, así que está pareada.

## 5. El nulo de vecindad es un nulo, y más limpio que el sellado

Aplicando la **regla R1** de ADR-013. El punto de operación `(k, alfa)` se elige en los bloques que no se miden, porque escogerlo mirando el resultado sería quedarse con el máximo del ruido.

| Base | F1-macro base | F1-macro refinado | Delta | IC 95 % | ¿Excluye el cero? | McNemar |
|---|---|---|---|---|---|---|
| Mejor individual (tsvit-pheno) | 0.7367 | 0.7360 | −0.0007 | [−0.0016, +0.0002] | no | p = 0.66 |
| Árbitro agrupado | 0.6794 | 0.6794 | +0.0000 | [+0.0000, +0.0000] | no | p = 1 |

El barrido completo es aún más claro: sobre el mejor individual, **ningún** punto con `alfa > 0` mejora al `alfa = 0`, y la degradación crece con `alfa` (0,7367 a `alfa = 0`, 0,6973 a `alfa = 0,5` con k = 5). Sobre el árbitro agrupado, los cinco bloques eligieron `alfa = 0` y el refinamiento queda en no-operación.

Frente a esto, el artefacto sellado `ec_neighborhood_result.json` daba un delta **positivo** de +0,0027 a +0,0068. La diferencia se explica sola: aquel barrido se aplicó sobre la posterior in-sample del campeón, y suavizar hacia los vecinos recupera parte de lo que un modelo sobreajustado pierde. Sobre un predictor libre de fuga no hay nada que recuperar.

Enunciado que el dato sostiene: **el refinamiento por vecindad entre parcelas no aporta sobre un predictor libre de fuga, y su intervalo incluye el cero.** No dice nada sobre el contexto intraparcela, que la literatura sí muestra útil sobre estos embeddings.

## 6. Las dos ramas FarSLIP no aportan nada medible

Medido en un solo universo, el del campeón sellado, con ambos ensambles estimados por el mismo agrupado espacial:

| Variante | Régimen | F1-macro | Exactitud |
|---|---|---|---|
| Tres miembros | held-out | 0.6789 | 0.8456 |
| Cinco miembros | held-out | 0.6794 | 0.8466 |
| Tres miembros | in-sample | 0.7470 | 0.8490 |
| Cinco miembros | in-sample | 0.7486 | 0.8495 |

Delta de cinco frente a tres, libre de fuga: **+0.0006**, IC 95 % [−0.0024, +0.0034], McNemar p = 0.18. El intervalo incluye el cero.

Las dos filas in-sample reproducen exactamente las cifras selladas y su delta de +0,0016. Es decir, el aporte de FarSLIP que el manuscrito reporta vive entero dentro del régimen in-sample; bajo un protocolo libre de fuga no se distingue de cero.

## 7. Un error propio, corregido y documentado

La primera implementación de la retirada de clases dejaba el argmax sobre las dieciocho columnas y promediaba el macro sobre la unión de las leyendas de todos los bloques. Con eso, retirar clases parecía **empeorar** el F1 de forma monótona y el rechazo por confianza dominaba en todos los puntos. Las dos cosas estaban mal: una leyenda que no se promete es una leyenda que el modelo no emite, y promediar sobre la unión de leyendas que discrepan puntúa un producto que nadie desplegaría. Corregido, el resultado se invierte y es el de la sección 4.

## 8. Preguntas abiertas que no se resuelven en local

- **La procedencia de `tsvit-pheno`.** Saca 0,7367 de F1-macro mientras `tsvit-pheno-fullm`, otra variante de la misma arquitectura, saca 0,2552 sobre las mismas parcelas. Una diferencia de 0,48 entre dos variantes obliga a comprobar con qué folds se entrenó cada checkpoint antes de publicar nada que dependa de ese miembro. El checkpoint y su registro de entrenamiento están en la VM H100. **Depende de Arthur.**
- **Volcados OOF de los folds 1 a 4.** Sin ellos el meta-modelo no puede entrenarse como es debido y la sección 2 no puede cerrarse en positivo ni en negativo.

## 9. Estado de la fase 2

Los cinco puntos del plan están hechos: individuales bajo un protocolo único, homogéneo
frente a heterogéneo con pruebas pareadas, curva calidad-cobertura contra su comparador,
aporte de FarSLIP con intervalo y nulo de vecindad con intervalo. Lo que queda depende de
Arthur: la procedencia de `tsvit-pheno` y los volcados OOF de los folds 1 a 4.

## 10. Qué le queda al artículo, en una frase

De las tres patas del ángulo A, la que sobrevive es la del punto de operación: **a igual
cobertura, retirar clases de la leyenda compra más calidad macro que rechazar parcelas por
confianza**, y el mecanismo se entiende porque un árbitro entrenado ya hace esa retirada
por su cuenta, en silencio y sin que nadie la elija.
