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

| K | Cobertura media (retirada / confianza) | F1-macro retirada | F1-macro confianza | Delta | IC 95 % | ¿Excluye el cero? |
|---|---|---|---|---|---|---|
| 18 | 1.000 / 1.000 | 0.5612 | 0.5613 | −0.0001 | [−0.0293, +0.0303] | no |
| 16 | 0.942 / 0.901 | 0.5866 | 0.5918 | −0.0052 | [−0.0407, +0.0269] | no |
| 14 | 0.890 / 0.857 | 0.6236 | 0.6101 | +0.0135 | [−0.0260, +0.0463] | no |
| 12 | 0.865 / 0.839 | 0.6715 | 0.6098 | **+0.0616** | [+0.0130, +0.0935] | sí |
| 10 | 0.839 / 0.820 | 0.7694 | 0.6139 | **+0.1554** | [+0.1047, +0.1911] | sí |
| 9 | 0.829 / 0.813 | 0.7783 | 0.6241 | **+0.1542** | [+0.1067, +0.1971] | sí |
| 8 | 0.810 / 0.802 | 0.8243 | 0.6268 | **+0.1975** | [+0.1441, +0.2284] | sí |

**La contribución central se sostiene.** A igual cobertura, recortar la leyenda compra más calidad macro que rechazar parcelas por confianza, y la ventaja crece según se acorta la leyenda: por debajo de doce clases el intervalo excluye el cero.

Tres advertencias que van al artículo con el resultado:

1. **Los dos mecanismos entregan productos distintos.** Uno promete una leyenda más corta en todas sus parcelas; el otro, la leyenda completa en menos parcelas. La comparación es de la frontera calidad-cobertura, no una prueba de equivalencia.
2. **La leyenda no es estable entre regiones.** A K = 12 solo nueve de las catorce clases retenidas aparecen en las cinco leyendas; a K = 16, once de dieciocho. Qué clases retirar depende de la región, y eso es parte del resultado, no una nota al pie.
3. **El F1-macro por bloque es más bajo que el agregado** (0,5612 frente a 0,7367 a K = 18) porque un bloque de dos a cuatro mil parcelas castiga más a las clases raras. Las dos cifras responden preguntas distintas: calidad media por región frente a calidad sobre el mapa completo. La comparación entre mecanismos se hace *dentro* de cada bloque, así que está pareada.

## 5. Un error propio, corregido y documentado

La primera implementación de la retirada de clases dejaba el argmax sobre las dieciocho columnas y promediaba el macro sobre la unión de las leyendas de todos los bloques. Con eso, retirar clases parecía **empeorar** el F1 de forma monótona y el rechazo por confianza dominaba en todos los puntos. Las dos cosas estaban mal: una leyenda que no se promete es una leyenda que el modelo no emite, y promediar sobre la unión de leyendas que discrepan puntúa un producto que nadie desplegaría. Corregido, el resultado se invierte y es el de la sección 4.

## 6. Preguntas abiertas que no se resuelven en local

- **La procedencia de `tsvit-pheno`.** Saca 0,7367 de F1-macro mientras `tsvit-pheno-fullm`, otra variante de la misma arquitectura, saca 0,2552 sobre las mismas parcelas. Una diferencia de 0,48 entre dos variantes obliga a comprobar con qué folds se entrenó cada checkpoint antes de publicar nada que dependa de ese miembro. El checkpoint y su registro de entrenamiento están en la VM H100. **Depende de Arthur.**
- **Volcados OOF de los folds 1 a 4.** Sin ellos el meta-modelo no puede entrenarse como es debido y la sección 2 no puede cerrarse en positivo ni en negativo.

## 7. Qué queda de la fase 2

- Aporte de FarSLIP, cinco frente a tres miembros, en un solo universo y con intervalo.
- Nulo de vecindad con intervalo pareado, sobre el mejor predictor libre de fuga (regla R1 de ADR-013).
