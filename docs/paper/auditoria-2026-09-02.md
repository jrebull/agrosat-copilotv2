# Auditoría completa del estado del artículo — 2 de septiembre de 2026

**Alcance**: todo lo producido en las fases 0, 1 y 2, incluido **mi propio código**, que es
de donde cuelga ahora la contribución del artículo.
**Modo del skill**: 2 (integridad científica) más el modificador de patrones de incidente.
**Regla que la gobierna**: un control que nunca se ha visto fallar no se sabe si funciona, y
una auditoría que solo mira el trabajo ajeno no es una auditoría.

Resumen: **dos hallazgos bloqueantes, uno de ellos mío y ya corregido**; tres importantes;
tres controles que pasan. Ninguna cifra de este documento viene de otro documento: todas se
recalcularon contra los artefactos.

---

## B1 · La cobertura que llamaba «igualada» no lo estaba. Error mío, corregido

**Severidad: bloqueante. Estado: corregido y reejecutado.**

La contribución central compara dos mecanismos «a igual cobertura». Al medir el desajuste
real entre las coberturas que cada mecanismo entregaba, salió **0,064 de media y 0,261 en el
peor bloque**: en el bloque 2 la retirada de clases entregaba el 71,5 % de las parcelas y el
rechazo por confianza solo el 45,3 %. Eso no es una comparación a igual cobertura, es una
comparación a la cobertura que salió.

**Causa.** El umbral de confianza se tomaba como cuantil de los **otros** bloques y se
aplicaba al bloque medido. Parecía lo prudente —no tocar los datos que se miden— y era
justo lo incorrecto: la distribución de confianza cambia de bloque a bloque, así que el
mismo umbral entrega fracciones muy distintas en cada uno.

**Arreglo.** El umbral pasa a ser el cuantil del **propio** bloque que entrega exactamente
el número de parcelas que entregó el otro mecanismo allí. **No es fuga**: acertar una
cobertura objetivo lee posteriores, nunca etiquetas, y cualquier despliegue puede ordenar
sus parcelas por confianza y responder el 80 % más confiado sin conocer una sola verdad.

**Consecuencia.** El desajuste pasa a ser **exactamente cero** y la conclusión sobrevive con
cifras ligeramente distintas: el delta a K = 12 baja de +0,0616 a +0,0504 y a K = 8 de
+0,1975 a +0,1944. La fila de K = 18 pasa a dar delta exactamente cero, que es la
comprobación de cordura que antes no cuadraba (daba −0,0001).

**Lección.** «Más prudente» y «correcto» no son sinónimos. La prudencia mal colocada rompió
el emparejamiento que era la razón de ser del experimento.

## B2 · `tsvit-pheno-fullm` da 0,7918 a nivel píxel y 0,2552 a nivel parcela

**Severidad: bloqueante. Estado: abierto, requiere a Arthur.**

| Fuente | Métrica | Valor |
|---|---|---|
| `reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv` | F1-macro píxel, fold 5 | **0,7918** |
| `ml/eval/oof/oof_parcel_tsvit-pheno-fullm_fold5.parquet` | F1-macro parcela, fold 5 | **0,2552** |

Para comparar, el mismo par en `tsvit-pheno` **sí concuerda**: 0,7401 a nivel píxel
(`model_comparison_fold5.csv`) y 0,7367 a nivel parcela. Una diferencia de 0,004 frente a
una de 0,54.

**Descartado**: no es un desplazamiento de clases. Probé el volcado con las predicciones
corridas ±1 y empeora a 0,011 y 0,010, y `pred_class` coincide con el `argmax` en las 16 640
filas. El volcado es internamente consistente; lo que no cuadra es con su propia evaluación
densa.

**Por qué importa.** Ese volcado es la base del universo «fullm» de la rejilla sellada
`us043_farslip_grid.csv`, donde aparece etiquetado como «mejor individual». **No lo es a
nivel de parcela: es el peor de los diez.** Y eso resuelve de paso una pregunta que el plan
llevaba abierta —por qué el mejor miembro individual produce el peor stacking—: la etiqueta
«mejor individual» venía de un número de píxel de otra evaluación, no del volcado que
alimenta el ensamble.

**Qué hacer.** Hasta que se explique, la rama «fullm» de la rejilla no se cita. Pregunta
concreta para Arthur: con qué checkpoint y con qué reconciliación píxel-parcela se generó
`oof_parcel_tsvit-pheno-fullm_fold5.parquet`.

## I1 · Tres cifras distintas de F1-macro píxel para variantes de TSViT en fold 5

**Severidad: importante.**

`model_comparison_fold5.csv` da 0,7401 para `tsvit-pheno`. `tsvit_pheno_vs_base_fold5.csv`
da 0,7942 para `tsvit-base-fullm` y 0,7918 para `tsvit-pheno-fullm`. Son tres números para
tres variantes, en dos ficheros, y el propio `export_avance4_metrics_us025.py` advierte que
las comparativas mezclan configuraciones «18 vs 20 clases, 128 vs 256 px, 10 vs 3 bandas».
Cualquier tabla del artículo que compare variantes de TSViT tiene que declarar de cuál de
las dos evaluaciones sale cada fila, o no compararlas.

## I2 · El criterio real con el que se retiraron seis clases, en palabras del equipo

**Severidad: importante, y a favor.**

Preguntado por qué `france-12` tiene doce clases y no dieciocho, el equipo responde: *«fue
porque se tenía muy poca muestra»* y *«bajaban mucho el F1 macro»*. Es decir, el criterio
real fue **soporte bajo más daño al macro**, exactamente los dos ejes que el artículo
formaliza. Deja de ser una regla que yo invento para la fase 3: es el criterio que un equipo
usó de verdad para poder desplegar. Va al artículo citado como práctica documentada.

## I3 · El buffer espacial de 1 km no excluye ninguna parcela

**Severidad: importante, con matiz que lo desactiva.**

`build_spatial_kfold` reporta `excluded=0`: el buffer de 1 km no aparta ni una parcela, lo
que a primera vista significa que la separación entre bloques es nominal. Lo comprobé por
otra vía y el resultado tranquiliza: **ningún parche de PASTIS se reparte entre bloques**,
los 496 caen enteros en uno solo. Como las parcelas de un parche están a menos de 1,3 km
entre sí y comparten la misma imagen, esa es la unidad de contaminación relevante y está
respetada. El buffer no hace nada porque no tiene nada que hacer.

## C1 · Los bloques espaciales son disjuntos y cubren todo — pasa

Cinco bloques, cero intersección entre entrenamiento y prueba en los cinco, y **las 16 640
parcelas evaluadas exactamente una vez**: ninguna se queda sin evaluar y ninguna se evalúa
dos veces. Tamaños de 1 313 a 4 533, desiguales por geografía, que es lo esperado con
teselación H3.

## C2 · Los volcados por parcela concuerdan con `pred_class` — pasa

En los diez miembros, la clase de máxima probabilidad coincide con la columna `pred_class`
almacenada. No hay desalineación entre lo que el volcado dice haber predicho y lo que sus
probabilidades implican.

## C3 · El ledger detecta un cambio y distingue lo que falta por `dvc pull` — pasa

Probado en negativo dos veces: alterar un byte de un CSV sellado pone el gate en rojo, y
restaurarlo lo devuelve a verde. Sobre un clon limpio, 23 de los 37 artefactos de entonces
verificaban solo con `git clone` y los demás se reportan como pendientes de `dvc pull`, no
como sello roto.

---

## Lo que esta auditoría cambia en las cifras publicables

| Cifra | Antes | Ahora | Motivo |
|---|---|---|---|
| Delta a K = 12 | +0,0616 | **+0,0504** | B1 |
| Delta a K = 10 | +0,1554 | **+0,1423** | B1 |
| Delta a K = 9 (criterio principal) | +0,1542 | **+0,1501** | B1 |
| Delta a K = 8 | +0,1975 | **+0,1944** | B1 |
| Delta a K = 18 (cordura) | −0,0001 | **+0,0000** | B1 |
| «Mejor individual» del universo fullm | tsvit-pheno-fullm | **no se cita** | B2 |

El criterio principal preregistrado, K = round(18/2) = 9, da **+0,1501 con intervalo
[+0,1081, +0,1907]**, que excluye el cero. La contribución central se sostiene después de
la auditoría, con la cobertura ahora exactamente igualada.

## Deuda que queda abierta

| Id | Qué | Quién |
|---|---|---|
| B2 | La inconsistencia píxel-parcela de `tsvit-pheno-fullm` | Arthur |
| — | La procedencia de folds de `tsvit-pheno` | Arthur, o se resuelve reentrenando |
| — | Tres checkpoints de FarSLIP sin versionar | Arthur |
| — | Seis artefactos sellados sin guion generador | nosotros, en la fase 3 |
| — | Validez externa: un solo conjunto de datos | nosotros, en la fase 4 |
