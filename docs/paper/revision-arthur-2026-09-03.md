# Revisión de Arthur, verificada punto por punto

**Documentos revisados**: `evaluacion-cuaderno-micai-2027.md` y `rutas-micai-2027-post-hallazgos-2026-09-03.md`, ambos del 3 de septiembre de 2026.
**Verificado contra**: este fork (`jrebull/agrosat-copilotv2`), no contra el `main` del que parte Arthur.
**Criterio**: cada punto se marca CONFIRMADO (lo comprobé y tiene razón), NO APLICA AQUÍ (es cierto en su árbol y no en este), o MATIZ (tiene razón en parte).

## 0. Lo primero, porque explica media lista

Arthur revisa contra `main @ 471d64a`. **Este trabajo vive en el fork**, y por eso su punto 0.1 —«ninguno de estos artefactos existe en `main`»— es cierto y no es un error suyo: es el problema de las dos fuentes de verdad que él mismo señala (área J, US-122). Los artefactos existen, con su MD5, en `jrebull/agrosat-copilotv2`. No hay nada que corregir en su observación salvo el destino.

## 1. CONFIRMADO y grave: la premisa del criterio de retirada es falsa, y está impresa en el artículo

Es el hallazgo más importante de su revisión y es contra mí.

El artículo afirma, en el abstract, en la sección 3.5, en la 4.4 y en la discusión, que retirar clases **por soporte bajo** es «what practitioners report doing». Lo comprobé en los artefactos de este repositorio y **no es cierto**:

`reports/voting_new/cardinalidad.json` da el orden real en que las clases entran al catálogo:

| k | clase añadida | soporte |
|---|---|---|
| 4 | Winter rapeseed | 387 |
| 5 | Grapevine | 2 067 |
| 6 | Beet | 167 |
| 7 | Soybeans | 209 |
| 8 | Winter barley | 573 |
| 12 | Orchard | 511 |
| 17 | Leguminous fodder | **660** |

Si el criterio fuera el soporte, el orden sería monótono decreciente en soporte. No lo es: la colza entra cuarta con 387 parcelas por delante de la vid con 2 067, y el forraje leguminoso, con 660, se retira el penúltimo. **El orden es por F1 por clase**, y `f1_ge_09_classes: 12` fija el umbral en 0,90.

Y hay más: `reports/ensemble/metrics/us043_winner_cardinality_curve.csv` trae una columna `cumulative_support_share` que vale **0,9054** con doce clases. **La cobertura sí se midió.** La frase «tomó la primera vía sin medirla» es falsa; lo que no se midió es la comparación contra la alternativa.

**Consecuencia para el artículo, y no es cosmética.** El criterio que el equipo usó de verdad es la retirada por F1, que en mis propios resultados es **la mejor** de las dos retiradas, no la peor. El artículo dice ahora mismo lo contrario de lo que la evidencia sostiene sobre este equipo. Lo que sí puedo sostener es que **elegir por soporte es marcadamente peor que elegir por calidad medida**, y que este equipo eligió lo segundo.

## 2. CONFIRMADO: el volcado de `tsvit-pheno-fullm` está roto también en este fork

`ml/eval/oof/dump_oof.py` **no menciona `n_timesteps` en ninguna línea** de este árbol. El único arreglo relacionado que hay commiteado es `d43412a`, anterior y distinto. El arreglo de Arthur vive sin commitear en su árbol, así que nunca llegó aquí.

Consecuencias, medidas:

- La fase 3 ordenó los diez miembros y tomó los dos primeros. Con `fullm` en 0,2552 quedó fuera; con 0,7883 sería **el primero**. La frase del artículo «we take the two strongest of ten available members» **es falsa**, y un revisor con el dato la tumba.
- **Lo que NO está contaminado**: el árbitro de la sección 4.5. Sus miembros son `tsvit-pheno, utae, xgb-alphaearth, farslip-ft18, farslip-zeroshot`, comprobado en `reports/paper_micai/fase2/arbitraje_pruebas.json`. `fullm` no está entre ellos, así que el cero de *Winter durum wheat* no es un artefacto del bug. Esto lo verifiqué antes de creerlo.
- **CORREGIDO el 3 de septiembre, y la corrección es contra mí.** Escribí que este fork no tiene
  `checkpoints/` y que por eso el re-volcado era imposible. **Es falso.** Existe
  `checkpoints/segmentation.dvc` —2,84 GB, 37 ficheros— y los pesos están en el remoto: verificado
  blob a blob, `tsvit-pheno-fullm-v1/best.pt` son 144 951 653 bytes en GCS y `fullm-v2` otros
  144 940 133. Están a un `dvc pull` de distancia. Repetí la afirmación sin comprobar el puntero
  DVC, que es exactamente lo que este documento reprocha en otros sitios. **El arreglo de `fullm`
  es posible aquí**, y con él la frase «los dos miembros más fuertes de diez» pasa de limitación a
  declarar a error a corregir.
- **Lo que sí está perdido de verdad** son otros tres, y son de FarSLIP: `parcel/18cls`,
  `parcel/04cls` e `incremental/08cls`. Ni fichero, ni puntero, ni blob. Cuatro scripts los
  referencian como valor por defecto, y `farslip-ft18` es uno de los cinco miembros del árbitro.

## 3. CONFIRMADO: el párrafo de las cinco peores IoU del sitio es falso

`docs/serving/copiloto-v2-12clases.md` lista las seis retiradas: Winter triticale, Fruits/veg/flowers, Potatoes, Leguminous fodder, Mixed cereal, Sorghum. De las cinco peores por IoU que el sitio nombra, **solo dos** están ahí. Orchard, Beet y Winter durum wheat se conservaron.

## 4. CONFIRMADO con matiz: el 0,6789 no «ya existía en la columna de al lado»

La columna existe y es `f1_macro_spatialcv` en `reports/ensemble/metrics/weighted_voting_pastis.csv`, pero vale **0,536** para el Stacking. El 0,6789 es una re-derivación mía, agrupando posteriores en vez de promediar macros por bloque.

El artículo lo presenta correctamente, como cálculo propio. **El sitio no**: dice que la cifra libre de fuga «ya existía guardada en la columna de al lado», y lo que existía es otra cifra. Hay que decir lo exacto: existía una columna libre de fuga, con un estimador distinto y peor, y el proyecto publicó la contaminada teniendo esa al lado.

## 5. CONFIRMADO: divulgación de asistencia de IA

Los cuatro auditores de la fase 2 ter fueron agentes, no personas, y tanto el código como el texto se produjeron con asistencia de IA. El artículo **no lo declara**, y Springer lo exige. Es un incumplimiento formal, no una cuestión de estilo.

## 6. CONFIRMADO: el bloque espacial hay que nombrarlo con precisión

La sección 3.3 dice «cinco bloques» sin más. Hay que decir que el fold 5 es la partición externa oficial de PASTIS y que los cinco bloques son **sub-folds internos** construidos con `build_spatial_kfold` sobre ese fold. Tal como está, se puede leer como validación cruzada de cinco folds del banco, que no es lo que se hizo.

## 7. MATIZ: U-TAE, el producto desplegado y el resto

- **U-TAE 0,19 frente a 0,63**: compara F1-macro por parcela con mIoU por píxel. Es una comparación inválida y está en el sitio, no en el artículo. El artículo no la hace, así que solo hay que corregir el sitio.
- **El producto desplegado ausente**: con el encuadre anterior era una omisión grave. Con el encuadre actual el artículo ya no trata del sistema, así que deja de ser fatal; pero la sección 4.5 gana si dice qué se desplegó y con qué cifra.
- **Consentimiento y crédito**: Isaac y Aaron ya pasaron a agradecimientos sin atribuirles autoría del código. Falta el patrocinador y falta que alguien les pida permiso de verdad, que no es algo que yo pueda hacer.

## 8. Lo que Arthur da por pendiente y ya está hecho en este fork

No para discutirle nada, sino para que no lo rehaga: ADR-013 existe (`docs/decisions/ADR-013-angulo-micai.md`), el esqueleto LNCS existe y compila, el preregistro con sus dos enmiendas existe, BreizhCrops está replicado de punta a punta, el gate de artefactos y el de doble ciego existen y están probados en negativo, y la compilación es reproducible byte a byte. Su Ruta 6 y buena parte de la 3 están cerradas.

Lo que sigue abierto de su lista y no depende de él: la corrección de la premisa (punto 1), la declaración del volcado roto (punto 2), la divulgación de IA (punto 5) y la precisión del bloque (punto 6).
