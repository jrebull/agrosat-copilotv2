# Los artefactos perdidos, uno por uno

**Fecha**: 3 de septiembre de 2026. **Origen**: investigación forense a ciegas, verificada por mí en lo comprobable.

## Corrección previa, y es contra mí

Dije que se habían perdido **tres** checkpoints de FarSLIP. La realidad es más matizada y peor
contada:

| Artefacto | Veredicto | Por qué |
|---|---|---|
| `parcel/04cls` | **REGENERABLE, con receta verificada** | Existe el entrenador, existen los datos, y la receta se recuperó y se comprobó |
| `parcel/18cls` | **Regenerable reinventando la receta, y no vale la pena** | N=18 **nunca entró al barrido**: no hay ninguna cifra contra la cual validarlo |
| `incremental/08cls` | **Nunca existió** | Cinco evidencias convergentes. Decir que se perdió es un error de registro |

Y una corrección menor: no son cuatro los scripts que los referencian por defecto, son **tres**.
El cuarto, `farslip_eval_phenology.py:56`, lo nombra en un ejemplo del docstring; su
`--checkpoint-path` es obligatorio y sin valor por defecto.

## `parcel/04cls`: la receta se recuperó y se verificó

El default del barrido apuntaba a `parcel_phenology_captions.parquet`, que no está ni en disco ni en
DVC. Lo que sí está versionado es `parcel_phenology_captions_69k.parquet`, con 69 297 parcelas.

La reconstrucción es determinista y **se comprobó contra dos puntos del barrido**: submuestrear
1 000 filas por clase, ordenando por `(int(patch_id), parcel_id)`, da exactamente **16 012 filas**,
que es el tamaño que la documentación atribuye al parquet perdido. Con ese conjunto:

| | reconstruido | `parcel_sweep.csv` dice |
|---|---|---|
| N = 4, fold 4, con caption y área ≥ 16 px | 1 301 | **1 301** |
| N = 12, ídem | 3 200 | **3 200** |

Coincidencia exacta en los dos extremos de la curva. El criterio de aceptación al regenerarlo es
**macro-F1 ≈ 0,7025 sobre exactamente 1 301 parcelas**.

> **Ojo con el criterio que circula.** La evaluación externa propone reentrenar si se reproduce
> «~0,6452 macro-F1». Esa cifra **no existe en ningún artefacto del repositorio**: la busqué en CSV,
> JSON, parquet y markdown y solo aparece dentro de `metadata.geojson` de PASTIS-R, donde es una
> coordenada. El criterio correcto es **0,7025**.

Coste medido, no estimado: unas **1 h de GPU** para N=4. Dos obstáculos concretos en este Mac: el
barrido construye su configuración sin pasar `device`, y el resolutor solo mapea a `cuda` o `cpu`,
nunca a `mps`; y `batch_size=256` con ViT-B/16 a 224² no cabe en 38,6 GB de memoria unificada.
Bajarlo cambiaría la composición de negativos del contraste, así que **dejaría de ser la misma
receta**: para reproducción fiel hace falta una GPU de 40 GB o más.

## `parcel/18cls`: se puede correr, pero no se puede validar

Se puede reutilizar la misma receta con `--n-values 18`, unas 4 h. Pero **N=18 no está en
`parcel_sweep.csv`** —el barrido fue 4, 6, 8, 10 y 12—, no hay corrida en MLflow, ni log, ni
reporte. Un checkpoint nuevo sin criterio de aceptación **no restaura la reproducibilidad, la
simula**. Declarar la pérdida es más honesto que producir algo que se *parece* al original.

## `incremental/08cls`: nunca existió

Cinco evidencias convergentes: solo hay puntero DVC de `incremental/04cls`; 06cls y 08cls no
aparecen en ningún reporte, métrica, figura ni documento; el resumen de la corrida dice literalmente
que fueron «solo 4 clases fáciles», que es lo que dispara el criterio de parada
`new_classes_unacceptable` en el escalón 06; la historia se cerró con otro checkpoint; y el propio
docstring que lo nombra dice «**una vez que exista** el checkpoint ganador».

**No se declara perdido: se corrige el docstring** para que apunte a `incremental/04cls`, que sí está
en DVC, y se anota que el currículum paró en el escalón 04.

## MLflow no tiene nada, y se puede probar

El servidor de `:5010` guarda en Postgres. La base `mlflow` **tiene cero tablas**, y su volumen se
creó el 2 de septiembre de 2026, así que no hay historial anterior. Además, la máquina virtual donde
se entrenó **no podía correr Docker** —virtualización anidada desactivada—, de modo que esos
entrenamientos nunca se registraron: sus métricas vivían en ficheros de log que tampoco están.

La tarjeta del modelo promete etiquetas de MLflow que no existen. Eso hay que corregirlo.

## Qué se pierde de verdad si no se regeneran

**Menos de lo que parece.** Todo lo que consume esos checkpoints ya está materializado aguas abajo y
versionado: los volcados OOF de `farslip-ft18` y `farslip-zeroshot` están en disco y en DVC, con
16 475 predicciones reales de 16 640. De ahí salen las cifras del Stacking-5. **Regenerar los
checkpoints no cambia ninguna cifra**: restaura la cadena que va del dato crudo al volcado.

Y `parcel/04cls` alimenta un experimento cuyo resultado documentado es **negativo** (F1-macro
0,2694). Regenerarlo reproduce un resultado negativo ya publicado como tal.

## Un hallazgo colateral que sí importa

`data/features/alphaearth_italia_2018.parquet` tiene su puntero `.dvc` en git, **pero el blob nunca
se subió al bucket**. La evaluación externa lo daba por versionado; el puntero lo está, el dato no.
Es la peor de las dos situaciones, porque parece resuelto.

El inventario completo son **56 rutas por defecto que apuntan a ficheros inexistentes**, en tres
grupos: recuperables con `dvc pull`, genuinamente ausentes, y literales de prueba que nunca se
abren. Los genuinamente ausentes incluyen toda la sección DE4 del manuscrito heredado, de la que
dependen unas treinta y cinco cifras.

## Recomendación

**Regenerar `parcel/04cls`. Declarar irreproducible `parcel/18cls`. Borrar `incremental/08cls` del
código.** Y en los tres casos, arreglar los valores por defecto para que apunten a algo
materializable o fallen con un mensaje explícito, en vez de fallar con un error de fichero sobre una
ruta que ya nadie puede satisfacer.
