# Preregistro v2 — borrador para firma

**Estado**: BORRADOR. Tras la auditoría externa, **tres parámetros vuelven a estar abiertos**: la banda de equivalencia, la función de pérdida y el estimando con su población. No se firma hasta cerrarlos. No vale hasta que esté commiteado y firmado. **Nada de las EPIC 20, 21, 22 ni 25 se computa antes de ese commit.**

Este documento existe porque el preregistro anterior se citó como credencial mientras se omitía que su hipótesis se había refutado y que la regla de entrega había cambiado. Aquí se declara todo antes, incluidos los grados de libertad que la vez pasada nadie sabía que lo eran.

## 1. La pregunta

Un mapa de cultivos que no alcanza calidad puede prometer menos de cuatro maneras: recortar el catálogo de clases, abstenerse por parcela, devolver un conjunto de etiquetas plausibles, o retroceder a una clase más gruesa. **¿Cuál conviene, y a costa de quién?**

## 2. Hipótesis sustantiva

No es solo un criterio, es una expectativa con su razón, y se declara para poder equivocarnos por escrito:

> **H1.** A igual coste esperado por parcela, los cuatro mecanismos **no se distinguirán** en calidad agregada dentro de una banda de equivalencia **por declarar**.
>
> **Corregido tras auditoría externa**: la banda estaba anclada en el efecto mínimo detectable del diseño (±0,033), y eso está mal. Un margen de equivalencia tiene que salir de la **menor diferencia prácticamente relevante para quien usa el mapa**, no de la resolución del instrumento; si se ancla en el MDE, es el experimento el que define qué cuenta como equivalente. Queda pendiente de fijar con los casos de uso, y hasta entonces H1 no está preregistrada.
>
> Porque todos operan sobre la misma posterior y solo redistribuyen su incertidumbre.
>
> **H2.** Sí se distinguirán en **el reparto**: el recorte de leyenda concentrará la promesa retirada en las clases que retira, y la abstención la repartirá de forma más uniforme, porque el recorte es una decisión por clase y la abstención es una decisión por parcela.

**Qué las refutaría.** H1 se refuta si algún contraste pareado excluye el cero **fuera** de la banda de equivalencia bajo el `k` preregistrado. H2 se refuta si la medida de disparidad declarada no separa a los mecanismos con el intervalo por bloque, o si separa en la dirección contraria.

## 3. Criterio principal

**Corregido tras auditoría externa: NO se decide después de ver dónde hay potencia.** Elegir la variable primaria según dónde el diseño la detecte es escoger el resultado antes de medirlo, con otro nombre. Se congela aquí, antes del banco confirmatorio, junto con el afectado, la pérdida, el margen práctico, la población y la regla de decisión. Lo hecho sobre PASTIS y BreizhCrops es **exploratorio** y se etiqueta así.

Lo que la medición sí aporta, y se declara: Hoy no puede fijarse honestamente: medida la potencia con cuatro medidas de disparidad declaradas, **ninguna la tiene con cinco bloques** —las cuatro incluyen el cero y necesitan entre doce y diecisiete—. Si tras fijar `k` por el criterio de §4 la disparidad tiene potencia, es el criterio principal; si no la tiene, el criterio principal es el mapa de decisión de la EPIC 25, que no depende de significancia, y el artículo lo dice.

## 4. Los grados de libertad, declarados

Esto es lo que la vez pasada no existía, y es donde se perdió el resultado anterior.

### 4.1 El número de bloques espaciales

`k` se fija por un **criterio espacial**, decidido y commiteado **antes** de mirar ningún contraste con ese `k`:

> **Criterio, en dos condiciones que no miran ningún contraste:**
> **(a) Independencia espacial.** La separación mínima entre una parcela de prueba y la más cercana de entrenamiento tiene que ser un orden de magnitud mayor que el colchón con que se construyen los pliegues (1 km), porque un colchón que solo separa lo que él mismo impuso no demuestra independencia.
> **(b) Estimabilidad del estimando.** El peor bloque tiene que conservar al menos **8 clases** con soporte suficiente. Un F1-macro sobre una leyenda de nueve clases calculado en un bloque donde solo dos son estimables no es una media macro: es otra cosa con el mismo nombre.

**Medido, y el resultado del criterio es incómodo para nosotros:**

| k | separación mínima | clases estimables en el peor bloque (S=20) | ¿cumple? |
|---:|---:|---:|---|
| **5** | **23,5 km** | **10** | **sí** |
| 8 | 2,9 km | 9 | no, por (a) |
| 10 | 2,4 km | 5 | no |
| 12 | 2,6 km | 2 | no |
| 15 | 2,2 km | 2 | no |
| 20 | 2,4 km | 1 | no |
| 25 | 2,0 km | 1 | no |

**El criterio selecciona `k = 5`.** Con cinco bloques la separación mediana entre prueba y entrenamiento es de 133 km; a partir de ocho se desploma a cinco, o sea que los bloques pasan a ser vecinos y la independencia espacial deja de estar demostrada. Y a partir de doce, el peor bloque conserva **dos** clases estimables de nueve.

**Y `k = 5` es exactamente el valor donde el contraste NO alcanza significancia.** Ese es el punto de haber declarado el criterio antes: si lo hubiéramos elegido mirando el resultado habríamos cogido el 15, que es a la vez el mayor `|delta|` y la menor desviación del barrido —la celda más favorable de siete— y que tiene una separación de 2,2 km y dos clases estimables en su peor bloque.

**¿Y no se pueden tener más bloques subiendo el colchón?** Se midió, porque era la única vía
legítima para ganar potencia por diseño en vez de por elección. **No existe.** El colchón sí arregla
la separación, pero se come las parcelas de los bordes, y las que se come son desproporcionadamente
de las clases raras:

| k | colchón | separación mínima | clases estimables | ¿cumple? |
|---:|---:|---:|---:|---|
| **5** | 1 km | **23,6 km** | **10** | **sí** |
| 10 | 15 km | 22,2 km | 4 | no |
| 12 | 15 km | 15,8 km | 2 | no |
| 15 | 15 km | 15,9 km | 2 | no |

Ninguna combinación por encima de cinco bloques pasa las dos condiciones. `k = 5` no es una
preferencia: es el único punto viable del diseño.

**Prohibido explícitamente**: revisar `k` después de ver un contraste.

**Se publica la curva entera** de sensibilidad a `k`, gane lo que gane, y `k` no se revisa después de ver un contraste. La sensibilidad **es un resultado**: el estimador se mueve 0,0122 entre valores de `k`, un 73 % del propio efecto y sin tendencia monótona, lo que significa que en validación cruzada espacial la granularidad de la partición decide la significancia — y la literatura del área la trata como un dato.

### 4.2 El suelo de soporte por bloque

Medido en el barrido: **en todos los valores de `k`, algún bloque tiene una sola parcela de alguna clase.** El F1 por bloque lleva calculándose desde la fase 3 sobre bloques donde una clase es un único ejemplo, y **ningún `k` lo arregla**.

> El estimando promedia solo sobre las clases con al menos **S = 20** parcelas en ese bloque.

**Por qué 20 y no otro.** Es el mayor suelo que conserva una leyenda utilizable en el peor bloque con `k = 5`: sobreviven **10 clases** y cubren el **96,7 %** de las parcelas. Con S = 30 bajan a 8 clases y 92,7 %; con S = 50, a 5 clases y 84,7 %, que ya no es una macro sobre el catálogo del producto. Y con S = 10 entran clases con una decena de ejemplos, donde el F1 binario es ruido.

### 4.3 El estadístico de disparidad

Se define **aquí**, no en la historia que lo calcula, porque una medida definida donde se computa no está preregistrada. **No es la mayor razón observada**: es una medida sobre todas las clases que superan el suelo `S`, y se declara qué hacer cuando una clase recibe cobertura cero en un bloque, porque el número de clases retenidas varía entre 6 y 9 según el bloque y una razón puede no estar definida.

### 4.4 La función de coste

**Dos correcciones de la auditoría externa, las dos de fondo.**

**La cardinalidad no es el coste.** `E[|C|]` trata igual a dos conjuntos del mismo tamaño aunque uno sea agronómicamente inútil, y el conjunto vacío le cuesta cero. La moneda común es una **tabla de pérdidas por acción, resultado y afectado** —etiqueta errónea, no respuesta, conjunto ambiguo, retroceso taxonómico—, y la cardinalidad queda como descriptor secundario. El precio de abstenerse se suma directamente, no se multiplica por la contención.

**Y son tres razones libres, no dos.** Cuatro costes menos una escala común dejan tres. Escribí dos sin justificarlo. O se declara aquí una segunda restricción sustantiva —una equivalencia concreta entre dos pérdidas— o el mapa se presenta como un símplex de tres, y en los dos casos se publica la sensibilidad fuera de esa sección.

### 4.5 El estimando y su población

**Falta declararlo, y el diseño hoy mezcla dos.** O inferencia **condicional al conjunto de datos**, con el parche como clúster y el alcance local reconocido; o inferencia **entre regiones y campañas**, con sitio-año como unidad y bancos genuinamente independientes. Partir el mismo territorio más fino no crea réplicas nuevas: los folds comparten entrenamiento y calibración con un Jaccard medio de 0,60 y máximo de 0,80.

## 5. Universos y multiplicidad

Los dos universos de clases se reportan los dos. El criterio principal se reporta sin corregir; la familia exploratoria, con Holm, excluyendo los puntos donde los mecanismos son idénticos por construcción. La multiplicidad entre bancos, predictores y universos se declara y se trata.

## 6. Bancos: qué es confirmatorio y qué no

**PASTIS y BreizhCrops son exploratorios.** Ya se miraron, y preregistrar una medida sobre datos vistos es un ritual, no un compromiso. El estudio confirmatorio se hace sobre el banco nuevo, y **solo entra como confirmatorio si su partición ajena define bloques suficientes** para el efecto mínimo detectable; si no, entra como descriptivo y se declara.

## 7. Enmiendas

Toda desviación posterior entra como enmienda fechada **antes** de calcular el contraste afectado, incluida cualquiera que toque la regla de entrega, el estimando o `k`.

**Enmienda 3 al preregistro anterior**, que faltaba: se declara que **H1 se refutó** y que la **regla de entrega cambió** del oráculo de etiqueta a la predicción, con el efecto medido de ese cambio.

## 8. Criterio de no envío

Si tras fijar `k` por §4.1 la disparidad no tiene potencia **y** el mapa de decisión no separa regiones, esto es un informe técnico y no un MICAI. **Se evalúa en la fecha atada al sellado de US-134**, con firma en el ADR, y no el día del cierre de envíos, que es cuando ya no se puede actuar.

**El protocolo no puede ser el resultado de reserva.**
