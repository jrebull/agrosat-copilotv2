# Preregistro v2 — borrador para firma

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md): las produjo `ml/eval/paper_micai_coverage.py`
> cuando aun tenia los tres defectos —denominador movil, punto de operacion elegido dentro del
> bloque evaluado, y remuestreo a nivel de parcela—. **Ninguna de esas cifras entra en el
> articulo** hasta regenerarlas (US-124, US-125). Se conservan sin retocar porque el registro de
> lo que creimos importa tanto como lo que resulte.

**Estado**: BORRADOR, y tras la **segunda** auditoría externa sigue siéndolo. **Tres** parámetros están abiertos: la **función de pérdida**, el **margen práctico** y el **criterio principal**. El **estimando y su población quedan cerrados** en §4.5 y en [`estimando-v1.json`](estimando-v1.json). No se firma hasta cerrar esos tres, y no vale hasta que esté commiteado y firmado. **Nada de las EPIC 20, 21, 22 ni 25 se computa antes de ese commit.**

Este documento existe porque el preregistro anterior se citó como credencial mientras se omitía que **el análisis de entonces reportaba que su hipótesis no replicaba** y que la regla de entrega había cambiado. Lo que se omitió es el hecho de haberlo reportado; **el veredicto en sí está hoy pendiente** (§7), porque las cifras que lo sostenían están invalidadas. (Esta frase decía primero «se había refutado» y después «no había replicado»: las dos afirmaban el veredicto, y la segunda lo afirmaba mientras el §7 lo declaraba pendiente.) Aquí se declara todo antes, incluidos los grados de libertad que la vez pasada nadie sabía que lo eran.

**Convención de nombres, y no es cosmética.** Las hipótesis de este documento se numeran `H1`, `H2`. Las del preregistro anterior se citan siempre como **`H1-2026`**, con año. Reutilizar el identificador para dos cosas distintas fue un hallazgo de la auditoría: el mismo documento decía que `H1` estaba por declarar y dos páginas después que «H1 se refutó».

## 1. La pregunta

Un mapa de cultivos que no alcanza calidad puede prometer menos de cuatro maneras: recortar el catálogo de clases, abstenerse por parcela, devolver un conjunto de etiquetas plausibles, o retroceder a una clase más gruesa. **¿Cuál conviene, y a costa de quién?**

## 2. Hipótesis sustantiva

No es solo un criterio, es una expectativa con su razón, y se declara para poder equivocarnos por escrito:

> **H1.** **En puntos de operación seleccionados en entrenamiento para un mismo presupuesto de pérdida**, los cuatro mecanismos **no se distinguirán** en calidad agregada dentro de una banda de equivalencia **por declarar**.
>
> La formulación anterior decía «a igual pérdida esperada por parcela», y eso solo se consigue igualando en la prueba, que es justo lo que §4.5 prohíbe y lo que el código ya impide. El presupuesto se fija en entrenamiento; lo que la prueba estime es el resultado realizado, y puede no ser igual entre mecanismos.
>
> Porque todos operan sobre la misma posterior y solo redistribuyen su incertidumbre.
>
> **H2.** Sí se distinguirán en **el reparto**: el recorte de leyenda concentrará la promesa retirada en las clases que retira, y la abstención la repartirá de forma más uniforme, porque el recorte es una decisión por clase y la abstención es una decisión por parcela.

**Dos cosas que faltan y sin las cuales `H1` no está preregistrada.** La primera, la banda: estaba anclada en el efecto mínimo detectable del diseño (±0,033) y eso está mal, porque un margen de equivalencia sale de la **menor diferencia prácticamente relevante para quien usa el mapa**, no de la resolución del instrumento; anclarlo en el MDE hace que sea el experimento el que decide qué cuenta como equivalente. La segunda, «igual pérdida»: mientras la moneda sea la cardinalidad, «igual coste» no significa nada, porque dos conjuntos del mismo tamaño no cuestan lo mismo y el conjunto vacío no es gratis. Las dos las cierra la EPIC 27.

**Qué las refutaría.** `H1` se refuta si algún contraste pareado excluye el cero **fuera** de la banda de equivalencia bajo el `k` preregistrado. `H2` se refuta si la medida de disparidad declarada no separa a los mecanismos con el intervalo por bloque, o si separa en la dirección contraria.

## 3. Criterio principal

**No se elige aquí, y esa es la razón de que este documento sea un borrador.** El criterio principal se congela en la firma, **uno solo y sin regla condicional**, junto con el afectado, la pérdida, el margen práctico, la población y la regla de decisión.

**Se retira explícitamente la regla que este documento traía**: «si tras fijar `k` la disparidad tiene potencia, es el criterio principal; si no la tiene, el criterio principal es el mapa de decisión». Esa frase es elegir la variable primaria según dónde el diseño la detecte, que es escoger el resultado antes de medirlo con otro nombre. Que estuviera escrita dos líneas después de prohibirlo es el hallazgo, no la excusa.

Lo que la medición sí aporta, y se declara: con cuatro medidas de disparidad declaradas, **ninguna tiene potencia con cinco bloques** — las cuatro incluyen el cero y necesitan entre doce y diecisiete. Eso es un dato del diseño, no un criterio de selección. Lo hecho sobre PASTIS y BreizhCrops es **exploratorio** y se etiqueta así en todas partes.

## 4. Los grados de libertad, declarados

Esto es lo que la vez pasada no existía, y es donde se perdió el resultado anterior.

### 4.1 El número de bloques espaciales

El universo retenido tiene **176 celdas H3 de resolución 5** —producido por el artefacto, no escrito a mano—, así que admite muchos más de cinco bloques. `k` se fija por un **criterio espacial**, decidido y commiteado **antes** de mirar ningún contraste con ese `k`:

> **Criterio, en dos condiciones que no miran ningún contraste:**
> **(a) Separación espacial.** La distancia mínima **entre centroides** de una parcela de prueba y la más cercana de entrenamiento tiene que ser un orden de magnitud mayor que el colchón con que se construyen los pliegues, porque un colchón que solo separa lo que él mismo impuso no demuestra nada. **Y se dice lo que es**: el universo sellado guarda centroides, así que esta distancia es una **cota superior** de la separación entre parcelas, y por sí sola no demuestra independencia — eso necesita un diagnóstico de autocorrelación residual, que no está hecho y entra como limitación declarada.
> **(b) Estimabilidad del estimando.** El peor bloque tiene que conservar al menos **8 clases** con soporte suficiente. Un F1-macro sobre una leyenda de nueve clases calculado en un bloque donde solo dos son estimables no es una media macro: es otra cosa con el mismo nombre.

**Los dos umbrales —el orden de magnitud y las ocho clases— son elegidos, no deducidos.** La auditoría externa tiene razón en esto y se corrige el lenguaje: no hay diagnóstico de autocorrelación que demuestre que diez veces el colchón implica independencia, ni fuente externa que fije ocho clases. Son criterios **razonables y declarados antes del contraste**, que es lo que los hace válidos como preregistro; no son una restricción del dato. Por eso se publica la curva entera y se declara la sensibilidad a los dos umbrales.

**Medido con distancia exacta al vecino más cercano, y el resultado del criterio es incómodo para nosotros** (`reports/paper_micai/prereg/parametros_diseno.csv`):

| k | separación mínima entre centroides | mediana | clases estimables en el peor bloque (S=20) | ¿cumple? |
|---:|---:|---:|---:|---|
| **5** | **22,972 km** | **122,056 km** | **10** | **sí** |
| 8 | 1,975 km | 4,635 km | 9 | no, por (a) |
| 10 | 1,850 km | 4,606 km | 5 | no |
| 12 | 2,506 km | 4,494 km | 2 | no |
| 15 | 2,009 km | 3,211 km | 2 | no |
| 20 | 2,142 km | 4,171 km | 1 | no |
| 25 | 1,850 km | 3,711 km | 1 | no |

**Corrección 3, de la segunda auditoría externa.** Las cifras que este documento publicaba antes —23,5 km con `k=5`, 2,9 con `k=8`, mediana 133,5— salían de submuestrear unos 300 puntos de cada lado antes de tomar el mínimo. Eso **sesga hacia arriba por construcción**: quitar candidatos solo puede mantener o subir un mínimo, así que el diseño parecía más separado de lo que está. Con KD-tree exacto la separación baja a 22,972, 1,975 y 122,056. **El veredicto del criterio no cambia**, pero el número publicado sí, y lo que se publica es el número.

**El criterio selecciona `k = 5`.** Con cinco bloques la separación mediana entre prueba y entrenamiento es de 122 km; a partir de ocho se desploma por debajo de cinco, o sea que los bloques pasan a ser vecinos. Y a partir de doce, el peor bloque conserva **dos** clases estimables.

**Y `k = 5` es exactamente el valor donde el contraste NO alcanza significancia.** Ese es el punto de haber declarado el criterio antes: si lo hubiéramos elegido mirando el resultado habríamos cogido el 15, que es a la vez el mayor `|delta|` y la menor desviación del barrido —la celda más favorable de siete— y que tiene 2,0 km de separación y dos clases estimables en su peor bloque.

**¿Y no se pueden tener más bloques subiendo el colchón?** Se midió, porque era la única vía legítima para ganar potencia por diseño en vez de por elección. Veinte combinaciones, con productor y artefacto sellado (`reports/paper_micai/prereg/barrido_colchon.csv`) — antes estos números vivían solo en esta prosa, que es el defecto que más veces se ha repetido en este proyecto:

| k | colchón | separación mínima entre centroides | separación / colchón | clases estimables | parcelas de prueba | ¿cumple? |
|---:|---:|---:|---:|---:|---:|---|
| **5** | **1 km** | **22,972 km** | **23,0×** | **10** | **16 640** | **sí** |
| 8 | 10 km | 11,281 km | 1,1× | 8 | 14 314 | no, por (a) |
| 8 | 15 km | 20,694 km | 1,4× | 7 | 12 806 | no, por (a) y (b) |
| 10 | 15 km | 15,628 km | 1,0× | 4 | 12 026 | no |
| 12 | 15 km | 15,478 km | 1,0× | 2 | 11 516 | no |
| 15 | 15 km | 14,989 km | 1,0× | 2 | 10 413 | no |

Subir el colchón **no compra separación relativa**: la separación mínima converge al colchón mismo, que es justo lo que la condición (a) descarta —un colchón que solo separa lo que él impuso—. Y cuesta parcelas: de 16 640 a 10 413 en el peor caso, y las que se come son desproporcionadamente de las clases raras, que es lo que hunde la columna de clases estimables. Con `k = 5` el colchón es irrelevante porque los bloques ya están a 23 km: las cuatro filas dan el mismo número.

**Ninguna combinación por encima de cinco bloques pasa las dos condiciones.** Dicho con el lenguaje correcto: `k = 5` es **el único punto que pasa los umbrales elegidos**, y esos umbrales se declaran antes del contraste. No es «el único viable del diseño»: eso era convertir una preferencia declarada en una restricción del dato.

**Prohibido explícitamente**: revisar `k` después de ver un contraste.

**Se publica la curva entera** de sensibilidad a `k`, gane lo que gane. La sensibilidad **es un resultado**: el estimador se mueve 0,0122 entre valores de `k`, un 73 % del propio efecto y sin tendencia monótona, es decir que **en este análisis exploratorio y en este territorio, la conclusión cambia con `k`**. No se escribe «la granularidad decide la significancia»: eso es una ley y esto es un territorio con folds solapados, medido además con el módulo que aún tenía los tres defectos.

### 4.2 El suelo de soporte por bloque, y el denominador móvil que introduce

Medido en el barrido: **en todos los valores de `k`, algún bloque tiene una sola parcela de alguna clase.** El F1 por bloque lleva calculándose desde la fase 3 sobre bloques donde una clase es un único ejemplo, y **ningún `k` lo arregla**.

La regla candidata era: promediar solo sobre las clases con al menos **S = 20** parcelas **en ese bloque**. Y aquí la segunda auditoría externa encontró algo que nosotros no vimos.

> **La regla del suelo por bloque reintroduce exactamente el defecto que el artículo denuncia.** Si cada bloque decide su universo de clases con sus propias etiquetas, cada bloque estima **una población distinta**, y promediar macros cuyos denominadores significan cosas distintas es el defecto del denominador no común con otro disfraz.

**Medido** (`universo_de_clases_por_bloque` en `parametros_prereg.json`), con `k = 5` y `S = 20`:

| | |
|---|---|
| clases que sobreviven en cada bloque | 11, 13, 14, 13, 10 |
| clases presentes en **los cinco** | **6** |
| clases en la unión | 18 |
| Jaccard mínimo entre dos universos | **0,400** |
| Jaccard medio entre universos | 0,565 |

Seis clases de dieciocho son comunes a los cinco bloques. **DECIDIDO: universo común de clases fijado exclusivamente desde ENTRENAMIENTO**, el mismo para todos los mecanismos y todos los bloques.

- Se deriva de las etiquetas de entrenamiento y **nunca de las del bloque evaluado**, que es lo que hacía móvil el denominador.
- Es **uno solo**: el mismo conjunto de clases para los cinco bloques y para los seis mecanismos, así que dos números del artículo siempre se refieren a la misma población de clases.
- Cuesta clases, y ese es su precio declarado.

**Se retira la alternativa** del estimando clase×bloque con *partial pooling*: era una salida legítima y dejarla abierta significaba elegirla después, que es el defecto que este documento existe para impedir.

Lo que ya no se puede hacer es promediar cinco macros de denominador distinto y llamarlo el mismo estimando. **Es la misma clase de defecto por la que se cayó el resultado anterior**, y esta vez lo encontró alguien de fuera.

Para el registro, la medición que justificaba `S = 20` sigue siendo correcta en lo suyo: es el mayor suelo que conserva **10 clases** y el **96,7 %** de las parcelas en el peor bloque con `k = 5`; con `S = 30` bajan a 8 clases y 92,7 %, y con `S = 50` a 5 y 84,7 %. Eso mide el precio del suelo, no resuelve el denominador.

### 4.3 El estadístico de disparidad

Se define **aquí**, no en la historia que lo calcula, porque una medida definida donde se computa no está preregistrada. **No es la mayor razón observada**: es una medida sobre todas las clases del universo común de §4.2, y se declara qué hacer cuando una clase recibe cobertura cero en un bloque, porque una razón puede no estar definida.

### 4.4 La función de pérdida

**Dos correcciones de la primera auditoría externa, las dos de fondo, y las dos siguen abiertas.**

**La cardinalidad no es la pérdida.** `E[|C|]` trata igual a dos conjuntos del mismo tamaño aunque uno sea agronómicamente inútil, y le pone precio cero al conjunto vacío. La moneda común es una **tabla de pérdidas por acción, resultado y afectado** —etiqueta errónea, no respuesta, conjunto ambiguo, retroceso taxonómico—, y la cardinalidad queda como descriptor secundario. Ya está así en el código: `ml/eval/set_valued.py` expone `cardinalidad_esperada`, que **no acepta ninguna función de coste**, precisamente para que la cardinalidad no pueda volver a ocupar el lugar de una pérdida que nadie declaró.

**Y son tres razones libres, no dos.** Cuatro costes menos una escala común dejan tres. Se escribieron dos sin justificarlo. O se declara una segunda restricción sustantiva —una equivalencia concreta entre dos pérdidas— o el mapa se presenta como un símplex de tres, y en los dos casos se publica la sensibilidad fuera de cualquier sección bidimensional.

### 4.5 El estimando y su población

**DECIDIDO: inferencia condicional al conjunto de datos.** El diseño mezclaba dos y había que
elegir una; se elige la que el diseño sostiene, no la que más nos convendría. La decisión vive
además en un contrato ejecutable, [`estimando-v1.json`](estimando-v1.json), que un gate comprueba
contra este apartado.

| | |
|---|---|
| **Alcance** | Condicional a cada banco, su región y campaña, su partición y su predictor |
| **Población** | Todas las parcelas elegibles del bloque de prueba, **incluidas las que no reciben entrega** |
| **Unidad observacional** | La parcela |
| **Clúster de dependencia** | `patch_id`. Con menos de **tres clústeres únicos pareados**, el resultado es descriptivo: sin intervalo, sin p y sin Holm |
| **Entre bancos** | **Ningún promedio inferencial.** Cada banco se reporta por separado |
| **Transporte** | **No se afirma.** Ni a otras regiones ni a otras campañas |
| **`k`** | Parámetro de **sensibilidad espacial**. No es un número de réplicas ni una palanca de potencia |
| **PASTIS y BreizhCrops** | Siguen siendo **exploratorios** |

**El estimando, simbólicamente**, sin inventar ninguna pérdida —la `L` la fija US-172—:

$$R_{d,a,m}=\frac{1}{N_d}\sum_{i=1}^{N_d} L_a\bigl(y_i, C_{m,\tau_m}(x_i)\bigr)$$

donde `d` es el banco con su partición, `a` el afectado, `m` el mecanismo, `N_d` **todas** las
parcelas elegibles del bloque de prueba y `C_{m,τ_m}` el conjunto que el mecanismo emite con su
punto de operación `τ_m`.

> **`τ_m` se obtiene ENTERAMENTE de entrenamiento y validación.** La prueba solo estima el resultado
> realizado. **Queda prohibido volver a igualar pérdida, cobertura o tasa usando la prueba**, que es
> exactamente la fuga que dos auditorías encontraron en `confidence_baseline` —primero el umbral,
> después su tasa objetivo— y que la invariancia ya implementada impide en el código.

**Qué se pierde al elegir esto, dicho aquí**: el artículo no puede afirmar que sus conclusiones se
transporten a otra región o campaña, y no lo afirmará. Lo que gana es que lo que sí afirme está
sostenido por el diseño que tiene, en vez de por el que nos habría gustado tener.

Partir el mismo territorio más fino no crea réplicas nuevas. **Medido, y con los dos conjuntos nombrados por separado**, porque la vez pasada se citó un número como si fuera el otro. La fuente única de estas cifras es la clave `solapamiento_entre_bloques` de [`reports/paper_micai/prereg/parametros_prereg.json`](../../reports/paper_micai/prereg/parametros_prereg.json); **no se vuelven a escribir desde ninguna otra**:

| k | Jaccard medio entre **entrenamientos** | máximo | Jaccard medio entre **entrenamiento + validación** | máximo |
|---:|---:|---:|---:|---:|
| 5 | 0,4273 | 0,5510 | 0,6000 | 0,8005 |
| 15 | 0,5906 | 0,6402 | 0,8667 | 0,9515 |
| 25 | 0,6215 | 0,6518 | 0,9200 | 0,9737 |

El 0,60 medio y el 0,80 máximo que este documento publicaba eran los de **entrenamiento más validación**, no los de entrenamiento. Los dos son ciertos y no son el mismo número. Y la tendencia dice lo que importa: **cuanto más se parte, más comparten los folds**, hasta un 0,92 con veinticinco bloques. Subdividir no produce réplicas; produce el mismo dato contado más veces.

## 5. Universos y multiplicidad

Los dos universos de clases se reportan los dos. El criterio principal se reporta sin corregir; la familia exploratoria, con Holm, excluyendo los puntos donde los mecanismos son idénticos por construcción. La multiplicidad **de toda la superficie** —bancos, predictores, universos y el mapa de decisión entero— se declara y se trata, no solo la de los contrastes tabulados. Y el MDE se recalcula con t no central. **Aquí no se cita el valor anterior**: salió del módulo con los tres defectos, su artefacto está marcado `OBSOLETO`, y un preregistro que ancla una expectativa en una cifra invalidada la está preregistrando. Se recalcula y se acepta lo que salga.

## 6. Bancos: qué es confirmatorio y qué no

**PASTIS y BreizhCrops son exploratorios.** Ya se miraron, y preregistrar una medida sobre datos vistos es un ritual, no un compromiso. El estudio confirmatorio se hace sobre el banco nuevo, y **solo entra como confirmatorio si su partición ajena define bloques suficientes** para el efecto mínimo detectable; si no, entra como descriptivo y se declara. La selección de predictores se hace sobre datos separados de los de evaluación, que hoy no ocurre.

## 7. Enmiendas

Toda desviación posterior entra como enmienda fechada **antes** de calcular el contraste afectado, incluida cualquiera que toque la regla de entrega, el estimando o `k`.

**Enmienda 3 al preregistro anterior**, que faltaba: se declara que **el veredicto sobre `H1-2026` está PENDIENTE** —no «no replicó», que es lo que decía aquí y era afirmar con las cifras que se reconocen invalidadas dos líneas más abajo—. Lo que sí se declara sin depender de ninguna cifra es que la **regla de entrega cambió**. Lo que se observó, y se recalcula antes de firmar, fue que `H1-2026` no replicaba bajo el análisis exploratorio corregido —el intervalo pasaba de (−0,0430, −0,0147) con p = 0,005 a (−0,0410, +0,0077) con p = 0,130 al componer las dos correcciones de protocolo; **las cuatro cifras salieron del módulo con los tres defectos y se recalculan antes de firmar**— y que la **regla de entrega cambió** del oráculo de etiqueta a la predicción, con el efecto medido de ese cambio.

**Ni «se refutó» ni «no replicó»: PENDIENTE.** Tres motivos, uno por auditoría:

1. El inferencial que produjo ese resultado conservaba el remuestreo defectuoso, hoy reparado pero
   sin regenerar los artefactos.
2. Un análisis exploratorio corregido a posteriori no refuta.
3. Y el que faltaba, de la sexta auditoría: **las cuatro cifras que sostienen el veredicto están
   invalidadas**, así que el veredicto lo está también. Decir «no replicó» mientras se reconoce que
   sus números no valen es exactamente lo que este documento existe para impedir. El veredicto se
   emite cuando se recalcule, y puede salir en cualquiera de los dos sentidos.

## 8. Criterio de no envío

Si tras fijar `k` por §4.1 el criterio principal no separa **y** el mapa de decisión no separa regiones, esto es un informe técnico y no un MICAI. **Se evalúa en la fecha atada al sellado de US-134**, con firma en el ADR, y no el día del cierre de envíos, que es cuando ya no se puede actuar.

**El protocolo no puede ser el resultado de reserva.**
