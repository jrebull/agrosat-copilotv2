# Respuesta a las auditorías externas

**Regla de esta página, y es la lección de la primera ronda**: *un hallazgo NO se cierra con una
historia de usuario*. Se cierra cuando **el comportamiento cambió** y hay dónde comprobarlo. Una
historia planificada es un compromiso, no una corrección, y contarla como cierre fue exactamente lo
que la segunda auditoría desmontó.

Seis rondas hasta hoy, y cada una verificó a la anterior en el código. La segunda encontró dos
cierres falsos. La tercera, tres cierres parciales contados enteros y un gate burlable moviendo una
frase de campo. La cuarta encontró **fugas dentro de las reparaciones de la tercera**: un umbral que
seguía leyendo el bloque de prueba por una rendija más estrecha, un gate de custodia que verificaba
los bytes de hoy pero no que el commit los hubiera producido, y un gate de plan que recorría listas
pero no diccionarios. Lo que sigue es el estado real.





---

## Ronda 6 — 4 de septiembre de 2026

La ronda 6 confirmó las cinco reparaciones de la 5 y dejó **dos reservas**, las dos sobre la misma
cosa: un control recién nacido y una propagación a medias.

| # | Hallazgo | Qué cambió de comportamiento |
|---|---|---|
| S1 | **El gate de publicación solo veía la RUTA del artefacto.** La prosa copia números, no rutas: `0,0326` estaba en el preregistro sin marca y el gate pasaba | El gate **extrae las cifras de los propios artefactos obsoletos** —1 007 de cuatro decimales o más, que son las que no salen por casualidad— y falla si un documento no declarado reproduce alguna. Al encenderlo saltaron **cinco documentos** que la búsqueda por ruta no veía. Test en negativo con `--docs` sobre un árbol temporal |
| S2 | **«Se había refutado» sobrevivía** en el propio preregistro y en el cuaderno, mientras el mismo preregistro lo prohíbe en §7 | Corregido en los tres sitios que quedaban, y en cada uno se deja dicho qué decía antes. El documento de revisores lo lleva inline |
| S3 | **El preregistro anclaba una expectativa en el MDE obsoleto** (0,0326) | Retirado el valor: un preregistro que ancla una expectativa en una cifra invalidada la está preregistrando. Se recalcula y se acepta lo que salga. Las cuatro cifras de la enmienda 3 quedan marcadas como pendientes de recalcular antes de firmar |

**Doce documentos** llevan ahora cuarentena, cuatro más que ayer: los encontró el gate, no una
lectura. Y las revisiones recibidas de terceros quedan **exentas por archivo ajeno**: marcarlas
sería editar lo que alguien nos escribió.

---

## Ronda 5 — 4 de septiembre de 2026

La ronda 4 arregló tres cosas y la 5 encontró que **dos de las tres reparaciones tenían el mismo
tipo de agujero que arreglaban**. No es mala suerte: es que reparábamos el caso que nos habían
enseñado y no la clase de casos.

### Cerrado en esta ronda, con su evidencia

| # | Hallazgo | Qué cambió de comportamiento |
|---|---|---|
| Q1 | **«Tres bloques» significaba tres elementos de una lista.** Tres puntos con `block=0` daban `n_unidades=3`, intervalo y p | `_exigir_bloques_pareados` rechaza bloques repetidos y brazos no pareados por `(k, bloque)`. Dos tests, los dos **comprobados fallando** sobre la versión anterior. El mínimo protegía la aritmética y no protegía nada: contaba posiciones |
| Q2 | **El delta restaba dos `nanmean` independientes** mientras el intervalo usaba pares completos, así que el delta publicado no era el centro de su propio intervalo | `delta = media de las diferencias pareadas`. Lo mismo en el bootstrap, que ahora promedia diferencias por bloque y cuenta `sorteos_indefinidos`. El test viejo no podía verlo porque su brazo derecho valía siempre 0,5; el nuevo **exige explícitamente** que las dos fórmulas difieran en el caso de prueba |
| Q3 | **Una fila del ledger con una raya en la celda de procedencia pasaba**: la verificación solo corría cuando *encontraba* un SHA | Una fila sellada declara su commit o dice explícitamente que no está versionada. No hay tercera opción, y hay test |
| Q4 | **El gate de dependencias excluía el campo `dep`**, así que la contradicción cabía dentro de la propia lista de dependencias | Se lee el objeto entero, `dep` incluido. Test |
| Q5 | **`OBSOLETO` avisaba y no impedía nada**: imprimía y devolvía éxito, mientras ocho documentos activos citaban esas cifras | **Gate de publicación nuevo**, `make paper-obsoletos-check`: un documento que cita cifras obsoletas lleva marca de cuarentena o falla, y uno que nombra una ruta obsoleta sin estar declarado también. Los ocho documentos llevan ya la marca. Probado quitándosela a uno |
| Q6 | **Un test fosilizaba la redacción retirada** («le refutaron la hipótesis») usándola como ancla | Las tres mutaciones se anclan en la **estructura** del objeto, no en frases del plan |
| Q7 | **El rol de US-140 contradecía a su propio criterio de aceptación** | Dice «H1-2026 no replicó bajo el análisis exploratorio corregido» |
| Q8 | **US-155 y US-171 conservaban como criterio de aceptación cifras de artefactos OBSOLETO** | Reescritos: se recalculan y **se acepta lo que salga**. Las cifras viejas quedan como lo que se creyó, no como expectativa |
| Q9 | **El cuaderno seguía titulando «La afirmación que el artículo puede sostener»** y afirmando que no se distinguen | Retitulado a «la afirmación **a la que aspira**, y por qué hoy no se sostiene», con las tres cosas que faltan nombradas |
| Q10 | **Dos porcentajes de esfuerzo incompatibles en la misma página**, ninguno derivable del plan | Retirados los dos. El reparto se lee de `make plan-check` y no se transcribe: transcribirlo es exactamente lo que produjo la incompatibilidad |
| Q11 | **El ledger llamaba «separación espacial exacta» a la distancia entre centroides** | Corregido en las tres filas, con la cota superior dicha. Y la pregunta a Arthur pasa de 2,2 a 2,009 km y nombra los centroides |
| Q12 | **El mínimo de tres bloques se presentaba como frontera de inferencia** | Declarado como **regla conservadora de publicación de este proyecto**, con su motivo, y separado de la independencia y del MDE, que son preguntas distintas |
| Q13 | **US-172 permitía cerrar cada precio como «supuesto del artículo»** — la salida que vaciaba el único bloqueante sustantivo | Eliminada. Ahora exige protocolo de elicitación escrito antes, **al menos tres informantes de cada uno de los dos grupos afectados**, respuestas crudas guardadas, síntesis trazable celda por celda, y publicación del desacuerdo cuando lo haya. Un precio que nadie de fuera del equipo haya dicho en voz alta no entra |

### Lo que sigue ABIERTO

US-172, US-173, US-174 y US-175, y con ellas el criterio principal. El MDE con t central. La
multiplicidad de toda la superficie. La selección de predictores sobre datos separados, que está
declarada pero no reparada. Y los trece artefactos `OBSOLETO`, que ahora al menos **no se pueden
citar sin decirlo**.

---

## Ronda 4 — 4 de septiembre de 2026

El patrón de esta ronda: **reparamos por el camino ancho y dejamos el estrecho abierto**. Las tres
correcciones de la ronda 3 eran correctas en lo que arreglaban y seguían teniendo una rendija.

### Cerrado en esta ronda, con su evidencia

| # | Hallazgo | Qué cambió de comportamiento |
|---|---|---|
| C1 | **El punto de operación seguía leyendo el bloque de prueba.** El umbral salía de entrenamiento, pero su cuantil se fijaba con `ref.delivered.mean()`, la cobertura que el otro mecanismo **realizó en el bloque de prueba** | `umbral_desde_entrenamiento()` calcula las dos mitades sobre entrenamiento: la tasa es la fracción de parcelas de ENTRENAMIENTO cuyo argmax cae en la leyenda. Dos tests: uno de **invariancia** —destrozar el bloque de prueba entero no mueve el umbral, y tocar el entrenamiento sí— y otro que comprueba que cambiar la máscara de la referencia no cambia lo que entrega el comparador. El segundo **falla sobre la implementación de ayer**, comprobado |
| C2 | **Con dos bloques se publicaban intervalo, p y Holm**, donde el contrato dice reportar solo los dos deltas | Por debajo de **tres bloques definidos**, `paired_interval` devuelve `ci=None`, `p=None` y el motivo. Los dos productores detectan el `None` y **no aplican Holm**: escriben `motivo_sin_holm` en su lugar. Test con n=2 |
| C3 | **La rama de varianza cero devolvía p=1 para cualquier constante**, y su único test usaba cuatro ceros — otra vez el único valor que no distingue | Separados los dos casos: todo cero es la autocomprobación del arnés (p=1); constante no nula da intervalo degenerado, `excluye_cero` verdadero y **`p_valor: None`**, porque sin varianza la t no está definida y no se inventa un p. Test para cada caso |
| C4 | **Una intersección vacía devolvía 0,0**, un valor con pinta de medida | `macro_over` devuelve **NaN**. `paired_interval` los descarta y publica `bloques_indefinidos`. Dos tests |
| C5 | **El gate de custodia no comprobaba procedencia**: verificaba los bytes de hoy, no que el commit declarado los hubiera producido. Sustituir el sello por el commit raíz, 467 atrás, daba OK | Verifica el **blob en el commit de cada fila** —o el MD5 dentro de su puntero `.dvc`— y exige que el sello **no sea anterior a ninguna fila que sella**. Al encenderlo saltaron **las diez filas** que la auditoría predijo. Corregidas con `scripts/paper_artifacts_seal.py`, que **calcula** la columna desde git en vez de rellenarla a mano. Tres tests nuevos: commit de fila inventado, sello en el commit raíz, y MD5 que ese commit nunca produjo |
| C6 | **El gate de dependencias no recorría diccionarios anidados**: `meta:{nota:"Sin dependencias"}` pasaba | Recorrido recursivo por cualquier estructura. Cuarto test parametrizado |
| C7 | **Los artefactos obsoletos seguían siendo `SELLADO`** y el gate anunciaba «84 sellados, OK» | Estado **`OBSOLETO`** ejecutable: 13 filas marcadas, verificadas igual, contadas aparte y con aviso explícito de que no son citables |
| C8 | **El cuaderno público presentaba esas cifras como «rehecho sin los tres defectos» y como transporte demostrado** | Los tres apartados marcados «pendiente de regenerar», con el texto viejo tachado y el motivo. El de BreizhCrops dice además que su intervalo y su Holm **no deberían existir** con dos regiones |
| C9 | **«22 951 metros entre parcelas» y «partición impecable»** | El número no está en ningún artefacto —el vigente da 22 972— y es entre **centroides**. Retirado, con la limitación escrita |
| C10 | **«No se distinguen» presentado como sostenible** | Reescrito: es la frase a la que aspiramos, y hoy no lo es. Faltan las tres cosas, y se nombran — sin pérdida «a igual coste» no significa nada, ausencia de evidencia no es equivalencia, y la disparidad tampoco tiene potencia |
| C11 | **Correcciones no propagadas**: «decide la significancia» en el preregistro y en un título; «un revisor lo criticará se haga como se haga» en dos documentos; el campo de tiro decía que el código conservaba los defectos; «ocho auditorías internas» | Las cinco corregidas en todas sus apariciones |
| C12 | **El presupuesto documental quedó obsoleto al reordenar el plan** | Fechado como fotografía del 3 de septiembre, con la fuente viva señalada: `make plan-check` da 89 historias, 255 SP pendientes y 96 SP de camino crítico |
| C13 | **La selección de predictores usa las etiquetas que luego evalúa** | No reparado —necesita datos separados, y es US-139— pero **declarado en el código y en el log**, para que nadie lo tome por una elección ciega |

### Lo que sigue ABIERTO

Sin cambios de dueño: **US-172** (pérdida por acción, resultado y afectado), **US-173** (estimando y
población), **US-174** (margen práctico), **US-175** (las tres razones libres) y, con ellas, el
criterio principal. El MDE sigue siendo aproximación con t central. La multiplicidad de toda la
superficie y la selección de predictores sobre datos separados siguen sin implementar. **Y los trece
artefactos marcados `OBSOLETO` siguen sin regenerar**: ninguna de sus cifras entra en el artículo.

---

## Ronda 3 — 3 de septiembre de 2026

La ronda 3 tiene un patrón propio y conviene decirlo antes que los hallazgos: **corregimos el
documento donde el auditor había señalado y no el resto de sus apariciones**. El preregistro decía
lo correcto y el plan seguía diciendo lo anterior, cuatro veces. La regla que faltaba es la sexta de
la guía del proyecto y la teníamos escrita: *agota el alcance del control*.

### Cerrado en esta ronda, con su evidencia

| # | Hallazgo | Qué cambió de comportamiento |
|---|---|---|
| T1 | **Los tres defectos de `paper_micai_coverage.py` seguían en el código, y el módulo no tenía tests** | Los tres reparados, y **cada reparación es un parámetro obligatorio**, porque los tres sobrevivieron como valores por defecto silenciosos: `macro_over` exige `presentes` (el universo es del BLOQUE, no de lo entregado); `confidence_baseline` toma el umbral de los bloques de ENTRENAMIENTO, así que la cobertura ya no coincide exacta con la referencia; `paired_interval` exige `unidad`, y `"bloque"` es el intervalo del artículo. Ocho tests nuevos en `tests/ml/eval/test_paper_micai_coverage.py`, y **dos de ellos comprobados fallando** sobre la implementación anterior |
| T2 | **US-155 conservaba «el criterio principal se mueve a donde haya potencia»**, contradiciendo a US-140 | Retirado del plan con su motivo. La potencia **evalúa** una variable elegida por la decisión sustantiva; no la elige |
| T3 | **«Distancia exacta entre parcelas» era exacta solo entre centroides** | Renombrada a `separacion_centroides_min_km` en el artefacto, en el preregistro y en US-171, con la nota de que es una **cota superior** y de que el diagnóstico de autocorrelación que sí demostraría independencia no está hecho |
| T4 | **El criterio de despliegue estaba mal descrito**: decíamos «F1 por clase ≥ 0,90» | El código dice otra cosa (`ml/eval/class_remap.py`): el catálogo de doce es el **último cuyo macro-F1 restringido se mantiene sobre 0,90** al añadir clases en orden de calidad resuelta — 0,9001 con doce, por debajo con trece. Corregido. Importa porque esa decisión real es el insumo de US-172 y no puede entrar deformada |
| T5 | **0,88269 seguía publicada como cobertura** en el reencuadre | Corregida ahí también. Era la cuarta aparición, y las tres primeras se habían arreglado |
| T6 | **El Jaccard seguía mal atribuido** en US-173 y en el cuaderno público | Corregido en los dos consumidores |
| T7 | **US-140 seguía exigiendo declarar que «H1 se refutó»** | Ahora dice `H1-2026` **no replicó**, con el intervalo antes y después |
| T8 | **El ledger tenía procedencia falsa y su gate no podía verla** | El gate comprueba ahora **procedencia, no solo bytes**: el commit de sellado tiene que existir y estar en la historia de HEAD, y una fila que dice «sin seguimiento en git» tiene que ser cierta. Al encenderlo saltaron **24 filas mintiendo**, no las tres que el auditor encontró a mano. Corregidas con el commit real de cada artefacto. Los dos controles, probados en negativo **y versionados** en `tests/scripts/test_gates_procedencia.py` |
| T9 | **El gate de dependencias solo miraba dos campos**, y burlarlo era mover la frase al título | Recorre el objeto entero —cualquier campo de texto, presente o futuro— y reconoce ocho variantes de la afirmación. Tres tests parametrizados mutan título, rol y criterio de aceptación y exigen que falle en los tres |
| T10 | **«La granularidad decide la significancia»** excedía lo que el diseño sostiene | Reescrito en sus tres apariciones: «en este análisis exploratorio, la conclusión cambia con k». Ni «decide» ni «demuestra» |
| T11 | **Las 176 celdas H3 estaban escritas a mano** en un docstring y en un JSON | El artefacto las **produce**: `celdas_h3_del_universo`, con su `h3_res`. Salen 176 |
| T12 | **La exclusión del rechazador aprendido se presentaba como inevitabilidad** («un revisor lo criticará se haga como se haga») | Declarada como exclusión de alcance con su coste: el artículo no dice nada sobre mecanismos **aprendidos** de renuncia. Y US-170 deja de afirmar que los cuatro «cubren el espacio» |
| T13 | **`u(y,C) = g(|C|)` reaparecía en el plan** como si la cardinalidad fuera la moneda | US-160 lo separa: `g` es el premio por acertar con un conjunto de un tamaño dado; el coste sale de la tabla de US-172. En el código ya estaba separado |
| T14 | **`k=15` seguía con 2,2 km** en el documento para Arthur tras recalcularlo | 2,009 km |
| T15 | **`ESTADO.md` decía nueve auditorías internas** y el plan decía ocho | Sin número: el recuento no sale de ningún artefacto |

### Lo que sigue ABIERTO, y no ha cambiado de dueño

`A1` a `A10` de la ronda 2 siguen como estaban, salvo `A6`, que se cierra con T1. En resumen:
**la tabla de pérdidas (US-172), el estimando y su población (US-173), el margen práctico (US-174)
y las tres razones libres (US-175)** siguen sin existir, y con ellas el criterio principal. El MDE
sigue siendo aproximación con t central. La multiplicidad de toda la superficie y la selección de
predictores sobre datos separados siguen sin implementar. **Y todo lo que produjo el módulo
reparado hoy queda pendiente de regenerar**, con el aviso escrito en la cabecera del ledger.

---

## Ronda 2 — 3 de septiembre de 2026

### Lo que la ronda 2 encontró de nuevo, y ya está corregido

| # | Hallazgo | Estado | Evidencia |
|---|---|---|---|
| N1 | **`g` seguía recibiendo el cero.** La firma prometía que la utilidad de un conjunto solo se evalúa sobre conjuntos no vacíos, y la implementación la aplicaba sobre TODOS los tamaños antes de descartar columnas. Una `g` legítima definida solo para tamaños positivos reventaba | **CERRADO** | `ml/eval/set_valued.py` evalúa `g` sobre `flatnonzero(acierta & ~vacios)`. Test `test_la_utilidad_nunca_le_pasa_un_cero_a_g` con un espía que **falla sobre la implementación anterior** — comprobado revirtiendo el módulo |
| N2 | **La cardinalidad seguía declarada como «el eje de coste»** en el docstring, y `coste_esperado` seguía siendo una función del tamaño | **CERRADO** | La función se llama `cardinalidad_esperada`, **no acepta ninguna `g`**, y el test comprueba que rechaza recibirla. El docstring dice que la moneda es la tabla de pérdidas de US-172 y que no vive en ese módulo |
| N3 | **La «distancia mínima» no era mínima**: se submuestreaban ~300 puntos por lado antes de tomar el mínimo, lo que sesga hacia arriba por construcción | **CERRADO** | KD-tree exacto en `scripts/run_paper_micai_parametros_prereg.py`. Los números publicados cambian: k=5 de 23,505 a **22,972 km**; k=8 de 2,877 a **1,975**; mediana de 133,489 a **122,056**. El veredicto del criterio no cambia; el número sí |
| N4 | **La tabla de colchones vivía solo en la prosa** — 23,6 / 22,2 / 15,8 / 15,9 km sin productor ni artefacto | **CERRADO** | `reports/paper_micai/prereg/barrido_colchon.csv`, veinte combinaciones selladas. Y los valores exactos no son los que la prosa decía |
| N5 | **El Jaccard estaba mal atribuido**: 0,60 / 0,80 es train+val, no train | **CERRADO** | El artefacto trae los dos con nombre propio: entre entrenamientos **0,4273 / 0,5510**; entre entrenamiento+validación **0,6000 / 0,8005**. Corregido en el preregistro, en US-171 y en el campo de tiro |
| N6 | **0,88269 tampoco es una cobertura.** El filtro que produce las 14 688 parcelas descarta por la etiqueta verdadera | **CERRADO en el nombre, ABIERTO en la medición** | Verificado en `ml/eval/perceiver_champion_eval.py`: `if int(label) not in kept: continue`. Es la **cuota de soporte de la verdad**, no cobertura. Retirada de las tres apariciones publicadas. **La cobertura de entrega sigue sin medirse**, y es trabajo de US-171 |
| N7 | **«k=5 es el único punto viable» convierte preferencias en restricciones** | **CERRADO** | El preregistro dice ahora «el único que pasa los umbrales **elegidos**», y declara que el orden de magnitud y las ocho clases son criterios razonables declarados antes del contraste, no deducidos del dato |
| N8 | **«H1 se refutó» reutilizaba el nombre de una hipótesis que aún no existe** | **CERRADO** | Convención explícita: `H1-2026` para la anterior, `H1` para la confirmatoria. Y ya no se dice «se refutó» sino «no replicó bajo el análisis exploratorio corregido», porque el inferencial que produjo ese resultado conserva el remuestreo que US-125 no ha reparado |
| N9 | **US-140 decía «sin dependencias» y declaraba cuatro** | **CERRADO, y con gate** | `scripts/plan_check.py` comprueba la contradicción. Probado en negativo: inyectada la frase, el gate falla; retirada, pasa. Encontró además tres falsos positivos propios («no depende de qué salga»), y se acotó |
| N10 | **La leyenda del plan decía que `idle` es «sin bloqueos»** con 48 historias `idle` con dependencias pendientes | **CERRADO** | La leyenda dice ahora «especificada y no empezada, puede tener dependencias pendientes» |
| N11 | **Tope editorial contradictorio**: cabecera 16–18, EPIC 23 veinte, presupuesto 20,5 | **CERRADO** | Una sola hipótesis de planificación —**20 páginas, a reverificar con la convocatoria**— y el presupuesto suma 20: resultados baja de 6,5 a 6, aplicando su propia regla de recorte |
| N12 | **«No hay asociación significativa bajo abstención, y sí bajo recorte»** | **CERRADO** | Ninguna de las dos lo es: bajo recorte p = 0,0716 (PASTIS) y p = 0,9661 (BreizhCrops). Leer un 0,0716 como un 0,05 es el error que el artículo denuncia en otros |
| N13 | **La corrección se vendía como cierre en el propio plan**: «reparado en US-124/125» sin cambio de código | **CERRADO como afirmación** | La frase está retirada y esta página lo dice; el código sigue con los tres defectos, y aparecen abajo como ABIERTO, no como cerrado |

### Lo que la ronda 2 encontró y sigue ABIERTO

| # | Hallazgo | Por qué no está cerrado | Dueño |
|---|---|---|---|
| A1 | **El suelo `S` por bloque reintroduce un denominador móvil.** Si cada bloque elige su universo de clases con sus propias etiquetas, cada bloque estima otra población | **Es el mismo defecto que tumbó el resultado anterior, con otro disfraz.** Medido y publicado, no resuelto: con k=5 y S=20 los bloques retienen 11, 13, 14, 13 y 10 clases, solo **6 son comunes a los cinco**, la unión es 18 y el Jaccard mínimo entre universos es **0,400**. Hay dos salidas honestas y hay que elegir una en la firma | US-173 |
| A2 | **La pérdida no existe.** La cardinalidad ya no ocupa su lugar en el código, pero la tabla no está | Ninguna cifra del artículo puede depender de «igual coste» hasta que exista | **US-172, cabeza del camino crítico** |
| A3 | **Tres razones libres, no dos.** Cuatro costes menos una escala común | Reconocido en el preregistro; falta la geometría ejecutable y la sensibilidad fuera de cualquier sección bidimensional | US-175 |
| A4 | **El estimando y su población.** El diseño mezcla condicional-al-conjunto con entre-regiones | Y de él depende que el artículo pueda o no afirmar transporte | US-173 |
| A5 | **La banda de equivalencia.** Retirada del productor, no sustituida | El artefacto ya no produce ningún número que pueda pasar por una banda: `banda_equivalencia: null` con su motivo | US-174 |
| A6 | **Los tres defectos de `paper_micai_coverage.py` siguen en el código**: `macro_over`, `confidence_baseline`, `paired_interval`. Y el módulo no tiene tests | Diagnosticados y medidos, no reparados. **Todo lo que ese módulo produjo hoy es exploratorio** | US-124, US-125 |
| A7 | **`tiene_potencia` del barrido de bloques trata las subdivisiones como réplicas** | El artefacto se conserva con su hash; lo que cambia es que **esa columna no se cita** y el ledger lo dice con su motivo. Se rehace con la unidad del estimando | US-171, tras US-173 |
| A8 | **El MDE es una aproximación con t central** | Etiquetado como aproximación en el ledger y en el preregistro; falta rehacerlo con t no central | US-128 |
| A9 | **Multiplicidad de toda la superficie**, no solo de los contrastes tabulados; y la selección de predictores usa hoy los datos de evaluación | Declarado en §5 y §6 del preregistro; sin implementar | US-128, US-139 |
| A10 | **El criterio principal no está fijado** | Se retiró la regla condicional, que era peor que no tenerlo. Se fija en la firma, uno solo, desde la tabla de pérdidas | US-140, tras US-172 |

---

## Ronda 1 — lo que quedó, revisado por la ronda 2

De los trece hallazgos de la primera ronda, la segunda confirmó **cinco cerrados**, degradó **dos que
habíamos dado por cerrados** (la utilidad y el presupuesto de páginas: la corrección estaba
incompleta) y dejó el resto abiertos. Los que siguen vivos están arriba, en la tabla de ABIERTO, con
su dueño. No se repiten aquí para que no haya dos versiones del mismo estado.

**Los cuatro patrones que se repiten**, porque son lo más caro del proyecto:

1. **Un número que solo existe en la prosa** — cinco veces, la última las 176 celdas H3.
2. **Una cifra correcta en el contexto equivocado** — cuatro veces, la última 0,88269.
3. **Un control incapaz de detectar aquello para lo que existe** — siete veces: un gate ciego a
   sus acentos; un test con el único valor que no distinguía, **dos veces**, porque la rama de
   varianza cero repitió exactamente el mismo error; dos tests que miraban el número de salida en
   vez del mecanismo; un gate que solo leía dos campos, y su sucesor que leía listas pero no
   diccionarios; y un gate de custodia que comparaba los bytes de hoy sin comprobar que el commit
   los hubiera producido.
4. **Corregir donde se señaló y no en el resto de las apariciones** — el patrón entero de la
   ronda 3, y cinco casos más en la ronda 4.
5. **Reparar por el camino ancho y dejar el estrecho abierto** — la ronda 4: el umbral salía de
   entrenamiento y su tasa no; el gate leía listas y no diccionarios; el gate de custodia comparaba
   bytes y no procedencia. Y la ronda 5, otra vez sobre las mismas reparaciones: el mínimo de
   bloques contaba posiciones de lista; el gate de custodia solo verificaba cuando *encontraba* un
   sha; el de plan leía todo el objeto menos el campo de dependencias.
6. **Reparar el caso que nos ensenaron en vez de la clase de casos.** Es la lectura de la ronda 5, y
   la regla que sale de ella: ante cada reparación, preguntar **de qué clase es el defecto** y
   buscar los demás miembros de esa clase antes de darla por hecha.

Los cuatro tienen la misma raíz: **verificar el resultado en vez de verificar el mecanismo**, y
después **verificar el ejemplo en vez de agotar el alcance**. Por eso ahora todo control nuevo se
prueba en negativo antes de creerle, los de esta ronda están **versionados** para que la prueba no
muera con la sesión, y toda corrección se busca en todas sus apariciones antes de darse por hecha.

---

## Veredicto vigente

**No se arranca.** Las cinco rondas coinciden en el mismo único cambio: **US-172, la tabla de
pérdidas por acción, resultado y afectado**, obtenida de usuarios reales y no de nuestra intuición.
La ronda 5 añadió lo que faltaba: su criterio de aceptación permitía cerrarla declarando cada precio
como «supuesto del artículo», que es cerrarla sin hacerla. Esa salida está eliminada.
