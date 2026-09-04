# Respuesta a las auditorías externas

**Regla de esta página, y es la lección de la primera ronda**: *un hallazgo NO se cierra con una
historia de usuario*. Se cierra cuando **el comportamiento cambió** y hay dónde comprobarlo. Una
historia planificada es un compromiso, no una corrección, y contarla como cierre fue exactamente lo
que la segunda auditoría desmontó.

Dos rondas hasta hoy. La segunda verificó la primera en el código, no en esta página, y **encontró
dos defectos nuevos graves y un cierre falso**. Lo que sigue es el estado real.

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

**El patrón que se repite, y hay que decirlo porque es el más caro**: cuatro veces hemos publicado
un número que solo existía en la prosa, tres veces una cifra correcta en el contexto equivocado, y
dos veces un control incapaz de detectar aquello para lo que existe —un gate ciego a sus propios
acentos, un test que usaba el único valor que no distinguía—. Los tres modos de fallo tienen la
misma raíz: **verificar el resultado en vez de verificar el mecanismo**. Por eso ahora todo control
nuevo se prueba en negativo antes de creerle, y por eso los dos gates de esta ronda se probaron
rompiéndolos.

---

## Veredicto vigente

**No se arranca.** El único cambio que más acerca el sí es el mismo que dijo la ronda 1 y repite la
ronda 2: **US-172, la tabla de pérdidas por acción, resultado y afectado**, obtenida de usuarios
reales y no de nuestra intuición. Todo lo demás está esperando a eso.
