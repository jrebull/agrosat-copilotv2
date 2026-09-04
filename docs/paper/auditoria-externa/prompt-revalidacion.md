# Re-validación externa · de la ronda 2 en adelante

**Para quién**: el mismo auditor externo de la ronda anterior, o uno nuevo si se prefiere sangre fresca. Se usa después de cada ronda de correcciones, hasta que la respuesta a la última pregunta sea «arrancamos».

---

## El prompt

> Auditaste este plan de investigación y tu veredicto fue **no arrancar todavía**. El equipo dice haber corregido. Tu trabajo ahora tiene tres partes, y la tercera es la que importa.
>
> **Dónde está todo**: el plan en `agrosat-micai-site/plan.html`, variable `EPICS`. Los artefactos en `agrosat-copilotv2/reports/paper_micai/`. El registro de custodia en `paper/ARTIFACTS.md`. El código en `ml/eval/`. Puedes correr `make plan-check`, `make paper-artifacts-check`, `pytest tests/ml/eval/` y `pytest tests/scripts/`.
>
> **Lo que el equipo dice haber hecho** está en `docs/paper/respuesta-auditoria-externa.md`, con un apartado por cada hallazgo tuyo.
>
> ---
>
> **Parte 1 · Verifica, no creas.**
>
> Para cada hallazgo tuyo, comprueba en el código y en los artefactos si está corregido de verdad. Un hallazgo se cierra cuando el comportamiento cambió, **no** cuando existe una historia de usuario que dice que se cambiará, ni cuando un documento afirma que se corrigió. Ese fue uno de tus propios hallazgos y merece la pena volver a aplicarlo aquí.
>
> Marca cada uno: **CERRADO** (con la evidencia que lo demuestra) · **PARCIAL** (qué falta) · **ABIERTO** (por qué no cuenta) · **PEOR** (la corrección introdujo algo nuevo).
>
> **Parte 2 · Lo que la corrección rompió.**
>
> Arreglar cosas rompe otras. Busca en concreto:
> - Cifras que dependían de lo corregido y no se han recalculado.
> - Afirmaciones en documentos o en el cuaderno público que la corrección deja obsoletas.
> - Dependencias del plan que ya no tienen sentido tras reordenar.
> - Tests que pasan porque siguen probando la conducta vieja.
>
> **Parte 3 · Lo nuevo, y es lo que de verdad pagamos.**
>
> No te limites a tu lista anterior. Cada ronda de correcciones es una oportunidad de meter defectos nuevos, y el equipo ya ha demostrado que los mete. Estos cinco patrones tienen cuenta abierta; el número dice cuántas veces han aparecido ya:
> - **Un número que exista solo en prosa.** Cinco veces.
> - **Una cifra tomada de un artefacto y atribuida a otro contexto.** Cuatro veces.
> - **Un control que no puede detectar aquello para lo que existe.** Cuatro veces: un gate ciego a sus propios acentos; un test que usaba el único valor que no distinguía; dos tests que comprobaban el número de salida en vez del mecanismo; y un gate que solo leía dos de los campos que debía cubrir. **Pregúntate siempre por el camino de al lado**: si el control se puede burlar moviendo la afirmación de sitio, no es un control.
> - **Una corrección aplicada donde se señaló y no en el resto de sus apariciones.** Cuatro veces en una sola ronda: el preregistro decía lo correcto y el plan seguía diciendo lo anterior. **Toda afirmación corregida se busca en todas partes** — plan, preregistro, cuaderno público, ledger, docstrings y el propio manuscrito.
> - **Una decisión presentada como restricción del problema**, o **una afirmación que el diseño no puede sostener**, escrita en indicativo.
>
> ---
>
> **Reglas**: verifica en vez de confiar; no inventes referencias; puedes correr código; ordena por severidad; y di también qué está bien, para que no lo rompan en la ronda siguiente.
>
> **Formato**: (a) el hallazgo original o el nuevo; (b) estado con su evidencia; (c) qué falta; (d) **BLOQUEANTE / IMPORTANTE / MENOR**.
>
> **Termina con una sola línea**: **¿arrancamos, o hay que tocar algo antes?** Y si es lo segundo, di cuál es el único cambio que más acerca el «sí».

---

## Cómo se cierra el ciclo

1. El auditor responde.
2. El equipo corrige y **actualiza `respuesta-auditoria-externa.md`** con un apartado por hallazgo, diciendo qué se cambió y dónde se puede comprobar.
3. Se vuelve a lanzar este mismo prompt.
4. Se repite hasta que la última línea diga **arrancamos**.

**Una regla para el equipo, no para el auditor**: no se responde a un hallazgo con una historia de usuario. Se responde con un cambio de comportamiento y con la manera de comprobarlo. Si el arreglo de verdad es futuro, se dice que sigue **ABIERTO** y se explica por qué, en vez de darlo por cerrado.

---

## Estado para la ronda 3

La ronda 2 encontró **dos cierres falsos y tres defectos nuevos**, y el veredicto siguió siendo *no
arrancamos*. Antes de relanzar el prompt de arriba, esto es lo que cambió, para que el auditor lo
verifique en vez de creerlo:

| Qué se tocó | Dónde comprobarlo |
|---|---|
| `g` ya no recibe el cero, y hay un espía que lo prueba | `ml/eval/set_valued.py`, `tests/ml/eval/test_set_valued.py::test_la_utilidad_nunca_le_pasa_un_cero_a_g` |
| La cardinalidad dejó de ser «el eje de coste»: `cardinalidad_esperada` no acepta ninguna `g` | `ml/eval/set_valued.py` |
| Distancia mínima con KD-tree exacto; las tres cifras publicadas cambian | `scripts/run_paper_micai_parametros_prereg.py`, `reports/paper_micai/prereg/parametros_diseno.csv` |
| Barrido de colchón con productor y artefacto (antes solo prosa) | `reports/paper_micai/prereg/barrido_colchon.csv` |
| Jaccard con los dos conjuntos nombrados por separado | `parametros_prereg.json`, clave `solapamiento_entre_bloques` |
| El denominador móvil del suelo `S`, medido y publicado como problema abierto | `parametros_prereg.json`, clave `universo_de_clases_por_bloque`; §4.2 del preregistro |
| Banda de equivalencia retirada del productor, con su motivo | `parametros_prereg.json`, `banda_equivalencia: null` |
| Gate de contradicción «sin dependencias» con dependencias declaradas | `scripts/plan_check.py`, `make plan-check` |
| `H1-2026` frente a `H1`; retirada la regla condicional del criterio principal | `docs/paper/preregistro-v2-borrador.md` §2 y §3 |
| 0,88269 renombrada a cuota de soporte de la verdad en las tres apariciones | `plan.html`, `index.html`, US-171 |

**Lo que NO se tocó, y se dice antes de que lo encuentre**: los tres defectos de
`ml/eval/paper_micai_coverage.py` siguen en el código y ese módulo sigue sin tests; el MDE sigue
siendo aproximación con t central; la multiplicidad de toda la superficie y la selección de
predictores siguen sin implementar; y la tabla de pérdidas, el estimando y el margen práctico —las
tres que bloquean la firma— siguen siendo historias, no comportamiento.

---

## Estado para la ronda 4

La ronda 3 encontró **tres cierres parciales contados como enteros** y un patrón nuestro que no
habíamos visto: *corregir donde el auditor señaló y dejar el resto de las apariciones*. El
preregistro decía lo correcto y el plan seguía diciendo lo anterior, cuatro veces.

Lo que cambió de comportamiento, para que se verifique en vez de creerse:

| Qué se tocó | Dónde comprobarlo |
|---|---|
| Los tres defectos de la frontera, reparados; cada reparación es un **parámetro obligatorio** | `ml/eval/paper_micai_coverage.py`; `tests/ml/eval/test_paper_micai_coverage.py` (8 tests) |
| El universo del macro es del BLOQUE, no de lo entregado | `macro_over(..., presentes=...)`, sin valor por defecto |
| El umbral de confianza sale de los bloques de entrenamiento | `confidence_baseline`; la cobertura ya **no** coincide exacta con la referencia |
| El intervalo declara su unidad; `"bloque"` es el del artículo | `paired_interval(..., unidad=...)`, sin valor por defecto |
| Gate de procedencia del ledger: commit de sellado y versionado real | `scripts/paper_artifacts_check.py`; saltaron 24 filas falsas |
| Gate de dependencias, recorriendo el objeto entero | `scripts/plan_check.py::_textos` |
| Los dos gates, probados en negativo **y versionados** | `tests/scripts/test_gates_procedencia.py` (7 tests) |
| Celdas H3 producidas, no tecleadas | `parametros_prereg.json`, clave `celdas_h3_del_universo` |
| Distancia renombrada a **entre centroides**, con su limitación | `parametros_diseno.csv`, US-171, §4.1 del preregistro |
| Criterio de despliegue descrito como lo describe el código | `ml/eval/class_remap.py`, reencuadre §«el único activo» |
| Retirada de US-155 la regla de mover el criterio a donde haya potencia | `plan.html`, US-155 |

**Lo que NO cambió**: la tabla de pérdidas, el estimando y su población, el margen práctico y las
tres razones libres siguen sin existir, y con ellas el criterio principal. El MDE sigue siendo
aproximación con t central. La multiplicidad de toda la superficie y la selección de predictores
sobre datos separados siguen sin implementar. **Y todos los artefactos que produjo el módulo
reparado hoy están pendientes de regenerar**, con el aviso en la cabecera del ledger.

**Para la ronda 4, además de las tres partes de arriba, busca específicamente el patrón de la
ronda 3**: una corrección aplicada en un documento y no en los otros que dicen lo mismo. Y
comprueba si los controles nuevos son burlables por el camino de al lado, como lo fue el anterior.

---

## Estado para la ronda 5

La ronda 4 encontró **fugas dentro de las reparaciones de la ronda 3**, y de ahí sale el quinto
patrón: *reparar por el camino ancho y dejar el estrecho abierto*.

| Qué se tocó | Dónde comprobarlo |
|---|---|
| El punto de operación entero sale de entrenamiento, tasa incluida | `ml/eval/paper_micai_coverage.py::umbral_desde_entrenamiento`; test de **invariancia** frente a cualquier cambio en el bloque de prueba |
| Con menos de tres bloques definidos no hay intervalo, ni p, ni Holm | `paired_interval`; `run_paper_micai_fase3.py` y `fase4.py` escriben `motivo_sin_holm` |
| Varianza cero: separados «todo cero» y «constante no nula»; la segunda no inventa p | `paired_interval`, dos tests |
| Lo indefinido es NaN, y los bloques indefinidos se cuentan | `macro_over`; clave `bloques_indefinidos` |
| El gate de custodia verifica el **blob en el commit de cada fila** y que el sello no preceda a sus filas | `scripts/paper_artifacts_check.py`; `scripts/paper_artifacts_seal.py` calcula la columna desde git |
| Estado **`OBSOLETO`** ejecutable: 13 filas verificadas pero no citables | `paper/ARTIFACTS.md`, salida del gate |
| El gate de dependencias recorre diccionarios anidados | `scripts/plan_check.py::_textos` |
| Retiradas del cuaderno las cifras del módulo defectuoso presentadas como corregidas | `index.html`, apartados de fase 3 y fase 4 |
| «22 951 m entre parcelas» y «partición impecable», retirados | `index.html` |
| «No se distinguen» deja de presentarse como sostenible | `docs/paper/recomendacion-final.md` §2 |
| La selección de predictores sobre datos de evaluación, **declarada** en código y log | `scripts/run_paper_micai_fase3.py` |

**Lo que NO cambió**: US-172, US-173, US-174 y US-175 siguen sin existir, y con ellas el criterio
principal. El MDE sigue con t central. La multiplicidad de toda la superficie y la selección de
predictores sobre datos separados siguen sin implementar. Los trece artefactos `OBSOLETO` siguen sin
regenerar.

**Para la ronda 5**, además de las tres partes, dos encargos concretos:

1. **El camino de al lado, otra vez.** Cada reparación de esta ronda es un control nuevo. Intenta
   burlarlo por donde no mira: el gate de custodia, el de plan, la invariancia del umbral y la regla
   de los tres bloques.
2. **La propagación.** Toma tres afirmaciones que esta ronda dice haber corregido y búscalas en
   TODAS partes: plan, preregistro, cuaderno, ledger, docstrings y manuscrito.

---

## Estado para la ronda 6

La ronda 5 encontró que **dos de las tres reparaciones de la ronda 4 tenían el mismo tipo de agujero
que arreglaban**, y de ahí sale el sexto patrón: *reparar el caso que nos enseñaron en vez de la
clase de casos*.

| Qué se tocó | Dónde comprobarlo |
|---|---|
| «Tres bloques» son tres bloques distintos y pareados, no tres posiciones de lista | `_exigir_bloques_pareados`; dos tests probados fallando |
| El delta es la media de las **diferencias pareadas**, no la diferencia de dos medias | `paired_interval`; el test exige que el caso distinga las dos fórmulas |
| Una fila del ledger declara commit o dice que no lo tiene: no hay tercera opción | `paper_artifacts_check.py`; test con una raya |
| El gate de plan lee también el campo `dep` | `plan_check.py::_textos`; test |
| **Gate de publicación nuevo**: `make paper-obsoletos-check` | Un documento que cite cifras `OBSOLETO` lleva cuarentena o falla |
| Los tests se anclan en la estructura del objeto, no en frases del plan | `tests/scripts/test_gates_procedencia.py` |
| US-155 y US-171 dejan de usar cifras obsoletas como criterio de aceptación | `plan.html` |
| El cuaderno deja de titular «la afirmación que el artículo puede sostener» | `index.html` |
| El mínimo de tres bloques, declarado como regla de publicación y no como frontera | docstring de `paired_interval` |
| **US-172 pierde la salida por «supuesto del artículo»** | `plan.html`, US-172 |

**Lo que NO cambió**: US-172 a US-175 siguen sin hacerse, y con ellas el criterio principal. El MDE
con t central. La multiplicidad de toda la superficie. La selección de predictores sobre datos
separados. Los trece artefactos `OBSOLETO` siguen sin regenerar.

**Para la ronda 6**, además de las tres partes:

1. **La clase, no el caso.** Por cada reparación de esta ronda, pregunta de qué clase es el defecto
   y busca los otros miembros de esa clase. Es lo que las rondas 4 y 5 encontraron dos veces.
2. **Los gates nuevos.** `paper-obsoletos-check` es un control recién nacido: intenta burlarlo.
3. **La propagación, otra vez**, con tres afirmaciones distintas de las que ya comprobaste.

---

## Estado para la ronda 7

La ronda 6 confirmó las cinco reparaciones de la 5 y dejó dos reservas. Las dos están cerradas:

| Qué se tocó | Dónde comprobarlo |
|---|---|
| El gate de publicación vigila **cifras**, no solo rutas: 1 007 valores de cuatro decimales extraídos de los propios artefactos obsoletos | `scripts/paper_obsoletos_check.py::cifras_distintivas`; test en negativo con `--docs` |
| Doce documentos con cuarentena, cuatro más que ayer — los encontró el gate | `docs/paper/*.md` |
| El gate escanea también el **cuaderno público**, donde una cifra obsoleta hace más daño: encontró 22 | `index.html`, `plan.html`, con banner visible y `data-cuarentena` |
| Las revisiones recibidas quedan exentas por **archivo ajeno**: marcarlas sería editar lo que alguien escribió | `ARCHIVO_AJENO` |
| El preregistro deja de anclar una expectativa en el MDE obsoleto (0,0326) | `docs/paper/preregistro-v2-borrador.md` §5 |
| Las cuatro cifras de la enmienda 3, marcadas como pendientes de recalcular antes de firmar | ídem §7 |
| «Se había refutado» retirado de los tres sitios que quedaban | preregistro §1, `index.html`, documento de revisores |

**Lo que NO cambió**: US-172 a US-175, y con ellas el criterio principal. El MDE con t central. La
multiplicidad de toda la superficie. La selección de predictores sobre datos separados. Los trece
artefactos `OBSOLETO`, sin regenerar.

**Para la ronda 7**, además de las tres partes:

1. **El gate de cifras es nuevo y por tanto sospechoso.** Dos cosas ya sabidas y dichas, para que
   no las cuentes como hallazgo: **no detecta redondeos ni paráfrasis** —0,0326 escrito «0,033» o
   «3,3 %» se le escapa, y bajar a tres decimales lo llenaría de coincidencias—, y esa frontera está
   escrita en su docstring. Lo que sí buscamos: si se le escapa una copia LITERAL por alguna vía que
   no vimos. Al escanear el cuaderno público encontró 22 cifras que la versión anterior no veía.
2. **La clase, no el caso**, otra vez: por cada reparación de esta ronda, busca los demás miembros
   de su clase.
