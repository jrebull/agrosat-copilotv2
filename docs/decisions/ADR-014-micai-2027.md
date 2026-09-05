# ADR-014 — MICAI 2027: alcance, modelo y destino de lo heredado

**Estado**: BORRADOR. Redactado por Javier Rebull; **pendiente de la firma de Arthur Zizumbo**.
**Fecha del borrador**: 4 de septiembre de 2026
**Decisores**: Arthur Jafed Zizumbo Velasco (primer autor) · Javier A. Rebull-Saucedo (segundo autor)
**Fecha de corte**: **18 de septiembre de 2026.** Si no está firmado, se ejecuta el plan con las
decisiones por defecto marcadas **[POR DEFECTO]** en cada apartado, y esta línea es su registro.
**Sustituye a**: [ADR-013](ADR-013-angulo-micai.md), cuya tesis no sobrevivió a la fase 3.
**Enmienda 1**, 4 de septiembre de 2026: el §6 pasa de «qué predictor» a «qué regla de selección o panel», porque el valor por defecto anterior contradecía a US-139.
**Enmienda 2**, 4 de septiembre de 2026: el panel usa `xgb-alphaearth-remat-v1` y no el componente histórico desplegado, que queda como referencia descriptiva.
**Enmienda 3**, 4 de septiembre de 2026: se congela el panel en cinco miembros (§6 bis) y se fija el orden de ejecución (§7 bis).
**Enmienda 3**, 4 de septiembre de 2026: US-119 retira `anysat` y `utae` del conjunto elegible
inferencial mientras no se identifique la causa de su caída; `segformer` se conserva porque la
premisa de que también caía fue desmentida por el artefacto.
**Evidencia**: ocho rondas de auditoría externa en
[`docs/paper/respuesta-auditoria-externa.md`](../paper/respuesta-auditoria-externa.md) · el
preregistro en [`docs/paper/preregistro-v2-borrador.md`](../paper/preregistro-v2-borrador.md) · el
contrato del estimando en [`docs/paper/estimando-v1.json`](../paper/estimando-v1.json) · el ledger
de custodia en [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md)

---

## Contexto

Es el tercer cambio de tesis. Este ADR existe para que dentro de seis meses nadie reconstruya de
memoria el ángulo viejo, y para que el cambio tenga una fecha y una firma en vez de una costumbre.

Lo que obligó al cambio, con su evidencia y no con una impresión:

| Qué se creía | Qué se encontró | Dónde |
|---|---|---|
| El resultado central se sostenía | Su aparato tenía **tres defectos** —denominador móvil, punto de operación elegido dentro del bloque evaluado, remuestreo en la unidad equivocada—; **reparados en el código, sin regenerar los artefactos** | 25 filas `OBSOLETO` en el ledger |
| «A igual cobertura» era comparable | **No está definido** para dos de los cuatro mecanismos: un conjunto conforme y una clase gruesa se entregan siempre | `ml/eval/set_valued.py` |
| La cardinalidad servía de coste | Dos conjuntos del mismo tamaño no cuestan lo mismo, y el vacío no es gratis. **No hay función de pérdida** | US-172, sin hacer |
| Había potencia en la disparidad | **Ninguna** de las cuatro medidas declaradas la tiene con cinco bloques | `reports/paper_micai/potencia/` |

## Decisión

### 1 · El ángulo

> **Las cuatro maneras de que un mapa de cultivos prometa menos —recortar el catálogo, abstenerse,
> devolver un conjunto, retroceder a una clase gruesa— no son comparables por acierto, y ordenarlas
> exige declarar una pérdida. Este artículo declara esa pérdida con procedencia, sitúa los cuatro
> mecanismos en el espacio de costes resultante, y mide quién paga cada uno.**

**No se afirma todavía** que no se distingan en calidad ni que se distingan en reparto: las dos
cosas son hipótesis, y la primera exige además una prueba de equivalencia contra una banda que no
existe. El artículo dirá lo que salga.

### 2 · Contribuciones, en orden

1. **La descomposición del recorte de catálogo**: cuánto de la mejora aparente es el cambio de
   denominador y cuánto el método, con el control que lo revela. Es la candidata más firme y
   **está pendiente de regenerar** con el módulo reparado.
2. **El marco de conjuntos con pérdida declarada**: los seis mecanismos como predictores de valor
   conjunto, puntuados sobre la población completa, con la tabla de pérdidas de US-172 como moneda.
3. **La contabilidad de equidad**: quién paga cada mecanismo, por clase.
4. **El caso de despliegue real**, anonimizado, situado en ese espacio.

### 3 · Lo que baja de rango, y lo que se retira

| | |
|---|---|
| **Baja a método y apéndice** | La autopsia de nuestros propios errores. Las correcciones son obligatorias; la confesión no es una contribución |
| **Baja a hipótesis** | La disparidad como criterio principal: no tiene potencia medida y elegirla por tenerla sería escoger el resultado |
| **Se retira** | Cualquier afirmación de **transporte** a otras regiones o campañas: el estimando es condicional al conjunto de datos y eso es su precio declarado ([`estimando-v1.json`](../paper/estimando-v1.json)) |
| **Se retira** | El refinamiento por vecindad y el árbitro entrenado como contribución: salen del artículo |
| **Fuera de alcance, declarado** | Los mecanismos **aprendidos** de renuncia. Es una decisión nuestra de presupuesto y riesgo de sobreajuste, no una imposibilidad |

### 4 · Sede

**MICAI 2027, y solo MICAI. [POR DEFECTO]** Sin artículo de revista en paralelo. Hipótesis de
planificación de **20 páginas** —el tope de la edición anterior—, **a reverificar cuando salga la
convocatoria**; si baja, el recorte sale de resultados hacia el apéndice y **nunca de robustez**.

### 5 · Destino de lo heredado

Ninguno de los dos queda en limbo:

| Documento | Destino |
|---|---|
| Manuscrito heredado de **24 páginas** | **Informe técnico interno**, sin publicar. Su tesis es la de ADR-013, que no sobrevivió a la fase 3, y publicarlo como preprint pondría en circulación una afirmación que ya sabemos que no se sostiene. **[POR DEFECTO]** |
| Borrador retirado de **15 páginas** | **Archivo del repositorio**, con [`paper/micai/ESTADO.md`](../../paper/micai/ESTADO.md) delante diciendo por qué no se envía. **No se publica como preprint** por el mismo motivo: su cadena inferencial no es válida. **[POR DEFECTO]** |

**No se destruyen.** El registro de lo que se creyó es parte del método de este proyecto, y borrarlo
sería el mismo gesto que retirar una cifra y dejar la conclusión.

### 6 · Qué regla de selección o panel predeclarado usa el artículo

**No se declara un predictor ganador**, y no por prudencia: hoy no se puede declarar honestamente.
El ranking que usaríamos se calcula **sobre las mismas etiquetas con las que después se evalúa**
(`scripts/run_paper_micai_fase3.py`, defecto declarado en el código), y elegir con él sesga hacia
arriba todo lo que venga después.

> **Regla que se firma**: no se declara un predictor ganador. Se reporta el **panel elegible
> predeclarado** —al menos **tres predictores de familias distintas por banco**— y el predictor se
> trata como **factor de sensibilidad**. Solo podrá seleccionarse uno mediante **selección anidada
> independiente de la evaluación**. **[POR DEFECTO: el panel, sin ganador.]**

**Y el panel no contiene el componente que se desplegó, sino su reconstrucción.** Hay que decirlo
con todas las letras porque es fácil de pasar por alto:

> El panel inferencial usa **`xgb-alphaearth-remat-v1`**, una reconstrucción reproducible con
> identidad propia. **No es el componente histórico desplegado.** Este permanece
> `legacy_unverified`; se reporta descriptivamente su acuerdo global de **0,945433** y la
> divergencia concentrada en baja confianza —0,5474 por debajo de 0,5 de confianza frente a 0,9999
> por encima de 0,9—, pero **no se usa para inferencia**.

Fingir que los dos son el mismo predictor sería el error que US-118 existe para impedir, cometido
en el documento que fija el alcance.

**US-119 reduce el conjunto elegible de segmentación, pero no selecciona un ganador.** Para PASTIS
quedan `tsvit-pheno`, `tsvit-pheno-fullm`, `deeplabv3plus`, `segformer` y `unet`; `anysat` y `utae`
se conservan como descriptivos y no entran en inferencia mientras la causa de sus deltas de 0,3597
y 0,4010 siga sin identificar. La decisión y la identidad de los pesos están en
[`sanidad_miembros.json`](../../reports/paper_micai/us119/sanidad_miembros.json), no solo en esta
prosa. US-139 todavía debe congelar el panel final y mantener al menos tres familias distintas.

**Corrección del borrador anterior**: decía «el artículo reporta los dos predictores», y eso
contradice a US-139, que exige un panel de al menos tres de familias distintas por banco. Dos
predictores del mismo linaje no son un panel: son una comparación con el mismo sesgo dos veces. El
error venía de que la pregunta estaba mal formulada —«qué predictor es el del artículo»— y una
pregunta que presupone un ganador se contesta con un ganador.

### 6 bis · El panel, congelado

**Cinco miembros y cuatro familias**, declarados en [`panel-v1.json`](../paper/panel-v1.json) y en
§4.6 del preregistro, con `make panel-check` comprobando que ninguno esté excluido, que las familias
lleguen al mínimo y que la prosa diga lo mismo:

`unet` · `deeplabv3plus` · `segformer` · `tsvit-pheno` · `tsvit-pheno-fullm`

**El margen sobre el mínimo es de una familia.** Dos miembros son de la misma, y perder uno más deja
el panel por debajo de su propio criterio. El gate lo imprime en cada ejecución.

**Fuera, y la distinción importa**: `anysat` y `utae` por superar el umbral de US-119 con la causa
**sin identificar**, así que su exclusión es **reversible**; los dos FarSLIP y el `xgb-alphaearth`
histórico por procedencia, y eso no se revierte solo.

> **Precisión que se mantiene**: `segformer` mejora en fold 5 y **sigue dentro del panel**. Eso
> **no invalida el criterio del umbral**, que sigue vigente; demuestra que la lista previa de
> sospechosos estaba parcialmente equivocada. **Un miembro se excluye por superar el umbral**,
> nunca por figurar en aquella lista.

**`xgb-alphaearth-remat-v1`, decidido el 4 de septiembre**: entra como **candidato nuevo con
procedencia verificada**, **nunca como reproducción del desplegado** —no lo reproduce, y el
desacuerdo está donde el artículo trabaja—. Fuera del panel de segmentación densa, que es otra
ruta. El **XGB histórico se queda `legacy_unverified` y fuera de inferencia**. De ninguno de los dos
se declara ganador: el predictor es factor de sensibilidad.

### 7 bis · El orden de ejecución

1. Congelar el panel *(hecho)*.
2. **US-172** con pérdidas obtenidas de usuarios reales.
3. Congelar el preregistro.
4. Regenerar **una sola vez** los 25 artefactos `OBSOLETO`.
5. Multiplicidad y análisis final.
6. Manuscrito.

**Regenerar ahora provocaría otra ronda de artefactos obsoletos**, porque cambiarían de nuevo al
fijar la pérdida. Por eso se hace una vez y al final de los tres cierres.

### 7 · Qué NO autoriza este ADR

Ningún cálculo de las EPIC 20, 21, 22 ni 25 antes de que el preregistro esté firmado, y el
preregistro no se firma antes de US-172. Este ADR fija el alcance; no abre la caja.

## Consecuencias

**Aceptadas:**

- El artículo pierde la afirmación de transporte, que era su gancho más vendible.
- El calendario depende de una elicitación con personas reales, que no se puede acelerar.
- Veinticinco artefactos hay que regenerarlos antes de citar una sola de sus cifras.

**A cambio:**

- Cada cifra que se imprima tendrá procedencia verificable por un gate, no por una lectura.
- La pérdida la habrá dicho alguien que usa mapas de cultivos, no nosotros.
- Y si el resultado es nulo, el artículo del resultado nulo ya está escrito antes de verlo.

**El riesgo real, dicho:** que la elicitación se retrase y no haya artículo para MICAI 2027. El
criterio de no envío está en el §8 del preregistro y se evalúa en la fecha atada al sellado de
US-134, no el día del cierre de envíos.

## Firmas

| | |
|---|---|
| Arthur Jafed Zizumbo Velasco | `[PENDIENTE]` |
| Javier A. Rebull-Saucedo | `[PENDIENTE]` |

Mientras las dos digan `[PENDIENTE]` y no se haya alcanzado la fecha de corte, este ADR es un
borrador. Pasada la fecha de corte sin firma, valen las decisiones **[POR DEFECTO]** y se anota aquí
la fecha en que empezaron a valer.
