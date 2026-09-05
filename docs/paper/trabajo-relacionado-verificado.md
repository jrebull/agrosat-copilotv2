# Trabajo relacionado, verificado en fuente primaria

**Estado**: EVIDENCIA PREVIA AL MANUSCRITO. Este documento no es la sección 2 del artículo: es lo
que se comprobó, dónde, y qué se puede escribir con ello. La integración en primera página es
US-144 y **no está hecha**.

**Regla que gobierna el documento**: cada fila dice qué afirmación **sí** sostiene su referencia y
cuál **no**, con la redacción permitida y la prohibida. Nace de un caso real: este proyecto citó a
Jones et al. para respaldar lo contrario de lo que demuestra, y la auditoría lo encontró.

---

## 1. Alcance y fecha

| | |
|---|---|
| Búsqueda sistemática | **2 de septiembre de 2026**, 61 consultas, 631 resultados |
| Frentes | `F1` ensambles y arbitraje · `F2` cardinalidad y selectiva · `F3` contexto espacial y modelos de fundación · `F4` copiloto LLM sobre observación de la Tierra |
| Búsqueda manual | 6 consultas registradas el mismo día |
| Verificación adicional | **4 y 5 de septiembre de 2026**, en fuente primaria, de las referencias que sostienen la novedad |
| Registro | `reports/paper_micai/fase0/search_log.csv` y `search_log_manual.csv`; respuestas crudas en `fase0/raw/` |

## 2. Fuentes primarias y sesgos conocidos

Consultadas: **arXiv** (32 consultas), **OpenAlex** (15), **Semantic Scholar** (14) y búsqueda web
manual (6). Para las referencias de §4 a §6 se leyó además el documento: el PDF de arXiv extraído
con `pdftotext`, la página de actas de PMLR o NeurIPS, y el BibTeX oficial de CVF.

**Sesgos que sabemos que tiene esta búsqueda**, y que hay que declarar en el artículo:

- **Idioma**: solo inglés. Literatura agronómica en otros idiomas queda fuera.
- **Indexación**: las tres fuentes indexan mal las actas de congresos de teledetección sin DOI y la
  literatura gris de agencias, que en este dominio es abundante.
- **Vocabulario**: las consultas se escribieron con el vocabulario de clasificación selectiva. Un
  trabajo que haga lo mismo con otras palabras puede no aparecer — **y de hecho pasó**: Rey et al.
  implementa rechazo por incertidumbre sin usar ninguno de esos términos, y se encontró por otra
  vía.
- **Ventana**: el barrido cubre 2019–2026 en los cuatro frentes, más los clásicos que se buscaron
  por nombre. Un precedente anterior a 2019 fuera de los clásicos conocidos puede faltar.

**Un resumen de búsqueda no es fuente primaria.** Durante esta verificación un resultado de
búsqueda devolvió el orden de autores de SHRUG-FM equivocado, y la página de arXiv y el BibTeX de
CVF lo desmintieron. Toda fila de §3 se comprobó en el documento o en su página de actas.

## 3. Tabla por referencia

Toda clave existe en `paper/micai2027/refs-candidates.bib`.

| clave | lo que **sí** sostiene | límite metodológico | redacción **permitida** | redacción **prohibida** | identificador |
|---|---|---|---|---|---|
| `rey2025uncertaintyeo` | Rechazo de píxeles por umbral de incertidumbre sobre PASTIS con TSViT, UNET3D y U-TAE, con el intercambio precisión–recall explícito | Lo evalúa como detección de error y de ruido, no como un mecanismo de decisión entre alternativas | «implementa rechazo por incertidumbre y explicita el intercambio precisión–recall, sin conectarlo con clasificación selectiva ni riesgo–cobertura» | «ignora la abstención» · «no rechaza» · «no reconoce el compromiso» | arXiv 2510.19586 |
| `ha1997classselective` | El rechazo **selectivo de clases** es el compromiso óptimo entre error y número medio de clases seleccionadas | Teoría de decisión con las posteriores **supuestas conocidas** | «el marco existe desde 1997 y nuestra contribución no es el marco» | «proponemos predecir conjuntos» · «es un marco nuevo» | 10.1109/34.601248 |
| `mortier2021setvalued` | Algoritmos para el subconjunto Bayes-óptimo de máxima utilidad esperada, con utilidad función del tamaño | La utilidad depende del **tamaño** del conjunto; no hay afectado ni reparto | «la forma de nuestra utilidad ya existe; lo que cambia es de dónde sale su precio» | «la cardinalidad como coste es nuestra» | 10.1007/s10618-021-00751-x |
| `chzhen2021setvalued` | Revisión unificada de las formulaciones de clasificación con valores de conjunto y sus compromisos | Óptimos a muestra infinita y principio *plug-in*; no toca el coste por afectado | «las formulaciones están sistematizadas; nuestra aportación no es una formulación nueva» | «no existe un marco unificado» | arXiv 2102.12318 |
| `ghassemi2025howdetail` | El **tamaño del catálogo** como variable de diseño medida: de 52 clases LUCAS, un esquema de **26** equilibra exactitud y detalle | Es sobre cobertura del suelo con Random Forest, no sobre mecanismos de renuncia | «el catálogo como variable de diseño ya está medido en la literatura» | «somos los primeros en tratar el catálogo como variable» | 10.3390/rs17081379 |
| `chow1970reject` | El compromiso error–rechazo y su regla óptima | Rechazo **simple**: se abstiene o responde, sin conjuntos ni clases gruesas | «el caso degenerado de Ha» | «cubre los cuatro mecanismos» | 10.1109/tit.1970.1054406 |
| `jones2020selectivedisparities` | La clasificación selectiva **magnifica** disparidades entre grupos; abstenerse más puede **empeorar** algunos | Grupos demográficos en visión y lenguaje, no clases agronómicas ni parcelas | «el reparto desigual del rechazo está documentado fuera de nuestro dominio» | «demuestra que abstenerse reparte mejor» · cualquier lectura con el signo invertido | ICLR 2021, OpenReview `N0M_4BkQ05i` |
| `gonzalezcalabuig2026shrugfm` | Único trabajo EO revisado que dice **«abstain»** con umbrales interpretables | Cero tareas agrícolas: cicatriz de incendio, inundación, deslizamiento | «la abstención llega a EO, y no por la vía agrícola» | «nadie abstiene en observación de la Tierra» | CVPR 2026 Workshops, arXiv 2511.10370 |
| `carvalho2023opensetcrop` | Reconocimiento de cultivos en conjunto abierto por exposición a *outliers* | Rechaza lo **desconocido**; es otra pregunta que repartir un presupuesto fijo de promesa | «el rechazo por novedad está resuelto aparte y queda fuera de alcance» | «cubre nuestro caso» | 10.1109/LGRS.2023.3244532 |
| `gimenez2023rejection` | Métodos de rechazo aplicados a cartografía de vegetación | Hiperespectral aerotransportado, no series Sentinel por parcela | «hay precedente de rechazo en cartografía de vegetación» | «hay precedente en mapeo de cultivos por parcela» | 10.1080/01431161.2023.2240520 |
| `turkoglu2021hierarchies` | Jerarquías de etiquetas multiescala para mapeo de cultivos desde series temporales | Usa la jerarquía para **entrenar mejor**, no como mecanismo de renuncia en inferencia | «el retroceso taxonómico tiene precedente como estructura de etiquetas» | «ya se usa como mecanismo de renuncia» | 10.1016/j.rse.2021.112603 |
| `barriere2024hierarchical` | Fusión jerárquica de señales satelitales, rotacionales y contextuales para clasificar cultivos | La jerarquía es de fusión de información, no de granularidad de la respuesta | «la jerarquía de cultivos se explota, con otro fin» | «predice a granularidad variable como nosotros» | 10.1016/j.rse.2024.114110 |
| `geifman2017selective`, `geifman2019selectivenet`, `liu2019deepgamblers`, `pugnana2022aucselective` | Clasificación selectiva profunda: garantía de riesgo, arquitectura con rechazo integrado, pérdida tipo apuesta, y criterio por AUC | Todos abstienen **por parcela**; ninguno emite conjuntos ni retrocede a una clase gruesa | «la abstención aprendida está desarrollada; nosotros no la reimplementamos» | «comparamos contra el estado del arte en abstención aprendida» — **está fuera de alcance por decisión propia** | actas NeurIPS y PMLR |

## 4. Rey et al. 2025: el precedente más cercano

Es el trabajo más próximo y conviene decir exactamente por qué, porque la versión anterior de esta
frase era falsa.

**Lo que hace**, verificado en el PDF: usa **PASTIS** y **nuestras tres arquitecturas** —TSViT,
UNET3D, U-TAE—, **rechaza píxeles por umbral de incertidumbre**, y **explicita el intercambio**:
no responder reduce la exhaustividad y puede aumentar la precisión si el píxel rechazado se habría
clasificado mal. Eso es el compromiso riesgo–cobertura, hecho y reconocido.

**Lo que no hace**, y es el hueco:

1. **No lo conecta con la clasificación selectiva.** Los términos *abstention*, *abstain*,
   *selective classification*, *reject option* y *risk-coverage* no aparecen; las dos apariciones
   de *coverage* son cobertura geográfica. Lo dibuja como precisión–recall.
2. **No compara los cuatro mecanismos como acciones alternativas.** Rechazar es la única acción;
   no hay recorte de catálogo, ni conjunto de etiquetas, ni retroceso a clase gruesa.
3. **No modela quién soporta la pérdida.** No hay afectado, ni coste por tipo de error, ni reparto.

**Redacción permitida**: «El trabajo más cercano rechaza por incertidumbre sobre nuestro mismo
banco y nuestras mismas arquitecturas, y explicita el compromiso; lo que no hace es situarlo en el
marco de la decisión selectiva ni comparar las alternativas entre sí.»
**Prohibida**: cualquier variante de «ignora la abstención».

## 5. Ha, Mortier y Chzhen: el marco ya existe

**El marco de predicción con valores de conjunto no es nuestra contribución, y el artículo lo dice
en primera página.** Ha (1997) da el criterio de optimalidad, Mortier et al. (2021) los algoritmos
para el subconjunto de máxima utilidad esperada, y Chzhen et al. (2021) la sistematización de las
formulaciones.

**La diferencia que sí podemos reclamar**, y solo esta: en los tres la utilidad o el coste es una
función del **tamaño** del conjunto. Nuestro artículo sustituye ese tamaño por una **tabla de
pérdidas por acción, resultado y afectado obtenida de usuarios reales** (US-172), y mide **el
reparto** —quién paga cada mecanismo— que ese marco no mide.

**Mientras US-172 no esté hecha, esta diferencia es un plan y no un resultado**, y así debe
escribirse.

## 6. Ghassemi et al. 2025: granularidad del catálogo

Trata el tamaño del catálogo como **variable de diseño medida**: sobre 52 clases LUCAS, un esquema
de 26 equilibra exactitud y detalle. Es el precedente directo de «cuántas clases prometer es un
punto de operación», y **desactiva** cualquier afirmación de que seamos los primeros en tratarlo
así. Lo que allí es cobertura del suelo con Random Forest, aquí es un mecanismo entre cuatro,
puntuado bajo una pérdida declarada.

## 7. Párrafo candidato para la primera página

> **BORRADOR NO INTEGRADO.** Texto propuesto para US-144. No está en ningún manuscrito y puede
> cambiar entero.

> Prometer menos para acertar más no es una idea nueva. El compromiso entre error y rechazo se
> formalizó en 1970, y su generalización a la selección de un **subconjunto** de clases —con el
> criterio de optimalidad que la gobierna— en 1997; la literatura reciente da algoritmos eficientes
> para el subconjunto de máxima utilidad esperada y una sistematización de las formulaciones. En
> mapeo de cultivos, el tamaño del catálogo ya se ha medido como variable de diseño, y el trabajo
> más cercano al nuestro rechaza píxeles por incertidumbre sobre el mismo banco y las mismas
> arquitecturas, explicitando el compromiso entre precisión y exhaustividad. Lo que ese trabajo no
> hace —y lo que este artículo se propone— es situar esas decisiones en un mismo marco: tratar el
> recorte del catálogo, la abstención, el conjunto de etiquetas y el retroceso taxonómico como
> **cuatro acciones alternativas**, puntuarlas con una pérdida obtenida de quienes usan el mapa en
> vez de con el tamaño del conjunto, y medir **a costa de quién** opera cada una.

## 8. Fórmula de novedad, limitada

**Se escribe así, y nunca «nadie ha…»:**

> **No encontramos trabajo que** compare el recorte de catálogo, la abstención por parcela, la
> predicción con conjuntos y el retroceso taxonómico **como acciones alternativas del mismo
> problema de decisión**, bajo una pérdida declarada por acción, resultado y afectado, con
> contabilidad del reparto entre clases.

**Universo y procedimiento**, que van con la frase:

> Búsqueda en arXiv, OpenAlex y Semantic Scholar el 2 de septiembre de 2026 —61 consultas
> automáticas y 6 manuales sobre cuatro frentes, 631 resultados—, restringida al inglés, con
> verificación en fuente primaria de las referencias que sostienen esta afirmación. Los términos de
> búsqueda proceden del vocabulario de la clasificación selectiva; un trabajo equivalente expresado
> con otro vocabulario podría no haber aparecido, **como de hecho ocurrió** con el precedente más
> cercano.

**No se declara exhaustividad.** La frase acota lo que buscamos y cómo, no lo que existe.
