# Novedad del ángulo A — artículo MICAI 2027

**Fase**: 0 de [`docs/plan-micai-2027.md`](../plan-micai-2027.md).
**Fecha de ejecución**: 2026-09-02.
**Responsable**: Javier A. Rebull-Saucedo. **Decisión conjunta pendiente**: Arthur Jafed Zizumbo Velasco.
**Veredicto**: reencuadre. Registrado en [`ADR-013`](../decisions/ADR-013-angulo-micai.md).
**Artefactos sellados de esta fase**: `reports/paper_micai/fase0/` (registro de consultas, respuestas crudas, matriz y verificación).

> Regla aplicada en todo el documento: ningún título, autor, año ni cifra procede de la
> memoria. Los metadatos bibliográficos vienen de la API que se indica en cada fila y las
> cifras propias vienen del artefacto que se cita con su ruta.

---

## 1. La tesis candidata, en una frase

> **Ángulo A (tal como lo propuso el equipo).** En mapeo de cultivos desbalanceado, un
> árbitro heterogéneo por clase mejora sobre el promedio homogéneo, cuántas clases
> prometer es un punto de operación medible, y el contexto espacial no aporta sobre
> embeddings de fundación anuales.

Esa es la frase que se sometió a prueba. La sección 6 dice en qué queda.

---

## 2. Cómo se buscó

Ventana 2019-2026, cuatro frentes del plan, tres APIs con registro de consulta, fecha,
código de estado y número de registros. Todo el material crudo queda en
`reports/paper_micai/fase0/raw/` (un JSON por consulta) y el registro en
`search_log.csv`. El código que lo produce es
[`scripts/paper_micai_lit_search.py`](../../scripts/paper_micai_lit_search.py).

| Fuente | Consultas | Con respuesta (HTTP 200) | Con bloqueo (HTTP 429) | Registros devueltos |
|---|---|---|---|---|
| arXiv (API Atom) | 32 | 32 | 0 | 252 |
| Semantic Scholar (Graph API) | 14 | 1 | 13 | 25 |
| OpenAlex (API works) | 15 | 15 | 0 | 354 |

| Frente | Consultas ejecutadas | Registros devueltos |
|---|---|---|
| Stacking heterogéneo y arbitraje por clase en series temporales satelitales | 16 | 127 |
| Número de clases y clasificación selectiva en cobertura del suelo | 15 | 133 |
| Contexto espacial sobre embeddings de modelos de fundación EO | 16 | 191 |
| Copilotos LLM anclados para observación de la Tierra | 14 | 180 |

### 2.1 Tres desviaciones respecto del plan, declaradas

1. **Semantic Scholar limita el caudal.** La API de Semantic Scholar sin clave comparte
   una cuota global y devolvió HTTP 429 en la mayoría de las consultas, incluso con
   reintentos y espera creciente. Las que sí respondieron están en el registro con su
   código 200. Para no dejar el frente 1 sin literatura de revista se añadió **OpenAlex**
   como tercera fuente, que cubre *Remote Sensing*, ISPRS y JAG y responde sin clave.
2. **Google Scholar no tiene API.** Sus consultas se ejecutaron a mano desde el buscador y
   quedan registradas con fecha en `search_log_manual.csv`. Se listan aquí íntegras porque
   una de ellas es la que cambió el veredicto.
3. **Un reintento destruyó tres respuestas buenas de Semantic Scholar.** La primera pasada
   obtuvo 3 respuestas con código 200; un reintento posterior, pensado para recuperar las
   bloqueadas, volvió a recibir 429 y sobrescribió aquellas tres filas y sus JSON crudos.
   El fallo estaba en la fusión del registro, que reemplazaba por clave sin mirar el
   resultado. Ya está corregido en `scripts/paper_micai_lit_search.py`: un fallo no puede
   pisar un acierto, ni en el CSV ni en el JSON crudo. Lo que el registro muestra hoy para
   Semantic Scholar es el resultado del reintento, no el de la primera pasada, y así consta.
   No afecta a la matriz: ninguna de sus 43 entradas depende de una consulta de Semantic
   Scholar, y arXiv y OpenAlex respondieron a las 47 suyas.

| Fecha | Frente | Consulta | Enlaces | Qué aportó |
|---|---|---|---|---|
| 2026-09-02 | F1 | `stacking ensemble heterogeneous meta-learner crop type classification satellite image time series per-class 2023..2026` | 9 | La literatura de stacking en cultivos vive en revistas (ScienceDirect, MDPI, Frontiers), no en arXiv; ninguna entrada contrasta ensamble homogéneo frente a heterogéneo sobre los mismos folds. |
| 2026-09-02 | F2 | `selective classification reject option coverage accuracy trade-off land cover crop type mapping remote sensing` | 8 | Aparece la línea clásica de Chow y las curvas de rechazo; en teledetección agrícola se documenta que agrupar clases sube la exactitud, pero como advertencia metodológica y no como punto de operación. |
| 2026-09-02 | F3 | `AlphaEarth Foundations embeddings spatial context neighborhood features crop mapping evaluation 2026` | 9 | El ecosistema de 2026 alrededor de AlphaEarth está muy poblado; localiza el artículo original arXiv:2507.22291 y evaluaciones de riego, tomate y clima urbano sobre sus embeddings. |
| 2026-09-02 | F4 | `grounded LLM agent earth observation geospatial tools benchmark hallucination GeoAnalystBench 2025 2026` | 8 | Familia completa de bancos de agentes geoespaciales (GeoAnalystBench, GeoBenchX, GeoAgentBench, GISAgentBench, GeoNatureAgent); ninguno ancla el agente en un clasificador de cultivos propio. |
| 2026-09-02 | F1 | `"homogeneous" versus "heterogeneous" ensemble comparison crop classification remote sensing macro F1 per-class gain` | 9 | No se localiza ningún trabajo que cuantifique la diferencia por clase entre ensamble homogéneo y heterogéneo en clasificación de cultivos. |
| 2026-09-02 | F2 | `effect of number of crop classes legend granularity on classification accuracy crop type mapping trade-off study` | 10 | Hallazgo decisivo: Turkoglu et al. 2021 ya publica curvas de cobertura frente a confianza en cultivos con retroceso jerárquico. Es la consulta que cambió el veredicto. |

---

## 3. Matriz de trabajo relacionado

43 entradas, todas con identificador resuelto por API: 17 con DOI y el resto
con identificador arXiv verificado. El CSV con la respuesta completa de cada API está en
`reports/paper_micai/fase0/related_work_verified.csv` y el verificador es
[`scripts/paper_micai_ref_verify.py`](../../scripts/paper_micai_ref_verify.py), que marca
`NOT_FOUND` o `TITLE_MISMATCH` cualquier fila que no case con la fuente. La corrida de
cierre da 0 filas fuera de `OK`.

### Frente 1 · Stacking heterogéneo y arbitraje por clase en series temporales satelitales

| # | Trabajo (título, autores y año resueltos por API) | Método | Fortaleza | Límite | Hueco que deja |
|---|---|---|---|---|---|
| 1 | **Wolpert 1992** — Stacked generalization<br>`DOI 10.1016/s0893-6080(05)80023-1` · verificado en crossref | Meta-aprendizaje: un modelo de segundo nivel se entrena sobre las predicciones out-of-fold de los modelos base. | Formula el marco general del stacking y la necesidad de particiones libres de fuga para entrenar el meta-modelo. | Trabajo teórico de 1992, sin dominio geoespacial ni desbalance de clases; no analiza qué hace el meta-modelo clase por clase. | No responde si el meta-modelo promedia o arbitra por clase, ni cómo distinguir ambas cosas empíricamente. |
| 2 | **Pelletier 2019** — Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series<br>`DOI 10.3390/rs11050523` · verificado en crossref | Red convolucional temporal sobre series Sentinel-2 a nivel de píxel y de parcela para clasificación de cultivos. | Línea base temporal de referencia, reproducible y barata, comparada con Random Forest y redes recurrentes bajo el mismo protocolo. | Modelo único, sin ensamble; la métrica dominante es la exactitud global, no el F1-macro sensible a clases raras. | No estudia la combinación de arquitecturas heterogéneas ni el punto de operación del conjunto de clases. |
| 3 | **Rußwurm 2020** — Self-attention for raw optical Satellite Time Series Classification<br>`DOI 10.1016/j.isprsjprs.2020.06.006` · verificado en crossref | Transformador de autoatención sobre series ópticas crudas, comparado con redes recurrentes, convolucionales temporales y Random Forest. | Comparación controlada de familias de modelos sobre el mismo dato y evidencia de que la atención tolera la nubosidad. | Compara modelos individuales; el ensamble aparece como referencia y no hay análisis por clase. | Deja abierto cuánto gana un árbitro entrenado sobre esos modelos frente a un promedio homogéneo. |
| 4 | **Garnot 2021** — Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks<br>`arXiv:2107.07933` · verificado en arxiv | U-TAE con atención temporal ligera y el banco PASTIS con particiones espacialmente disjuntas. | Define el protocolo de evaluación del dominio: folds disjuntos, 18 clases y métricas densas y panópticas. | Evalúa arquitecturas individuales; no publica estudio de ensambles ni de cardinalidad de clases. | El banco existe, pero nadie lo usa para medir arbitraje heterogéneo por clase con pruebas pareadas. |
| 5 | **Tarasiou 2023** — ViTs for SITS: Vision Transformers for Satellite Image Time Series<br>`arXiv:2301.04944` · verificado en arxiv | Transformador que factoriza la atención en un eje temporal y otro espacial para series de imágenes satelitales. | Estado del arte en PASTIS; muestra que el eje temporal domina al espacial en cultivos. | Modelo único; no combina troncales ni analiza el desempeño en las clases minoritarias. | No mide cuánto queda por ganar combinando modelos decorrelacionados sobre sus propias posteriores. |
| 6 | **Mena 2024** — In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data<br>`arXiv:2403.16582` · verificado en arxiv | Comparación sistemática de estrategias de fusión multivista (entrada, característica y decisión) para clasificación de cultivos. | Barrido honesto de fusiones sobre datos globales que incluye la fusión a nivel de decisión, pariente cercano del voto. | Fusiona vistas del dato, no modelos heterogéneos entrenados por separado; sin meta-modelo entrenado ni delta por clase. | La fusión por decisión se evalúa en promedio; no se contrasta contra un árbitro entrenado ni se reporta el efecto por clase. |
| 7 | **Abdali 2023** — A Parallel-Cascaded Ensemble of Machine Learning Models for Crop Type Classification in Google Earth Engine Using Multi-Temporal Sentinel-1/2 and Landsat-8/9 Remote Sensing Data<br>`DOI 10.3390/rs16010127` · verificado en crossref | Ensamble en paralelo y en cascada de modelos clásicos de aprendizaje automático para tipo de cultivo sobre Google Earth Engine. | Ensamble heterogéneo real sobre datos operativos, con implementación reproducible en la nube. | Miembros tabulares clásicos, sin arquitecturas densas de series temporales; sin análisis por clase ni pruebas pareadas. | Combina modelos, pero no pregunta si la ganancia viene del promedio o del arbitraje por clase. |
| 8 | **Zhang 2022** — A Review of Ensemble Learning Algorithms Used in Remote Sensing Applications<br>`DOI 10.3390/app12178654` · verificado en crossref | Revisión de bagging, boosting, stacking y voto aplicados a teledetección. | Cataloga el uso real de cada familia de ensamble en el dominio y sus resultados típicos. | Revisión descriptiva; no unifica protocolos ni contrasta homogéneo contra heterogéneo bajo los mismos folds. | Confirma que el stacking se usa y que casi nadie publica el contraste controlado que aquí se propone. |

### Frente 2 · Número de clases y clasificación selectiva en cobertura del suelo

| # | Trabajo (título, autores y año resueltos por API) | Método | Fortaleza | Límite | Hueco que deja |
|---|---|---|---|---|---|
| 9 | **Chow 1970** — On optimum recognition error and reject tradeoff<br>`DOI 10.1109/tit.1970.1054406` · verificado en crossref | Regla óptima de rechazo por umbral sobre la probabilidad a posteriori. | Fija el marco clásico error-rechazo que sigue siendo la referencia de la clasificación selectiva. | Supone probabilidades verdaderas conocidas y decide muestra a muestra; no contempla retirar clases enteras. | No existe el análogo por clase: qué pasa si en lugar de rechazar muestras se recorta el conjunto de etiquetas. |
| 10 | **Geifman 2017** — Selective Classification for Deep Neural Networks<br>`arXiv:1705.08500` · verificado en arxiv | Selector sobre una red ya entrenada con garantía de riesgo para una cobertura objetivo. | Formaliza la curva riesgo-cobertura y da garantías estadísticas para la cobertura elegida. | Rechazo por muestra en visión genérica; sin dominio geoespacial ni clases desbalanceadas por naturaleza. | La cobertura se recorta por confianza, nunca por decisión sobre qué clases se prometen en producción. |
| 11 | **Geifman 2019** — SelectiveNet: A Deep Neural Network with an Integrated Reject Option<br>`arXiv:1901.09192` · verificado en arxiv | Arquitectura con cabeza de selección entrenada de forma conjunta para una cobertura objetivo. | Optimiza predicción y rechazo a la vez y mejora la frontera riesgo-cobertura frente al umbral posterior. | Exige reentrenar; en teledetección agrícola obligaría a reentrenar cada miembro del ensamble. | No compara recortar cobertura por muestra contra recortar el espacio de etiquetas a igual cobertura. |
| 12 | **Ziyin 2019** — Deep Gamblers: Learning to Abstain with Portfolio Theory<br>`arXiv:1907.00208` · verificado en arxiv | Clase de abstención adicional entrenada con una pérdida derivada de la teoría de carteras. | Abstención sin hiperparámetro explícito de cobertura y con interpretación económica del riesgo. | Abstención por muestra; la clase extra compite con las clases raras, justo las críticas en cultivos. | No dice nada sobre cuántas clases prometer cuando el soporte por clase es muy desigual. |
| 13 | **Pugnana 2022** — AUC-based Selective Classification<br>`arXiv:2210.10703` · verificado en arxiv | Criterio de selección que optimiza el área bajo la curva en lugar de la exactitud bajo cobertura. | Muestra que la métrica objetivo cambia la frontera de selección, algo decisivo cuando importa la clase rara. | Escenario binario y tabular; no aborda F1-macro multiclase ni el desbalance extremo del mapeo de cultivos. | Falta el equivalente multiclase con F1-macro, que es la métrica que gobierna este dominio. |
| 14 | **Fischer 2023** — Precision and Recall Reject Curves for Classification<br>`arXiv:2308.08381` · verificado en arxiv | Curvas de precisión y exhaustividad frente a la tasa de rechazo, en lugar de solo exactitud. | Expone que la exactitud bajo rechazo oculta el comportamiento en las clases minoritarias. | Datos tabulares de referencia; sin aplicación a teledetección ni a conjuntos de etiquetas jerárquicos. | La curva se traza sobre rechazo por muestra; no existe la versión por clase retirada. |
| 15 | **Hasan 2023** — Survey on Leveraging Uncertainty Estimation Towards Trustworthy Deep Neural Networks: The Case of Reject Option and Post-training Processing<br>`arXiv:2304.04906` · verificado en arxiv | Revisión de estimación de incertidumbre y de opción de rechazo posterior al entrenamiento. | Mapa completo del estado del arte del rechazo y de las técnicas que no exigen reentrenar. | Revisión generalista; los ejemplos de teledetección son marginales y no tratan cultivos por parcela. | Confirma que el rechazo por muestra es el estándar y que la cardinalidad de etiquetas no se estudia como palanca. |
| 16 | **Hendrickx 2024** — Machine learning with a reject option: a survey<br>`DOI 10.1007/s10994-024-06534-x` · verificado en crossref | Revisión unificada del rechazo por ambigüedad y por novedad, con taxonomía de métodos y métricas. | Referencia canónica y actual del área; define con precisión cobertura, riesgo y sus curvas. | Todos los mecanismos revisados operan muestra a muestra; ninguno reduce el conjunto de etiquetas prometidas. | Deja explícito el hueco: recortar el espacio de clases no figura en la taxonomía de mecanismos de rechazo. |
| 17 | **Jones 2020** — Selective Classification Can Magnify Disparities Across Groups<br>`arXiv:2010.14134` · verificado en arxiv | Análisis de cómo el rechazo por confianza degrada a los subgrupos peor representados. | Resultado negativo sólido: recortar cobertura por confianza puede empeorar al grupo débil en vez de protegerlo. | Los subgrupos son demográficos, no clases minoritarias de un mapa de cultivos. | Sugiere, sin medirlo en este dominio, que el rechazo por confianza sacrificaría justo las clases raras. |
| 18 | **Turkoglu 2021** — Crop mapping from image time series: Deep learning with multi-scale label hierarchies<br>`arXiv:2102.08820` · verificado en crossref | convSTAR multiescala que predice tres niveles de una jerarquía de cultivos y retrocede al nivel grueso cuando la confianza no alcanza un umbral. | Es el trabajo previo más cercano: publica curvas de cobertura frente a confianza y demuestra que retroceder a etiqueta gruesa sube la cobertura sin perder exactitud. | El mecanismo es retroceso jerárquico por píxel, la métrica es exactitud global sobre el área cubierta y la evaluación usa un solo fold sin intervalos. | Nadie contrasta ese retroceso jerárquico contra retirar clases enteras de un conjunto plano a igual cobertura, ni lo mide en F1-macro con intervalos pareados. |
| 19 | **Li 2021** — Cost-effective Land Cover Classification for Remote Sensing Images<br>`arXiv:2107.12016` · verificado en arxiv | Clasificación de cobertura del suelo con presupuesto acotado de etiquetado y de cómputo. | Trata el coste operativo como restricción de diseño y no como nota al pie. | El presupuesto es de etiquetas y de cómputo, no del conjunto de clases que se entrega al usuario. | No convierte el número de clases en un punto de operación medible frente a la cobertura. |
| 20 | **Schneider 2021** — EuroCrops: A Pan-European Dataset for Time Series Crop Type Classification<br>`DOI 10.14459/2021mp1615987` · verificado en openalex | Armonización de registros parcelarios europeos mediante la taxonomía jerárquica HCAT. | Hace explícito que el conjunto de clases es una decisión de armonización y no un dato natural. | Aporta la jerarquía pero no mide el coste en calidad de operar en cada uno de sus niveles. | La jerarquía existe; falta la curva que diga qué se gana y qué se pierde al subir o bajar de nivel. |
| 21 | **Claverie 2026** — EuroCrops v2.0: multi-annual harmonized parcel level crop type data linked to European Union-wide survey, statistical, and Earth Observation products<br>`DOI 10.5194/essd-18-4075-2026` · verificado en crossref | Conjunto parcelario europeo multianual armonizado y enlazado a productos estadísticos y de observación de la Tierra. | Convierte el conjunto de clases en un objeto explícito, versionado y trazable a la declaración administrativa. | Aporta taxonomía y dato, no el análisis de qué nivel conviene entregar en producción. | La jerarquía está disponible; el punto de operación sobre ella sigue sin medirse. |
| 22 | **Rußwurm 2023** — End-to-end learned early classification of time series for in-season crop type mapping<br>`DOI 10.1016/j.isprsjprs.2022.12.016` · verificado en crossref | Clasificación temprana aprendida que decide cuándo dejar de observar la serie. | Precedente claro de un punto de operación explícito en mapeo de cultivos: precocidad frente a exactitud. | El eje del compromiso es el tiempo de observación, no el número de clases prometidas. | Legitima la figura del compromiso operativo y deja libre el eje de la cardinalidad de clases. |
| 23 | **Lei 2025** — FineCrop: Mapping fine-grained crops using class-aware feature decoupling and parcel-aware class rebalancing with Sentinel-2 time series<br>`DOI 10.1016/j.isprsjprs.2025.07.041` · verificado en crossref | Desacoplamiento de rasgos por clase y reequilibrio consciente de la parcela para cultivos de grano fino. | Ataca de frente el desbalance en cultivos finos y reporta el desempeño por clase. | Resuelve el desbalance dentro del modelo; el conjunto de clases permanece fijo y completo. | Mejora las clases raras pero no plantea que a veces la decisión correcta sea no prometerlas. |
| 24 | **Donmez 2025** — Satellite remote sensing-based crop cover classification over Europe: accuracy of different methodological approaches<br>`DOI 10.1080/01431161.2025.2565837` · verificado en crossref | Comparación metodológica de clasificación de cobertura de cultivos en Europa con distintos niveles de agregación. | Documenta que las clases agrupadas alcanzan mayor exactitud y que el nivel de agregación invalida comparaciones ingenuas. | Lo trata como advertencia metodológica para comparar estudios, no como un punto de operación que se elige y se mide. | Reconoce el efecto de la agregación sin convertirlo en una frontera calidad-cobertura con intervalos. |

### Frente 3 · Contexto espacial sobre embeddings de modelos de fundación EO

| # | Trabajo (título, autores y año resueltos por API) | Método | Fortaleza | Límite | Hueco que deja |
|---|---|---|---|---|---|
| 25 | **Brown 2025** — AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data<br>`arXiv:2507.22291` · verificado en arxiv | Modelo de campo de embeddings que fusiona múltiples sensores en 64 dimensiones anuales de alcance global. | Embeddings listos para analizar, gratuitos en Earth Engine, que superan featurizaciones operativas en quince tareas de observación de la Tierra. | La evaluación interna es sobre todo de cobertura y uso del suelo; el embedding es anual y comprime el eje temporal. | No responde cuánto contexto espacial queda por explotar sobre el embedding en tareas de cultivo por parcela. |
| 26 | **Ma 2025** — Harvesting AlphaEarth: Benchmarking the Geospatial Foundation Model for Agricultural Downstream Tasks<br>`arXiv:2601.00857` · verificado en arxiv | Evaluación de AlphaEarth en rendimiento de cultivo, laboreo y cultivos de cobertura en Estados Unidos, con esquemas de transferencia espacial, de escala y anual. | Primer banco sistemático de un modelo de fundación de observación de la Tierra en tareas agrícolas, con modelos de teledetección entrenados como comparador. | Tres tareas en un solo país; la transferencia espacial falla en rendimiento pero es competitiva en laboreo y en cultivos de cobertura. | No evalúa clasificación de tipo de cultivo por parcela ni ensambles, y enuncia la limitación temporal sin cuantificarla en términos fenológicos. |
| 27 | **Corley 2026** — From Pixels to Patches: Pooling Strategies for Earth Embeddings<br>`arXiv:2603.02080` · verificado en arxiv | Comparación de once métodos de agrupamiento sin entrenamiento sobre embeddings de AlphaEarth, OlmoEarth y Tessera en EuroSAT-Embed. | Mide directamente cuánto contexto espacial dentro del parche se pierde con el promedio y cuánto se recupera con agrupamientos estadísticos y de covarianza. | La agregación es dentro del parche y la tarea es clasificación de escenas; no hay vecindad entre parcelas ni ensamble de modelos. | Es el trabajo más cercano en el eje espacial y obliga a separar el contexto intraparcela, que sí aporta, del contexto entre parcelas, que es donde cae nuestro nulo. |
| 28 | **Gilch 2026** — How to Embed Matters: Evaluation of EO Embedding Design Choices<br>`arXiv:2603.10658` · verificado en arxiv | Análisis sistemático de troncal, preentrenamiento, profundidad, agregación espacial y combinación de representaciones con NeuCo-Bench. | Aísla la agregación espacial como decisión de diseño con impacto medible en tareas de observación de la Tierra. | Tareas de banco genérico; sin series temporales de cultivo por parcela ni métrica macro con clases raras. | Confirma que la agregación importa, pero no en el escenario donde ya hay miembros temporales que consumen el eje del tiempo. |
| 29 | **Benavides-Martinez 2026** — What on Earth is AlphaEarth? Hierarchical structure and functional interpretability for global land cover<br>`arXiv:2603.16911` · verificado en arxiv | Ingeniería inversa del papel de cada dimensión del embedding por importancia de rasgos y ablación progresiva sobre cobertura del suelo. | Muestra una organización funcional jerárquica con dimensiones especialistas y generalistas. | El objeto es cobertura del suelo global, no cultivos por parcela; es interpretabilidad, no rendimiento de ensambles. | Da el lenguaje para explicar por qué el embedding ya trae contexto, sin medirlo en la tarea de cultivo. |
| 30 | **Lisaius 2026** — Embedding -based Crop Type Classification in the Groundnut Basin of Senegal<br>`arXiv:2601.16900` · verificado en arxiv | Comparación de TESSERA y AlphaEarth contra líneas base para tipo de cultivo en agricultura de pequeños productores. | Criterios explícitos de utilidad: desempeño, plausibilidad, transferibilidad y accesibilidad. | Una sola región, pocas clases, sin ensamble heterogéneo ni análisis de cobertura frente a número de clases. | Compara embeddings entre sí, no mecanismos de combinación sobre un mismo embedding. |
| 31 | **Adjei 2026** — Do Foundation Model Embeddings Improve Cross-Country Crop Yield Generalisation? A Leave-One-Country-Out Evaluation in Sub-Saharan Africa<br>`arXiv:2605.08113` · verificado en arxiv | Validación dejando un país fuera con embeddings Prithvi-EO frente a rasgos espectrales de Sentinel-2. | Publica un banco negativo reproducible: los embeddings congelados no ganan fuera del país de entrenamiento. | Es regresión de rendimiento, no clasificación, y el modelo de fundación es Prithvi, no AlphaEarth. | Refuerza que los resultados negativos con embeddings de fundación son publicables y necesarios. |
| 32 | **Lehmann 2026** — Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts<br>`arXiv:2608.16614` · verificado en arxiv | Evaluación de la calibración de dieciséis codificadores congelados bajo corrupciones y desplazamiento de distribución. | Demuestra que el orden por exactitud cambia al mirar calibración y que estos modelos se vuelven sobreconfiados bajo desplazamiento. | Clasificación y segmentación de banco; no aborda abstención ni conjunto de etiquetas operativo. | La calibración es requisito de cualquier curva calidad-cobertura y nadie la ha cruzado con la elección de clases en cultivos. |
| 33 | **Roberts 2017** — Cross‐validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure<br>`DOI 10.1111/ecog.02881` · verificado en crossref | Estrategias de validación cruzada por bloques para datos con dependencia espacial y temporal. | Referencia estándar de por qué la validación aleatoria infla el desempeño cuando hay autocorrelación espacial. | El dominio es ecología; no cubre ensambles ni modelos de fundación. | Justifica el protocolo de folds disjuntos, pero no dice cuánto contexto espacial queda utilizable tras respetarlo. |
| 34 | **Ploton 2020** — Spatial validation reveals poor predictive performance of large-scale ecological mapping models<br>`DOI 10.1038/s41467-020-18321-y` · verificado en crossref | Comparación de validación aleatoria frente a validación espacial en modelos de mapeo a gran escala. | Cuantifica el sesgo optimista de la validación aleatoria en mapeo ecológico. | No hay cultivos ni embeddings de fundación; el foco es el sesgo de evaluación, no la ganancia por contexto. | Delimita cómo debe medirse un aporte espacial para que resulte creíble. |
| 35 | **Shah 2026** — Embeddings based Anomaly Detection for Cleaning Global Crop Type Reference Datasets<br>`arXiv:2607.23908` · verificado en arxiv | Detección de anomalías sobre embeddings de un codificador de observación de la Tierra para limpiar etiquetas de referencia de cultivos. | Usa la vecindad local en el espacio de embeddings de forma operativa y honesta sobre el ruido de etiqueta. | El objetivo es limpiar etiquetas, no clasificar; no mide ganancia de contexto en la predicción final. | Muestra un uso de vecindad que sí aporta, lo que obliga a acotar nuestro nulo al refinamiento de posteriores entre parcelas. |
| 36 | **Chang 2024** — On the Generalizability of Foundation Models for Crop Type Mapping<br>`arXiv:2409.09451` · verificado en arxiv | Evaluación de modelos de fundación de observación de la Tierra en transferencia entre regiones para mapeo de tipo de cultivo. | Aborda de frente la generalización en la tarea exacta de este trabajo y no en cobertura del suelo. | Se centra en la transferencia entre regiones; no evalúa mecanismos de combinación ni contexto espacial añadido. | Confirma que la pregunta abierta es qué hacer sobre el embedding, no si el embedding sirve. |

### Frente 4 · Copilotos LLM anclados para observación de la Tierra

| # | Trabajo (título, autores y año resueltos por API) | Método | Fortaleza | Límite | Hueco que deja |
|---|---|---|---|---|---|
| 37 | **Huang 2025** — Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration<br>`arXiv:2511.19417` · verificado en arxiv | Marco multiagente en el que un modelo de visión y lenguaje perceptor, afinado con datos sintéticos, conversa con un modelo de lenguaje razonador congelado. | El razonador no se toca y aun así gana en tareas multimodales intensivas en conocimiento; el perceptor es intercambiable. | Solo visión genérica, matemáticas y medicina; ninguna evaluación en teledetección, agricultura ni geoespacial, y ninguna mención de alucinación. | El patrón está probado fuera de la observación de la Tierra; su traslado a percepción geoespacial con herramientas deterministas no lo ha evaluado nadie. |
| 38 | **Li 2025** — Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind<br>`arXiv:2505.12207` · verificado en arxiv | Banco de evaluación de modelos multimodales en teledetección agrícola con cuatro dimensiones y trece tipos de tarea. | Cobertura amplia de escenas y tareas, con evidencia de que estos modelos fallan en percepción espacial fina. | Solo evaluación, solo en inglés y sin partición de entrenamiento; la clasificación de tipo de cultivo por parcela no es su tarea central. | No evalúa un copiloto anclado a un clasificador especialista: mide el modelo multimodal solo, no la arquitectura perceptor-razonador. |
| 39 | **Zhang 2025** — GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation<br>`arXiv:2509.05881` · verificado en arxiv | Banco de flujos de análisis espacial y generación de código para modelos de lenguaje. | Tareas realistas de sistemas de información geográfica con evaluación de flujo completo y de código ejecutable. | Análisis espacial genérico; no hay percepción de cultivos ni anclaje a un modelo entrenado propio. | Evalúa al agente como programador geoespacial, no como comunicador de los resultados de un clasificador especialista. |
| 40 | **Yu 2026** — GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis<br>`arXiv:2604.13888` · verificado en arxiv | Banco de ejecución dinámica para agentes con herramientas en análisis espacial. | Ejecuta herramientas de verdad en lugar de comparar texto, y mide si se elige la herramienta correcta. | Herramientas geoespaciales de propósito general, sin herramientas de percepción agrícola ni métricas agronómicas. | Ningún banco evalúa un agente cuyas herramientas son modelos de cultivo entrenados y auditables. |
| 41 | **Lewis 2020** — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks<br>`arXiv:2005.11401` · verificado en arxiv | Generación condicionada a documentos recuperados de un índice denso. | Patrón canónico de anclaje que reduce las respuestas sin soporte en tareas intensivas en conocimiento. | Texto puro; la recuperación no tiene filtro espacial ni noción de vecindad geográfica. | El híbrido de filtro espacial más similitud vectorial sobre embeddings de observación de la Tierra no está evaluado en esta literatura. |
| 42 | **Yao 2022** — ReAct: Synergizing Reasoning and Acting in Language Models<br>`arXiv:2210.03629` · verificado en arxiv | Intercalado de razonamiento explícito y llamadas a herramientas. | Base del patrón de planificar y actuar que usan hoy los agentes con herramientas. | Dominios de texto y navegación, sin herramientas numéricas deterministas de teledetección. | No cubre el caso en que la herramienta es un clasificador cuya salida no debe ser reinterpretada por el modelo. |
| 43 | **Boudiaf 2026** — CropVLM: A Domain-Adapted Vision-Language Model for Open-Set Crop Analysis<br>`arXiv:2605.03259` · verificado en arxiv | Modelo de visión y lenguaje adaptado al dominio para análisis de cultivos de conjunto abierto. | Adaptación de dominio explícita para agricultura, con vocabulario abierto. | El propio modelo clasifica: no separa percepción de razonamiento ni ancla las cifras en herramientas deterministas. | Es la ruta contraria a nuestro patrón y nadie ha comparado ambas bajo el mismo protocolo agronómico. |


---

## 4. Las tres referencias ancla: qué dicen de verdad

Las tres entradas del `.bib` del manuscrito tienen título y autores inventados. Esta
sección deja el registro de lo que dicen las fuentes reales, leídas de su texto completo.

### 4.1 arXiv:2511.19417 — Be My Eyes

| | |
|---|---|
| Título real | *Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration* |
| Autores | James Y. Huang, Sheng Zhang, Qianchu Liu, Guanghui Qin, Tinghui Zhu, Tristan Naumann, Muhao Chen, Hoifung Poon |
| Publicado | 24 de noviembre de 2025, v1 |
| En el repositorio | `refs.bib` y `farslip_refs.bib` traen **dos títulos distintos, ambos inventados**, para la misma clave, y el autor como «Huang and colleagues» |

Qué dice, comprobado sobre el texto completo del PDF:

- El razonador está congelado, tal cual lo usa el manuscrito: «Note that BeMyEyes does not
  require any fine-tuning or architectural modification of the reasoner agent».
- **El perceptor sí se entrena.** Hay un pipeline de síntesis de datos con GPT-4o sobre
  imágenes de CoSyn-400K —«a large collection of computer-generated charts, diagrams, and
  figures»— que produce 12 145 preguntas multimodales con conversaciones simuladas, y un
  ajuste supervisado del perceptor con pérdida de entropía cruzada.
- **La palabra *hallucination* no aparece ni una vez en el artículo.** La cota de
  alucinación que el manuscrito le atribuye es una afirmación nuestra, no de la fuente.
- **No hay ninguna evaluación geoespacial.** Las cadenas «remote sensing», «agriculture»,
  «geospatial» y «satellite» no aparecen en el texto. Los bancos son MMMU, MMMU-Pro,
  MathVista, MathVision y MMMU-Med.

Consecuencia: «adoptamos este patrón al pie de la letra» es falso, porque no afinamos
perceptor; y el anclaje contra alucinación hay que defenderlo con evidencia propia. A
cambio, el hueco es genuino: nadie ha llevado el patrón a percepción geoespacial con
herramientas deterministas.

### 4.2 arXiv:2601.00857 — Harvesting AlphaEarth

| | |
|---|---|
| Título real | *Harvesting AlphaEarth: Benchmarking the Geospatial Foundation Model for Agricultural Downstream Tasks* |
| Autores | Yuchi Ma, Yawen Shen, Anu Swatantran, David B. Lobell |
| DOI de revista | 10.1016/j.jag.2026.105258 |
| En el repositorio | autor «Anonymous» y un título inventado sobre transferibilidad espacial para mapeo de cultivos |

Qué dice, comprobado sobre el texto completo:

- Evalúa **tres tareas en Estados Unidos**: predicción de rendimiento, mapeo de laboreo y
  mapeo de cultivos de cobertura. **No evalúa clasificación de tipo de cultivo.**
- La limitación de transferencia **depende de la tarea**: «AEF-based had competitive
  spatial transferability in tillage and cover crop mapping but consistently underperformed
  RS-based models in yield prediction».
- Sobre el eje temporal: «AEF lacks temporal sensitivity and therefore cannot support
  time-critical tasks such as in-season yield prediction».
- **No propone la adaptación few-shot como remedio.** Lo que propone es que versiones
  futuras generen «monthly/seasonal embeddings».

Consecuencias para el manuscrito: la frase «frozen global embeddings transfer poorly across
distant agronomic regions» generaliza más de lo que la fuente dice, y «few-shot adaptation
is required, in line with reported limits» le atribuye una recomendación que no hace. Más
grave para el ángulo A: la fuente contradice en el eje temporal la idea de que el embedding
anual codifica fenología.

### 4.3 arXiv:2505.12207 — AgroMind

| | |
|---|---|
| Título real | *Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind* |
| Primer autor | Qingmei Li (la clave del `.bib` inventa a «Ruan») |
| Versiones | v1 18-may-2025, v2 4-ago-2025, v3 13-ago-2025 |

Sobre los conteos, que el manuscrito usa: la **misma versión v3** se contradice consigo
misma. El resumen registrado en arXiv dice «27 247 QA pairs and 19 615 images» integrando
ocho conjuntos públicos; el texto completo de esa v3 dice «28 482 QA pairs and 20 850
images» integrando nueve conjuntos públicos y un conjunto parcelario privado. La v1 decía
25 026 y 15 556. La cifra 28 482 que usa el manuscrito es defendible **citando el texto
completo de la v3**, y conviene decirlo así en lugar de dejarla suelta.

Lo demás se confirma: es un banco **solo de evaluación**, **solo en inglés**, y la
clasificación de tipo de cultivo por parcela no es su tarea central. En consecuencia,
cualquier derivado bilingüe italiano-español es construcción nuestra y hoy no está medido.

---

## 5. Las tres patas del ángulo A frente a la literatura

### 5.1 Arbitraje heterogéneo por clase

**Qué hay.** El stacking en clasificación de cultivos está muy publicado: ensambles en
paralelo y cascada sobre Earth Engine, revisiones específicas de ensambles en teledetección
y comparativas de fusión multivista. El marco lo fija Wolpert en 1992.

**Qué no hay.** Ningún trabajo de la matriz contrasta ensamble **homogéneo** frente a
**heterogéneo** manteniendo fijos los mismos miembros, el mismo fold y el mismo arnés, con
delta por clase y prueba pareada. Las comparativas existentes cambian a la vez el conjunto
de miembros y el protocolo, de modo que no separan promedio de arbitraje.

**Qué podemos correr hoy en CPU.** `ml/eval/oof/` tiene diez tablas OOF por parcela para
Francia —ocho miembros más las dos ramas FarSLIP— cuya intersección es de **16 640
parcelas** en el fold 5 held-out, con el manifiesto declarando 18 clases y `code_version`
`086c4b53`. Sobre ellas se calcula media simple, voto ponderado y stacking, con bootstrap
pareado y McNemar, sin GPU.

**Veredicto parcial.** Hueco real pero estrecho. Sostiene un mecanismo, no un artículo.

### 5.2 Cuántas clases prometer

**Qué hay, y es más de lo que suponía la propuesta.** La clasificación selectiva tiene una
línea completa —Chow, Geifman y El-Yaniv, SelectiveNet, Deep Gamblers, curvas de rechazo,
y la revisión de Hendrickx de 2024— en la que **todos** los mecanismos recortan cobertura
muestra a muestra. Y en cultivos existe ya un precedente muy cercano: Turkoglu et al. 2021
predice tres niveles de una jerarquía y **retrocede a la etiqueta gruesa** cuando la
confianza no alcanza un umbral, publicando curvas de cobertura frente a confianza; su
propio texto dice que con umbral 0,9 «accuracy (computed over covered areas) stays the
same, coverage improves by ≈ 0.20 − 0.25». La literatura europea de mapeo, además, documenta
que agrupar clases sube la exactitud, aunque lo trate como advertencia para comparar
estudios y no como decisión de diseño.

**Qué no hay.** Nadie compara los **dos mecanismos de recorte de cobertura entre sí**:
retirar clases enteras de un conjunto plano frente a rechazar muestras por confianza, **a
igual cobertura** y bajo **F1-macro**, con intervalos pareados. Turkoglu mide exactitud
global sobre el área cubierta, sobre un solo fold y sin intervalos, y su mecanismo es
retroceso jerárquico, no retirada de clases.

**Qué podemos correr hoy en CPU.** `reports/ensemble/metrics/us043_honest_dropout_curve.csv`
ya trae la rama de retirada de clases sobre el fold 5 held-out: 18 clases dan F1-macro
0,7486 con las 16 640 parcelas; 12 clases dan 0,8573 con 14 925; 9 clases dan 0,9121 con
13 624; 8 clases dan 0,9201 con 13 311. Falta el comparador de rechazo por confianza a
igual cobertura y los intervalos, que son CPU pura sobre las mismas posteriores.

**Veredicto parcial.** Es la contribución central, pero solo con el comparador. Sin él, la
curva es una repetición de lo ya sabido.

### 5.3 El contexto espacial

**Qué hay.** El ecosistema alrededor de AlphaEarth es denso en 2026: interpretabilidad de
sus dimensiones, calibración bajo desplazamiento, limpieza de etiquetas por vecindad en el
espacio de embeddings y, sobre todo, dos trabajos que atacan directamente la agregación
espacial. Corley et al. 2026 comparan once métodos de agrupamiento sobre embeddings de
AlphaEarth y reportan que los esquemas ricos reducen la brecha de generalización geográfica
más de un 50 % respecto a la media y suben la exactitud hasta un 6 % en particiones
espaciales. Gilch et al. 2026 aíslan la agregación espacial como decisión de diseño con
impacto medible.

**Qué dice nuestro propio artefacto, leído con cuidado.**
`reports/ensemble/metrics/ec_neighborhood_result.json` **no** contiene un cero. El punto que
mejora ambos ejes (k = 10, alfa = 0,1) da F1-macro a 18 clases 0,7513, es decir **+0,0027**
sobre el campeón, y 0,9122 en france-9, **+0,0002**. El óptimo en el eje de 18 clases
(k = 10, alfa = 0,3) da **+0,0068** pero degrada france-9 en −0,0020. El propio JSON marca
`improves_18: true` y `material_both: false` contra un umbral de 0,01, sobre 16 640
parcelas y **sin intervalo de confianza**.

**Veredicto parcial.** La afirmación «el contexto espacial no aporta» es insostenible tal
como está escrita: hay trabajo previo que muestra que el contexto **dentro** del parche sí
aporta sobre estos mismos embeddings, y nuestro dato da un delta positivo pequeño, no un
cero. Lo que podrá sostenerse, **una vez calculado el intervalo que hoy no existe**, es un
enunciado acotado: el refinamiento por vecindad **entre parcelas** de las posteriores no
produce una mejora material sobre un ensamble ya apilado. Mientras el intervalo no esté
calculado, ni siquiera esa versión estrecha puede imprimirse.

---

## 6. Veredicto

**Reencuadre.** Criterio del plan: hay contribución si existe al menos un experimento
ejecutable en CPU que aísle el mecanismo y ningún trabajo previo lo reporte en este dominio
con este protocolo.

- El experimento en CPU existe y está en disco: diez tablas OOF por parcela, 16 640
  parcelas en la intersección, artefactos de cardinalidad y de vecindad ya calculados.
- El protocolo no está reportado: el contraste entre mecanismos de recorte de cobertura a
  igual cobertura y bajo F1-macro no aparece en ninguna de las 43 entradas.
- Pero dos de las tres afirmaciones del ángulo A, tal como estaban redactadas, no
  sobreviven a la verificación.

Tesis reformulada:

> En mapeo de cultivos por parcela con clases desbalanceadas, una vez que las posteriores
> de miembros heterogéneos se combinan con un árbitro entrenado, la palanca que queda no es
> más contexto espacial sino la decisión de qué clases se prometen: retirar clases enteras
> del conjunto plano y el rechazo por confianza son dos mecanismos distintos de recortar
> cobertura, se pueden comparar a igual cobertura bajo F1-macro, y el refinamiento por
> vecindad entre parcelas no aporta una mejora material sobre el ensamble apilado.

El reparto de papeles, las afirmaciones prohibidas y las reglas de decisión
pre-registradas quedan en [`ADR-013`](../decisions/ADR-013-angulo-micai.md).

---

## 7. Riesgos que pueden tumbar este veredicto

| Riesgo | Qué lo dispara | Qué se hace |
|---|---|---|
| El nulo de vecindad deja de ser nulo | El intervalo bootstrap del delta excluye el cero | Regla R1 del ADR: se reporta como mejora pequeña y no accionable, con su cifra; no se retira el experimento |
| La retirada de clases no domina al rechazo por confianza | El comparador a igual cobertura empata o pierde | Regla R2: la contribución pasa a ser el protocolo de comparación y se reporta el empate |
| Un revisor considera Turkoglu 2021 demasiado cercano | La diferencia de mecanismo y de métrica no queda explícita en el texto | El trabajo relacionado abre con esa comparación, con la cita, y el artículo declara la diferencia en la introducción |
| La contribución se percibe estrecha para MICAI | Una sola región, un año, un fold held-out | Se declara el alcance en el título y en limitaciones; la fuerza está en el protocolo pareado, no en la escala |
| Cifras que hoy no tienen artefacto | Bretaña, WorldCereal, DE4, métricas de herramientas | Fase 1: lo que no tenga artefacto sellado sale del texto |

---

## 8. Qué cambia en el plan

- **Fase 2** gana un experimento obligatorio: rechazo por confianza a igual cobertura como
  comparador de la curva de cardinalidad. Sin él la contribución central no es defendible.
- **Fase 2** debe producir intervalos para el barrido de vecindad, no solo el delta puntual.
- **Fase 2** no hereda la curva de retirada de clases del CSV: el sellado de la fase 1
  muestra que `us043_honest_dropout_curve.csv` y `us043_farslip_grid.csv` **no tienen driver
  versionado**, aunque su cálculo sí lo está (`ml.eval.per_class_analysis` y
  `ml/ensemble/`). La curva se vuelve a generar llamando a esas funciones desde las
  posteriores OOF selladas, y el CSV pasa a ser comprobación cruzada. El eje de cobertura
  ya reproduce: los siete valores de `n_parcels_fold5` salen exactos del ground truth
  sellado en `reports/paper_micai/fase1/parcel_gt_fold5.parquet`.
- **Fase 3** hereda cuatro afirmaciones prohibidas y su redacción correcta.
- **Fase 4** reordena Resultados: la curva calidad-cobertura pasa a ser 4.1.
- **Fase 5** parte de 43 referencias con identificador ya verificado; quedan por resolver
  los DOI de actas para las entradas que hoy solo tienen identificador arXiv.

---

## 9. Artefactos de esta fase

| Artefacto | Qué contiene |
|---|---|
| `reports/paper_micai/fase0/search_log.csv` | Una fila por consulta automática: frente, fuente, consulta, fecha, código HTTP, registros |
| `reports/paper_micai/fase0/search_log_manual.csv` | Las seis consultas de buscador ejecutadas a mano, con fecha y hallazgo |
| `reports/paper_micai/fase0/raw/` | Respuesta cruda de cada consulta, un JSON por consulta |
| `reports/paper_micai/fase0/search_candidates.csv` | Todos los candidatos devueltos, sin filtrar |
| `reports/paper_micai/fase0/related_work_matrix.csv` | La matriz redactada: método, fortaleza, límite y hueco |
| `reports/paper_micai/fase0/related_work_verified.csv` | La matriz con título, autores, año, DOI y estado devueltos por la API |
| `scripts/paper_micai_lit_search.py` | Ejecuta y sella la búsqueda |
| `scripts/paper_micai_ref_verify.py` | Verifica cada referencia contra arXiv, Crossref y OpenAlex |
| `reports/paper_micai/fase1/parcel_gt_fold5.parquet` | Las 16 640 etiquetas del fold 5 contra las que se mide todo, derivadas de PASTIS-R |
| `scripts/paper_micai_seal_fold5.py` | Deriva y sella ese ground truth, sus centroides y su procedencia |
