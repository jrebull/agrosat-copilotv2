# Qué artículo sale de este proyecto, escrito desde cero

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md): las produjo `ml/eval/paper_micai_coverage.py`
> cuando aun tenia los tres defectos —denominador movil, punto de operacion elegido dentro del
> bloque evaluado, y remuestreo a nivel de parcela—. **Ninguna de esas cifras entra en el
> articulo** hasta regenerarlas (US-124, US-125). Se conservan sin retocar porque el registro de
> lo que creimos importa tanto como lo que resulte.

**Modo**: 1 del skill `micai-paper` — encuadre de la contribución y estructura.
**Fecha**: 2 de septiembre de 2026. **Responsable**: Javier A. Rebull-Saucedo. **Decide con**: Arthur Jafed Zizumbo Velasco.
**Base**: [`novedad.md`](novedad.md) (fase 0), [`fase2-hallazgos.md`](fase2-hallazgos.md) (fase 2) y [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md) (49 artefactos sellados).

> Este documento no propone escribir todavía. Responde a una pregunta anterior: **con lo que hay medido y sellado, ¿qué artículo se sostiene ante un revisor?** Ninguna cifra de aquí procede del manuscrito heredado.

---

## 1. Lo que sí tenemos, y que un revisor puede replicar

Todo lo siguiente está sellado con MD5, es libre de fuga, se reproduce en CPU y vive sobre un banco **público**.

| Activo | Qué permite afirmar |
|---|---|
| PASTIS-R, fold 5 held-out, 16 640 parcelas, 18 clases | Banco público con particiones espacialmente disjuntas ya establecidas por la literatura |
| Ground truth y centroides sellados, 420 KB | La evaluación completa se replica sin descargar los 68 GB del dataset |
| Diez miembros heterogéneos con posteriores por parcela | Comparar reglas de combinación sin reentrenar nada |
| Protocolo de bloques espaciales (H3 res 5, cinco bloques, buffer de 1 km) con predicciones agrupadas | Toda decisión —pesos, leyenda, umbral, punto de operación— se toma fuera del bloque que se mide |
| Tabla de individuales bajo un protocolo único | El mejor miembro es tsvit-pheno con F1-macro 0,7367 |
| Cuatro reglas de combinación en sus dos regímenes, con bootstrap pareado y McNemar | Ninguna combinación libre de fuga mejora al mejor miembro |
| F1 por clase de las cuatro reglas | El árbitro entrenado deja la clase 10 en F1 exactamente 0 sobre 355 parcelas |
| Frontera calidad-cobertura de dos mecanismos a igual cobertura, por bloque, con IC | Retirar clases domina al rechazo por confianza: +0,050 a K=12 y +0,194 a K=8 |
| Nulo de vecindad con IC y barrido completo | Ningún alfa mayor que cero mejora; el intervalo incluye el cero |
| Delta de FarSLIP con IC | +0,0006, IC [−0,0024, +0,0034] |
| Desbalance medido: 6 128 parcelas en la clase mayor, 103 en la menor | La razón por la que la métrica tiene que ser macro y no exactitud |

## 2. Lo que no tenemos, y condiciona el alcance

1. **Un segundo conjunto de datos con posteriores por parcela.** Es la objeción número uno de cualquier revisor. No se resuelve con lo que hay en disco: EuroCropsML solo conserva la curva agregada y Sen4AgriNet solo un punto denso de diez parches. Salir de una región exige entrenamiento nuevo.
2. **Volcados OOF de los folds 1 a 4.** Sin ellos el meta-modelo solo puede entrenarse con bloques del propio fold 5, así que el resultado negativo del ensamble mide *el stacking que se puede validar con lo que hay*, no el stacking en general.
3. **La procedencia de `tsvit-pheno`.** 0,7367 frente a 0,2552 de otra variante de la misma arquitectura. Hasta que Arthur confirme con qué folds se entrenó, cualquier cifra que dependa de ese miembro es provisional.
4. **La capa conversacional.** Su evaluación no existe: fixtures de tres líneas y una tabla marcador de posición.
5. **Multi-región y DE4.** Sin artefactos sellados.

## 3. El hueco, tal como quedó verificado en la fase 0

De las 43 entradas de la matriz, con título y autores resueltos por API:

- La clasificación selectiva es una línea madura —Chow 1970, Geifman y El-Yaniv, SelectiveNet, Deep Gamblers, las curvas de rechazo de Fischer 2023— y la revisión de Hendrickx 2024 deja explícito que **todos** sus mecanismos recortan cobertura muestra a muestra. Retirar clases del espacio de etiquetas no figura en esa taxonomía.
- En cultivos, el precedente más cercano es Turkoglu et al. 2021: predice tres niveles de una jerarquía y retrocede a la etiqueta gruesa cuando la confianza no alcanza un umbral, publicando curvas de cobertura frente a confianza. Mide **exactitud global** sobre el área cubierta, sobre **un solo fold** y **sin intervalos**, y su mecanismo es retroceso jerárquico, no retirada de una leyenda plana.
- La literatura europea de mapeo documenta que agrupar clases sube la exactitud, pero lo trata como advertencia para comparar estudios, no como un punto de operación que se elige y se mide.

**Nadie compara los dos mecanismos entre sí, a igual cobertura, bajo F1-macro y con intervalos pareados.** Ese es el hueco, y es estrecho: hay que decirlo con esas palabras y no con adjetivos.

## 4. Cuatro encuadres posibles

| | Encuadre | Evidencia disponible | Riesgo principal | Coste |
|---|---|---|---|---|
| **A** | El punto de operación: qué leyenda prometer frente a qué parcelas responder | Completa y sellada | Una sola región y un solo dataset | Cero experimentos obligatorios |
| **B** | Qué mide de verdad la ganancia de un ensamble en mapeo de cultivos | Sólida, pero es un resultado negativo sobre nuestro propio código | Se lee como auditoría interna, no como aporte | Exigiría auditar código publicado de terceros para generalizar |
| **C** | **A como contribución, con B dentro como protocolo y controles negativos** | Completa | El de A, mitigable | Dos experimentos recomendados, ambos en CPU |
| **D** | El copiloto anclado perceptor-razonador | Inexistente | El núcleo evaluativo no está medido | Dataset nuevo más ventana H100 |

**B no sostiene un artículo por sí solo.** Que nuestro meta-modelo se evaluara in-sample es un error nuestro; convertirlo en tesis exigiría demostrar que la práctica está extendida, y eso pide auditar repositorios ajenos, que es otro proyecto. Como **sección de protocolo dentro de A**, en cambio, es lo que hace creíble todo lo demás: explica por qué nuestras cifras no coinciden con las que el propio equipo publicó antes.

## 5. Encuadre vigente, reescrito tras las fases 3 y 4

> **Este apartado se reescribió el 3 de septiembre de 2026.** La versión anterior sostenía que
> «recortar la leyenda compra más calidad macro que rechazar parcelas por confianza a igual
> cobertura» y que «domina por debajo de doce clases». Las fases 3 y 4 **falsificaron** esa
> afirmación, y el encuadre cambia con ella en lugar de sobrevivirla. Lo que era la tesis pasa a
> ser el resultado negativo; lo que era un control auxiliar pasa a ser la contribución.

El artículo trata de **cuánto de una mejora de F1-macro es el método y cuánto es el denominador**,
y usa el recorte de leyenda en mapeo de cultivos como el caso donde eso se puede medir y separar.

El hilo, en una frase: *cuando se reporta F1-macro sobre un catálogo de clases recortado, casi toda
la mejora es el cambio de denominador y no el mecanismo que la produjo; el control que lo revela
cuesta una línea —puntuar el predictor intacto sobre la misma leyenda— y sin él la conclusión se
invierte.*

### Título de trabajo

> *How much of that gain is the denominator? A control for legend shrinking in imbalanced crop mapping*

Sin nombre del sistema ni de la institución: nace listo para doble ciego.

### Contribuciones, con verbos honestos

Inline, no en lista con viñetas: (i) formalizamos dos mecanismos para cambiar cobertura por calidad
macro —recortar la leyenda y abstenerse por confianza— y damos un protocolo por bloques espaciales
que elige el punto de operación fuera de los datos que lo miden, con un **estimando alineado** que
puntúa los dos sobre el mismo conjunto de clases y una entrega que **no lee la etiqueta**;
(ii) mostramos que la mayor parte de la mejora aparente al recortar la leyenda es el cambio de
denominador y no el mecanismo —88 % en el banco primario, 95 % en el segundo y **100 % en el tramo
donde la cobertura sigue completa**—, y damos el control de una línea que lo revela;
(iii) mostramos que a igual cobertura **abstenerse por confianza supera a recortar la leyenda**, con
intervalos pareados y corrección por multiplicidad, y que **retirar por soporte bajo** —que es lo que
se hace en la práctica— es el peor de los tres criterios y además no es monótono; (iv) mostramos que
un meta-modelo entrenado ejecuta ese recorte de forma implícita y no declarada, y que la diferencia
entre medirlo dentro y fuera de muestra basta para invertir la conclusión sobre si el ensamble aporta.

### Por qué esto es un artículo y no un resultado negativo suelto

Porque la contribución no es «el mecanismo no funciona», sino **el control que permite saberlo**, más
la demostración de que sin ese control la literatura del área publica el denominador como si fuera
método. Se sostiene en dos bancos, dos años, dos regiones y dos familias de modelo, con el mismo
protocolo y sin tocar su método entre uno y otro.

### Esqueleto de doce páginas

| Sección | Contenido | Páginas |
|---|---|---|
| 1 Introducción | El desbalance real (6 128 frente a 103, y 19 207 a 1 en el segundo banco), por qué la leyenda es una decisión y no un dato, contribuciones inline | 1,25 |
| 2 Trabajo relacionado, por limitaciones | Clasificación selectiva y su taxonomía por muestra; Turkoglu y el retroceso jerárquico; agregación de clases en mapeo europeo; validación espacial | 1,5 |
| 3 Materiales y método | Los dos bancos y el universo compartido; el protocolo de bloques; el **estimando alineado** y por qué el F1 nativo no es comparable; la entrega sin oráculo; el control sin mecanismo; el bootstrap pareado y Holm | 2,75 |
| 4 Resultados | 4.1 la descomposición y el pasillo donde las cuatro series coinciden; 4.2 la réplica en el segundo banco; 4.3 el orden de los tres mecanismos y la no monotonía del soporte; 4.4 la retirada implícita del árbitro, dentro frente a fuera de muestra | 3,25 |
| 5 Discusión | Qué compra cada mecanismo y a quién; qué habría que reportar para que una cifra macro sea interpretable | 1,25 |
| 6 Limitaciones y conclusión | Dos bancos pero una geografía; bloques por región en el segundo; imputación del 1,62 % | 0,75 |
| Referencias | Unas 30 entradas con DOI | 1,25 |

Cabe en doce con holgura de media página, que es lo que pide la experiencia cuando una figura crece al final.

## 6. Lo que faltaba, y ya está hecho

Los cinco puntos que este apartado listaba el 2 de septiembre están cerrados por las fases 3 y 4.
Se dejan con su estado para que se vea qué respondió cada uno.

| # | Qué era | Estado |
|---|---|---|
| 1 | Segundo predictor (`xgb-alphaearth`) | **Hecho** en la fase 3. La conclusión se mantiene en los dos predictores |
| 2 | Corrección por multiplicidad y declarar confirmatorio frente a exploratorio | **Hecho**: Holm sobre la familia de siete K, con el criterio principal preregistrado |
| 3 | Tercer mecanismo, retirar por soporte | **Hecho**, y resultó el peor de los tres en los dos bancos, además de no monótono |
| 4 | Curva de rechazo por confianza con más puntos | **Hecho**: la frontera completa, con el control sin mecanismo añadido |
| 5 | Figura de la frontera en vectorial | **Hecha** para los dos bancos, legible en blanco y negro |

Lo que queda por delante es escribir, no medir. La fase 5 —reentrenamiento OOF de cinco folds— sigue
siendo opcional y **no bloquea**: la contribución vigente no depende del marcador del ensamble.

## 7. Las objeciones que va a recibir, y su respuesta

| Objeción | Respuesta que el dato sostiene |
|---|---|
| «Un solo dataset y una sola región» | **Ya no aplica.** Dos bancos, dos años, dos regiones y dos familias de modelo, con el mismo protocolo y sin tocar su método entre uno y otro |
| «Esto ya lo hace Turkoglu 2021» | Mecanismo distinto (leyenda plana frente a retroceso jerárquico), métrica distinta (macro frente a exactitud) y aquí hay comparador a igual cobertura, estimando alineado, intervalos pareados y control sin mecanismo |
| «¿Por qué F1-macro?» | Porque la clase mayor tiene 6 128 parcelas y la menor 103 en el primer banco, y 19 207 a 1 en el segundo: la exactitud premia ignorar las raras, que es exactamente la decisión que el artículo estudia |
| «Es un resultado negativo» | La contribución no es «el mecanismo no funciona», sino **el control que permite saberlo**, más la demostración de que sin él se publica el denominador como si fuera método. Y hay un resultado positivo: a igual cobertura la abstención por confianza gana, con IC que excluye el cero |
| «¿Por qué su ensamble no mejora?» | Se reporta con su causa: los volcados OOF solo existen para el fold 5, así que el meta-modelo se entrena con muy poco. Es una limitación de los artefactos, no una afirmación sobre el stacking en general |

## 8. Restricciones de MICAI, verificadas hoy

**MICAI 2027 no tiene sitio ni convocatoria todavía.** Es la edición conmemorativa de los cuarenta años de la SMIA y siguen buscando institución anfitriona. Como referencia, MICAI 2026 (`micai.org/2026/authors/`, consultado el 2 de septiembre de 2026):

- «The recommended length of a paper is up to 12 pages, but it can be larger, not exceeding 20 pages.»
- «The review procedure is double blind. Thus the papers submitted for review must not contain the authors' names, affiliations, or any information that may disclose the authors' identity.»
- Envío en `.zip` con el proyecto LaTeX completo.
- Calendario 2026: envío el 30 de junio, notificación el 5 de agosto, versión final el 23 de agosto.

Todo esto **se vuelve a verificar** cuando salga la convocatoria de 2027. Doce páginas es el objetivo de diseño; veinte es el techo, no la meta.

## 9. Desde cero, y por qué

El manuscrito heredado tiene veinticuatro páginas —no treinta y seis, como se dijo primero; lo dice `pdfinfo`—, citas con título y autores inventados, cifras del régimen equivocado y una tesis central que la fase 2 desmontó. Repararlo cuesta más que escribir doce páginas nuevas sobre artefactos sellados, y deja rastros: el que repara arrastra frases que ya no corresponden a ningún dato.

Se reutiliza, eso sí, lo que costó trabajo y sigue siendo válido: las 43 referencias verificadas por API, el ledger de custodia con sus 49 artefactos, las tablas de la fase 2 y la plantilla LNCS ya prototipada.

## 10. Qué hace falta decidir

1. **Aceptar el encuadre C** o proponer otro. Es una enmienda a [`ADR-013`](../decisions/ADR-013-angulo-micai.md), porque cambia título, contribuciones y la mitad de las secciones.
2. **La procedencia de `tsvit-pheno`**, que solo Arthur puede comprobar. Si ese checkpoint vio el fold 5, el mejor individual se cae y hay que rehacer las cifras del apartado de resultados.
3. **Si el artículo se envía a MICAI 2027 o antes a otro sitio**, dado que 2027 aún no tiene fechas.
