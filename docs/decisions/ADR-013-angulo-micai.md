# ADR-013 — Ángulo del artículo MICAI 2027: reencuadre del ángulo A

**Estado**: Enmendado el 2 de septiembre de 2026 tras la fase 2. Encuadre aceptado por Javier Rebull; pendiente de la firma de Arthur Zizumbo.
**Fecha del borrador**: 2 de septiembre de 2026
**Decisores**: Arthur Jafed Zizumbo Velasco (primer autor) · Javier A. Rebull-Saucedo (segundo autor)
**Fase del plan**: [Fase 0 de `docs/plan-micai-2027.md`](../plan-micai-2027.md)
**Evidencia**: [`docs/paper/novedad.md`](../paper/novedad.md) · matriz verificada en `reports/paper_micai/fase0/related_work_verified.csv` (43 entradas, 43 con identificador resuelto por API) · registro de consultas en `reports/paper_micai/fase0/search_log.csv` y `search_log_manual.csv`
**Sustituye en la práctica a**: el encuadre de [ADR-010](ADR-010-ensamble-ec-geocontext-future.md) sobre el eje estructural, que queda acotado por lo que se dice abajo.

---

> **CUARENTENA** — Este ADR cita cifras derivadas de artefactos marcados `OBSOLETO` en el registro
> de custodia: las produjo el módulo de evaluación cuando aún tenía tres defectos —denominador
> móvil, punto de operación elegido dentro del bloque que lo mide, y remuestreo a nivel de
> parcela—. La decisión que registra sigue en pie; **los números que la acompañan no se citan**
> hasta regenerarlos. Se conserva sin retocar porque un ADR es el registro de lo que se decidió y
> con qué se decidió. Lo comprueba `make paper-obsoletos-check`.

## Contexto

La fase 0 del plan pide decidir, antes de reescribir una línea del manuscrito, si el
artículo aporta algo. La tesis candidata era el **ángulo A**:

> En mapeo de cultivos desbalanceado, un árbitro heterogéneo por clase mejora sobre el
> promedio homogéneo, cuántas clases prometer es un punto de operación medible, y el
> contexto espacial no aporta sobre embeddings de fundación anuales.

Se ejecutó la búsqueda sistemática de 2019 a 2026 en arXiv, Semantic Scholar y OpenAlex,
más seis consultas de buscador registradas a mano, sobre los cuatro frentes del plan; se
construyó una matriz de 43 trabajos con método, fortaleza, límite y hueco, con el título,
los autores y el año tomados de la API y nunca de memoria; y se leyeron las tres
referencias ancla para comprobar qué dicen de verdad.

## Decisión

**Reencuadre.** El ángulo A no se abandona y tampoco se confirma tal como está escrito.
Se reordenan sus tres patas y se estrecha cada afirmación hasta lo que el dato sostiene.

Tesis que sustituye a la anterior:

> En mapeo de cultivos por parcela con clases desbalanceadas, una vez que las posteriores
> de miembros heterogéneos se combinan con un árbitro entrenado, la palanca que queda no
> es más contexto espacial sino la decisión de qué clases se prometen: retirar clases
> enteras del conjunto plano y el rechazo por confianza son dos mecanismos distintos de
> recortar cobertura, se pueden comparar a igual cobertura bajo F1-macro y el refinamiento
> por vecindad entre parcelas no aporta una mejora material sobre el ensamble apilado.

Reparto de papeles en el artículo:

| Pata | Papel nuevo | Motivo |
|---|---|---|
| Cardinalidad y cobertura | **Contribución central** | Es el único eje donde existe un contraste que la literatura no reporta: retirada de clases frente a rechazo por confianza, a igual cobertura y en F1-macro. |
| Arbitraje heterogéneo por clase | **Mecanismo que la habilita**, no contribución independiente | El stacking en cultivos está muy publicado; lo que no está es el contraste controlado homogéneo frente a heterogéneo sobre los mismos miembros, folds y arnés, con delta por clase y pruebas pareadas. |
| Contexto espacial | **Control negativo acotado**, con intervalo y con alcance declarado | El artefacto propio da una mejora **positiva** por debajo del umbral de materialidad, no un cero; y hay trabajo previo que muestra que el contexto **intraparcela** sí aporta sobre estos embeddings. |

## Afirmaciones que quedan prohibidas

1. **«El contexto espacial no aporta sobre embeddings de fundación».** Demasiado ancha.
   `reports/ensemble/metrics/ec_neighborhood_result.json` da +0.0027 de F1-macro a 18
   clases en el punto que mejora ambos ejes y +0.0068 en el óptimo de 18 clases, ambos
   positivos y por debajo del umbral de 0.01 que el propio artefacto fija. La afirmación
   admisible es: *el refinamiento por vecindad de posteriores entre parcelas no produce
   una mejora material sobre un ensamble ya apilado*, con intervalo y sobre PASTIS-R.
2. **«AlphaEarth ya codifica la fenología».** La referencia ancla dice lo contrario en el
   eje temporal: «AEF lacks temporal sensitivity and therefore cannot support time-critical
   tasks such as in-season yield prediction». Se puede afirmar que el embedding anual
   absorbe contexto suficiente para que la vecindad no añada, nunca que codifica fenología.
3. **«Cuántas clases prometer es un punto de operación medible» como novedad a secas.**
   Turkoglu et al. 2021 ya publican curvas de cobertura frente a confianza en cultivos con
   retroceso jerárquico. La novedad es el contraste entre mecanismos a igual cobertura y
   bajo F1-macro, no la existencia del compromiso.
4. **«Adoptamos el patrón Be My Eyes al pie de la letra».** El artículo original afina el
   perceptor con un pipeline de datos sintéticos y no evalúa nada geoespacial; nosotros no
   afinamos perceptor. Y la cota de alucinación es una afirmación nuestra: la palabra
   *hallucination* no aparece ni una vez en ese artículo.

## Reglas de decisión pre-registradas

Se fijan antes de correr los experimentos de la fase 2 para que el resultado no elija la
afirmación:

- **R1 — Nulo de vecindad.** Se calcula un intervalo bootstrap pareado por parcela
  (B = 1000, semilla fija) sobre el delta de F1-macro del mejor punto del barrido. Si el
  intervalo del 95 % **excluye el cero**, la pata deja de ser un nulo y se reporta como
  «mejora pequeña y no accionable», con su cifra, en Resultados. Si lo **incluye**, se
  reporta como nulo acotado. En ningún caso se retira el experimento.
- **R2 — Cardinalidad.** El baseline obligatorio es el rechazo por confianza a **igual
  cobertura**, no una curva sin comparador. Si la retirada de clases no domina al rechazo
  por confianza en F1-macro dentro del intervalo, se reporta el empate y la contribución
  pasa a ser el protocolo de comparación, no el resultado.
- **R3 — Arbitraje.** El contraste homogéneo frente a heterogéneo usa exactamente los
  mismos miembros, el mismo fold y el mismo arnés. Si el heterogéneo no gana con
  significación pareada, se reporta así y la pata baja a discusión.
- **R4 — Cifras.** Ninguna cifra entra al manuscrito sin artefacto en `reports/` con MD5
  en `paper/ARTIFACTS.md`. Las que hoy solo viven en markdown o en títulos de PNG se
  retiran hasta que exista el artefacto.

## Qué entra y qué sale del cuerpo

**Entra**: protocolo libre de fuga sobre PASTIS-R fold-5; miembros y árbitro; contraste
homogéneo frente a heterogéneo con delta por clase; frontera calidad-cobertura por
retirada de clases frente a rechazo por confianza; nulo acotado de vecindad; limitaciones.

**Sale del cuerpo** (a un segundo artículo o a un párrafo de contexto sin cifras): capa
conversacional y patrón perceptor-razonador, transferencia multirregión, Baja Sajonia,
FinOps y toda la narrativa de proceso interno.

## Consecuencias

- La fase 2 gana un experimento que antes no estaba: el **baseline de rechazo por
  confianza a igual cobertura**, sin el cual la contribución central no es defendible.
- El sellado de la fase 1 obliga además a **volver a generar** la curva de retirada de
  clases desde las posteriores OOF: `us043_honest_dropout_curve.csv` y
  `us043_farslip_grid.csv` están sellados y son reales, y su cálculo sí está versionado
  (`ml.eval.per_class_analysis.honest_class_dropout_curve` y `ml/ensemble/`), pero el guion
  que los escribió no lo está. Los CSV quedan como comprobación cruzada del resultado
  nuevo, no como su fuente. El eje de cobertura ya se comprobó: los siete valores de
  `n_parcels_fold5` reproducen exactos desde el ground truth sellado del fold 5.
- La fase 5 debe reconstruir seis entradas bibliográficas con datos falsos; la fase 0 ya
  deja resueltos y verificados los tres anclajes y otras cuarenta referencias.
- El alcance geográfico se declara: una región, un año, un fold held-out para los modelos
  densos. La generalización a otras regiones es trabajo futuro, no una promesa del título.
- ADR-010 mantiene su diagnóstico pero su lectura se acota: el delta nulo se midió en el
  espacio tabular y en el refinamiento estructural ligero, no en toda forma de contexto.

## Alternativas descartadas

- **Confirmar el ángulo A tal cual.** Se descarta porque dos de sus tres afirmaciones se
  caen ante la literatura verificada y ante el propio artefacto.
- **Abandonar el ángulo.** Se descarta porque el criterio del plan se cumple: existe al
  menos un experimento ejecutable en CPU que aísla el mecanismo (las diez tablas OOF por
  parcela de `ml/eval/oof/`, 16 640 parcelas en la intersección) y ningún trabajo de la
  matriz lo reporta en este dominio con este protocolo.
- **Pivotar al artículo del copiloto.** Se descarta para este envío: su núcleo evaluativo
  no existe todavía y depende de la ventana H100.

## Firmas

- [ ] Arthur Jafed Zizumbo Velasco — fecha:
- [ ] Javier A. Rebull-Saucedo — fecha:

---

## Enmienda del 2 de septiembre de 2026, tras medir la fase 2

La fase 2 midió las tres patas y el resultado obliga a mover la decisión otra vez. Esto no
sustituye lo anterior: lo estrecha.

### Qué cambió respecto al reencuadre original

1. **La cifra campeona no era un held-out.** `StackingEnsemble` reentrena su meta-modelo
   sobre todas las parcelas del fold 5 antes de predecirlas. Reproducido al sexto decimal:
   0,748614 y 0,849459. Las cuatro cifras selladas del proyecto salen de ese régimen.
2. **La pata del arbitraje se cae, y con ella la premisa del ensamble.** Libre de fuga
   ninguna combinación mejora al mejor miembro individual (tsvit-pheno, 0,7367): el árbitro
   empata con el promedio (+0,0005, el intervalo cruza el cero) y pierde contra el voto
   ponderado (−0,0437) y contra el propio miembro (−0,0572). Se aplicó la regla R3.
3. **La pata de la vecindad es un nulo limpio**, más limpio que el sellado: ningún alfa
   mayor que cero mejora y el intervalo incluye el cero. Regla R1 satisfecha.
4. **La contribución central se sostiene** (regla R2): a igual cobertura, retirar clases
   domina al rechazo por confianza, con delta de +0,050 a +0,194 e intervalo que excluye el
   cero por debajo de doce clases.
5. **Apareció el mecanismo que une las piezas.** El árbitro entrenado deja la clase 10 en
   F1 exactamente cero sobre 355 parcelas: ya retira clases por su cuenta, sin declararlo.

### Decisión enmendada

**Se escribe un artículo nuevo, desde cero, sobre el punto de operación.** El manuscrito
heredado no se repara: treinta y seis páginas, citas con título y autores inventados,
cifras del régimen equivocado y una tesis que la fase 2 desmontó. Se conserva como material
de consulta, no como base.

Tesis del artículo:

> Un meta-modelo entrenado sobre miembros heterogéneos acaba retirando clases por su
> cuenta, sin declararlo y sin que nadie elija el punto de operación. Conviene sacar esa
> decisión del modelo y medirla: a igual cobertura, recortar la leyenda compra más calidad
> macro que abstenerse por confianza, y la ventaja crece según se acorta la leyenda.

Título de trabajo, contribuciones, esqueleto de doce páginas, experimentos que faltan y
objeciones previsibles con su respuesta: [`docs/paper/que-paper-sale.md`](../paper/que-paper-sale.md).

### Qué papel tiene ahora cada pata

| Pata | Papel enmendado |
|---|---|
| Punto de operación | **Contribución central**, sin cambios |
| Arbitraje heterogéneo | **Caso que revela el problema**, no contribución. Su resultado negativo entra en la sección de protocolo con su causa declarada |
| Contexto espacial | **Control negativo**, ya medido y cerrado con intervalo |

### Afirmaciones prohibidas, ampliadas

A las cuatro anteriores se añaden tres:

5. **«El ensamble mejora al mejor miembro».** Libre de fuga no lo hace. Cualquier cifra de
   ganancia del ensamble pertenece al régimen in-sample y debe decirlo.
6. **«0,7486 held-out».** Es in-sample para el meta-modelo. Si se imprime, se etiqueta.
7. **«FarSLIP aporta señal complementaria».** +0,0006 con intervalo de [−0,0024, +0,0034].
   El aporte que reportaba el manuscrito vive entero dentro del régimen in-sample.

### Reglas nuevas para las fases 3 y 4

- **R5 — Confirmatorio antes de mirar.** La decisión de qué contrastes son confirmatorios y
  cuáles exploratorios, y la corrección por multiplicidad de los siete valores de K, se
  escriben **antes** de correr los experimentos nuevos.
- **R6 — El segundo conjunto de datos no se elige por su resultado.** BreizhCrops se fija
  como réplica antes de mirar nada, y su resultado se reporta se transporte o no. Si la
  conclusión no se transporta, el artículo gana un matiz; no se cambia de dataset.
- **R7 — Ninguna cifra sin fila sellada.** Ya vigente, se reafirma para el manuscrito nuevo.

### Firmas de la enmienda

- [ ] Arthur Jafed Zizumbo Velasco — fecha:
- [x] Javier A. Rebull-Saucedo — 2 de septiembre de 2026
