# ADR-013 — Ángulo del artículo MICAI 2027: reencuadre del ángulo A

**Estado**: Borrador para firma. Pendiente de aprobación de Arthur Zizumbo y Javier Rebull.
**Fecha del borrador**: 2 de septiembre de 2026
**Decisores**: Arthur Jafed Zizumbo Velasco (primer autor) · Javier A. Rebull-Saucedo (segundo autor)
**Fase del plan**: [Fase 0 de `docs/plan-micai-2027.md`](../plan-micai-2027.md)
**Evidencia**: [`docs/paper/novedad.md`](../paper/novedad.md) · matriz verificada en `reports/paper_micai/fase0/related_work_verified.csv` (43 entradas, 43 con identificador resuelto por API) · registro de consultas en `reports/paper_micai/fase0/search_log.csv` y `search_log_manual.csv`
**Sustituye en la práctica a**: el encuadre de [ADR-010](ADR-010-ensamble-ec-geocontext-future.md) sobre el eje estructural, que queda acotado por lo que se dice abajo.

---

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
- El sellado de la fase 1 obliga además a **reimplementar** la curva de retirada de clases
  desde las posteriores OOF: `us043_honest_dropout_curve.csv` y `us043_farslip_grid.csv`
  están sellados y son reales, pero su productor no está en el repositorio, que solo
  contiene lectores. Los CSV quedan como comprobación cruzada del resultado nuevo, no como
  su fuente.
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
