# Preregistro de las fases 3 y 4

**Escrito el**: 2 de septiembre de 2026, **antes** de ejecutar ningún experimento de estas dos fases y **antes** de descargar BreizhCrops.
**Obliga**: regla R5 de [`ADR-013`](../decisions/ADR-013-angulo-micai.md).
**Responsable**: Javier A. Rebull-Saucedo.
**Congelado en**: el commit que introduce este archivo. Cualquier cambio posterior se añade como enmienda fechada al final, nunca editando lo de arriba.

> Para qué sirve esto. Con siete valores de K, tres mecanismos y dos conjuntos de datos hay
> sitio de sobra para encontrar el resultado que a uno le gusta y llamarlo hallazgo. Este
> documento fija qué se va a mirar, con qué prueba y qué contaría como refutación, antes de
> poder mirarlo.

---

## 1. Qué fue exploratorio y qué será confirmatorio

Hay que decirlo sin adornos: **la fase 2 sobre PASTIS-R fue exploratoria.** Generó la
hipótesis; no puede confirmarla, porque las decisiones de diseño —restringir el argmax a
la leyenda, medir por bloque en lugar de agrupado, qué valores de K barrer— se tomaron
mientras se veían los resultados. Presentarla como confirmatoria sería falso.

| Fase | Conjunto de datos | Papel |
|---|---|---|
| 2 | PASTIS-R, fold 5 | **Exploratoria.** Generó la hipótesis |
| 3 | PASTIS-R, fold 5 | **Robustez preespecificada.** Comprueba que el hallazgo no depende de un modelo ni de un criterio de retirada |
| 4 | BreizhCrops | **Confirmatoria.** Réplica sobre datos que hoy no he mirado |

La distinción va al artículo tal cual. Un revisor que la vea declarada confía más que ante
un trabajo que presenta ocho contrastes como si los ocho fueran hipótesis previas.

## 2. La hipótesis, fijada

**H1.** En mapeo de cultivos por parcela con clases desbalanceadas, y a igual cobertura,
recortar la leyenda produce mayor F1-macro que la abstención por confianza a nivel de
parcela.

**H0.** No hay diferencia, o la diferencia favorece a la abstención por confianza.

## 3. El criterio de valoración principal, uno solo

- **Estadístico**: diferencia de la media entre bloques del F1-macro, retirada de clases
  menos rechazo por confianza, a cobertura igualada.
- **Punto de operación único**: `K = round(C / 2)`, donde `C` es el número de clases del
  conjunto de datos. En PASTIS-R son 18 y da `K = 9`; en BreizhCrops se calculará igual,
  con el `C` que traiga el dataset.
- **Por qué la mitad de la leyenda y no el mejor punto**: es una regla escribible sin mirar
  el resultado. El mejor delta observado en la fase 2 estaba en `K = 8`, no en `K = 9`, así
  que esta regla **no** selecciona el máximo. Declaro que la regla se fijó conociendo los
  resultados de PASTIS-R; sobre BreizhCrops es genuinamente ciega.
- **Un solo contraste principal por conjunto de datos**, así que el criterio principal no
  tiene problema de multiplicidad.

### Definición exacta de la métrica

Para un bloque `b` y una leyenda `L_b`:

1. `L_b` son las `K` clases con mejor F1 binario en los bloques distintos de `b`. Nunca se
   elige con el bloque que se mide.
2. La retirada de clases entrega las parcelas de `b` cuya etiqueta verdadera está en `L_b`,
   y emite `argmax` **restringido a las columnas de `L_b`**.
3. El rechazo por confianza entrega las parcelas de `b` cuya confianza máxima supera el
   umbral que alcanza en los otros bloques la misma cobertura que logró la retirada en `b`.
   Emite `argmax` sobre la leyenda completa.
4. El F1-macro de cada mecanismo se promedia sobre la intersección entre la leyenda que
   promete y las clases presentes entre sus parcelas entregadas. Una clase prometida que no
   ocurre en ese bloque no entra como cero.
5. El estadístico es la media simple entre bloques, y su intervalo sale de un bootstrap que
   remuestrea parcelas dentro de cada bloque y recalcula ambos mecanismos en cada sorteo.

### Parámetros congelados

Semilla 42. Mil remuestreos. Cinco bloques espaciales por teselación H3 de resolución 5
con exclusión de 1 km. Intervalos de percentil al 95 %.

## 4. Qué contaría como refutación

Esto es lo que hace que el preregistro valga algo.

- **Si en BreizhCrops el intervalo del criterio principal incluye el cero**, H1 no replica.
  El artículo lo reporta así y la contribución pasa a ser el protocolo y la réplica
  negativa. No se cambia de conjunto de datos, ni de K, ni de métrica.
- **Si el intervalo excluye el cero pero con el signo contrario**, el hallazgo de PASTIS-R
  queda como específico de ese dataset y el artículo lo dice en el título.
- **Si BreizhCrops resulta no tener desbalance apreciable** —criterio: la razón entre la
  clase mayor y la menor por debajo de 10— deja de ser una réplica válida de una hipótesis
  sobre desbalance. En ese caso se declara así, se reporta igual como contexto, y la
  limitación de un solo dataset se asume en el artículo. **No se sustituye por otro dataset
  elegido después.**

## 5. Análisis secundarios, declarados exploratorios

Se reportan siempre, con corrección de multiplicidad de Holm dentro de cada familia y con
el intervalo sin corregir al lado. Ninguno sostiene la afirmación principal.

1. **Familia de K**: los siete valores `{18, 16, 14, 12, 10, 9, 8}` en PASTIS-R y su
   equivalente en BreizhCrops. Familia de siete contrastes.
2. **Segundo predictor**: repetir todo sobre `xgb-alphaearth` (F1-macro 0,5913 en PASTIS-R)
   para separar «propiedad del desbalance» de «propiedad de un modelo».
3. **Tercer mecanismo**: retirar clases **por soporte** en lugar de por F1, que es lo que
   se hace en la práctica.
4. **Curva completa de abstención**, con más puntos que los igualados, para dibujar la
   frontera y no solo sus intersecciones.
5. **Análisis por clase** de qué entrega y qué descarta cada mecanismo.
6. **Estabilidad de la leyenda** entre bloques, que en PASTIS-R ya salió baja y es parte
   del resultado.

## 6. Lo que no se va a hacer

- No se añaden valores de K después de ver los resultados.
- No se cambia la métrica principal por accuracy si el macro no acompaña.
- No se elige el predictor base por su resultado: es el de mejor F1-macro bajo el protocolo
  único, y en PASTIS-R eso ya está fijado en `tsvit-pheno`.
- No se descarta un bloque espacial por dar un número incómodo.
- No se sustituye BreizhCrops por otro conjunto de datos si el resultado no gusta.

## 7. Orden de ejecución

### Fase 3 · PASTIS-R, en CPU

Salida en `reports/paper_micai/fase3/`, sellada en `paper/ARTIFACTS.md`.

1. Segundo predictor: los tres mecanismos sobre `xgb-alphaearth`.
2. Tercer mecanismo: retirada por soporte, sobre ambos predictores.
3. Corrección de Holm sobre la familia de K, en los dos predictores.
4. Curva completa de abstención.
5. Figura de la frontera, vectorial y legible en blanco y negro.

### Fase 4 · BreizhCrops, en CPU

Salida en `reports/paper_micai/fase4/`.

1. `dvc pull data/breizhcrops` (1,66 GB, seis archivos).
2. Inspección mínima y honesta: número de clases, soporte por clase, razón mayor a menor.
   **Esta inspección se hace antes de entrenar y se reporta**, porque de ella depende que
   el dataset sea una réplica válida según el criterio de la sección 4.
3. Entrenar un clasificador por parcela en CPU, con particiones espaciales del mismo tipo.
4. Sellar ground truth, posteriores y soporte por clase igual que se hizo con el fold 5.
5. Correr **exactamente** el mismo protocolo, sin tocar una línea del método.
6. Reportar el criterio principal y la familia exploratoria, se transporte o no.

## 8. Enmiendas

### Enmienda 1 · 2 de septiembre de 2026, tras inspeccionar BreizhCrops y antes de entrenar

**Motivo**: la inspección que la seccion 7 obliga a hacer antes de entrenar revela un
reparto de clases que hace inservible el criterio principal tal como estaba escrito.

Medido con `ml.ingest.breizhcrops_loader.breizhcrops_parcel_index`:

| Región | Parcelas | Clases | Mayor | Menor | Razón |
|---|---|---|---|---|---|
| frh01 | 178 632 | 9 | 52 013 (praderas temporales) | **1** (girasol) | 52 013 : 1 |
| frh04 | 122 708 | 9 | 38 414 (praderas temporales) | **2** (girasol) | 19 207 : 1 |

El criterio de escape de la sección 4 —razón por debajo de 10— **no se dispara**: el
desbalance es mucho mayor que en PASTIS-R, que era de 60 a 1. Pero aparece el problema
contrario, que el preregistro no previó: con clases de una y dos parcelas, un F1-macro sobre
nueve clases lo decide el azar de en qué bloque cae esa única parcela. No es una medida, es
ruido con nombre.

**Qué se cambia**, y solo esto:

1. El contraste se reporta en **dos universos**, ambos publicados: las **nueve** clases tal
   como vienen, y las **siete** con soporte de al menos cien parcelas, que es el suelo que el
   conjunto primario tenía por su cuenta. El umbral se elige por mensurabilidad y se declara
   aquí antes de ver un solo resultado; no se ajustará después.
2. El criterio principal `K = round(C/2)` se aplica a cada universo con su propio `C`: K = 5
   sobre nueve clases y K = 4 sobre siete. Ambos se reportan.
3. Si los dos universos discrepan, se reportan los dos y se dice cuál es más interpretable y
   por qué, sin elegir el que convenga.

4. **Los bloques dejan de ser hexágonos y pasan a ser regiones.** El índice de BreizhCrops
   **no trae coordenadas por parcela** —sus columnas son `idx, meanCLD, id, CODE_CULTU,
   path, sequencelength, classid, classname, region`—, así que la teselación H3 del conjunto
   primario es imposible aquí. La única estructura espacial disponible es la región, que es
   además como el propio banco está diseñado para partirse. Se usan las dos regiones
   descargadas, `frh01` y `frh04`, como los dos bloques. Son menos que los cinco del conjunto
   primario y el intervalo lo acusará: se reporta así, no se disimula.
5. **Submuestreo estratificado por presupuesto de cómputo.** Extraer las 185 características
   por parcela cuesta unos 33 milisegundos, o sea cerca de tres horas para las 301 340
   parcelas de las dos regiones. Se toma un submuestreo **proporcional** de 30 000 parcelas
   por región con semilla fija, que preserva el reparto de clases —que es justo el objeto de
   estudio— y deja la clase de huertos por encima de las cien parcelas en las dos. El motivo
   es el reloj, no el resultado, y el tamaño se fija aquí antes de ver ninguna cifra.

**Qué no se cambia**: ni la métrica, ni la definición de los mecanismos, ni el estimando
alineado, ni el remuestreo pareado, ni el hecho de que la decisión de cada bloque se tome
fuera de él.

**Nota de honestidad sobre la ceguera.** La sección 1 decía que la fase 4 era confirmatoria
sobre datos «que hoy no he mirado». Eso era inexacto y hay que dejarlo escrito: BreizhCrops
está versionado en el repositorio desde mayo y su distribución de clases se publicó en un
notebook commiteado con salidas. No lo había leído, pero estaba disponible, así que la fase 4
**no es una réplica ciega: es una réplica preespecificada sobre un conjunto conocido**. Lo
señaló la auditoría ciega y se corrige aquí en lugar de defenderlo.

**Y el papel de la fase 4 cambia**, porque H1 se cayó en la fase 3. Ya no confirma que un
mecanismo domine al otro. Comprueba si **la descomposición** se transporta: cuánto de la
mejora aparente al recortar la leyenda es el denominador y cuánto el mecanismo, en otro
conjunto de datos, otra región y otro reparto de clases.

### Corrección 1 al punto 5 de la enmienda · 3 de septiembre de 2026, con el extractor corriendo

La enmienda 1 justificó el submuestreo diciendo que extraer las 185 características cuesta
«unos 33 milisegundos por parcela, o sea cerca de tres horas para las 301 340 parcelas». **Esa
cifra estaba mal por un factor de veintitrés.** Medido contra el reloj, no estimado: frh01
tardó de 22:53:20 a 05:19:11, es decir **6 h 26 min para 30 000 parcelas**, unos **772 ms por
parcela**. Extraer las dos regiones completas habría costado del orden de **sesenta y cinco
horas**, no tres.

Qué cambia y qué no. **La decisión no cambia**: el tamaño de 30 000 por región se fijó antes de
ver ningún resultado y sigue fijado; el coste real solo refuerza el motivo que ya se había
declarado. Lo que cambia es la cifra impresa, que era falsa y queda corregida aquí en lugar de
sobrevivir hasta el artículo. La regla de la casa —ninguna cifra sin artefacto que la sostenga—
también vale para las cifras de coste, y ésta no lo tenía.

Un defecto de ingeniería que la corrección deja anotado, porque afecta a la reproducibilidad y
no al resultado: `scripts/build_breizhcrops_features.py` escribe el parquet **solo al final**,
después de las dos regiones. Con un coste real de doce horas eso significa que un fallo en la
hora once lo pierde todo. Quien repita esto debe materializar cada región por separado antes de
concatenar.

### Corrección 2 · 3 de septiembre de 2026, con la tabla ya extraída y antes de leer contraste alguno

Al primer intento de entrenar, XGBoost se negó: la tabla de características contiene valores no
finitos. Medido sobre el parquet sellado: **179 747 celdas de 11 100 000, un 1,62 %**, repartidas
en **22 de las 185 columnas** y tocando **38 034 de las 60 000 parcelas**.

No es ruido aleatorio, tiene causa y conviene dejarla escrita porque es un hallazgo sobre el
propio banco de datos:

- `GCVI` (29 523 parcelas) y `MCARI` (21 570) son cocientes cuyo denominador lleva la banda verde
  o la del borde rojo. BreizhCrops codifica el pixel sin dato como cero, así que cualquier fecha
  enmascarada mete un infinito, y basta una para que la media y el máximo de la serie sean
  infinitos.
- `PSRI` aporta 133 más por la misma vía, con signo a ambos lados.
- Las columnas fenológicas (`senescence_doy`, `ndvi_slope_post_peak`, 5 886 cada una) son NaN
  cuando la serie no llega a tener senescencia detectable dentro de la ventana.

**Tratamiento, y por qué éste.** Se imputa cada celda no finita con la **mediana de su columna
calculada solo sobre la región de entrenamiento**, que es exactamente lo que el baseline tabular
del proyecto ya hacía con estas mismas columnas (`_column_medians` y `_impute_with` de
`ml/train/baseline.py`, reutilizados y no reescritos). Se elige por dos razones: es la convención
que el repositorio ya tenía, así que no es una decisión inventada para esta corrida; y calcular la
mediana **solo en la región de entrenamiento** deja la región evaluada fuera de la imputación, que
es la misma disciplina de fuga que el resto del protocolo.

**Qué no cambia**: ni el protocolo, ni los mecanismos, ni el estimando, ni los universos, ni los
valores de K. Esto ocurre aguas arriba del protocolo, en la entrada del clasificador, y el
protocolo recibe posteriores igual que antes.

Queda anotado como limitación honesta: un 1,62 % de las celdas de este segundo conjunto son
imputadas, cosa que no ocurría en el conjunto primario, y eso puede empujar hacia abajo la
calidad del clasificador de la fase 4. No afecta a la comparación **entre mecanismos**, que es lo
que la fase 4 mide, porque todos los mecanismos leen la misma posterior.
