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

Ninguna todavía. Cualquier cambio a lo anterior se añade aquí, con fecha y motivo, y nunca
borrando lo que sustituye.
