# Preregistro v2 — borrador para firma

**Estado**: BORRADOR. No vale hasta que esté commiteado y firmado. **Nada de las EPIC 20, 21, 22 ni 25 se computa antes de ese commit.**

Este documento existe porque el preregistro anterior se citó como credencial mientras se omitía que su hipótesis se había refutado y que la regla de entrega había cambiado. Aquí se declara todo antes, incluidos los grados de libertad que la vez pasada nadie sabía que lo eran.

## 1. La pregunta

Un mapa de cultivos que no alcanza calidad puede prometer menos de cuatro maneras: recortar el catálogo de clases, abstenerse por parcela, devolver un conjunto de etiquetas plausibles, o retroceder a una clase más gruesa. **¿Cuál conviene, y a costa de quién?**

## 2. Hipótesis sustantiva

No es solo un criterio, es una expectativa con su razón, y se declara para poder equivocarnos por escrito:

> **H1.** A igual coste esperado por parcela, los cuatro mecanismos **no se distinguirán** en calidad agregada dentro de una banda de equivalencia de ±0,03, porque todos operan sobre la misma posterior y solo redistribuyen su incertidumbre.
>
> **H2.** Sí se distinguirán en **el reparto**: el recorte de leyenda concentrará la promesa retirada en las clases que retira, y la abstención la repartirá de forma más uniforme, porque el recorte es una decisión por clase y la abstención es una decisión por parcela.

**Qué las refutaría.** H1 se refuta si algún contraste pareado excluye el cero **fuera** de la banda de equivalencia bajo el `k` preregistrado. H2 se refuta si la medida de disparidad declarada no separa a los mecanismos con el intervalo por bloque, o si separa en la dirección contraria.

## 3. Criterio principal

**Se decide después de US-155 y antes de correr nada más.** Hoy no puede fijarse honestamente: medida la potencia con cuatro medidas de disparidad declaradas, **ninguna la tiene con cinco bloques** —las cuatro incluyen el cero y necesitan entre doce y diecisiete—. Si tras fijar `k` por el criterio de §4 la disparidad tiene potencia, es el criterio principal; si no la tiene, el criterio principal es el mapa de decisión de la EPIC 25, que no depende de significancia, y el artículo lo dice.

## 4. Los grados de libertad, declarados

Esto es lo que la vez pasada no existía, y es donde se perdió el resultado anterior.

### 4.1 El número de bloques espaciales

`k` se fija por un **criterio espacial**, decidido y commiteado **antes** de mirar ningún contraste con ese `k`:

> `k` es el mayor valor tal que todo bloque conserva un área mínima declarada y el colchón entre bloques garantiza una separación mínima declarada entre una parcela de prueba y la más cercana de entrenamiento.

**Prohibido explícitamente**: fijar `k` en 15. Es el valor donde el contraste sale significativo, y es a la vez el mayor `|delta|` y la menor desviación de las siete del barrido, o sea la celda más favorable. Elegirlo por eso es p-hacking con otro nombre.

**Se publica la curva entera** de sensibilidad a `k`, gane lo que gane, y `k` no se revisa después de ver un contraste. La sensibilidad **es un resultado**: el estimador se mueve 0,0122 entre valores de `k`, un 73 % del propio efecto y sin tendencia monótona, lo que significa que en validación cruzada espacial la granularidad de la partición decide la significancia — y la literatura del área la trata como un dato.

### 4.2 El suelo de soporte por bloque

Medido en el barrido: **en todos los valores de `k`, algún bloque tiene una sola parcela de alguna clase.** El F1 por bloque lleva calculándose desde la fase 3 sobre bloques donde una clase es un único ejemplo, y **ningún `k` lo arregla**.

> El estimando promedia solo sobre las clases con al menos **S** parcelas en ese bloque, con `S` declarado aquí antes de mirar.

### 4.3 El estadístico de disparidad

Se define **aquí**, no en la historia que lo calcula, porque una medida definida donde se computa no está preregistrada. **No es la mayor razón observada**: es una medida sobre todas las clases que superan el suelo `S`, y se declara qué hacer cuando una clase recibe cobertura cero en un bloque, porque el número de clases retenidas varía entre 6 y 9 según el bloque y una razón puede no estar definida.

### 4.4 La función de coste

`g(|C|)` de la familia de pérdidas se declara aquí, con su análisis de sensibilidad, y la normalización deja el problema en **dos** razones libres.

## 5. Universos y multiplicidad

Los dos universos de clases se reportan los dos. El criterio principal se reporta sin corregir; la familia exploratoria, con Holm, excluyendo los puntos donde los mecanismos son idénticos por construcción. La multiplicidad entre bancos, predictores y universos se declara y se trata.

## 6. Bancos: qué es confirmatorio y qué no

**PASTIS y BreizhCrops son exploratorios.** Ya se miraron, y preregistrar una medida sobre datos vistos es un ritual, no un compromiso. El estudio confirmatorio se hace sobre el banco nuevo, y **solo entra como confirmatorio si su partición ajena define bloques suficientes** para el efecto mínimo detectable; si no, entra como descriptivo y se declara.

## 7. Enmiendas

Toda desviación posterior entra como enmienda fechada **antes** de calcular el contraste afectado, incluida cualquiera que toque la regla de entrega, el estimando o `k`.

**Enmienda 3 al preregistro anterior**, que faltaba: se declara que **H1 se refutó** y que la **regla de entrega cambió** del oráculo de etiqueta a la predicción, con el efecto medido de ese cambio.

## 8. Criterio de no envío

Si tras fijar `k` por §4.1 la disparidad no tiene potencia **y** el mapa de decisión no separa regiones, esto es un informe técnico y no un MICAI. **Se evalúa en la fecha atada al sellado de US-134**, con firma en el ADR, y no el día del cierre de envíos, que es cuando ya no se puede actuar.

**El protocolo no puede ser el resultado de reserva.**
