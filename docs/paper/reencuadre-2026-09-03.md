# Reencuadre del artículo MICAI 2027

**Fecha**: 3 de septiembre de 2026. **Motivo**: cuatro revisores a ciegas recomendaron rechazo por caminos distintos, y la revisión de Arthur encontró una premisa falsa. Tres de los defectos estadísticos los reproduje yo mismo. Ver [`auditoria-revisores-2026-09-03.md`](auditoria-revisores-2026-09-03.md) y [`revision-arthur-2026-09-03.md`](revision-arthur-2026-09-03.md).

**Decisión**: no se parchea. Se reencuadra, con el calendario largo que la convocatoria de 2027 permite.

---

## 1. Qué muere, qué sobrevive

| Del artículo actual | Destino |
|---|---|
| «Casi toda la mejora es el denominador» como **contribución central** | **Baja de rango.** El efecto de agregar clases está documentado en teledetección desde los noventa. Pasa a ser una recomendación de reporte con su precedente citado |
| «El F1-macro no es comparable entre catálogos» | Sobrevive como **premisa**, no como hallazgo |
| «Retirar por soporte es lo que hace la práctica» | **Muere.** Es falso: el equipo retiró por F1 por clase con umbral 0,90, y midió la cobertura |
| «A igual cobertura gana la abstención» | Sobrevive **si y solo si** se rehace con umbral simétrico, denominador común y remuestreo por bloque |
| El orden entre criterios de retirada y su no monotonía | **Sube a contribución.** Es lo único que un revisor llamó no trivial. Le faltan intervalos |
| El árbitro que retira una clase sin declararlo | **Sale del artículo.** Es un fenómeno de otro tema, con n=1 y una explicación más simple sin excluir |
| El estimando alineado, el remuestreo pareado, la entrega sin oráculo | Sobreviven **corregidos**, y su corrección pasa a ser contribución: se demuestra qué cuesta cada error |

## 2. El ángulo nuevo

> **Prometer menos clases, responder menos parcelas, o responder con un conjunto.** Puntos de operación para mapeo de cultivos con desbalance extremo.

Un producto de mapeo que no alcanza calidad tiene tres salidas, no dos, y la tercera es la que nadie ha medido en este dominio: **devolver un conjunto de cultivos plausibles** en lugar de una etiqueta o de nada. Es el comparador moderno —predicción conforme y clasificación con conjuntos— y su ausencia es lo que hacía estrecho el artículo anterior.

**Contribuciones, en orden de fuerza:**

1. **Un protocolo que hace comparables los mecanismos, y la medida de lo que cuesta cada error al construirlo.** Tres defectos concretos, cada uno con su efecto medido: denominador no común (invierte el signo de un bloque y encoge el contraste un 42 %), unidad de remuestreo equivocada (decide si el intervalo cruza el cero), y elección asimétrica del punto de operación (favorece a un brazo). Esto no es metodología de relleno: es el resultado de habernos equivocado en los tres y haberlo medido.
2. **El tercer y cuarto mecanismo**: conjuntos de etiquetas y retroceso jerárquico, frente a los dos que ya teníamos, todos a igual cobertura.
3. **Contabilidad de equidad: quién paga.** Cobertura por clase bajo cada mecanismo. Ya medido, y **con la cifra corregida el 3 de septiembre**, porque la primera versión mezclaba dos filas: en la clase 5, con 198 parcelas, el recorte atiende el **20,2 %** y la abstención el **63,6 %**; en la clase 12, con 103 parcelas, el 27,2 % frente al 62,1 %. Ninguna fila daba «20 y 62». Y el recorte entrega respuestas **garantizadamente falsas** para las clases que retiró, mientras la abstención no entrega nada.
4. **El orden entre criterios de retirada**, con cobertura igualada e intervalos, que hoy no tiene ninguno.
5. **La descomposición denominador/mecanismo** como recomendación de reporte, con su precedente reconocido en la primera página.

## 3. Qué exige el reencuadre, y no es poco

- **Cuatro bancos de datos, y al menos uno fuera de Europa occidental.** Dos bancos franceses no demuestran transporte, y así lo dijeron.
- **Varios predictores por banco**, no uno.
- **Líneas base de verdad**: la regla de Chow, rechazo aprendido, predicción conforme, retroceso jerárquico. Hoy el artículo no compara contra ninguna línea base de la literatura que dice extender.
- **Un preregistro nuevo** del experimento central, escrito antes de correrlo, que declare la hipótesis y qué la refutaría.
- **Reconocer el precedente** de los noventa en la primera página, no en una nota.

## 4. Contraste con la lista de Arthur

Su diagnóstico y el de los revisores **no se solapan y los dos son necesarios**. Él revisó el proyecto; ellos revisaron el artículo.

| Lo que Arthur pedía | Estado en este plan |
|---|---|
| Ruta 0: cerrar la deuda del harness OOF (áreas A, B, C, D, G, H) | **EPIC 18**, y sube a bloqueante: sin ella la frase «los dos miembros más fuertes» seguirá siendo falsa |
| Área D: sanidad de `utae`, `anysat`, `segformer` antes de afirmar nada | **US-119**, antes de la tabla de miembros |
| Área E: tres columnas de régimen en toda tabla de ensambles | **US-118**, y el árbitro sale del artículo, así que deja de ser crítico para el paper |
| Área F: identidad de `tsvit-pheno-fullm-v2` | **US-120** |
| Área J: fuente única de verdad | **US-123**, y es lo que explica la mitad de su evaluación |
| Área K: gobernanza, consentimiento, divulgación de IA | **US-121 y US-122**, y los revisores lo confirmaron como incumplimiento formal |
| Ruta 4: plan de cómputo para el reentrenamiento OOF | **US-138**, con las cuatro opciones que él costeó |
| US-098 relabelada: el criterio del equipo fue F1, no soporte | Hecho, y va más lejos: **la premisa falsa sale del artículo entero** |
| Incluir el producto desplegado (Voting-3 v2) | **US-139**, como uno de los predictores |
| Calendario holgado hasta mayo de 2027 | Adoptado, con las fases nuevas encajadas |

Lo que él no vio, porque revisaba el proyecto y no el manuscrito: los tres defectos del aparato inferencial, el reporte selectivo entre universos preregistrados, y que el gate de anonimato no normaliza acentos.

Lo que los revisores no vieron, porque no tenían su repositorio: que el volcado de `fullm` está roto, que la VM se perdió con tres checkpoints, y que hay dos fuentes de verdad.


---

## 5. El hueco, afilado por el barrido bibliográfico (3 de septiembre, tarde)

El barrido de 2024–2026 **estrecha el hueco y lo mejora**, porque un hueco preciso se defiende y uno amplio se tumba. Los nueve DOI se verificaron contra Crossref y los dos de arXiv contra su API, por mí, no solo por el barrido.

**Lo que ya no podemos decir:**

- *«Nadie abstiene en mapeo de cultivos.»* Rey et al. 2025 excluyen píxeles por umbral de incertidumbre **sobre PASTIS, con U-TAE, UNET3D y TSViT** —nuestro banco y nuestras arquitecturas—. Lo decisivo es cómo: en todo el trabajo no aparecen «abstention», «selective classification» ni «risk-coverage», y lo conforme se menciona una vez en trabajo relacionado. Es abstención de facto, sin marco, sin curva riesgo-cobertura y sin preguntar quién paga.
- *«Recortar el catálogo no se ha medido.»* Ghassemi et al. 2025 tratan el tamaño del catálogo como variable de diseño medida: de las 52 clases de LUCAS 2022 concluyen que **26 equilibran exactitud y detalle**. En cobertura del suelo general, con clasificación plana y jerárquica, sin marco de abstención y sin contabilidad de quién pierde.
- *«El desbalance extremo es nuestra premisa original.»* Wang et al. 2026 sacan **F1 de 36,72 % sobre 101 variedades** en H2Crop intentando salvar las clases raras. Es la mejor evidencia publicada de que un catálogo puede exceder lo que el modelo distingue, y sus autores no sacan la conclusión de diseño. Nosotros sí podemos.

**Lo que sí queda, y es más defendible que lo anterior:**

> Las tres piezas existen en la literatura de cultivos y **ninguna se habla con las otras**. La jerarquía normativa está montada y validada —HCAT4, EuroCrops v2.0 con 47 millones de parcelas— pero solo se usa como supervisión, nunca como opción de repliegue. El rechazo existe, pero por **novedad** (Carvalho 2023, Xu 2026, Giménez 2023) o como umbral ad hoc de un solo punto. Y lo conforme llega a cobertura del suelo, pero no a tipo de cultivo por parcela desde series temporales. El único trabajo de observación de la Tierra que dice «abstain» con umbrales interpretables, SHRUG-FM en CVPR EarthVision 2026, evalúa incendio, inundación y deslizamiento: **ninguna tarea agrícola**.

**La contribución, reformulada:** no es el mecanismo, es **la contabilidad**. Medir, para cada uno de los tres mecanismos, **qué cultivos y qué tamaños de parcela absorben la promesa retirada**. Eso, en teledetección, no lo ha hecho nadie, y encaja exactamente con lo que ya medimos en `reports/paper_micai/equidad/`.

**Y una regla de redacción que sale de aquí.** Las búsquedas negativas de este barrido tienen un sesgo declarado: OpenAlex carece de resumen para buena parte de Elsevier e IEEE, así que un cero suyo infravalora. En el artículo se escribe **«no encontramos trabajo que…», nunca «nadie ha…»**, y se declara el alcance de la búsqueda. Un revisor con un contraejemplo tumba una afirmación absoluta, y no la tumbaría dos veces.


## 6. Los bancos ya están elegidos, y con partición espacial ajena

El barrido de bancos cierra el punto que más pesaba: **CropHarvest** (Zenodo, CC BY-SA 4.0, con separación espacial explícita y documentada) y **GEO-Bench `m-SA-crop-type`** (Sudáfrica, CC BY 4.0). Los dos traen su propia partición, así que no la construimos nosotros — que es exactamente lo que un revisor cuestionaría si la inventásemos. Detalle y descartes en [`bancos-candidatos.md`](bancos-candidatos.md).

Cuatro bancos, dos continentes, y ninguno con la partición hecha en casa.

---

## 7. Segunda vuelta: de describir a decidir (3 de septiembre, noche)

Una revisión estratégica a ciegas del plan devolvió un juicio que la aritmética confirma sin
discusión: **de los 150 puntos planificados, solo 27 producían conocimiento nuevo. El 18 %.** El
resto era rehabilitación. Es lo que pasa después de cuatro rechazos, y es exactamente la trampa.

### Lo que cambia

**La primera contribución dejaba de ser un resultado para ser una confesión.** El plan abría con
«un protocolo que hace comparables los mecanismos, y la medida de lo que cuesta cada error al
construirlo», es decir, con la autopsia de tres errores propios. Ningún revisor recuerda una
enmienda; recuerda un resultado. Y un artículo que abre explicando qué hizo mal invita al comité a
evaluar la versión anterior. **La autopsia baja a método y apéndice.**

**El artículo se mordía la cola.** Denuncia que el F1-macro no es comparable entre catálogos, y
luego iba a comparar con F1-macro cuatro mecanismos que cambian el catálogo. Los cuatro producen
errores **cualitativamente distintos** —una etiqueta segura y falsa, un silencio, una ambigüedad
acotada, una verdad gruesa— y ninguna métrica escalar de acierto los ordena.

**La EPIC 25 es la respuesta**: evaluación en espacio de costes, no de aciertos. Una familia de
pérdidas con cuatro parámetros, un barrido, y un **diagrama de fases** que dice qué mecanismo gana
en qué región. Encima, cuatro casos de uso agrícolas con sus razones de coste. **Cero horas de
GPU**: es re-puntuar predicciones que ya existen. Es el mejor cociente entre impacto y esfuerzo de
todo el plan.

**Y recupera el único activo que nadie puede reproducir.** Este equipo tiene algo que casi nadie en
esta literatura tiene: **una decisión de punto de operación tomada de verdad en un producto
desplegado** —de 18 clases a 12, criterio F1 por clase ≥ 0,90, cobertura resultante 0,88269 —14 688 parcelas de 16 640—, no 0,9054: esa cifra es el cumulative_support_share de la curva del Stacking-5, y se nos cruzó entre artefactos— con su
contrafactual medible. El reencuadre la había borrado («el artículo ya no trata del sistema») y eso
era un error: **el doble ciego exige quitar el nombre, no el hecho**. Cualquiera con una GPU y
PASTIS reproduce el resto del artículo. La decisión que este equipo ya tomó en producción no la
reproduce nadie. Va al mapa, como estudio de caso, con la pregunta que nadie publica: dado su coste
implícito, ¿acertó?

### Lo que se recorta, y por qué

- **US-138, reentrenamiento OOF de cinco folds.** Su rol declarado empieza «como equipo cuyo
  meta-modelo se entrena con un solo fold», y el reencuadre ya sacó el árbitro del artículo. Ocho
  puntos y presupuesto de nube al servicio de una sección que no existe.
- **US-137, cuarto banco.** Tres bancos y dos continentes ya responden la objeción de transporte.
- **US-150**, gobernanza disfrazada de trabajo de artículo.
- **La reimplementación de un rechazador aprendido**, dentro de US-132. Chow, conforme y retroceso
  jerárquico ya son un conjunto honesto de líneas base; reimplementar un SelectiveNet es un
  proyecto de investigación en riesgo de ajuste que un revisor criticará se haga como se haga.

### Las tres escotillas, cerradas

El plan tenía una salida en forma de protocolo para cada desenlace empírico: «si el resultado no
sobrevive a la simetría, se dice», «si la conclusión no se transporta, el artículo gana un matiz»,
«si no replica, la contribución pasa a ser el protocolo». **Un plan con una escotilla para cada
desenlace aterriza en la escotilla.** Eso no es honestidad, es un compromiso previo con el
resultado menos interesante. Queda escrito en US-126: **el protocolo no puede ser el resultado de
reserva**. Si el experimento central no da nada, se va a revista y se espera al año siguiente.

### Dos vehículos

Doce páginas de actas no aguantan la rejilla que este plan produce, y unas actas rara vez se
convierten en la referencia metodológica de nadie.

- **MICAI 2027, doce páginas: el artículo de decisión.** Tres bancos, cuatro mecanismos, el
  diagrama de fases, la contabilidad de quién paga y la decisión de despliegue real como estudio de
  caso. Una figura que la gente fotografía.
- **Revista, el artículo de protocolo.** Los cuatro bancos, las multiplicidades, los nulos, la
  autopsia y el arnés. Una revista tiene sitio para eso y lo premia.

Y una advertencia que conviene no olvidar: MICAI no es la sede donde vive la comunidad de mapeo de
cultivos. El techo de este trabajo en MICAI es estar bien recordado **en MICAI**. Si el objetivo es
cambiar cómo trabaja el área, el vehículo que llega a esa gente es la revista. Se envía a MICAI el
artículo con postura, y no se confunde la aceptación con haber llegado.

### El presupuesto, después

| | antes | después |
|---|---|---|
| Puntos planificados | 150 | 152 |
| Puntos que producen conocimiento nuevo | 27 (**18 %**) | 48 (**32 %**) |

---

## 8. Tercera vuelta: al reparar el protocolo, el hallazgo se disuelve (3 de septiembre, noche)

Un *red team* a ciegas encontró lo que ninguna de las auditorías anteriores había compuesto, y lo he
verificado con el artefacto sellado del diagnóstico. **Es el hecho más importante de todo el
proyecto.**

### El hallazgo titular ya no existe

Las dos correcciones de la EPIC 19 nunca se habían aplicado **juntas**. Compuestas:

| | media | IC t sobre bloques | p |
|---|---|---|---|
| deltas publicados | −0,028838 | (−0,0430, −0,0147) | **0,005** |
| con denominador común | −0,016691 | (−0,0410, **+0,0077**) | **0,130** |

**Incluye el cero.** Y falta aplicar la simetría del umbral, que por construcción empuja aún más
hacia cero. La EPIC 19 no repara el hallazgo: **lo disuelve**. El plan estaba escrito como si la
reparación fuera un impuesto metodológico que se paga y se sigue; en realidad es la refutación.

De paso, el objetivo de la EPIC 19 decía «los tres empujaban hacia la conclusión que publiqué», y
es falso: el del remuestreo empuja al contrario, y la asimetría del umbral solo valía 0,00127.
Corregido.

### El diseño no puede detectar lo que busca

Con sd entre bloques de 0,0196 y cinco bloques:

| escenario | efecto mínimo detectable |
|---|---|
| un contraste, α = 0,05 | **0,0326** |
| seis contrastes con Holm, que es lo que traen cuatro mecanismos | **0,0508** |
| efecto observado | **0,0167** |

Harían falta **13 bloques** con un contraste y **20** bajo Holm. Hay 5 y 2. Y la EPIC 20, al pasar de
dos mecanismos a cuatro, **sube el umbral de 0,033 a 0,051 mientras el efecto real es 0,017**: añade
multiplicidad sin añadir unidad de análisis.

### Dónde sí hay efecto

> **RETIRADO por la enmienda §8 de este mismo documento, y se deja escrito para que se vea el
> error.** Lo que sigue en este apartado es falso por dos caminos: ocho es el máximo de dieciocho
> razones cuya mediana es 1,00, y ninguna de las cuatro medidas de disparidad declaradas tiene
> potencia con cinco bloques. Además, «el criterio principal se mueve donde hay potencia» es elegir
> el resultado antes de medirlo.

En la disparidad, y por goleada. La clase 10 del banco primario, con 355 parcelas, recibe **0,090 de
cobertura bajo recorte y 0,741 bajo abstención: un factor de ocho.** Frente a un delta de calidad de
0,0167 que no llega a significativo.

**El criterio principal del artículo se mueve ahí**, que es la magnitud donde el diseño tiene
potencia. La contabilidad de equidad deja de ser la tercera contribución y pasa a ser la tesis.

### Un banco no puede alojar el experimento

**CropHarvest no sirve**, y es verificable en diez minutos: sus tareas de referencia —Kenia, Togo,
Brasil— son **binarias**. Con K = 2 el recorte a K = 1 es degenerado, el retroceso jerárquico no
tiene taxonomía sobre la que retroceder, y el conjunto conforme `{0,1}` **es** la abstención: los
cuatro mecanismos colapsan en uno. `bancos-candidatos.md` eligió por partición espacial y licencia,
que son criterios correctos, y **no comprobó la cardinalidad del espacio de etiquetas**, que para
este artículo es el criterio de admisión número uno. Añadido, y el barrido se rehace.

### El eje común que falta, y que existe desde 1997

«A igual cobertura» **no está definido** para dos de los cuatro mecanismos: un conjunto conforme y
una clase gruesa se entregan en el 100 % de las parcelas siempre. La solución no es un parche sino
reformular los cuatro como **predictores con valores de conjunto**, con un eje único
`E[|C(x)|]` definido para todos. Eso está publicado: **Ha 1997** da la regla óptima de rechazo
selectivo de clases, y **Mortier et al. 2021 y 2022** ya comparan el retroceso jerárquico con los
conjuntos. Nuestra línea base clásica era Chow 1970, que es el **caso degenerado** de Ha. Citábamos
el caso particular y omitíamos el general.

### El sesgo del superviviente, admitido

El *red team* lo señala y tiene razón: la sección 1 de este documento es literalmente una tabla de
supervivencia («del artículo actual → destino»), y la sección 5 dice que la contribución de equidad
«encaja exactamente con lo que ya medimos». Un reencuadre dirigido por la pregunta empieza por la
pregunta; este empezó por los artefactos y buscó qué pregunta los admitía. Queda dicho, y la
US-157 —manipular el desbalance dentro de un banco, con la geografía fija— es la respuesta que no
depende de qué artefacto sobrevivió.


---

## 9. Enmiendas, fechadas (3 de septiembre de 2026, noche)

Este documento se escribió en tres vueltas y el plan siguió cambiando después. Estas son las
diferencias, para que quien ejecute no lea una versión superada.

| Dice arriba | Vale esto |
|---|---|
| «Cuatro bancos de datos» (§3, §6) | **Tres.** CropHarvest se cayó por cardinalidad binaria y el cuarto se recortó. El transporte se apoya en un solo sistema de etiquetado ajeno, y eso se declara en el artículo |
| «MICAI 2027, doce páginas» (§7) | **Veinte páginas**, decidido el 3 de septiembre |
| «Dos vehículos: … revista, el artículo de protocolo» (§7) | **Un solo vehículo: MICAI.** Con veinte páginas cabe todo y no hay artículo de revista en paralelo |
| «El criterio principal se mueve a la disparidad» (§8) | **Es una hipótesis, no un hallazgo.** Medida la potencia, ninguna de las cuatro medidas de disparidad la tiene con cinco bloques. La decisión depende de US-155 |
| «Un factor de ocho» (§8) | Es el **máximo de dieciocho razones cuya mediana es 1,00**, leído tras mirar la tabla, y no es monótono en soporte |

Y una cosa que no estaba en ninguna vuelta y cambia el diagnóstico: **cinco bloques era el valor por
defecto de una función, no una restricción del dato**. El fold retenido tiene 176 celdas H3, y con
quince bloques el contraste excluye el cero y tiene potencia. Eso abre la puerta a recuperar el
hallazgo y abre a la vez un grado de libertad que nadie había preregistrado. Ver US-171.
