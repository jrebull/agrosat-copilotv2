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
