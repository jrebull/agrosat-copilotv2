# Recomendación final: qué artículo escribir, y qué cuesta

**Fecha**: 3 de septiembre de 2026. **Decisión del equipo**: sede única MICAI 2027, de 16 a 18 páginas, sin artículo de revista en paralelo.

Esto es mi mejor recomendación, con el coste declarado y sin suavizar lo que hay que rehacer.

## 1. Qué cambia el tope de 18 páginas

Cambia más de lo que parece. La recomendación de partir el trabajo en dos vehículos existía porque
**doce páginas no aguantaban la rejilla**. Con dieciocho, cabe entera: cuatro mecanismos, tres o más
bancos, líneas base, equidad por clase y por tamaño de parcela, diagrama de fases y caso de
despliegue. **No hay que partir nada.** Y lo que sigue sin caber no es una sección: es la autopsia de
nuestros propios errores como primera contribución.

## 2. El artículo, en una frase

> A igual coste, las cuatro maneras de que un mapa de cultivos prometa menos **no se distinguen en
> calidad agregada dentro de la resolución del diseño, y se distinguen muchísimo en quién paga**;
> el reparto es predecible, tiene consecuencias medibles sobre la estadística de superficie, y hay
> una región de costes donde cada mecanismo es el correcto.

Esa afirmación **es sostenible con lo que hay**, y es la única que lo es. La que perseguíamos —«a
igual cobertura gana la abstención»— **no lo es**, y no por falta de trabajo: por falta de potencia.

## 3. Por qué no es sostenible la anterior, con los números

Al componer las dos correcciones de protocolo, que nunca se habían aplicado juntas:

| | media | IC t sobre bloques | p |
|---|---|---|---|
| deltas publicados | −0,0288 | (−0,0430, −0,0147) | 0,005 |
| con denominador común | −0,0167 | (−0,0410, **+0,0077**) | **0,130** |

Y el diseño no puede detectar un efecto de ese tamaño: con cinco bloques el efecto mínimo detectable
es **0,0326** con un contraste y **0,0508** bajo Holm con seis, contra un efecto observado de
**0,0167**. Harían falta trece y veinte bloques. Hay cinco y dos.

**En cambio, la disparidad es un factor de ocho**: la clase 10 recibe 0,090 de cobertura bajo recorte
y 0,741 bajo abstención. Ahí el diseño sí tiene potencia de sobra, y ahí va el criterio principal.

## 4. Lo que hay que rehacer, y no es poco

Cuesta lo que cuesta. En orden de impacto:

1. **El estimando, entero.** «A igual cobertura» no está definido para dos de los cuatro mecanismos,
   porque un conjunto conforme y una clase gruesa se entregan en el 100 % de las parcelas siempre.
   Hay que reformular los cuatro como **predictores con valores de conjunto**, con eje único
   `E[|C|]` y una utilidad declarada. Y el estimando actual condiciona sobre una variable posterior
   al tratamiento —evalúa solo lo entregado, y cada brazo entrega otra cosa—, que es **el mismo
   pecado que el artículo denuncia, un nivel más abajo**.
2. **La línea base clásica.** Era Chow 1970, que es el **caso degenerado** de Ha 1997. Citábamos el
   particular y omitíamos el general.
3. **Un banco entero.** CropHarvest no sirve: sus tareas de referencia son **binarias**, y con dos
   clases los cuatro mecanismos colapsan en uno. Se rehace el barrido con la cardinalidad como
   criterio de admisión primero.
4. **El criterio principal**, que se mueve del delta de calidad a la disparidad, y **no se puede
   preregistrar sobre los dos bancos ya mirados**. Se declaran exploratorios y el preregistro
   confirmatorio se hace para los nuevos.
5. **Las figuras**, en inglés y a tamaño legible. Las seis estaban en español y a 4 pt.

## 5. Lo que se añade, y por qué vale la pena

- **El diagrama de fases en espacio de costes.** Ordena lo que ninguna métrica de acierto ordena, y
  **no depende de que ningún contraste salga significativo**. Cuesta cero horas de GPU. Es la figura
  que un revisor recuerda.
- **La decisión de despliegue real, situada en ese mapa.** Es el único activo del proyecto que nadie
  más puede reproducir. El doble ciego exige quitar el nombre, no el hecho.
- **El barrido del desbalance con la geografía fija.** Convierte cuatro bancos en más de cuarenta
  puntos de diseño sin reentrenar nada, y es la única forma de separar «hallazgo» de «artefacto
  europeo» — que es la objeción que un revisor hará y que más bancos no resuelven, porque geografía
  y forma del desbalance están perfectamente confundidas con n = 4.
- **El sesgo inducido en la estadística de superficie.** Es lo que convierte la equidad de tabla
  descriptiva en consecuencia medible, y conecta con el uso real: subvenciones, seguro, estadística
  agrícola.
- **El artículo del resultado nulo, escrito antes de correr**, con banda de equivalencia y criterio
  de no envío. Porque el nulo es probable y hay que llegar a él con la salida escrita, no
  improvisándola en marzo.

## 6. Lo que se recorta, con su motivo

| | motivo |
|---|---|
| Reentrenamiento OOF de cinco folds | existía para alimentar al árbitro, y el árbitro salió del artículo |
| Cuarto banco de datos | tres bancos y el barrido de desbalance ya responden el transporte |
| Reimplementar un rechazador aprendido | riesgo de ajuste que el revisor criticará se haga como se haga |
| Gate sobre el HTML del cuaderno | gobernanza de proyecto, no trabajo de artículo |
| `parcel/18cls` | se puede correr y **no se puede validar**: su configuración nunca entró al barrido |
| La autopsia como primera contribución | las correcciones son obligatorias; la confesión no |

## 7. El presupuesto

| | |
|---|---|
| Historias vivas | 40 |
| Puntos | 191 |
| Camino crítico de lo pendiente | 50 SP |
| Conocimiento nuevo | **76 SP, el 40 %** — era el 18 % |

Y una advertencia sobre la escala que conviene no olvidar: los puntos de este proyecto se
calibraron sobre trabajo de CPU y prosa, donde un punto son unos diez minutos. **Nada de la EPIC 22
se parece a eso.** Con descargas, cargadores nuevos y tres personas, esto son entre diez y catorce
semanas de reloj, y el plazo contra el que corren no existe todavía porque MICAI 2027 aún no tiene
convocatoria ni sede.

## 8. El riesgo que asumo al recomendar esto

Que la disparidad tampoco separe en los bancos nuevos. Si eso pasa, **no hay artículo de MICAI**, y
el criterio de no envío de US-156 obliga a decirlo en vez de enviar el andamio. Prefiero ese riesgo
declarado a la alternativa, que es la que ya nos pasó dos veces: llegar a marzo con un resultado
débil y buscarle un encuadre que lo admita.
