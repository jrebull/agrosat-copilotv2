# Recomendación final: qué artículo escribir, y qué cuesta

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md): las produjo `ml/eval/paper_micai_coverage.py`
> cuando aun tenia los tres defectos —denominador movil, punto de operacion elegido dentro del
> bloque evaluado, y remuestreo a nivel de parcela—. **Ninguna de esas cifras entra en el
> articulo** hasta regenerarlas (US-124, US-125). Se conservan sin retocar porque el registro de
> lo que creimos importa tanto como lo que resulte.

**Fecha**: 3 de septiembre de 2026. **Decisión del equipo**: sede única MICAI 2027, **veinte páginas**, sin artículo de revista en paralelo. Con ese tope cabe además una sección de robustez de verdad: ver [`campo-de-tiro.md`](campo-de-tiro.md).

Esto es mi mejor recomendación, con el coste declarado y sin suavizar lo que hay que rehacer.

## 1. Qué cambia el tope de 20 páginas

Cambia más de lo que parece. La recomendación de partir el trabajo en dos vehículos existía porque
**doce páginas no aguantaban la rejilla**. Con veinte cabe entera, **y sobra sitio para robustez**: cuatro mecanismos, tres o más
bancos, líneas base, equidad por clase y por tamaño de parcela, diagrama de fases y caso de
despliegue. **No hay que partir nada.** Y lo que sigue sin caber no es una sección: es la autopsia de
nuestros propios errores como primera contribución.

## 2. El artículo, en una frase

> A igual coste, las cuatro maneras de que un mapa de cultivos prometa menos **no se distinguen en
> calidad agregada dentro de la resolución del diseño, y se distinguen muchísimo en quién paga**;
> el reparto es predecible, tiene consecuencias medibles sobre la estadística de superficie, y hay
> una región de costes donde cada mecanismo es el correcto.

**Es la frase que aspiramos a poder escribir, y hoy NO es sostenible.** Decía aquí que sí, y era la
misma sobreafirmación que el artículo denuncia en otros. Tres cosas faltan y ninguna es menor:

1. **«A igual coste» no significa nada todavía**, porque no hay función de pérdida: la cardinalidad
   estuvo ocupando ese lugar sin justificación (US-172).
2. **«No se distinguen» no se deduce de no haber encontrado diferencia.** Ausencia de evidencia no
   es equivalencia; hace falta una prueba de equivalencia contra una banda declarada, y la banda
   sale del usuario, no del instrumento (US-174). Con cinco y dos bloques, además, el diseño
   difícilmente podrá establecerla.
3. **«Se distinguen muchísimo en quién paga» tampoco está medido**: ninguna de las cuatro medidas
   de disparidad declaradas tiene potencia con cinco bloques.

Lo que sí se puede decir hoy: la que perseguíamos —«a igual cobertura gana la abstención»— **no se
sostiene**, y no por falta de trabajo, sino por falta de potencia y por tres defectos de protocolo.

## 3. Por qué no es sostenible la anterior, con los números

Al componer las dos correcciones de protocolo, que nunca se habían aplicado juntas:

| | media | IC t sobre bloques | p |
|---|---|---|---|
| deltas publicados | −0,0288 | (−0,0430, −0,0147) | 0,005 |
| con denominador común | −0,0167 | (−0,0410, **+0,0077**) | **0,130** |

Y el diseño no puede detectar un efecto de ese tamaño: con cinco bloques el efecto mínimo detectable
es **0,0326** con un contraste y **0,0508** bajo Holm con seis, contra un efecto observado de
**0,0167**. Harían falta trece y veinte bloques. Hay cinco y dos.

**La disparidad se midió, y NO tiene potencia de sobra.** Esta frase decía lo contrario y era falsa.
La clase 10 recibe 0,090 de cobertura bajo recorte y 0,741 bajo abstención, que es una razón de ocho
— pero es la **mayor de dieciocho razones cuya mediana es 1,00**, y con las cuatro medidas de
disparidad declaradas el intervalo por bloque **incluye el cero en las cuatro**: harían falta entre
doce y diecisiete bloques. Ni la disparidad ni ningún otro criterio se elige por dónde haya
potencia; el criterio principal se fija en la firma, uno solo, desde la tabla de pérdidas.

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
| Reimplementar un rechazador aprendido | **exclusión de alcance nuestra**, por presupuesto y riesgo de sobreajuste. El precio: el artículo no dice nada sobre mecanismos aprendidos de renuncia |
| Gate sobre el HTML del cuaderno | gobernanza de proyecto, no trabajo de artículo |
| `parcel/18cls` | se puede correr y **no se puede validar**: su configuración nunca entró al barrido |
| La autopsia como primera contribución | las correcciones son obligatorias; la confesión no |

## 7. El presupuesto

> **Fotografía del 3 de septiembre de 2026, obsoleta desde el reencuadre de la EPIC 27.** La
> fuente viva es `make plan-check`, que hoy da **quince épicas, ochenta y nueve historias, 376 SP
> en total, 255 pendientes y un camino crítico de 96 SP** que empieza en US-172. Se conserva la
> tabla vieja fechada en vez de borrarla, porque el motivo del cambio es parte del registro.

| | 3 de septiembre (obsoleto) |
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
