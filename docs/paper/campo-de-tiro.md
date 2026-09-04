# El campo de tiro: cómo nos destruiría un revisor estricto, y qué prueba lo impide

**Fecha**: 3 de septiembre de 2026. **Decisión**: veinte páginas, MICAI y solo MICAI.

Con veinte páginas hay sitio para una sección de robustez **de verdad**, no para una frase de
cortesía. Esta es la lista de ataques que un revisor estricto de MICAI puede montar, cada uno con la
prueba que lo desactiva. Se corren **antes** de escribir: lo que sobreviva va al artículo, lo que no,
cambia la afirmación.

El orden es por letalidad, no por comodidad.

| # | El ataque, tal como lo escribiría | La prueba | Historia |
|---|---|---|---|
| 1 | «En vez de todo esto, entrenen un clasificador mejor» | La ganancia de cada mecanismo frente a la de pasar del peor al mejor predictor, en el mismo eje de coste | US-164 |
| 2 | «Su rechazo por confianza y su conforme suponen posteriores calibradas y no enseñan ni un diagrama de fiabilidad» | Calibración por clase, no agregada, y cobertura **condicional** del conforme, que es la que dice algo del reparto | US-162 |
| 3 | «Todo corre con la semilla 42 y un submuestreo que eligieron ustedes» | Diez semillas sobre todo el pipeline; si una conclusión depende de la semilla, se retira | US-161 |
| 4 | «Su disparidad es la definición de su mecanismo, no un hallazgo» | Descomponer la disparidad en la parte implicada analíticamente y la que no, y declarar cuál es cuál | US-168 |
| 5 | «El 1,62 % de celdas imputadas toca al 63 % de las parcelas» | Repetir sobre las parcelas sin ninguna celda imputada. Y reportar la fracción de **parcelas**, no la de celdas, que subestima el alcance en un factor de treinta y nueve | US-163 |
| 6 | «Eligieron el punto de operación que les conviene» | La curva completa, con el punto preregistrado marcado y distinguido, y el **rango** de la magnitud titular en vez de su máximo | US-169 |
| 7 | «Su punto de operación se elige en un año y se aplica en otro» | Elegirlo en un año y evaluarlo en otro sobre un banco multianual; si no hay banco, se declara la limitación con nombre | US-165 |
| 8 | «Su colchón de un kilómetro es arbitrario y sus bloques filtran» | Varios colchones y resoluciones, con la distancia mínima real entre prueba y entrenamiento por bloque | US-167 |
| 9 | «Las declaraciones de cultivo tienen error conocido: su disparidad puede ser ruido» | Inyección de ruido a tasas documentadas, y a qué tasa deja de sostenerse | US-166 |
| 10 | «¿Por qué estos cuatro mecanismos?» | Situarlos como puntos del retículo de predictores con valores de conjunto, y nombrar los que quedan fuera | US-170 |

## Los que ya tienen respuesta, y de dónde salió

No todos los ataques son nuevos. Estos ya se desactivaron durante las auditorías, y la respuesta
está medida:

| El ataque | La respuesta, con su número |
|---|---|
| «Su estimando alineado no está alineado» | Cierto, y **AÚN NO REPARADO**: US-124 sigue pendiente y el código conserva la conducta. Medido: encogía el contraste un 42,1 % y un bloque cambiaba de signo |
| «Su intervalo remuestrea la unidad equivocada» | Cierto, y **AÚN NO REPARADO**: US-125 sigue pendiente. Y una auditoría externa añade algo peor: partir el mismo territorio más fino no crea réplicas independientes, con Jaccard de 0,4273 medio entre los entrenamientos de los folds, y 0,6000 si se cuenta entrenamiento mas validacion — el 0,60 que publicabamos como «entre entrenamientos» era el segundo |
| «Un brazo elige su punto de operación dentro del bloque que lo mide» | Cierto, y **medido**: valía 0,00127, un 4,4 % del delta. Real y casi inconsecuente |
| «Citan a Jones et al. para respaldar lo contrario de lo que demuestra» | Cierto, y ahora **medido**: **ninguna de las dos asociaciones es significativa**. Bajo recorte, Spearman 0,434 con p = 0,0716 en PASTIS y 0,017 con p = 0,9661 en BreizhCrops; bajo abstención, 0,286 (p = 0,250) y −0,167 (p = 0,668). La versión anterior de esta fila decía «y sí bajo recorte», que es leer un 0,0716 como si fuera un 0,05 |
| «No tienen potencia» | Cierto: efecto mínimo detectable **aproximado** 0,033 contra un efecto de 0,017, y pendiente de rehacer con t no central. **Y la respuesta que estaba escrita aquí era peor que el ataque**: mudar el criterio principal a donde sí hay potencia es elegir el resultado antes de medirlo. El criterio principal se fija en la firma, uno solo y sin regla condicional |
| «Dos bancos franceses no demuestran transporte» | Cierto, y peor: comparten el sistema de etiquetado. Por eso el barrido de desbalance dentro de banco |
| «Su F1-macro no es comparable entre catálogos y lo usan para comparar catálogos» | Cierto. Por eso el eje de coste único y la utilidad declarada |

## El reparto de las veinte páginas

| Sección | Páginas |
|---|---|
| Introducción | 1,5 |
| Trabajo relacionado, por limitaciones | 2 |
| Método: marco de conjuntos, estimando, protocolo, inferencia | 3,5 |
| Resultados | 6 |
| **Robustez: las diez pruebas de arriba** | **2,5** |
| Discusión | 1,5 |
| Limitaciones y conclusión | 1 |
| Apéndice | 0,5 |
| Referencias | 1,5 |
| **Total** | **20**, que es el tope. Los seis puntos y medio de resultados bajaron a seis, aplicando la regla de recorte de abajo: sale de resultados, nunca de robustez |

**Regla de recorte**: si sobran páginas se recorta de resultados hacia el apéndice, **nunca de
robustez**. Un resultado más en la mesa no compensa un flanco abierto, y este equipo ya perdió un
artículo por flancos abiertos, no por falta de resultados.

## Advertencia sobre el tope

Las veinte páginas son el techo que **MICAI 2026** permitía —«up to 12 pages, but it can be larger,
not exceeding 20»—. La convocatoria de 2027 no existe todavía: es la edición del cuarenta
aniversario y siguen buscando sede. **Hay que reverificar el tope en cuanto se publique**, y diseñar
sabiendo que doce sigue siendo la longitud recomendada. Un artículo de veinte páginas que podía ser
de catorce se lee como un artículo que no supo elegir.
