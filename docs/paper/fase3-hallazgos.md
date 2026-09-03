# Fase 3 — la frontera rehecha, y la hipótesis no se sostiene

**Fase**: 3 de [`docs/plan-micai-2027.md`](../plan-micai-2027.md), bajo el preregistro de
[`preregistro-fases-3-4.md`](preregistro-fases-3-4.md) y tras la
[auditoría ciega](auditoria-2026-09-02.md).
**Fecha**: 2 de septiembre de 2026. **Artefactos**: `reports/paper_micai/fase3/`.
**Código**: [`ml/eval/paper_micai_coverage.py`](../../ml/eval/paper_micai_coverage.py) y
[`scripts/run_paper_micai_fase3.py`](../../scripts/run_paper_micai_fase3.py).

> Rehecho con los tres defectos corregidos: estimando alineado, entrega sin oráculo de
> etiqueta y remuestreo pareado con mil réplicas. Más lo que el preregistro añadía: segundo
> predictor, tercer mecanismo anclado en la práctica del equipo, corrección por
> multiplicidad y un control sin mecanismo.

---

## 1. El veredicto sobre H1

**H1 era**: a igual cobertura, recortar la leyenda produce mayor F1-macro que la abstención
por confianza. **Criterio principal preregistrado**: K = round(C/2) = 9, media entre bloques
del delta, un solo contraste por predictor.

| Predictor | Delta a K = 9 | IC 95 % | ¿Excluye el cero? | p de Holm |
|---|---|---|---|---|
| tsvit-pheno | **−0,0288** | [−0,0414, +0,0080] | no | 0,336 |
| xgb-alphaearth | **−0,0016** | [−0,0237, +0,0252] | no | 1,000 |

**H1 no se sostiene.** El intervalo incluye el cero en los dos predictores y el punto
estimado va, si acaso, en dirección contraria. Y con la corrección de Holm sobre la familia
de siete valores de K **ningún contraste sobrevive** en ninguno de los dos predictores: los
tres que a K = 16, 14 y 12 tenían el intervalo crudo fuera del cero pasan a p = 0,084.

Esto es exactamente lo que el preregistro contemplaba en su sección de refutación. Se
reporta tal cual, y la contribución pasa a ser el protocolo y la explicación del efecto.

## 2. Lo que sí demuestra el experimento, y es más interesante

El control que la versión retirada no tenía —**el predictor intacto, puntuado sobre la misma
leyenda, entregando todas las parcelas**— es el que da la respuesta.

Sobre `tsvit-pheno`, al pasar de dieciocho clases a ocho:

| Concepto | F1-macro | Aporte |
|---|---|---|
| Leyenda completa, sin tocar nada | 0,5612 | punto de partida |
| Ocho clases, **sin mecanismo alguno** | 0,7767 | **+0,2155 solo por cambiar el denominador** |
| Ocho clases, retirando por F1 | 0,8052 | +0,0285 sobre lo anterior |
| Ocho clases, rechazando por confianza | 0,8394 | +0,0627 sobre lo anterior |

**El 87 % de la mejora aparente al recortar la leyenda no es el mecanismo: es la métrica.**
Basta con dejar de promediar sobre las clases difíciles para que el número suba dos décimas,
sin que el producto haya mejorado en nada ni haya dejado de responder una sola parcela.

Ese es el resultado del artículo, y sobrevive a la corrección por multiplicidad porque no es
un contraste: es una descomposición.

## 3. El criterio que el equipo usó de verdad es el peor de los tres

El tercer mecanismo —retirar por **soporte bajo**, que es lo que el equipo declaró haber
hecho para desplegar— queda sistemáticamente último:

| K | Retirada por F1 | Retirada por soporte | Rechazo por confianza |
|---|---|---|---|
| 12 | 0,6570 | **0,6075** | 0,6760 |
| 10 | 0,7529 | **0,6234** | 0,7677 |
| 9 | 0,7645 | **0,6449** | 0,7933 |
| 8 | 0,8052 | **0,6845** | 0,8394 |

Retirar las clases con menos muestra es intuitivo y es la práctica documentada, pero retirar
las que el modelo confunde da entre 0,12 y 0,13 más de F1-macro a la misma cobertura. Es un
hallazgo accionable y no depende de la hipótesis que se cayó.

## 4. El control que prueba que el arnés está bien

A K = 18 los dos mecanismos son literalmente el mismo objeto. El delta sale **+0,0000** y el
intervalo sale **[+0,0000, +0,0000]**, degenerado como debe ser. Esa era justamente la fila
que en la versión retirada devolvía [−0,0294, +0,0308] y delataba el remuestreo sin parear.
El arnés nuevo la pasa.

## 5. La réplica en el segundo predictor

Sobre `xgb-alphaearth` (F1-macro 0,5913) el patrón se repite en lo esencial y con deltas aún
más pequeños: todos los intervalos incluyen el cero y todos los p de Holm valen 1,000. La
descomposición también se sostiene: de 0,4106 con leyenda completa a 0,5564 con ocho clases
**sin mecanismo**, frente a 0,6059 retirando por F1.

Que el efecto sea igual de nulo en dos predictores de calidad muy distinta refuerza que no
es una propiedad de un modelo: **no hay efecto que atribuir al mecanismo**.

## 6. Qué cambia esto en el artículo

La tesis deja de ser «un mecanismo domina al otro» y pasa a ser:

> El F1-macro no es comparable entre catálogos de distinto tamaño. En mapeo de cultivos
> desbalanceado, recortar la leyenda de dieciocho clases a ocho sube el F1-macro en 0,22 sin
> que el modelo cambie ni deje de responder una parcela, mientras que los mecanismos que se
> le atribuyen esa mejora aportan menos de 0,07. Comparar sistemas con catálogos distintos
> por su F1-macro mide sobre todo el catálogo.

Y como corolario accionable: si aun así se decide recortar, hacerlo **por confusión y no por
soporte** vale entre 0,12 y 0,13 de F1-macro.

## 7. Lo que queda

- La figura de la frontera está en `frontera.svg` y `frontera.png`, con los tres mecanismos,
  el control y el criterio principal marcado.
- El intervalo por clúster de parche se calcula y se guarda junto al de parcela; ninguno
  cambia un veredicto.
- Falta la réplica en BreizhCrops (fase 4). El preregistro obliga a correrla igual, y ahora
  su papel cambia: ya no confirma H1, sino que comprueba si la **descomposición** se
  transporta a otro conjunto de datos y a otro reparto de clases.
