# Auditoría de cuatro revisores estrictos MICAI · veredicto y verificación

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md): las produjo `ml/eval/paper_micai_coverage.py`
> cuando aun tenia los tres defectos —denominador movil, punto de operacion elegido dentro del
> bloque evaluado, y remuestreo a nivel de parcela—. **Ninguna de esas cifras entra en el
> articulo** hasta regenerarlas (US-124, US-125). Se conservan sin retocar porque el registro de
> lo que creimos importa tanto como lo que resulte.

**Fecha**: 3 de septiembre de 2026. **Objeto**: `paper/micai/main.pdf`, 15 páginas.
**Método**: cuatro revisores a ciegas, sin acceso a nuestras conclusiones ni a la revisión de Arthur, con ejes disjuntos: validez estadística, afirmación contra evidencia, novedad, y cumplimiento editorial.
**Resultado**: **los cuatro recomiendan rechazo**, con evidencia que apenas se solapa.

Lo que sigue son solo los hallazgos que **yo mismo he reproducido**. Lo que no verifiqué queda marcado.

---

## 1. El estimando alineado no está alineado — VERIFICADO

Es el defecto más grave, porque es el método central del artículo cometiendo el error que el artículo denuncia.

La ecuación 1 promedia sobre `L ∩ Y`, donde `Y` son las clases presentes **entre las parcelas entregadas**. Los dos mecanismos entregan conjuntos distintos por construcción, así que `L` es común pero `L ∩ Y` **no lo es**. Reproducido sobre el banco primario, `tsvit-pheno`, K = 9:

| bloque | \|L∩Y\| retirada | \|L∩Y\| confianza | δ publicado | δ con denominador común |
|---|---|---|---|---|
| b0 | 9 | 9 | −0,031256 | −0,031256 |
| b1 | 7 | 7 | −0,017962 | −0,017962 |
| **b2** | **9** | **8** | **−0,044753** | **+0,015978** |
| b3 | 8 | 8 | −0,017519 | −0,017519 |
| b4 | 6 | 6 | −0,032698 | −0,032698 |
| | | | **−0,028838** | **−0,016691** |

En b2 la retirada paga un F1 de clase que a la confianza no se le cobra. Ese bloque, el de mayor δ, **cambia de signo** al igualar el denominador, y el contraste principal se encoge un 42 %.

## 2. El remuestreo usa la unidad equivocada — VERIFICADO, y va en las dos direcciones

El estimando declarado es la media **por bloque**, pero `paired_interval` remuestrea parcelas dentro de cada bloque y **nunca remuestrea bloques**. La varianza entre bloques, que es la razón de haber hecho bloqueo espacial, queda fuera.

Los cinco δ del banco primario son todos negativos: −0,0313, −0,0180, −0,0448, −0,0175, −0,0327. Un intervalo t sobre la unidad correcta da **(−0,0430, −0,0147), que excluye el cero**. El artículo publica (−0,0414, **+0,0080**) y concluye «no distinguible de cero».

En el banco de réplica el problema es el contrario: con **dos** bloques la componente entre bloques no es estimable, y ahí está el único resultado que el artículo llama significativo.

## 3. El mecanismo ganador elige su punto de operación dentro del bloque que lo mide — VERIFICADO

`confidence_baseline` usa `confidence[test_pos]`: el umbral sale de la distribución de confianzas **del bloque de evaluación**. La retirada de leyenda elige `L` con `labels[train_pos]`, fuera. El abstract afirma que el protocolo «elige el punto de operación fuera de los datos que lo miden», y eso es cierto para un brazo y falso para el otro — el que gana.

## 4. La premisa sobre la práctica es falsa — VERIFICADO (coincide con Arthur)

Seis apariciones en el artículo, cero citas. El origen es una frase oral de nuestro propio equipo, y el producto desplegado hizo **lo contrario**: `reports/voting_new/cardinalidad.json` ordena por F1 por clase con umbral 0,90, y la columna `cumulative_support_share` prueba que la cobertura sí se midió.

## 5. El abstract contradice a la sección 4.3 — VERIFICADO

Abstract: «abstention outperforms legend shrinking **on both benchmarks**». Sección 4.3: en el primario el contraste es indistinguible de cero, p = 0,336. Y el segundo predictor, cuyo δ es −0,0016 con p de Holm **1,000**, desaparece del apartado donde sería un nulo plano.

## 6. Reporte selectivo entre dos universos preregistrados — VERIFICADO

La enmienda 1 obliga a publicar los dos universos y prohíbe elegir. El artículo imprime el 95,1 % del universo de nueve clases. El universo de siete, que el preregistro declaró **el mensurable**, da **85,5 %**, y esa cifra no aparece en ninguna parte.

## 7. El preregistro se cita como credencial y se omite lo que dice — VERIFICADO

Faltan dos cosas en el apéndice: el **veredicto sobre H1-2026**, hoy pendiente —este apartado decía «H1 se refutó», que es la palabra que después prohibimos; luego «no replicó», que tampoco vale mientras sus cuatro cifras estén invalidadas— y que la regla de entrega **cambió** respecto de la preregistrada (que era el oráculo de etiqueta). El cambio es correcto y es justo el que invirtió el resultado, y no está declarado como enmienda.

## 8. Una contribución anunciada que no existe — VERIFICADO

La introducción dice «ordenamos los tres criterios con intervalos pareados y corrección por multiplicidad». El único contraste calculado es retirada-por-F1 contra confianza. **No hay ni un intervalo para el orden entre los dos criterios de retirada**, que es todo el contenido de la sección 4.4.

Peor: el criterio por soporte se declara «peor» comparándolo a **cobertura distinta** (0,968 frente a 0,816), violando el principio que el propio artículo impone.

## 9. Cumplimiento editorial — VERIFICADO

- **Sin declaración de asistencia de IA.** Springer la exige. Cuatro agentes produjeron las tres correcciones que definen el resultado actual.
- **El envío es un `.zip` del proyecto LaTeX, y el fuente lleva los nombres, los dos ORCID, la institución y los correos** en la rama `\else`. El gate solo mira el PDF. La anonimidad se rompe por el sobre, no por la carta.
- **El gate no normaliza acentos**: `tecnologico de monterrey` nunca puede coincidir con «Tecnológico de Monterrey». Tres de dieciséis tokens son inertes, y falta el ORCID del segundo autor. La autoprueba pasaba porque inyecta el token tal cual: **una prueba que no puede detectar un token que no corresponde a la realidad**.
- **Las seis figuras del artículo en inglés están rotuladas en español**, y su tipografía queda entre 4,0 y 4,8 pt al tamaño final.
- **`siunitx` agrupa la parte decimal**: `0.2155` se imprime `0.215 5`, en unos veinticinco números.
- **Once de veinticinco referencias** salen sin venue ni DOI porque el generador emitió `@misc`.
- **El apéndice cuenta mal**: dice 62 en inglés, 65 en español, y el ledger tiene 73.

## 10. Cosas que no verifiqué y hay que comprobar antes de aceptar

- Que el efecto de agregar clases esté documentado en teledetección desde 1995 (Foody y Embashi; GOFC-GOLD; Gudmann 2019).
- Que el rechazo jerárquico o por clase ya esté formalizado (Deng CVPR 2012; Huang MVA 2014; Goren, Galil y El-Yaniv 2024).
- Que `jones2020selectivedisparities` diga lo contrario de lo que le atribuimos. **Esto sí lo medimos**: ver `reports/paper_micai/equidad/`, y el resultado nos favorece, pero solo porque ahora está medido.
- Que PASTIS-R se introduzca en el ISPRS 2022 y no en el ICCV 2021 que citamos.

---

## Veredicto

El artículo, tal como está, sería rechazado y con razón. Tres de sus defectos estadísticos empujan en la dirección de sus propias conclusiones, y el que no —el del remuestreo— la empuja en la contraria.

Lo que **no** está tocado: la descomposición denominador/mecanismo sigue en pie como fenómeno, las cifras verificadas contra artefactos son exactas (el revisor de estadística confirmó más de treinta), la construcción reproducible byte a byte y los gates son buenos, y la medición de equidad que exigió un revisor salió a favor.

Lo que hay que decidir es si esto se arregla o se reencuadra.
