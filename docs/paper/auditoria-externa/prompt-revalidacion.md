# Re-validación externa · ronda 2 en adelante

**Para quién**: el mismo auditor externo de la ronda anterior, o uno nuevo si se prefiere sangre fresca. Se usa después de cada ronda de correcciones, hasta que la respuesta a la última pregunta sea «arrancamos».

---

## El prompt

> Auditaste este plan de investigación y tu veredicto fue **no arrancar todavía**. El equipo dice haber corregido. Tu trabajo ahora tiene tres partes, y la tercera es la que importa.
>
> **Dónde está todo**: el plan en `agrosat-micai-site/plan.html`, variable `EPICS`. Los artefactos en `agrosat-copilotv2/reports/paper_micai/`. El registro de custodia en `paper/ARTIFACTS.md`. El código en `ml/eval/`. Puedes correr `make plan-check`, `make paper-artifacts-check` y `pytest tests/ml/eval/`.
>
> **Lo que el equipo dice haber hecho** está en `docs/paper/respuesta-auditoria-externa.md`, con un apartado por cada hallazgo tuyo.
>
> ---
>
> **Parte 1 · Verifica, no creas.**
>
> Para cada hallazgo tuyo, comprueba en el código y en los artefactos si está corregido de verdad. Un hallazgo se cierra cuando el comportamiento cambió, **no** cuando existe una historia de usuario que dice que se cambiará, ni cuando un documento afirma que se corrigió. Ese fue uno de tus propios hallazgos y merece la pena volver a aplicarlo aquí.
>
> Marca cada uno: **CERRADO** (con la evidencia que lo demuestra) · **PARCIAL** (qué falta) · **ABIERTO** (por qué no cuenta) · **PEOR** (la corrección introdujo algo nuevo).
>
> **Parte 2 · Lo que la corrección rompió.**
>
> Arreglar cosas rompe otras. Busca en concreto:
> - Cifras que dependían de lo corregido y no se han recalculado.
> - Afirmaciones en documentos o en el cuaderno público que la corrección deja obsoletas.
> - Dependencias del plan que ya no tienen sentido tras reordenar.
> - Tests que pasan porque siguen probando la conducta vieja.
>
> **Parte 3 · Lo nuevo, y es lo que de verdad pagamos.**
>
> No te limites a tu lista anterior. Cada ronda de correcciones es una oportunidad de meter defectos nuevos, y el equipo ya ha demostrado que los mete. Busca especialmente:
> - **Un número que exista solo en prosa.** Ya ha pasado cuatro veces en este proyecto.
> - **Una cifra tomada de un artefacto y atribuida a otro contexto.** Ha pasado tres veces.
> - **Un control que no puede detectar aquello para lo que existe.** Ha pasado dos veces: un gate ciego a sus propios acentos, y un test que usaba el único valor que no distinguía.
> - **Una decisión presentada como restricción del problema.** Ha pasado con el número de bloques.
> - **Una afirmación que el diseño no puede sostener**, escrita en indicativo.
>
> ---
>
> **Reglas**: verifica en vez de confiar; no inventes referencias; puedes correr código; ordena por severidad; y di también qué está bien, para que no lo rompan en la ronda siguiente.
>
> **Formato**: (a) el hallazgo original o el nuevo; (b) estado con su evidencia; (c) qué falta; (d) **BLOQUEANTE / IMPORTANTE / MENOR**.
>
> **Termina con una sola línea**: **¿arrancamos, o hay que tocar algo antes?** Y si es lo segundo, di cuál es el único cambio que más acerca el «sí».

---

## Cómo se cierra el ciclo

1. El auditor responde.
2. El equipo corrige y **actualiza `respuesta-auditoria-externa.md`** con un apartado por hallazgo, diciendo qué se cambió y dónde se puede comprobar.
3. Se vuelve a lanzar este mismo prompt.
4. Se repite hasta que la última línea diga **arrancamos**.

**Una regla para el equipo, no para el auditor**: no se responde a un hallazgo con una historia de usuario. Se responde con un cambio de comportamiento y con la manera de comprobarlo. Si el arreglo de verdad es futuro, se dice que sigue **ABIERTO** y se explica por qué, en vez de darlo por cerrado.

---

## Estado para la ronda 3

La ronda 2 encontró **dos cierres falsos y tres defectos nuevos**, y el veredicto siguió siendo *no
arrancamos*. Antes de relanzar el prompt de arriba, esto es lo que cambió, para que el auditor lo
verifique en vez de creerlo:

| Qué se tocó | Dónde comprobarlo |
|---|---|
| `g` ya no recibe el cero, y hay un espía que lo prueba | `ml/eval/set_valued.py`, `tests/ml/eval/test_set_valued.py::test_la_utilidad_nunca_le_pasa_un_cero_a_g` |
| La cardinalidad dejó de ser «el eje de coste»: `cardinalidad_esperada` no acepta ninguna `g` | `ml/eval/set_valued.py` |
| Distancia mínima con KD-tree exacto; las tres cifras publicadas cambian | `scripts/run_paper_micai_parametros_prereg.py`, `reports/paper_micai/prereg/parametros_diseno.csv` |
| Barrido de colchón con productor y artefacto (antes solo prosa) | `reports/paper_micai/prereg/barrido_colchon.csv` |
| Jaccard con los dos conjuntos nombrados por separado | `parametros_prereg.json`, clave `solapamiento_entre_bloques` |
| El denominador móvil del suelo `S`, medido y publicado como problema abierto | `parametros_prereg.json`, clave `universo_de_clases_por_bloque`; §4.2 del preregistro |
| Banda de equivalencia retirada del productor, con su motivo | `parametros_prereg.json`, `banda_equivalencia: null` |
| Gate de contradicción «sin dependencias» con dependencias declaradas | `scripts/plan_check.py`, `make plan-check` |
| `H1-2026` frente a `H1`; retirada la regla condicional del criterio principal | `docs/paper/preregistro-v2-borrador.md` §2 y §3 |
| 0,88269 renombrada a cuota de soporte de la verdad en las tres apariciones | `plan.html`, `index.html`, US-171 |

**Lo que NO se tocó, y se dice antes de que lo encuentre**: los tres defectos de
`ml/eval/paper_micai_coverage.py` siguen en el código y ese módulo sigue sin tests; el MDE sigue
siendo aproximación con t central; la multiplicidad de toda la superficie y la selección de
predictores siguen sin implementar; y la tabla de pérdidas, el estimando y el margen práctico —las
tres que bloquean la firma— siguen siendo historias, no comportamiento.
