# Encargo de auditoría externa · plan de investigación MICAI 2027

**Para quién es esto**: alguien —persona o sistema— **ajeno al proyecto**, con acceso de lectura al repositorio. No hemos participado en su elección y no queremos que nos dé la razón.

**Por qué existe**: este plan ya pasó ocho auditorías internas. Todas útiles, y todas ejecutadas por el mismo equipo con las mismas herramientas y, probablemente, los mismos puntos ciegos. Lo que buscamos aquí es lo que ese conjunto no puede ver por construcción.

**Cómo usarlo**: la parte A se responde **antes** de leer la parte B. Es importante en ese orden, y más abajo se explica por qué.

---

## Contexto mínimo, sin nuestras conclusiones

Un equipo prepara un artículo para MICAI 2027 (proceedings Springer LNAI, hasta veinte páginas, revisión a doble ciego) sobre **mapeo de cultivos por satélite a nivel de parcela**.

La pregunta del artículo: cuando un mapa de cultivos no alcanza la calidad necesaria, puede **prometer menos** de varias maneras —retirar clases del catálogo, abstenerse en parcelas dudosas, devolver un conjunto de etiquetas plausibles, o retroceder a una clase más gruesa—. El artículo quiere decir cuál conviene, y a costa de quién.

Un intento anterior de este mismo artículo fue retirado.

**Dónde está todo:**

| Qué | Dónde |
|---|---|
| El plan que auditas | `agrosat-micai-site/plan.html`, variable JavaScript `EPICS`. **Épicas 18 a 26**; las 13 a 17 son historia cerrada |
| Los artefactos con sus cifras | `agrosat-copilotv2/reports/paper_micai/` |
| El registro de custodia | `agrosat-copilotv2/paper/ARTIFACTS.md` |
| El código de evaluación | `agrosat-copilotv2/ml/eval/` |
| El manuscrito retirado | `agrosat-copilotv2/paper/micai/main.pdf` |
| Documentación del proyecto | `agrosat-copilotv2/docs/paper/` |

Hay dos comprobaciones mecánicas que puedes correr: `make plan-check` valida el grafo del plan y `make paper-artifacts-check` recalcula los MD5 del registro.

---

## Parte A · Sin leer nada nuestro

Responde esto **antes** de abrir cualquier documento de `docs/paper/` que contenga la palabra *auditoría*, *reencuadre* o *revisión*. Mira el plan, el código y los artefactos, y forma tu propio juicio.

1. **¿Este plan produce un artículo que un comité de MICAI recordaría?** Sé concreto sobre qué lo impide.

2. **¿Las afirmaciones que el plan promete están sostenidas por lo que hay en `reports/`?** Toma al menos seis cifras que el plan dé por establecidas y verifícalas contra su artefacto. Reporta cada discrepancia, por pequeña que sea.

3. **¿El diseño experimental puede responder la pregunta que se hace?** Mira el tamaño de muestra, la unidad de análisis, la estructura de dependencia de los datos y la multiplicidad. Si el diseño no puede detectar el efecto que persigue, dilo con números.

4. **¿Hay algún criterio de aceptación que prejuzgue su resultado**, o que no sea verificable objetivamente? Nómbralos.

5. **¿Qué grados de libertad tiene este análisis que el plan no declara?** Es decir: qué decisiones podrían tomarse de más de una manera, donde la elección cambiaría la conclusión, y el plan no dice cómo se elige ni cuándo.

6. **Si este plan fracasa, ¿por qué será?** Una sola cosa, la más probable.

7. **¿Hay algo que el equipo está tratando como una restricción del problema y que en realidad es una elección suya?**

Escribe tus respuestas antes de continuar.

---

## Parte B · Ahora sí, contrasta

Lee `docs/paper/reencuadre-2026-09-03.md`, `docs/paper/auditoria-revisores-2026-09-03.md`, `docs/paper/campo-de-tiro.md` y `docs/paper/recomendacion-final.md`.

8. **¿Qué encontraste en la parte A que no está ahí?** Es lo más valioso de este encargo.

9. **¿En qué discrepas de lo que el equipo concluyó?** Especialmente donde tengan una cifra que los favorece.

10. **De lo que el equipo dice haber corregido, ¿qué sigue sin corregir de verdad?** Comprueba, no confíes en la declaración.

11. **El equipo declara varios errores propios.** ¿Los interpretaron bien, o hay alguno cuya causa real es distinta de la que se atribuyen?

---

## Reglas del encargo

- **Verifica, no confíes.** Si el plan dice que algo está medido, ábrelo. Varios errores de este proyecto han sido cifras que solo existían en prosa.
- **No inventes referencias.** Si citas literatura, tienes que haberla abierto.
- **Puedes correr código.** El repositorio tiene sus datos; los artefactos de `reports/paper_micai/` son pequeños y bastan para reproducir casi todo.
- **Di también lo que está bien**, en una línea cada cosa, para que el equipo no lo rompa al arreglar lo demás.
- **Ordena por severidad**, no por el orden en que lo encontraste.

## Lo que NO necesitamos

Estilo de prosa, nombres de variables, cobertura de tests, o si el inglés del manuscrito es idiomático. Eso está cubierto.

## Formato

Para cada hallazgo: (a) dónde, con ruta y línea o cita textual; (b) qué está mal; (c) qué evidencia lo demuestra; (d) el arreglo concreto; (e) severidad **BLOQUEANTE / IMPORTANTE / MENOR**.

Termina con una línea: **¿arrancamos, o hay que tocar algo antes?**
