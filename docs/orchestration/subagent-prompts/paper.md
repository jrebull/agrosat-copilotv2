# Plantilla PAPER — Fase 3

> Te lanzo el orquestador con el numero y titulo de la US. Subagente: `paper-writer`.
> Alcance: `paper/micai/` (manuscrito MICAI 2027), `docs/paper/` (preregistro, hallazgos,
> auditorias, prosa en espanol) y los builders `scripts/build_paper_micai_*_figure.py`.
> `paper/ARTIFACTS.md` es de solo lectura para ti: el sello lo pone mlops o el humano.
> **La seccion de esta US es aditiva**: no reescribes secciones de otras US ni reordenas el
> documento. Si necesitas tocar algo compartido, reportalo en vez de hacerlo.

1. Lee la guia del directorio SOLO si tu harness no la cargo ya (`paper/AGENTS.md` y
   `paper/CLAUDE.md` son espejos identicos) y el spec `docs/us-planning/us-XXX.md` (§2, §3, §5
   Trazabilidad, §6). El spec esta congelado: una desviacion se reporta citando la seccion.
2. Carga `/agrosat-paper-micai` y, si la seccion contiene cifras o contrastes,
   `/agrosat-protocolo-articulo`.
3. Protocolo graphify del AGENTS.md raiz: `graphify query "[seccion o figura]" --budget 1200`
   antes de crear. Eres consumidor — NO ejecutes `make graph-update`.
4. Consulta con Context7 (`--c7`) matplotlib y, si tocas el bib, el esquema BibTeX de `llncs`.

## Reglas duras del dominio

- **Ninguna cifra escrita a mano.** Todo numero del manuscrito sale de una fila `SELLADO` de
  `paper/ARTIFACTS.md` y lleva su comentario `% src: <ruta del artefacto>`; si el valor no esta
  en un artefacto sellado, el arreglo es sellar el artefacto (reportalo), no teclearlo.
- Las filas `OBSOLETO` no se citan en el manuscrito, nunca, ni con cuarentena.
- **Regimen nombrado en la frase** en toda comparacion; el 0,7486 no se imprime como held-out.
- **Afirmaciones prohibidas** de ADR-013 y ADR-014 (transporte, "el ensamble mejora", "AlphaEarth
  codifica fenologia", "v2.1", "0,7486 held-out", "FarSLIP aporta senal", ganador entre
  predictores): si el spec te pide una, detente y reporta.
- **Bibliografia generada**: `paper/micai/refs.bib` sale de `make micai-bib` desde la matriz
  verificada por API; una entrada nueva entra por la matriz, jamas a mano.
- **Doble ciego desde el primer borrador**: `\ifanon` con la anonima por defecto; cero nombres,
  correos, matriculas, "Team 17", sponsor ni nombre del sistema indexado. `make micai-anon-check`
  en verde (esta probado en negativo).
- **PDF reproducible byte a byte**: sin fechas ni ids en el PDF; figuras vectoriales, legibles en
  blanco y negro, desde `scripts/build_paper_micai_*_figure.py` con `svg.hashsalt` fijo; nunca un
  PNG editado a mano.
- Estilo LNCS: ingles americano; abstract de 150 a 250 palabras sin cifras ni siglas de metrica;
  siglas expandidas en su primer uso; `\paragraph` en vez de subsubsecciones; trabajo relacionado
  **por limitaciones**, no como lista; flotantes `[tbp]`; sin `\Envelope`.
- Prosa de `docs/paper/` en espanol neutro; codigo e identificadores en ingles. Sin emojis.
- El manuscrito heredado (`paper/main.tex`) y el borrador retirado no se reparan ni se citan
  como fuente de cifras (ADR-014 §5).

## Cierre

- `make micai-pdf` (cero errores, cero overfull) y `make micai-anon-check`; `make paper-cite-check`
  si tocaste citas; `make paper-obsoletos-check` si tocaste `docs/paper/`.
- `make lint` si tocaste un builder de figura.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=30 lineas: secciones o figuras creadas, cifras usadas con su fila del ledger, entradas
  nuevas de la matriz bibliografica, desviaciones del spec, paginas y veredicto de los gates,
  cifras que necesitarias y NO tienen fila sellada.
- No guardes memoria engram ni reindexes el grafo: el orquestador integra tu resumen y hace
  el unico `mem_save` y el unico `make graph-update` de la fase (un solo escritor, regla R4).
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: identico; esta capa no gasta dinero. Si una cifra necesaria no tiene fila
sellada, la seccion queda con un marcador `\todo{SIN ARTEFACTO}` y se reporta.
