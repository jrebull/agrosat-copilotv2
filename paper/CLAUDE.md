# Paper — AgroSatCopilot v2 (MICAI 2027)

Scope `paper/`. Hereda el orquestador root ([../AGENTS.md](../AGENTS.md)) — no se repiten aquí los NON-NEGOTIABLE ni las reglas científicas (§Reglas científicas del artículo). Las reglas de escritura viven en la skill [`agrosat-paper-micai`](../.claude/skills/agrosat-paper-micai/SKILL.md); aquí, lo operativo de la carpeta.

## Qué hay aquí

| Ruta | Qué es | Estado |
|---|---|---|
| `micai/main.tex`, `micai/main_es.tex`, `micai/sections/`, `micai/sections_es/`, `micai/refs.bib`, `micai/figures/` | Manuscrito MICAI 2027 bajo `llncs`: `\ifanon` con la anónima por defecto, PDF reproducible byte a byte, bib **generado** por `make micai-bib` | Borrador de 15 páginas **RETIRADO** (EPIC 16, [`micai/ESTADO.md`](micai/ESTADO.md)): la maquinaria es la vigente; el contenido se reescribe desde el preregistro firmado (US-144) |
| [`ARTIFACTS.md`](ARTIFACTS.md) | Ledger de custodia: elemento, ruta, MD5, bytes, commit, estado por fila | Vigente. Se edita **solo** con `make paper-artifacts-seal`; nunca un MD5 a mano |
| `main.tex`, `sections/`, `sections_es/`, `arxiv/`, `PRIMEarxiv.sty`, `bib/`, `tables/`, `figures/us-070/`, `notebooks/` | Manuscrito heredado E11 (24 páginas) | Informe técnico interno (ADR-014 §5): **no se repara, no se publica, no se cita** como fuente de cifras |
| `avance1_eda_report.html` | Artefacto del curso | Historia |

## Comandos

```bash
make micai-pdf              # anonimo: pdflatex -> bibtex -> pdflatex x2; cero errores, cero overfull; imprime paginas
make micai-pdf-es           # version en espanol
make micai-anon-check       # gate de doble ciego sobre texto y metadatos, con autoprueba en negativo
make micai-bib              # regenera micai/refs.bib desde reports/paper_micai/fase0/related_work_verified.csv
make micai-pdf-cr           # camera-ready en main_cr.pdf; comprueba que SI revela identidad
make paper-cite-check       # cada \cite{} con entrada en el bib (sin LaTeX)
make paper-artifacts-check  # recalcula el MD5 de cada fila sellada
make paper-artifacts-seal   # sella artefactos nuevos (mlops o humano)
make paper-obsoletos-check  # ningun documento activo cita OBSOLETO sin cuarentena

make paper-pdf / paper-pdf-docker / paper-tables / paper-figures   # SOLO para el informe heredado
```

## Reglas de esta carpeta

- **Ninguna cifra sin fila `SELLADO`** en `ARTIFACTS.md` y sin comentario `% src: <ruta>` en el `.tex`. Las filas `OBSOLETO` no entran ni con cuarentena; lo que no existe sellado se sella primero (mlops), no se teclea.
- **Régimen nombrado en la frase** de toda comparación. El 0,7486 es in-sample para el meta-modelo y nunca se imprime como held-out.
- **Afirmaciones prohibidas** (ADR-013 y ADR-014): transporte, "el ensamble mejora", "AlphaEarth codifica fenología", "v2.1", "FarSLIP aporta señal", ganador entre predictores, "retirada por poca muestra" como premisa.
- **Bib generado, nunca a mano**: una referencia nueva entra por la matriz verificada por API (`scripts/paper_micai_ref_verify.py`) y `make micai-bib`. Sin campos `note`; DOI de actas cuando exista.
- **Doble ciego desde el primer borrador**: cero nombres, correos, matrículas, "Team 17", sponsor ni nombre del sistema indexado en el PDF anónimo. Autor de correspondencia con `\thanks{Corresponding author.}` solo en camera-ready.
- **Figuras** desde `scripts/build_paper_micai_*_figure.py`: vectoriales, legibles en blanco y negro, inglés, `svg.hashsalt` fijo, sin fechas. Nunca un PNG editado a mano.
- **Robustez nunca se recorta**: si el tope de páginas baja, el recorte va de resultados a apéndice.
- Prosa en inglés americano en el manuscrito; español neutro en `docs/paper/`. Sin emojis.
- Atribuciones obligatorias: AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0 (no "v2.1"); PASTIS-R (Garnot et al., ICCV 2021); BreizhCrops; Sen4AgriNet y EuroCropsML CC-BY-SA-4.0; Gemini 2.5 Pro (no "3.5 Flash"); Qwen3-30B-A3B servido con llama.cpp; U-TAE cita a su propio paper, no al de L-TAE.

## No tocar

- `ARTIFACTS.md` a mano (solo `make paper-artifacts-seal`); `micai/ESTADO.md` (registro histórico de por qué el borrador no se envía).
- `micai/refs.bib` a mano; figuras generadas (`micai/figures/`, `figures/**`).
- El manuscrito heredado (`main.tex`, `sections/`, `arxiv/`): archivo, no base.
- Auxiliares LaTeX (`*.aux *.bbl *.blg *.log *.out`) y PDFs compilados — gitignored; se regeneran con los targets.
- `notebooks/0{1..4}_figures_*.ipynb` se editan en su builder `scripts/build_paper_us070_notebooks.py` (informe heredado).

## Tests y gates

```bash
make micai-anon-check                                       # incluye autoprueba en negativo
make paper-cite-check
poetry run pytest tests/ml/eval/test_paper_micai_fold5_seal.py -q   # el ground truth sellado del fold 5
poetry run pytest tests/ml/analysis/test_paper_methods.py -q         # metodos del informe heredado
```

## Skills

- [`agrosat-paper-micai`](../.claude/skills/agrosat-paper-micai/SKILL.md) — LNCS, doble ciego, bib, figuras, gates, entrega.
- [`agrosat-protocolo-articulo`](../.claude/skills/agrosat-protocolo-articulo/SKILL.md) — régimen, unidad, ledger: qué cifra puede imprimirse y cómo.
- [`agrosat-ml-evaluation`](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — métricas y figuras interpretadas.
- [`agrosat-dvc-mlflow`](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — sellado y reproducibilidad de artefactos.
