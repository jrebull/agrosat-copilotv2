# Paper — AgroSatCopilot

Scope sub-agente `paper/`. Hereda el orquestador root ([../AGENTS.md](../AGENTS.md)) — no se repiten aqui los NON-NEGOTIABLE (idioma, secrets, DVC/MLflow, sin emojis).

## Estado

MANUSCRITO ESTRUCTURADO (US-071). Paper Track es **opcional** y arranca **post-presentacion** (semanas 10-11). No compromete ningun Avance del curso.

- Manuscrito modular: [`main.tex`](main.tex) (preambulo + `\input{}`), `sections/NN_*.tex` (9 secciones nucleares, prosa en ingles) mas las secciones profundas `sections/method_farslip.tex` (US-072), `sections/experiments_multiregion.tex` (US-073) y `sections/experiments_de4.tex` (US-082), y [`bib/refs.bib`](bib/refs.bib) (BibTeX con ids arXiv reales). Compila a 24 paginas con `make paper-pdf` (BibTeX, requiere LaTeX) o `make paper-pdf-docker` (texlive en contenedor, sin LaTeX local).
- `main.tex` es el espejo en ingles de [docs/final_doc/Avance7_equipo17.tex](../docs/final_doc/Avance7_equipo17.tex) (curso, ES): **derivado, no duplicado**. Corrige los errores factuales del Avance7 — Gemini 2.5-pro (no Flash), FarSLIP `arXiv:2511.14901` (no `2502.xxxxx`), AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 (no "v2.1"), sin Swin-UNETR entrenado (AnySat lo sustituye), sin Gemma 4 LoRA, patron Be My Eyes (reasoner frozen).
- Cada cifra del Results lleva comentario `% src: <artefacto real>`; ninguna a mano sin fuente. Validacion de citas sin LaTeX: `make paper-cite-check` (`scripts/paper_cite_check.py`).
- Preambulo: `PRIMEarxiv.sty` (copia local). El destino del fork es MICAI 2027 (Springer LNCS/LNAI, `llncs.cls`); la migracion de plantilla es una fase de la hoja de ruta del README raiz, no un blocker MDPI. Usa el paquete `import` para `\subimport` de la seccion US-072 (cuyas rutas `../tables/...` son relativas a `sections/`).
- Bloqueos (Overleaf, template MDPI, Grammarly, revision Camacho, arXiv, venue, cifras LLM US-069, toolchain LaTeX): [docs/blockers/epic11-notas.md](../docs/blockers/epic11-notas.md).
- Tabla escrita a mano canonica: [tables/us-023-preview/baseline_v2_comparison.tex](tables/us-023-preview/baseline_v2_comparison.tex). Tablas/figuras reproducibles (US-070): `tables/us-070/`, `figures/us-070/`. Todo lo demas en `figures/` es **generado**, mas `avance1_eda_report.html`.

## Comandos

```bash
poetry install --with paper            # deps opcionales del paper (grupo poetry "paper")
make paper-pdf                         # US-071: compila main.tex (pdflatex -> bibtex -> pdflatex x2). Requiere LaTeX
make paper-pdf-docker                  # US-071: compila main.tex en contenedor texlive (no requiere LaTeX local)
make paper-cite-check                  # US-071: valida que cada \cite{} tenga entrada en bib/refs.bib (sin LaTeX)
make paper-pdf-clean                   # US-071: borra auxiliares LaTeX (.aux .bbl .blg .log ...), conserva main.pdf
make avance2-figures                   # extrae figuras inline de los 3 nb FE -> figures/feature-engineering/
make eda-figures-paper-methods         # copia 5 PNG de reports/paper_methods/ -> figures/paper-methods/
make paper-methods-notebook            # regenera + ejecuta notebooks/eda/02e_eda_metodos_paper.ipynb (papermill)
make paper-tables                      # US-070: regenera las 6 tablas .tex desde reports/ (sin hardcode)
make paper-figures                     # US-070: regenera + ejecuta (papermill) los 4 nb de figuras del paper
```

## US-071 - Manuscrito modular

- `main.tex` une las secciones en orden de paper: Abstract, Introduction, Related Work, Method, Experiments, Results, (US-072 FarSLIP), (US-073 multi-region), Discussion, Conclusion, Appendix + References (`\bibliography{bib/refs}`).
- `sections/00_abstract.tex` .. `08_appendix.tex`: las 9 secciones nucleares en ingles. Related Work es **nuevo** (el Avance7 no lo tiene).
- `sections/method_farslip.tex` (US-072) y `sections/experiments_multiregion.tex` (US-073) son secciones profundas auto-contenidas, contribuidas por sus US; `main.tex` las integra (la primera via `\subimport`).
- `bib/refs.bib`: 22 entradas con atribuciones reales (AlphaEarth, FarSLIP, Be My Eyes `arXiv:2511.19417`, Harvesting AlphaEarth `arXiv:2601.00857`, Phenology Wen 2025, Sen4AgriNet CC-BY-SA-4.0, EuroCropsML CC-BY-SA-4.0, U-TAE/TSViT/SegFormer/AnySat, Gemma/Qwen/Gemini). `paper/bib/farslip_refs.bib` es el subset de US-072 (no lo usa `main.tex`; las 3 citas ya viven en `refs.bib`).
- Claims defendibles unicamente: NO "VLM supera a Gemini", NO "F1>=0.80 en Mexico", NO "zero-shot fuera de Francia", NO "TSViT 0.75+ mIoU". La tabla de benchmark LLM queda como placeholder honesto pendiente de US-069 (H100).

## US-070 - Figuras y tablas reproducibles

- `ml/report/paper_figures.py` - **plantilla unica** cientifica (CVPR/ISPRS:
  rcParams serif, 300 DPI, `PAPER_SEED=17`) + exportador `save_fig_svg_png`
  (SVG vector + PNG 300 DPI) + figuras que **recomponen** desde CSV/JSON
  (`fig_benchmark_barplot`, `fig_farslip_sweep_curve`, `fig_transfer_catalonia`,
  `fig_llm_benchmark_barplot`) o **promueven** PNG reales (`PROMOTED_FIGURES`,
  `promote_png`). Fuente ausente -> `None`, nunca se fabrica.
- `ml/report/paper_tables.py` - 6 tablas `.tex` (`booktabs`) **data-driven**:
  `build_fm_comparison_table` (T1), `build_segmentation_table` (T2),
  `build_ensemble_table` (T3), `build_llm_benchmark_table` (T4),
  `build_tool_ablation_table` (T5), `build_farslip_band_ablation_table` (Tx).
  Cero literales numericos: cada celda se lee de `reports/**`.
- `paper/notebooks/0{1..4}_figures_*.ipynb` - **solo orquestan** los modulos de
  arriba (no reimplementan plots ni hardcodean cifras). Se regeneran desde el
  builder `scripts/build_paper_us070_notebooks.py` y se ejecutan con papermill
  (outputs poblados). Editar el builder, no el `.ipynb` a mano.
- Salidas: `paper/figures/us-070/*.{svg,png}` y `paper/tables/us-070/*.tex`.
  Bloqueos (H100/GEE/revision): `docs/blockers/epic11-notas.md` (B-070-1..5).
- Atribuciones obligatorias en captions: AlphaEarth = `SATELLITE_EMBEDDING/V1/ANNUAL`
  v1.1 CC-BY-4.0 (NO "v2.1"); SegFormer = B0 RGB 3-banda; AnySat sustituye a
  Swin-UNETR (nunca entrenado); Gemini 2.5-pro = reasoner frozen (Be My Eyes);
  Qwen on-prem = Qwen3-30B-A3B (`Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`; el id "Qwen3.5-35B-A3B" del plan v8 no existe, ver CLAUDE.md raiz y docs/serving/qwen35.md). Sin sobre-claims.

## Stack local

- `ml/analysis/paper_methods.py` — 8 funciones que materializan metodos de papers REALES sobre el dato del proyecto: `boundary_pixel_mask`, `boundary_interior_stats`, `compute_boundary_ratio`, `temporal_sampling_stats`, `confusion_symmetry_analysis`, `aggregate_rare_classes`, `phenology_calendar_features`, `cloud_gap_robustness`.
- Methods cita los modelos que **de hecho** se entrenan: TSViT, U-TAE, AnySat, DeepLabv3+, SegFormer. **No** Gemma 4 LoRA aqui.
- Atribucion AlphaEarth: `Satellite Embedding V1 Annual, data version 1.1, CC-BY-4.0` (no "v2.1").

## Convenciones

- ✅ Funciones de `paper_methods.py` retornan estructuras Polars / dicts; logging via `structlog`.
- ✅ Cita por-funcion al paper origen (Russwurm & Korner, Tarasiou et al., Phenology-Aware Transformer, Qin et al. STCLN).
- ✅ Figuras reproducibles solo desde los targets `make` de arriba (incluido `make paper-figures` / `make paper-tables`); nunca editar el PNG/`.tex` a mano ni el `.ipynb` (editar su builder).
- ✅ Toda cifra del manuscrito anclada a un artefacto real via `% src:`; citas validadas con `make paper-cite-check`.
- ❌ No atribuir AlphaEarth como "v2.1"; no listar Gemma 4 LoRA en Methods; no Gemini 2.5-Flash (es 2.5-pro).
- ❌ No sacrificar entregables del curso por trabajo del paper.

## No tocar

- Figuras generadas (`figures/**`) y `avance1_eda_report.html` — son artefactos; regenerar via target, no editar.
- El notebook `notebooks/eda/02e_eda_metodos_paper.ipynb` se edita en su **builder** (`scripts/build_paper_methods_notebook.py`), no a mano.
- `ml/analysis/paper_methods.py` — al cambiar firmas, sincronizar `tests/ml/analysis/test_paper_methods.py`.
- Auxiliares LaTeX (`*.aux *.bbl *.blg *.log *.out`, `main.pdf`) — gitignored; regenerar con `make paper-pdf`, no commitear.

## Tests

```bash
poetry run pytest tests/ml/analysis/test_paper_methods.py            # unit (datos sinteticos)
poetry run pytest tests/ml/analysis/test_paper_methods.py -m empirical   # valida sobre dato real (skip si falta)
make paper-cite-check                                                # valida \cite{} <-> bib/refs.bib (US-071)
```

## Skills

- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — benchmarks y figuras interpretadas.
- [agrosat-dvc-mlflow](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — reproducibilidad de datos/runs.
