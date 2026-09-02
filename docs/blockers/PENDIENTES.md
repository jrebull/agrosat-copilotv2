# PENDIENTES — indice unico de lo que falta (cierre validacion US-040..077)

> Rama `fix/blockers-validacion-us040-077`. Este documento CONSOLIDA en un solo
> lugar lo que quedo genuinamente pendiente tras la sesion de validacion +
> correcciones. Los 8 `docs/blockers/epic*.md` quedan como historico detallado;
> ESTE archivo es el indice de lo que falta. Regla del proyecto: cifras REALES o
> nada; un dato que no existe se anota aqui, no se inventa.

---

## Resumen

**Se cerro esta sesion (la gran mayoria).** EPICs 6-12 quedaron validados con
datos reales y resultados verificados en disco. Lo cerrado incluye:

- **EPIC 6 (ensambles)**: champion Stacking-5 +FarSLIP F1-macro 0.7486 reproducible;
  E-a/E-b documentados como resultado negativo honesto (no bug).
- **EPIC 7 (agente)**: arreglado el hallazgo central — el perceiver percibia con el
  baseline `xgb-alphaearth` en vez del champion Stacking-5; re-cableo medido en H100
  (accuracy 0.831 -> 0.941, macro-F1 0.687 -> 0.901, +1.490 parcelas correctas).
- **EPIC 8 (backend)**: 146 tests verdes (29 testcontainers con Postgres real); mypy
  limpio; RLS multi-tenant aplicada y verificada.
- **EPIC 9 (frontend)**: todos los gates verdes (vitest 53/7, typecheck, i18n parity,
  eslint, build de produccion).
- **EPIC 10 (observabilidad/docs)**: tokens Qwen/vLLM en streaming arreglados en codigo
  (`include_usage`); drift Evidently corrido sobre corpus real y subido a GCS
  (`gs://agrosat-reports/drift/2026-W26/report.html`); xlsx costo-beneficio generado.
- **EPIC 11 (paper)**: manuscrito LaTeX compilado en local (15 pag, 0 refs indefinidas);
  `make paper-cite-check` verde (20/20 keys).
- **EPIC 12 (transfer)**: 4 US con notebook ejecutado; finetune FR->Catalonia real
  (few-shot mIoU 0.2468); EuroCropsML crudo versionado en DVC.
- **Transfer-mejora + modelo multi-region (esta sesion, commits c46abde y 93f8a9d)**:
  AlphaEarth+campeon cierra el domain gap en few-shot (delta max +0.111 F1 en k=1);
  WorldCereal tropical recupera con few-shot (Brasil k=20 F1=0.626); taxonomia
  ampliada 18->30 hojas HCAT reales sin mapeos falsos; macro NO degrada
  (0.658 vs 0.6535). Hallazgo clave: el cuello de botella es la separabilidad
  fenologica, no el numero de regiones.

**Queda pendiente (poco, casi todo de terceros o infra).** Ninguno bloquea la
presentacion del 27-jun. Se agrupa abajo en 4 categorias. El paper compila HOY con
placeholders honestos en las celdas que dependen de la eval LLM.

---

## 1. Requiere terceros (humano externo)

### 1.1 Revision humana nativa AgroMind-IT/ES (US-068)
- **Que es**: los 250 pares `it` necesitan un revisor italiano de Scuola Superiore
  Sant'Anna (via sponsor) y los 250 `es` un hablante de espanol del equipo. Solo los
  pares aceptados/editados por humano nativo (`source=human-edited`) entran al
  benchmark publicado.
- **Por que esta pendiente**: no hay reviewers humanos asignados; la app de revision
  esta lista pero no se ha corrido con ellos.
- **Como completar**: `streamlit run ml/eval/agromind_it_es/review_app.py`; cada
  reviewer acepta/edita/rechaza; exportar el split aceptado a
  `data/benchmark/agromind_it_es/agromind_it_es_500.jsonl`.
- Detalle: `epic11-notas.md` B-068-2.

### 1.2 Zenodo DOI del benchmark (US-068)
- **Que es**: subir `agromind_it_es_500.jsonl` + `.zenodo.json` a Zenodo y obtener el
  DOI para anclarlo en el README del dataset y en el paper.
- **Por que esta pendiente**: requiere cuenta/token Zenodo del sponsor (`ZENODO_TOKEN`,
  ya declarado en `Settings`); el builder de metadata NO incluye la llamada de upload
  a proposito.
- **Como completar**: generar `.zenodo.json` con `write_zenodo_metadata`, subir con el
  token, tomar el DOI, anclarlo en README + paper (US-070/071).
- Detalle: `epic11-notas.md` B-068-3.

### 1.3 Precio IBM Cloud H100 no publicado (US-063)
- **Que es**: la celda de precio por GPU-hora de H100 NVL en IBM Cloud (perfil `gx3`)
  en `docs/cloud/comparativa_proveedores.md`. IBM es proveedor OPCIONAL (la rubrica
  exige GCP vs Azure como minimo).
- **Por que esta pendiente**: IBM no publica un precio oficial por GPU-hora de H100
  (re-investigado 2026-06-25: ComputePrices, GPUPerHour, Spheron, IntuitionLabs y los
  docs IBM VPC confirman familia `gx3` H100 NVL pero sin tarifa). Un dato suelto de
  ~$0.99/GPU-h SIN fuente oficial NO se usa (regla de datos reales).
- **Como completar**: rellenar la celda cuando IBM publique tarifa oficial. Hoy queda
  como "Sin precio oficial publicado / No confirmado" con la nota de fuentes + fecha.
- Detalle: `epic10-notas.md` US-063 B15.

### 1.4 Revisiones de manuscrito y submission (US-071)
- **Que es**: pasada Grammarly sobre la prosa EN, revision academica del Dr. Camacho,
  submission a arXiv cs.CV (cuenta + endorsement) y a venue (Remote Sensing MDPI
  rolling / CVPR EarthVision 2026 / ISPRS).
- **Por que esta pendiente**: todos son pasos humanos post-redaccion.
- **Como completar**: ejecucion humana del equipo/sponsor. El build local +
  `make paper-pdf-docker` ya cubren la reproducibilidad.
- Detalle: `epic11-notas.md` B-GRAMMARLY / B-CAMACHO / B-ARXIV / B-VENUE.

---

## 2. Requiere infra / ejecucion (GPU, tunel, descarga)

### 2.1 Eval LLM multi-benchmark (US-069) — el grande
- **Que es**: poblar la tabla de benchmark LLM (`paper/tables/us-069/` y la
  `Table~\ref{tab:llm-bench}` del manuscrito) con accuracy / grounding / LLM-judge de
  Gemini 2.5-pro vs Qwen3.5-35B-A3B sobre 3 benchmarks x 3 seeds (con Wilcoxon).
- **Por que esta pendiente** (varias dependencias de infra simultaneas):
  - **Gemini 504 timeout con imagenes**: la eval multimodal (clasificar el tile) hace
    timeout en gateway con los items que cargan imagen.
  - **Qwen necesita tunel TCP al `:8002`** de la VM (serving vLLM GPTQ-Int4 single-GPU
    de US-048) + ventana H100.
  - **GEO-Bench-2 no descargado**: no esta en `data/`; descarga + inferencia de vision
    exige GPU/cuota (`poetry add geobench`, materializar `data/geobench2/manifest.json`,
    `dvc add`).
  - **VM F: desincronizada**: el repo VM esta atras de `origin/main`.
- **Estado del paper**: COMPILA HOY con placeholder honesto — las celdas LLM llevan
  `% src: pendiente US-069` / `\textit{pendiente}`, nunca numeros fabricados.
- **Como completar**: correr `python -m ml.eval.paper_bench --variants gemini qwen
  --seeds 0 1 2 ...` en la VM, **con retry en el 504** (o usar `gemini-2.5-flash` para
  los items con imagen); levantar vLLM + tunel TCP al `:8002`; descargar GEO-Bench-2.
  `--checkpoint`/`--resume` evitan re-pagar variantes ya hechas.
- Detalle: `epic11-notas.md` B-069-1..5, B-070-3, B-LLM-BENCH-NUMS; `epic7-notas.md`
  B-E7-3 (subset AgroMind casi todo multimodal).

### 2.2 Las 3 variantes de banda FarSLIP (US-070/072)
- **Que es**: la tabla de ablacion de bandas (rgb vs nir-rgb falso-color vs 4band-pheno)
  con F1-macro / mIoU por variante.
- **Por que esta pendiente**: los logs de destilacion solo registran perdidas, no un
  bloque de evaluacion (F1/mIoU). Falta re-extraer embeddings FarSLIP por variante —
  job de GPU sobre los checkpoints que viven en la VM (`F:\checkpoints\farslip\...`).
- **Como completar**: `make farslip-extract-embeddings` por variante en H100, volcar
  metricas a `reports/farslip/metrics/band_ablation.csv`, extender
  `build_farslip_band_ablation_table` y `fig_farslip_band_ablation`. Es re-EVALUACION,
  no re-entrenamiento.
- Detalle: `epic11-notas.md` B-072-1, B-070-5.

### 2.3 WorldCereal multiclase real (transfer)
- **Que es**: transfer multiclase real sobre WorldCereal; hoy son mascaras BINARIAS
  (cultivo vs no-cultivo), no multiclase de tipo de cultivo.
- **Por que esta pendiente**: el producto WorldCereal `ESA/WorldCereal/2021/MODELS/v100`
  entrega mascaras binarias; el transfer multiclase requiere ingesta de tipo de cultivo
  + GPU. Lo entregado (zero-shot Europa->Brasil falla, few-shot recupera) usa las
  mascaras disponibles.
- **Como completar**: ingesta WorldCereal RDM / Harmonized Global Crops con etiqueta de
  tipo de cultivo; declarado FUTURE en el AC (ver tambien seccion 4.3).
- Detalle: `epic11-notas.md` B-073-5; `epic12-notas.md` B-E12-4.

### 2.4 Verificaciones de infra que no bloquean (resumen)
- **End-to-end Qwen/vLLM vivo**: el fix de `include_usage` esta probado con dobles que
  replican el chunk de usage de OpenAI/vLLM; falta confirmar el wire real contra el
  `:8002` del H100. Codigo y tests no lo requieren (`epic10-notas.md` B6).
- **MLflow `:5010` lineage**: el server Docker estaba caido; el finetune FR->Catalonia
  y otros runs cayeron al fallback `file:./mlruns`. Re-registrar en `:5010` cuando este
  arriba, o aceptar el JSON como evidencia (`epic10-notas.md` B7; `epic12-notas.md`
  B-E12-1).
- **Paneles Grafana / alertas Cloud Monitoring / scrape Prometheus / pen-test staging /
  SMTP drift**: plantillas listas; se pueblan/wiran cuando exista el deploy con scrape
  activo. No se fabrica trafico ni salida de escaneo sintetica (`epic10-notas.md`
  B1-B4, B8, B12; US-064 B17).
- **E2E Playwright en vivo (frontend)**: requiere `backend/.env.local` (DB + creds LLM)
  y una sesion sembrada con `alphaearth_embedding`. El frontend ya esta verde por sus
  gates (`epic9-notas.md` B-E9-1).
- **DVC pull Sen4AgriNet a local**: checkpoint `best.pt` + subset `.nc` (943 MB) solo
  en la VM; 4 tests del adapter quedan skipped en local. El resultado vive en el JSON
  (`epic12-notas.md` B-E12-2).

---

## 3. Deuda tecnica menor (rapida, no bloquea)

### 3.1 `dvc add` + commit de los parquets nuevos de transfer — PARCIALMENTE RESUELTO
- Verificado 2026-09-02 en el fork: `multiregion_*.parquet` y `worldcereal_*.parquet` ya tienen
  `.dvc` y estan en el remoto; `eurocropsml_fewshot_results.parquet`, `mexico_demo_ndvi.parquet` y
  `mexico_demo_alphaearth.parquet` (4-20 KB) estan versionados directamente en git.
- Siguen sin existir en ningun sitio (ni disco, ni git, ni remoto DVC):
  `eurocropsml_alphaearth_fewshot_results.parquet` y `pastis_only_*.parquet`; solo la VM H100
  del equipo puede regenerarlos. Tambien falta en el remoto `data/features/alphaearth_italia_2018.parquet`
  (su `.dvc` existe, nunca se hizo `dvc push`).

### 3.2 `method_farslip.tex` esta en espanol — RESUELTO
- Traducido a ingles en el upstream (commit `1134c3b`, 2026-06-30). Sin accion pendiente.

### 3.3 `multiregion_fine_leaf_provenance.parquet` fuera del modulo
- **Que es**: el parquet de procedencia de hojas finas del modelo multi-region vive en
  `data/transfer/multiregion_fine_leaf_provenance.parquet`, separado del modulo que lo
  produce.
- **Por que esta pendiente**: quedo en la ruta de datos, no junto al codigo del modulo.
- **Como completar**: re-ubicar junto al modulo multi-region o documentar su procedencia
  en `docs/transfer/modelo-multiregion.md`. Cosmetico.

### 3.4 PNG `leaf_vs_macro` mal nombrado
- **Que es**: `docs/transfer/figures/multiregion_leaf_vs_macro_f1.png` (junto con
  `multiregion_leaf_f1.png`) tiene un nombre que no refleja con claridad su contenido.
- **Por que esta pendiente**: nombrado apresurado al generar la figura.
- **Como completar**: renombrar a un nombre descriptivo y actualizar la referencia en
  `docs/transfer/modelo-multiregion.md`. Cosmetico.

### 3.5 Otras inexactitudes de documentacion ya menores — RESUELTO (fork, 2026-09-02)
- Test obsoleto US-043: ya alineado con `tsvit-pheno-fullm` (suite `tests/ml/ensemble` verde).
- Tests de `tests/ml` con regex en espanol contra mensajes en ingles, fixtures y dobles
  desactualizados: alineados en el commit `45c3f9a` del fork (15 tests).
- `frontend/CLAUDE.md` ya no dice "SKELETON"; `frontend/AGENTS.md` sincronizado como espejo.
- Registro Prometheus: `backend/app/middleware/metrics.py` acepta `CollectorRegistry` dedicado;
  la suite de backend pasa completa (137 tests).
- `db/CLAUDE.md`: describe la RLS aplicada y las 6 tablas reales.
- `docs/STATUS.md`: creado con migraciones aplicadas, artefactos y gates como fuente de verdad.

---

## 4. Trabajo futuro (ADR / decision de equipo)

### 4.1 E-c geo-context completo (CRF / GNN)
- **Que es**: el ensamble geo-context E-c (ADR-010) quedo como DISENO; lo ejecutado y
  campeon es Stacking-5 +FarSLIP (US-043).
- **Por que es futuro**: alcance separado, post-presentacion.
- **Como completar**: implementar el geo-context (CRF / GNN) segun ADR-010 como trabajo
  futuro dedicado.
- Detalle: `epic6-notas.md` B-E6-3.

### 4.2 Features temporales multi-fecha (rescate de clases finas)
- **Que es**: el siguiente paso que apunta el hallazgo del modelo multi-region. El
  cuello de botella NO es el numero de regiones sino la SEPARABILIDAD FENOLOGICA:
  AlphaEarth ANUAL no distingue cultivos fenologicamente similares (0 clases finas
  rescatadas sobre 0.85; las 2 que cruzan ya estaban en PASTIS).
- **Por que es futuro**: requiere re-extraer features temporales multi-fecha (no el
  embedding anual) y re-entrenar; cambia el insumo del modelo.
- **Como completar**: incorporar features temporales multi-fecha (FFT/fenologia
  multi-temporal) para separar cultivos similares; decision de equipo + GPU.
- Detalle: `docs/transfer/modelo-multiregion.md`.

### 4.3 Transfer multiclase WorldCereal / HGC
- **Que es**: transfer multiclase real sobre WorldCereal / Harmonized Global Crops
  (las clases tropicales NO mapean a PASTIS-18; solo maiz cruza).
- **Por que es futuro**: declarado FUTURE en el AC; las mascaras WorldCereal hoy son
  binarias (ver 2.3).
- **Como completar**: ingesta WorldCereal RDM / HGC con tipo de cultivo, post-paper.
- Detalle: `epic11-notas.md` B-073-5.

### 4.4 US-077 validacion metrica F1 de la demo Mexico
- **Que es**: F1/accuracy formal de la demo Mexico (aguacate/guayaba), hoy zero-shot
  CUALITATIVO.
- **Por que es futuro**: por diseno — no hay ground truth de campo mexicano.
- **Como completar**: requiere etiquetas de campo de Mexico. Queda FUTURE.
- Detalle: `epic12-notas.md` B-E12-4.

---

## Nota final

Los 8 `docs/blockers/epic*.md` (epic6..epic12 + epic12-vm-setup) quedan como
**historico detallado** de la sesion de validacion: ahi viven los blockers ya
RESUELTOS, las causas raiz, las cifras de cada correccion y el contexto de la VM H100.
ESTE `PENDIENTES.md` es el **indice unico de lo que falta** — si solo se va a leer un
archivo para saber que sigue pendiente, es este. Nada de lo listado aqui bloquea la
presentacion del 27-jun; el paper compila hoy con placeholders honestos en las celdas
que dependen de la eval LLM (seccion 2.1).

### Atribuciones (obligatorias en paper / cards / captions)
- AlphaEarth: Khanna et al. arXiv:2310.03425, GEE `SATELLITE_EMBEDDING/V1/ANNUAL`
  data v1.1, 64-dim, CC-BY-4.0 (NUNCA "v2.1").
- Sen4AgriNet / EuroCropsML: CC-BY-SA-4.0.
- WorldCereal: `ESA/WorldCereal/2021/MODELS/v100`, CC-BY-4.0.
- PASTIS-R: Garnot et al. ICCV 2021.
- Gemini 2.5-pro: GA, 1M ctx (NO 2M, NO "3.1"). AgroMind / AgroMind-IT/ES: eval-only.
- Qwen3.5-35B-A3B: vLLM GPTQ-Int4 single-GPU (on-prem, soberania de datos).
