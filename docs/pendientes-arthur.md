# Pendientes de Arthur para el paper de MICAI 2027

Lista única de lo que solo Arthur puede entregar o decidir. Cada punto dice para qué sirve y qué desbloquea. Actualizada el 2 de septiembre de 2026; se tacha o se marca cuando se cierre.

## Decisiones

- [ ] **Autoría.** Confirmar que el artículo va firmado por Arthur Jafed Zizumbo Velasco (primero) y Javier A. Rebull-Saucedo (segundo), y avisar a Isaac Ávila y Aaron Bocanegra, que quedan acreditados como autores del código en el README, `LICENSE` y los créditos del camera-ready, no del artículo.
- [ ] **Ángulo.** Aprobar el ángulo A (arbitraje por clase, punto de operación calidad contra cobertura y nulo de vecindad) tras leer `docs/paper/novedad.md`, o proponer otro. Queda registrado en `docs/decisions/ADR-013-angulo-micai.md`.
- [ ] **Autor de correspondencia.** Decidir quién firma a mano la licencia de Springer; no se puede cambiar después del camera-ready.
- [ ] **Reasoner que nombra el paper.** Explicar cómo se ejecutó `gemini-3.5-flash` (`reports/copilot_backends/gemini-3.5-flash.json`): en Vertex AI hoy devuelve 404 y solo responden `gemini-2.5-pro` y `gemini-2.5-flash`. Con esa respuesta se fija un único modelo en todo el texto y en la cita.
- [ ] **Serving de Qwen.** Confirmar qué produjo los artefactos del copiloto: llama.cpp con GGUF Q4_K_M (lo que documenta `docs/serving/qwen35.md`) o vLLM GPTQ-Int4 (lo que dice el manuscrito).
- [ ] **Sitio live.** Crear el sitio en Netlify (o dar acceso al equipo de Netlify de Javier) y cargar `NETLIFY_AUTH_TOKEN` y `NETLIFY_SITE_ID` como secretos del repositorio en GitHub.

## Artefactos que solo existen en la VM H100 o en su disco

- [ ] `dvc push` de `data/features/alphaearth_italia_2018.parquet`: su `.dvc` está en git desde julio pero el archivo nunca subió al bucket. Sin él no se reproduce la transferencia a Italia.
- [ ] DE4 (Baja Sajonia): `checkpoints/transfer/voting-italia/de4_2023/report.json`, `data/pastis_de4_2023/`, `data/features/alphaearth_de4_2023_full.parquet`, `data/reference/eurocrops_v2/de4_2023.parquet`. Toda la sección DE4 del manuscrito (unas 35 cifras) solo existe hoy en un markdown.
- [ ] Checkpoint few-shot de Sen4AgriNet `tsvit-pheno-sen4agri-cat-ft-v1/best.pt` y el subset `.nc` si no está completo en DVC: el mIoU 0.2468 solo se reproduce en la VM; en local salen 0.246 y 0.280 según la corrida.
- [ ] Parquets que ninguna copia conserva (disco, git ni DVC): `eurocropsml_alphaearth_fewshot_results.parquet`, `eurocropsml_alphaearth_vs_s2_delta.parquet`, `eurocropsml_per_class.parquet`, `sen4agrinet_per_class.parquet`, `pastis_to_breizhcrops.parquet`, `sen4agrinet_es_alphaearth.parquet`, `sen4agrinet_fr_alphaearth.parquet`, `worldcereal_fewshot_results.parquet`, `worldcereal_fewshot_india.parquet`, `pastis_only_*.parquet`. Si están en `F:\`, `dvc add` y `dvc push`; si no, se regeneran en CPU los que se pueda y el resto se retira del texto.
- [ ] `reports/agent_bench/us049_system_eval.json` (métricas de tools del copiloto) y los logs de la ablación de bandas FarSLIP (`reports/farslip/logs/*.log`): sin ellos las tablas correspondientes se eliminan.
- [ ] Si el paper conserva la evaluación conversacional: correr en la VM la evaluación multi-benchmark (US-069) y la re-extracción de embeddings por variante de banda (US-070). Si se sigue el ángulo A, no hacen falta.

## Accesos

- [ ] Rol `roles/secretmanager.secretAccessor` para `javirebull@gmail.com` en el proyecto `agrosat-copilot`, solo si se va a levantar el chat con Clerk o el serving de Qwen:
  ```bash
  gcloud projects add-iam-policy-binding agrosat-copilot \
    --member="user:javirebull@gmail.com" --role="roles/secretmanager.secretAccessor"
  ```
- [ ] Su `.env.local` compartido trae `GEMINI_MODEL=gemini-3.1-pro`, que no existe en Vertex, y rutas a dos JSON de service account bajo `./.env/` que no se distribuyen. Con ADC funciona todo; conviene que corrija el suyo para no arrastrar el error a otros.

## Metadatos y repositorio

- [ ] ORCID de Arthur y forma exacta de su nombre para paper, sistema del congreso y licencia.
- [ ] Valorar mover el repositorio a una organización neutra antes del camera-ready (barato ahora, caro después), o al menos: `LICENSE` en formato que GitHub reconozca como MIT, `CITATION.cff` y una release con DOI en Zenodo.
- [ ] Decidir si los 21 commits de arreglos del fork (tests, mypy, lock para Mac, Postgres arm64, CI, `/readyz`, documentación) se suben al upstream como pull request.
- [ ] `docs/licenses/DATA_LICENSE.md` sigue marcando WorldCereal como "no ingerido" aunque el paper lo usa; corregir con la atribución `ESA/WorldCereal/2021/MODELS/v100`, CC-BY-4.0.

## Terceros (heredado de `docs/blockers/PENDIENTES.md`)

- [ ] Revisión humana nativa del benchmark AgroMind-IT/ES y su DOI en Zenodo, solo si el benchmark entra en el paper.
- [ ] Revisión académica del Dr. Camacho antes del envío.
