# Pendientes de Arthur para el paper de MICAI 2027

Lista única de lo que solo Arthur puede entregar o decidir. Cada punto dice para qué sirve y qué desbloquea. Actualizada el 2 de septiembre de 2026 tras la fase 0 y el sellado de la fase 1, verificando cada ruta contra el disco y contra `git ls-files` en lugar de contra el markdown heredado; se tacha o se marca cuando se cierre.

## Lo urgente, en orden (2 de septiembre de 2026, tras la fase 2)

Lo demás de este documento sigue siendo cierto, pero el reencuadre bajó su prioridad. Esto
es lo que bloquea de verdad hoy.

1. **Procedencia de `tsvit-pheno`. Es lo único que puede invalidar resultados ya escritos.**
   Sobre las mismas 16 640 parcelas del fold 5 saca F1-macro 0,7367 mientras
   `tsvit-pheno-fullm` saca 0,2552. Una diferencia de 0,48 entre dos variantes de la misma
   arquitectura no es normal. Hace falta el registro de entrenamiento del checkpoint: **con
   qué folds se entrenó**. Si vio el fold 5, el mejor individual se cae y hay que rehacer
   toda la sección de resultados. Para Arthur son minutos en la VM; para el artículo es la
   diferencia entre poder enviarlo y no.
2. **Firmar o rebatir [`ADR-013`](decisions/ADR-013-angulo-micai.md) con su enmienda.** El
   encuadre cambió: artículo nuevo desde cero sobre el punto de operación, el manuscrito
   heredado no se repara. Sin esa firma no se escribe la introducción ni el título.
3. **Autoría, confirmada por escrito.** Arthur primero, Javier segundo. Y avisar a Isaac
   Ávila y Aaron Bocanegra de que quedan acreditados como autores del código en README,
   `LICENSE` y los créditos del camera-ready, no del artículo.
4. **Autor de correspondencia.** Firma la licencia de Springer **a mano** y no se puede
   cambiar después del camera-ready. Decidirlo ahora cuesta un minuto; decidirlo tarde
   cuesta el envío.
5. ~~Ventana H100 para el reentrenamiento OOF.~~ **Retirado el 2 de septiembre de 2026: no
   hace falta.** El run de MLflow de TSViT-pheno (`0eef8a60`) tardó 1 915,4 s, unos 32
   minutos, en una RTX 4070. Cinco folds son 2,7 horas de GPU de consumo y el dataset denso
   solo necesita 36 GB de PASTIS-R, no 68. Lo hacemos nosotros en una L4 spot. Lo único que
   sí seguiría siendo suyo es el acceso a Azure, y para esto no se necesita Azure.
6. **`dvc push` de `data/features/alphaearth_italia_2018.parquet`.** Es lo único que
   `dvc status --cloud` reporta como ausente en el remoto. Barato y cierra un hueco.

Lo que **bajó de prioridad** con el reencuadre, porque su contenido sale del cuerpo del
artículo: los artefactos de DE4, la evaluación conversacional, el modelo que nombra el
reasoner y el serving de Qwen. Siguen listados abajo por si el segundo artículo los
recupera.

## Para reproducir el repositorio, al margen del artículo

Medido el 2 de septiembre de 2026 contra disco, `git ls-files` y `dvc status --cloud`, no
contra el markdown heredado. Esto no bloquea el envío del artículo; bloquea que otra
persona pueda regenerar lo que el repositorio dice tener.

1. **Tres checkpoints de FarSLIP que el código usa por defecto y no están versionados en
   ningún sitio.** Ni en git, ni en DVC, ni en disco:
   - `checkpoints/farslip/parcel/18cls/best.safetensors`, que es el que
     `scripts/run_us043_farslip_ensembles.py` usa para materializar el miembro
     `farslip-ft18`.
   - `checkpoints/farslip/parcel/04cls/best.safetensors`, el que
     `ml/ensemble/farslip_ft18.py` y `dual_head_fusion.py` traen como defecto.
   - `checkpoints/farslip/incremental/08cls/best.safetensors`, que usa
     `scripts/farslip_eval_phenology.py`.

   Ojo con la confusión de rutas: sí existe `checkpoints/farslip/incremental/04cls/best.safetensors.dvc`,
   que es **otra** ruta. Sin los tres de arriba, los dos miembros FarSLIP del ensamble no se
   pueden regenerar. Si están en la VM o en `F:\`, basta `dvc add` y `dvc push`.
2. **`data/features/alphaearth_italia_2018.parquet`.** Es lo único que `dvc status --cloud`
   reporta ausente en el remoto sobre 80 punteros. El resto del catálogo está completo.
3. **Acceso a Azure**, si alguien más va a operar la VM: `az` CLI con credenciales de la
   suscripción (grupo `agrosat-rg`, VM `agrosat-h100-prod`) y la clave SSH de `agrosat@`.
   Hoy no las tenemos, así que `make train-h100` no se puede lanzar desde fuera.
4. **Rol `roles/secretmanager.secretAccessor`** en el proyecto GCP, solo si se levanta el
   chat con Clerk o el serving de Qwen.

Lo que **sí** está bien y no hace falta pedir: `checkpoints/segmentation` está en DVC y se
recupera con `dvc pull`; los once parquets OOF coinciden con el `md5` de su propio `.dvc`;
PASTIS-R crudo se baja de Zenodo con su MD5 documentado.

## Decisiones

- [ ] **Autoría.** Confirmar que el artículo va firmado por Arthur Jafed Zizumbo Velasco (primero) y Javier A. Rebull-Saucedo (segundo), y avisar a Isaac Ávila y Aaron Bocanegra, que quedan acreditados como autores del código en el README, `LICENSE` y los créditos del camera-ready, no del artículo.
- [ ] **Ángulo.** La fase 0 ya está hecha y su veredicto es **reencuadre**, no confirmación: dos de las tres afirmaciones del ángulo A no sobreviven a la literatura verificada ni al propio artefacto. Leer [`docs/paper/novedad.md`](paper/novedad.md) y firmar o rebatir el borrador de [`ADR-013`](decisions/ADR-013-angulo-micai.md), que incluye cuatro afirmaciones prohibidas y cuatro reglas de decisión pre-registradas para la fase 2.
- [ ] **Autor de correspondencia.** Decidir quién firma a mano la licencia de Springer; no se puede cambiar después del camera-ready.
- [ ] **Reasoner que nombra el paper.** Explicar cómo se ejecutó `gemini-3.5-flash` (`reports/copilot_backends/gemini-3.5-flash.json`): en Vertex AI hoy devuelve 404 y solo responden `gemini-2.5-pro` y `gemini-2.5-flash`. Con esa respuesta se fija un único modelo en todo el texto y en la cita.
- [ ] **Serving de Qwen.** Confirmar qué produjo los artefactos del copiloto: llama.cpp con GGUF Q4_K_M (lo que documenta `docs/serving/qwen35.md`) o vLLM GPTQ-Int4 (lo que dice el manuscrito).
- [ ] **Sitio live.** Crear el sitio en Netlify (o dar acceso al equipo de Netlify de Javier) y cargar `NETLIFY_AUTH_TOKEN` y `NETLIFY_SITE_ID` como secretos del repositorio en GitHub.

## Bloqueantes nuevos que salieron de la fase 2

- [ ] **Procedencia de `tsvit-pheno`.** Sobre las mismas 16 640 parcelas del fold 5 saca F1-macro 0,7367 mientras `tsvit-pheno-fullm` saca 0,2552. Una diferencia de 0,48 entre dos variantes de la misma arquitectura obliga a comprobar con qué folds se entrenó cada checkpoint antes de publicar cualquier cifra que dependa de ese miembro. El checkpoint y su registro están en la VM.
- [ ] **Volcados OOF de los folds 1 a 4.** Hoy solo existen los del fold 5, así que el meta-modelo del stacking solo puede entrenarse con bloques del propio fold 5. Con los volcados de los otros folds el stacking podría validarse como es debido; sin ellos la sección de ensambles no se puede cerrar ni en positivo ni en negativo.

## Artefactos que solo existen en la VM H100 o en su disco

- [ ] `dvc push` de `data/features/alphaearth_italia_2018.parquet`: su `.dvc` está en git desde julio pero el archivo nunca subió al bucket. Sin él no se reproduce la transferencia a Italia.
- [ ] DE4 (Baja Sajonia): `checkpoints/transfer/voting-italia/de4_2023/report.json`, `data/pastis_de4_2023/`, `data/features/alphaearth_de4_2023_full.parquet`, `data/reference/eurocrops_v2/de4_2023.parquet`. Toda la sección DE4 del manuscrito (unas 35 cifras) solo existe hoy en un markdown.
- [ ] Checkpoint few-shot de Sen4AgriNet `tsvit-pheno-sen4agri-cat-ft-v1/best.pt` y el subset `.nc` si no está completo en DVC: el mIoU 0.2468 solo se reproduce en la VM; en local salen 0.246 y 0.280 según la corrida.
- [ ] Parquets que no aparecen ni en disco, ni en git, ni en DVC, comprobado el 2 de septiembre: `eurocropsml_alphaearth_vs_s2_delta.parquet`, `eurocropsml_per_class.parquet`, `sen4agrinet_per_class.parquet`, `pastis_to_breizhcrops.parquet`, `sen4agrinet_es_alphaearth.parquet`, `sen4agrinet_fr_alphaearth.parquet`, `worldcereal_fewshot_results.parquet`, `worldcereal_fewshot_india.parquet`, `pastis_only_*.parquet`. Si están en `F:\`, `dvc add` y `dvc push`; si no, se regeneran en CPU los que se pueda y el resto se retira del texto.

  Tres correcciones a esta lista, que estaba equivocada:

  - **`eurocropsml_fewshot_results.parquet` no está perdido.** Está en git desde el commit `bc019e5` (20 de junio de 2026), tiene 63 filas reales (LV→EE, tres semillas, k de 1 a 500) y ya está sellado en [`paper/ARTIFACTS.md`](../paper/ARTIFACTS.md). La lista lo nombraba `eurocropsml_alphaearth_fewshot_results.parquet`, que no existe con ese nombre.
  - **Los insumos crudos de WorldCereal sí están.** `worldcereal_brazil_cerrado.parquet` (212 KB) y `worldcereal_india_karnataka.parquet` (200 KB) están en disco y en DVC, y también sellados. Lo que falta son los **resultados** del barrido few-shot, que `scripts/build_worldcereal_tropical_figure.py` puede regenerar en CPU desde esos crudos: no hacen falta ni la VM ni Arthur.
  - **Los parquets multirregión sí están.** `multiregion_paired_delta.parquet`, `multiregion_per_class_macro.parquet` y los demás `multiregion_*` están en disco con su `.dvc`.

  Bajo el ángulo reencuadrado de [`ADR-013`](decisions/ADR-013-angulo-micai.md) toda esta rama multirregión sale del cuerpo del artículo, así que ninguna de estas ausencias bloquea el envío: bloquean el segundo artículo.
- [ ] `reports/agent_bench/us049_system_eval.json` (métricas de tools del copiloto) y los logs de la ablación de bandas FarSLIP (`reports/farslip/logs/*.log`): sin ellos las tablas correspondientes se eliminan.
- [ ] Si el paper conserva la evaluación conversacional (bajo el ángulo reencuadrado **no** la conserva): correr en la VM la evaluación multi-benchmark (US-069) y la re-extracción de embeddings por variante de banda (US-070). Si se sigue el ángulo A, no hacen falta.

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
