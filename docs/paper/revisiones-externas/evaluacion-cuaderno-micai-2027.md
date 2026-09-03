# Evaluación crítica del cuaderno público "AgroSatCopilot · Cuaderno MICAI 2027"

**Sitio evaluado**: <https://agrosat2027.netlify.app/> (pestañas `#p-que`, `#estado`, `#resultados`, `#hallazgos`, `#pendientes`) y `/plan` (EPIC 13 a 17, US-084 a US-116). Revisión del sitio: 2-sep-2026.
**Base de comparación**: repo `main` @ `471d64a` + working tree (3-sep-2026), engram del proyecto (observaciones #156, #158, #373, #822, #823) y los CSV/JSON reales de `reports/`.
**Fecha de esta evaluación**: 3-sep-2026.
**Criterio**: cada afirmación del sitio se marca como APLICA (coincide con repo/engram), NO APLICA (contradice repo/engram o es un error), NO VERIFICABLE (el artefacto vive fuera de este repo) u OBSOLETA (era cierta el 2-sep y dejó de serlo).

---

## 0. Veredicto en una página

1. **El sitio NO es el estado del proyecto integrador**: es el cuaderno de un artículo nuevo (MICAI 2027) con una contribución mucho más estrecha que el sistema. Eso es legítimo, pero ninguno de sus artefactos (`reports/paper_micai/`, `paper/ARTIFACTS.md`, `paper/micai/`, `docs/paper/novedad.md`, `docs/paper/auditoria-2026-09-02.md`, `scripts/paper_artifacts_check.py`, `tests/ml/eval/test_paper_micai_fold5_seal.py`, `docs/plan-micai-2027.md`, `CITATION.cff`) existe en `main`, en las 40 ramas locales ni en `origin` (única rama remota de paper: `feature/E11-paper-track-docs`). El plan v8 termina en EPIC 12 / US-082 y no menciona MICAI en ningún documento; la única aparición de "MICAI" en el repo es una cita dentro de `notebooks/segmentation/Avance4.Equipo17.ipynb`. **El "sello MD5" que el sitio invoca como garantía no es verificable desde este repo.**
2. **La tesis central del sitio es correcta y el propio código la confirma**: el 0.7486 / 0.7470 del Stacking es in-sample para el meta-modelo. `ml/ensemble/stacking.py::fit` hace CV espacial interna, guarda `oof_cv_metrics_` y **reajusta sobre todas las filas**; `scripts/run_us043_farslip_ensembles.py::_stacking_metrics` puntúa ese reajuste sobre las mismas parcelas y su docstring dice literalmente que devolver `oof_cv_metrics_` "would report the pessimistic spatial sub-fold CV ... so it is NOT used here". La columna libre de fuga que ya existía es `f1_macro_spatialcv` en `reports/ensemble/metrics/weighted_voting_pastis.csv` (Stacking **0.536**, Voting **0.5581**), no el 0.6794 que imprime el sitio (ese es una re-derivación nueva, agrupando bloques, y su artefacto no está aquí).
3. **Las dos "urgencias" de la pestaña Pendientes ya están resueltas y el sitio quedó obsoleto en 24 h**: (01) los folds de `tsvit-pheno-v1` son train (1,2,3) / val 4, fold 5 nunca entró, sin fuga (engram #822, 2-sep); (02) el 0.2552 de `tsvit-pheno-fullm` era un bug del harness (`dump_oof.py` no pasaba `n_timesteps`, el modelo T=37 recibía T=10) y corregido da **F1-macro 0.7883 / acc 0.8811** (engram #823, 3-sep). Consecuencia dura: **cambia el mejor individual** (fullm 0.7883 > tsvit-pheno 0.7367), la tabla de diez modelos, la frase "ninguna regla de combinación lo mejora" y todas las filas de Stacking/Blending/E-a que usaron a fullm como miembro deben recalcularse. El fix vive **sin commitear** en el working tree de `main` (`M ml/eval/oof/dump_oof.py`).
4. **La premisa del encuadre está mal citada**: el equipo no retiró seis clases "porque tenían muy poca muestra". Retiró en orden de peor F1 por clase hasta que el macro-F1 cruzó 0.90 (`reports/voting_new/cardinalidad.json`, `us043_winner_cardinality_curve.csv`, que además trae `cumulative_support_share`, o sea que la cobertura SÍ se midió: 90.5 % de las parcelas con 12 clases). De las seis retiradas (Potatoes, Winter triticale, Fruits/veg/flowers, Sorghum, Leguminous fodder, Mixed cereal) solo tres están entre las seis de menor soporte; Beet (167 parcelas), Spring barley (198) y Soybeans (209) se conservaron con F1 0.94 / 0.80 / 0.93. La US-098 del plan ("retirada por soporte, que es el criterio declarado por el equipo") parte de una premisa falsa.
5. **Hay un error factual visible en Resultados**: "las cinco últimas [IoU] (Mixed cereal, Orchard, Beet, Potatoes, Winter durum wheat) son casi exactamente las que el equipo retiró". Solo Mixed cereal y Potatoes se retiraron; Orchard, Beet y Winter durum wheat están dentro de `france-12`.
6. **Autoría**: el sitio firma "A. J. Zizumbo Velasco · J. A. Rebull-Saucedo". El segundo nombre no aparece en el repo ni en engram. El equipo registrado es Arthur Zizumbo, Carlos Aaron Bocanegra y Carlos Isaac Ávila, con el Dr. Camacho como sponsor (H100). Varios checkpoints que alimentan la tabla de diez modelos llevan su firma en el nombre (`utae-isaac`, `segformer-isaac`, `unet-aaron`, `anysat-aaron`). Esto no es un detalle de "versión final": es una decisión de consentimiento y crédito que debe cerrarse antes de escribir.

---

## 1. Pestaña "Qué es" (`#p-que`)

| Afirmación del sitio | Estado | Evidencia |
|---|---|---|
| Universo PASTIS-R, fold 5 held-out, 16 640 parcelas, 18 clases | APLICA | `reports/ensemble/us043_farslip_summary.json` (`n_universe: 16640`), `ml/eval/oof/manifest.json` (`fold: 5`, 496 patches, 18 clases) |
| Conteos por clase (6 128 ... 103), razón 60:1 | APLICA (suma 16 640) | Coherente con el universo; el mapeo `cl N` es el `semantic18` del repo (0 = Meadow, 12 = Potatoes) |
| "Cuatro clases se comen dos tercios" (72 %) y "seis raras < 7 %" (6.5 %) | APLICA | Aritmética sobre los conteos del propio sitio |
| "Pasó de 18 a 12 porque las seis retiradas «tenían muy poca muestra» y «bajaban mucho el F1 macro»" | NO APLICA | Las comillas no aparecen en ningún documento del repo (`git grep` sin resultados). Criterio real: curva de cardinalidad ordenada por F1 por clase con umbral 0.90 (`cardinalidad.json`: 12 clases = 0.9001; 13 = 0.8858). La presentación lo formula como "los doce cultivos que el sistema separa con claridad ... los seis restantes son cultivos raros y parecidos entre sí" (`docs/presentation/content/es.json:1033`) |
| "Tomó la primera vía sin medirla" | PARCIAL | La cobertura sí se midió (`cumulative_support_share`, 0.9054 con 12 clases). Lo no medido es la comparación contra la abstención por confianza. Reformular: "sin compararla con la alternativa" |
| "Revisión de 43 trabajos ... nunca cuantificada frente a su alternativa" | NO VERIFICABLE + RIESGO | No hay matriz en el repo. La literatura de *selective classification / reject option* (Chow 1970; Geifman y El-Yaniv 2017), *open-set* en mapeo de cultivos y el propio *coarsening* de taxonomías (HCAT, EuroCrops) debe estar citada explícitamente; si no, un revisor tumba la novedad en la primera ronda |
| "Cifras reales o nada; sellado MD5" | NO VERIFICABLE | No existe `paper/ARTIFACTS.md` ni `make paper-artifacts-check` en este repo. La regla coincide con la del proyecto (`docs/blockers/PENDIENTES.md`: "cifras REALES o nada"), pero el gate vive en otro sitio |
| "Las auditorías se hacen a ciegas ... revisores" | AMBIGUO | En `/plan` la US-093 se llama "Auditoría ciega multiagente": son agentes de IA, no personas. El sitio y el paper deben decirlo así (política de Springer sobre asistencia de IA) |

## 2. Pestaña "Estado" (`#estado`)

### 2.1 Registro de custodia (51 sellados, 8 sin artefacto, 43 refs, 61 consultas, 420 kB vs 68 GB)

NO VERIFICABLE desde este repo. El orden de magnitud "68 GB" corresponde a PASTIS-R completo (el archivo de la VM registra 37 GB solo S2). Los 43/61 corresponden a artefactos de US-084 que no están aquí.

### 2.2 Tabla "Los diez modelos, bajo un solo protocolo"

Las cifras coinciden con `reports/ensemble/metrics/headline_voting4.csv` (tsvit-pheno 0.7367/0.8579; utae 0.188/0.5407; xgb-alphaearth 0.5913/0.7811; farslip-zeroshot 0.0306/0.0948). Los demás miembros (deeplabv3plus, segformer, unet, anysat, farslip-ft18, fullm) salen de los parquets `oof_parcel_*_fold5.parquet` y son plausibles frente a `model_comparison_fold5.csv`.

Problemas:

- **`tsvit-pheno-fullm 0,2552 / 0,5334` es un bug, no un resultado.** OBSOLETO desde el 3-sep. Con el fix de `n_timesteps` en `dump_oof.py` la misma GT sobre las mismas 16 640 parcelas da **0.7883 / 0.8811** (engram #823; el parquet corregido está en scratchpad, aún no promovido a `ml/eval/oof/` ni añadido al `manifest.json`, que hoy solo lista 6 modelos).
- Por lo anterior, **"El mejor miembro individual es tsvit-pheno"** deja de ser cierto (fullm +0.0516) y **"ninguna de las cuatro reglas de combinación lo mejora"** debe recalcularse con el miembro corregido antes de imprimirse.
- **"Es la primera vez que el proyecto los compara así"**: NO APLICA. La US-030 fue exactamente "Harness único de métrica de segmentación (re-score apples-to-apples)" (`reports/segmentation/metrics/model_comparison_fold5.csv`, 6 arquitecturas, mismo fold, misma GT) y `headline_voting4.csv` ya tenía miembros a nivel parcela bajo el mismo arnés. Lo nuevo es la tabla de diez filas a nivel parcela, no el protocolo.
- **"U-TAE rinde 0,19 donde la literatura le da alrededor de 0,63"**: compara F1-macro por parcela (0.188) con mIoU por píxel de Garnot et al. 2021 (63.1). Métricas y niveles distintos. En el repo el mIoU denso de `utae` en fold 5 es 0.1605 (harness US-030) y 0.4742 en el fold 4 legado (plan v8 §US-026): el problema real es que el checkpoint `utae-isaac` está subentrenado o mal configurado, y así hay que decirlo, en mIoU contra mIoU.

### 2.3 "Los dos regímenes de la cifra campeona"

- **0.7486 in-sample: APLICA y está confirmado por código** (ver Veredicto, punto 2). El repo además lo etiqueta mal en dos sitios que hay que corregir: `reports/ensemble/us043_farslip_summary.json` llama `stacking_5_oof_cv` a un número que sale de la misma función in-sample (`_stacking_metrics`), y el comentario del abstract (`paper/sections/00_abstract.tex`) dice que "0.6477 is the out-of-fold CV comparison". Ambos son el mismo régimen in-sample con distinto miembro base (fullm roto).
- **0.6794 "libre de fuga, ya existía en un CSV del repositorio, guardada en la columna de al lado"**: PARCIAL. La columna existe (`f1_macro_spatialcv` en `weighted_voting_pastis.csv`) pero su valor es **0.536** para Stacking y **0.5581** para Voting (promedio de macros por sub-fold). El 0.6794 no aparece en ningún CSV, JSON, MD, TEX ni PY del repo: es una re-derivación agrupando predicciones de bloques (lo que US-107 llama "agrupar predicciones en lugar de promediar macros"). Es un número defendible, pero no "ya existía": hay que presentarlo como cálculo propio con su artefacto.
- **0.7367 "mejor individual libre de fuga"**: OBSOLETO (ver 2.2): con el fix es 0.7883 (fullm).

## 3. Pestaña "Resultados" (`#resultados`)

| Afirmación | Estado | Evidencia |
|---|---|---|
| Figuras densas de 19 filas (con Background) sobre 496 parches, de `tsvit` base | APLICA | `reports/lote_us030_040/figures/` y `manifest.json` (496 patches). Correcto aclarar que son de la variante base |
| Matriz: mayoritarias sólidas, Mixed cereal se confunde con Background/Meadow/Soft wheat | APLICA (cualitativo) | Coherente con `model_comparison_fold5.csv` y per-class CSVs |
| "Las cinco últimas (Mixed cereal, Orchard, Beet, Potatoes, Winter durum wheat) son casi exactamente las que el equipo retiró" | **NO APLICA** | Retiradas en `france-12`: Winter triticale, Fruits/veg/flowers, Potatoes, Leguminous fodder, Mixed cereal, Sorghum (`docs/serving/copiloto-v2-12clases.md`). Orchard (F1 0.786, clase 12 de la curva), Beet (0.940) y Winter durum wheat (0.820) se conservaron (engram #373) |
| Residuos espaciales: 2 326 errores sobre 14 314 aciertos | APLICA (suma 16 640; acc 0.860 = régimen Blending-3) | `comparison_us040.csv`, figura `spatial_residuals_blending` |
| El ensamble desplegado | **FALTA** | El sitio nunca cuantifica el producto que motiva el artículo: Voting-3 v2 (`tsvit-pheno-fullm-v2` + `utae` + `xgb-alphaearth`, pesos 0.902/0/0.098), `france-12` F1-macro 0.9001 en curva y **0.8992 / acc 0.9375 sobre 14 688 parcelas** en la evaluación del perceiver (`reports/agent_bench/perceiver_champion_eval_v2.json`). Si el artículo habla de "la decisión que el sistema tomó para desplegarse", el sistema desplegado tiene que estar en la tabla o justificarse su ausencia |

## 4. Pestaña "Hallazgos" (`#hallazgos`)

### Lo que cayó (los tres bloqueantes)

- **Delta = aritmética del denominador**: razonamiento correcto y consistente con la semántica de la curva de cardinalidad del repo (macro sobre las K mejores). El "0,8956 sobre 8 clases" no es verificable aquí; el repo tiene Stacking-5 top-8 = 0.9029 (`us043_winner_cardinality_curve.csv`) y Voting-3 v2 top-8 = 0.9494 (`cardinalidad.json`), mismo orden de magnitud.
- **Un mecanismo miraba la etiqueta**: válido. Nota útil: la variante desplegable ya existe en código. `ClassifyParcelInput` renormaliza el posterior al label-space activo y expone `out_of_vocabulary_classes` y `unresolved_candidate` (`ml/agent/tools/classify.py`, `ml/eval/class_remap.py`, `docs/serving/copiloto-v2-12clases.md`). US-095 debe reutilizar exactamente esa lógica, no reinventarla.
- **Bootstrap no pareado**: válido; el control "a cobertura completa el intervalo debe ser cero" es el correcto.

### Lo que aguanta

- **In-sample**: confirmado por código (Veredicto, punto 2).
- **Tres nulos**: consistentes con el repo. Vecindad: `ec_neighborhood_result.json` da deltas de +0.0027 a +0.0032 en 18 clases (k=5, alfa 0.1-0.2), dentro del ruido, y ADR-010 ya dejó E-c como FUTURE. Árbitro vs promedio: Stacking 0.7470 vs Voting ponderado 0.7444 (`weighted_voting_pastis.csv`). Rama contrastiva: +0.0016 in-sample (`us043_farslip_grid.csv`, fila 2), compatible con el +0.0006 agrupado. **Salvedad**: cualquier combinación que incluyó a `tsvit-pheno-fullm` (filas 5-8 del grid, `us043_farslip_summary.json`, y E-a `dual_head_fusion.py` con `DEFAULT_TSVIT_MEMBER = "tsvit-pheno-fullm"`, resultado 0.2694) consumió el OOF roto y debe rehacerse antes de citar los nulos como "firmes".
- **Partición espacial (5 bloques, 22 951 m)**: NO VERIFICABLE aquí. Precisar en el texto que el fold 5 es el split externo oficial de PASTIS y que los "cinco bloques" son sub-folds internos (`build_spatial_kfold`) usados solo por el meta-modelo.

### Redacción

"Cuatro revisores independientes auditaron el experimento" se lee como personas. Son agentes (US-093). Decirlo.

## 5. Pestaña "Pendientes" (`#pendientes`)

| Ítem del sitio | Estado | Qué hay |
|---|---|---|
| 01 Registro de folds de tsvit-pheno (Urgente) | **RESUELTO 2-sep** (engram #822) | Defaults del CLI `ml/train/train_segmentation.py:291-292` (train "1,2,3", val "4"); el notebook 5b lanzó sin `--train-folds`; `best_metrics` del ckpt (0.6253/0.7500/0.8759) se reproducen exacto sobre el fold 4 (482 patches); fold 5 sale peor que fold 4 (0.7401 vs 0.7500): con fuga saldría mejor. El run MLflow `alt-tsvit-pheno-v1` no loguea folds (eso sí es una deuda de provenance) |
| 02 Inconsistencia fullm 0.7918 píxel vs 0.2552 parcela (Urgente) | **RESUELTO 3-sep** (engram #823) | `dump_oof.py` no pasaba `n_timesteps` (modelo T=37, datos T=10; el guard existía en `dense_metrics.py:482`). Fix aplicado localmente, `pytest tests/ml/eval/oof/` 47 verdes, re-dump en scratchpad. Pendiente: commit, promover parquet, `dvc add/push`, añadir fullm al `manifest.json`, recalcular downstream |
| 03 Firmar o rebatir el encuadre | DECISIÓN REAL | Ver §7: el encuadre debe corregir la premisa (criterio F1, no soporte) antes de firmarse |
| 04 Autoría y autor de correspondencia | DECISIÓN REAL + RIESGO | Ver Veredicto, punto 6 |
| 05 "Cuatro archivos sin versionar" | **PARCIAL, MAL DESCRITO** (corregido 3-sep tras `hallazgos-harness-oof-fold5-2026-09-03.md` y engram #827) | Los tres checkpoints (`checkpoints/farslip/parcel/04cls`, `parcel/18cls`, `incremental/08cls`) existieron **solo en la VM H100** (`F:\projects\agrosat-copilot`, ver `_vm_scratch/gen_farslip_oof.py` y `run_ea_eb_only.bat`), nunca en DVC, y la VM ya no está disponible: se perdió el generador, no el artefacto (los OOF `farslip-ft18`/`farslip-zeroshot` sobreviven en local y DVC). Siguen referenciados como defaults en `ml/ensemble/dual_head_fusion.py:75`, `ml/ensemble/farslip_ft18.py:88`, `scripts/run_us043_farslip_ensembles.py:623` y `scripts/farslip_eval_phenology.py:56`. Lo que sí hay: `checkpoints/farslip/{4band-pheno, baseline-nir, baseline-rgb, faithful_v2, incremental/04cls}`. El parquet de Italia **sí está versionado** (`data/features/alphaearth_italia_2018.parquet.dvc`, commit `f131165`). Pendiente correcto: (a) declarar la pérdida en la sección de reproducibilidad, (b) decidir si se reentrena `parcel/04cls` (criterio: reproducir ~0.6452 macro-F1) o se descarta E-a, (c) arreglar los defaults |
| "Ventana H100 fuera: el modelo tardó 32 minutos en una tarjeta de consumo; cinco reentrenamientos en < 3 h" | **RIESGO** | MLflow `alt-tsvit-pheno-v1`: 31 min, pero en la **L4 de GCP** (VM dada de baja el 25-ago), con la config chica (T=10, dim 128). El miembro campeón es `tsvit-pheno-fullm-v2` (T=32-37, dim 192, 40 épocas, `best_epoch=36`, horas por corrida en H100). US-104 con Full-M son ~20-30 GPU-h y no cabe en la RTX local con batch 20 a 128 px. **Actualización 3-sep**: la H100 del sponsor ya no está disponible (engram #827), así que "ya no hace falta pedirla" es cierto por la razón equivocada. Opciones reales: reprovisionar la L4 (`farslip_vm_enabled=true` en Terraform, restaurar desde `gs://agrosat-artifacts-dev/vm-archive/`) o declarar que el artículo reentrena la config chica. Ver `rutas-micai-2027-post-hallazgos-2026-09-03.md` §4 |
| Multirregión y evaluación conversacional fuera | COHERENTE con el encuadre | Entonces el título y la introducción no pueden apoyarse en "AgroSatCopilot" como sistema; el sitio ya lo dice ("el artículo no trata del sistema") |

## 6. `/plan` por épicas

### Consistencia interna

- 33 US y 118 SP: suman (10+7+5+7+4; 44+20+21+24+9).
- **EPIC 16 y EPIC 17 "Bloqueada" contradice la convención del propio plan**: US-106, US-109, US-110 (EPIC 16) y US-113, US-114, US-116 (EPIC 17) no dependen de nadie o están "En curso". Deberían ser "Lista"/"En curso".
- EPIC 13 "Hecha 9/10" con US-090 retirada: correcto y honesto.

### Por US

- **US-084/085** (búsqueda y referencias ancla; "bib con datos inventados"): el bib heredado pasó `make paper-cite-check` 20/20 (`PENDIENTES.md`). Si hubo afirmaciones mal atribuidas, nombrarlas; la frase genérica desacredita el trabajo previo sin evidencia visible.
- **US-088** ("el resultado contradice la etiqueta heredada de mejor individual"): OBSOLETA. Tras el fix, fullm vuelve a ser el mejor individual (0.7883). Rehacer.
- **US-089**: los "dos regímenes separados" están bien; añadir que el repo ya trae `oof_cv_metrics_` y `f1_macro_spatialcv`, y explicar por qué el agrupado (0.6794) difiere del promedio de macros (0.536).
- **US-094 a US-100** (EPIC 14): diseño correcto. Añadir explícitamente: (i) reutilizar la restricción por predicción del producto (`class_remap.py`); (ii) incluir Voting-3 v2 como predictor bajo estudio, porque la "decisión de despliegue" se tomó sobre él; (iii) US-097 "segundo mejor miembro": tras el fix es `tsvit-pheno` (0.7367), tercero `xgb-alphaearth` (0.5913).
- **US-098** ("retirada por soporte bajo, que es el criterio declarado por el equipo"): premisa falsa. El criterio declarado es F1 por clase con umbral 0.90 (`cardinalidad.json`, `f1_ge_09_classes: 12`). Mantener el mecanismo como tercer brazo, pero etiquetarlo como "criterio alternativo", no como "el del equipo".
- **US-101** (BreizhCrops "Hecha"): APLICA. `data/breizhcrops.dvc`, `ml/ingest/breizhcrops_loader.py`, `ml/features/breizhcrops_features.py`, `notebooks/eda/02d_eda_breizhcrops.ipynb`, POC de transferencia (F1 tabular 0.21 sobre 2 145 parcelas, 7 cultivos compartidos).
- **US-102** ("entrenamiento en CPU"): viable solo para un clasificador tabular/1D; declararlo.
- **US-104** (OOF 5 folds, 2 433 parches, 36 GB): ver riesgo de cómputo en §5. 2 433 parches es el total oficial de PASTIS; 83 000 parcelas de US-105 cuadra con 16 640 x 5.
- **US-106 a US-112** (LNCS, 12 páginas): coherente con MICAI (Springer LNAI). "El manuscrito heredado no se repara": aun así, `paper/sections/03_method.tex`, licencias de datos y agradecimientos son reutilizables y ya están en dos idiomas.
- **US-113** (Netlify, noindex): APLICA. Cabecera `robots=noindex,nofollow` presente. Detalle: `favicon.ico` da 404 en consola.
- **US-115** (`CITATION.cff`, DOI Zenodo): APLICA como pendiente; el repo no tiene `CITATION.cff` y el DOI de Zenodo ya estaba pendiente en E11 (`PENDIENTES.md` §1.2).

### Lo que falta en el plan (US que no existen y deberían)

1. **US-0xx Corrección del harness OOF y recálculo** (bloqueante de todo lo demás, ~3 SP): commitear `dump_oof.py`, promover el parquet de fullm, `dvc add/push`, añadir fullm al `manifest.json`, regenerar la tabla de diez modelos, las cuatro reglas de combinación, E-a y los tres nulos; actualizar `docs/model_cards/tsvit-pheno.md`. Sin esto, la pestaña Estado imprime un bug como resultado.
2. **US-0xx Etiquetado de regímenes en el repo**: renombrar `stacking_5_oof_cv` en `us043_farslip_summary.json`, corregir el comentario del abstract, y documentar en `docs/model_cards/ensemble-final-e6.md` las tres columnas (in-sample, promedio de macros por bloque, agrupado).
3. **ADR-013 "Pivote a MICAI 2027"**: el repo registra como destino arXiv cs.CV + Remote Sensing MDPI / CVPR EarthVision / ISPRS (`PENDIENTES.md` §1.4). No hay ADR que documente el cambio de venue, el recorte de contribución ni el cambio de autores. Debe existir en `docs/decisions/`.
4. **US-0xx Consentimiento y crédito**: coautores del integrador (Isaac, Aaron), sponsor (Dr. Camacho, H100), Scuola Sant'Anna si se usa algo de Italia. "Créditos solo en la versión final" (US-114) es tarde: el consentimiento se pide antes de enviar.
5. **US-0xx Divulgación de IA**: los cuatro "revisores" son agentes; el código y el texto se produjeron con asistencia de IA. Springer exige declararlo.
6. **US-0xx Plan de cómputo de US-104**: qué config se reentrena, dónde (H100 del sponsor, sigue disponible según ADR-009; L4 ya no existe), horas y costo de GCS/DVC.
7. **US-0xx Fuente única de verdad**: o los artefactos MICAI aterrizan en este monorepo (los paths del plan ya apuntan aquí: `reports/paper_micai/`, `paper/micai/`, `docs/paper/*.md`), o el sitio deja de citar rutas de este repo. Hoy hay dos repos y el fix crítico está en el que el sitio no mira.
8. **US-0xx Trabajo relacionado de selective classification / reject option / open-set crop mapping**: sin esto la frase "nunca cuantificada" no sobrevive.
9. **US-0xx Defaults fantasma y provenance FarSLIP**: arreglar las cuatro rutas inexistentes y registrar qué checkpoint generó cada OOF de FarSLIP.
10. **Opcional**: incluir Voting-3 v2 en la tabla de diez modelos (once) o justificar su ausencia; hoy el "producto desplegado" del que habla el artículo no aparece cuantificado.

---

## 7. Lista consolidada de correcciones al sitio (ordenada por impacto)

| # | Pestaña | Dice | Realidad (repo / engram) | Acción |
|---|---|---|---|---|
| 1 | Estado, Pendientes | fullm 0.2552 / 0.5334; inconsistencia "urgente" | Bug `n_timesteps`; corregido 0.7883 / 0.8811 (engram #823) | Recalcular tabla, "mejor individual", combinaciones y nulos; cerrar pendiente 02 |
| 2 | Pendientes | Folds de tsvit-pheno "urgente" | Resuelto: train (1,2,3), val 4, sin fuga (engram #822) | Cerrar pendiente 01; anotar la deuda de provenance en MLflow |
| 3 | Qué es, /plan US-098 | Retiro por "poca muestra", criterio del equipo | Criterio real: F1 por clase, umbral 0.90; cobertura medida (90.5 %) | Corregir premisa y comillas; relabel US-098 |
| 4 | Resultados | Las 5 peores IoU "son casi exactamente las retiradas" | Solo 2 de 5 (Mixed cereal, Potatoes); Orchard, Beet, Durum se conservan | Corregir párrafo |
| 5 | Estado | 0.6794 "ya existía en un CSV, columna de al lado" | La columna vecina vale 0.536 (Stacking) / 0.5581 (Voting); 0.6794 es re-derivación propia | Presentarlo como cálculo nuevo con su artefacto |
| 6 | Estado | "Primera vez que se comparan así" | US-030 y `headline_voting4.csv` ya lo hacían | Reformular: primera tabla de diez a nivel parcela |
| 7 | Estado | U-TAE 0.19 vs 0.63 literatura | F1 parcela vs mIoU píxel; en mIoU el repo tiene 0.1605 (fold 5) / 0.4742 (fold 4) | Comparar mIoU con mIoU y hablar del checkpoint |
| 8 | Pendientes | "Cuatro archivos sin versionar" | 3 checkpoints perdidos con la VM H100 (nunca en DVC; sus OOF sobreviven) + parquet Italia sí versionado | Redefinir el ítem como pérdida de generador, no de artefacto |
| 9 | Pendientes | H100 fuera: 32 min, 5 reentrenos < 3 h | 31 min en L4 con config chica; Full-M son horas de GPU; la H100 ya no existe para el equipo | Decidir config y reprovisionar L4 (Terraform) o alquilar GPU |
| 10 | Qué es, Hallazgos | "Revisores", "auditorías a ciegas" | Agentes de IA (US-093) | Decirlo explícitamente |
| 11 | Portada | Autores | Segundo autor no consta; coautores y sponsor del integrador sin crédito | Resolver consentimiento antes de escribir |
| 12 | Todo el sitio | Sello MD5 como garantía | No verificable desde el repo principal | Aterrizar artefactos aquí o publicar el repo del sello |
| 13 | /plan | EPIC 16 y 17 "Bloqueada" | Varias US sin dependencias | Ajustar estados |
| 14 | Resultados | Producto desplegado ausente | Voting-3 v2 `france-12` 0.8992 / 0.9375 (14 688 parcelas) | Añadir o justificar |
| 15 | /plan | Sin ADR del pivote a MICAI | E11 apuntaba a arXiv / MDPI / EarthVision / ISPRS | ADR-013 |
| 16 | Todo | Cifras impresas el 2-sep | Fix del 3-sep sin commitear en `main` | Commit + PR antes de tocar el sitio |

---

## 8. Lo que el sitio hace bien y conviene conservar

- Detectar y decir en público que el 0.7486 es in-sample: el repo lo sabía (`weighted_voting_pastis.csv`, docstring de `_stacking_metrics`) y la presentación lo omitió.
- Retirar el resultado principal (US-090) tras la auditoría y dejar la retractación visible.
- Publicar los tres nulos con intervalo, en lugar de esconderlos en "trabajo futuro".
- Convertir una decisión de producto (recorte de catálogo) en una pregunta medible (frontera calidad-cobertura a igual conjunto de clases). Es una contribución modesta pero defendible, siempre que la premisa se corrija (§1) y el trabajo relacionado de *selective classification* se cite.
- `noindex,nofollow` mientras dura el doble ciego.

## 9. Fuentes consultadas

- Sitio: extracción completa de `/` y `/plan` con Playwright (3-sep-2026, 17:07 y 17:12 UTC).
- Repo: `context/RefinamientoPlaneacionAgroSatCopilot_v8.md`; `docs/blockers/PENDIENTES.md`; `docs/VALIDACION-US040-077.md`; `docs/serving/copiloto-v2-12clases.md`; `docs/model_cards/tsvit-pheno.md`; `docs/us-resolved/us-039.md`, `us-081.md`; `docs/presentation/content/es.json`; `paper/sections/00_abstract.tex`; `ml/ensemble/stacking.py`; `scripts/run_us043_farslip_ensembles.py`; `ml/eval/oof/manifest.json`; `reports/ensemble/metrics/{headline_voting4, comparison_us040, us043_farslip_grid, us043_farslip_stacking_blending, weighted_voting_pastis, us043_winner_cardinality_curve, ec_neighborhood_result}`; `reports/ensemble/us043_farslip_summary.json`; `reports/voting_new/cardinalidad.json`; `mlruns/965679031955557780/*/meta.yaml`; `git branch -a`, `git ls-remote --heads origin`.
- Engram: #156 (resultados US-040), #158 (lote US-036..040), #373 (campeón Voting-3 v2), #822 (folds tsvit-pheno-v1, 2-sep), #823 (fix `n_timesteps`, 3-sep).
