# ML — Guía de agente (AgroSatCopilot v2)

> Scope `ml/`. Sobreescribe al orquestador root solo en contexto ML. NO repite las reglas NON-NEGOTIABLE ni las científicas: ver [`../AGENTS.md`](../AGENTS.md) (Polars, MLflow `data_version` + `code_version`, split espacial, DVC, régimen nombrado, unidad parcela / clúster `patch_id`, ledger). Antes de cualquier comparación o cifra: skill `agrosat-protocolo-articulo`.

## Estado

ACTIVE y grande (~110 `.py`). Lo que existe y se usa hoy, por capa:

- **Evaluación del artículo** (`eval/`): `paper_micai_coverage.py` (`macro_over` con universo del bloque, `paired_interval` con unidad obligatoria, los tres defectos reparados y testeados), `paper_micai_arbitration.py`, `set_valued.py` (los mecanismos como predictores con valores de conjunto), `checkpoint_registry.py`, `class_remap.py` (`semantic18`, crosswalk HCAT), `per_class_analysis.py`, `dense_metrics.py`, `metrics.py`, `comparison.py`. Los runners viven en `scripts/run_paper_micai_*.py` y escriben en `reports/paper_micai/<fase>/`. `decision_cost.py` (US-172, US-175) está por crear.
- **Arnés OOF** (`eval/oof/`): `dump_oof.py` (el bug que no pasaba `n_timesteps` y daba T=10 a un modelo T=37 se corrigió el 3-sep-2026), `manifest.json` (identidad de cada miembro; `make oof-manifest-check`), `inventario.py` / `inventario.json`, `parquet_io.py`. Parquets por DVC: `oof_parcel_<miembro>_fold5.parquet` (parcela) y `oof_<miembro>_fold5.parquet` (píxel). Los sellados en `paper/ARTIFACTS.md` son de solo lectura.
- **Miembros densos** (`models/`): U-Net ResNet-50 y DeepLabv3+ vía `smp` (`segmentation.py`, `deeplabv3plus.py`), U-TAE (`utae.py`), TSViT y TSViT-pheno (`tsvit_wrapper.py`, `pheno_semantic_branch.py`), AnySat frozen + linear head (`anysat_wrapper.py`), `temporal.py`. SegFormer es B0 RGB 3-banda. Swin-UNETR nunca se entrenó: AnySat lo sustituye.
- **Predictores tabulares y entrenamiento** (`train/`): `baseline.py` / `train_baseline.py` (RF + XGBoost sobre AlphaEarth 64-dim + índices), `train_segmentation.py` (`build_and_train` recibe `train_folds` y `val_folds`; la CLI `main` solo admite `unet` y `anysat`, TSViT se entrena por `build_and_train`), `phenology_models.py` (TempCNN, InceptionTime), `finetune_sen4agrinet.py`.
- **Ensambles** (`ensemble/`): `voting.py`, `voting_weighted.py`, `bagging.py`, `blending.py` (Optuna), `stacking.py` (**`fit` reajusta el meta-modelo sobre todas las parcelas: sus métricas son in-sample**), `ec_neighborhood.py` (nulo de vecindad), `dual_head_fusion.py`, `farslip_ft18.py`, `farslip_zeroshot.py`, variantes Italia. Libre de fuga, ninguno mejora al mejor miembro individual.
- **Datos y features** (`ingest/`, `features/`, `data/`, `transfer/`): cargadores PASTIS-R y BreizhCrops, AlphaEarth vía GEE, índices espectrales y temporales, `features/spatial_split.py` (`build_spatial_kfold`), transferencia Italia (histórica).
- **FarSLIP** (`farslip/`, `extractors/farslip_extractor.py`): destilación CLIP region-aware. Los tres checkpoints `parcel/04cls`, `parcel/18cls`, `incremental/08cls` **existieron solo en la VM H100 y están perdidos**; sobreviven sus OOF.
- **Agente** (`agent/`): guía propia en [`agent/AGENTS.md`](agent/AGENTS.md). `report/`, `analysis/`, `serving/`, `workers/`, `monitoring/`, `losses/` (`dirpa.py`, probado y degradado, conservado como herramienta): sistema y curso, en mantenimiento.

NO implementado y fuera de alcance (no inventar, no referenciar como existente): Swin-UNETR, fine-tune Gemma 4 LoRA, serving vLLM en H100, extractor DINOv3 (solo existe FarSLIP), mecanismos aprendidos de renuncia.

## Comandos

```bash
make test-ml                  # pytest tests/ml (excluye slow)
make oof-manifest-check       # cada oof_parcel_* con entrada en manifest.json
make paper-artifacts-check    # el ledger
make baseline-test / ensembles-test / interpretability-test / learning-curves-test
make train-baseline           # RF + XGB con tuning, registra runs MLflow (CPU)
make train-l4 epic=Ex us=US-xxx script=train_segmentation.py   # L4 spot, con tope en el spec
make mlflow-up / mlflow-down / mlflow-ui   # server MLflow Docker (:5010)
make farslip-extract-embeddings            # embeddings FarSLIP desde el student registrado
poetry run python scripts/run_paper_micai_fase3.py --help      # runners del articulo
```

## Stack local

- **Polars 1.x** `LazyFrame` para todo pipeline tabular/feature; `parcel_id` canónico `pl.Utf8` vía `canonical_parcel_id`.
- **PyTorch + `segmentation_models_pytorch`** para U-Net / DeepLabv3+; XGBoost + scikit-learn para tabulares y contrastes.
- **MLflow** server en Docker (puerto **5010**), no `./mlruns`.
- **H3 + KMeans** para el split espacial (`features/spatial_split.py`, colchón de 1 km).
- **Hardware**: CPU para todo el protocolo (`_resolve_device` admite `cuda` o `cpu`; **MPS no sirve**); RTX 4070 o L4 spot para reentrenar un miembro (TSViT-pheno, 30 épocas, ~32 min en 4070). No existe H100.
- AlphaEarth = `SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1, 64-dim, CC-BY-4.0.

## Convenciones (✅/❌)

- ✅ `import polars as pl`; `LazyFrame`.   ❌ pandas dentro de pipelines.
- ✅ `build_spatial_kfold` para cualquier split.   ❌ random/IID split sobre parcelas.
- ✅ Toda comparación declara régimen y unidad; `paired_interval(unit=...)`; universo desde entrenamiento; punto de operación desde train/val.   ❌ Una métrica sin régimen, o un umbral elegido mirando la prueba.
- ✅ Runner versionado → artefacto con semilla, versiones, commit e intervalo → fila en el ledger.   ❌ Cifras desde notebooks o a mano; sobrescribir un archivo sellado.
- ✅ `@track_experiment` (MLflow) con `data_version` + `code_version`, y `train_folds` / `val_folds` en todo entrenamiento.   ❌ runs sin esos tags.
- ✅ Una reparación de protocolo trae un test que falla sobre la versión anterior.

## No tocar

- Parquets OOF y artefactos con fila `SELLADO`: nunca sobrescribir; archivo nuevo + sello nuevo + estado del viejo.
- `eval/oof/manifest.json` a mano: se actualiza por el runner y lo vigila `make oof-manifest-check`.
- Nombres de keys del checkpoint de `models/utae.py` (`model_state_dict`, config del checkpoint de Isaac) — cargar pesos depende de ellos.
- `farslip/cap_vocabulary.yaml`; default `weights_uri` de `farslip_extractor` (`gs://agrosat-models/farslip/farslip-clip-italy-v1/`).
- Pesos frozen: AnySat (`torch.hub gastruc/anysat`), teacher CLIP (`openai/clip-vit-base-patch16`), MiniLM (`sentence-transformers/all-MiniLM-L6-v2`), ResNet-50 ImageNet.
- `losses/dirpa.py`: se probó en US-079 (tau=1 degradó a F1 0,722 in-domain); se conserva como herramienta con tau=0, no se reintenta sin prior shift real.

## Tests

- Suite en `tests/ml/` (pytest): `eval/` (incluido `test_paper_micai_coverage.py`, que falla sobre la versión con los tres defectos, y `test_paper_micai_fold5_seal.py`), `ensemble/`, `models/`, `features/`, `ingest/`, `farslip/`, `extractors/`, `utils/`, `train/`, `tune/`, `transfer/`, `agent/` + tests sueltos.
- Fixtures con filas reales (ground truth sellado del fold 5, subconjuntos de OOF, `tests/ml/eval/fixtures/`); datos mínimos solo para mecánica pura y marcados como tales.
- Marcadores: `slow`, `integration`, `requires_gee`, `empirical`. Ningún test llama a GEE, Vertex ni al LLM local.
- Validación va en `tests/ml/` — **nunca** scripts ad-hoc `scripts/_*.py`.

## Skills

- `agrosat-protocolo-articulo` — **antes de cualquier comparación, intervalo o artefacto.**
- `agrosat-ml-evaluation` — métricas, figuras interpretadas, sanidad de miembros.
- `agrosat-ml-ensemble` — arnés OOF, voting/bagging/blending/stacking y la trampa in-sample.
- `agrosat-ml-segmentation` — miembros densos implementados y su reentrenamiento.
- `agrosat-ml-baseline` — XGBoost sobre AlphaEarth + índices.
- `agrosat-ml-features` — cargadores, índices, split espacial, `canonical_parcel_id`.
- `agrosat-gee-alphaearth` — ingesta AlphaEarth vía GEE (ADC, proyecto `agrosat-copilot`).
- `agrosat-dvc-mlflow` — `@track_experiment`, DVC, sellado.

## Gotchas

- **MLflow**: el lineage vive en el server Docker en `:5010`, no en `./mlruns`. Runs lanzados por subprocess quedan `RUNNING` si el hijo no cierra el run — verificar y cerrarlos.
- **Tres checkpoints llamados "tsvit-pheno"**: v1 (T=10, dim 128), fullm-v1 (T=37, dim 192), fullm-v2 (T=32, dim 192, el desplegado en Voting-3 v2). La identidad única por miembro es US-120; hasta entonces, comprobar `n_timesteps` y dim antes de puntuar.
- **Sanidad de miembros** (US-119): una caída > 0,15 entre `best_metrics` del checkpoint y su F1 en fold 5 (caso `utae-isaac`: val_miou 0,4826 frente a 0,1605 denso) exige explicación antes de imprimir nada.
- `poetry run` sin `poetry install` previo crea un venv vacío y falla con `ModuleNotFoundError`; en una máquina nueva, `make bootstrap` primero.
