# ML — Guia de agente (AgroSatCopilot)

> Scope `ml/`. Sobreescribe al orquestador root solo en contexto ML. NO repite las reglas NON-NEGOTIABLE globales: ver [`../AGENTS.md`](../AGENTS.md) (Polars, MLflow data_version+code_version, spatial CV, DVC, sin pandas/pip/emojis, notebooks con outputs).

## Estado

ACTIVE y grande (~106 `.py`). Lo realmente implementado y entrenable hoy:

- **Baseline tabular** (`train/baseline.py`, `train/train_baseline.py`): RF + XGBoost sobre features fusionadas (AlphaEarth Satellite Embedding V1 Annual v1.1 64-dim + indices espectrales).
- **Segmentacion** (`models/segmentation.py` U-Net ResNet-50 #1 + DeepLabv3+ #2 via `smp`; `models/utae.py` U-TAE; `models/tsvit_wrapper.py` TSViT; `models/anysat_wrapper.py` AnySat frozen + linear head; `models/pheno_semantic_branch.py` rama semantica fenologica). CLI: `train/train_segmentation.py`.
- **Fenologia** (`train/phenology_models.py`): TempCNN + InceptionTime con spatial CV.
- **FarSLIP** (`farslip/{distill,train,dataset,extract_embeddings}.py`): destilacion CLIP region-aware con checkpoints reales; extractor en `extractors/farslip_extractor.py`.
- **Eval** (`eval/`): metrics, dense_metrics, interpretability, learning_curves, feature_ablation, avance4_figures, segmentation_inference.

NO implementado (no inventar, no referenciar como existente): Swin-UNETR, SegFormer-B2 entrenable, fine-tune Gemma 4 LoRA / Qwen3-VL / vLLM serving, ensambles (voting/bagging/stacking/blending — `ensemble/` es solo `__init__.py` vacio), extractor DINOv3 (solo FarSLIP existe).

## Comandos

```bash
make baseline-test            # pytest baseline.py + metrics.py + mlflow_utils.py
make interpretability-test    # pytest eval/interpretability.py
make learning-curves-test     # pytest eval/learning_curves.py
make baseline-notebook        # reconstruye + ejecuta notebooks/baseline/04_baseline.ipynb
make ml-train-image           # build local imagen ml-train (smoke, sin push)
make mlflow-up                # server MLflow Docker (UI localhost:5010)
make mlflow-down              # detiene server MLflow
make mlflow-ui                # abre UI MLflow
make train-baseline           # RF + XGB con tuning, registra runs MLflow
make farslip-extract-embeddings   # embeddings FarSLIP desde student @Production
```

## Stack local

- **Polars 1.x** `LazyFrame` para todo pipeline tabular/feature.
- **PyTorch + `segmentation_models_pytorch` (smp)** para U-Net / DeepLabv3+.
- **MLflow** server en Docker (puerto **5010**), no `./mlruns`.
- **H3 + scikit-learn KMeans** para spatial CV (`features/spatial_split.py`).
- **transformers / HF** para teachers frozen (CLIP, MiniLM).
- AlphaEarth via GEE = **Satellite Embedding V1 Annual v1.1**, 64-dim.

## Convenciones (✅/❌)

- ✅ `import polars as pl`; trabajar con `LazyFrame`.   ❌ pandas dentro de pipelines.
- ✅ Spatial CV con `build_spatial_kfold` (H3 res 5 + KMeans + buffer 1 km entre folds).   ❌ random/IID split en datos espaciales.
- ✅ `parcel_id` canonico `pl.Utf8` via `canonical_parcel_id` (cast idempotente antes de cada LEFT JOIN).   ❌ `parcel_id` entero.
- ✅ Training decorado con `@track_experiment` (MLflow) → escribe tags `data_version` (DVC) + `code_version` (git SHA).   ❌ runs sin esos tags.

## No tocar

- Nombres de keys del checkpoint de `models/utae.py` (`model_state_dict`, config del checkpoint 04j de Isaac) — cargar pesos depende de ellos.
- `farslip/cap_vocabulary.yaml`.
- Parquets materializados versionados con DVC: `data/farslip/*`, `data/features/*` (no regenerar/sobrescribir a ciegas; respetar los `.dvc`).
- Default `weights_uri` de `farslip_extractor` (`gs://agrosat-models/farslip/farslip-clip-italy-v1/`).
- Pesos frozen: AnySat (`torch.hub gastruc/anysat`), teacher CLIP (`openai/clip-vit-base-patch16`), text-encoder MiniLM (`sentence-transformers/all-MiniLM-L6-v2`), encoder ResNet-50 (ImageNet, U-Net #1).

## Tests

- Suite en `tests/ml/` (pytest): `models/`, `eval/`, `features/`, `farslip/`, `extractors/`, `utils/`, `train/`, `tune/` + tests sueltos (`test_segmentation_models.py`, `test_dense_metrics.py`, `test_pheno_semantic_branch.py`, etc.).
- Validacion va en `tests/ml/` o inline en notebook con `display()` — **nunca** scripts ad-hoc `scripts/_*.py`.
- Atajos: `make baseline-test`, `make interpretability-test`, `make learning-curves-test`.

## Skills

- `agrosat-ml-baseline` — baseline tabular (RF/XGB/LightGBM) sobre AlphaEarth + indices.
- `agrosat-ml-features` — indices espectrales, features temporales, fusion multisensor Polars.
- `agrosat-ml-segmentation` — arquitecturas densas (las implementadas: U-Net, DeepLabv3+, U-TAE, TSViT, AnySat).
- `agrosat-ml-evaluation` — metricas + plots interpretados + benchmarks.
- `agrosat-gee-alphaearth` — ingesta AlphaEarth via GEE.
- `agrosat-dvc-mlflow` — `@track_experiment`, DVC, Model Registry.

## Gotcha MLflow

El lineage vive en el server Docker en `:5010`, **no** en `./mlruns`. Runs lanzados por subprocess quedan en estado `RUNNING` si el proceso hijo no cierra el run — verificar y cerrarlos manualmente.
