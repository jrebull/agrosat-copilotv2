.PHONY: help bootstrap bootstrap-gpu bootstrap-gpu-linux verify-structure dev stop test lint format check secrets-scan notebooks-strip notebooks-check i18n-check db-migrate db-rollback db-new db-status db-seed db-test-us015 features-extract-demo features-persist features-fuse-demo features-fuse-italy dagster-materialize-features feature-selection-subset feature-selection-build feature-selection-notebook feature-selection-test feature-fusion-build feature-fusion-notebook avance2-figures avance2-build mlflow-up mlflow-down train-baseline baseline-test ensembles-test baseline-notebook baseline-notebook-check baseline-v2-full s2-raw-parcels interpretability-test learning-curves-test ml-train-image train-l4 train-l4-smoke train-h100 azure-h100-start azure-h100-stop azure-h100-status mlflow-ui dagster-ui dvc-push dvc-pull eda-sentinel2 eda-alphaearth eda-bivariado eda-figures-avance1 eda-figures-paper-methods us073-transfer-figures eda-pastis-subset eda-notebook-avance1 paper-methods-notebook eda-pdf eda-dashboard eda-dashboard-test eval-agromind eval-geoanalyst serve-qwen35 cost-audit deploy-staging deploy-prod tf-init tf-plan tf-apply tf-fmt tf-validate farslip-dataset-build farslip-dataset-check farslip-train farslip-eval-pastis farslip-smoke-eval farslip-extract-embeddings feature-ablation phenology-train phenology-description-test reencuadre-notebook reencuadre-notebook-check reencuadre-notebook-full docs-pdf docs-pdf-clean docs-pdf-docker paper-tables paper-figures paper-pdf paper-pdf-clean paper-pdf-docker paper-cite-check paper-artifacts-check paper-artifacts-seal paper-obsoletos-check protocolo-check us172-adjuntos micai-pdf micai-anon-check micai-bib micai-pdf-cr micai-pdf-es plan-check

help:
	@echo "AgroSatCopilot — comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# === Bootstrap (reemplaza el hook post_gen_project.py de cookiecutter) ===
bootstrap:  ## Instala deps Python + Node (poetry + pnpm) — sin GPU
	poetry install --with dev,test,ml,geo,dagster,paper
	cd frontend && pnpm install

bootstrap-gpu:  ## Como bootstrap + torch CUDA 13.0 + bitsandbytes (Win/Linux con GPU NVIDIA)
	poetry install --with dev,test,ml,ml-gpu,geo,dagster,paper
	cd frontend && pnpm install

bootstrap-gpu-linux:  ## Como bootstrap-gpu + flash-attn + vllm (solo Linux, replica cloud)
	poetry install --with dev,test,ml,ml-gpu,ml-gpu-linux,geo,dagster,paper
	cd frontend && pnpm install

fix-libomp-macos:  ## macOS: torch usa el libomp de Homebrew (evita segfault con xgboost/lightgbm)
	@TL=$$(poetry run python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib'))"); \
	BREW=/opt/homebrew/opt/libomp/lib/libomp.dylib; \
	if [ ! -f "$$BREW" ]; then echo "falta libomp: brew install libomp"; exit 1; fi; \
	if [ -L "$$TL" ]; then echo "ya enlazado: $$TL"; else mv "$$TL" "$$TL.orig" && ln -s "$$BREW" "$$TL" && echo "enlazado: $$TL -> $$BREW"; fi

verify-structure:  ## Valida estructura de directorios (AC-4 de US-001)
	@bash scripts/verify_structure.sh

# === Dev ===
dev:  ## Levanta docker-compose con 8 servicios (carga puertos desde .env.local)
	docker compose --env-file .env.local up -d
	@echo "API: http://localhost:$${API_HOST_PORT:-8010}  Frontend: http://localhost:$${FRONTEND_HOST_PORT:-3010}  Dagster: http://localhost:$${DAGSTER_HOST_PORT:-3011}  MLflow: http://localhost:$${MLFLOW_HOST_PORT:-5010}"

stop:  ## Detiene docker-compose
	docker compose --env-file .env.local down

# === Demo (presentacion) ===
demo:  ## Levanta la app completa para demo (datos Docker + back/front nativos). Probado.
	pwsh scripts/demo_up.ps1

demo-vm:  ## Igual que demo + tunel al Qwen on-prem de la VM H100 (requiere tunel cloudflared)
	pwsh scripts/demo_up.ps1 -WithVM

demo-down:  ## Baja la demo (servicios nativos + tuneles; conserva datos Docker)
	pwsh scripts/demo_down.ps1 -KeepDocker

# === Lint & format ===
lint:  ## ruff + ruff format check + mypy (backend, ml, dagster_project) + pnpm lint
	cd backend && poetry run ruff check .
	cd backend && poetry run ruff format --check .
	cd backend && poetry run mypy app/
	poetry run ruff check ml dagster_project scripts
	poetry run ruff format --check ml dagster_project scripts tests
	poetry run mypy ml dagster_project
	cd frontend && pnpm lint

format:  ## ruff format
	cd backend && poetry run ruff format .
	cd ml && poetry run ruff format .

secrets-scan:  ## gitleaks secret scanning (reemplazo del hook pre-commit)
	gitleaks detect --no-banner --redact

notebooks-strip:  ## nbstripout on-demand (NO usar en quality gates - notebooks commitean con outputs)
	poetry run nbstripout notebooks/*.ipynb notebooks/eda/*.ipynb

notebooks-check:  ## papermill end-to-end (smoke modo degradado, ~3 min)
	poetry run papermill notebooks/02b_eda_alphaearth.ipynb /tmp/02b_check.ipynb \
		-p sample_size 1000 -p n_pastis_patches 2 -p tsne_subsample 500 --no-progress-bar
	poetry run papermill notebooks/eda/02a_eda_sentinel2.ipynb /tmp/02a_check.ipynb \
		-p n_patches 3 -p sample_size 2000 -p use_gee False --no-progress-bar

i18n-check:  ## valida que las claves i18n existan en it/es/en
	cd frontend && pnpm i18n:check

check: lint secrets-scan i18n-check  ## suite local previa a PR (reemplaza pre-commit)

# === Tests ===
test:  ## pytest backend con cobertura
	cd backend && poetry run pytest --cov=app --cov-report=term-missing --cov-fail-under=70

test-unit:
	cd backend && poetry run pytest tests/unit -v

test-integration:
	cd backend && poetry run pytest tests/integration -v

test-e2e:  ## Playwright E2E
	cd frontend && pnpm test:e2e

test-frontend:  ## vitest con cobertura (umbral 50 % lineas en vitest.config.ts)
	cd frontend && pnpm test:coverage

test-ml:  ## pytest tests/ml (excluye `slow`; usa -m "" para todo)
	poetry run pytest tests/ml -q -m "not slow" -p no:cacheprovider

test-all: test test-ml test-frontend  ## backend + ml + frontend (la promesa de CLAUDE.md)

# === DB ===
db-migrate:  ## dbmate up
	dbmate up

db-rollback:
	dbmate down

db-new:  ## make db-new name=create_xxx
	dbmate new $(name)

db-status:
	dbmate status

db-seed:
	poetry run python scripts/seed.py

db-shell:
	docker compose exec postgres psql -U agrosat -d agrosat

# === Features (US-015) ===
features-extract-demo:  ## Ejecuta extract_temporal_features sobre el fixture demo
	poetry run python -c "import xarray as xr; from ml.features import extract_temporal_features; \
ds = xr.open_dataset('data/test_fixtures/parcel_demo_ts.nc'); \
da = ds['parcel_indices']; da.attrs.setdefault('parcel_id', 42); da.attrs.setdefault('year', 2024); \
da = da.assign_coords(band=[b.decode() if isinstance(b, bytes) else b for b in da.coords['band'].values]) if da.coords['band'].dtype.kind == 'S' else da; \
df = extract_temporal_features(da); \
import structlog; log = structlog.get_logger(); log.info('features extracted', shape=df.shape, sample_cols=df.columns[:8])"

features-persist:  ## TODO US-016: invoca load_features_parcels con DSN de .env.local (stub)
	@echo "US-015 stub: load_features_parcels esperando parquet operativo desde US-016"
	@echo "Uso futuro: poetry run python -m ml.features.persist_features --input <parquet> --dsn $$DATABASE_URL"

db-test-us015:  ## Ejecuta tests round-trip de migraciones US-015
	poetry run pytest tests/db/test_migrations_us015.py -q

# === Features US-016 (fusión multisensor a nivel parcela) ===
features-fuse-demo:  ## US-016 — Fusión sobre fixture demo 9 parcelas (3 regiones italianas)
	poetry run python scripts/build_parcel_features.py \
	  --year 2024 \
	  --regions pianura_padana,toscana,puglia \
	  --parcels-path data/test_fixtures/parcels_demo_3regions.parquet \
	  --out data/features/features_fused_v1_demo.parquet \
	  --scaler-out artifacts/scaler_v1_demo.pkl \
	  --splits-out data/splits/spatial_kfold_v1_demo/ \
	  --no-farslip

features-fuse-italy:  ## US-016 — Fusión completa Italia 3 regiones (requiere parcels Postgres + GEE)
	poetry run python scripts/build_parcel_features.py \
	  --year 2024 \
	  --regions pianura_padana,toscana,puglia \
	  --out data/features/features_fused_v1.parquet \
	  --scaler-out artifacts/scaler_v1.pkl \
	  --splits-out data/splits/spatial_kfold_v1/ \
	  --no-farslip

dagster-materialize-features:  ## US-016 — Materializa los 3 assets en orden (features → splits → scaler)
	poetry run dagster asset materialize -m dagster_project.definitions \
	  --select parcel_features_fused+

# === Feature Selection US-018 (Avance 2 CRISP-ML(Q) Data Preparation) ===
feature-selection-subset:  ## US-018 — Regenera subset PASTIS-R estratificado (>=500 muestras x 187 cols)
	poetry run python scripts/generate_feature_selection_subset.py \
	  --root data/PASTIS-R \
	  --out data/test_fixtures/feature_selection_subset.parquet \
	  --min-per-class 10 \
	  --max-samples 500

feature-selection-build:  ## US-018 — Reconstruye notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb desde scripts/build_us018_notebook.py
	poetry run python scripts/build_us018_notebook.py

feature-selection-notebook:  ## US-018 — Papermill end-to-end sobre 03b_fe_spectral_temporal_pastis.ipynb
	MPLBACKEND=Agg poetry run papermill notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb \
	  notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb --no-progress-bar

feature-selection-test:  ## US-018 — pytest selection.py con cobertura
	poetry run pytest tests/ml/features/test_selection.py \
	  --cov=ml.features.selection --cov-report=term-missing

feature-fusion-build:  ## US-018 ext — Reconstruye notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb desde scripts/build_fusion_notebook.py
	poetry run python scripts/build_fusion_notebook.py

feature-fusion-notebook:  ## US-018 ext — Papermill end-to-end sobre 03c_fe_alphaearth_pastis.ipynb
	MPLBACKEND=Agg poetry run papermill notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb \
	  notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb --no-progress-bar

# === Avance 2 — Feature Engineering (notebook integrador del curso) ===
avance2-figures:  ## Extrae figuras inline de los 3 notebooks FE a paper/figures/feature-engineering/
	poetry run python -m ml.report.extract_notebook_figures \
	  notebooks/feature_engineering/03a_fe_sentinel2.ipynb \
	  --output paper/figures/feature-engineering/sentinel2
	poetry run python -m ml.report.extract_notebook_figures \
	  notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb \
	  --output paper/figures/feature-engineering/spectral-temporal
	poetry run python -m ml.report.extract_notebook_figures \
	  notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb \
	  --output paper/figures/feature-engineering/alphaearth

avance2-build: avance2-figures  ## Regenera el notebook integrador Avance2.Equipo17.ipynb (figuras embebidas, sin papermill)
	poetry run python scripts/build_avance2_notebook.py \
	  --out notebooks/feature_engineering/Avance2.Equipo17.ipynb

# === US-019 — Baseline RF/XGB (EPIC 4) ===
mlflow-up:  ## US-019 — Levanta el servidor MLflow local en Docker (UI en localhost:5010)
	docker compose up -d mlflow

mlflow-down:  ## US-019 — Detiene el servidor MLflow local
	docker compose stop mlflow

train-baseline:  ## US-019 — Entrena RF + XGB con tuning y registra runs MLflow
	poetry run python ml/train/train_baseline.py --model both --tune

baseline-test:  ## US-019 — pytest baseline.py + metrics.py + mlflow_utils.py
	poetry run python -m pytest tests/ml/train tests/ml/eval tests/ml/utils -q

ensembles-test:  ## US-040 — pytest los 4 ensambles + anti-fuga + figuras
	poetry run python -m pytest tests/ml/ensemble tests/ml/eval/test_ensemble_figures.py -q

baseline-notebook:  ## US-019/020/021/022 + US-023-preview v2 — Reconstruye y ejecuta notebooks/baseline/04_baseline.ipynb
	poetry run python scripts/build_baseline_notebooks_v2.py --only 04_baseline
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04_baseline.ipynb \
	  notebooks/baseline/04_baseline.ipynb --no-progress-bar

baseline-notebooks-v2-build:  ## US-023-preview v2 — Reconstruye los 6 notebooks de baseline (sin ejecutar)
	poetry run python scripts/build_baseline_notebooks_v2.py

baseline-notebooks-v2-run:  ## US-023-preview v2 — Ejecuta los 6 notebooks de baseline con papermill (orden secuencial, requiere GEMINI_API_KEY + GEE)
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04b_baseline.ipynb notebooks/baseline/04b_baseline.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04_baseline.ipynb notebooks/baseline/04_baseline.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04c_baseline.ipynb notebooks/baseline/04c_baseline.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04_farslip_eval_pastis.ipynb notebooks/baseline/04_farslip_eval_pastis.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb notebooks/baseline/05_reencuadre_fenologico.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/Avance3.Equipo17.ipynb notebooks/baseline/Avance3.Equipo17.ipynb --no-progress-bar

baseline-notebook-check:  ## US-022 — papermill end-to-end de 04_baseline.ipynb con parametros reducidos (CI, ~5 min)
	poetry run python scripts/build_baseline_notebook.py --out notebooks/baseline/04_baseline.ipynb
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/04_baseline.ipynb /tmp/04_baseline_check.ipynb \
	  -p MAX_SAMPLES 4000 -p TUNE False -p COMPARISON_MAX_SAMPLES 4000 \
	  -p COMPARISON_K_FOLDS 3 --no-progress-bar

baseline-v2-full:  ## US-023-preview P8 — papermill notebook 04 v2 con CUDA (3 modelos sobre conjunto ganador post-ablation, ~90 min)
	poetry run python scripts/build_baseline_notebook.py --out notebooks/baseline/04_baseline.ipynb
	poetry run papermill notebooks/baseline/04_baseline.ipynb \
	  notebooks/baseline/04_baseline.ipynb \
	  -p RUN_BASELINE_V2 True -p V2_MAX_SAMPLES 0 -p V2_K_FOLDS 5 -p V2_BUFFER_KM 1.0 \
	  -p V2_TEMPORAL_EPOCHS 200 -p V2_TEMPORAL_BATCH_SIZE 128 -p V2_DEVICE auto --no-progress-bar

s2-raw-parcels:  ## US-022 — genera el escenario (b): Sentinel-2 crudo a nivel parcela
	poetry run python scripts/build_s2_raw_parcels.py \
	  --pastis-root data/PASTIS-R \
	  --parcels data/processed/pastis_parcels_full.geoparquet \
	  --out data/cache/pastis/s2_raw_parcels_2019_85951.parquet \
	  --n-jobs -1

interpretability-test:  ## US-020 — pytest del modulo ml/eval/interpretability.py
	poetry run python -m pytest tests/ml/eval/test_interpretability.py -q

learning-curves-test:  ## US-021 — pytest del modulo ml/eval/learning_curves.py
	poetry run python -m pytest tests/ml/eval/test_learning_curves.py -q

# === US-022-b — Reencuadre fenologico (C + D) ===
feature-ablation:  ## US-022b-C — ablation de features (5 sets x N modelos) sobre el subset US-018
	poetry run python -c "from pathlib import Path; from ml.eval.feature_ablation import run_feature_ablation, export_ablation_table; \
results = run_feature_ablation(features_path='data/test_fixtures/feature_selection_parcels_subset.parquet', models=('xgb',), max_samples=8000, k_folds=5, buffer_km=1.0, seed=42); \
export_ablation_table(results, Path('reports/baseline/feature_ablation'))"

phenology-train:  ## US-022b-C — entrena TempCNN + InceptionTime con spatial CV (CPU smoke; en L4 cambiar device)
	poetry run python scripts/train_phenology_models.py --device cpu --n-epochs 5 --batch-size 128 --n-parcels 4000

phenology-description-test:  ## US-022b-D — pytest del modulo phenology_description (Gemini mockeado)
	poetry run python -m pytest tests/ml/features/test_phenology_description.py -q

reencuadre-notebook:  ## US-022b-C/D + US-023-preview P1 — Reconstruye y ejecuta notebooks/baseline/05_reencuadre_fenologico.ipynb
	poetry run python scripts/build_reencuadre_notebook.py \
	  --out notebooks/baseline/05_reencuadre_fenologico.ipynb
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb \
	  notebooks/baseline/05_reencuadre_fenologico.ipynb --no-progress-bar

reencuadre-notebook-check:  ## US-022b — papermill smoke con parametros reducidos (~3 min CI)
	poetry run python scripts/build_reencuadre_notebook.py \
	  --out notebooks/baseline/05_reencuadre_fenologico.ipynb
	MPLBACKEND=Agg poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb \
	  /tmp/05_reencuadre_check.ipynb \
	  -p MAX_SAMPLES 800 -p K_FOLDS 3 -p BUFFER_KM 0.5 -p TEMPORAL_EPOCHS 2 -p TEMPORAL_BATCH_SIZE 32 \
	  -p DEVICE cpu -p RUN_SEMANTIC_BRANCH False --no-progress-bar

reencuadre-notebook-full:  ## US-022b — corrida real GPU local (full dataset, 200 ep + early stopping, CUDA)
	poetry run python scripts/build_reencuadre_notebook.py \
	  --out notebooks/baseline/05_reencuadre_fenologico.ipynb
	poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb \
	  notebooks/baseline/05_reencuadre_fenologico.ipynb \
	  -p MAX_SAMPLES 0 -p K_FOLDS 5 -p BUFFER_KM 1.0 -p TEMPORAL_EPOCHS 200 -p TEMPORAL_BATCH_SIZE 128 \
	  -p DEVICE auto -p RUN_SEMANTIC_BRANCH False --no-progress-bar

# === ML / Training ===
# US-022b-A — Imagen `ml-train` (CUDA 13.0 + grupos ml/ml-gpu/ml-gpu-linux/geo).
# Build local sin GPU (smoke A-1) — el push a Artifact Registry lo hace
# Cloud Build (infrastructure/cloudbuild.yaml step `build-ml-train`).
ml-train-image:  ## US-022b-A — build local de la imagen ml-train (AC-1 smoke sin push)
	docker build \
	  -f infrastructure/docker/ml-train.Dockerfile \
	  --target=runtime \
	  -t ml-train:dev \
	  .
	@echo "Smoke check (sin GPU): import torch + mlflow + breizhcrops"
	docker run --rm ml-train:dev \
	  python -c "import torch, mlflow, breizhcrops; print('OK torch', torch.__version__, 'mlflow', mlflow.__version__, 'breizhcrops', breizhcrops.__version__)"

train-l4:  ## US-022b-A — Spot L4 24GB (baselines, dev). Requires: epic=Ex us=US-xxx script=path/to/script.py + MLFLOW_TRACKING_URI export
	@if [ -z "$$MLFLOW_TRACKING_URI" ]; then \
	  echo "ERROR: export MLFLOW_TRACKING_URI=\$$(terraform -chdir=infrastructure/terraform/environments/dev output -raw mlflow_tracking_uri)"; exit 1; \
	fi
	@if [ -z "$(epic)" ] || [ -z "$(us)" ]; then \
	  echo "ERROR: usage 'make train-l4 epic=E5 us=US-022b script=ml/farslip/train_student.py'"; exit 1; \
	fi
	@echo "Lanzando job en GCP L4 spot para epic=$(epic) us=$(us) script=$(script)"
	@tmpfile=$$(mktemp); trap "rm -f $$tmpfile" EXIT; \
	  TRAIN_SCRIPT="$(script)" envsubst '$$MLFLOW_TRACKING_URI $$TRAIN_SCRIPT' < ml/configs/l4_spot.yaml > $$tmpfile; \
	  gcloud ai custom-jobs create \
	    --region=$${GCP_REGION:-us-central1} \
	    --display-name=train-$(epic)-$(us) \
	    --config=$$tmpfile \
	    --args="--epic=$(epic),--us=$(us)"

train-l4-smoke:  ## US-022b-A AC-5 — smoke job 1 epoca TempCNN sintetica (~10 min, <$0.20). Requires MLFLOW_TRACKING_URI
	@if [ -z "$$MLFLOW_TRACKING_URI" ]; then \
	  echo "ERROR: export MLFLOW_TRACKING_URI=\$$(terraform -chdir=infrastructure/terraform/environments/dev output -raw mlflow_tracking_uri)"; exit 1; \
	fi
	@echo "Lanzando smoke job US-022b-A en L4 spot (timeout 20 min) con MLFLOW_TRACKING_URI=$$MLFLOW_TRACKING_URI"
	@tmpfile=$$(mktemp); trap "rm -f $$tmpfile" EXIT; \
	  envsubst '$$MLFLOW_TRACKING_URI' < ml/configs/l4_smoke.yaml > $$tmpfile; \
	  gcloud ai custom-jobs create \
	    --region=$${GCP_REGION:-us-central1} \
	    --display-name=train-E5-US-022b-smoke \
	    --config=$$tmpfile
	@echo "Sigue el estado en: https://console.cloud.google.com/vertex-ai/training/custom-jobs"

train-h100:  ## Azure H100 96GB ventana=Vn script=xxx.py
	@echo "Lanzando $(script) en H100 ventana $(window)"
	ssh agrosat@$(shell az vm show -d -g agrosat-rg -n agrosat-h100-prod --query publicIps -o tsv) \
	  "cd ~/agro_sat_copilot && poetry run python ml/train/$(script)"

azure-h100-start:
	bash scripts/azure_h100_start.sh

azure-h100-stop:
	bash scripts/azure_h100_stop.sh

azure-h100-status:
	bash scripts/azure_h100_status.sh

serve-qwen35:  ## Lanza vLLM Qwen3.5-35B-A3B en H100
	bash scripts/serve_qwen35.sh

# === DVC / MLflow / Dagster ===
dvc-push:
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

dagster-ui:
	dagster dev -m dagster_project.definitions

# === EDA / Notebooks ===
eda-sentinel2:  ## Ejecuta el notebook US-010 con papermill (sample_size=100000)
	poetry run papermill notebooks/02a_eda_sentinel2.ipynb /tmp/02a_out.ipynb -p sample_size 100000

eda-alphaearth:  ## Ejecuta el notebook US-011 con papermill (sample_size=100000, year=2024)
	poetry run papermill notebooks/02b_eda_alphaearth.ipynb /tmp/02b_out.ipynb -p sample_size 100000 -p year 2024

eda-bivariado:  ## Ejecuta el notebook US-012 bivariado/multivariado/temporal (n_parcels=200)
	poetry run papermill notebooks/eda/02c_eda_bivariado_temporal.ipynb notebooks/eda/02c_eda_bivariado_temporal.ipynb -p n_parcels 200

eda-figures-avance1:  ## Extrae figuras inline del notebook Avance1.Equipo17 a paper/figures/avance1/
	poetry run python -m ml.report.extract_notebook_figures notebooks/eda/Avance1.Equipo17.ipynb

eda-figures-paper-methods:  ## Copia las figuras de 02e_eda_metodos_paper a paper/figures/paper-methods/
	mkdir -p paper/figures/paper-methods
	cp reports/paper_methods/boundary_interior_histograms.png \
	   reports/paper_methods/temporal_gap_distribution.png \
	   reports/paper_methods/confusion_symmetry_scatter.png \
	   reports/paper_methods/phenology_calendar_distribution.png \
	   reports/paper_methods/cloud_gap_drift.png \
	   paper/figures/paper-methods/

us073-transfer-figures:  ## Genera las 2 tablas .tex + 2 figuras de transferencia multi-region (US-073) desde artefactos REALES E12
	poetry run python -m scripts.build_us073_transfer_figures

eda-pastis-subset:  ## Genera subset compacto de PASTIS-R (~500KB) para el mapa folium del dashboard
	poetry run python -m ml.report.generate_pastis_subset

eda-notebook-avance1:  ## Regenera notebooks/eda/Avance1.Equipo17.ipynb desde notebook_content.py + figure_narratives.py
	poetry run python scripts/build_avance1_notebook.py

paper-methods-notebook:  ## Regenera y ejecuta notebooks/eda/02e_eda_metodos_paper.ipynb (7 metodos de 4 papers)
	poetry run python scripts/build_paper_methods_notebook.py
	MPLBACKEND=Agg poetry run papermill notebooks/eda/02e_eda_metodos_paper.ipynb \
	  notebooks/eda/02e_eda_metodos_paper.ipynb --no-progress-bar

paper-tables:  ## US-070: regenera las 6 tablas .tex del paper desde reports/ (sin hardcode)
	poetry run python -m ml.report.paper_tables

paper-figures: paper-tables  ## US-070: regenera + ejecuta (papermill) los 4 notebooks de figuras del paper
	poetry run python scripts/build_paper_us070_notebooks.py
	MPLBACKEND=Agg poetry run papermill paper/notebooks/01_figures_segmentation.ipynb \
	  paper/notebooks/01_figures_segmentation.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill paper/notebooks/02_figures_ensemble_farslip.ipynb \
	  paper/notebooks/02_figures_ensemble_farslip.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill paper/notebooks/03_figures_embeddings_fm.ipynb \
	  paper/notebooks/03_figures_embeddings_fm.ipynb --no-progress-bar
	MPLBACKEND=Agg poetry run papermill paper/notebooks/04_figures_agent_llm.ipynb \
	  paper/notebooks/04_figures_agent_llm.ipynb --no-progress-bar

eda-pdf:  ## Genera el reporte PDF del Avance 1 con las 7 fichas (S2, AlphaEarth, Bivariado, PASTIS, BreizhCrops, Literatura, Globales)
	poetry run python -m ml.report.export_pdf --output paper/avance1_eda_report.pdf

eda-dashboard:  ## Arranca el dashboard Streamlit del Avance 1 (8 tabs: 7 fichas + mapa espacial)
	poetry run streamlit run app/eda_dashboard.py --server.port 8501 --server.headless true

eda-dashboard-test:  ## Smoke test opcional con Playwright para el dashboard (US-013 AC-11 bonus)
	@echo "Playwright smoke optional (AC-11)"

# === Eval ===
eval-agromind:  ## make eval-agromind variant=gemini
	poetry run python ml/agent/eval/eval_agromind.py --variant=$(variant)

eval-geoanalyst:
	poetry run python ml/agent/eval/eval_geoanalyst.py --variant=$(variant)

# === FinOps ===
cost-audit:
	bash scripts/cost_audit.sh

scale-to-zero-check:
	gcloud run services list --format='table(metadata.name,spec.template.spec.containers[0].resources.limits.cpu,spec.template.metadata.annotations.\"autoscaling.knative.dev/minScale\")'

# === Terraform ===
tf-init:  ## make tf-init env=dev
	cd infrastructure/terraform/environments/$(env) && terraform init

tf-plan:
	cd infrastructure/terraform/environments/$(env) && terraform plan -out tfplan

tf-apply:
	cd infrastructure/terraform/environments/$(env) && terraform apply tfplan

tf-fmt:
	terraform fmt -recursive infrastructure/terraform/

tf-validate:
	cd infrastructure/terraform/environments/$(env) && terraform validate

# === Deploy ===
deploy-staging:  ## Cloud Build → staging
	gcloud builds submit --config=infrastructure/cloudbuild.yaml --substitutions=_ENV=staging

deploy-prod:
	@[ "$(shell git rev-parse --abbrev-ref HEAD)" = "main" ] || (echo "ERROR: deploy-prod solo desde main"; exit 1)
	gcloud builds submit --config=infrastructure/cloudbuild.yaml --substitutions=_ENV=prod

# === FarSLIP (US-017 / US-016b) ===
farslip-dataset-build:  ## US-017 — Construye dataset pares imagen-texto (3 ROIs italianas)
	poetry run python -c "from pathlib import Path; from ml.farslip.dataset import build_farslip_pairs; build_farslip_pairs(rois=('pianura_padana','toscana','puglia'), output_root=Path('data/farslip_pairs'), vocabulary_path=Path('ml/farslip/cap_vocabulary.yaml'))"

farslip-dataset-check:  ## US-017 AC-3 gate — n_pairs>=30k + balance min/max ROI>=0.20
	poetry run python -m ml.farslip.dataset_audit

farslip-train:  ## US-017 AC-4 — entrena FarSLIP (CPU smoke local o GCP L4 spot)
	poetry run python -m ml.farslip.train --rois italy --epochs 4 --batch-size 64 --lr 1e-5 --seed 42 --output-dir artifacts/farslip

farslip-extract-embeddings:  ## US-022-c P1 B-4 — extrae embeddings FarSLIP (85951 x 514) desde student MLflow @Production
	poetry run python -m ml.farslip.extract_embeddings \
	  --student-checkpoint mlflow://Models/farslip-clip-italy-v1@Production \
	  --parcels-parquet data/features/features_fused_v1.parquet \
	  --rois italy \
	  --output data/farslip/embeddings_pastis.parquet \
	  --batch-size 256 \
	  --device auto \
	  --seed 42

farslip-eval-pastis:  ## US-022-c P1 B-3 — eval mIoU FarSLIP vs RemoteCLIP en PASTIS-R (gate +0.05)
	poetry run python scripts/build_farslip_eval_notebook.py
	MPLBACKEND=Agg poetry run papermill notebooks/features/04_farslip_eval_pastis.ipynb \
	  notebooks/features/04_farslip_eval_pastis.ipynb --no-progress-bar

farslip-smoke-eval:  ## US-017 — smoke eval extractor desde GCS o cache local
	poetry run python scripts/farslip_smoke_eval.py --n-patches 10

# === Security ===
security-audit:
	bash scripts/security_audit.sh

# === Documentation (Paper Track / Avance 7, US-071) ===
DOCS_DIR := docs/final_doc
PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error -file-line-error

docs-pdf:  ## Compila los PDFs de docs/final_doc (Avance7 ES + EN) con pdflatex (2 pasadas c/u). Requiere LaTeX (MiKTeX/TeX Live).
	cd $(DOCS_DIR) && $(PDFLATEX) Avance7_equipo17.tex
	cd $(DOCS_DIR) && $(PDFLATEX) Avance7_equipo17.tex
	cd $(DOCS_DIR) && $(PDFLATEX) Avance7_equipo17_english.tex
	cd $(DOCS_DIR) && $(PDFLATEX) Avance7_equipo17_english.tex
	@echo PDFs generados en $(DOCS_DIR): Avance7_equipo17.pdf y Avance7_equipo17_english.pdf

docs-pdf-clean:  ## Borra auxiliares LaTeX (.aux .log .out .toc ...) de docs/final_doc, conserva los .pdf
	cd $(DOCS_DIR) && rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.pdf

DOCS_IMAGE := agrosat-docs-latex:dev

docs-pdf-docker:  ## Compila los PDFs en un contenedor texlive (no requiere LaTeX local). Solo necesita Docker.
	docker build -f infrastructure/docker/docs-latex.Dockerfile -t $(DOCS_IMAGE) infrastructure/docker
	docker run --rm -v "$(CURDIR):/repo" -w /repo/$(DOCS_DIR) $(DOCS_IMAGE)

# === Paper Track manuscript (EPIC 11, US-071) ===
# Manuscrito modular en paper/ (main.tex + sections/ + bib/refs.bib).
# Usa BibTeX, asi que la secuencia es pdflatex -> bibtex -> pdflatex x2.
PAPER_DIR := paper
PAPER_IMAGE := agrosat-paper-latex:dev

paper-pdf:  ## Compila paper/main.tex (manuscrito EPIC 11) con BibTeX. Requiere LaTeX (MiKTeX/TeX Live) + bibtex.
	cd $(PAPER_DIR) && $(PDFLATEX) main.tex
	cd $(PAPER_DIR) && bibtex main
	cd $(PAPER_DIR) && $(PDFLATEX) main.tex
	cd $(PAPER_DIR) && $(PDFLATEX) main.tex
	@echo PDF generado en $(PAPER_DIR): main.pdf

paper-pdf-clean:  ## Borra auxiliares LaTeX (.aux .bbl .blg .log ...) de paper/, conserva main.pdf
	cd $(PAPER_DIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	cd $(PAPER_DIR) && rm -f sections/*.aux

paper-pdf-docker:  ## Compila paper/main.tex en un contenedor texlive (no requiere LaTeX local). Solo necesita Docker.
	docker build -f infrastructure/docker/paper-latex.Dockerfile -t $(PAPER_IMAGE) infrastructure/docker
	docker run --rm -v "$(CURDIR):/repo" -w /repo/$(PAPER_DIR) $(PAPER_IMAGE)

paper-cite-check:  ## Valida que cada \cite{} del manuscrito tenga entrada en paper/bib/refs.bib (sin LaTeX).
	poetry run python scripts/paper_cite_check.py

paper-artifacts-check:  ## Recalcula el MD5 de cada artefacto sellado en paper/ARTIFACTS.md y falla si cambio.
	poetry run python scripts/paper_artifacts_check.py

paper-artifacts-seal:  ## Recalcula desde git la columna de procedencia del ledger (usa --escribir).
	poetry run python scripts/paper_artifacts_seal.py --escribir

paper-obsoletos-check:  ## Falla si un documento activo cita cifras de artefactos OBSOLETO sin cuarentena.
	poetry run python scripts/paper_obsoletos_check.py

protocolo-check:  ## Impide congelar el protocolo de US-172 con campos operativos sin rellenar.
	poetry run python scripts/protocolo_check.py

us172-adjuntos:  ## Genera los cuatro PDF de la consulta al comite de etica desde el protocolo.
	poetry run python scripts/build_us172_adjuntos.py --salida reports/us172

micai-pdf:  ## Compila el manuscrito MICAI (paper/micai) con bibliografia y devuelve su numero de paginas.
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	cd paper/micai && bibtex main >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
	@cd paper/micai && test "$$(grep -c '^!' main.log)" = "0" || (echo "micai-pdf: LaTeX reporto errores" && exit 1)
	@cd paper/micai && test "$$(grep -c Overfull main.log)" = "0" || (echo "micai-pdf: hay cajas overfull" && exit 1)
	@cd paper/micai && test "$$(grep -c 'Citation.*undefined' main.log)" = "0" || (echo "micai-pdf: hay citas sin resolver" && exit 1)
	@cd paper/micai && test "$$(grep -c 'Reference.*undefined' main.log)" = "0" || (echo "micai-pdf: hay referencias cruzadas sin resolver" && exit 1)
	@pdfinfo paper/micai/main.pdf | grep -iE "pages|page size"

micai-pdf-es:  ## Compila la version en espanol (lectura y revision del equipo), anonima.
	cd paper/micai && rm -f camera-ready.flag
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main_es.tex >/dev/null
	cd paper/micai && bibtex main_es >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main_es.tex >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error main_es.tex >/dev/null
	@cd paper/micai && test "$$(grep -c '^!' main_es.log)" = "0" || (echo "micai-pdf-es: LaTeX reporto errores" && exit 1)
	@cd paper/micai && test "$$(grep -c Overfull main_es.log)" = "0" || (echo "micai-pdf-es: hay cajas overfull" && exit 1)
	@pdfinfo paper/micai/main_es.pdf | grep -iE "pages|page size"

micai-anon-check:  ## Gate de doble ciego sobre los dos PDF anonimos, con autoprueba en negativo.
	poetry run python scripts/paper_micai_anon_check.py paper/micai/main.pdf
	poetry run python scripts/paper_micai_anon_check.py paper/micai/main_es.pdf

micai-bib:  ## Regenera paper/micai/refs.bib desde la matriz verificada de la fase 0.
	poetry run python scripts/build_paper_micai_bib.py

micai-pdf-cr:  ## Compila el camera-ready (no anonimo) del manuscrito MICAI, en main_cr.pdf.
	cd paper/micai && touch camera-ready.flag
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr main.tex >/dev/null
	cd paper/micai && bibtex main_cr >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr main.tex >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr main.tex >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr_es main_es.tex >/dev/null
	cd paper/micai && bibtex main_cr_es >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr_es main_es.tex >/dev/null
	cd paper/micai && pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_cr_es main_es.tex >/dev/null
	cd paper/micai && rm -f camera-ready.flag
	@cd paper/micai && test "$$(grep -c '^!' main_cr.log)" = "0" || (echo "micai-pdf-cr: LaTeX reporto errores" && exit 1)
	@cd paper/micai && test "$$(grep -c Overfull main_cr.log)" = "0" || (echo "micai-pdf-cr: hay cajas overfull" && exit 1)
	@echo "El camera-ready NO debe pasar el gate de anonimato; se comprueba que efectivamente falla:"
	@poetry run python scripts/paper_micai_anon_check.py paper/micai/main_cr.pdf; test $$? -eq 1 || (echo "micai-pdf-cr: el camera-ready no revela identidad, algo va mal" && exit 1)
	@pdfinfo paper/micai/main_cr.pdf | grep -iE "pages|page size"

plan-check:  ## Comprueba el plan por epicas: dependencias, ciclos, estados y camino critico.
	poetry run python scripts/plan_check.py
