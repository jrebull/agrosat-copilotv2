# Runbook — Levantar AgroSatCopilot en local

Guía paso a paso para que cualquier integrante levante el sistema completo (backend + frontend + chat conversacional con clasificación de cultivos) en su máquina, evitando los problemas que ya diagnosticamos. Probado en **Windows 11 + NVIDIA GPU**; las notas para CPU/Linux están señaladas.

> TL;DR de arquitectura recomendada en Windows:
> - **PostgreSQL + Redis** → en Docker.
> - **Backend (FastAPI)** → en **local con Poetry** (NO en Docker: la imagen `api` excluye el grupo `ml`/`torch` a propósito y no puede correr el chat).
> - **Frontend (Nuxt)** → en **local con `pnpm dev`** (la imagen Docker del frontend está rota).
> - **Gemini + Earth Engine** → vía **ADC de gcloud** (Vertex AI), sin API keys.

---

## 0. Prerrequisitos (instalar una sola vez)

| Herramienta | Versión | Notas |
|---|---|---|
| Docker Desktop | reciente | Para `postgres`, `redis`. |
| Python | **3.12** | El lock fija 3.12. Recomendado `pyenv-win`. |
| Poetry | **2.2.1** | `pipx install poetry==2.2.1`. |
| Node | **20+** | Para el frontend. |
| pnpm | **10+** | `corepack enable && corepack prepare pnpm@10 --activate`. Nunca npm/yarn. |
| gcloud CLI | reciente | Google Cloud SDK (incluye `bq`). |
| dbmate | reciente | Migraciones SQL. (O usar los `make db-*`.) |
| GPU NVIDIA + driver | CUDA **13.x** | Para `torch 2.11.0+cu130`. Sin GPU: ver [Apéndice B](#apéndice-b--sin-gpu-cpu). |
| DVC | (viene con `poetry install`) | Para descargar los datos versionados. |

Accesos Google necesarios (pídelos si no los tienes):
- Cuenta con acceso al proyecto GCP **`agrosat-copilot`**.
- **Vertex AI API** y **Earth Engine** habilitados en ese proyecto.

---

## 1. Clonar y configurar `.env.local`

```bash
git clone <repo-url> agrosat-copilot
cd agrosat-copilot
```

Crea **`.env.local`** en la raíz (es el archivo de configuración del backend; `pydantic-settings` con `extra="forbid"`, así que **toda** variable debe estar declarada en `backend/app/core/config.py`). Contenido mínimo que funciona:

```dotenv
# Postgres en Docker, puerto host 55432 (5432 suele estar ocupado por un PG nativo).
DATABASE_URL=postgresql+asyncpg://agrosat:agrosat@localhost:55432/agrosat

# Redis en Docker. Usar 6381 (ver paso 4: arrancar con REDIS_HOST_PORT=6381).
REDIS_URL=redis://localhost:6381/0

# Earth Engine / Vertex usan ADC (no service-account JSON en dev). Solo el proyecto.
GEE_PROJECT_ID=agrosat-copilot

# Reasoner Gemini por Vertex AI usando ADC. gemini-2.5-pro está en Vertex para
# este proyecto (gemini-3.5-flash da 404). NO se necesita API key de AI Studio.
GEMINI_MODEL=gemini-2.5-pro
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=agrosat-copilot
GOOGLE_CLOUD_LOCATION=us-central1

# El frontend corre en :3001 (3000 suele estar ocupado por otro proyecto).
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

> **Importante:** `REDIS_HOST_PORT` NO va en `.env.local` (no es un campo de `Settings`; rompería el arranque por `extra="forbid"`). Es una variable de Docker Compose que se pasa al levantar (paso 4).

---

## 2. Dependencias del backend (Python)

Instala **todos los grupos** que el chat necesita en una sola pasada (esto evita el goteo de "No module named X"):

```bash
poetry install --with dev,test,ml,ml-gpu,geo
```

- `ml` → clasificadores, mlflow, polars, transformers, etc.
- `ml-gpu` → **torch 2.11.0+cu130** (Windows/Linux con GPU NVIDIA).
- `geo` → rasterio, earthengine-api, spyndex, eemont, shapely, h3, etc.

> Por qué tantos grupos: el tool de clasificación importa `ml.train`, que arrastra **todo** el stack de modelado (smp, monai, spyndex…). El backend no arranca el `/chat` sin ese stack completo.

### 2.1 Gotcha torch/torchvision (CRÍTICO en GPU)

El lock fija `torch ...+cu130` pero **no** fija `torchvision` desde el índice cu130, así que `pip`/`poetry` puede dejar una `torchvision +cpu` incompatible → error en runtime: `operator torchvision::nms does not exist`.

Verifica y, si no casan, instala la torchvision del índice cu130:

```bash
poetry run python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__)"
# Si torchvision termina en +cpu (no +cu130):
poetry run pip install "torchvision==0.26.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
```

### 2.2 Gotcha polars sin binario

Si `polars.__version__` sale vacío o ves `UserWarning: Polars binary is missing!` (rompe mlflow con `Invalid version: ''`), reinstala polars a la versión del lock (trae el runtime nativo):

```bash
poetry run pip install --force-reinstall --no-cache-dir "polars==1.40.1"
poetry run python -c "import polars as pl; print('polars', pl.__version__)"  # debe imprimir 1.40.1
```

### 2.3 Verificación rápida del entorno

```bash
poetry run python -c "import importlib.util as u; mods=['torch','torchvision','xgboost','h3','mlflow','polars','ee','rasterio','spyndex','eemont','segmentation_models_pytorch','monai','sentence_transformers','google.genai','litellm']; m=[x for x in mods if not u.find_spec(x)]; print('FALTAN:', m or 'ninguna')"
```

Debe imprimir `FALTAN: ninguna`.

---

## 3. Dependencias del frontend

```bash
cd frontend
pnpm install
cd ..
```

---

## 4. Infra local: Postgres + Redis (Docker)

```bash
REDIS_HOST_PORT=6381 docker compose --env-file .env.local up -d postgres redis
docker compose --env-file .env.local ps   # ambos "healthy"
```

- Postgres queda en `localhost:55432`, Redis en `localhost:6381` (casa con `.env.local`).
- **No** levantes los servicios `api`/`frontend` de Docker (ver TL;DR): corren en local.

---

## 5. Datos versionados (DVC)

El clasificador ajusta XGBoost sobre `data/features/features_fused_pastis.parquet` (124 MB) y lee los OOF de stacking/voting. Descárgalos:

```bash
dvc pull
# o, si solo quieres lo mínimo del clasificador:
dvc pull data/features/features_fused_pastis.parquet.dvc
```

---

## 6. Migraciones y datos demo

```bash
dbmate up                                   # crea chat_sessions, aois, parcels, features_parcels, RLS...
poetry run python scripts/seed.py           # sesión demo + AOI Tuscany
poetry run python scripts/seed_demo_parcels.py   # parcelas demo dentro del AOI
```

---

## 7. Autenticar Google (Vertex reasoner + Earth Engine)

El reasoner (Gemini por Vertex) y el muestreo AlphaEarth (Earth Engine) usan **ADC**:

```bash
gcloud config set project agrosat-copilot
gcloud auth application-default login        # abre navegador; usa tu cuenta con acceso al proyecto
gcloud auth application-default set-quota-project agrosat-copilot
```

> Si tienes varias cuentas: `gcloud config set account <tu-correo>` y repite el `application-default login` con `--account=<tu-correo>`.

---

## 8. Levantar backend y frontend

### Backend (local, SIN `--reload`)

```bash
poetry run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

- El primer arranque tarda ~60 s (importa torch + stack ml).
- **NO uses `--reload` en Windows**: deja procesos huérfanos que retienen el puerto 8000 con código viejo (ver [Troubleshooting](#9-troubleshooting)).

### Frontend (local)

En otra terminal:

```bash
cd frontend
NUXT_PUBLIC_API_BASE_URL=http://localhost:8000 pnpm exec nuxt dev --port 3001
```

URLs:
- Frontend: **http://localhost:3001**
- Backend docs: **http://localhost:8000/docs** · health: `/healthz`, `/readyz`

---

## 9. Sesiones de chat (automáticas, US-080)

Ya **no hay que crear la sesión a mano**. Al abrir el frontend, el cliente llama a **`POST /sessions`** y registra la sesión en la BD automáticamente (`useSessions.ensureActiveSession`). Además:

- **Múltiples chats en pestañas**: el switcher arriba del panel permite crear (+), cambiar y cerrar (x) chats; cada pestaña es **su propia sesión** con su propio historial.
- **Historial en Postgres**: cada turno (user + assistant) se persiste en `chat_messages` y se restaura al cambiar de pestaña o recargar (`GET /sessions/{id}/messages`).
- **Mapa aislado por chat**: la zona dibujada (AOI) se guarda por sesión; marcar una zona en el Chat 1 no afecta al Chat 2.
- **Pestañas desde el servidor (sin auth)**: al abrir, el front llama `GET /sessions` con un `X-User-ID` **estable** (`local-user`, constante mientras no haya auth) y reconstruye las pestañas desde la BD — así los chats se recuperan en cualquier navegador de este despliegue. El listado respeta RLS vía la función `list_chat_sessions(user_id)` (`SECURITY DEFINER`). Cuando entre Clerk, basta cambiar ese id constante por el id real del usuario.

> Requiere la migración `20260628120000_create_chat_messages` aplicada (tabla `chat_messages` + `chat_sessions.title`). Si clonas limpio: `dbmate up` (o aplica el bloque `migrate:up` con psql como el resto del runbook).

El INSERT manual de antes **ya no es necesario**.

---

## 10. Verificación end-to-end (smoke)

```bash
# Salud
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/readyz   # 200

# Chat con un AOI dibujado (ejemplo: Tuscany). Debe responder con un cultivo,
# no con "needs_gee_sampling".
curl -s -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <TU-UUID>" \
  -d '{"messages":[{"role":"user","content":"que cultivo hay en esta zona?"}],"aoi":{"type":"Polygon","coordinates":[[[11.10,43.30],[11.11,43.30],[11.11,43.31],[11.10,43.31],[11.10,43.30]]]},"year":2019,"locale":"es"}'
```

En los logs del backend deberías ver `classify_new_parcel_embedding_resolved source=gee` (buscó en BD, no encontró parcela persistida y **descargó** el embedding de Earth Engine).

---

## 11. Troubleshooting (problemas que ya resolvimos)

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named '<x>'` al arrancar el back (`torch`, `polars`, `mlflow`, `xgboost`, `h3`, `spyndex`, `segmentation_models_pytorch`…) | El `.venv` no tiene el grupo `ml`/`geo` completo | `poetry install --with dev,test,ml,ml-gpu,geo` (paso 2). Si poetry falla por red, reintenta o instala los faltantes con `poetry run pip install <pkg>`. |
| `operator torchvision::nms does not exist` | `torchvision +cpu` no casa con `torch +cu130` | Paso [2.1](#21-gotcha-torchtorchvision-crítico-en-gpu). |
| `packaging.version.InvalidVersion: Invalid version: ''` / `Polars binary is missing!` | polars sin binario nativo | Paso [2.2](#22-gotcha-polars-sin-binario). |
| `/chat` → **403** `chat_session_forbidden` | La sesión del front no está en `chat_sessions` | Paso [9](#9-registrar-la-sesión-del-chat-necesario-una-vez). |
| Chat responde "dibuja el área" / perceiver `crop_class=needs_gee_sampling` | El AOI no intersecta una parcela persistida **y** el muestreo GEE no pudo correr | Verifica ADC (paso 7), `GEE_PROJECT_ID`, y que `dvc pull` haya traído el parquet de features (paso 5). |
| `/chat` → `No API key was provided` | El reasoner cayó en modo AI Studio sin key | Activa Vertex en `.env.local` (`GOOGLE_GENAI_USE_VERTEXAI=true` + project/location) y completa el ADC (paso 7). Reinicia el back. |
| Cambios en `.env.local`/código no surten efecto; el puerto 8000 sigue con comportamiento viejo | Worker **huérfano** de `uvicorn --reload` reteniendo el socket | No uses `--reload`. Para matarlo: `Get-NetTCPConnection -LocalPort 8000 -State Listen \| % { taskkill /F /T /PID $_.OwningProcess }` y relanza. |
| `api` en Docker crashea con `ModuleNotFoundError: torch` | La imagen `api` instala solo `dev,test` (sin `ml`) por diseño | Corre el backend en **local** (paso 8), no en Docker. |
| Redis no conecta (rate-limit) | Compose mapea Redis a `63790` por defecto, pero `.env.local` usa `6381` | Levanta con `REDIS_HOST_PORT=6381 docker compose ... up -d` (paso 4). |
| Frontend Docker no buildea (`COPY --from=deps /pnpm` not found) | Bug en `frontend.Dockerfile` (cache-mount no persistente) + `node_modules` win32 | Corre el frontend en **local** con `pnpm dev` (paso 8). |
| `/chat` devuelve "texto vacío" en la UI | Falso positivo: el `text_delta` sí trae texto; suele ser el reasoner pidiendo dibujar el AOI porque el perceiver dio `needs_gee_sampling` | Revisa la causa real arriba (ADC/GEE/sesión). |

---

## Apéndice A — Cambios en el working tree pendientes de commit

Para que el flujo "buscar → si no está, descargar de GEE" y el resto funcione, estos cambios deben estar en la rama (si haces `git pull` y no los ves, pídelos / commitéalos):

- `ml/ingest/gee_sampler.py` → nueva `sample_alphaearth_aoi_mean(...)` (muestreo AlphaEarth de un AOI vía Earth Engine).
- `ml/agent/tools/classify.py` → fallback que, si no hay embedding persistido que intersecte el AOI, **descarga** el embedding de GEE antes de caer a `needs_gee_sampling`.
- `backend/app/services/chat_service.py` → en `_agent_messages`, la observación del perceiver se inyecta con una directiva explícita ("el usuario ya seleccionó el AOI; responde con esta observación; no pidas dibujar"); sin esto, el reasoner respondía "dibuja el área" pese a tener una clasificación válida.
- `frontend/layouts/default.vue` → `<ChatDock>` envuelto en `<ClientOnly>`. El store del chat persiste en `localStorage` (`pick: ["messages"]`); con SSR el servidor renderizaba el transcript vacío y el cliente hidrataba con mensajes → **hydration mismatch** que dejaba el texto del asistente (streaming `text_delta`) **sin pintar** aunque sí llegaba al store. `ClientOnly` elimina el desajuste.
- `docker-compose.yml` → `command` del `api` corregido a `uvicorn backend.app.main:app` (solo relevante si algún día se corre el api en Docker con el stack ml).

## Apéndice B — Sin GPU (CPU)

Si no tienes GPU NVIDIA, omite `ml-gpu` y usa torch CPU:

```bash
poetry install --with dev,test,ml,geo
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

El chat funciona igual (la clasificación tabular XGBoost es CPU); solo afecta a entrenamientos pesados.

## Apéndice C — Hacer reproducible lo que hoy parchamos con pip

Lo ideal es que el equipo deje el entorno consistente desde el lock para no depender de los `pip install` manuales:

- Considerar **fijar `torchvision==0.26.0+cu130`** en el grupo `ml-gpu` (source `pytorch-cu130`) del `pyproject.toml`, para que `poetry install` traiga la torchvision correcta.
- Revisar el pin de `polars` (que el wheel traiga `polars-runtime-*`).
- Tras ajustar, `poetry lock` + commit, y todos hacen `poetry install --with dev,test,ml,ml-gpu,geo` sin parches.

## Apéndice D — macOS Apple Silicon (verificado 2-sep-2026, M3 Pro, macOS 26)

Todo el repo corre en Mac arm64 sin GPU NVIDIA. Estos son los pasos y parches que
hicieron falta además de los del cuerpo del runbook; ninguno toca el `poetry.lock`.

```bash
# 1) Herramientas (Homebrew)
brew install poetry pnpm gitleaks dbmate libomp librsvg
brew tap hashicorp/tap && brew install hashicorp/tap/terraform   # solo para `terraform validate`

# 2) Entorno Python (venv dentro del repo, Python 3.12 de Homebrew)
export POETRY_VIRTUALENVS_IN_PROJECT=true
poetry env use /opt/homebrew/opt/python@3.12/bin/python3.12
poetry install --with dev,test,ml,geo,dagster,paper    # FALLA en torch: el lock lo fija a +cu130
poetry run pip install "torch==2.11.0" "torchvision==0.26.0"   # ruedas macOS arm64 de PyPI (MPS)
poetry install --with dev,test,ml,geo,dagster,paper --dry-run  # lista lo que quedó sin instalar;
#   instalar esa lista con `poetry run pip install pkg==ver` EXCEPTO torch y tree-sitter-python
poetry install --only-root                              # instala el paquete raíz (ml, backend, dagster_project)

# 3) OpenMP: torch trae su propio libomp.dylib y xgboost/lightgbm usan el de Homebrew.
#    Con dos runtimes cargados el proceso hace segfault (exit 139) o se cuelga.
#    Solución: que torch use el mismo libomp que el resto. (scikit-learn trae una
#    tercera copia en `sklearn/.dylibs/`, que no ha dado problemas.)
TL=.venv/lib/python3.12/site-packages/torch/lib/libomp.dylib
mv "$TL" "$TL.orig" && ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib "$TL"

# 4) Frontend
cd frontend && pnpm install && cd ..                    # pnpm 11 cambia solo a la 10.24.0 del packageManager

# 5) Infra local. NO copiar .env.example tal cual: trae variables que Settings rechaza
#    (extra="forbid") y publica Postgres/Redis en 5432/6379. Este mínimo funciona:
cat > .env.local <<'EOF'
DATABASE_URL=postgresql+asyncpg://agrosat:agrosat@localhost:55432/agrosat
REDIS_URL=redis://localhost:63790/0
GEE_PROJECT_ID=agrosat-copilot
GEMINI_MODEL=gemini-2.5-pro
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=agrosat-copilot
GOOGLE_CLOUD_LOCATION=us-central1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3010
EOF
docker compose --env-file .env.local up -d --build postgres redis   # 55432 y 63790 (defaults del compose)
DATABASE_URL='postgres://agrosat:agrosat@localhost:55432/agrosat?sslmode=disable' dbmate --no-dump-schema up
#   `--no-dump-schema`: sin `pg_dump` local dbmate fallaría al volcar db/schema.sql tras migrar.
#   La imagen de Postgres es amd64 y corre emulada en arm64: funciona, pero es más lenta.

# 6) Paper (BasicTeX es de root; instalar paquetes en modo usuario desde el repositorio histórico 2025;
#    `units` aporta nicefrac.sty, que main.tex carga)
tlmgr init-usertree
tlmgr --usermode --repository https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final install units multirow import
#   Los PNG de las figuras del paper están gitignorados; se regeneran desde reports/ (ver paper/AGENTS.md)
#   o se rasterizan desde los SVG versionados:
cd paper && for s in $(git ls-files 'figures/*/*.svg' | grep -v '_es\.svg$'); do
  [ -f "${s%.svg}.png" ] || rsvg-convert --dpi-x 300 --dpi-y 300 --zoom 3 -o "${s%.svg}.png" "$s"; done; cd ..
make paper-pdf
```

Acceso a GCP (validado el 2-sep-2026 con la cuenta que autorizó el equipo):

```bash
# Configuración de gcloud aparte, para no pisar otros proyectos de la misma cuenta
gcloud config configurations create agrosat --activate
gcloud auth login                              # cuenta Gmail autorizada en el proyecto
gcloud config set project agrosat-copilot
gcloud auth application-default login          # ADC para Python, Earth Engine y DVC (fija el quota project)
# Verificación (la que pide el equipo)
gcloud auth application-default print-access-token > /dev/null && gcloud config get-value project   # agrosat-copilot
poetry run python -c "import ee; ee.Initialize(project='agrosat-copilot'); print(ee.Number(42).getInfo())"  # 42
dvc status -c                                  # compara con gs://agrosat-dvc-remote
dvc pull -R data/features ml/eval/oof reports  # artefactos ligeros (< 3 GB) que alimentan tablas y figuras; -R para directorios
```

- El `.env.local` del equipo apunta a dos JSON de service account bajo `./.env/` que no se
  distribuyen; en local se comentan esas dos líneas y todo usa ADC. Los secretos viven en
  Secret Manager del proyecto (8 secretos), pero el rol otorgado no incluye leerlos.
- En Vertex AI responde `gemini-2.5-pro` (y `gemini-2.5-flash`); `gemini-3.1-pro` devuelve 404.
  Ajustar `GEMINI_MODEL` en `.env.local` si el archivo compartido trae otro valor.
- El remoto DVC tiene todo salvo `data/features/alphaearth_italia_2018.parquet` (su `.dvc`
  existe pero nunca se hizo `dvc push`). Las imágenes crudas de PASTIS-R no están en DVC, solo
  sus derivados; el dataset completo (53.7 GB) está en EOTDL y solo hace falta para reentrenar.

Limitaciones conocidas en Mac:

- `tree-sitter-python==0.21.0` (grupo `dev`, dependencia de `codebleu`) no publica rueda para
  macOS arm64 ni sdist (sí para Linux, Windows y macOS x86_64). Solo lo usa la métrica CodeBLEU,
  importada de forma diferida en `ml/eval/agent_metrics.py`, que devuelve 0.0 con aviso si falta.
- `make secrets-scan` marcaba un falso positivo histórico en `docs/us-planning/us-017.md`
  (docstring "CLS token output" junto a una URI `gs://`); está listado en `.gitleaksignore`.
- Sin credenciales del equipo (GCS del DVC, GEE, Vertex) no se descargan datos ni pesos: los tests
  marcados `empirical`/`requires_gee` se saltan y `make notebooks-check` no aplica.
- `mypy -p backend.app` desde la raíz reportaba errores en `ml/` mientras el paquete raíz no
  estaba instalado; tras `poetry install --only-root` queda limpio, igual que `make lint`.
- Quedan 14 tests de `tests/ml` que fallan por deuda del upstream, independiente del sistema
  operativo y ya anotada en `docs/blockers/PENDIENTES.md` §3.5: regex en español contra mensajes
  de error en inglés (`analysis`, `ingest`, `models`, `report`, `utils`), lambdas de prueba sin el
  parámetro `region_prefix` en `tests/ml/transfer`, CodeBLEU sin `tree-sitter-python`
  (`tests/ml/eval/test_agent_metrics.py`) y el benchmark LLM que pasó de Gemini 2.5-pro a
  2.5-flash sin actualizar `tests/ml/eval/test_paper_bench.py`.
