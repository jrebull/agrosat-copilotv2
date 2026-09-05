# infrastructure/ — Guía del agente

> Scope: IaC del monorepo (Terraform GCP, Dockerfiles, Cloud Build). Reglas NON-NEGOTIABLE en el orquestador root [`../AGENTS.md`](../AGENTS.md) / [`../CLAUDE.md`](../CLAUDE.md) — no se repiten aquí.

## Estado

**DORMIDO — solo `dev`, y detenido.** Por [ADR-002](../docs/decisions/ADR-002-single-env-dev.md) hay un único entorno con Terraform real. `environments/staging/` y `environments/prod/` son solo `README.md` (out of scope, sin `.tf`). El artículo MICAI 2027 corre en CPU y no necesita nada de esta capa: se reactiva solo si una US lo pide, con presupuesto en su spec §7 y consumo en la bitácora §1.5.

- Capa GCP (`modules/gcp/`): Cloud Run x4 (api, frontend, tiling, inference-worker), Cloud SQL PG15, GCS, Pub/Sub, Secret Manager, Artifact Registry, IAM least-privilege. **Aplicada; Cloud SQL con `db_activation_policy = "NEVER"`** (instancia detenida, datos preservados). Subir a `"ALWAYS"` antes de retomar trabajo que use la DB, y volver a `"NEVER"` al terminar.
- Capa Azure (`modules/azure/`): **código histórico, no se reactiva.** La VM H100 del sponsor se perdió con tres checkpoints FarSLIP dentro ([`docs/paper/artefactos-perdidos.md`](../docs/paper/artefactos-perdidos.md)); el `provider "azurerm"` y `module "azure"` siguen comentados en `environments/dev/main.tf` y así se quedan (AGENTS.md raíz §Descartados). Los scripts `scripts/azure_h100_*.sh` y `scripts/bootstrap_sponsor_h100.ps1` se conservan como historia y ya no tienen target Make.
- GPU para reentrenamientos: L4 spot por US (`make train-l4`, definiciones en `compute/`), creada y destruida dentro de la ventana de esa US. Sin VM permanente.

## Comandos

```bash
make tf-init env=dev          # terraform init en environments/dev
make tf-plan env=dev          # plan -out tfplan
make tf-apply env=dev         # apply tfplan (revisar plan antes)
make tf-fmt                   # fmt -recursive infrastructure/terraform/
make tf-validate env=dev      # validate
make deploy-staging           # Cloud Build → staging (dormido)
make deploy-prod              # Cloud Build → prod (gate: rama main; dormido)
make train-l4 epic=Ex us=US-xxx script=xxx.py   # L4 spot con MLFLOW_TRACKING_URI exportado
make cost-audit               # scripts/cost_audit.sh (GCP)
make scale-to-zero-check      # lista Cloud Run con minScale por servicio
```

## Stack local

- `docker-compose.yml` vive en la **raíz del repo**, NO en `infrastructure/`. Los Dockerfiles sí viven aquí en `docker/`: `backend.Dockerfile`, `frontend.Dockerfile`, `dagster.Dockerfile`, `inference-worker.Dockerfile`, `mlflow.Dockerfile`, `postgres.Dockerfile`, `ml-train.Dockerfile`.
- El servicio Cloud Run `tiling` usa la **imagen del backend** (TiTiler montado en FastAPI). NO existe `titiler.Dockerfile`.
- `cloudbuild.yaml` (build + push + migrate + deploy), `cloudbuild-ml.yaml` y `cloudbuild-mlflow-only.yaml`.

## Convenciones (✅/❌)

- ✅ Módulos en `terraform/modules/gcp/` (y `terraform/modules/azure/` como historia); entornos en `terraform/environments/`. NO existe `modules/vertex/` (Vertex vive dentro de `gcp/`).
- ✅ Region GCP default `us-central1` (en `modules/gcp/variables.tf` y `environments/dev/variables.tf`). ❌ NO es `europe-west1` (cualquier doc que lo diga está desactualizado).
- ✅ SAs least-privilege con `for_each` sobre listas de roles explícitos. ❌ Nunca `roles/owner` ni `roles/editor` en SAs runtime.
- ✅ Cloud Run `min_instances = 0` (scale-to-zero) en todos los servicios. ❌ Ningún recurso con instancias mínimas ni VM permanente; una GPU vive dentro de la ventana de una US.
- ✅ Secretos: TF crea solo el contenedor `google_secret_manager_secret` (los 7 no-DB en `secret_ids` quedan **vacíos**, se rellenan a mano fuera de TF). Excepción: `agrosat-db-password` sí lleva `secret_version` desde `random_password`.
- ❌ Nunca `0.0.0.0/0` salvo el puerto público HTTPS de Cloud Run.

## No tocar

- `.terraform/`, `*.tfstate`, `terraform.tfvars`, `*.auto.tfvars`, `.terraform.lock.hcl` — todos gitignored en `environments/dev/.gitignore`. Nunca commitear ni editar a mano.
- Backend state en `gs://agrosat-tfstate` (versionado). No migrar ni borrar objetos de estado manualmente.
- Secretos ya provisionados en Secret Manager: editar el **valor** por consola/CLI, no por TF.
- `modules/azure/` y los scripts `azure_h100_*`: historia. Ni se borran ni se reactivan.

## FinOps gotchas

- `db_activation_policy = "NEVER"` evita drift que reencendería la instancia en cada `apply` (var dedicada, no toques el recurso directo).
- GCP no encoge discos in-place: para reducir tamaño de un PD hay que snapshot → disco nuevo → rsync → reimportar a TF (ver memoria `disk-shrink-finops-procedure`).
- Objetivo mientras el artículo corre en CPU: **coste fijo = solo almacenamiento** (remoto DVC `gs://agrosat-dvc-remote`, `gs://agrosat-tfstate`, disco archivado de la VM FarSLIP L4 dada de baja el 26-ago-2026). Todo gasto variable se presupuesta por US.

## Tests

No hay suite pytest para esta capa. La validación es Terraform nativo:

```bash
cd infrastructure/terraform/environments/dev
terraform init -backend=false && terraform validate   # CI sin credenciales
terraform fmt -check -recursive ../..
```

`make check` (root) cubre lint/secrets-scan/i18n/espejos; no ejecuta `terraform plan`.

## Skills

- [`agrosat-terraform`](../.claude/skills/agrosat-terraform/SKILL.md) — módulos, backend state, workspaces.
- [`agrosat-gcp-services`](../.claude/skills/agrosat-gcp-services/SKILL.md) — Cloud Run, Cloud SQL, Pub/Sub, Secret Manager, IAM.
- [`agrosat-finops`](../.claude/skills/agrosat-finops/SKILL.md) — auditoría de costos, scale-to-zero, GPU por US.
- [`agrosat-security-audit`](../.claude/skills/agrosat-security-audit/SKILL.md) — CIS GCP pre-deploy.
