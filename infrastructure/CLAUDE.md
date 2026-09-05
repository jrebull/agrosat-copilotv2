# infrastructure/ — Guía del agente

> Scope: IaC del monorepo (Terraform GCP + Azure, Dockerfiles, Cloud Build). Reglas NON-NEGOTIABLE en el orquestador root [`../AGENTS.md`](../AGENTS.md) / [`../CLAUDE.md`](../CLAUDE.md) — no se repiten aquí.

## Estado

**ACTIVE — solo `dev`.** Por [ADR-002](../docs/decisions/ADR-002-single-env-dev.md) hay un único entorno con Terraform real. `environments/staging/` y `environments/prod/` son solo `README.md` (out of scope, sin `.tf`).

- Capa GCP (`modules/gcp/`): Cloud Run x4 (api, frontend, tiling, inference-worker), Cloud SQL PG15, GCS, Pub/Sub, Secret Manager, Artifact Registry, IAM least-privilege. **Aplicada y viva.**
- Capa Azure H100 (`modules/azure/`): el módulo existe completo, pero el `provider "azurerm"` y `module "azure"` están **COMENTADOS** en `environments/dev/main.tf` (US-022-c). Re-activar solo al abrir ventana H100 (rellenar `azure_subscription_id`, `allowed_ssh_cidrs`, `admin_ssh_public_key` en `terraform.tfvars`).
- FinOps: dev SQL corre con `db_activation_policy = "NEVER"` (instancia detenida, datos preservados). Subir a `"ALWAYS"` antes de retomar trabajo que use la DB.

## Comandos

```bash
make tf-init env=dev          # terraform init en environments/dev
make tf-plan env=dev          # plan -out tfplan
make tf-apply env=dev         # apply tfplan (revisar plan antes)
make tf-fmt                   # fmt -recursive infrastructure/terraform/
make tf-validate env=dev      # validate
make deploy-staging           # Cloud Build → staging
make deploy-prod              # Cloud Build → prod (gate: rama main)
make azure-h100-start         # enciende VM H100 spot
make azure-h100-stop          # apaga VM H100
make azure-h100-status        # estado + auto-shutdown timer
make cost-audit               # scripts/cost_audit.sh (GCP + Azure)
make scale-to-zero-check      # lista Cloud Run con minScale por servicio
```

## Stack local

- `docker-compose.yml` vive en la **raíz del repo**, NO en `infrastructure/`. Los Dockerfiles sí viven aquí en `docker/`: `backend.Dockerfile`, `frontend.Dockerfile`, `dagster.Dockerfile`, `inference-worker.Dockerfile`, `mlflow.Dockerfile`, `postgres.Dockerfile`, `ml-train.Dockerfile`.
- El servicio Cloud Run `tiling` usa la **imagen del backend** (TiTiler montado en FastAPI). NO existe `titiler.Dockerfile`.
- `cloudbuild.yaml` (build + push + migrate + deploy), `cloudbuild-ml.yaml` y `cloudbuild-mlflow-only.yaml`.

## Convenciones (✅/❌)

- ✅ Módulos en `terraform/modules/gcp/` y `terraform/modules/azure/`; entornos en `terraform/environments/`. NO existe `modules/vertex/` (Vertex vive dentro de `gcp/`).
- ✅ Region GCP default `us-central1` (en `modules/gcp/variables.tf` y `environments/dev/variables.tf`). ❌ NO es `europe-west1` (cualquier doc que lo diga está desactualizado).
- ✅ SAs least-privilege con `for_each` sobre listas de roles explícitos. ❌ Nunca `roles/owner` ni `roles/editor` en SAs runtime.
- ✅ Cloud Run `min_instances = 0` (scale-to-zero) en todos los servicios.
- ✅ Azure H100 spot + auto-shutdown (`shutdown_time_utc`) + NSG con SSH whitelist por CIDR. ❌ Nunca `0.0.0.0/0` salvo el puerto público HTTPS de Cloud Run.
- ✅ Secretos: TF crea solo el contenedor `google_secret_manager_secret` (los 7 no-DB en `secret_ids` quedan **vacíos**, se rellenan a mano fuera de TF). Excepción: `agrosat-db-password` sí lleva `secret_version` desde `random_password`.

## No tocar

- `.terraform/`, `*.tfstate`, `terraform.tfvars`, `*.auto.tfvars`, `.terraform.lock.hcl` — todos gitignored en `environments/dev/.gitignore`. Nunca commitear ni editar a mano.
- Backend state en `gs://agrosat-tfstate` (versionado). No migrar ni borrar objetos de estado manualmente.
- Secretos ya provisionados en Secret Manager: editar el **valor** por consola/CLI, no por TF.

## FinOps gotchas

- `db_activation_policy = "NEVER"` evita drift que reencendería la instancia en cada `apply` (var dedicada, no toques el recurso directo).
- GCP no encoge discos in-place: para reducir tamaño de un PD hay que snapshot → disco nuevo → rsync → reimportar a TF (ver memoria `disk-shrink-finops-procedure`).
- Target operativo ~$115 USD/mes: scale-to-zero + db-f1-micro + H100 spot puntual.

## Tests

No hay suite pytest para esta capa. La validación es Terraform nativo:

```bash
cd infrastructure/terraform/environments/dev
terraform init -backend=false && terraform validate   # CI sin credenciales
terraform fmt -check -recursive ../..
```

`make check` (root) cubre lint/secrets-scan/i18n; no ejecuta `terraform plan`.

## Skills

- [`agrosat-terraform`](../.claude/skills/agrosat-terraform/SKILL.md) — módulos, backend state, workspaces.
- [`agrosat-gcp-services`](../.claude/skills/agrosat-gcp-services/SKILL.md) — Cloud Run, Cloud SQL, Pub/Sub, Secret Manager, IAM.
- [`agrosat-azure-h100`](../.claude/skills/agrosat-azure-h100/SKILL.md) — VM `Standard_NC40ads_H100_v5`, scripts start/stop.
- [`agrosat-finops`](../.claude/skills/agrosat-finops/SKILL.md) — auditoría de costos, scale-to-zero.
- [`agrosat-security-audit`](../.claude/skills/agrosat-security-audit/SKILL.md) — CIS GCP/Azure pre-deploy.
