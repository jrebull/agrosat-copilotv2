# STATUS — estado del repositorio

> Documento referenciado por el plan v8 y por varias US (051, 062, 064) que nunca
> llegó a existir en el upstream. Esta versión lo materializa en el fork con lo que
> hay en disco y en la base de datos; para el detalle histórico de cada US ver
> [`us-resolved/`](us-resolved/) y para lo abierto [`blockers/PENDIENTES.md`](blockers/PENDIENTES.md).

Última actualización: 2 de septiembre de 2026.

## Base de datos (fuente de verdad: `dbmate status`)

| Migración | Qué aporta |
|---|---|
| `20260511213942_initial_schema` | extensiones, `chat_sessions`, `aois` |
| `20260516210000_create_parcels` | `parcels` |
| `20260516210100_create_features_parcels` | `features_parcels` con `VECTOR(64)` |
| `20260615082041_create_rag_documents` | `rag_documents` para Spatial-RAG |
| `20260620000418_rls_multi_tenant` | **RLS aplicada**: `FORCE ROW LEVEL SECURITY` y política `tenant_isolation` por `session_id`; rol `agrosat_app` |
| `20260620002624_alter_chat_sessions_llm_model` | `llm_variant` pasa a `llm_model` con 4 variantes |
| `20260628120000_create_chat_messages` | `chat_messages` con RLS |
| `20260628130000_list_chat_sessions_fn` | función `list_chat_sessions(text)` |
| `20260628233613_add_canonical_parcel_id_to_parcels` | `canonical_parcel_id` |
| `20260630120000_widen_chat_sessions_llm_model_qwen_vl` | variante `qwen-vl` |

Tablas con RLS forzada: `aois`, `chat_messages`, `chat_sessions`, `features_parcels`, `parcels`.

## Modelos y artefactos

- Campeón desplegado: Voting-3 v2, 12 clases (`france-12`), F1-macro 0.8992 (`reports/agent_bench/perceiver_champion_eval_v2.json`).
- Modelo final del curso: Stacking-5 heterogéneo con FarSLIP, F1-macro 0.7486 held-out fold-5 (`reports/ensemble/metrics/us043_farslip_grid.csv`); el Stacking-3 sin FarSLIP de US-040 da 0.7470 (`comparison_us040.csv`).
- Datos versionados en DVC (`gs://agrosat-dvc-remote`): features, OOF, embeddings FarSLIP, checkpoints y datasets de transferencia. Falta en el remoto `data/features/alphaearth_italia_2018.parquet`. PASTIS-R crudo no está en DVC: se descarga de Zenodo (53.7 GB, MD5 `4887513d6c2d2b07fa935d325bd53e09`) y en esta máquina ya está extraído en `data/PASTIS-R/` (gitignorado).

## Entorno y gates

- Corre en macOS Apple Silicon y en Linux/Windows con GPU (ver [`runbook-local-setup.md`](runbook-local-setup.md), Apéndice D para Mac).
- Gates locales: `make check` (ruff, gitleaks, i18n), `make test` (backend, cobertura ≥ 70 %), `make test-ml`, `make test-frontend` (vitest, cobertura ≥ 50 %), `make paper-pdf`.
- CI (`.github/workflows/ci.yml`): lint Python y frontend, migraciones dbmate, Terraform validate, gitleaks e i18n. No ejecuta pytest por decisión del equipo.

## Pendiente

Ver [`blockers/PENDIENTES.md`](blockers/PENDIENTES.md). Lo que requiere terceros o la VM H100 sigue igual; la deuda de tests y documentación se está cerrando en el fork (ver el historial de `main`).
