---
name: agrosat-finops
description: Audit and optimize cloud costs for AgroSatCopilot on GCP (Azure retired). Target while the article runs on CPU: fixed cost = storage only (Cloud Run scale-to-zero, Cloud SQL dev stopped); every variable spend (L4 spot GPU, GEE exports, LLM batches) is budgeted per US in its spec. Use for budget alerts, scale-to-zero verification, GPU rental windows, and cost reports.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot FinOps Skill

## Targets

| Categoría | Mensual operativo | Notas |
|-----------|-------------------|-------|
| Cloud Run × 4 | $33 | api, frontend, tiling, worker scale-to-zero |
| Cloud SQL PostGIS+pgvector | $14 | db-f1-micro 20 GB |
| GCS Standard | $6 | 250 GB |
| Redis Memorystore | $15 | Basic 1 GB |
| Pub/Sub + Tasks | $3 | <10 GB |
| Vertex AI (Gemini 3.1 Pro) | $12 | ~500k tokens/mes |
| Secret Manager + CDN | $3 | |
| GEE / NVIDIA Earth-2 | $0-5 | research tier |
| **TOTAL** | **solo almacenamiento mientras el articulo corre en CPU** | |

| Training único | Costo |
|----------------|-------|
| GCP L4 spot ~50h | $14 |
| GCP storage 200 GB×3m | $12 |
| **Total training** | **$262-602** |

## Auditoría Mensual

```bash
# scripts/cost_audit.sh
#!/usr/bin/env bash

echo "=== GCP costs last 30 days ==="
gcloud billing accounts list
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID

# Reporte por servicio
bq query --use_legacy_sql=false "
  SELECT service.description, ROUND(SUM(cost), 2) AS total_usd
  FROM \`agrosat-prod.billing.gcp_billing_export_v1_XXX\`
  WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY service.description ORDER BY total_usd DESC
"

  --output table

  --query "[?contains(name, 'Spot')]"
```

## Budget Alerts

```hcl
# Terraform GCP budget
resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account
  display_name    = "AgroSatCopilot Monthly $200"
  amount {
    specified_amount {
      currency_code = "USD"
      units         = "200"
    }
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }
  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.email.id]
  }
}
```

## Comandos Periódicos

```bash
make cost-audit               # gcloud + az: costos últimos 30 días
make scale-to-zero-check      # verifica que Cloud Run min=0
```

## Checklist Mensual

- [ ] Cost audit ejecutado
- [ ] Cloud Run min_instances=0 verificado
- [ ] Cloud SQL no over-provisioned
- [ ] DVC remote sin archivos huérfanos (>30 días sin referencia)
- [ ] MLflow artifact store con cleanup runs >90 días
- [ ] Budget alerts activos
