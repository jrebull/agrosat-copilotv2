---
name: agrosat-finops
description: Audit and optimize cloud costs for AgroSatCopilot (GCP + Azure H100). Target ~$115 USD/month operational + ~$262-602 USD one-time training. Use for budget alerts, scale-to-zero verification, spot price monitoring, and cost reports.
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
| Azure H100 spot operativo | $30 | ~3 h/día activo |
| Secret Manager + CDN | $3 | |
| GEE / NVIDIA Earth-2 | $0-5 | research tier |
| **TOTAL** | **~$115** | |

| Training único | Costo |
|----------------|-------|
| Azure H100 spot 80h | $220 |
| Azure H100 on-demand fallback | $560 |
| GCP L4 spot ~50h | $14 |
| GCP storage 200 GB×3m | $12 |
| Azure Blob checkpoints 150 GB×3m | $9 |
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

echo "=== Azure costs last 30 days ==="
az consumption usage list --top 100 --query "[].{date:usageStart, service:meterCategory, cost:pretaxCost}" \
  --output table

# Spot price check H100
az vm list-skus --location swedencentral --size Standard_NC40ads_H100_v5 \
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
make azure-h100-status        # verifica si VM está allocated
make azure-h100-stop          # si está accidentalmente arriba
```

## Checklist Mensual

- [ ] Cost audit ejecutado
- [ ] Cloud Run min_instances=0 verificado
- [ ] Cloud SQL no over-provisioned
- [ ] Azure H100 deallocated cuando no se usa
- [ ] DVC remote sin archivos huérfanos (>30 días sin referencia)
- [ ] MLflow artifact store con cleanup runs >90 días
- [ ] Budget alerts activos
