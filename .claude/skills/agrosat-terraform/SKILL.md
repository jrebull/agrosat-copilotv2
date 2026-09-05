---
name: agrosat-terraform
description: Define infrastructure as code with Terraform 1.9+ for AgroSatCopilot — GCP primary (Cloud Run, Cloud SQL PostGIS+pgvector, GCS, Pub/Sub, Vertex AI, Artifact Registry, Secret Manager) + Azure H100 secondary. Use when creating modules, workspaces, or backend state.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Terraform Skill

## Rules — NON-NEGOTIABLE

- Workspaces separados: `dev`, `staging`, `prod`
- Backend state en `gs://agrosat-tfstate` con versionado activado
- Variables sensibles vía `.tfvars` (gitignored) o Secret Manager
- Plan antes de apply, revisión humana obligatoria
- Cloud Run `min_instances=0` (scale-to-zero)
- IAM principle of least privilege

## Estructura

```
infrastructure/terraform/
├── modules/
│   ├── gcp/
│   │   ├── cloud_run/        # services: api, frontend, tiling, inference-worker
│   │   ├── cloud_sql/        # PostgreSQL 15 + PostGIS + pgvector
│   │   ├── pubsub/           # inference-jobs, inference-results
│   │   ├── gcs/              # data, artifacts, dvc-remote, tfstate
│   │   ├── secret_manager/   # 6 secrets base
│   │   ├── vertex_ai/        # Agent Engine binding
│   │   ├── artifact_registry/
│   │   └── iam/
│   └── azure/
│       ├── h100_vm/          # Standard_NC40ads_H100_v5 spot + on-demand
│       ├── blob_storage/
│       ├── vnet/
│       └── key_vault/
└── environments/
    ├── dev/    {main.tf, variables.tf, backend.tf, terraform.tfvars(gitignored)}
    ├── staging/
    └── prod/
```

## Cloud Run Module

```hcl
# modules/gcp/cloud_run/main.tf
resource "google_cloud_run_v2_service" "this" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0  # scale-to-zero
      max_instance_count = var.max_instances
    }
    containers {
      image = var.image
      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }
      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = var.secret_env_vars
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }
    }
    service_account = var.service_account_email
  }
}
```

## Cloud SQL Module

```hcl
# modules/gcp/cloud_sql/main.tf
resource "google_sql_database_instance" "postgres" {
  name             = "agrosat-${var.env}"
  database_version = "POSTGRES_15"
  region           = var.region
  settings {
    tier              = var.tier  # db-f1-micro dev, db-custom-2-7680 prod
    availability_type = var.env == "prod" ? "REGIONAL" : "ZONAL"
    backup_configuration {
      enabled            = true
      point_in_time_recovery_enabled = var.env == "prod"
      backup_retention_settings { retained_backups = 7 }
    }
    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
    database_flags {
      name  = "shared_preload_libraries"
      value = "vector,pg_stat_statements"
    }
  }
  deletion_protection = var.env == "prod"
}
```

## Backend State

```hcl
# environments/dev/backend.tf
terraform {
  required_version = ">= 1.9"
  backend "gcs" {
    bucket = "agrosat-tfstate"
    prefix = "dev"
  }
}
```

## Azure H100 Module

```hcl
# modules/azure/h100_vm/main.tf
resource "azurerm_linux_virtual_machine" "h100" {
  name                = "agrosat-h100-${var.env}"
  resource_group_name = var.resource_group
  location            = var.location
  size                = "Standard_NC40ads_H100_v5"
  priority            = "Spot"  # 4× cheaper
  eviction_policy     = "Deallocate"
  max_bid_price       = -1  # current spot price

  admin_username                   = "agrosat"
  disable_password_authentication  = true

  admin_ssh_key {
    username   = "agrosat"
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 256
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-2204"
    sku       = "2204-gen2"
    version   = "latest"
  }

  network_interface_ids = [azurerm_network_interface.this.id]
}

# Auto-shutdown
resource "azurerm_dev_test_global_vm_shutdown_schedule" "auto_off" {
  virtual_machine_id    = azurerm_linux_virtual_machine.h100.id
  location              = var.location
  enabled               = true
  daily_recurrence_time = "2300"
  timezone              = "Europe/Rome"
  notification_settings { enabled = false }
}
```

## Comandos

```bash
make tf-init env=dev
make tf-plan env=dev
make tf-apply env=dev          # con confirmación
make tf-destroy env=dev        # solo dev
make tf-fmt                    # terraform fmt -recursive
make tf-validate
```

## QA Checklist

- [ ] Workspace correcto
- [ ] Backend state versionado
- [ ] Secret Manager para credenciales
- [ ] min_instances=0 en Cloud Run
- [ ] PITR en Cloud SQL prod
- [ ] Auto-shutdown VM H100
- [ ] IAM principle of least privilege
- [ ] `terraform validate` y `tflint` pass
