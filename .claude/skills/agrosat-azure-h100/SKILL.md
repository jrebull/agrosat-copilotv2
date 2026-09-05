---
name: agrosat-azure-h100
description: Manage Azure VM Standard_NC40ads_H100_v5 (1×H100 NVL 96GB) for AgroSatCopilot intensive training and serving workloads. Use when starting/stopping the H100 VM, configuring spot instance, scripts for ventana windows V1-V6, Azure Blob checkpoints.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Azure H100 Skill

## Rules — NON-NEGOTIABLE

- VM Standard_NC40ads_H100_v5 spot (4× más barato que on-demand)
- Auto-shutdown timer 12 h por defecto
- Checkpoint cada 30 min a Azure Blob `az://agrosat-checkpoints/`
- VNet privada, NSG SSH solo desde IPs de los 3 devs
- 6 ventanas planificadas: V1 (8h baselines), V2 (12h temporal), V3 (24h Gemma 4), V4 (12h Qwen3-VL), V5 (16h Qwen3.5 serving + LoRA), V6 (8h demo warm)
- Total presupuesto: ~80 h spot ($220) o ~$560 on-demand

## VM Config

| Atributo | Valor |
|----------|-------|
| SKU | Standard_NC40ads_H100_v5 |
| GPU | 1× H100 NVL 96 GB |
| CPU | 40 vCPU (AMD EPYC) |
| RAM | 320 GB |
| Disk | 1 TB Premium SSD |
| Image | Microsoft DSVM Ubuntu 22.04 |
| Priority | Spot (eviction policy: Deallocate) |
| Region | swedencentral o westeurope (donde haya cuota) |

## Script Start

```bash
#!/usr/bin/env bash
# scripts/azure_h100_start.sh
set -euo pipefail

RG="agrosat-rg"
VM="agrosat-h100-prod"

echo "Starting Azure H100 spot VM..."
az vm start --resource-group $RG --name $VM

# Configurar auto-shutdown 12h
SHUTDOWN_TIME=$(date -u -d '+12 hours' +'%Y-%m-%dT%H:%M:%SZ')
az vm auto-shutdown -g $RG -n $VM --time $SHUTDOWN_TIME

IP=$(az vm show -d -g $RG -n $VM --query publicIps -o tsv)
echo "H100 ready at $IP"
echo "Auto-shutdown set for $SHUTDOWN_TIME (12 h)"

# Smoke test GPU
ssh agrosat@$IP "nvidia-smi" || echo "SSH not ready yet, retry in 30s"
```

## Script Stop

```bash
#!/usr/bin/env bash
# scripts/azure_h100_stop.sh
set -euo pipefail
RG="agrosat-rg"; VM="agrosat-h100-prod"

az vm deallocate --resource-group $RG --name $VM
echo "VM deallocated. Spot price no longer accumulating."
```

## Status

```bash
# scripts/azure_h100_status.sh
az vm show -g agrosat-rg -n agrosat-h100-prod --show-details --query \
  "{PowerState:powerState, IP:publicIps, AutoShutdown:tags.autoShutdown}"
```

## Azure Blob Checkpoint Pattern

```python
from azure.storage.blob import BlobServiceClient

blob_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
container = blob_client.get_container_client("agrosat-checkpoints")

def upload_checkpoint(local_path: str, remote_path: str):
    with open(local_path, "rb") as f:
        container.upload_blob(remote_path, f, overwrite=True)
```

## Ventanas Planificadas

| Ventana | Fecha | Duración | Uso | VRAM esperada |
|---------|-------|----------|-----|--------------|
| V1 | 18-20 may | 8 h | Baselines + preliminar | ~40 GB |
| V2 | 25-27 may | 12 h | U-TAE + TSViT + Swin-UNETR | ~60 GB |
| V3 | 28-30 may | 24 h | Gemma 4 26B LoRA | ~82 GB |
| V4 | 1-3 jun | 12 h | Qwen3-VL LoRA + ensambles | ~92 GB |
| V5 | 5-7 jun | 16 h | Qwen3.5 serving + LoRA tool traces | ~91 GB |
| V6 | 18-20 jun | 8 h | Warm vLLM para demo | ~91 GB |

## QA Checklist H100

- [ ] Spot price < $3/h al iniciar
- [ ] Auto-shutdown configurado
- [ ] SSH key del lead deployado
- [ ] NSG con solo IPs del equipo
- [ ] `nvidia-smi` reporta H100 96GB
- [ ] Azure Blob mounted o accesible
- [ ] Checkpoint testing antes del run real
- [ ] Log de uso en `docs/h100_log.md`
