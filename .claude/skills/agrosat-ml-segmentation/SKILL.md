---
name: agrosat-ml-segmentation
description: Implement and train 6 segmentation architectures for AgroSatCopilot crop mapping — U-Net, DeepLabv3+, SegFormer-B2, U-TAE, TSViT, Swin-UNETR. Use when implementing dense semantic segmentation on Sentinel-2 patches with PASTIS-R labels.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot ML Segmentation Skill

## Rules — NON-NEGOTIABLE

- 6 arquitecturas obligatorias por rúbrica Avance 4
- TSViT del paper del profesor (`michaeltrs/DeepSatModels`) es obligatorio
- Métricas: mIoU + F1-macro + pixel accuracy
- Patches Sentinel-2 con labels PASTIS-R
- Training en L4 dev / H100 final (ventana V2)
- MLflow run por modelo con tag `architecture`
- Optuna ajuste fino top-2 con ≥30 trials

## Las 6 Arquitecturas

| # | Modelo | Lib | Tipo |
|---|--------|-----|------|
| 1 | U-Net ResNet-50 | `segmentation_models.pytorch` | CNN clásica |
| 2 | DeepLabv3+ MobileNetV3 | `segmentation_models.pytorch` | CNN eficiente ASPP |
| 3 | SegFormer-B2 | `transformers` | Transformer spatial |
| 4 | U-TAE | GitHub `VSainteuf/utae-paps` | Temporal Attention |
| 5 | TSViT | GitHub `michaeltrs/DeepSatModels` | Paper 1 profesor (obligatorio) |
| 6 | Swin-UNETR | `monai` | Transformer moderno |

## U-Net Implementation

```python
import segmentation_models_pytorch as smp
import torch.nn as nn

def build_unet(num_classes: int):
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=10,  # B02..B12
        classes=num_classes,
        activation=None,
    )
```

## SegFormer Implementation

```python
from transformers import SegformerForSemanticSegmentation, SegformerConfig

def build_segformer(num_classes: int):
    config = SegformerConfig(
        num_labels=num_classes,
        num_channels=10,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
    )
    return SegformerForSemanticSegmentation(config)
```

## TSViT Wrapper

```python
# ml/models/tsvit_wrapper.py
from DeepSatModels.models.TSViT.TSViTcls import TSViT

def build_tsvit(num_classes: int, t_dim: int, h: int, w: int):
    return TSViT(
        model_config={
            "img_res": h, "patch_size": 4, "patch_size_time": 1,
            "num_channels": 10, "num_classes": num_classes,
            "max_seq_len": t_dim,
            "dim": 128, "temporal_depth": 6, "spatial_depth": 2,
            "heads": 4, "pool": "cls", "dropout": 0.0, "scale_dim": 4,
        }
    )
```

## Training Loop

```python
import torch
import mlflow
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, F1Score
from ml.utils.mlflow_utils import track_experiment

@track_experiment("alt-models-segmentation")
def train_model(model_name: str, build_fn, train_loader, val_loader, num_classes, epochs=50):
    device = torch.device("cuda")
    model = build_fn(num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    miou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    mlflow.set_tag("architecture", model_name)
    mlflow.log_params({"epochs": epochs, "lr": 1e-4, "model": model_name})

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                miou.update(pred, y); f1.update(pred, y)
        mlflow.log_metric("val_miou", miou.compute().item(), step=epoch)
        mlflow.log_metric("val_f1_macro", f1.compute().item(), step=epoch)
        miou.reset(); f1.reset()
    return model
```

## Optuna Ajuste Fino top-2

```python
import optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # ... train + return val_miou

study = optuna.create_study(
    direction="maximize",
    storage="postgresql://postgres@localhost/optuna",
    study_name=f"tune-{model_name}",
)
study.optimize(objective, n_trials=30)
```

## QA Checklist

- [ ] 6 arquitecturas implementadas
- [ ] mIoU + F1-macro + pixel accuracy reportados
- [ ] Optuna ≥30 trials sobre top-2
- [ ] MLflow tag `architecture`
- [ ] Tabla comparativa exportada
- [ ] Selección modelo individual final justificada
