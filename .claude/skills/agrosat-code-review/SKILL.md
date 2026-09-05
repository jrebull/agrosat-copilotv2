---
name: agrosat-code-review
description: Review pull requests with checklists by epica, security gates, and AgroSatCopilot quality standards. Use when reviewing PRs before merging to develop or main.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Code Review Skill

## Checklist Universal

- [ ] Branch name `feature/E{epic}-US-XXX-{slug}`
- [ ] Conventional Commit `feat(EX): ...`
- [ ] Sin emojis en código, commits ni logs
- [ ] Idioma: todo el código en inglés (comentarios y docstrings); texto del lector en notebooks y UI en es/it/en según corresponda
- [ ] Sin secretos hardcodeados (`make secrets-scan` con gitleaks)
- [ ] `make check` limpio (lint + secrets + i18n)
- [ ] Si tocó notebook: ejecutado end-to-end con papermill, commit con outputs poblados
- [ ] Tests con cobertura mínima
- [ ] Sin archivos binarios (>1 MB) en Git

## Checklist Backend

- [ ] `Depends(get_current_user)` en endpoints protegidos
- [ ] `_check_session_owner` en endpoints multi-tenant
- [ ] Pydantic validation antes de service
- [ ] Inferencia >2s delegada a Pub/Sub
- [ ] SSE para streaming chat
- [ ] structlog (no print)
- [ ] Errores tipados
- [ ] Migración dbmate si tocó schema

## Checklist Frontend

- [ ] i18n: keys en it/es/en simultáneo
- [ ] `<script setup lang="ts">` con types
- [ ] Pinia para estado compartido
- [ ] SSR-safe
- [ ] A11y básica (focus, aria, alt)
- [ ] Dark mode soportado

## Checklist ML

- [ ] Spatial CV en datos espaciales
- [ ] MLflow run con tags
- [ ] DVC para data nueva
- [ ] VRAM validada si tocó H100
- [ ] Notebooks tocados ejecutados end-to-end (papermill) y commiteados con outputs poblados
- [ ] Atribución de licencia

## Checklist Agente

- [ ] Tools con FunctionTool + Pydantic
- [ ] Citaciones en final_answer
- [ ] Latencias dentro de objetivo
- [ ] Tests de tools con mocks

## Checklist Infra

- [ ] terraform plan revisado
- [ ] Workspace correcto
- [ ] IAM least privilege
- [ ] Scale-to-zero
- [ ] Secret Manager

## Decision Tree

```
PR toca código → checklist universal + dominio
PR toca schema → checklist universal + backend + db
PR toca infra → checklist universal + infra + finops
PR toca ML training → checklist universal + ml + h100 si aplica
PR toca agente → checklist universal + backend + agente
PR toca frontend → checklist universal + frontend + a11y
```
