---
name: agrosat-security-audit
description: Run OWASP Top 10 (code-level), CIS GCP + CIS Azure benchmarks (Prowler-style checklist), pre-deploy security checklist, secret scanning, dependency CVE audits, and cross-session isolation tests for AgroSatCopilot. Use before each deploy to staging/prod and on a weekly schedule. Posture checks are documented (run by humans during reviews) — not yet automated against the live cloud.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Security Audit Skill

Esta skill cubre **tres capas** de auditoría que se ejecutan secuencialmente antes de cada deploy a `staging` o `prod`:

1. **Código** — OWASP Top 10, gitleaks, pip-audit / pnpm audit, ruff `--select=S`.
2. **Posture cloud** — checklist CIS GCP + CIS Azure (inspirado en Prowler, ejecutado manualmente por ahora; ver §"Roadmap" para automatización opcional).
3. **Integración** — RLS por `session_id`, cross-session isolation tests, deploy gate.

Complementa (no reemplaza) `agrosat-security` (implementación de controles).

## OWASP Top 10 (2021) Checklist

| ID | Categoría | Verificación |
|----|-----------|--------------|
| A01 | Broken Access Control | RLS por session_id + `_check_session_owner` en cada endpoint |
| A02 | Cryptographic Failures | TLS 1.3 en Cloud Run, secretos en Secret Manager |
| A03 | Injection | Pydantic validators, no raw SQL string format |
| A04 | Insecure Design | Rate limit, audit logging, principle of least privilege |
| A05 | Security Misconfiguration | CSP, security headers, no default creds |
| A06 | Vulnerable Components | `pip-audit`, `pnpm audit`, Dependabot |
| A07 | Auth Failures | Clerk OAuth, JWT validation server-side, MFA opcional |
| A08 | Data Integrity Failures | Signed COG uploads, hash verification de checkpoints |
| A09 | Logging Failures | Structured logging, audit log de mutaciones |
| A10 | SSRF | Validate AOI GeoJSON, allowlist de URIs externas |

## CIS GCP Posture Checklist (Prowler-inspired)

Cada ítem se revisa manualmente con `gcloud` antes del deploy. Severidades alineadas a Prowler 5.x. Findings `critical` o `high` **bloquean** el deploy.

| ID | Severidad | Check | Verificación rápida |
|----|-----------|-------|---------------------|
| iam_no_primitive_roles | critical | Sin `roles/owner` ni `roles/editor` en SAs de runtime | `gcloud projects get-iam-policy $PROJECT --flatten=bindings --filter="bindings.role:(roles/owner OR roles/editor)"` |
| iam_sa_no_user_managed_keys | high | Sin keys JSON de SA en repo ni laptops | `gcloud iam service-accounts keys list --iam-account=...` (debe estar vacío salvo `system_managed`) |
| iam_workload_identity_federation | high | CI usa WIF, no JSON estático | revisar `.github/workflows/*.yml` |
| cloudsql_iam_auth | high | Cloud SQL con IAM auth + `cloudsql.iam_authentication=on` | `gcloud sql instances describe $INSTANCE` |
| cloudsql_backups_pitr | high | Backups automáticos + PITR habilitado | idem |
| cloudsql_public_ip | critical | Sin IP pública; conexión vía VPC connector | idem |
| gcs_uniform_bucket_level_access | high | Buckets con UBLA + versioning | `gsutil ubla get gs://$BUCKET` |
| gcs_public_access_prevention | critical | `publicAccessPrevention=enforced` | `gsutil pap get gs://$BUCKET` |
| run_no_unauthenticated | high | Cloud Run privado salvo frontend público | `gcloud run services describe ... --format=json` revisar `iam` |
| run_min_tls12 | medium | TLS mínimo 1.2 | revisar config Cloud Run |
| secretmanager_rotation | medium | Rotación ≤ 90 días en secretos críticos | `gcloud secrets describe ...` |
| logging_admin_activity | high | Audit logs Admin Activity activos a nivel proyecto | `gcloud logging sinks list` |
| network_no_default_vpc | medium | VPC custom; default VPC eliminada | `gcloud compute networks list` |
| compute_no_legacy_metadata | high | `enable-oslogin=TRUE` + sin metadata v1 legacy | revisar VMs |

**Score objetivo**: ≥ 80% pass antes de `deploy-prod`.

## CIS Azure Posture Checklist (subscripción H100)

| ID | Severidad | Check |
|----|-----------|-------|
| rbac_no_owner_on_runtime | critical | SP de Prowler/CI con `Reader` + `Security Reader` solo |
| keyvault_soft_delete | high | Soft delete + purge protection activos |
| storage_secure_transfer | high | `Secure transfer required` ON |
| storage_public_access_disabled | critical | Blobs sin acceso anónimo |
| vm_h100_managed_identity | medium | VM H100 usa managed identity para Blob; sin keys en disco |
| network_nsg_restrict_ssh | high | NSG limita SSH a IP del equipo |
| monitor_diagnostic_settings | medium | Diagnostic settings exportando a Log Analytics |

## Pre-Deploy Script

```bash
# scripts/security_audit.sh
#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Secret scanning ==="
gitleaks detect --no-banner

echo "=== 2. Backend deps ==="
cd backend && poetry run pip-audit --strict

echo "=== 3. Frontend deps ==="
cd ../frontend && pnpm audit --audit-level=high

echo "=== 4. Lint security plugin ==="
cd ../backend && poetry run ruff check --select=S

echo "=== 5. SQL injection grep ==="
if grep -rE "f\"SELECT|f'SELECT" backend/app/; then
  echo "FAIL: f-string SQL found"; exit 1
fi

echo "=== 6. Hardcoded secret grep ==="
if grep -rE "(api[_-]?key|password|secret).*=.*['\"][a-zA-Z0-9]{20,}" backend/app/ frontend/; then
  echo "FAIL: possible hardcoded secret"; exit 1
fi

echo "=== 7. Cross-session isolation tests ==="
cd ../backend && poetry run pytest tests/integration/test_session_isolation.py -v

echo "=== 8. CIS posture checklist (manual) ==="
echo "Run docs/security/posture_checklist.md sign-off before continuing."
```

## Cross-Session Isolation Test

```python
# backend/tests/integration/test_session_isolation.py
async def test_user_cannot_access_other_session(client, session_a_token, session_b_id):
    """User A no debe poder leer AOIs de session B."""
    headers = {"Authorization": f"Bearer {session_a_token}"}
    r = await client.get(f"/api/v1/aois?session_id={session_b_id}", headers=headers)
    assert r.status_code == 403


async def test_rls_blocks_direct_query(db_session, session_b_id):
    """SET LOCAL app.current_session_id=A debe bloquear queries a session B."""
    await db_session.execute(text("SET LOCAL app.current_session_id = 'A'"))
    result = await db_session.execute(
        select(AOI).where(AOI.session_id == session_b_id)
    )
    assert result.scalars().all() == []
```

## Finding Format (OCSF-style, lightweight)

Cuando una revisión humana levanta un finding, se documenta en `docs/security/findings/${YYYY-MM-DD}-${slug}.md` con este shape:

```yaml
check_id: cloudsql_public_ip
severity: critical          # critical | high | medium | low | info
status: FAIL                # FAIL | PASS | MANUAL
resource: agrosat-pg-staging
title: Cloud SQL exposes public IP
finding: "Instance has assignedIpAddresses.type=PRIMARY,ipAddress=34.x.x.x"
remediation: "Disable public IP; use VPC connector + private IP. See infra/terraform/modules/cloudsql/main.tf."
owner: arthur
due: 2026-05-15
```

`scripts/security_audit_gate.py` agrega cualquier finding con `severity in {critical,high} and status=FAIL` no resuelto y bloquea `deploy-prod` con exit code 1.

## Roadmap — Automatización opcional

El proyecto **no usa Prowler como dependencia** hoy (curso, presupuesto, complejidad). Si se quisiera automatizar el checklist CIS contra la cuenta real, el path estándar sería ejecutar `prowler gcp` y `prowler azure` con SA / SP **read-only** desde un workflow programado y mapear su salida OCSF al gate de arriba — pero esa adopción debería discutirse con el equipo antes y agregarse como una skill aparte (`agrosat-prowler-audit`). Mientras tanto:

- Cubrir el checklist CIS arriba manualmente cada sprint.
- Tomar prestadas las **categorías de checks** y la **escala de severidad** de Prowler para que un swap futuro sea barato.
- Revisar `infrastructure/terraform/` con `tfsec` o `checkov` (zero-cost, ya local) para cubrir IaC.

## Pre-Deploy Checklist

- [ ] `gitleaks` sin findings
- [ ] `pip-audit` / `pnpm audit` sin Critical/High
- [ ] `ruff --select=S` limpio
- [ ] Sin f-string SQL ni secretos hardcodeados
- [ ] Cross-session isolation tests pasan
- [ ] CSP headers + rate limit verificados en staging
- [ ] CIS GCP posture checklist firmado para este commit
- [ ] CIS Azure posture checklist firmado (si se modificó infra H100)
- [ ] `tfsec` / `checkov` sobre `infrastructure/terraform/`
- [ ] Workload Identity Federation en CI (no JSON SA en GitHub Secrets)
- [ ] Findings `critical/high` sin resolver → 0
