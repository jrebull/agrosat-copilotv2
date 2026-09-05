---
name: agrosat-engram-memory
description: Configure Engram (Gentleman-Programming/engram) as DEV-TIME persistent memory for Claude Code on the AgroSatCopilot project. Local-only Go binary + SQLite + FTS5, cloud is opt-in and disabled here. No integration with runtime ADK agent. Use to persist decisions, sprint context, gotchas and US closures across coding sessions. Never store secrets, credentials, real session_ids or production data.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Engram Memory Skill (dev tooling)

## Scope and Hard Boundary

Engram is a **developer productivity** tool that persists memory across Claude Code sessions. It is **NOT** part of the production runtime.

| Layer | Memory store | Owner |
|-------|--------------|-------|
| Runtime agent (`/chat` SSE, ADK) | PostgreSQL `chat_sessions` + pgvector `rag_documents` with **RLS per `session_id`** | `agrosat-google-adk-agent` + `agrosat-spatial-rag` |
| Spatial-RAG retrieval | pgvector HNSW + PostGIS ST_DWithin | `agrosat-spatial-rag` |
| **Dev-time IDE memory** | **Engram (local SQLite, FTS5)** | **this skill** |

Engram **MUST NOT** be wired into FastAPI routers, ADK tools, or any production code path. If a future need for cross-session product memory appears, it goes through Postgres+pgvector under RLS, never Engram.

## Rules — NON-NEGOTIABLE

- **Local install only.** Single Go binary (`engram.exe` on Windows). DB at `%USERPROFILE%\.engram\engram.db` (override via `ENGRAM_DATA_DIR`). No external dependencies.
- **Cloud OFF.** Cloud is an opt-in `engram cloud enroll <project>` flow — we **do not enroll** the AgroSatCopilot project. Team synchronization is disabled while ADR-015 remains proposed.
- **Project scoping is automatic.** Engram detects the current project from the cwd of the MCP subprocess; if ambiguous, it returns a recovery token. Use `mem_current_project` MCP tool at session start to confirm the project context is `agrosat-copilotv2`.
- **Never persist**: real `session_id` UUIDs, JWTs, Clerk tokens, HF tokens, GEE service-account JSON, GCP/Azure secret values, customer parcel boundaries, real user emails. Engram entries are **internal team knowledge**, not data.
- **No native chunks in Git.** A future reviewed export format may live under `docs/engram/` only after ADR-015 is accepted. The raw `~/.engram/engram.db` SQLite file is git-ignored and never committed.
- **Per-laptop only**: do not check the SQLite DB into the repo, ever (already in `.gitignore`).

## Install on Windows (current method, v1.15.x — May 2026)

The `install.sh` script no longer exists. **Recommended for this project** (used during bootstrap on 2026-05-11): Go install + Claude Code plugin marketplace.

### Step 1 — install the binary

```powershell
# Recommended: Go install (puts engram.exe in %USERPROFILE%\go\bin, must be on PATH)
go install github.com/Gentleman-Programming/engram/cmd/engram@latest
engram --version
```

Alternatives:

```powershell
# Pre-built zip release (no Go toolchain needed)
$ver = (gh release view --repo Gentleman-Programming/engram --json tagName -q .tagName).TrimStart("v")
Invoke-WebRequest -Uri "https://github.com/Gentleman-Programming/engram/releases/download/v$ver/engram_${ver}_windows_amd64.zip" -OutFile "$env:TEMP\engram.zip"
Expand-Archive -Force "$env:TEMP\engram.zip" -DestinationPath "$env:USERPROFILE\.local\bin"
```

```bash
# macOS / Linux (Homebrew)
brew install gentleman-programming/tap/engram
```

### Step 2 — register with Claude Code via plugin marketplace

```powershell
claude plugin marketplace add Gentleman-Programming/engram
claude plugin install engram
```

The plugin handles MCP server registration **and** hooks automatically. After install, `claude mcp list` should show:

```
plugin:engram:engram: engram mcp --tools=agent - Connected
```

### Step 3 — restart Claude Code

Updating the `engram` binary on disk does not replace an already-running stdio MCP subprocess. Restart Claude Code after install or after any `engram` binary upgrade.

## Project allowlist (`.claude/settings.json`)

The plugin exposes its MCP tools under the namespace `mcp__plugin_engram_engram__*`. Allow only the non-destructive operations we use:

```jsonc
{
  "enabledPlugins": {},
  "permissions": {
    "allow": [
      "mcp__plugin_engram_engram__mem_save",
      "mcp__plugin_engram_engram__mem_update",
      "mcp__plugin_engram_engram__mem_search",
      "mcp__plugin_engram_engram__mem_context",
      "mcp__plugin_engram_engram__mem_timeline",
      "mcp__plugin_engram_engram__mem_get_observation",
      "mcp__plugin_engram_engram__mem_session_start",
      "mcp__plugin_engram_engram__mem_session_end",
      "mcp__plugin_engram_engram__mem_session_summary",
      "mcp__plugin_engram_engram__mem_current_project",
      "mcp__plugin_engram_engram__mem_stats",
      "mcp__plugin_engram_engram__mem_doctor",
      "mcp__plugin_engram_engram__mem_suggest_topic_key",
      "mcp__plugin_engram_engram__mem_save_prompt",
      "mcp__plugin_engram_engram__mem_capture_passive"
    ]
  }
}
```

Do **not** add a manual `mcpServers.engram` entry in `.claude/settings.json` when the plugin is installed — it duplicates the MCP subprocess and the manual entry exposes tools under a different namespace (`mcp__engram__*`) that won't match this allowlist. Pick one path (plugin marketplace is the official one) and stick to it.

The allowlist alone does **not** install or enable the plugin. On 2026-09-04 this repository has
`enabledPlugins: {}`, and `engram` is not on the current macOS `PATH`; therefore no project memory is
active in that environment. Agents must verify the binary and MCP connection before claiming that a
memory was read or saved.

We intentionally **do not allow** `mem_delete` or `mem_merge_projects` — destructive operations require explicit per-call approval.

Cloud stays OFF unless someone runs `engram cloud enroll agrosat-copilotv2`. Do not enroll without team agreement.

## MCP Tools available (19 total, May 2026)

| Category | Tools |
|----------|-------|
| Save & Update | `mem_save`, `mem_update`, `mem_delete`, `mem_suggest_topic_key` |
| Search & Retrieve | `mem_search`, `mem_context`, `mem_timeline`, `mem_get_observation` |
| Session Lifecycle | `mem_session_start`, `mem_session_end`, `mem_session_summary` |
| Conflict Surfacing | `mem_judge`, `mem_compare` |
| Utilities | `mem_save_prompt`, `mem_stats`, `mem_capture_passive`, `mem_merge_projects`, `mem_current_project`, `mem_doctor` |

We **do not** allow `mem_delete` or `mem_merge_projects` by default — destructive operations require explicit per-call approval.

## What to Save (and not save)

### SAVE — internal knowledge that decays slowly

- US closure rationale: *"TSViT trained on V1 window with batch 4 grad-accum 8 = effective 32, BF16, peak 78 GB VRAM"*.
- Trade-off decisions and **why** (paired with the ADR if one exists).
- Cross-laptop gotchas: *"GEE export to GCS asia-northeast1 fails silently; use us-central1 mirror"*.
- Sprint retro insights: *"Sprint 4 underestimated SegFormer-B2 by 2 SP — base future SP on actuals"*.
- Pointers to canonical docs: *"For Avance 3 rubric see docs/general/Rubricas Integrador.html section 4"*.

### DO NOT SAVE — runtime / sensitive / volatile

- Anything from `.env*`, Secret Manager, Key Vault.
- Real user session_ids, Clerk JWTs, OAuth refresh tokens.
- COG/parcel boundaries that are user data.
- Anything already in code or git history (memory is for **why**, not **what**).
- Anything that changes weekly (sprint board state — use Linear/GitHub).

## Suggested `mem_save` Pattern (via MCP)

When closing a US the agent should call `mem_save` with this shape:

```json
{
  "title": "Gemma 4 26B-MoE LoRA fits 1xH100 NVL 96GB",
  "type": "architecture",
  "what": "Gemma 4 26B-MoE LoRA rank 16 BF16 + FlashAttention-2 + grad checkpointing fits 1xH100 NVL 96GB at effective batch 16 (b=2 x ga=8).",
  "why": "V3 budget is 24h; this config peaks at 82 GB VRAM with 78%/22% util, leaving headroom for serving warm-up.",
  "where": "ml/train/train_gemma4_lora.py + scripts/azure_h100_train.sh",
  "learned": "Higher batch size triggers OOM during eval pass; keep eval batch at b=1."
}
```

CLI invocation (for ad-hoc terminal use, not through Claude Code):

```bash
engram save \
  "Gemma 4 26B-MoE LoRA fits 1xH100 NVL 96GB" \
  "Gemma 4 26B-MoE LoRA rank 16 BF16 + FlashAttention-2 + grad checkpointing fits 1xH100 NVL 96GB at effective batch 16 (b=2 x ga=8). Peak VRAM 82 GB. 24h fits in V3 window." \
  --type architecture --project agrosat-copilotv2
```

## Search Pattern

From inside Claude Code, prefer the MCP `mem_search` / `mem_context` tools (results land in the model context). From the terminal:

```bash
engram search "h100 vram gemma" --project agrosat-copilotv2 --limit 5
```

## Team Sync (disabled until a reviewable export policy exists)

The upstream `engram sync` command writes `.engram/manifest.json` and compressed chunks under
`.engram/chunks/`. It can export sessions, prompts and every observation in the selected project;
upstream documents that `scope: personal` is not excluded automatically from project sync.

For this repository, `.engram/` is deliberately ignored and must **not** be force-added. There is no
`make memory-sync` or `make memory-import` target. Until the team approves a redaction workflow and a
gate that inspects the actual chunk payload, share durable knowledge through reviewed ADRs,
`docs/us-resolved/` and other ordinary Markdown sources. Cloud enrollment remains prohibited without
an explicit team decision.

## `.gitignore` Additions

```gitignore
# Engram — local only
.engram/
*.engram.db
*.engram.db-wal
*.engram.db-shm
```

(Already pre-emptively included in the project `.gitignore`.)

## Verification Checklist

- [ ] `engram --version` works on each developer laptop (>= v1.15.10)
- [ ] The Engram plugin is enabled and `claude mcp list` reports its process as connected
- [ ] `engram cloud status` shows **not enrolled** for `agrosat-copilotv2`
- [ ] `mem_current_project` returns `agrosat-copilotv2` from inside Claude Code
- [ ] SQLite DB outside the repo (`%USERPROFILE%\.engram\engram.db`)
- [ ] No `mem_save` content contains `sk-`, `Bearer`, `clerk_`, real UUIDs, or paths to creds
- [ ] No `.engram/` chunk is tracked; reviewed shared decisions live in ordinary project docs
- [ ] `.gitignore` excludes Engram local state
- [ ] No FastAPI / ADK / TiTiler code imports or shells out to `engram`

## When NOT to Use This Skill

- Implementing user-facing chat memory → that's `agrosat-google-adk-agent` (ADK session memory) + `agrosat-spatial-rag` (Postgres+pgvector under RLS).
- Storing model checkpoints or experiment metadata → `agrosat-dvc-mlflow`.
- Storing dataset versions → DVC remote.
- Sprint/issue tracking → Linear (`SEC`, `ML`, `MLOPS` teams).
