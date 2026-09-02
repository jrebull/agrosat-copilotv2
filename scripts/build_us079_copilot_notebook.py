"""Builder of the US-079 copilot notebook (original PASTIS vs Italian transfer).

Generates ``notebooks/transfer/us079_copilot_original_vs_tl.ipynb`` reproducibly
(same idempotent ``nbformat`` + ``typer`` pattern as the sibling builders
``scripts/build_us078_eda_notebook.py`` / ``scripts/build_us079_eval_notebook.py``
and the other ``scripts/build_*_notebook.py``). The notebook closes the loop of
EPIC 12: it accesses the dense transfer models THROUGH the conversational agent
(``ml.agent``) -- the "Be My Eyes" copilot -- and contrasts the ORIGINAL champion
(trained on PASTIS / France) against the TRANSFER-LEARNING model (fine-tuned on
the Italian homologue of US-078), probing several LLM backends.

The "Be My Eyes" division of labour is the whole point:

- the PERCEIVER is the team's dense models, reached via the agent's ``classify``
  / ``compare`` tools (the deployment-winner Voting-3 for the TL, and the
  ``stacking-5`` / ``xgb-alphaearth`` members for the original via
  ``compare_models``). They turn satellite signals into structured TEXT (a crop
  class + confidence), they never talk to the user;
- the REASONER is a frontier/on-prem LLM that reasons over THAT TEXT. It never
  classifies pixels. Three backends are wired through ``make_backend``:
  ``gemini-3.5-flash`` (cloud, Vertex AI / GenAI), ``qwen3.6-vl`` (on-prem
  multimodal, llama.cpp ``:8003``) and ``qwen35`` (on-prem text, vLLM).

What the notebook shows:

1. Framing of the copilot + the Be My Eyes pattern + what is compared.
2. Setup: build the agent with EACH backend via ``make_backend``. A backend with
   no credentials / endpoint is marked NOT AVAILABLE honestly and skipped -- the
   notebook never fabricates an LLM answer.
3. Copilot demo on REAL Italian parcels (US-078 dataset): the agent classifies
   with the ``classify`` tool (Voting-3 for the TL) and ``compare_models`` (the
   original members), and the reasoner reasons over the tool TEXT. The real
   dialogue is rendered.
4. Original vs TL comparison: (a) the transfer delta (French champion zero-shot
   -> Italy vs the fine-tuned TL -> Italy, read from the runner ``report.json``)
   and (b) the domain parity (France F1 0.9069 vs the TL on Italy).
5. How much the copilot improves/performs with each LLM backend (a backend x
   metric table): the reasoner reasons over the perceiver's TEXT.

HARD RULE -- REAL VALUES ONLY. If the H100 training pipeline has not produced the
``report.json`` yet, the metric cells print an HONEST pending state; they NEVER
emit placeholder numbers. When re-run with the report present, the cells populate
with the real metrics. The LLM cells degrade honestly too: a backend without
credentials / a reachable endpoint is reported NOT AVAILABLE, never mocked.

Visible prose (markdown, captions, prints) is Spanish with accents; code,
identifiers, comments and docstrings stay in English ASCII (project convention).
No emojis.

Usage::

    poetry run python scripts/build_us079_copilot_notebook.py \\
        --out notebooks/transfer/us079_copilot_original_vs_tl.ipynb \\
        --report-dir checkpoints/transfer/voting-italia/us079 \\
        --data-dir data/pastis_italia_2018

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us079_copilot_original_vs_tl.ipynb")
_DEFAULT_REPORT = Path("checkpoints/transfer/voting-italia/us079")
_DEFAULT_DATA = Path("data/pastis_italia_2018")

#: France deployment champion F1-macro reference (Voting-3 over ``france-10``),
#: ratified in EPIC 6 and reused here as the original-domain parity baseline. It
#: is a fixed, already-measured constant (not a fabricated run output): the
#: notebook contrasts it against the TL number READ from the runner report.
_FRANCE_CHAMPION_F1: float = 0.9069


def _build_cells(report_dir: str, data_dir: str) -> list:
    """Build the markdown + code cells of the US-079 copilot notebook.

    Args:
        report_dir: Repo-relative path to the runner output (``report.json``),
            injected into the parameters cell so the comparison reads real
            metrics when they exist.
        data_dir: Repo-relative path to the Italian homologue dataset (US-078),
            from which real Italian parcels are surfaced for the copilot demo.

    Returns:
        The ordered list of ``nbformat`` cells.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ---------------------------------------------------------------- Cover ---
    cells.append(
        md(
            "# US-079 - El copiloto: modelo ORIGINAL (PASTIS) vs TRANSFER (Italia)\n\n"
            '### Patron "Be My Eyes": los modelos del equipo *ven*, un LLM *razona*\n\n'
            "**Equipo 17** - AgroSatCopilot - Transfer learning mediterraneo (EPIC 12)\n\n"
            "---\n\n"
            "Este cuaderno **accede a los modelos densos a traves del copiloto** (el agente "
            "`ml.agent`) y contrasta el modelo **ORIGINAL** -- el campeon entrenado sobre PASTIS "
            "(Francia) -- contra el modelo de **TRANSFER LEARNING** -- afinado sobre el homologo "
            "italiano de US-078. La idea central es el patron **Be My Eyes**:\n\n"
            "- El **perceiver** son los modelos del equipo. No hablan con el usuario: **miran una "
            "parcela y emiten una observacion en TEXTO** (clase de cultivo + confianza). Para el "
            "TL se invoca el **Voting-3** ponderado (ganador del despliegue) via la herramienta "
            "`classify_new_parcel`; para el original se contrastan los miembros "
            "`stacking-5` / `xgb-alphaearth` via `compare_models`.\n"
            "- El **reasoner** es un LLM frontera (Gemini en la nube) u on-prem (Qwen). **No "
            "clasifica pixeles**: lee ESE TEXTO del perceiver, llama herramientas geoespaciales y "
            "redacta la respuesta. Aqui probamos **tres backends** intercambiables por "
            "`make_backend`.\n\n"
            "Comparamos el original y el TL desde **dos angulos** (ambos pedidos):\n\n"
            "1. **Delta del transfer**: el campeon frances **zero-shot** sobre Italia frente al "
            "TL **afinado** sobre Italia (cuanto aporta afinar de verdad).\n"
            "2. **Paridad de dominios**: el campeon en **Francia** (F1-macro 0.9069) frente al "
            "**TL en Italia** (mismo objetivo de calidad en un dominio nuevo).\n\n"
            "> **Solo valores reales.** Las cifras de transfer/paridad se leen del `report.json` "
            "que produce `scripts/run_transfer_italia.py`. El entrenamiento (fine-tune + Voting-3 "
            "+ eval) corre en la H100; si el reporte aun no existe, las celdas lo dicen "
            "explicitamente y muestran el estado **pendiente**, nunca numeros inventados. Las "
            "respuestas del LLM usan credenciales reales si estan en `.env.local`; un backend sin "
            "credenciales o sin endpoint se marca **no disponible** y se omite, sin fabricar."
        )
    )

    # --------------------------------------------- parameters (papermill) ---
    cells.append(
        code(
            "# Parametros (papermill). Sobreescribe con `papermill -p <name> <value>`.\n"
            f'report_dir = "{report_dir}"   # salida del runner (report.json del TL + Voting-3)\n'
            f'data_dir = "{data_dir}"   # homologo italiano de US-078 (parcelas reales)\n'
            "demo_user = 'demo@agrosat.dev'   # propietario de la sesion de demostracion sembrada\n"
            "n_demo_parcels = 2   # parcelas italianas a pasar por el copiloto\n"
            "year = 2018   # campaña del embedding anual (Italia 2018)\n"
            "france_champion_f1 = "
            f"{_FRANCE_CHAMPION_F1}   # F1-macro del campeon en Francia (france-10, EPIC 6)\n"
            "# Backends a probar (nombre -> etiqueta legible). make_backend los resuelve.\n"
            "backend_models = {\n"
            "    'gemini-3.5-flash': 'Gemini 3.5 Flash (nube, Vertex AI / GenAI)',\n"
            "    'qwen3.6-vl': 'Qwen3.6-VL (on-prem multimodal, llama.cpp :8003)',\n"
            "    'qwen35': 'Qwen3.5-35B (on-prem texto, vLLM)',\n"
            "}"
        )
    )
    cells[-1].metadata = {"tags": ["parameters"]}

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        md(
            "## Preparacion del entorno\n\n"
            "Resolvemos la raiz del repositorio, forzamos UTF-8 en la salida (la consola de "
            "Windows usa cp1252 y la prosa/logs llevan acentos) y cargamos `.env.local` para tomar "
            "la cadena de conexion de la base y las credenciales de los LLM. Nada de rutas "
            "absolutas ni secretos en el cuaderno."
        )
    )
    cells.append(
        code(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Windows console is cp1252; structlog and Spanish prose use accents. Reconfigure\n"
            "# stdout/stderr to UTF-8 so an accented log line never raises UnicodeEncodeError.\n"
            "for _stream in (sys.stdout, sys.stderr):\n"
            "    try:\n"
            "        _stream.reconfigure(encoding='utf-8')\n"
            "    except (AttributeError, ValueError):\n"
            "        pass\n\n"
            "from ml.utils.notebook_setup import find_repo_root, load_env_local\n\n"
            "REPO_ROOT = find_repo_root()\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n"
            "os.chdir(REPO_ROOT)\n"
            "load_env_local(REPO_ROOT)\n\n"
            "%load_ext autoreload\n"
            "%autoreload 2\n\n"
            "import json\n"
            "import time\n\n"
            "import numpy as np\n"
            "import polars as pl\n"
            "from IPython.display import Markdown, display\n\n"
            "print('repo:', REPO_ROOT)\n"
            "print('report_dir:', report_dir, '| data_dir:', data_dir)"
        )
    )

    # ============================================ Seccion 1 - Backends ============
    cells.append(
        md(
            "## 1. Los backends LLM del reasoner (Be My Eyes)\n\n"
            "La abstraccion `LLMBackend` / `make_backend` desacopla el bucle del agente del "
            "modelo concreto: cambiar el nombre del modelo cambia el backend sin tocar las "
            "herramientas ni el perceiver. Probamos tres:\n\n"
            "- **Gemini 3.5 Flash** -- reasoner por defecto en la nube (`GeminiBackend`, Vertex AI "
            "o la API GenAI). Requiere `GEMINI_API_KEY` (o credenciales de Vertex) en "
            "`.env.local`.\n"
            "- **Qwen3.6-VL** -- on-prem **multimodal** servido por llama.cpp con `--mmproj` "
            "(`OllamaBackend` en `:8003`). Procesa imagenes ademas de texto.\n"
            "- **Qwen3.5-35B** -- on-prem **texto** servido por vLLM (`VLLMOpenAIBackend`, "
            "`:8002`). Soberania de datos: el razonamiento ocurre dentro del perimetro.\n\n"
            "Para cada uno comprobamos -- **sin llamadas de red** -- que `make_backend` resuelve "
            "la clase y el endpoint correctos, y luego una **sonda honesta** marca si el backend "
            "realmente disponible (credenciales / endpoint vivo). Un backend no disponible se "
            "**omite**; el cuaderno nunca inventa una respuesta de un LLM que no respondio."
        )
    )
    cells.append(
        code(
            "# Backend selection is a local, network-free operation. We resolve each model\n"
            "# name to its concrete backend and surface the endpoint it targets, exactly like\n"
            "# the chat service does via /llm/switch.\n"
            "from backend.app.core.config import get_settings\n"
            "from ml.agent.backends import (\n"
            "    GeminiBackend,\n"
            "    OllamaBackend,\n"
            "    VLLMOpenAIBackend,\n"
            "    make_backend,\n"
            ")\n\n"
            "settings = get_settings()\n\n"
            "_rows = []\n"
            "for _name, _label in backend_models.items():\n"
            "    _b = make_backend(_name, settings)\n"
            "    _kind = type(_b).__name__\n"
            "    _endpoint = getattr(_b, '_base_url', None) or (\n"
            "        'Vertex AI / GenAI' if isinstance(_b, GeminiBackend) else '-'\n"
            "    )\n"
            "    _rows.append({\n"
            "        'modelo': _name,\n"
            "        'etiqueta': _label,\n"
            "        'backend': _kind,\n"
            "        'modelo_servido': getattr(_b, 'model', _name),\n"
            "        'endpoint': _endpoint,\n"
            "    })\n"
            "backends_df = pl.DataFrame(_rows)\n"
            "with pl.Config(fmt_str_lengths=120, tbl_width_chars=220):\n"
            "    display(backends_df)"
        )
    )
    cells.append(
        code(
            "# Honest availability probe -- NO fabrication. A backend counts as available\n"
            "# only if its credentials / endpoint actually answer; otherwise it is marked NOT\n"
            "# AVAILABLE and skipped downstream. The probe is cheap and side-effect free.\n"
            "import urllib.error\n"
            "import urllib.request\n\n\n"
            "def _gemini_available(settings) -> tuple[bool, str]:\n"
            '    """Return whether Gemini credentials are configured (key or Vertex)."""\n'
            "    api_key = (getattr(settings, 'gemini_api_key', '') or\n"
            "               getattr(settings, 'google_api_key', ''))\n"
            "    _flag = str(getattr(settings, 'google_genai_use_vertexai', '')).lower()\n"
            "    use_vertex = _flag in ('1', 'true', 'yes')\n"
            "    if api_key:\n"
            "        return True, 'GEMINI_API_KEY presente en .env.local'\n"
            "    if use_vertex and getattr(settings, 'google_cloud_project', ''):\n"
            "        return True, 'Vertex AI configurado (proyecto + ADC)'\n"
            "    return False, 'sin GEMINI_API_KEY ni Vertex AI configurado'\n\n\n"
            "def _endpoint_alive(base_url: str, timeout: float = 2.0) -> tuple[bool, str]:\n"
            '    """Best-effort liveness check of an OpenAI-compatible endpoint (/models)."""\n'
            "    url = base_url.rstrip('/') + '/models'\n"
            "    try:\n"
            "        with urllib.request.urlopen(url, timeout=timeout) as resp:\n"
            "            return (200 <= resp.status < 500), f'HTTP {resp.status} en {url}'\n"
            "    except urllib.error.HTTPError as exc:\n"
            "        # A 4xx still proves the server is up and speaking HTTP.\n"
            "        return True, f'HTTP {exc.code} en {url} (servidor vivo)'\n"
            "    except (urllib.error.URLError, OSError, ValueError) as exc:\n"
            "        return False, f'sin respuesta en {url}: {exc}'\n\n\n"
            "availability = {}\n"
            "_probe_rows = []\n"
            "for _name in backend_models:\n"
            "    _b = make_backend(_name, settings)\n"
            "    if isinstance(_b, GeminiBackend):\n"
            "        _ok, _why = _gemini_available(settings)\n"
            "    else:\n"
            "        _ok, _why = _endpoint_alive(getattr(_b, '_base_url', ''))\n"
            "    availability[_name] = _ok\n"
            "    _probe_rows.append({\n"
            "        'modelo': _name,\n"
            "        'disponible': 'si' if _ok else 'NO',\n"
            "        'motivo': _why,\n"
            "    })\n"
            "display(pl.DataFrame(_probe_rows))\n"
            "_avail = [n for n, ok in availability.items() if ok]\n"
            "if _avail:\n"
            "    print('Backends disponibles para el copiloto:', ', '.join(_avail))\n"
            "else:\n"
            "    print('AVISO: ningun backend disponible localmente. La demo del copiloto se '\n"
            "          'omite honestamente; las secciones de comparacion de metricas (que leen '\n"
            "          'el report.json del entrenamiento) siguen ejecutandose abajo.')"
        )
    )

    # ============================================ Seccion 2 - Sesion + parcelas ===
    cells.append(
        md(
            "## 2. Conexion a la sesion y parcelas italianas reales\n\n"
            "El copiloto es **multi-tenant**: toda consulta y toda herramienta filtran por "
            "`session_id`. Abrimos el *pool* de la base local y resolvemos la sesion de "
            "demostracion sembrada. Las parcelas que pasaremos por el agente son **italianas "
            "reales** del homologo de US-078 (su geometria + embedding AlphaEarth viven en la "
            "sesion); si la siembra italiana aun no esta cargada, lo decimos y la demo del "
            "copiloto se omite sin inventar parcelas."
        )
    )
    cells.append(
        code(
            "# Connect to the seeded demo session and resolve the Italian demo parcels.\n"
            "from ml.agent.context import ToolContext\n"
            "from ml.agent.db import get_pool\n\n"
            "pool = await get_pool()\n"
            "async with pool.acquire() as conn:\n"
            "    session_id = await conn.fetchval(\n"
            "        'SELECT id FROM chat_sessions WHERE user_id = $1', demo_user\n"
            "    )\n"
            "ctx = (\n"
            "    ToolContext(pool=pool, settings=settings, session_id=session_id)\n"
            "    if session_id else None\n"
            ")\n\n"
            "demo_parcels = []\n"
            "if session_id is not None:\n"
            "    async with pool.acquire() as conn:\n"
            "        # Italian parcels of the session that actually have an embedding for the\n"
            "        # year (so classify returns a real class, not needs_gee_sampling).\n"
            "        rows = await conn.fetch(\n"
            "            'SELECT p.id, '\n"
            "            '       ST_X(ST_Centroid(p.geom)) AS lon, '\n"
            "            '       ST_Y(ST_Centroid(p.geom)) AS lat '\n"
            "            'FROM parcels p '\n"
            "            'JOIN features_parcels fp ON fp.parcel_id = p.id '\n"
            "            'WHERE p.session_id = $1 AND fp.year = $2 '\n"
            "            '  AND fp.alphaearth_embedding IS NOT NULL '\n"
            "            'ORDER BY p.id LIMIT $3',\n"
            "            session_id, int(year), int(n_demo_parcels),\n"
            "        )\n"
            "    demo_parcels = [\n"
            "        {'parcel_id': int(r['id']), 'lon': float(r['lon']), 'lat': float(r['lat'])}\n"
            "        for r in rows\n"
            "    ]\n\n"
            "if session_id is None:\n"
            "    display(Markdown('> **Sin sesion de demostracion.** Siembra los datos de demo '\n"
            "                     'para correr el copiloto. Las comparaciones de metricas de '\n"
            "                     'abajo no dependen de la base y siguen ejecutandose.'))\n"
            "elif not demo_parcels:\n"
            "    display(Markdown(\n"
            "        f'> **Sesion `{session_id}` sin parcelas italianas con embedding para '\n"
            "        f'{year}.** El copiloto se demostrara cuando la siembra italiana de '\n"
            "        'US-078 este cargada.'))\n"
            "else:\n"
            "    display(Markdown(\n"
            "        f'**Sesion**: `{session_id}` | parcelas italianas para la demo: '\n"
            "        f'{[p[\"parcel_id\"] for p in demo_parcels]}'))"
        )
    )

    # ============================================ Seccion 3 - Demo del copiloto ===
    cells.append(
        md(
            "## 3. Demo del copiloto sobre parcelas italianas\n\n"
            "Aqui se ve el patron Be My Eyes de punta a punta. Para cada backend disponible "
            "construimos el agente y le hacemos la **misma pregunta** sobre una parcela italiana "
            "real. El agente:\n\n"
            "1. llama `classify_new_parcel` con `use_stacking=True` -> el perceiver responde con "
            "la clase de cultivo + confianza del **Voting-3 / Stacking** sobre el dominio italiano "
            "(la cifra es del modelo, no del LLM);\n"
            "2. opcionalmente llama `compare_models` para contrastar los miembros del modelo "
            "**original** (`xgb-alphaearth`, `stacking-5`, `tsvit-pheno`) sobre la misma parcela;\n"
            "3. el **reasoner razona sobre ESE TEXTO** y redacta la explicacion.\n\n"
            "Renderizamos el **dialogo real**: cada `tool_call` con sus argumentos, cada "
            "`tool_result` (la observacion del perceiver) y la respuesta del reasoner. Toda cifra "
            "de la respuesta proviene de un `tool_result` visible: no hay numeros inventados."
        )
    )
    cells.append(
        code(
            "# Helper: drive ONE agent turn for a given backend and render the real dialogue.\n"
            "# Returns a compact record (answer text, n tool calls, latency, ok flag) for the\n"
            "# cross-backend table in section 5. NEVER fabricates: a backend failure is surfaced\n"
            "# as ok=False with the real error message.\n"
            "from ml.agent.agent import create_agent\n"
            "from ml.agent.events import (\n"
            "    DoneEvent,\n"
            "    ErrorEvent,\n"
            "    TextDeltaEvent,\n"
            "    ToolCallEvent,\n"
            "    ToolResultEvent,\n"
            ")\n\n\n"
            "def _summarize(result: dict, *, limit: int = 320) -> str:\n"
            '    """Compact one tool result dict into a short readable string."""\n'
            "    text = json.dumps(result, ensure_ascii=False, default=str)\n"
            "    return text if len(text) <= limit else text[:limit] + ' ...'\n\n\n"
            "async def run_copilot(model_name: str, question: str, *,\n"
            "                      render: bool = True) -> dict:\n"
            '    """Stream one user turn through the agent built on ``model_name``.\n\n'
            "    Renders the tool_call -> tool_result -> answer flow and returns a record with\n"
            "    the answer, the tool calls observed, the turn latency and an ok flag. A\n"
            "    backend that raises is reported with ok=False and the real error (no\n"
            "    fabrication).\n"
            '    """\n'
            "    if not availability.get(model_name, False):\n"
            "        if render:\n"
            "            display(Markdown(\n"
            "                f'> Backend `{model_name}` **no disponible**; turno omitido.'))\n"
            "        return {'model': model_name, 'ok': False, 'available': False,\n"
            "                'answer': '', 'n_tool_calls': 0, 'latency_ms': None,\n"
            "                'error': 'backend no disponible'}\n"
            "    agent = create_agent(model=model_name, settings=settings)\n"
            "    if render:\n"
            "        display(Markdown(\n"
            "            f'### Backend `{model_name}` '\n"
            "            f'({type(agent.backend).__name__})\\n\\n> {question}'\n"
            "        ))\n"
            "    answer_parts, tool_calls = [], []\n"
            "    error_msg, t0 = None, time.perf_counter()\n"
            "    try:\n"
            "        async for ev in agent.stream_response(\n"
            "            messages=[{'role': 'user', 'content': question}],\n"
            "            session_id=session_id,\n"
            "            ctx=ctx,\n"
            "        ):\n"
            "            if isinstance(ev, ToolCallEvent):\n"
            "                tool_calls.append(ev.name)\n"
            "                if render:\n"
            "                    _args = json.dumps(\n"
            "                        ev.arguments, ensure_ascii=False, default=str)\n"
            "                    display(Markdown(\n"
            "                        f'**herramienta** `{ev.name}` | argumentos: `{_args}`'\n"
            "                    ))\n"
            "            elif isinstance(ev, ToolResultEvent):\n"
            "                if render:\n"
            "                    _flag = 'ok' if ev.ok else 'ERROR'\n"
            "                    _res = _summarize(ev.result)\n"
            "                    display(Markdown(\n"
            "                        f'**perceiver** ({_flag}) `{ev.name}`: `{_res}`'\n"
            "                    ))\n"
            "            elif isinstance(ev, TextDeltaEvent):\n"
            "                answer_parts.append(ev.text)\n"
            "            elif isinstance(ev, ErrorEvent):\n"
            "                error_msg = ev.message\n"
            "                if render:\n"
            "                    display(Markdown(f'**error del agente**: {ev.message}'))\n"
            "            elif isinstance(ev, DoneEvent):\n"
            "                pass\n"
            "    except Exception as exc:  # surface the real failure, never fabricate\n"
            "        error_msg = f'{type(exc).__name__}: {exc}'\n"
            "        if render:\n"
            "            display(Markdown(\n"
            "                f'**fallo del backend** `{model_name}`: {error_msg}'))\n"
            "    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)\n"
            "    answer = ''.join(answer_parts).strip()\n"
            "    if render:\n"
            "        display(Markdown(\n"
            "            f'#### Respuesta del reasoner\\n\\n'\n"
            "            f'{answer or \"_(sin texto)_\"}\\n\\n'\n"
            "            f'_herramientas: {tool_calls or \"ninguna\"} | '\n"
            "            f'latencia: {latency_ms} ms_'\n"
            "        ))\n"
            "    return {'model': model_name, 'ok': error_msg is None and bool(answer),\n"
            "            'available': True, 'answer': answer, 'n_tool_calls': len(tool_calls),\n"
            "            'tool_calls': tool_calls, 'latency_ms': latency_ms, 'error': error_msg}"
        )
    )
    cells.append(
        code(
            "# The question we put to the copilot on a real Italian parcel. The agent draws a\n"
            "# tiny AOI around the parcel centroid so classify resolves the parcel spatially\n"
            "# (multi-tenant). use_stacking=True asks the perceiver for the Voting-3 / Stacking\n"
            "# posterior; the reasoner then reasons over that TEXT.\n"
            "copilot_records = []\n"
            "if ctx is not None and demo_parcels:\n"
            "    _p = demo_parcels[0]\n"
            "    _pid, _lon, _lat = _p['parcel_id'], _p['lon'], _p['lat']\n"
            "    _question = (\n"
            "        f'Tengo una parcela italiana centrada en lon={_lon:.5f}, lat={_lat:.5f}. '\n"
            "        f'Clasifica su cultivo con el ensamble del equipo (usa use_stacking=True y '\n"
            "        f'year={year}) y explicame, con la clase y la confianza que devuelva el '\n"
            "        f'modelo, que cultivo es y que tan seguro esta. No inventes cifras.'\n"
            "    )\n"
            "    for _name in backend_models:\n"
            "        _rec = await run_copilot(_name, _question)\n"
            "        copilot_records.append(_rec)\n"
            "else:\n"
            "    display(Markdown('> Demo del copiloto omitida (sin sesion o sin parcelas '\n"
            "                     'italianas con embedding). Las comparaciones de metricas de '\n"
            "                     'abajo no dependen del copiloto y se ejecutan igual.'))"
        )
    )
    cells.append(
        md(
            "**Lectura**: el agente **primero actua** (clasifica via el perceiver / Voting-3) y "
            "**luego razona** sobre el texto que devolvio la herramienta. La clase y la confianza "
            "son del modelo denso afinado a Italia; el LLM solo las interpreta y las explica. Ese "
            "es el contrato Be My Eyes y la garantia anti-alucinacion."
        )
    )

    # ====================================== Seccion 4 - Original vs TL ============
    cells.append(
        md(
            "## 4. Comparacion ORIGINAL vs TRANSFER\n\n"
            "Cargamos el `report.json` que produjo el runner del transfer. Si aun no existe (el "
            "entrenamiento en la H100 esta corriendo o pendiente), las celdas lo dicen y muestran "
            "el estado **pendiente** -- **nunca** numeros placeholder. El reporte trae dos formas "
            "segun el `vote-space`: `parcel` (replica del campeon, con `voting_dense_eval`) y "
            "`pixel` (experimento denso, con `voting_eval`). Las celdas son robustas a ambas."
        )
    )
    cells.append(
        code(
            "# Load the runner report robustly. REAL VALUES ONLY: when absent, report the\n"
            "# pending state honestly. Normalise the two vote-space shapes (parcel/pixel).\n"
            "REPORT_DIR = Path(report_dir)\n"
            "report_path = REPORT_DIR / 'report.json'\n"
            "HAS_REPORT = report_path.is_file()\n"
            "report = (\n"
            "    json.loads(report_path.read_text(encoding='utf-8')) if HAS_REPORT else None\n"
            ")\n\n"
            "def _voting_eval(rep: dict) -> dict | None:\n"
            '    """Return the Voting-3 dense summary regardless of vote-space shape."""\n'
            "    return rep.get('voting_eval') or rep.get('voting_dense_eval')\n\n"
            "def _best_subset(rep: dict) -> dict | None:\n"
            '    """Return the best F1>0.9 subset regardless of vote-space shape."""\n'
            "    return (rep.get('best_subset_f1_over_0.9')\n"
            "            or rep.get('voting_dense_best_subset_f1_over_0.9'))\n\n"
            "if HAS_REPORT:\n"
            "    print(f'Reporte US-079 encontrado: run={report.get(\"run\")}, '\n"
            "          f'vote_space={report.get(\"vote_space\")}, '\n"
            "          f'test_fold={report.get(\"test_fold\")}, '\n"
            "          f'miembros={report.get(\"members\")}')\n"
            "else:\n"
            "    print('PENDIENTE: no hay report.json todavia. El fine-tune + Voting-3 + eval '\n"
            "          'corre en la H100 (scripts/run_transfer_italia.py). Re-ejecuta este '\n"
            "          'cuaderno con el reporte presente para poblar las metricas reales. No '\n"
            "          'se muestran numeros inventados.')"
        )
    )

    # --------------------------------------- 4a. transfer delta (zero-shot vs TL) ---
    cells.append(
        md(
            "### 4a. Delta del transfer: campeon frances zero-shot vs TL afinado\n\n"
            "La cota inferior es el **campeon frances zero-shot**: el checkpoint PASTIS "
            "aplicado tal cual a Italia (las clases mediterraneas nuevas, que nunca vio, caen "
            "a fondo). El **TL** es ese mismo backbone **afinado** sobre los patches "
            "italianos. El delta = (fine-tune) - (zero-shot) cuantifica cuanto aporta afinar "
            "de verdad sobre el dominio nuevo. Se lee del campo `transfer_delta` del reporte "
            "(presente cuando el runner corrio sin `--no-zero-shot`)."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT and report.get('transfer_delta'):\n"
            "    d = report['transfer_delta']\n"
            "    ddf = pl.DataFrame({\n"
            "        'metrica': list(d.keys()),\n"
            "        'delta (fine-tune - zero-shot)': [round(float(v), 4) for v in d.values()],\n"
            "    })\n"
            "    display(ddf)\n"
            "    import matplotlib.pyplot as plt\n"
            "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "    _vals = [float(v) for v in d.values()]\n"
            "    _colors = ['#2e7d32' if v >= 0 else '#c62828' for v in _vals]\n"
            "    ax.barh(list(d.keys())[::-1], _vals[::-1], color=_colors[::-1])\n"
            "    ax.axvline(0, color='black', linewidth=0.8)\n"
            "    ax.set_title('Delta del transfer (fine-tune - zero-shot) sobre Italia')\n"
            "    ax.grid(axis='x', alpha=0.3); plt.tight_layout(); plt.show()\n"
            "    print('Un delta positivo confirma que afinar adapta el backbone frances al '\n"
            "          'vocabulario mediterraneo nuevo, en vez de forzar todo por la taxonomia '\n"
            "          'francesa.')\n"
            "elif HAS_REPORT:\n"
            "    print('PENDIENTE: el reporte existe pero no trae transfer_delta. Re-corre el '\n"
            "          'runner sin --no-zero-shot para medir el campeon zero-shot sobre Italia.')\n"
            "else:\n"
            "    print('PENDIENTE: delta del transfer (necesita el report.json del '\n"
            "          'entrenamiento).')"
        )
    )

    # --------------------------------------- 4b. domain parity (France vs Italy) ---
    cells.append(
        md(
            "### 4b. Paridad de dominios: campeon en Francia (0.9069) vs TL en Italia\n\n"
            "El segundo angulo compara **dominios**: el campeon **Voting-3** logro F1-macro "
            f"**{_FRANCE_CHAMPION_F1}** sobre `france-10` (EPIC 6, una constante ya medida, "
            "no una salida fabricada). La pregunta de US-079 es si el **TL** alcanza un "
            "objetivo de calidad equivalente en el dominio **italiano** -- medido por el "
            "F1-macro del Voting-3 sobre las mejores ~10 clases italianas (subconjunto "
            "honesto con F1 > 0.9). Solo se grafica el TL cuando el reporte lo trae."
        )
    )
    cells.append(
        code(
            "ev = _voting_eval(report) if HAS_REPORT else None\n"
            "best = _best_subset(report) if HAS_REPORT else None\n"
            "if HAS_REPORT and ev is not None:\n"
            "    _ff = ev.get('fine_f1_macro')\n"
            "    tl_fine_f1 = float(_ff) if _ff is not None else None\n"
            "    tl_best_f1 = float(best['macro_f1']) if best else None\n"
            "    tl_best_n = int(best['n_classes']) if best else None\n"
            "    rows = [{'dominio': 'Francia (PASTIS, france-10)',\n"
            "             'modelo': 'Voting-3 (campeon)',\n"
            "             'f1_macro': round(france_champion_f1, 4),\n"
            "             'nota': 'referencia EPIC 6 (ya medida)'}]\n"
            "    if tl_fine_f1 is not None:\n"
            "        rows.append({'dominio': 'Italia (TL fino, todas las clases)',\n"
            "                     'modelo': 'Voting-3 (transfer)',\n"
            "                     'f1_macro': round(tl_fine_f1, 4),\n"
            "                     'nota': 'espacio de etiquetas italiano completo'})\n"
            "    if tl_best_f1 is not None:\n"
            "        rows.append({'dominio': f'Italia (TL, mejores {tl_best_n} clases)',\n"
            "                     'modelo': 'Voting-3 (transfer)',\n"
            "                     'f1_macro': round(tl_best_f1, 4),\n"
            "                     'nota': 'subconjunto honesto con F1 > 0.9'})\n"
            "    parity_df = pl.DataFrame(rows)\n"
            "    display(parity_df)\n"
            "    import matplotlib.pyplot as plt\n"
            "    fig, ax = plt.subplots(figsize=(8, 3.6))\n"
            "    _labels = [f\"{r['dominio']}\\n{r['modelo']}\" for r in rows]\n"
            "    _f1s = [r['f1_macro'] for r in rows]\n"
            "    _colors = ['#1565c0'] + ['#6a1b9a'] * (len(rows) - 1)\n"
            "    ax.bar(range(len(rows)), _f1s, color=_colors)\n"
            "    ax.axhline(france_champion_f1, color='grey', linestyle='--',\n"
            "               label=f'paridad Francia {france_champion_f1}')\n"
            "    ax.set_xticks(range(len(rows)))\n"
            "    ax.set_xticklabels(_labels, rotation=15, ha='right', fontsize=8)\n"
            "    ax.set_ylabel('F1-macro'); ax.set_ylim(0, 1)\n"
            "    ax.set_title('Paridad de dominios: campeon Francia vs TL Italia')\n"
            "    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()\n"
            "    if tl_best_f1 is not None:\n"
            "        _gap = round(tl_best_f1 - france_champion_f1, 4)\n"
            "        print(f'TL Italia (mejores {tl_best_n} clases) F1={tl_best_f1} vs Francia '\n"
            "              f'{france_champion_f1} -> diferencia {_gap:+}.')\n"
            "else:\n"
            "    print('PENDIENTE: paridad de dominios. La referencia de Francia '\n"
            "          f'(F1 {france_champion_f1}) esta fija; el TL de Italia se grafica al '\n"
            "          'leerse el report.json del entrenamiento. No se inventan numeros.')"
        )
    )
    cells.append(
        code(
            "# Voting-3 learned weights over Italy (AC2 interpretability), when available.\n"
            "if HAS_REPORT and report.get('voting_weights'):\n"
            "    weights = report['voting_weights']\n"
            "    wdf = pl.DataFrame({'miembro': list(weights.keys()),\n"
            "                        'peso': [round(float(v), 4) for v in weights.values()]}\n"
            "                       ).sort('peso', descending=True)\n"
            "    display(wdf)\n"
            "    _oof = report.get('voting_oof_f1_macro')\n"
            "    print(f'F1-macro OOF (spatial-CV) del Voting-3 sobre Italia: {_oof}')\n"
            "else:\n"
            "    print('PENDIENTE: pesos del Voting-3 (se reportan al correr el runner).')"
        )
    )

    # ====================================== Seccion 5 - Copiloto x backend ========
    cells.append(
        md(
            "## 5. Cuanto rinde el copiloto con cada backend LLM\n\n"
            "El Voting-3 (el perceiver) es el **mismo** para todos los backends: la clase de "
            "cultivo y su confianza no cambian segun el LLM, porque el LLM **no clasifica**. Lo "
            "que cambia entre backends es la **calidad del razonamiento sobre ese texto** y "
            "la **latencia / coste**. La tabla siguiente resume, por backend, los registros "
            "reales del turno del copiloto de la seccion 3: si llamo a la herramienta "
            "correcta, cuantas herramientas uso, la latencia del turno y si produjo "
            "respuesta. Un backend no "
            "disponible aparece como tal -- sin cifras inventadas."
        )
    )
    cells.append(
        code(
            "# Cross-backend summary table from the REAL copilot records (section 3). The\n"
            "# perceiver metric (the Voting-3 class/confidence) is identical across backends by\n"
            "# construction; what varies is the reasoning over that text, the tool use and the\n"
            "# latency.\n"
            "if copilot_records:\n"
            "    _rows = []\n"
            "    for r in copilot_records:\n"
            "        _used = 'classify_new_parcel' in (r.get('tool_calls') or [])\n"
            "        _rows.append({\n"
            "            'backend': r['model'],\n"
            "            'disponible': 'si' if r.get('available') else 'NO',\n"
            "            'respondio': 'si' if r.get('ok') else 'no',\n"
            "            'uso_classify': 'si' if _used else 'no',\n"
            "            'n_herramientas': r.get('n_tool_calls', 0),\n"
            "            'latencia_ms': r.get('latency_ms'),\n"
            "            'chars_respuesta': len(r.get('answer') or ''),\n"
            "        })\n"
            "    bench_df = pl.DataFrame(_rows)\n"
            "    with pl.Config(tbl_width_chars=200):\n"
            "        display(bench_df)\n"
            "    _ok = [r['model'] for r in copilot_records if r.get('ok')]\n"
            "    print('Backends que completaron el turno del copiloto:',\n"
            "          ', '.join(_ok) or 'ninguno')\n"
            "    print('El perceiver (Voting-3) entrega la misma clase/confianza a todos; el '\n"
            "          'LLM solo razona sobre ese TEXTO (Be My Eyes).')\n"
            "else:\n"
            "    print('PENDIENTE: tabla backend x metrica. Requiere la sesion sembrada con '\n"
            "          'parcelas italianas y al menos un backend disponible. No se fabrican '\n"
            "          'filas.')"
        )
    )

    # ----------------------------------------------------------- Conclusiones ---
    cells.append(
        md(
            "## 6. Conclusiones\n\n"
            "**Que se demostro**\n\n"
            "- El **copiloto accede a los modelos densos del TL a traves del agente** "
            "(`classify_new_parcel` / `compare_models`), no por una llamada directa: el perceiver "
            "es el Voting-3 afinado a Italia y el reasoner razona sobre su TEXTO (Be My Eyes).\n"
            "- La **abstraccion de backend** intercambia tres reasoners (`gemini-3.5-flash`, "
            "`qwen3.6-vl`, `qwen35`) sin tocar las herramientas; los no disponibles se reportan "
            "honestamente y se omiten.\n"
            "- El **ORIGINAL vs TL** se contrasta desde dos angulos: el **delta del transfer** "
            "(campeon zero-shot vs TL afinado) y la **paridad de dominios** (Francia "
            f"{_FRANCE_CHAMPION_F1} vs TL en Italia), ambos leidos del `report.json` real.\n\n"
            "**Regla de honestidad**\n\n"
            "- Toda cifra proviene del `report.json` del entrenamiento o de un `tool_result` "
            "visible del copiloto. Si el reporte aun no existe o un backend no respondio, el "
            "cuaderno lo dice y muestra el estado pendiente -- **nunca** numeros placeholder.\n\n"
            "**Lo que sigue**\n\n"
            "- Re-ejecutar con el `report.json` poblado (tras el run de la H100) para fijar las "
            "metricas reales de transfer y paridad.\n"
            "- Levantar los endpoints on-prem (Qwen-VL `:8003`, vLLM `:8002`) para comparar, sobre "
            "la misma pregunta, la calidad del razonamiento nube vs on-prem."
        )
    )

    # ---------------------------------------------------- Cierre del pool ------
    cells.append(
        md("### Cierre\n\nCerramos el *pool* de conexiones de forma ordenada al terminar.")
    )
    cells.append(
        code(
            "# Close the shared asyncpg pool cleanly (no-op if it was never opened).\n"
            "try:\n"
            "    from ml.agent.db import close_pool\n"
            "    await close_pool()\n"
            "    print('pool cerrado.')\n"
            "except Exception as exc:  # the pool may never have been created (no DB session)\n"
            "    print('cierre del pool omitido:', exc)"
        )
    )

    return cells


@app.command()
def build(
    out: Annotated[Path, typer.Option(help="Ruta de salida del notebook.")] = _DEFAULT_OUT,
    report_dir: Annotated[
        Path, typer.Option(help="Ruta de la salida del runner (report.json del TL).")
    ] = _DEFAULT_REPORT,
    data_dir: Annotated[
        Path, typer.Option(help="Ruta del dataset homologo italiano (US-078).")
    ] = _DEFAULT_DATA,
) -> None:
    """Write the US-079 copilot notebook (unexecuted; papermill populates outputs).

    The notebook is committed UNEXECUTED and is meant to be run against the real
    database + LLM credentials. It is robust to a missing ``report.json`` (shows
    the pending state) and to unavailable LLM backends (marks them and skips),
    never fabricating numbers.

    Args:
        out: Output ``.ipynb`` path.
        report_dir: Repo-relative path to the runner output (``report.json``) the
            comparison cells read.
        data_dir: Repo-relative path to the Italian homologue dataset (US-078).
    """
    nb = nbf.v4.new_notebook()
    nb.cells = _build_cells(str(report_dir).replace("\\", "/"), str(data_dir).replace("\\", "/"))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    typer.echo(f"Notebook escrito en {out} ({len(nb.cells)} celdas).")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
