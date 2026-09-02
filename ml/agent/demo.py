"""Reusable driver + renderers for the conversational-copilot demo notebooks.

The Avance 6 demo (``notebooks/final_model/Avance6.Demo.Copiloto.Equipo17.ipynb``)
and the US-079 transfer copilot (``notebooks/transfer/us079_copilot_original_vs_tl
.ipynb``) used to each declare the SAME helpers inline -- the availability probe,
the one-turn agent driver that renders ``tool_call -> tool_result -> answer``, the
result summariser and the cross-backend table. This module centralises them so the
notebooks stay short (a cell becomes a single call) and the logic is unit-tested
once instead of copy-pasted into two ``.ipynb`` files (notebooks/CLAUDE.md: any
inline cell logic above a few lines belongs in ``ml/``).

Design:

- **Display is optional.** Every renderer returns the plain data (a ``list[dict]``
  or a record) AND, when ``display=True`` (the notebook default), shows it via
  ``IPython.display`` + Polars. Tests call with ``display=False`` so they assert on
  the data with no notebook dependency. The heavy display imports (IPython, Polars)
  are therefore lazy -- imported inside the functions, never at module load -- so
  the module imports cleanly in a headless test process.
- **Honest by construction.** :func:`run_agent_turn` never fabricates: a backend
  that raises is returned as ``ok=False`` with the real error string; an
  unavailable backend is skipped. The token/latency numbers are measured, never
  invented.
- **Backend-agnostic.** The driver consumes the typed
  :class:`~ml.agent.events.AgentEvent` stream of :meth:`Agent.stream_response`, so
  it works identically for the Gemini, vLLM-Qwen and Qwen-VL backends selected by
  :func:`ml.agent.backends.make_backend`.

Visible strings rendered to the reader are Spanish (project convention); code,
identifiers and docstrings stay English ASCII. No emojis.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from pydantic import BaseModel

from ml.agent.agent import create_agent
from ml.agent.backends import GeminiBackend, make_backend
from ml.agent.events import (
    DoneEvent,
    ErrorEvent,
    PerceiverObservationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.app.core.config import Settings
    from ml.agent.agent import Agent
    from ml.agent.context import ToolContext
    from ml.agent.perceiver import PerceiverLayer, PerceiverObservation
    from ml.agent.rag import RAGDocument

logger = structlog.get_logger(__name__)

__all__ = [
    "BACKEND_CATALOG",
    "backend_overview",
    "cross_backend_table",
    "endpoint_alive",
    "gemini_available",
    "load_persisted_records",
    "observe_parcels",
    "perceiver_table",
    "probe_availability",
    "quiet_logging",
    "rag_table",
    "run_agent_turn",
    "run_backend_turn",
    "run_tool",
    "save_backend_record",
    "summarize_result",
    "tool_inventory",
]

#: The three reasoner backends the copilot demos contrast (name -> human label).
#: ``make_backend`` resolves each name to its concrete backend and endpoint:
#: ``gemini-3.5-flash`` -> :class:`GeminiBackend` (cloud), ``qwen3.6-vl`` ->
#: multimodal ``OllamaBackend`` (llama.cpp ``:8003``), ``qwen35`` ->
#: :class:`VLLMOpenAIBackend` (vLLM ``:8002``). Reused by both copilot notebooks.
BACKEND_CATALOG: dict[str, str] = {
    "gemini-3.5-flash": "Gemini 3.5 Flash (nube, Vertex AI / GenAI)",
    "qwen3.6-vl": "Qwen3.6-VL (on-prem multimodal, llama.cpp :8003)",
    "qwen35": "Qwen3-30B-A3B (on-prem texto, vLLM :8002)",
}

#: Default per-result summary length for :func:`summarize_result`.
_SUMMARY_LIMIT: int = 320


# ---------------------------------------------------------------------------
# Logging hygiene
# ---------------------------------------------------------------------------
def quiet_logging(level: int = logging.WARNING) -> None:
    """Silence the agent's INFO/DEBUG structlog noise inside a notebook.

    The agent, perceiver, tools and DB layer emit a structured INFO/DEBUG line
    per step (``perceiver_observe_started``, ``agent_db_session_scoped``,
    ``classify_voting3_unavailable`` ...). In a live demo those flood the cell
    output and bury the actual answer. This reconfigures structlog with a
    filtering bound logger at ``level`` (default WARNING), mirroring
    ``backend/app/core/logging.py`` but quieter and with colours off (ANSI codes
    render as garbage in exported notebook HTML). Warnings and errors still show.

    Args:
        level: Minimum stdlib level to keep (``logging.WARNING`` by default).
    """
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


# ---------------------------------------------------------------------------
# Small display helpers (lazy imports so the module stays headless-importable)
# ---------------------------------------------------------------------------
def _md(text: str) -> None:
    """Render Markdown in a notebook (no-op-safe outside IPython)."""
    from IPython.display import Markdown, display

    display(Markdown(text))


def _display_df(
    rows: Sequence[dict[str, Any]],
    *,
    fmt_str_lengths: int = 120,
    tbl_width: int = 220,
) -> Any:
    """Build a Polars DataFrame from ``rows`` and display it; return the frame.

    Args:
        rows: Records to tabulate.
        fmt_str_lengths: Polars string-truncation width for the display config.
        tbl_width: Polars table width (chars) for the display config.

    Returns:
        The built :class:`polars.DataFrame` (also displayed as a side effect).
    """
    import polars as pl
    from IPython.display import display

    frame = pl.DataFrame(list(rows))
    with pl.Config(fmt_str_lengths=fmt_str_lengths, tbl_width_chars=tbl_width):
        display(frame)
    return frame


# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------
def tool_inventory(*, display: bool = True) -> list[dict[str, Any]]:
    """Tabulate the geospatial tools straight from the LLM-facing declarations.

    Built from :func:`ml.agent.tools.build_function_declarations` -- the single
    source of truth advertised to the model -- so the table can never drift from
    what the agent actually exposes.

    Args:
        display: When ``True`` (notebook default), also render the table.

    Returns:
        One record per tool: ``herramienta`` / ``tipo`` (sincrona|diferida) /
        ``comportamiento`` (BLOCKING|NON_BLOCKING) / ``descripcion``.
    """
    from ml.agent.tools import TOOL_SPECS, build_function_declarations

    rows: list[dict[str, Any]] = []
    for decl in build_function_declarations():
        name = decl.name
        if name is None:  # declarations always carry a name; narrow for the type checker
            continue
        deferred = TOOL_SPECS[name][3]
        rows.append(
            {
                "herramienta": name,
                "tipo": "diferida" if deferred else "sincrona",
                "comportamiento": "NON_BLOCKING" if deferred else "BLOCKING",
                "descripcion": decl.description,
            }
        )
    rows.sort(key=lambda r: (r["tipo"], r["herramienta"]))
    if display:
        _display_df(rows)
        n_sync = sum(1 for r in rows if r["tipo"] == "sincrona")
        _md(
            f"**{len(rows)} herramientas** | sincronas: **{n_sync}** | "
            f"diferidas: **{len(rows) - n_sync}**"
        )
    return rows


# ---------------------------------------------------------------------------
# Backend resolution + availability
# ---------------------------------------------------------------------------
def backend_overview(
    names: Iterable[str],
    settings: Settings,
    *,
    display: bool = True,
) -> list[dict[str, Any]]:
    """Resolve each model name to its backend class + endpoint (network-free).

    Mirrors what the chat service does through ``/llm/switch``: it only resolves
    the class and the endpoint, it does not call the model.

    Args:
        names: Reasoner model names (e.g. the keys of :data:`BACKEND_CATALOG`).
        settings: Typed settings carrying the on-prem URLs / API keys.
        display: When ``True``, also render the resolution table.

    Returns:
        One record per backend: ``modelo`` / ``etiqueta`` / ``backend`` (class) /
        ``modelo_servido`` / ``endpoint``.
    """
    rows: list[dict[str, Any]] = []
    for name in names:
        backend = make_backend(name, settings)
        endpoint = getattr(backend, "_base_url", None) or (
            "Vertex AI / GenAI" if isinstance(backend, GeminiBackend) else "-"
        )
        rows.append(
            {
                "modelo": name,
                "etiqueta": BACKEND_CATALOG.get(name, name),
                "backend": type(backend).__name__,
                "modelo_servido": getattr(backend, "model", name),
                "endpoint": endpoint,
            }
        )
    if display:
        _display_df(rows)
    return rows


def gemini_available(settings: Settings) -> tuple[bool, str]:
    """Return whether Gemini credentials are configured (API key or Vertex).

    Args:
        settings: Typed settings read from ``.env.local``.

    Returns:
        ``(available, reason)``; ``available`` is ``True`` when a usable key or a
        Vertex AI project is present, ``False`` otherwise. Never makes a network
        call (a real generation would cost a token round-trip).
    """
    # Defensive getattr over optionally-present credential fields (test stubs may
    # omit some); mirrors ml.agent.backends.make_backend.
    api_key = getattr(settings, "gemini_api_key", "") or getattr(settings, "google_api_key", "")
    use_vertex = str(getattr(settings, "google_genai_use_vertexai", "")).lower() in (
        "1",
        "true",
        "yes",
    )
    if api_key:
        return True, "GEMINI_API_KEY presente en .env.local"
    if use_vertex and getattr(settings, "google_cloud_project", ""):
        return True, "Vertex AI configurado (proyecto + ADC)"
    return False, "sin GEMINI_API_KEY ni Vertex AI configurado"


def endpoint_alive(
    base_url: str,
    *,
    timeout: float = 2.0,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bool, str]:
    """Best-effort liveness probe of an OpenAI-compatible endpoint (``/models``).

    A 2xx-4xx response proves the server is up and speaking HTTP (a 4xx still
    means "alive but rejected"); a transport error (refused / timeout / DNS)
    means down. ``urlopen`` is injectable so tests assert the logic offline.

    Args:
        base_url: The ``.../v1`` base URL of the server.
        timeout: Socket timeout in seconds.
        urlopen: The opener (defaults to :func:`urllib.request.urlopen`).

    Returns:
        ``(alive, reason)``.
    """
    if not base_url:
        return False, "endpoint sin URL"
    url = base_url.rstrip("/") + "/models"
    try:
        with urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            return (200 <= status < 500), f"HTTP {status} en {url}"
    except urllib.error.HTTPError as exc:
        # A 4xx still proves the server is up and speaking HTTP.
        return True, f"HTTP {exc.code} en {url} (servidor vivo)"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"sin respuesta en {url}: {exc}"


def probe_availability(
    names: Iterable[str],
    settings: Settings,
    *,
    display: bool = True,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Probe which reasoner backends are actually reachable here and now.

    Gemini is checked by credentials (no token spent); the OpenAI-compatible
    on-prem backends are checked by a ``/models`` liveness call. Nothing is
    fabricated: a down backend is reported ``NO`` and is meant to be skipped
    downstream.

    Args:
        names: Reasoner model names to probe.
        settings: Typed settings (credentials + on-prem URLs).
        display: When ``True``, render the probe table.
        urlopen: Injectable opener for the endpoint liveness check (tests).

    Returns:
        ``(availability, rows)`` where ``availability`` maps name -> bool and
        ``rows`` is the renderable table (``modelo`` / ``disponible`` / ``motivo``).
    """
    availability: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for name in names:
        backend = make_backend(name, settings)
        if isinstance(backend, GeminiBackend):
            ok, reason = gemini_available(settings)
        else:
            ok, reason = endpoint_alive(getattr(backend, "_base_url", ""), urlopen=urlopen)
        availability[name] = ok
        rows.append({"modelo": name, "disponible": "si" if ok else "NO", "motivo": reason})
    if display:
        _display_df(rows)
        ready = [n for n, ok in availability.items() if ok]
        _md(
            "**Backends disponibles:** " + (", ".join(f"`{n}`" for n in ready) or "_ninguno_")
            if ready
            else "> Ningun backend disponible localmente; los turnos del copiloto se "
            "omiten honestamente (sin inventar respuestas)."
        )
    return availability, rows


# ---------------------------------------------------------------------------
# Agent turn driver + event rendering
# ---------------------------------------------------------------------------
def summarize_result(result: dict, *, limit: int = _SUMMARY_LIMIT) -> str:
    """Compact one tool-result mapping into a short, readable one-line string.

    Args:
        result: The tool result mapping (already JSON-able).
        limit: Maximum characters before truncating with an ellipsis.

    Returns:
        A single-line JSON rendering, truncated to ``limit`` chars.
    """
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def _blank_record(model: str, *, available: bool, error: str) -> dict[str, Any]:
    """Return the record shape for a turn that did not run (skipped/unavailable)."""
    return {
        "model": model,
        "ok": False,
        "available": available,
        "answer": "",
        "n_tool_calls": 0,
        "tool_calls": [],
        "latency_ms": None,
        "error": error,
    }


async def run_agent_turn(
    agent: Agent,
    question: str,
    *,
    ctx: ToolContext,
    session_id: UUID,
    render: bool = True,
    title: str | None = None,
) -> dict[str, Any]:
    """Stream ONE user turn through a pre-built agent and render its event flow.

    Drives :meth:`Agent.stream_response`, rendering each ``tool_call`` (with its
    arguments), each ``tool_result`` (a compact summary) and finally the
    reasoner's answer. It NEVER raises: any backend/tool failure is captured and
    returned with ``ok=False`` and the real error message, so a live demo degrades
    honestly instead of crashing the notebook.

    Args:
        agent: A ready :class:`~ml.agent.agent.Agent` (use :func:`run_backend_turn`
            to build one from a model name with an availability guard).
        question: The user question for this turn.
        ctx: Shared tool execution context (session-scoped).
        session_id: Tenant session id (must match ``ctx.session_id``).
        render: When ``True`` (notebook default), render the dialogue inline.
        title: Optional Markdown header shown before the question (e.g. the
            backend label); defaults to a plain "Pregunta" heading.

    Returns:
        A record: ``model`` / ``ok`` / ``available`` / ``answer`` / ``n_tool_calls``
        / ``tool_calls`` (names) / ``latency_ms`` / ``error``.
    """
    model = getattr(agent.backend, "model", "?")
    if render:
        header = title or "### Pregunta"
        _md(f"{header}\n\n> {question}")

    answer_parts: list[str] = []
    tool_calls: list[str] = []
    error_msg: str | None = None
    start = time.perf_counter()
    try:
        async for event in agent.stream_response(
            messages=[{"role": "user", "content": question}],
            session_id=session_id,
            ctx=ctx,
        ):
            if isinstance(event, ToolCallEvent):
                tool_calls.append(event.name)
                if render:
                    args = json.dumps(event.arguments, ensure_ascii=False, default=str)
                    _md(f"**herramienta** `{event.name}` | argumentos: `{args}`")
            elif isinstance(event, ToolResultEvent):
                if render:
                    flag = "ok" if event.ok else "ERROR"
                    summary = summarize_result(event.result)
                    _md(f"**resultado** ({flag}) `{event.name}`: `{summary}`")
            elif isinstance(event, PerceiverObservationEvent):
                if render:
                    _md("**observacion del perceiver inyectada al reasoner.**")
            elif isinstance(event, TextDeltaEvent):
                answer_parts.append(event.text)
            elif isinstance(event, ErrorEvent):
                error_msg = event.message
                if render:
                    _md(f"**error del agente**: {event.message}")
            elif isinstance(event, DoneEvent):
                pass
    except Exception as exc:  # noqa: BLE001 - boundary guard: never crash the demo
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("demo_agent_turn_failed", model=model, error=str(exc))
        if render:
            _md(f"**fallo del backend** `{model}`: {error_msg}")

    latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
    answer = "".join(answer_parts).strip()
    if render:
        _md(
            f"#### Respuesta del reasoner\n\n{answer or '_(sin texto)_'}\n\n"
            f"_herramientas: {tool_calls or 'ninguna'} | latencia: {latency_ms} ms_"
        )
    return {
        "model": model,
        "ok": error_msg is None and bool(answer),
        "available": True,
        "answer": answer,
        "n_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "latency_ms": latency_ms,
        "error": error_msg,
    }


async def run_backend_turn(
    model: str,
    question: str,
    *,
    settings: Settings,
    ctx: ToolContext,
    session_id: UUID,
    availability: dict[str, bool] | None = None,
    render: bool = True,
    title: str | None = None,
) -> dict[str, Any]:
    """Build the agent for ``model`` and run one turn, guarded by availability.

    Convenience wrapper over :func:`run_agent_turn` for the cross-backend
    comparison: it skips a model flagged unavailable in ``availability`` (returning
    a blank record), otherwise builds the agent with :func:`create_agent` and
    drives one turn with a backend-labelled header.

    Args:
        model: Reasoner model name (selects the backend via ``make_backend``).
        question: The user question.
        settings: Typed settings injected into the backend.
        ctx: Shared tool execution context.
        session_id: Tenant session id.
        availability: Optional map from :func:`probe_availability`; when it marks
            ``model`` unavailable, the turn is skipped honestly.
        render: When ``True``, render the dialogue inline.
        title: Optional Markdown header to show before the question; defaults to a
            backend-labelled header (model + backend class).

    Returns:
        The same record shape as :func:`run_agent_turn`.
    """
    if availability is not None and not availability.get(model, False):
        if render:
            _md(f"> Backend `{model}` **no disponible**; turno omitido.")
        return _blank_record(model, available=False, error="backend no disponible")
    agent = create_agent(model=model, settings=settings)
    if title is None:
        label = BACKEND_CATALOG.get(model, model)
        title = f"### Backend `{model}` -- {label} ({type(agent.backend).__name__})"
    return await run_agent_turn(
        agent, question, ctx=ctx, session_id=session_id, render=render, title=title
    )


async def run_tool(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    render: bool = True,
) -> dict[str, Any]:
    """Run ONE tool directly (bypassing the LLM) and return its result honestly.

    Used to demonstrate the tools the conversational loop does not exercise inline
    -- the AOI-based synchronous tools and the deferred ones (``compare_models``,
    ``add_aoi`` ...), which Gemini's standard generate API rejects in the loop
    because of their ``NON_BLOCKING`` behavior. Mirrors the agent's own dispatch:
    inject the tenant ``session_id``, validate ``args`` with the tool's Pydantic
    input model, run it. NEVER raises -- a validation/execution failure is captured
    and returned with ``ok=False`` and the real error (so a parcel that is not in
    the OOF, say, degrades honestly instead of crashing the notebook).

    Args:
        name: Registered tool name (e.g. ``"classify_new_parcel"``).
        args: Tool arguments WITHOUT ``session_id`` (it is injected from ``ctx``);
            nested geometries may be passed as plain GeoJSON dicts.
        ctx: Shared tool execution context (session-scoped).
        render: When ``True``, render the result (or the honest failure) inline.

    Returns:
        ``{name, ok, result, error}``: ``result`` is the tool output as a
        JSON-able mapping (empty on failure), ``error`` the real message or ``None``.
    """
    from ml.agent.tools import get_tool

    spec = get_tool(name)
    raw = dict(args)
    if "session_id" in spec.input_model.model_fields:
        raw["session_id"] = ctx.session_id
    try:
        inp = spec.input_model.model_validate(raw)
        output = await spec.fn(inp, ctx)
        result = output.model_dump(mode="json") if isinstance(output, BaseModel) else dict(output)
        ok, error = True, None
    except Exception as exc:  # noqa: BLE001 - honest degradation, never fabricate
        result, ok, error = {}, False, f"{type(exc).__name__}: {exc}"
        logger.warning("demo_tool_failed", tool=name, error=str(exc))
    if render:
        if ok:
            _md(f"**`{name}`** -> `{summarize_result(result)}`")
        else:
            _md(f"**`{name}`** no disponible (degradacion honesta): {error}")
    return {"name": name, "ok": ok, "result": result, "error": error}


def save_backend_record(
    record: dict[str, Any],
    out_dir: Path | str,
    *,
    only_ok: bool = True,
) -> Path | None:
    """Persist a per-backend turn record to ``<out_dir>/<model>.json``.

    Lets the three reasoners be evaluated **one at a time** (the on-prem Qwen
    text and Qwen-VL share the single H100 GPU, so only one can serve at once):
    each live pass saves its backend's real record, and
    :func:`load_persisted_records` reassembles the complete cross-backend table
    from the accumulated files. By default only a successful run is persisted
    (``only_ok``), so a transient failure never overwrites a good prior record.

    Args:
        record: A record from :func:`run_backend_turn` / :func:`run_agent_turn`.
        out_dir: Directory to write the per-model JSON into (created if missing).
        only_ok: When ``True`` (default), skip records that did not produce an
            answer (``ok`` is falsy), returning ``None`` without writing.

    Returns:
        The written path, or ``None`` when the record was skipped.
    """
    if only_ok and not record.get("ok"):
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['model']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_persisted_records(
    models: Iterable[str],
    out_dir: Path | str,
) -> dict[str, dict[str, Any]]:
    """Load the persisted per-backend records for ``models`` (if present).

    Counterpart of :func:`save_backend_record`: reads back the records written by
    earlier one-at-a-time passes so the cross-backend table can show every
    backend's real result even though they never served simultaneously.

    Args:
        models: Reasoner model names to look for.
        out_dir: Directory the per-model JSON files were written to.

    Returns:
        A mapping ``model -> record`` for every model whose JSON exists and
        parses; models without a file are simply absent.
    """
    out = Path(out_dir)
    loaded: dict[str, dict[str, Any]] = {}
    for model in models:
        path = out / f"{model}.json"
        if path.is_file():
            try:
                loaded[model] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("demo_record_load_failed", model=model, error=str(exc))
    return loaded


def cross_backend_table(
    records: Sequence[dict[str, Any]],
    *,
    display: bool = True,
) -> list[dict[str, Any]]:
    """Summarise the per-backend turn records into one comparison table.

    The perceiver (the dense ensemble) is identical across backends by
    construction, so what varies is the reasoning over that text, the tool use and
    the latency. This surfaces exactly that, from the REAL records of
    :func:`run_backend_turn`.

    Args:
        records: The per-backend records.
        display: When ``True``, render the table.

    Returns:
        One row per backend: ``backend`` / ``disponible`` / ``respondio`` /
        ``n_herramientas`` / ``herramientas`` / ``latencia_ms`` / ``chars_respuesta``.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "backend": record["model"],
                "disponible": "si" if record.get("available") else "NO",
                "respondio": "si" if record.get("ok") else "no",
                "n_herramientas": record.get("n_tool_calls", 0),
                "herramientas": ", ".join(record.get("tool_calls") or []) or "-",
                "latencia_ms": record.get("latency_ms"),
                "chars_respuesta": len(record.get("answer") or ""),
            }
        )
    if display:
        _display_df(rows, tbl_width=200)
        ready = [r["model"] for r in records if r.get("ok")]
        _md(
            "Backends que completaron el turno: "
            + (", ".join(f"`{m}`" for m in ready) or "_ninguno_")
            + ". El perceiver entrega la misma clase/confianza a todos; el LLM solo "
            "razona sobre ese TEXTO (Be My Eyes)."
        )
    return rows


# ---------------------------------------------------------------------------
# Perceiver + RAG renderers
# ---------------------------------------------------------------------------
async def observe_parcels(
    perceiver: PerceiverLayer,
    parcel_ids: Sequence[int],
) -> list[tuple[PerceiverObservation, float]]:
    """Run the perceiver over each parcel, timing every call.

    Args:
        perceiver: The :class:`~ml.agent.perceiver.PerceiverLayer` for the session.
        parcel_ids: Stored parcel ids to observe.

    Returns:
        ``[(observation, latency_ms), ...]`` in ``parcel_ids`` order.
    """
    observations: list[tuple[PerceiverObservation, float]] = []
    for parcel_id in parcel_ids:
        start = time.perf_counter()
        obs = await perceiver.observe(int(parcel_id))
        observations.append((obs, round((time.perf_counter() - start) * 1000.0, 1)))
    return observations


def perceiver_table(
    observations: Sequence[tuple[PerceiverObservation, float]],
    *,
    display: bool = True,
) -> list[dict[str, Any]]:
    """Tabulate the structured TEXT fields the perceiver exposes (no tensors).

    Args:
        observations: ``[(observation, latency_ms), ...]`` from
            :func:`observe_parcels`.
        display: When ``True``, render the table.

    Returns:
        One row per observation: ``parcela`` / ``cultivo`` / ``confianza`` /
        ``vigor`` / ``latencia_ms``.
    """
    rows = [
        {
            "parcela": obs.parcel_id,
            "cultivo": obs.crop_class,
            "confianza": round(obs.confidence, 3),
            "vigor": obs.vigor,
            "latencia_ms": ms,
        }
        for obs, ms in observations
    ]
    if display:
        _display_df(rows)
    return rows


def rag_table(
    documents: Sequence[RAGDocument],
    *,
    display: bool = True,
) -> list[dict[str, Any]]:
    """Tabulate retrieved Spatial-RAG documents with their fused score + distance.

    Args:
        documents: The :class:`~ml.agent.rag.RAGDocument` list from
            :func:`ml.agent.rag.spatial_rag`.
        display: When ``True``, render the table (or a "no neighbours" note).

    Returns:
        One row per document: ``doc_id`` / ``fuente`` / ``parcela`` /
        ``distancia_m`` / ``score`` / ``contenido`` (truncated).
    """
    rows = [
        {
            "doc_id": doc.id,
            "fuente": doc.source,
            "parcela": doc.parcel_id,
            "distancia_m": round(doc.distance_m, 1) if doc.distance_m is not None else None,
            "score": round(doc.score, 4),
            "contenido": doc.content[:90] + ("..." if len(doc.content) > 90 else ""),
        }
        for doc in documents
    ]
    if display:
        if rows:
            _display_df(rows)
        else:
            _md("> No se hallaron vecinos dentro del radio. Aumenta `rag_radius_m` y reejecuta.")
    return rows
