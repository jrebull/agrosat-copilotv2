"""Agent factory and the manual function-calling loop (US-047).

This module is the reasoner orchestrator of the "Be My Eyes" agent. It owns the
manual function-calling loop required by ``google-genai`` 2.6 once the SDK's
*automatic* function calling is disabled (it must be, because the tools are async
and need the :class:`~ml.agent.context.ToolContext` for session-scoped DB access /
RLS, which automatic calling cannot thread through).

Loop shape
----------
:meth:`Agent.stream_response` drives the loop:

#. Build the ``contents`` list (``list[google.genai.types.Content]``) from the
   chat ``messages`` (roles ``user`` -> ``user`` and ``assistant``/``model`` ->
   ``model``).
#. Ask the backend to generate against the tool declarations and the system
   instruction. The backend (:mod:`ml.agent.backends`) abstracts Gemini vs the
   OpenAI-compatible vLLM endpoint and yields :class:`BackendChunk`-like chunks
   (text delta | function call | done).
#. If the turn produced function calls: for each one, inject the tenant
   ``session_id`` (never trusted from the model), validate the arguments with the
   tool's Pydantic ``input_model`` (invalid -> :class:`ErrorEvent`, skip), run
   ``await spec.fn(inp, ctx)`` under the shared context, emit
   :class:`ToolCallEvent` + :class:`ToolResultEvent`, and append the model's
   function-call turn plus the ``FunctionResponse`` turn to ``contents``. Then
   loop back to the backend.
#. When the turn is text-only, stream :class:`TextDeltaEvent` chunks and finish
   with :class:`DoneEvent`.

A hard :data:`MAX_TURNS` cap bounds the loop so a misbehaving model cannot spin
forever. Any backend or tool exception is surfaced as an :class:`ErrorEvent`
instead of crashing the async generator.

The module imports cleanly even before :mod:`ml.agent.backends` and
:mod:`ml.agent.prompts` exist (sibling sub-task ml/A): the dependency on
``make_backend`` / ``ANALYST_SYSTEM_PROMPT`` is resolved lazily inside
:func:`create_agent`, while the type-only imports are guarded by
``TYPE_CHECKING``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from google.genai import types
from pydantic import BaseModel, ValidationError

from ml.agent.context import ToolContext
from ml.agent.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ml.agent.tools import ToolSpec, build_function_declarations, get_sync_tools

if TYPE_CHECKING:
    from backend.app.core.config import Settings
    from ml.agent.backends import LLMBackend

logger = structlog.get_logger(__name__)

__all__ = ["MAX_TURNS", "Agent", "create_agent"]

#: Default reasoner model. Gemini 3.5 Flash on Vertex AI / the GenAI API (US-052
#: conscious deviation from ``gemini-2.5-pro`` for cost/latency); swapped per
#: request via ``/llm/switch`` without touching the factory (AC-5). The chat
#: service always passes the model explicitly, so this is the bare-call default.
DEFAULT_MODEL: str = "gemini-3.5-flash"

#: Hard cap on backend round-trips per ``stream_response`` call. Guards against a
#: model that keeps requesting tools forever. Each turn is one ``generate`` call;
#: the loop ends earlier as soon as a turn yields no function calls.
MAX_TURNS: int = 8

#: Field name of the tenant session id present on every tool ``*Input`` model.
#: The agent injects ``ctx.session_id`` here so the model never controls the
#: tenant boundary (multi-tenant safety); models that carry no such field are
#: left untouched.
_SESSION_ID_FIELD: str = "session_id"

#: Tool name of the Spatial-RAG grounding tool. It is synchronous (a fast pgvector
#: query) but is exposed in the loop ONLY when ``rag_enabled``, so the default agent
#: stays ungrounded exactly as before US-046 and only grounds itself on request.
_RAG_TOOL_NAME: str = "retrieve_context"


def _rag_enabled(settings: Settings | None) -> bool:
    """Return whether the Spatial-RAG grounding tool should be in the agent loop.

    Args:
        settings: Typed settings (``None`` in tests that do not build them).

    Returns:
        ``True`` when ``settings.rag_enabled`` is truthy; ``False`` otherwise.
    """
    return bool(getattr(settings, "rag_enabled", False)) if settings is not None else False


@dataclass
class _ToolCall:
    """A single function call requested by the model in one turn.

    Attributes:
        name: Tool name the model wants to invoke.
        args: Raw argument mapping produced by the model (pre-validation).
        call_id: Provider-supplied call identifier, if any (Gemini omits it).
        thought_signature: Opaque per-part signature Gemini 3.x attaches to a
            function-call part. It MUST be echoed back verbatim when the model's
            turn is rebuilt into ``contents`` for the next turn, or Gemini 3.x
            rejects the follow-up with ``400 INVALID_ARGUMENT``. ``None`` for
            backends/models that do not emit one (Qwen/vLLM and older Gemini),
            in which case the rebuilt part is unchanged from before.
    """

    name: str
    args: dict[str, Any]
    call_id: str | None = None
    thought_signature: bytes | None = None


class Agent:
    """Reasoner that drives the manual function-calling loop over the tools.

    The agent is backend-agnostic: it is constructed with an injected
    :class:`~ml.agent.backends.LLMBackend`, the list of :class:`ToolSpec` it may
    call, and the system instruction. :meth:`stream_response` yields the typed
    :data:`~ml.agent.events.AgentEvent` stream that :class:`ChatService`
    serialises to SSE.
    """

    def __init__(
        self,
        backend: LLMBackend,
        tools: list[ToolSpec],
        instruction: str,
    ) -> None:
        """Initialise the agent with its backend, tools and system instruction.

        Args:
            backend: LLM backend abstraction (Gemini or vLLM); injected so the
                model variant is swappable without touching the loop.
            tools: Tools the reasoner may call, in declaration order.
            instruction: System instruction (the analyst "Be My Eyes" prompt).
        """
        self.backend = backend
        self.tools = tools
        self.instruction = instruction
        # Name -> spec for O(1) dispatch and the function declarations the
        # backend advertises to the model, both derived once at construction.
        self._tools_by_name: dict[str, ToolSpec] = {spec.name: spec for spec in tools}
        self._declarations: list[types.FunctionDeclaration] = self._build_declarations(tools)

    @staticmethod
    def _build_declarations(tools: list[ToolSpec]) -> list[types.FunctionDeclaration]:
        """Build the ``FunctionDeclaration`` list advertised to the model.

        Reuses the registry's :func:`build_function_declarations` (single source
        of truth for the Pydantic-derived schemas) and keeps only the
        declarations whose tool is in ``tools`` so the model is never offered a
        tool the agent will not execute.

        Args:
            tools: The tools this agent exposes.

        Returns:
            The matching declarations, in the order of ``tools``.
        """
        by_name = {decl.name: decl for decl in build_function_declarations()}
        return [by_name[spec.name] for spec in tools if spec.name in by_name]

    async def stream_response(
        self,
        messages: list[dict],
        session_id: UUID,
        ctx: ToolContext,
    ) -> AsyncIterator[AgentEvent]:
        """Run the function-calling loop and yield the typed event stream.

        Args:
            messages: Chat history as ``{"role": ..., "content": ...}`` dicts;
                roles ``user`` and ``assistant``/``model`` are mapped to the
                backend's ``user`` / ``model`` roles.
            session_id: Tenant session; injected into every tool's arguments and
                used for structured logging. Must match ``ctx.session_id``.
            ctx: Shared tool execution context (asyncpg pool, settings, session,
                defer hook) threaded into every tool's ``run`` coroutine.

        Yields:
            :data:`~ml.agent.events.AgentEvent` values: ``tool_call`` /
            ``tool_result`` per executed tool, ``text_delta`` chunks of the final
            answer, then a terminal ``done`` (or ``error`` on failure).
        """
        start = time.perf_counter()
        logger.info(
            "agent_turn_started",
            session_id=str(session_id),
            n_messages=len(messages),
            n_tools=len(self.tools),
            model=getattr(self.backend, "model", None),
        )

        contents: list[types.Content] = self._contents_from_messages(messages)

        try:
            async for event in self._run_loop(contents, session_id, ctx):
                yield event
        except Exception as exc:  # last-resort guard for the stream
            logger.exception(
                "agent_turn_failed",
                session_id=str(session_id),
                error=str(exc),
            )
            yield ErrorEvent(message=f"agent loop failed: {exc}")
            return

        logger.info(
            "agent_turn_finished",
            session_id=str(session_id),
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    async def _run_loop(
        self,
        contents: list[types.Content],
        session_id: UUID,
        ctx: ToolContext,
    ) -> AsyncIterator[AgentEvent]:
        """Drive the bounded backend round-trips until a text-only answer.

        On each turn the backend is asked to generate; the chunks are split into
        text deltas (buffered) and function calls. Any buffered text is streamed
        out as :class:`TextDeltaEvent` for the turn (even when the same turn also
        requested tools, so the model's interleaved reasoning is never lost). If
        the turn requested tools, they are executed -- with the model's text kept
        in the reconstructed ``model`` content -- and their responses fed back;
        otherwise the loop terminates with :class:`DoneEvent`.

        Args:
            contents: The conversation contents accumulated so far (mutated in
                place as tool turns are appended).
            session_id: Tenant session for argument injection and logging.
            ctx: Shared tool execution context.

        Yields:
            The events produced across the turns (tool calls/results, text
            deltas, and the terminal done/error event).
        """
        usage: dict[str, int] | None = None
        for turn in range(MAX_TURNS):
            text_parts: list[str] = []
            tool_calls: list[_ToolCall] = []

            async for chunk in self.backend.generate_stream(
                contents=contents,
                tools=self._declarations,
                system_instruction=self.instruction,
            ):
                text, call = self._read_chunk(chunk)
                if text:
                    text_parts.append(text)
                if call is not None:
                    tool_calls.append(call)
                # Carry the provider's token accounting forward when a chunk
                # reports it (FinOps, US-065). Only the last turn's usage reaches
                # ``DoneEvent``; ``None`` stays ``None`` (never synthesised).
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

            # Stream any text the model emitted this turn *before* deciding what
            # to do with it. Gemini and Qwen frequently interleave reasoning text
            # with the function calls in the same turn; that text must reach the
            # SSE stream (and the persisted history) instead of being dropped when
            # the turn also requested tools.
            for piece in text_parts:
                if piece:
                    yield TextDeltaEvent(text=piece)

            if not tool_calls:
                logger.info(
                    "agent_loop_completed",
                    session_id=str(session_id),
                    turns=turn + 1,
                )
                yield DoneEvent(usage=usage)
                return

            # Record the model's turn (its reasoning text plus the function calls)
            # before the responses so the backend sees a well-formed call/response
            # pairing on the next turn and the history keeps the model's text.
            contents.append(self._model_function_call_content(tool_calls, text_parts))
            response_parts: list[types.Part] = []
            for call in tool_calls:
                async for event in self._execute_tool(call, session_id, ctx, response_parts):
                    yield event
            contents.append(types.Content(role="tool", parts=response_parts))

        # Exhausted the turn budget without a final text answer.
        logger.warning(
            "agent_loop_max_turns",
            session_id=str(session_id),
            max_turns=MAX_TURNS,
        )
        yield ErrorEvent(
            message=(
                f"el agente no produjo una respuesta final tras {MAX_TURNS} turnos de herramientas"
            )
        )

    async def _execute_tool(
        self,
        call: _ToolCall,
        session_id: UUID,
        ctx: ToolContext,
        response_parts: list[types.Part],
    ) -> AsyncIterator[AgentEvent]:
        """Validate, run one tool call and append its ``FunctionResponse``.

        The tenant ``session_id`` is injected before validation so the model can
        never widen the tenant boundary. Validation and execution failures are
        surfaced as controlled events and as a ``{"error": ...}`` function
        response (so the model can recover) -- never as an exception that aborts
        the stream.

        Args:
            call: The function call requested by the model.
            session_id: Tenant session injected into the tool arguments.
            ctx: Shared tool execution context.
            response_parts: Accumulator the function-response part is appended to
                (mutated in place) for the single ``tool`` content of this turn.

        Yields:
            A :class:`ToolCallEvent` then a :class:`ToolResultEvent`, or an
            :class:`ErrorEvent` when the arguments fail validation.
        """
        spec = self._tools_by_name.get(call.name)
        if spec is None:
            message = f"el modelo pidio una herramienta desconocida: {call.name!r}"
            logger.warning("agent_unknown_tool", session_id=str(session_id), tool=call.name)
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"error": message})
            )
            yield ErrorEvent(message=message)
            return

        # Inject the tenant session before validation; the model never supplies
        # it. Only set it when the input model declares the field. The raw
        # ``UUID`` (not its string form) is injected because the tool inputs use
        # ``strict=True`` and reject a ``str`` for a ``UUID`` field.
        raw_args = dict(call.args)
        if _SESSION_ID_FIELD in spec.input_model.model_fields:
            raw_args[_SESSION_ID_FIELD] = session_id

        try:
            inp = spec.input_model.model_validate(raw_args)
        except ValidationError as exc:
            message = f"argumentos invalidos para {call.name}: {exc.error_count()} error(es)"
            logger.warning(
                "agent_tool_args_invalid",
                session_id=str(session_id),
                tool=call.name,
                errors=exc.error_count(),
            )
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"error": message})
            )
            yield ErrorEvent(message=message)
            return

        yield ToolCallEvent(
            name=call.name,
            arguments=self._jsonable(raw_args),
            call_id=call.call_id,
        )

        tool_start = time.perf_counter()
        try:
            output = await spec.fn(inp, ctx)
        except Exception as exc:  # controlled per-tool failure
            duration_ms = round((time.perf_counter() - tool_start) * 1000.0, 2)
            message = f"la herramienta {call.name} fallo: {exc}"
            logger.exception(
                "agent_tool_call_failed",
                session_id=str(session_id),
                tool=call.name,
                duration_ms=duration_ms,
                error=str(exc),
            )
            error_payload = {"error": message}
            response_parts.append(
                types.Part.from_function_response(name=call.name, response=error_payload)
            )
            yield ToolResultEvent(name=call.name, result=error_payload, ok=False)
            return

        duration_ms = round((time.perf_counter() - tool_start) * 1000.0, 2)
        result_payload = self._dump_output(output)
        logger.info(
            "agent_tool_call",
            session_id=str(session_id),
            tool=call.name,
            deferred=spec.deferred,
            duration_ms=duration_ms,
        )
        response_parts.append(
            types.Part.from_function_response(name=call.name, response=result_payload)
        )
        yield ToolResultEvent(name=call.name, result=result_payload, ok=True)

    # ------------------------------------------------------------------
    # contents / chunk helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _contents_from_messages(messages: list[dict]) -> list[types.Content]:
        """Map chat messages to ``google-genai`` ``Content`` turns.

        ``user`` messages become ``role="user"``; ``assistant``/``model``
        messages become ``role="model"``. Any other role (e.g. an injected
        ``system``/``perceiver`` grounding block) is folded into the ``user``
        role so the backend still sees its text as context. Empty-content
        messages are skipped.

        Args:
            messages: Chat history dicts with ``role`` and ``content`` keys.

        Returns:
            The equivalent list of :class:`google.genai.types.Content`.
        """
        contents: list[types.Content] = []
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            role = "model" if message.get("role") in {"assistant", "model"} else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=str(content))])
            )
        return contents

    @staticmethod
    def _model_function_call_content(
        tool_calls: list[_ToolCall], text_parts: list[str] | None = None
    ) -> types.Content:
        """Render the model's mixed turn as a single ``model`` content.

        The reasoning text the model emitted alongside its function calls is kept
        as leading text parts so the reconstructed ``Content`` mirrors what the
        model actually produced. Without it, the model's reasoning would vanish
        from the history and the next backend turn would see a call/response
        pairing stripped of its rationale.

        Args:
            tool_calls: The calls requested by the model in this turn.
            text_parts: The interleaved reasoning-text deltas emitted in the same
                turn, in order; empty/absent when the turn was pure function calls.

        Returns:
            A ``role="model"`` :class:`google.genai.types.Content` whose parts are
            the (non-empty) text deltas followed by the function calls, mirroring
            what the model emitted so the backend sees a coherent call/response
            pairing and the history keeps the model's reasoning. Each function-call
            part re-attaches its ``thought_signature`` when the model emitted one
            (Gemini 3.x), since the API requires it echoed back verbatim on the
            next turn; calls without a signature (Qwen/vLLM, older Gemini) are
            rebuilt exactly as before.
        """
        parts: list[types.Part] = [
            types.Part.from_text(text=piece) for piece in (text_parts or []) if piece
        ]
        for call in tool_calls:
            part = types.Part.from_function_call(name=call.name, args=call.args)
            if call.thought_signature is not None:
                # Round-trip Gemini 3.x's opaque signature verbatim; required for
                # the follow-up request to be accepted (otherwise 400).
                part.thought_signature = call.thought_signature
            parts.append(part)
        return types.Content(role="model", parts=parts)

    @staticmethod
    def _read_chunk(chunk: Any) -> tuple[str | None, _ToolCall | None]:
        """Extract a text delta and/or a function call from a backend chunk.

        The backend abstraction yields duck-typed chunks (``BackendChunk``):
        ``chunk.text`` carries an incremental text delta and
        ``chunk.function_call`` carries a requested call (with ``name`` / ``args``
        and an optional ``id``). Either may be absent on a given chunk.

        Args:
            chunk: One chunk yielded by ``LLMBackend.generate_stream``.

        Returns:
            ``(text, call)`` where each element is ``None`` when not present.
        """
        text = getattr(chunk, "text", None)
        call_obj = getattr(chunk, "function_call", None)
        call: _ToolCall | None = None
        if call_obj is not None:
            name = getattr(call_obj, "name", None)
            if name:
                args = getattr(call_obj, "args", None) or {}
                call_id = getattr(call_obj, "id", None) or getattr(call_obj, "call_id", None)
                # Carry Gemini 3.x's per-part thought_signature through so the
                # rebuilt model turn can echo it back (required for multi-turn
                # tool calls). Absent on non-Gemini backends -> None.
                sig = getattr(call_obj, "thought_signature", None)
                call = _ToolCall(name=name, args=dict(args), call_id=call_id, thought_signature=sig)
        return (text if text else None), call

    @staticmethod
    def _dump_output(output: Any) -> dict:
        """Serialise a tool output into a JSON-able mapping for the wire/model.

        Pydantic models are dumped in JSON mode (dates -> ISO strings, etc.);
        plain mappings pass through; anything else is wrapped under ``"result"``.

        Args:
            output: The value returned by a tool's ``run`` coroutine.

        Returns:
            A JSON-serialisable mapping.
        """
        if isinstance(output, BaseModel):
            return output.model_dump(mode="json")
        if isinstance(output, dict):
            return output
        return {"result": output}

    @staticmethod
    def _jsonable(args: dict[str, Any]) -> dict:
        """Coerce raw tool arguments into a JSON-serialisable mapping.

        Used for the ``ToolCallEvent.arguments`` payload so the SSE wire only ever
        carries plain scalars/mappings (e.g. the injected ``UUID`` session id is
        already a string). Non-serialisable values are stringified defensively.

        Args:
            args: The raw (post-injection) argument mapping.

        Returns:
            A JSON-serialisable copy of ``args``.
        """
        jsonable: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str | int | float | bool | type(None) | list | dict):
                jsonable[key] = value
            else:
                jsonable[key] = str(value)
        return jsonable


def create_agent(
    model: str = DEFAULT_MODEL,
    tools: list[ToolSpec] | None = None,
    instruction: str | None = None,
    settings: Settings | None = None,
) -> Agent:
    """Build an :class:`Agent` wired to a backend selected by model name.

    The model variant is injected (AC-5): ``"gemini-*"`` resolves to the Gemini
    backend and ``"qwen*"`` to the OpenAI-compatible vLLM backend via
    :func:`ml.agent.backends.make_backend`, without the caller touching the loop.
    The default tool set is the 5 synchronous demo tools (the deferred tools need
    the Pub/Sub ``defer`` hook to be wired); the default instruction is the
    analyst "Be My Eyes" system prompt.

    Args:
        model: Reasoner model name (default :data:`DEFAULT_MODEL`).
        tools: Tools to expose; defaults to :func:`get_sync_tools`.
        instruction: System instruction; defaults to ``ANALYST_SYSTEM_PROMPT``.
        settings: Typed settings injected into the backend (API key, vLLM URL);
            ``None`` lets the backend read them from configuration.

    Returns:
        A ready-to-stream :class:`Agent`.
    """
    # Imported lazily so this module stays importable before sibling sub-task
    # ml/A (``backends.py`` / ``prompts.py``) lands, and so tests can stub the
    # backend without importing the real ``google-genai`` client.
    from ml.agent.backends import make_backend
    from ml.agent.prompts import ANALYST_SYSTEM_PROMPT

    resolved_tools = tools if tools is not None else get_sync_tools()
    # Spatial-RAG grounding (US-046 ``retrieve_context``) is in the loop ONLY when
    # ``rag_enabled``, so the default copilot is unchanged; when on, the reasoner can
    # call it to ground its answer in cited corpus evidence (anti-hallucination).
    if tools is None and not _rag_enabled(settings):
        resolved_tools = [spec for spec in resolved_tools if spec.name != _RAG_TOOL_NAME]
    resolved_instruction = instruction if instruction is not None else ANALYST_SYSTEM_PROMPT
    backend = make_backend(model, settings)

    logger.info(
        "agent_created",
        model=model,
        n_tools=len(resolved_tools),
        tool_names=[spec.name for spec in resolved_tools],
    )
    return Agent(
        backend=backend,
        tools=resolved_tools,
        instruction=resolved_instruction,
    )
