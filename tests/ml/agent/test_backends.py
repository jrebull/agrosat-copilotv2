"""Tests for the dual LLM backends (US-047, sub-task ml/A).

:mod:`ml.agent.backends` abstracts the reasoner LLM behind a single
``LLMBackend.generate_stream(contents, tools, system_instruction)`` coroutine that
yields duck-typed ``BackendChunk`` items (``.text`` deltas and/or
``.function_call`` with ``.name`` / ``.args``). Two concrete backends exist:

* :class:`GeminiBackend` over the ``google-genai`` client (``gemini-*`` models);
* :class:`VLLMOpenAIBackend` over an OpenAI-compatible endpoint (Qwen on vLLM,
  US-048).

Both are exercised with their network client fully **mocked** -- the genai
``Client`` is replaced by a fake whose ``models.generate_content`` /
``generate_content_stream`` return scripted responses, and the OpenAI/vLLM HTTP
client is replaced likewise. Zero real network. :func:`make_backend` routing is
asserted by model-name prefix.

The whole module is skipped with a clear reason until ``ml.agent.backends`` lands
(sibling sub-task ml/A): writing the tests against the planning's section-4
contract now means they go green the moment the module exists, without polluting
the suite while it does not.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

# ml/A ('backends.py') may not have landed yet on this branch. Skip the whole
# module (with an explicit reason) rather than erroring at collection time.
backends = pytest.importorskip(
    "ml.agent.backends",
    reason="ml.agent.backends (sub-task ml/A) not implemented yet",
)

from google.genai import types  # noqa: E402  (after importorskip on purpose)

from tests.ml.agent.conftest import FakeSettings  # noqa: E402


# ---------------------------------------------------------------------------
# google-genai client doubles
# ---------------------------------------------------------------------------
@dataclass
class FakeGenAIResponse:
    """Stand-in for a ``google-genai`` ``GenerateContentResponse``.

    Exposes the two attributes :class:`GeminiBackend` reads off a non-streaming
    response: ``function_calls`` (the parsed calls) and ``text`` (the answer).
    """

    function_calls: list[types.FunctionCall] = field(default_factory=list)
    text: str | None = None


@dataclass
class FakeGenAIStreamChunk:
    """One streamed ``google-genai`` chunk (text delta and/or function call)."""

    text: str | None = None
    function_calls: list[types.FunctionCall] = field(default_factory=list)


class FakeModels:
    """Double for ``client.models`` recording the config it was called with."""

    def __init__(
        self,
        *,
        response: FakeGenAIResponse | None = None,
        stream_chunks: list[FakeGenAIStreamChunk] | None = None,
    ) -> None:
        self._response = response or FakeGenAIResponse()
        self._stream_chunks = stream_chunks or []
        # ``calls`` records every model call (streaming or not); ``stream_calls``
        # counts only the streaming surface so the B-4 single-generation contract
        # can assert the streaming generation is never invoked for a text turn.
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: int = 0

    def generate_content(self, **kwargs: Any) -> FakeGenAIResponse:
        """Record the call and return the scripted non-streaming response."""
        self.calls.append(kwargs)
        return self._response

    def generate_content_stream(self, **kwargs: Any):
        """Record the call and yield the scripted stream chunks (sync iterator)."""
        self.calls.append(kwargs)
        self.stream_calls += 1
        yield from self._stream_chunks


class FakeGenAIClient:
    """Minimal ``google.genai.Client`` double exposing ``.models``."""

    def __init__(self, models: FakeModels) -> None:
        self.models = models


def _make_function_call(name: str, args: dict[str, Any]) -> types.FunctionCall:
    """Build a real ``types.FunctionCall`` (the SDK type the backend parses)."""
    return types.FunctionCall(name=name, args=args)


async def _drain(backend: Any) -> list[Any]:
    """Drive ``generate_stream`` with throwaway args and collect the chunks."""
    declarations = backends_declarations()
    chunks = [
        chunk
        async for chunk in backend.generate_stream(
            contents=[types.Content(role="user", parts=[types.Part.from_text(text="hola")])],
            tools=declarations,
            system_instruction="system-prompt",
        )
    ]
    return chunks


def backends_declarations() -> list[types.FunctionDeclaration]:
    """A single declaration; the backends only forward it to the client."""
    return [
        types.FunctionDeclaration(
            name="list_parcels",
            description="list parcels",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        )
    ]


# ---------------------------------------------------------------------------
# genai Schema -> JSON Schema normaliser (Bug 1: llama.cpp 400 on UPPERCASE types)
# ---------------------------------------------------------------------------
def test_genai_schema_lowercases_nested_types() -> None:
    """``_genai_schema_to_json_schema`` lowercases every nested ``type`` value.

    The ``google.genai`` ``Schema.model_dump`` emits the genai enum names in
    UPPERCASE (``OBJECT``/``STRING``/``ARRAY``/...), which the OpenAI-compatible
    vLLM/llama.cpp grammar compiler rejects with a ``400``. The normaliser must
    recurse into ``properties`` and ``items`` and lowercase each ``type`` while
    preserving ``description``/``enum``/``required``/``nullable``.
    """
    raw = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING", "description": "n", "enum": ["a", "b"]},
            "limit": {"type": "INTEGER"},
            "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
            "flag": {"type": "BOOLEAN", "nullable": True},
        },
        "required": ["name"],
    }

    out = backends._genai_schema_to_json_schema(raw)

    assert out["type"] == "object"
    props = out["properties"]
    assert props["name"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"  # nested items recursed
    assert props["flag"]["type"] == "boolean"
    # Non-type annotations preserved verbatim.
    assert props["name"]["description"] == "n"
    assert props["name"]["enum"] == ["a", "b"]
    assert props["flag"]["nullable"] is True
    assert out["required"] == ["name"]


def test_genai_schema_normaliser_is_defensive() -> None:
    """The normaliser tolerates missing keys, lowercase types and non-dict nodes."""
    # Already-lowercase + missing properties: returned equivalent, not mangled.
    assert backends._genai_schema_to_json_schema({"type": "object"}) == {"type": "object"}
    # Non-dict node returned untouched.
    assert backends._genai_schema_to_json_schema("STRING") == "STRING"
    assert backends._genai_schema_to_json_schema(None) is None
    # anyOf list of schemas is recursed.
    out = backends._genai_schema_to_json_schema({"anyOf": [{"type": "STRING"}, {"type": "NULL"}]})
    assert [s["type"] for s in out["anyOf"]] == ["string", "null"]


def test_tools_from_declarations_emit_lowercase_json_schema() -> None:
    """The OpenAI tool envelope carries a lowercase-typed JSON Schema (Bug 1).

    Before the fix the genai dump leaked ``"type": "OBJECT"`` into the
    ``function.parameters`` sent to llama.cpp, which 400'd. The envelope shape
    (``type``/``function`` keys) is unchanged; only the schema is normalised.
    """
    decl = types.FunctionDeclaration(
        name="list_parcels",
        description="list parcels",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"limit": types.Schema(type=types.Type.INTEGER)},
        ),
    )

    tools = backends.VLLMOpenAIBackend._tools_from_declarations([decl])

    assert tools[0]["type"] == "function"
    fn = tools[0]["function"]
    assert fn["name"] == "list_parcels"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["limit"]["type"] == "integer"
    # No UPPERCASE genai type names survive anywhere in the schema.
    assert "OBJECT" not in json.dumps(params)
    assert "INTEGER" not in json.dumps(params)


# ---------------------------------------------------------------------------
# make_backend routing
# ---------------------------------------------------------------------------
def test_make_backend_gemini_returns_gemini_backend() -> None:
    """A ``gemini-*`` model name routes to :class:`GeminiBackend`."""
    backend = backends.make_backend("gemini-2.5-pro", FakeSettings())  # type: ignore[arg-type]
    assert isinstance(backend, backends.GeminiBackend)
    assert backend.model == "gemini-2.5-pro"


def test_make_backend_gemini_flash_variant() -> None:
    """The flash variant also routes to :class:`GeminiBackend` (AC-5 swap)."""
    backend = backends.make_backend("gemini-2.5-flash", FakeSettings())  # type: ignore[arg-type]
    assert isinstance(backend, backends.GeminiBackend)
    assert backend.model == "gemini-2.5-flash"


def test_make_backend_qwen_returns_vllm_backend() -> None:
    """A ``qwen*`` model name routes to :class:`VLLMOpenAIBackend`."""
    backend = backends.make_backend("qwen35", FakeSettings())  # type: ignore[arg-type]
    assert isinstance(backend, backends.VLLMOpenAIBackend)


# ---------------------------------------------------------------------------
# GeminiBackend: function-call parsing + text streaming
# ---------------------------------------------------------------------------
async def test_gemini_backend_parses_function_calls() -> None:
    """A response carrying ``function_calls`` is surfaced as call chunks.

    The mocked client returns a response with one ``FunctionCall``; the backend
    must yield at least one chunk whose ``function_call`` exposes ``name`` and
    ``args`` matching what the model requested.
    """
    call = _make_function_call("list_parcels", {"limit": 5})
    models = FakeModels(response=FakeGenAIResponse(function_calls=[call]))
    backend = backends.GeminiBackend(model="gemini-2.5-pro", client=FakeGenAIClient(models))

    chunks = await _drain(backend)

    parsed = [c.function_call for c in chunks if getattr(c, "function_call", None)]
    assert parsed, "expected at least one function-call chunk"
    assert parsed[0].name == "list_parcels"
    assert dict(parsed[0].args) == {"limit": 5}


@dataclass
class _CandidateContent:
    """Stand-in for ``candidate.content`` exposing ``parts``."""

    parts: list[types.Part] = field(default_factory=list)


@dataclass
class _Candidate:
    """Stand-in for one ``response.candidates[i]`` exposing ``content``."""

    content: _CandidateContent | None = None


@dataclass
class FakeGenAIResponseWithParts:
    """Response double that exposes BOTH the flat ``function_calls`` surface and
    the per-candidate ``parts`` carrying Gemini 3.x ``thought_signature`` (the
    signature lives on the part, not on the flat ``FunctionCall``).
    """

    function_calls: list[types.FunctionCall] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    text: str | None = None


async def test_gemini_backend_captures_thought_signature_from_part() -> None:
    """Gemini 3.x signature on the function-call PART reaches the chunk (bug fix).

    ``response.function_calls`` flattens the parts and drops the per-part
    ``thought_signature``; the backend must recover it from
    ``candidates[0].content.parts`` and expose it on the emitted
    :class:`BackendFunctionCall` so the agent can echo it back next turn.
    """
    sig = b"opaque-signature-bytes"
    part = types.Part.from_function_call(name="classify_new_parcel", args={"id": 7})
    part.thought_signature = sig
    flat_call = _make_function_call("classify_new_parcel", {"id": 7})
    response = FakeGenAIResponseWithParts(
        function_calls=[flat_call],
        candidates=[_Candidate(content=_CandidateContent(parts=[part]))],
    )
    backend = backends.GeminiBackend(
        model="gemini-3.5-pro",
        client=FakeGenAIClient(FakeModels(response=response)),  # type: ignore[arg-type]
    )

    chunks = await _drain(backend)

    parsed = [c.function_call for c in chunks if getattr(c, "function_call", None)]
    assert parsed, "expected at least one function-call chunk"
    assert parsed[0].name == "classify_new_parcel"
    assert parsed[0].thought_signature == sig


async def test_gemini_backend_thought_signature_none_without_part() -> None:
    """No candidate parts (flat double / older Gemini) -> signature is ``None``.

    Back-compat: a response exposing only the flat ``function_calls`` surface must
    still parse, with ``thought_signature`` defaulting to ``None`` so the Qwen and
    legacy-Gemini paths are unchanged.
    """
    call = _make_function_call("list_parcels", {"limit": 5})
    models = FakeModels(response=FakeGenAIResponse(function_calls=[call]))
    backend = backends.GeminiBackend(model="gemini-2.5-pro", client=FakeGenAIClient(models))

    chunks = await _drain(backend)

    parsed = [c.function_call for c in chunks if getattr(c, "function_call", None)]
    assert parsed, "expected at least one function-call chunk"
    assert parsed[0].thought_signature is None


async def test_gemini_backend_streams_text_chunks() -> None:
    """A text-only response is surfaced from the SINGLE non-streaming call (B-4).

    Contract (B-4 fix): a turn is resolved with one ``generate_content`` call.
    The answer text is re-emitted from that response (via ``_chunks_from_response``)
    instead of running a second streaming generation, so cost/latency stay at one
    model call and the text shown is exactly the one inspected for tool calls.

    The stream chunks scripted here are intentionally DIFFERENT from
    ``response.text`` to prove the output comes from the non-streaming response,
    not from a second streaming generation.
    """
    models = FakeModels(
        stream_chunks=[
            FakeGenAIStreamChunk(text="STREAM-SHOULD-NOT-BE-USED"),
        ],
        response=FakeGenAIResponse(text="Hola, mundo."),
    )
    backend = backends.GeminiBackend(model="gemini-2.5-pro", client=FakeGenAIClient(models))

    chunks = await _drain(backend)

    text = "".join(c.text for c in chunks if getattr(c, "text", None))
    assert text == "Hola, mundo."
    assert not any(getattr(c, "function_call", None) for c in chunks)


async def test_gemini_backend_single_generation_no_tool_calls() -> None:
    """Regression for B-4: a text turn must call the model exactly ONCE.

    The previous implementation ran ``_generate`` (non-stream) to detect function
    calls and THEN ``_stream_text`` (a second full generation) to surface the
    text, doubling cost/latency and risking an inconsistent sample. The fix
    re-emits the text already present on the non-streaming response, so the
    streaming surface must never be invoked.
    """
    models = FakeModels(
        stream_chunks=[FakeGenAIStreamChunk(text="unused")],
        response=FakeGenAIResponse(text="respuesta unica"),
    )
    backend = backends.GeminiBackend(model="gemini-2.5-pro", client=FakeGenAIClient(models))

    chunks = await _drain(backend)

    text = "".join(c.text for c in chunks if getattr(c, "text", None))
    assert text == "respuesta unica"
    # Exactly one model call total, and it is the non-streaming surface.
    assert len(models.calls) == 1, "the text turn must hit the model exactly once"
    assert models.stream_calls == 0, "the streaming surface must not be invoked"


async def test_gemini_backend_disables_automatic_function_calling() -> None:
    """The backend must pass a config that disables automatic function calling.

    The manual loop (US-047) requires ``automatic_function_calling.disable=True``;
    the backend builds the ``GenerateContentConfig`` and the system instruction,
    so we inspect the config object handed to the client.
    """
    models = FakeModels(
        stream_chunks=[FakeGenAIStreamChunk(text="ok")],
        response=FakeGenAIResponse(text="ok"),
    )
    backend = backends.GeminiBackend(model="gemini-2.5-pro", client=FakeGenAIClient(models))

    await _drain(backend)

    assert models.calls, "the backend never called the genai client"
    config = models.calls[-1].get("config")
    assert config is not None, "the backend must pass a GenerateContentConfig"
    afc = getattr(config, "automatic_function_calling", None)
    assert afc is not None and afc.disable is True
    # The system instruction is threaded through to the model.
    assert getattr(config, "system_instruction", None) == "system-prompt"


# ---------------------------------------------------------------------------
# VLLMOpenAIBackend: OpenAI-compatible parsing (HTTP fully mocked)
# ---------------------------------------------------------------------------
def _vllm_backend_with_client(client: Any) -> Any:
    """Build a :class:`VLLMOpenAIBackend`, injecting the mocked OpenAI client.

    The backend's constructor signature varies (it may accept a ``client`` kwarg
    or build its own from ``base_url``); we set the attribute the backend uses to
    talk to the endpoint so no real HTTP happens regardless.
    """
    backend = backends.VLLMOpenAIBackend(
        base_url="http://vllm.invalid/v1", model="qwen35", api_key="EMPTY"
    )
    # Inject the fake OpenAI/HTTP client on whichever private attr the backend
    # holds it in (``_client`` by convention).
    for attr in ("_client", "client", "_openai", "_aclient"):
        if hasattr(backend, attr):
            setattr(backend, attr, client)
            break
    else:  # pragma: no cover - defensive: contract changed
        backend._client = client  # type: ignore[attr-defined]
    return backend


@dataclass
class _OAFunction:
    name: str | None = None
    arguments: str = ""


@dataclass
class _OAToolCall:
    index: int = 0
    id: str | None = None
    function: _OAFunction = field(default_factory=_OAFunction)
    type: str = "function"


@dataclass
class _OADelta:
    content: str | None = None
    tool_calls: list[_OAToolCall] | None = None


@dataclass
class _OAChoice:
    delta: _OADelta = field(default_factory=_OADelta)
    finish_reason: str | None = None


@dataclass
class _OAUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class _OAChunk:
    choices: list[_OAChoice] = field(default_factory=list)
    usage: _OAUsage | None = None


class _FakeAsyncStream:
    """Async iterator over scripted OpenAI streaming chunks."""

    def __init__(self, chunks: list[_OAChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, chunks: list[_OAChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        """Return an async stream of the scripted chunks (mirrors AsyncOpenAI)."""
        self.calls.append(kwargs)
        return _FakeAsyncStream(self._chunks)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Minimal ``AsyncOpenAI`` double exposing ``chat.completions.create``."""

    def __init__(self, chunks: list[_OAChunk]) -> None:
        self.completions = _FakeCompletions(chunks)
        self.chat = _FakeChat(self.completions)


async def _drain_vllm(backend: Any) -> list[Any]:
    """Drive the vLLM backend's ``generate_stream`` and collect chunks."""
    return await _drain(backend)


async def test_vllm_backend_streams_text() -> None:
    """OpenAI-format ``delta.content`` chunks become text-delta chunks."""
    client = FakeOpenAIClient(
        [
            _OAChunk(choices=[_OAChoice(delta=_OADelta(content="Hola"))]),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(content=" Qwen"))]),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(), finish_reason="stop")]),
        ]
    )
    backend = _vllm_backend_with_client(client)

    chunks = await _drain_vllm(backend)
    text = "".join(c.text for c in chunks if getattr(c, "text", None))
    assert text == "Hola Qwen"


async def test_vllm_backend_requests_include_usage() -> None:
    """The streaming request carries ``stream_options={"include_usage": True}``.

    Without it, vLLM never emits the ``usage`` chunk and the FinOps observability
    of US-065 logs ``None`` tokens for Qwen (B6).
    """
    client = FakeOpenAIClient([_OAChunk(choices=[_OAChoice(delta=_OADelta(content="hola"))])])
    backend = _vllm_backend_with_client(client)

    await _drain_vllm(backend)
    kwargs = client.completions.calls[0]
    assert kwargs.get("stream") is True
    assert kwargs.get("stream_options") == {"include_usage": True}


async def test_vllm_backend_surfaces_streamed_usage() -> None:
    """The final ``usage`` chunk is forwarded onto the last emitted chunk (B6).

    With ``include_usage`` the endpoint emits a trailing chunk with empty
    ``choices`` and a populated ``usage``; the backend reads it into the neutral
    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` mapping so the
    agent forwards real token counts onto the terminal ``DoneEvent``.
    """
    client = FakeOpenAIClient(
        [
            _OAChunk(choices=[_OAChoice(delta=_OADelta(content="Hola"))]),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(), finish_reason="stop")]),
            _OAChunk(
                choices=[],
                usage=_OAUsage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            ),
        ]
    )
    backend = _vllm_backend_with_client(client)

    chunks = await _drain_vllm(backend)
    usages = [c.usage for c in chunks if getattr(c, "usage", None)]
    assert usages, "expected the streamed usage to be surfaced"
    assert usages[-1] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


async def test_vllm_backend_usage_on_tool_call_chunk() -> None:
    """Usage rides the LAST tool-call chunk when the turn ends in tool calls."""
    client = FakeOpenAIClient(
        [
            _OAChunk(
                choices=[
                    _OAChoice(
                        delta=_OADelta(
                            tool_calls=[
                                _OAToolCall(
                                    index=0,
                                    id="call_1",
                                    function=_OAFunction(name="list_parcels", arguments="{}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(), finish_reason="tool_calls")]),
            _OAChunk(
                choices=[],
                usage=_OAUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
            ),
        ]
    )
    backend = _vllm_backend_with_client(client)

    chunks = await _drain_vllm(backend)
    call_chunks = [c for c in chunks if getattr(c, "function_call", None)]
    assert call_chunks, "expected the parsed tool call"
    assert call_chunks[-1].usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }


async def test_vllm_backend_no_usage_when_server_omits_it() -> None:
    """A server that does not report usage yields ``None`` (no synthesis)."""
    client = FakeOpenAIClient(
        [
            _OAChunk(choices=[_OAChoice(delta=_OADelta(content="Hola"))]),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(), finish_reason="stop")]),
        ]
    )
    backend = _vllm_backend_with_client(client)

    chunks = await _drain_vllm(backend)
    assert all(getattr(c, "usage", None) is None for c in chunks)


async def test_vllm_backend_parses_tool_calls() -> None:
    """OpenAI streamed ``tool_calls`` (name + JSON-arg fragments) are parsed.

    The OpenAI wire streams a tool call as an id/name in the first delta and the
    JSON ``arguments`` in fragments across subsequent deltas. The backend must
    accumulate them and surface a single function call with the decoded ``args``.
    """
    client = FakeOpenAIClient(
        [
            _OAChunk(
                choices=[
                    _OAChoice(
                        delta=_OADelta(
                            tool_calls=[
                                _OAToolCall(
                                    index=0,
                                    id="call_1",
                                    function=_OAFunction(name="list_parcels", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            _OAChunk(
                choices=[
                    _OAChoice(
                        delta=_OADelta(
                            tool_calls=[
                                _OAToolCall(
                                    index=0,
                                    function=_OAFunction(arguments='{"limit"'),
                                )
                            ]
                        )
                    )
                ]
            ),
            _OAChunk(
                choices=[
                    _OAChoice(
                        delta=_OADelta(
                            tool_calls=[
                                _OAToolCall(
                                    index=0,
                                    function=_OAFunction(arguments=": 5}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _OAChunk(choices=[_OAChoice(delta=_OADelta(), finish_reason="tool_calls")]),
        ]
    )
    backend = _vllm_backend_with_client(client)

    chunks = await _drain_vllm(backend)
    calls = [c.function_call for c in chunks if getattr(c, "function_call", None)]
    assert calls, "expected one parsed tool call"
    call = calls[0]
    assert call.name == "list_parcels"
    args = call.args if isinstance(call.args, dict) else json.loads(call.args)
    assert args == {"limit": 5}


# ---------------------------------------------------------------------------
# VLLMOpenAIBackend: history serialisation (tool_call_id round-trip, US-048)
# ---------------------------------------------------------------------------
def test_vllm_messages_carry_unique_tool_call_ids() -> None:
    """Tool messages reference the assistant ``tool_calls`` via ``tool_call_id``.

    The OpenAI/vLLM API returns 400 for a ``tool`` message without a
    ``tool_call_id`` and requires each assistant ``tool_calls[].id`` to be unique.
    ``genai`` function-call/response parts carry no ids, so the backend must
    synthesise unique ids on the model turn and replay them, in order, onto the
    following tool messages -- even when the same tool is requested twice in one
    turn (regression for the US-048 multi-turn tool bug).
    """
    backend = backends.VLLMOpenAIBackend(
        base_url="http://vllm.invalid/v1", model="qwen35", api_key="EMPTY"
    )
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="dos AOIs")]),
        types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(name="list_parcels", args={"aoi": 1}),
                types.Part.from_function_call(name="list_parcels", args={"aoi": 2}),
            ],
        ),
        types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(name="list_parcels", response={"n": 1}),
                types.Part.from_function_response(name="list_parcels", response={"n": 2}),
            ],
        ),
    ]

    messages = backend._messages_from_contents(contents, "system-prompt")

    assistant = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(assistant) == 1
    call_ids = [tc["id"] for tc in assistant[0]["tool_calls"]]
    assert len(call_ids) == 2
    assert len(set(call_ids)) == 2, "tool_call ids must be unique within a turn"

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m.get("tool_call_id") for m in tool_msgs), (
        "every tool message must carry a tool_call_id"
    )
    # Each response maps back to its call, in emission order.
    assert [m["tool_call_id"] for m in tool_msgs] == call_ids


# ---------------------------------------------------------------------------
# Contract guards (independent of the network)
# ---------------------------------------------------------------------------
def test_generate_stream_is_async_generator() -> None:
    """Both backends expose ``generate_stream`` as an async-generator function."""
    for cls in (backends.GeminiBackend, backends.VLLMOpenAIBackend):
        fn = cls.generate_stream
        assert inspect.isasyncgenfunction(fn) or inspect.iscoroutinefunction(fn), (
            f"{cls.__name__}.generate_stream must be async"
        )
