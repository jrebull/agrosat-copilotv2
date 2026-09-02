"""Tests for availability-aware LLM routing (US-081 AC10).

The probe is INJECTABLE so the on-prem -> Gemini fallback is exercised with zero
network: a stub probe returns ``True``/``False`` deterministically. The Gemini and
vLLM backends are never actually called (no Vertex AI / vLLM); we only assert the
route decision and the constructed backend TYPE.
"""

from __future__ import annotations

from ml.agent.backends import GeminiBackend, VLLMOpenAIBackend
from ml.agent.llm_routing import (
    RouteDecision,
    SocketAvailabilityProbe,
    make_backend_for_variant_available,
    resolve_route_available,
)


class _Settings:
    """Minimal settings stub carrying the route env attributes."""

    # Gemini (always-resolvable fallback target).
    gemini_api_key = "test-key"
    gemini_model = "gemini-3.5-flash"
    google_api_key = ""
    google_genai_use_vertexai = ""
    google_cloud_project = ""
    gcp_project_id = ""
    google_cloud_location = ""
    vertex_ai_location = ""
    # On-prem Qwen (US-048).
    vllm_qwen35_url = "http://127.0.0.1:8002/v1"
    vllm_api_key = "EMPTY"
    # Other variants' envs (unused here but read by the route table).
    qwen_api_url = "https://qwen.example/v1"
    qwen_api_key = "qkey"
    qwen_api_model = "qwen-api"
    gemma_api_url = "http://127.0.0.1:11435/v1"
    gemma_api_key = ""
    gemma_model = "gemma4:31b-it-q8_0"
    ollama_base_url = ""


def _always(value: bool):
    """Build a probe stub that always returns ``value`` (and records calls)."""

    calls: list[str] = []

    def _probe(base_url: str) -> bool:
        calls.append(base_url)
        return value

    _probe.calls = calls  # type: ignore[attr-defined]
    return _probe


def test_gemini_variant_needs_no_probe() -> None:
    """A native Gemini variant resolves without probing (cloud is the fallback)."""
    settings = _Settings()
    probe = _always(False)  # would force a fallback IF it were consulted

    decision = resolve_route_available("gemini", settings, probe=probe)  # type: ignore[arg-type]

    assert isinstance(decision, RouteDecision)
    assert decision.fell_back is False
    assert decision.reason == "gemini_native"
    assert decision.route.backend_type == "gemini"
    # The probe must NOT be consulted for a native Gemini route.
    assert probe.calls == []  # type: ignore[attr-defined]


def test_onprem_reachable_is_used() -> None:
    """When the on-prem endpoint is reachable the on-prem route is kept."""
    settings = _Settings()
    probe = _always(True)

    decision = resolve_route_available("qwen-onprem", settings, probe=probe)  # type: ignore[arg-type]

    assert decision.fell_back is False
    assert decision.reason == "available"
    assert decision.route.backend_type == "openai_compat"
    assert decision.route.base_url == "http://127.0.0.1:8002/v1"
    assert probe.calls == ["http://127.0.0.1:8002/v1"]  # type: ignore[attr-defined]


def test_onprem_unreachable_falls_back_to_gemini() -> None:
    """An unreachable on-prem endpoint degrades honestly to Gemini (AC10)."""
    settings = _Settings()
    probe = _always(False)

    decision = resolve_route_available("qwen-onprem", settings, probe=probe)  # type: ignore[arg-type]

    assert decision.fell_back is True
    assert decision.reason == "onprem_unreachable"
    assert decision.requested_variant == "qwen-onprem"
    # The selected route is now the Gemini cloud route.
    assert decision.route.backend_type == "gemini"
    assert decision.route.variant == "gemini"
    assert probe.calls == ["http://127.0.0.1:8002/v1"]  # type: ignore[attr-defined]


def test_make_backend_falls_back_to_gemini_backend() -> None:
    """The backend builder returns a GeminiBackend when on-prem is unreachable."""
    settings = _Settings()
    backend, decision = make_backend_for_variant_available(
        "qwen-onprem",
        settings,
        probe=_always(False),  # type: ignore[arg-type]
    )
    assert isinstance(backend, GeminiBackend)
    assert decision.fell_back is True


def test_make_backend_keeps_vllm_when_reachable() -> None:
    """The backend builder returns a vLLM backend when on-prem is reachable."""
    settings = _Settings()
    backend, decision = make_backend_for_variant_available(
        "qwen-onprem",
        settings,
        probe=_always(True),  # type: ignore[arg-type]
    )
    assert isinstance(backend, VLLMOpenAIBackend)
    assert backend.model == "qwen35"
    assert decision.fell_back is False


def test_gemma_unreachable_also_falls_back() -> None:
    """The Gemma on-prem variant also degrades to Gemini when unreachable."""
    settings = _Settings()
    decision = resolve_route_available("gemma", settings, probe=_always(False))  # type: ignore[arg-type]
    assert decision.fell_back is True
    assert decision.route.backend_type == "gemini"


def test_socket_probe_returns_false_on_bad_url() -> None:
    """The default socket probe never raises and reports unreachable on junk."""
    probe = SocketAvailabilityProbe(timeout=0.05)
    assert probe("") is False
    assert probe("not-a-url") is False
    # A closed/unused high port on localhost should refuse fast (no raise).
    assert isinstance(probe("http://127.0.0.1:59999/v1"), bool)
