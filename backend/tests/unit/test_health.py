"""Unit tests for the liveness/readiness probes (no Docker: checks are swapped)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.app.api import health


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    return TestClient(app)


def test_healthz_is_always_ok() -> None:
    body = _client().get("/healthz").json()
    assert body["status"] == "ok"
    assert body["service"] == "agrosat-api"


def test_readyz_ready_when_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok() -> str:
        return "ok"

    monkeypatch.setattr(health, "READINESS_CHECKS", {"postgres": ok, "redis": ok})
    resp = _client().get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["checks"] == {"postgres": "ok", "redis": "ok"}


def test_readyz_503_when_a_dependency_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok() -> str:
        return "ok"

    async def boom() -> str:
        raise ConnectionRefusedError("redis down")

    monkeypatch.setattr(health, "READINESS_CHECKS", {"postgres": ok, "redis": boom})
    resp = _client().get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["redis"].startswith("error: ConnectionRefusedError")


def test_readyz_skipped_check_counts_as_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok() -> str:
        return "ok"

    async def skipped() -> str:
        return "skipped"

    monkeypatch.setattr(health, "READINESS_CHECKS", {"postgres": ok, "redis": skipped})
    resp = _client().get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["checks"]["redis"] == "skipped"
