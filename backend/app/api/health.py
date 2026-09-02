"""Healthcheck endpoints for Cloud Run liveness/readiness probes.

``/healthz`` only proves the process is alive. ``/readyz`` proves the service can
actually serve traffic: Postgres (the RLS-scoped application pool) and Redis (the
rate-limit storage) must both answer within a short timeout, otherwise the probe
returns 503 so the platform keeps routing traffic elsewhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import cast

import redis.asyncio as redis_asyncio
import structlog
from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.core.db import get_pool

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)

#: Per-dependency timeout. A probe must be cheap: a hung dependency is "not ready".
CHECK_TIMEOUT_S = 2.0


class HealthResponse(BaseModel):
    """Response of the liveness endpoint."""

    status: str
    service: str
    timestamp: datetime


class ReadyResponse(HealthResponse):
    """Response of the readiness endpoint: liveness fields plus one entry per check."""

    checks: dict[str, str] = Field(default_factory=dict)


async def check_postgres() -> str:
    """Round-trip ``SELECT 1`` through the application pool."""
    pool = await get_pool()
    await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=CHECK_TIMEOUT_S)
    return "ok"


async def check_redis() -> str:
    """``PING`` the Redis behind ``REDIS_URL`` (rate-limit storage).

    Deployments that use the Upstash REST API instead of a TCP Redis keep the dev
    default ``REDIS_URL``; there is nothing to ping, so the check is reported as
    skipped rather than failed.
    """
    settings = get_settings()
    if settings.upstash_redis_rest_url and settings.redis_url.startswith("redis://localhost"):
        return "skipped"
    client = redis_asyncio.from_url(
        settings.redis_url,
        socket_connect_timeout=CHECK_TIMEOUT_S,
        socket_timeout=CHECK_TIMEOUT_S,
    )
    try:
        # redis-py types ``ping()`` as ``Awaitable[bool] | bool``; the async client
        # always returns an awaitable.
        await asyncio.wait_for(cast("Awaitable[bool]", client.ping()), timeout=CHECK_TIMEOUT_S)
    finally:
        await client.aclose()
    return "ok"


#: Registry of readiness checks; tests swap entries to simulate outages.
READINESS_CHECKS: dict[str, Callable[[], Awaitable[str]]] = {
    "postgres": check_postgres,
    "redis": check_redis,
}


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness probe: 200 if the process is alive."""
    return HealthResponse(
        status="ok",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/readyz",
    response_model=ReadyResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
)
async def readyz(response: Response) -> ReadyResponse:
    """Readiness probe: 200 only when Postgres and Redis answer, 503 otherwise."""
    checks: dict[str, str] = {}
    for name, check in READINESS_CHECKS.items():
        try:
            checks[name] = await check()
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            checks[name] = f"error: {type(exc).__name__}"
            logger.warning("readiness_check_failed", dependency=name, error=str(exc))
    ready = all(result in ("ok", "skipped") for result in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if ready else "not_ready",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
        checks=checks,
    )
