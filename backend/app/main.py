"""Entry point of the AgroSatCopilot API.

Startup: ``poetry run uvicorn backend.app.main:app --reload --port 8000``.

Routers are mounted progressively as the US are closed:
- /healthz, /readyz             — operational from the bootstrap
- /chat (SSE)                   — EPIC 7 (Google ADK agent)
- /aois, /timeseries            — EPIC 2 (satellite data)
- /stac/search, /tiles          — EPIC 2 (catalog + TiTiler)
- /llm/switch                   — EPIC 7 (A/B switch Gemini <-> Qwen3 on-prem)
- /jobs                         — EPIC 8 (asynchronous inference via Pub/Sub)
"""

# isort: off
# US-055 PROJ fix: pin PROJ_DATA to rasterio's bundled DB BEFORE titiler /
# rio-tiler initialise GDAL's PROJ (Riesgo R1 on the Windows dev box). This MUST
# stay above the ``titiler.core`` import below -- do not reorder.
from backend.app.services import proj_env as _proj_env  # noqa: F401

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# slowapi ships an async-friendly handler that builds the ``429`` response (with
# the ``Retry-After`` / ``X-RateLimit-*`` headers when enabled). Imported under a
# private alias so it is not re-exported from this module.
from slowapi import _rate_limit_exceeded_handler as _slowapi_rate_limit_handler
from slowapi.errors import RateLimitExceeded
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers

# isort: on
from backend.app.api import (
    aois,
    chat,
    health,
    llm,
    metrics,
    sessions,
    stac,
    tiles,
    timeseries,
)
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.core.rate_limit import limiter
from backend.app.middleware.metrics import PrometheusMiddleware
from backend.app.services.cog_tiler import cog_tiler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: connection setup and cleanup."""
    settings = get_settings()
    configure_logging(env=settings.env, log_level=settings.log_level)
    logger = structlog.get_logger()
    logger.info("startup", env=settings.env, region=settings.gcp_region)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    app = FastAPI(
        title="AgroSatCopilot API",
        version="0.1.0",
        description="SaaS conversacional agrícola con Foundation Models satelitales.",
        lifespan=lifespan,
    )
    # Per-session rate limiting (US-052). slowapi reads the limiter from
    # ``app.state.limiter``; the ``@limiter.limit`` decorator on ``/chat`` does
    # the enforcement and the registered handler renders a JSON ``429`` (no
    # global ``SlowAPIMiddleware`` -- only ``/chat`` is limited, keyed per
    # session). The handler is evaluated before the SSE stream opens.
    app.state.limiter = limiter
    # slowapi types its handler against the concrete ``RateLimitExceeded`` while
    # Starlette's signature expects the base ``Exception``; the registration is
    # the documented slowapi pattern, so the variance is narrowed here.
    app.add_exception_handler(RateLimitExceeded, _slowapi_rate_limit_handler)  # type: ignore[arg-type]
    # CORS with explicit allow_headers (SEC hardening): combining allow_credentials=True
    # with allow_headers=["*"] exposes the API to abuse. Whitelist the minimum headers
    # that the Nuxt frontend + SSE client actually send.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-Session-ID",
            # US-080: the session switcher sends an anonymous browser/user id on
            # POST/GET /sessions so chats restore from the server (no auth yet).
            "X-User-ID",
        ],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )
    # US-059 observability: outermost middleware (added after CORS so it runs
    # first/last in the stack) records real per-process HTTP latency + request
    # counters by route template. The ``/metrics`` scrape endpoint below exposes
    # them in Prometheus text format; the middleware excludes ``/metrics`` from
    # its own histogram to avoid skewing the percentiles with scrape traffic.
    app.add_middleware(PrometheusMiddleware)
    app.include_router(health.router)
    # US-059 Prometheus scrape endpoint (process metrics, not session-scoped).
    app.include_router(metrics.router)
    app.include_router(chat.router)
    # US-080 chat-session lifecycle + server-side transcript (create/list
    # messages/rename/delete) for the in-app multi-chat UI.
    app.include_router(sessions.router)
    # US-054 hot-swap of the per-session reasoner variant (session-scoped, RLS).
    app.include_router(llm.router)
    # US-053 geospatial data endpoints (all session-scoped via RLS).
    app.include_router(aois.router)
    app.include_router(timeseries.router)
    app.include_router(stac.router)
    # US-055 tiling -- two surfaces, one render path:
    #   /cog  -> the full TiTiler ``TilerFactory`` (literal AC endpoint
    #            ``/cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}.png?url&expression&
    #            rescale&colormap_name`` + /cog/info, /cog/tilejson.json, ...).
    #   /tiles -> the stable US-053 contract ``GET /tiles/{z}/{x}/{y}.png?url&index``
    #            (adapter fixes WebMercatorQuad and delegates to the shared render).
    # ``add_exception_handlers`` maps rio-tiler/titiler errors to typed HTTP
    # responses (no bare 500 without a body). The existing hardened CORS
    # middleware already covers GET/OPTIONS for both surfaces.
    app.include_router(cog_tiler.router, prefix="/cog", tags=["cog"])
    add_exception_handlers(app, DEFAULT_STATUS_CODES)
    app.include_router(tiles.router)
    return app


app = create_app()
