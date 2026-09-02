"""Central application configuration.

Loads environment variables via Pydantic Settings. Never read ``os.environ``
directly from routers or services — always via ``get_settings()``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_DATABASE_URL = "postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat"
# Application-role DSN (role ``agrosat_app``, NOBYPASSRLS) used by the backend
# pool so RLS policies actually enforce (US-051). Separate from the superuser
# ``agrosat`` migration role, which bypasses RLS.
_DEV_APP_DATABASE_URL = "postgresql+asyncpg://agrosat_app:agrosat_app@localhost:55432/agrosat"
_DEV_REDIS_URL = "redis://localhost:6379/0"
# Placeholder rejected by the validator if env != dev.
_JWT_PLACEHOLDER = "change-me-in-prod"


class Settings(BaseSettings):
    """Typed configuration of the AgroSatCopilot backend.

    ``extra="forbid"`` detects typos in ``.env.local`` (variable defined but not
    declared here) and aborts startup. Add any new variable from
    ``.env.example`` that the backend must read.
    """

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    app_name: str = "agrosatcopilot"
    debug: bool = False

    # Connections — local docker-compose defaults. In staging/prod they are
    # mandatory and validated in ``_require_real_urls_in_cloud``.
    database_url: str = Field(default=_DEV_DATABASE_URL)
    # DSN of the non-superuser application role ``agrosat_app`` (NOBYPASSRLS):
    # the backend pool connects with this so the multi-tenant RLS policies
    # enforce (US-051). The superuser ``agrosat`` (``database_url`` /
    # ``dbmate_database_url``) is the migration role and bypasses RLS.
    app_database_url: str = Field(default=_DEV_APP_DATABASE_URL)
    dbmate_database_url: str = Field(default="")
    redis_url: str = Field(default=_DEV_REDIS_URL)
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    db_pass: str = ""

    # Cloud
    gcp_project_id: str = "agrosat-prod"
    gcp_region: str = "europe-west1"
    google_application_credentials: str = ""
    gcs_data_bucket: str = ""
    gcs_artifacts_bucket: str = ""
    gcs_dvc_bucket: str = ""
    pubsub_inference_topic: str = "inference-jobs"
    azure_subscription_id: str = ""
    azure_resource_group: str = "agrosat-rg"
    azure_h100_vm_name: str = "agrosat-h100-prod"
    azure_storage_connection_string: str = ""
    azure_blob_checkpoints_container: str = "agrosat-checkpoints"
    # H100 VM access via Cloudflare tunnel (declared for extra=forbid; the tunnel
    # is operated outside the backend).
    azure_tunnel_cloudflare_token: str = ""
    cf_access_client_id: str = ""
    cf_access_client_secret: str = ""

    # Earth Engine / CDSE
    gee_service_account_path: str = ""
    gee_project_id: str = ""
    # CDSE OAuth2 client-credentials (server-side confidential client, NOT a SPA).
    # Created at https://shapps.dataspace.copernicus.eu (Sentinel Hub OAuth client).
    # The secret is shown once at creation -- keep it only in .env.local (gitignored).
    cdse_client_id: str = ""
    cdse_client_secret: str = ""
    # Token endpoint of the CDSE Keycloak realm (verified against the official docs:
    # eu-cdse/documentation, sh_token_url). Public URL, not a secret.
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # noqa: S105
    )
    # Legacy username/password grant (kept for backward compatibility; prefer the
    # client-credentials pair above for automation).
    cdse_username: str = ""
    cdse_password: str = ""

    # HuggingFace
    huggingface_token: str = ""
    hf_home: str = ""

    # LLM backends
    # US-054: startup/fallback variant only -- the per-request reasoner reads the
    # variant from ``chat_sessions.llm_model`` (D3), so this is no longer the
    # per-request source. The four values are 1:1 with the routing table
    # (``ml.agent.llm_routing.VARIANTS``) and the DB CHECK constraint.
    llm_variant_default: Literal["gemini", "qwen-api", "qwen-onprem", "gemma"] = "gemini"
    vertex_ai_location: str = "us-central1"
    # Reasoner model for the ``gemini`` LLM variant (single source of truth read
    # by ``ChatService._reasoner_model``). US-052: ``gemini-3.5-flash`` is a
    # CONSCIOUS DEVIATION from the original AC (``gemini-2.5-pro``) decided by
    # Arthur for cost/latency of the copilot; the legacy ``gemini-3.1-pro``
    # hardcode (never read by the service) is dropped.
    gemini_model: str = "gemini-3.5-flash"
    vllm_qwen35_url: str = ""
    vllm_api_key: str = ""
    # Ollama OpenAI-compatible endpoint for the local Gemma variant (US-049).
    ollama_base_url: str = ""
    # US-054 routing table env vars (backend-agnostic). Each names WHERE a
    # variant's OpenAI-compatible host lives, so moving a model H100 -> L4 -> a
    # hosted API is an env edit, zero code (AC-4). Safe empty dev defaults: an
    # unset variant degrades to ``gemini`` at request time (the resolver logs
    # ``llm_route_env_missing``). ``qwen-onprem`` reuses ``vllm_qwen35_url`` /
    # ``vllm_api_key``; ``gemma`` falls back to ``ollama_base_url`` when
    # ``gemma_api_url`` is empty.
    # ``qwen-api``: hosted OpenAI-compatible Qwen (Together / Fireworks / OpenRouter).
    qwen_api_url: str = ""
    qwen_api_key: str = ""
    qwen_api_model: str = ""
    # ``gemma``: Google AI Studio or on-prem Ollama serving Gemma 4.
    gemma_api_url: str = ""
    gemma_api_key: str = ""
    gemma_model: str = ""
    # ``qwen-vl`` (E12): on-prem multimodal Qwen3.6-VL served by llama.cpp + mmproj
    # (OpenAI-compatible, served-model alias ``qwen36-vl``). Empty by default: an
    # unset host degrades to ``gemini`` at request time (availability-aware probe
    # logs ``llm_route_onprem_unreachable``). Reachable only behind the demo VM
    # tunnel (``make demo-vm``), so leave it blank in dev.
    qwen36_vl_url: str = ""
    # Gemini / google-genai credentials (read by the SDK via the environment;
    # declared here so ``extra="forbid"`` accepts them in ``.env.local``).
    gemini_api_key: str = ""
    google_genai_use_vertexai: str = ""
    # Zenodo deposition token for the AgroMind-IT/ES benchmark upload (US-068,
    # EPIC 11 Paper Track). Only the metadata builder runs locally; the actual
    # upload (blocker B3) reads this token. Empty by default (sponsor-provided).
    zenodo_token: str = ""
    google_cloud_project: str = ""
    google_cloud_location: str = ""
    agrosat_llm_provider: str = ""

    # Spatial-RAG lite (US-046). Feature flag gating the deferred
    # ``retrieve_context`` tool. Default off: with it disabled the reasoner runs
    # ungrounded and the agent loop never touches the ``rag_documents`` corpus
    # (AC-5, AC-10). Set ``RAG_ENABLED=true`` in ``.env.local`` to opt in.
    rag_enabled: bool = False

    # Active crop label-space for the copilot's perceiver/classifier. ``None`` (the
    # default) resolves to ``ml.eval.class_remap.DEFAULT_LABEL_SPACE`` (the v2
    # champion's ``france-12``); set ``LABEL_SPACE=france-9`` in ``.env.local`` to
    # serve a narrower vocabulary without touching code. The name must be a
    # registered label-space (``france-9`` / ``france-12`` / future HCAT spaces).
    label_space: str | None = None

    # MLflow / Dagster
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_store: str = ""
    dagster_home: str = ""

    # Auth (Clerk)
    clerk_secret_key: str = ""
    clerk_publishable_key: str = ""

    # Frontend
    frontend_url: str = "http://localhost:3000"
    nuxt_public_api_url: str = "http://localhost:8000"

    # Observability
    prometheus_pushgateway: str = ""
    sentry_dsn: str = ""
    # US-065: gate the Prometheus export of the per-turn chat metrics
    # (``chat_turn_metrics``). Off by default so structlog is always emitted but
    # the exporter is only touched when an operator opts in; the export degrades
    # to a no-op when ``prometheus-client`` is absent (honest degradation).
    chat_metrics_prometheus_enabled: bool = False

    # JWT / security
    jwt_secret: str = _JWT_PLACEHOLDER
    jwt_algorithm: str = "HS256"
    cors_allowed_origins: str = "http://localhost:3000"
    # SSRF allowlist for the COG ``url`` tile param (US-055): comma-separated
    # http(s) hosts the tiler may fetch server-side. ``gs://`` is gated separately
    # by ``gcs_data_bucket``; local/``file://`` COGs are always allowed. Empty in
    # prod = no arbitrary remote http(s) COGs.
    tile_url_allowed_hosts: str = "localhost,127.0.0.1"
    rate_limit_chat_per_min: int = 10
    rate_limit_llm_switch_per_min: int = 5

    # Terraform passthrough (not used in backend, declared for extra=forbid)
    tf_var_project_id: str = ""
    tf_var_gcp_region: str = ""
    tf_var_azure_location: str = ""
    tf_var_allowed_ssh_cidrs: str = ""

    # DVC
    dvc_remote_name: str = ""
    dvc_remote_url: str = ""

    # Docker base image for the Postgres service (docker-compose build arg; declared
    # for extra=forbid). arm64 hosts set `imresamu/postgis:15-3.4`.
    postgis_image: str = "postgis/postgis:15-3.4"

    # Host ports (docker-compose, not used by backend but declared)
    postgres_host_port: int = 5432
    redis_host_port: int = 6379
    api_host_port: int = 8000
    frontend_host_port: int = 3000
    titiler_host_port: int = 8001
    mlflow_host_port: int = 5000
    dagster_host_port: int = 3001
    ollama_host_port: int = 11434

    @property
    def cors_allow_origins(self) -> list[str]:
        """Parsed list of CORS origins from the CSV string in .env.local."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _require_real_urls_in_cloud(self) -> "Settings":
        """Refuse to start with local defaults if ``env`` is staging/prod.

        Defends against the case where a real deploy starts without a valid
        ``.env.local`` and ends up connecting to Postgres with default
        credentials.
        """
        if self.env != "dev":
            if self.database_url == _DEV_DATABASE_URL:
                raise ValueError(
                    f"DATABASE_URL is required when env={self.env!r} "
                    "(do not use the development default in cloud)."
                )
            if self.app_database_url == _DEV_APP_DATABASE_URL:
                raise ValueError(
                    f"APP_DATABASE_URL is required when env={self.env!r} "
                    "(do not ship the dev agrosat_app password to cloud)."
                )
            if self.redis_url == _DEV_REDIS_URL and not self.upstash_redis_rest_url:
                raise ValueError(
                    f"REDIS_URL or UPSTASH_REDIS_REST_URL is required when "
                    f"env={self.env!r} (do not use the development default in cloud)."
                )
            if self.jwt_secret == _JWT_PLACEHOLDER:
                raise ValueError(f"JWT_SECRET cannot be the placeholder in env={self.env!r}.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton configuration instance."""
    return Settings()
