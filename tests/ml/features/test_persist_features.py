"""Tests integración de :func:`ml.features.persist_features.load_features_parcels` (US-015).

Requiere ``testcontainers[postgres]`` con PostGIS + pgvector. Si la dependencia
o Docker no están disponibles, los tests se saltan limpiamente.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError

testcontainers_postgres = pytest.importorskip("testcontainers.postgres")
# Cualquier flavor de psycopg debe estar disponible para que SQLAlchemy abra
# la conexión. Si el entorno no lo tiene, saltamos limpiamente.
psycopg_available = False
for _candidate in ("psycopg", "psycopg2"):
    try:
        __import__(_candidate)
        psycopg_available = True
        break
    except ImportError:
        continue

if not psycopg_available:
    pytest.skip("psycopg / psycopg2 not installed", allow_module_level=True)

from ml.features.persist_features import load_features_parcels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


#: Imagen custom buildeada en `infrastructure/docker/postgres.Dockerfile`
#: (PostGIS 15-3.4 + pgvector). Si no está en el registry local, el fixture
#: hace skip clean — `docker compose build postgres` la genera.
_DEFAULT_IMAGE = "agrosat-postgres:15-3.4-pgvector"


@pytest.fixture(scope="module")
def pg_engine() -> Iterator[Engine]:
    """Levanta un contenedor Postgres + PostGIS + pgvector efímero."""
    try:
        container = testcontainers_postgres.PostgresContainer(
            image=_DEFAULT_IMAGE,
            username="agrosat",
            password="agrosat",
            dbname="agrosat",
        )
        container.start()
    except Exception as exc:  # noqa: BLE001 - imagen no buildeada localmente o Docker apagado
        pytest.skip(f"agrosat-postgres custom image no disponible: {exc}")
    try:
        url = container.get_connection_url().replace("postgresql+psycopg2", "postgresql+psycopg")
        engine = create_engine(url, future=True)
        _bootstrap_schema(engine)
        yield engine
        engine.dispose()
    finally:
        container.stop()


def _bootstrap_schema(engine: Engine) -> None:
    """Aplica el initial_schema completo + las migraciones US-015 vía sus archivos SQL."""
    # Cada CREATE EXTENSION va en su propia transacción: si una falla, no
    # contamina las siguientes (evita InFailedSqlTransaction en cascada).
    for ext in ("postgis", "vector"):
        with engine.begin() as conn:
            conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext}"))

    applied = _try_apply_migrations(engine)
    if not applied:
        raise RuntimeError(
            f"No migrations found in {MIGRATIONS_DIR}. US-015 requires the "
            "create_parcels and create_features_parcels migrations to be present."
        )


def _try_apply_migrations(engine: Engine) -> bool:
    """Aplica todas las migraciones de ``db/migrations/`` en orden lexicográfico.

    Replica el efecto de ``dbmate up`` sin requerir el binario externo.
    Salta los ``CREATE EXTENSION`` ya ejecutados arriba para evitar
    advertencias duplicadas (idempotente con ``IF NOT EXISTS``).
    """
    if not MIGRATIONS_DIR.exists():
        return False
    candidates = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not candidates:
        return False
    for path in candidates:
        sql = path.read_text(encoding="utf-8")
        up_section = sql
        if "-- migrate:up" in sql:
            up_section = sql.split("-- migrate:up", 1)[1]
        if "-- migrate:down" in up_section:
            up_section = up_section.split("-- migrate:down", 1)[0]
        with engine.begin() as conn:
            # exec_driver_sql: the raw migration text is sent as-is. ``text()`` would
            # parse ``:8002`` / ``:11434`` inside SQL comments as bind parameters.
            conn.exec_driver_sql(up_section)
    return True


@pytest.fixture
def parcel_id(pg_engine: Engine) -> int:
    """Crea una parcela con geometría dummy y devuelve su id."""
    with pg_engine.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO parcels (geom, year)
                VALUES (ST_GeomFromText('POLYGON((0 0,0 1,1 1,1 0,0 0))', 4326), 2024)
                RETURNING id
                """
            )
        ).first()
    assert row is not None
    pid = int(row[0])
    yield pid
    with pg_engine.begin() as conn:
        conn.execute(text("DELETE FROM features_parcels WHERE parcel_id = :pid"), {"pid": pid})
        conn.execute(text("DELETE FROM parcels WHERE id = :pid"), {"pid": pid})


def _build_df(parcel_id: int, *, peak_value: float = 0.85, year: int = 2024) -> pl.DataFrame:
    """Construye un DF mínimo con todas las columnas que persiste el módulo."""
    data: dict[str, object] = {
        "parcel_id": [parcel_id],
        "year": [year],
        "NDVI_mean": [0.45],
        "NDVI_std": [0.2],
        "EVI_mean": [0.35],
        "NDVI_fft_amp_0": [0.45],
        "NDVI_fft_phase_0": [0.0],
        "NDVI_fft_amp_1": [0.3],
        "NDVI_fft_phase_1": [-1.57],
        "sog_doy": [120],
        "peak_doy": [180],
        "peak_value": [peak_value],
        "senescence_doy": [240],
        "ndvi_auc": [80.5],
        "ndvi_slope_pre_peak": [0.01],
        "ndvi_slope_post_peak": [-0.01],
        "maturity_duration_days": [30],
    }
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_inserts_new_rows(pg_engine: Engine, parcel_id: int) -> None:
    df = _build_df(parcel_id)
    n = load_features_parcels(df, pg_engine, on_conflict="update")
    assert n == 1
    with pg_engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT peak_value, ndvi_stats, phenology FROM features_parcels "
                "WHERE parcel_id = :pid"
            ),
            {"pid": parcel_id},
        ).one()
    assert row[0] == pytest.approx(0.85, abs=1e-3)
    stats = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    assert stats["NDVI_mean"] == pytest.approx(0.45)


def test_upsert_updates_existing(pg_engine: Engine, parcel_id: int) -> None:
    load_features_parcels(_build_df(parcel_id, peak_value=0.7), pg_engine, on_conflict="update")
    load_features_parcels(_build_df(parcel_id, peak_value=0.9), pg_engine, on_conflict="update")
    with pg_engine.begin() as conn:
        rows = conn.execute(
            text("SELECT peak_value FROM features_parcels WHERE parcel_id = :pid"),
            {"pid": parcel_id},
        ).all()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.9, abs=1e-3)


def test_on_conflict_skip(pg_engine: Engine, parcel_id: int) -> None:
    load_features_parcels(_build_df(parcel_id, peak_value=0.7), pg_engine, on_conflict="update")
    load_features_parcels(_build_df(parcel_id, peak_value=0.9), pg_engine, on_conflict="skip")
    with pg_engine.begin() as conn:
        peak = conn.execute(
            text("SELECT peak_value FROM features_parcels WHERE parcel_id = :pid"),
            {"pid": parcel_id},
        ).scalar_one()
    assert peak == pytest.approx(0.7, abs=1e-3)


def test_on_conflict_raise(pg_engine: Engine, parcel_id: int) -> None:
    load_features_parcels(_build_df(parcel_id), pg_engine, on_conflict="update")
    with pytest.raises(IntegrityError):
        load_features_parcels(_build_df(parcel_id), pg_engine, on_conflict="raise")


def test_unique_violation_on_duplicate_with_raise_mode(pg_engine: Engine, parcel_id: int) -> None:
    load_features_parcels(_build_df(parcel_id, year=2023), pg_engine, on_conflict="raise")
    # Re-insertar con mismo (parcel_id, year) debe violar UNIQUE.
    with pytest.raises(IntegrityError):
        load_features_parcels(_build_df(parcel_id, year=2023), pg_engine, on_conflict="raise")


def test_load_empty_frame_returns_zero(pg_engine: Engine) -> None:
    empty = pl.DataFrame(
        {"parcel_id": pl.Series([], dtype=pl.Int64), "year": pl.Series([], dtype=pl.Int32)}
    )
    assert load_features_parcels(empty, pg_engine, on_conflict="update") == 0
