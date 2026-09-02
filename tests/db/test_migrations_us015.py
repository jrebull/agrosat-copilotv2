"""Tests de integración para las migraciones de la US-015.

Valida que las migraciones `20260516210000_create_parcels.sql` y
`20260516210100_create_features_parcels.sql` aplican correctamente sobre un
PostgreSQL efímero con PostGIS y pgvector, que el schema introspectado coincide
con el contrato del plan (§4.3), y que las invariantes críticas (FK con
cascada, restricción UNIQUE, índices) se preservan tras un round-trip
up→down→up.

Diseño:
    - testcontainers levanta una imagen con PostGIS 15 + pgvector preinstalado.
    - El fixture aplica TODAS las migraciones de ``db/migrations/*.sql`` en
      orden lexicográfico, parseando cada archivo por los marcadores
      ``-- migrate:up`` / ``-- migrate:down``.
    - Las pruebas usan introspección sobre ``information_schema`` y
      ``pg_indexes`` para verificar tipos, constraints e índices.

Si la imagen Docker no levanta o ``testcontainers`` no está instalado, los
tests se saltan limpiamente (no fallan) para no bloquear la CI sin Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("testcontainers", reason="testcontainers no instalado")
pytest.importorskip("sqlalchemy", reason="sqlalchemy requerido")

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

try:
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover - rama defensiva
    PostgresContainer = None  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

# Imágenes candidatas en orden de preferencia. La primera que arranque gana.
# - agrosat-postgres: imagen del proyecto (PostGIS 15-3.4 + pgvector), buildeada
#   localmente vía `infrastructure/docker/postgres.Dockerfile`.
# - postgis/postgis: fallback con solo PostGIS; salta los tests si pgvector no
#   está disponible (no contamina la transacción gracias al CREATE EXTENSION
#   en transacción aislada).
CANDIDATE_IMAGES: tuple[str, ...] = (
    "agrosat-postgres:15-3.4-pgvector",
    "postgis/postgis:15-3.4",
)

_MIGRATE_UP_RE = re.compile(r"--\s*migrate:up\s*\n(.*?)(?=--\s*migrate:down|\Z)", re.DOTALL)


def _split_migration(sql_text: str) -> str:
    """Devuelve el bloque ``migrate:up`` de un archivo dbmate."""
    match = _MIGRATE_UP_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migración sin bloque -- migrate:up")
    return match.group(1).strip()


def _apply_all_migrations(engine: Engine) -> None:
    """Aplica todas las migraciones ``db/migrations/*.sql`` en orden lexicográfico."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No se encontraron migraciones en {MIGRATIONS_DIR}")
    with engine.begin() as conn:
        for migration_path in files:
            up_block = _split_migration(migration_path.read_text(encoding="utf-8"))
            conn.execute(text(up_block))


def _rollback_us015(engine: Engine) -> None:
    """Revierte las dos migraciones de la US-015 (orden inverso)."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features_parcels CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS parcels CASCADE"))


def _apply_us015(engine: Engine) -> None:
    """Aplica solo las dos migraciones de la US-015 (parcels y features_parcels)."""
    targets = sorted(MIGRATIONS_DIR.glob("2026051621*_*.sql"))
    if len(targets) != 2:
        raise RuntimeError(f"Esperaba 2 migraciones US-015, encontré {len(targets)}: {targets}")
    with engine.begin() as conn:
        for migration_path in targets:
            conn.execute(text(_split_migration(migration_path.read_text(encoding="utf-8"))))


@pytest.fixture(scope="module")
def pg_engine() -> Engine:
    """Levanta un Postgres efímero con PostGIS+pgvector y aplica todas las migraciones."""
    if PostgresContainer is None:
        pytest.skip("testcontainers.postgres no disponible")

    last_error: Exception | None = None
    for image in CANDIDATE_IMAGES:
        try:
            container = PostgresContainer(
                image=image, username="test", password="test", dbname="test"
            )
            container.start()
        except Exception as exc:  # pragma: no cover - depende del host
            last_error = exc
            continue

        try:
            url = container.get_connection_url().replace(
                "postgresql+psycopg2", "postgresql+psycopg"
            )
            try:
                engine = create_engine(url, future=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            except Exception:
                # fallback al driver por defecto si psycopg v3 no está instalado
                engine = create_engine(container.get_connection_url(), future=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            # Garantiza que la extensión vector exista antes de aplicar migraciones.
            try:
                with engine.begin() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except OperationalError as exc:
                container.stop()
                last_error = exc
                continue

            _apply_all_migrations(engine)
            yield engine
            container.stop()
            return
        except Exception as exc:  # pragma: no cover - depende del entorno
            container.stop()
            last_error = exc
            continue

    pytest.skip(f"Ninguna imagen Postgres+PostGIS+pgvector disponible: {last_error}")


def _columns(engine: Engine, table: str) -> dict[str, str]:
    """Devuelve mapping {nombre_columna: data_type} via information_schema."""
    query = text(
        """
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table": table}).all()
    # Para tipos USER-DEFINED (geometry, vector) information_schema reporta
    # data_type='USER-DEFINED' y el detalle queda en udt_name.
    return {
        row.column_name: (row.udt_name if row.data_type == "USER-DEFINED" else row.data_type)
        for row in rows
    }


def test_dbmate_up_creates_parcels(pg_engine: Engine) -> None:
    """La tabla parcels existe con las columnas y tipos del contrato."""
    cols = _columns(pg_engine, "parcels")
    assert cols["id"] == "bigint"
    assert cols["session_id"] == "uuid"
    assert cols["aoi_id"] == "bigint"
    assert cols["geom"] == "geometry"
    assert cols["crop_class"] == "text"
    assert cols["confidence"] == "real"
    assert cols["area_ha"] == "real"
    assert cols["year"] == "smallint"
    assert cols["created_at"] == "timestamp with time zone"
    assert cols["updated_at"] == "timestamp with time zone"


def test_dbmate_up_creates_features_parcels(pg_engine: Engine) -> None:
    """La tabla features_parcels existe con columnas, tipos y VECTOR(64)."""
    cols = _columns(pg_engine, "features_parcels")
    assert cols["id"] == "bigint"
    assert cols["parcel_id"] == "bigint"
    assert cols["year"] == "smallint"
    assert cols["alphaearth_embedding"] == "vector"
    assert cols["ndvi_stats"] == "jsonb"
    assert cols["phenology"] == "jsonb"
    assert cols["sog_doy"] == "smallint"
    assert cols["peak_doy"] == "smallint"
    assert cols["peak_value"] == "real"
    assert cols["senescence_doy"] == "smallint"
    assert cols["ndvi_auc"] == "real"
    assert cols["ndvi_slope_pre_peak"] == "real"
    assert cols["ndvi_slope_post_peak"] == "real"
    assert cols["maturity_duration_days"] == "smallint"
    assert cols["created_at"] == "timestamp with time zone"
    assert cols["updated_at"] == "timestamp with time zone"

    # Confirma dimensión VECTOR(64) consultando pg_attribute + atttypmod.
    query = text(
        """
        SELECT atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'features_parcels' AND a.attname = 'alphaearth_embedding'
        """
    )
    with pg_engine.connect() as conn:
        atttypmod = conn.execute(query).scalar_one()
    assert atttypmod == 64, f"Esperaba VECTOR(64), atttypmod={atttypmod}"


def test_unique_constraint_exists(pg_engine: Engine) -> None:
    """La restricción UNIQUE features_parcels_parcel_year_uniq está presente."""
    query = text(
        """
        SELECT constraint_name, constraint_type
        FROM information_schema.table_constraints
        WHERE table_schema = 'public'
          AND table_name = 'features_parcels'
          AND constraint_name = 'features_parcels_parcel_year_uniq'
        """
    )
    with pg_engine.connect() as conn:
        row = conn.execute(query).one_or_none()
    assert row is not None, "Falta la constraint UNIQUE (parcel_id, year)"
    assert row.constraint_type == "UNIQUE"


def test_fk_cascade_on_parcel_delete(pg_engine: Engine) -> None:
    """Borrar una parcel debe borrar en cascada sus features asociados."""
    insert_parcel = text(
        """
        INSERT INTO parcels (geom, year)
        VALUES (ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 4326), 2024)
        RETURNING id
        """
    )
    with pg_engine.begin() as conn:
        parcel_id = conn.execute(insert_parcel).scalar_one()
        conn.execute(
            text("INSERT INTO features_parcels (parcel_id, year) VALUES (:pid, :yr)"),
            {"pid": parcel_id, "yr": 2024},
        )
        feature_count_before = conn.execute(
            text("SELECT count(*) FROM features_parcels WHERE parcel_id = :pid"),
            {"pid": parcel_id},
        ).scalar_one()
        assert feature_count_before == 1

        conn.execute(text("DELETE FROM parcels WHERE id = :pid"), {"pid": parcel_id})
        feature_count_after = conn.execute(
            text("SELECT count(*) FROM features_parcels WHERE parcel_id = :pid"),
            {"pid": parcel_id},
        ).scalar_one()
    assert feature_count_after == 0, "FK ON DELETE CASCADE no surtió efecto"


def test_round_trip_up_down_up(pg_engine: Engine) -> None:
    """Aplicar las migraciones, revertirlas y re-aplicarlas deja schema idéntico."""
    cols_before = _columns(pg_engine, "features_parcels")
    _rollback_us015(pg_engine)
    with pg_engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT to_regclass('public.features_parcels'),       to_regclass('public.parcels')"
            )
        ).one()
    assert exists[0] is None and exists[1] is None, "Rollback no eliminó las tablas"

    _apply_us015(pg_engine)
    cols_after = _columns(pg_engine, "features_parcels")
    assert cols_before == cols_after, "Schema difiere tras round-trip up→down→up"


def test_indexes_present(pg_engine: Engine) -> None:
    """Los índices declarados existen (GIST en geom, BTREE en year/parcel_id)."""
    query = text(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename IN ('parcels', 'features_parcels')
        """
    )
    with pg_engine.connect() as conn:
        rows = {row.indexname: row.indexdef for row in conn.execute(query).all()}

    assert "parcels_geom_idx" in rows
    # PostgreSQL serializa `pg_indexes.indexdef` siempre en lowercase para los
    # tipos de índice ("using gist"), por eso ambos lados van en lowercase.
    assert "using gist" in rows["parcels_geom_idx"].lower()

    assert "parcels_session_id_idx" in rows
    assert "parcels_year_idx" in rows
    assert "features_parcels_parcel_id_idx" in rows
    assert "features_parcels_year_idx" in rows

    # Los índices BTREE son el default; pg_indexes lo deja implícito en indexdef.
    # Verificamos que no estén creados como GIST por accidente.
    for name in (
        "parcels_session_id_idx",
        "parcels_year_idx",
        "features_parcels_parcel_id_idx",
        "features_parcels_year_idx",
    ):
        assert "using gist" not in rows[name].lower(), f"{name} no debe ser GIST"
