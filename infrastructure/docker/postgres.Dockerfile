# Imagen base parametrizable: la oficial `postgis/postgis:15-3.4` solo publica amd64
# (en Apple Silicon corre emulada y lenta). `imresamu/postgis:15-3.4` es la misma
# receta publicada para amd64 + arm64; se elige con POSTGIS_IMAGE en .env.local.
ARG POSTGIS_IMAGE=postgis/postgis:15-3.4
FROM ${POSTGIS_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-15-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Crea la base de datos dedicada `mlflow` para el backend store de MLflow,
# separada de la DB `agrosat` de la aplicacion (evita mezclar el schema
# interno de MLflow con las migraciones dbmate del negocio).
COPY infrastructure/docker/init-mlflow-db.sql /docker-entrypoint-initdb.d/10-init-mlflow-db.sql
