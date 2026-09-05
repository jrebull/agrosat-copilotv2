# Plantilla GEO-DATA — Fase 3

> Te lanzo el orquestador con el numero y titulo de la US. Subagente: `geo-data-engineer`.
> Alcance: `ml/ingest/`, `ml/features/`, `ml/data/`, `ml/utils/`, `dagster_project/` y, solo si el
> spec lo marca, `db/migrations/`. No toques otros directorios.

1. Lee la guia del directorio SOLO si tu harness no la cargo ya (`ml/AGENTS.md` y `ml/CLAUDE.md`
   son espejos identicos) y el spec `docs/us-planning/us-XXX.md` (en especial §2 Arquitectura,
   §3 Interfaces y §6 Falsabilidad). El spec esta congelado: no lo edites; si tienes que
   desviarte, reportalo en tu resumen citando la seccion.
2. Carga `/agrosat-ml-features` y, si tocas AlphaEarth o GEE, `/agrosat-gee-alphaearth`.
3. Protocolo graphify del AGENTS.md raiz: `query` + `affected` antes de crear; grep solo con
   0 nodos. Eres consumidor del grafo — NO ejecutes `make graph-update`.
4. Consulta con Context7 (`--c7`) Polars, GeoPandas/Shapely 2.x, earthengine-api, rasterio.

## Reglas duras del dominio

- Polars `LazyFrame`, no pandas. `parcel_id` canonico `pl.Utf8` via `canonical_parcel_id`
  (cast idempotente antes de cada LEFT JOIN).
- Split espacial con `build_spatial_kfold` (H3 res 5 + KMeans + colchon de 1 km). Jamas un
  split aleatorio sobre parcelas.
- AlphaEarth es `SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1, CC-BY-4.0, 64 dims. Nunca "v2.1";
  nunca "codifica fenologia".
- GEE solo por ADC sobre el proyecto `agrosat-copilot`; ningun test llama a GEE (`requires_gee`
  se salta en CI). Un export masivo (> 50 patches) exige confirmacion del humano: cuota y
  egress cuestan.
- PASTIS-R crudo vive en `data/PASTIS-R/` (68 GB, gitignorado); BreizhCrops en DVC
  (`dvc pull data/breizhcrops`). Los cargadores existen (`ml/ingest/pastis_loader.py`,
  `ml/ingest/breizhcrops_loader.py`): se extienden, no se duplican.
- Todo parquet nuevo sale con `dvc add`; el `.dvc` es parte de la custodia. Un archivo con
  fila `SELLADO` en `paper/ARTIFACTS.md` es de solo lectura.
- Un banco nuevo se sella como el fold 5: ground truth por parcela, centroides, soporte por clase,
  procedencia con MD5 de origen, commit y versiones.

## Cierre

- Ejecuta las pruebas de falsabilidad del spec §6 que toquen datos y reporta el resultado
  sea cual sea.
- Todo funcional, cero stubs ni TODOs.
- `make lint && make test-ml` — si fallan, corrigelos antes de reportar.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=30 lineas: archivos creados/extendidos, columnas y tipos del parquet, decisiones,
  desviaciones del spec, falsabilidad (resultado vs umbral), `.dvc` generados, pendientes o
  conflictos de frontera con modeling.
- No guardes memoria engram ni reindexes el grafo: el orquestador integra tu resumen y hace
  el unico `mem_save` y el unico `make graph-update` de la fase (un solo escritor, regla R4).
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: cero exports de GEE y cero descargas masivas; implementa la logica y los
tests con fixtures reales pequenas; la corrida real la lanza el humano al despertar.
