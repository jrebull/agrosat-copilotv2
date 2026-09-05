# Plantilla TESTS — Fase 3 (post-integracion, foreground)

> Te lanzo el orquestador cuando los dominios ya integraron. Escribes los tests faltantes
> de la US en curso — y SOLO de ella.

1. Lee el spec `docs/us-planning/us-XXX.md` §7 (plan de tests) y §8 (candados) y el resumen
   de archivos que te pase el orquestador.
2. Carga `/agrosat-testing`.

## Reglas duras

- Testea SOLO los archivos nuevos o modificados de esta US; los de otras US tienen su propia
  suite y su propio gate.
- Dos suites separadas: `backend/tests` (con `cd backend`) y `tests/` (raiz). No se cargan juntas.
- Mockea SIEMPRE Vertex/Gemini, GEE, el LLM local, MLflow server y GCS — ningun test gasta
  dinero ni red. Marcadores: `slow`, `integration`, `requires_gee`, `empirical`.
- Fixtures de evaluacion con **filas reales**: el ground truth sellado del fold 5
  (`reports/paper_micai/fase1/`), un subconjunto pequeno de un parquet OOF, o el sample
  versionado de `tests/ml/eval/fixtures/`. Nada de parcelas ni posteriores inventadas para una
  metrica titular; los datos minimos solo valen para mecanica pura y se marcan como tales.
- **Cada candado del spec §8 tiene un test que falla si se viola**: regimen no declarado,
  universo desde parcelas entregadas, umbral elegido en prueba, intervalo sin unidad, menos de
  3 clusteres. Una reparacion de protocolo lleva un test que falla sobre la version anterior.
- Incluye los tests anti-fuga espacial — no los marques como skip.
- Un gate nuevo (`scripts/*_check.py`) lleva su test en negativo: un byte alterado, un campo
  vacio o una cita rota deben ponerlo en rojo.
- Cada test cierra un hueco nombrado; nada de tests para inflar cobertura.

## Cierre

- `make test-ml` (o `make test` si es backend) — si fallan, corrigelos antes de reportar.
  Cobertura como manda AGENTS.md: directorio acotado, fila por archivo (>= 70 % por archivo de
  la US), jamas el TOTAL.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=20 lineas: tests agregados, hueco que cierra cada uno, cobertura por archivo de la US.
- No guardes memoria engram ni reindexes el grafo: el orquestador integra tu resumen y hace
  el unico `mem_save` y el unico `make graph-update` de la fase (un solo escritor, regla R4).
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: identico; todo mockeado por regla.
