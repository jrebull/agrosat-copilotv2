# Plantilla MODELING — Fase 3

> Te lanzo el orquestador con el numero y titulo de la US. Subagente: `ml-engineer`.
> Alcance: `ml/eval/`, `ml/ensemble/`, `ml/train/`, `ml/models/`, `ml/tune/`,
> `scripts/run_paper_micai_*.py` y las salidas en `reports/paper_micai/<fase>/`. Utilidades
> compartidas se refactorizan a `ml/utils/`, no se duplican. No toques otros directorios.

1. Lee la guia del directorio SOLO si tu harness no la cargo ya (`ml/AGENTS.md` y `ml/CLAUDE.md`
   son espejos identicos) y el spec `docs/us-planning/us-XXX.md` (en especial §2 Arquitectura,
   §3 Interfaces, §5 Trazabilidad, §6 Falsabilidad y §8 Candados). El spec esta congelado: no
   lo edites; si tienes que desviarte, reportalo en tu resumen citando la seccion.
2. Carga `/agrosat-protocolo-articulo` ANTES de escribir cualquier comparacion, intervalo o
   contraste. Para miembros densos, `/agrosat-ml-segmentation`; para tabulares,
   `/agrosat-ml-baseline`; para combinar, `/agrosat-ml-ensemble`.
3. Si Graphify está disponible, úsalo para orientar `query` + `affected` y confirma con `rg`,
   imports y tests. Eres consumidor del grafo: no ejecutes `make graph-update`.
4. Consulta con Context7 (`--c7`) scikit-learn, XGBoost, SciPy/statsmodels, MAPIE o la libreria
   de conformal que uses.

## Reglas duras del dominio

- **Un regimen por comparacion, nombrado** (fold-5 held-out por parcela · OOF · pixel). El
  0,7486 es in-sample para el meta-modelo: cualquier cifra que salga de `_stacking_metrics` o
  de Optuna sobre el fold 5 se etiqueta como tal.
- **Unidad = parcela, cluster = `patch_id`.** `paired_interval` exige declarar la unidad y no
  publica intervalo ni p por debajo de 3 clusteres pareados. K es sensibilidad, no replica.
- **Universo de clases desde entrenamiento** (`macro_over` exige el universo del bloque).
  **Punto de operacion desde train/val**, aplicado sin tocarlo en prueba: igualar perdida,
  cobertura o tasa usando la prueba es la fuga que dos auditorias encontraron.
- Multiplicidad (Holm) y separacion confirmatorio / exploratorio **declaradas en el spec antes
  de correr**; si el spec no las trae, detente y reporta.
- Semillas fijas y registradas; cada artefacto sale con semilla, versiones de computo, commit
  y prueba pareada con intervalo, en `reports/paper_micai/<fase>/`, y se reporta como
  PENDIENTE DE SELLAR: el sello (`make paper-artifacts-seal`) lo pone mlops o el humano.
- **Nunca sobrescribas un archivo con fila `SELLADO` u `OBSOLETO`**: archivo nuevo. Nunca cites
  un artefacto `OBSOLETO` fuera de un bloque CUARENTENA.
- **ADR-014 §7**: nada de las EPIC 20, 21, 22 ni 25 antes del preregistro firmado. Si el spec
  lo pide, implementa la mecanica con tests y NO ejecutes la corrida; reportalo.
- **Panel, no ganador**: nunca elijas "el predictor del articulo" con las etiquetas que despues
  evaluan; el predictor es factor de sensibilidad (>= 3 familias por banco).
- Ninguna afirmacion de transporte entre bancos, ni en un docstring.
- Split espacial con `build_spatial_kfold`; MLflow con `data_version` + `code_version`;
  checkpoints y OOF con `dvc add`. Polars, no pandas.
- Hardware del artículo: CPU, RTX 4070 o L4 spot. La H100 sigue gobernada por ADR-009, pero una US
  MICAI no depende de ella sin autorización explícita ni de un checkpoint perdido
  (`docs/paper/artefactos-perdidos.md`).
- Si una metrica sube mas de lo que el spec anticipa, NO lo celebres: audita regimen,
  denominador y punto de operacion primero y reportalo.

## Cierre

- Ejecuta las pruebas de falsabilidad del spec §6 y reporta el resultado sea cual sea,
  incluidos los nulos.
- `make lint && make test-ml` — si fallan, corrigelos antes de reportar. Una reparacion de
  protocolo lleva un test que falla sobre la version anterior.
- NO escribas en el spec ni en `docs/us-work/`. Devuelve al orquestador un resumen de
  <=30 lineas: archivos, decisiones, desviaciones del spec, artefactos generados (ruta,
  regimen, unidad, semilla, commit, PENDIENTE DE SELLAR), metricas con intervalo, MLflow run id,
  pendientes o conflictos de frontera (columnas del parquet, claves del JSON que consume paper).
- No sincronices Engram ni reindexes el grafo: el orquestador integra tu resumen en fuentes
  revisables y decide si actualiza herramientas locales.
- El limite NO aplica a advertencias que QA necesita: deprecations, workarounds, fallos
  intermitentes o tracebacks residuales van tras el resumen como "ANEXO TECNICO".

**Modo nocturno**: el computo en CPU no gasta dinero y puede correr; nada de GPU alquilada, nada
de sellar, nada de las EPIC 20/21/22/25. Los artefactos quedan PENDIENTES DE SELLAR.
