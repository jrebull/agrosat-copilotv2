---
name: agrosat-protocolo-articulo
description: La skill mas importante del articulo MICAI 2027 — traduce el preregistro, el contrato del estimando (docs/paper/estimando-v1.json) y el ledger de custodia (paper/ARTIFACTS.md) a reglas ejecutables para cualquier comparacion, intervalo, contraste, artefacto o cifra. Use ANTES de escribir o auditar evaluacion (ml/eval/paper_micai_*, set_valued, decision_cost), de generar un artefacto en reports/paper_micai/, de sellar, o de interpretar una metrica que parezca buena.
allowed-tools: Read, Bash, Glob, Grep
---

# Protocolo del articulo — estimando, regimen, intervalos, custodia

## Los tres defectos que ya nos costaron un manuscrito

| Defecto | Sintoma en codigo | Regla que lo impide |
|---|---|---|
| **Denominador movil** | El universo de clases del macro sale de las parcelas *entregadas* por el mecanismo | `macro_over(universe=...)` exige el universo del bloque, calculado **desde entrenamiento** (`class_universe_source: training_only`, `common_class_universe: true`) |
| **Punto de operacion elegido dentro del bloque evaluado** | Un umbral, una tasa o una igualacion "a igual cobertura" se ajusta mirando la prueba | `operating_point_source: training_validation_only`, `rematch_on_test: false`. Invariancia del umbral comprobada por test |
| **Remuestreo en la unidad equivocada** | Bootstrap sobre parcelas dentro del bloque: 16 640 replicas que no existen | `paired_interval(unit=...)` obligatorio; `dependence_cluster: patch_id`; por debajo de `minimum_unique_paired_clusters: 3` no hay intervalo ni p |

Los tres estan reparados en `ml/eval/paper_micai_coverage.py` con tests que fallan sobre la
version anterior (`tests/ml/eval/test_paper_micai_coverage.py`). Los artefactos que produjo la
version defectuosa (`fase3/*`, `fase4/replica_*`, `potencia/*`, `bloques/*`, `equidad/*`) son
`OBSOLETO` hasta regenerarse (US-124, US-125): se citan solo en bloques CUARENTENA y nunca en el
manuscrito. `make paper-obsoletos-check` lo vigila.

## El contrato del estimando (fuente normativa: `docs/paper/estimando-v1.json`)

| Campo | Valor | Consecuencia practica |
|---|---|---|
| `scope` | `dataset_conditional` | La inferencia es condicional a cada banco; **`transport_claim: false`** |
| `population` | `all_eligible_test_parcels` | `include_non_delivery: true`: la parcela no entregada cuenta, no se descarta |
| `analysis_unit` / `dependence_cluster` | `parcel` / `patch_id` | Intervalos pareados por cluster; K (`k_role`) es sensibilidad espacial, **no** replica |
| `pool_across_datasets` | `false` | Nunca se agrupan bancos en una sola cifra |
| Funcion de perdida L | **no esta en el JSON** | La fija US-172 con usuarios reales; hasta entonces no hay "quien gana" |

`make preregistro-check` falla si el JSON y la seccion 4.5 del preregistro divergen. Cambiar el
contrato es una revision del preregistro, no un parametro del script.

## Regimenes — uno por comparacion, nombrado en la frase

| Regimen | Que es | Ejemplo de etiqueta |
|---|---|---|
| fold-5 held-out por parcela | miembros entrenados en folds 1-4, puntuados sobre las 16 640 parcelas del fold 5 | "F1-macro, fold-5 held-out, parcela" |
| **in-sample para el meta-modelo** | `StackingEnsemble.fit` reajusta sobre todas las parcelas del fold 5; Optuna optimiza pesos sobre esas etiquetas | "0,7486 (in-sample, meta-modelo)" |
| out-of-fold (OOF) | posteriores por parcela de un modelo que nunca vio esa parcela | "OOF, cinco folds" |
| pixel | metrica densa por pixel (mIoU) | no se compara con F1 por parcela |

Un numero sin regimen no se reporta. El 0,7486 / 0,7470 nunca se imprime como held-out.

## Reglas de cada artefacto en `reports/paper_micai/<fase>/`

- Sale de un runner versionado (`scripts/run_paper_micai_*.py`), nunca de un notebook ni a mano.
- Lleva semilla, versiones de computo (`xgboost`, `scikit-learn`, `polars`, `numpy`), commit y,
  si compara, la prueba pareada con su intervalo y su unidad.
- **Multiplicidad**: Holm sobre los K contrastes, y la lista confirmatorio / exploratorio escrita
  en el spec **antes** de correr (regla R5 de ADR-013).
- **Semillas**: diez sobre todo el pipeline cuando la US lo pide (US-161); una conclusion que
  depende de la semilla se retira.
- **Sellado**: fila en `paper/ARTIFACTS.md` via `make paper-artifacts-seal` (MD5, bytes, commit,
  estado). Sin fila no se imprime. Nunca se sobrescribe un archivo sellado: archivo nuevo y el
  viejo cambia de estado con motivo.
- Gate: `make paper-artifacts-check` (probado en negativo), `make oof-manifest-check` para los
  parquets OOF (cada `oof_parcel_*` con entrada en `ml/eval/oof/manifest.json`).

## Candados de alcance

- **ADR-014 §7**: ningun calculo de las EPIC 20, 21, 22 ni 25 antes del preregistro firmado; el
  preregistro no se firma antes de US-172. La mecanica se implementa y se testea; la corrida no.
- **Panel, no ganador** (ADR-014 §6, US-139): >= 3 predictores de familias distintas por banco;
  el predictor es factor de sensibilidad. Un ganador solo por seleccion anidada independiente.
- **Afirmaciones prohibidas** (ADR-013 y su enmienda, ampliadas en ADR-014): «el contexto
  espacial no aporta», «AlphaEarth codifica la fenologia», «cuantas clases prometer es medible»
  como novedad, «adoptamos Be My Eyes al pie de la letra», «el ensamble mejora al mejor miembro»,
  «0,7486 held-out», «FarSLIP aporta senal complementaria», cualquier transporte.
- **Los cuatro mecanismos** (recortar catalogo, abstenerse, conjunto conforme, retroceso
  jerarquico) son predictores con valores de conjunto (`ml/eval/set_valued.py`); "a igual
  cobertura" no esta definido para conjunto y clase gruesa sin una perdida declarada.

## Cuando una metrica parece demasiado buena

1. Regimen: es in-sample o held-out? Que vio el meta-modelo?
2. Denominador: el universo de clases cambio entre brazos?
3. Punto de operacion: se eligio mirando la prueba?
4. Unidad: el intervalo remuestreo parcelas o clusteres?
5. Semilla: sobrevive a diez semillas?
6. Provenance del checkpoint: con que folds se entreno (`MLflow train_folds`/`val_folds`,
   manifest con identidad unica)? Una caida > 0,15 entre `best_metrics` del checkpoint y el
   fold 5 exige explicacion antes de imprimir nada (US-119).

## Salida esperada al auditar

Tabla por cifra o contraste: regimen · unidad y cluster · universo · punto de operacion · semilla
· fila del ledger (SELLADO / PENDIENTE / OBSOLETO) · seccion del preregistro que lo autoriza ·
veredicto. Sin recomendaciones genericas de estilo.
