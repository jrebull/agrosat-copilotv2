# Hallazgos del harness OOF fold-5: fuga descartada, bug de `n_timesteps` y su impacto en EPIC 6

> **CUARENTENA** — Este documento cita cifras derivadas de artefactos marcados `OBSOLETO` en
> [`paper/ARTIFACTS.md`](../../paper/ARTIFACTS.md), entre ellas las de EPIC 6 que salen de
> `reports/paper_micai/fase3/`. **Ninguna de esas cifras entra en el artículo** hasta regenerarlas
> (US-124, US-125). El hallazgo del documento —el bug de `n_timesteps` y su impacto— **no depende
> de ellas**: sus mediciones propias son de esta sesión y se sostienen solas.
>
> *Banner añadido por el gate de publicación, que detectó las citas. El contenido no se ha tocado.*

**Alcance**: auditoría de la sospecha de fuga de datos en `checkpoints/segmentation/tsvit-pheno-v1/best.pt` y del salto anómalo de F1-macro entre las dos variantes de TSViT-pheno sobre fold 5.
**Base**: repo `main` @ `471d64a` + working tree, `data/PASTIS-R` local (2468 patches), GPU RTX 4070 Laptop.
**Fecha**: 3-sep-2026.
**Naturaleza de las cifras**: todas medidas en esta sesión sobre las mismas 16 640 parcelas de fold 5, con la misma verdad de terreno (`ml.agent.tools.classify._build_parcel_ground_truth`). Ninguna cifra viene de reportes previos salvo donde se marca como "publicado".

---

## 0. Veredicto en una página

1. **No hay fuga de datos.** `tsvit-pheno-v1` se entrenó con folds de train `(1,2,3)` y val `(4)`. El fold 5 nunca entró. El mejor modelo individual del paper no está contaminado y la sección de resultados no hay que rehacerla por esa causa.
2. **Pero no existe registro a nivel de corrida que lo demuestre.** El run de MLflow `alt-tsvit-pheno-v1` no registra `train_folds` ni `val_folds`, y la VM L4 donde se entrenó ya fue dada de baja sin conservar log. La conclusión se sostiene por cadena de código y por huella numérica, no por provenance. Es una debilidad real y corregible.
3. **El salto de 0,48 entre variantes era un bug del harness, no una propiedad de los modelos.** `ml/eval/oof/dump_oof.py` no propagaba `n_timesteps` al dataset: `tsvit-pheno-fullm` fue entrenado con T=37 y el harness le entregaba T=10, desalineando la codificación posicional ordinal. Corregido, pasa de **0,2552 a 0,7883** de F1-macro.
4. **Cambia el mejor modelo individual**: `tsvit-pheno-fullm` (0,7883) supera a `tsvit-pheno` (0,7367).
5. **El bug contaminaba toda la línea US-043.** Recalculada, Stacking-5 sube de 0,6477 a **0,7840** y Blending-5 de 0,5866 a **0,7884**.
6. **Dos conclusiones del paper quedan debilitadas y hay que reescribirlas**: (a) el aporte de los miembros FarSLIP se desploma al arreglar el miembro denso, y (b) ningún ensamble supera ya al modelo individual corregido.
7. **El modo de fallo estaba documentado en el propio repo desde US-039** y aun así se publicó una cifra afectada por él. Eso es lo más accionable de todo este informe.

---

## 1. Pregunta 1: ¿con qué folds se entrenó `tsvit-pheno-v1`?

**Respuesta: train `(1,2,3)`, val `(4)`. Fold 5 excluido.**

### 1.1 Lo que NO existe

| Fuente esperada | Estado |
|---|---|
| Params de folds en el run MLflow `alt-tsvit-pheno-v1` | **No existen**. Params registrados: `amp`, `batch_size`, `ce_weight`, `device`, `dice_weight`, `epochs`, `ignore_index`, `lambda_contrast`, `lr`, `num_classes`, `optimizer`, `use_phenology` |
| Log de la corrida de entrenamiento | **No existe**. Se entrenó en la VM L4 `agrosat-farslip-trainer-dev`, dada de baja |
| Folds embebidos en el checkpoint | **No**. Claves: `epoch`, `model_state`, `optimizer_state`, `scaler_state`, `scheduler_state`, `best_metrics` |

### 1.2 Evidencia que sí sostiene la conclusión

1. **Cadena de código, sin ruta alternativa.** El notebook [`5b_tsvit.ipynb`](../../notebooks/segmentation/5b_tsvit.ipynb) invoca el CLI en `_train_via_cli` **sin** `--train-folds` ni `--val-folds`, por lo que rigen los defaults `"1,2,3"` y `"4"` de [`train_segmentation.py`](../../ml/train/train_segmentation.py). `build_and_train` construye `PASTISSegmentationDataset(folds=train_folds)` y `(folds=val_folds)` de forma directa: no pasa por `pastis_fold_split`, así que el fold 5 no llega a instanciarse por ninguna vía.
2. **Huella numérica.** El `best_metrics` del checkpoint (mIoU 0,6253 / F1 0,7500 / pixAcc 0,8759, best_epoch 28) fue reproducido de forma exacta por una evaluación local independiente **sobre fold 4** (482 patches). Si el fold de validación hubiera sido el 5, una evaluación sobre fold 4 no coincidiría a cuatro decimales.
3. **Prueba por rendimiento.** Sobre fold 5 el modelo da 0,7401 (denso) y 0,7367 (parcela), **por debajo** de su fold de validación 4 (0,7500). Una fuga produciría lo contrario: el fold visto rendiría mejor que el de validación.
4. **Log de re-scoring limpio.** [`us030_rescore_run.log`](../../reports/segmentation/metrics/us030_rescore_run.log) registra `folds=(5,)`, `rescore_norm_train_only train_folds=(1, 2, 3)` (normalización derivada solo de train) y `n_missing_keys=0`.

---

## 2. Pregunta 2: el salto de 0,48 entre variantes

**Respuesta: bug de desalineación temporal en el volcado OOF.**

### 2.1 Mecanismo

Topología real leída de los pesos de cada checkpoint:

| Checkpoint | `dim` | `depth_temporal` | `temporal_pos_ordinal` | T entrenado |
|---|---|---|---|---|
| `tsvit-pheno-v1` | 128 | 4 | `(1, 10, 128)` | **10** |
| `tsvit-pheno-fullm-v1` | 192 | 6 | `(1, 37, 192)` | **37** |

`ml/eval/oof/dump_oof.py` construía `ds_kwargs` **sin la clave `n_timesteps`**, de modo que el dataset entregaba T=10 por defecto. En paralelo, `load_checkpoint_model` sí reconstruía el modelo con `TSVIT_FULLM_CONFIG` (T=37). El resultado: un modelo con codificación posicional ordinal dimensionada a 37 recibiendo series de 10, y una carga con `strict=False` que no aborta.

`tsvit-pheno-v1` era inmune porque nació con T=10: coincide con el default.

### 2.2 El modo de fallo ya estaba documentado en el repo

Este es el punto más incómodo del informe. El fallo estaba descrito, con su magnitud, en dos sitios:

- [`ml/models/tsvit_wrapper.py`](../../ml/models/tsvit_wrapper.py): *"el harness DEBE reconstruir con el mismo 37 o el PE ordinal se desalinea y el mIoU colapsa (mismatch 64/10 -> 0.17 en vez de 0.68, cierre de US-039)"*.
- [`ml/eval/dense_metrics.py`](../../ml/eval/dense_metrics.py): *"CRITICAL (US-038/039) ... TSViT Full-M entrenado con T=37 puntuó 0.17 cuando el harness le entregó T=10"*.

Y `dense_metrics.py` **sí tenía el guard**:

```python
ds_kwargs["n_timesteps"] = int(spec.model_kwargs.get("n_timesteps", 10))
```

`dump_oof.py` no. Un `grep` de esa línea la encontraba en un solo archivo de los dos que puntúan fold 5.

### 2.3 Corrección aplicada

Se replicó el guard en `dump_oof.py`, dentro de la rama `if is_temporal:` de `_dump_one`, con el comentario que explica el porqué y remite a `dense_metrics`. Validación: `ruff` y `mypy` limpios; `pytest tests/ml/eval/oof/` con **47 tests en verde**.

---

## 3. Impacto medido

### 3.1 Modelo individual (fold 5, 16 640 parcelas)

| Modelo | F1-macro | Accuracy |
|---|---|---|
| `tsvit-pheno` (v1, T=10) | 0,7367 | 0,8579 |
| `tsvit-pheno-fullm` — **antes** (bug) | 0,2552 | 0,5334 |
| `tsvit-pheno-fullm` — **después** (fix) | **0,7883** | **0,8811** |

Delta del fix: **+0,5331**. Nuevo mejor individual: `tsvit-pheno-fullm`, por +0,0516 sobre `tsvit-pheno`.

### 3.2 Ensambles US-043 (mismas 16 640 parcelas)

| Ensamble | Publicado (bug) | Recalculado (fix) | Delta |
|---|---|---|---|
| Stacking-3 | 0,6359 | **0,7793** | +0,1434 |
| Stacking-5 (+FarSLIP) | 0,6477 | **0,7840** | +0,1363 |
| Blending-3 | 0,5651 | **0,7875** | +0,2224 |
| Blending-5 (+FarSLIP) | 0,5866 | **0,7884** | +0,2018 |

### 3.3 Las dos conclusiones que se caen

**(a) El aporte de FarSLIP era en gran medida artefacto.** Era la tesis de US-043:

| Aporte de los 2 miembros FarSLIP | Con miembro denso roto | Con miembro denso sano |
|---|---|---|
| en Stacking | +0,0118 | **+0,0047** |
| en Blending | +0,0215 | **+0,0009** |

**(b) Los ensambles ya no superan al modelo individual.** `tsvit-pheno-fullm` solo da 0,7883; el mejor ensamble (Blending-5) da 0,7884. Empate técnico. La narrativa de EPIC 6 sobre la ganancia del ensamblado necesita reescribirse sobre estas cifras.

---

## 4. Alcance acotado: qué NO está afectado

Se midió cada miembro contra la misma verdad de terreno para delimitar el daño.

| Componente | Miembro denso | Estado |
|---|---|---|
| Voting-3 v2 (campeón del agente) | `tsvit-pheno` desde `ml/eval/oof_new32` (fullm-v2 @ T=32) → **0,7839** | Sano, no afectado |
| Stacking/Blending US-040 | `tsvit-pheno` v1 → 0,7367 | Sano |
| **US-043 Stacking-5 / Blending-5** | **`tsvit-pheno-fullm`** | **Afectado** (recalculado arriba) |
| E-a fusión dual-head (US-041) | `tsvit-pheno-fullm` | Afectado y **bloqueado** (ver §6) |

Miembros no temporales o sin `n_timesteps` en `model_kwargs` (`utae` 0,1880; `anysat` 0,1873; `segformer` 0,3382; `xgb-alphaearth` 0,5913) **no** sufren este fallo: el guard no altera su ruta. Sus cifras bajas son su calidad real y son coherentes con la tabla densa publicada.

---

## 5. Áreas de oportunidad

Ordenadas por riesgo de que vuelvan a producir una cifra incorrecta en el paper.

### 5.1 Dos rutas de scoring paralelas con lógica divergente

`dense_metrics.py` y `dump_oof.py` puntúan ambas fold 5 y construyen ambas un `PASTISSegmentationDataset`, pero con código independiente. El guard crítico vivía en una y no en la otra. Es una violación de DRY con consecuencia directa sobre resultados publicados.

**Solución propuesta**: extraer un único constructor compartido, por ejemplo `build_scoring_dataset_kwargs(spec, fold, ...)` en `ml/eval/`, y que ambas rutas lo consuman. El guard deja de poder existir "en una sola de las dos".

### 5.2 Ningún test detectaba la desalineación

Los 47 tests de `tests/ml/eval/oof/` pasaban con el bug presente. Ninguno afirma la coherencia entre el T del spec y el T que el dataset entrega.

**Solución propuesta**: test de regresión que, para cada spec con `n_timesteps` en `model_kwargs`, verifique que el dataset construido por el volcado devuelve tensores con esa T. Es un test barato y habría bloqueado esto.

### 5.3 Carga silenciosa con `strict=False`

`load_checkpoint_model` hace `model.load_state_dict(state, strict=False)` y solo registra `n_missing_keys` / `n_unexpected_keys` en el log. Una desalineación de forma en un buffer crítico como el PE ordinal se degrada a silencio.

**Solución propuesta**: elevar a error cuando falten claves de una lista de tensores críticos (`temporal_pos_ordinal`, `temporal_pos_embedding`, `spatial_pos_embedding`), conservando la tolerancia para buffers accesorios.

### 5.4 Provenance de folds ausente en MLflow

La ruta TSViT no registra `train_folds` / `val_folds`. La ruta U-Net/DeepLab (`train_and_eval`) sí los registra. Por eso la pregunta que originó esta auditoría no pudo responderse con un run.

**Solución propuesta**: añadir `train_folds`, `val_folds`, `n_train` y `n_val` a `mlflow.log_params` en `build_and_train`. Para un paper cuya afirmación central es "fold 5 held-out", que eso no esté en el tracking es una carencia de primer orden.

### 5.5 Artefactos publicados sin entrada en el manifiesto

`ml/eval/oof/manifest.json` listaba 6 modelos y **no incluía `tsvit-pheno-fullm`**, pese a que su parquet se consumía en US-043. Se materializó en una corrida aparte, sin registrar `code_version` ni `data_version`. Un artefacto sin provenance es exactamente el que nadie audita.

**Solución propuesta**: ya corregido (manifiesto con 7 modelos). Como control permanente, un chequeo que falle si existe un `oof_parcel_*.parquet` sin entrada correspondiente.

### 5.6 Ausencia de control de sanidad entre miembros

Un miembro denso que rinde 0,2552 cuando su propia validación dice 0,8078 es una discrepancia de 0,55 que ningún control señaló.

**Solución propuesta**: al cerrar un volcado, comparar el F1 de cada miembro contra el `best_metrics` de su checkpoint y emitir advertencia si la caída supera un umbral (por ejemplo 0,15). Es cinco líneas y cubre toda una familia de fallos de harness.

### 5.7 Artefactos críticos fuera de DVC

Ver §6. Es el riesgo abierto de mayor severidad.

---

## 6. Riesgo abierto: artefactos perdidos con la VM H100

Tres checkpoints existían **únicamente** en la VM H100 (`F:\projects\agrosat-copilot`), nunca versionados en DVC. La VM ya no está disponible:

- `checkpoints/farslip/parcel/04cls/best.safetensors` (macro-F1 0,6452)
- `checkpoints/farslip/parcel/18cls/best.safetensors`
- `checkpoints/farslip/incremental/08cls/best.safetensors`

**El impacto es de reproducibilidad, no de resultados.** `parcel/04cls` fue el generador de los dos miembros FarSLIP OOF (`gen_farslip_oof.py`), y esos parquets sí sobreviven en local y en DVC: por eso US-043 pudo recalcularse sin la VM. Lo que se pierde es la capacidad de regenerar el artefacto desde los pesos. Para un paper, es un hueco de provenance: el artefacto existe, su generador no.

**Qué sobrevive**: cinco checkpoints FarSLIP (`4band-pheno`, `baseline-nir`, `baseline-rgb`, `faithful_v2`, `incremental/04cls`), ambas torres base `FarSLIP{1,2}_ViT-B-16.pt`, las captions 69k, los splits filtrados y PASTIS-R completo. Los dos entrenadores (`run_us036b_parcel_sweep.py`, `run_us036a_farslip_full_incremental.py`) están en el repo.

**Recomendación**:

1. **Re-entrenar solo `parcel/04cls`**, con criterio de éxito explícito: reproducir ~0,6452 macro-F1. Cierra el hueco y desbloquea E-a. Advertencia: se entrenó en H100 y el hardware disponible es una RTX 4070 Laptop; requiere prueba de humo de VRAM y tiempo por época antes de comprometerse.
2. **Descartar `parcel/18cls`**: ningún archivo del repo lo referencia.
3. **`incremental/08cls` solo si la evaluación fenológica entra al paper**: su único consumidor es `scripts/farslip_eval_phenology.py`.
4. **No resucitar E-a (US-041)**: su resultado fue un fracaso (0,2694) por un problema de diseño de espacio de clases (fusión 4-vs-18), independiente de este bug. Con los ensambles ya sin ventaja sobre el individual, conviene reportarla como resultado negativo declarando explícitamente que su miembro denso estaba comprometido.

---

## 7. Qué debe cambiar en el paper

| Elemento | Cambio |
|---|---|
| Mejor modelo individual | `tsvit-pheno-fullm` 0,7883 (no `tsvit-pheno` 0,7367) |
| Cifra de `tsvit-pheno-fullm` | 0,7883 en toda tabla y figura; la de 0,2552 es inválida |
| Tabla de ensambles US-043 | Cuatro filas nuevas (§3.2) |
| Tesis sobre FarSLIP | El aporte cae a +0,0009 / +0,0047. La afirmación de que FarSLIP vía stacking es el campeón no se sostiene |
| Narrativa de EPIC 6 | Los ensambles no superan al individual (0,7884 vs 0,7883) |
| `docs/model_cards/tsvit-pheno.md` | Actualizar la comparativa fold-5 |
| Sección de reproducibilidad | Declarar la pérdida de los tres checkpoints y qué se conserva |

---

## 8. Trazabilidad

**Código modificado**: `ml/eval/oof/dump_oof.py` (guard `n_timesteps` en `_dump_one`).
**Manifiesto**: `ml/eval/oof/manifest.json`, ampliado de 6 a 7 modelos.

**Artefactos regenerados, versionados y subidos** (`dvc push` devolvió `2 files pushed`; `dvc status --cloud` devolvió `Cache and remote 'gcs-remote' are in sync`):

| Artefacto | md5 nuevo | md5 anterior |
|---|---|---|
| `ml/eval/oof/oof_parcel_tsvit-pheno-fullm_fold5.parquet` | `491df530cc03f34cd68268987758f76a` | `16758cb30ec2fc818bf0bed56e4f3a1a` |
| `ml/eval/oof/oof_tsvit-pheno-fullm_fold5.parquet` (246 MB) | `f1acb08c8bc6898b3e64149905c374a5` | no existía versionado |

**Reproducción del volcado corregido**:

```python
from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
from ml.eval.oof.dump_oof import dump_oof

dump_oof({"tsvit-pheno-fullm": CHECKPOINT_REGISTRY["tsvit-pheno-fullm"]},
         fold=5, out_dir=OUT, device="cuda", write_parcel=True)
```

**Reproducción de los ensambles**:

```bash
python -m scripts.run_us043_farslip_ensembles \
  --oof-dir ml/eval/oof --pastis-root data/PASTIS-R \
  --out-dir <out> --no-materialize --no-mlflow --device cuda
```

---

## 9. Salvedades de este informe

1. **Los CSV recalculados de US-043 no están publicados.** Quedaron fuera de `reports/`; `reports/ensemble/metrics/us043_farslip_stacking_blending.csv` conserva las cifras antiguas. La corrida usó `--no-mlflow`, así que carece de lineage: antes de publicar conviene repetirla de forma canónica con MLflow.
2. **El fix no está commiteado.** `ml/eval/oof/dump_oof.py`, `manifest.json` y los dos `.dvc` siguen en el working tree.
3. **La conclusión sobre los folds es inferencial, no documental.** Se apoya en cadena de código y coincidencia numérica exacta, no en un registro de la corrida. Es sólida, pero conviene declararla como tal y cerrar la carencia con §5.4.
4. **Las cifras de §3 y §4 son de nivel parcela** sobre 16 640 parcelas de fold 5. No son intercambiables con las métricas densas por píxel que aparecen en otras tablas del proyecto.
