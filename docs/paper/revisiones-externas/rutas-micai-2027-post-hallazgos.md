# Contraste del informe de hallazgos del harness OOF y rutas para continuar hacia MICAI 2027

**Documentos contrastados**: `docs/paper/hallazgos-harness-oof-fold5-2026-09-03.md` (informe del harness, 3-sep 12:06) frente a `docs/paper/evaluacion-cuaderno-micai-2027.md` (evaluación del cuaderno público, 3-sep 11:24) y el cuaderno <https://agrosat2027.netlify.app> (revisión 2-sep).
**Base**: repo `main` @ `471d64a` + working tree (fix de `dump_oof.py`, `manifest.json` con 7 modelos, dos `.dvc` nuevos, todo sin commit), engram #822, #823, #824, #826, #827, checkpoints locales leídos directamente.
**Fecha**: 3-sep-2026.

---

## 0. Resumen

El informe de hallazgos es sólido, está bien trazado y coincide en lo esencial con la evaluación del cuaderno. Aporta tres cosas que el cuaderno no sabía el 2-sep: el fix del harness, el recálculo de la línea US-043 y la pérdida de la VM H100 con tres checkpoints FarSLIP. Con eso, cinco de los seis "urgentes/decisiones" de la pestaña Pendientes cambian de estado.

Tiene cuatro puntos ciegos que hay que cerrar antes de que sus cifras entren a cualquier paper:

1. **Sus recálculos de ensambles siguen en el régimen in-sample.** Stacking-5 0.7840 y Blending-5 0.7884 salen de la misma función que produjo el 0.7486 (`_stacking_metrics`: meta-modelo reajustado sobre todas las parcelas y puntuado sobre ellas; en Blending, Optuna optimiza los pesos sobre las mismas etiquetas del fold 5). El informe no lo dice. La consecuencia es favorable a su conclusión: si ni siquiera in-sample los ensambles superan a un individual que no ajustó nada sobre el fold 5 (0.7883), en régimen libre de fuga quedan por debajo. El "empate técnico" es en realidad una derrota del ensamblado, y refuerza el nulo del árbitro del cuaderno (US-089).
2. **Declara "calidad real" para `utae` y `anysat` sin aplicar su propio control §5.6.** El checkpoint `checkpoints/segmentation/utae-isaac/best_model.pt` guarda `val_miou 0.4826 / val_f1 0.6163` (epoch 38) y en fold 5 rinde mIoU 0.1605 (denso) y F1 0.188 (parcela). Una caída de 0.32 en mIoU es exactamente el patrón que el informe propone detectar con umbral 0.15. AnySat: 0.4459 en fold 4 (plan v8) frente a 0.1684 en fold 5. Hasta que no se explique esa caída (folds distintos en el entrenamiento, preprocesado distinto, checkpoint equivocado), la frase del cuaderno "U-TAE rinde 0,19, no hay con qué combinarlo" no puede imprimirse.
3. **El miembro desplegado no existe como identidad.** `tsvit-pheno-fullm-v2` (T=32, el que sirve el copiloto en Voting-3 v2) no está en `CHECKPOINT_REGISTRY` (8 claves), no está en `ml/eval/oof/manifest.json`, y en `ml/eval/oof_new32/manifest.json` aparece bajo el nombre `tsvit-pheno` con rutas `F:\projects\...` de la VM perdida. Hay tres checkpoints distintos llamados "tsvit-pheno" en distintos sitios (v1 T=10 dim128; fullm-v1 T=37 dim192; fullm-v2 T=32 dim192). Es el mismo tipo de ambigüedad que produjo el bug.
4. **"El paper" es ambiguo.** La sección 7 del informe habla del manuscrito E11 (`paper/main.tex`, 24 páginas) pero el trabajo activo es el cuaderno MICAI. Hoy ninguno de los dos refleja el fix.

---

## 1. Contraste punto por punto

| Tema | Informe de hallazgos | Evaluación del cuaderno | Veredicto conjunto |
|---|---|---|---|
| Fuga en `tsvit-pheno-v1` | No hay; train (1,2,3) / val 4; conclusión inferencial, sin provenance | Igual (engram #822); marca obsoleto el pendiente 01 del cuaderno | Coinciden. Cerrar pendiente 01 y abrir la deuda de provenance (§5.4 del informe) |
| Bug `n_timesteps` | Mecanismo, fix, 47 tests, re-dump, fullm 0.2552 → 0.7883 | Igual (engram #823); exige commit y recálculo antes de tocar el sitio | Coinciden. El fix sigue sin commit: es el primer bloqueante de todo |
| Mejor individual | `tsvit-pheno-fullm` 0.7883 | Igual, y añade que el cuaderno lo tiene al revés | Coinciden. Falta decidir cuál de los tres "tsvit-pheno" es el modelo del artículo (ver §3, Ruta 1) |
| US-043 recalculado | Stacking-5 0.7840, Blending-5 0.7884; FarSLIP cae a +0.0047 / +0.0009 | No lo tenía; sí tenía que el 0.7486 es in-sample por código | Cifras válidas pero **del mismo régimen in-sample**. Publicar las tres columnas (in-sample, promedio de macros por bloque, agrupado) o no publicar |
| Aporte FarSLIP | Se desploma | Nulo 2 del cuaderno (+0.0006 agrupado) | Coinciden en la dirección. US-092 se cierra con cifras nuevas bajo régimen agrupado |
| `utae` / `anysat` | "Su calidad real" | Discrepancia F1 parcela vs mIoU literatura; checkpoint subentrenado o mal configurado | **Discrepan.** `utae-isaac` guarda val_miou 0.4826; aplicar §5.6 antes de afirmar nada |
| Voting-3 v2 | Sano (0.7839 con `oof_new32`) | Ausente del cuaderno; es el producto cuya decisión motiva el artículo | Coinciden en que está sano. Debe entrar al registry, al manifest y a la tabla de miembros |
| Checkpoints FarSLIP perdidos | Existieron solo en la VM H100; VM no disponible; OOF sobreviven | Decía "nunca existieron" (evaluación corregida el 3-sep) | El informe tiene razón. Pérdida de generador, no de artefacto. Declararlo en reproducibilidad |
| E-a (US-041) | No resucitar; resultado negativo con salvedad | Igual | Coinciden |
| H100 | Pérdida como riesgo de reproducibilidad | Riesgo de cómputo para US-104 | Ambos: la pestaña Pendientes del cuaderno ("ya no hace falta pedir la H100") es cierta por la razón equivocada. Hace falta plan de cómputo |
| Régimen de métricas | No lo trata | Eje central | Único punto en que la evaluación va más lejos que el informe |
| Provenance MLflow | §5.4 folds ausentes en `build_and_train` | Igual | Coinciden. Añadir también md5 del checkpoint y `n_timesteps` |
| Manifest | Ampliado a 7 modelos | Pedía exactamente eso | Hecho, sin commit. Falta `tsvit-pheno-fullm-v2` |
| Qué cambia en el paper | Tabla para el manuscrito E11 | 16 correcciones al cuaderno MICAI | Son dos destinos distintos. Decidir el destino del E11 (Ruta 1) |

---

## 2. Áreas de oportunidad consolidadas (unión de ambos documentos, por riesgo)

| # | Área | Qué hacer | Fuente |
|---|---|---|---|
| A | Harness con dos rutas de scoring | Constructor único `build_scoring_dataset_kwargs(spec, fold)` en `ml/eval/`, consumido por `dense_metrics.py` y `dump_oof.py` | Informe §5.1 |
| B | Sin test de coherencia T | Test de regresión: para cada spec con `n_timesteps`, el dataset del volcado devuelve tensores con esa T | Informe §5.2 |
| C | `strict=False` silencioso | Error si faltan `temporal_pos_ordinal`, `temporal_pos_embedding`, `spatial_pos_embedding` | Informe §5.3 |
| D | Sanidad entre miembros | Al cerrar un volcado, comparar F1/mIoU de cada miembro con `best_metrics` de su checkpoint; advertir si cae más de 0.15. **Aplicarlo ya a `utae` y `anysat`** | Informe §5.6 + este contraste |
| E | Régimen de cada cifra | Toda tabla de ensambles con tres columnas etiquetadas; renombrar `stacking_5_oof_cv` en `us043_farslip_summary.json`; corregir el comentario del abstract E11 | Evaluación §2.3 |
| F | Identidad de modelos | Clave `tsvit-pheno-fullm-v2` en registry y manifest; renombrar el alias `tsvit-pheno` de `oof_new32`; tabla de miembros con checkpoint, md5, T, dim, folds de entrenamiento | Este contraste |
| G | Provenance de entrenamiento | `train_folds`, `val_folds`, `n_train`, `n_val`, `n_timesteps`, md5 del checkpoint en `mlflow.log_params` de `build_and_train` | Informe §5.4 |
| H | Manifest sin huérfanos | Chequeo que falle si existe `oof_parcel_*.parquet` sin entrada | Informe §5.5 |
| I | Reproducibilidad FarSLIP | Declarar pérdida de `parcel/04cls`, `parcel/18cls`, `incremental/08cls`; reentrenar `parcel/04cls` solo si E-a o la fenología entran al artículo (para MICAI no entran) | Informe §6 |
| J | Fuente única de verdad | Los artefactos MICAI (`reports/paper_micai/`, `paper/micai/`, `docs/paper/*.md`, gate MD5) aterrizan en este monorepo; el cuaderno se despliega desde aquí o cita este repo | Evaluación §6 |
| K | Gobernanza | ADR-013 (venue, contribución acotada, autores, destino del E11), consentimiento y crédito de coautores y sponsor, divulgación de IA (agentes auditores y asistencia) | Evaluación §6 |
| L | Cómputo | Plan explícito para US-104 (ver §4) | Ambos |

---

## 3. Rutas para avanzar hacia MICAI 2027

Orden de dependencia. Las estimaciones en SP siguen la escala del plan v8.

### Ruta 0. Cerrar la deuda del harness en `main` (esta semana, ~8 SP)

Rama `fix/E5-US-117-oof-harness-n-timesteps`. Contenido:

1. Commit del fix de `ml/eval/oof/dump_oof.py`, `manifest.json` (7 modelos) y los dos `.dvc` ya pusheados a GCS.
2. Áreas A, B, C, D, G, H de la tabla anterior. Cada una es pequeña; juntas evitan que el mismo fallo vuelva a salir por otra puerta.
3. Re-correr US-043 de forma canónica (con MLflow, sin `--no-mlflow`) y publicar en `reports/ensemble/metrics/` los CSV con las tres columnas de régimen. No pisar los CSV antiguos: nueva versión con sufijo y nota en el `README` de la carpeta.
4. Actualizar `docs/model_cards/tsvit-pheno.md` y `docs/model_cards/ensemble-final-e6.md`.
5. Correr la sanidad D sobre `utae`, `anysat` y `segformer` y documentar la causa de cada caída (aunque sea "checkpoint de un fold distinto, no comparable").

Aparte, commit separado `chore(E10)` con el Terraform de FinOps (`farslip_vm_enabled`) y `docs/infra/archivo-disco-farslip-vm.md`, que llevan desde el 25-ago sin commit y no tienen nada que ver con el paper.

### Ruta 1. Decisiones que solo el equipo puede tomar (ADR-013, ~3 SP)

1. **Venue y contribución**: MICAI 2027, contribución acotada (F1-macro no comparable entre catálogos; frontera retirar-clases vs abstenerse con estimando alineado). Registrarlo en `docs/decisions/ADR-013-micai-2027.md`.
2. **Destino del manuscrito E11** (24/25 páginas, cifras previas al fix): congelarlo como informe técnico interno, o publicarlo como preprint del sistema tras aplicar el fix. No dejarlo en limbo con cifras inválidas.
3. **Modelo del artículo**: recomendación, `tsvit-pheno-fullm-v2` (T=32, el desplegado), porque la premisa del artículo es la decisión de despliegue; `fullm-v1` (0.7883) y `tsvit-pheno-v1` (0.7367) como réplicas en US-097. Si se elige v1 o fullm-v1 hay que decir por qué el artículo no estudia el producto real.
4. **Autoría, consentimiento y crédito**: coautores del integrador cuyos checkpoints alimentan la tabla (`utae-isaac`, `segformer-isaac`, `unet-aaron`, `anysat-aaron`), sponsor Dr. Camacho (H100), segundo autor del cuaderno. Antes de escribir una línea del manuscrito, no "solo en la versión final".
5. **Divulgación de IA**: los cuatro auditores de US-093 son agentes; el código y el texto se produjeron con asistencia de IA. Redactar el párrafo ahora.
6. **Fuente única de verdad**: mover el cuaderno y sus artefactos a este monorepo (`paper/micai/`, `reports/paper_micai/`, `docs/paper/`), con el gate MD5 como target de `Makefile` junto a `paper-cite-check`.

### Ruta 2. Rehacer EPIC 13 con el protocolo endurecido (~8 SP, CPU)

- US-088: tabla de miembros con once filas (los diez del cuaderno más `fullm-v2`), cada una con régimen, checkpoint, md5, T y folds de entrenamiento. Solo después de la sanidad D.
- US-089: cuatro reglas de combinación con las tres columnas de régimen. Se espera que el resultado sea "ninguna combinación supera al individual en régimen libre de fuga", ahora con el miembro sano.
- US-091 y US-092: repetir con el miembro sano; el nulo de FarSLIP queda más firme (+0.0047 / +0.0009 in-sample).
- Aplicar al cuaderno las 16 correcciones de la evaluación (tabla §7) y cambiar la fecha de revisión.

### Ruta 3. Núcleo del artículo: EPIC 14 (~20 SP, CPU, con los OOF que ya existen)

- US-094 a US-100 tal como están planteadas, con tres ajustes: (i) el mecanismo "desplegable, sin oráculo" reutiliza la restricción por predicción del producto (`ml/eval/class_remap.py`, `ClassifyParcelInput`, campos `out_of_vocabulary_classes` y `unresolved_candidate`); (ii) US-098 se relabela: el criterio declarado del equipo fue F1 por clase con umbral 0.90 y cobertura medida (`reports/voting_new/cardinalidad.json`, `cumulative_support_share` 0.9054), la retirada por soporte es un tercer brazo alternativo; (iii) US-097 replica sobre `fullm-v1`, `tsvit-pheno-v1` y `xgb-alphaearth`, no solo sobre "el segundo mejor".
- Bootstrap pareado por parche (US-096) y multiplicidad (US-099) preregistrados antes de mirar resultados; el preregistro vive en este repo.

### Ruta 4. Cómputo para US-104 (reentrenamiento OOF de cinco folds)

La H100 ya no existe para el equipo y la L4 de GCP fue destruida el 25-ago (recuperable: `farslip_vm_enabled = true` en Terraform y `gcloud storage rsync` desde `gs://agrosat-artifacts-dev/vm-archive/farslip-data-dev-125/`). Opciones, con estimaciones que hay que confirmar con una prueba de humo de una época:

| Opción | Config | Tiempo estimado | Costo estimado | Observación |
|---|---|---|---|---|
| a | `tsvit-pheno-v1` (T=10, dim 128) en RTX 4070 Laptop | 31 min por fold en L4; del orden de 1 h por fold en la laptop; ~5 h total | 0 USD | No es el modelo desplegado. Solo vale si Ruta 1 elige v1 |
| b | Full-M (T=32/37, dim 192) en L4 reprovisionada | Horas por fold en H100 (US-039, best_epoch 36 de 40); en L4 varias veces más; del orden de 100 a 150 GPU-h en total | ~85 a 130 USD a ~0.85 USD/h (g2-standard-8) más disco | Requiere bajar batch (24 GB de VRAM) |
| c | Full-M en A100 40 GB (GCP spot o similar) | Del orden de 30 a 40 GPU-h | ~40 a 60 USD spot (verificar precio del día) | Mejor relación tiempo/costo si hay cuota |
| d | Volver a pedir H100 al sponsor | Según disponibilidad | 0 USD | Fuera del control del equipo |

Recomendación: (c) si hay cuota, si no (b). Cabe en el presupuesto one-time de entrenamiento del plan (262 a 602 USD) sin tocar el operativo. Precondiciones: decidir el modelo (Ruta 1), tener el harness endurecido (Ruta 0) para que los cinco volcados salgan con folds, T y md5 registrados en MLflow, y `dvc push` por fold. Sin Ruta 4, la validez externa se apoya solo en BreizhCrops y el meta-modelo sigue entrenándose con bloques de un solo fold (US-105 no se hace).

### Ruta 5. Validez externa (EPIC 15, ~21 SP)

- US-102 y US-103 (BreizhCrops, clasificador por parcela en CPU) pueden correr en paralelo con Ruta 3; los datos ya están en DVC (`data/breizhcrops.dvc`, loader y features en el repo).
- US-105 depende de Ruta 4.

### Ruta 6. Manuscrito y entrega (EPIC 16 y 17, ~33 SP)

- Crear ya el esqueleto LNCS en `paper/micai/` de este repo (US-106) para que el gate de artefactos, el de citas y el de anonimato vivan con el código.
- `CITATION.cff` y licencia legible por GitHub (US-115) no dependen de nada: hacerlo en Ruta 0.
- US-116 (vigilancia de la convocatoria): históricamente el envío a MICAI cae entre finales de mayo y junio; verificar en cuanto se publique la convocatoria 2027.

---

## 4. Calendario hacia atrás (tentativo, hasta confirmar la convocatoria)

| Ventana | Rutas | Entregable |
|---|---|---|
| sep-2026 | 0, 1 | PR del harness fusionada; ADR-013; cuaderno corregido y movido al monorepo |
| oct-2026 | 2, 3 (inicio), 5 (BreizhCrops) | Tabla de once miembros con tres regímenes; preregistro EPIC 14 |
| nov a dic-2026 | 3, 4 | Frontera con estimando alineado sobre `fullm-v2`; cinco volcados OOF con provenance |
| ene a feb-2027 | 3 (cierre), 5 (US-105) | Réplicas, multiplicidad, figura de la frontera; universo ampliado |
| mar a abr-2027 | 6 | Manuscrito completo bajo LNCS, gates en verde |
| may-2027 | 6 | Envío con una semana de holgura sobre el plazo real |

Es un calendario holgado a propósito: el cuaderno ya retiró un resultado por prisa, y el harness publicó una cifra con un fallo documentado desde junio.

---

## 5. US nuevas que el plan `/plan` no tiene (propuesta de numeración)

| US | Título | SP | Depende | Salida |
|---|---|---|---|---|
| US-117 | Endurecimiento del harness OOF (áreas A, B, C, D, G, H) y re-corrida canónica de US-043 con tres regímenes | 8 | — | PR en `main`, CSV nuevos, model cards |
| US-118 | ADR-013: venue, contribución, modelo del artículo, destino del E11, autoría, divulgación de IA | 3 | — | `docs/decisions/ADR-013-micai-2027.md`, `docs/paper/metadatos-autoria.md` |
| US-119 | Sanidad de miembros heredados (`utae`, `anysat`, `segformer`): causa de la caída entre validación propia y fold 5 | 3 | US-117 | Informe por miembro; decisión de incluir, corregir o excluir |
| US-120 | Identidad `tsvit-pheno-fullm-v2`: registry, manifest único, renombrado del alias en `oof_new32`, tabla de miembros con md5/T/folds | 2 | US-117 | Registry y manifest actualizados |
| US-121 | Plan de cómputo y reprovisión (Terraform L4 o A100), prueba de humo de una época, presupuesto aprobado | 2 | US-118 | `docs/operations/computo-micai-2027.md` |
| US-122 | Migración del cuaderno y del gate MD5 al monorepo; despliegue de Netlify desde este repo | 3 | US-118 | `paper/micai/`, target `make paper-artifacts-check`, `netlify.toml` |

---

## 6. Riesgos que quedan abiertos

1. **Volver a imprimir cifras del régimen equivocado.** Mitigación: tres columnas obligatorias en todo CSV de ensambles y en el gate de artefactos.
2. **La sanidad de `utae`/`anysat` cambia la tabla otra vez.** Mitigación: US-119 antes de US-088.
3. **Sin GPU no hay US-104 ni US-105.** Mitigación: Ruta 4 decidida en septiembre, no en enero.
4. **Autoría sin consentimiento.** Mitigación: US-118 antes del primer borrador.
5. **Dos repos con cifras distintas.** Mitigación: US-122; hasta entonces el cuaderno no publica ninguna cifra que no exista en `main`.
6. **El fix crítico sigue sin commit.** Mitigación: Ruta 0 hoy.
