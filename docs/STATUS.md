# STATUS — estado del repositorio

> Documento referenciado por el plan v8 y por varias US (051, 062, 064) que nunca
> llegó a existir en el upstream. Esta versión lo materializa en el fork con lo que
> hay en disco y en la base de datos; para el detalle histórico de cada US ver
> [`us-resolved/`](us-resolved/) y para lo abierto [`blockers/PENDIENTES.md`](blockers/PENDIENTES.md).

Última actualización: 2 de septiembre de 2026 (fase 0 del artículo MICAI cerrada, fase 1 en curso).

## Base de datos (fuente de verdad: `dbmate status`)

| Migración | Qué aporta |
|---|---|
| `20260511213942_initial_schema` | extensiones, `chat_sessions`, `aois` |
| `20260516210000_create_parcels` | `parcels` |
| `20260516210100_create_features_parcels` | `features_parcels` con `VECTOR(64)` |
| `20260615082041_create_rag_documents` | `rag_documents` para Spatial-RAG |
| `20260620000418_rls_multi_tenant` | **RLS aplicada**: `FORCE ROW LEVEL SECURITY` y política `tenant_isolation` por `session_id`; rol `agrosat_app` |
| `20260620002624_alter_chat_sessions_llm_model` | `llm_variant` pasa a `llm_model` con 4 variantes |
| `20260628120000_create_chat_messages` | `chat_messages` con RLS |
| `20260628130000_list_chat_sessions_fn` | función `list_chat_sessions(text)` |
| `20260628233613_add_canonical_parcel_id_to_parcels` | `canonical_parcel_id` |
| `20260630120000_widen_chat_sessions_llm_model_qwen_vl` | variante `qwen-vl` |

Tablas con RLS forzada: `aois`, `chat_messages`, `chat_sessions`, `features_parcels`, `parcels`.

## Modelos y artefactos

- Campeón desplegado: Voting-3 v2, 12 clases (`france-12`), F1-macro 0.8992 (`reports/agent_bench/perceiver_champion_eval_v2.json`).
- Modelo final del curso: Stacking-5 heterogéneo con FarSLIP, F1-macro 0.7486 **in-sample para el meta-modelo**, no held-out (`reports/ensemble/metrics/us043_farslip_grid.csv`); el Stacking-3 sin FarSLIP de US-040 da 0.7470 (`comparison_us040.csv`).
- Datos versionados en DVC (`gs://agrosat-dvc-remote`): features, OOF, embeddings FarSLIP, checkpoints y datasets de transferencia. Falta en el remoto `data/features/alphaearth_italia_2018.parquet`. PASTIS-R crudo no está en DVC: se descarga de Zenodo (53.7 GB, MD5 `4887513d6c2d2b07fa935d325bd53e09`) y en esta máquina ya está extraído en `data/PASTIS-R/` (gitignorado).

## Artículo MICAI 2027

- **Fase 0 cerrada** con veredicto de **reencuadre**: el ángulo A no se abandona pero se
  reordena. La contribución central pasa a ser el contraste entre dos mecanismos de recorte
  de cobertura (retirar clases enteras frente a rechazo por confianza) a igual cobertura y
  bajo F1-macro; el arbitraje heterogéneo queda como mecanismo y la vecindad espacial como
  control negativo acotado. Evidencia en [`paper/novedad.md`](paper/novedad.md); decisión en
  [`ADR-013`](decisions/ADR-013-angulo-micai.md), pendiente de firma de Arthur y Javier.
- Búsqueda sistemática sellada en `reports/paper_micai/fase0/`: 61 consultas automáticas en
  arXiv, Semantic Scholar y OpenAlex más seis manuales, con respuesta cruda por consulta, y
  una matriz de 43 trabajos con método, fortaleza, límite y hueco, todos con identificador
  resuelto por API (0 filas fuera de estado `OK`).
- Las tres referencias ancla del manuscrito tenían título y autores inventados. Ya están
  leídas y contrastadas: el artículo de Be My Eyes nunca menciona alucinación ni evalúa nada
  geoespacial, «Harvesting AlphaEarth» no recomienda adaptación few-shot y dice que el
  embedding anual carece de sensibilidad temporal, y AgroMind se contradice consigo mismo en
  el conteo de pares QA entre el resumen de arXiv y el texto completo de su v3.
- **Fase 1 en curso**: [`paper/ARTIFACTS.md`](../paper/ARTIFACTS.md) sella 51 artefactos con
  MD5 y declara 8 cifras sin artefacto. Gate `make paper-artifacts-check` en verde y probado
  en negativo. Los once parquets OOF coinciden con el `md5` de su propio `.dvc`.
- Corregido en el camino: `docs/pendientes-arthur.md` daba por perdidos artefactos que sí
  están en disco (la curva k-shot de EuroCropsML lleva en git desde `bc019e5`; los crudos de
  WorldCereal están en DVC).
- Hallazgo de reproducibilidad: seis artefactos sellados **no tienen driver versionado**,
  entre ellos `us043_honest_dropout_curve.csv` y `us043_farslip_grid.csv`, que sostienen la
  contribución central. Su cálculo sí está en `ml/` (`per_class_analysis`, `ml/ensemble/`);
  lo que falta es el guion que lo invocó. La fase 2 los vuelve a generar desde las
  posteriores OOF selladas y usa los CSV como comprobación cruzada.
- Sellado el ground truth del fold 5 en `reports/paper_micai/fase1/`: 16 640 etiquetas,
  sus centroides y el soporte por clase, 420 KB en total, derivados de PASTIS-R. Con eso la
  evaluación se reproduce sin los 68 GB del dataset. El eje de cobertura de la curva de
  cardinalidad ya reproduce exacto contra el CSV sellado en sus siete valores.

- **Fase 2 hecha.** Reproducido al sexto decimal: la cifra campeona 0,7486 es el meta-modelo
  reentrenado sobre las mismas parcelas que puntúa, no un held-out. Libre de fuga ninguna
  combinación mejora al mejor miembro individual (tsvit-pheno, F1-macro 0,7367). La
  contribución central se sostiene: a igual cobertura, retirar clases domina al rechazo por
  confianza **RETIRADA**: una auditoría ciega multiagente mostró que ese delta es un artefacto de
  promediar sobre conjuntos de clases distintos; al igualarlos el signo se invierte.
  El nulo de vecindad es un nulo limpio y el aporte de FarSLIP no se distingue de cero fuera
  del régimen in-sample. Detalle y advertencias en
  [`paper/fase2-hallazgos.md`](paper/fase2-hallazgos.md).
- **Encuadre aceptado**: artículo nuevo desde cero sobre el punto de operación, con el
  resultado negativo del ensamble dentro como sección de protocolo. El manuscrito heredado
  no se repara. Enmienda en [`ADR-013`](decisions/ADR-013-angulo-micai.md), firmada por
  Javier y pendiente de Arthur; justificación en [`paper/que-paper-sale.md`](paper/que-paper-sale.md).
  Plan reformulado en diez fases: robustez en CPU, BreizhCrops como segundo conjunto de
  datos, reentrenamiento OOF opcional en GPU, y escritura desde cero.
- Corroboración: `reports/ensemble/metrics/weighted_voting_pastis.csv` ya traía la cifra
  libre de fuga al lado de la publicada (0,536 junto a 0,747). La fila del voto simple `1/N`,
  que no ajusta nada, permite separar el coste de la fuga (0,068) del de promediar macros por
  bloque (0,143): el número honesto del stacking es **0,679**, no 0,536.
- El campeón desplegado es un Voting-3 sobre `france-12` con doce clases: el equipo ya
  **retiró seis clases a mano para desplegar**. La decisión que el artículo formaliza ya se
  toma en la práctica sin medirla.
- Pregunta abierta para Arthur: `tsvit-pheno` saca 0,7367 y `tsvit-pheno-fullm` 0,2552 sobre
  las mismas parcelas. Antes de publicar hay que comprobar con qué folds se entrenó cada
  checkpoint.

- **Fase 3 hecha.** La frontera rehecha sin los tres defectos dice que **H1 no se sostiene**:
  a K = 9 el intervalo incluye el cero en los dos predictores y nada sobrevive a Holm. Lo que
  sí demuestra es la descomposición: de dieciocho clases a ocho el F1-macro sube 0,2440 y
  **0,2155 de esa subida es solo el denominador**, sin mecanismo alguno. Y retirar por soporte,
  el criterio que el equipo usó al desplegar, es el peor de los tres. Detalle en
  [`paper/fase3-hallazgos.md`](paper/fase3-hallazgos.md).

## Entorno y gates

- Corre en macOS Apple Silicon y en Linux/Windows con GPU (ver [`runbook-local-setup.md`](runbook-local-setup.md), Apéndice D para Mac).
- Gates locales: `make check` (ruff, gitleaks, i18n), `make test` (backend, cobertura ≥ 70 %), `make test-ml`, `make test-frontend` (vitest, cobertura ≥ 50 %), `make paper-pdf`.
- CI (`.github/workflows/ci.yml`): lint Python y frontend, migraciones dbmate, Terraform validate, gitleaks e i18n. No ejecuta pytest por decisión del equipo.

## Pendiente

Ver [`blockers/PENDIENTES.md`](blockers/PENDIENTES.md). Lo que requiere terceros o la VM H100 sigue igual; la deuda de tests y documentación se está cerrando en el fork (ver el historial de `main`).
