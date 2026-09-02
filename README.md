# AgroSatCopilot v2 — Rumbo a MICAI 2027

> **Objetivo de este repositorio:** llevar un artículo científico derivado del proyecto
> AgroSatCopilot a **MICAI 2027** (Mexican International Conference on Artificial
> Intelligence, memorias en Springer LNAI).

**Mantenedor de este fork:** Javier Rebull ([jrebull](https://github.com/jrebull))
**Origen:** fork de [ArthurZizumbo/agrosat-copilot](https://github.com/ArthurZizumbo/agrosat-copilot),
Proyecto Integrador MNA del Equipo 17 (Tec de Monterrey), cerrado en julio de 2026.
**Licencia:** MIT (código). Datasets y modelos bajo sus licencias respectivas
([`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md)).

---

## 1. Qué se persigue aquí

El proyecto original entregó un sistema completo (percepción satelital, ensambles,
copiloto conversacional, transferencia multi-región) y dejó un manuscrito de journal
de unas 15 páginas en formato arXiv que nunca se sometió. Este fork existe para
convertir ese material en **una contribución publicable en MICAI 2027**:

1. Elegir **un ángulo** defendible con los artefactos que ya existen en el repo, no
   un paper de sistema que lo cuente todo.
2. Migrar el manuscrito al formato **Springer LNCS/LNAI** y ajustarlo al límite de
   páginas de la convocatoria (MICAI 2026 recomendó 12 y admitió hasta 20; la cifra
   de 2027 se confirma en la página oficial de autores cuando se publique).
3. Preparar la versión **doble ciego** para revisión y, si se acepta, el camera-ready
   con la licencia Springer firmada.

Las fechas de la convocatoria 2027 no están publicadas al momento de escribir esto y
**no se inventan**: se anotarán aquí en cuanto salgan.

## 2. El proyecto en una página

AgroSatCopilot cuantifica superficie por tipo de cultivo a partir de series temporales
de Sentinel-2 y explica sus predicciones en lenguaje natural. La hipótesis central es
que la señal que distingue cultivos vive en su **evolución fenológica**, no en una
imagen estática, y los resultados la confirman: los encoders temporales dominan a los
baselines densos 2D.

El copiloto sigue el patrón **Be My Eyes**: los modelos del equipo *perciben* cada
parcela y emiten observaciones en texto; un LLM congelado (Gemini 2.5 Pro en nube o
Qwen3.5-35B-A3B on-prem) *razona* sobre esas observaciones, invoca herramientas
geoespaciales y responde. El LLM nunca clasifica píxeles.

**Stack:** FastAPI + Polars · Nuxt 4 · PostgreSQL + PostGIS + pgvector · Google ADK ·
PyTorch (TSViT, U-TAE, AnySat, DeepLabv3+, SegFormer, U-Net) · AlphaEarth Foundations
(`SATELLITE_EMBEDDING/V1/ANNUAL` v1.1, CC-BY-4.0) · FarSLIP · DVC + MLflow + Dagster ·
Terraform GCP + Azure H100.

## 3. Cifras canónicas disponibles para el paper

Regla heredada del proyecto y que se mantiene: **cifras reales o nada**. Cada número
del manuscrito lleva un comentario `% src:` que apunta a su artefacto en `reports/`.

| Concepto | Valor | Fuente verificada |
|---|---|---|
| Mejor modelo individual: TSViT-pheno, PASTIS-R 18 clases, fold held-out | mIoU 0.6253 · F1-macro 0.7500 | `paper/sections/05_results.tex` |
| Modelo final: Stacking-5 heterogéneo con FarSLIP, held-out fold-5 | F1-macro 0.7470 · acc 0.8490 | `reports/ensemble/metrics/comparison_us040.csv` |
| Ganancia del ensamble sobre el mejor individual | +12.3 pp F1-macro | `paper/sections/05_results.tex` |
| Stacking-5 out-of-fold (libre de fuga) | F1-macro 0.6477 · acc 0.7935 | `reports/ensemble/us043_farslip_summary.json` |
| Aporte de FarSLIP (5 vs 3 miembros, OOF) | +0.0118 F1-macro | `reports/ensemble/us043_farslip_summary.json` |
| Modelo desplegado: Voting-3 v2, 12 clases (`france-12`) | F1-macro 0.8992 · acc 0.9375 | `reports/agent_bench/perceiver_champion_eval_v2.json` |
| Recableo del perceiver del agente al campeón (14 688 parcelas) | acc 0.8539 → 0.9375 | `reports/agent_bench/perceiver_champion_eval_v2.json` |
| Curva calidad-cobertura: 18 → 9 clases (campeón re-evaluado) | F1-macro 0.7486 → 0.9121 (~82 % de parcelas) | `reports/ensemble/metrics/ec_neighborhood_result.json` |
| Vecindad espacial k-NN sobre el campeón (E-c) | Δ +0.0002, no material | `reports/ensemble/metrics/ec_neighborhood_result.json` |
| Transferencia Francia → Cataluña (Sen4AgriNet) | zero-shot mIoU 0.0 → few-shot 0.2468 | `reports/segmentation/sen4agrinet_transfer_result.json` |
| Transferencia Francia → Alemania DE4 (Voting-3) | F1-macro 0.119 → 0.266 | `paper/sections/experiments_de4.tex` |
| Transferencia Italia, dataset completo (US-082) | **pendiente**: corrida bloqueada en la VM H100 | `docs/us-resolved/us-082.md` |
| Benchmark LLM multi-modelo (US-069) | **pendiente**: tabla con placeholders honestos | `docs/blockers/PENDIENTES.md` §2.1 |

## 4. Material de partida

| Recurso | Qué contiene |
|---|---|
| [`paper/`](paper/) | Manuscrito modular en inglés (`main.tex` + 12 secciones, ~12 000 palabras, 22 referencias), espejo en español en `sections_es/`, figuras y tablas reproducibles (`make paper-tables`, `make paper-figures`). Formato actual: `PRIMEarxiv.sty`. |
| [`docs/paper/papers-adicionales-propuesta.md`](docs/paper/papers-adicionales-propuesta.md) | Propuesta del equipo original de tres papers derivados con ángulo propio, cada uno con evidencia disponible y lo que falta. Punto de partida para elegir el ángulo de MICAI. |
| [`docs/blockers/PENDIENTES.md`](docs/blockers/PENDIENTES.md) | Índice único de lo que quedó abierto en el proyecto. Nada de esto se reclama en el paper sin cerrarlo. |
| [`docs/final_doc/`](docs/final_doc/) | Entregable final del curso (Avance 7) en LaTeX, ES y EN. |
| [`docs/presentation/`](docs/presentation/) | Presentación de defensa (Reveal.js, 65 láminas, ES/EN). |
| [`reports/`](reports/) | Métricas JSON/CSV y figuras de cada experimento. Es la fuente de verdad numérica. |
| [`docs/README-upstream-agrosat-copilot.md`](docs/README-upstream-agrosat-copilot.md) | README original del proyecto, conservado íntegro. |

## 5. Plan de trabajo hacia MICAI 2027

| Fase | Entregable | Estado |
|---|---|---|
| 0. Contexto | Fork, clon local, lectura del repo y de las reglas del proyecto | Hecho (2026-09-02) |
| 1. Ángulo | Decidir la contribución única del paper y su tesis en una frase | Pendiente |
| 2. Evidencia | Auditar cada cifra contra `reports/`; listar lo que falta y si es reproducible sin la VM H100 | Pendiente |
| 3. Manuscrito LNCS | Migrar a `llncs.cls`, recortar al límite de páginas, apéndice antes de referencias | Pendiente |
| 4. Doble ciego | Versión anónima, sin nombres, afiliaciones ni enlaces identificables | Pendiente |
| 5. Envío | Registro en el sistema de MICAI 2027, PDF anónimo | Pendiente (fechas por confirmar) |
| 6. Camera-ready | Fuentes LaTeX, figuras vectoriales, licencia Springer firmada a mano | Condicional a aceptación |

## 6. Qué se puede hacer desde este clon

- **Sí:** compilar el manuscrito (`pdflatex` o `tectonic` están instalados), regenerar
  tablas y figuras desde `reports/`, editar secciones, auditar citas
  (`scripts/paper_cite_check.py`).
- **No sin credenciales del equipo original:** descargar datos y pesos (DVC remoto
  `gs://agrosat-dvc-remote`, privado), reentrenar modelos (viven en la VM H100 del
  sponsor), correr la evaluación LLM pendiente.
- **Setup mínimo para el paper:**

```bash
git clone https://github.com/jrebull/agrosat-copilotv2.git
cd agrosat-copilotv2
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Para el stack completo (Docker, Poetry, pnpm) sigue el README original en
[`docs/README-upstream-agrosat-copilot.md`](docs/README-upstream-agrosat-copilot.md).

## 7. Reglas de trabajo en este fork

- **Autoría de commits:** todo commit y push se firma únicamente como `jrebull`
  (Javier Rebull). Sin trailers `Co-Authored-By` de asistentes de IA, sin pies
  "generado con" en PRs. Es regla del proyecto original y de este fork.
- **Integridad científica:** ninguna cifra sin artefacto real. Lo que no existe se
  declara pendiente, no se estima. No se concluye más de lo que miden los datos.
- **Idioma:** código, identificadores y docstrings en inglés; prosa de documentación
  en español neutro; el manuscrito en inglés.
- **Sin emojis** en código, commits, logs ni documentación.
- **Atribuciones obligatorias:** AlphaEarth (Khanna et al., GEE
  `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1, CC-BY-4.0), FarSLIP (arXiv:2511.14901),
  Be My Eyes (arXiv:2511.19417), PASTIS-R (Garnot et al., ICCV 2021), Sen4AgriNet y
  EuroCropsML (CC-BY-SA-4.0), TSViT, U-TAE, SegFormer, AnySat.
- Las guías operativas del proyecto original (`CLAUDE.md`, `AGENTS.md` y las de cada
  carpeta) siguen vigentes para el código.

## 8. Crédito al equipo original

AgroSatCopilot fue desarrollado por Carlos Isaac Ávila Gutiérrez, Carlos Aaron
Bocanegra Buitrón y Arthur Jafed Zizumbo Velasco (Equipo 17, Maestría en Inteligencia
Artificial Aplicada, Tec de Monterrey), con el Dr. Gerardo Jesús Camacho González como
sponsor académico. Este fork no altera esa autoría: la conserva y construye sobre ella.
La autoría del artículo para MICAI se acordará con ellos antes del envío.
