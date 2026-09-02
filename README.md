<div align="center">

<img src="docs/presentation/assets/figs/cover.png" alt="Parcelas agrícolas vistas desde satélite" width="100%">

<br>

# AgroSatCopilot v2

**Rumbo a MICAI 2027**

Un artículo científico derivado de AgroSatCopilot para la
*Mexican International Conference on Artificial Intelligence*

<br>

[![Meta](https://img.shields.io/badge/Meta-MICAI%202027-1f6f43?style=flat-square)](#hoja-de-ruta)
[![Formato](https://img.shields.io/badge/Formato-Springer%20LNAI-4b6584?style=flat-square)](#hoja-de-ruta)
[![Fase](https://img.shields.io/badge/Fase-1%20de%206-b9770e?style=flat-square)](#hoja-de-ruta)
[![Licencia](https://img.shields.io/badge/C%C3%B3digo-MIT-2d3436?style=flat-square)](LICENSE)
[![Origen](https://img.shields.io/badge/Fork%20de-ArthurZizumbo%2Fagrosat--copilot-6c5ce7?style=flat-square)](https://github.com/ArthurZizumbo/agrosat-copilot)

<br>

Mantenido por **Javier Rebull** · [jrebull](https://github.com/jrebull)

</div>

<br>

---

<br>

## Objetivo

El proyecto original entregó un sistema completo de percepción satelital, ensambles,
copiloto conversacional y transferencia multi-región, y dejó un manuscrito de journal
de unas 15 páginas en formato arXiv que nunca se sometió.

Este fork existe para una sola cosa: **convertir ese material en una contribución
publicable en MICAI 2027**. No un paper de sistema que lo cuente todo, sino un ángulo
defendible con los artefactos que ya existen en el repositorio.

<br>

<table>
<tr>
<td width="33%" valign="top">

**Punto de partida**

Manuscrito modular en inglés, 12 secciones, 22 referencias, cada cifra anclada a un
artefacto real en `reports/`.

</td>
<td width="33%" valign="top">

**Meta**

Artículo en formato Springer LNCS/LNAI, dentro del límite de páginas de la
convocatoria, con versión doble ciego para revisión.

</td>
<td width="33%" valign="top">

**Restricción**

Cifras reales o nada. Lo que no existe se declara pendiente. Las fechas de la
convocatoria 2027 se anotarán cuando se publiquen.

</td>
</tr>
</table>

<br>

## El proyecto en breve

AgroSatCopilot cuantifica superficie por tipo de cultivo a partir de series temporales
de Sentinel-2 y explica sus predicciones en lenguaje natural. La hipótesis central es
que la señal que distingue cultivos vive en su **evolución fenológica**, no en una
imagen estática. Los resultados la confirman: los encoders temporales dominan a los
baselines densos 2D.

<div align="center">
<img src="docs/presentation/assets/figs/seg_triptych.png" alt="Entrada RGB, verdad de campo y predicción de TSViT-pheno sobre un parche PASTIS-R" width="85%">
<br>
<sub>Entrada Sentinel-2, verdad de campo y predicción del mejor segmentador individual (TSViT-pheno) sobre un parche de PASTIS-R.</sub>
</div>

<br>

El copiloto sigue el patrón **Be My Eyes**. Los modelos del equipo *perciben* cada
parcela y emiten observaciones en texto. Un LLM congelado, Gemini 2.5 Pro en nube o
Qwen3.5-35B-A3B on-prem, *razona* sobre esas observaciones, invoca herramientas
geoespaciales y responde. El LLM nunca clasifica píxeles.

<details>
<summary><b>Stack técnico</b></summary>
<br>

| Capa | Tecnología |
|:--|:--|
| Percepción | TSViT, U-TAE, AnySat, DeepLabv3+, SegFormer, U-Net (PyTorch) |
| Embeddings | AlphaEarth Foundations `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1, CC-BY-4.0 |
| Visión-lenguaje | FarSLIP contrastivo-fenológico |
| Agente | Google ADK con 9 tools geoespaciales, patrón perceiver-reasoner |
| Backend / Frontend | FastAPI + Polars · Nuxt 4 SSR + MapLibre |
| Datos | PostgreSQL 15 + PostGIS + pgvector · DVC + MLflow + Dagster |
| Infraestructura | Terraform GCP · Azure H100 NVL 96 GB |

</details>

<br>

## Resultados canónicos

Regla heredada del proyecto y que se mantiene: cada número del manuscrito lleva un
comentario `% src:` que apunta a su artefacto. Estas son las cifras sobre las que
puede construirse el paper.

| Resultado | Métrica | Fuente |
|:--|:--|:--|
| Mejor individual · TSViT-pheno, PASTIS-R 18 clases | mIoU **0.6253** · F1-macro **0.7500** | `paper/sections/05_results.tex` |
| Modelo final · Stacking-5 heterogéneo, held-out fold-5 | F1-macro **0.7470** · acc 0.8490 | `reports/ensemble/metrics/comparison_us040.csv` |
| Ganancia del ensamble sobre el mejor individual | **+12.3 pp** F1-macro | `paper/sections/05_results.tex` |
| Modelo desplegado · Voting-3 v2, 12 clases | F1-macro **0.8992** · acc 0.9375 | `reports/agent_bench/perceiver_champion_eval_v2.json` |
| Recableo del perceiver al campeón, 14 688 parcelas | acc 0.8539 → **0.9375** | `reports/agent_bench/perceiver_champion_eval_v2.json` |
| Transferencia Francia → Cataluña | zero-shot 0.0 → few-shot mIoU **0.2468** | `reports/segmentation/sen4agrinet_transfer_result.json` |

<details>
<summary><b>Más cifras y lo que sigue pendiente</b></summary>
<br>

| Resultado | Métrica | Fuente |
|:--|:--|:--|
| Stacking-5 out-of-fold, libre de fuga | F1-macro 0.6477 · acc 0.7935 | `reports/ensemble/us043_farslip_summary.json` |
| Aporte de FarSLIP, 5 vs 3 miembros | +0.0118 F1-macro | `reports/ensemble/us043_farslip_summary.json` |
| Curva calidad-cobertura, 18 → 9 clases | F1-macro 0.7486 → 0.9121 con ~82 % de parcelas | `reports/ensemble/metrics/ec_neighborhood_result.json` |
| Vecindad espacial k-NN sobre el campeón | Δ +0.0002, no material | `reports/ensemble/metrics/ec_neighborhood_result.json` |
| Transferencia Francia → Alemania DE4, Voting-3 | F1-macro 0.119 → 0.266 | `paper/sections/experiments_de4.tex` |
| Transferencia Italia, dataset completo | Pendiente: corrida bloqueada en la VM H100 | `docs/us-resolved/us-082.md` |
| Benchmark LLM multi-modelo | Pendiente: tabla con placeholders honestos | `docs/blockers/PENDIENTES.md` |

</details>

<br>

## Hoja de ruta

- [x] **Contexto.** Fork, clon local, lectura del repositorio y de sus reglas. *2 sep 2026*
- [ ] **Ángulo.** Decidir la contribución única del paper y su tesis en una frase.
- [ ] **Evidencia.** Auditar cada cifra contra `reports/` y listar qué es reproducible sin la VM H100.
- [ ] **Manuscrito LNCS.** Migrar a `llncs.cls`, recortar al límite de páginas, apéndice antes de referencias.
- [ ] **Doble ciego.** Versión anónima sin nombres, afiliaciones ni enlaces identificables.
- [ ] **Envío.** Registro en el sistema de MICAI 2027 con el PDF anónimo. Fechas por confirmar.
- [ ] **Camera-ready.** Fuentes LaTeX, figuras vectoriales, licencia Springer firmada a mano. Condicional a aceptación.

> MICAI 2026 recomendó 12 páginas y admitió hasta 20. El límite de 2027 se confirma en
> la página oficial de autores cuando se publique.

<br>

## Material de partida

| | |
|:--|:--|
| [`paper/`](paper/) | Manuscrito modular en inglés con espejo en español, figuras y tablas regenerables desde `reports/`. Formato actual `PRIMEarxiv.sty`. |
| [`docs/paper/papers-adicionales-propuesta.md`](docs/paper/papers-adicionales-propuesta.md) | Tres ángulos de paper propuestos por el equipo original, con evidencia disponible y lo que falta. Punto de partida para la fase 1. |
| [`docs/blockers/PENDIENTES.md`](docs/blockers/PENDIENTES.md) | Índice único de lo que quedó abierto. Nada de esto se reclama sin cerrarlo. |
| [`reports/`](reports/) | Métricas JSON y CSV y figuras de cada experimento. Fuente de verdad numérica. |
| [`docs/final_doc/`](docs/final_doc/) · [`docs/presentation/`](docs/presentation/) | Entregable final del curso en LaTeX y presentación de defensa de 65 láminas, ambos en ES y EN. |
| [`docs/README-upstream-agrosat-copilot.md`](docs/README-upstream-agrosat-copilot.md) | README original del proyecto, conservado íntegro. |

<br>

## Trabajar desde este clon

Lo que se puede hacer sin credenciales del equipo original: compilar el manuscrito,
regenerar tablas y figuras desde `reports/`, editar secciones y auditar citas con
`scripts/paper_cite_check.py`.

Lo que no: descargar datos y pesos (el remoto DVC `gs://agrosat-dvc-remote` es
privado), reentrenar modelos (viven en la VM H100 del sponsor) y correr la evaluación
LLM pendiente.

```bash
git clone https://github.com/jrebull/agrosat-copilotv2.git
cd agrosat-copilotv2/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Para el stack completo con Docker, Poetry y pnpm, sigue el
[README original](docs/README-upstream-agrosat-copilot.md). El setup verificado en
macOS Apple Silicon, con sus parches, está en el
[Apéndice D del runbook](docs/runbook-local-setup.md#apéndice-d--macos-apple-silicon-verificado-2-sep-2026-m3-pro-macos-26).

<br>

## Reglas de este fork

- **Autoría.** Todo commit y push se firma únicamente como `jrebull`. Sin trailers
  `Co-Authored-By` de asistentes de IA ni pies "generado con" en PRs.
- **Integridad.** Ninguna cifra sin artefacto real. No se concluye más de lo que miden
  los datos.
- **Idioma.** Código y docstrings en inglés, documentación en español neutro, manuscrito
  en inglés. Sin emojis en código, commits ni documentación.
- **Atribuciones.** AlphaEarth (Khanna et al., CC-BY-4.0), FarSLIP (arXiv:2511.14901),
  Be My Eyes (arXiv:2511.19417), PASTIS-R (Garnot et al., ICCV 2021), Sen4AgriNet y
  EuroCropsML (CC-BY-SA-4.0), TSViT, U-TAE, SegFormer, AnySat.
- Las guías operativas del proyecto original (`CLAUDE.md`, `AGENTS.md` y las de cada
  carpeta) siguen vigentes para el código.

<br>

---

<div align="center">
<sub>

AgroSatCopilot fue desarrollado por **Carlos Isaac Ávila Gutiérrez**, **Carlos Aaron Bocanegra Buitrón**
y **Arthur Jafed Zizumbo Velasco** (Equipo 17, Maestría en Inteligencia Artificial Aplicada, Tec de Monterrey),
con el **Dr. Gerardo Jesús Camacho González** como sponsor académico.
Este fork conserva esa autoría y construye sobre ella. La autoría del artículo para MICAI se acordará con ellos antes del envío.

Código MIT · Datasets y modelos bajo sus licencias respectivas, ver [`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md)

</sub>
</div>
