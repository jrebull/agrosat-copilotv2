# Plan MICAI 2027 — AgroSatCopilot v2

**Autores del artículo:** Arthur Jafed Zizumbo Velasco (primero) y Javier A. Rebull-Saucedo (segundo).
**Estado de partida:** 2 de septiembre de 2026, tres auditorías independientes del manuscrito heredado (contenido científico, reproducibilidad y formato LNCS). Resumen de hallazgos en la sección 2.
**Pendientes que dependen de Arthur:** [`pendientes-arthur.md`](pendientes-arthur.md).
**Regla de oro heredada y vigente:** cifras reales o nada; cada número impreso se rederiva desde un artefacto versionado; no se concluye más de lo que mide el dato.
**Infraestructura lista (2 de septiembre de 2026):** entorno completo en macOS (Apéndice D del runbook), acceso al proyecto GCP `agrosat-copilot` con ADC, artefactos ligeros de DVC en disco, PASTIS-R crudo extraído en `data/PASTIS-R/` (2 433 parches, 68 GB), `make lint` y `make test-all` en verde, mypy en cero. La fase 0 puede empezar sin esperar nada de terceros.

Las fechas de la convocatoria MICAI 2027 no están publicadas. Este plan se ordena por dependencias, no por calendario; cuando salga la convocatoria se fija el calendario hacia atrás desde la fecha de envío (MICAI 2026 recomendó 12 páginas y admitió 20; confirmar para 2027).

---

## 1. Principios de trabajo

1. **Un solo régimen de evaluación por comparación**, nombrado en la frase: fold-5 held-out a nivel parcela, out-of-fold, o píxel. Nunca se cruzan sin decirlo.
2. **Custodia sellada**: las cifras salen de un paquete con hash (`paper/ARTIFACTS.md` con MD5 de cada CSV, JSON y parquet), nunca de rutas vivas ni de markdowns.
3. **Resultados negativos se reportan** con su matiz; a "future work" solo va lo que quedó fuera de alcance por diseño.
4. **Doble ciego desde el primer borrador**: un `main.tex` con toggle, cero tokens de identidad en el PDF de envío.
5. **Verificación en negativo de cada control** antes de confiar en él (un gate que nunca ha fallado no se sabe si funciona).
6. **Commits como jrebull**, sin coautoría de asistentes; cada fase cierra con commit y `make lint` en verde.

---

## 2. Diagnóstico consolidado (auditorías del 2 de septiembre)

| Ámbito | Hallazgo | Severidad |
|---|---|---|
| Extensión | 36 páginas bajo `llncs`; secciones 1 a 5 ocupan 11, FarSLIP, multi-región y DE4 suman 17 | Bloqueante |
| Cifra estrella | 0.7470 es el Stacking-3 sin FarSLIP; el Stacking-5 da 0.7486 (`us043_farslip_grid.csv`) | Bloqueante |
| Régimen | "held-out 0.7470 vs OOF 0.6477" es un cambio de miembro TSViT, no de régimen; ambos son fold-5 held-out | Bloqueante |
| Signos y lecturas | Blending +0.0215 y no −0.0117; delta FarSLIP vs AlphaEarth −0.0896 F1 (−0.0024 es silhouette); la rama fenológica resta 0.33 mIoU, no suma | Bloqueante |
| Cifras sin artefacto | Bretaña 0.21, AlphaEarth vs S2 +0.111, WorldCereal few-shot, métricas de tools, toda la sección DE4, conteos por escena, ablación de bandas | Bloqueante |
| Referencias | Be My Eyes, Harvesting AlphaEarth y AgroMind con título y autores inventados; coautores falsos en EuroCropsML; U-TAE cita al paper de L-TAE; Wen 2025 no localizable | Bloqueante |
| Modelos | Gemini 3.5 Flash en 9 sitios citando el informe de 2.5; código y A/B usan 2.5 Pro; Qwen descrito con vLLM cuando se sirvió con llama.cpp; "Qwen 3.6-VL" sin cita | Bloqueante |
| Pendientes en el texto | Cuatro secciones y una tabla dicen "still pending" | Bloqueante |
| Doble ciego | Nombres, correos, matrículas, "Team 17" en cabecera, sponsor, 35 menciones de jerga interna, nombre del sistema indexado en GitHub | Bloqueante |
| Reproducibilidad del PDF | 11 PNG ignorados por git; 6 dependen de fuentes perdidas; un clon limpio compila con 6 cajas vacías; AOI consulta GEE en vivo | Bloqueante |
| Licencias | Texto dice Apache 2.0, repo MIT; datasets CC-BY-SA llamados "permisivos"; sin URL ni DOI; GitHub no detecta la licencia | Bloqueante |
| Tablas | Individuales con folds mezclados, sin U-TAE ni AnySat fold-5; Sen4AgriNet con tres corridas distintas entre tabla, texto y figura | Importante |
| Estadística | Ninguna prueba ni intervalo; bootstrap pareado y McNemar son posibles en CPU con 16 640 parcelas | Importante |
| Related Work | Lista de usos sin limitaciones; faltan TempCNN, Rußwurm y Körner, ReAct, RAG, Wolpert, Roberts, clasificación selectiva | Importante |
| Estilo | Abstract 272 palabras con 8 citas; inglés mixto; 20 siglas sin expandir; bib sin DOI en 21 de 22 | Importante |
| Metadatos | Sin autor de correspondencia, ORCID solo de uno, nombres distintos entre archivos | Importante |

Las cifras que se rederivan hoy desde disco: 61 %. Las que contradicen el artefacto: 8 %. Las que solo existen en markdown o PNG: 14 %. Las que dependen de archivos perdidos: 17 %.

---

## 3. Fases

### Fase 0 · Aporte real: ¿el paper contribuye algo?

Antes de reescribir una línea. Responsable: Javier. Salida: `docs/paper/novedad.md`.

1. Formular la tesis candidata en una frase. Candidata principal (ángulo A de la propuesta del equipo): *en mapeo de cultivos desbalanceado, un árbitro heterogéneo por clase mejora sobre el promedio homogéneo, cuántas clases prometer es un punto de operación medible, y el contexto espacial no aporta sobre embeddings de fundación anuales.*
2. Búsqueda sistemática (arXiv, Semantic Scholar, Google Scholar; 2019 a 2026) en cuatro frentes: stacking heterogéneo y arbitraje por clase en series temporales de satélite; selección de número de clases y clasificación selectiva en cobertura del suelo; contexto espacial sobre embeddings de modelos de fundación EO; copilotos LLM anclados para observación de la Tierra. Registrar consulta, fecha y resultados.
3. Matriz de trabajo relacionado: para cada paper, método, fortaleza, límite y qué hueco deja. Mínimo 25 entradas, con DOI o id arXiv verificado por API.
4. Verificar las tres referencias ancla con título real y leerlas: comprobar que dicen lo que el manuscrito les atribuye.
5. Veredicto explícito con tres salidas posibles: contribución confirmada (seguir con A), reencuadre (qué cambia), o abandono del ángulo. Criterio: existe al menos un experimento ejecutable en CPU que aísla el mecanismo y ningún trabajo previo lo reporta en este dominio con este protocolo.
6. Decisión conjunta con Arthur registrada en `docs/decisions/ADR-013-angulo-micai.md`.

### Fase 1 · Sellado de artefactos

Responsable: Javier, con entregas de Arthur. Salida: `paper/ARTIFACTS.md` y DVC actualizado.

1. `dvc add` y `dvc push` de todo lo que alimenta una cifra y hoy no está versionado: `worldcereal_fewshot_{results,india}.parquet`, `eurocropsml_alphaearth_vs_s2_delta.parquet`, `eurocropsml_per_class.parquet`, `sen4agrinet_per_class.parquet`, `pastis_to_breizhcrops.parquet`, `sen4agrinet_{es,fr}_alphaearth.parquet`, `us049_system_eval.json`, logs de la ablación de bandas. Los que no existan en ningún sitio se regeneran en CPU (WorldCereal, Bretaña) o se retiran del texto.
2. Recibir de Arthur los artefactos de DE4 e Italia y el checkpoint few-shot de Sen4AgriNet (ver pendientes) y versionarlos.
3. Una sola corrida sellada de Sen4AgriNet: parquet, JSON per-clase y checkpoint; regenerar texto y figura desde ella.
4. Emitir CSV con los conteos por escena que hoy solo viven en títulos de PNG.
5. Figuras: `make paper-figures` emite SVG y PDF versionados; retirar `*.png` del flujo; `aoi_italy` desde caché sellada, no GEE en vivo.
6. `paper/ARTIFACTS.md`: tabla elemento del paper, artefacto, MD5, commit, fecha. Gate: `scripts/paper_artifacts_check.py` compara los MD5 y falla si algo cambió sin registro.
7. Fijar versiones de cómputo en el ledger (`xgboost`, `scikit-learn`, `polars`, `matplotlib`) y el SHA del commit que produjo cada artefacto.

### Fase 2 · Experimentos del ángulo A (CPU, datos en disco)

Responsable: Javier. Insumos: `ml/eval/oof/oof_parcel_*_fold5.parquet` (8 miembros, 16 640 parcelas), `us043_farslip_grid.csv`, `us043_winner_cardinality_curve.csv`, `cardinalidad.json`, `ec_neighborhood_result.json`, geometrías de PASTIS-R.

1. Homogéneo contra heterogéneo sobre las mismas OOF: media simple, voto ponderado y stacking; delta por clase; bootstrap pareado por parcela (B = 1000) y McNemar.
2. Curva calidad contra cobertura con intervalo bootstrap y, como baseline de clasificación selectiva, rechazo por umbral de confianza a igual cobertura.
3. Aporte de FarSLIP 5 contra 3 miembros en un solo universo, con intervalo; declarar por qué se elige ese universo.
4. Nulo de vecindad espacial con intervalo (`ml/ensemble/ec_neighborhood.py`).
5. Tabla de individuales bajo un solo protocolo (fold-5, todos los miembros, mismo harness).
6. Cada experimento produce CSV o JSON bajo `reports/paper_micai/` con semilla, versiones y SHA; entra al ledger de la fase 1.

### Fase 3 · Corrección de cifras y afirmaciones

Responsable: Javier. Barrido completo tras cada cambio (abstract, introducción, resultados, discusión, limitaciones, conclusión, pies y tablas).

1. 0.7486 y 0.8495 para Stacking-5; 0.7470 solo como Stacking-3 si se conserva.
2. Una tabla única de ensambles con las cuatro celdas del grid y su régimen; explicar por qué el mejor miembro individual produce el peor stacking.
3. Blending +0.0215; delta FarSLIP −0.0896 F1; ganancia del ensamble +1.2 pp a nivel parcela.
4. Retirar Bretaña, AlphaEarth vs S2, WorldCereal, tools y por-escena hasta que existan artefactos sellados; DE4 solo si Arthur entrega `report.json` y datos.
5. Un solo reasoner en todo el texto y en la cita: Gemini 2.5 Pro (Vertex AI); repetir la medición de intercambio de backends con n mayor o igual a 30 consultas, o reducirla a una frase.
6. Serving de Qwen descrito como fue (llama.cpp, GGUF Q4_K_M) o eliminado; sin "Qwen 3.6-VL" sin cita; sin Gemma.
7. Eliminar toda mención de resultados pendientes, jerga interna (US-0xx, EPIC, H100, blockers, engram), rutas del repositorio y costos FinOps.
8. Limitaciones con lo que debilita las conclusiones: un solo fold held-out para los densos, un par denso con 10 parches, artefactos regenerados.

### Fase 4 · Reescritura en 12 a 15 páginas LNCS

Responsable: Javier redacta, Arthur revisa. Plan de secciones:

| Sección | Contenido | Páginas |
|---|---|---|
| 1 Introducción | Problema, desbalance, por qué prometer K clases es una decisión medible; contribuciones inline (i) (ii) (iii) con verbos honestos | 1.5 |
| 2 Trabajo relacionado | Por limitaciones: SITS y PASTIS, stacking, clasificación selectiva, modelos de fundación EO, validación espacial | 1.5 |
| 3 Materiales y método | Datos y protocolo libre de fuga (ranking en OOF, medida en held-out, un fold), miembros, árbitro y mecanismo de punto de operación | 3 |
| 4 Resultados | 4.1 heterogéneo contra homogéneo por clase; 4.2 curva calidad contra cobertura frente a rechazo por confianza; 4.3 aporte FarSLIP; 4.4 nulo de vecindad | 4 |
| 5 Discusión | Qué compra el arbitraje; por qué el contexto ya está en el embedding; confirmatorio contra exploratorio | 1.5 |
| 6 Limitaciones y conclusión | Solo lo que debilita las conclusiones | 1 |
| Referencias | Unas 30 entradas con DOI | 1.5 |

Fuera del cuerpo: capa conversacional (un párrafo de contexto, sin cifras), multi-región y DE4 (segundo artículo), narrativa del bug de FarSLIP (un párrafo), apéndices largos.

Reglas de estilo: abstract de 150 a 250 palabras sin cifras ni siglas de métrica; inglés americano consistente; em-dash para incisos y en-dash para rangos; Oxford comma; siglas expandidas en su primer uso y el abstract como ámbito propio; separador de miles con espacio fino; `\paragraph` en lugar de subsubsecciones.

### Fase 5 · Bibliografía

Responsable: Javier. Gate: `scripts/paper_cite_check.py` más verificación por API de cada id.

1. Reconstruir desde la fuente las seis entradas con datos falsos o dudosos: `huang2025bemyeyes`, `harvesting2026alphaearth`, `ruan2025agromind`, `reuss2025eurocropsml`, `li2025farslip`, `wen2025phenology`; corregir `garnot2021utae`.
2. DOI en las 22 entradas actuales y en las nuevas; retirar los campos `note`; proteger siglas con llaves; más de seis autores se listan seis y "et al.".
3. Añadir las referencias que un revisor de IA espera: TempCNN (Pelletier 2019), Rußwurm y Körner 2020, ReAct, Toolformer, RAG (Lewis 2020), LLM-as-judge (Zheng 2023), stacking (Wolpert 1992), validación espacial (Roberts 2017), clasificación selectiva (Geifman y El-Yaniv 2017), EuroCrops y HCAT (Schneider 2023), Prithvi, SatMAE, Optuna.

### Fase 6 · Plantilla LNCS, figuras y doble ciego

Responsable: Javier. Prototipo ya compilado en scratch: `llncs` 2.26, A4 real, cero errores, cero referencias indefinidas.

1. Preámbulo: `\documentclass[runningheads]{llncs}`, `\AtBeginDocument{\pdfpagewidth=210mm \pdfpageheight=297mm}`, `cmap` antes de `fontenc`, `xurl`, `hyperref` con `hidelinks`, `\UrlFont` en negro, `\emergencystretch=1em`, `\keywords` dentro del abstract, apéndice antes de referencias, flotantes `[tbp]`, `splncs04`.
2. Figuras vectoriales PDF desde SVG en el build; texto legible en blanco y negro; medir tamaño de glifo sobre el PDF ensamblado.
3. Retirar `PRIMEarxiv.sty`, `fancyhdr`, `import`, `\And`, `\thanks` de cita sugerida.
4. Un `main.tex` con `\newif\ifanon`; `\sysname` que en envío es "the copilot"; bloques `% >>> REVIEW:` y `% >>> CAMERA-READY:` para autores, `\institute`, `\orcidID`, `\email`, `credits` y URL de código. Generador de dos PDF (`make paper-submission`, `make paper-camera-ready`) con cuerpo byte-idéntico.
5. Gate de identidad: `pdftotext` del PDF de envío con cero coincidencias de la lista de tokens (nombres, correos, instituciones, ORCID, Team 17, AgroSatCopilot, organizaciones de GitHub, US-0xx, H100, rutas).
6. Empaquetador que excluya `main_es.tex`, `sections_es/`, `AGENTS.md`, `CLAUDE.md`, notebooks, HTML, tablas no usadas y comentarios `% src:` del paquete de envío, y que verifique cero overfull mayores de 5 pt.

### Fase 7 · Sitio live del paper en Netlify

Responsable: Javier construye, Arthur revisa. Objetivo: que cada iteración del manuscrito se vea en vivo, sin abrir LaTeX, con su estado de auditoría.

**Qué muestra el sitio**

- Portada: versión actual (commit corto, fecha, páginas, tamaño A4 verificado), botón al PDF de envío y al de camera-ready, y semáforo de gates (compila, cero referencias indefinidas, cero tokens de identidad, ledger de artefactos íntegro, páginas dentro del límite).
- Lector: el PDF más reciente embebido (pdf.js) y una versión HTML del manuscrito (LaTeXML o pandoc) para lectura en móvil.
- Iteraciones: línea de tiempo de cada push a `main` que tocó `paper/`, con mensaje de commit, delta de páginas, palabras y cifras cambiadas (diff del ledger), y enlace al PDF de esa iteración.
- Ledger de cifras: cada número impreso con su artefacto, MD5 y estado (verificado, cambiado, sin fuente).
- Galería de figuras y tablas con su generador y fecha.
- Tablero de auditoría: los hallazgos de este plan como checklist con estado abierto, en curso, cerrado y commit que lo cierra.
- Comentarios: un enlace por sección a un issue de GitHub para que Arthur comente sin tocar LaTeX.

**Cómo se construye**

- Carpeta `site/` con generador estático propio (`scripts/build_paper_site.py`, Python del repo, plantillas Jinja2 ya disponibles en el grupo `paper`) que lee `paper/ARTIFACTS.md`, el log de git, el `.log` de LaTeX y los PDF. Sin framework de frontend: HTML, CSS y pdf.js desde CDN.
- Workflow `paper-live.yml` en GitHub Actions: en cada push a `main` que toque `paper/`, `scripts/` o `reports/`: compila con texlive en contenedor (`make paper-pdf-docker`), corre los gates, construye `site/` y despliega con `netlify deploy --prod --dir site` usando `NETLIFY_AUTH_TOKEN` y `NETLIFY_SITE_ID` como secretos del repositorio. Cada iteración se guarda en `site/iteraciones/<sha>/`.
- Netlify: sitio creado por Arthur o Javier, dominio `agrosat-micai.netlify.app` o similar, con contraseña de Netlify o cabecera `noindex` porque el sitio muestra autores y por tanto no puede enlazarse desde el paper ni ser indexable durante la revisión.
- Prueba en negativo: un push con una referencia rota debe poner el semáforo en rojo y no publicar el PDF.

**Entregables**: `site/` generado, `scripts/build_paper_site.py`, `.github/workflows/paper-live.yml`, `docs/site-live.md` con la operación.

### Fase 8 · Metadatos y entrega

Responsables: ambos.

1. Autor de correspondencia decidido antes del envío (firma la licencia Springer a mano y no se puede cambiar después).
2. ORCID de Arthur; nombres idénticos carácter a carácter en paper, sistema del congreso, `LICENSE`, `pyproject.toml` y README; afiliación única "Tecnológico de Monterrey".
3. `credits` con agradecimientos al sponsor y a los desarrolladores originales (Isaac Ávila y Aaron Bocanegra como autores del código, no del artículo) y "Disclosure of Interests"; solo en camera-ready.
4. Repositorio citable: `LICENSE` en formato que GitHub reconozca, `CITATION.cff`, release con DOI de Zenodo, declaración de disponibilidad con licencias reales (MIT código, CC-BY-SA-4.0 derivados de datos), sin URL en la versión de envío.
5. Seguimiento de la convocatoria MICAI 2027: fechas, límite de páginas, formato de envío; calendario hacia atrás desde la fecha de envío con una semana de holgura.

---

## 4. Cronograma tentativo y esfuerzo

| Fase | Esfuerzo | Depende de |
|---|---|---|
| 0 Aporte real | 3 días | nada |
| 1 Sellado de artefactos | 1 día más entregas de Arthur | 0 |
| 2 Experimentos CPU | 2 días | 1 |
| 3 Corrección de cifras | 1 día | 2 |
| 4 Reescritura | 3 a 5 días | 0, 2, 3 |
| 5 Bibliografía | 1 día | 0 |
| 6 LNCS y doble ciego | 2 días | 4 |
| 7 Sitio live | 2 días | 6 (puede empezar en paralelo con el PDF actual) |
| 8 Metadatos y entrega | 1 día más trámites | 6, convocatoria |

Total orientativo: 16 a 18 días de trabajo, sin contar esperas de terceros.

---

## 5. Criterios de cierre por fase

- Fase 0: `novedad.md` con matriz de al menos 25 trabajos y veredicto firmado por ambos en un ADR.
- Fase 1: `make paper-artifacts-check` en verde; un clon limpio con `dvc pull` regenera todas las tablas y figuras.
- Fase 2: cada experimento con CSV o JSON sellado, semilla, versiones y prueba estadística con n.
- Fase 3: cero menciones de "pending", jerga interna o rutas en el texto; cada cifra con `% src:` verificable por script.
- Fase 4: PDF de 12 a 15 páginas bajo `llncs`; abstract de 150 a 250 palabras; `make paper-pdf` con código de salida cero.
- Fase 5: cada entrada del bib verificada por API; cero `note`; DOI en todas.
- Fase 6: dos PDF generados desde un master; cero tokens de identidad en el de envío; cero overfull mayores de 5 pt.
- Fase 7: cada push a `main` publica la iteración en Netlify en menos de 15 minutos; la prueba en negativo pone el semáforo en rojo.
- Fase 8: licencia lista para firmar, nombres unificados, repositorio citable con DOI.
