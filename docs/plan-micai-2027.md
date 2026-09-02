# Plan MICAI 2027 — AgroSatCopilot v2

**Autores del artículo:** Arthur Jafed Zizumbo Velasco (primero) y Javier A. Rebull-Saucedo (segundo).
**Encuadre aceptado (2 de septiembre de 2026):** se escribe un artículo **desde cero** sobre el punto de operación, con el resultado negativo del ensamble dentro como sección de protocolo. Justificación en [`docs/paper/que-paper-sale.md`](paper/que-paper-sale.md); enmienda en [`ADR-013`](decisions/ADR-013-angulo-micai.md).
**Estado de partida:** 2 de septiembre de 2026, tres auditorías independientes del manuscrito heredado (contenido científico, reproducibilidad y formato LNCS). Resumen de hallazgos en la sección 2. El manuscrito heredado **no se repara**: se conserva como fuente de material, no como base.
**Pendientes que dependen de Arthur:** [`pendientes-arthur.md`](pendientes-arthur.md).
**Regla de oro heredada y vigente:** cifras reales o nada; cada número impreso se rederiva desde un artefacto versionado; no se concluye más de lo que mide el dato.
**Fases 0, 1 y 2 cerradas (2 de septiembre de 2026).** La 0 dio veredicto de reencuadre ([`novedad.md`](paper/novedad.md), 43 trabajos verificados por API). La 1 dejó `paper/ARTIFACTS.md` con 49 artefactos sellados por MD5, 8 cifras declaradas sin artefacto y el gate `make paper-artifacts-check` probado en negativo. La 2 midió los cinco experimentos con intervalos pareados ([`fase2-hallazgos.md`](paper/fase2-hallazgos.md)) y encontró que la cifra campeona 0,7486 es **in-sample** para el meta-modelo: libre de fuga ninguna combinación mejora al mejor miembro individual (0,7367), pero la contribución central **sí se sostiene**.
**Infraestructura lista (2 de septiembre de 2026):** entorno completo en macOS (Apéndice D del runbook), acceso al proyecto GCP `agrosat-copilot` con ADC, artefactos ligeros de DVC en disco, PASTIS-R crudo extraído en `data/PASTIS-R/` (2 433 parches, 68 GB), `make lint` y `make test-all` en verde, mypy en cero. La fase 0 puede empezar sin esperar nada de terceros.

Las fechas de la convocatoria MICAI 2027 no están publicadas. Este plan se ordena por dependencias, no por calendario; cuando salga la convocatoria se fija el calendario hacia atrás desde la fecha de envío MICAI 2027 sigue sin sede ni convocatoria: es la edición conmemorativa de los cuarenta años de la SMIA y todavía buscan institución anfitriona. Referencia verificada el 2 de septiembre de 2026 en la página de autores de MICAI 2026: doce páginas recomendadas, veinte de techo, doble ciego y envío en `.zip` con el proyecto LaTeX completo. **Se vuelve a verificar cuando salga la convocatoria de 2027.**

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

Las fases 0, 1 y 2 están cerradas y su detalle vive en sus documentos. Lo que sigue es el
plan del artículo nuevo, ordenado por dependencias.

### Fase 0 · ¿Aporta algo? — CERRADA

Veredicto: reencuadre. Búsqueda sistemática en tres fuentes, matriz de 43 trabajos con
identificador resuelto por API, y las tres referencias ancla leídas y contrastadas.
Detalle en [`docs/paper/novedad.md`](paper/novedad.md).

### Fase 1 · Sellado de artefactos — CERRADA en lo que no depende de Arthur

`paper/ARTIFACTS.md` con 49 artefactos sellados, 8 filas `SIN_ARTEFACTO` y el gate
`make paper-artifacts-check` probado en negativo y contra un clon limpio. Sellado además
el ground truth del fold 5 (16 640 etiquetas, 420 KB) para reproducir la evaluación sin
los 68 GB de PASTIS-R. Sigue pendiente lo de Arthur: el `dvc push` de
`alphaearth_italia_2018.parquet` y el informe de DE4.

### Fase 2 · Experimentos en CPU — CERRADA

Los cinco puntos hechos, cada uno con artefacto sellado, semilla, versiones, commit y
prueba pareada con intervalo. Detalle en
[`docs/paper/fase2-hallazgos.md`](paper/fase2-hallazgos.md).

### Fase 3 · Robustez del resultado sobre PASTIS-R (CPU, sin terceros)

Cierra las objeciones baratas antes de tocar datos nuevos. Salida:
`reports/paper_micai/fase3/`.

1. **Segundo predictor.** Repetir la comparación de cobertura sobre `xgb-alphaearth`
   (F1-macro 0,5913). Si el resultado se mantiene, la conclusión es una propiedad del
   desbalance y no de un modelo concreto. Es la respuesta más barata a la objeción de
   validez externa.
2. **Tercer mecanismo.** Retirar clases **por soporte** en lugar de por F1, que es lo que
   se hace en la práctica y lo primero que va a preguntar un revisor.
3. **Multiplicidad.** Corregir los siete contrastes de K (Holm o Benjamini-Hochberg) y
   declarar por escrito qué es confirmatorio y qué exploratorio, decidido **antes** de
   mirar los resultados nuevos.
4. **Curva completa de rechazo por confianza**, con más puntos que los siete igualados,
   para dibujar la frontera y no solo sus intersecciones.
5. **Figura de la frontera** en vectorial, legible en blanco y negro.

### Fase 4 · Segundo conjunto de datos: BreizhCrops (CPU, sin terceros)

Es la respuesta real a «un solo dataset», y no necesita GPU. Salida:
`reports/paper_micai/fase4/`.

1. `dvc pull data/breizhcrops` (1,66 GB, seis archivos). El cargador ya existe:
   [`ml/ingest/breizhcrops_loader.py`](../ml/ingest/breizhcrops_loader.py) y
   [`ml/features/breizhcrops_features.py`](../ml/features/breizhcrops_features.py).
2. Entrenar un clasificador por parcela en CPU sobre otra región de Francia y otro año
   (2017), con su propio desbalance y su propia leyenda.
3. Sellar su ground truth, sus posteriores y su soporte por clase como se hizo con el
   fold 5 de PASTIS-R.
4. Repetir **exactamente** el protocolo de la fase 2: bloques espaciales, los tres
   mecanismos, intervalos pareados. Sin tocar una línea del método.
5. Reportar si la conclusión se transporta o no. **Si no se transporta, se reporta igual**
   y el artículo gana un matiz en lugar de perder una tesis.

### Fase 5 · Reentrenamiento OOF de cinco folds (GPU, depende de la ventana H100)

No bloquea la escritura. Es lo que convierte un artículo correcto en uno sólido.

1. Entrenar el miembro elegido dejando cada fold fuera, para tener posteriores OOF sobre
   los **2 433 parches** en lugar de 496. El universo pasa de 16 640 parcelas a unas 83 000.
2. Con eso el meta-modelo por fin se entrena como es debido y el resultado negativo del
   ensamble deja de estar limitado por los artefactos.
3. **Beneficio lateral que vale por sí solo**: resuelve para siempre la duda sobre la
   procedencia de `tsvit-pheno`, porque el entrenamiento sería nuestro y con folds conocidos.
4. Si la ventana no se abre, el artículo se envía con el universo de 16 640 y la limitación
   declarada. No es un bloqueante.

### Fase 6 · Escritura desde cero, doce páginas LNCS

Redacta Javier, revisa Arthur. Título de trabajo, contribuciones y reparto de páginas en
[`docs/paper/que-paper-sale.md`](paper/que-paper-sale.md), sección 5.

- `paper/micai/` nuevo, no `paper/`. El manuscrito heredado se queda donde está como
  material de consulta.
- Ninguna cifra entra sin fila sellada en `paper/ARTIFACTS.md`.
- Reglas de estilo: inglés americano consistente, abstract de 150 a 250 palabras sin
  cifras ni siglas de métrica, em-dash para incisos y en-dash para rangos, Oxford comma,
  siglas expandidas en su primer uso con el abstract como ámbito propio, `\paragraph` en
  lugar de subsubsecciones.
- Trabajo relacionado **por limitaciones**, no como lista: cada cita con su fortaleza y su
  límite. La matriz de la fase 0 ya está redactada así.

### Fase 7 · Bibliografía

Parte de `reports/paper_micai/fase0/related_work_verified.csv`: 43 entradas con título,
autores y año resueltos por arXiv, Crossref u OpenAlex.

1. Completar el DOI de actas de las entradas que hoy solo tienen identificador arXiv.
2. Reconstruir desde la fuente cualquier entrada heredada que se reutilice; las tres
   anclas ya están verificadas y leídas.
3. Gate `make paper-cite-check` más verificación por API de cada identificador.
4. Sin campos `note`, siglas protegidas con llaves, más de seis autores se listan seis y
   «et al.».

### Fase 8 · Plantilla LNCS, doble ciego y paquete

1. `\documentclass[runningheads]{llncs}`, `\AtBeginDocument{\pdfpagewidth=210mm
   \pdfpageheight=297mm}` para el A4 real, `cmap` antes de `fontenc`, `xurl`, `hyperref`
   con `hidelinks`, `\emergencystretch=1em`, `\keywords` dentro del abstract, apéndice
   antes de las referencias, flotantes `[tbp]`.
2. Autor de correspondencia con `\thanks{Corresponding author.}`, nunca `\Envelope`.
3. Un `main.tex` con `\newif\ifanon` y dos salidas byte-idénticas en el cuerpo.
4. Gate de identidad: `pdftotext` del PDF de envío con cero coincidencias de la lista de
   tokens, **probado en negativo** antes de confiar en él.
5. Empaquetado en `.zip` con el proyecto LaTeX completo, como pide MICAI, y verificación
   de cero overfull mayores de 5 pt sobre el PDF ensamblado, no sobre el `.tex`.

### Fase 9 · Sitio live en Netlify — OPCIONAL

Sin cambios respecto al plan anterior, y sin prioridad hasta que haya manuscrito. Depende
de que Arthur cree el sitio y cargue los secretos.

### Fase 10 · Metadatos y entrega

1. Autor de correspondencia decidido antes del envío: firma la licencia **a mano** y
   Springer no permite cambiarlo después del camera-ready.
2. ORCID de Arthur; nombres idénticos carácter a carácter en artículo, sistema del
   congreso, `LICENSE`, `pyproject.toml` y README.
3. `credits` con agradecimientos a Isaac Ávila y Aaron Bocanegra como autores del código,
   y «Disclosure of Interests»; solo en camera-ready.
4. Repositorio citable: `LICENSE` que GitHub reconozca, `CITATION.cff`, release con DOI de
   Zenodo, y declaración de disponibilidad sin URL en la versión de envío.
5. Seguimiento de la convocatoria de MICAI 2027 y calendario hacia atrás con una semana de
   holgura.

---

## 4. Orden y esfuerzo

| Fase | Esfuerzo | Depende de | Bloquea el envío |
|---|---|---|---|
| 3 Robustez en CPU | 1 día | nada | sí, dos de sus cinco puntos |
| 4 BreizhCrops | 2 a 3 días | `dvc pull` | no, pero es la mejor defensa |
| 5 Reentrenamiento OOF | 1 día de cómputo | ventana H100, Arthur | no |
| 6 Escritura | 4 a 6 días | 3, y 4 si llega a tiempo | sí |
| 7 Bibliografía | 1 día | nada | sí |
| 8 LNCS y doble ciego | 2 días | 6 | sí |
| 9 Sitio live | 2 días | Arthur | no |
| 10 Metadatos y entrega | 1 día más trámites | 8, convocatoria | sí |

Las fases 3, 4 y 7 no dependen de nadie y pueden empezar hoy. La 5 entra cuando se abra la
ventana; si no se abre, se declara la limitación y se envía igual.

---

## 5. Criterios de cierre por fase

- Fase 0: **cumplida salvo la firma** de `ADR-013`.
- Fase 1: `make paper-artifacts-check` en verde y probado en negativo. **Cumplida** en lo
  que no depende de Arthur.
- Fase 2: cada experimento con artefacto sellado, semilla, versiones y prueba pareada con
  su intervalo. **Cumplida.**
- Fase 3: los cinco puntos con artefacto sellado; la decisión de confirmatorio frente a
  exploratorio escrita **antes** de mirar los resultados.
- Fase 4: el protocolo de la fase 2 corre sobre BreizhCrops sin modificarlo, y su
  resultado se reporta se transporte o no.
- Fase 5: posteriores OOF de los cinco folds sellados, con la procedencia del checkpoint
  documentada por nosotros.
- Fase 6: PDF de doce páginas bajo `llncs`, abstract de 150 a 250 palabras, cada cifra con
  fila en el ledger.
- Fase 7: cada entrada verificada por API, con DOI, sin `note`.
- Fase 8: dos PDF desde un master, cero tokens de identidad en el de envío con el gate
  probado en negativo, cero overfull mayores de 5 pt medidos sobre el PDF.
- Fase 9: cada push publica la iteración en menos de quince minutos y la prueba en negativo
  pone el semáforo en rojo.
- Fase 10: licencia lista para firmar, nombres unificados, repositorio citable con DOI.
