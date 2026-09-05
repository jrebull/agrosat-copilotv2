---
name: agrosat-paper-micai
description: Manuscrito MICAI 2027 en paper/micai/ — plantilla Springer LNCS (llncs), doble ciego con \ifanon, bibliografia generada desde la matriz verificada (make micai-bib), cifras solo desde filas SELLADO del ledger con comentario % src:, figuras vectoriales reproducibles byte a byte, gates make micai-pdf / micai-anon-check / paper-cite-check / paper-obsoletos-check, empaquetado .zip y camera-ready. Use al escribir, revisar o compilar cualquier seccion, figura, tabla, cita o metadato del articulo, y al preparar el envio.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Manuscrito MICAI 2027 — `paper/micai/`

## Que hay y que es cada cosa

| Ruta | Estado | Regla |
|---|---|---|
| `paper/micai/main.tex`, `sections/`, `refs.bib`, `figures/` | Borrador retirado de 15 paginas (EPIC 16) con `ESTADO.md` delante: **no se envia**, pero su maquinaria (plantilla, gate de identidad, compilacion reproducible, bib generado) es la vigente y se reutiliza | Las secciones se reescriben desde el preregistro firmado (US-144); ninguna cifra vieja sobrevive sin fila `SELLADO` vigente |
| `paper/main.tex`, `sections/`, `arxiv/` | Manuscrito heredado de 24 paginas (E11) → informe tecnico interno (ADR-014 §5) | No se repara, no se publica, no se cita como fuente de cifras |
| `paper/ARTIFACTS.md` | Ledger de custodia | Solo lectura desde esta skill; sellar es `make paper-artifacts-seal` (mlops / humano) |

## Reglas de contenido

- **Ninguna cifra sin fila `SELLADO`** y sin `% src: <ruta>` en el `.tex`. `OBSOLETO` no entra ni
  con cuarentena. Si el valor no existe sellado, se sella primero (reportar), no se teclea.
- **Regimen en la frase** de toda comparacion (fold-5 held-out por parcela · in-sample para el
  meta-modelo · OOF · pixel).
- **Afirmaciones prohibidas** (ADR-013, ADR-014): transporte, "el ensamble mejora", "AlphaEarth
  codifica fenologia", "v2.1", "0,7486 held-out", "FarSLIP aporta senal", ganador entre
  predictores, "retirada por poca muestra" como premisa.
- **Resultados negativos** con su matiz, en resultados; "future work" solo para lo excluido por diseno.
- Doce a veinte paginas bajo `llncs`; si la convocatoria baja el tope, el recorte va de
  resultados a apendice, **nunca de robustez**.
- Atribuciones obligatorias: AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0; PASTIS-R
  (Garnot et al., ICCV 2021); BreizhCrops; Sen4AgriNet y EuroCropsML CC-BY-SA-4.0 (no "permisivos");
  Gemini 2.5 Pro como reasoner (nunca 3.5 Flash); Qwen3-30B-A3B servido con vLLM.

## Reglas de forma (LNCS y estilo)

- `\documentclass[runningheads]{llncs}`; A4 real (`\pdfpagewidth=210mm \pdfpageheight=297mm`);
  `cmap` antes de `fontenc`; `xurl`; `hyperref` con `hidelinks`; `\emergencystretch=1em`;
  `\keywords` dentro del abstract; apendice antes de las referencias; flotantes `[tbp]`.
- Ingles americano consistente; abstract de 150 a 250 palabras **sin cifras ni siglas de
  metrica**; siglas expandidas en su primer uso (el abstract es ambito propio); em-dash para
  incisos, en-dash para rangos; Oxford comma; `\paragraph` en lugar de subsubsecciones.
- Trabajo relacionado **por limitaciones** (cada cita con su fortaleza y su limite; el precedente
  del recorte de leyenda en la primera pagina, US-141), no como lista de usos.
- Autor de correspondencia con `\thanks{Corresponding author.}`, nunca `\Envelope`; solo en
  camera-ready. Creditos a Isaac Avila y Aaron Bocanegra como autores del codigo y "Disclosure of
  Interests" solo en camera-ready.
- Doble ciego: `\newif\ifanon` con la anonima por defecto; cero nombres, correos, matriculas,
  "Team 17", sponsor, jerga interna ni nombre del sistema indexado. Declaracion de disponibilidad
  sin URL en la version de envio.

## Bibliografia

- `paper/micai/refs.bib` se **genera** con `make micai-bib` desde
  `reports/paper_micai/fase0/related_work_verified.csv` (titulo, autores y ano resueltos por
  arXiv, Crossref u OpenAlex). Una entrada nueva entra por la matriz (con identificador resuelto
  por API: `scripts/paper_micai_ref_verify.py`), jamas a mano.
- Sin campos `note`; siglas protegidas con llaves; mas de seis autores se listan seis y "et al.";
  DOI de actas cuando exista (US-142).
- `make paper-cite-check`: cada `\cite{}` con entrada; cero `Citation undefined`.

## Figuras y tablas

- Figuras desde `scripts/build_paper_micai_*_figure.py`: vectoriales, legibles en blanco y negro
  (el color nunca es el unico canal), texto en ingles, `svg.hashsalt` fijo y
  `metadata={"CreationDate": None}` para reproducibilidad byte a byte (US-143). Nunca un PNG
  editado a mano.
- La figura central es el diagrama de fases en espacio de costes (US-152); la curva completa con
  el punto preregistrado marcado (US-169), no un solo punto de operacion.
- Tablas data-driven desde artefactos sellados; cero literales numericos en el `.tex` que no
  lleven `% src:`.

## Gates y comandos

```bash
make micai-pdf            # anonimo: pdflatex -> bibtex -> pdflatex x2, cero errores, cero overfull
make micai-anon-check     # busca 16 tokens de identidad en texto y metadatos; autoprueba en negativo
make micai-bib            # regenera refs.bib desde la matriz verificada
make micai-pdf-cr         # camera-ready; comprueba que SI revela identidad (el anonimo, no)
make paper-cite-check     # \cite{} <-> refs.bib
make paper-obsoletos-check
make paper-artifacts-check
```

El PDF de envio es reproducible byte a byte (`\pdfinfoomitdate`, `\pdftrailerid{}`,
`\pdfsuppressptexinfo`); el cuerpo del camera-ready es identico al anonimo. El empaquetado `.zip`
con el proyecto LaTeX completo y la verificacion de overfull sobre el PDF ensamblado son US-145.

## Salida esperada

Al escribir: la seccion o figura, la lista de cifras usadas con su fila del ledger, las entradas
nuevas que la matriz necesita, y el resultado de los gates con paginas. Al auditar: tabla por
cifra (fila del ledger, regimen, `% src:`), tokens de identidad, citas sin resolver, afirmaciones
prohibidas detectadas.
