# Presentación AgroSatCopilot (Reveal.js, ES/EN)

Presentación de 78 láminas para la defensa final, bilingüe (español/inglés) con
**switch de idioma** en vivo, lista para **GitHub Pages**.

## Arquitectura (contenido separado del HTML)

El contenido NO está embebido en el HTML. Vive en archivos JSON, uno por idioma, y
un motor JavaScript construye las láminas. Así, **corregir un texto = editar una
línea de JSON**, sin pelear con el HTML.

```
docs/presentation/
  index.html          # shell minimo (cabecera, marca, switch; SIN laminas)
  content/
    es.json           # TODO el texto en espanol (una entrada por lamina)
    en.json           # TODO el texto en ingles (mismas laminas, mismo orden)
  js/render.js        # motor: lee el JSON activo y construye los <section>; switch ES/EN
  css/theme.css       # tema agro-satelital (paleta crema/verde/tierra)
  assets/figs/        # figuras reales + ilustraciones hero + logo del Tec (logo_tec.png)
  .nojekyll
```

### Como editar el contenido

- Cambiar un texto: abre `content/es.json` (o `en.json`), busca la lámina y edita el
  campo (`title`, `body`, `paras`, etc.). Recarga el navegador. Listo.
- Cada lámina es un objeto con un `layout` (`cover`, `divider`, `cards`, `kpi`,
  `table`, `fig`, `twocol`, `text`, `closing`) y los campos de ese layout. **El
  número de láminas y el orden deben coincidir entre `es.json` y `en.json`** (es lo
  que mantiene el switch alineado).
- Cambiar una figura: edita el campo `img` de la lámina (ruta dentro de `assets/figs/`).
- Agregar una lámina: añade el mismo objeto en `es.json` y en `en.json`, en la misma
  posición.

### Logo del Tecnológico de Monterrey

La marca institucional usa `assets/figs/logo_tec.png` si existe; si no, muestra un
wordmark tipográfico de respaldo. **Para poner el logo real, guarda el PNG como
`assets/figs/logo_tec.png`** (la presentación lo toma automáticamente).

## Ver localmente

```bash
cd docs/presentation
python -m http.server 8765
# abrir http://127.0.0.1:8765/index.html
```

- Switch ES/EN: botón flotante arriba a la derecha (o `?lang=es` / `?lang=en`).
- Navegación: flechas, `Esc` (vista general), `S` (notas), `F` (pantalla completa).

## Desplegar en GitHub Pages

Sitio estático (HTML + CSS + JS + JSON + PNG, Reveal.js por CDN). En **Settings →
Pages**, elige *Deploy from a branch* y la carpeta `/docs`. El `.nojekyll` evita que
Pages procese el sitio con Jekyll.

## Contenido (6 secciones, con láminas divisorias ilustradas)

1. **Negocio** — el problema, el impacto económico, la métrica correcta.
2. **Datos y exploración** — la señal que separa cultivos vive en el tiempo; AlphaEarth, PASTIS.
3. **Modelado** — de modelos simples a seis arquitecturas y cuatro formas de combinarlas.
4. **El modelo ganador** — Voting-3 con doce cultivos bien resueltos (puntaje 0.86, 90% de cobertura), que mejora con cada nueva región vía aprendizaje por transferencia.
5. **El copiloto y la aplicación** — los modelos ven; un modelo de lenguaje razona y responde.
6. **Llevarlo a nuevas regiones** — aprendizaje por transferencia, buenas prácticas y lo aprendido.

Todas las cifras provienen de artefactos reales del proyecto (cero placeholders).
Lenguaje llano, con las siglas técnicas explicadas la primera vez.
