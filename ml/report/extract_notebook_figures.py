"""Extract inline PNG figures from a Jupyter notebook to a target directory.

Walks the outputs of each cell looking for ``image/png`` (base64) and exports
them with a stable name ``cell_{idx:03d}_{output_idx}.png``. Used to feed the
Avance 1 PDF report from notebooks that generate their figures inline
(without explicit ``plt.savefig``), like ``Avance1.Equipo17.ipynb``.

Usage::

    python -m ml.report.extract_notebook_figures \\
        notebooks/eda/Avance1.Equipo17.ipynb \\
        --output paper/figures/avance1/
"""

from __future__ import annotations

import base64
import json
import re
import sys
import unicodedata
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, help="Extrae figuras PNG inline de notebooks Jupyter.")


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a markdown heading to an ASCII slug safe for filenames.

    Args:
        text: Source text (may contain accents, ``#``, section numbers).
        max_len: Maximum length of the resulting slug.

    Returns:
        Lowercase slug with underscores. Empty if the input is empty.
    """
    text = text.lstrip("#").strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len]


def _nearest_heading(cells: list[dict], cell_idx: int) -> str:
    """Return the nearest markdown heading (backwards) to ``cell_idx``.

    Args:
        cells: List of notebook cells.
        cell_idx: Index of the cell with the figure.

    Returns:
        First non-empty line of the previous markdown, or an empty string if none.
    """
    for j in range(cell_idx, -1, -1):
        cell = cells[j]
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            line = line.strip()
            if line:
                return line
    return ""


def extract_png_outputs(notebook_path: Path, output_dir: Path) -> list[Path]:
    """Extract all inline PNG images from a notebook into ``output_dir``.

    Args:
        notebook_path: Path to the .ipynb file.
        output_dir: Target directory where the PNGs are written.

    Returns:
        List of created PNG paths, ordered by cell and output_idx.

    Raises:
        FileNotFoundError: If the notebook does not exist.
        ValueError: If the notebook JSON is corrupted.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook does not exist: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as fh:
        try:
            nb = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid notebook JSON: {notebook_path}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    cells: list[dict] = nb.get("cells", [])
    used_slugs: dict[str, int] = {}

    for cell_idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs") or []
        for output in outputs:
            data = output.get("data") or {}
            png_b64 = data.get("image/png")
            if not png_b64:
                continue
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            try:
                png_bytes = base64.b64decode(png_b64)
            except (ValueError, TypeError):
                continue
            heading = _nearest_heading(cells, cell_idx)
            slug = _slugify(heading) or "figure"
            count = used_slugs.get(slug, 0)
            used_slugs[slug] = count + 1
            target = output_dir / f"cell_{cell_idx:03d}_{slug}_{count}.png"
            if count == 0:
                # To avoid always having the _0 suffix in the common case.
                target = output_dir / f"cell_{cell_idx:03d}_{slug}.png"
            target.write_bytes(png_bytes)
            created.append(target)

    return created


@app.command()
def main(
    notebook: Path = typer.Argument(
        ...,
        help="Path al notebook .ipynb a procesar.",
    ),
    output: Path = typer.Option(
        REPO_ROOT / "paper" / "figures" / "avance1",
        "--output",
        "-o",
        help="Directorio destino para las figuras PNG.",
    ),
) -> None:
    """Extract inline PNG figures from a notebook to the target directory."""
    notebook = notebook.resolve()
    output = output.resolve()

    try:
        created = extract_png_outputs(notebook, output)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"OK {len(created)} figuras extraidas a {output}")
    for path in created:
        typer.echo(f"  - {path.name}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
