"""Shared figure infrastructure for the MICAI manuscript.

The retired manuscript generated figures at 7.2 inches and placed them at the
4.8-inch LNCS text width. That reduced a requested 7 pt font to roughly 4.7 pt
on paper. This module makes the physical output width, printed font size,
redundant line-series channels, and byte reproducibility executable rules.

Labels remain data selected by language; callers do not translate individual
figures ad hoc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

__all__ = [
    "LNCS_TEXT_WIDTH_INCHES",
    "MIN_FONT_SIZE_PT",
    "SERIES_STYLES",
    "SeriesStyle",
    "apply_manuscript_style",
    "find_series_channel_violations",
    "find_texts_below_minimum",
    "save_figure",
    "translate_label",
    "validate_figure",
]

# Measured by compiling ``\the\textwidth`` with the llncs class. TeX uses
# 72.27 points per inch; PDF and SVG use 72 PostScript points per inch.
LNCS_TEXT_WIDTH_INCHES: float = 347.12354 / 72.27
MIN_FONT_SIZE_PT: float = 8.0
WIDTH_TOLERANCE_INCHES: float = 1e-4

SUPPORTED_FORMATS: frozenset[str] = frozenset({"pdf", "svg"})
DETERMINISTIC_METADATA: dict[str, dict[str, str | None]] = {
    "svg": {"Date": None},
    "pdf": {"CreationDate": None, "ModDate": None},
}


@dataclass(frozen=True)
class SeriesStyle:
    """Visual channels assigned to one plotted series.

    Attributes:
        color: Series colour, which is never its only distinguishing channel.
        marker: Marker shape used when colour is unavailable.
        line_style: Stroke pattern used when colour and markers are unavailable.
    """

    color: str
    marker: str
    line_style: str


SERIES_STYLES: dict[str, SeriesStyle] = {
    "catalog_reduction": SeriesStyle("#B4522F", "o", "-"),
    "abstention": SeriesStyle("#3E6B89", "s", "--"),
    "label_set": SeriesStyle("#6B8E3E", "^", "-."),
    "taxonomic_backoff": SeriesStyle("#8A6D3B", "D", ":"),
}

_LABELS: dict[str, dict[str, str]] = {
    "es": {
        "coverage": "Cobertura",
        "quality": "Calidad",
        "catalog_reduction": "recorte de leyenda",
        "abstention": "abstención",
        "label_set": "conjunto de clases",
        "taxonomic_backoff": "retroceso taxonómico",
        "no_mechanism": "sin mecanismo",
        "classes": "clases prometidas",
        "parcels": "parcelas",
    },
    "en": {
        "coverage": "Coverage",
        "quality": "Quality",
        "catalog_reduction": "legend shrinking",
        "abstention": "abstention",
        "label_set": "class set",
        "taxonomic_backoff": "taxonomic back-off",
        "no_mechanism": "no mechanism",
        "classes": "promised classes",
        "parcels": "parcels",
    },
}


def translate_label(key: str, language: str) -> str:
    """Return one label in the requested language.

    Args:
        key: Stable label key.
        language: Supported language code, currently ``"es"`` or ``"en"``.

    Returns:
        The translated label.

    Raises:
        KeyError: If the language or key is unknown. Missing translations never
            fall back silently to another language.
    """
    if language not in _LABELS:
        raise KeyError(f"idioma desconocido: {language!r}; hay {sorted(_LABELS)}")
    if key not in _LABELS[language]:
        raise KeyError(f"no hay rótulo {key!r} en {language!r}")
    return _LABELS[language][key]


def apply_manuscript_style(*, base_font_size_pt: float = 9.0) -> None:
    """Set the rcParams shared by manuscript figures.

    Derived sizes are clamped to the printed minimum. Output cropping is
    disabled because a tight bounding box changes the physical canvas width
    and silently reintroduces a placement scale.

    Args:
        base_font_size_pt: Base printed font size in points.

    Raises:
        ValueError: If the requested base font is already below the minimum.
    """
    if base_font_size_pt < MIN_FONT_SIZE_PT:
        raise ValueError(
            f"la tipografía base es {base_font_size_pt} pt y el mínimo impreso es "
            f"{MIN_FONT_SIZE_PT}"
        )
    derived_size = max(base_font_size_pt - 0.5, MIN_FONT_SIZE_PT)
    plt.rcParams.update(
        {
            "font.size": base_font_size_pt,
            "axes.titlesize": base_font_size_pt + 1,
            "axes.labelsize": base_font_size_pt,
            "xtick.labelsize": derived_size,
            "ytick.labelsize": derived_size,
            "legend.fontsize": derived_size,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "svg.hashsalt": "agrosat-micai",
            "pdf.compression": 6,
        }
    )


def find_texts_below_minimum(
    figure: Figure,
    *,
    scale_factor: float = 1.0,
) -> list[tuple[str, float]]:
    """Find text whose printed size would fall below the minimum.

    Args:
        figure: Fully constructed figure.
        scale_factor: Downstream placement scale. The normal value is exactly
            one because :func:`save_figure` enforces the LNCS canvas width.

    Returns:
        Pairs containing the text and its resulting printed size.

    Raises:
        ValueError: If the scale factor is not positive.
    """
    if scale_factor <= 0:
        raise ValueError("el factor de escala debe ser positivo")

    offenders: list[tuple[str, float]] = []
    for artist in figure.findobj():
        if not isinstance(artist, Text):
            continue
        content = artist.get_text().strip()
        if not content:
            continue
        printed_size = float(artist.get_fontsize()) * scale_factor
        if printed_size < MIN_FONT_SIZE_PT:
            offenders.append((content, round(printed_size, 2)))
    return offenders


def find_series_channel_violations(figure: Figure) -> list[str]:
    """Find line series that are distinguishable only by colour.

    The check examines the artists that will actually be saved, not the shared
    style table. When an axes contains multiple visible lines, every line must
    have a marker and a stroke pattern, and each non-colour signature must be
    unique. Decorative lines may be explicitly marked with the
    ``micai-decoration`` gid and are excluded from the series set.

    Args:
        figure: Fully constructed figure.

    Returns:
        Human-readable violations, or an empty list when redundant channels
        distinguish every line series.
    """
    violations: list[str] = []
    for axes_index, axes in enumerate(figure.axes, start=1):
        lines = [
            line
            for line in axes.lines
            if line.get_visible() and line.get_gid() != "micai-decoration"
        ]
        if len(lines) <= 1:
            continue

        owners_by_signature: dict[tuple[str, str], str] = {}
        for line_index, line in enumerate(lines, start=1):
            raw_label = str(line.get_label())
            label = (
                raw_label
                if raw_label and not raw_label.startswith("_")
                else f"eje {axes_index}, serie {line_index}"
            )
            marker = str(line.get_marker())
            line_style = str(line.get_linestyle())
            has_marker = marker.lower() not in {"", "none", " ", "nan"}
            has_line_style = line_style.lower() not in {"", "none", " ", "nan"}
            if not has_marker or not has_line_style:
                missing = []
                if not has_marker:
                    missing.append("marcador")
                if not has_line_style:
                    missing.append("trazo")
                violations.append(f"{label!r} no tiene {' ni '.join(missing)}")
                continue

            signature = (marker, line_style)
            previous_owner = owners_by_signature.get(signature)
            if previous_owner is not None:
                violations.append(
                    f"{label!r} repite marcador y trazo de {previous_owner!r}: {signature}"
                )
            else:
                owners_by_signature[signature] = label
    return violations


def validate_figure(figure: Figure) -> None:
    """Validate the physical and accessibility contract of one figure.

    Args:
        figure: Fully constructed figure.

    Raises:
        ValueError: If width, printed font size, or redundant channels violate
            the manuscript contract.
    """
    width_inches = float(figure.get_size_inches()[0])
    if not math.isclose(
        width_inches,
        LNCS_TEXT_WIDTH_INCHES,
        rel_tol=0.0,
        abs_tol=WIDTH_TOLERANCE_INCHES,
    ):
        placement_scale = LNCS_TEXT_WIDTH_INCHES / width_inches
        raise ValueError(
            f"el ancho es {width_inches:.4f} in, no {LNCS_TEXT_WIDTH_INCHES:.4f} in; "
            f"al colocarlo a textwidth reaparecería una escala {placement_scale:.4f}"
        )

    text_offenders = find_texts_below_minimum(figure)
    if text_offenders:
        detail = ", ".join(f"{text!r} a {size} pt" for text, size in text_offenders[:5])
        raise ValueError(
            f"{len(text_offenders)} texto(s) por debajo de {MIN_FONT_SIZE_PT} pt impresos: {detail}"
        )

    channel_violations = find_series_channel_violations(figure)
    if channel_violations:
        detail = "; ".join(channel_violations[:5])
        raise ValueError(f"el color es el único canal de una o más series: {detail}")


def save_figure(
    figure: Figure,
    destination: Path,
    *,
    formats: tuple[str, ...] = ("pdf", "svg"),
) -> list[Path]:
    """Validate and save one figure with a fixed physical canvas.

    Args:
        figure: Fully constructed figure.
        destination: Output path without an extension.
        formats: Vector formats to write.

    Returns:
        Paths written in the same order as ``formats``.

    Raises:
        ValueError: If the figure contract fails, no format is requested, or an
            unsupported format is requested.
    """
    if not formats:
        raise ValueError("se requiere al menos un formato de salida")
    unsupported_formats = sorted(set(formats) - SUPPORTED_FORMATS)
    if unsupported_formats:
        raise ValueError(f"formatos no soportados: {unsupported_formats}")

    validate_figure(figure)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas = figure.bbox_inches.frozen()
    written_paths: list[Path] = []
    for extension in formats:
        output_path = destination.with_suffix(f".{extension}")
        figure.savefig(
            output_path,
            format=extension,
            bbox_inches=canvas,
            metadata=DETERMINISTIC_METADATA[extension],
        )
        written_paths.append(output_path)
    return written_paths
