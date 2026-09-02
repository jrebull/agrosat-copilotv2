"""Visual cover and per-member conclusions for the course notebooks.

Generates the Markdown/HTML of:

- ``build_cover_markdown``: a standardized cover with visual identity
  (institution/program/course badges, technology stack, team data,
  sponsor and date) for the four AvanceX.Equipo17 deliverables.
- ``build_team_conclusions_markdown``: the closing block with an individual
  conclusion per member about the results of the avance.

The text lives here (not embedded in each builder) so that the four covers
are identical and editing is DRY. The strings render the same in Jupyter,
papermill and the GitHub viewer (inline HTML + shields.io badges).
"""

from __future__ import annotations

from dataclasses import dataclass

# Team 17 members (stable order in cover and conclusions).
# Each member: (name, student id, role, photo file under img/).
TEAM_MEMBERS: tuple[tuple[str, str, str, str], ...] = (
    ("Carlos Isaac Ávila Gutiérrez", "A01796035", "ML / Data Scientist", "IsaacAvila.jpg"),
    ("Carlos Aaron Bocanegra Buitrón", "A01796345", "Full-Stack / Backend", "AaronBocanegra.jpg"),
    ("Arthur Jafed Zizumbo Velasco", "A01796363", "MLOps / Platform", "ArthurZizumbo.png"),
)

# Relative path to img/ from each deliverable. The four AvanceX.Equipo17 live
# two levels below the root (notebooks/<phase>/), so ``../../img`` resolves in
# the GitHub viewer and in local Jupyter without depending on the published branch.
_IMG_BASE = "../../img"

_SPONSOR = "Dr. Gerardo Jesús Camacho González · gjcamacho@tec.mx"
_COURSE = "MNA — Tec de Monterrey · 20-abr → 3-jul-2026"


@dataclass(frozen=True)
class MemberConclusion:
    """Individual conclusion of a member about the results of the avance.

    Attributes:
        name: Member name.
        role: Short role in parentheses (e.g. ``"ML / Data Scientist"``).
        text: Paragraph with the individual reading of the obtained results.
    """

    name: str
    role: str
    text: str


def _badge(label: str, message: str, color: str) -> str:
    """Return a shields.io ``<img>`` for a label-message badge."""
    label_enc = label.replace(" ", "%20").replace("-", "--")
    msg_enc = message.replace(" ", "%20").replace("-", "--")
    return (
        f'<img alt="{label}: {message}" '
        f'src="https://img.shields.io/badge/{label_enc}-{msg_enc}-{color}">'
    )


def build_cover_markdown(
    avance: str,
    title: str,
    subtitle: str,
    delivery_date: str,
    tech_badges: tuple[tuple[str, str], ...] = (),
) -> str:
    """Build the standardized visual cover of a deliverable.

    Args:
        avance: Avance label (e.g. ``"Avance 4"``).
        title: Deliverable title (e.g. ``"Segmentacion semantica densa"``).
        subtitle: One-line tagline with the notebook scope.
        delivery_date: Delivery date (free text, e.g. ``"2026-05-31"``).
        tech_badges: (label, message) pairs of additional stack badges.

    Returns:
        Markdown string with inline HTML ready for a cover cell.
    """
    nav_badges = "&nbsp;".join(
        (
            _badge("Institucion", "Tec de Monterrey", "002D72"),
            _badge("Programa", "MNA", "0072CE"),
            _badge("Materia", "Proyecto Integrador", "F2A900"),
            _badge("Entregable", avance, "1E40AF"),
        )
    )
    default_tech = (
        ("Python", "3.12"),
        ("Jupyter", "notebook"),
        ("PyTorch", "2.4"),
        ("Polars", "1.x"),
    )
    tech = tech_badges or default_tech
    tech_row = "&nbsp;".join(_badge(label, msg, "3776AB") for label, msg in tech)

    photos = "".join(
        f'<td align="center" width="200">'
        f'<img src="{_IMG_BASE}/{photo}" width="130" height="130" '
        f'style="border-radius:50%;object-fit:contain;background:#f0f0f0"><br>'
        f"<strong>{name}</strong><br>"
        f"<code>{mid}</code><br>"
        f"<sub>{role}</sub></td>\n"
        for name, mid, role, photo in TEAM_MEMBERS
    )

    members = "<br>".join(
        f"&nbsp;&nbsp;• <strong>{name}</strong> — <code>{mid}</code>"
        for name, mid, _role, _photo in TEAM_MEMBERS
    )

    return (
        '<div align="center">\n\n'
        f"{nav_badges}\n\n"
        f"{tech_row}\n\n"
        "</div>\n\n"
        f'<h1 align="center">{avance} — {title}</h1>\n'
        f'<p align="center"><strong>AgroSatCopilot</strong> · '
        "Cuantificación de superficies de cultivo por satélite</p>\n\n"
        '<table align="center"><tr>\n'
        f"{photos}"
        "</tr></table>\n\n"
        "---\n\n"
        f"<strong>{subtitle}</strong>\n\n"
        "<table>\n"
        f"<tr><td><strong>Equipo 17</strong></td><td>{members}</td></tr>\n"
        f"<tr><td><strong>Curso</strong></td><td>{_COURSE}</td></tr>\n"
        f"<tr><td><strong>Sponsor académico</strong></td><td>{_SPONSOR}</td></tr>\n"
        f"<tr><td><strong>Fecha de entrega</strong></td><td>{delivery_date}</td></tr>\n"
        "</table>\n"
    )


def build_team_conclusions_markdown(conclusions: tuple[MemberConclusion, ...]) -> str:
    """Build the block of individual per-member conclusions.

    Args:
        conclusions: Per-member conclusions about the results of the avance.

    Returns:
        Markdown string with one subsection per member.
    """
    blocks = [
        "## Conclusiones individuales del equipo\n\n"
        "Cada integrante cierra el avance con su lectura propia de los resultados "
        "obtenidos, desde el ángulo de su rol en el proyecto.\n"
    ]
    for c in conclusions:
        blocks.append(f"\n### {c.name} — _{c.role}_\n\n{c.text}\n")
    return "".join(blocks)
