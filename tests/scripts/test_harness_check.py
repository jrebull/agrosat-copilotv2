"""Tests para scripts/harness_check.py.

Un control que nunca ha fallado no se sabe si funciona: cada test rompe a proposito lo que el
gate dice vigilar y comprueba que lo reporta, y que el PASS no aparece junto al FAIL. Todos
fallan sobre la version anterior, donde tres checks imprimian su linea PASS incondicional, el
escaneo de secretos solo miraba `observations`, y un CLAUDE.md huerfano pasaba en verde.

Los chunks y guias son sinteticos: esto es mecanica del arnes, no evaluacion del articulo.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "harness_check.py"


@pytest.fixture
def harness(tmp_path: Path, monkeypatch):
    """Carga harness_check.py con ROOT apuntando a un repo de mentira."""
    spec = importlib.util.spec_from_file_location("harness_check", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["harness_check"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def _audit(harness):
    return harness.Audit()


def _lines(audit) -> str:
    return "\n".join(audit.lines)


# --- el PASS no puede convivir con el FAIL del mismo check -----------------------------------


def test_routing_no_imprime_pass_cuando_encuentra_skill_retirada(
    harness, tmp_path, monkeypatch
) -> None:
    """Con una tabla que enruta a la skill retirada, no debe quedar ninguna linea PASS."""
    docs = tmp_path / "docs" / "orchestration"
    docs.mkdir(parents=True)
    monkeypatch.setattr(harness, "RETIRED_SKILLS", frozenset({"agrosat-retired"}))
    for nombre in ("auto-invoke.md", "skill-owners.md", "skills-catalog.md", "commands.md"):
        (docs / nombre).write_text("usa agrosat-retired", encoding="utf-8")
    audit = _audit(harness)
    harness.check_routing(audit)
    assert audit.failures == 4
    assert "PASS" not in _lines(audit)


def test_skills_no_imprime_pass_cuando_el_frontmatter_miente(harness, tmp_path) -> None:
    """Un name que no coincide con el directorio invalida el PASS del check."""
    skill = tmp_path / ".claude" / "skills" / "agrosat-demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: otra-cosa\ndescription: x\n---\ncuerpo\n", encoding="utf-8"
    )
    audit = _audit(harness)
    harness.check_skills(audit)
    assert audit.failures == 1
    assert "PASS" not in _lines(audit)


def test_graph_config_fails_when_local_engram_is_not_ignored(harness, tmp_path) -> None:
    """Native Engram state must not become versionable before ADR-015 is accepted."""
    (tmp_path / ".graphifyignore").write_text("data/\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("graphify-out/\n", encoding="utf-8")
    audit = _audit(harness)
    harness.check_graph_config(audit)
    assert audit.failures == 1
    assert "PASS" not in _lines(audit)


def test_graph_config_accepts_local_engram_boundary(harness, tmp_path) -> None:
    """Ignoring both reconstructed graph output and local memory is the current policy."""
    (tmp_path / ".graphifyignore").write_text("data/\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("graphify-out/\n.engram/\n", encoding="utf-8")
    audit = _audit(harness)
    harness.check_graph_config(audit)
    assert audit.failures == 0
    assert "PASS" in _lines(audit)


# --- espejos en las dos direcciones ----------------------------------------------------------


def test_claude_md_sin_agents_md_hermano_falla(harness, tmp_path, monkeypatch) -> None:
    """Borrar el AGENTS.md deja a Codex, Copilot y Cursor sin guia; eso debe verse."""
    (tmp_path / "ml").mkdir()
    (tmp_path / "ml" / "CLAUDE.md").write_text("guia de ml\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("raiz\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("raiz\n", encoding="utf-8")
    monkeypatch.setattr(
        harness,
        "guide_files",
        lambda: [
            tmp_path / "AGENTS.md",
            tmp_path / "CLAUDE.md",
            tmp_path / "ml" / "CLAUDE.md",
        ],
    )
    audit = _audit(harness)
    harness.check_guides(audit, sync=False)
    assert audit.failures == 1
    assert "ml/CLAUDE.md" in _lines(audit)


def test_claude_md_puntero_permitido_no_falla(harness, tmp_path, monkeypatch) -> None:
    """.claude/CLAUDE.md es un puntero declarado, no un espejo huerfano."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "CLAUDE.md").write_text("puntero\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("raiz\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("raiz\n", encoding="utf-8")
    monkeypatch.setattr(
        harness,
        "guide_files",
        lambda: [
            tmp_path / "AGENTS.md",
            tmp_path / "CLAUDE.md",
            tmp_path / ".claude" / "CLAUDE.md",
        ],
    )
    audit = _audit(harness)
    harness.check_guides(audit, sync=False)
    assert audit.failures == 0


def test_tracked_native_engram_state_fails(harness, monkeypatch) -> None:
    """A tracked chunk must fail even if its payload looks harmless."""
    monkeypatch.setattr(
        harness, "_tracked_engram_files", lambda: [".engram/chunks/aaaa1111.jsonl.gz"]
    )
    audit = _audit(harness)
    harness.check_memory(audit)
    assert audit.failures == 1
    assert "versionado" in _lines(audit)
    assert "PASS" not in _lines(audit)


def test_absent_tracked_engram_state_passes(harness, monkeypatch) -> None:
    """Local or absent Engram state is valid because Git sees neither one."""
    monkeypatch.setattr(harness, "_tracked_engram_files", lambda: [])
    audit = _audit(harness)
    harness.check_memory(audit)
    assert audit.failures == 0
    assert "PASS" in _lines(audit)


# --- ids retirados en la superficie que se carga sola -----------------------------------------


def test_descripcion_con_id_retirado_falla(harness, tmp_path) -> None:
    """La description de una skill entra en la lista de cada sesion: ahi no vive un id muerto."""
    skill = tmp_path / ".claude" / "skills" / "agrosat-demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: agrosat-demo\ndescription: sirve Qwen3.5-35B-A3B en H100\n---\n",
        encoding="utf-8",
    )
    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    audit = _audit(harness)
    harness.check_retired_terms(audit)
    assert audit.failures == 1
    assert "PASS" not in _lines(audit)


def test_mencion_que_niega_el_termino_no_falla(harness, tmp_path) -> None:
    """ "Sin H100" es la guia haciendo su trabajo, no una reintroduccion."""
    skill = tmp_path / ".claude" / "skills" / "agrosat-demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: agrosat-demo\ndescription: corre en CPU. Sin Azure ni H100.\n---\n",
        encoding="utf-8",
    )
    (tmp_path / ".claude" / "agents").mkdir(parents=True)
    audit = _audit(harness)
    harness.check_retired_terms(audit)
    assert audit.failures == 0


# --- enlaces y permisos ----------------------------------------------------------------------


def test_enlace_roto_fuera_del_agents_raiz_se_detecta(harness, tmp_path) -> None:
    """El check miraba solo AGENTS.md; un agente que entra por otra puerta veia el enlace roto."""
    docs = tmp_path / "docs" / "orchestration"
    docs.mkdir(parents=True)
    (docs / "auto-invoke.md").write_text("ver [loop](no-existe.md)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("sin enlaces\n", encoding="utf-8")
    audit = _audit(harness)
    harness.check_links(audit)
    assert audit.failures == 1
    assert "auto-invoke.md" in _lines(audit)


def test_percent_encoded_relative_link_resolves(harness, tmp_path, monkeypatch) -> None:
    """A valid Markdown link with an encoded space must not be reported as missing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "Rubrica Integrador.html"
    target.write_text("rubrica\n", encoding="utf-8")
    source = tmp_path / "README.md"
    source.write_text("[rubrica](docs/Rubrica%20Integrador.html)\n", encoding="utf-8")
    monkeypatch.setattr(harness, "_link_docs", lambda: [source])
    audit = _audit(harness)
    harness.check_links(audit)
    assert audit.failures == 0
    assert "PASS" in _lines(audit)


def test_permiso_make_comodin_falla(harness, tmp_path) -> None:
    """Bash(make:*) auto-aprueba sellado, deploy y dvc push."""
    settings = tmp_path / ".claude"
    settings.mkdir()
    (settings / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(make:*)"]}, "hooks": {"SessionStart": []}}),
        encoding="utf-8",
    )
    audit = _audit(harness)
    harness.check_settings(audit)
    assert audit.failures == 1
    assert "Bash(make:*)" in _lines(audit)


# --- ausencias que antes reventaban con traceback ---------------------------------------------


def test_skills_ausente_produce_fail_no_traceback(harness, tmp_path) -> None:
    """En un worktree sin .claude/skills el gate reporta, no muere."""
    audit = _audit(harness)
    harness.check_skills(audit)
    assert audit.failures == 1
    assert "skills ausente" in _lines(audit)


def test_gitignore_ausente_produce_fail_no_traceback(harness, tmp_path) -> None:
    """Lo mismo para el .gitignore que lee check_graph_config."""
    audit = _audit(harness)
    harness.check_graph_config(audit)
    assert audit.failures >= 1
    assert "PASS" not in _lines(audit)
