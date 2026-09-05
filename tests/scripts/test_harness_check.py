"""Tests para scripts/harness_check.py.

Un control que nunca ha fallado no se sabe si funciona: cada test rompe a proposito lo que el
gate dice vigilar y comprueba que lo reporta, y que el PASS no aparece junto al FAIL. Todos
fallan sobre la version anterior, donde tres checks imprimian su linea PASS incondicional, el
escaneo de secretos solo miraba `observations`, y un CLAUDE.md huerfano pasaba en verde.

Los chunks y guias son sinteticos: esto es mecanica del arnes, no evaluacion del articulo.
"""

from __future__ import annotations

import gzip
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


def _write_chunk(engram: Path, chunk_id: str, row: dict[str, object]) -> None:
    """Escribe un chunk gz y lo lista en el manifest."""
    chunks = engram / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    with gzip.open(chunks / f"{chunk_id}.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    (engram / "config.json").write_text(
        json.dumps({"project_name": "agrosat-copilotv2"}), encoding="utf-8"
    )
    (engram / "manifest.json").write_text(
        json.dumps({"version": 1, "chunks": [{"id": chunk_id, "memories": 1}]}), encoding="utf-8"
    )


# --- el PASS no puede convivir con el FAIL del mismo check -----------------------------------


def test_routing_no_imprime_pass_cuando_encuentra_skill_retirada(harness, tmp_path) -> None:
    """Con una tabla que enruta a la skill retirada, no debe quedar ninguna linea PASS."""
    docs = tmp_path / "docs" / "orchestration"
    docs.mkdir(parents=True)
    for nombre in ("auto-invoke.md", "skill-owners.md", "skills-catalog.md", "commands.md"):
        (docs / nombre).write_text("usa agrosat-azure-h100 para la VM", encoding="utf-8")
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


def test_graph_config_no_imprime_pass_cuando_engram_esta_ignorado(harness, tmp_path) -> None:
    """Ignorar .engram/ entero parte la memoria del equipo; el check no puede decir PASS."""
    (tmp_path / ".graphifyignore").write_text("data/\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(
        "graphify-out/\n.engram/\n.engram/engram.db\n", encoding="utf-8"
    )
    audit = _audit(harness)
    harness.check_graph_config(audit)
    assert audit.failures == 1
    assert "PASS" not in _lines(audit)


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


# --- secretos y proyectos ajenos en las cuatro listas del chunk -------------------------------


@pytest.mark.parametrize("array", ["observations", "prompts", "sessions", "mutations"])
def test_token_en_cualquier_lista_del_chunk_se_detecta(harness, tmp_path, array) -> None:
    """Un token pegado en un prompt guardado viaja igual que uno en una observacion."""
    campo = "payload" if array == "mutations" else "content"
    fila = {
        array: [
            {
                "id": "1",
                "project": "agrosat-copilotv2",
                campo: "usa sk-abcdefghijklmnop12345 para entrar",
            }
        ]
    }
    _write_chunk(tmp_path / ".engram", "aaaa1111", fila)
    audit = _audit(harness)
    harness.check_memory(audit)
    assert audit.failures == 1
    assert "posibles secretos" in _lines(audit)


@pytest.mark.parametrize("array", ["observations", "prompts", "sessions", "mutations"])
def test_proyecto_ajeno_en_cualquier_lista_se_detecta(harness, tmp_path, array) -> None:
    """Un `engram sync --all` mete memorias de otros proyectos en cualquiera de las listas."""
    fila = {array: [{"id": "1", "project": "notas-personales", "content": "algo"}]}
    _write_chunk(tmp_path / ".engram", "aaaa1111", fila)
    audit = _audit(harness)
    harness.check_memory(audit)
    assert audit.failures == 1
    assert "otros proyectos" in _lines(audit)


def test_chunk_limpio_pasa(harness, tmp_path) -> None:
    """El gate no puede ser ruido: un chunk correcto pasa en verde."""
    fila = {
        "observations": [{"id": "1", "project": "agrosat-copilotv2", "content": "una decision"}],
        "prompts": [{"id": "2", "project": "agrosat-copilotv2", "content": "un prompt"}],
    }
    _write_chunk(tmp_path / ".engram", "aaaa1111", fila)
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
    assert audit.failures == 2
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
