"""Auto-generate ``docs/agent/tools.md`` from the agent tool registry (AC-6).

Single source of truth for the agent tool catalogue documentation: this script
introspects :data:`ml.agent.tools.TOOL_SPECS` (the static descriptor table) and
emits a Markdown reference listing every tool with its deferred flag, description
and the JSON schema of its Pydantic input/output models (via
``model_json_schema()``). Re-run it whenever a tool's schema or description
changes so the docs never drift from the contracts.

``TOOL_SPECS`` is used rather than ``TOOL_REGISTRY`` so the generator depends only
on the schemas/metadata and does **not** import the per-tool ``run`` modules:
the documentation can be regenerated even while sibling sub-tasks are still
landing their ``ml/agent/tools/<name>.py`` modules.

The prose in the generated document is Spanish (reader-facing), per the project
language policy; identifiers, JSON keys and code fences stay verbatim.

Usage:

    poetry run python scripts/gen_tools_doc.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import BaseModel

from ml.agent.tools import TOOL_SPECS

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _ToolDoc:
    """Documentation view of a tool, derived from a ``TOOL_SPECS`` descriptor.

    Attributes:
        name: Stable tool name exposed to the LLM.
        deferred: ``True`` for background/non-blocking tools.
        description: One-line natural-language description.
        input_model: Pydantic model validating the tool arguments.
        output_model: Pydantic model of the tool result.
    """

    name: str
    deferred: bool
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]


def _tool_docs() -> list[_ToolDoc]:
    """Build the documentation views from the static descriptor table.

    Returns:
        One :class:`_ToolDoc` per registered tool, in registry order. No per-tool
        ``run`` module is imported.
    """
    docs: list[_ToolDoc] = []
    for name, (_module, input_model, output_model, deferred, description) in TOOL_SPECS.items():
        docs.append(
            _ToolDoc(
                name=name,
                deferred=deferred,
                description=description,
                input_model=input_model,
                output_model=output_model,
            )
        )
    return docs


# Repository root: this file lives in ``<root>/scripts/gen_tools_doc.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_PATH = _REPO_ROOT / "docs" / "agent" / "tools.md"


def _schema_block(model: type) -> str:
    """Render a Pydantic model's JSON schema as a fenced JSON code block.

    Args:
        model: A Pydantic ``BaseModel`` subclass (a tool ``*Input``/``*Output``).

    Returns:
        A Markdown ``json`` fenced block with the indented schema.
    """
    schema = model.model_json_schema()
    rendered = json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=False)
    return f"```json\n{rendered}\n```"


def _summary_table(docs: list[_ToolDoc]) -> str:
    """Build the summary Markdown table (name, deferred, description).

    Args:
        docs: Tool documentation views in registry order.

    Returns:
        A Markdown table summarising every tool.
    """
    lines = [
        "| Tool | Diferida | Descripcion |",
        "|------|----------|-------------|",
    ]
    for doc in docs:
        deferred = "si" if doc.deferred else "no"
        lines.append(f"| `{doc.name}` | {deferred} | {doc.description} |")
    return "\n".join(lines)


def _tool_section(doc: _ToolDoc) -> str:
    """Render the detailed section for a single tool.

    Args:
        doc: The tool documentation view to render.

    Returns:
        A Markdown section with metadata and input/output JSON schemas.
    """
    mode = "diferida (background, `Behavior.NON_BLOCKING`)" if doc.deferred else "sincrona"
    return "\n".join(
        [
            f"## `{doc.name}`",
            "",
            f"- **Modo**: {mode}.",
            f"- **Descripcion**: {doc.description}",
            f"- **Modelo de entrada**: `{doc.input_model.__name__}`",
            f"- **Modelo de salida**: `{doc.output_model.__name__}`",
            "",
            "### Esquema de entrada",
            "",
            _schema_block(doc.input_model),
            "",
            "### Esquema de salida",
            "",
            _schema_block(doc.output_model),
            "",
        ]
    )


def render_document() -> str:
    """Render the full ``tools.md`` document from the tool registry.

    Returns:
        The complete Markdown document as a string.
    """
    docs = _tool_docs()
    n_deferred = sum(1 for doc in docs if doc.deferred)
    n_sync = len(docs) - n_deferred

    header = "\n".join(
        [
            "# Catalogo de FunctionTools del agente",
            "",
            "> Documento generado automaticamente por `scripts/gen_tools_doc.py` a",
            "> partir de `ml.agent.tools.TOOL_SPECS`. No editar a mano: cualquier",
            "> cambio se sobrescribe al regenerar. Los esquemas provienen de los",
            "> modelos Pydantic v2 (`model_json_schema()`) de `ml/agent/schemas.py`.",
            "",
            f"Total de tools: **{len(docs)}** ({n_sync} sincronas, {n_deferred} diferidas).",
            "",
            "## Resumen",
            "",
            _summary_table(docs),
            "",
        ]
    )
    sections = "\n".join(_tool_section(doc) for doc in docs)
    return f"{header}\n{sections}"


def main() -> int:
    """Generate ``docs/agent/tools.md`` and report the outcome.

    Returns:
        Process exit code (``0`` on success).
    """
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = render_document()
    _OUTPUT_PATH.write_text(document, encoding="utf-8")
    logger.info(
        "tools_doc_generated",
        output=str(_OUTPUT_PATH),
        n_tools=len(TOOL_SPECS),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
