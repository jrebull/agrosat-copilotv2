"""Canonical bootstrap for AgroSatCopilot project notebooks.

Centralizes the pattern that would otherwise be duplicated in each `.ipynb`:

- Robust resolution of the repo root (re-export of `notebook_setup.find_repo_root`).
- Loading of `.env.local`.
- Configuration of `sys.path` so `import ml.*` works from any subfolder of
  `notebooks/`.
- Configuration of Polars (rich HTML rendering), matplotlib (DPI, inline) and
  autoreload (`%autoreload 2`).
- Creation of the notebook figures directory (`paper/figures/{slug}`).

Returns a `NotebookEnv` dataclass with the useful paths so each notebook does not
have to rebuild them.

Typical usage in cell 3 of the notebook:

```python
from ml.utils.notebook_bootstrap import setup_notebook

env = setup_notebook(figures_subdir="us-023-preview/04_baseline")
display(env.summary_markdown())
```
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ml.utils.notebook_setup import (
    configure_ee_from_env,
    find_repo_root,
    load_env_local,
)

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

__all__ = ["NotebookEnv", "setup_notebook"]


@dataclass(frozen=True)
class NotebookEnv:
    """Paths and configuration derived from the notebook bootstrap.

    Attributes:
        repo: Absolute path to the repo root (resolved via `pyproject.toml`).
        figures_dir: Directory where the notebook persists PNG plots.
        reports_dir: Directory where the notebook persists tables and parquets.
        data_dir: Shortcut to `repo / "data"`.
        cache_dir: Shortcut to `repo / "data/cache"`.
        has_gemini_api_key: True if `GEMINI_API_KEY` or `GOOGLE_API_KEY` or
            `GOOGLE_GENAI_USE_VERTEXAI=true` are present in the env.
        has_ee_credentials: True if Earth Engine can be initialized with SA or ADC.
        gee_project: GCP project for EE (may be None).
        gee_sa_path: Path to the EE service account JSON (may be None).
        env_warnings: List of actionable messages for the user.
    """

    repo: Path
    figures_dir: Path
    reports_dir: Path
    data_dir: Path
    cache_dir: Path
    has_gemini_api_key: bool
    has_ee_credentials: bool
    gee_project: str | None
    gee_sa_path: Path | None
    env_warnings: list[str] = field(default_factory=list)

    def summary_markdown(self) -> str:
        """Build a readable Markdown summary for `display(Markdown(...))`.

        Returns:
            Markdown text with a table of paths and credential status.
        """
        rows = [
            "| Recurso | Estado |",
            "|---|---|",
            f"| Repo root | `{self.repo}` |",
            f"| Figures dir | `{self.figures_dir.relative_to(self.repo)}` |",
            f"| Reports dir | `{self.reports_dir.relative_to(self.repo)}` |",
            f"| Gemini API key | {'presente' if self.has_gemini_api_key else 'ausente'} |",
            f"| Earth Engine | {'configurado' if self.has_ee_credentials else 'no configurado'} |",
        ]
        if self.gee_project:
            rows.append(f"| GEE project | `{self.gee_project}` |")
        text = "\n".join(rows)
        if self.env_warnings:
            text += "\n\n**Avisos**:\n" + "\n".join(f"- {w}" for w in self.env_warnings)
        return text


def setup_notebook(
    figures_subdir: str = "default",
    reports_subdir: str = "default",
    *,
    enable_autoreload: bool = True,
    matplotlib_inline: bool = True,
    polars_rich_html: bool = True,
    load_dotenv: bool = True,
    ipython: InteractiveShell | None = None,
) -> NotebookEnv:
    """Apply the canonical bootstrap and return the environment ready to use.

    Follows the order documented in `notebooks/CLAUDE.md` Section "Estructura
    estandar de notebook" Cell 3.

    Args:
        figures_subdir: Subfolder under `paper/figures/` for the PNGs. The
            effective path is left in `env.figures_dir`.
        reports_subdir: Subfolder under `reports/` for tables/parquets.
        enable_autoreload: If True runs `%load_ext autoreload` and
            `%autoreload 2` (changes in `ml/*.py` are reflected without
            restarting the kernel).
        matplotlib_inline: If True runs `%matplotlib inline`.
        polars_rich_html: If True configures Polars for formatted HTML rendering
            (`ASCII_MARKDOWN`, 20 rows, 60 chars).
        load_dotenv: If True (default) loads `.env.local` into `os.environ`.
            Deterministic tests can set it to False to avoid overwriting their
            monkeypatches.
        ipython: IPython shell (auto-detected if None). Used for the magics
            `%load_ext`, `%autoreload`, `%matplotlib`. If we are not inside
            IPython, the magics are silently skipped.

    Returns:
        `NotebookEnv` with resolved paths and credential status.
    """
    repo = find_repo_root()

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if load_dotenv:
        load_env_local(repo)
        gee_project, gee_sa_path = configure_ee_from_env(repo)
    else:
        gee_project = os.environ.get("GEE_PROJECT_ID") or None
        gee_sa_env = os.environ.get("GEE_SERVICE_ACCOUNT_PATH")
        gee_sa_path = Path(gee_sa_env) if gee_sa_env and Path(gee_sa_env).is_file() else None

    figures_dir = repo / "paper" / "figures" / figures_subdir
    reports_dir = repo / "reports" / reports_subdir
    data_dir = repo / "data"
    cache_dir = data_dir / "cache"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if polars_rich_html:
        _configure_polars()

    if matplotlib_inline:
        _configure_matplotlib(ipython)

    if enable_autoreload:
        _enable_autoreload(ipython)

    has_gemini_api_key = _detect_gemini_credentials()
    has_ee_credentials = gee_project is not None or gee_sa_path is not None

    env_warnings: list[str] = []
    if not has_gemini_api_key:
        env_warnings.append(
            "Gemini API key ausente. Exporta `GEMINI_API_KEY` en `.env.local` "
            "antes de ejecutar celdas que llamen a `materialize_phenology_text`."
        )
    if not has_ee_credentials:
        env_warnings.append(
            "Earth Engine no configurado. Define `GEE_PROJECT_ID` y opcionalmente "
            "`GEE_SERVICE_ACCOUNT_PATH` en `.env.local`, o ejecuta "
            "`earthengine authenticate` localmente."
        )

    return NotebookEnv(
        repo=repo,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        data_dir=data_dir,
        cache_dir=cache_dir,
        has_gemini_api_key=has_gemini_api_key,
        has_ee_credentials=has_ee_credentials,
        gee_project=gee_project,
        gee_sa_path=gee_sa_path,
        env_warnings=env_warnings,
    )


def _configure_polars() -> None:
    """Configure Polars for rich rendering in notebooks."""
    import polars as pl

    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_fmt_str_lengths(60)


def _configure_matplotlib(ipython: InteractiveShell | None) -> None:
    """Configure matplotlib (high DPI + inline backend) for notebooks."""
    import matplotlib.pyplot as plt

    shell = ipython or _get_ipython()
    if shell is not None:
        try:
            shell.run_line_magic("matplotlib", "inline")
        except (ValueError, AttributeError):
            pass

    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200


def _enable_autoreload(ipython: InteractiveShell | None) -> None:
    """Enable `%autoreload 2` if we are inside IPython."""
    shell = ipython or _get_ipython()
    if shell is None:
        return
    try:
        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
    except (ValueError, AttributeError):
        pass


def _get_ipython() -> InteractiveShell | None:
    """Return the active IPython shell or None if we are not in a notebook."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _detect_gemini_credentials() -> bool:
    """Detect whether any environment variable enables the Gemini call.

    Returns:
        True if at least one of the following is present: GEMINI_API_KEY,
        GOOGLE_API_KEY, or (GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT).
    """
    if os.environ.get("GEMINI_API_KEY"):
        return True
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    has_project = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return use_vertex and has_project
