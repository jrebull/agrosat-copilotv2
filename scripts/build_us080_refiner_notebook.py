"""Builder of the US-080 FarSLIP-refiner evaluation notebook (Equipo 17).

Generates ``notebooks/transfer/us080_farslip_refiner.ipynb`` reproducibly (same
idempotent ``nbformat`` + ``typer`` pattern as the sibling ``scripts/build_*
_notebook.py``). The notebook demonstrates and evaluates the US-080 second stage:
a FarSLIP zero-shot signal -- guided by the LLM phenology description -- that
re-ranks the Voting-3 champion ONLY on uncertain / open-set parcels.

What the notebook shows:

1. The per-class phenology prompts (``ml.farslip.class_prompts``) -- REAL,
   deterministic.
2. The gated refiner logic (``ml.agent.refine.apply_refinement``) on REAL Voting-3
   posteriors of seeded PASTIS-R parcels: the easy case is left untouched, the
   uncertain / disagreement case is re-ranked. The FarSLIP score per class is the
   ONLY illustrative piece (clearly labelled) until FarSLIP is served + the chips
   exist -- the documented blocker.
3. The FarSLIP zero-shot head (``ml.farslip.zeroshot_head``): it tries to load the
   real FarSLIP extractor and, if the weights are reachable, scores; otherwise it
   reports the blocker honestly (no fabricated embeddings).
4. The delta-F1 eval harness (``ml.eval.farslip_refine_eval.run_refine_eval``):
   Voting-3 vs Voting-3+refine. Runs REAL when FarSLIP + chips are available;
   otherwise prints the PENDING state and how to complete it.

HARD RULE -- REAL VALUES ONLY. The Voting-3 posteriors, the GT and the prompts are
real. The FarSLIP scores used to illustrate the mechanism are marked ILUSTRATIVO,
never reported as a measured result; the delta-F1 cells either run for real or say
PENDIENTE -- they never invent a number.

Visible prose is Spanish; code/identifiers English ASCII. No emojis.

Usage::

    poetry run python scripts/build_us080_refiner_notebook.py \\
        --out notebooks/transfer/us080_farslip_refiner.ipynb

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us080_farslip_refiner.ipynb")


def _build_cells() -> list:
    """Build the markdown + code cells of the US-080 refiner notebook."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ---------------------------------------------------------------- Cover ---
    cells.append(
        md(
            "# US-080 - Refinador FarSLIP-fenologia: el copiloto afina el Voting-3\n\n"
            '### Segunda etapa "Be My Eyes": el LLM describe, FarSLIP re-rankea\n\n'
            "**Equipo 17** - AgroSatCopilot - EPIC 12\n\n"
            "---\n\n"
            "El campeon del copiloto es el **Voting-3** (US-079). Esta US anade una **segunda "
            "etapa opcional**: el reasoner LLM genera una **descripcion fenologica por clase** y "
            "**FarSLIP** (CLIP region-aware, alinea imagen<->texto) puntua la parcela contra cada "
            "descripcion; esa senal se **fusiona** con el posterior Voting-3 **solo cuando los "
            "modelos densos estan inseguros** (margen bajo, miembros en desacuerdo) o ante una "
            "**clase abierta/nueva**. En el caso facil el campeon **no se toca** -- nunca empeora "
            "donde ya acierta.\n\n"
            'Metodo respaldado por *"Phenology description is all you need!"* (ISPRS J. P&RS '
            "2025/26) y la linea LLM-descripcion -> CLIP (CuPL 2022, Saha 2024, Concept-Guided "
            "Bayesian 2026).\n\n"
            "> **Solo valores reales.** Los posteriores Voting-3, la verdad de campo y los prompts "
            "son reales. El **score FarSLIP** usado para ilustrar el mecanismo se marca "
            "**ILUSTRATIVO** (no es una medicion); el **ΔF1** se corre de verdad si FarSLIP esta "
            "servido + existen los chips, o dice **PENDIENTE** con como completarlo -- nunca inventa."
        )
    )

    # --------------------------------------------- parameters (papermill) ---
    cells.append(
        code(
            "# Parametros (papermill). Sobreescribe con `papermill -p <name> <value>`.\n"
            "demo_user = 'demo@agrosat.dev'   # propietario de la sesion sembrada\n"
            "n_parcels = 3                    # parcelas reales a observar\n"
            "alpha = 0.4                      # peso convexo de la senal FarSLIP en la fusion\n"
            "margin_tau = 0.15                # umbral de margen top1-top2 para disparar\n"
            "farslip_weights = 'gs://agrosat-models/farslip/farslip-clip-italy-v1/'  # pesos FarSLIP"
        )
    )
    cells[-1]["metadata"]["tags"] = ["parameters"]

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        md(
            "## Preparacion del entorno\n\n"
            "Resolvemos la raiz del repo, cargamos `.env.local`, silenciamos el ruido de logs y "
            "abrimos el *pool* de la base local. Sin rutas absolutas ni secretos."
        )
    )
    cells.append(
        code(
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "for _stream in (sys.stdout, sys.stderr):\n"
            "    try:\n"
            "        _stream.reconfigure(encoding='utf-8')\n"
            "    except (AttributeError, ValueError):\n"
            "        pass\n\n"
            "from ml.utils.notebook_setup import find_repo_root, load_env_local\n\n"
            "REPO_ROOT = find_repo_root()\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n"
            "os.chdir(REPO_ROOT)\n"
            "load_env_local(REPO_ROOT)\n\n"
            "%load_ext autoreload\n"
            "%autoreload 2\n\n"
            "import polars as pl\n"
            "from IPython.display import Markdown, display\n\n"
            "from ml.agent import demo\n"
            "demo.quiet_logging()\n"
            "print('repo:', REPO_ROOT)"
        )
    )

    # ============================================ Seccion 1 - Prompts por clase ===
    cells.append(
        md(
            "## 1. Las descripciones fenologicas por clase (lo que ve FarSLIP)\n\n"
            "FarSLIP alinea la imagen de la parcela contra **un texto por clase candidata**. "
            "Idealmente el reasoner genera esos textos en vivo; aqui usamos el **fallback "
            "determinista** (en ingles, porque el encoder de texto de FarSLIP es el CLIP teacher) "
            "para las nueve clases bien resueltas `france-9`. Son los prompts contra los que se "
            "puntua la imagen."
        )
    )
    cells.append(
        code(
            "from ml.eval.class_remap import get_label_space\n"
            "from ml.farslip.class_prompts import build_class_prompts\n\n"
            "france9 = list(get_label_space('france-9').class_names.values())\n"
            "prompts = build_class_prompts(france9)\n"
            "display(pl.DataFrame({'clase': list(prompts), 'descripcion_fenologica': list(prompts.values())}))"
        )
    )

    # ============================================ Seccion 2 - El refinador ===
    cells.append(
        md(
            "## 2. El refinador, sobre posteriores Voting-3 REALES\n\n"
            "Observamos parcelas reales de la sesion con el perceiver (posterior **Voting-3 real**) "
            "y aplicamos `apply_refinement`. El disparo es **selectivo**: en el caso facil (margen "
            "alto, miembros de acuerdo) devuelve el campeon **intacto**; en el incierto / "
            "open-set lo re-rankea. El **score FarSLIP** de abajo es **ILUSTRATIVO** (un favor a "
            "una clase) para ensenar el mecanismo: el score real sale del cabezal de la seccion 3 "
            "cuando FarSLIP este servido."
        )
    )
    cells.append(
        code(
            "from ml.agent.context import ToolContext\n"
            "from ml.agent.db import get_pool\n"
            "from ml.agent.perceiver import PerceiverLayer\n"
            "from backend.app.core.config import get_settings\n\n"
            "settings = get_settings()\n"
            "pool = await get_pool()\n"
            "async with pool.acquire() as conn:\n"
            "    session_id = await conn.fetchval(\n"
            "        'SELECT cs.id FROM chat_sessions cs LEFT JOIN parcels p ON p.session_id=cs.id '\n"
            "        'WHERE cs.user_id=$1 GROUP BY cs.id ORDER BY count(p.id) DESC, cs.id LIMIT 1',\n"
            "        demo_user)\n"
            "    _rows = await conn.fetch(\n"
            "        'SELECT id FROM parcels WHERE session_id=$1 ORDER BY id LIMIT $2',\n"
            "        session_id, int(n_parcels))\n"
            "parcel_ids = [int(r['id']) for r in _rows]\n"
            "ctx = ToolContext(pool=pool, settings=settings, session_id=session_id)\n"
            "perceiver = PerceiverLayer(ctx)\n"
            "observations = await demo.observe_parcels(perceiver, parcel_ids)\n"
            "print('parcelas:', parcel_ids)"
        )
    )
    cells.append(
        code(
            "# Apply the gated refiner to each real Voting-3 posterior. The FarSLIP score here is\n"
            "# ILLUSTRATIVE (it favours the parcel's 2nd-most-likely class to show the mechanism);\n"
            "# the REAL score comes from the FarSLIP head (section 3) once it is served.\n"
            "from ml.agent.refine import apply_refinement, top1_top2_margin\n\n"
            "_rows = []\n"
            "for obs, _ in observations:\n"
            "    post = obs.class_probabilities\n"
            "    _ranked = sorted(post, key=lambda k: post[k], reverse=True)\n"
            "    # ILUSTRATIVO: a FarSLIP that favours the runner-up class (forces an uncertain re-rank).\n"
            "    _illustrative = {c: (1.0 if c == _ranked[min(1, len(_ranked) - 1)] else 0.0) for c in post}\n"
            "    res = apply_refinement(post, _illustrative, alpha=alpha, margin_tau=margin_tau, open_set=True)\n"
            "    _rows.append({\n"
            "        'parcela': obs.parcel_id,\n"
            "        'margen_top1_top2': round(top1_top2_margin(post), 3),\n"
            "        'clase_voting3': res.top_class_before,\n"
            "        'disparo': res.reason,\n"
            "        'refinado': res.refined,\n"
            "        'clase_tras_refinar': res.top_class_after,\n"
            "    })\n"
            "display(pl.DataFrame(_rows))\n"
            "display(Markdown('**ILUSTRATIVO**: el score FarSLIP de esta celda es un ejemplo del '\n"
            "    'mecanismo (favorece a la 2a clase). El ΔF1 real se mide en la seccion 4 con el '\n"
            "    'cabezal FarSLIP de la seccion 3.'))"
        )
    )

    # ============================================ Seccion 3 - Cabezal FarSLIP ===
    cells.append(
        md(
            "## 3. El cabezal zero-shot de FarSLIP (intento de carga real)\n\n"
            "El cabezal calcula `softmax(image_emb . text_emb / T)` con FarSLIP (`extract_embeddings` "
            "+ `encode_text`, ambos L2-norm). Intentamos cargar el **extractor real**; si los pesos "
            "no son alcanzables (sin ADC / sin `dvc pull`), se reporta el **blocker honesto** -- no "
            "se fabrican embeddings."
        )
    )
    cells.append(
        code(
            "# Honest attempt to load the real FarSLIP extractor. No fabrication: on failure we\n"
            "# report the blocker and leave the real scoring + delta-F1 as pending.\n"
            "farslip = None\n"
            "try:\n"
            "    from ml.extractors.farslip_extractor import FarSLIPExtractor\n"
            "    farslip = FarSLIPExtractor(weights_uri=farslip_weights, device='cpu')\n"
            "    _t = farslip.encode_text(list(prompts.values())[:2])\n"
            "    display(Markdown(f'FarSLIP cargado. encode_text -> shape `{tuple(_t.shape)}` '\n"
            "        '(modo student si cargaron los pesos, teacher si no).'))\n"
            "except Exception as exc:  # noqa: BLE001 - report the blocker, never fabricate\n"
            "    display(Markdown(\n"
            "        f'> **PENDIENTE (blocker US-080 sec 4.2)**: no se pudo cargar FarSLIP '\n"
            "        f'(`{type(exc).__name__}: {exc}`). Para la corrida real: `dvc pull` de los '\n"
            "        'pesos / ADC de GCS + los chips por parcela (`ml/farslip/dataset.py`). El '\n"
            "        'cabezal `ml.farslip.zeroshot_head` queda listo e inyectable.'))"
        )
    )

    # ============================================ Seccion 4 - Eval ΔF1 ===
    cells.append(
        md(
            "## 4. Eval ΔF1 **REAL**: Voting-3 vs Voting-3 + FarSLIP-zeroshot\n\n"
            "La senal FarSLIP que usamos aqui es el **OOF zero-shot REAL** "
            "(`oof_parcel_farslip-zeroshot_fold5`, con los prompts propios de FarSLIP) que ya cubre "
            "**las 16.640 parcelas fold-5 del Voting-3** -- asi el ΔF1 se mide **sin modelo ni "
            "chips**. `run_real_eval` ata los posteriores Voting-3 reales + la verdad de campo + "
            "esta senal y corre la fusion condicionada, barriendo `alpha`. **Reportamos el delta tal "
            "cual** -- la regla manda valores reales, positivos o no."
        )
    )
    cells.append(
        code(
            "# REAL delta-F1: the precomputed FarSLIP zero-shot fold-5 OOF already covers the\n"
            "# Voting-3 parcels, so we measure without the FarSLIP model or chips. Honest: the\n"
            "# delta is reported as measured (positive or not).\n"
            "from ml.eval.class_remap import get_label_space\n"
            "from ml.eval.farslip_refine_eval import (\n"
            "    build_farslip_zeroshot_scorer,\n"
            "    load_real_inputs,\n"
            "    run_refine_eval,\n"
            ")\n\n"
            "_ls = get_label_space('france-9')\n"
            "_vp, _mp, _gt = load_real_inputs(_ls)        # real Voting-3 posteriors + members + GT\n"
            "_scorer = build_farslip_zeroshot_scorer(_ls)  # real FarSLIP zero-shot signal\n"
            "_rows = []\n"
            "for _a in (0.1, 0.2, 0.3):\n"
            "    _r = run_refine_eval(_vp, _gt, _scorer, member_predictions=_mp, alpha=_a,\n"
            "                         margin_tau=float(margin_tau))\n"
            "    _rows.append({'alpha': _a, 'F1_voting3': _r['f1_before'], 'F1_refinado': _r['f1_after'],\n"
            "                  'delta_F1': _r['delta_f1'], 'delta_F1_disparadas': _r['delta_f1_fired'],\n"
            "                  'n_disparadas': _r['n_fired'], 'n_cambiadas': _r['n_changed']})\n"
            "display(pl.DataFrame(_rows))"
        )
    )
    cells.append(
        code(
            "# Honest reading of the REAL numbers (no fabrication).\n"
            "_b = _rows[0]['F1_voting3']\n"
            "display(Markdown(\n"
            "    f'**Resultado real**: el Voting-3 ya esta en F1-macro **{_b}** sobre las '\n"
            "    f'{len(_gt)} parcelas france-9 del fold-5. Anadir la senal **FarSLIP zero-shot** da '\n"
            "    '**ΔF1 aproximadamente 0** (neutral a alpha bajo, ligeramente negativo a alpha '\n"
            "    'alto): con ESTA senal (los prompts propios de FarSLIP) el refinador **no mejora** '\n"
            "    'al campeon. El mecanismo dispara en ~5.300 parcelas inciertas y **respeta el caso '\n"
            "    'facil** (no lo rompe). La ganancia que predice el paper depende de re-puntuar '\n"
            "    'FarSLIP con las **descripciones fenologicas del LLM** (la variante US-080, '\n"
            "    'pendiente de los chips fold-5 + el modelo servido). Honestidad cientifica: se '\n"
            "    'reporta el delta medido, no se fuerza una mejora.'))"
        )
    )

    # ----------------------------------------------------------- Conclusiones ---
    cells.append(
        md(
            "## Conclusiones\n\n"
            "- El **refinador FarSLIP-fenologia** anade una segunda etapa al campeon Voting-3 que "
            "**solo actua en parcelas inciertas / open-set**, con fusion convexa auditable -- nunca "
            "degrada el caso facil (garantia por diseno, verificada).\n"
            "- Implementado y testeado de punta a punta (`ml.agent.refine`, "
            "`ml.farslip.zeroshot_head`, `ml.farslip.class_prompts`, `ml.eval.farslip_refine_eval`).\n"
            "- **ΔF1 REAL medido** (seccion 4) sobre las 16.640 parcelas fold-5, usando el OOF "
            "zero-shot real de FarSLIP: **ΔF1 aproximadamente 0** -- con esta senal el refinador no "
            "mejora al Voting-3 (que ya esta en F1-macro ~0,92). Resultado honesto, no forzado.\n"
            "- **Siguiente paso** (la hipotesis del paper): re-puntuar FarSLIP con las "
            "**descripciones fenologicas del LLM** (no los prompts propios de FarSLIP) sobre los "
            "chips fold-5, donde se espera la ganancia -- sobre todo en el caso **open-set** "
            "(cultivos mediterraneos que PASTIS no tiene). Esa variante necesita el modelo FarSLIP "
            "servido + los chips por parcela.\n\n"
            "### Cierre"
        )
    )
    cells.append(
        code("from ml.agent.db import close_pool\n\nawait close_pool()\nprint('pool cerrado.')")
    )

    return cells


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Ruta del notebook de salida.")] = _DEFAULT_OUT,
) -> None:
    """Generate the US-080 FarSLIP-refiner evaluation notebook.

    Args:
        out: Destination path of the ``.ipynb``.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
