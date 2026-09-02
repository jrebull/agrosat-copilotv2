"""Unit tests for the reproducible paper figures (US-070).

The figures are recomposed from real artifacts (CSV/JSON) or promoted from
existing PNGs. We verify on synthetic fixtures that a figure produced from a
source yields both SVG and PNG, and that a missing source returns ``None``
(blocked figure, never fabricated). Real-artifact smoke tests run when the
artifact exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=False)

from ml.report import paper_figures as pf

REPORTS_DIR = Path("reports")


def test_set_paper_style_is_idempotent() -> None:
    pf.set_paper_style()
    pf.set_paper_style()
    assert matplotlib.rcParams["savefig.dpi"] == 300
    assert matplotlib.rcParams["font.family"] == ["serif"]


def test_save_fig_svg_png_writes_both(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    pf.set_paper_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = pf.save_fig_svg_png(fig, "smoke", out_dir=tmp_path)
    assert out["svg"].exists() and out["svg"].suffix == ".svg"
    assert out["png"].exists() and out["png"].suffix == ".png"


def test_benchmark_barplot_from_fixture(tmp_path: Path) -> None:
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "model,miou,f1_macro,pixel_accuracy,fold,n_patches,status,model_kind,"
        "needs_resize,in_channels,cohen_kappa,balanced_acc\n"
        "m1,0.5,0.6,0.7,5,10,ok,m1,false,10,0.5,0.5\n"
        "m2,0.3,0.4,0.6,5,10,ok,m2,false,10,0.5,0.4\n",
        encoding="utf-8",
    )
    out = pf.fig_benchmark_barplot(csv, out_dir=tmp_path)
    assert out is not None
    assert out["png"].exists() and out["svg"].exists()


def test_farslip_sweep_from_fixture(tmp_path: Path) -> None:
    csv = tmp_path / "sweep.csv"
    csv.write_text(
        "n_classes,macro_f1,macro_iou,n_well_resolved,n_eval_parcels,best_ckpt\n"
        "4,0.70,0.55,4,1301,a\n6,0.45,0.31,2,1843,b\n",
        encoding="utf-8",
    )
    out = pf.fig_farslip_sweep_curve(csv, out_dir=tmp_path)
    assert out is not None and out["png"].exists()


def test_transfer_catalonia_from_fixture(tmp_path: Path) -> None:
    js = tmp_path / "transfer.json"
    js.write_text(
        json.dumps(
            {
                "zero_shot_metrics": {
                    "miou": 0.0,
                    "f1_macro": 0.0,
                    "pixel_accuracy": 0.0,
                },
                "few_shot_metrics": {
                    "miou": 0.2468,
                    "f1_macro": 0.3005,
                    "pixel_accuracy": 0.9179,
                },
            }
        ),
        encoding="utf-8",
    )
    out = pf.fig_transfer_catalonia(js, out_dir=tmp_path)
    assert out is not None and out["svg"].exists()


def test_llm_benchmark_barplot_from_fixture(tmp_path: Path) -> None:
    js = tmp_path / "eval.json"
    js.write_text(
        json.dumps(
            {
                "gemini": {
                    "tool_calling": {
                        "tool_selection_accuracy": {"mean": 0.55},
                        "arg_match_accuracy": {"mean": 0.95},
                    },
                    "grounded_crop": {
                        "crop_match_accuracy": {"mean": 0.92},
                        "routing_accuracy": {"mean": 1.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    out = pf.fig_llm_benchmark_barplot(js, out_dir=tmp_path)
    assert out is not None and out["png"].exists()


def test_stem_for_lang_suffix() -> None:
    assert pf.stem_for_lang("fig", "en") == "fig"
    assert pf.stem_for_lang("fig", "es") == "fig_es"


def test_save_fig_svg_png_es_suffix(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    pf.set_paper_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = pf.save_fig_svg_png(fig, "smoke", out_dir=tmp_path, lang="es")
    assert out["svg"].name == "smoke_es.svg"
    assert out["png"].name == "smoke_es.png"
    assert out["svg"].exists() and out["png"].exists()


def test_benchmark_barplot_bilingual_distinct_titles(tmp_path: Path) -> None:
    """Both languages emit files with the right suffix and different titles."""
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "model,miou,f1_macro,pixel_accuracy,fold,n_patches,status,model_kind,"
        "needs_resize,in_channels,cohen_kappa,balanced_acc\n"
        "m1,0.5,0.6,0.7,5,10,ok,m1,false,10,0.5,0.5\n",
        encoding="utf-8",
    )
    en = pf.fig_benchmark_barplot(csv, out_dir=tmp_path, lang="en")
    es = pf.fig_benchmark_barplot(csv, out_dir=tmp_path, lang="es")
    assert en is not None and es is not None
    assert en["png"].name == "benchmark_barplot_fold5.png"
    assert es["png"].name == "benchmark_barplot_fold5_es.png"
    assert en["png"].exists() and es["png"].exists()
    # English/Spanish titles must actually differ (real translation, not a copy).
    assert pf._STR_BENCHMARK["en"]["title"] != pf._STR_BENCHMARK["es"]["title"]


def test_build_all_figures_emits_both_languages(tmp_path: Path) -> None:
    """build_all_figures produces base (EN) and _es keys for real figures."""
    results = pf.build_all_figures(out_dir=tmp_path)
    # farslip_band_ablation source exists in the repo -> both variants present.
    if results.get("farslip_band_ablation") is not None:
        assert results.get("farslip_band_ablation_es") is not None
        base = results["farslip_band_ablation"]
        es = results["farslip_band_ablation_es"]
        assert base["png"].name == "farslip_band_ablation.png"
        assert es["png"].name == "farslip_band_ablation_es.png"


def test_missing_source_returns_none(tmp_path: Path) -> None:
    assert pf.fig_benchmark_barplot(tmp_path / "nope.csv", out_dir=tmp_path) is None
    assert pf.promote_png(tmp_path / "nope.png", "x", out_dir=tmp_path) is None


def test_promote_png_from_fixture(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    src = tmp_path / "src.png"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    fig.savefig(src)
    plt.close(fig)
    out = pf.promote_png(src, "promoted", out_dir=tmp_path)
    assert out is not None and out["svg"].exists() and out["png"].exists()


def test_conversational_examples_from_fixture(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    rec = {"benchmark": "GeoAnalystBench", "task": "t", "prediction": "p"}
    (traces / "trace_gemini_x.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = pf.export_conversational_examples(out_dir=tmp_path, traces_dir=traces)
    assert out is not None and out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "pendiente" in payload["note_it"].lower()  # IT not fabricated


# --------------------------------------------------------------------------- #
# Real-artifact smoke tests.
# --------------------------------------------------------------------------- #
def test_real_benchmark_barplot_if_present(tmp_path: Path) -> None:
    src = REPORTS_DIR / "segmentation" / "metrics" / "model_comparison_fold5.csv"
    if not src.exists():
        pytest.skip("real fold-5 metrics missing")
    out = pf.fig_benchmark_barplot(src, out_dir=tmp_path)
    assert out is not None and out["png"].exists()
