"""Run only the P8 block of baseline v2 (US-023-preview).

Equivalent to section 9 of the notebook ``04_baseline.ipynb`` but without the
heavy prior sections (AlphaEarth vs S2 raw comparison, learning curves,
SHAP). Trains the 3 canonical A3 models (XGBoost + TempCNN +
InceptionTime) with 5-fold spatial CV buffer 1 km over the post-ablation
winning set, logs 3 MLflow runs and emits the artifacts required by
AC-P8-1..AC-P8-8.

Usage:
    poetry run python scripts/run_baseline_v2_standalone.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import polars as pl

from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe, train_one_model
from ml.train.phenology_models import train_temporal_model
from ml.utils.git_meta import dvc_data_version, git_sha


FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"
V2_OUTPUT_DIR = Path("reports/baseline/model_comparison_v2")
V2_FEATURE_ABLATION_CSV = Path("reports/baseline/feature_ablation.csv")
PAPER_TABLES_DIR = Path("paper/tables/us-023-preview")
PAPER_FIGURES_DIR = Path("paper/figures/us-023-preview")

V2_K_FOLDS = 5
V2_BUFFER_KM = 1.0
V2_SEED = 42
V2_TEMPORAL_EPOCHS = 200
V2_TEMPORAL_BATCH_SIZE = 128
V2_DEVICE = "auto"  # autodetect CUDA


def _resolve_winner_set(ablation_csv: Path) -> str:
    """Decide the post-ablation winning set.

    If the report does not exist or all its F1-macro are NaN, returns the
    fallback documented in D-9: ``"no_geom"`` (discards the 3 ``geom_*``
    cols due to spatial leakage).
    """
    if not ablation_csv.exists():
        print(f"[winner] reporte {ablation_csv} no existe; fallback 'no_geom'")
        return "no_geom"
    abl = pl.read_csv(ablation_csv)
    abl_sorted = abl.filter(pl.col("f1_macro").is_finite()).sort("f1_macro", descending=True)
    if abl_sorted.height == 0:
        print(f"[winner] reporte sin F1-macro finito; fallback 'no_geom'")
        return "no_geom"
    winner = abl_sorted.row(0, named=True)["feature_set"]
    print(f"[winner] conjunto ganador post-ablation: {winner}")
    return winner


def _filter_dataset_by_winner(df: pl.DataFrame, winner_set: str) -> pl.DataFrame:
    """Filter the dataset according to the winning set.

    For ``no_geom`` simply discards the ``geom_*`` cols. Other future sets
    (with FarSLIP, pheno_text, spectral signature) would require explicit
    logic here.
    """
    if winner_set == "no_geom":
        geom_cols = [c for c in df.columns if c.startswith("geom_")]
        if geom_cols:
            df = df.drop(geom_cols)
            print(f"[filter] descartadas {len(geom_cols)} cols geom_*")
    return df


def _log_run(run_name: str, metrics: dict, tags: dict, params: dict) -> str:
    """Open an MLflow run and log params/metrics/tags. Return run_id."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        for k, v in metrics.items():
            if v == v:  # NaN-safe (NaN != NaN)
                mlflow.log_metric(k, float(v))
        return mlflow.active_run().info.run_id


def main() -> int:
    t0 = time.time()
    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== US-023-preview P8: baseline v2 standalone ===")
    print(f"Dataset: {FEATURES_PATH}")
    print(f"Spatial CV: k={V2_K_FOLDS}, buffer={V2_BUFFER_KM} km")
    print(
        f"Temporal: epochs={V2_TEMPORAL_EPOCHS}, batch={V2_TEMPORAL_BATCH_SIZE}, device={V2_DEVICE}"
    )

    # --- 1. Load + filter ------------------------------------------------
    df_raw = _load_baseline_dataset(FEATURES_PATH)
    df = _prepare_dataframe(df_raw)
    print(f"[data] post-prepare shape: {df.shape}")

    winner_set = _resolve_winner_set(V2_FEATURE_ABLATION_CSV)
    df = _filter_dataset_by_winner(df, winner_set)
    print(f"[data] post-filter shape: {df.shape}  | winner_set={winner_set}")
    n_parcels = df.height

    # --- 2. MLflow setup -------------------------------------------------
    data_ver = dvc_data_version(FEATURES_PATH)
    code_ver = git_sha(short=True)
    tags = {
        "data_version": data_ver,
        "code_version": code_ver,
        "us": "US-023-preview",
        "bloque": "P8",
        "feature_set": winner_set,
    }
    mlflow.set_experiment("baseline-v2-us-023-preview")
    device_str = "cuda" if V2_DEVICE == "auto" else V2_DEVICE
    source_str = (
        f"US-023-preview P8 baseline v2 (CUDA RTX 4070, spatial CV "
        f"{V2_K_FOLDS}-fold buffer {V2_BUFFER_KM} km, feature_set={winner_set})"
    )
    common_params = {
        "k_folds": V2_K_FOLDS,
        "buffer_km": V2_BUFFER_KM,
        "seed": V2_SEED,
        "feature_set": winner_set,
        "n_parcels": n_parcels,
        "device": device_str,
    }

    v2_results: dict[str, dict] = {}

    # --- 3. XGBoost ------------------------------------------------------
    print("\n[XGB] entrenando spatial CV 5-fold sobre full dataset...")
    t_xgb = time.time()
    xgb_res = train_one_model(
        df,
        model="xgb",
        k_folds=V2_K_FOLDS,
        buffer_km=V2_BUFFER_KM,
        random_state=V2_SEED,
    )
    xgb_time = time.time() - t_xgb
    xgb_metrics = {
        "f1_macro": float(xgb_res.metrics["f1_macro"]),
        "f1_weighted": float(xgb_res.metrics["f1_weighted"]),
        "miou": float(xgb_res.metrics["miou"]),
        "accuracy": float(xgb_res.metrics.get("accuracy", 0.0)),
        "kappa": float(xgb_res.metrics.get("cohen_kappa", 0.0)),
        "train_time_s": xgb_time,
    }
    xgb_run_id = _log_run(
        "baseline-v2-xgb",
        xgb_metrics,
        tags,
        {**common_params, "model_kind": "xgb"},
    )
    v2_results["xgboost"] = {
        **xgb_metrics,
        "mlflow_run_id": xgb_run_id,
        "source": source_str,
    }
    print(f"[XGB] F1-macro={xgb_metrics['f1_macro']:.4f}  t={xgb_time:.1f}s  run={xgb_run_id[:8]}")

    # --- 4. TempCNN + InceptionTime --------------------------------------
    for kind in ("tempcnn", "inceptiontime"):
        print(
            f"\n[{kind}] entrenando {V2_TEMPORAL_EPOCHS} epocas "
            f"batch={V2_TEMPORAL_BATCH_SIZE} en {device_str}..."
        )
        t_tk = time.time()
        temp_res = train_temporal_model(
            df=df,
            model_kind=kind,
            n_epochs=V2_TEMPORAL_EPOCHS,
            batch_size=V2_TEMPORAL_BATCH_SIZE,
            seed=V2_SEED,
            device=V2_DEVICE,
            k_folds=V2_K_FOLDS,
            buffer_km=V2_BUFFER_KM,
            dropout=0.2,
            use_class_weights=True,
            use_weighted_sampler=True,
            use_lr_scheduler=True,
            warmup_epochs=5,
            early_stopping_patience=20,
            val_fraction=0.15,
        )
        tk_time = time.time() - t_tk
        tk_metrics = {
            "f1_macro": float(temp_res.f1_macro),
            "f1_weighted": float(temp_res.f1_weighted),
            "miou": float(temp_res.miou),
            "accuracy": float("nan"),  # train_temporal_model does not expose accuracy
            "kappa": float(temp_res.cohen_kappa),
            "train_time_s": tk_time,
        }
        tk_run_id = _log_run(
            f"baseline-v2-{kind}",
            tk_metrics,
            tags,
            {
                **common_params,
                "model_kind": kind,
                "n_epochs": V2_TEMPORAL_EPOCHS,
                "batch_size": V2_TEMPORAL_BATCH_SIZE,
            },
        )
        v2_results[kind] = {
            **tk_metrics,
            "mlflow_run_id": tk_run_id,
            "source": source_str,
        }
        print(
            f"[{kind}] F1-macro={tk_metrics['f1_macro']:.4f}  t={tk_time:.1f}s  run={tk_run_id[:8]}"
        )

    wall = time.time() - t0
    print(f"\nWall clock total v2 standalone: {wall:.1f}s (target <= 5400s)")

    # --- 5. Persistence --------------------------------------------------
    rows = [{"model": k, **v} for k, v in v2_results.items()]
    table = pl.DataFrame(rows).sort("f1_macro", descending=True)
    table.write_parquet(V2_OUTPUT_DIR / "model_comparison_v2.parquet")
    table.write_csv(V2_OUTPUT_DIR / "model_comparison_v2.csv")
    print("\n[persist] tabla v2 escrita:")
    print(table)

    # LaTeX (without mlflow_run_id/source cols for readability).
    latex_cols = [c for c in table.columns if c not in ("mlflow_run_id", "source")]
    latex = (
        table.select(latex_cols)
        .to_pandas()
        .to_latex(
            index=False,
            float_format="%.4f",
            caption="Baseline v2 - 3 modelos canonicos US-023-preview P8",
            label="tab:baseline_v2_comparison",
        )
    )
    (PAPER_TABLES_DIR / "baseline_v2_comparison.tex").write_text(latex, encoding="utf-8")
    print(f"[persist] LaTeX en {PAPER_TABLES_DIR / 'baseline_v2_comparison.tex'}")

    # --- 6. Decision D-10 + plot ----------------------------------------
    winner_v2 = table.row(0, named=True)["model"]
    print(f"\nModelo ganador v2 (D-10): {winner_v2}")
    print(f"F1-macro = {table.row(0, named=True)['f1_macro']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    models = table["model"].to_list()
    f1s = table["f1_macro"].to_list()
    bars = ax.bar(models, f1s, color=["#117733", "#4477aa", "#cc6677"])
    ax.bar_label(bars, fmt="%.4f", fontsize=10)
    ax.set_ylabel("F1-macro (spatial CV 5-fold)")
    ax.set_title(f"Baseline v2 - modelo ganador: {winner_v2}")
    ax.set_ylim(0, max(f1s) * 1.2 if max(f1s) > 0 else 0.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES_DIR / "model_comparison_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[persist] plot en {PAPER_FIGURES_DIR / 'model_comparison_v2.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
