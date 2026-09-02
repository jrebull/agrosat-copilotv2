"""Retry MLflow logging with file backend (local) after main P4 run."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import mlflow  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "reports/baseline/feature_ablation/us023_p4_summary.json"

s = json.loads(SUMMARY.read_text(encoding="utf-8"))

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("baseline-pheno-text-ablation")
with mlflow.start_run(run_name="baseline-pheno-text-ablation-v1") as run:
    mlflow.set_tag("us", "US-023-preview")
    mlflow.set_tag("bloque", "P4")
    mlflow.set_tag("code_version", s["git_sha"])
    mlflow.set_tag("data_version", "phenology-text-italy-v1")
    mlflow.log_params(
        {
            "n_parcels": s["n_parcels"],
            "n_classes_balanced": s["n_classes_balanced"],
            "gemini_model": s["gemini_model"],
            "gemini_n_requests": s["gemini_n_requests"],
            "gemini_cost_usd": s["gemini_cost_usd_est"],
            "target_per_class": s["target_per_class"],
            "k_folds": 5,
            "buffer_km": 1.0,
        }
    )
    mlflow.log_metrics(
        {
            "f1_macro_full": s["f1_macro_full"],
            "f1_macro_with_pheno_text": s["f1_macro_with_pheno_text"],
            "f1_macro_pheno_text_only": s["f1_macro_pheno_text_only"],
            "delta_pheno_text_vs_full": s["delta_pheno_text_vs_full"],
            "delta_pheno_text_only_vs_full": s["delta_pheno_text_only_vs_full"],
        }
    )
    mlflow.log_artifact(
        str(REPO / "reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet")
    )
    mlflow.log_artifact(str(SUMMARY))
    print(f"mlflow_run_id: {run.info.run_id}")
    s["mlflow_run_id"] = run.info.run_id
    s["mlflow_tracking_uri"] = "file:./mlruns"
    SUMMARY.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary updated.")
