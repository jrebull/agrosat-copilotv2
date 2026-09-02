"""US-023-preview P4 — real pheno_text ablation with Gemini Flash 3.5.

Production run (not smoke). Builds a stratified subset of >=1000 parcels
balanced by class, generates descriptions with Gemini Flash 3.5
(temperature=0, per-parcel cache), encodes them with sentence-transformers
all-MiniLM-L6-v2 (384 dim) and persists to data/features/phenology_text_pastis.parquet.

Then it runs the ablation with XGBoost spatial CV 5-fold over 3 sets:
- full (185 base features without geom_*)
- with_pheno_text (185 + 384)
- pheno_text_only (384)

Persists results to reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet,
logs the Gemini cost and creates an MLflow run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

load_dotenv(".env.local")
os.environ.setdefault("AGROSAT_LLM_PROVIDER", "google-genai")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.eval.feature_ablation import run_feature_ablation  # noqa: E402
from ml.features.phenology_description import (  # noqa: E402
    DEFAULT_TEXT_EMBED_DIM,
    encode_descriptions,
    generate_phenology_description,
)
from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe  # noqa: E402
from ml.train.phenology_models import _reconstruct_curve  # noqa: E402

# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------

FEATURES_PATH = REPO / "data/test_fixtures/feature_selection_parcels_subset.parquet"
OUT_PARQUET = REPO / "data/features/phenology_text_pastis.parquet"
ABLATION_OUT = REPO / "reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet"
CACHE_DIR = REPO / "data/cache/phenology_descriptions"
SEED = 42
TARGET_PER_CLASS = 60  # 60 x 18 = 1080 parcels (>= 1000 AC-P4-2)
MODEL_NAME = "gemini-3.5-flash"
GEMINI_INPUT_USD_PER_1M = 0.30  # Gemini 2.5/3.5 Flash input rate
GEMINI_OUTPUT_USD_PER_1M = 2.50  # Gemini 2.5/3.5 Flash output rate


def log(msg: str) -> None:
    print(f"[us023-p4] {msg}", flush=True)


def main() -> None:
    t0 = time.time()

    # Validate API key.
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key.startswith("AIzaSy"):
        raise SystemExit("GEMINI_API_KEY not loaded from .env.local")
    log(f"GEMINI_API_KEY ok (len={len(key)})")

    # 1) Load dataset and build balanced subset.
    log("cargando dataset baseline...")
    df_raw = _load_baseline_dataset(FEATURES_PATH)
    df = _prepare_dataframe(df_raw)
    log(f"dataset full: {df.shape}")

    rng = np.random.default_rng(SEED)
    parts: list[pl.DataFrame] = []
    for class_id, sub in df.group_by("class_id"):
        n = min(TARGET_PER_CLASS, sub.height)
        idx = rng.choice(sub.height, size=n, replace=False)
        parts.append(sub[idx.tolist()])
    subset = pl.concat(parts).sort("parcel_id")
    log(f"subset balanceado: {subset.shape} ({subset['class_id'].n_unique()} clases)")
    if subset.height < 1000:
        raise SystemExit(f"AC-P4-2 violated: subset {subset.height} < 1000")

    # 2) Reconstruct NDVI curves from FFT.
    log("reconstruyendo curvas NDVI desde FFT...")
    curves = _reconstruct_curve(subset, index_name="NDVI", sequence_length=72)
    log(f"curvas: shape {curves.shape}")

    # 3) Generate descriptions with Gemini Flash 3.5.
    parcel_ids = subset["parcel_id"].to_list()
    descriptions: list[str] = []
    n_total = len(parcel_ids)
    n_cache_hits = 0
    n_api_calls = 0
    api_t0 = time.time()
    chars_in = 0  # proxy of tokens_in for prompt
    chars_out = 0  # proxy of tokens_out for description

    for i, (pid, curve) in enumerate(zip(parcel_ids, curves, strict=True)):
        # Detect cache hit before calling (same internal helper).
        from ml.features.phenology_description import _hash_curve

        cache_key = _hash_curve(pid, curve.astype(np.float32), MODEL_NAME)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        was_cached = cache_file.exists()

        desc = generate_phenology_description(
            curve.astype(np.float32),
            parcel_id=pid,
            model=MODEL_NAME,
            temperature=0.0,
            cache_dir=CACHE_DIR,
        )
        descriptions.append(desc)

        if was_cached:
            n_cache_hits += 1
        else:
            n_api_calls += 1
            # Rough approximation tokens = chars/4.
            chars_out += len(desc)
            # The prompt includes the template ~700 chars + curve serialization
            # ~24 points x ~18 chars = ~430 chars. Total ~1200 chars.
            chars_in += 1200

        if (i + 1) % 50 == 0:
            dt = time.time() - api_t0
            log(
                f"  pheno {i + 1}/{n_total} (api={n_api_calls}, cache={n_cache_hits}, dt={dt:.0f}s)"
            )

    api_dt = time.time() - api_t0
    log(f"descripciones listas: api={n_api_calls}, cache={n_cache_hits}, dt={api_dt:.0f}s")

    # Cost estimation: tokens ~ chars/4.
    tokens_in = chars_in // 4
    tokens_out = chars_out // 4
    cost_in = tokens_in / 1_000_000 * GEMINI_INPUT_USD_PER_1M
    cost_out = tokens_out / 1_000_000 * GEMINI_OUTPUT_USD_PER_1M
    cost_total = cost_in + cost_out
    log(
        f"costo Gemini estimado: in={tokens_in} tok (${cost_in:.4f}), out={tokens_out} tok (${cost_out:.4f}), total=${cost_total:.4f}"
    )
    if cost_total > 5.0:
        raise SystemExit(f"AC-P4-4 violated: ${cost_total:.4f} > $5.0")

    # 4) Encode with sentence-transformers.
    log("encoding con sentence-transformers all-MiniLM-L6-v2...")
    embeddings = encode_descriptions(descriptions, encoder="sentence-transformers")
    log(f"embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (n_total, DEFAULT_TEXT_EMBED_DIM)

    # 5) Persist extended parquet.
    log(f"persistiendo {OUT_PARQUET}...")
    block: dict[str, list] = {
        "parcel_id": parcel_ids,
        "year": subset["year"].to_list(),
    }
    text_cols = [f"pheno_text_{j:03d}" for j in range(embeddings.shape[1])]
    for j, name in enumerate(text_cols):
        block[name] = embeddings[:, j].tolist()
    schema: dict = {
        "parcel_id": subset.schema["parcel_id"],
        "year": subset.schema["year"],
    }
    for name in text_cols:
        schema[name] = pl.Float32()
    text_df = pl.DataFrame(block, schema=schema)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    text_df.write_parquet(OUT_PARQUET)
    log(f"parquet shape: {text_df.shape}")
    assert text_df.shape == (n_total, 1 + 1 + DEFAULT_TEXT_EMBED_DIM)

    # 6) Build fused frame for ablation: subset + pheno_text cols.
    log("fusionando para ablation...")
    fused = subset.join(text_df.drop("year"), on="parcel_id", how="inner")
    log(f"fused shape: {fused.shape}")

    # 7) Build feature_sets for the 3 sets.
    cols = fused.columns
    drop_meta = {
        "parcel_id",
        "year",
        "patch_id",
        "instance_id",
        "class_id",
        "class_name",
        "fold",
        "n_pixels",
    }
    geom_cols = {c for c in cols if c.startswith("geom_")}
    pheno_text_cols = tuple(c for c in cols if c.startswith("pheno_text_"))
    base_full_cols = tuple(
        c
        for c in cols
        if c not in drop_meta and c not in geom_cols and not c.startswith("pheno_text_")
    )
    feature_sets = {
        "full": base_full_cols,
        "with_pheno_text": base_full_cols + pheno_text_cols,
        "pheno_text_only": pheno_text_cols,
    }
    log(
        f"feature_sets: full={len(base_full_cols)}, "
        f"with_pheno_text={len(base_full_cols) + len(pheno_text_cols)}, "
        f"pheno_text_only={len(pheno_text_cols)}"
    )

    # 8) Run ablation with spatial CV 5-fold.
    log("corriendo run_feature_ablation (XGB, spatial CV 5-fold)...")
    results = run_feature_ablation(
        df=fused,
        feature_sets=feature_sets,
        models=("xgb",),
        seed=SEED,
        k_folds=5,
        buffer_km=1.0,
    )

    # Convert list of FeatureAblationResult to Polars DataFrame.
    result_rows = [
        {
            "feature_set": r.feature_set,
            "model": r.model_kind,
            "n_features": int(r.n_features),
            "f1_macro": float(r.f1_macro),
            "f1_weighted": float(r.f1_weighted),
            "miou": float(r.miou),
            "delta_vs_full": (float(r.delta_vs_full) if not np.isnan(r.delta_vs_full) else None),
        }
        for r in results
    ]
    result_df = pl.DataFrame(result_rows)
    log("ablation result:")
    print(result_df)

    ABLATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(ABLATION_OUT)
    log(f"ablation persistido -> {ABLATION_OUT}")

    # 9) Compute deltas.
    rows_by_set = {r["feature_set"]: r for r in result_rows}
    f1_full = float(rows_by_set["full"]["f1_macro"])
    f1_with = float(rows_by_set["with_pheno_text"]["f1_macro"])
    f1_only = float(rows_by_set["pheno_text_only"]["f1_macro"])
    delta_with = f1_with - f1_full
    delta_only = f1_only - f1_full
    log(f"f1_full={f1_full:.6f}  f1_with={f1_with:.6f}  delta={delta_with:+.6f}")
    log(f"f1_pheno_only={f1_only:.6f}  delta_only={delta_only:+.6f}")

    if delta_with >= 0.01:
        decision = "PROMOVER al baseline (delta >= +0.01)"
    elif delta_with >= -0.01:
        decision = "BASE LEARNER stacking EPIC 6 (delta en [-0.01, +0.01])"
    else:
        decision = "DEUDA US-024 (delta < -0.01, escalar a full 85951)"
    log(f"decision: {decision}")

    # 10) Persist run summary JSON for report.
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "us": "US-023-preview",
        "bloque": "P4",
        "n_parcels": n_total,
        "n_classes_balanced": int(subset["class_id"].n_unique()),
        "target_per_class": TARGET_PER_CLASS,
        "gemini_model": MODEL_NAME,
        "gemini_n_requests": n_api_calls,
        "gemini_n_cache_hits": n_cache_hits,
        "gemini_tokens_in_est": tokens_in,
        "gemini_tokens_out_est": tokens_out,
        "gemini_cost_usd_est": round(cost_total, 6),
        "gemini_wall_seconds": round(api_dt, 1),
        "f1_macro_full": f1_full,
        "f1_macro_with_pheno_text": f1_with,
        "f1_macro_pheno_text_only": f1_only,
        "delta_pheno_text_vs_full": round(delta_with, 6),
        "delta_pheno_text_only_vs_full": round(delta_only, 6),
        "decision": decision,
        "git_sha": git_sha,
        "ablation_out": str(ABLATION_OUT.relative_to(REPO)),
        "pheno_text_parquet": str(OUT_PARQUET.relative_to(REPO)),
        "wall_seconds_total": round(time.time() - t0, 1),
    }
    summary_path = REPO / "reports/baseline/feature_ablation/us023_p4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"summary -> {summary_path.relative_to(REPO)}")

    # 11) Optional MLflow.
    try:
        import mlflow

        tracking = os.environ.get("MLFLOW_TRACKING_URI", "")
        if tracking:
            mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment("baseline-pheno-text-ablation")
        with mlflow.start_run(run_name="baseline-pheno-text-ablation-v1") as run:
            mlflow.set_tag("us", "US-023-preview")
            mlflow.set_tag("bloque", "P4")
            mlflow.set_tag("code_version", git_sha)
            mlflow.set_tag("data_version", "phenology-text-italy-v1")
            mlflow.log_params(
                {
                    "n_parcels": n_total,
                    "n_classes_balanced": summary["n_classes_balanced"],
                    "gemini_model": MODEL_NAME,
                    "gemini_n_requests": n_api_calls,
                    "gemini_cost_usd": round(cost_total, 6),
                    "target_per_class": TARGET_PER_CLASS,
                    "k_folds": 5,
                    "buffer_km": 1.0,
                }
            )
            mlflow.log_metrics(
                {
                    "f1_macro_full": f1_full,
                    "f1_macro_with_pheno_text": f1_with,
                    "f1_macro_pheno_text_only": f1_only,
                    "delta_pheno_text_vs_full": delta_with,
                    "delta_pheno_text_only_vs_full": delta_only,
                }
            )
            mlflow.log_artifact(str(ABLATION_OUT))
            mlflow.log_artifact(str(summary_path))
            log(f"mlflow run id: {run.info.run_id}")
            summary["mlflow_run_id"] = run.info.run_id
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    except Exception as exc:  # mlflow optional
        log(f"mlflow skip (motivo: {exc!r})")

    log(f"P4 done en {summary['wall_seconds_total']}s")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
