"""Compare each checkpoint's declared metric with its held-out fold-5 dump.

Both compared metrics are pixel-level macro F1. Parcel-level macro F1 is
reported separately and is never subtracted from either pixel metric.

The producer also proves that the checkpoint carrying the metric has the same
weights as the checkpoint loaded by the harness. This matters for U-Net,
AnySat and SegFormer, whose metric and harness weights live at different paths.

Usage:
    poetry run python scripts/run_us119_sanidad_miembros.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import f1_score

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = REPO_ROOT / "ml" / "eval" / "oof"
PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
OUT_DIR = REPO_ROOT / "reports" / "paper_micai" / "us119"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "segmentation"

# Difference above which a member is withheld from the inferential panel until
# its cause is identified and the held-out dump is regenerated.
THRESHOLD = 0.15

# Member -> (checkpoint that records the metric, dotted metric key).
METRIC_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "unet": ("unet-aaron/unet_ckpt.pt", "best.f1_macro"),
    "anysat": ("anysat-aaron/anysat_ckpt.pt", "best.f1_macro"),
    "utae": ("utae-isaac/best_model.pt", "val_f1"),
    "segformer": ("segformer-isaac/best_model.pt", "val_f1"),
    "deeplabv3plus": ("deeplab-18/best.pt", "best_metrics.f1_macro"),
    "tsvit-pheno": ("tsvit-pheno-v1/best.pt", "best_metrics.f1_macro"),
    "tsvit-pheno-fullm": ("tsvit-pheno-fullm-v1/best.pt", "best_metrics.f1_macro"),
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_file(path: Path) -> Path:
    """Resolve a checkpoint path to the file that actually carries its weights."""
    if path.is_dir():
        candidate = path / "model.safetensors"
        if not candidate.exists():
            raise FileNotFoundError(f"no existe {candidate}")
        return candidate
    return path


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load tensors from a raw, wrapped, or Hugging Face checkpoint."""
    source = _weights_file(path)
    loaded: object
    if source.suffix == ".safetensors":
        loaded = load_safetensors(str(source), device="cpu")
    else:
        loaded = torch.load(source, map_location="cpu", weights_only=False)

    if not isinstance(loaded, Mapping):
        raise TypeError(f"{source}: el checkpoint no es un mapping")
    for key in ("model_state", "model_state_dict", "state_dict"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping):
            loaded = candidate
            break

    tensors = {str(key): value for key, value in loaded.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError(f"{source}: no contiene tensores de pesos")
    return tensors


def _display_path(path: Path) -> str:
    """Return a repo-relative path when possible, otherwise the resolved path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _compare_weight_sources(metric_path: Path, harness_path: Path) -> dict[str, Any]:
    """Return executable evidence that metric and harness sources carry equal weights."""
    metric_file = _weights_file(metric_path)
    harness_file = _weights_file(harness_path)
    metric_state = _load_state_dict(metric_path)
    if metric_file.resolve() == harness_file.resolve():
        harness_state = metric_state
    else:
        harness_state = _load_state_dict(harness_path)
    common = metric_state.keys() & harness_state.keys()
    equal = sum(
        1
        for key in common
        if metric_state[key].shape == harness_state[key].shape
        and torch.equal(metric_state[key], harness_state[key])
    )
    same = (
        metric_state.keys() == harness_state.keys()
        and equal == len(metric_state)
        and equal == len(harness_state)
    )
    return {
        "checkpoint_metrica": _display_path(metric_path),
        "checkpoint_arnes": _display_path(harness_path),
        "sha256_checkpoint_metrica": _sha256(metric_file),
        "sha256_checkpoint_arnes": _sha256(harness_file),
        "tensores_checkpoint_metrica": len(metric_state),
        "tensores_checkpoint_arnes": len(harness_state),
        "tensores_identicos": equal,
        "pesos_identicos": same,
    }


def _fold_fields(value: object, prefix: str = "") -> dict[str, Any]:
    """Find fold-related fields recursively in checkpoint mappings and sequences."""
    found: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if "fold" in key.lower() and isinstance(child, (str, int, float, bool, list, tuple)):
                found[path] = list(child) if isinstance(child, tuple) else child
            if not isinstance(child, torch.Tensor):
                found.update(_fold_fields(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list, tuple)):
                found.update(_fold_fields(child, f"{prefix}[{index}]"))
    return found


def _declared_metric(path: Path, key: str) -> tuple[float, dict[str, Any]]:
    """Return a checkpoint metric and all recursively recorded fold metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    value: object = checkpoint
    for part in key.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
        if value is None:
            break
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: no declara una metrica numerica en {key}")
    folds = _fold_fields(checkpoint)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
    return float(value), {"epoch": epoch, "campos_fold": folds}


def _pixel_ground_truth(patch_ids: list[str]) -> np.ndarray:
    """Return flattened semantic18 ground truth for the requested patches."""
    from ml.data.pastis_seg_dataset import _build_semantic18_lut

    lut = _build_semantic18_lut(255)
    pieces = []
    for patch_id in patch_ids:
        target = np.load(PASTIS_ROOT / "ANNOTATIONS" / f"TARGET_{patch_id}.npy")[0]
        pieces.append(lut[np.clip(target.astype(np.int64), 0, 19)].ravel())
    return np.concatenate(pieces)


def _git_head() -> str:
    """Return the short Git revision used to produce the artifact."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    """Build the seven-member sanity table and fail on incomplete provenance."""
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.oof.inventario import cargar_inventario
    from ml.utils.parcel_reconcile import PROB_COLUMNS

    inventory = cargar_inventario()
    parcel_truth = pl.read_parquet(
        REPO_ROOT / "reports/paper_micai/fase1/parcel_gt_fold5.parquet"
    ).sort("canonical_parcel_id")
    rows: list[dict[str, Any]] = []

    for member, (relative_metric_path, metric_key) in METRIC_CHECKPOINTS.items():
        dense_path = OOF_DIR / f"oof_{member}_fold5.parquet"
        parcel_path = OOF_DIR / f"oof_parcel_{member}_fold5.parquet"
        inventory_state = inventory["ficheros"].get(parcel_path.name, {}).get("estado")
        if inventory_state != "canonical":
            raise ValueError(f"{member}: el posterior por parcela no es canonical")
        if not dense_path.exists() or not parcel_path.exists():
            raise FileNotFoundError(f"{member}: faltan el volcado denso o el de parcela")

        dense = pl.read_parquet(dense_path, columns=["patch_id", "pred"]).sort("patch_id")
        patch_ids = dense["patch_id"].to_list()
        prediction = np.concatenate(
            [np.asarray(item, dtype=np.int64) for item in dense["pred"].to_list()]
        )
        truth = _pixel_ground_truth(patch_ids)
        if prediction.shape != truth.shape:
            raise ValueError(f"{member}: prediccion y verdad no tienen la misma forma")
        valid = truth != 255
        pixel_f1 = float(
            f1_score(truth[valid], prediction[valid], average="macro", zero_division=0)
        )

        parcel = pl.read_parquet(parcel_path).sort("canonical_parcel_id")
        joined = parcel_truth.join(parcel, on="canonical_parcel_id", how="inner")
        if joined.height != parcel_truth.height:
            raise ValueError(
                f"{member}: el posterior por parcela no cubre toda la poblacion elegible"
            )
        probabilities = joined.select(PROB_COLUMNS).to_numpy()
        parcel_f1 = float(
            f1_score(
                joined["label"].to_numpy(),
                probabilities.argmax(axis=1),
                average="macro",
                zero_division=0,
            )
        )

        metric_path = CHECKPOINT_ROOT / relative_metric_path
        metric, record = _declared_metric(metric_path, metric_key)
        harness_path = CHECKPOINT_REGISTRY[member].path
        weight_evidence = _compare_weight_sources(metric_path, harness_path)
        if not weight_evidence["pesos_identicos"]:
            raise ValueError(f"{member}: la metrica y el arnes no usan los mismos pesos")

        delta = round(metric - pixel_f1, 4)
        exceeds = bool(abs(delta) > THRESHOLD)
        row = {
            "miembro": member,
            "estado_inventario": inventory_state,
            "clave_declarada": metric_key,
            "f1_macro_pixel_declarado": round(metric, 4),
            "f1_macro_pixel_fold5": round(pixel_f1, 4),
            "delta_pixel": delta,
            "supera_umbral": exceeds,
            "decision_panel": "excluir" if exceeds else "incluir",
            "f1_macro_parcela_fold5": round(parcel_f1, 4),
            "parcelas_fuera_poblacion_elegible": parcel.height - joined.height,
            "n_patches": len(patch_ids),
            "epoch_del_checkpoint": record["epoch"],
            "campos_fold_checkpoint": json.dumps(
                record["campos_fold"], ensure_ascii=False, sort_keys=True
            ),
            "declara_folds_de_entrenamiento": bool(record["campos_fold"]),
            "metricas_en_otro_fichero": (
                weight_evidence["checkpoint_metrica"] != weight_evidence["checkpoint_arnes"]
            ),
            **weight_evidence,
        }
        rows.append(row)
        logger.info("member_checked", **row)

    if {row["miembro"] for row in rows} != set(METRIC_CHECKPOINTS):
        raise RuntimeError("la tabla de sanidad no contiene exactamente los siete miembros")

    included = [row["miembro"] for row in rows if row["decision_panel"] == "incluir"]
    excluded = [row["miembro"] for row in rows if row["decision_panel"] == "excluir"]
    disproved = [row["miembro"] for row in rows if row["delta_pixel"] < 0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(OUT_DIR / "sanidad_miembros.csv")
    (OUT_DIR / "sanidad_miembros.json").write_text(
        json.dumps(
            {
                "para_que": (
                    "Comparar la metrica declarada por cada checkpoint con su volcado sobre el "
                    "fold 5 y probar que ambos proceden de los mismos pesos."
                ),
                "umbral": THRESHOLD,
                "las_dos_metricas_comparadas_son_por_pixel": True,
                "nota_parcela": (
                    "f1_macro_parcela_fold5 se publica aparte y no se compara con las de pixel."
                ),
                "miembros_incluidos": included,
                "miembros_excluidos": excluded,
                "premisa_desmentida_por": disproved,
                "miembros": rows,
                "code_version": _git_head(),
                "generado": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact_written", out=str(OUT_DIR), members=len(rows))


if __name__ == "__main__":
    main()
