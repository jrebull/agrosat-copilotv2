"""Canonical selection of the winning feature set for downstream EPIC 5.

Closes the US-023-preview baseline loop: once the ablations have been run
(FarSLIP, pheno_text, spectral_signature, geom_only), this function decides
which blocks are promoted and persists the final parquet
`data/features/features_fused_winning_pastis.parquet` consumed by the dense
models of EPIC 5 (U-Net, U-TAE, TSViT, Swin-UNETR) and the EPIC 6 ensembles.
The content is French PASTIS-R (parcel_id format `10000_1`), hence the
canonical `_pastis` naming.

Decision rule (aligned with the US-023-preview plan):

- Mandatory base block: `phenology_only` (8 cols) + `indices_stats` (85 cols
  of the US-018 subset: NDVI/NDWI/EVI x stats + FFT NDVI).
- AlphaEarth block (`ae_*`): always include if present (Foundation Model free
  via GEE).
- ERA5 block (`era5_*`) + SRTM (`srtm_*`): include unless the ablation shows
  that they contribute negatively.
- `geom_*`: discard by default (US-022-b decision: spatial leakage).
- Optional blocks (`farslip`, `pheno_text`, `spectral_signature`): include only
  if the ablation promotes them (delta >= +0.005 vs `full`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "WinningFeatureSet",
    "persist_winning_features",
    "select_winning_features",
]


@dataclass(frozen=True)
class WinningFeatureSet:
    """Result of the winning set selection.

    Attributes:
        name: Short label (e.g. `"phenology+ae+farslip"`).
        feature_cols: Ordered tuple of selected columns.
        decisions: Mapping `{block: bool}` with the promote/discard decisions.
        rationale: Accessible-language text explaining the selection.
        delta_vs_full: F1-macro delta reported by the ablation for each promoted
            optional block.
    """

    name: str
    feature_cols: tuple[str, ...]
    decisions: dict[str, bool]
    rationale: str
    delta_vs_full: dict[str, float]


def select_winning_features(
    ablation_table: pl.DataFrame,
    available_cols: Sequence[str],
    *,
    promote_threshold: float = 0.005,
    discard_geom: bool = True,
) -> WinningFeatureSet:
    """Select the winning set based on the ablation table.

    Args:
        ablation_table: DataFrame with columns `feature_set`, `model`,
            `f1_macro`, `delta_vs_full`. Typically the output of
            :func:`ml.utils.baseline_notebook_helpers.run_ablation_and_persist`.
        available_cols: List of columns available in the fused dataset.
        promote_threshold: Minimum delta to promote an optional block.
        discard_geom: If True, discards `geom_*` due to spatial leakage.

    Returns:
        `WinningFeatureSet` with the decision and the column list.
    """
    deltas = _read_deltas(ablation_table)
    decisions: dict[str, bool] = {
        "geom": not discard_geom,
        "farslip": deltas.get("with_farslip", float("nan")) >= promote_threshold,
        "pheno_text": deltas.get("with_pheno_text", float("nan")) >= promote_threshold,
        "spectral_signature": (
            deltas.get("with_spectral_signature", float("nan")) >= promote_threshold
        ),
    }

    base_cols = _base_block_cols(available_cols, include_geom=not discard_geom)
    optional_cols: list[str] = []
    for block, promoted in decisions.items():
        if not promoted:
            continue
        optional_cols.extend(_optional_block_cols(available_cols, block))
    selected = tuple(sorted(set(base_cols + optional_cols)))

    promoted_blocks = [b for b, ok in decisions.items() if ok and b != "geom"]
    name_parts = ["phenology", "ae", "indices"]
    name_parts.extend(promoted_blocks)
    name = "+".join(name_parts)

    rationale_lines = [
        f"Conjunto ganador: `{name}` con {len(selected)} columnas.",
        ("Bloque base: AlphaEarth, indices espectrales x stats, fenologia y FFT NDVI."),
    ]
    if discard_geom:
        rationale_lines.append("`geom_*` descartado (leakage espacial confirmado en US-022-b).")
    for block, promoted in decisions.items():
        if block == "geom":
            continue
        if promoted:
            rationale_lines.append(
                f"`{block}` promovido (delta={deltas.get(f'with_{block}', float('nan')):+.4f})."
            )
        else:
            rationale_lines.append(
                f"`{block}` descartado (delta={deltas.get(f'with_{block}', float('nan')):+.4f} "
                f"< {promote_threshold:+.3f})."
            )

    rationale = "\n".join(rationale_lines)
    logger.info(
        "winning_features_selected",
        name=name,
        n_cols=len(selected),
        decisions=decisions,
    )
    return WinningFeatureSet(
        name=name,
        feature_cols=selected,
        decisions=decisions,
        rationale=rationale,
        delta_vs_full={k: v for k, v in deltas.items() if k.startswith("with_")},
    )


def persist_winning_features(
    winning: WinningFeatureSet,
    fused_df: pl.DataFrame,
    *,
    output_path: Path | str = Path("data/features/features_fused_winning_pastis.parquet"),
    overwrite: bool = False,
) -> Path:
    """Persist the subset of winning features from the fused dataset.

    Keeps the metadata columns (`parcel_id`, `year`, `class_id`, `patch_id`) in
    addition to the selected features.

    Args:
        winning: Result of :func:`select_winning_features`.
        fused_df: Polars DataFrame with all features.
        output_path: Destination path of the parquet.
        overwrite: If False and the file exists, it does not write.

    Returns:
        Path of the written parquet.
    """
    output = Path(output_path)
    if output.exists() and not overwrite:
        logger.info("winning_features_already_exists", path=str(output))
        return output

    meta_cols = [
        c for c in ("parcel_id", "year", "class_id", "patch_id", "fold") if c in fused_df.columns
    ]
    feature_cols_present = [c for c in winning.feature_cols if c in fused_df.columns]
    keep = meta_cols + feature_cols_present
    subset = fused_df.select(keep)

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_parquet(output)

    manifest_path = output.with_suffix(".manifest.json")
    import json

    manifest_path.write_text(
        json.dumps(
            {
                "name": winning.name,
                "n_features": len(feature_cols_present),
                "feature_cols": feature_cols_present,
                "meta_cols": meta_cols,
                "decisions": winning.decisions,
                "delta_vs_full": winning.delta_vs_full,
                "rationale": winning.rationale,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "winning_features_persisted",
        parquet=str(output),
        manifest=str(manifest_path),
        n_features=len(feature_cols_present),
    )
    return output


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _read_deltas(ablation_table: pl.DataFrame) -> dict[str, float]:
    """Extract the mapping `{feature_set: delta_vs_full}` (reference model)."""
    if "delta_vs_full" not in ablation_table.columns:
        return {}
    # If there are several models, take the one with the best f1_macro in `full`.
    ref_model: str | None = None
    if "model" in ablation_table.columns:
        full_rows = ablation_table.filter(pl.col("feature_set") == "full")
        if full_rows.height > 0:
            ref_model = full_rows.sort("f1_macro", descending=True).select("model").row(0)[0]
    filtered = ablation_table
    if ref_model is not None:
        filtered = filtered.filter(pl.col("model") == ref_model)
    out: dict[str, float] = {}
    for row in filtered.iter_rows(named=True):
        delta = row.get("delta_vs_full")
        if delta is None:
            continue
        out[row["feature_set"]] = float(delta)
    return out


def _base_block_cols(available_cols: Sequence[str], *, include_geom: bool) -> list[str]:
    """Return the mandatory base columns present in `available_cols`."""
    pheno_known = {
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    }
    cols: list[str] = []
    for c in available_cols:
        if c in pheno_known:
            cols.append(c)
        elif "_fft_" in c:
            cols.append(c)
        elif c.startswith(("ae_", "dim_", "emb_", "alphaearth_")):
            cols.append(c)
        elif c.startswith("era5_") or c.startswith("srtm_"):
            cols.append(c)
        elif c.startswith("s1_"):
            cols.append(c)
        elif any(
            c.startswith(f"{idx.lower()}_")
            for idx in ("NDVI", "NDWI", "EVI", "MSAVI2", "MCARI", "CCCI", "NDRE")
        ):
            # Spectral indices x stats (e.g. ndvi_mean, ndwi_p95, ...).
            cols.append(c)
        elif include_geom and c.startswith("geom_"):
            cols.append(c)
    return cols


def _optional_block_cols(available_cols: Sequence[str], block: str) -> list[str]:
    """Return the cols of the indicated optional block."""
    prefix_map = {
        "farslip": "farslip_",
        "pheno_text": "pheno_text_",
        "spectral_signature": "spectral_signature_",
    }
    prefix = prefix_map.get(block)
    if prefix is None:
        return []
    return [c for c in available_cols if c.startswith(prefix)]
