"""Typer CLI to train FarSLIP (US-017 / US-016b).

Launches the trainer with the config validated in planning. Expected VRAM on
GCP L4 24 GB: ~22 GB. Hard cap 8 h (warning 6 h).

Typical usage::

    poetry run python -m ml.farslip.train \\
        --rois italy --epochs 4 --batch-size 64 --lr 1e-5 --seed 42 \\
        --output-dir artifacts/farslip --gcs-output-uri gs://agrosat-models/farslip/v1/

The ``--resume`` flag loads a previous checkpoint (local path or GCS) and resumes.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the train CLI. poetry add typer") from exc

from torch.utils.data import ConcatDataset

from ml.farslip.bands import (
    BandSelection,
    n_in_channels_for,
    select_and_reorder_bands,
)
from ml.farslip.cap_pastis_mapping import expand_to_cap, load_cap_to_pastis
from ml.farslip.dataset import FarSLIPDataset
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.utils.seed import propagate_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = structlog.get_logger(__name__)

ProtoSource = Literal["pastis", "random"]

#: US-035 recommended (band_selection, proto_source) pairing per variant. The
#: two flags stay INDEPENDENT on the CLI (so any combination is reproducible);
#: this map only documents the canonical pairing logged for the 3 runs.
_RECOMMENDED_PROTO_SOURCE: dict[str, ProtoSource] = {
    "rgb": "random",
    "nir_rgb": "random",
    "4band": "pastis",
}

# Italian ROIs hardcoded for --rois italy (the 3 zones of the US-017 paper).
# When France is added (us-022-e), expand this mapping to {"italy": [...], "france": [...]}.
_ROIS_BY_KEY: dict[str, tuple[str, ...]] = {
    "italy": ("pianura_padana", "toscana", "puglia"),
}


def _build_dataset(
    dataset_root: Path, rois_key: str, band_selection: BandSelection = "4band"
) -> tuple[ConcatDataset, int, int, list[str]]:
    """Concatenate the Italian ROI manifests into a PyTorch Dataset.

    Important: we pass global canonical `cap_classes` and `regions` (unified
    from the 3 manifests) so that `region_id` and `category_id` are in a
    consistent namespace across the 3 child FarSLIPDataset instances. Without
    this each child dataset derives its own indices and region_id would be
    ambiguous when concatenating.

    US-035: the band-ablation slice is wired as the per-crop ``transform`` of
    each child :class:`FarSLIPDataset` (applied AFTER the resize, on the float
    ``(C, H, W)`` crop) so the dataset yields exactly ``n_in_channels_for(
    band_selection)`` channels.

    Args:
        dataset_root: path to `data/farslip_pairs/`.
        rois_key: key in `_ROIS_BY_KEY` (default "italy").
        band_selection: US-035 band variant; selects/reorders the crop channels.

    Returns:
        Tuple (ConcatDataset, n_regions, n_categories, all_cap_classes) where
        n_regions and n_categories are the real sizes of the global vocabulary
        (needed to dimension text_prototypes correctly) and all_cap_classes is
        the canonical CAP-slug order (``category_id`` indexing) consumed by
        ``build_text_prototypes`` / ``expand_to_cap`` to keep the prototype rows
        aligned with the loss targets.

    Raises:
        FileNotFoundError: if any of the manifests does not exist.
        KeyError: if rois_key is not in `_ROIS_BY_KEY`.
    """
    import polars as pl

    if rois_key not in _ROIS_BY_KEY:
        raise KeyError(f"rois={rois_key!r} not recognized. Valid: {list(_ROIS_BY_KEY)}")
    roi_slugs = _ROIS_BY_KEY[rois_key]

    # Pre-scan: unify cap_classes and regions across the 3 manifests.
    all_cap_classes: list[str] = []
    all_regions: list[str] = []
    seen_caps: set[str] = set()
    seen_regs: set[str] = set()
    for roi in roi_slugs:
        manifest = dataset_root / roi / "manifest.parquet"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest does not exist: {manifest}")
        df = pl.read_parquet(manifest, columns=["cap_class", "region"])
        for c in df["cap_class"].to_list():
            if c not in seen_caps:
                all_cap_classes.append(c)
                seen_caps.add(c)
        for r in df["region"].to_list():
            if r not in seen_regs:
                all_regions.append(r)
                seen_regs.add(r)

    band_transform = partial(select_and_reorder_bands, sel=band_selection)
    parts = []
    for roi in roi_slugs:
        manifest = dataset_root / roi / "manifest.parquet"
        parts.append(
            FarSLIPDataset(
                manifest_path=manifest,
                cap_classes=all_cap_classes,
                regions=all_regions,
                transform=band_transform,
            )
        )
    return ConcatDataset(parts), len(all_regions), len(all_cap_classes), all_cap_classes


def _expand_cap_classes(cap_classes: Sequence[str], n_categories: int) -> list[str]:
    """Returns the canonical CAP class order padded to ``n_categories`` slots.

    The dataset derives ``n_categories`` from the union of ``cap_class`` values
    across the 3 manifests; the prototype matrix must have one row per category
    in the SAME order (``category_id`` ``0..n_categories-1``). If fewer distinct
    CAP slugs were observed than ``n_categories`` (defensive: should not happen),
    the missing tail slots are filled with ``"altro"`` (neutral, mapped to Meadow)
    so the cardinality always matches without a silent index shift.

    Args:
        cap_classes: distinct CAP slugs observed in the dataset, canonical order.
        n_categories: target number of categories (loss expects this many rows
            per region).

    Returns:
        List of length ``n_categories`` of CAP slugs in canonical order.
    """
    ordered = list(cap_classes)
    if len(ordered) < n_categories:
        ordered = ordered + ["altro"] * (n_categories - len(ordered))
    return ordered[:n_categories]


def build_text_prototypes(
    *,
    n_regions: int,
    n_categories: int,
    hidden_dim: int,
    seed: int,
    proto_source: ProtoSource = "pastis",
    cap_classes: Sequence[str] | None = None,
    prototype_path: Path | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Builds the ``(n_regions * n_categories, D)`` text prototypes for the loss.

    US-034 fix: replaces the legacy ``torch.randn`` prototypes (contrastive
    alignment against noise — the bug that made FarSLIP lose 0.163 vs 0.233) with
    the REAL phenological prototypes of US-033 (per-class MiniLM-384 embeddings).
    Flow when ``proto_source == "pastis"``:

        load_class_prototype_embeddings()  -> proto_18 (18, 384) L2-norm
        expand_to_cap(proto_18, cap_classes(32))  -> proto_cap (32, 384)
        np.tile(proto_cap, (n_regions, 1))  -> proto_tiled (96, 384), region-major

    The 384-dim tile is returned RAW; :meth:`set_text_prototypes` reprojects it to
    ``hidden_dim`` (768) via a frozen orthogonal map. Row order is region-major
    (``region * n_categories + category``), matching the loss target
    ``region_ids * n_categories + category_ids``.

    Falls back to deterministic ``torch.randn(hidden_dim)`` (legacy behaviour) ONLY
    if ``proto_source == "random"`` or the US-033 parquet is unavailable (with a
    warning), so tests / dry-runs never break.

    Args:
        n_regions: number of regions (3 for the Italian ROIs).
        n_categories: number of CAP categories (32).
        hidden_dim: student CLS dimension (768); the random fallback uses it.
        seed: determinism seed (random fallback + reprojection).
        proto_source: ``"pastis"`` (real prototypes) or ``"random"`` (legacy).
        cap_classes: CAP slugs in the dataset's canonical order (required for the
            ``"pastis"`` path; defines the row order of ``expand_to_cap``).
        prototype_path: optional override of the US-033 parquet path.

    Returns:
        Tuple ``(prototypes, meta)``. ``prototypes`` is ``(n_regions * n_categories,
        D)``: ``D == 384`` for the ``"pastis"`` path (reprojected later) or
        ``hidden_dim`` for the random fallback. ``meta`` carries MLflow params
        (``proto_source``, ``proto_proj``, ``caveat``, ``n_protos``,
        ``proto_dim_in``, ``proto_dim_out``).
    """
    n_protos = n_regions * n_categories

    def _random_fallback(reason: str) -> tuple[torch.Tensor, dict[str, object]]:
        gen = torch.Generator().manual_seed(seed)
        protos = torch.randn(n_protos, hidden_dim, generator=gen)
        meta: dict[str, object] = {
            "proto_source": "random",
            "proto_proj": "none",
            "caveat": reason,
            "n_protos": n_protos,
            "proto_dim_in": hidden_dim,
            "proto_dim_out": hidden_dim,
        }
        _log.warning("text_prototypes random fallback", reason=reason, n_protos=n_protos)
        return protos, meta

    if proto_source == "random":
        return _random_fallback("proto_source=random (explicit A/B baseline)")

    if cap_classes is None:
        raise ValueError("cap_classes is required for proto_source='pastis'")

    try:
        from ml.features.phenology_class_prototypes import (
            load_class_prototype_embeddings,
        )

        if prototype_path is not None:
            proto_18, class_ids = load_class_prototype_embeddings(prototype_path)
        else:
            proto_18, class_ids = load_class_prototype_embeddings()
    except (FileNotFoundError, ImportError, OSError) as exc:
        return _random_fallback(f"US-033 parquet unavailable: {exc}")

    ordered_caps = _expand_cap_classes(cap_classes, n_categories)
    proto_cap = expand_to_cap(
        proto_18,
        ordered_caps,
        mapping=load_cap_to_pastis(),
        pastis_class_ids=class_ids,
    )  # (n_categories, 384)
    proto_tiled = np.tile(proto_cap, (n_regions, 1))  # (n_protos, 384), region-major
    if proto_tiled.shape[0] != n_protos:
        raise ValueError(f"tiled prototypes rows={proto_tiled.shape[0]} != expected {n_protos}")
    dim_in = int(proto_tiled.shape[1])
    prototypes = torch.from_numpy(proto_tiled).float()
    meta = {
        "proto_source": "pastis_prototypes",
        "proto_proj": f"ortho_{dim_in}_{hidden_dim}",
        "caveat": "ortho_proj_crude_approx",
        "n_protos": n_protos,
        "proto_dim_in": dim_in,
        "proto_dim_out": hidden_dim,
    }
    _log.info(
        "text_prototypes built from pastis prototypes",
        n_protos=n_protos,
        proto_dim_in=dim_in,
        proto_dim_out=hidden_dim,
        n_pastis=int(proto_18.shape[0]),
        n_categories=n_categories,
    )
    return prototypes, meta


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def train(
    rois: Annotated[str, typer.Option(help="Identificador de ROI set, e.g. 'italy'")] = "italy",
    epochs: Annotated[int, typer.Option(help="Numero de epochs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size logico")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    output_dir: Annotated[Path, typer.Option(help="Directorio local para checkpoints")] = Path(
        "artifacts/farslip"
    ),
    gcs_output_uri: Annotated[
        str | None, typer.Option(help="URI GCS para subir pesos finales")
    ] = None,
    dataset_root: Annotated[Path, typer.Option(help="Raiz dataset farslip_pairs")] = Path(
        "data/farslip_pairs"
    ),
    teacher_model_id: Annotated[
        str, typer.Option(help="HF id del CLIP teacher")
    ] = "openai/clip-vit-base-patch16",
    resume: Annotated[str | None, typer.Option(help="Ruta/URI a checkpoint para reanudar")] = None,
    time_cap_hours: Annotated[float, typer.Option(help="Hard cap horas")] = 8.0,
    proto_source: Annotated[
        str,
        typer.Option(help="Fuente de prototipos: 'pastis' (fenologicos US-033) o 'random' (A/B)"),
    ] = "pastis",
    band_selection: Annotated[
        str,
        typer.Option(
            help=(
                "Ablacion de bandas US-035: 'rgb' (B04-B03-B02, 3 canales), "
                "'nir_rgb' (B08-B04-B03 falso color, 3 canales) o '4band' "
                "(B02-B03-B04-B08, 4 canales, identidad). Recomendado: rgb/nir_rgb "
                "con --proto-source random, 4band con --proto-source pastis "
                "(flags independientes)."
            )
        ),
    ] = "4band",
    mlflow_run_name: Annotated[
        str, typer.Option(help="Nombre del run MLflow")
    ] = "farslip-pheno-fix-v1",
) -> None:
    """Train FarSLIP with the provided configuration."""
    propagate_seed(seed)
    # US-035: validate band_selection and DERIVE n_in_channels from it (single
    # source of truth, R-NCHAN). rgb/nir_rgb -> 3 (patch_embed no-op);
    # 4band -> 4 (3->4 mean-RGB adaptation, US-034).
    if band_selection not in ("rgb", "nir_rgb", "4band"):
        raise typer.BadParameter("band-selection must be 'rgb', 'nir_rgb' or '4band'")
    band_selection_typed: BandSelection = band_selection  # type: ignore[assignment]
    n_in_channels = n_in_channels_for(band_selection_typed)
    recommended_proto = _RECOMMENDED_PROTO_SOURCE[band_selection_typed]
    if proto_source != recommended_proto:
        # Informational only: the flags are intentionally independent so any
        # combination remains reproducible; we just surface the canonical pairing.
        _log.info(
            "band_selection/proto_source pairing differs from recommended",
            band_selection=band_selection_typed,
            proto_source=proto_source,
            recommended_proto_source=recommended_proto,
        )
    _log.info(
        "starting farslip training",
        rois=rois,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        band_selection=band_selection_typed,
        n_in_channels=n_in_channels,
        proto_source=proto_source,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # US-022-c P1 fix (2026-05-24): instantiate FarSLIPDataset + ConcatDataset for the
    # 3 Italian ROIs. The previous CLI only instantiated the trainer without a dataset, which
    # triggered RuntimeError("dataset y dataloader nulos: nada que entrenar") in distill.py:534.
    dataset, n_regions, n_categories, cap_classes = _build_dataset(
        dataset_root, rois, band_selection=band_selection_typed
    )
    _log.info(
        "dataset built",
        n_samples=len(dataset),
        rois=rois,
        n_regions=n_regions,
        n_categories=n_categories,
        band_selection=band_selection_typed,
    )

    # US-034: build prototypes BEFORE the trainer so the meta (caveat params) can
    # be logged to MLflow via cfg.extra_params.
    if proto_source not in ("pastis", "random"):
        raise typer.BadParameter("proto-source must be 'pastis' or 'random'")
    hidden_dim_probe = 768  # ViT-B/16 hidden_size; confirmed against the trainer below.
    prototypes, proto_meta = build_text_prototypes(
        n_regions=n_regions,
        n_categories=n_categories,
        hidden_dim=hidden_dim_probe,
        seed=seed,
        proto_source=proto_source,  # type: ignore[arg-type]
        cap_classes=cap_classes,
    )

    # US-035: surface band_selection in extra_params too (the dedicated config
    # field is already logged as a top-level MLflow param; this keeps the band
    # ablation context together with the prototype meta in the tags table).
    extra_params = dict(proto_meta)
    extra_params["band_selection"] = band_selection_typed
    extra_params["recommended_proto_source"] = recommended_proto
    cfg = FarSLIPTrainerConfig(
        teacher_model_id=teacher_model_id,
        dataset_root=dataset_root,
        output_dir=output_dir,
        gcs_output_uri=gcs_output_uri,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        time_cap_hours=time_cap_hours,
        n_in_channels=n_in_channels,
        band_selection=band_selection_typed,
        n_regions=n_regions,
        n_categories=n_categories,
        mlflow_run_name=mlflow_run_name,
        extra_params=extra_params,
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=dataset)
    if resume:
        _log.info("resume from checkpoint", uri=resume)
        path = Path(resume)
        if path.exists():
            sd = torch.load(path, map_location=trainer.device, weights_only=True)
            trainer.student.load_state_dict(sd, strict=False)

    # US-034 fix: inject the REAL phenological prototypes (US-033) instead of
    # torch.randn (which aligned the contrastive InfoNCE against noise). The
    # prototypes were built above; set_text_prototypes reprojects the MiniLM-384
    # tile to the student CLS dim (768) via a frozen orthogonal map and asserts D.
    hidden_dim = int(trainer.teacher.config.hidden_size)
    if proto_meta.get("proto_source") == "random" and hidden_dim != hidden_dim_probe:
        # Random fallback was sized with the probe; rebuild at the true hidden dim.
        prototypes, proto_meta = build_text_prototypes(
            n_regions=n_regions,
            n_categories=n_categories,
            hidden_dim=hidden_dim,
            seed=seed,
            proto_source="random",
        )
        # Preserve the US-035 band-ablation keys when refreshing the proto meta.
        rebuilt = dict(proto_meta)
        rebuilt["band_selection"] = band_selection_typed
        rebuilt["recommended_proto_source"] = recommended_proto
        cfg.extra_params = rebuilt
    trainer.set_text_prototypes(prototypes)
    _log.info(
        "text_prototypes initialized",
        n_protos=prototypes.shape[0],
        hidden_dim=hidden_dim,
        **{k: proto_meta[k] for k in ("proto_source", "proto_proj", "caveat")},
    )

    metrics = trainer.train()
    _log.info("training done", **metrics)


if __name__ == "__main__":  # pragma: no cover
    app()
