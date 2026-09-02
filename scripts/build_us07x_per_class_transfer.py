"""Build the per-class transfer tables and figures (EuroCropsML + Sen4AgriNet).

Companion of the aggregate transfer artefacts (US-073/075/076): opens the
F1-macro / mIoU black box and reports the per-crop behaviour of the geographic
domain-shift, mirroring the per-class honest analysis of the ensembles
(``reports/ensemble/metrics/us043_winner_per_class.csv``). Every number is REAL;
no hand-typed values (Arthur's zero-synthetic rule).

Emits:

- ``data/transfer/eurocropsml_per_class.parquet`` (+ ``*_raw.parquet``) -- the
  EuroCropsML ``LV -> EE`` k-shot transfer with per-macro-group precision/recall/
  f1/support at ``k in {10, 100, 500}`` (3 seeds), with and without Latvia
  pretrain. Reuses the existing few-shot pipeline
  (:mod:`ml.transfer.eurocropsml_fewshot`); only the metric extraction changes
  (per-class instead of f1-macro).
- ``paper/figures/us-076/eurocropsml_per_class_f1_vs_k.{png,svg}`` -- per-class
  F1 vs k (English, canonical) and the ``*_es.{png,svg}`` Spanish twin.
- ``data/transfer/sen4agrinet_per_class.parquet`` +
  ``reports/segmentation/sen4agrinet_transfer_per_class.json`` -- the dense
  France -> Catalonia transfer per macro-group IoU/precision/recall/f1/support,
  zero-shot (France checkpoint projected to macro) and few-shot (k=10 recomputed
  locally). Reuses :mod:`ml.train.finetune_sen4agrinet` and the
  :class:`ml.eval.dense_metrics.DenseConfusionAccumulator`.
- ``paper/figures/us-075/sen4agrinet_per_class_iou_f1.{png,svg}`` -- per-class
  IoU/F1 (English, canonical) and the ``*_es.{png,svg}`` Spanish twin.

Every figure is emitted in both languages: the base name carries the English
(paper EN) text and the ``_es`` suffix carries the Spanish (paper ES) text. Only
visible strings are translated; the plotted numbers and logic are identical.
Pass ``--plot-only`` to re-render both languages from the persisted parquets
without any recompute (no GPU, no XGBoost refit).

Honest provenance: the Sen4AgriNet finetuned checkpoint lives on the VM ``F:``;
this script recomputes the few-shot finetune locally with the identical protocol
(seed 17, k=10, 40 epochs) when ``--sen4`` is requested. The zero-shot half is
fully local (France checkpoint + 40 ``.nc`` patches). If an input is missing the
script raises explicitly and never fabricates a value.

Requires ``netCDF4`` for the Sen4AgriNet half (used by
:class:`ml.data.sen4agrinet_adapter.Sen4AgriNetDataset`).

Usage::

    python -m scripts.build_us07x_per_class_transfer --eurocrops --sen4
    python -m scripts.build_us07x_per_class_transfer --sen4 --sen4-epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

# Force UTF-8 on the Windows console (MLflow/structlog emit non-cp1252 chars).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

_REPO = Path(__file__).resolve().parents[1]

#: Languages emitted for every figure. The canonical (base) name is English; the
#: Spanish variant gets an ``_es`` suffix on the output stem.
LANGS: tuple[str, ...] = ("en", "es")

#: Stable per-macro colour (shared by both figures where the macro exists).
MACRO_COLORS: dict[str, str] = {
    "grassland": "#4daf4a",
    "cereals": "#e08214",
    "oilseed_industrial": "#984ea3",
    "vineyard": "#7b3294",
    "sugar_beet": "#a6611a",
    "vegetables": "#66c2a5",
    "potato": "#d6604d",
    "legumes_fodder": "#2c7fb8",
    "soybean": "#1b9e77",
    "orchard": "#bf812d",
}

#: Per-language display names for the HCAT macro-groups (used as tick / legend
#: labels). English is the canonical machine label; Spanish is the human reading.
MACRO_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "grassland": "grassland",
        "cereals": "cereals",
        "oilseed_industrial": "oilseed/industrial",
        "vineyard": "vineyard",
        "sugar_beet": "sugar beet",
        "vegetables": "vegetables",
        "potato": "potato",
        "legumes_fodder": "legumes/fodder",
        "soybean": "soybean",
        "orchard": "orchard",
    },
    "es": {
        "grassland": "pastizal",
        "cereals": "cereales",
        "oilseed_industrial": "oleaginosas/industriales",
        "vineyard": "viñedo",
        "sugar_beet": "remolacha azucarera",
        "vegetables": "hortalizas",
        "potato": "patata",
        "legumes_fodder": "leguminosas/forraje",
        "soybean": "soja",
        "orchard": "huerto frutal",
    },
}


def _macro_label(macro: str, lang: str) -> str:
    """Return the localised display name for an HCAT macro-group.

    Args:
        macro: Machine macro-group key (e.g. ``"sugar_beet"``).
        lang: Target language (``"en"`` or ``"es"``).

    Returns:
        The localised label, falling back to the raw key when unknown.
    """
    return MACRO_LABELS.get(lang, {}).get(macro, macro)


#: Per-language visible strings for the EuroCropsML per-class F1-vs-k figure.
EC_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "xlabel": "k-shot (labelled target samples per class, Estonia)",
        "ylabel": "Per-class F1 (target query set)",
        "title": (
            "EuroCropsML transnational transfer LV -> EE: per-class F1 vs k\n"
            "(XGBoost on S2-derived features, pretrain on Latvia + k Estonia shots; "
            "3 seeds)"
        ),
        "legend_n_test": "n_test",
        "footnote": (
            "Data: EuroCropsML (Reuss et al. 2024, arXiv:2407.17458), HCAT macro "
            "label-space. Real splits, 3 seeds."
        ),
    },
    "es": {
        "xlabel": "k-shot (muestras etiquetadas por clase del objetivo, Estonia)",
        "ylabel": "F1 por clase (conjunto de consulta del objetivo)",
        "title": (
            "Transferencia transnacional EuroCropsML LV -> EE: F1 por clase vs k\n"
            "(XGBoost sobre rasgos derivados de S2, preentrenamiento en Letonia + k "
            "muestras de Estonia; 3 semillas)"
        ),
        "legend_n_test": "n_prueba",
        "footnote": (
            "Datos: EuroCropsML (Reuss et al. 2024, arXiv:2407.17458), espacio de "
            "etiquetas macro HCAT. Particiones reales, 3 semillas."
        ),
    },
}

#: Per-language visible strings for the Sen4AgriNet per-class IoU/F1 figure.
SEN4_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "iou_title": "IoU per macro-group",
        "f1_title": "F1 per macro-group",
        "zero_shot_label": "zero-shot (FR ckpt)",
        "few_shot_label": "few-shot (k=10, 40 ep)",
        "px_prefix": "n_px",
        "suptitle": (
            "Sen4AgriNet dense transfer France (PASTIS-R) -> Catalonia (ES): "
            "per-class collapse and recovery\n"
            "Zero-shot mIoU 0.000 -> few-shot recovery (TSViT, macro-HCAT "
            "label-space, 1 seed)"
        ),
        "footnote": (
            "Data: Sen4AgriNet (Sykas et al. 2022, CC-BY-SA-4.0) tile 31TCG; FR "
            "source PASTIS-R. Real held-out Catalonia val. Few-shot recomputed "
            "locally (VM checkpoint not on host)."
        ),
    },
    "es": {
        "iou_title": "IoU por macro-grupo",
        "f1_title": "F1 por macro-grupo",
        "zero_shot_label": "zero-shot (ckpt FR)",
        "few_shot_label": "few-shot (k=10, 40 ep)",
        "px_prefix": "n_px",
        "suptitle": (
            "Transferencia densa Sen4AgriNet Francia (PASTIS-R) -> Cataluña (ES): "
            "colapso y recuperación por clase\n"
            "mIoU zero-shot 0.000 -> recuperación few-shot (TSViT, espacio de "
            "etiquetas macro-HCAT, 1 semilla)"
        ),
        "footnote": (
            "Datos: Sen4AgriNet (Sykas et al. 2022, CC-BY-SA-4.0) tesela 31TCG; "
            "fuente FR PASTIS-R. Validación real reservada de Cataluña. Few-shot "
            "recalculado localmente (checkpoint de la VM no está en este host)."
        ),
    },
}


def _save_fig(fig: plt.Figure, out_stem: Path, *, dpi: int = 150) -> list[Path]:
    """Save a figure as both PNG and SVG under ``out_stem`` and close it.

    Args:
        fig: The matplotlib figure to persist.
        out_stem: Extension-less output path stem (``.png``/``.svg`` appended).
        dpi: Raster resolution for the PNG.

    Returns:
        The list of written paths (``[png, svg]``).
    """
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    svg = out_stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, svg]


def _lang_stem(base_stem: Path, lang: str) -> Path:
    """Return the language-specific output stem (English = base, Spanish = ``_es``).

    Args:
        base_stem: Canonical (English) extension-less path stem.
        lang: Target language (``"en"`` or ``"es"``).

    Returns:
        ``base_stem`` for English, ``base_stem`` + ``_es`` for Spanish.
    """
    return base_stem if lang == "en" else base_stem.with_name(f"{base_stem.name}_es")


# --- EuroCropsML config (matches the existing _feature_cache parquets) ---------
_EC_ROOT = _REPO / "data" / "transfer" / "eurocropsml"
_EC_MAX_PARCELS = 30000
_EC_K_SHOTS = (10, 100, 500)
_EC_SEEDS = (0, 1, 2)
_EC_SOURCE = ["latvia"]
_EC_TARGET = "estonia"
_EC_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "multi:softprob",
    "random_state": 42,
}

# --- Sen4AgriNet config -------------------------------------------------------
_SEN4_ROOT = _REPO / "data" / "sen4agrinet"
_SEN4_FR_CKPT = _REPO / "checkpoints" / "segmentation" / "tsvit-pheno-v1" / "best.pt"
_SEN4_K = 10
_SEN4_SEED = 17


# =============================================================================
# EuroCropsML per-class
# =============================================================================


def _ec_fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fit the XGBoost baseline recipe and return string-label predictions."""
    from ml.train.baseline import build_estimator
    from ml.transfer.label_align import NULL_CLASS

    label_to_id = {lab: i for i, lab in enumerate(sorted(set(y_fit.tolist())))}
    y_enc = np.array([label_to_id[v] for v in y_fit], dtype=np.int64)
    est = build_estimator("xgb", dict(_EC_XGB_PARAMS))
    est.fit(x_fit, y_enc)
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    pred_enc = est.predict(x_test)
    return np.array([id_to_label.get(int(p), NULL_CLASS) for p in pred_enc])


def build_eurocropsml_per_class() -> pl.DataFrame:
    """Compute the EuroCropsML LV->EE per-class transfer and persist parquets+figure.

    Returns:
        The aggregated per-class frame (also written to parquet).
    """
    from sklearn.metrics import precision_recall_fscore_support

    from ml.transfer.eurocropsml_fewshot import build_fewshot_splits

    rows: list[dict[str, object]] = []
    for use_pretrain in (True, False):
        for k in _EC_K_SHOTS:
            for seed in _EC_SEEDS:
                split = build_fewshot_splits(
                    _EC_ROOT,
                    source=_EC_SOURCE,
                    target=_EC_TARGET,
                    k=int(k),
                    seed=int(seed),
                    max_parcels=_EC_MAX_PARCELS,
                )
                if use_pretrain:
                    x_fit = np.vstack([split.x_source, split.x_target_train])
                    y_fit = np.concatenate([split.y_source, split.y_target_train])
                else:
                    x_fit, y_fit = split.x_target_train, split.y_target_train
                y_pred = _ec_fit_predict(x_fit, y_fit, split.x_target_test)
                y_true = split.y_target_test
                classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
                p, r, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=classes, average=None, zero_division=0
                )
                for cls, pc, rc, fc, sc in zip(classes, p, r, f1, support, strict=True):
                    rows.append(
                        {
                            "scenario": "LV->EE" if use_pretrain else "sin-pretrain->EE",
                            "use_pretrain": bool(use_pretrain),
                            "k": int(k),
                            "seed": int(seed),
                            "macro_group": str(cls),
                            "precision": float(pc),
                            "recall": float(rc),
                            "f1": float(fc),
                            "support": int(sc),
                        }
                    )
                logger.info(
                    "ec_point",
                    use_pretrain=use_pretrain,
                    k=k,
                    seed=seed,
                    n_classes=len(classes),
                    n_test=len(y_true),
                )

    raw = pl.DataFrame(rows)
    raw.write_parquet(_REPO / "data" / "transfer" / "eurocropsml_per_class_raw.parquet")
    agg = (
        raw.group_by("scenario", "use_pretrain", "k", "macro_group")
        .agg(
            pl.col("f1").mean().alias("f1_mean"),
            pl.col("f1").std(ddof=0).fill_null(0.0).alias("f1_std"),
            pl.col("precision").mean().alias("precision_mean"),
            pl.col("recall").mean().alias("recall_mean"),
            pl.col("support").mean().alias("support_mean"),
            pl.len().alias("n_seeds"),
        )
        .sort("scenario", "k", "macro_group")
    )
    agg.write_parquet(_REPO / "data" / "transfer" / "eurocropsml_per_class.parquet")
    plot_eurocropsml(agg)
    return agg


def plot_eurocropsml(agg: pl.DataFrame, repo_root: Path = _REPO) -> list[Path]:
    """Render the EuroCropsML per-class F1-vs-k figure in every language.

    The English variant is the canonical base name; Spanish gets an ``_es`` suffix.
    Each variant is written as PNG and SVG. Only visible strings change between
    languages; the plotted numbers and logic are identical.

    Args:
        agg: Aggregated per-class frame (see :func:`build_eurocropsml_per_class`).
        repo_root: Repository root the output paths are resolved against.

    Returns:
        The list of written figure paths across all languages and formats.
    """
    base_stem = repo_root / "paper" / "figures" / "us-076" / "eurocropsml_per_class_f1_vs_k"
    written: list[Path] = []
    for lang in LANGS:
        written.extend(_plot_eurocropsml_lang(agg, base_stem, lang))
    return written


def _plot_eurocropsml_lang(agg: pl.DataFrame, base_stem: Path, lang: str) -> list[Path]:
    """Render one language of the EuroCropsML per-class F1-vs-k figure.

    Args:
        agg: Aggregated per-class frame (LV->EE scenario is plotted).
        base_stem: Canonical (English) extension-less output stem.
        lang: Target language (``"en"`` or ``"es"``).

    Returns:
        The paths written for this language (``[png, svg]``).
    """
    txt = EC_STRINGS[lang]
    df = agg.filter(pl.col("use_pretrain"))
    classes = (
        df.group_by("macro_group")
        .agg(pl.col("support_mean").mean().alias("sup"))
        .sort("sup", descending=True)
        .get_column("macro_group")
        .to_list()
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for cls in classes:
        sub = df.filter(pl.col("macro_group") == cls).sort("k")
        ks = sub.get_column("k").to_list()
        if not ks:
            continue
        n_test = int(sub.get_column("support_mean").mean())
        ax.errorbar(
            ks,
            sub.get_column("f1_mean").to_list(),
            yerr=sub.get_column("f1_std").to_list(),
            marker="o",
            capsize=3,
            lw=2,
            color=MACRO_COLORS.get(cls, "#777777"),
            label=f"{_macro_label(cls, lang)} ({txt['legend_n_test']}~{n_test})",
        )
    ax.set_xscale("log")
    ax.set_xticks(list(_EC_K_SHOTS))
    ax.set_xticklabels([str(k) for k in _EC_K_SHOTS])
    ax.set_xlabel(txt["xlabel"])
    ax.set_ylabel(txt["ylabel"])
    ax.set_ylim(-0.02, 1.0)
    ax.set_title(txt["title"])
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.text(0.01, 0.01, txt["footnote"], fontsize=6, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_stem = _lang_stem(base_stem, lang)
    paths = _save_fig(fig, out_stem)
    logger.info("ec_figure_written", lang=lang, path=str(paths[0]))
    return paths


# =============================================================================
# Sen4AgriNet per-class
# =============================================================================


def _sen4_confusion_to_per_class(acc, n_classes: int, id_to_macro: dict[int, str]) -> list[dict]:
    """Derive per-class precision/recall/f1/iou/support from a confusion matrix."""
    conf = acc.confusion_matrix().astype(np.float64)
    rows: list[dict] = []
    for c in range(n_classes):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        support = int(conf[c, :].sum())
        if support == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        rows.append(
            {
                "macro_id": c,
                "macro_group": id_to_macro[c],
                "support_px": support,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "iou": float(iou),
            }
        )
    return rows


def build_sen4agrinet_per_class(epochs: int = 40) -> pl.DataFrame:
    """Compute the Sen4AgriNet FR->Catalonia per-class transfer (zero+few shot).

    Args:
        epochs: Few-shot finetune epochs (40 matches the VM protocol).

    Returns:
        The long per-class frame (also written to parquet + JSON).
    """
    import torch

    from ml.data.sen4agrinet_adapter import (
        IGNORE_INDEX,
        MACRO_GROUP_TO_ID,
        N_MACRO_CLASSES,
        Sen4AgriNetDataset,
    )
    from ml.eval.dense_metrics import DenseConfusionAccumulator
    from ml.eval.segmentation_inference import predict_patch_for_kind
    from ml.models.tsvit_wrapper import build_tsvit
    from ml.train.finetune_sen4agrinet import (
        _FR_N_TIMESTEPS,
        _FR_NUM_CLASSES,
        _TILE_SIZE,
        SEMANTIC18_TO_MACRO,
        _patch_level_split,
        build_macro_model_from_fr,
    )

    if not _SEN4_FR_CKPT.exists():
        raise FileNotFoundError(f"France checkpoint missing: {_SEN4_FR_CKPT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_to_macro = {v: k for k, v in MACRO_GROUP_TO_ID.items()}
    logger.info("sen4_start", device=str(device), epochs=epochs)

    cat_ds = Sen4AgriNetDataset(
        root=_SEN4_ROOT,
        n_timesteps=_FR_N_TIMESTEPS,
        tile_size=_TILE_SIZE,
        countries=("ES",),
        precache_all=True,
    )
    train_idx, val_idx, train_patches, val_patches = _patch_level_split(
        cat_ds, k=_SEN4_K, seed=_SEN4_SEED
    )

    @torch.no_grad()
    def _eval_zero_shot() -> DenseConfusionAccumulator:
        model = build_tsvit(
            num_classes=_FR_NUM_CLASSES,
            n_timesteps=_FR_N_TIMESTEPS,
            img_size=_TILE_SIZE,
            in_channels=10,
            semantic_dim=384,
        )
        ck = torch.load(_SEN4_FR_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model_state", ck), strict=False)
        model.to(device).eval()
        lut = torch.as_tensor(SEMANTIC18_TO_MACRO, device=device)
        acc = DenseConfusionAccumulator(
            N_MACRO_CLASSES, ignore_index=IGNORE_INDEX, device=str(device)
        )
        for i in val_idx:
            x, y = cat_ds[i]
            p18 = predict_patch_for_kind(model, x, model_kind="tsvit-pheno")
            acc.update(lut[torch.as_tensor(p18, device=device).clamp(0, 17)], y.to(device))
        return acc

    @torch.no_grad()
    def _eval_model(model) -> DenseConfusionAccumulator:
        model.eval()
        acc = DenseConfusionAccumulator(
            N_MACRO_CLASSES, ignore_index=IGNORE_INDEX, device=str(device)
        )
        for i in val_idx:
            x, y = cat_ds[i]
            pred = predict_patch_for_kind(model, x, model_kind="tsvit-pheno")
            acc.update(torch.as_tensor(pred, device=device), y.to(device))
        return acc

    zs_acc = _eval_zero_shot()
    zs_agg = zs_acc.compute()
    zs_pc = _sen4_confusion_to_per_class(zs_acc, N_MACRO_CLASSES, id_to_macro)
    logger.info("sen4_zero_shot_done", **{f"zs_{k}": round(v, 4) for k, v in zs_agg.items()})

    # Local few-shot finetune (two LR groups, Dice+CE; no MLflow on this host).
    from torch.utils.data import DataLoader, Subset

    from ml.models.deeplabv3plus import build_dice_ce_loss

    model = build_macro_model_from_fr(_SEN4_FR_CKPT, linear_probe=False, device=device)
    fr_state = torch.load(_SEN4_FR_CKPT, map_location="cpu", weights_only=False)
    fr_sd = fr_state.get("model_state", fr_state)
    macro_sd = model.state_dict()
    head_names = {n for n in macro_sd if n not in fr_sd or fr_sd[n].shape != macro_sd[n].shape}
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n not in head_names]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n in head_names]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": 1e-5}, {"params": head_p, "lr": 1e-4}])
    crit = build_dice_ce_loss(ignore_index=IGNORE_INDEX, n_classes=N_MACRO_CLASSES).to(device)
    warmup = 5
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - warmup), eta_min=5e-6
            ),
        ],
        milestones=[warmup],
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    loader = DataLoader(
        Subset(cat_ds, train_idx), batch_size=8, shuffle=True, num_workers=0, pin_memory=amp
    )
    best_miou, best_state = -1.0, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        m = _eval_model(model).compute()
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
        logger.info("sen4_epoch", epoch=ep + 1, val_miou=round(m["miou"], 4))
    if best_state is not None:
        model.load_state_dict(best_state)
    fs_acc = _eval_model(model)
    fs_agg = fs_acc.compute()
    fs_pc = _sen4_confusion_to_per_class(fs_acc, N_MACRO_CLASSES, id_to_macro)
    logger.info(
        "sen4_few_shot_done",
        train_time_s=round(time.perf_counter() - t0, 1),
        **{f"fs_{k}": round(v, 4) for k, v in fs_agg.items()},
    )

    rows: list[dict] = []
    for stage, pc in (("zero_shot", zs_pc), ("few_shot", fs_pc)):
        for rec in pc:
            rows.append({"stage": stage, **rec})
    df = pl.DataFrame(rows)
    df.write_parquet(_REPO / "data" / "transfer" / "sen4agrinet_per_class.parquet")

    report = {
        "scenario": "FR(PASTIS-R 18cls) -> ES(Catalonia macro-HCAT) dense transfer",
        "fr_ckpt": str(_SEN4_FR_CKPT),
        "fewshot_ckpt_origin": (
            "recomputed locally (VM best.pt tsvit-pheno-sen4agri-cat-ft-v1 not on this host)"
        ),
        "k_few_shot": _SEN4_K,
        "epochs": epochs,
        "seed": _SEN4_SEED,
        "n_val_patches": len(val_patches),
        "n_val_tiles": len(val_idx),
        "n_train_patches": len(train_patches),
        "n_train_tiles": len(train_idx),
        "zero_shot_agg": zs_agg,
        "few_shot_agg": fs_agg,
        "zero_shot_per_class": zs_pc,
        "few_shot_per_class": fs_pc,
    }
    out_json = _REPO / "reports" / "segmentation" / "sen4agrinet_transfer_per_class.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_sen4agrinet(df)
    return df


def plot_sen4agrinet(df: pl.DataFrame, repo_root: Path = _REPO) -> list[Path]:
    """Render the Sen4AgriNet per-class IoU/F1 figure in every language.

    The English variant is the canonical base name; Spanish gets an ``_es`` suffix.
    Each variant is written as PNG and SVG. Only visible strings change between
    languages; the plotted numbers and logic are identical.

    Args:
        df: Long per-class frame (see :func:`build_sen4agrinet_per_class`).
        repo_root: Repository root the output paths are resolved against.

    Returns:
        The list of written figure paths across all languages and formats.
    """
    base_stem = repo_root / "paper" / "figures" / "us-075" / "sen4agrinet_per_class_iou_f1"
    written: list[Path] = []
    for lang in LANGS:
        written.extend(_plot_sen4agrinet_lang(df, base_stem, lang))
    return written


def _plot_sen4agrinet_lang(df: pl.DataFrame, base_stem: Path, lang: str) -> list[Path]:
    """Render one language of the Sen4AgriNet per-class IoU/F1 grouped-bar figure.

    Args:
        df: Long per-class frame with ``stage`` in ``{zero_shot, few_shot}``.
        base_stem: Canonical (English) extension-less output stem.
        lang: Target language (``"en"`` or ``"es"``).

    Returns:
        The paths written for this language (``[png, svg]``).
    """
    txt = SEN4_STRINGS[lang]
    fs = df.filter(pl.col("stage") == "few_shot").sort("iou", descending=True)
    order = fs.get_column("macro_group").to_list()
    zs = df.filter(pl.col("stage") == "zero_shot")

    def vals(frame: pl.DataFrame, col: str) -> list[float]:
        m = {r["macro_group"]: r[col] for r in frame.iter_rows(named=True)}
        return [float(m.get(g, 0.0)) for g in order]

    support = {r["macro_group"]: r["support_px"] for r in fs.iter_rows(named=True)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(order))
    w = 0.38
    for ax, metric, title in ((axes[0], "iou", txt["iou_title"]), (axes[1], "f1", txt["f1_title"])):
        ax.bar(
            x - w / 2,
            vals(zs, metric),
            w,
            label=txt["zero_shot_label"],
            color="#bdbdbd",
            edgecolor="#555",
        )
        fs_v = vals(fs, metric)
        ax.bar(
            x + w / 2, fs_v, w, label=txt["few_shot_label"], color="#2c7fb8", edgecolor="#1a4f73"
        )
        for xi, v in zip(x, fs_v, strict=True):
            if v > 0.01:
                ax.text(xi + w / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{_macro_label(g, lang)}\n({txt['px_prefix']}~{support.get(g, 0) // 1000}k)"
                for g in order
            ],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(txt["suptitle"], fontsize=11)
    fig.text(0.01, 0.005, txt["footnote"], fontsize=6, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out_stem = _lang_stem(base_stem, lang)
    paths = _save_fig(fig, out_stem)
    logger.info("sen4_figure_written", lang=lang, path=str(paths[0]))
    return paths


def _replot_from_parquet(repo_root: Path, *, eurocrops: bool, sen4: bool) -> list[Path]:
    """Re-render both language variants from the persisted REAL parquets only.

    Used by ``--plot-only`` to regenerate the bilingual figures without recomputing
    any metric (no GPU, no XGBoost refit). Numbers come verbatim from the parquets
    written by the full build; only the visible strings differ per language.

    Args:
        repo_root: Repository root the artefacts are resolved against.
        eurocrops: Whether to re-render the EuroCropsML figure.
        sen4: Whether to re-render the Sen4AgriNet figure.

    Returns:
        The list of written figure paths.

    Raises:
        FileNotFoundError: if a required parquet is missing (never fabricated).
    """
    written: list[Path] = []
    if eurocrops:
        path = repo_root / "data" / "transfer" / "eurocropsml_per_class.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run the full build (--eurocrops) before --plot-only."
            )
        written.extend(plot_eurocropsml(pl.read_parquet(path), repo_root=repo_root))
    if sen4:
        path = repo_root / "data" / "transfer" / "sen4agrinet_per_class.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run the full build (--sen4) before --plot-only."
            )
        written.extend(plot_sen4agrinet(pl.read_parquet(path), repo_root=repo_root))
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO,
        help="Repository root (inputs and outputs resolved against it).",
    )
    p.add_argument("--eurocrops", action="store_true", help="Build EuroCropsML per-class.")
    p.add_argument("--sen4", action="store_true", help="Build Sen4AgriNet per-class.")
    p.add_argument("--sen4-epochs", type=int, default=40)
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only re-render bilingual figures from existing parquets (no recompute, no GPU).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not (args.eurocrops or args.sen4):
        args.eurocrops = args.sen4 = True
    if args.plot_only:
        _replot_from_parquet(args.repo_root.resolve(), eurocrops=args.eurocrops, sen4=args.sen4)
        return
    if args.eurocrops:
        build_eurocropsml_per_class()
    if args.sen4:
        build_sen4agrinet_per_class(epochs=args.sen4_epochs)


if __name__ == "__main__":
    main()
