"""'El modelo en accion': fichas de inferencia del Stacking por parcela (06c).

Para hacer tangible el ensamble ganador, este modulo arma, sobre patches REALES
del fold-5 held-out, una ficha por patch con cuatro piezas alineadas:

1. **Fuente de la verdad** (RGB real del patch + verdad de campo por parcela).
2. **Lo predicho** por el Stacking, parcela a parcela, con su etiqueta.
3. **La fenologia** de cada parcela: su curva NDVI temporal real (lo que el
   modelo "ve" en el tiempo) y una descripcion derivada de esa curva.
4. **El embedding** que usa el miembro tabular (AlphaEarth 64-dim por parcela).

La prediccion del Stacking se reconstruye desde las OOF reales de sus tres base
learners (``tsvit-pheno``, ``utae``, ``xgb-alphaearth``) promediando sus matrices
post-softmax por parcela (consenso; el argmax coincide con el del meta-modelo en
la inmensa mayoria de parcelas). Todo es OOF de fold-5: el modelo no vio estas
parcelas al entrenar.

Reusa los helpers de render de :mod:`ml.report.lote_figures`
(``_rgb_from_peak_ndvi``) y la reconciliacion parcela de
:mod:`ml.utils.parcel_reconcile`. Conventions: Polars, numpy/torch solo en el
borde de datos, matplotlib Agg, structlog, type hints, English docstrings,
Spanish prose, no emojis.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["InferenceCard", "build_stacking_inference_cards"]

#: Base learners of the E3 Stacking whose parcel OOF are averaged.
_STACKING_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")
#: PASTIS non-agronomic ids excluded from the cards (same policy as extract_regions).
_BACKGROUND, _VOID = 0, 19
#: Minimum parcel area (px) to show -- slivers carry no readable signal.
_MIN_AREA_PX = 32
#: Sentinel-2 NDVI band indices in the PASTIS (T, 10, H, W) layout.
_B04_IDX, _B08_IDX = 2, 6
#: Minimum NIR reflectance (PASTIS int16 scale, ~0.05) for a step to count as a
#: real observation. Below it the step is cloud/shadow/no-data: B08 near zero with
#: B04 clipped to 0 yields a spurious NDVI of 1.0 that would dominate the peak.
_MIN_NIR = 500.0


class InferenceCard:
    """One patch's inference card: figure path + per-parcel table.

    Attributes:
        patch_id: PASTIS-R patch id (fold-5 held-out).
        figure_path: path to the saved 3-panel PNG (RGB / truth / prediction).
        table: per-parcel :class:`polars.DataFrame` (parcel, true, pred, hit,
            area, NDVI peak, phenology text).
        n_parcels: number of agronomic parcels shown.
        n_correct: how many the Stacking got right.
        n_alphaearth_dims: AlphaEarth embedding size the tabular member consumes.
    """

    def __init__(
        self,
        patch_id: str,
        figure_path: Path,
        table: pl.DataFrame,
        n_parcels: int,
        n_correct: int,
        n_alphaearth_dims: int = 0,
    ) -> None:
        self.patch_id = patch_id
        self.figure_path = figure_path
        self.table = table
        self.n_parcels = n_parcels
        self.n_correct = n_correct
        self.n_alphaearth_dims = n_alphaearth_dims


def _describe_phenology(ndvi: np.ndarray, doy: np.ndarray | None = None) -> str:
    """Derive a short Spanish phenology description from an NDVI curve.

    Rule-based and deterministic (no LLM): reads the peak position, amplitude and
    early/late vigor of the temporal NDVI to describe the growth pattern. This is
    the same information a phenology caption encodes, computed live for a fold-5
    parcel (which has no pre-generated caption, since captions cover train folds).

    Args:
        ndvi: ``(T,)`` mean NDVI of the parcel over the time series.
        doy: optional ``(T,)`` day-of-year per step (unused in the summary text
            but kept for signature parity / future use).

    Returns:
        A one-line Spanish description of the phenological pattern.
    """
    if ndvi.size == 0:
        return "sin observaciones NDVI útiles"
    # Robust peak/amplitude: the p90/p10 ignore 1-2 residual cloud/shadow spikes
    # that the band-level filter cannot catch (e.g. thin haze), so the summary
    # reflects the real growth envelope, not a single noisy step.
    peak = float(np.percentile(ndvi, 90))
    low = float(np.percentile(ndvi, 10))
    amp = peak - low
    # Position the peak at the step closest to the robust peak value.
    peak_i = int(np.argmin(np.abs(ndvi - peak)))
    frac = peak_i / max(ndvi.size - 1, 1)
    when = "temprano" if frac < 0.35 else ("tardío" if frac > 0.65 else "a media temporada")
    vigor = "alto" if peak > 0.6 else ("moderado" if peak > 0.4 else "bajo")
    dyn = "marcada" if amp > 0.4 else ("moderada" if amp > 0.2 else "plana")
    return (
        f"pico de vigor {vigor} (NDVI {peak:.2f}) {when}; "
        f"dinámica estacional {dyn} (amplitud {amp:.2f})"
    )


def _parcel_ndvi(s2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean NDVI curve ``(T,)`` of the parcel pixels from the S2 time series.

    PASTIS-R surface reflectances are raw int16 that include atmospheric-correction
    artifacts (small or negative B04/B08), which make the naive ratio overflow past
    the physical NDVI range. We clip reflectances to be non-negative, guard the
    denominator, drop steps with no usable signal and clip the result to [-1, 1] so
    the phenology summary reflects real vigor, not numerical noise.
    """
    b04 = np.clip(s2[:, _B04_IDX, :, :][:, mask].mean(axis=1).astype(float), 0.0, None)
    b08 = np.clip(s2[:, _B08_IDX, :, :][:, mask].mean(axis=1).astype(float), 0.0, None)
    # Keep only steps with real NIR signal: cloud/shadow/no-data steps (B08 ~ 0,
    # B04 clipped to 0) would otherwise yield a spurious NDVI of 1.0 that dominates
    # the peak and breaks the phenology summary.
    valid = b08 >= _MIN_NIR
    if not valid.any():
        return np.array([])
    b04, b08 = b04[valid], b08[valid]
    ndvi = np.clip((b08 - b04) / (b08 + b04 + 1e-6), -1.0, 1.0)
    return np.asarray(ndvi, dtype=float)


def _natural_rgb(s2: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Bright, natural true-color RGB ``(H, W, 3)`` from the S2 time series.

    The peak-NDVI composite picks a SINGLE timestep optimized for vegetation
    contrast (great as a model input, but it renders dark forest/water/shadow
    areas almost black and may carry clouds). For a readable visual reference we
    instead take the TEMPORAL MEDIAN of the true-color bands (B04/B03/B02) over
    all steps -- robust to clouds and shadows -- then stretch each channel on its
    own 2-98 percentile and lift midtones with a mild gamma.

    Args:
        s2: int16 tensor ``(T, 10, H, W)`` (PASTIS 10-band layout).
        gamma: <1 brightens midtones.

    Returns:
        ``(H, W, 3)`` RGB in [0, 1], bands [B04, B03, B02].
    """
    s2f = np.clip(s2.astype(np.float32), 0.0, None)
    # True-color band order R=B04(idx2), G=B03(idx1), B=B02(idx0).
    bands = s2f[:, [2, 1, 0], :, :]  # (T, 3, H, W)
    rgb = np.median(bands, axis=0).transpose(1, 2, 0)  # (H, W, 3)
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        if hi <= lo:
            hi = lo + 1e-3
        out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(np.power(out, gamma), dtype=np.float32)


def _slot_map(class_map: np.ndarray, id_to_slot: dict[int, int]) -> np.ndarray:
    """Remap a class-id map to contiguous colormap slots (NaN where no parcel).

    Args:
        class_map: ``(H, W)`` map of PASTIS class ids (0 = no parcel painted).
        id_to_slot: ``{class_id: colormap_slot}`` for the classes present.

    Returns:
        ``(H, W)`` float map with the slot index per pixel and NaN elsewhere.
    """
    out = np.full(class_map.shape, np.nan)
    for cid, slot in id_to_slot.items():
        out[class_map == cid] = slot
    return out


def _stacking_pred_by_parcel(
    canonical_ids: list[str],
    oof_dir: Path,
    members: tuple[str, ...] = _STACKING_MEMBERS,
) -> dict[str, int]:
    """Reconstruct an ensemble's consensus argmax per parcel from member OOF.

    Averages the ``members``' post-softmax ``(18,)`` rows per
    ``canonical_parcel_id`` and returns the argmax mapped to the PASTIS class id
    (``prob_000 -> class 1``). Only parcels present in ALL members are returned.

    This uniform-average consensus matches the Stacking meta-model argmax in the
    vast majority of parcels and is the cheap fallback when no real per-parcel
    prediction is injected. For a non-uniform ensemble (e.g. Blending's Optuna
    weights) pass the real prediction via ``pred_by_parcel`` instead.

    Args:
        canonical_ids: parcels of interest (``{patch}_{ParcelIDs_local}``).
        oof_dir: directory holding ``oof_parcel_{member}_fold5.parquet``.
        members: the base learners whose parcel OOF are averaged.

    Returns:
        Mapping ``canonical_parcel_id -> predicted PASTIS class id (1..18)``.
    """
    prob_cols = [f"prob_{i:03d}" for i in range(18)]
    acc: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    for member in members:
        path = oof_dir / f"oof_parcel_{member}_fold5.parquet"
        d = pl.read_parquet(path).filter(pl.col("canonical_parcel_id").is_in(canonical_ids))
        for row in d.select(["canonical_parcel_id", *prob_cols]).iter_rows(named=True):
            cid = row["canonical_parcel_id"]
            vec = np.array([row[c] for c in prob_cols], dtype=float)
            acc[cid] = acc.get(cid, np.zeros(18)) + vec
            count[cid] = count.get(cid, 0) + 1
    preds: dict[str, int] = {}
    for cid, vec in acc.items():
        if count[cid] == len(members):  # present in every member
            preds[cid] = int(np.argmax(vec)) + 1
    return preds


def build_stacking_inference_cards(
    patch_ids: list[str],
    *,
    pastis_root: Path,
    oof_dir: Path,
    features_path: Path,
    out_dir: Path,
    class_names: dict[int, str] | None = None,
    members: tuple[str, ...] = _STACKING_MEMBERS,
    ensemble_label: str = "Stacking",
    pred_by_parcel: dict[str, int] | None = None,
) -> list[InferenceCard]:
    """Build the per-patch 'ensemble in action' cards over fold-5 patches.

    For each patch: render the 3-panel figure (real RGB / per-parcel truth /
    per-parcel ensemble prediction) and a per-parcel table joining the true
    class, the predicted class, the hit flag, the area, the live NDVI phenology
    description and a summary of the AlphaEarth embedding the tabular member uses.
    The phenology description lives ONLY in the returned table (column
    ``fenologia``) and is NEVER drawn on the figure, so the notebook can surface
    it with its own ``display`` separate from the image.

    Works for ANY ensemble, not just the winner: pass its ``members`` (for the
    embedding/legend and the consensus fallback) and its ``ensemble_label`` (for
    the panel title). When ``pred_by_parcel`` is given (the REAL per-parcel
    argmax of the fitted ensemble, e.g. ``Blending``/``Stacking.predict_proba``),
    it is used verbatim; otherwise the uniform-average consensus over ``members``
    is reconstructed (honest for a uniform stacking, an approximation otherwise).

    Args:
        patch_ids: fold-5 patch ids to render.
        pastis_root: real PASTIS-R root.
        oof_dir: directory with the base-learner parcel OOF parquets.
        features_path: ``features_fused_pastis.parquet`` (AlphaEarth dims +
            ``class_name`` per parcel, keyed by ``{patch}_{instance}``; used only
            for the embedding summary, matched by patch + area rank).
        out_dir: where the card PNGs are written.
        class_names: optional ``{id: name}`` override; defaults to PASTIS names.
        members: the ensemble's base learners (default = the E3 Stacking trio).
        ensemble_label: readable ensemble name for the prediction panel title.
        pred_by_parcel: optional REAL ``{canonical_parcel_id: PASTIS class id}``
            of the fitted ensemble; when ``None`` the consensus over ``members``
            is used.

    Returns:
        One :class:`InferenceCard` per patch that had agronomic parcels.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from ml.ingest.pastis_loader import PASTIS_R_CLASSES, load_pastis_patch
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    names = dict(PASTIS_R_CLASSES)
    if class_names:
        names.update(class_names)
    pastis_root = Path(pastis_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # AlphaEarth embedding dimensionality the tabular member consumes (reported
    # in the notebook prose). Read once for the count; the per-parcel join across
    # the two disjoint id namespaces is not 1:1, so the embedding is described at
    # the member level, not per parcel.
    n_alphaearth_dims = sum(
        1 for c in pl.scan_parquet(features_path).collect_schema().names() if c.startswith("dim_")
    )

    cards: list[InferenceCard] = []
    for pid in patch_ids:
        parcels = load_pastis_parcel_ids(pid, pastis_root)
        patch = load_pastis_patch(pid, root=pastis_root, load_annotations=True)
        sem = np.asarray(patch["semantic"])
        s2 = np.asarray(patch["s2"])

        # Agronomic parcels of this patch (ParcelIDs local ids, class 1..18, area).
        local_ids = sorted(int(x) for x in np.unique(parcels) if x != 0)
        rows: list[dict] = []
        truth_map = np.zeros_like(sem)
        pred_map = np.zeros_like(sem)
        canon_ids = [f"{pid}_{lid}" for lid in local_ids]
        if pred_by_parcel is not None:
            preds = {c: pred_by_parcel[c] for c in canon_ids if c in pred_by_parcel}
        else:
            preds = _stacking_pred_by_parcel(canon_ids, Path(oof_dir), members)

        for lid in local_ids:
            mask = parcels == lid
            area = int(mask.sum())
            true_cls = Counter(sem[mask].tolist()).most_common(1)[0][0]
            if true_cls in (_BACKGROUND, _VOID) or area < _MIN_AREA_PX:
                continue
            cid = f"{pid}_{lid}"
            pred_cls = preds.get(cid)
            if pred_cls is None:
                continue
            ndvi = _parcel_ndvi(s2, mask)
            truth_map[mask] = true_cls
            pred_map[mask] = pred_cls
            rows.append(
                {
                    "parcela": cid,
                    "clase_real": names.get(true_cls, str(true_cls)),
                    "clase_predicha": names.get(pred_cls, str(pred_cls)),
                    "acierto": bool(pred_cls == true_cls),
                    "area_px": area,
                    "ndvi_pico": (round(float(np.percentile(ndvi, 90)), 3) if ndvi.size else 0.0),
                    "fenologia": _describe_phenology(ndvi),
                }
            )

        if not rows:
            continue
        table = pl.DataFrame(rows).sort("area_px", descending=True)
        n_correct = int(table["acierto"].sum())

        # --- 3-panel figure: RGB / truth / prediction ---
        rgb = _natural_rgb(s2)
        present = sorted(
            set(table["clase_real"].to_list()) | set(table["clase_predicha"].to_list())
        )
        present_ids = sorted({k for k, v in names.items() if v in present})
        cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(len(present_ids), 1))))
        id_to_slot = {cid: i for i, cid in enumerate(present_ids)}
        bounds = list(range(len(present_ids) + 1))
        norm = BoundaryNorm(bounds, cmap.N) if present_ids else None

        # Larger panels + reserved bottom strip for the legend so titles never
        # overlap the images. constrained_layout keeps the three axes balanced.
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), layout="constrained")
        axes[0].imshow(np.clip(rgb, 0, 1))
        axes[0].set_title(f"RGB real (Patch {pid})", fontsize=13, pad=10)
        axes[0].axis("off")
        axes[1].imshow(
            _slot_map(truth_map, id_to_slot), cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes[1].set_title("Verdad de campo (por parcela)", fontsize=13, pad=10)
        axes[1].axis("off")
        axes[2].imshow(
            _slot_map(pred_map, id_to_slot), cmap=cmap, norm=norm, interpolation="nearest"
        )
        axes[2].set_title(
            f"Prediccion {ensemble_label}: {n_correct}/{len(rows)} correctas",
            fontsize=13,
            pad=10,
        )
        axes[2].axis("off")
        handles = [
            Patch(color=cmap(id_to_slot[c]), label=names.get(c, str(c))) for c in present_ids
        ]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=min(6, len(handles)),
            fontsize=11,
            frameon=False,
        )
        slug = "".join(ch if ch.isalnum() else "-" for ch in ensemble_label.lower()).strip("-")
        fig_path = out_dir / f"inference_{slug}_{pid}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        cards.append(InferenceCard(pid, fig_path, table, len(rows), n_correct, n_alphaearth_dims))
        logger.info(
            "ensemble_inference_card",
            ensemble=ensemble_label,
            patch_id=pid,
            n_parcels=len(rows),
            n_correct=n_correct,
            figure=str(fig_path),
        )

    return cards
