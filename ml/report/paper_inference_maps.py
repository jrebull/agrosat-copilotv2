"""Publication-grade per-parcel inference maps (4-panel, colour-blind safe, scaled).

This module renders the "model in action" figure at journal quality, going beyond
:mod:`ml.report.ensemble_inference_cards` (which produces the 3-panel notebook
card) by adding the three things a remote-sensing reviewer expects:

1. **A fourth panel (d): the error map.** Correct parcels in green, wrong ones in
   red, so the figure shows *where* the model fails instead of leaving the reader
   to diff panels (b) and (c) by eye.
2. **A real, per-patch scale bar.** PASTIS-R patches are reprojected to Lambert-93
   (EPSG:2154) and are NOT a fixed physical size: the metadata bounding boxes span
   1279-1376 m for a 128 px patch, i.e. 9.99-10.75 m/px depending on the patch.
   The bar length is therefore computed from each patch's true ground sampling
   distance (:func:`patch_ground_resolution`), not the nominal 10 m.
3. **A colour-blind-safe palette** (Paul Tol / Okabe-Ito) instead of ``tab20``,
   plus subpanel tags ``(a)..(d)`` for caption referencing and 300 dpi output.

The per-parcel aggregation (majority-vote ground truth, ``pred_by_parcel`` overlay,
background dropped) mirrors :func:`ml.report.ensemble_inference_cards
.build_stacking_inference_cards` so the parcel-hit counts match the published
champion numbers exactly.

Real PASTIS-R data only; nothing is fabricated. A parcel below ``min_area_px`` or
whose true class is Background/Void is dropped and excluded from the hit count.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PaperInferenceMap",
    "build_paper_inference_maps",
    "patch_ground_resolution",
]

#: PASTIS-R semantic ids dropped from the parcel hit count.
_BACKGROUND: int = 0
_VOID: int = 19

#: Minimum parcel area (pixels) to be scored, matching ensemble_inference_cards.
_DEFAULT_MIN_AREA_PX: int = 32

#: Colour-blind-safe palette (Paul Tol "muted" + Okabe-Ito), indexed by the
#: PASTIS semantic id minus one. Eighteen distinct, print-safe hues.
_CB_PALETTE: tuple[str, ...] = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#882255",
    "#44AA99",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#117733",
    "#332288",
    "#AA4499",
    "#88CCEE",
    "#999999",
)

_INK = "#1a1a2e"
_GREEN = "#15803d"
_ERR_RED = "#d62728"
_OK_GREEN = "#2ca02c"
#: Neutral grey for out-of-scope parcels (classes the model cannot resolve).
_OOS_GREY = "#bdbdbd"


@dataclass(frozen=True)
class PaperInferenceMap:
    """Result of rendering one patch's publication figure.

    Attributes:
        patch_id: PASTIS-R patch identifier.
        path: Written PNG path.
        n_correct: Parcels the model classified correctly (in-scope only).
        n_parcels: Scored in-scope parcels (Background/Void, tiny and out-of-scope
            parcels excluded).
        n_out_of_scope: Parcels whose true class the model cannot resolve (only
            non-zero when ``resolved_classes`` is given); not counted as errors.
        ground_resolution_m: True metres-per-pixel of this patch (Lambert-93).
    """

    patch_id: str
    path: Path
    n_correct: int
    n_parcels: int
    n_out_of_scope: int
    ground_resolution_m: float

    @property
    def parcel_accuracy(self) -> float:
        """Fraction of scored parcels classified correctly."""
        return self.n_correct / self.n_parcels if self.n_parcels else 0.0


def patch_ground_resolution(patch_id: str | int, pastis_root: Path) -> float:
    """Return the true metres-per-pixel of a PASTIS-R patch.

    PASTIS-R ships ``metadata.geojson`` in Lambert-93 (EPSG:2154, metres). Each
    patch is a 128x128 raster, but its physical extent varies with location
    (reprojection from the Sentinel-2 UTM grid), so the ground sampling distance
    is NOT a constant 10 m. This reads the patch's bounding box and divides the
    mean side length by 128.

    Args:
        patch_id: PASTIS-R patch identifier (``ID_PATCH``).
        pastis_root: Root of the PASTIS-R dataset (holds ``metadata.geojson``).

    Returns:
        Metres per pixel for this patch; falls back to ``10.0`` if the patch is
        not found in the metadata.
    """
    geo = json.loads((Path(pastis_root) / "metadata.geojson").read_text())
    target = str(patch_id)
    for feature in geo["features"]:
        if str(feature["properties"].get("ID_PATCH", "")) == target:
            pts: list[list[float]] = []
            _flatten_coords(feature["geometry"]["coordinates"], pts)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return ((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2 / 128
    return 10.0


def _flatten_coords(node: object, out: list[list[float]]) -> None:
    """Collect ``[x, y]`` pairs from an arbitrarily nested GeoJSON ring."""
    if isinstance(node, (list, tuple)):
        if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
            out.append([float(node[0]), float(node[1])])
        else:
            for child in node:
                _flatten_coords(child, out)


def _rgb_from_s2(s2: np.ndarray) -> np.ndarray:
    """Build a contrast-stretched RGB ``(H, W, 3)`` from an S2 patch.

    Collapses the temporal axis by median if present, takes B4/B3/B2 and rescales
    by the 2-98 percentile for a readable composite.
    """
    arr = np.asarray(s2)
    if arr.ndim == 4:
        arr = np.median(arr, axis=0)
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    stretched: np.ndarray = np.clip((rgb - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return stretched


def build_paper_inference_maps(
    patch_ids: list[str],
    *,
    pastis_root: Path,
    pred_by_parcel: dict[str, int],
    out_dir: Path,
    model_label: str = "Voting-3",
    class_names: dict[int, str] | None = None,
    resolved_classes: tuple[int, ...] | None = None,
    min_area_px: int = _DEFAULT_MIN_AREA_PX,
    scale_bar_m: float = 500.0,
    dpi: int = 300,
) -> list[PaperInferenceMap]:
    """Render the 4-panel publication inference figure for each patch.

    For every patch: aggregate the per-pixel ground truth to parcels (majority
    vote over the ParcelID raster), overlay ``pred_by_parcel`` as the prediction,
    and draw four panels -- (a) RGB with a true scale bar, (b) per-parcel truth,
    (c) per-parcel prediction, (d) correct/incorrect error map -- with a
    colour-blind-safe palette at ``dpi``.

    Args:
        patch_ids: PASTIS-R patch ids to render.
        pastis_root: PASTIS-R root (``ANNOTATIONS`` + ``metadata.geojson``).
        pred_by_parcel: ``{canonical_parcel_id -> PASTIS class id (1..18)}`` of the
            fitted ensemble (e.g. ``VotingReport.pred_by_parcel``). Keys are
            ``f"{patch_id}_{local_parcel_id}"``.
        out_dir: Directory for the PNGs (created if absent).
        model_label: Readable model name shown in panel (c) and the title.
        class_names: Optional ``{id: name}`` override; defaults to PASTIS names.
        resolved_classes: PASTIS class ids the model actually resolves (the
            ``france-N`` label-space, e.g. the 12 champion classes). A parcel whose
            true class is NOT in this set is *out of scope*: the model cannot
            predict it by construction, so it is drawn in a neutral hatched grey in
            panel (d) and EXCLUDED from the hit count (``n_correct``/``n_parcels``),
            matching the restricted F1. When ``None`` every class is in scope (the
            legacy behaviour: out-of-scope parcels then count as errors).
        min_area_px: Minimum parcel area to score (smaller parcels are dropped).
        scale_bar_m: Length of the scale bar in metres.
        dpi: Output resolution.

    Returns:
        One :class:`PaperInferenceMap` per rendered patch (those with parcels).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from ml.ingest.pastis_loader import PASTIS_R_CLASSES, load_pastis_patch
    from ml.utils.parcel_reconcile import load_pastis_parcel_ids

    names = dict(PASTIS_R_CLASSES)
    if class_names:
        names.update(class_names)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "text.color": _INK,
            "font.size": 12,
        }
    )

    in_scope = set(resolved_classes) if resolved_classes is not None else None
    results: list[PaperInferenceMap] = []
    for pid in patch_ids:
        parcels = load_pastis_parcel_ids(pid, pastis_root)
        patch = load_pastis_patch(pid, root=pastis_root, load_annotations=True)
        sem = np.asarray(patch["semantic"])
        s2 = np.asarray(patch["s2"])

        truth = np.zeros_like(sem)
        pred = np.zeros_like(sem)
        # err panel encoding: 0.0 error, 1.0 hit, 0.5 out-of-scope, NaN no parcel.
        err = np.full(sem.shape, np.nan)
        n_ok = n_tot = n_oos = 0
        for lid in (int(x) for x in np.unique(parcels) if x != 0):
            mask = parcels == lid
            area = int(mask.sum())
            true_cls = Counter(sem[mask].tolist()).most_common(1)[0][0]
            if true_cls in (_BACKGROUND, _VOID) or area < min_area_px:
                continue
            pred_cls = pred_by_parcel.get(f"{pid}_{lid}")
            if pred_cls is None:
                continue
            truth[mask] = true_cls
            pred[mask] = pred_cls
            if in_scope is not None and true_cls not in in_scope:
                # The model cannot resolve this class by construction: out of
                # scope. Not a model error -> excluded from the count, drawn grey.
                err[mask] = 0.5
                n_oos += 1
                continue
            n_tot += 1
            hit = pred_cls == true_cls
            n_ok += int(hit)
            err[mask] = 1.0 if hit else 0.0

        if n_tot == 0 and n_oos == 0:  # patch had no agronomic parcels; skip
            continue

        present = sorted(
            set(np.unique(truth[truth > 0]).tolist()) | set(np.unique(pred[pred > 0]).tolist())
        )
        cmap = ListedColormap([_CB_PALETTE[c - 1] for c in present])
        norm = BoundaryNorm(range(len(present) + 1), cmap.N)

        def _slot(class_map: np.ndarray, _present: list[int] = present) -> np.ndarray:
            out = np.full(class_map.shape, np.nan)
            for slot, cid in enumerate(_present):
                out[class_map == cid] = slot
            return out

        rgb = _rgb_from_s2(s2)
        # Error panel: 0->red (error), 0.5->grey (out of scope), 1->green (hit).
        err_cmap = ListedColormap([_ERR_RED, _OOS_GREY, _OK_GREEN])
        err_norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], err_cmap.N)
        mpp = patch_ground_resolution(pid, pastis_root)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), gridspec_kw={"wspace": 0.08})
        axes[0].imshow(rgb)
        axes[1].imshow(_slot(truth), cmap=cmap, norm=norm, interpolation="nearest")
        axes[2].imshow(_slot(pred), cmap=cmap, norm=norm, interpolation="nearest")
        axes[3].imshow(err, cmap=err_cmap, norm=err_norm, interpolation="nearest")
        # Panel titles are short and fixed-width so they never overlap; the model
        # name lives in the figure suptitle, not inside the narrow panel header.
        titles = (
            "(a) Sentinel-2 (RGB)",
            "(b) Verdad de campo",
            "(c) Prediccion",
            "(d) Aciertos y errores",
        )
        for ax, title in zip(axes, titles, strict=True):
            colour = _GREEN if title.startswith("(c)") else _INK
            ax.set_title(title, fontsize=12.5, fontweight="bold", color=colour, pad=10, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#cfcfcf")
                spine.set_linewidth(1.0)

        # Real scale bar: scale_bar_m / mpp pixels long.
        sb_px = scale_bar_m / mpp
        x0, y0 = 8, 120
        axes[0].plot(
            [x0, x0 + sb_px],
            [y0, y0],
            color="white",
            lw=4,
            path_effects=[pe.withStroke(linewidth=6, foreground="black")],
        )
        axes[0].text(
            x0 + sb_px / 2,
            y0 - 6,
            f"{int(scale_bar_m)} m",
            color="white",
            fontsize=10,
            ha="center",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        )

        # Crop legend: classes outside the model's scope are tagged with an
        # asterisk so the reader knows they appear in (b) but the model never
        # predicts them (and they are not scored).
        crop_handles = []
        for i, c in enumerate(present):
            oos = in_scope is not None and c not in in_scope
            lbl = names.get(c, str(c)) + (" *" if oos else "")
            crop_handles.append(Patch(facecolor=cmap(i), edgecolor="white", label=lbl))
        crop_title = (
            "Cultivos  (* = fuera del alcance del modelo)"
            if (in_scope is not None and n_oos)
            else "Cultivos"
        )
        fig.legend(
            handles=crop_handles,
            loc="lower center",
            ncol=min(7, len(crop_handles)),
            fontsize=9.5,
            frameon=False,
            bbox_to_anchor=(0.40, -0.06),
            title=crop_title,
            title_fontsize=10,
        )
        err_handles = [
            Patch(facecolor=_OK_GREEN, label="Acierto"),
            Patch(facecolor=_ERR_RED, label="Error"),
        ]
        if n_oos:
            err_handles.append(Patch(facecolor=_OOS_GREY, label="Fuera de alcance"))
        fig.legend(
            handles=err_handles,
            loc="lower center",
            ncol=len(err_handles),
            fontsize=9.5,
            frameon=False,
            bbox_to_anchor=(0.88, -0.06),
        )
        acc = n_ok / n_tot if n_tot else 0.0
        oos_note = f"  -  {n_oos} fuera de alcance" if n_oos else ""
        fig.suptitle(
            f"Escena {pid} - {model_label}: {n_ok}/{n_tot} parcelas correctas "
            f"({acc:.0%}){oos_note}  -  {mpp:.1f} m/px",
            fontsize=14.5,
            fontweight="bold",
            color=_INK,
            y=1.01,
        )
        path = out_dir / f"paper_inference_{pid}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        results.append(
            PaperInferenceMap(
                patch_id=str(pid),
                path=path,
                n_correct=n_ok,
                n_parcels=n_tot,
                n_out_of_scope=n_oos,
                ground_resolution_m=mpp,
            )
        )
    return results
