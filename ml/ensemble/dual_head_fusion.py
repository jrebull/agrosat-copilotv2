"""E-a -- Dual-head fusion TSViT-pheno + FarSLIP (US-041, EPIC 6).

The incremental ensemble ordered by the sponsor: fuse the strongest dense
segmenter (TSViT-pheno Full-M, US-039) with the phenology-contrastive FarSLIP
branch (US-036-b parcel-level student). FarSLIP no longer enters as a negative
ablation but as a POSITIVE member, via a learnable convex coefficient ``alpha``::

    P_fused = alpha * P_tsvit + (1 - alpha) * P_farslip   (alpha in [0, 1])

The TSViT side is the per-pixel ``(18, 128, 128)`` POST-softmax from the OOF dump
(``tsvit-pheno-fullm``, re-dumped in Fase 1). The FarSLIP side is built here: a
per-parcel CLS-768 embedding is broadcast over the parcel's pixels and scored by
cosine against 18 VISUAL class prototypes (mean CLS-768 per class over TRAIN
parcels), then softmaxed per pixel into a ``(18, 128, 128)`` map. ``alpha`` is
learned by maximizing F1-macro on OOF spatial sub-folds of fold-5 (anti-leakage,
``spatial_subfolds`` of :class:`EnsembleModel`). Metrics are reported on fold-5
ONLY (``EnsembleModel.HELD_OUT_FOLD``).

Correcciones verificadas en recon (plan US-041 Seccion "Correcciones"):

- **R-DIM-768**: the FarSLIP embedding is the 768-dim CLS
  (``CLIPVisionModel.last_hidden_state[:, 0, :]``), NOT 512. The 512 space
  (``visual_projection`` / ``pooler``) used by the embeddings extractor is a
  DIFFERENT (CLIP image-text shared) space and is not the student's training/loss
  space. The cosine and the prototypes live in 768.
- **R-PROTO-VISUAL**: :func:`build_class_prototypes` averages the VISUAL CLS-768
  embeddings of TRAIN parcels per real class, distinct from the checkpoint's
  MiniLM TEXT bank (``set_category_prototypes``).
- **R-PURITY**: the per-pixel broadcast assumes PASTIS-R parcel purity (~98%); it
  breaks on mixed margin pixels. Documented caveat, not a bug.
- **R-HONEST-GAIN**: FarSLIP-pheno does NOT beat AlphaEarth (F1-macro 0.555 vs
  0.645) and TSViT-pheno saturated (fold-5 delta -0.0033, noise); ``alpha`` is
  expected high (~0.85-0.95) and the phenology branch adds ~0.3 %, not 5 %.

Project conventions: ``polars`` (never pandas) for tabular access, ``numpy`` only
at the array boundary, ``torch`` for the student, ``structlog`` for logging, type
hints and Google-style docstrings; visible prose Spanish, code identifiers
English; real PASTIS-R French data only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
import torch.nn.functional as F

from ml.ensemble.base import EnsembleModel

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import geopandas as gpd

    from ml.farslip.parcel_crop_dataset import ParcelCropDataset

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_FARSLIP_CHECKPOINT",
    "DEFAULT_TSVIT_MEMBER",
    "DualHeadFusionHead",
    "build_class_prototypes",
    "farslip_cosine_map",
]

#: Default dense member: the strong segmenter re-dumped to Full-M in Fase 1.
DEFAULT_TSVIT_MEMBER: str = "tsvit-pheno-fullm"

#: Default parcel-level FarSLIP student (US-036-b). Relative path -> lands on F:
#: on the VM. Override per run via the constructor.
DEFAULT_FARSLIP_CHECKPOINT: str = "checkpoints/farslip/parcel/04cls/best.safetensors"

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: FarSLIP student CLS dimension (ViT-B/16 hidden_size). R-DIM-768: NOT 512.
_EMBED_DIM: int = 768

#: Class axis of a dense ``(18, H, W)`` softmax map.
_CLASS_AXIS: int = 0

#: PASTIS-R patch side (the OOF dump resamples to 128).
_PATCH_SIDE: int = 128

#: Float16 storage tolerance for the OOF dump (matches ``VotingEnsemble``).
_FLOAT16_SUM_TOL: float = 5e-3

#: Floor avoiding divide-by-zero when renormalizing a pixel's class vector.
_RENORM_EPS: float = 1e-12

#: Grid resolution for the alpha line search in ``[0, 1]`` (21 -> step 0.05).
_ALPHA_GRID: int = 21


def _resolve_device(device: str) -> torch.device:
    """Resolve ``"auto"`` to cuda when available, else the literal device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def build_class_prototypes(
    student: torch.nn.Module,
    dataset: ParcelCropDataset,
    *,
    class_ids: Sequence[int],
    device: str = "auto",
    batch_size: int = 32,
) -> np.ndarray:
    """Mean per class of L2-normalized CLS-768 FarSLIP embeddings over TRAIN parcels.

    For every parcel crop in ``dataset`` (TRAIN folds), runs the FarSLIP student,
    takes the CLS token (``last_hidden_state[:, 0, :]``, 768-dim, R-DIM-768),
    L2-normalizes it, and accumulates the mean per real ``class_id``. The returned
    bank rows follow ``class_ids`` order and are L2-normalized again so the cosine
    in :func:`farslip_cosine_map` is a plain dot product.

    These are VISUAL prototypes (R-PROTO-VISUAL): the mean of the student's image
    embeddings per class, NOT the MiniLM text bank the checkpoint was trained
    against. A class with no TRAIN parcel keeps a zero row (its cosine is 0 for
    every pixel, so it never wins the argmax -- an honest "unknown" rather than a
    fabricated prototype).

    Args:
        student: FarSLIP ``CLIPVisionModel`` student in ``eval()`` (4-band input).
        dataset: :class:`ParcelCropDataset` over the TRAIN folds (each item gives
            ``image (4, 224, 224)`` + RAW PASTIS ``class_id``).
        class_ids: Ordered PASTIS class ids (e.g. ``active_classes(18)`` -> 1..18).
            Row ``i`` of the bank corresponds to ``class_ids[i]``.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        batch_size: Inference batch size.

    Returns:
        ``(C, 768)`` ``float32`` prototype bank; row order == ``class_ids``; each
        non-empty row L2-normalized.

    Raises:
        ValueError: if ``dataset`` is empty or ``class_ids`` is empty.
    """
    from torch.utils.data import DataLoader

    from ml.farslip.parcel_crop_dataset import collate_parcel_batch

    ids = [int(c) for c in class_ids]
    if not ids:
        raise ValueError("class_ids cannot be empty.")
    if len(dataset) == 0:
        raise ValueError("dataset is empty; cannot build class prototypes.")

    dev = _resolve_device(device)
    student.eval()
    student.to(dev)

    row_of = {cid: i for i, cid in enumerate(ids)}
    sums = np.zeros((len(ids), _EMBED_DIM), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_parcel_batch,
    )
    for batch in loader:
        images = batch["images"].to(dev)
        cls = _student_cls(student, images)  # (B, 768) L2-normalized
        cls_np = cls.cpu().numpy().astype(np.float64)
        for emb, cid in zip(cls_np, batch["class_ids"].tolist(), strict=True):
            row = row_of.get(int(cid))
            if row is None:
                continue
            sums[row] += emb
            counts[row] += 1

    bank = np.zeros((len(ids), _EMBED_DIM), dtype=np.float32)
    for row, n in enumerate(counts):
        if n == 0:
            continue
        mean = sums[row] / float(n)
        norm = float(np.linalg.norm(mean))
        bank[row] = (mean / norm if norm > _RENORM_EPS else mean).astype(np.float32)
    logger.info(
        "farslip_class_prototypes",
        n_classes=len(ids),
        per_class_counts=counts.tolist(),
        empty_classes=[ids[i] for i, n in enumerate(counts) if n == 0],
    )
    return bank


def _student_cls(student: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Forward the student and return the L2-normalized CLS-768 token.

    Args:
        student: FarSLIP ``CLIPVisionModel`` student.
        images: ``(B, 4, 224, 224)`` parcel crops on the student's device.

    Returns:
        ``(B, 768)`` L2-normalized CLS embeddings (R-DIM-768).
    """
    out = student(pixel_values=images)
    cls = out.last_hidden_state[:, 0, :]  # (B, 768) CLS token
    return F.normalize(cls, p=2, dim=-1)


@torch.no_grad()
def farslip_cosine_map(
    student: torch.nn.Module,
    prototypes: np.ndarray,
    *,
    patch_id: str | int,
    dataset: ParcelCropDataset,
    parcel_ids_map: np.ndarray,
    class_ids: Sequence[int] | None = None,
    device: str = "auto",
) -> np.ndarray:
    """Per-pixel cosine vs visual prototypes via spatial broadcast of parcel CLS.

    Embeds every parcel crop of ``patch_id`` (one CLS-768 per parcel), scores each
    against the prototype bank (dot product of L2-normalized vectors == cosine),
    softmaxes per parcel over the bank's classes, and broadcasts it to all pixels
    of that parcel using ``parcel_ids_map``. The result is ALWAYS placed in the
    harness 18-class space: when the bank covers fewer classes (e.g. the N=4
    parcel-level FarSLIP), each class is scattered to its PASTIS slot
    (``class_id - 1``) and the absent classes stay near zero. This is what makes
    the map fusible with the TSViT ``(18, H, W)`` dense softmax (the fusion bug
    R-CLASSES-MISMATCH: a ``(4, H, W)`` map cannot broadcast onto ``(18, H, W)``).

    Background pixels (parcel id 0) and pixels of parcels absent from ``dataset``
    get a uniform distribution over the 18 classes (no information).

    R-PURITY: the broadcast assumes PASTIS-R parcel purity (~98%); mixed margin
    pixels inherit the parcel-level distribution.

    Args:
        student: FarSLIP ``CLIPVisionModel`` student in ``eval()``.
        prototypes: ``(C, 768)`` bank from :func:`build_class_prototypes`.
        patch_id: PASTIS-R patch id.
        dataset: :class:`ParcelCropDataset` (gives per-parcel crops + ids).
        parcel_ids_map: ``(128, 128)`` local ParcelIDs map (0 = Background), from
            :func:`ml.utils.parcel_reconcile.load_pastis_parcel_ids`.
        class_ids: PASTIS class ids (1..18) of the bank rows, in row order. When
            given (and the bank is not already 18-wide), the per-parcel softmax is
            scattered into the 18-class space at ``class_id - 1``. When ``None``
            the bank is assumed to already span the 18 classes.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.

    Returns:
        ``(18, 128, 128)`` POST-softmax map (sum-to-1 per pixel).

    Raises:
        ValueError: if ``prototypes`` shape is not ``(C, 768)`` or ``class_ids``
            length mismatches the bank rows.
    """
    proto = np.asarray(prototypes, dtype=np.float32)
    if proto.ndim != 2 or proto.shape[1] != _EMBED_DIM:
        raise ValueError(f"prototypes must be (C, {_EMBED_DIM}); got {proto.shape}.")
    n_bank = proto.shape[0]
    if class_ids is not None and len(class_ids) != n_bank:
        raise ValueError(
            f"class_ids length ({len(class_ids)}) must match prototype rows ({n_bank})."
        )
    # Row index in the bank -> 18-class slot. Identity when the bank is already
    # 18-wide and no class_ids given; otherwise scatter by PASTIS id (class-1).
    if class_ids is None:
        if n_bank != _NUM_CLASSES:
            raise ValueError(
                f"prototypes have {n_bank} rows but no class_ids given; cannot "
                f"place them in the {_NUM_CLASSES}-class space."
            )
        slots = list(range(_NUM_CLASSES))
    else:
        slots = [int(c) - 1 for c in class_ids]
        if any(not 0 <= s < _NUM_CLASSES for s in slots):
            raise ValueError(f"class_ids must be in [1, {_NUM_CLASSES}]; got {list(class_ids)}.")

    dev = _resolve_device(device)
    proto_t = torch.from_numpy(proto).to(dev)

    pid = str(patch_id)
    parcel_to_probs = _embed_patch_parcels(student, proto_t, dataset, pid, dev)

    side = parcel_ids_map.shape[0]
    uniform = np.full(_NUM_CLASSES, 1.0 / _NUM_CLASSES, dtype=np.float64)
    out = np.empty((_NUM_CLASSES, side, side), dtype=np.float64)
    out[:] = uniform[:, None, None]
    for local_id, probs in parcel_to_probs.items():
        mask = parcel_ids_map == local_id
        if not mask.any():
            continue
        # Scatter the bank softmax into the 18-class vector, then renormalize so
        # the pixel still sums to 1 (absent classes get 0).
        full = np.zeros(_NUM_CLASSES, dtype=np.float64)
        for row, slot in enumerate(slots):
            full[slot] = probs[row]
        total = full.sum()
        if total > 0:
            full /= total
        else:
            full = uniform
        out[:, mask] = full[:, None]
    return EnsembleModel.validate_probs(out, class_axis=_CLASS_AXIS, name=f"farslip_cosine:{pid}")


def _embed_patch_parcels(
    student: torch.nn.Module,
    proto_t: torch.Tensor,
    dataset: ParcelCropDataset,
    patch_id: str,
    dev: torch.device,
) -> dict[int, np.ndarray]:
    """Return ``{local_parcel_id: softmax_18}`` for every parcel of ``patch_id``.

    Args:
        student: FarSLIP student.
        proto_t: ``(C, 768)`` prototype bank on ``dev`` (L2-normalized rows).
        dataset: :class:`ParcelCropDataset` whose samples carry ``parcel_id`` as
            ``"{patch_id}_{local_id}"``.
        patch_id: target patch id.
        dev: torch device.

    Returns:
        Mapping ``{local_id (int): (C,) post-softmax numpy}``.
    """
    student.eval()
    student.to(dev)
    result: dict[int, np.ndarray] = {}
    for idx in range(len(dataset)):
        sample_pid, _src, local_id, _cat = dataset._samples[idx]
        if not sample_pid.startswith(f"{patch_id}_"):
            continue
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(dev)
        cls = _student_cls(student, image)  # (1, 768)
        logits = cls @ proto_t.t()  # (1, C) cosine (both normalized)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        result[int(local_id)] = probs.astype(np.float64)
    return result


class DualHeadFusionHead(EnsembleModel):
    """E-a: learnable convex fusion of TSViT-pheno dense softmax + FarSLIP cosine.

    ``P_fused = alpha * P_tsvit + (1 - alpha) * P_farslip``, ``alpha in [0, 1]``,
    learned by maximizing fold-5 F1-macro over geographic OOF sub-folds
    (:meth:`EnsembleModel.spatial_subfolds`, anti-leakage). Reports fold-5 ONLY
    (the base :meth:`EnsembleModel.evaluate` rejects any other fold). Subclasses
    :class:`EnsembleModel` to reuse ``validate_probs``, ``spatial_subfolds``,
    ``assert_oof_only``, ``compute_metrics``, ``evaluate``, ``log_to_mlflow``.

    Attributes:
        tsvit_member: OOF member name for the dense head (``tsvit-pheno-fullm``).
        farslip_checkpoint: parcel-level FarSLIP student checkpoint path.
        n_classes: number of agronomic classes (18).
        data_root: PASTIS-R root for the FarSLIP crops, parcel-id maps and GT.
    """

    def __init__(
        self,
        *,
        tsvit_member: str = DEFAULT_TSVIT_MEMBER,
        farslip_checkpoint: str | Path = DEFAULT_FARSLIP_CHECKPOINT,
        n_classes: int = _NUM_CLASSES,
        n_spatial_folds: int = 5,
        buffer_km: float = 1.0,
        device: str = "auto",
        data_root: Path | str | None = None,
        **kw: object,
    ) -> None:
        """Initialize the dual-head fusion ensemble.

        Args:
            tsvit_member: OOF member name of the dense head (default
                ``tsvit-pheno-fullm``, re-dumped in Fase 1).
            farslip_checkpoint: parcel-level FarSLIP student checkpoint (relative
                path lands on F: on the VM).
            n_classes: number of classes (default 18).
            n_spatial_folds: geographic OOF sub-folds for the alpha search.
            buffer_km: inter-sub-fold exclusion buffer (km).
            device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
            data_root: PASTIS-R root; ``None`` uses the dataset default.
            **kw: forwarded to :class:`EnsembleModel` (``oof_dir``, ``random_state``).
        """
        super().__init__(**kw)  # type: ignore[arg-type]
        self.tsvit_member = str(tsvit_member)
        self.farslip_checkpoint = Path(farslip_checkpoint)
        self.n_classes = int(n_classes)
        self.n_spatial_folds = int(n_spatial_folds)
        self.buffer_km = float(buffer_km)
        self.device = str(device)
        self.data_root = Path(data_root) if data_root is not None else None
        self._alpha: float | None = None
        self._prototypes: np.ndarray | None = None
        self._student: torch.nn.Module | None = None
        self._fold5_dataset: ParcelCropDataset | None = None
        logger.debug(
            "dual_head_init",
            tsvit_member=self.tsvit_member,
            farslip_checkpoint=str(self.farslip_checkpoint),
        )

    @property
    def alpha(self) -> float:
        """Learned fusion coefficient in ``[0, 1]`` (raises if not fitted)."""
        if self._alpha is None:
            raise RuntimeError("alpha is not set; call fit(...) first.")
        return self._alpha

    # ------------------------------------------------------------------
    # Lazy student + prototypes.
    # ------------------------------------------------------------------

    def _ensure_student(self) -> torch.nn.Module:
        """Load (once) the FarSLIP student via the embeddings-extractor loader."""
        if self._student is None:
            from ml.farslip.extract_embeddings import _load_student

            dev = _resolve_device(self.device)
            self._student = _load_student(self.farslip_checkpoint, device=dev)
        return self._student

    def _resolved_root(self) -> Path:
        """Return the PASTIS-R root (``data_root`` or the dataset default).

        ``load_pastis_parcel_ids`` requires a concrete root (no default), so this
        resolves ``None`` to the canonical ``data/PASTIS-R`` used by the datasets.
        """
        if self.data_root is not None:
            return self.data_root
        from ml.farslip.pastis_pair_dataset import _DEFAULT_PASTIS_ROOT

        return _DEFAULT_PASTIS_ROOT

    def set_prototypes(self, prototypes: np.ndarray) -> DualHeadFusionHead:
        """Inject a pre-built ``(C, 768)`` visual prototype bank (DRY for tests).

        Args:
            prototypes: ``(n_classes, 768)`` bank from
                :func:`build_class_prototypes`.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: if the bank shape is not ``(n_classes, 768)``.
        """
        proto = np.asarray(prototypes, dtype=np.float32)
        if proto.shape != (self.n_classes, _EMBED_DIM):
            raise ValueError(
                f"prototypes must be ({self.n_classes}, {_EMBED_DIM}); got {proto.shape}."
            )
        self._prototypes = proto
        return self

    # ------------------------------------------------------------------
    # Per-patch dense maps for each head.
    # ------------------------------------------------------------------

    def _tsvit_map(self, patch_id: str, tsvit_index: dict[str, np.ndarray]) -> np.ndarray:
        """Return the TSViT-pheno post-softmax ``(18, 128, 128)`` for a patch."""
        sm = tsvit_index.get(patch_id)
        if sm is None:
            raise ValueError(
                f"patch_id {patch_id!r} absent from member {self.tsvit_member!r} "
                "OOF; re-dump tsvit-pheno-fullm fold-5 (Fase 1)."
            )
        arr = np.asarray(sm, dtype=np.float64)
        self.validate_probs(
            arr,
            class_axis=_CLASS_AXIS,
            name=f"{self.tsvit_member}:{patch_id}",
            tol=_FLOAT16_SUM_TOL,
        )
        return _renormalize(arr)

    def _fold5_crop_dataset(self) -> ParcelCropDataset:
        """Build (once) the fold-5 :class:`ParcelCropDataset` for FarSLIP crops."""
        if self._fold5_dataset is None:
            from ml.farslip.parcel_crop_dataset import ParcelCropDataset
            from ml.farslip.pastis_pair_dataset import active_classes

            ds_kwargs: dict[str, object] = {
                "captions": {},
                "folds": (self.HELD_OUT_FOLD,),
                "active_class_ids": active_classes(self.n_classes),
            }
            if self.data_root is not None:
                ds_kwargs["root"] = self.data_root
            self._fold5_dataset = ParcelCropDataset(**ds_kwargs)  # type: ignore[arg-type]
        return self._fold5_dataset

    def _farslip_map(self, patch_id: str) -> np.ndarray:
        """Build the FarSLIP cosine ``(18, 128, 128)`` for a patch (lazy)."""
        from ml.utils.parcel_reconcile import load_pastis_parcel_ids

        if self._prototypes is None:
            raise RuntimeError(
                "prototypes are not set; call build_class_prototypes + "
                "set_prototypes, or fit(...) which builds them."
            )
        from ml.farslip.pastis_pair_dataset import active_classes

        student = self._ensure_student()
        dataset = self._fold5_crop_dataset()
        parcel_ids_map = load_pastis_parcel_ids(patch_id, self._resolved_root())
        # The bank covers active_classes(n_classes) in row order (e.g. N=4 -> 4
        # rows). Pass those ids so the map is scattered into the 18-class space
        # and stays fusible with the TSViT (18, H, W) softmax.
        return farslip_cosine_map(
            student,
            self._prototypes,
            patch_id=patch_id,
            dataset=dataset,
            parcel_ids_map=parcel_ids_map,
            class_ids=active_classes(self.n_classes),
            device=self.device,
        )

    def _fuse(self, p_tsvit: np.ndarray, p_farslip: np.ndarray, alpha: float) -> np.ndarray:
        """Convex per-pixel fusion ``alpha*P_tsvit + (1-alpha)*P_farslip``."""
        fused = alpha * p_tsvit + (1.0 - alpha) * p_farslip
        return self.validate_probs(_renormalize(fused), class_axis=_CLASS_AXIS, name="fused")

    # ------------------------------------------------------------------
    # fit: learn alpha via OOF spatial sub-folds (anti-leakage).
    # ------------------------------------------------------------------

    def fit(
        self,
        patch_ids: Sequence[str],
        parcel_geoms: gpd.GeoDataFrame,
        *,
        prototype_dataset: ParcelCropDataset | None = None,
    ) -> DualHeadFusionHead:
        """Learn ``alpha`` on OOF spatial sub-folds of fold-5 (anti-leakage).

        Builds the visual prototypes (if not injected) from ``prototype_dataset``
        (TRAIN folds), partitions the fold-5 parcels geographically via
        :meth:`EnsembleModel.spatial_subfolds`, and for each sub-fold searches the
        ``alpha`` grid that maximizes F1-macro on the held-out sub-fold parcels.
        The final ``alpha`` is the mean over sub-folds. ``assert_oof_only`` guards
        that the alpha picked for a sub-fold never used its own held-out parcels.

        Args:
            patch_ids: fold-5 patch ids to fuse and score.
            parcel_geoms: GeoDataFrame of fold-5 parcels with the integer
                ``parcel_id`` surrogate, the ``canonical_parcel_id``
                (``"{patch}_{local}"``) and ``geometry`` (EPSG:4326). The
                surrogate drives the geographic partition; the canonical id maps
                a sub-fold back to the parcel's pixels (mirror of US-040 blending).
            prototype_dataset: TRAIN-fold :class:`ParcelCropDataset` to build the
                visual prototypes. Required unless ``set_prototypes`` was called.

        Returns:
            ``self`` with ``alpha`` learned.

        Raises:
            ValueError: if prototypes are missing and no ``prototype_dataset`` is
                given, if ``patch_ids`` is empty, or if ``parcel_geoms`` lacks the
                ``canonical_parcel_id`` column.
        """
        ids = [str(p) for p in patch_ids]
        if not ids:
            raise ValueError("fit needs at least one patch_id.")
        if "canonical_parcel_id" not in parcel_geoms.columns:
            raise ValueError(
                "parcel_geoms must carry 'canonical_parcel_id' ('{patch}_{local}') "
                "to map spatial sub-folds back to parcel pixels."
            )
        surrogate_to_canonical = {
            int(s): str(c)
            for s, c in zip(
                parcel_geoms["parcel_id"].tolist(),
                parcel_geoms["canonical_parcel_id"].tolist(),
                strict=True,
            )
        }
        if self._prototypes is None:
            if prototype_dataset is None:
                raise ValueError(
                    "prototypes are not set and no prototype_dataset given; "
                    "pass a TRAIN-fold ParcelCropDataset or call set_prototypes."
                )
            from ml.farslip.pastis_pair_dataset import active_classes

            self._prototypes = build_class_prototypes(
                self._ensure_student(),
                prototype_dataset,
                class_ids=active_classes(self.n_classes),
                device=self.device,
            )

        # Precompute both heads once per patch (alpha search is cheap arithmetic).
        tsvit_index = self._load_tsvit_index()
        p_tsvit = {pid: self._tsvit_map(pid, tsvit_index) for pid in ids}
        p_farslip = {pid: self._farslip_map(pid) for pid in ids}

        subfolds = self.spatial_subfolds(
            parcel_geoms, n_folds=self.n_spatial_folds, buffer_km=self.buffer_km
        )
        grid = np.linspace(0.0, 1.0, _ALPHA_GRID)
        per_fold_alpha: list[float] = []
        for fa in subfolds:
            # Held-out sub-fold = test_ids; the rest (train|val) is the OOF
            # context, mirroring StackingEnsemble. assert_oof_only guards leakage.
            # Translate the integer surrogate ids back to canonical "{patch}_{local}".
            held_ids = {
                surrogate_to_canonical[int(x)]
                for x in fa.test_ids
                if int(x) in surrogate_to_canonical
            }
            train_ids = {
                surrogate_to_canonical[int(x)]
                for x in (set(fa.train_ids) | set(fa.val_ids))
                if int(x) in surrogate_to_canonical
            }
            self.assert_oof_only(list(train_ids), list(held_ids), context="alpha-subfold")
            best_alpha, best_f1 = self._search_alpha(ids, p_tsvit, p_farslip, held_ids, grid)
            per_fold_alpha.append(best_alpha)
            logger.info(
                "dual_head_subfold_alpha",
                best_alpha=round(best_alpha, 3),
                best_f1=round(best_f1, 4),
                n_held_parcels=len(held_ids),
            )
        self._alpha = float(np.mean(per_fold_alpha)) if per_fold_alpha else 0.5
        logger.info(
            "dual_head_alpha_learned",
            alpha=round(self._alpha, 4),
            per_fold=[round(a, 3) for a in per_fold_alpha],
        )
        return self

    def _search_alpha(
        self,
        patch_ids: Sequence[str],
        p_tsvit: dict[str, np.ndarray],
        p_farslip: dict[str, np.ndarray],
        held_parcel_ids: set[str],
        grid: np.ndarray,
    ) -> tuple[float, float]:
        """Grid-search alpha maximizing F1-macro on held-out sub-fold parcels.

        Args:
            patch_ids: fold-5 patch ids.
            p_tsvit/p_farslip: ``{patch_id: (18,128,128)}`` precomputed heads.
            held_parcel_ids: parcel ids of the held-out sub-fold (``"{pid}_{lid}"``).
            grid: alpha candidates in ``[0, 1]``.

        Returns:
            ``(best_alpha, best_f1_macro)``.
        """
        from ml.utils.parcel_reconcile import load_pastis_parcel_ids

        # Restrict scoring to patches that contain a held-out parcel.
        held_patches = sorted({pid.rsplit("_", 1)[0] for pid in held_parcel_ids})
        held_patches = [p for p in held_patches if p in p_tsvit]
        if not held_patches:
            return 0.5, 0.0

        gt = {pid: self._ground_truth_patch(pid) for pid in held_patches}
        root = self._resolved_root()
        pid_maps = {pid: load_pastis_parcel_ids(pid, root) for pid in held_patches}
        local_held: dict[str, set[int]] = {}
        for parcel in held_parcel_ids:
            base, _, lid = parcel.rpartition("_")
            if base in pid_maps:
                local_held.setdefault(base, set()).add(int(lid))

        best_alpha, best_f1 = 0.5, -1.0
        for alpha in grid:
            yt_all, yp_all = [], []
            for pid in held_patches:
                fused = self._fuse(p_tsvit[pid], p_farslip[pid], float(alpha))
                pred = fused.argmax(axis=_CLASS_AXIS)  # (128,128)
                mask = np.isin(pid_maps[pid], list(local_held.get(pid, set())))
                if not mask.any():
                    continue
                yt_all.append(gt[pid][mask])
                yp_all.append(pred[mask])
            if not yt_all:
                continue
            metrics = self.compute_metrics(
                np.concatenate(yt_all),
                np.concatenate(yp_all),
                num_classes=self.n_classes,
            )
            if metrics["f1_macro"] > best_f1:
                best_f1, best_alpha = metrics["f1_macro"], float(alpha)
        return best_alpha, best_f1

    # ------------------------------------------------------------------
    # predict / evaluate.
    # ------------------------------------------------------------------

    def predict_proba(self, patch_ids: Sequence[str]) -> np.ndarray:
        """Return fused POST-softmax maps aligned to ``patch_ids``.

        Args:
            patch_ids: fold-5 patch ids.

        Returns:
            ``(18, 128, 128)`` for one id, else ``(N, 18, 128, 128)``.

        Raises:
            RuntimeError: if ``alpha`` is not learned (call :meth:`fit`).
        """
        ids = [str(p) for p in patch_ids]
        if not ids:
            raise ValueError("predict_proba needs at least one patch_id.")
        alpha = self.alpha  # raises if unfitted
        tsvit_index = self._load_tsvit_index()
        fused = [
            self._fuse(self._tsvit_map(pid, tsvit_index), self._farslip_map(pid), alpha)
            for pid in ids
        ]
        stacked = np.stack(fused, axis=0)
        logger.info(
            "dual_head_predict_proba",
            alpha=round(alpha, 4),
            n_patches=len(ids),
            shape=tuple(stacked.shape),
        )
        return stacked[0] if stacked.shape[0] == 1 else stacked

    def predict(self, patch_ids: Sequence[str]) -> np.ndarray:
        """Hard per-pixel labels: argmax over the fused class axis."""
        proba = self.predict_proba(patch_ids)
        class_axis = 0 if proba.ndim == 3 else 1
        return proba.argmax(axis=class_axis).astype(np.int64)

    def evaluate_patches(
        self, patch_ids: Sequence[str], *, fold: int = EnsembleModel.HELD_OUT_FOLD
    ) -> dict[str, float]:
        """Fuse ``patch_ids`` and score against the fold-5 semantic18 ground truth.

        Args:
            patch_ids: fold-5 patch ids to fuse and score.
            fold: must be the held-out fold 5 (delegated guard).

        Returns:
            ``{"f1_macro": float, "accuracy": float}`` over fold-5 pixels.
        """
        ids = [str(p) for p in patch_ids]
        y_true = np.stack([self._ground_truth_patch(p) for p in ids], axis=0)
        y_pred = self.predict(ids)
        return self.evaluate(
            y_true=np.asarray(y_true).reshape(-1),
            y_pred=np.asarray(y_pred).reshape(-1),
            fold=fold,
        )

    def mlflow_params(self) -> dict[str, object]:
        """Params for the MLflow run (``ensemble-Ea-tsvit-pheno-farslip``)."""
        return {
            "alpha": self._alpha,
            "tsvit_member": self.tsvit_member,
            "farslip_checkpoint": str(self.farslip_checkpoint),
            "n_spatial_folds": self.n_spatial_folds,
            "embedding_dim": _EMBED_DIM,
            "n_classes": self.n_classes,
        }

    # ------------------------------------------------------------------
    # Helpers (OOF index + ground truth, mirror VotingEnsemble).
    # ------------------------------------------------------------------

    def _load_tsvit_index(self) -> dict[str, np.ndarray]:
        """Load and index the TSViT-pheno-fullm pixel OOF (``{patch_id: softmax}``)."""
        loaded = self.load_oof_members([self.tsvit_member], space="pixel")
        df = loaded[self.tsvit_member]
        if "patch_id" not in df.columns or "softmax" not in df.columns:
            raise ValueError(
                f"member {self.tsvit_member!r} pixel OOF must carry 'patch_id' "
                f"and 'softmax'; got {df.columns}."
            )
        index: dict[str, np.ndarray] = {}
        for pid, sm in zip(df["patch_id"].to_list(), df["softmax"].to_list(), strict=True):
            if sm is not None:
                index[str(pid)] = np.asarray(sm)
        return index

    def _ground_truth_patch(self, patch_id: str) -> np.ndarray:
        """Load the PASTIS-R semantic18 fold-5 label map ``(128, 128)`` for a patch."""
        from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

        ds_kwargs: dict[str, object] = {
            "folds": (self.HELD_OUT_FOLD,),
            "collapse_time": "median",
            "target": "semantic18",
            "ignore_index": 255,
        }
        if self.data_root is not None:
            ds_kwargs["root"] = self.data_root
        dataset = PASTISSegmentationDataset(**ds_kwargs)  # type: ignore[arg-type]
        pos_of = {pid: i for i, pid in enumerate(dataset.patch_ids)}
        pos = pos_of.get(str(patch_id))
        if pos is None:
            raise ValueError(f"patch_id {patch_id!r} not in fold-{self.HELD_OUT_FOLD} split.")
        _x, y = dataset[pos]
        return np.asarray(y, dtype=np.int64)


def _renormalize(softmax_18: np.ndarray) -> np.ndarray:
    """Renormalize a ``(18, H, W)`` map so each pixel's class axis sums to 1."""
    denom = softmax_18.sum(axis=_CLASS_AXIS, keepdims=True)
    return softmax_18 / np.where(denom < _RENORM_EPS, 1.0, denom)
