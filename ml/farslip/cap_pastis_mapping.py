"""CAP-32 -> PASTIS-18 cardinality bridge for FarSLIP prototypes (US-034).

The FarSLIP contrastive loss (:class:`ml.farslip.distill.RegionCategoryAlignmentLoss`)
expects ``n_regions * n_categories`` text prototypes, where ``n_categories = 32``
(the Italian CAP vocabulary of ``ml/farslip/cap_vocabulary.yaml``). The
phenological prototypes of US-033
(``data/features/phenology_class_prototypes_pastis.parquet``) only cover the **18
PASTIS-R crop classes** (class_id 1..18). Tiling the 18 prototypes by region
yields ``(54, D)``, not the ``(96, D) = (3 * 32, D)`` the loss requires ->
``ValueError``.

This module is the missing semantic bridge: an explicit, documented map from
each of the 32 CAP categories to the closest PASTIS-18 crop class, plus
:func:`expand_to_cap`, which materializes a ``(32, D)`` matrix where each CAP
category receives the embedding of its mapped PASTIS prototype. CAP categories
without a clean PASTIS analogue (e.g. ``olivo``, ``riso``, ``agrumi``,
``tabacco``) fall back to the nearest agronomic PASTIS class, documented in
:data:`CAP_TO_PASTIS`.

The mapping is an artifact aside from ``cap_vocabulary.yaml`` (which is NOT
edited, per ml/CLAUDE.md "No tocar") and aside from the US-033 parquet (which is
only read). It is the puente CAP-italiano <-> PASTIS-R that the repo lacked.

PASTIS-18 reference (``data/reference/pastis_class_mapping.json``):
    1 Meadow, 2 Soft winter wheat, 3 Corn, 4 Winter barley, 5 Winter rapeseed,
    6 Spring barley, 7 Sunflower, 8 Grapevine, 9 Beet, 10 Winter triticale,
    11 Winter durum wheat, 12 Fruits/vegetables/flowers, 13 Potatoes,
    14 Leguminous fodder, 15 Soybeans, 16 Orchard, 17 Mixed cereal, 18 Sorghum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

#: Explicit CAP-32 -> PASTIS-18 class_id mapping (US-034 R-CARD / R-FALLBACK).
#:
#: Keys are the 32 CAP class slugs of ``cap_vocabulary.yaml`` (Italian agronomic
#: vocabulary). Values are the PASTIS-R ``class_id`` (1..18) whose phenological
#: prototype best matches each CAP category. Direct matches map to their PASTIS
#: twin; CAP categories without a clean PASTIS analogue fall back to the closest
#: agronomic PASTIS class, with the rationale documented inline (NEVER a silent
#: ``class 0`` / Background). The fallbacks are an explicit approximation (caveat
#: US-034 AC-8); US-037 measures their impact on FarSLIP retrieval.
CAP_TO_PASTIS: dict[str, int] = {
    # --- Direct PASTIS analogues -------------------------------------------
    "mais": 3,  # Corn
    "frumento": 2,  # Soft winter wheat
    "vite": 8,  # Grapevine
    "riso": 1,  # rice -> Meadow (flooded herbaceous; no PASTIS rice, see note)
    "foraggio": 14,  # Leguminous fodder
    "ortaggi": 12,  # Fruits, vegetables, flowers
    "girasole": 7,  # Sunflower
    "soia": 15,  # Soybeans
    "colza": 5,  # Winter rapeseed
    "orzo": 4,  # Winter barley
    "sorgo": 18,  # Sorghum
    "pomodoro": 12,  # tomato -> Fruits, vegetables, flowers
    "patata": 13,  # Potatoes
    "barbabietola": 9,  # Beet
    "leguminose": 14,  # legumes -> Leguminous fodder
    # --- Permanent / woody crops -> Orchard or Grapevine -------------------
    "olivo": 16,  # olive grove -> Orchard (woody permanent, no PASTIS olive)
    "agrumi": 16,  # citrus orchard -> Orchard
    "melo": 16,  # apple orchard -> Orchard
    "pero": 16,  # pear orchard -> Orchard
    "pesco": 16,  # peach orchard -> Orchard
    "mandorlo": 16,  # almond orchard -> Orchard
    "noce": 16,  # walnut orchard -> Orchard
    # --- Grassland / set-aside -> Meadow -----------------------------------
    "prato_permanente": 1,  # permanent grassland -> Meadow
    "pascolo": 1,  # pasture -> Meadow
    "set_aside": 1,  # set-aside (fallow herbaceous) -> Meadow (neutral)
    # --- Special / horticultural crops -> Fruits, vegetables, flowers -------
    "tabacco": 12,  # tobacco -> Fruits, vegetables, flowers (no PASTIS analogue)
    "lino": 12,  # flax -> Fruits, vegetables, flowers (industrial herbaceous)
    "canapa": 12,  # hemp -> Fruits, vegetables, flowers (industrial herbaceous)
    "serra": 12,  # greenhouse -> Fruits, vegetables, flowers (horticulture)
    "vivai": 12,  # plant nursery -> Fruits, vegetables, flowers
    "floricoltura": 12,  # floriculture -> Fruits, vegetables, flowers
    # --- Catch-all neutral -> Meadow ---------------------------------------
    "altro": 1,  # generic agricultural crop -> Meadow (neutral, documented)
}


def load_cap_to_pastis() -> dict[str, int]:
    """Returns the explicit CAP-32 -> PASTIS-18 class_id mapping.

    The mapping covers all 32 CAP categories of ``cap_vocabulary.yaml`` with
    direct PASTIS analogues where they exist and documented fallbacks (nearest
    agronomic PASTIS class) for CAP categories without a clean PASTIS twin
    (olive, citrus, tobacco, flax, hemp, greenhouse, nursery, floriculture,
    set-aside, generic). See :data:`CAP_TO_PASTIS` for the per-class rationale.

    Returns:
        A copy of :data:`CAP_TO_PASTIS` (``{cap_slug: pastis_class_id}``), where
        every value is a valid PASTIS crop class_id in ``1..18``.
    """
    return dict(CAP_TO_PASTIS)


def expand_to_cap(
    proto_pastis: np.ndarray,
    cap_classes: Sequence[str],
    *,
    mapping: dict[str, int] | None = None,
    pastis_class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Maps each CAP category to its PASTIS-18 prototype embedding.

    Builds a ``(len(cap_classes), D)`` matrix where row ``i`` holds the PASTIS
    prototype assigned to ``cap_classes[i]`` via :data:`CAP_TO_PASTIS`. The row
    order MUST match the dataset's CAP category indexing (``category_id``
    ``0..n_categories-1``, i.e. the canonical ``all_cap_classes`` order produced
    in ``ml/farslip/train.py``), otherwise each visual CLS would align with the
    wrong prototype with no error (silent loss degradation, US-034 R-ORDER).

    Args:
        proto_pastis: ``(P, D)`` prototype matrix from
            :func:`ml.features.phenology_class_prototypes.load_class_prototype_embeddings`
            (P == 18, D == 384 for MiniLM).
        cap_classes: CAP category slugs in the dataset's canonical order. Each
            slug must be a key of ``mapping``.
        mapping: CAP-slug -> PASTIS class_id map. Defaults to
            :data:`CAP_TO_PASTIS`.
        pastis_class_ids: class_id of each row of ``proto_pastis`` (the second
            element returned by ``load_class_prototype_embeddings``). Defaults to
            ``[1, 2, ..., P]`` (the canonical PASTIS crop order 1..18).

    Returns:
        ``(len(cap_classes), D)`` float32 matrix; row ``i`` is the prototype of
        the PASTIS class mapped from ``cap_classes[i]``.

    Raises:
        ValueError: if ``proto_pastis`` is not 2-D, if a CAP slug is missing from
            ``mapping``, or if a mapped PASTIS class_id is absent from
            ``pastis_class_ids``.
    """
    if proto_pastis.ndim != 2:
        raise ValueError(f"proto_pastis must be 2-D (P, D); got shape {proto_pastis.shape}")
    resolved_mapping = mapping if mapping is not None else CAP_TO_PASTIS
    n_pastis, dim = proto_pastis.shape
    if pastis_class_ids is None:
        # Canonical PASTIS crop order: class_id 1..P aligned with rows 0..P-1.
        class_ids: list[int] = list(range(1, n_pastis + 1))
    else:
        class_ids = list(pastis_class_ids)
        if len(class_ids) != n_pastis:
            raise ValueError(f"pastis_class_ids len={len(class_ids)} != proto rows={n_pastis}")
    class_id_to_row = {cid: row for row, cid in enumerate(class_ids)}

    out = np.empty((len(cap_classes), dim), dtype=np.float32)
    for i, cap_slug in enumerate(cap_classes):
        if cap_slug not in resolved_mapping:
            raise ValueError(
                f"CAP class {cap_slug!r} (index {i}) has no entry in the "
                f"CAP->PASTIS mapping. Add it to CAP_TO_PASTIS."
            )
        pastis_id = resolved_mapping[cap_slug]
        if pastis_id not in class_id_to_row:
            raise ValueError(
                f"CAP class {cap_slug!r} maps to PASTIS class_id={pastis_id}, "
                f"absent from the prototype matrix class_ids {class_ids}."
            )
        out[i] = proto_pastis[class_id_to_row[pastis_id]]

    logger.info(
        "cap_prototypes_expanded",
        n_cap=len(cap_classes),
        n_pastis=n_pastis,
        emb_dim=dim,
        n_fallback_orchard=sum(1 for c in cap_classes if resolved_mapping[c] == 16),
    )
    return out
