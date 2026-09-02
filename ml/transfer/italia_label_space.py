"""Italian transfer label space + kept-class flag (US-079).

Generalizes :mod:`ml.transfer.finetune_baltico` (``BalticLabelSpace`` +
``warm_start_head``) to the Mediterranean homologue of US-078. The dense champion
members (TSViT-pheno, U-TAE) were trained on PASTIS-R France (18 semantic
classes); to transfer them to Italy 2018 the target label space must:

1. Reuse the Italian dense classes materialised by the US-078 builder
   (``data/pastis_italia_2018/class_mapping.json`` / ``class_table.parquet``),
   keeping the contiguous ids ``[1, K]`` with id ``0`` = background.
2. Mark the CONSERVED classes -- the Italian leaves that map to a PASTIS-18 class
   (e.g. ``vineyards -> Grapevine``, ``permanent_grassland -> Meadow``,
   ``durum_hard_wheat -> Winter durum wheat``) -- whose head rows are
   WARM-STARTED from the PASTIS head (the "kept-class flag"), versus the NEW
   Mediterranean leaves (``olive``, ``tree_wood_forest``, ...) that PASTIS never
   saw and start random. This is the exact pattern of
   :func:`ml.transfer.finetune_baltico.warm_start_head`.
3. Provide a fine -> coarse collapse map so the transfer can be evaluated at
   BOTH the FINE (Italian) level and a COARSE level shared with PASTIS (the
   hierarchical "papaya/fruits" eval of US-079).

Honesty
-------
- The conserved mapping is the agronomically defensible crosswalk between the
  Italian HCAT4 leaves and the PASTIS-18 names; a leaf with no PASTIS counterpart
  is genuinely NEW and is reported as such (its F1 measures what the French
  backbone learns about a class it never saw).
- The label space is BUILT FROM the on-disk class table (not hardcoded), so it
  stays consistent with whatever ``min_support`` US-078 used to materialise the
  dataset; classes absent from the table simply never appear.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "CONSERVED_LEAF_TO_PASTIS",
    "FINE_TO_COARSE",
    "ItaliaLabelSpace",
    "build_italia_label_space",
    "stratified_pixel_patch_sample",
    "warm_start_head",
]

#: Repo root (this file is ``<root>/ml/transfer/italia_label_space.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default US-078 homologue dataset root (where ``class_mapping.json`` lives).
DEFAULT_ITALIA_ROOT: Path = _REPO_ROOT / "data" / "pastis_italia_2018"

#: Conserved classes: Italian HCAT4 leaf -> PASTIS-18 class name (the
#: "kept-class flag"). These rows of the new head are warm-started from the
#: PASTIS head so the model retains what France already taught it. The mapping is
#: the agronomic crosswalk verified against ``class_mapping.json`` (US-078) and
#: ``ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`` (PASTIS names). Italian leaves
#: NOT listed here are genuinely new Mediterranean classes (olive, forest, ...)
#: and start random.
CONSERVED_LEAF_TO_PASTIS: dict[str, str] = {
    "vineyards_wine_vine_rebland_grapes": "Grapevine",
    "permanent_grassland": "Meadow",
    "durum_hard_wheat": "Winter durum wheat",
    "common_soft_wheat": "Soft winter wheat",
    "barley": "Winter barley",
    "maize_corn_popcorn": "Corn",
    "sunflower": "Sunflower",
    "alfalfa_lucerne": "Leguminous fodder",
    "clover": "Leguminous fodder",
    "sorghum": "Sorghum",
    "potatoes": "Potatoes",
    "oats": "Mixed cereal",
    "spelt": "Mixed cereal",
    "triticale": "Winter triticale",
    "beans": "Leguminous fodder",
    "fresh_vegetables": "Fruits, vegetables, flowers",
    "apples": "Orchard",
    "peach": "Orchard",
    "plums": "Orchard",
    "pears": "Orchard",
}

#: Collapse map for the hierarchical eval: a FINE Italian leaf -> the COARSE
#: bucket a model without the granularity would use. The conserved leaves collapse
#: to their PASTIS parent (so the fine prediction is scorable at the level PASTIS
#: labels); the genuinely-new Mediterranean leaves collapse to an explicit coarse
#: agronomic group. A leaf absent here keeps its own name as its coarse label
#: (it is already its coarsest sensible bucket, e.g. ``olive``).
FINE_TO_COARSE: dict[str, str] = {
    # New Mediterranean leaves -> coarse agronomic group.
    "olive": "Permanent woody crop",
    "tree_wood_forest": "Forest",
    "other_tree_wood_forest": "Forest",
    "sweet_chestnuts": "Forest",
    "unmaintained": "Unmaintained / fallow",
    "not_known_and_other": "Other",
    "other": "Other",
    "arable_crops": "Arable land",
    "poaceae_grasses": "Meadow",
    "flowers_ornamental_plants": "Fruits, vegetables, flowers",
    "kitchen_gardens": "Fruits, vegetables, flowers",
    "nurseries_nursery": "Permanent woody crop",
    "chickpeas": "Leguminous fodder",
    "tomato": "Fruits, vegetables, flowers",
    "artichoke": "Fruits, vegetables, flowers",
    "lolium_ryegrass": "Meadow",
    "tobacco": "Other",
    "vetches": "Leguminous fodder",
    "onobrychis_sainfoins": "Leguminous fodder",
}


@dataclass
class ItaliaLabelSpace:
    """The Italian fine-tune target label space (conserved + new), PASTIS-mapped.

    Generalizes :class:`ml.transfer.finetune_baltico.BalticLabelSpace`: instead of
    per-parcel Baltic leaves it carries the DENSE Italian class ids materialised by
    the US-078 builder (id ``0`` = background, ``[1, K]`` = crops). The dense head
    therefore has ``num_classes = K + 1`` outputs (background included), and the
    warm-start operates on the conserved class ids.

    Attributes:
        leaves: All Italian leaf names in id order (index = ``class_id``); index 0
            is the reserved background sentinel ``"__background__"``.
        class_ids: The crop class ids ``[1, K]`` (background excluded).
        conserved: Subset of leaves that maps to a PASTIS-18 class (warm-started).
        new: Subset that is genuinely new Mediterranean (random init).
        leaf_to_pastis: Conserved leaf -> PASTIS-18 class name.
        index: leaf name -> dense class id (``leaves`` is its inverse).
        num_classes: Dense head size = ``len(leaves)`` = ``K + 1`` (incl.
            background).
        background_id: The background sentinel id (always 0).
    """

    leaves: tuple[str, ...]
    class_ids: tuple[int, ...]
    conserved: tuple[str, ...]
    new: tuple[str, ...]
    leaf_to_pastis: dict[str, str]
    index: dict[str, int] = field(default_factory=dict)
    background_id: int = 0

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {leaf: i for i, leaf in enumerate(self.leaves)}

    @property
    def num_classes(self) -> int:
        """Dense head size (background + ``K`` crops)."""
        return len(self.leaves)

    def id_to_leaf(self) -> dict[int, str]:
        """Return the inverse map ``dense class id -> leaf name``."""
        return {i: leaf for leaf, i in self.index.items()}

    def coarse_of(self, leaf: str) -> str:
        """Collapse a fine leaf to its coarse (PASTIS-comparable) bucket.

        A conserved leaf collapses to its PASTIS parent; a new leaf to its
        explicit agronomic group (:data:`FINE_TO_COARSE`); a leaf with no entry
        keeps its own name (already its coarsest bucket).

        Args:
            leaf: A fine Italian leaf name.

        Returns:
            The coarse bucket name used for the hierarchical eval.
        """
        if leaf in self.leaf_to_pastis:
            return self.leaf_to_pastis[leaf]
        return FINE_TO_COARSE.get(leaf, leaf)


def _load_class_table(italia_root: Path) -> pl.DataFrame:
    """Load the US-078 ``class_table.parquet`` (``class_id``, ``hcat4_name``).

    Args:
        italia_root: The homologue dataset root (US-078 output).

    Returns:
        The Polars class table sorted by ``class_id`` (crop ids ``[1, K]``).

    Raises:
        FileNotFoundError: if neither ``class_table.parquet`` nor
            ``class_mapping.json`` is present (the US-078 builder must run first).
    """
    table_path = italia_root / "class_table.parquet"
    if table_path.is_file():
        return pl.read_parquet(table_path).sort("class_id")
    mapping_path = italia_root / "class_mapping.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        return pl.DataFrame(mapping["classes"]).sort("class_id")
    raise FileNotFoundError(
        f"no class_table.parquet / class_mapping.json under {italia_root}; run the "
        "US-078 builder (scripts/build_italia_pastis.py) first."
    )


def build_italia_label_space(
    *,
    italia_root: Path = DEFAULT_ITALIA_ROOT,
    conserved: dict[str, str] = CONSERVED_LEAF_TO_PASTIS,
    background_name: str = "__background__",
) -> ItaliaLabelSpace:
    """Assemble the Italian dense label space from the US-078 class table.

    Reads the contiguous Italian class ids materialised by US-078 and tags each
    crop leaf as CONSERVED (maps to a PASTIS-18 name in ``conserved`` -> the head
    row is warm-started) or NEW (Mediterranean leaf PASTIS never saw -> random
    init). The dense head keeps the background id ``0`` so the model's class axis
    aligns 1:1 with the on-disk ``TARGET_<id>.npy`` masks.

    Args:
        italia_root: The US-078 homologue dataset root.
        conserved: Italian leaf -> PASTIS-18 class name (the kept-class flag).
        background_name: Sentinel name for the reserved id-0 background row.

    Returns:
        An :class:`ItaliaLabelSpace` whose ``leaves`` index equals the dense
        ``class_id`` (index 0 = background).
    """
    table = _load_class_table(italia_root)
    crop_ids = [int(c) for c in table["class_id"].to_list()]
    crop_names = [str(n) for n in table["hcat4_name"].to_list()]
    max_id = max(crop_ids) if crop_ids else 0

    # Build the id-ordered leaf list with index 0 = background; any gap in the
    # crop ids (defensive) is filled with a placeholder so indexing stays dense.
    id_to_name: dict[int, str] = dict(zip(crop_ids, crop_names, strict=True))
    leaves: list[str] = [background_name]
    for cid in range(1, max_id + 1):
        leaves.append(id_to_name.get(cid, f"__unused_{cid}__"))

    conserved_leaves = tuple(name for name in crop_names if name in conserved)
    new_leaves = tuple(name for name in crop_names if name not in conserved)
    leaf_to_pastis = {leaf: conserved[leaf] for leaf in conserved_leaves}

    space = ItaliaLabelSpace(
        leaves=tuple(leaves),
        class_ids=tuple(crop_ids),
        conserved=conserved_leaves,
        new=new_leaves,
        leaf_to_pastis=leaf_to_pastis,
        background_id=0,
    )
    logger.info(
        "italia_label_space_built",
        num_classes=space.num_classes,
        n_crops=len(crop_ids),
        n_conserved=len(conserved_leaves),
        n_new=len(new_leaves),
        conserved=list(conserved_leaves),
    )
    return space


def warm_start_head(
    new_head_weight: np.ndarray,
    new_head_bias: np.ndarray | None,
    pastis_head_weight: np.ndarray,
    pastis_head_bias: np.ndarray | None,
    *,
    label_space: ItaliaLabelSpace,
    pastis_class_names: dict[int, str],
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Copy PASTIS head rows into the conserved rows of the Italian head.

    Generalization of :func:`ml.transfer.finetune_baltico.warm_start_head` to the
    Italian dense label space. For each CONSERVED Italian class, find its PASTIS
    class id (by the PASTIS name) and copy that row of the PASTIS classification
    head into the corresponding Italian row, so the model starts knowing the
    classes France already taught it. Background and NEW rows are left as
    initialised. This is the concrete "kept-class flag": conserved classes are
    warm-started, new ones learn from scratch.

    The PASTIS names map to the contiguous ``[0, 17]`` head ids of
    ``ml.data.pastis_filter.SEMANTIC18_CLASS_NAMES`` (the semantic-18 head of the
    PASTIS checkpoints), NOT the raw ``[0, 19]`` ids.

    Args:
        new_head_weight: Italian head weight ``(K_new, D)`` (modified + returned).
        new_head_bias: Italian head bias ``(K_new,)`` or ``None``.
        pastis_head_weight: PASTIS head weight ``(18, D)``.
        pastis_head_bias: PASTIS head bias ``(18,)`` or ``None``.
        label_space: The Italian label space.
        pastis_class_names: PASTIS contiguous id (``[0, 17]``) -> class name.

    Returns:
        ``(weight, bias, warmed_leaves)`` -- the head with conserved rows copied
        and the list of leaves actually warm-started (a conserved leaf whose PASTIS
        row is missing or whose dims mismatch stays random and is omitted, logged).
    """
    name_to_pastis_id = {name: cid for cid, name in pastis_class_names.items()}
    if new_head_weight.shape[1] != pastis_head_weight.shape[1]:
        logger.warning(
            "warm_start_dim_mismatch",
            new_dim=new_head_weight.shape[1],
            pastis_dim=pastis_head_weight.shape[1],
        )
        return new_head_weight, new_head_bias, []
    warmed: list[str] = []
    for leaf in label_space.conserved:
        pastis_name = label_space.leaf_to_pastis.get(leaf)
        pastis_id = name_to_pastis_id.get(pastis_name) if pastis_name else None
        if pastis_id is None:
            logger.info("warm_start_skip_no_pastis_row", leaf=leaf, pastis_name=pastis_name)
            continue
        row = label_space.index[leaf]
        new_head_weight[row] = pastis_head_weight[pastis_id]
        if new_head_bias is not None and pastis_head_bias is not None:
            new_head_bias[row] = pastis_head_bias[pastis_id]
        warmed.append(leaf)
    logger.info("warm_start_done", n_warmed=len(warmed), warmed=warmed)
    return new_head_weight, new_head_bias, warmed


def stratified_pixel_patch_sample(
    patch_classes: list[set[int]],
    *,
    class_ids: tuple[int, ...],
    min_patches_per_class: int,
    seed: int,
) -> list[int]:
    """Pick patch indices covering each crop class at least ``min_patches`` times.

    The dense Italian patches are few (a pilot of ~20, a full run ~1.2k); a
    per-class patch sample guarantees the rarer Mediterranean leaves (e.g.
    ``durum_hard_wheat``, ``barley``) are not starved of training pixels, mirroring
    the stratified-parcel intent of
    :func:`ml.transfer.finetune_baltico.stratified_parcel_sample` at the patch
    granularity.

    Args:
        patch_classes: Per-patch set of present crop class ids (row-aligned with
            the patch order).
        class_ids: The crop class ids the sample must cover.
        min_patches_per_class: Target number of patches per class (capped at the
            available support).
        seed: RNG seed.

    Returns:
        Sorted list of selected patch indices (a patch may cover several classes,
        so the union is typically smaller than ``len(class_ids) * min_patches``).
    """
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    for cid in class_ids:
        carriers = [i for i, present in enumerate(patch_classes) if cid in present]
        if not carriers:
            continue
        take = min(min_patches_per_class, len(carriers))
        chosen = rng.choice(carriers, size=take, replace=False).tolist()
        selected.update(int(i) for i in chosen)
    return sorted(selected)
