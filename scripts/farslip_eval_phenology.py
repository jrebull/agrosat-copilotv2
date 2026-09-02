"""FarSLIP-pheno (US-036-a) vs AlphaEarth embedding evaluation over real PASTIS-R.

This helper closes the honest evaluation gap of the FarSLIP track (US-037): it
measures, apples-to-apples, whether the embedding space of the US-036-a student
(the full incremental fine-tune over the **real French PASTIS-R** dataset) beats
the previous FarSLIP separability (``0.163``) and approaches AlphaEarth's
(``0.233``). The two numbers are per-class silhouette / separability of the
embedding space (NOT dense mIoU); the objective is to MEASURE and REPORT
honestly, never to guarantee a number (R-NOGAIN / R-CLAIM).

Responsibilities (planning section 4.1):

1. :func:`extract_pheno_embeddings` -- load the winning incremental checkpoint of
   US-036-a via :class:`ml.extractors.farslip_extractor.FarSLIPExtractor`, rebuild
   the held-out PASTIS patches (folds 4/5) with the US-036 peak-NDVI builder
   (:func:`ml.farslip.pastis_pair_dataset.create_incremental_dataset`), run the
   student and write a parquet ``parcel_id`` (= patch_id) + ``class_id``
   (dominant PASTIS class) + ``emb_NNN`` columns (CLS 768 by default, 512 teacher
   projection as a secondary space). Italian / synthetic checkpoints are rejected
   with a clear ``ValueError`` (AC-1).
2. :func:`silhouette_per_class` -- per-class mean silhouette of one space,
   deterministic, returned as a Polars frame whose n-weighted mean approximates
   the global silhouette.
3. :func:`compare_to_alphaearth` -- align the FarSLIP-pheno space and AlphaEarth
   on the shared patches (:func:`ml.eval.embedding_separability.align_spaces_on_parcels`),
   run :func:`ml.eval.embedding_separability.eval_space` per space, and build the
   comparative table with the deltas vs ``0.163`` and vs AlphaEarth-here.
4. A Typer CLI ``eval`` that wires the three steps together, logs one MLflow run
   per space (``:5010`` with ``data_version`` + ``code_version``, runs CLOSED) and
   prints the honest verdict.

Scope (critical, ordered by the user 2026-06-07): ONLY real French PASTIS-R. The
checkpoint is the US-036-a one (``checkpoints/farslip/incremental/<NN>cls/
best.safetensors``), NEVER the Italian US-034/035 student
(``checkpoints/farslip/4band-pheno``), NEVER the published official
``FarSLIP2_ViT-B-16.pt``, NEVER ``data/farslip_pairs`` (Italian synthetic). The
eval uses folds 4/5 held-out, disjoint from the US-036-a train folds (1/2/3).

This module IMPORTS ``eval_space`` / ``align_spaces_on_parcels`` /
``build_balanced_eval_set`` / ``load_alphaearth_embeddings`` from
:mod:`ml.eval.embedding_separability` and ``create_incremental_dataset`` from
:mod:`ml.farslip.pastis_pair_dataset`; it never reimplements them. AlphaEarth
loading goes through :func:`load_alphaearth_for_eval`, a thin schema-aware
wrapper that handles the real PASTIS-aligned parquet (``px_id``/``tile``/
``fold``, where ``px_id`` IS the PASTIS ``ID_PATCH``) on top of the legacy
per-parcel one (``parcel_id``), delegating the legacy case to the shared loader.

Project convention: ``polars`` (no pandas); ``structlog`` (no ``print``); type
hints everywhere; docstrings in English, prose in Spanish; no emojis; MLflow on
the Docker server ``:5010`` with ``data_version`` + ``code_version``.

Typical productive usage on the H100 (run ``nvidia-smi`` first), once the
US-036-a winning checkpoint exists::

    poetry run python -m scripts.farslip_eval_phenology eval \\
        --checkpoint-path checkpoints/farslip/incremental/08cls/best.safetensors \\
        --n-classes 8 --embedding-space cls768 --eval-folds 4,5 \\
        --pastis-root data/PASTIS-R \\
        --alphaearth-path data/cache/gee/alphaearth_at_pastis_fr_full_2019_2433.parquet \\
        --mlflow-uri http://localhost:5010
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import polars as pl
import structlog

from ml.eval.embedding_separability import (
    SeparabilityResult,
    align_spaces_on_parcels,
    build_balanced_eval_set,
    eval_space,
    load_alphaearth_embeddings,
    space_matrix,
)
from ml.ingest.pastis_loader import PASTIS_R_CLASSES
from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.parcel_id import canonical_parcel_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_log = structlog.get_logger(__name__)

#: Golden separability baselines (per-class silhouette of the embedding space,
#: NOT dense mIoU). ``0.163`` is the previous FarSLIP space; ``0.233`` is
#: AlphaEarth in its own eval space. The comparative table contrasts both.
PREV_FARSLIP_SILHOUETTE: float = 0.163
ALPHAEARTH_GOLDEN_SILHOUETTE: float = 0.233

#: Default MLflow tracking server (Docker on :5010); the lineage lives here, NOT
#: in ``./mlruns``. Overridable for CI/dry-runs (e.g. a SQLite ``file://`` URI).
_DEFAULT_MLFLOW_URI: str = "http://localhost:5010"

#: Number of Sentinel-2 input channels of the composite (B02, B03, B04, B08).
_N_IN_CHANNELS: int = 4

#: Embedding column prefix for the FarSLIP-pheno space (matches the notebook and
#: ``align_spaces_on_parcels``); AlphaEarth uses ``dim_``.
_EMB_PREFIX: str = "emb_"
_AE_PREFIX: str = "dim_"

#: Default output parquet of the FarSLIP-pheno embeddings (insumo de US-041).
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_PASTIS_ROOT: Path = _REPO_ROOT / "data" / "PASTIS-R"
_DEFAULT_OUTPUT: Path = _REPO_ROOT / "data" / "farslip" / "embeddings_pheno_pastis.parquet"
_DEFAULT_ALPHAEARTH: Path = (
    _REPO_ROOT / "data" / "cache" / "gee" / "alphaearth_at_pastis_fr_full_2019_2433.parquet"
)

#: Forbidden checkpoint / data roots (US-034/035 Italian + official + synthetic).
#: Pointing the eval at any of these is a scope violation and fails fast (AC-1).
_FORBIDDEN_CKPT_TOKENS: tuple[str, ...] = (
    "4band-pheno",  # US-034 Italian student
    "farslip_pairs",  # US-034/035 Italian synthetic crops
    "farslip2_vit-b-16",  # official published checkpoint
    "farslip-clip-italy",  # Italian distill (US-022 family)
)
_FORBIDDEN_ROOT_NAME: str = "farslip_pairs"

#: Embedding space dimensionality per choice (documented asymmetry vs 64, R-DIM).
_SPACE_DIM: dict[str, int] = {"cls768": 768, "proj512": 512}

EmbeddingSpace = Literal["cls768", "proj512"]


@dataclass(frozen=True)
class EvalEmbeddingsResult:
    """Typed result of :func:`extract_pheno_embeddings`.

    Attributes:
        n_patches: Number of held-out PASTIS patches embedded.
        n_dims: Embedding dimensionality (768 for ``cls768``, 512 for ``proj512``).
        output_path: Absolute path of the generated parquet.
        code_version: Repo git SHA at extraction time.
        data_version: DVC ``data_version`` of the US-036-a checkpoint.
        device_used: ``"cuda"`` or ``"cpu"`` actually used.
        embedding_space: ``"cls768"`` or ``"proj512"``.
        eval_folds: Held-out PASTIS folds used (disjoint from train 1/2/3).
    """

    n_patches: int
    n_dims: int
    output_path: Path
    code_version: str
    data_version: str
    device_used: str
    embedding_space: str
    eval_folds: tuple[int, ...]


@dataclass(frozen=True)
class ComparisonReport:
    """Typed result of :func:`compare_to_alphaearth`.

    Attributes:
        results: Per-space :class:`SeparabilityResult` keyed by space name
            (``"farslip_pheno"``, ``"alphaearth_2019"``).
        per_class_silhouette: Per-space per-class silhouette tables (Polars).
        comparative_table: Comparative table with columns ``space``,
            ``silhouette``, ``f1_macro_mean``, ``f1_macro_std``, ``n_dims``,
            ``n_samples``, ``delta_vs_0163``, ``delta_vs_alphaearth_here``.
        n_shared_parcels: Number of patches shared by every space after the join.
        verdict: Honest one-line verdict (beats / does not beat 0.163; approaches
            / does not approach AlphaEarth-here).
    """

    results: dict[str, SeparabilityResult]
    per_class_silhouette: dict[str, pl.DataFrame]
    comparative_table: pl.DataFrame
    n_shared_parcels: int
    verdict: str = field(default="")


# ---------------------------------------------------------------------------
# Scope guards (AC-1): reject Italian / synthetic / official checkpoints.
# ---------------------------------------------------------------------------


def _validate_checkpoint(checkpoint_path: Path) -> None:
    """Reject Italian / synthetic / official checkpoints; require US-036-a.

    US-037 only evaluates the US-036-a incremental student over real PASTIS-R.
    Pointing the eval at the Italian US-034 student
    (``checkpoints/farslip/4band-pheno``), the Italian synthetic crops
    (``data/farslip_pairs``), the published official checkpoint
    (``FarSLIP2_ViT-B-16.pt``) or any Italian distill is a scope violation and
    fails fast (AC-1).

    Args:
        checkpoint_path: Candidate student checkpoint path.

    Raises:
        ValueError: if the path matches a forbidden Italian/synthetic/official
            token.
    """
    needle = str(checkpoint_path).replace("\\", "/").lower()
    for token in _FORBIDDEN_CKPT_TOKENS:
        if token in needle:
            raise ValueError(
                f"checkpoint_path {checkpoint_path!s} matches the forbidden "
                f"token '{token}' (Italian US-034/035, official published or "
                "synthetic). US-037 evaluates ONLY the US-036-a incremental "
                "student over real PASTIS-R, e.g. "
                "checkpoints/farslip/incremental/<NN>cls/best.safetensors."
            )


def _validate_pastis_root(pastis_root: Path) -> None:
    """Reject the Italian/synthetic data root; require a real PASTIS-R root.

    Args:
        pastis_root: Candidate PASTIS-R root.

    Raises:
        ValueError: if the path is the forbidden Italian/synthetic root.
    """
    parts = {p.lower() for p in pastis_root.parts}
    if _FORBIDDEN_ROOT_NAME in pastis_root.name.lower() or _FORBIDDEN_ROOT_NAME in parts:
        raise ValueError(
            f"pastis_root {pastis_root!s} points at the Italian/synthetic "
            f"'{_FORBIDDEN_ROOT_NAME}' data (US-034/035, discarded). US-037 is "
            "PASTIS-R-only: pass a real PASTIS-R root (e.g. data/PASTIS-R)."
        )


# ---------------------------------------------------------------------------
# Per-class silhouette (AC-4): the only metric NOT already in
# ml.eval.embedding_separability. Everything else is IMPORTED, not reimplemented.
# ---------------------------------------------------------------------------


def silhouette_per_class(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = 10000,
    random_state: int = 42,
) -> pl.DataFrame:
    """Per-class mean silhouette of one embedding space.

    Computes :func:`sklearn.metrics.silhouette_samples` and aggregates the mean
    coefficient per ``class_id``. The n-weighted mean of the per-class values
    approximates the global silhouette of the space (consistency check). The
    result is deterministic for a fixed ``random_state``: when the matrix exceeds
    ``sample_size`` a seeded random subset is drawn before computing the samples.

    Args:
        matrix: Embedding matrix ``(n_samples, n_dims)``.
        labels: Integer class vector ``(n_samples,)`` aligned with ``matrix``.
        sample_size: Cap on samples for the silhouette computation (it is
            ``O(n^2)``); above this a seeded random subset is used.
        random_state: Seed for the subsample (determinism).

    Returns:
        A Polars frame with columns ``class_id`` (Int64), ``silhouette_class``
        (Float64) and ``n`` (Int64), one row per distinct class, sorted by
        ``class_id``.

    Raises:
        ValueError: if ``matrix`` and ``labels`` differ in length or there are
            fewer than two distinct classes (silhouette is undefined).
    """
    from sklearn.metrics import silhouette_samples

    mat = np.asarray(matrix, dtype=np.float64)
    lab = np.asarray(labels)
    if mat.shape[0] != lab.shape[0]:
        raise ValueError(f"matrix rows ({mat.shape[0]}) must equal labels length ({lab.shape[0]}).")
    if len(set(lab.tolist())) < 2:
        raise ValueError("silhouette is undefined for fewer than two distinct classes.")

    if mat.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(mat.shape[0], size=sample_size, replace=False)
        idx.sort()
        mat = mat[idx]
        lab = lab[idx]

    samples = silhouette_samples(mat, lab)
    df = pl.DataFrame(
        {
            "class_id": lab.astype(np.int64),
            "silhouette_sample": samples.astype(np.float64),
        }
    )
    out = (
        df.group_by("class_id")
        .agg(
            pl.col("silhouette_sample").mean().alias("silhouette_class"),
            pl.len().alias("n"),
        )
        .sort("class_id")
    )
    return out


def attach_class_names(per_class: pl.DataFrame, class_names: dict[int, str]) -> pl.DataFrame:
    """Add a ``class_name`` column to a per-class silhouette table.

    Args:
        per_class: Output of :func:`silhouette_per_class` (``class_id`` column).
        class_names: ``class_id -> readable name`` map (e.g. PASTIS classes).

    Returns:
        The frame with a ``class_name`` column right after ``class_id``.
    """
    return per_class.with_columns(
        pl.col("class_id")
        .map_elements(lambda c: class_names.get(int(c), f"c{int(c)}"), return_dtype=pl.Utf8)
        .alias("class_name")
    ).select(["class_id", "class_name", "silhouette_class", "n"])


# ---------------------------------------------------------------------------
# Embedding extraction (AC-2): FarSLIP-pheno over real PASTIS-R held-out folds.
# ---------------------------------------------------------------------------


def _emb_columns(n_dims: int) -> list[str]:
    """Return ``["emb_000", ..., f"emb_{n_dims - 1:03d}"]``."""
    return [f"{_EMB_PREFIX}{i:03d}" for i in range(n_dims)]


def _student_space_forward(
    extractor: object,
    images: torch.Tensor,
    embedding_space: EmbeddingSpace,
) -> np.ndarray:
    """Run the student on a batch and return the requested embedding space.

    For ``cls768`` it uses the pooled CLS token of the vision model
    (``pooler_output``, the 768-dim representation the US-036-a InfoNCE
    optimized); for ``proj512`` it uses ``extract_embeddings`` (the teacher's
    ``visual_projection``, NOT fine-tuned, reported only as a secondary space,
    R-EXTR-512).

    Args:
        extractor: A loaded :class:`FarSLIPExtractor` (typed ``object`` so the
            heavy import stays local and the tests can pass a light double).
        images: ``(B, 4, H, W)`` float crops in ``[0, 1]``.
        embedding_space: ``"cls768"`` or ``"proj512"``.

    Returns:
        ``(B, n_dims)`` float32 numpy array.
    """
    import torch as _torch

    if embedding_space == "proj512":
        emb = extractor.extract_embeddings(images)  # type: ignore[attr-defined]
        return emb.detach().cpu().float().numpy()

    # cls768: the pooled CLS the contrastive loss consumed. We replicate the
    # extractor's prep (uint16/10000, 4 bands, 224) and read pooler_output.
    prepped = extractor._prep_crops(images)  # type: ignore[attr-defined]
    with _torch.inference_mode():
        vision_out = extractor.model.vision_model(pixel_values=prepped)  # type: ignore[attr-defined]
        pooled = vision_out.pooler_output  # (B, 768)
    return pooled.detach().cpu().float().numpy()


def extract_pheno_embeddings(
    *,
    student_checkpoint: Path,
    n_classes: int,
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    eval_folds: Sequence[int] = (4, 5),
    embedding_space: EmbeddingSpace = "cls768",
    output_path: Path = _DEFAULT_OUTPUT,
    batch_size: int = 64,
    ratio: float = 3.0,
    device: str = "auto",
    seed: int = 42,
) -> EvalEmbeddingsResult:
    """Extract FarSLIP-pheno (US-036-a student) embeddings over real PASTIS-R.

    Loads the winning incremental checkpoint via :class:`FarSLIPExtractor`,
    rebuilds the held-out PASTIS patches with the US-036 peak-NDVI builder
    (:func:`create_incremental_dataset`, ``folds=eval_folds``), runs the student
    and writes a parquet with ``parcel_id`` (= patch_id), ``class_id`` (dominant
    PASTIS class of the patch) and ``emb_NNN`` columns. Rejects Italian /
    synthetic / official checkpoints (``ValueError``).

    The unit of evaluation is the PATCH (the unit the student saw at train time,
    R-GRAN): one composite pico-NDVI per patch, one embedding per patch, the
    patch's dominant PASTIS class as ``class_id``; ``parcel_id`` carries the
    ``patch_id`` so AlphaEarth can be aggregated to the same unit downstream.

    Args:
        student_checkpoint: Best of the US-036-a winning step
            (``checkpoints/farslip/incremental/<NN>cls/best.safetensors``).
        n_classes: Cardinality of the US-036-a winning step (active classes).
        pastis_root: Real PASTIS-R root (Italian/synthetic root rejected).
        eval_folds: Held-out PASTIS folds (default ``(4, 5)``; disjoint of train).
        embedding_space: ``"cls768"`` (default, the fine-tuned CLS) or
            ``"proj512"`` (teacher projection, secondary).
        output_path: Output parquet (parent created if missing).
        batch_size: Student forward batch size.
        ratio: 3:1 Meadow dominance filter ratio (matches the US-036 builder).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` forwarded to the extractor.
        seed: Determinism seed forwarded to the builder.

    Returns:
        An :class:`EvalEmbeddingsResult` with the parquet metadata and lineage.

    Raises:
        ValueError: if the checkpoint or root is Italian/synthetic/official.
        FileNotFoundError: if the checkpoint does not exist yet (US-036-a still
            running) -- a clear error, never a silent fallback.
    """
    import torch

    from ml.extractors.farslip_extractor import FarSLIPExtractor
    from ml.farslip.pastis_pair_dataset import (
        active_classes,
        create_incremental_dataset,
    )

    _validate_checkpoint(student_checkpoint)
    _validate_pastis_root(pastis_root)
    if not student_checkpoint.exists():
        raise FileNotFoundError(
            f"student_checkpoint not found: {student_checkpoint!s}. The US-036-a "
            "winning checkpoint must exist (full incremental run closed, "
            "best.safetensors materialized in DVC) before US-037 can extract. "
            "This is a hard block, not a silent fallback (R-DEP)."
        )

    n_dims = _SPACE_DIM[embedding_space]
    folds = tuple(int(f) for f in eval_folds)
    active = active_classes(n_classes)

    _log.info(
        "extract_pheno_embeddings_start",
        student_checkpoint=str(student_checkpoint),
        extractor_mode="real",
        n_classes=n_classes,
        eval_folds=list(folds),
        embedding_space=embedding_space,
        n_dims=n_dims,
        dataset_source="pastis_real",
        n_regions=1,
        active_classes=list(active),
    )

    dataset, n_regions, _n_categories, _proto = create_incremental_dataset(
        n_classes,
        root=pastis_root,
        folds=folds,
        ratio=ratio,
        seed=seed,
    )
    extractor = FarSLIPExtractor(
        weights_uri=str(student_checkpoint),
        device=device,
        n_in_channels=_N_IN_CHANNELS,
    )

    patch_ids: list[str] = []
    class_ids: list[int] = []
    emb_chunks: list[np.ndarray] = []
    batch_imgs: list[torch.Tensor] = []

    def _flush() -> None:
        if not batch_imgs:
            return
        images = torch.stack(batch_imgs, dim=0)
        emb_chunks.append(_student_space_forward(extractor, images, embedding_space))
        batch_imgs.clear()

    for idx in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[idx]
        # The dataset exposes (patch_id, category_id) pairs; recover both the
        # patch_id and the dominant PASTIS class_id (active index -> class_id).
        pid, category_id = dataset._samples[idx]  # type: ignore[attr-defined]
        patch_ids.append(str(pid))
        class_ids.append(int(active[int(category_id)]))
        batch_imgs.append(item["image"])
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()

    embeddings = (
        np.concatenate(emb_chunks, axis=0)
        if emb_chunks
        else np.empty((0, n_dims), dtype=np.float32)
    )
    if embeddings.shape[0] != len(patch_ids):  # pragma: no cover - defensive
        raise RuntimeError(f"embedding rows ({embeddings.shape[0]}) != patches ({len(patch_ids)}).")

    cols = _emb_columns(n_dims)
    out_df = pl.DataFrame({"parcel_id": patch_ids, "class_id": class_ids}).hstack(
        pl.DataFrame(embeddings, schema=cols)
    )
    out_df = canonical_parcel_id(out_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)

    result = EvalEmbeddingsResult(
        n_patches=out_df.height,
        n_dims=n_dims,
        output_path=output_path.resolve(),
        code_version=git_sha(short=True),
        data_version=dvc_data_version(str(student_checkpoint)),
        device_used=str(extractor.device),  # type: ignore[attr-defined]
        embedding_space=embedding_space,
        eval_folds=folds,
    )
    _log.info(
        "extract_pheno_embeddings_done",
        n_patches=result.n_patches,
        n_dims=result.n_dims,
        n_regions=n_regions,
        output=str(result.output_path),
        device=result.device_used,
        code_version=result.code_version,
        data_version=result.data_version,
    )
    return result


# ---------------------------------------------------------------------------
# AlphaEarth loading + aggregation to patch level (AC-3): same unit as FarSLIP.
# ---------------------------------------------------------------------------

#: Pixel/sample id of the real PASTIS-aligned AlphaEarth parquet. It is NOT a
#: pixel within a tile: ``px_id`` equals the PASTIS ``ID_PATCH`` 1:1 (verified on
#: ``alphaearth_at_pastis_fr_full_2019_2433.parquet``: 2433 rows, 2433 unique
#: px_id, exact match against ``metadata.geojson`` ID_PATCH, zero fold
#: mismatches). The Sentinel-2 MGRS ``tile`` column (only 4 values) is the tile,
#: NOT the patch, so AlphaEarth must NEVER be aggregated by ``tile``.
_AE_PXID_COL: str = "px_id"
_AE_TILE_COL: str = "tile"


def load_alphaearth_for_eval(path: str | Path) -> pl.DataFrame:
    """Load an AlphaEarth parquet at PASTIS-patch level, schema-aware.

    Two AlphaEarth schemas coexist in the project and this loader normalizes both
    to ``parcel_id`` (= PASTIS ``ID_PATCH``) + the 64 ``dim_NN`` columns:

    1. The real PASTIS-aligned parquet (US-037 default,
       ``alphaearth_at_pastis_fr_full_2019_2433.parquet``) is keyed by
       ``px_id`` -- which IS the PASTIS ``ID_PATCH`` (one row per patch, verified
       1:1 against ``metadata.geojson``) -- plus ``lon``/``lat``/``year``/
       ``tile``/``fold``. The patch key is ``px_id``; ``tile`` is the Sentinel-2
       MGRS tile (only 4 values), NOT the patch, so it is dropped.
    2. The legacy per-parcel parquet
       (``alphaearth_parcels_pastis_parcels_2019_85951.parquet``) is keyed by
       ``parcel_id == "{patch_id}_{instance}"`` (many rows per patch). For this
       schema the shared :func:`load_alphaearth_embeddings` is reused.

    Args:
        path: Path to the AlphaEarth parquet (either schema).

    Returns:
        A DataFrame with ``parcel_id`` (Utf8) and the 64 ``dim_NN`` columns. For
        schema 1 the patch key is materialized from ``px_id``; for schema 2 the
        original ``parcel_id`` is preserved (collapse to patch level afterwards
        with :func:`aggregate_alphaearth_to_patch`).

    Raises:
        ValueError: if the parquet carries neither ``parcel_id`` nor ``px_id``
            (an unknown AlphaEarth schema), or if it has no ``dim_*`` columns.
    """
    raw = pl.read_parquet(path)
    dim_cols = sorted(c for c in raw.columns if c.startswith(_AE_PREFIX))
    if not dim_cols:
        raise ValueError(
            f"AlphaEarth parquet {path!s} has no '{_AE_PREFIX}*' embedding "
            f"columns; available columns: {raw.columns}."
        )
    if "parcel_id" in raw.columns:
        # Legacy per-parcel schema: reuse the shared, tested loader as-is.
        return load_alphaearth_embeddings(path)
    if _AE_PXID_COL in raw.columns:
        # Real PASTIS-aligned schema: px_id IS the patch_id (ID_PATCH). Rename it
        # to parcel_id and keep only the dims; tile/fold/lon/lat are dropped.
        out = raw.select([_AE_PXID_COL, *dim_cols]).rename({_AE_PXID_COL: "parcel_id"})
        return canonical_parcel_id(out)
    raise ValueError(
        f"AlphaEarth parquet {path!s} carries neither 'parcel_id' nor "
        f"'{_AE_PXID_COL}'; cannot derive the PASTIS patch key. Available "
        f"columns: {raw.columns}."
    )


def aggregate_alphaearth_to_patch(
    alphaearth_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate AlphaEarth embeddings to PASTIS patch level (idempotent).

    Two AlphaEarth grains reach this function and both are collapsed to one row
    per PASTIS patch (R-GRAN), so the inner join with FarSLIP-pheno (whose
    ``parcel_id`` IS the ``patch_id``) lines the two spaces up on the same unit:

    - Legacy per-parcel grain (``parcel_id == "{patch_id}_{instance}"``): the
      patch key is the prefix before the first ``_`` and the dims are averaged
      over the parcels of each patch.
    - Real PASTIS-aligned grain (``parcel_id`` already a bare ``ID_PATCH`` from
      :func:`load_alphaearth_for_eval`, no ``_``): the prefix is the whole id, so
      the group-by collapses each (already unique) patch to itself -- an
      identity, never an over-aggregation to the Sentinel-2 ``tile``.

    Args:
        alphaearth_df: AlphaEarth table with ``parcel_id`` and the 64 ``dim_NN``
            columns (either grain, as returned by
            :func:`load_alphaearth_for_eval`).

    Returns:
        A patch-level table with ``parcel_id`` (= patch_id, Utf8) and the mean
        ``dim_NN`` columns, one row per distinct patch.
    """
    df = canonical_parcel_id(alphaearth_df)
    dim_cols = sorted(c for c in df.columns if c.startswith(_AE_PREFIX))
    df = df.with_columns(pl.col("parcel_id").str.split("_").list.first().alias("__patch"))
    aggregated = (
        df.group_by("__patch")
        .agg(*(pl.col(c).mean().alias(c) for c in dim_cols))
        .rename({"__patch": "parcel_id"})
    )
    return canonical_parcel_id(aggregated)


# ---------------------------------------------------------------------------
# Apples-to-apples comparison (AC-3, AC-5): FarSLIP-pheno vs AlphaEarth.
# ---------------------------------------------------------------------------


def _build_verdict(pheno: SeparabilityResult, alphaearth: SeparabilityResult) -> str:
    """Build the honest one-line verdict (R-NOGAIN / R-CLAIM).

    Args:
        pheno: FarSLIP-pheno separability.
        alphaearth: AlphaEarth (here) separability.

    Returns:
        A reader-facing Spanish verdict that does NOT over-claim: it states
        whether the previous 0.163 is beaten and whether AlphaEarth-here is
        approached, with the asymmetry caveat (dims differ).
    """
    beats_prev = pheno.silhouette > PREV_FARSLIP_SILHOUETTE
    gap_here = alphaearth.silhouette - pheno.silhouette
    approaches = pheno.silhouette >= alphaearth.silhouette or gap_here <= 0.02

    prev_clause = (
        f"supera el 0.163 previo (silhouette={pheno.silhouette:.3f})"
        if beats_prev
        else f"NO supera el 0.163 previo (silhouette={pheno.silhouette:.3f})"
    )
    ae_clause = (
        f"se acerca a AlphaEarth-aqui ({alphaearth.silhouette:.3f})"
        if approaches
        else (f"no alcanza AlphaEarth-aqui ({alphaearth.silhouette:.3f}; brecha {gap_here:.3f})")
    )
    return (
        f"FarSLIP-pheno {prev_clause}; {ae_clause}. "
        f"Caveat: dimensiones {pheno.n_dims} vs {alphaearth.n_dims} no son "
        "invariantes para silhouette; la senal complementaria se confirma con "
        "F1-macro LogReg, no solo con silhouette."
    )


def compare_to_alphaearth(
    *,
    pheno_df: pl.DataFrame,
    alphaearth_df: pl.DataFrame,
    class_names: dict[int, str] | None = None,
    per_class_cap: int = 500,
    min_class_samples: int = 50,
    prev_farslip_silhouette: float = PREV_FARSLIP_SILHOUETTE,
    random_state: int = 42,
    n_splits: int = 5,
    balanced: bool = True,
) -> ComparisonReport:
    """Apples-to-apples FarSLIP-pheno vs AlphaEarth on the shared patches.

    Inner-joins both spaces with :func:`align_spaces_on_parcels` so they are
    evaluated on IDENTICAL rows and labels (the ``class_id`` is taken from the
    FarSLIP-pheno table, the single source of truth for the patch's dominant
    class -- R-LABEL), runs :func:`eval_space` per space, computes the per-class
    silhouette for both, and builds the comparative table with the deltas vs
    ``prev_farslip_silhouette`` (0.163) and vs AlphaEarth-here.

    Args:
        pheno_df: FarSLIP-pheno embeddings (``parcel_id`` + ``class_id`` +
            ``emb_*``). ``parcel_id`` is the PASTIS ``patch_id``.
        alphaearth_df: AlphaEarth embeddings at PATCH level (``parcel_id`` =
            patch_id + ``dim_*``). Use :func:`aggregate_alphaearth_to_patch` to
            collapse the per-parcel parquet first.
        class_names: Optional ``class_id -> name`` map for the per-class tables.
        per_class_cap: Max patches per class in the balanced eval set.
        min_class_samples: Min patches for a class to survive the balancing.
        prev_farslip_silhouette: Golden FarSLIP separability to delta against.
        random_state: Seed for the balancing and the eval.
        n_splits: Stratified CV folds for the LogReg F1-macro.
        balanced: If ``True`` (default) evaluate on the per-class capped set; if
            ``False`` evaluate on the full shared universe.

    Returns:
        A :class:`ComparisonReport`.

    Raises:
        ValueError: if either table lacks the required columns or the join is
            empty (no shared patches).
    """
    if "class_id" not in pheno_df.columns:
        raise ValueError("pheno_df must carry a 'class_id' column.")
    pheno = canonical_parcel_id(pheno_df)
    alphaearth = canonical_parcel_id(alphaearth_df)

    emb_cols = sorted(c for c in pheno.columns if c.startswith(_EMB_PREFIX))
    ae_cols = sorted(c for c in alphaearth.columns if c.startswith(_AE_PREFIX))
    if not emb_cols:
        raise ValueError("pheno_df has no 'emb_*' embedding columns.")
    if not ae_cols:
        raise ValueError("alphaearth_df has no 'dim_*' embedding columns.")

    # The label universe and the single source of class_id is the FarSLIP-pheno
    # table (patch-level dominant class). AlphaEarth contributes only its dims.
    labels_universe = pheno.select(["parcel_id", "class_id"])
    if balanced:
        labels_df, dropped = build_balanced_eval_set(
            labels_universe,
            per_class_cap=per_class_cap,
            min_class_samples=min_class_samples,
            random_state=random_state,
            class_names=class_names,
        )
        if dropped:
            _log.info("compare_dropped_rare_classes", dropped=dropped)
    else:
        labels_df = labels_universe
        if class_names is not None:
            labels_df = labels_df.with_columns(
                pl.col("class_id")
                .map_elements(
                    lambda c: class_names.get(int(c), f"c{int(c)}"),
                    return_dtype=pl.Utf8,
                )
                .alias("class_name")
            )

    merged, prefixed_cols = align_spaces_on_parcels(
        labels_df,
        {
            "farslip_pheno": (pheno.select(["parcel_id", *emb_cols]), _EMB_PREFIX),
            "alphaearth_2019": (
                alphaearth.select(["parcel_id", *ae_cols]),
                _AE_PREFIX,
            ),
        },
    )
    n_shared = merged.height
    if n_shared == 0:
        raise ValueError(
            "no shared patches between FarSLIP-pheno and AlphaEarth after the "
            "inner join; check the parcel_id schema (patch-level on both sides)."
        )
    labels = merged["class_id"].to_numpy()

    results: dict[str, SeparabilityResult] = {}
    per_class: dict[str, pl.DataFrame] = {}
    for space_key, cols in prefixed_cols.items():
        matrix = space_matrix(merged, cols)
        results[space_key] = eval_space(
            matrix,
            labels,
            label=space_key,
            n_splits=n_splits,
            random_state=random_state,
        )
        table = silhouette_per_class(matrix, labels, random_state=random_state)
        if class_names is not None:
            table = attach_class_names(table, class_names)
        per_class[space_key] = table

    pheno_res = results["farslip_pheno"]
    ae_res = results["alphaearth_2019"]
    comparative_table = pl.DataFrame(
        [
            {
                "space": space_key,
                "silhouette": res.silhouette,
                "f1_macro_mean": res.f1_macro_mean,
                "f1_macro_std": res.f1_macro_std,
                "n_dims": res.n_dims,
                "n_samples": res.n_samples,
                "delta_vs_0163": res.silhouette - prev_farslip_silhouette,
                "delta_vs_alphaearth_here": res.silhouette - ae_res.silhouette,
            }
            for space_key, res in results.items()
        ]
    )

    verdict = _build_verdict(pheno_res, ae_res)
    _log.info(
        "compare_to_alphaearth_done",
        n_shared_parcels=n_shared,
        farslip_silhouette=round(pheno_res.silhouette, 4),
        farslip_f1_macro=round(pheno_res.f1_macro_mean, 4),
        alphaearth_silhouette=round(ae_res.silhouette, 4),
        alphaearth_f1_macro=round(ae_res.f1_macro_mean, 4),
        delta_vs_0163=round(pheno_res.silhouette - prev_farslip_silhouette, 4),
        delta_vs_alphaearth_here=round(pheno_res.silhouette - ae_res.silhouette, 4),
        verdict=verdict,
    )
    return ComparisonReport(
        results=results,
        per_class_silhouette=per_class,
        comparative_table=comparative_table,
        n_shared_parcels=n_shared,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# MLflow logging (AC-8): one run per space, CLOSED, data/code version.
# ---------------------------------------------------------------------------


def _log_space_run(
    *,
    mlflow_uri: str,
    space_key: str,
    result: SeparabilityResult,
    per_class_table: pl.DataFrame,
    ae_result: SeparabilityResult,
    student_checkpoint: Path,
    n_classes: int,
    embedding_space: str,
    eval_folds: tuple[int, ...],
    per_class_cap: int,
    regime: str,
) -> None:
    """Log one CLOSED MLflow run for a single embedding space.

    Tags ``code_version`` (git SHA) + ``data_version`` (DVC hash of the US-036-a
    checkpoint); logs the separability metrics, the deltas vs 0.163 and vs
    AlphaEarth-here, and the per-class silhouette table as a Polars artifact. If
    MLflow is not installed or the server is down the function degrades to a
    warning (the run still lives in the logs); it NEVER leaves a run ``RUNNING``
    (gotcha ml/AGENTS.md).

    Args:
        mlflow_uri: Tracking URI (Docker :5010 or a SQLite file for CI).
        space_key: ``"farslip_pheno"`` or ``"alphaearth_2019"``.
        result: The space's separability metrics.
        per_class_table: The space's per-class silhouette table.
        ae_result: AlphaEarth-here separability (for the delta).
        student_checkpoint: US-036-a checkpoint (drives ``data_version``).
        n_classes: Cardinality of the US-036-a winning step.
        embedding_space: ``"cls768"`` or ``"proj512"``.
        eval_folds: Held-out PASTIS folds.
        per_class_cap: Balancing cap used.
        regime: ``"balanced"`` or ``"full"``.
    """
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow optional
        _log.warning("mlflow not installed; space run not logged", space=space_key)
        return

    import tempfile

    run_name = f"farslip-eval-{space_key.replace('_', '-')}-{regime}"
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("farslip")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "code_version": git_sha(),
                    "data_version": dvc_data_version(str(student_checkpoint)),
                    "us": "US-037",
                    "space": space_key,
                    "regime": regime,
                }
            )
            mlflow.log_params(
                {
                    "student_checkpoint": str(student_checkpoint),
                    "n_classes": n_classes,
                    "embedding_space": embedding_space,
                    "eval_folds": ",".join(str(f) for f in eval_folds),
                    "per_class_cap": per_class_cap,
                    "space": space_key,
                    "regime": regime,
                }
            )
            mlflow.log_metrics(
                {
                    "silhouette": result.silhouette,
                    "f1_macro_mean": result.f1_macro_mean,
                    "f1_macro_std": result.f1_macro_std,
                    "n_samples": float(result.n_samples),
                    "n_dims": float(result.n_dims),
                    "n_classes": float(result.n_classes),
                    "delta_vs_0163": result.silhouette - PREV_FARSLIP_SILHOUETTE,
                    "delta_vs_alphaearth_here": result.silhouette - ae_result.silhouette,
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                art = Path(tmp) / f"silhouette_per_class_{space_key}.parquet"
                per_class_table.write_parquet(art)
                mlflow.log_artifact(str(art))
        _log.info("space mlflow run logged and closed", run=run_name)
    except Exception as exc:  # noqa: BLE001 - never let logging kill the eval
        _log.warning("mlflow space run failed", run=run_name, error=str(exc))


def run_eval(
    *,
    checkpoint_path: Path,
    n_classes: int,
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    alphaearth_path: Path = _DEFAULT_ALPHAEARTH,
    eval_folds: Sequence[int] = (4, 5),
    embedding_space: EmbeddingSpace = "cls768",
    output_path: Path = _DEFAULT_OUTPUT,
    per_class_cap: int = 500,
    min_class_samples: int = 50,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 42,
    mlflow_uri: str = _DEFAULT_MLFLOW_URI,
    log_mlflow: bool = True,
) -> ComparisonReport:
    """End-to-end US-037 evaluation: extract, align, compare, log.

    Extracts the FarSLIP-pheno embeddings of the US-036-a winning checkpoint over
    the held-out PASTIS folds, aggregates AlphaEarth to patch level, compares the
    two spaces apples-to-apples (silhouette per class + F1-macro 5-fold) and logs
    one CLOSED MLflow run per space. Returns the honest :class:`ComparisonReport`.

    Args:
        checkpoint_path: US-036-a winning checkpoint (best.safetensors).
        n_classes: Cardinality of the US-036-a winning step.
        pastis_root: Real PASTIS-R root.
        alphaearth_path: Per-parcel AlphaEarth parquet (2019, 64-dim).
        eval_folds: Held-out PASTIS folds (default ``(4, 5)``).
        embedding_space: ``"cls768"`` (default) or ``"proj512"``.
        output_path: Output parquet of the FarSLIP-pheno embeddings.
        per_class_cap: Max patches per class in the balanced eval set.
        min_class_samples: Min patches per class to survive the balancing.
        batch_size: Student forward batch size.
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        seed: Determinism seed.
        mlflow_uri: MLflow tracking URI.
        log_mlflow: If ``True`` log one CLOSED run per space.

    Returns:
        A :class:`ComparisonReport` with the comparative table and verdict.
    """
    extracted = extract_pheno_embeddings(
        student_checkpoint=checkpoint_path,
        n_classes=n_classes,
        pastis_root=pastis_root,
        eval_folds=eval_folds,
        embedding_space=embedding_space,
        output_path=output_path,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    pheno_df = pl.read_parquet(extracted.output_path)
    alphaearth_raw = load_alphaearth_for_eval(alphaearth_path)
    alphaearth_patch = aggregate_alphaearth_to_patch(alphaearth_raw)

    report = compare_to_alphaearth(
        pheno_df=pheno_df,
        alphaearth_df=alphaearth_patch,
        class_names=PASTIS_R_CLASSES,
        per_class_cap=per_class_cap,
        min_class_samples=min_class_samples,
        random_state=seed,
    )

    if log_mlflow:
        ae_result = report.results["alphaearth_2019"]
        folds = tuple(int(f) for f in eval_folds)
        for space_key, result in report.results.items():
            _log_space_run(
                mlflow_uri=mlflow_uri,
                space_key=space_key,
                result=result,
                per_class_table=report.per_class_silhouette[space_key],
                ae_result=ae_result,
                student_checkpoint=checkpoint_path,
                n_classes=n_classes,
                embedding_space=embedding_space,
                eval_folds=folds,
                per_class_cap=per_class_cap,
                regime="balanced",
            )

    _log.info(
        "run_eval_done",
        verdict=report.verdict,
        n_shared_parcels=report.n_shared_parcels,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the eval CLI. poetry add typer") from exc

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Eval FarSLIP-pheno vs AlphaEarth (US-037). Subcomando: ``eval``.

    El callback fuerza a Typer a exponer ``eval`` como subcomando explicito (sin
    el, un Typer de comando unico ignora el nombre e interpreta ``eval`` como
    argumento extra).
    """


def _parse_folds(folds: str) -> tuple[int, ...]:
    """Parse a comma-separated fold string into a sorted unique tuple.

    Args:
        folds: e.g. ``"4,5"``.

    Returns:
        Sorted unique tuple of ints.
    """
    return tuple(sorted({int(f.strip()) for f in folds.split(",") if f.strip()}))


@app.command(name="eval")
def eval_command(
    checkpoint_path: Annotated[
        Path,
        typer.Option(
            help=(
                "Best del escalon ganador de US-036-a "
                "(checkpoints/farslip/incremental/<NN>cls/best.safetensors)"
            )
        ),
    ],
    n_classes: Annotated[int, typer.Option(help="Cardinalidad del escalon ganador de US-036-a")],
    pastis_root: Annotated[
        Path, typer.Option(help="Raiz PASTIS-R (frances real)")
    ] = _DEFAULT_PASTIS_ROOT,
    alphaearth_path: Annotated[
        Path,
        typer.Option(
            help=(
                "Parquet AlphaEarth 2019 (64-dim). Acepta el formato real "
                "alineado a PASTIS (px_id=ID_PATCH, tile, fold) o el legado "
                "por-parcela (parcel_id={patch}_{i})"
            )
        ),
    ] = _DEFAULT_ALPHAEARTH,
    eval_folds: Annotated[
        str, typer.Option(help="Folds held-out PASTIS (disjuntos de train 1/2/3)")
    ] = "4,5",
    embedding_space: Annotated[
        str,
        typer.Option(help="Espacio FarSLIP: cls768 (default) o proj512 (secundario)"),
    ] = "cls768",
    output_path: Annotated[
        Path, typer.Option(help="Parquet de salida FarSLIP-pheno (insumo US-041)")
    ] = _DEFAULT_OUTPUT,
    per_class_cap: Annotated[
        int, typer.Option(help="Tope de patches por clase (eval balanceada)")
    ] = 500,
    min_class_samples: Annotated[
        int, typer.Option(help="Minimo de patches por clase para conservarla")
    ] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch del forward del student")] = 64,
    device: Annotated[str, typer.Option(help="Device: auto | cuda | cpu")] = "auto",
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    mlflow_uri: Annotated[
        str, typer.Option(help="MLflow tracking URI (Docker :5010)")
    ] = _DEFAULT_MLFLOW_URI,
    log_mlflow: Annotated[
        bool, typer.Option(help="Registra un run MLflow por espacio (cerrado)")
    ] = True,
) -> None:
    """Evalua FarSLIP-pheno (US-036-a) vs AlphaEarth sobre PASTIS-R real.

    Extrae los embeddings del best del escalon ganador de US-036-a sobre los folds
    held-out (4/5), agrega AlphaEarth a nivel patch, compara ambos espacios
    apples-to-apples (silhouette por clase + F1-macro LogReg 5-fold) y registra un
    run MLflow por espacio (cerrado, con ``data_version`` + ``code_version``).
    Imprime la tabla comparativa y el veredicto honesto (supera/no el 0.163;
    se acerca/no a AlphaEarth-aqui), SIN sobre-afirmar.

    Args:
        checkpoint_path: Best del escalon ganador de US-036-a.
        n_classes: Cardinalidad del escalon ganador.
        pastis_root: Raiz PASTIS-R real.
        alphaearth_path: Parquet AlphaEarth 2019 por parcela.
        eval_folds: Folds held-out coma-separados.
        embedding_space: ``cls768`` o ``proj512``.
        output_path: Parquet de salida de los embeddings FarSLIP-pheno.
        per_class_cap: Tope de patches por clase.
        min_class_samples: Minimo de patches por clase.
        batch_size: Batch del forward.
        device: ``auto`` | ``cuda`` | ``cpu``.
        seed: Semilla determinismo.
        mlflow_uri: MLflow tracking URI.
        log_mlflow: Si registra runs MLflow.
    """
    if embedding_space not in _SPACE_DIM:
        raise typer.BadParameter(
            f"embedding_space must be one of {sorted(_SPACE_DIM)}; got {embedding_space!r}."
        )
    report = run_eval(
        checkpoint_path=checkpoint_path,
        n_classes=n_classes,
        pastis_root=pastis_root,
        alphaearth_path=alphaearth_path,
        eval_folds=_parse_folds(eval_folds),
        embedding_space=embedding_space,  # type: ignore[arg-type]
        output_path=output_path,
        per_class_cap=per_class_cap,
        min_class_samples=min_class_samples,
        batch_size=batch_size,
        device=device,
        seed=seed,
        mlflow_uri=mlflow_uri,
        log_mlflow=log_mlflow,
    )
    typer.echo("Tabla comparativa (FarSLIP-pheno vs AlphaEarth):")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        typer.echo(str(report.comparative_table))
    typer.echo(f"Patches compartidos: {report.n_shared_parcels}")
    typer.echo(f"Veredicto honesto: {report.verdict}")


if __name__ == "__main__":  # pragma: no cover
    app()


__all__ = [
    "ALPHAEARTH_GOLDEN_SILHOUETTE",
    "PREV_FARSLIP_SILHOUETTE",
    "ComparisonReport",
    "EvalEmbeddingsResult",
    "aggregate_alphaearth_to_patch",
    "attach_class_names",
    "compare_to_alphaearth",
    "extract_pheno_embeddings",
    "load_alphaearth_for_eval",
    "run_eval",
    "silhouette_per_class",
]
