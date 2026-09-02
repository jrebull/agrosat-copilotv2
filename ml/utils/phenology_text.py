"""High-level wrapper to materialize the ``pheno_text_*`` block.

This module orchestrates the generation of phenology descriptions with
Gemini 3.5 Flash over the full dataset (PASTIS-R, ~85951 parcels;
``parcel_id`` format ``10000_1``, not Italian) and persists the result
as a parquet ready for ``LEFT JOIN`` in ``ml.features.fusion``.

Contract (US-023-preview v2):

1. **No mocks or silent skips**: if the API key is missing, an explicit
   ``RuntimeError`` is raised with configuration instructions.
2. **Idempotent cache**: if the parquet already exists and ``overwrite=False``,
   it is reused without invoking the LLM again.
3. **Optional stratified sampling**: ``balanced_by_class=True`` takes
   ``min_per_class`` rows per ``class_id`` (fixed seed for
   reproducibility).
4. **Budget tracking**: estimates the Gemini cost at the start of the run
   (``COST_PER_DESCRIPTION_USD * N``) and logs it via structlog.
5. **Canonical schema**: ``parcel_id`` is always persisted as
   ``pl.Utf8`` (see :mod:`ml.utils.parcel_id`).

The wrapper deliberately does NOT accept the ``skip_llm`` flag in its
signature: notebooks and pipelines must run real Gemini. For unit tests a
mock client is injected via
``ml.features.phenology_description.set_llm_client``.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import structlog

from ml.features.phenology_description import (
    COST_PER_DESCRIPTION_USD,
    DEFAULT_TEXT_EMBED_DIM,
    _has_credentials,
    build_phenology_text_block,
)
from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

__all__ = ["materialize_phenology_text"]


def _check_credentials_or_raise() -> None:
    """Check Gemini credentials; raise ``RuntimeError`` if missing.

    Raises:
        RuntimeError: if there is no ``GEMINI_API_KEY`` nor ``GOOGLE_API_KEY``
            nor Vertex AI configuration present.
    """
    if not _has_credentials():
        raise RuntimeError(
            "Gemini is not configured. Define one of these options in "
            ".env.local before invoking materialize_phenology_text:\n"
            "  1. GEMINI_API_KEY=...        (Google AI Studio)\n"
            "  2. GOOGLE_API_KEY=...        (historical alias)\n"
            "  3. GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT=...\n"
            "If you need a dry-run, inject a mock client with "
            "ml.features.phenology_description.set_llm_client(callable)."
        )


def _stratified_sample(
    df: pl.DataFrame,
    *,
    class_col: str,
    min_per_class: int,
    seed: int,
) -> pl.DataFrame:
    """Sample up to ``min_per_class`` rows per value of ``class_col``.

    For classes with fewer parcels than ``min_per_class``, all are taken
    (no upsampling). The result is shuffled globally with the same
    ``seed`` to avoid residual per-class ordering.

    Args:
        df: Input DataFrame.
        class_col: Column with the class id (Int / Utf8).
        min_per_class: Target number of rows per class.
        seed: Sampling seed.

    Returns:
        Sampled and shuffled DataFrame.
    """
    parts: list[pl.DataFrame] = []
    for class_value, sub in df.group_by(class_col, maintain_order=True):
        del class_value  # only for debugging in pdb if needed.
        n = min(sub.height, min_per_class)
        if sub.height <= min_per_class:
            parts.append(sub)
        else:
            parts.append(sub.sample(n=n, seed=seed, with_replacement=False))
    if not parts:
        return df.clear()
    combined = pl.concat(parts, how="vertical_relaxed")
    return combined.sample(fraction=1.0, seed=seed, with_replacement=False, shuffle=True)


def materialize_phenology_text(
    parcels_features_path: Path | str,
    *,
    output_path: Path = Path("data/features/phenology_text_pastis.parquet"),
    max_parcels: int | None = None,
    balanced_by_class: bool = True,
    min_per_class: int = 30,
    seed: int = 42,
    model: str = "gemini-3.5-flash",
    overwrite: bool = False,
    enforce_api_key: bool = True,
    class_col: str = "class_id",
    parcel_id_col: str = "parcel_id",
    year_col: str = "year",
    progress_every: int = 100,
) -> Path:
    """Materialize the ``pheno_text_*`` block over the real parcels.

    Canonical behavior:

    1. Reads the full features dataset (PASTIS-R, ~85951 parcels) from
       ``parcels_features_path``.
    2. If ``enforce_api_key=True``: checks Gemini credentials with
       :func:`_check_credentials_or_raise`. If there is no injected client
       nor env vars, raises ``RuntimeError``.
    3. If ``balanced_by_class=True``: stratifies by ``class_col``
       taking ``min_per_class`` per class (those with fewer use all
       available rows).
    4. If ``max_parcels`` is ``int > 0``: applies a subsample after the
       balanced sampling.
    5. Cache: if ``output_path`` exists and ``overwrite=False`` returns
       without recomputing (logging ``phenology_text_cache_hit``).
    6. Calls :func:`build_phenology_text_block` with forced
       ``skip_llm=False``.
    7. Persists parquet with canonical schema:
       ``parcel_id`` (Utf8) + ``year`` (Int16) +
       ``pheno_text_000..pheno_text_{D-1}`` (Float32, D=384 by default).

    Args:
        parcels_features_path: Path to the full features parquet
            (PASTIS-R full, ~85951 parcels).
        output_path: Destination path of the parquet with text embeddings.
        max_parcels: Upper limit of parcels to process after the
            balanced sampling. ``None`` = no limit.
        balanced_by_class: If ``True`` stratifies by ``class_col``.
        min_per_class: Minimum rows per class after stratification.
        seed: Sampling seed.
        model: Gemini model identifier (default
            ``"gemini-3.5-flash"``).
        overwrite: If ``True`` recomputes even if ``output_path`` exists.
        enforce_api_key: If ``True`` and there are no credentials nor
            injected client, raises ``RuntimeError``. Only set it to
            ``False`` for tests with a mocked client.
        class_col: Column with the class id (default ``"class_id"``).
        parcel_id_col: Parcel identifier column.
        year_col: Agronomic year column.
        progress_every: Frequency (in number of parcels) of the progress
            log.

    Returns:
        ``Path`` pointing to the generated or reused parquet.

    Raises:
        RuntimeError: if ``enforce_api_key=True`` and there are no credentials
            nor injected client.
        FileNotFoundError: if ``parcels_features_path`` does not exist.
    """
    from ml.features.phenology_description import _LLM_CLIENT

    parcels_path = Path(parcels_features_path)
    if not parcels_path.exists():
        raise FileNotFoundError(f"parcels_features_path does not exist: {parcels_path}")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        cached = pl.read_parquet(output_path)
        logger.info(
            "phenology_text_cache_hit",
            output_path=str(output_path),
            n_parcels=cached.height,
        )
        return output_path

    # Credentials validation: the hard barrier lives in
    # build_phenology_text_block, but we bring the error forward here to
    # avoid reading the full parquet if we are going to fail.
    if enforce_api_key and _LLM_CLIENT is None:
        _check_credentials_or_raise()

    df = pl.read_parquet(parcels_path)
    df = canonical_parcel_id(df, col=parcel_id_col)
    logger.info(
        "phenology_text_input_loaded",
        path=str(parcels_path),
        n_rows=df.height,
        n_cols=len(df.columns),
    )

    sample = df
    if balanced_by_class:
        if class_col not in df.columns:
            raise KeyError(
                f"balanced_by_class=True requires the {class_col!r} column. "
                f"Available columns: {df.columns}"
            )
        sample = _stratified_sample(df, class_col=class_col, min_per_class=min_per_class, seed=seed)
        logger.info(
            "phenology_text_balanced_sample",
            n_after_balance=sample.height,
            min_per_class=min_per_class,
        )

    if max_parcels is not None and max_parcels > 0 and sample.height > max_parcels:
        sample = sample.sample(n=max_parcels, seed=seed, with_replacement=False)
        logger.info("phenology_text_subsampled", n=sample.height)

    n_total = sample.height
    est_cost_usd = n_total * COST_PER_DESCRIPTION_USD
    logger.info(
        "phenology_text_budget_estimate",
        n_total=n_total,
        cost_per_call_usd=COST_PER_DESCRIPTION_USD,
        est_total_cost_usd=round(est_cost_usd, 4),
        model=model,
    )

    t_start = time.monotonic()
    block = build_phenology_text_block(
        sample,
        parcel_id_col=parcel_id_col,
        year_col=year_col,
        model=model,
        skip_llm=False,
        progress_every=progress_every,
    )
    elapsed_s = time.monotonic() - t_start

    # Validation of the canonical output schema.
    if parcel_id_col in block.columns and block.schema[parcel_id_col] != pl.Utf8:
        block = canonical_parcel_id(block, col=parcel_id_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    block.write_parquet(output_path)
    logger.info(
        "phenology_text_materialized",
        output_path=str(output_path),
        n_parcels=block.height,
        n_text_cols=DEFAULT_TEXT_EMBED_DIM,
        elapsed_s=round(elapsed_s, 2),
        est_cost_usd=round(n_total * COST_PER_DESCRIPTION_USD, 4),
    )
    return output_path
