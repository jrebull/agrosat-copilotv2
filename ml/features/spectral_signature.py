"""Compact per-parcel spectral signature descriptors (US-023-preview P5).

Generates ``spectral_signature_*`` as an optional block for ``fusion.py`` with
agronomically justified descriptors of the spectral curve sampled per parcel:

- ``rep`` (default, **Frampton et al. 2013**, DOI 10.1016/j.isprsjprs.2013.04.007):
  seasonal Red Edge Position. Position (in nm) of the inflection point of
  the reflectance curve between red and near-infrared. The REP varies with
  chlorophyll content and crop phenology; agronomic remote-sensing
  literature documents it as one of the most reliable compact descriptors
  of crop condition.
- ``sam`` (Spectral Angle Mapper): cosine of the spectral angle between the
  parcel signature and the centroid of the majority class observed during
  fit. It is ``1.0`` when the parcel "resembles" the mean signature of the
  majority class, ``-1.0`` when orthogonal. Useful as a contrastive base
  learner.
- ``redge_moments``: statistical moments (mean, var, skew) of the red-edge
  reflectance aggregated per parcel. Captures the shape of the red-edge
  curve in 3 compact numbers.

Canonical decisions (US-023-preview plan §11 D-3):

- ``rep`` as default: well established in the literature, computable from
  already-sampled S2, requires no new GEE ingestion.
- Sklearn-compatible (``BaseEstimator`` + ``TransformerMixin``) to fit into
  Pipelines and satisfy the contract of the US-022b tests.
- Polars in / Polars out: the input DataFrame is already clean and
  filtered; the caller (notebook 05 / ``fusion.py``) performs the joins.
- **Consumes no GEE quota**: the bands and stats come from the fused
  features parquet (``data/features/*``). The module only combines
  already-sampled columns.

Output layout (stable order, downstream depends on it):

::

    parcel_id (i64) | year (i16) |
    spectral_signature_000 .. spectral_signature_{K-1} (K)

Where ``K`` depends on the descriptor:
- ``rep``: ``K = len(phenology_anchors)`` (default 3 — SOG/peak/senescence
  -> ``spectral_signature_000, 001, 002``).
- ``sam``: ``K = 1`` (a single scalar angle).
- ``redge_moments``: ``K = 3 * len(phenology_anchors)`` (mean, var, skew
  per anchor; default 9 cols).

The block enters ``fusion.py`` via ``LEFT JOIN`` on ``(parcel_id, year)``,
the same pattern as FarSLIP and phenology_text.

Agronomic references
--------------------
- Frampton, W.J. et al. (2013), *Evaluating the capabilities of Sentinel-2
  for quantitative estimation of biophysical variables in vegetation*,
  ISPRS J. 82, 83-92. DOI 10.1016/j.isprsjprs.2013.04.007.
- Kruse, F.A. et al. (1993), *The Spectral Image Processing System (SIPS)*,
  Remote Sens. Environ. 44, 145-163 (SAM).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import polars as pl
import structlog
from sklearn.base import BaseEstimator, TransformerMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_PHENOLOGY_ANCHORS",
    "DEFAULT_REDGE_BANDS",
    "SpectralSignatureFeatures",
    "compute_rep",
]


#: Phenology anchors over which the spectral signature is computed. Each
#: anchor maps to a set of columns of the input DataFrame (see
#: :meth:`SpectralSignatureFeatures._extract_anchor_bands`).
DEFAULT_PHENOLOGY_ANCHORS: Final[tuple[str, ...]] = ("sog", "peak", "senescence")

#: Sentinel-2 red-edge bands that dominate the REP computation. The
#: central wavelengths (nm) come from the official
#: ESA Sentinel-2 MSI documentation: B05=703.9, B06=740.2, B07=782.5, B08=835.1.
DEFAULT_REDGE_BANDS: Final[tuple[str, ...]] = ("b05", "b06", "b07", "b08")

#: Central wavelengths in nanometers (ESA S2 MSI specs).
_BAND_WAVELENGTHS_NM: Final[dict[str, float]] = {
    "b04": 664.6,
    "b05": 703.9,
    "b06": 740.2,
    "b07": 782.5,
    "b08": 835.1,
    "b8a": 864.7,
}

#: Supported descriptor types.
Descriptor = Literal["rep", "sam", "redge_moments"]


def compute_rep(
    reflectance_b04: np.ndarray,
    reflectance_b05: np.ndarray,
    reflectance_b06: np.ndarray,
    reflectance_b07: np.ndarray,
) -> np.ndarray:
    """Compute the linear-4-bands Red Edge Position (REP) (Frampton et al. 2013).

    Implements the linearized formula of the "Red Edge Position Linear
    4-bands" version from Frampton's paper (eq. 1):

    .. math::

        REP = 705 + 35 \\times
        \\frac{(R_{B04} + R_{B07}) / 2 - R_{B05}}{R_{B06} - R_{B05}}

    The result is in nm and usually ranges between 700 and 740 nm for
    healthy vegetation. Stressed crops or bare soil produce values outside
    that range — the formula tolerates them (no artificial clipping).

    Args:
        reflectance_b04: B04 reflectance (red, ~665 nm) shape ``(N,)``.
        reflectance_b05: B05 reflectance (red-edge 1, ~704 nm) shape ``(N,)``.
        reflectance_b06: B06 reflectance (red-edge 2, ~740 nm) shape ``(N,)``.
        reflectance_b07: B07 reflectance (red-edge 3, ~783 nm) shape ``(N,)``.

    Returns:
        REP vector in nm, shape ``(N,)``, dtype ``float64``. ``NaN`` values
        where the formula degenerates (denominator ~0 or NaN inputs).

    Raises:
        ValueError: if the 4 arrays do not have the same shape.
    """
    arrays = (reflectance_b04, reflectance_b05, reflectance_b06, reflectance_b07)
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"The 4 bands must have the same shape; got {shapes!r}.")
    b04 = reflectance_b04.astype(np.float64, copy=False)
    b05 = reflectance_b05.astype(np.float64, copy=False)
    b06 = reflectance_b06.astype(np.float64, copy=False)
    b07 = reflectance_b07.astype(np.float64, copy=False)

    denom = b06 - b05
    # Avoid division by zero by producing explicit NaN values.
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, np.nan)
    numerator = (b04 + b07) / 2.0 - b05
    rep = 705.0 + 35.0 * (numerator / safe_denom)
    return rep


class SpectralSignatureFeatures(BaseEstimator, TransformerMixin):
    """Generate compact features derived from the per-parcel spectral signature.

    Sklearn-compatible: fits into ``sklearn.pipeline.Pipeline`` with
    ``StandardScaler``, ``XGBRegressor``, etc. The ``fit`` method learns
    (when applicable) the centroid of the majority class for the ``sam``
    descriptor; ``transform`` always returns a :class:`polars.DataFrame`
    with columns ``parcel_id, year, spectral_signature_NNN``.

    Args:
        descriptor: Descriptor type. One of ``"rep"`` (default, Frampton
            et al. 2013 Red Edge Position), ``"sam"`` (Spectral Angle Mapper
            vs the majority-class centroid) or ``"redge_moments"``
            (mean/var/skew of the red-edge reflectance at each anchor).
        phenology_anchors: Temporal anchors over which each descriptor is
            computed. Default ``("sog", "peak", "senescence")``. For each
            anchor, columns ``{anchor}_{band}`` (e.g. ``sog_b05``) are
            looked up; if absent, the anchor is filled with NaN.
        bands: Required red-edge bands (default
            ``("b05", "b06", "b07", "b08")``). For ``rep``, B04..B07 are used.
        parcel_id_col: Name of the identifier column (default
            ``"parcel_id"``).
        year_col: Name of the year column (default ``"year"``).
        class_col: Class column used by ``sam`` to compute the centroid
            during ``fit``. If ``None``, ``sam`` computes against a vector
            of ones (mere fallback) and warns via a structured warning.
    """

    def __init__(
        self,
        descriptor: Descriptor = "rep",
        phenology_anchors: tuple[str, ...] = DEFAULT_PHENOLOGY_ANCHORS,
        bands: tuple[str, ...] = DEFAULT_REDGE_BANDS,
        parcel_id_col: str = "parcel_id",
        year_col: str = "year",
        class_col: str | None = "class_id",
    ) -> None:
        self.descriptor = descriptor
        self.phenology_anchors = phenology_anchors
        self.bands = bands
        self.parcel_id_col = parcel_id_col
        self.year_col = year_col
        self.class_col = class_col

    # ------------------------------------------------------------------
    # Sklearn API.
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pl.DataFrame,
        y: object | None = None,
    ) -> SpectralSignatureFeatures:
        """Learn the centroid of the majority class (``sam`` only).

        Args:
            X: Polars DataFrame with at least ``parcel_id``, ``year`` and the
                spectral columns required by the descriptor.
            y: Ignored (sklearn signature).

        Returns:
            The ``self`` instance for chaining.

        Raises:
            ValueError: if ``descriptor`` is not one of the supported values.
        """
        if self.descriptor not in ("rep", "sam", "redge_moments"):
            raise ValueError(
                f"`descriptor` must be 'rep', 'sam' or 'redge_moments'; got {self.descriptor!r}."
            )

        self.centroid_: np.ndarray | None = None
        if self.descriptor == "sam":
            self.centroid_ = self._fit_centroid(X)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Produce the ``parcel_id, year, spectral_signature_NNN`` DataFrame.

        Args:
            X: Polars DataFrame with the required spectral columns.

        Returns:
            Polars DataFrame with shape ``(N, 2 + K)`` where ``K`` depends
            on the descriptor (3 for default ``rep``, 1 for ``sam``, 9 for
            default ``redge_moments``).

        Raises:
            ValueError: if ``parcel_id`` or ``year`` are not in ``X``.
        """
        self._validate_input(X)

        n = X.height
        if self.descriptor == "rep":
            feats = self._transform_rep(X)
        elif self.descriptor == "sam":
            feats = self._transform_sam(X)
        else:  # redge_moments
            feats = self._transform_redge_moments(X)

        k = feats.shape[1]
        feat_cols = [f"spectral_signature_{i:03d}" for i in range(k)]

        out_dict: dict[str, list[object]] = {
            self.parcel_id_col: X.get_column(self.parcel_id_col).to_list(),
            self.year_col: X.get_column(self.year_col).to_list(),
        }
        for j, name in enumerate(feat_cols):
            out_dict[name] = feats[:, j].tolist()

        schema: dict[str, pl.DataType] = {
            self.parcel_id_col: X.schema[self.parcel_id_col],
            self.year_col: X.schema[self.year_col],
        }
        for name in feat_cols:
            schema[name] = pl.Float32()

        logger.info(
            "spectral_signature_transformed",
            descriptor=self.descriptor,
            n_rows=n,
            n_features=k,
        )
        return pl.DataFrame(out_dict, schema=schema)

    def fit_transform(  # type: ignore[override]
        self,
        X: pl.DataFrame,
        y: object | None = None,
        **fit_params: object,
    ) -> pl.DataFrame:
        """Sklearn fit_transform: ``self.fit(X, y).transform(X)``."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Private helpers.
    # ------------------------------------------------------------------

    def _validate_input(self, X: pl.DataFrame) -> None:
        """Validate that the DataFrame carries the minimum columns."""
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"`X` must be a polars.DataFrame; got {type(X)!r}.")
        missing = [c for c in (self.parcel_id_col, self.year_col) if c not in X.columns]
        if missing:
            raise ValueError(
                f"`X` does not contain required columns: {missing}. "
                f"Expected at least: ['{self.parcel_id_col}', '{self.year_col}']."
            )

    def _extract_anchor_bands(
        self,
        X: pl.DataFrame,
        anchor: str,
        bands: Sequence[str],
    ) -> np.ndarray:
        """Return a ``(N, len(bands))`` matrix of reflectances.

        Looks for columns with prefix ``{anchor}_{band}``. If absent, it
        tries the US-018 subset-style column fallback ``{band}_mean``
        (ignores the anchor). If those are missing too, it fills with NaN
        to preserve the shape contract.
        """
        n = X.height
        out = np.full((n, len(bands)), np.nan, dtype=np.float64)
        for j, band in enumerate(bands):
            candidate_cols = [
                f"{anchor}_{band}",
                f"{band}_{anchor}",
                f"{band}_mean",
                band,
            ]
            for col in candidate_cols:
                if col in X.columns and X.schema[col].is_numeric():
                    out[:, j] = X.get_column(col).cast(pl.Float64).to_numpy()
                    break
        return out

    def _transform_rep(self, X: pl.DataFrame) -> np.ndarray:
        """Compute REP at each phenology anchor.

        Requires the 4 bands B04/B05/B06/B07 per anchor. If any is missing,
        the resulting column stays NaN for that anchor.
        """
        n = X.height
        out = np.full((n, len(self.phenology_anchors)), np.nan, dtype=np.float64)
        required_bands = ("b04", "b05", "b06", "b07")
        for j, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, required_bands)
            out[:, j] = compute_rep(
                bands_matrix[:, 0],
                bands_matrix[:, 1],
                bands_matrix[:, 2],
                bands_matrix[:, 3],
            )
        return out

    def _transform_sam(self, X: pl.DataFrame) -> np.ndarray:
        """Compute Spectral Angle Mapper vs the centroid learned during fit.

        Returns a scalar per parcel: the cosine of the angle between the
        parcel's mean signature (concatenation of red-edge bands at the
        anchors) and the centroid. With no learned centroid (fit not called
        or no ``class_col``), it yields the cosine vs a vector of ones.
        """
        signatures = self._stack_signatures(X)
        if self.centroid_ is None:
            centroid = np.ones(signatures.shape[1], dtype=np.float64)
            logger.warning(
                "spectral_signature_sam_no_centroid",
                hint="llamar fit() con class_col valido para SAM significativo",
            )
        else:
            centroid = self.centroid_

        # Cosine similarity row-wise, robust to NaN (replaces them with 0).
        sig = np.where(np.isfinite(signatures), signatures, 0.0)
        cen = np.where(np.isfinite(centroid), centroid, 0.0)
        num = sig @ cen
        denom = np.linalg.norm(sig, axis=1) * (np.linalg.norm(cen) + 1e-12)
        safe_denom = np.where(denom > 1e-12, denom, np.nan)
        return (num / safe_denom).reshape(-1, 1)

    def _transform_redge_moments(self, X: pl.DataFrame) -> np.ndarray:
        """Compute mean/var/skew of the red-edge bands per anchor.

        Returns ``K = 3 * len(phenology_anchors)`` columns: for each anchor,
        the 3 statistical moments over the red-edge curve.
        """
        n = X.height
        out = np.full((n, 3 * len(self.phenology_anchors)), np.nan, dtype=np.float64)
        for j, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, self.bands)
            # Impute NaN within each row with the row's own mean.
            row_means = np.nanmean(bands_matrix, axis=1)
            imputed = bands_matrix.copy()
            for i in range(n):
                if np.isnan(imputed[i]).any():
                    fill_value = row_means[i] if np.isfinite(row_means[i]) else 0.0
                    imputed[i] = np.where(np.isnan(imputed[i]), fill_value, imputed[i])
            mean_row = imputed.mean(axis=1)
            var_row = imputed.var(axis=1)
            # Classic skewness: m3 / m2^(3/2). Tolerates degenerate distributions.
            centred = imputed - mean_row[:, None]
            m2 = (centred**2).mean(axis=1)
            m3 = (centred**3).mean(axis=1)
            safe_m2 = np.where(m2 > 1e-12, m2, np.nan)
            skew_row = m3 / np.power(safe_m2, 1.5)
            out[:, j * 3] = mean_row
            out[:, j * 3 + 1] = var_row
            out[:, j * 3 + 2] = skew_row
        return out

    def _stack_signatures(self, X: pl.DataFrame) -> np.ndarray:
        """Build the concatenated signature (anchors x bands) per parcel.

        Returns:
            Matrix ``(N, len(phenology_anchors) * len(bands))`` with all the
            reflectances in the order ``anchor0_band0, anchor0_band1, ...,
            anchorK_bandJ``.
        """
        n = X.height
        out = np.full(
            (n, len(self.phenology_anchors) * len(self.bands)),
            np.nan,
            dtype=np.float64,
        )
        for a_idx, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, self.bands)
            start = a_idx * len(self.bands)
            out[:, start : start + len(self.bands)] = bands_matrix
        return out

    def _fit_centroid(self, X: pl.DataFrame) -> np.ndarray:
        """Compute the centroid of the majority class for SAM."""
        if self.class_col is None or self.class_col not in X.columns:
            logger.warning(
                "spectral_signature_no_class_col",
                class_col=self.class_col,
                hint="SAM degrada a coseno vs vector de unos",
            )
            return np.ones(len(self.phenology_anchors) * len(self.bands), dtype=np.float64)

        class_counts = X.group_by(self.class_col).len().sort("len", descending=True)
        majority_class = class_counts.row(0, named=True)[self.class_col]
        subset = X.filter(pl.col(self.class_col) == majority_class)
        signatures = self._stack_signatures(subset)
        # Centroid = column-by-column mean, ignoring NaN.
        centroid = np.nanmean(signatures, axis=0)
        # If the whole column was NaN, fill it with 0 (contributes no angle).
        centroid = np.where(np.isfinite(centroid), centroid, 0.0)
        logger.info(
            "spectral_signature_centroid_fitted",
            majority_class=majority_class,
            n_parcels_class=subset.height,
            n_signature_dims=centroid.size,
        )
        return centroid
