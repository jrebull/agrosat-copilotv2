"""``explain_prediction`` tool: structured explanation of a parcel (synchronous).

This is the PERCEIVER entry point of the "Be My Eyes" pattern: it emits a
structured TEXT description of an existing, session-scoped parcel (crop class,
phenology, vigor, confidence and a natural-language summary). The LLM never
classifies pixels itself -- the textual phenology comes from the real Wen et al.
(2025) descriptor (:func:`ml.features.phenology_description.generate_phenology_description`)
run over the parcel's reconstructed NDVI curve.

Data sources (all session-scoped, multi-tenant):
- ``parcels`` -> ``crop_class`` and ``confidence`` of the stored prediction.
- ``features_parcels.phenology`` JSONB -> the NDVI FFT harmonics, from which the
  measured annual NDVI curve is reconstructed (the same inverse-FFT used by the
  phenology models, never a synthetic curve).
- ``features_parcels`` scalar columns (``sog_doy``, ``peak_doy``, ``peak_value``,
  ``senescence_doy``, ``ndvi_auc``, ``maturity_duration_days``) -> the structured
  phenology text block and the qualitative vigor assessment.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import structlog

from ml.agent.context import ToolContext
from ml.agent.schemas import ExplainPredictionInput, Explanation

logger = structlog.get_logger(__name__)

__all__ = ["run"]

#: Spectral index whose curve drives the phenology description.
_INDEX: str = "NDVI"

#: Reconstructed daily-curve length (T) for the inverse-FFT reconstruction.
_SEQUENCE_LENGTH: int = 72

#: ``peak_value`` (max NDVI) thresholds for the qualitative vigor label.
_VIGOR_HIGH: float = 0.7
_VIGOR_MODERATE: float = 0.4

#: Fallback label when a parcel has no stored crop prediction yet.
_UNKNOWN_CROP: str = "unknown"


class _ParcelNotFoundError(Exception):
    """Raised when the parcel does not exist for the current session."""


def _vigor_from_peak(peak_value: float | None) -> str:
    """Map the parcel's peak NDVI to a qualitative vigor label.

    Args:
        peak_value: Maximum NDVI of the season (``features_parcels.peak_value``),
            or ``None`` when not computed.

    Returns:
        ``"high"``, ``"moderate"``, ``"low"`` or ``"unknown"`` (when ``None``).
    """
    if peak_value is None:
        return "unknown"
    if peak_value >= _VIGOR_HIGH:
        return "high"
    if peak_value >= _VIGOR_MODERATE:
        return "moderate"
    return "low"


def _clamp_ndvi(value: float) -> float:
    """Clamp an NDVI-like value to the physical [-1, 1] range.

    The aggregated ``peak_value`` from the feature pipeline can carry saturated
    Sentinel-2 pixels (values >1), which are not physically meaningful NDVI. For a
    user-facing description we clamp to the valid range so the text never reports
    an impossible "NDVI 2.3"; the raw feature still feeds the model unchanged.

    Args:
        value: The raw aggregated NDVI-like value.

    Returns:
        The value clamped to ``[-1.0, 1.0]``.
    """
    return max(-1.0, min(1.0, float(value)))


def _build_phenology_text(record: dict[str, Any]) -> str:
    """Build the structured phenology text block from the real scalar columns.

    Args:
        record: Row of ``features_parcels`` scalar phenology columns.

    Returns:
        A one-line Spanish phenology block citing the measured SOG / peak /
        senescence landmarks and the AUC (no invented numbers).
    """
    sog = record.get("sog_doy")
    peak_doy = record.get("peak_doy")
    peak_value = record.get("peak_value")
    senescence = record.get("senescence_doy")
    ndvi_auc = record.get("ndvi_auc")
    maturity = record.get("maturity_duration_days")

    parts: list[str] = []
    if sog is not None:
        parts.append(f"inicio de crecimiento (SOG) en el dia {int(sog)}")
    if peak_doy is not None and peak_value is not None:
        parts.append(f"pico NDVI {_clamp_ndvi(peak_value):.2f} en el dia {int(peak_doy)}")
    elif peak_value is not None:
        parts.append(f"pico NDVI {_clamp_ndvi(peak_value):.2f}")
    if senescence is not None:
        parts.append(f"senescencia hacia el dia {int(senescence)}")
    if maturity is not None:
        parts.append(f"madurez estimada de {int(maturity)} dias")
    if ndvi_auc is not None:
        parts.append(f"area bajo la curva NDVI de {float(ndvi_auc):.1f}")

    if not parts:
        return "Sin metricas fenologicas registradas para esta parcela."
    return "Fenologia: " + "; ".join(parts) + "."


def _reconstruct_ndvi_curve(phenology_json: dict[str, Any]) -> np.ndarray | None:
    """Reconstruct the parcel's annual NDVI curve from its FFT harmonics.

    The ``features_parcels.phenology`` JSONB stores the NDVI FFT amplitude/phase
    coefficients (``NDVI_fft_amp_k`` / ``NDVI_fft_phase_k``). The inverse FFT is
    delegated to :func:`ml.train.phenology_models._reconstruct_curve` (the same
    reconstruction used by the phenology models), so the curve is the measured
    seasonal signal, never a synthetic one.

    Args:
        phenology_json: Parsed ``phenology`` JSONB (FFT coefficient columns).

    Returns:
        A ``(T,)`` ``float64`` NDVI curve, or ``None`` if the JSONB carries no
        usable FFT coefficients for the NDVI index.
    """
    from ml.train.phenology_models import _reconstruct_curve

    fft_keys = [
        k
        for k in phenology_json
        if k.startswith(f"{_INDEX}_fft_amp_") or k.startswith(f"{_INDEX}_fft_phase_")
    ]
    if not fft_keys:
        return None

    frame = pl.DataFrame({k: [float(phenology_json[k])] for k in fft_keys})
    matrix = _reconstruct_curve(frame, index_name=_INDEX, sequence_length=_SEQUENCE_LENGTH)
    if matrix.shape[0] == 0:
        return None
    curve: np.ndarray = matrix[0].astype(np.float64)
    if not np.any(np.isfinite(curve)) or float(np.nanmax(np.abs(curve))) == 0.0:
        return None
    return curve


async def _fetch_parcel(ctx: ToolContext, parcel_id: int) -> dict[str, Any]:
    """Fetch the parcel prediction and phenology features (session-scoped).

    Args:
        ctx: Tool execution context (pool, session id).
        parcel_id: Parcel to explain.

    Returns:
        A dict with the parcel ``crop_class`` / ``confidence`` and the latest
        ``features_parcels`` row (phenology JSONB + scalar columns).

    Raises:
        _ParcelNotFoundError: if no parcel with that id belongs to the session.
    """
    from ml.agent.db import session_scoped_conn

    query = """
        SELECT
            p.crop_class,
            p.confidence,
            fp.phenology,
            fp.sog_doy,
            fp.peak_doy,
            fp.peak_value,
            fp.senescence_doy,
            fp.ndvi_auc,
            fp.maturity_duration_days
        FROM parcels p
        LEFT JOIN LATERAL (
            SELECT *
            FROM features_parcels f
            WHERE f.parcel_id = p.id
            ORDER BY f.year DESC
            LIMIT 1
        ) fp ON true
        WHERE p.id = $1
          AND p.session_id = $2
        LIMIT 1
    """
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(query, parcel_id, ctx.session_id)

    if row is None:
        raise _ParcelNotFoundError(f"parcel {parcel_id} not found for the current session.")
    return dict(row)


async def run(inp: ExplainPredictionInput, ctx: ToolContext) -> Explanation:
    """Explain a parcel's prediction with phenology, vigor and a description.

    Args:
        inp: Validated arguments (session id, parcel id).
        ctx: Tool execution context (asyncpg pool, settings, session id).

    Returns:
        An :class:`Explanation` carrying the stored crop class and confidence, the
        structured phenology text, a qualitative vigor label and a natural-language
        description generated by the Wen et al. (2025) phenology descriptor over
        the parcel's reconstructed NDVI curve.

    Raises:
        _ParcelNotFoundError: if the parcel is not visible to the session.
    """
    from ml.features.phenology_description import generate_phenology_description

    logger.info(
        "explain_prediction_started",
        session_id=str(inp.session_id),
        parcel_id=inp.parcel_id,
    )

    record = await _fetch_parcel(ctx, inp.parcel_id)

    crop_class = record.get("crop_class") or _UNKNOWN_CROP
    confidence = float(record["confidence"]) if record.get("confidence") is not None else 0.0
    vigor = _vigor_from_peak(record.get("peak_value"))
    phenology_text = _build_phenology_text(record)

    raw_phenology = record.get("phenology")
    phenology_json: dict[str, Any] = {}
    if isinstance(raw_phenology, str):
        try:
            phenology_json = json.loads(raw_phenology)
        except json.JSONDecodeError:
            logger.warning("explain_phenology_json_unparseable", parcel_id=inp.parcel_id)
    elif isinstance(raw_phenology, dict):
        phenology_json = raw_phenology

    curve = _reconstruct_ndvi_curve(phenology_json)
    if curve is not None:
        crop_hint = crop_class if crop_class != _UNKNOWN_CROP else None
        description = generate_phenology_description(
            curve,
            parcel_id=inp.parcel_id,
            crop_type_hint=crop_hint,
            temperature=0.0,
        )
    else:
        # No FFT harmonics stored: fall back to the structured landmark text so
        # the explanation stays grounded in the real scalar metrics (no LLM call,
        # no invented curve).
        logger.info("explain_no_ndvi_curve", parcel_id=inp.parcel_id)
        description = phenology_text

    explanation = Explanation(
        parcel_id=inp.parcel_id,
        crop_class=crop_class,
        confidence=confidence,
        phenology_text=phenology_text,
        vigor=vigor,
        description=description,
    )
    logger.info(
        "explain_prediction_finished",
        session_id=str(inp.session_id),
        parcel_id=inp.parcel_id,
        crop_class=crop_class,
        vigor=vigor,
    )
    return explanation
