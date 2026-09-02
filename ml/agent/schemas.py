"""Pydantic v2 contracts for the nine geospatial agent tools.

This module is the single source of truth for tool input/output schemas. Each
tool exposes one ``*Input`` model (validated arguments coming from the LLM) and
one ``*Output`` model (typed result returned to the agent loop). Shared
value-object types (geometry, bounding box, parcel/AOI references) live here too
so that every tool speaks the same vocabulary.

All models use ``strict`` typing so that the LLM-provided JSON is validated
exactly (no silent ``"3"`` -> ``3`` coercions on critical numeric fields) before
any tool runs. The JSON schema of every ``*Input`` model is later derived by
``ml.agent.tools.build_function_declarations`` to register the tools with the
``google-genai`` SDK.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ml.eval.class_remap import DEFAULT_LABEL_SPACE

__all__ = [
    "AddAoiInput",
    "AoiRef",
    "AoiStats",
    "AoiStatsInput",
    "BBox",
    "ClassificationResult",
    "ClassifyParcelInput",
    "CompareModelsInput",
    "CropModel",
    "ExplainPredictionInput",
    "Explanation",
    "GeoJSONGeometry",
    "GetTilesInput",
    "ListParcelsInput",
    "ModelComparison",
    "ParcelList",
    "ParcelRef",
    "ParcelTimeseriesInput",
    "SceneList",
    "SearchStacInput",
    "TileUrl",
    "TimeSeries",
]

# ``strict`` rejects implicit type coercion (e.g. str -> int); ``extra="forbid"``
# rejects unknown keys hallucinated by the LLM. Shared by every contract below.
_STRICT_CONFIG = ConfigDict(strict=True, extra="forbid")


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------
class GeoJSONGeometry(BaseModel):
    """A GeoJSON geometry as produced by the frontend draw tools.

    Only the geometry object is modelled (not a full ``Feature``). ``type`` is
    constrained to the OGC geometry primitives the agent accepts; ``coordinates``
    keeps the raw nested list because its depth depends on ``type``.

    Attributes:
        type: GeoJSON geometry type (e.g. ``"Polygon"``, ``"MultiPolygon"``).
        coordinates: Raw GeoJSON coordinate array (nesting depends on ``type``).
    """

    model_config = _STRICT_CONFIG

    type: str
    coordinates: list

    # ``ClassVar`` so Pydantic v2 treats this as a plain class constant rather
    # than a ``ModelPrivateAttr`` (a bare leading-underscore attribute would be a
    # private attr, which is not iterable inside the validator).
    _ALLOWED_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
        }
    )

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        """Reject geometry types outside the supported OGC primitives."""
        if value not in cls._ALLOWED_TYPES:
            allowed = ", ".join(sorted(cls._ALLOWED_TYPES))
            raise ValueError(f"unsupported GeoJSON geometry type {value!r}; allowed: {allowed}")
        return value

    @field_validator("coordinates")
    @classmethod
    def _validate_coordinates(cls, value: list) -> list:
        """Reject an empty coordinate array (a geometry must have vertices)."""
        if not value:
            raise ValueError("coordinates must not be empty")
        return value


class BBox(BaseModel):
    """Axis-aligned bounding box in EPSG:4326 (lon/lat degrees).

    Attributes:
        minx: Minimum longitude (west edge).
        miny: Minimum latitude (south edge).
        maxx: Maximum longitude (east edge).
        maxy: Maximum latitude (north edge).
    """

    model_config = _STRICT_CONFIG

    minx: float
    miny: float
    maxx: float
    maxy: float

    @field_validator("minx", "maxx")
    @classmethod
    def _validate_lon(cls, value: float) -> float:
        """Constrain longitude to the valid [-180, 180] range."""
        if not -180.0 <= value <= 180.0:
            raise ValueError(f"longitude {value} out of range [-180, 180]")
        return value

    @field_validator("miny", "maxy")
    @classmethod
    def _validate_lat(cls, value: float) -> float:
        """Constrain latitude to the valid [-90, 90] range."""
        if not -90.0 <= value <= 90.0:
            raise ValueError(f"latitude {value} out of range [-90, 90]")
        return value


class ParcelRef(BaseModel):
    """Lightweight reference to a parcel returned by listing/search tools.

    Attributes:
        parcel_id: Primary key of the parcel in the ``parcels`` table.
        crop_class: Predicted crop class label, if known.
        confidence: Classifier confidence in ``[0, 1]``, if known.
    """

    model_config = _STRICT_CONFIG

    parcel_id: int
    crop_class: str | None = None
    confidence: float | None = None


class AoiRef(BaseModel):
    """Reference to a persisted Area Of Interest.

    Doubles as the output of ``add_aoi`` (the created AOI).

    Attributes:
        aoi_id: Primary key of the AOI in the ``aois`` table.
        label: Human-readable AOI label, if any.
        area_ha: AOI area in hectares, if computed.
    """

    model_config = _STRICT_CONFIG

    aoi_id: int
    label: str | None = None
    area_ha: float | None = None


# ---------------------------------------------------------------------------
# list_parcels
# ---------------------------------------------------------------------------
class ListParcelsInput(BaseModel):
    """Arguments for ``list_parcels``.

    Attributes:
        session_id: Tenant session; every DB query filters by it.
        aoi: Optional polygon to spatially restrict the listing.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    aoi: GeoJSONGeometry | None = None


class ParcelList(BaseModel):
    """Result of ``list_parcels``.

    Attributes:
        parcels: Parcels visible to the session (optionally within the AOI).
        count: Number of parcels returned.
    """

    model_config = _STRICT_CONFIG

    parcels: list[ParcelRef]
    count: int


# ---------------------------------------------------------------------------
# get_parcel_timeseries
# ---------------------------------------------------------------------------
class ParcelTimeseriesInput(BaseModel):
    """Arguments for ``get_parcel_timeseries``.

    Attributes:
        session_id: Tenant session.
        parcel_id: Parcel whose temporal index is requested.
        start: Inclusive start date of the window.
        end: Inclusive end date of the window.
        index: Spectral index to extract.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    parcel_id: int
    start: date
    end: date
    index: Literal["ndvi", "ndwi", "evi"]

    @field_validator("end")
    @classmethod
    def _validate_window(cls, value: date, info: ValidationInfo) -> date:
        """Ensure the end date is not before the start date."""
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError(f"end {value} must not be before start {start}")
        return value


class TimeSeries(BaseModel):
    """Result of ``get_parcel_timeseries``.

    Attributes:
        parcel_id: Parcel the series belongs to.
        index: Spectral index name echoed back.
        dates: Observation dates (ascending), aligned with ``values``.
        values: Index values aligned one-to-one with ``dates``.
    """

    model_config = _STRICT_CONFIG

    parcel_id: int
    index: str
    dates: list[date]
    values: list[float]

    @field_validator("values")
    @classmethod
    def _validate_aligned(cls, value: list[float], info: ValidationInfo) -> list[float]:
        """Ensure ``values`` and ``dates`` have matching length."""
        dates = info.data.get("dates")
        if dates is not None and len(value) != len(dates):
            raise ValueError(f"values length {len(value)} != dates length {len(dates)}")
        return value


# ---------------------------------------------------------------------------
# get_aoi_stats
# ---------------------------------------------------------------------------
class AoiStatsInput(BaseModel):
    """Arguments for ``get_aoi_stats``.

    Attributes:
        session_id: Tenant session.
        aoi: Polygon over which crop statistics are aggregated.
        year: Campaign year of the AlphaEarth annual embedding.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    aoi: GeoJSONGeometry
    year: int

    @field_validator("year")
    @classmethod
    def _validate_year(cls, value: int) -> int:
        """Constrain the year to the AlphaEarth annual coverage range."""
        if not 2017 <= value <= 2100:
            raise ValueError(f"year {value} out of supported range [2017, 2100]")
        return value


class AoiStats(BaseModel):
    """Result of ``get_aoi_stats``.

    Attributes:
        area_ha: Total AOI area in hectares.
        dominant_crop: Most frequent crop class inside the AOI.
        crop_fractions: Per-class area fraction in ``[0, 1]`` summing to ~1.
        n_parcels: Number of parcels intersecting the AOI.
    """

    model_config = _STRICT_CONFIG

    area_ha: float
    dominant_crop: str
    crop_fractions: dict[str, float]
    n_parcels: int


# ---------------------------------------------------------------------------
# search_stac (deferred)
# ---------------------------------------------------------------------------
class SearchStacInput(BaseModel):
    """Arguments for ``search_stac`` (pgstac scene search).

    Attributes:
        bbox: Bounding box to search within.
        datetime_range: RFC 3339 interval string (e.g. ``"2019-01-01/2019-12-31"``).
        cloud_cover_max: Maximum acceptable cloud cover percentage.
    """

    model_config = _STRICT_CONFIG

    bbox: BBox
    datetime_range: str
    cloud_cover_max: float = 20.0

    @field_validator("datetime_range")
    @classmethod
    def _validate_datetime_range(cls, value: str) -> str:
        """Require a non-empty STAC datetime interval string."""
        if not value.strip():
            raise ValueError("datetime_range must not be empty")
        return value

    @field_validator("cloud_cover_max")
    @classmethod
    def _validate_cloud_cover(cls, value: float) -> float:
        """Constrain cloud cover to a valid percentage."""
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"cloud_cover_max {value} out of range [0, 100]")
        return value


class SceneList(BaseModel):
    """Result of ``search_stac``.

    Attributes:
        scenes: STAC item dictionaries matching the query.
        count: Number of scenes returned.
    """

    model_config = _STRICT_CONFIG

    scenes: list[dict]
    count: int


# ---------------------------------------------------------------------------
# get_tiles (deferred)
# ---------------------------------------------------------------------------
class GetTilesInput(BaseModel):
    """Arguments for ``get_tiles`` (TiTiler tile-template URL).

    Attributes:
        scene_id: STAC scene identifier to render.
        index: Visual product to render (spectral index or natural color).
    """

    model_config = _STRICT_CONFIG

    scene_id: str
    index: Literal["ndvi", "ndwi", "evi", "rgb"]

    @field_validator("scene_id")
    @classmethod
    def _validate_scene_id(cls, value: str) -> str:
        """Require a non-empty scene identifier."""
        if not value.strip():
            raise ValueError("scene_id must not be empty")
        return value


class TileUrl(BaseModel):
    """Result of ``get_tiles``.

    Attributes:
        scene_id: Scene identifier echoed back.
        index: Rendered product echoed back.
        tile_url: XYZ tile template URL (contains ``{z}/{x}/{y}`` placeholders).
    """

    model_config = _STRICT_CONFIG

    scene_id: str
    index: str
    tile_url: str


# ---------------------------------------------------------------------------
# classify_new_parcel
# ---------------------------------------------------------------------------

#: Serving models selectable for ``classify_new_parcel``. SINGLE SOURCE OF TRUTH
#: for the crop-model tag: :class:`ClassifyParcelInput.model`, the perceiver's
#: AOI path and the ``/chat`` request body all reuse this alias instead of
#: re-declaring the ``Literal`` (the frontend mirror lives in
#: ``frontend/types/agent.ts``). Adding a model = editing this line only.
CropModel = Literal["xgb", "voting3", "stacking5"]


class ClassifyParcelInput(BaseModel):
    """Arguments for ``classify_new_parcel`` (honest per-parcel crop classifier).

    By default this serves the ``voting3`` EPIC 12 deployment champion (the
    weighted soft-vote of ``tsvit-pheno-v2`` + ``utae`` + ``xgb-alphaearth``,
    US-081 AC4a), NOT a generic "stacking ensemble". The active model is selected
    by ``model`` (and the back-compat ``use_stacking`` flag), and two independent
    flags refine the posterior:

    - ``model`` (default ``"voting3"``) picks the serving model for a parcel
      already materialized in the cached fold-5 OOF: ``"voting3"`` (the real
      EPIC 12 deployment champion -- the v2 weighted soft-vote of ``tsvit-pheno``
      + ``utae`` + ``xgb-alphaearth``, france-10 F1 0.9069 -> 0.9264, france-12
      0.9001 > Stacking-5 0.8927; the copilot default since US-081), ``"xgb"``
      (the tabular member, the historical default kept for back-compat) or
      ``"stacking5"`` (the EPIC 6 Stacking-5 meta, kept as LEGACY). Any model that
      cannot resolve the parcel (a fresh polygon) or whose OOF artifacts are
      unavailable degrades cleanly to ``xgb-alphaearth`` with a structured
      warning -- never a fabricated posterior. With the ``voting3`` default a
      fresh AOI therefore behaves exactly as the historical ``xgb`` default did.
    - ``restrict_to_resolved_classes`` (default ON) masks the posterior down to
      the well-resolved classes of ``label_space`` (``france-12`` by default, the
      twelve classes with restricted macro-F1 >= 0.90 OOF fold-5) and
      renormalizes. This trades 18-class breadth for honesty about which classes
      the model resolves.
    - ``use_stacking`` (default OFF) is the LEGACY selector kept for back-compat:
      when ``True`` and ``model`` is set explicitly to ``"xgb"``, it is treated as
      ``model="stacking5"``. With the new ``"voting3"`` default the legacy flag is
      IGNORED (the champion is authoritative). Prefer ``model`` for new callers.

    Attributes:
        session_id: Tenant session.
        aoi: Polygon of the new parcel to classify.
        year: Campaign year of the AlphaEarth annual embedding (default 2019).
        restrict_to_resolved_classes: When ``True`` (default) the posterior is
            masked + renormalized over the active label-space's resolved classes;
            when ``False`` the full 18-class posterior is returned.
        model: Serving model for a fold-5 parcel: ``"voting3"`` (default EPIC 12
            weighted-vote champion), ``"xgb"`` (tabular member, historical
            default) or ``"stacking5"`` (EPIC 6 Stacking-5 meta, legacy). Each
            degrades cleanly to ``xgb-alphaearth`` when it cannot resolve the
            parcel or its OOF artifacts are unavailable. NOT the last word: when
            the user pinned a model in the UI,
            :attr:`ml.agent.context.ToolContext.crop_model` OVERRIDES this
            argument (the switch is a hard choice enforced at the tool boundary).
            Read ``ClassificationResult.served_model`` for what actually ran.
        use_stacking: LEGACY back-compat flag. Honoured ONLY when ``model`` is set
            explicitly to ``"xgb"`` (then promoted to ``model="stacking5"`` so an
            old caller passing ``use_stacking=True`` still gets the Stacking-5
            posterior). Ignored under the new ``"voting3"`` default and for any
            other explicit ``model``. Default ``False``.
        label_space: Name of the registered label-space whose resolved classes
            gate the posterior when ``restrict_to_resolved_classes`` is on.
            Defaults to :data:`~ml.eval.class_remap.DEFAULT_LABEL_SPACE` (the
            copilot's configured crop vocabulary) rather than a hardcoded name.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    aoi: GeoJSONGeometry
    year: int = 2019
    restrict_to_resolved_classes: bool = True
    model: CropModel = "voting3"
    use_stacking: bool = False
    label_space: str = DEFAULT_LABEL_SPACE

    @property
    def resolved_model(self) -> str:
        """Resolve the effective serving model from ``model`` + ``use_stacking``.

        ``model`` is authoritative and now defaults to the ``"voting3"`` champion
        (US-081 AC4a). The legacy ``use_stacking`` flag is honoured ONLY when
        ``model`` is set explicitly to ``"xgb"`` (so an old caller passing
        ``model="xgb", use_stacking=True`` still gets the Stacking-5 posterior),
        and is otherwise ignored -- under the new ``"voting3"`` default the flag is
        a no-op. This keeps the historical legacy behaviour intact while serving
        the champion by default and letting new callers select ``"voting3"`` /
        ``"stacking5"`` explicitly.

        Returns:
            One of ``"xgb"``, ``"voting3"`` or ``"stacking5"``.
        """
        if self.model == "xgb" and self.use_stacking:
            return "stacking5"
        return self.model

    @field_validator("year")
    @classmethod
    def _validate_year(cls, value: int) -> int:
        """Constrain the year to the AlphaEarth annual coverage range."""
        if not 2017 <= value <= 2100:
            raise ValueError(f"year {value} out of supported range [2017, 2100]")
        return value

    @field_validator("label_space")
    @classmethod
    def _validate_label_space(cls, value: str) -> str:
        """Require a non-empty label-space name (resolved at run time)."""
        if not value.strip():
            raise ValueError("label_space must not be empty")
        return value


class ClassificationResult(BaseModel):
    """Result of ``classify_new_parcel``.

    Attributes:
        crop_class: Argmax crop class predicted by the ensemble.
        confidence: Probability of ``crop_class`` in ``[0, 1]``.
        class_probabilities: Full posterior over crop classes.
        out_of_vocabulary_classes: Crop names the active label-space does NOT
            resolve reliably (its dropped set). Populated when restricting so the
            reasoner knows the model's vocabulary boundary and can hand an
            out-of-scope crop to RAG + phenology instead of forcing a label.
        unresolved_candidate: When restricting and the model's RAW (unrestricted)
            top class falls OUTSIDE the resolved vocabulary, the name of that
            out-of-vocabulary crop the raw signal leaned toward -- the reasoner's
            explicit cue that ``crop_class`` may be a renormalization artifact and
            should be hedged with neighbouring-parcel grounding, not reported as
            confident. ``None`` when the raw top class is in vocabulary.
        served_model: The ensemble member that actually produced this posterior
            (``"voting-3"``, ``"xgb-alphaearth"`` or ``"stacking-5"``), reflecting
            any degradation. When the caller requested Voting-3 but the parcel
            fell outside the fold-5 OOF universe, this is exactly
            ``"xgb-alphaearth"`` (not ``"voting-3"``) so the reasoner and the UI
            stay honest about the active model. Empty string for sentinel results
            (e.g. ``needs_gee_sampling``) that did not run a model.
    """

    model_config = _STRICT_CONFIG

    crop_class: str
    confidence: float
    class_probabilities: dict[str, float]
    out_of_vocabulary_classes: list[str] = Field(default_factory=list)
    unresolved_candidate: str | None = None
    served_model: str = ""


# ---------------------------------------------------------------------------
# add_aoi (deferred)
# ---------------------------------------------------------------------------
class AddAoiInput(BaseModel):
    """Arguments for ``add_aoi`` (persist an AOI for the session).

    Attributes:
        session_id: Tenant session that owns the AOI.
        aoi: Polygon geometry to persist.
        name: Human-readable label for the AOI.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    aoi: GeoJSONGeometry
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Require a non-empty AOI name."""
        if not value.strip():
            raise ValueError("name must not be empty")
        return value


# ``add_aoi`` returns an ``AoiRef`` (the created AOI); no dedicated output model.


# ---------------------------------------------------------------------------
# compare_models (deferred)
# ---------------------------------------------------------------------------
class CompareModelsInput(BaseModel):
    """Arguments for ``compare_models``.

    Attributes:
        session_id: Tenant session.
        parcel_id: Parcel whose predictions are compared across models.
        models: Model member names to compare (e.g. ``["tsvit-pheno", "utae"]``).
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    parcel_id: int
    models: list[str]

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        """Require at least two distinct models to compare."""
        if len(value) < 2:
            raise ValueError("compare_models requires at least two models")
        if len(set(value)) != len(value):
            raise ValueError("models must be unique")
        return value


class ModelComparison(BaseModel):
    """Result of ``compare_models``.

    Attributes:
        parcel_id: Parcel the comparison refers to.
        predictions: Mapping of model name -> predicted crop class.
        agreement: Fraction of models agreeing with the majority in ``[0, 1]``.
    """

    model_config = _STRICT_CONFIG

    parcel_id: int
    predictions: dict[str, str]
    agreement: float


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------
class ExplainPredictionInput(BaseModel):
    """Arguments for ``explain_prediction``.

    Attributes:
        session_id: Tenant session.
        parcel_id: Parcel whose prediction is explained.
    """

    model_config = _STRICT_CONFIG

    session_id: UUID
    parcel_id: int


class Explanation(BaseModel):
    """Result of ``explain_prediction`` (entry point of the Be My Eyes pattern).

    Attributes:
        parcel_id: Parcel the explanation refers to.
        crop_class: Predicted crop class being explained.
        confidence: Confidence of the prediction in ``[0, 1]``.
        phenology_text: Structured phenology text block (SOG/peak/senescence).
        vigor: Qualitative vigor assessment (e.g. ``"high"``, ``"moderate"``).
        description: Natural-language explanation suitable for the final answer.
    """

    model_config = _STRICT_CONFIG

    parcel_id: int
    crop_class: str
    confidence: float
    phenology_text: str
    vigor: str
    description: str
