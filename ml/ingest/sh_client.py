"""Sentinel Hub Process API client -- real raster crops with spatial texture.

The :mod:`ml.ingest.cdse_client` ``CDSEClient`` only DISCOVERS scenes (the OData
catalogue: which products exist for an area/date). It does not return pixels. The
dense champion members (TSViT-pheno, U-TAE) need a spatial PATCH with real
texture, not a single pixel -- and EuroCropsML ships only the per-parcel
pixel-reduced series. This client closes that gap: it pulls on-the-fly raster
crops from the Sentinel Hub Process API hosted on CDSE
(``https://sh.dataspace.copernicus.eu/api/v1/process``), authenticated with the
SAME CDSE OAuth client-credentials pair (confirmed: the CDSE confidential client
created at ``shapps.dataspace.copernicus.eu`` is Sentinel-Hub-enabled).

It builds, per parcel, a temporal stack ``(T, n_bands, H, W)`` of L2A surface
reflectance with the 10 PASTIS-R bands in the order the dense models expect, so a
parcel can be fed to TSViT/U-TAE with the texture they were trained on.

Honesty
-------
- Credentials live only in ``.env.local`` (gitignored); this module reads them
  via settings and never hardcodes them.
- Every crop is a real Process API response (a GeoTIFF decoded with rasterio); an
  empty/failed crop is skipped and counted, never fabricated.
- Network + quota cost is real: each (parcel, date) is one Process API request.
  Callers cap the parcel count and the date count for a pilot.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["SHCrop", "SentinelHubClient", "sh_client_from_settings"]

#: Sentinel Hub Process API endpoint hosted on CDSE.
_PROCESS_URL: str = "https://sh.dataspace.copernicus.eu/api/v1/process"

#: CDSE Keycloak token endpoint (same realm the OData client uses). Public URL.
_TOKEN_URL: str = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # noqa: S105
)

#: The 10 PASTIS-R bands (surface reflectance) in the order the dense models read
#: them. Matches ``ml.transfer.ensemble_full_tl._PASTIS_BANDS``.
PASTIS_BANDS: tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

#: Evalscript returning the 10 PASTIS bands as FLOAT32 surface reflectance. The
#: Process API multiplies DN by the L2A scale internally, so the output is already
#: in [0, 1]-ish reflectance.
_EVALSCRIPT: str = (
    "//VERSION=3\n"
    "function setup(){return {input:["
    + ",".join(f'"{b}"' for b in PASTIS_BANDS)
    + "],output:{bands:"
    + str(len(PASTIS_BANDS))
    + ',sampleType:"FLOAT32"}};}\n'
    "function evaluatePixel(s){return [" + ",".join(f"s.{b}" for b in PASTIS_BANDS) + "];}"
)


def _orbit_evalscript(n_frames: int) -> str:
    """Build a multi-temporal evalscript that returns ``n_frames`` x 10 bands.

    Uses ``mosaicking: "ORBIT"`` so ``evaluatePixel`` receives one sample per
    acquisition. It emits the most-recent ``n_frames`` orbits (newest first),
    each as the 10 PASTIS bands, concatenated into a single ``n_frames * 10``-band
    output. When fewer than ``n_frames`` orbits exist the missing frames are
    zero-filled (the caller drops all-zero frames). This collapses a whole season
    into ONE Process API request per parcel instead of one request per window.

    Per-PIXEL cloud masking (coherence): besides the scene-level
    ``maxCloudCoverage`` filter, each pixel is zeroed when the Scene
    Classification Layer (SCL) marks it cloud/shadow/cirrus/snow (SCL in
    {3 shadow, 8 cloud-medium, 9 cloud-high, 10 cirrus, 11 snow}). This keeps the
    temporal stack physically coherent -- a cloudy pixel injects spurious
    reflectance that corrupts the phenology signal the dense models read.

    Args:
        n_frames: Number of temporal frames to emit.

    Returns:
        The evalscript source string.
    """
    bands_in = ",".join(f'"{b}"' for b in PASTIS_BANDS)
    nb = len(PASTIS_BANDS)
    total = n_frames * nb
    # For each frame f and band b: 0 when the frame is absent OR the pixel is
    # cloud/shadow/cirrus/snow per SCL; otherwise the reflectance value.
    out_terms: list[str] = []
    for f in range(n_frames):
        clear = (
            f"(n>{f}"
            f"&&samples[{f}].SCL!=3&&samples[{f}].SCL!=8&&samples[{f}].SCL!=9"
            f"&&samples[{f}].SCL!=10&&samples[{f}].SCL!=11)"
        )
        for b in PASTIS_BANDS:
            out_terms.append(f"({clear}?samples[{f}].{b}:0)")
    return (
        "//VERSION=3\n"
        f'function setup(){{return {{input:[{{bands:[{bands_in},"SCL"]}}],'
        f'output:{{bands:{total},sampleType:"FLOAT32"}},mosaicking:"ORBIT"}};}}\n'
        "function evaluatePixel(samples){var n=samples.length;return ["
        + ",".join(out_terms)
        + "];}"
    )


_TOKEN_REFRESH_MARGIN_S: float = 30.0

#: Base backoff (seconds) for the Process API 429 retry; attempt i waits
#: ``base * 2**i`` -> 1, 2, 4, 8 s.
_RETRY_BACKOFF_BASE_S: float = 1.0

#: Cap on the honoured ``Retry-After`` (seconds). SH may ask for ~200s; with
#: concurrent workers that stalls the run, so we cap and rely on the per-minute
#: budget recovering while a few workers trickle through.
_RETRY_MAX_WAIT_S: float = 20.0


def _jitter(spread: float = 5.0) -> float:
    """Return a small per-call jitter (seconds) to de-synchronise workers.

    Derived from the sub-second fraction of the monotonic clock (no RNG), so two
    workers hitting a 429 in the same instant wait slightly different amounts and
    stop retrying in lockstep.

    Args:
        spread: Maximum jitter in seconds.

    Returns:
        A value in ``[0, spread)``.
    """
    import time as _t

    return (_t.monotonic() % 1.0) * spread


@dataclass(frozen=True)
class SHCrop:
    """A single Sentinel Hub raster crop for a parcel at one date window.

    Attributes:
        data: Reflectance array ``(n_bands, H, W)`` float32 in PASTIS band order.
        date: The mid date of the requested window (ISO ``YYYY-MM-DD``).
    """

    data: np.ndarray
    date: str


class SentinelHubClient:
    """Process API client: real raster crops authenticated with CDSE OAuth.

    The access token is fetched lazily and cached until just before expiry. All
    HTTP goes through one injected :class:`httpx.Client` so tests can stub it.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        token_url: str = _TOKEN_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialise with the CDSE confidential OAuth pair.

        Args:
            client_id: CDSE OAuth client id (Sentinel-Hub-enabled).
            client_secret: CDSE OAuth client secret (only in ``.env.local``).
            token_url: Keycloak token endpoint of the CDSE realm.
            http_client: Optional injected HTTP client.

        Raises:
            ValueError: if either credential is empty.
        """
        if not client_id or not client_secret:
            raise ValueError(
                "SentinelHubClient needs the CDSE client-credentials pair "
                "(cdse_client_id / cdse_client_secret in .env.local)."
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._http = http_client or httpx.Client(timeout=120.0)
        self._access_token: str | None = None
        self._token_expiry: float = 0.0

    def _ensure_token(self) -> str:
        """Return a valid bearer token, fetching/refreshing as needed."""
        now = time.monotonic()
        if self._access_token is not None and now < self._token_expiry:
            return self._access_token
        response = self._http.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        self._access_token = str(payload["access_token"])
        self._token_expiry = now + float(payload.get("expires_in", 600.0)) - _TOKEN_REFRESH_MARGIN_S
        return self._access_token

    def post_process(
        self, payload: dict[str, Any], *, max_attempts: int = 5
    ) -> httpx.Response | None:
        """POST to the Process API, retrying on ``429`` with exponential backoff.

        Public entry point so other modules (e.g. :mod:`ml.ingest.sh_path`) reuse
        the token + URL + retry logic instead of touching private members or
        hardcoding the endpoint.

        The Process API enforces a per-second request quota; under fan-out a
        ``429 RATE_LIMIT_EXCEEDED`` is expected and transient. This retries it
        (honouring a ``Retry-After`` header when present, else exponential
        backoff), so a rate-limited request is not silently lost -- which would
        bias the experiment by dropping parcels.

        Args:
            payload: The Process API request body.
            max_attempts: Maximum attempts before giving up.

        Returns:
            The successful :class:`httpx.Response`, or ``None`` when every attempt
            was rate-limited / failed.
        """
        import time as _time

        for attempt in range(max_attempts):
            token = self._ensure_token()
            response = self._http.post(
                _PROCESS_URL,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
            if response.status_code == 200:
                return response
            if response.status_code == 429 and attempt < max_attempts - 1:
                # SH returns a per-minute rate-limit ``Retry-After`` (often ~200s).
                # Respecting the full value with concurrent workers makes them all
                # sleep in lockstep and the run stalls. Cap the wait and add a
                # per-worker jitter so requests de-synchronise and trickle through
                # under the rate budget instead of hammering it in bursts.
                retry_after = response.headers.get("Retry-After")
                raw = float(retry_after) if retry_after else _RETRY_BACKOFF_BASE_S * (2**attempt)
                wait = min(raw, _RETRY_MAX_WAIT_S) + _jitter()
                logger.info("sh_rate_limited_retry", attempt=attempt + 1, wait_s=round(wait, 1))
                _time.sleep(wait)
                continue
            logger.warning(
                "sh_process_failed", status=response.status_code, body=response.text[:160]
            )
            return None
        return None

    def crop(
        self,
        bbox: tuple[float, float, float, float],
        *,
        date_from: str,
        date_to: str,
        size: int = 16,
        max_cloud: float = 30.0,
    ) -> np.ndarray | None:
        """Fetch one ``(n_bands, size, size)`` reflectance crop for a bbox/window.

        Args:
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
            date_from: Window start ISO date (``YYYY-MM-DD``).
            date_to: Window end ISO date.
            size: Output side in pixels (the patch is ``size x size``).
            max_cloud: Maximum scene cloud-cover percentage for the mosaic.

        Returns:
            The crop ``(n_bands, size, size)`` float32, or ``None`` when the
            Process API returns no data (logged, never fabricated).
        """
        token = self._ensure_token()
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_from}T00:00:00Z",
                                "to": f"{date_to}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud,
                        },
                    }
                ],
            },
            "output": {
                "width": size,
                "height": size,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": _EVALSCRIPT,
        }
        response = self._http.post(
            _PROCESS_URL,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        if response.status_code != 200:
            logger.warning(
                "sh_crop_failed",
                status=response.status_code,
                body=response.text[:200],
            )
            return None
        arr = _decode_tiff(response.content)
        if arr is None or arr.size == 0 or not np.isfinite(arr).any() or arr.max() == 0.0:
            logger.info("sh_crop_empty", date_from=date_from, date_to=date_to)
            return None
        return arr

    def parcel_series(
        self,
        lon: float,
        lat: float,
        *,
        windows: list[tuple[str, str]],
        size: int = 16,
        half_side_deg: float = 0.0008,
        max_cloud: float = 30.0,
    ) -> np.ndarray | None:
        """Build a temporal patch stack ``(T, n_bands, size, size)`` for a parcel.

        Centres a small bbox on the parcel centroid and pulls one crop per date
        window, stacking the successful crops over time. A window that yields no
        data is skipped; if fewer than two windows succeed the parcel is dropped
        (a dense temporal model needs at least two dates).

        Args:
            lon: Parcel centroid longitude.
            lat: Parcel centroid latitude.
            windows: List of ``(date_from, date_to)`` ISO windows spanning the
                season (e.g. monthly).
            size: Patch side in pixels.
            half_side_deg: Half the bbox side in degrees (~90 m at 0.0008).
            max_cloud: Maximum cloud cover per window.

        Returns:
            A ``(T, n_bands, size, size)`` float32 stack, or ``None`` when fewer
            than two windows return data.
        """
        bbox = (
            lon - half_side_deg,
            lat - half_side_deg,
            lon + half_side_deg,
            lat + half_side_deg,
        )
        frames: list[np.ndarray] = []
        for date_from, date_to in windows:
            crop = self.crop(
                bbox, date_from=date_from, date_to=date_to, size=size, max_cloud=max_cloud
            )
            if crop is not None:
                frames.append(crop)
        if len(frames) < 2:
            logger.info("sh_parcel_series_insufficient", n_frames=len(frames))
            return None
        return np.stack(frames, axis=0).astype(np.float32)

    def parcel_series_orbit(
        self,
        lon: float,
        lat: float,
        *,
        date_from: str,
        date_to: str,
        n_frames: int = 12,
        size: int = 128,
        half_side_deg: float = 0.0008,
        max_cloud: float = 25.0,
    ) -> np.ndarray | None:
        """Build a parcel's temporal stack in ONE request (mosaicking=ORBIT).

        ``max_cloud`` defaults to 25 (keep scenes that are >= 75% cloud-free).
        This is the scene-level gate; on top of it the evalscript masks each
        cloudy/shadow/cirrus PIXEL via SCL, so a 25% scene still yields a coherent
        stack (the cloudy pixels are zeroed, not used). The looser scene gate keeps
        MORE usable frames in cloudy regions like the Baltic while the per-pixel
        SCL mask preserves coherence. ``n_frames`` candidates + dropping empty
        frames absorb the rest.

        Pulls the most-recent ``n_frames`` cloud-filtered acquisitions of the
        season in a single Process API call (10 bands x ``n_frames`` frames in one
        multi-band GeoTIFF), then reshapes to ``(T, 10, size, size)`` and drops
        all-zero frames (missing orbits). This is ~``len(windows)``x cheaper than
        :meth:`parcel_series` (one request instead of one per window).

        Args:
            lon: Parcel centroid longitude.
            lat: Parcel centroid latitude.
            date_from: Season start ISO date.
            date_to: Season end ISO date.
            n_frames: Max temporal frames to request.
            size: Patch side in pixels.
            half_side_deg: Half the bbox side in degrees.
            max_cloud: Maximum scene cloud cover.

        Returns:
            A ``(T, 10, size, size)`` float32 stack (``T <= n_frames`` after
            dropping empty frames), or ``None`` when fewer than two frames carry
            data.
        """
        bbox = (
            lon - half_side_deg,
            lat - half_side_deg,
            lon + half_side_deg,
            lat + half_side_deg,
        )
        nb = len(PASTIS_BANDS)
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_from}T00:00:00Z",
                                "to": f"{date_to}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud,
                        },
                    }
                ],
            },
            "output": {
                "width": size,
                "height": size,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": _orbit_evalscript(n_frames),
        }
        response = self.post_process(payload)
        if response is None:
            return None
        flat = _decode_tiff(response.content)  # (n_frames*10, H, W)
        if flat is None or flat.shape[0] != n_frames * nb:
            logger.info("sh_orbit_unexpected_bands", got=None if flat is None else flat.shape)
            return None
        stack = flat.reshape(n_frames, nb, flat.shape[1], flat.shape[2])
        keep = [f for f in range(n_frames) if np.abs(stack[f]).sum() > 0.0]
        if len(keep) < 2:
            logger.info("sh_orbit_insufficient", n_frames=len(keep))
            return None
        return stack[keep].astype(np.float32)

    def parcel_series_batch(
        self,
        coords: list[tuple[float, float]],
        *,
        date_from: str,
        date_to: str,
        n_frames: int = 12,
        size: int = 128,
        half_side_deg: float = 0.0008,
        max_cloud: float = 25.0,
        max_workers: int = 2,
    ) -> list[np.ndarray | None]:
        """Download many parcels' ORBIT stacks concurrently (thread pool).

        Each parcel is one :meth:`parcel_series_orbit` call; the pool runs
        ``max_workers`` of them at a time. A single token is fetched up front so
        the workers share it (the Process API is the bottleneck, not auth).

        Args:
            coords: Parcel centroids ``[(lon, lat), ...]``.
            date_from: Season start ISO date.
            date_to: Season end ISO date.
            n_frames: Max temporal frames per parcel.
            size: Patch side in pixels.
            half_side_deg: Half the bbox side in degrees.
            max_cloud: Maximum scene cloud cover.
            max_workers: Concurrent Process API requests.

        Returns:
            A list aligned with ``coords``; each item is the parcel stack or
            ``None`` when it had insufficient data.
        """
        from concurrent.futures import ThreadPoolExecutor

        self._ensure_token()  # warm the shared token before fanning out

        def _one(lonlat: tuple[float, float]) -> np.ndarray | None:
            lon, lat = lonlat
            return self.parcel_series_orbit(
                lon,
                lat,
                date_from=date_from,
                date_to=date_to,
                n_frames=n_frames,
                size=size,
                half_side_deg=half_side_deg,
                max_cloud=max_cloud,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_one, coords))
        n_ok = sum(1 for r in results if r is not None)
        logger.info("sh_batch_done", n_total=len(coords), n_ok=n_ok, max_workers=max_workers)
        return results


def _decode_tiff(content: bytes) -> np.ndarray | None:
    """Decode a Process API GeoTIFF response to a ``(bands, H, W)`` array.

    Args:
        content: Raw ``image/tiff`` bytes from the Process API.

    Returns:
        The decoded array, or ``None`` when rasterio cannot read the bytes.
    """
    try:
        import rasterio

        with rasterio.open(io.BytesIO(content)) as ds:
            decoded: np.ndarray = ds.read().astype(np.float32)
            return decoded
    except Exception as exc:  # noqa: BLE001 -- a bad tiff is dropped, logged
        logger.warning("sh_tiff_decode_failed", error=str(exc))
        return None


def sh_client_from_settings(settings: Any) -> SentinelHubClient:
    """Build a :class:`SentinelHubClient` from the application settings.

    Args:
        settings: Settings exposing ``cdse_client_id`` / ``cdse_client_secret``
            (and optionally ``cdse_token_url``).

    Returns:
        A ready :class:`SentinelHubClient`.

    Raises:
        ValueError: if the client-credentials pair is unset.
    """
    return SentinelHubClient(
        client_id=getattr(settings, "cdse_client_id", "") or "",
        client_secret=getattr(settings, "cdse_client_secret", "") or "",
        token_url=getattr(settings, "cdse_token_url", _TOKEN_URL),
    )
