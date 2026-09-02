"""FarSLIP embedding extractor for inference (US-017 / US-016b).

Class consumed by US-016 (multisensor fusion) and US-025 (SegFormer-B2
open-vocab head). Loads student weights from GCS with local cache + optional
checksum; graceful fallback to the cache if GCS is offline.

API:
    extract_embeddings(crops) -> (B, 512) float32 L2-norm  (CLIP projection)
    extract_patch_features(crops) -> (B, 196, 768) float32 (vision last hidden)
    encode_text(texts) -> (N, 512) float32 L2-norm

Notes:
    - The student was trained with 4 Sentinel-2 bands (B02/B03/B04/B08).
    - The text encoder is the teacher's (frozen) -- that is why ``encode_text``
      loads ``openai/clip-vit-base-patch16`` by default.
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers required for FarSLIPExtractor") from exc

from ml.farslip.distill import adapt_patch_embed_to_n_channels
from ml.utils.gcs_errors import is_gcs_auth_error

_log = structlog.get_logger(__name__)


DEFAULT_TEACHER_ID = "openai/clip-vit-base-patch16"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "agrosat" / "farslip"


class FarSLIPExtractor:
    """Lazy loader of FarSLIP student weights + teacher text encoder.

    Args:
        weights_uri: ``gs://...`` or local path to the ``student.safetensors``
            file (or a directory containing it). If ``None``, uses the CLIP
            teacher weights without distillation (degraded placeholder mode).
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``.
        cache_dir: local cache folder (default ``~/.cache/agrosat/farslip/``).
        n_in_channels: Sentinel-2 bands (default 4).
        teacher_model_id: HF id used for the text encoder + base architecture.
        expected_sha1: optional SHA1 checksum of student.safetensors.
    """

    def __init__(
        self,
        weights_uri: str | None = "gs://agrosat-models/farslip/farslip-clip-italy-v1/",
        device: str = "auto",
        cache_dir: Path | None = None,
        n_in_channels: int = 4,
        teacher_model_id: str = DEFAULT_TEACHER_ID,
        expected_sha1: str | None = None,
    ) -> None:
        self.weights_uri = weights_uri
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_in_channels = n_in_channels
        self.teacher_model_id = teacher_model_id
        self.expected_sha1 = expected_sha1

        # Base model (vision + text projection). We load the full CLIPModel
        # to have access to visual_projection and text_projection (512-dim).
        base = CLIPModel.from_pretrained(teacher_model_id)
        base.eval()
        # Adapt vision_model.embeddings.patch_embedding to n_in_channels.
        adapt_patch_embed_to_n_channels(base.vision_model, n_in_channels)
        self.model = base.to(self.device)  # type: ignore[arg-type]

        self.tokenizer = CLIPTokenizer.from_pretrained(teacher_model_id)

        # Load student weights if available.
        weights_local = self._resolve_weights_local()
        if weights_local is not None:
            self._load_student_weights(weights_local)
        else:
            _log.warning(
                "FarSLIP weights no disponibles; corriendo en modo teacher (degradado)",
                weights_uri=weights_uri,
            )

        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_weights_local(self) -> Path | None:
        """Resolves to a local path. Downloads from GCS if needed; local cache fallback."""
        if self.weights_uri is None:
            return None
        uri = str(self.weights_uri)

        # If it is a direct local path (file://, absolute or existing relative path)
        local_candidate = Path(uri)
        if local_candidate.is_file():
            return local_candidate
        if local_candidate.is_dir():
            cand = local_candidate / "student.safetensors"
            if cand.is_file():
                return cand
            return None

        if uri.startswith("gs://"):
            cached = self._maybe_use_cache(uri)
            if cached is not None:
                # Before downloading, try to use a valid cache.
                if self._validate_checksum(cached):
                    _log.info("usando cache local valido (sin GCS)", path=str(cached))
                    return cached

            try:
                return self._download_from_gcs(uri)
            except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
                _log.warning(
                    "GCS download fallido; intentando cache local",
                    error=str(exc),
                    uri=uri,
                )
                if cached is not None and cached.exists():
                    return cached
                return None
            except Exception as exc:
                # google.api_core / google.auth: 403, 401, NotFound, missing
                # creds. Degrade to local cache or teacher mode instead
                # of propagating and breaking the constructor. Any other error
                # (real AttributeError, KeyError, ValueError) bubbles up.
                if is_gcs_auth_error(exc):
                    _log.warning(
                        "GCS auth/permiso denegado; intentando cache local",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        uri=uri,
                    )
                    if cached is not None and cached.exists():
                        return cached
                    return None
                raise
        return None

    def _cache_path_for(self, uri: str) -> Path:
        # sha1 without usedforsecurity flag: cache deduplication only, not crypto.
        h = hashlib.sha1(uri.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return self.cache_dir / f"{h}_student.safetensors"

    def _maybe_use_cache(self, uri: str) -> Path | None:
        cached = self._cache_path_for(uri)
        return cached if cached.exists() else None

    def _validate_checksum(self, path: Path) -> bool:
        if self.expected_sha1 is None:
            return True
        try:
            h = hashlib.sha1(usedforsecurity=False)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest() == self.expected_sha1
        except OSError:  # pragma: no cover
            return False

    #: Retries for auto-download in case of invalid checksum after wait.
    _DOWNLOAD_MAX_ATTEMPTS = 3
    #: Seconds to wait per polling iteration when another process is downloading.
    _DOWNLOAD_WAIT_INTERVAL_S = 0.5
    #: Maximum polling iterations (= 30 s with interval 0.5 s).
    _DOWNLOAD_WAIT_MAX_ITERATIONS = 60

    def _is_complete_download(self, path: Path) -> bool:
        """Validate that ``path`` exists, has ``size > 0`` and matches the checksum.

        This covers the case where a parallel process died mid-download
        leaving a partial file (size=0 or invalid checksum). The checksum is
        only evaluated if ``self.expected_sha1`` is set; in the absence of a
        known hash we rely on the size check.
        """
        if not path.exists():
            return False
        try:
            if path.stat().st_size == 0:
                return False
        except OSError:  # pragma: no cover
            return False
        return self._validate_checksum(path)

    def _download_from_gcs(self, uri: str) -> Path:
        from google.cloud import storage  # type: ignore[import-untyped]

        if not uri.startswith("gs://"):
            raise ValueError(f"Non-GCS URI: {uri}")
        without_scheme = uri[len("gs://") :]
        parts = without_scheme.split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""
        # If it points to a folder, append student.safetensors.
        if blob_path.endswith("/") or not blob_path.endswith(".safetensors"):
            blob_path = blob_path.rstrip("/") + "/student.safetensors"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        dest = self._cache_path_for(uri)

        import time as _time

        for attempt in range(1, self._DOWNLOAD_MAX_ATTEMPTS + 1):
            lock_path = dest.with_suffix(".safetensors.lock")
            try:
                with lock_path.open("x"):
                    try:
                        blob.download_to_filename(str(dest))
                    except BaseException:
                        # MT-3 fix: if download_to_filename blows up (403,
                        # NotFound, network drop) it may have created the
                        # destination file empty or partial. Clean up before propagating
                        # so the next invocation does not see that garbage as
                        # a valid cache (without expected_sha1 _validate_checksum
                        # returns True and load_file blows up with
                        # SafetensorError: header too small).
                        with suppress(OSError):
                            dest.unlink()
                        raise
            except FileExistsError:
                # Another execution is already downloading. Poll the final file.
                for _ in range(self._DOWNLOAD_WAIT_MAX_ITERATIONS):
                    if dest.exists():
                        # Wait for the other process to release the lock before
                        # validating (avoids reading during the final flush).
                        if not lock_path.exists():
                            break
                    _time.sleep(self._DOWNLOAD_WAIT_INTERVAL_S)
            finally:
                with suppress(OSError):
                    lock_path.unlink()

            # Post-wait/post-download validation (Q7): size>0 + checksum.
            if self._is_complete_download(dest):
                return dest

            _log.warning(
                "download incompleto/checksum invalido; reintentando",
                attempt=attempt,
                dest=str(dest),
            )
            with suppress(OSError):
                dest.unlink()

        raise RuntimeError(
            f"GCS download failed after {self._DOWNLOAD_MAX_ATTEMPTS} attempts: {uri}"
        )

    def _load_student_weights(self, path: Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(path), device=str(self.device))
        # The weights come from CLIPVisionModel (student state). We load into the
        # internal vision_model; we ignore missing/unexpected keys (text encoder).
        missing, unexpected = self.model.vision_model.load_state_dict(state, strict=False)
        if missing:
            _log.warning("missing keys al cargar student", n=len(missing))
        if unexpected:
            _log.warning("unexpected keys al cargar student", n=len(unexpected))
        _log.info("FarSLIP student weights cargados", path=str(path))

    # ------------------------------------------------------------------ API

    @torch.inference_mode()
    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        """Extract CLS embeddings projected to 512-dim, L2-norm.

        Args:
            crops: ``(B, 4, H, W)`` float [0,1] or raw uint16 (normalized).

        Returns:
            ``(B, 512)`` float32 L2-normalized.
        """
        crops = self._prep_crops(crops)
        vision_out = self.model.vision_model(pixel_values=crops)
        pooled = vision_out.pooler_output  # (B, 768)
        embeds = self.model.visual_projection(pooled)  # (B, 512)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds.float()

    @torch.inference_mode()
    def extract_patch_features(self, crops: torch.Tensor) -> torch.Tensor:
        """Extract patch features (without CLS) ``(B, 196, 768)`` for SegFormer US-025."""
        crops = self._prep_crops(crops)
        vision_out = self.model.vision_model(pixel_values=crops)
        # last_hidden_state: (B, 1+P, 768). Remove CLS.
        hidden: torch.Tensor = vision_out.last_hidden_state
        return hidden[:, 1:, :].float()

    def load_crops_batch(self, paths: list[str | Path]) -> torch.Tensor:
        """Read Sentinel-2 crops from TIFF paths and return a batch tensor.

        Public helper consumed by the Dagster asset ``farslip_embeddings_italy``
        (asset name kept for lineage; its content is PASTIS-R, not Italian) to
        abstract the TIFF I/O (4 bands B02/B03/B04/B08 at 10 m, uint16 scaled
        reflectance). The fine preprocessing (uint16 -> [0,1], resize to
        224x224) is applied inside ``extract_embeddings`` via ``_prep_crops``.

        Args:
            paths: list of paths to ``.tif`` (4 bands, same expected shape).

        Returns:
            ``(B, 4, H, W)`` ``torch.int32`` with raw uint16 values (the
            normalization happens downstream).

        Raises:
            ImportError: if ``rasterio`` is not installed.
            FileNotFoundError: if any path does not exist.
        """
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rasterio required for load_crops_batch") from exc

        if not paths:
            raise ValueError("paths is empty")

        arrays = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"crop TIFF does not exist: {path}")
            with rasterio.open(path) as src:
                arr = src.read()  # (C, H, W)
            if arr.shape[0] != self.n_in_channels:
                raise ValueError(
                    f"crop {path} has {arr.shape[0]} bands; expected {self.n_in_channels}"
                )
            arrays.append(arr)

        # Stack (B, C, H, W). Cast to int32 to preserve uint16 without overflow
        # in torch (torch has no native uint16).
        import numpy as np

        batch = np.stack(arrays, axis=0).astype(np.int32)
        return torch.from_numpy(batch)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode texts via the text encoder + text_projection. L2-norm."""
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_out.pooler_output
        embeds = self.model.text_projection(pooled)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds.float()

    # ------------------------------------------------------------------ utils

    def _prep_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """Move crops to device, normalize uint16 -> [0,1], ensure 4 channels and 224x224."""
        if crops.dtype in (torch.int16, torch.int32, torch.uint8):
            crops = crops.to(torch.float32) / 10000.0
        elif crops.dtype == torch.float64:
            crops = crops.to(torch.float32)
        crops = crops.to(self.device)
        if crops.dim() != 4:
            raise ValueError(f"crops must be (B,C,H,W); got {crops.shape}")
        if crops.shape[1] != self.n_in_channels:
            raise ValueError(f"expected C={self.n_in_channels}; got C={crops.shape[1]}")
        target = 224
        if crops.shape[-1] != target or crops.shape[-2] != target:
            crops = F.interpolate(
                crops, size=(target, target), mode="bilinear", align_corners=False
            )
        return crops


__all__ = ["FarSLIPExtractor"]
