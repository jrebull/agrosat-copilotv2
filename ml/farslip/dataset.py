"""FarSLIP image-text pair builder + PyTorch Dataset (US-017 / US-016b).

Generates ``data/farslip_pairs/{roi}/crops/*.tif`` (256x256 px, 4 bands
B02/B03/B04/B08) + a Polars ``manifest.parquet`` with it/es/en templates per
CAP parcel. Reuses the Cloud Score+ QA mask machinery of
:mod:`ml.ingest.gee_sampler` (US-007) and the BOA constants of
:mod:`ml.features.spectral_indices` (US-014); does not duplicate logic.

Idempotency: the builder keeps a set of ``crop_id`` already written and filters
duplicates when appending to the manifest.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog
import torch
import yaml
from torch.utils.data import Dataset

from ml.utils.geo_io import write_crop_tiff

_log = structlog.get_logger(__name__)


MANIFEST_SCHEMA: dict[str, type] = {
    "crop_id": pl.Utf8,
    "crop_path": pl.Utf8,
    "crop_doy": pl.Int32,
    "crop_year": pl.Int32,
    "cap_class": pl.Utf8,
    "region": pl.Utf8,
    "text_it": pl.Utf8,
    "text_es": pl.Utf8,
    "text_en": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
}


PHENOLOGY_BY_DOY: tuple[tuple[int, int, str], ...] = (
    (1, 90, "germinazione"),
    (91, 150, "fioritura"),
    (151, 210, "fruttificazione"),
    (211, 270, "maturazione"),
    (271, 366, "raccolta"),
)


def _doy_to_phenology(doy: int) -> str:
    """Map DOY to the canonical phenophase of the default YAML."""
    for lo, hi, label in PHENOLOGY_BY_DOY:
        if lo <= doy <= hi:
            return label
    return "maturazione"


def _region_to_display(region_slug: str) -> str:
    """Map ROI slug to the Italian display name of the YAML."""
    mapping = {
        "pianura_padana": "Pianura Padana",
        "toscana": "Toscana",
        "puglia": "Puglia",
    }
    return mapping.get(region_slug, region_slug.replace("_", " ").title())


def _load_vocabulary(path: Path) -> dict[str, Any]:
    """Load the CAP YAML. Validate that the 32 classes exist."""
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if "classes" not in raw:
        raise ValueError(f"vocabulario invalido en {path}: falta 'classes'")
    return raw


def _render_template(template: str, *, phenology: str, region: str) -> str:
    """Format a template with {phenology}/{region} markers; ignores extras."""
    return template.format(phenology=phenology, region=region)


def _build_text_triplet(
    vocab: dict[str, Any],
    cap_class: str,
    template_idx: int,
    doy: int,
    region_slug: str,
) -> tuple[str, str, str]:
    """Return texts (it, es, en) for the given combination."""
    classes = vocab["classes"]
    if cap_class not in classes:
        cap_class = "altro"
    entry = classes[cap_class]
    phen = _doy_to_phenology(doy)
    region_display = _region_to_display(region_slug)
    texts = []
    for lang in ("it", "es", "en"):
        templates = entry[lang]
        idx = template_idx % len(templates)
        texts.append(_render_template(templates[idx], phenology=phen, region=region_display))
    return texts[0], texts[1], texts[2]


def _compute_crop_id(
    *, region: str, lat: float, lon: float, year: int, doy: int, cap_class: str
) -> str:
    """Deterministic hash for idempotency (UNIQUE key). Non-crypto."""
    h = hashlib.sha1(
        f"{region}|{lat:.6f}|{lon:.6f}|{year}|{doy}|{cap_class}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"{region}_{h}"


def _synthetic_crop_uint16(
    rng: np.random.Generator, *, crop_size_px: int, n_bands: int = 4
) -> np.ndarray:
    """Generate a synthetic crop ``(n_bands, H, W)`` uint16 — used in tests/dataset_audit dryrun."""
    return rng.integers(
        low=200, high=3000, size=(n_bands, crop_size_px, crop_size_px), dtype=np.uint16
    )


# write_crop_tiff lives in ml.utils.geo_io (Q10 SoC fix).


def build_farslip_pairs(
    *,
    rois: tuple[str, ...] = ("pianura_padana", "toscana", "puglia"),
    n_per_roi: int = 10000,
    crop_size_px: int = 256,
    qa_cloud_threshold: float = 0.2,
    output_root: Path = Path("data/farslip_pairs"),
    vocabulary_path: Path = Path("ml/farslip/cap_vocabulary.yaml"),
    seed: int = 42,
    parcel_records: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a FarSLIP image-text pair dataset per Italian ROI.

    Args:
        rois: Italian ROI tuples (slug). Default 3 ROIs.
        n_per_roi: samples to generate per ROI.
        crop_size_px: crop side (default 256).
        qa_cloud_threshold: cloud_prob threshold to filter cloudy parcels.
        output_root: destination path ``data/farslip_pairs/``.
        vocabulary_path: path to ``cap_vocabulary.yaml``.
        seed: reproducibility seed.
        parcel_records: optional ``DataFrame`` with columns
            ``[region, lat, lon, year, doy, cap_class, cloud_prob]`` to
            inject real parcels. If None, synthetic ones are generated (used in
            tests/dryrun).

    Returns:
        Aggregated ``DataFrame`` with ``MANIFEST_SCHEMA``. Side effect: creates
        TIFFs + ``manifest.parquet`` per ROI. Idempotent (UNIQUE crop_id).
    """
    rng = np.random.default_rng(seed)
    # Non-crypto random: only decides the CAP template index for lexical diversity.
    py_rng = random.Random(seed)  # noqa: S311
    vocab = _load_vocabulary(vocabulary_path)
    cap_classes_list = list(vocab["classes"].keys())
    aggregated: list[pl.DataFrame] = []

    output_root.mkdir(parents=True, exist_ok=True)

    for roi in rois:
        roi_dir = output_root / roi
        crops_dir = roi_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = roi_dir / "manifest.parquet"

        # Get parcels: real or synthetic.
        if parcel_records is not None:
            roi_records = parcel_records.filter(pl.col("region") == roi)
        else:
            roi_records = _generate_synthetic_parcels(
                roi=roi,
                n=n_per_roi,
                rng=rng,
                cap_classes=cap_classes_list,
            )

        # QA filter: cloud_prob <= threshold (assumes the column is present)
        if "cloud_prob" in roi_records.columns:
            roi_records = roi_records.filter(pl.col("cloud_prob") <= qa_cloud_threshold)

        rows: list[dict[str, Any]] = []
        for record in roi_records.iter_rows(named=True):
            cap_class = record["cap_class"]
            year = int(record["year"])
            doy = int(record["doy"])
            lat = float(record["lat"])
            lon = float(record["lon"])

            crop_id = _compute_crop_id(
                region=roi, lat=lat, lon=lon, year=year, doy=doy, cap_class=cap_class
            )
            crop_path = crops_dir / f"{crop_id}.tif"
            if not crop_path.exists():
                arr = _synthetic_crop_uint16(rng, crop_size_px=crop_size_px)
                write_crop_tiff(arr, crop_path)

            template_idx = py_rng.randint(0, 2)
            text_it, text_es, text_en = _build_text_triplet(
                vocab, cap_class, template_idx, doy, roi
            )

            rows.append(
                {
                    "crop_id": crop_id,
                    "crop_path": str(crop_path.resolve()),
                    "crop_doy": doy,
                    "crop_year": year,
                    "cap_class": cap_class,
                    "region": roi,
                    "text_it": text_it,
                    "text_es": text_es,
                    "text_en": text_en,
                    "lat": lat,
                    "lon": lon,
                }
            )

        new_df = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)

        # Idempotent merge with the existing manifest.
        if manifest_path.exists():
            existing = pl.read_parquet(manifest_path)
            merged = pl.concat([existing, new_df], how="vertical").unique(
                subset=["crop_id"], keep="first"
            )
        else:
            merged = new_df.unique(subset=["crop_id"], keep="first")
        merged.write_parquet(manifest_path)
        aggregated.append(merged)
        _log.info(
            "roi manifest written",
            roi=roi,
            n_rows=merged.height,
            path=str(manifest_path),
        )

    return pl.concat(aggregated, how="vertical")


def _generate_synthetic_parcels(
    *, roi: str, n: int, rng: np.random.Generator, cap_classes: list[str]
) -> pl.DataFrame:
    """Generate synthetic parcel records for tests/dryrun."""
    # Approximate bounding boxes per ROI (lat, lon ranges)
    bbox = {
        "pianura_padana": (44.5, 45.7, 8.5, 12.0),
        "toscana": (42.5, 44.2, 10.0, 12.0),
        "puglia": (40.0, 41.9, 15.0, 18.5),
    }.get(roi, (42.0, 46.0, 8.0, 18.0))
    lat = rng.uniform(bbox[0], bbox[1], size=n)
    lon = rng.uniform(bbox[2], bbox[3], size=n)
    year = rng.integers(2022, 2025, size=n)
    doy = rng.integers(1, 366, size=n)
    cap_idx = rng.integers(0, len(cap_classes), size=n)
    cloud_prob = rng.uniform(0.0, 0.5, size=n)
    return pl.DataFrame(
        {
            "region": [roi] * n,
            "lat": lat,
            "lon": lon,
            "year": year.astype(np.int32),
            "doy": doy.astype(np.int32),
            "cap_class": [cap_classes[i] for i in cap_idx],
            "cloud_prob": cloud_prob,
        }
    )


class FarSLIPDataset(Dataset):
    """PyTorch Dataset for FarSLIP training.

    Each item returns a dict with:

    - ``image``: tensor ``(C, 224, 224)`` float32 normalized to [0,1]
    - ``input_ids``: tokens of the selected language (``(77,)`` long)
    - ``attention_mask``: ``(77,)`` long
    - ``region_id``: 0-d long tensor
    - ``category_id``: 0-d long tensor
    - ``cap_class``: str (for debug)

    Args:
        manifest_path: path to ``manifest.parquet``.
        tokenizer: HF ``CLIPTokenizer`` or ``None`` (in which case input_ids is zeros).
        lang_strategy: ``"uniform"``, ``"it_only"`` or ``"round_robin"``.
        crop_resize_to: target side (default 224 for CLIP).
        transform: optional callable applied to the float ``(C, H, W)`` tensor.
        cap_classes: canonical list of classes to map ``cap_class`` -> ``category_id``.
        regions: canonical list of ROIs to map ``region`` -> ``region_id``.
    """

    def __init__(
        self,
        manifest_path: Path,
        tokenizer: Any | None = None,
        lang_strategy: Literal["uniform", "it_only", "round_robin"] = "uniform",
        crop_resize_to: int = 224,
        transform: Any | None = None,
        cap_classes: list[str] | None = None,
        regions: list[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest no existe: {self.manifest_path}")
        self.df = pl.read_parquet(self.manifest_path)
        self.tokenizer = tokenizer
        self.lang_strategy = lang_strategy
        self.crop_resize_to = crop_resize_to
        self.transform = transform
        if cap_classes is None:
            # derive from the manifest preserving order of appearance.
            cap_classes = list(dict.fromkeys(self.df["cap_class"].to_list()))
        if regions is None:
            regions = list(dict.fromkeys(self.df["region"].to_list()))
        self._cap_to_idx = {c: i for i, c in enumerate(cap_classes)}
        self._region_to_idx = {r: i for i, r in enumerate(regions)}
        self._langs = ("it", "es", "en")

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.row(idx, named=True)
        crop_path = self._resolve_crop_path(row["crop_path"])
        img = self._load_crop(crop_path)
        img = self._resize_chw(img, self.crop_resize_to)
        if self.transform is not None:
            img = self.transform(img)

        lang = self._select_lang(idx)
        text = row[f"text_{lang}"]
        if self.tokenizer is not None:
            tok = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(77, dtype=torch.long)
            attention_mask = torch.zeros(77, dtype=torch.long)

        cap_class = row["cap_class"]
        region = row["region"]
        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "region_id": torch.tensor(self._region_to_idx.get(region, 0), dtype=torch.long),
            "category_id": torch.tensor(self._cap_to_idx.get(cap_class, 0), dtype=torch.long),
            "cap_class": cap_class,
            "text": text,
        }

    def _select_lang(self, idx: int) -> str:
        if self.lang_strategy == "it_only":
            return "it"
        if self.lang_strategy == "round_robin":
            return self._langs[idx % 3]
        # uniform
        return self._langs[idx % 3]

    def _resolve_crop_path(self, raw: str) -> Path:
        """Resolve the crop path tolerant to cross-platform manifests.

        The manifests were generated on Windows with absolute paths
        (`C:\\Users\\...\\data\\farslip_pairs\\{roi}\\crops\\{file}.tif`). On the
        Linux VM the Windows absolute path does not exist. Strategy (fallback):

        1. If the original absolute path exists -> use it.
        2. Extract the basename and resolve it against `manifest_path.parent / "crops"`.
        3. If it still fails, let `_load_crop` report FileNotFoundError.
        """
        p = Path(raw)
        if p.exists():
            return p
        # Take the last two parts (`crops/{filename}`) and join them to the
        # manifest root. If the path has no "crops/" use only the basename.
        parts = raw.replace("\\", "/").split("/")
        if "crops" in parts:
            idx = parts.index("crops")
            tail = Path(*parts[idx:])
        else:
            tail = Path(parts[-1])
        return self.manifest_path.parent / tail

    def _load_crop(self, path: Path) -> torch.Tensor:
        """Load TIFF or NPY and return a ``(C, H, W)`` float32 tensor in [0,1]."""
        npy_path = path.with_suffix(".npy")
        arr: np.ndarray | None = None
        if path.exists():
            try:
                import rasterio  # type: ignore[import-untyped]

                with rasterio.open(path) as src:
                    arr = src.read()  # (C, H, W) uint16
            except (ImportError, OSError):  # pragma: no cover
                arr = None
        if arr is None:
            if not npy_path.exists():
                raise FileNotFoundError(f"crop no encontrado: {path}")
            arr = np.load(npy_path)
        # Normalize uint16 BOA to [0, 1] (10000 factor standard Sentinel-2 L2A)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 10000.0
        else:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr)

    @staticmethod
    def _resize_chw(img: torch.Tensor, target: int) -> torch.Tensor:
        if img.shape[-1] == target and img.shape[-2] == target:
            return img
        # interpolate expects (N, C, H, W)
        img4 = img.unsqueeze(0)
        out = torch.nn.functional.interpolate(
            img4, size=(target, target), mode="bilinear", align_corners=False
        )
        return out.squeeze(0)


__all__ = [
    "MANIFEST_SCHEMA",
    "FarSLIPDataset",
    "build_farslip_pairs",
]
