"""Tests for the US-037 FarSLIP-pheno vs AlphaEarth evaluation helper.

Zero GPU, zero network, zero real dataset, zero checkpoint: the heavy
:class:`FarSLIPExtractor` and the US-036 dataset builder
(``create_incremental_dataset``) are MOCKED via monkeypatch; the embedding
matrices are synthetic Gaussian clusters with a KNOWN separability so the
silhouette has an expected golden shape. The tests verify the LOGIC of the eval
(per-class silhouette determinism + consistency with the global, apples-to-apples
alignment, the comparative-table deltas, the scope guards and the patch-level
AlphaEarth aggregation), never an actual run.

Section map (docs/us-planning/us-037.md section 6):
    6.1 per-class silhouette deterministic and consistent (AC-4),
    6.2 apples-to-apples alignment + comparative deltas (AC-3, AC-5),
    6.3 scope guards (AC-1),
    plus the AlphaEarth patch-level aggregation (R-GRAN) and the honest verdict
    (R-NOGAIN / R-CLAIM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import scripts.farslip_eval_phenology as ev

# ---------------------------------------------------------------------------
# Synthetic data helpers (clusters with a known separability).
# ---------------------------------------------------------------------------


def _gaussian_clusters(
    *, n_per_class: int, n_dims: int, centers: list[float], scale: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build well/poorly separated Gaussian blobs and their integer labels.

    Args:
        n_per_class: Samples per class.
        n_dims: Embedding dimensionality.
        centers: Per-class mean offset (one per class).
        scale: Gaussian std (small -> separated, large -> overlapping).
        seed: RNG seed.

    Returns:
        ``(matrix, labels)`` with ``matrix (len(centers)*n_per_class, n_dims)``.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    labels: list[int] = []
    for cls, center in enumerate(centers):
        blocks.append(rng.normal(loc=center, scale=scale, size=(n_per_class, n_dims)))
        labels.extend([cls] * n_per_class)
    return np.vstack(blocks), np.asarray(labels)


def _emb_frame(
    matrix: np.ndarray, labels: np.ndarray, *, prefix: str, id_prefix: str = "patch"
) -> pl.DataFrame:
    """Wrap a matrix + labels into a parcel_id/class_id/<prefix>NNN frame."""
    n, d = matrix.shape
    cols = {f"{prefix}{i:03d}": matrix[:, i] for i in range(d)}
    return pl.DataFrame(
        {
            "parcel_id": [f"{id_prefix}{r}" for r in range(n)],
            "class_id": labels.astype(np.int64),
            **cols,
        }
    )


# ---------------------------------------------------------------------------
# 6.1 Per-class silhouette deterministic and consistent (AC-4).
# ---------------------------------------------------------------------------


def test_silhouette_per_class_shape_and_columns() -> None:
    """Returns one row per class with columns [class_id, silhouette_class, n]."""
    matrix, labels = _gaussian_clusters(
        n_per_class=40, n_dims=8, centers=[0.0, 6.0, 12.0], scale=0.2, seed=0
    )
    table = ev.silhouette_per_class(matrix, labels)
    assert table.columns == ["class_id", "silhouette_class", "n"]
    assert table.height == 3
    assert table["class_id"].to_list() == [0, 1, 2]
    assert table["n"].sum() == 120


def test_silhouette_per_class_high_for_separated_low_for_overlapping() -> None:
    """Separated clusters -> high per-class silhouette; overlapping -> low."""
    sep_m, sep_l = _gaussian_clusters(
        n_per_class=60, n_dims=8, centers=[0.0, 8.0], scale=0.1, seed=1
    )
    sep = ev.silhouette_per_class(sep_m, sep_l)
    assert sep["silhouette_class"].min() > 0.8  # both clusters cleanly separated

    ov_m, ov_l = _gaussian_clusters(
        n_per_class=60, n_dims=8, centers=[0.0, 0.05], scale=3.0, seed=2
    )
    ov = ev.silhouette_per_class(ov_m, ov_l)
    assert ov["silhouette_class"].max() < 0.3  # heavily overlapping -> near 0


def test_silhouette_per_class_consistent_with_global() -> None:
    """The n-weighted mean of the per-class values approximates eval_space's global."""
    from ml.eval.embedding_separability import eval_space

    matrix, labels = _gaussian_clusters(
        n_per_class=50, n_dims=8, centers=[0.0, 5.0, 10.0], scale=0.3, seed=3
    )
    table = ev.silhouette_per_class(matrix, labels)
    weighted = (table["silhouette_class"] * table["n"]).sum() / table["n"].sum()
    global_sil = eval_space(matrix, labels, label="toy", n_splits=5).silhouette
    assert abs(weighted - global_sil) < 0.02


def test_silhouette_per_class_deterministic() -> None:
    """Two calls with the same random_state give the identical table."""
    # Large matrix to force the seeded subsample path (> sample_size).
    matrix, labels = _gaussian_clusters(
        n_per_class=300, n_dims=6, centers=[0.0, 5.0, 10.0], scale=0.4, seed=4
    )
    a = ev.silhouette_per_class(matrix, labels, sample_size=200, random_state=42)
    b = ev.silhouette_per_class(matrix, labels, sample_size=200, random_state=42)
    assert a["silhouette_class"].to_list() == b["silhouette_class"].to_list()
    assert a["n"].to_list() == b["n"].to_list()


def test_silhouette_per_class_validates() -> None:
    """Mismatched lengths or a single class raise a clear error."""
    matrix, labels = _gaussian_clusters(
        n_per_class=10, n_dims=4, centers=[0.0, 5.0], scale=0.1, seed=5
    )
    with pytest.raises(ValueError, match="must equal labels length"):
        ev.silhouette_per_class(matrix, labels[:-1])
    single = np.zeros(matrix.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="two distinct classes"):
        ev.silhouette_per_class(matrix, single)


def test_attach_class_names_adds_readable_column() -> None:
    """attach_class_names inserts a class_name column from the map."""
    matrix, labels = _gaussian_clusters(
        n_per_class=20, n_dims=4, centers=[0.0, 5.0], scale=0.2, seed=6
    )
    table = ev.silhouette_per_class(matrix, labels)
    named = ev.attach_class_names(table, {0: "Meadow", 1: "Corn"})
    assert named.columns == ["class_id", "class_name", "silhouette_class", "n"]
    assert set(named["class_name"].to_list()) == {"Meadow", "Corn"}


# ---------------------------------------------------------------------------
# 6.2 Apples-to-apples alignment + comparative deltas (AC-3, AC-5).
# ---------------------------------------------------------------------------


def test_aligned_spaces_same_rows_and_labels() -> None:
    """compare_to_alphaearth evaluates both spaces on the same patches/labels."""
    pheno_m, labels = _gaussian_clusters(
        n_per_class=80, n_dims=12, centers=[0.0, 6.0], scale=0.3, seed=7
    )
    ae_m, _ = _gaussian_clusters(n_per_class=80, n_dims=8, centers=[0.0, 5.0], scale=0.5, seed=8)
    pheno_df = _emb_frame(pheno_m, labels, prefix="emb_")
    ae_df = _emb_frame(ae_m, labels, prefix="dim_").drop("class_id")
    report = ev.compare_to_alphaearth(
        pheno_df=pheno_df,
        alphaearth_df=ae_df,
        per_class_cap=200,
        min_class_samples=10,
    )
    pheno_res = report.results["farslip_pheno"]
    ae_res = report.results["alphaearth_2019"]
    # Same number of samples (apples-to-apples) and the dims documented (R-DIM).
    assert pheno_res.n_samples == ae_res.n_samples == report.n_shared_parcels
    assert pheno_res.n_dims == 12
    assert ae_res.n_dims == 8


def test_comparative_table_has_deltas() -> None:
    """The comparative table carries delta_vs_0163 and delta_vs_alphaearth_here."""
    pheno_m, labels = _gaussian_clusters(
        n_per_class=80, n_dims=12, centers=[0.0, 8.0], scale=0.1, seed=9
    )
    ae_m, _ = _gaussian_clusters(n_per_class=80, n_dims=8, centers=[0.0, 0.1], scale=3.0, seed=10)
    pheno_df = _emb_frame(pheno_m, labels, prefix="emb_")
    ae_df = _emb_frame(ae_m, labels, prefix="dim_").drop("class_id")
    report = ev.compare_to_alphaearth(
        pheno_df=pheno_df,
        alphaearth_df=ae_df,
        per_class_cap=200,
        min_class_samples=10,
    )
    table = report.comparative_table
    assert "delta_vs_0163" in table.columns
    assert "delta_vs_alphaearth_here" in table.columns
    pheno_row = table.filter(pl.col("space") == "farslip_pheno")
    # FarSLIP-pheno is the well-separated space -> beats 0.163 (positive delta).
    assert pheno_row["delta_vs_0163"].item() > 0
    # ... and out-separates the overlapping AlphaEarth here (positive delta).
    assert pheno_row["delta_vs_alphaearth_here"].item() > 0


def test_compare_empty_join_raises() -> None:
    """No shared patches between the two spaces is a hard error."""
    pheno_m, labels = _gaussian_clusters(
        n_per_class=30, n_dims=6, centers=[0.0, 5.0], scale=0.2, seed=11
    )
    pheno_df = _emb_frame(pheno_m, labels, prefix="emb_", id_prefix="pheno")
    ae_m, _ = _gaussian_clusters(n_per_class=30, n_dims=6, centers=[0.0, 5.0], scale=0.2, seed=12)
    ae_df = _emb_frame(ae_m, labels, prefix="dim_", id_prefix="other").drop("class_id")
    with pytest.raises(ValueError, match="no shared patches"):
        ev.compare_to_alphaearth(
            pheno_df=pheno_df,
            alphaearth_df=ae_df,
            per_class_cap=100,
            min_class_samples=5,
        )


def test_verdict_is_honest_does_not_overclaim() -> None:
    """When FarSLIP-pheno does NOT beat 0.163 the verdict says so (R-NOGAIN)."""
    # Overlapping FarSLIP-pheno (silhouette ~0) vs separated AlphaEarth.
    pheno_m, labels = _gaussian_clusters(
        n_per_class=80, n_dims=12, centers=[0.0, 0.1], scale=3.0, seed=13
    )
    ae_m, _ = _gaussian_clusters(n_per_class=80, n_dims=8, centers=[0.0, 8.0], scale=0.1, seed=14)
    pheno_df = _emb_frame(pheno_m, labels, prefix="emb_")
    ae_df = _emb_frame(ae_m, labels, prefix="dim_").drop("class_id")
    report = ev.compare_to_alphaearth(
        pheno_df=pheno_df,
        alphaearth_df=ae_df,
        per_class_cap=200,
        min_class_samples=10,
    )
    assert "NO supera el 0.163" in report.verdict
    # The verdict surfaces the dimensionality caveat, never claims "+5pp".
    assert "Caveat" in report.verdict
    assert "+5pp" not in report.verdict


def test_compare_missing_columns_raise() -> None:
    """Missing class_id / emb_* / dim_* columns are explicit errors."""
    matrix, labels = _gaussian_clusters(
        n_per_class=20, n_dims=6, centers=[0.0, 5.0], scale=0.2, seed=15
    )
    pheno_df = _emb_frame(matrix, labels, prefix="emb_")
    ae_df = _emb_frame(matrix, labels, prefix="dim_").drop("class_id")
    with pytest.raises(ValueError, match="class_id"):
        ev.compare_to_alphaearth(pheno_df=pheno_df.drop("class_id"), alphaearth_df=ae_df)
    with pytest.raises(ValueError, match="emb_"):
        ev.compare_to_alphaearth(
            pheno_df=pheno_df.select(["parcel_id", "class_id"]),
            alphaearth_df=ae_df,
        )


# ---------------------------------------------------------------------------
# AlphaEarth patch-level aggregation (R-GRAN).
# ---------------------------------------------------------------------------


def test_aggregate_alphaearth_to_patch_averages_parcels() -> None:
    """Per-parcel AlphaEarth collapses to patch level by mean, keyed on patch_id."""
    df = pl.DataFrame(
        {
            "parcel_id": ["100_1", "100_2", "200_1"],
            "dim_00": [0.0, 2.0, 5.0],
            "dim_01": [10.0, 20.0, 50.0],
        }
    )
    out = ev.aggregate_alphaearth_to_patch(df).sort("parcel_id")
    assert out["parcel_id"].to_list() == ["100", "200"]
    patch100 = out.filter(pl.col("parcel_id") == "100")
    assert patch100["dim_00"].item() == pytest.approx(1.0)  # mean(0, 2)
    assert patch100["dim_01"].item() == pytest.approx(15.0)  # mean(10, 20)
    assert out.filter(pl.col("parcel_id") == "200")["dim_00"].item() == 5.0


def test_load_alphaearth_for_eval_real_pxid_schema(tmp_path: Path) -> None:
    """The real PASTIS-aligned parquet (px_id/tile/fold) loads keyed on px_id.

    ``px_id`` IS the PASTIS ``ID_PATCH`` (one row per patch), so it becomes
    ``parcel_id`` directly; ``tile`` (the Sentinel-2 MGRS tile, NOT the patch),
    ``fold``, ``lon``, ``lat`` and ``year`` are dropped. This is the schema that
    triggered the US-037 KeyError (no ``parcel_id`` column).
    """
    real = pl.DataFrame(
        {
            "px_id": ["10000", "10002", "10003"],
            "lon": [-1.26, -1.30, -0.36],
            "lat": [49.6, 49.6, 49.3],
            "year": [2019, 2019, 2019],
            "dim_00": [0.1, 0.2, 0.3],
            "dim_01": [1.0, 2.0, 3.0],
            "tile": ["t30uxv", "t30uxv", "t30uxv"],
            "fold": [1, 4, 5],
        }
    )
    p = tmp_path / "alphaearth_at_pastis_fr_full_2019.parquet"
    real.write_parquet(p)

    out = ev.load_alphaearth_for_eval(p)
    assert "parcel_id" in out.columns
    assert out.schema["parcel_id"] == pl.Utf8
    # Keyed on px_id (= ID_PATCH), NOT collapsed to the 1 tile.
    assert sorted(out["parcel_id"].to_list()) == ["10000", "10002", "10003"]
    assert set(c for c in out.columns if c.startswith("dim_")) == {"dim_00", "dim_01"}
    # tile / fold / lon / lat / year are dropped.
    assert "tile" not in out.columns and "fold" not in out.columns


def test_load_alphaearth_for_eval_rejects_unknown_schema(tmp_path: Path) -> None:
    """A parquet without parcel_id nor px_id is a clear error, never a KeyError."""
    bad = pl.DataFrame({"foo": ["a"], "dim_00": [0.0]})
    p = tmp_path / "bad.parquet"
    bad.write_parquet(p)
    with pytest.raises(ValueError, match="px_id"):
        ev.load_alphaearth_for_eval(p)


def test_load_alphaearth_for_eval_then_aggregate_is_patch_level(
    tmp_path: Path,
) -> None:
    """Real schema -> aggregate is an identity (already one row per patch)."""
    real = pl.DataFrame(
        {
            "px_id": ["10000", "10002", "10003"],
            "dim_00": [0.1, 0.2, 0.3],
            "tile": ["t30uxv", "t30uxv", "t30uxv"],
            "fold": [1, 4, 5],
        }
    )
    p = tmp_path / "real.parquet"
    real.write_parquet(p)
    loaded = ev.load_alphaearth_for_eval(p)
    patch = ev.aggregate_alphaearth_to_patch(loaded).sort("parcel_id")
    # One row per patch survives; the single tile is NOT collapsed into one row.
    assert patch["parcel_id"].to_list() == ["10000", "10002", "10003"]
    assert patch["dim_00"].to_list() == pytest.approx([0.1, 0.2, 0.3])


def test_aggregate_then_compare_aligns_on_patch() -> None:
    """Aggregated AlphaEarth (patch-level) aligns with the FarSLIP-pheno patches."""
    pheno_df = pl.DataFrame(
        {
            "parcel_id": ["100", "200", "300", "400"],
            "class_id": [0, 1, 0, 1],
            "emb_000": [0.0, 5.0, 0.1, 5.1],
            "emb_001": [0.0, 5.0, 0.1, 5.1],
        }
    )
    ae_parcels = pl.DataFrame(
        {
            "parcel_id": ["100_1", "100_2", "200_1", "300_1", "400_1", "400_2"],
            "dim_00": [0.0, 0.2, 4.8, 0.1, 5.2, 5.0],
        }
    )
    ae_patch = ev.aggregate_alphaearth_to_patch(ae_parcels)
    report = ev.compare_to_alphaearth(
        pheno_df=pheno_df,
        alphaearth_df=ae_patch,
        per_class_cap=10,
        min_class_samples=2,
        n_splits=2,  # only 4 tiny patches in this alignment-focused check
    )
    assert report.n_shared_parcels == 4  # patches 100,200,300,400


# ---------------------------------------------------------------------------
# 6.3 Scope guards (AC-1): reject Italian / synthetic / official.
# ---------------------------------------------------------------------------


def test_rejects_italian_checkpoint() -> None:
    """Pointing at the Italian US-034 student is a hard ValueError."""
    with pytest.raises(ValueError, match="4band-pheno"):
        ev._validate_checkpoint(Path("checkpoints/farslip/4band-pheno/best.safetensors"))


def test_rejects_official_and_synthetic_checkpoints() -> None:
    """The official published and the synthetic checkpoints are also rejected."""
    with pytest.raises(ValueError, match=r"farslip2_vit-b-16|forbidden"):
        ev._validate_checkpoint(Path("data/farslip/checkpoints/FarSLIP2_ViT-B-16.pt"))
    with pytest.raises(ValueError, match=r"farslip_pairs|forbidden"):
        ev._validate_checkpoint(Path("data/farslip_pairs/student.safetensors"))


def test_accepts_incremental_us036a_checkpoint() -> None:
    """The US-036-a incremental checkpoint passes the guard."""
    ev._validate_checkpoint(Path("checkpoints/farslip/incremental/08cls/best.safetensors"))


def test_rejects_farslip_pairs_root() -> None:
    """Pointing pastis_root at the Italian/synthetic root is a hard ValueError."""
    with pytest.raises(ValueError, match="farslip_pairs"):
        ev._validate_pastis_root(Path("data/farslip_pairs"))
    with pytest.raises(ValueError, match="farslip_pairs"):
        ev._validate_pastis_root(Path("data") / "farslip_pairs")


def test_extract_rejects_italian_before_loading_anything() -> None:
    """extract_pheno_embeddings rejects the Italian checkpoint before any I/O."""
    with pytest.raises(ValueError, match="4band-pheno"):
        ev.extract_pheno_embeddings(
            student_checkpoint=Path("checkpoints/farslip/4band-pheno/best.safetensors"),
            n_classes=4,
        )


def test_extract_missing_checkpoint_is_clear_error(tmp_path: Path) -> None:
    """A not-yet-existing checkpoint raises FileNotFoundError, not a crash (R-DEP)."""
    missing = tmp_path / "incremental" / "08cls" / "best.safetensors"
    with pytest.raises(FileNotFoundError, match="US-036-a winning checkpoint"):
        ev.extract_pheno_embeddings(student_checkpoint=missing, n_classes=8)


# ---------------------------------------------------------------------------
# extract_pheno_embeddings with mocked extractor + builder (AC-2).
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Light stand-in for PastisPairDataset (patch_id, category_id, image)."""

    def __init__(self, n: int, n_classes: int) -> None:
        import torch

        rng = np.random.default_rng(0)
        self._samples = [(f"patch{r}", int(rng.integers(0, n_classes))) for r in range(n)]
        self._images = [
            torch.from_numpy(rng.normal(size=(4, 224, 224)).astype(np.float32)) for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        _pid, cat = self._samples[idx]
        return {
            "image": self._images[idx],
            "region_id": torch.tensor(0, dtype=torch.long),
            "category_id": torch.tensor(cat, dtype=torch.long),
        }


class _FakeExtractor:
    """Light stand-in for FarSLIPExtractor returning deterministic embeddings."""

    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.device = "cpu"

    def _prep_crops(self, crops: Any) -> Any:
        return crops

    def extract_embeddings(self, crops: Any) -> Any:
        import torch

        b = crops.shape[0]
        return torch.arange(b * 512, dtype=torch.float32).reshape(b, 512)


def test_extract_pheno_embeddings_writes_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the extractor and builder mocked it writes a parcel_id/class_id/emb_* parquet."""
    pytest.importorskip("torch")
    import ml.extractors.farslip_extractor as fe_mod
    import ml.farslip.pastis_pair_dataset as ds_mod

    ckpt = tmp_path / "incremental" / "08cls" / "best.safetensors"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake-weights")

    n_classes = 4
    fake_ds = _FakeDataset(10, n_classes)

    def _fake_builder(n: int, **_k: Any) -> tuple[Any, int, int, Any]:
        import torch

        assert n == n_classes
        return fake_ds, 1, n, torch.zeros((n, 384))

    monkeypatch.setattr(ds_mod, "create_incremental_dataset", _fake_builder)
    monkeypatch.setattr(fe_mod, "FarSLIPExtractor", _FakeExtractor)

    out = tmp_path / "emb.parquet"
    result = ev.extract_pheno_embeddings(
        student_checkpoint=ckpt,
        n_classes=n_classes,
        pastis_root=Path("data/PASTIS-R"),
        eval_folds=(4, 5),
        embedding_space="proj512",
        output_path=out,
        batch_size=4,
        device="cpu",
    )
    assert result.n_patches == 10
    assert result.n_dims == 512
    assert result.eval_folds == (4, 5)
    df = pl.read_parquet(out)
    assert df.height == 10
    assert "parcel_id" in df.columns
    assert df.schema["parcel_id"] == pl.Utf8
    assert "class_id" in df.columns
    assert "emb_000" in df.columns and "emb_511" in df.columns
    # class_id is a real PASTIS class_id (1..18), not the active index.
    active = ds_mod.active_classes(n_classes)
    assert set(df["class_id"].unique().to_list()).issubset(set(active))


def test_extract_does_not_touch_italian_builder_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder is called with the REAL PASTIS root, never farslip_pairs."""
    pytest.importorskip("torch")
    import ml.extractors.farslip_extractor as fe_mod
    import ml.farslip.pastis_pair_dataset as ds_mod

    ckpt = tmp_path / "incremental" / "06cls" / "best.safetensors"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"fake")

    seen_roots: list[Path] = []

    def _fake_builder(n: int, *, root: Path, **_k: Any) -> tuple[Any, int, int, Any]:
        import torch

        seen_roots.append(root)
        return _FakeDataset(4, n), 1, n, torch.zeros((n, 384))

    monkeypatch.setattr(ds_mod, "create_incremental_dataset", _fake_builder)
    monkeypatch.setattr(fe_mod, "FarSLIPExtractor", _FakeExtractor)

    ev.extract_pheno_embeddings(
        student_checkpoint=ckpt,
        n_classes=6,
        pastis_root=Path("data/PASTIS-R"),
        embedding_space="proj512",
        output_path=tmp_path / "e.parquet",
        batch_size=2,
        device="cpu",
    )
    assert seen_roots == [Path("data/PASTIS-R")]
    assert all("farslip_pairs" not in str(r) for r in seen_roots)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
