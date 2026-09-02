"""Tests for :mod:`ml.transfer.finetune_italia` (US-079 dense transfer).

The fine-tune builds a dense model with an Italian-sized head whose CONSERVED
rows are warm-started from a PASTIS checkpoint. The most important behaviour is
the per-architecture WARM-START of the DENSE head:

- TSViT-pheno: the per-class "head" is ``temporal_cls_tokens (1, K, dim)``; the
  conserved class tokens must equal the PASTIS tokens of the matching class.
- U-TAE: the head is ``out_conv.2 (K, 32, 1, 1)``; the conserved rows must equal
  the PASTIS ``out_conv.2`` rows (U-TAE id namespace = semantic id + 1).

We build the REAL torch models (cheap to instantiate on CPU) and load a SYNTHETIC
PASTIS checkpoint with the right head keys, then assert the conserved head rows
match and the new rows stay at init. No forward pass, no GPU, no real checkpoint.

``load_italia_patches`` / ``_equispaced_indices`` are tested against a toy on-disk
PASTIS-layout dataset, and ``run_italia_finetune`` end-to-end is exercised with a
TINY fake model (the real TSViT forward is too heavy for a unit test) so the
split / checkpoint / softmax-dump plumbing is verified without training a net.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
from ml.transfer import finetune_italia
from ml.transfer.finetune_italia import (
    DenseFineTuneConfig,
    _equispaced_indices,
    build_italia_finetune_model,
    load_italia_patches,
)
from ml.transfer.italia_label_space import ItaliaLabelSpace, build_italia_label_space

_CONSERVED = ["common_soft_wheat", "maize_corn_popcorn", "barley"]
_NEW = ["olive", "tree_wood_forest"]


# --------------------------------------------------------------------------- #
# Toy US-078 dataset + label space.
# --------------------------------------------------------------------------- #
def _write_class_table(root: Path) -> None:
    names = _CONSERVED + _NEW
    pl.DataFrame(
        {
            "class_id": list(range(1, len(names) + 1)),
            "hcat4_name": names,
            "n_parcels": [500] * len(names),
        }
    ).write_parquet(root / "class_table.parquet")


def _toy_label_space(root: Path) -> ItaliaLabelSpace:
    _write_class_table(root)
    return build_italia_label_space(italia_root=root)


def _write_toy_patches(root: Path, *, folds: dict[int, int]) -> None:
    """Write toy PASTIS-layout patches (small T, small dense masks) + metadata."""
    s2_dir = root / "DATA_S2"
    ann_dir = root / "ANNOTATIONS"
    s2_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for pid in folds:
        t = 6
        s2 = (rng.uniform(0, 1, (t, 10, 128, 128)) * 10000).astype(np.int16)
        np.save(s2_dir / f"S2_{pid}.npy", s2)
        mask = np.zeros((128, 128), dtype=np.int32)
        mask[:64] = 1  # class 1
        mask[64:] = 2  # class 2
        np.save(ann_dir / f"TARGET_{pid}.npy", mask)
        np.save(ann_dir / f"dates_{pid}.npy", np.linspace(60, 300, t).astype(np.int32))
    pl.DataFrame(
        {"patch_id": list(folds), "fold_espacial": [folds[p] for p in folds]}
    ).write_parquet(root / "metadata.parquet")


# --------------------------------------------------------------------------- #
# Synthetic PASTIS checkpoints with the right head keys.
# --------------------------------------------------------------------------- #
def _write_tsvit_pastis_ckpt(path: Path, *, dim: int = 128, seed: int = 5) -> torch.Tensor:
    """Write a tiny TSViT PASTIS checkpoint; return its (18, dim) cls-token bank."""
    rng = np.random.default_rng(seed)
    tokens = torch.from_numpy(rng.normal(size=(1, 18, dim)).astype(np.float32))
    state = {"temporal_cls_tokens": tokens}
    torch.save({"model_state": state}, path)
    return tokens[0]  # (18, dim)


def _write_utae_pastis_ckpt(path: Path, *, seed: int = 6) -> torch.Tensor:
    """Write a tiny U-TAE PASTIS checkpoint; return its (20, 32) out_conv.2 rows."""
    rng = np.random.default_rng(seed)
    w = torch.from_numpy(rng.normal(size=(20, 32, 1, 1)).astype(np.float32))
    b = torch.from_numpy(rng.normal(size=(20,)).astype(np.float32))
    state = {"out_conv.2.weight": w, "out_conv.2.bias": b}
    torch.save({"model_state_dict": state}, path)
    return w.reshape(20, 32)


# --------------------------------------------------------------------------- #
# _equispaced_indices.
# --------------------------------------------------------------------------- #
def test_equispaced_indices_subsamples_and_keeps_all_when_short() -> None:
    # Asking for more than available keeps every real frame in order and PADS
    # by repeating the last one, so the length is always exactly ``n_select``
    # (fixed temporal window, US-082; see the function docstring).
    short = _equispaced_indices(3, 10)
    assert len(short) == 10
    np.testing.assert_array_equal(short[:3], np.arange(3))
    assert set(short[3:].tolist()) == {2}
    # Subsampling spans the full range, ascending and unique.
    idx = _equispaced_indices(20, 5)
    assert len(idx) == 5
    assert idx[0] == 0 and idx[-1] == 19
    assert list(idx) == sorted(set(idx.tolist()))  # ascending + unique


# --------------------------------------------------------------------------- #
# load_italia_patches.
# --------------------------------------------------------------------------- #
def test_load_italia_patches_shapes_and_fold_filter(tmp_path: Path) -> None:
    """Patches load with subsampled T, int64 masks, and the fold filter applies."""
    _write_toy_patches(tmp_path, folds={0: 0, 1: 1, 2: 1})
    patches = load_italia_patches(italia_root=tmp_path, n_timesteps=4)
    assert len(patches) == 3
    assert patches.images[0].shape == (4, 10, 128, 128)
    assert patches.masks[0].shape == (128, 128)
    assert patches.masks[0].dtype == np.int64
    assert patches.doys[0].shape == (4,)
    # Reflectance scaled back from DN (/10000) -> in [0, ~1].
    assert patches.images[0].max() <= 1.5
    # Fold filter keeps only the wanted fold.
    only_f1 = load_italia_patches(italia_root=tmp_path, n_timesteps=4, folds=(1,))
    assert sorted(only_f1.patch_ids) == [1, 2]
    assert set(only_f1.folds) == {1}


def test_load_italia_patches_present_classes(tmp_path: Path) -> None:
    """present_classes excludes background and lists the crop ids per patch."""
    _write_toy_patches(tmp_path, folds={0: 0})
    patches = load_italia_patches(italia_root=tmp_path, n_timesteps=3)
    assert patches.present_classes() == [{1, 2}]


def test_load_italia_patches_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="homologue dataset incomplete"):
        load_italia_patches(italia_root=tmp_path / "absent")


# --------------------------------------------------------------------------- #
# WARM-START of the dense head (TSViT cls-tokens + U-TAE out_conv.2).
# --------------------------------------------------------------------------- #
def test_build_finetune_tsvit_warmstarts_cls_tokens(tmp_path: Path) -> None:
    """TSViT conserved class tokens equal the matching PASTIS tokens; new stay init."""
    root = tmp_path / "italia"
    root.mkdir()
    space = _toy_label_space(root)
    ckpt = tmp_path / "tsvit_pastis.pt"
    pastis_tokens = _write_tsvit_pastis_ckpt(ckpt, dim=128)  # (18, 128)

    model = build_italia_finetune_model(
        space,
        model_kind="tsvit-pheno",
        pastis_checkpoint=ckpt,
        n_timesteps=10,
        device="cpu",
    )
    own = model.state_dict()
    cls = own["temporal_cls_tokens"].detach().cpu().numpy()[0]  # (K, 128)

    name_to_sem = {name: cid for cid, name in SEMANTIC18_CLASS_NAMES.items()}
    pastis_np = pastis_tokens.detach().cpu().numpy()
    for leaf in space.conserved:
        row = space.index[leaf]
        sem_id = name_to_sem[space.leaf_to_pastis[leaf]]  # TSViT namespace = semantic id
        np.testing.assert_allclose(cls[row], pastis_np[sem_id], rtol=1e-5, atol=1e-6)
    # A new class token is NOT any PASTIS token (genuine random init).
    new_row = space.index["olive"]
    assert not any(np.allclose(cls[new_row], pastis_np[i]) for i in range(18))


def test_build_finetune_utae_warmstarts_out_conv(tmp_path: Path) -> None:
    """U-TAE conserved out_conv.2 rows equal PASTIS rows at semantic_id+1."""
    root = tmp_path / "italia"
    root.mkdir()
    space = _toy_label_space(root)
    ckpt = tmp_path / "utae_pastis.pt"
    pastis_rows = _write_utae_pastis_ckpt(ckpt)  # (20, 32)

    model = build_italia_finetune_model(
        space,
        model_kind="utae",
        pastis_checkpoint=ckpt,
        device="cpu",
    )
    own = model.state_dict()
    head = own["out_conv.2.weight"].detach().cpu().numpy()  # (K, 32, 1, 1)
    head2 = head.reshape(head.shape[0], -1)  # (K, 32)

    name_to_sem = {name: cid for cid, name in SEMANTIC18_CLASS_NAMES.items()}
    for leaf in space.conserved:
        row = space.index[leaf]
        native_id = name_to_sem[space.leaf_to_pastis[leaf]] + 1  # U-TAE namespace
        np.testing.assert_allclose(
            head2[row], pastis_rows[native_id].detach().cpu().numpy(), rtol=1e-5, atol=1e-6
        )
    # Background row (id 0) is not warmed -> not equal to PASTIS bg row by copy.
    new_row = space.index["tree_wood_forest"]
    assert not np.allclose(head2[new_row], pastis_rows.detach().cpu().numpy()[1])


def test_build_finetune_unsupported_kind_raises(tmp_path: Path) -> None:
    root = tmp_path / "italia"
    root.mkdir()
    space = _toy_label_space(root)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state": {}}, ckpt)
    with pytest.raises(ValueError, match="unsupported model_kind"):
        build_italia_finetune_model(space, model_kind="nope", pastis_checkpoint=ckpt, device="cpu")


def test_build_finetune_missing_checkpoint_raises(tmp_path: Path) -> None:
    root = tmp_path / "italia"
    root.mkdir()
    space = _toy_label_space(root)
    with pytest.raises(FileNotFoundError, match="PASTIS checkpoint"):
        build_italia_finetune_model(
            space, model_kind="utae", pastis_checkpoint=tmp_path / "absent.pt", device="cpu"
        )


# --------------------------------------------------------------------------- #
# run_italia_finetune plumbing with a tiny FAKE model (no heavy forward).
# --------------------------------------------------------------------------- #
class _TinyDenseModel(torch.nn.Module):
    """A trivial dense model: 1x1 conv on the time-mean of the stack.

    Stands in for TSViT/U-TAE so ``run_italia_finetune`` exercises its split /
    checkpoint / softmax-dump plumbing on CPU without a real transformer forward.
    Has the warmed head names so ``_is_head_param`` partitions params sensibly.
    """

    #: Must match the ``semantic_dim`` of the phenology branch (384): the training
    #: loop asks TSViT-like models for ``return_visual_proj=True`` and aligns the
    #: per-pixel projection with the projected class prototypes.
    SEMANTIC_DIM = 384

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.temporal_cls_tokens = torch.nn.Parameter(torch.zeros(1, num_classes, 4))
        self.to_seg = torch.nn.Conv2d(10, num_classes, 1)
        self.to_proj = torch.nn.Conv2d(10, self.SEMANTIC_DIM, 1)

    def forward(
        self, x: torch.Tensor, *, return_visual_proj: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # (B, T, 10, H, W)
        mean = x.mean(dim=1)  # (B, 10, H, W)
        logits = self.to_seg(mean)  # (B, K, H, W)
        if return_visual_proj:
            return logits, self.to_proj(mean)  # (B, semantic_dim, H, W)
        return logits


def test_run_italia_finetune_end_to_end_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fine-tune wires split -> train -> per-epoch ckpt -> softmax dump on CPU."""
    root = tmp_path / "italia"
    root.mkdir()
    _toy_label_space(root)
    # Two folds so the spatial split has a non-empty train and test side.
    _write_toy_patches(root, folds={0: 0, 1: 0, 2: 3, 3: 0})

    space_classes = len(_CONSERVED) + len(_NEW) + 1

    def _fake_build(label_space, **_kw):  # type: ignore[no-untyped-def]
        return _TinyDenseModel(label_space.num_classes)

    monkeypatch.setattr(finetune_italia, "build_italia_finetune_model", _fake_build)

    config = DenseFineTuneConfig(
        model_kind="tsvit-pheno",
        n_timesteps=3,
        head_warmup_epochs=1,
        finetune_epochs=2,
        batch_size=2,
    )
    ckpt_root = tmp_path / "checkpoints"
    summary = finetune_italia.run_italia_finetune(
        config,
        italia_root=root,
        pastis_checkpoint=tmp_path / "ignored.pt",
        test_fold=3,
        ckpt_root=ckpt_root,
        run_name="smoke",
        device="cpu",
    )

    assert summary["test_fold"] == 3
    assert summary["n_test"] == 1  # only patch 2 is in fold 3
    assert summary["n_train"] == 3
    assert summary["num_classes"] == space_classes
    assert summary["test_patch_ids"] == [2]
    # Per-epoch checkpoints + best/last written under the relative ckpt root.
    run_dir = ckpt_root / "tsvit-pheno-italia" / "smoke"
    assert (run_dir / "epoch_00.pt").is_file()
    assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
    assert len(summary["epoch_ckpts"]) == 2
    # The softmax dump is a real .npz with one post-softmax map per test patch.
    npz_path = Path(summary["softmax_path"])
    assert npz_path.is_file()
    with np.load(npz_path) as data:
        assert set(data.files) == {"2"}
        probs = data["2"]
        assert probs.shape == (space_classes, 128, 128)
        np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-4)


def test_run_italia_finetune_empty_split_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test_fold absent from the data yields an empty side -> ValueError."""
    root = tmp_path / "italia"
    root.mkdir()
    _toy_label_space(root)
    _write_toy_patches(root, folds={0: 0, 1: 0})  # all fold 0
    monkeypatch.setattr(
        finetune_italia,
        "build_italia_finetune_model",
        lambda label_space, **_kw: _TinyDenseModel(label_space.num_classes),
    )
    with pytest.raises(ValueError, match="empty side"):
        finetune_italia.run_italia_finetune(
            DenseFineTuneConfig(n_timesteps=3, head_warmup_epochs=0, finetune_epochs=1),
            italia_root=root,
            pastis_checkpoint=tmp_path / "ignored.pt",
            test_fold=99,
            ckpt_root=tmp_path / "ckpt",
            device="cpu",
        )
