"""Mechanism tests for the US-119 member-sanity producer."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts import run_us119_sanidad_miembros as sanity


def _state(value: float = 1.0) -> dict[str, torch.Tensor]:
    """Return a minimal deterministic state dict."""
    return {
        "layer.weight": torch.tensor([[value, 2.0]]),
        "layer.bias": torch.tensor([3.0]),
    }


def test_weight_check_compares_wrapped_metric_with_raw_harness(tmp_path: Path) -> None:
    """Equal tensors in different checkpoint conventions are executable evidence."""
    metric = tmp_path / "metric.pt"
    harness = tmp_path / "harness.pt"
    torch.save({"best": {"f1_macro": 0.5}, "model_state_dict": _state()}, metric)
    torch.save(_state(), harness)

    evidence = sanity._compare_weight_sources(metric, harness)

    assert evidence["pesos_identicos"] is True
    assert evidence["tensores_checkpoint_metrica"] == 2
    assert evidence["tensores_checkpoint_arnes"] == 2
    assert evidence["tensores_identicos"] == 2
    assert evidence["sha256_checkpoint_metrica"] != evidence["sha256_checkpoint_arnes"]


def test_weight_check_rejects_one_changed_tensor(tmp_path: Path) -> None:
    """The test value distinguishes equality from same-shaped checkpoints."""
    metric = tmp_path / "metric.pt"
    harness = tmp_path / "harness.pt"
    torch.save({"model_state": _state()}, metric)
    torch.save(_state(9.0), harness)

    evidence = sanity._compare_weight_sources(metric, harness)

    assert evidence["pesos_identicos"] is False
    assert evidence["tensores_identicos"] == 1


def test_weight_check_supports_hugging_face_safetensors_directory(tmp_path: Path) -> None:
    """SegFormer's metric file is compared with the HF directory the harness loads."""
    metric = tmp_path / "best_model.pt"
    harness = tmp_path / "hf_model"
    harness.mkdir()
    torch.save({"model_state_dict": _state()}, metric)
    save_file(_state(), harness / "model.safetensors")

    evidence = sanity._compare_weight_sources(metric, harness)

    assert evidence["pesos_identicos"] is True
    assert evidence["tensores_identicos"] == 2


def test_fold_search_reaches_nested_mappings_and_lists() -> None:
    """Moving fold metadata beside the old top-level path cannot bypass the check."""
    checkpoint = {
        "config": {"training": {"folds": [1, 2, 3]}},
        "runs": [{"held_out_fold": 5}],
        "model_state": _state(),
    }

    assert sanity._fold_fields(checkpoint) == {
        "config.training.folds": [1, 2, 3],
        "runs[0].held_out_fold": 5,
    }
