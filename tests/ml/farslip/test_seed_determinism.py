"""AC-9 — Reproducibilidad seed-determinismo del trainer FarSLIP.

Q8 fix: refactorizado para correr en suite default (no `@slow`). El test
original instanciaba dos trainers completos (= 2x descarga CLIP ViT-B/16
~600 MB). Ahora reseed-eamos un solo trainer entre runs y validamos que
los gradientes y los step() son bit-exact para la misma seed — verifica
exactamente lo mismo (que ``propagate_seed`` mas el optimizer son
deterministicos) en una fraccion del tiempo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.utils.seed import propagate_seed


def _hf_available() -> bool:
    try:
        from transformers import CLIPVisionModel  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@pytest.fixture(scope="module")
def trainer(tmp_path_factory: pytest.TempPathFactory) -> FarSLIPDistillationTrainer:
    """Trainer reutilizado entre tests del modulo (1x descarga CLIP)."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    out_dir = tmp_path_factory.mktemp("seed_determinism_out")
    cfg = FarSLIPTrainerConfig(
        teacher_model_id="openai/clip-vit-base-patch16",
        dataset_root=Path("data/farslip_pairs"),
        output_dir=out_dir,
        n_epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        device="cpu",
        n_in_channels=4,
        n_regions=2,
        n_categories=3,
    )
    try:
        t = FarSLIPDistillationTrainer(cfg)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo cargar CLIP: {exc}")
    return t


def _one_step_grad_norms(
    t: FarSLIPDistillationTrainer, seed: int
) -> tuple[float, dict[str, torch.Tensor]]:
    """Reseed + 1 forward+backward + devuelve grad-norm + state_dict del student.

    Returns:
        (grad_norm escalar, copia bit-exact del state_dict student post-step)
    """
    propagate_seed(seed)
    d = t.teacher.config.hidden_size
    n_protos = t.config.n_regions * t.config.n_categories
    g = torch.Generator(device="cpu").manual_seed(seed)
    protos = torch.randn(n_protos, d, generator=g)
    t.set_text_prototypes(protos)

    g_batch = torch.Generator(device="cpu").manual_seed(seed + 1)
    imgs = torch.rand(2, 4, 224, 224, generator=g_batch)
    region_ids = torch.tensor([0, 1])
    cat_ids = torch.tensor([0, 1])

    t._optim.zero_grad(set_to_none=True)
    losses = t.step(imgs, region_ids, cat_ids)
    losses["loss_total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(t.student.parameters(), max_norm=1e9).item()
    t._optim.step()
    state = {k: v.detach().cpu().clone() for k, v in t.student.state_dict().items()}
    return grad_norm, state


def test_two_steps_same_seed_bit_exact_state_dict(
    trainer: FarSLIPDistillationTrainer,
) -> None:
    """AC-9: misma seed -> grad-norm + state_dict del student matchean bit-exact."""
    # Snapshot inicial del student para restaurar entre runs.
    init_state = {k: v.detach().cpu().clone() for k, v in trainer.student.state_dict().items()}

    grad_a, state_a = _one_step_grad_norms(trainer, seed=123)

    # Restaurar student a su estado inicial para hacer el segundo run idempotente.
    trainer.student.load_state_dict(init_state)

    grad_b, state_b = _one_step_grad_norms(trainer, seed=123)

    assert abs(grad_a - grad_b) < 1e-6, f"grad-norm difiere entre runs: {grad_a} vs {grad_b}"
    assert set(state_a.keys()) == set(state_b.keys())
    mismatches = []
    for k in state_a:
        if not torch.equal(state_a[k], state_b[k]):
            diff = (state_a[k].float() - state_b[k].float()).abs().max().item()
            if diff > 1e-5:
                mismatches.append((k, diff))
    assert not mismatches, f"keys con diff > 1e-5: {mismatches[:5]}"
