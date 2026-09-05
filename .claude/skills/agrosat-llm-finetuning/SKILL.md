---
name: agrosat-llm-finetuning
description: Fine-tune Gemma 4 26B-MoE and Qwen3-VL-30B-A3B with LoRA rank 16 BF16, serve LLMs with vLLM or llama.cpp. FUTURE (ADR-011, ADR-014): the Azure H100 is gone and this is out of the MICAI 2027 article scope; use only when a US budgets a GPU for it.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot LLM Fine-tuning Skill

> **FUTURE — sin hardware y fuera del alcance del articulo (ADR-011, ADR-014).** La VM Azure
> H100 del sponsor ya no existe (4-sep-2026); ninguna ventana V1-V6 volvera a abrirse. Esta
> skill conserva los hechos verificados (id real, QLoRA bloqueado por el layout MoE 3D, via
> real `target_parameters`, AgroMind eval-only) para cuando exista GPU presupuestada en una US.
> El reasoner del copiloto es **Gemini 2.5-pro**; el on-prem es Qwen3-30B-A3B servido con
> llama.cpp (el id `Qwen3.5-35B-A3B` no existe). Ver [ADR-011](../../../docs/decisions/ADR-011-gemma4-lora-future.md).

## Rules — NON-NEGOTIABLE

- LoRA rank 16 BF16 con FSDP + FlashAttention-2 + gradient checkpointing
- **MoE de Gemma 4: los expertos son tensores 3D fused (`nn.Parameter`), NO `nn.Linear`.**
  `target_modules=[gate/up/down_proj]` NO los matchea (solo engancha attention,
  ~0.91% params). La via real es **`target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`** (PEFT >= 0.17).
- **QLoRA BLOQUEADO en Gemma 4 MoE**: `bitsandbytes` no cuantiza el layout 3D fused
  de los expertos. Usar LoRA BF16 (sin cuantizar los expertos), no QLoRA.
- Validar VRAM ANTES de lanzar (Gemma 4: ~82 GB / 96; Qwen3-VL: ~92 GB / 96; Qwen3.5 serving: ~91 GB / 96)
- Checkpoint cada 30 min a Azure Blob
- **AgroMind es EVAL-ONLY (28482 QA, sin train split) -> NUNCA fine-tune sobre el (= leakage).**
  AgroMind-IT/ES (500) tambien es eval. El SFT debe ser sintetico PROPIO (trazas de tool calls).
- vLLM continuous batching para serving (OpenAI-compatible)
- vLLM args: `--max-model-len 65536 --gpu-memory-utilization 0.92 --enable-prefix-caching`

## Modelos (IDs verificados HF 15-jun-2026)

| Modelo | HF ID | Licencia | Uso |
|--------|-------|----------|-----|
| Gemma 4 26B-A4B-MoE | `google/gemma-4-26B-A4B-it` | Apache 2.0 | Perceiver fine-tuned (FUTURE, ADR-011) — el id `gemma-4-26b-it` NO existe |
| Qwen3-VL 30B-A3B | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Apache 2.0 | VLM comparativo |
| Qwen MoE-A3B Int4 (on-prem) | `Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` | Apache 2.0 | Orquestador on-prem vLLM (el id `Qwen3.5-35B-A3B` no existe; sustitucion en US-048) |

## Training Script Gemma 4

```python
# ml/train/train_gemma4_lora.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_ID = "google/gemma-4-26B-A4B-it"  # NOT "gemma-4-26b-it" (no existe)

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum"],
        mixed_precision="bf16",
    )

    # NO QLoRA: bitsandbytes cannot quantize Gemma 4's fused 3D MoE expert
    # tensors. Load in BF16 (LoRA, not QLoRA).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    num_experts = getattr(model.config, "num_local_experts", None) or 128
    effective_r = max(1, 16 // num_experts)
    lora = LoraConfig(
        r=16, lora_alpha=32,
        # Attention via nn.Linear modules ...
        target_modules=["q_proj", "v_proj"],
        # ... and the MoE experts via their fused 3D nn.Parameter tensors
        # (target_modules would NOT match these — they have no forward()).
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        rank_pattern={"experts.gate_up_proj": effective_r,
                      "experts.down_proj": effective_r},
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # SFT on SYNTHETIC tool-call traces, never AgroMind (eval-only = leakage).
    train_loader = build_synthetic_sft_loader(cfg["batch"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out.loss)
                optim.step(); optim.zero_grad()
            if step % 500 == 0 and accelerator.is_main_process:
                accelerator.save_state(f"azure-blob://checkpoints/gemma4-ep{epoch}-s{step}")
                mlflow.log_metric("train_loss", out.loss.item(), step=step)
```

## Config H100

```yaml
# configs/gemma4_h100.yaml
model_id: google/gemma-4-26B-A4B-it   # NOT gemma-4-26b-it (no existe)
batch: 2
grad_accum: 8     # effective 16
lr: 1.0e-4
epochs: 3
max_seq_len: 32768
lora_rank: 16
lora_alpha: 32
checkpoint_every_steps: 500
checkpoint_path: az://agrosat-checkpoints/gemma4/
flash_attention: 2
gradient_checkpointing: true
fsdp:
  sharding_strategy: FULL_SHARD
  cpu_offload: false
```

## Config L4 Fallback (Gemma 4 E4B)

```yaml
# configs/gemma4_l4_fallback.yaml
model_id: google/gemma-4-e4b-it
batch: 1
grad_accum: 16
quantization: 4bit_nf4
lora_rank: 16
epochs: 3
```

## vLLM Serving (Qwen on-prem, US-048)

El id `Qwen/Qwen3.5-35B-A3B` NO existe en HF. El real mas cercano de la familia
MoE-A3B Int4 es `Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` (single-GPU GPTQ-Int4,
no BF16 ~70GB). El script real es [`scripts/serve_qwen35.sh`](../../../scripts/serve_qwen35.sh)
(con pre-flight de GPU + health wait); ver [`docs/serving/qwen35.md`](../../../docs/serving/qwen35.md).

```bash
# scripts/serve_qwen35.sh (resumen)
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4 \
  --served-model-name qwen35 \
  --quantization gptq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --port 8002
```

> Bloqueo de ejecucion (US-048): la VM H100 es Windows-guest sin virtualizacion
> anidada -> ni WSL2 ni Docker -> vLLM no corre hasta que el sponsor habilite
> `ExposeVirtualizationExtensions`. El codigo esta listo para cualquier Linux+GPU.

## Validación VRAM

```python
def validate_vram_budget(model_id: str, batch: int, ctx_len: int, h100_total_gb: float = 96):
    """Aborta si presupuesto estimado >94 GB."""
    weights = {"gemma-4-26b": 52, "qwen3-vl-30b": 60, "qwen3.5-35b": 70}[model_id]
    kv_cache_per_ctx = {"gemma-4-26b": 0.00025, "qwen3.5-35b": 0.0002}
    kv = batch * ctx_len * kv_cache_per_ctx[model_id]
    activations = 15  # con grad ckpt
    overhead = 8
    total = weights + 1.5 + kv + activations + overhead
    assert total < 94, f"VRAM exceeded: {total:.1f} GB > 94 GB safe limit"
    return total
```

## QA Checklist Fine-tuning

- [ ] VRAM validada antes de launch
- [ ] LoRA rank 16, target modules correctos
- [ ] FlashAttn-2 + grad ckpt + FSDP
- [ ] Checkpoints cada 30 min a Azure Blob
- [ ] MLflow run con todos los hiperparámetros
- [ ] Auto-shutdown VM H100
- [ ] Eval post-train con AgroMind + AgroMind-IT/ES
- [ ] Atribución Apache 2.0 (Gemma 4, Qwen) en DATA_LICENSE.md
