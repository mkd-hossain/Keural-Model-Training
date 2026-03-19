# Keural Model Training

> **Full training pipeline for the Keural MoE Foundation Model**
> A 14.83B parameter Mixture-of-Experts LLM trained from scratch on Korean + English + Code.
> Built and maintained by **MKD CO., LTD.**

---

## Table of Contents

1. [What is Keural?](#what-is-keural)
2. [Model Architecture](#model-architecture)
3. [Environment Setup](#environment-setup)
4. [Full Pipeline — Step by Step](#full-pipeline)
5. [Training in Detail](#training-in-detail)
6. [Checkpoint Format](#checkpoint-format)
7. [Binary Dataset Format](#binary-dataset-format)
8. [Inference & Testing](#inference--testing)
9. [Convert to HuggingFace](#convert-to-huggingface)
10. [Serve with vLLM](#serve-with-vllm)
11. [Troubleshooting](#troubleshooting)
12. [Future Roadmap](#future-roadmap)
13. [Author & License](#author--license)

---

## What is Keural?

**Keural** is a Korean-English multilingual foundation LLM built entirely from scratch — custom tokenizer, custom architecture, custom training pipeline. It is not a fine-tuned version of any existing model.

- **Language focus:** Korean (primary) + English + Code
- **Architecture:** Mixture of Experts (MoE) Transformer
- **Scale:** 14.83B total parameters, ~3.7B active per token
- **Training data:** 43B+ tokens (Stage 1), targeting 50B+
- **Tokenizer:** Custom SentencePiece Unigram, 131,072 vocab

Comparable models: Mistral 22B, Gemma 2 27B, Qwen 2.5 14B

---

## Model Architecture

### Overview

```
Input tokens
    ↓
Token Embedding [131072 × 4096]
    ↓
24 × TransformerBlock
    ├── RMSNorm
    ├── Attention (GQA, RoPE)
    │     ├── Even layers: Sliding window (512) — local context
    │     └── Odd layers: Full causal — global context
    ├── RMSNorm
    └── MoE FFN (8 experts, top-2)
          └── SwiGLU per expert
    ↓
RMSNorm
    ↓
LM Head (tied to embedding weights)
    ↓
Logits [vocab_size = 131072]
```

### Locked Hyperparameters (DO NOT CHANGE after training starts)

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 131,072 | Must match tokenizer exactly |
| `hidden_size` | 4,096 | Model width |
| `intermediate_size` | 5,632 | Per-expert SwiGLU width |
| `num_hidden_layers` | 24 | Depth |
| `num_attention_heads` | 32 | Query heads |
| `num_key_value_heads` | 8 | GQA — 4:1 ratio |
| `head_dim` | 128 | = hidden / num_heads = 4096/32 |
| `num_local_experts` | 8 | Total experts per layer |
| `num_experts_per_tok` | 2 | Active experts per token |
| `max_position_embeddings` | 4,096 | Stage 1 context length |
| `sliding_window` | 512 | Even-layer attention window |
| `rope_theta` | 500,000.0 | LLaMA-3 style RoPE base |
| `router_aux_loss_coef` | 0.02 | MoE load balance coefficient |
| `router_z_loss_coef` | 0.001 | Router stability coefficient |

### Parameter Count Breakdown

| Component | Parameters |
|---|---|
| Token embedding | 537M |
| Attention (all layers) | 1,811M |
| MoE FFN (all layers) | 12,386M |
| Norms + output | 98M |
| **Total** | **14,832M (14.83B)** |

### Key Design Choices

**GQA (Grouped Query Attention):** 32 query heads share 8 KV heads (4:1 ratio). Reduces KV cache memory by 4× with minimal quality loss. Used in LLaMA 3, Mistral, Gemma.

**SwiGLU Expert FFN:**
```python
output = down_proj(silu(gate_proj(x)) * up_proj(x))
# gate_proj:  4096 → 5632
# up_proj:    4096 → 5632
# down_proj:  5632 → 4096
```

**Alternating attention:** Even layers use sliding window (efficient local), odd layers use full causal (global context). Halves attention compute cost.

**Weight tying:** LM head shares weights with token embedding. Reduces parameters and improves training signal.
```python
logits = F.linear(x, embed_tokens.weight)  # FSDP-safe
```

**MoE routing:**
```python
router_logits = gate(x)                          # [T, 8]
probs = softmax(router_logits)
top2_weights, top2_indices = topk(probs, k=2)   # select 2 experts
top2_weights = top2_weights / top2_weights.sum() # normalize weights
output = Σ(weight_i × expert_i(x))              # weighted sum
```

---

## Environment Setup

### Prerequisites

You need a Linux server with NVIDIA GPUs. This has been tested on:
- **OS:** Ubuntu 22.04
- **Python:** 3.12
- **CUDA:** 12.4
- **PyTorch:** 2.6.0+cu124

### Step 1 — Install Python dependencies

```bash

# Activate your virtual environment first
python -m venv venv
source venv/bin/activate  # or: conda activate your_env

# Core training dependencies
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Data and tokenizer
pip install sentencepiece datasets huggingface_hub tqdm numpy

# Optional but recommended
pip install wandb  # for experiment tracking
```

### Step 2 — Verify GPU setup

```bash
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available'); print(torch.version.cuda)"
# Expected: 2 GPUs available, 12.4
```

### Step 3 — Clone and set up directories

```bash
git clone https://github.com/mkd-hossain/Keural-Model-Training.git
cd Keural-Model-Training

# Download tokenizer
git clone https://github.com/mkd-hossain/keural-tokenizer.git

# Create required directories
mkdir -p data/raw_stage1 data/binary checkpoints/stage1 logs
```

### Verified working environment

```
Python:    3.12.x
PyTorch:   2.6.0+cu124
CUDA:      12.4
GPU:       NVIDIA H200 140GB × 2  (also works on A100 80GB × 2)
RAM:       480GB
Storage:   2TB NVMe
```

---

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Collect corpus (~50B tokens)                       │
│  python scripts/collect_stage1_50b_production_2.py          │
│  Output: data/raw_stage1/*.txt  (~205GB)                    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Build binary dataset                               │
│  python scripts/build_tokenization_binary_dataset.py        │
│  Output: data/binary/keural_*.bin  (~155GB, 102 shards)     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Train model (FSDP, multi-GPU)                      │
│  torchrun --nproc_per_node=2 scripts/train_keural_v2.py     │
│  Output: checkpoints/stage1/checkpoint_*.pt  (~83GB each)   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Test inference on checkpoint                       │
│  python scripts/infer_keural.py --checkpoint ...            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Convert to HuggingFace format                      │
│  python scripts/upload_dataset_hf.py                        │
│  → Deploy with vLLM / OpenAI-compatible API                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Full Pipeline — Step by Step

### Step 1 — Collect Training Corpus

```bash
python scripts/collect_stage1_50b_production_2.py
```

**What it collects:**

| Dataset | Language | Type | Target |
|---|---|---|---|
| FineWeb (HuggingFace) | English | Web text | ~20B tokens |
| CC100 Korean | Korean | Web text | ~3B tokens |
| Korean WebText | Korean | Web text | ~4B tokens |
| WanJuan Korean | Korean | Web text | ~5B tokens |
| Wikipedia Korean | Korean | Encyclopedia | ~1B tokens |
| ArXiv | English | Science | ~4B tokens |
| PubMed | English | Medical science | ~3B tokens |
| The Stack v1 | Code | Programming | ~8B tokens |
| PG19 Literature | English | Books | ~1B tokens |
| **Total** | | | **~50B tokens** |

**Key features:**
- ✅ Safe to interrupt and restart — shard-based resume
- ✅ Hash deduplication per shard — no duplicate documents
- ✅ Timeout protection — stuck downloads automatically skip
- ✅ Progress saved to JSON — never lose progress

**Output:** `data/raw_stage1/` — one `.txt` file per dataset source

---

### Step 2 — Build Binary Dataset

```bash
python scripts/build_tokenization_binary_dataset.py \
  --input_dir data/raw_stage1 \
  --output_dir data/binary \
  --tokenizer keural-tokenizer/tokenizer/keural_tokenizer.model \
  --seq_len 4096
```

**What it does:**
1. Reads raw `.txt` files from `data/raw_stage1/`
2. Tokenizes each document using the Keural tokenizer
3. Packs sequences into fixed 4096-token chunks with BOS/EOS
4. Writes binary shards: `keural_000.bin`, `keural_001.bin`, ...
5. Creates index (`.idx`) and metadata (`.meta`) for each shard

**Binary format spec** (critical — must match exactly):

```
File: keural_NNN.bin
─────────────────────────────────────────────
HEADER (36 bytes):
  bytes  0- 7: magic = b"KEURAL\x00\x00"  (8 bytes)
  bytes  8-11: version = 1                (uint32, little-endian)
  bytes 12-19: num_sequences              (uint64, little-endian)
  bytes 20-27: seq_length = 4096          (uint64, little-endian)
  bytes 28-35: padding = 0                (uint64, little-endian)

BODY:
  num_sequences × seq_length × 4 bytes
  Each token stored as uint32, little-endian

File: keural_NNN.idx
─────────────────────────────────────────────
  bytes 0-3:  num_sequences  (uint32)
  bytes 4-7:  seq_length     (uint32)
  For each sequence:
    8 bytes: file offset     (uint64)
    4 bytes: sequence length (uint32)

File: keural_NNN.meta  (JSON)
─────────────────────────────────────────────
{
  "num_sequences": 100000,
  "seq_length": 4096,
  "source": "keural_NNN"
}
```

> ⚠️ **CRITICAL:** The header is 36 bytes (`struct.calcsize("<8sIQQQ")`).
> Reading only 32 bytes will cause a silent crash. Always use `HEADER_SIZE = struct.calcsize("<8sIQQQ")`.

**Output stats (current):**
- 102 shards
- 10,067,450 total sequences
- 43.17B tokens
- ~155GB total

---

### Step 3 — Train the Model

#### First time (no checkpoint):

```bash
torchrun --nproc_per_node=2 scripts/train_keural_v2.py \
  --batch_size 4 \
  --grad_accum 8
```

#### Resume from checkpoint (automatic):

```bash
torchrun --nproc_per_node=2 scripts/train_keural_v2.py \
  --batch_size 4 \
  --grad_accum 8
# --resume defaults to "auto" — finds latest checkpoint automatically
```

#### Resume from specific checkpoint:

```bash
torchrun --nproc_per_node=2 scripts/train_keural_v2.py \
  --batch_size 4 \
  --grad_accum 8 \
  --resume checkpoints/stage1/checkpoint_5000.pt
```

#### Run safely in tmux (recommended for long training):

```bash
tmux new -s keural-train
torchrun --nproc_per_node=2 scripts/train_keural_v2.py --batch_size 4 --grad_accum 8
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t keural-train
```

---

## Training in Detail

### All Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 8 | Sequences per GPU per step |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--max_steps` | 100,000 | Total optimizer steps |
| `--lr` | 3e-4 | Peak learning rate |
| `--save_interval` | 500 | Save checkpoint every N steps |
| `--log_interval` | 10 | Log metrics every N steps |
| `--resume` | auto | `auto`=latest, `None`=scratch, or path |
| `--checkpoint_dir` | checkpoints/stage1 | Checkpoint save directory |
| `--data_dir` | data/binary | Binary dataset directory |
| `--data_prefix` | keural | Shard filename prefix |
| `--cpu_offload` | False | FSDP CPU offload (saves VRAM, slower) |
| `--no_grad_ckpt` | False | Disable gradient checkpointing (faster, more VRAM) |

### Effective Batch Size Formula

```
eff_batch = batch_size × grad_accum × num_gpus
          = 4 × 8 × 2 = 64 sequences

tokens_per_step = eff_batch × seq_len
               = 64 × 4096 = 262,144 tokens
```

### Learning Rate Schedule

```
Phase 1 (steps 0-2000):   Linear warmup: 0 → 3e-4
Phase 2 (steps 2000-100000): Cosine decay: 3e-4 → 3e-5
```

### What the Training Log Means

```
[Rank0] Step 1000/100000 | Loss 8.2341 | Aux 0.8123 | GradNorm 2.34 | LR 1.50e-04 | 12.0K tok/s | 22.0s/step
```

| Field | Meaning | Healthy range |
|---|---|---|
| `Loss` | Cross-entropy language model loss | Decreasing over time |
| `Aux` | MoE load balancing loss | 0.5–1.5 |
| `GradNorm` | Gradient norm (clipped at 1.0) | 0.5–5.0 |
| `LR` | Current learning rate | Follows warmup/cosine schedule |
| `tok/s` | Training throughput | Higher is better |

### Memory Usage (2× H200 140GB)

| Component | Per GPU |
|---|---|
| Model weights (bf16) | ~14.83GB |
| Optimizer states (fp32) | ~54GB |
| Gradients | ~14.83GB |
| Activations (grad ckpt) | ~10GB |
| **Total per GPU** | **~94GB** |

### Training Time Estimates

| GPUs | GPU Type | tok/s | 100K steps | Time |
|---|---|---|---|---|
| 2× | H200 140GB | ~12K | 100,000 | ~14 days |
| 4× | H200 140GB | ~25K | 100,000 | ~7 days |
| 8× | H200 140GB | ~50K | 100,000 | ~3.5 days |
| 8× | A100 80GB | ~30K | 100,000 | ~6 days |

### Expected Loss Curve

| Step | Expected Loss | Quality |
|---|---|---|
| 0 | ~11.78 | Random (log 131072) |
| 500 | ~10.0 | Word-level patterns |
| 2,000 | ~7.0 | Basic language structure |
| 10,000 | ~4.5 | Sentences forming |
| 30,000 | ~3.2 | Coherent paragraphs |
| 100,000 | ~2.5 | Stage 1 complete |

---

## Checkpoint Format

Each checkpoint file (`checkpoints/stage1/checkpoint_STEP.pt`):

```python
{
    "model":             dict,   # Model state dict — ~29.66GB in bf16
    "optimizer":         dict,   # AdamW state — ~54GB in fp32 (m + v buffers)
    "step":              int,    # Current training step
    "loss":              float,  # Loss at this step
    "epoch":             int,    # Dataset epoch number
    "tokens_processed":  int,    # Total tokens seen so far
}
```

**Checkpoint size:** ~83GB per file
**Save frequency:** Every 500 steps (configurable)
**Auto-resume:** Script automatically finds latest checkpoint on restart
**Recommendation:** Keep at least 2 most recent checkpoints (delete older ones to save quota)

### Loading a checkpoint manually

```python
import torch
ckpt = torch.load("checkpoints/stage1/checkpoint_5000.pt", map_location="cpu")
print(f"Step: {ckpt['step']}")
print(f"Loss: {ckpt['loss']:.4f}")
print(f"Tokens seen: {ckpt['tokens_processed']:,}")
```

---

## Inference & Testing

```bash
python scripts/infer_keural.py \
  --checkpoint checkpoints/stage1/checkpoint_5000.pt \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
```

Tests 4 prompts automatically:
- English: `"The future of artificial intelligence"`
- Korean: `"인공지능의 미래는"`
- Korean Science: `"한국어 자연어처리 기술의 발전은"`
- Code: `"def fibonacci(n):"`

**Note:** The inference script loads the full checkpoint including optimizer states (~83GB). This is normal — only model weights (~29.66GB) are used for inference.

---

## Convert to HuggingFace

After Stage 1 training completes, convert to HuggingFace format:

```bash
python scripts/upload_dataset_hf.py
```

This strips optimizer states and saves only model weights in HuggingFace format:
- `config.json` — model architecture
- `pytorch_model.bin` or `model.safetensors` — weights (~29.66GB)
- `tokenizer_config.json` — tokenizer config
- `tokenizer.model` — SentencePiece model

**Final model sizes after conversion:**

| Format | Size |
|---|---|
| BF16 (full precision) | ~29.66GB |
| FP16 | ~29.66GB |
| INT8 quantized | ~15GB |
| INT4 quantized (GPTQ/AWQ) | ~7-8GB |

---

## Serve with vLLM

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model ./keural-hf \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 2  # use 2 GPUs
```

**OpenAI-compatible API:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="keural",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "안녕하세요! 자기소개 해주세요."}
    ]
)
print(response.choices[0].message.content)
```

---

## Troubleshooting

### OOM (Out of Memory)

```
torch.OutOfMemoryError: CUDA out of memory
```
**Fix:** Reduce batch size. Safe values:
- H200 140GB: `--batch_size 4` ✅
- A100 80GB: `--batch_size 2` ✅
- V100 32GB: not supported (model too large)

### Training stuck / NCCL timeout

```
WorkNCCL timeout: BROADCAST operation timed out
```
**Fix:** This was caused by `sync_module_states=True` in FSDP. Already fixed in `train_keural_v2.py`. If it happens again, restart training — it will resume from the last checkpoint.

### Checkpoint save fails

```
PytorchStreamWriter failed writing file: file write failed
```
**Fix:** Disk quota full. Delete old checkpoints, keep only the 2 most recent.
```bash
rm checkpoints/stage1/checkpoint_500.pt
rm checkpoints/stage1/checkpoint_*.pt.tmp  # remove partial saves
```

### Data loading crash (silent)

```
ValueError: Bad magic in keural_000.bin
```
**Fix:** Header size mismatch. Always use `struct.calcsize("<8sIQQQ")` = 36 bytes, never hardcode 32.

### Loss not decreasing

- Check `GradNorm` — if it's always 1.0 exactly, gradient clipping may be too aggressive
- Check `Aux` loss — if it's 0, MoE routing is broken
- Check `LR` — warmup should be increasing for the first 2000 steps

---

## Future Roadmap

### Training Stages

| Stage | Context | Tokens | Status | Goal |
|---|---|---|---|---|
| Stage 1 | 4,096 | 50B | 🔄 In progress | Foundation language understanding |
| Stage 2 | 32,768 | 20B | ⏳ Planned | Long context extension (RoPE scaling) |
| Stage 3 | 131,072 | 10B | ⏳ Planned | Ultra-long context (1M target) |

### Fine-tuning Stages

| Stage | Method | Goal |
|---|---|---|
| SFT | Supervised Fine-tuning | Follow instructions |
| DPO | Direct Preference Optimization | Human preference alignment |
| RLHF | Reinforcement Learning from Human Feedback | Advanced alignment |

### GPU Scaling Plan

| Phase | GPUs | Expected throughput | Training time (50B) |
|---|---|---|---|
| Current | 2× H200 | ~12K tok/s | ~50 days |
| Near-term | 4× H200 | ~25K tok/s | ~25 days |
| Mid-term | 8× H200 | ~50K tok/s | ~12 days |
| Long-term | 16× H200 | ~100K tok/s | ~6 days |

**GPU scaling notes:**
- 4× H200: Change `--nproc_per_node=4`, increase `--batch_size 8`
- 8× H200: Add `--nnodes` if multi-node, consider tensor parallelism
- 16×+: Switch to Megatron-LM or DeepSpeed for 3D parallelism

### Model Scaling Plan

| Version | Params | Experts | Layers | Hidden |
|---|---|---|---|---|
| Keural-14B (current) | 14.83B | 8 | 24 | 4096 |
| Keural-30B (planned) | ~30B | 16 | 32 | 5120 |
| Keural-70B (planned) | ~70B | 32 | 48 | 8192 |

### Product Roadmap

- [ ] **Keural-Chat** — instruction-tuned version for conversation
- [ ] **Keural-Code** — code-specialized fine-tune
- [ ] **Keural-Ko** — Korean-focused fine-tune for Korean NLP tasks
- [ ] **Keural API** — OpenAI-compatible hosted API
- [ ] **Keural on HuggingFace** — public model release
- [ ] **Quantized versions** — INT4/INT8 for consumer hardware

---

## Bugs Fixed in v2 (vs original scripts)

| Bug | Impact | Fix Applied |
|---|---|---|
| Header reads 32 bytes (needs 36) | Silent crash on data load | `HEADER_SIZE = struct.calcsize("<8sIQQQ")` |
| Double label shift | Off-by-2 prediction, broken training | Remove model-side shift |
| No causal mask on full-attention layers | Model sees future tokens | `is_causal=True` |
| Bidirectional sliding window mask | Wrong attention pattern | Block future + too-far-past only |
| Missing SwiGLU (2-linear expert) | Wrong FFN, reduced capacity | Add `gate_proj` |
| Weight tying breaks FSDP | Training crash | Use `F.linear(x, embed.weight)` |
| `use_reentrant=True` default | FSDP + gradient checkpoint error | `use_reentrant=False` |
| `sync_module_states=True` | NCCL broadcast timeout (10 min hang) | Removed — ranks build independently |
| GradScaler with bfloat16 | Unnecessary, minor overhead | Removed |

---

## Author & License

MKD CO., LTD.
Email: hossain.najmul@mkd.kr

**Keural** is an original foundation model built from scratch. The architecture, training pipeline, tokenizer, and all code are original work.

Related repositories:
- Tokenizer: [github.com/mkd-hossain/keural-tokenizer](https://github.com/MKD-CORP/keural-tokenizer)

```bibtex
@misc{keural-2026,
  author = {Md Najmul Hossain},
  title  = {Keural: A Korean-English Multilingual MoE Foundation Model},
  year   = {2026},
  url    = {https://github.com/mkd-hossain/Keural-Model-Training}
}
```
