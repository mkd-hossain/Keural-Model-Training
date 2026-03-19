# Keural Model Training

Full training pipeline for the **Keural MoE Foundation Model** — a 14.83B parameter Mixture-of-Experts LLM trained from scratch on Korean + English + Code.

---

## Model Specs

| Property | Value |
|---|---|
| Architecture | Mixture of Experts (MoE) Transformer |
| Parameters | 14.83B total / ~3.7B active per token |
| Layers | 24 |
| Hidden size | 4096 |
| Attention heads | 32 (GQA: 8 KV heads) |
| Experts | 8 total, top-2 per token |
| Expert FFN | SwiGLU (gate × silu(up) → down) |
| Context length | 4096 tokens (Stage 1) |
| Sliding window | 512 (every other layer) |
| Vocab size | 131,072 |
| Tokenizer | SentencePiece Unigram |
| RoPE theta | 500,000 |
| Training dtype | bfloat16 |
| Parallelism | FSDP (Fully Sharded Data Parallel) |

---

## Hardware Requirements

| Component | Minimum | Used |
|---|---|---|
| GPU | 2× A100 80GB | 2× H200 140GB |
| GPU Memory | 160GB total | 280GB total |
| RAM | 256GB | 480GB |
| Storage | 1TB | 2TB |
| CPU | 16 cores | 32 cores |

---

## Pipeline Overview

```
Step 1: Collect corpus (50B tokens)
    ↓
Step 2: Build binary dataset
    ↓
Step 3: Train model (FSDP, 2× GPU)
    ↓
Step 4: Test inference
    ↓
Step 5: Convert to HuggingFace format
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/collect_stage1_50b_production_2.py` | Collect 50B token training corpus |
| `scripts/train_keural_v2.py` | Main training script (FSDP) |
| `scripts/infer_keural.py` | Test inference on checkpoints |
| `scripts/upload_dataset_hf.py` | Upload dataset to HuggingFace Hub |

---

## Step 1 — Collect Training Data

```bash
python scripts/collect_stage1_50b_production_2.py
```

**Target corpus: ~50B tokens**

| Dataset | Language | Tokens (approx) |
|---|---|---|
| FineWeb | English | ~20B |
| CC100 Korean | Korean | ~3B |
| Korean WebText | Korean | ~4B |
| WanJuan Korean | Korean | ~5B |
| Wikipedia KO | Korean | ~1B |
| ArXiv | English (Science) | ~4B |
| PubMed | English (Science) | ~3B |
| The Stack v1 | Code | ~8B |
| PG19 Literature | English | ~1B |

**Features:**
- Shard-based resume — safe to interrupt and restart
- Hash deduplication per shard
- Timeout protection per document
- Progress tracking via JSON

**Output:** `data/raw_stage1/*.txt`

---

## Step 2 — Build Binary Dataset

```bash
python scripts/build_tokenization_binary_dataset.py \
  --input_dir data/raw_stage1 \
  --output_dir data/binary \
  --tokenizer keural-tokenizer/keural_tokenizer.model \
  --seq_len 4096
```

**Output format:** `data/binary/keural_NNN.bin` + `.idx` + `.meta`
**Header format:** `<8s I Q Q Q>` = 36 bytes (KEURAL magic + version + num_seqs + seq_len + pad)
Each shard: ~100,000 sequences × 4096 tokens = ~1.6GB

---

## Step 3 — Train

```bash
torchrun --nproc_per_node=2 scripts/train_keural_v2.py \
  --batch_size 4 \
  --grad_accum 8
```

**Auto-resumes from latest checkpoint automatically.**

### Key training args

| Arg | Default | Description |
|---|---|---|
| `--batch_size` | 8 | Per-GPU batch size |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--max_steps` | 100000 | Total training steps |
| `--lr` | 3e-4 | Peak learning rate |
| `--save_interval` | 500 | Save checkpoint every N steps |
| `--resume` | auto | Auto-load latest checkpoint |
| `--checkpoint_dir` | checkpoints/stage1 | Where to save checkpoints |

### Effective batch size

```
batch_size × grad_accum × num_gpus = effective_batch
4 × 8 × 2 = 64 sequences per optimizer step
64 × 4096 = 262,144 tokens per step
```

### Training time estimate (2× H200)

| Throughput | Steps | Time |
|---|---|---|
| ~12K tok/s | 100,000 | ~14 days |
| ~25K tok/s | 100,000 | ~7 days |

---

## Architecture Details

### MoE (Mixture of Experts)

Each transformer block contains:
- **Attention**: Multi-head with GQA (32 heads, 8 KV heads, head_dim=128)
- **MoE FFN**: 8 experts, top-2 routing per token

Only 2 out of 8 experts activate per token → efficient compute with large capacity.

### SwiGLU Expert FFN

```python
def forward(self, x):
    return down_proj(silu(gate_proj(x)) * up_proj(x))
```

### Attention Pattern

- **Even layers** (0,2,4,...): Sliding window attention (window=512) — local context
- **Odd layers** (1,3,5,...): Full causal attention — global context

### Weight Tying

```python
# Embedding and LM head share weights (FSDP-safe)
logits = F.linear(x, self.embed_tokens.weight)
```

### Auxiliary Loss (MoE Load Balancing)

```python
aux_loss = 0.02 × (num_experts × Σ(prob_i × freq_i))  # load balance
z_loss   = 0.001 × mean(logsumexp(router_logits)²)     # router stability
```

---

## Step 4 — Test Inference

```bash
python scripts/infer_keural.py \
  --checkpoint checkpoints/stage1/checkpoint_5000.pt \
  --max_new_tokens 200
```

Tests 4 prompts automatically:
- English: `"The future of artificial intelligence"`
- Korean: `"인공지능의 미래는"`
- Korean Science: `"한국어 자연어처리 기술의 발전은"`
- Code: `"def fibonacci(n):"`

**Expected quality by step:**

| Step | Expected output |
|---|---|
| 500 | Random words, no structure |
| 5,000 | Basic sentence patterns |
| 20,000 | Coherent sentences |
| 50,000 | Good quality text |
| 100,000 | Stage 1 complete |

---

## Step 5 — Convert to HuggingFace Format

After training completes, convert for vLLM / OpenAI-compatible API serving:

```bash
python scripts/upload_dataset_hf.py
```

Then serve with vLLM:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model ./keural-hf \
  --tokenizer ./keural-tokenizer \
  --dtype bfloat16
```

OpenAI-compatible API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="keural",
    messages=[{"role": "user", "content": "안녕하세요!"}]
)
```

---

## Checkpoint Format

Each checkpoint (`checkpoint_STEP.pt`) contains:

```python
{
    "model":     model_state_dict,      # ~29.66GB bf16 weights
    "optimizer": optimizer_state_dict,  # ~54GB fp32 Adam moments
    "step":      int,
    "loss":      float,
    "epoch":     int,
    "tokens_processed": int,
}
```

Total checkpoint size: ~83GB per save.

---

## Bugs Fixed in v2 (vs original)

| Bug | Impact | Fix |
|---|---|---|
| Header reads 32 bytes (should be 36) | Silent crash on data load | `struct.calcsize("<8sIQQQ")` = 36 |
| Double label shift | Off-by-2 prediction error | Remove model-side shift |
| No causal mask on full-attention layers | Model sees future tokens | `is_causal=True` |
| Bidirectional sliding window | Wrong attention pattern | Block future + too-far-past |
| Missing SwiGLU (2-linear expert) | Wrong FFN architecture | Add gate_proj |
| Weight tying breaks FSDP | Training crash | Use `F.linear` |
| `use_reentrant` missing | FSDP + grad checkpoint error | `use_reentrant=False` |

---

## Author
**MKD CO, LTD.
**Md Najmul Hossain**
Keural Foundation Model — trained from scratch, 2026

- Tokenizer: [github.com/mkd-hossain/keural-tokenizer](https://github.com/mkd-hossain/keural-tokenizer)