#!/usr/bin/env python3
"""
Keural MoE - Training Script v2
=================================
Fixes from v1:
  - Binary header reads 36 bytes (was 32, caused silent crash)
  - No double label shift (dataset shifts, model does not)
  - Proper causal mask on ALL attention layers
  - Causal + sliding window mask (not bidirectional)
  - SwiGLU FFN per expert (gate x up -> down)
  - use_reentrant=False for gradient checkpointing
  - Weight tying via F.linear (FSDP-safe, no shared tensor)
  - Removed GradScaler (not needed for bfloat16)
  - Per-layer FSDP wrapping for better memory efficiency

Model: ~14.8B params | MoE-8 experts, top-2 | 24 layers | hidden=4096
Hardware target: 2xH200 (282GB total VRAM) with FSDP
Data: data/binary/keural_*.bin (102 shards, 43B tokens)

Launch:
  torchrun --nproc_per_node=2 scripts/train_keural_v2.py
  torchrun --nproc_per_node=2 scripts/train_keural_v2.py --resume checkpoints/stage1/checkpoint_1000.pt
"""

import os
import sys
import json
import math
import time
import mmap
import glob
import struct
import logging
import argparse
import functools
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, List, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 131072          # Keural tokenizer vocab
    hidden_size: int = 4096
    intermediate_size: int = 5632     # SwiGLU per expert (gate+up+down)
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 8      # GQA: 4:1 ratio
    head_dim: int = 128               # 4096 / 32 = 128

    # MoE
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    router_aux_loss_coef: float = 0.02
    router_z_loss_coef: float = 0.001

    # Context
    max_position_embeddings: int = 4096
    sliding_window: int = 512         # Every other layer uses sliding window

    # RoPE
    rope_theta: float = 500000.0      # Llama-3 style base


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/binary"
    data_prefix: str = "keural"

    # Training
    batch_size: int = 8               # Per GPU
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # System
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    fsdp_cpu_offload: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints/stage1"
    save_interval: int = 500
    log_interval: int = 10
    resume_from: Optional[str] = None

    # Distributed (auto-set from env)
    local_rank: int = 0
    world_size: int = 1


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: str = "logs", rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers = [logging.FileHandler(os.path.join(log_dir, f"train_{ts}_rank{rank}.log"))]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank{rank}] %(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("keural")


logger = logging.getLogger("keural")


# =============================================================================
# BINARY DATASET  (fix: header reads 36 bytes, not 32)
# =============================================================================

class BinaryShard:
    """Memory-mapped binary shard reader."""

    MAGIC = b"KEURAL\x00\x00"
    # Header format: <8s I Q Q Q> = 8+4+8+8+8 = 36 bytes
    HEADER_FMT = "<8sIQQQ"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 36

    def __init__(self, bin_path: str, idx_path: str):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.num_sequences: int = 0
        self.seq_length: int = 0
        self.offsets: List[tuple] = []
        self._load_metadata()

    def _load_metadata(self):
        # Read binary header (FIX: was f.read(32), needs 36)
        with open(self.bin_path, "rb") as f:
            raw = f.read(self.HEADER_SIZE)
            magic, _ver, num_seqs, seq_len, _pad = struct.unpack(self.HEADER_FMT, raw)
            if magic != self.MAGIC:
                raise ValueError(f"Bad magic in {self.bin_path}: {magic}")
            self.num_sequences = num_seqs
            self.seq_length = seq_len

        # Read index
        with open(self.idx_path, "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
            _sl = struct.unpack("<I", f.read(4))[0]
            self.offsets = []
            for _ in range(n):
                off = struct.unpack("<Q", f.read(8))[0]
                length = struct.unpack("<I", f.read(4))[0]
                self.offsets.append((off, length))

    def __len__(self) -> int:
        return self.num_sequences

    def iter_sequences(self, shuffle: bool = True, seed: int = 42) -> Iterator[Dict]:
        """Opens mmap in the calling process (pickle-safe)."""
        fh = open(self.bin_path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            indices = list(range(self.num_sequences))
            if shuffle:
                np.random.default_rng(seed).shuffle(indices)
            for idx in indices:
                off, _ = self.offsets[idx]
                raw = mm[off: off + self.seq_length * 4]
                seq = np.frombuffer(raw, dtype=np.uint32).copy()
                # Dataset shifts labels by 1; model does NOT shift again (fix)
                input_ids = torch.from_numpy(seq[:-1]).long()   # [seq_len-1]
                labels    = torch.from_numpy(seq[1:]).long()    # [seq_len-1]
                labels    = labels.masked_fill(labels == 0, -100)
                yield {"input_ids": input_ids, "labels": labels}
        finally:
            mm.close()
            fh.close()


class TrainingDataset(IterableDataset):
    def __init__(self, data_dir: str, prefix: str, shuffle: bool = True, seed: int = 42):
        self.shuffle = shuffle
        self.seed = seed
        bins = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.bin")))
        if not bins:
            raise FileNotFoundError(f"No binary shards in {data_dir}/{prefix}_*.bin")

        self.shards: List[Dict] = []
        self.total_sequences = 0
        for b in bins:
            idx = b.replace(".bin", ".idx")
            meta = b.replace(".bin", ".meta")
            if not os.path.exists(idx) or not os.path.exists(meta):
                raise FileNotFoundError(f"Missing .idx or .meta for {b}")
            with open(meta) as f:
                m = json.load(f)
            self.shards.append({"bin": b, "idx": idx, "num_sequences": m["num_sequences"]})
            self.total_sequences += m["num_sequences"]
        self.num_shards = len(self.shards)

    def __len__(self) -> int:
        return self.total_sequences

    def __iter__(self) -> Iterator[Dict]:
        worker = torch.utils.data.get_worker_info()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1

        # Assign shards: each rank gets every world-th shard
        shard_ids = [i for i in range(self.num_shards) if i % world == rank % world]

        # Further split if multiple dataloader workers
        if worker is not None:
            per = max(1, len(shard_ids) // worker.num_workers)
            s = worker.id * per
            e = s + per if worker.id < worker.num_workers - 1 else len(shard_ids)
            shard_ids = shard_ids[s:e]
            worker_seed = self.seed + worker.id
        else:
            worker_seed = self.seed + rank

        for sid in shard_ids:
            info = self.shards[sid]
            shard = BinaryShard(info["bin"], info["idx"])
            for sample in shard.iter_sequences(self.shuffle, seed=worker_seed + sid * 1000):
                yield sample


# =============================================================================
# MODEL
# =============================================================================

class RoPEEmbedding(nn.Module):
    """Standard RoPE; no YaRN scaling needed for Stage 1 (4K context)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(config.max_position_embeddings)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.num_heads    = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim     = config.head_dim
        self.kv_groups    = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = RoPEEmbedding(config)

        # Odd layers use full causal attention; even layers use sliding window
        self.sliding_window = config.sliding_window if (layer_idx % 2 == 0) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        # Expand KV for GQA
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        if self.sliding_window is not None:
            # Causal + sliding window mask (FIX: was bidirectional / missing)
            i = torch.arange(S, device=x.device)
            j = torch.arange(S, device=x.device)
            # Block: future tokens (j > i) OR too-far past (i - j > window)
            blocked = (j.unsqueeze(0) > i.unsqueeze(1)) | \
                      ((i.unsqueeze(1) - j.unsqueeze(0)) > self.sliding_window)
            attn_mask = torch.zeros(S, S, dtype=q.dtype, device=x.device)
            attn_mask.masked_fill_(blocked, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Full causal attention — uses FlashAttention automatically
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Expert(nn.Module):
    """SwiGLU FFN: output = down(silu(gate(x)) * up(x))"""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k       = config.num_experts_per_tok
        self.gate        = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts     = nn.ModuleList([
            Expert(config.hidden_size, config.intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x_flat = x.view(-1, shape[-1])                              # [T, H]
        T = x_flat.size(0)

        logits  = self.gate(x_flat)                                 # [T, E]
        probs   = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)       # [T, k]
        top_w   = (top_w / top_w.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.experts):
            mask = (top_i == e_idx).any(dim=-1)                     # [T]
            if not mask.any():
                continue
            e_in  = x_flat[mask]
            e_out = expert(e_in)
            pos   = (top_i[mask] == e_idx).nonzero(as_tuple=True)
            w     = top_w[mask][pos[0], pos[1]]
            out[mask] += e_out * w.unsqueeze(-1)

        aux = self._aux_loss(logits, top_i)
        return out.view(shape), aux

    def _aux_loss(self, logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1).mean(0)
        freq  = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            freq[i] = (selected == i).float().mean()
        aux    = self.num_experts * (probs * freq).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return 0.02 * aux + 0.001 * z_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.moe_norm  = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.attn      = Attention(config, layer_idx)
        self.moe       = MoELayer(config)

    def forward(self, x: torch.Tensor):
        # Attention block
        x = x + self.attn(self.attn_norm(x))
        # MoE block
        h = self.moe_norm(x)
        B, S, H = h.shape
        h_flat, aux = self.moe(h.view(-1, H))
        x = x + h_flat.view(B, S, H)
        return x, aux


class KeuralMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        # No separate lm_head weight: use F.linear with embed_tokens.weight (FSDP-safe tying)

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids)

        total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # FIX: use_reentrant=False required for FSDP in PyTorch 2.x
                x, aux = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x, aux = layer(x)
            total_aux = total_aux + aux

        x = self.norm(x)
        # FIX: use F.linear with embed weight directly (FSDP-safe, avoids shared tensor issue)
        logits = F.linear(x, self.embed_tokens.weight)

        loss = None
        if labels is not None:
            # FIX: NO shift here — dataset already shifted labels by 1
            # input_ids  = seq[:-1]  → model predicts each next token
            # labels      = seq[1:]  → ground truth
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + total_aux / self.config.num_hidden_layers

        return {"loss": loss, "logits": logits, "aux_loss": total_aux}


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig):
        self.mc = model_config
        self.tc = train_config

        self._setup_distributed()

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        global logger
        logger = setup_logging(rank=self.rank)

        if self.is_main:
            logger.info(f"World={self.world_size} | Device={self.device} | PyTorch={torch.__version__}")

        # Dataset
        self.dataset = TrainingDataset(train_config.data_dir, train_config.data_prefix)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=train_config.batch_size,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        if self.is_main:
            logger.info(f"Dataset: {self.dataset.total_sequences:,} sequences across {self.dataset.num_shards} shards")

        # Build model
        self._build_model()

        # Optimizer
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=train_config.weight_decay,
                fused=True,
            )
            if self.is_main:
                logger.info("Optimizer: AdamW (fused)")
        except TypeError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=train_config.weight_decay,
            )
            if self.is_main:
                logger.info("Optimizer: AdamW")

        self.step = 0
        self.tokens_processed = 0
        self.epoch = 0

        if train_config.resume_from:
            self._resume(train_config.resume_from)

        if self.is_main:
            os.makedirs(train_config.checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()

    def _setup_distributed(self):
        self.rank       = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main    = (self.rank == 0)

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
            )

    def _build_model(self):
        if self.is_main:
            logger.info("Building model on CPU (all ranks)...")

        dtype = torch.bfloat16 if self.tc.dtype == "bfloat16" else torch.float32

        # All ranks build real weights — avoids large NCCL broadcast on init
        torch.manual_seed(42)
        model = KeuralMoE(self.mc).to(dtype)

        total = sum(p.numel() for p in model.parameters())
        if self.is_main:
            logger.info(f"Parameters: {total:,} ({total/1e9:.2f}B)")
            logger.info(f"Model BF16: {total*2/1e9:.2f}GB | Per-GPU est (FSDP): {total*2/1e9/self.world_size:.2f}GB")

        # FSDP wrapping
        mp = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=torch.float32,
        )

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )

        self.model = FSDP(
            model,
            device_id=self.device if torch.cuda.is_available() else None,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            auto_wrap_policy=wrap_policy,
            cpu_offload=CPUOffload(offload_params=self.tc.fsdp_cpu_offload),
            use_orig_params=True,
        )

        if self.is_main:
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"FSDP ready | GPU memory allocated: {allocated:.2f}GB")

    def _resume(self, path: str):
        if self.is_main:
            logger.info(f"Resuming from {path}")

        ckpt = torch.load(path, map_location="cpu")

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            self.model.load_state_dict(ckpt["model"])

        optim_state = FSDP.optim_state_dict_to_load(
            self.model, self.optimizer, ckpt["optimizer"]
        )
        self.optimizer.load_state_dict(optim_state)

        self.step             = ckpt["step"]
        self.tokens_processed = ckpt["tokens_processed"]
        self.epoch            = ckpt.get("epoch", 0)

        if self.is_main:
            logger.info(f"Resumed at step {self.step}, tokens {self.tokens_processed:,}")

    def _save(self):
        if self.is_main:
            logger.info(f"Saving checkpoint at step {self.step}...")

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = self.model.state_dict()
            optim_state = FSDP.optim_state_dict(self.model, self.optimizer)

        if self.is_main:
            path = os.path.join(self.tc.checkpoint_dir, f"checkpoint_{self.step}.pt")
            tmp  = path + ".tmp"
            torch.save({
                "step":             self.step,
                "epoch":            self.epoch,
                "tokens_processed": self.tokens_processed,
                "model":            model_state,
                "optimizer":        optim_state,
                "model_config":     asdict(self.mc),
                "train_config":     asdict(self.tc),
            }, tmp)
            os.rename(tmp, path)
            logger.info(f"Checkpoint saved: {path}")

    def _get_lr(self) -> float:
        if self.step < self.tc.warmup_steps:
            return self.tc.learning_rate * (self.step + 1) / self.tc.warmup_steps
        progress = (self.step - self.tc.warmup_steps) / max(1, self.tc.max_steps - self.tc.warmup_steps)
        coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.tc.min_lr + (self.tc.learning_rate - self.tc.min_lr) * coeff

    def train(self):
        if self.is_main:
            logger.info("=" * 60)
            logger.info("STARTING TRAINING")
            logger.info(f"  Steps:     {self.tc.max_steps:,}")
            logger.info(f"  Batch:     {self.tc.batch_size} x {self.tc.gradient_accumulation_steps} accum x {self.world_size} GPUs")
            logger.info(f"  Eff batch: {self.tc.batch_size * self.tc.gradient_accumulation_steps * self.world_size} sequences")
            logger.info("=" * 60)
            sys.stdout.flush()

        # Enable gradient checkpointing
        if isinstance(self.model, FSDP):
            self.model.module.gradient_checkpointing = self.tc.gradient_checkpointing
        else:
            self.model.gradient_checkpointing = self.tc.gradient_checkpointing

        self.model.train()

        running_loss = 0.0
        running_aux  = 0.0
        t0           = time.time()
        t_step       = time.time()
        data_iter    = iter(self.dataloader)

        while self.step < self.tc.max_steps:
            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_aux  = 0.0

            for micro in range(self.tc.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                    if self.is_main:
                        logger.info(f"Epoch {self.epoch}")

                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                # Sync gradients only on last micro-step
                ctx = self.model.no_sync() if micro < self.tc.gradient_accumulation_steps - 1 else \
                      torch.contextlib.nullcontext() if hasattr(torch, 'contextlib') else \
                      __import__('contextlib').nullcontext()

                with ctx:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out  = self.model(input_ids, labels)
                        loss = out["loss"] / self.tc.gradient_accumulation_steps
                        aux  = out["aux_loss"] / self.tc.gradient_accumulation_steps
                    loss.backward()

                accum_loss += loss.item()
                accum_aux  += aux.item()
                self.tokens_processed += input_ids.numel() * self.world_size

            # Gradient clip + optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.max_grad_norm
            )

            lr = self._get_lr()
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.step()
            self.step += 1

            running_loss += accum_loss
            running_aux  += accum_aux

            # Logging
            if self.is_main and self.step % self.tc.log_interval == 0:
                avg_loss = running_loss / self.tc.log_interval
                avg_aux  = running_aux  / self.tc.log_interval
                elapsed  = time.time() - t0
                tok_s    = self.tokens_processed / elapsed if elapsed > 0 else 0
                step_s   = (time.time() - t_step) / self.tc.log_interval

                logger.info(
                    f"Step {self.step:6d}/{self.tc.max_steps} | "
                    f"Loss {avg_loss:.4f} | Aux {avg_aux:.4f} | "
                    f"GradNorm {grad_norm:.2f} | LR {lr:.2e} | "
                    f"{tok_s/1e3:.1f}K tok/s | {step_s:.1f}s/step"
                )
                running_loss = 0.0
                running_aux  = 0.0
                t_step       = time.time()

            # Checkpoint
            if self.step % self.tc.save_interval == 0:
                self._save()
                if self.world_size > 1:
                    dist.barrier()

        # Final checkpoint
        self._save()
        if self.is_main:
            total_h = (time.time() - t0) / 3600
            logger.info(f"Training complete. {self.tokens_processed/1e9:.2f}B tokens in {total_h:.2f}h")

        if dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Keural MoE Training v2")
    parser.add_argument("--data_dir",        default="data/binary")
    parser.add_argument("--data_prefix",     default="keural")
    parser.add_argument("--checkpoint_dir",  default="checkpoints/stage1")
    parser.add_argument("--resume",          default="auto") #for manual put None and for automatic add auto
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--grad_accum",      type=int,   default=4)
    parser.add_argument("--max_steps",       type=int,   default=100000)
    parser.add_argument("--save_interval",   type=int,   default=500)
    parser.add_argument("--log_interval",    type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--cpu_offload",     action="store_true", help="FSDP CPU offload (slower, less VRAM)")
    parser.add_argument("--no_grad_ckpt",    action="store_true", help="Disable gradient checkpointing (faster, more VRAM)")
    args = parser.parse_args()

    # Auto-resume: find latest checkpoint automatically
    if args.resume == "auto":
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt")))
        args.resume = ckpts[-1] if ckpts else None
        if args.resume:
            print(f"Auto-resuming from: {args.resume}")
        else:
            print("No checkpoint found, starting from scratch.")

    mc = ModelConfig()
    tc = TrainConfig(
        data_dir               = args.data_dir,
        data_prefix            = args.data_prefix,
        checkpoint_dir         = args.checkpoint_dir,
        resume_from            = args.resume,
        batch_size             = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        max_steps              = args.max_steps,
        save_interval          = args.save_interval,
        log_interval           = args.log_interval,
        learning_rate          = args.lr,
        fsdp_cpu_offload       = args.cpu_offload,
        gradient_checkpointing = not args.no_grad_ckpt,
    )

    trainer = Trainer(mc, tc)
    trainer.train()


if __name__ == "__main__":
    main()