#!/usr/bin/env python3
"""
Keural MoE - Quick Inference Script
Usage:
  python scripts/infer_keural.py --checkpoint checkpoints/stage1/checkpoint_1500.pt
"""

import os, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False

TOKENIZER_PATH = "keural-tokenizer/keural_tokenizer.model"


@dataclass
class ModelConfig:
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    max_position_embeddings: int = 4096
    sliding_window: int = 512
    rope_theta: float = 500000.0


class RoPEEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(config.max_position_embeddings).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, x, seq_len):
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = RoPEEmbedding(config)
        self.sliding_window = config.sliding_window if (layer_idx % 2 == 0) else None

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, S), self.rope(k, S)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)
        if self.sliding_window is not None:
            i = torch.arange(S, device=x.device)
            j = torch.arange(S, device=x.device)
            blocked = (j.unsqueeze(0) > i.unsqueeze(1)) | ((i.unsqueeze(1) - j.unsqueeze(0)) > self.sliding_window)
            mask = torch.zeros(S, S, dtype=q.dtype, device=x.device).masked_fill_(blocked, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, -1))


class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate  = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config.hidden_size, config.intermediate_size)
                                      for _ in range(config.num_local_experts)])

    def forward(self, x):
        shape = x.shape
        xf = x.view(-1, shape[-1])
        probs = F.softmax(self.gate(xf), dim=-1, dtype=torch.float32)
        top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_w = (top_w / top_w.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(xf)
        for e_idx, expert in enumerate(self.experts):
            mask = (top_i == e_idx).any(dim=-1)
            if not mask.any(): continue
            pos = (top_i[mask] == e_idx).nonzero(as_tuple=True)
            w   = top_w[mask][pos[0], pos[1]]
            out[mask] += expert(xf[mask]) * w.unsqueeze(-1)
        return out.view(shape)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.moe_norm  = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.attn      = Attention(config, layer_idx)
        self.moe       = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.moe_norm(x)
        B, S, H = h.shape
        return x + self.moe(h.view(-1, H)).view(B, S, H)


class KeuralMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm   = nn.RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return F.linear(self.norm(x), self.embed_tokens.weight)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=150, temperature=0.8, top_p=0.9):
        self.eval()
        gen = input_ids.clone()
        for _ in range(max_new_tokens):
            ctx = gen[:, -self.config.max_position_embeddings:]
            logits = self(ctx)[:, -1, :] / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
            next_tok = sorted_idx.gather(-1, torch.multinomial(F.softmax(sorted_logits, dim=-1), 1))
            gen = torch.cat([gen, next_tok], dim=-1)
        return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/stage1/checkpoint_1500.pt")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    if SPM_AVAILABLE and os.path.exists(TOKENIZER_PATH):
        sp = spm.SentencePieceProcessor()
        sp.Load(TOKENIZER_PATH)
        encode = lambda t: torch.tensor([sp.Encode(t, out_type=int)], dtype=torch.long)
        decode = lambda ids: sp.Decode(ids.tolist())
        print(f"Tokenizer: vocab={sp.GetPieceSize()}")
    else:
        print("ERROR: sentencepiece not found. Install with: pip install sentencepiece")
        sys.exit(1)

    # Build model
    config = ModelConfig()
    print("Building model...")
    model = KeuralMoE(config).to(torch.bfloat16)

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    # Strip FSDP prefixes
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print(f"Missing keys: {len(missing)}")
    if unexpected: print(f"Unexpected keys: {len(unexpected)}")

    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    print(f"Step={step} | Loss={loss:.4f}" if isinstance(loss, float) else f"Step={step}")
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print("=" * 60)

    prompts = [
        ("English",        "The future of artificial intelligence"),
        ("Korean",         "인공지능의 미래는"),
        ("Korean Science", "한국어 자연어처리 기술의 발전은"),
        ("Code",           "def fibonacci(n):"),
    ]

    for label, prompt in prompts:
        print(f"\n[{label}] >>> {prompt}")
        input_ids = encode(prompt).to(device)
        output    = model.generate(input_ids, args.max_new_tokens, args.temperature, args.top_p)
        new_ids   = output[0, input_ids.shape[1]:].cpu()
        print(prompt + decode(new_ids))
        print()


if __name__ == "__main__":
    main()