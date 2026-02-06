"""
Training script for A+B+C Architecture Combination Study (v3.4 - BPE TOKENIZATION)

Trains 4 conditions on the SAME data for fair comparison:
- C0: baseline (no MoD, no COCONUT, no Memory)
- C1: coconut_only (COCONUT only)
- C2: lahr_only (MoD + Memory, no COCONUT)
- C3: full_abc (MoD + Memory + COCONUT)

All conditions use the same CoT training data.
For non-COCONUT models: train on full text (Q + CoT + A)
For COCONUT models: use curriculum training (replace CoT with latent tokens)

FIXES from v2 review (v3):
1. HARDER DATA: Multi-step word problems (5-9 steps avg), not simple arithmetic
2. VALIDATION SPLIT: Report both train and val PPL
3. RESTORE ITERATIONS: Back to 4 latent iterations (full COCONUT)
4. RESTORE CURRICULUM: Back to 4 stages (full compression)
5. INCREASE SEEDS: Default to 5 seeds for statistical power
6. FLOP TRACKING: Document compute difference between conditions

v3.4 EXTENSION:
7. BPE TOKENIZATION: Add --tokenizer {char,bpe} flag to test COCONUT with subword tokens
   - Character-level defeats COCONUT (compresses characters, not reasoning)
   - BPE (GPT-2 encoding) allows proper reasoning step compression
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tokenizer factory (v3.4)
try:
    from data.bpe_tokenizer import get_tokenizer, SimpleTokenizer, BPETokenizer
    TOKENIZER_FACTORY_AVAILABLE = True
except ImportError:
    TOKENIZER_FACTORY_AVAILABLE = False


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device with CUDA optimizations."""
    if torch.cuda.is_available():
        # Enable cudnn.benchmark for faster training (slightly non-deterministic)
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS device (limited determinism)")
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Simple Tokenizer for CoT Data
# ============================================================================

class SimpleTokenizer:
    """Character-level tokenizer for fair comparison."""

    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<eos>": 1,
            "<bot>": 2,      # Beginning of thought
            "<thought>": 3,  # Thought token (for COCONUT)
            "<eot>": 4,      # End of thought
        }

        # Build vocab (ASCII printable + special)
        self.char_to_idx = dict(self.special_tokens)
        for i in range(32, 127):  # Printable ASCII
            self.char_to_idx[chr(i)] = len(self.char_to_idx)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.char_to_idx[" "]) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in ids)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<eos>"]

    @property
    def thought_token_id(self) -> int:
        return self.special_tokens["<thought>"]

    @property
    def bot_token_id(self) -> int:
        return self.special_tokens["<bot>"]

    @property
    def eot_token_id(self) -> int:
        return self.special_tokens["<eot>"]


# ============================================================================
# Dataset for All Conditions
# ============================================================================

class ABCDataset(Dataset):
    """
    Dataset for A+B+C study.

    For non-COCONUT models: Returns full text (Q + CoT + A)
    For COCONUT models: Supports curriculum with latent tokens
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,  # v3: increased for longer problems
        use_curriculum: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_curriculum = use_curriculum
        self.current_stage = 0

        # Load data
        with open(data_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def set_stage(self, stage: int):
        """Set curriculum stage (only for COCONUT)."""
        self.current_stage = stage

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        question = sample["question"]
        steps = sample.get("steps", [])
        answer = sample.get("answer", "")

        if self.use_curriculum and self.current_stage > 0:
            # COCONUT curriculum: replace k steps with k latent tokens
            n_latent = min(self.current_stage, len(steps))
            kept_steps = steps[n_latent:]

            # Build sequence: Q <bot> <thought>*n_latent <eot> [remaining CoT] A
            text = question + " "
            input_ids = self.tokenizer.encode(text)
            input_ids.append(self.tokenizer.bot_token_id)
            input_ids.extend([self.tokenizer.thought_token_id] * n_latent)
            input_ids.append(self.tokenizer.eot_token_id)

            # Add remaining steps and answer
            if kept_steps:
                cot_text = " ".join(kept_steps) + " "
                input_ids.extend(self.tokenizer.encode(cot_text))
            input_ids.extend(self.tokenizer.encode(answer))

            # Labels: mask out latent tokens (-100)
            labels = input_ids.copy()
            # Find thought token positions and mask them
            for i, tok in enumerate(input_ids):
                if tok == self.tokenizer.thought_token_id:
                    labels[i] = -100

        else:
            # Non-curriculum: full text
            cot_text = " ".join(steps) if steps else ""
            full_text = f"{question} {cot_text} {answer}"
            input_ids = self.tokenizer.encode(full_text)
            labels = input_ids.copy()

        # Truncate/pad
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # Don't compute loss on padding

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================================
# Model Definitions (Simplified for Fair Comparison)
# ============================================================================

@dataclass
class ModelConfig:
    """Unified config for all models."""
    vocab_size: int = 100  # Will be set from tokenizer
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512  # v3: increased

    # Component flags
    use_mod: bool = False
    use_coconut: bool = False
    use_memory: bool = False

    # MoD settings
    mod_capacity: float = 0.125
    mod_skip_compute: bool = True  # v3.1: If False, process all tokens but only update selected

    # COCONUT settings (v3: RESTORED to 4 iterations)
    n_latent_iterations: int = 4

    # Memory settings
    n_memory_slots: int = 64
    memory_top_k: int = 8

    # Special tokens
    thought_token_id: int = 3
    bot_token_id: int = 2
    eot_token_id: int = 4


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.d_model * config.ff_mult
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MoDLayer(nn.Module):
    """Mixture-of-Depths layer with router."""

    def __init__(self, config: ModelConfig, skip_compute: bool = True):
        super().__init__()
        self.block = TransformerBlock(config)
        self.router = nn.Linear(config.d_model, 1, bias=False)
        self.capacity = config.mod_capacity
        self.skip_compute = skip_compute  # v3.1: Option to process all but update selectively

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        k = max(1, int(T * self.capacity))

        # Router with tie-breaking
        scores = self.router(x).squeeze(-1) + torch.rand_like(x[:, :, 0]) * 1e-6

        # Top-k selection
        _, indices = torch.topk(scores, k, dim=1)
        indices_sorted, _ = torch.sort(indices, dim=1)

        # Build mask for selected tokens
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        mask = torch.zeros(B, T, 1, device=x.device, dtype=torch.bool)
        mask[batch_idx, indices_sorted] = True

        if self.skip_compute:
            # ORIGINAL: Only process selected tokens (faster but breaks causal attention)
            selected = x[batch_idx, indices_sorted]
            processed = self.block(selected)
            full = torch.zeros_like(x)
            full[batch_idx, indices_sorted] = processed
            return torch.where(mask.expand(-1, -1, D), full, x)
        else:
            # v3.1 FIX: Process ALL tokens, but only UPDATE selected ones
            # This preserves causal attention but loses efficiency benefit
            processed_all = self.block(x)
            return torch.where(mask.expand(-1, -1, D), processed_all, x)


class MemoryModule(nn.Module):
    """Differentiable memory retrieval."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )
        self.top_k = config.memory_top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(x)

        keys_norm = F.normalize(self.keys, dim=-1)
        queries_norm = F.normalize(queries, dim=-1)
        sims = torch.matmul(queries_norm, keys_norm.T)

        topk_sims, topk_idx = torch.topk(sims, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_sims, dim=-1)

        topk_values = self.values[topk_idx]
        retrieved = (topk_weights.unsqueeze(-1) * topk_values).sum(dim=2)
        retrieved = self.out_proj(retrieved)

        gate = self.gate(torch.cat([x, retrieved], dim=-1))
        return x + gate * retrieved


class ABCModel(nn.Module):
    """
    Unified model for A+B+C study.

    Components enabled via config flags:
    - use_mod: Mixture-of-Depths
    - use_coconut: COCONUT latent reasoning
    - use_memory: Differentiable memory
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Layers (MoD or standard)
        if config.use_mod:
            # Alternate between MoD and standard layers
            self.layers = nn.ModuleList([
                MoDLayer(config, skip_compute=config.mod_skip_compute) if i % 2 == 1 else TransformerBlock(config)
                for i in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ])

        # Memory (optional)
        self.memory = MemoryModule(config) if config.use_memory else None

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape

        # Embeddings
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos_ids)
        x = self.dropout(x)

        # Memory retrieval
        if self.memory and self.config.use_memory:
            x = self.memory(x)

        # Check for COCONUT processing
        if self.config.use_coconut:
            thought_mask = (input_ids == self.config.thought_token_id)
            if thought_mask.any():
                x = self._coconut_forward(x, thought_mask)
            else:
                for layer in self.layers:
                    x = layer(x)
        else:
            for layer in self.layers:
                x = layer(x)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def _coconut_forward(self, x: torch.Tensor, thought_mask: torch.Tensor) -> torch.Tensor:
        """
        COCONUT: iteratively replace thought embeddings with shifted hidden states.

        v3: RESTORED to 4 iterations for full COCONUT mechanism.
        """
        B, T, D = x.shape

        # Build replacement mask (thought positions > 0)
        replace_mask = thought_mask & (torch.arange(T, device=x.device).unsqueeze(0) > 0)
        replace_mask = replace_mask.unsqueeze(-1)  # [B, T, 1]

        # Working embeddings for iterative refinement
        working_embeds = x

        # Iterate: process through transformer, then replace thought positions
        for _ in range(self.config.n_latent_iterations):
            # Forward through all layers
            hidden = working_embeds
            for layer in self.layers:
                hidden = layer(hidden)

            # Replace thought positions with shifted hidden states
            # (hidden state at position p-1 becomes embedding at position p)
            shifted = torch.cat([hidden[:, :1, :], hidden[:, :-1, :]], dim=1)
            working_embeds = torch.where(replace_mask.expand(-1, -1, D), shifted, working_embeds)

        # Final forward pass after all iterations
        final_hidden = working_embeds
        for layer in self.layers:
            final_hidden = layer(final_hidden)

        return final_hidden


# ============================================================================
# Condition Factory
# ============================================================================

def create_model(
    condition: str,
    tokenizer,  # SimpleTokenizer or BPETokenizer
    size: str = "tiny",
    mod_capacity: float = None,  # v3.1: Allow overriding MoD capacity
    mod_skip_compute: bool = None,  # v3.1: If False, process all tokens but only update selected
    max_seq_len: int = 512,  # v3.4: Configurable sequence length
    dropout: float = None,  # v4.1: Override dropout for FLOP controls
) -> ABCModel:
    """Create model for given condition."""

    size_configs = {
        "tiny": {"d_model": 128, "n_heads": 4, "n_layers": 4},      # ~7.5M with BPE
        "small": {"d_model": 256, "n_heads": 8, "n_layers": 8},     # ~15M with BPE
        "medium": {"d_model": 384, "n_heads": 6, "n_layers": 8},    # ~40M with BPE
        "large": {"d_model": 512, "n_heads": 8, "n_layers": 12},    # ~75M with BPE
        "base": {"d_model": 640, "n_heads": 10, "n_layers": 14},    # ~124M with BPE (125M scale)
    }

    base = size_configs[size]

    # v3.1: Added mod_only and memory_only for component isolation
    conditions = {
        "baseline": {
            "use_mod": False, "use_coconut": False, "use_memory": False,
        },
        "coconut_only": {
            "use_mod": False, "use_coconut": True, "use_memory": False,
        },
        "mod_only": {  # v3.1: Isolate MoD component
            "use_mod": True, "use_coconut": False, "use_memory": False,
        },
        "memory_only": {  # v3.1: Isolate Memory component
            "use_mod": False, "use_coconut": False, "use_memory": True,
        },
        "coconut_memory": {  # v3.2: COCONUT + Memory (no MoD)
            "use_mod": False, "use_coconut": True, "use_memory": True,
        },
        "lahr_only": {
            "use_mod": True, "use_coconut": False, "use_memory": True,
        },
        "full_abc": {
            "use_mod": True, "use_coconut": True, "use_memory": True,
        },
    }

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=base["d_model"],
        n_heads=base["n_heads"],
        n_layers=base["n_layers"],
        max_seq_len=max_seq_len,  # v3.4: Use configurable seq len
        thought_token_id=tokenizer.thought_token_id,
        bot_token_id=tokenizer.bot_token_id,
        eot_token_id=tokenizer.eot_token_id,
    )

    # Apply condition-specific flags
    for k, v in conditions[condition].items():
        setattr(config, k, v)

    # v3.1: Override MoD capacity if specified
    if mod_capacity is not None:
        config.mod_capacity = mod_capacity

    # v3.1: Override MoD skip_compute if specified
    if mod_skip_compute is not None:
        config.mod_skip_compute = mod_skip_compute

    # v4.1: Override dropout if specified (for FLOP control experiments)
    if dropout is not None:
        config.dropout = dropout

    return ABCModel(config)


# ============================================================================
# FLOP Estimation
# ============================================================================

def estimate_flops_per_forward(config: ModelConfig, seq_len: int) -> int:
    """Estimate FLOPs for a single forward pass."""
    d = config.d_model
    h = config.n_heads
    L = config.n_layers
    T = seq_len
    V = config.vocab_size
    ff = d * config.ff_mult

    # Per transformer layer:
    # Attention: 4*T*d^2 (Q,K,V,O projections) + 2*T^2*d (attention scores + output)
    # FFN: 3*T*d*ff (SwiGLU)
    attn_flops = 4 * T * d * d + 2 * T * T * d
    ffn_flops = 3 * T * d * ff
    layer_flops = attn_flops + ffn_flops

    # Total layers
    total = L * layer_flops

    # Embedding + LM head
    total += T * d  # Embed lookup
    total += T * d * V  # LM head

    return int(total)


def estimate_condition_flops(condition: str, config: ModelConfig, seq_len: int, has_thought_tokens: bool) -> Dict[str, int]:
    """Estimate total FLOPs for a condition considering COCONUT iterations."""
    base_flops = estimate_flops_per_forward(config, seq_len)

    if condition in ["coconut_only", "full_abc"] and has_thought_tokens:
        # COCONUT does n_latent_iterations + 1 forward passes
        n_passes = config.n_latent_iterations + 1
    else:
        n_passes = 1

    return {
        "flops_per_forward": base_flops,
        "n_forward_passes": n_passes,
        "total_flops": base_flops * n_passes,
    }


# ============================================================================
# Training Loop
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Learning rate with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Cosine decay to 10% of max
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(model: ABCModel, dataloader: DataLoader, device: torch.device, max_batches: int = None) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches and n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 10))

    return {"loss": avg_loss, "ppl": ppl}


def train_condition(
    condition: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    size: str = "tiny",
    max_steps: int = 500,
    batch_size: int = 4,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_stages: int = 4,  # v3: RESTORED to 4 stages
    warmup_steps: int = 50,
    eval_interval: int = 100,
    mod_capacity: float = None,  # v3.1: Allow overriding MoD capacity
    mod_skip_compute: bool = None,  # v3.1: If False, process all tokens but only update selected
    tokenizer_type: str = "char",  # v3.4: "char" or "bpe"
    max_seq_len: int = None,  # v3.4: Override sequence length (default: 512 char, 256 bpe)
    checkpoint_path: str = None,  # v3.5: Load from checkpoint for warm-start
    warmstart_from_condition: str = None,  # v3.5: Which condition the checkpoint was trained as
    dropout: float = None,  # v4.1: Override dropout for FLOP controls
) -> Dict[str, Any]:
    """Train a single condition."""

    set_seed(seed)
    device = get_device()

    # Tokenizer (v3.4: support BPE)
    if TOKENIZER_FACTORY_AVAILABLE:
        tokenizer = get_tokenizer(tokenizer_type)
    else:
        if tokenizer_type == "bpe":
            raise ImportError("BPE tokenizer requires bpe_tokenizer.py. Run from code/ directory.")
        tokenizer = SimpleTokenizer()

    # Determine sequence length (v3.4: shorter for BPE since it's more efficient)
    if max_seq_len is None:
        max_seq_len = 256 if tokenizer_type == "bpe" else 512

    # Dataset (use curriculum for COCONUT conditions)
    use_curriculum = condition in ["coconut_only", "coconut_memory", "full_abc"]
    train_dataset = ABCDataset(
        train_data_path,
        tokenizer,
        max_seq_len=max_seq_len,
        use_curriculum=use_curriculum,
    )
    val_dataset = ABCDataset(
        val_data_path,
        tokenizer,
        max_seq_len=max_seq_len,
        use_curriculum=use_curriculum,
    )

    # CUDA optimization: pin_memory for faster host-to-device transfer
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    # Model (v3.1: pass mod_capacity and mod_skip_compute overrides, v3.4: pass max_seq_len, v4.1: dropout)
    model = create_model(condition, tokenizer, size, mod_capacity=mod_capacity, mod_skip_compute=mod_skip_compute, max_seq_len=max_seq_len, dropout=dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # v3.5: Warm-start - load checkpoint from a previous training run
    warmstart_info = None
    if checkpoint_path is not None:
        print(f"\n🔥 WARM-START: Loading checkpoint from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # The checkpoint was trained as a different condition (e.g., baseline)
        # We need to handle potential architecture differences
        if warmstart_from_condition:
            print(f"   Checkpoint was trained as: {warmstart_from_condition}")
            print(f"   Continuing training as: {condition}")

            # Load only compatible weights (embedding, layers, norms)
            model_state = model.state_dict()
            loaded_count = 0
            skipped_count = 0
            for k, v in checkpoint_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_count += 1
                else:
                    skipped_count += 1
                    print(f"   Skipped: {k} (shape mismatch or not in model)")
            model.load_state_dict(model_state)
            print(f"   Loaded {loaded_count} parameters, skipped {skipped_count}")
        else:
            model.load_state_dict(checkpoint_state)
            print(f"   Loaded full checkpoint")

        warmstart_info = {
            "checkpoint_path": checkpoint_path,
            "warmstart_from_condition": warmstart_from_condition,
            "target_condition": condition,
        }

    # FLOP estimation
    flop_info = estimate_condition_flops(condition, model.config, max_seq_len, use_curriculum)

    print(f"\n{'='*60}")
    print(f"Condition: {condition} (seed={seed})")
    if warmstart_info:
        print(f"🔥 WARM-START from: {warmstart_from_condition} checkpoint")
    print(f"Tokenizer: {tokenizer_type} (vocab_size={tokenizer.vocab_size})")
    print(f"Sequence length: {max_seq_len}")
    print(f"Parameters: {n_params:,}")
    print(f"Use curriculum: {use_curriculum} (stages={n_stages if use_curriculum else 1})")
    print(f"FLOPs per step: {flop_info['total_flops']:,} ({flop_info['n_forward_passes']} forward passes)")
    print(f"{'='*60}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        eps=1e-8,
    )

    # Training
    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    running_tokens = 0
    start_time = time.time()

    metrics_history = []
    val_history = []

    # For COCONUT: curriculum stages
    effective_stages = n_stages if use_curriculum else 1
    steps_per_stage = max_steps // effective_stages

    while step < max_steps:
        epoch += 1
        current_stage = min(step // steps_per_stage, effective_stages - 1) if use_curriculum else 0

        if use_curriculum:
            train_dataset.set_stage(current_stage)
            val_dataset.set_stage(current_stage)

        for batch in train_loader:
            if step >= max_steps:
                break

            # Apply learning rate warmup
            lr = get_lr(step, warmup_steps, max_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            running_tokens += (labels != -100).sum().item()
            step += 1

            # Log every 50 steps
            if step % 50 == 0:
                avg_loss = running_loss / 50
                elapsed = time.time() - start_time
                tok_per_sec = running_tokens / elapsed
                ppl = math.exp(min(avg_loss, 10))

                print(f"Step {step}/{max_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | {tok_per_sec:.0f} tok/s | Stage: {current_stage} | LR: {lr:.2e}")

                metrics_history.append({
                    "step": step,
                    "loss": avg_loss,
                    "ppl": ppl,
                    "tok_per_sec": tok_per_sec,
                    "stage": current_stage,
                    "lr": lr,
                })

                running_loss = 0.0
                running_tokens = 0
                start_time = time.time()

            # Evaluate on validation set
            if step % eval_interval == 0 or step == max_steps:
                val_metrics = evaluate(model, val_loader, device, max_batches=20)
                print(f"  → Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['ppl']:.2f}")
                val_history.append({
                    "step": step,
                    "val_loss": val_metrics["loss"],
                    "val_ppl": val_metrics["ppl"],
                    "stage": current_stage,
                })

    # Final metrics
    final_train_loss = metrics_history[-1]["loss"] if metrics_history else float("inf")
    final_train_ppl = metrics_history[-1]["ppl"] if metrics_history else float("inf")
    final_val_loss = val_history[-1]["val_loss"] if val_history else float("inf")
    final_val_ppl = val_history[-1]["val_ppl"] if val_history else float("inf")
    final_tok_per_sec = sum(m["tok_per_sec"] for m in metrics_history[-5:]) / max(1, min(5, len(metrics_history)))

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "condition": condition,
        "n_params": n_params,
        "final_train_loss": final_train_loss,
        "final_train_ppl": final_train_ppl,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "avg_tok_per_sec": final_tok_per_sec,
        "max_steps": max_steps,
        "n_stages": effective_stages,
        "flops_per_step": flop_info["total_flops"],
        "n_forward_passes": flop_info["n_forward_passes"],
        # v3.4: Tokenizer metadata
        "tokenizer": {
            "type": tokenizer_type,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": max_seq_len,
        },
        # v3.5: Warm-start metadata
        "warmstart": warmstart_info,
        "train_history": metrics_history,
        "val_history": val_history,
    }

    with open(f"{output_dir}/{condition}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), f"{output_dir}/{condition}_model.pt")

    return results


def run_abc_study(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    size: str = "tiny",
    max_steps: int = 500,
    seeds: List[int] = None,
    n_stages: int = 4,  # v3: RESTORED to 4 stages
) -> Dict[str, Any]:
    """
    Run the full A+B+C study (4 conditions).

    v3: Uses 5 seeds by default for proper statistical power.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1001]  # v3: default 5 seeds

    conditions = ["baseline", "coconut_only", "lahr_only", "full_abc"]

    # Store results per condition per seed
    all_runs = {cond: [] for cond in conditions}

    print("\n" + "=" * 70)
    print("A+B+C ARCHITECTURE COMBINATION STUDY (v3 - FULL FIXES)")
    print("=" * 70)
    print(f"Train data: {train_data_path}")
    print(f"Val data: {val_data_path}")
    print(f"Size: {size}")
    print(f"Steps per condition: {max_steps}")
    print(f"Seeds: {seeds} (n={len(seeds)})")
    print(f"Curriculum stages: {n_stages}")
    print("=" * 70 + "\n")

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        for condition in conditions:
            results = train_condition(
                condition=condition,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=f"{output_dir}/seed_{seed}",
                size=size,
                max_steps=max_steps,
                seed=seed,
                n_stages=n_stages,
            )
            all_runs[condition].append(results)

    # Compute statistics across seeds
    stats = {}
    for condition in conditions:
        train_ppls = [r["final_train_ppl"] for r in all_runs[condition]]
        val_ppls = [r["final_val_ppl"] for r in all_runs[condition]]
        train_losses = [r["final_train_loss"] for r in all_runs[condition]]
        val_losses = [r["final_val_loss"] for r in all_runs[condition]]
        throughputs = [r["avg_tok_per_sec"] for r in all_runs[condition]]
        flops = all_runs[condition][0]["flops_per_step"]
        n_passes = all_runs[condition][0]["n_forward_passes"]

        stats[condition] = {
            "n_params": all_runs[condition][0]["n_params"],
            "train_ppl_mean": np.mean(train_ppls),
            "train_ppl_std": np.std(train_ppls, ddof=1) if len(train_ppls) > 1 else 0,
            "val_ppl_mean": np.mean(val_ppls),
            "val_ppl_std": np.std(val_ppls, ddof=1) if len(val_ppls) > 1 else 0,
            "train_loss_mean": np.mean(train_losses),
            "val_loss_mean": np.mean(val_losses),
            "throughput_mean": np.mean(throughputs),
            "flops_per_step": flops,
            "n_forward_passes": n_passes,
            "n_seeds": len(seeds),
        }

    # Summary table with statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (v3 - WITH VALIDATION)")
    print("=" * 70)

    print(f"\n{'Condition':<15} {'Train PPL':>15} {'Val PPL':>15} {'vs Baseline':>12} {'FLOPs':>15}")
    print("-" * 70)
    baseline_val_ppl = stats["baseline"]["val_ppl_mean"]
    baseline_flops = stats["baseline"]["flops_per_step"]

    for condition, s in stats.items():
        vs_baseline = (s["val_ppl_mean"] / baseline_val_ppl - 1) * 100
        flop_ratio = s["flops_per_step"] / baseline_flops
        train_ppl_str = f"{s['train_ppl_mean']:.2f} ± {s['train_ppl_std']:.2f}"
        val_ppl_str = f"{s['val_ppl_mean']:.2f} ± {s['val_ppl_std']:.2f}"
        print(f"{condition:<15} {train_ppl_str:>15} {val_ppl_str:>15} {vs_baseline:>+10.1f}% {flop_ratio:>13.1f}x")

    # Compute fairness analysis
    print("\n" + "-" * 70)
    print("COMPUTE FAIRNESS ANALYSIS")
    print("-" * 70)
    for condition, s in stats.items():
        flop_ratio = s["flops_per_step"] / baseline_flops
        if flop_ratio > 1.5:
            print(f"⚠️  {condition}: {flop_ratio:.1f}x more FLOPs than baseline ({s['n_forward_passes']} forward passes)")
        else:
            print(f"✓  {condition}: {flop_ratio:.1f}x FLOPs (fair comparison)")

    # Decision logic
    print("\n" + "-" * 70)
    print("DECISION ANALYSIS")
    print("-" * 70)

    abc_val_ppl = stats["full_abc"]["val_ppl_mean"]
    baseline_val_ppl = stats["baseline"]["val_ppl_mean"]
    min_other_val_ppl = min(
        stats["baseline"]["val_ppl_mean"],
        stats["coconut_only"]["val_ppl_mean"],
        stats["lahr_only"]["val_ppl_mean"],
    )

    # Statistical confidence check
    if len(seeds) > 1:
        abc_std = stats["full_abc"]["val_ppl_std"]
        baseline_std = stats["baseline"]["val_ppl_std"]
        combined_se = math.sqrt(abc_std**2 + baseline_std**2) / math.sqrt(len(seeds))
        diff = abc_val_ppl - baseline_val_ppl
        significant = abs(diff) > 2 * combined_se
        print(f"Statistical check: diff={diff:.3f}, 2×SE={2*combined_se:.3f}, significant={significant}")

    if abc_val_ppl < 0.95 * min_other_val_ppl:
        print("STRONG SIGNAL: A+B+C shows synergy (>5% better than best individual)")
        print("Recommendation: PROCEED to Phase 2")
    elif abc_val_ppl < baseline_val_ppl:
        print("WEAK SIGNAL: Combination helps, but not clearly synergistic")
        print("Recommendation: INVESTIGATE further")
    elif abc_val_ppl > 1.2 * baseline_val_ppl:
        print("REJECT: Components interfere destructively (>20% worse)")
        print("Recommendation: STOP")
    else:
        print("INCONCLUSIVE: Need more data or different setup")
        print("Recommendation: Extend training or adjust parameters")

    # Save summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump({
            "version": "v3_full_fixes",
            "fixes_applied": [
                "Multi-step word problems (5-9 steps avg)",
                "Validation split (15% held out)",
                "Restored 4 latent iterations",
                "Restored 4 curriculum stages",
                "5 seeds for statistical power",
                "FLOP tracking for fairness",
            ],
            "stats": stats,
            "all_runs": {
                k: [{
                    "train_ppl": r["final_train_ppl"],
                    "val_ppl": r["final_val_ppl"],
                    "train_loss": r["final_train_loss"],
                    "val_loss": r["final_val_loss"],
                    "throughput": r["avg_tok_per_sec"],
                } for r in v]
                for k, v in all_runs.items()
            },
            "baseline_val_ppl": baseline_val_ppl,
            "abc_val_ppl": abc_val_ppl,
            "min_other_val_ppl": min_other_val_ppl,
            "abc_vs_baseline": (abc_val_ppl / baseline_val_ppl - 1) * 100,
            "abc_vs_best_other": (abc_val_ppl / min_other_val_ppl - 1) * 100,
            "n_seeds": len(seeds),
            "seeds": seeds,
        }, f, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A+B+C Study Training (v3.4 - BPE Tokenization)")
    parser.add_argument("--train_data", type=str, default="data/multistep_train.json")
    parser.add_argument("--val_data", type=str, default="data/multistep_val.json")
    parser.add_argument("--output", type=str, default="../results/abc_study_v3")
    parser.add_argument("--size", type=str, choices=["tiny", "small", "medium", "large", "base"], default="tiny")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42, help="Single seed (use --seeds for multiple)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds (default: 42 123 456 789 1001)")
    parser.add_argument("--n_stages", type=int, default=4,
                        help="Number of curriculum stages for COCONUT (default: 4)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["baseline", "coconut_only", "mod_only", "memory_only", "coconut_memory", "lahr_only", "full_abc"],
                        help="Run single condition (default: run all)")
    # v3.1: Add mod_capacity override
    parser.add_argument("--mod_capacity", type=float, default=None,
                        help="Override MoD capacity (default: 0.125, try 0.5 for less aggressive)")
    # v3.1: Add mod_skip_compute override
    parser.add_argument("--mod_skip_compute", type=str, default=None, choices=["true", "false"],
                        help="MoD skip_compute mode: true=only process selected tokens (faster), false=process all tokens but only update selected (preserves attention)")
    # v3.4: Add tokenizer selection
    parser.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default="char",
                        help="Tokenizer type: 'char' for character-level (default), 'bpe' for GPT-2 BPE subword")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Maximum sequence length (default: 512 for char, 256 for bpe)")
    # v3.5: Warm-start support
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file for warm-start (e.g., baseline_model.pt)")
    parser.add_argument("--warmstart_from", type=str, default=None,
                        choices=["baseline", "coconut_only", "mod_only", "memory_only", "coconut_memory", "lahr_only", "full_abc"],
                        help="Which condition the checkpoint was trained as (for architecture matching)")
    # v4.1: FLOP control experiments
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout rate (default: 0.1, try 0.2 for regularization control)")

    args = parser.parse_args()

    # Determine seeds to use
    if args.seeds:
        seeds = args.seeds
    elif args.condition:
        seeds = [args.seed]  # Single condition uses single seed
    else:
        seeds = [42, 123, 456, 789, 1001]  # v3: default 5 seeds

    if args.condition:
        # Single condition (use first seed or all specified seeds)
        for seed in seeds:
            train_condition(
                condition=args.condition,
                train_data_path=args.train_data,
                val_data_path=args.val_data,
                output_dir=f"{args.output}/seed_{seed}" if len(seeds) > 1 else args.output,
                size=args.size,
                max_steps=args.max_steps,
                seed=seed,
                n_stages=args.n_stages,
                mod_capacity=args.mod_capacity,
                mod_skip_compute=args.mod_skip_compute == "false" if args.mod_skip_compute else None,
                tokenizer_type=args.tokenizer,  # v3.4
                max_seq_len=args.max_seq_len,   # v3.4
                checkpoint_path=args.checkpoint,  # v3.5
                warmstart_from_condition=args.warmstart_from,  # v3.5
                dropout=args.dropout,  # v4.1
            )
    else:
        # Full study with all seeds
        # Note: run_abc_study doesn't yet support tokenizer_type/max_seq_len
        # Use single condition mode for BPE experiments
        if args.tokenizer == "bpe":
            print("WARNING: Full study mode not yet updated for BPE. Use --condition flag.")
        run_abc_study(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_dir=args.output,
            size=args.size,
            max_steps=args.max_steps,
            seeds=seeds,
            n_stages=args.n_stages,
        )
