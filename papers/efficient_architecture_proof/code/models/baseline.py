"""
Baseline Transformer for comparison.

Standard GPT-style transformer with full attention at every layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .layers import RMSNorm, FeedForward


@dataclass
class TransformerConfig:
    """Configuration for baseline transformer."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_expand: int = 4
    dropout: float = 0.0
    max_seq_len: int = 1024


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        assert config.d_model % config.n_heads == 0

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(
            d_model=config.d_model,
            expand=config.ff_expand,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Standard GPT-style transformer.

    Full attention at every layer, no adaptive computation.
    Used as baseline for SSH comparison.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        # Initialize
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
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            return_metrics: whether to return efficiency metrics

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Forward through layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if return_metrics:
            result["metrics"] = {
                "avg_layers": self.config.n_layers,
                "attention_usage": 1.0,  # Always uses full attention
            }

        return result

    def count_flops(self, seq_len: int) -> Dict[str, int]:
        """
        Estimate FLOPs for a forward pass.
        """
        d = self.config.d_model
        n = seq_len
        v = self.config.vocab_size
        L = self.config.n_layers

        # Attention: QKV projection + attention computation + output projection
        # QKV: 3 * n * d * d
        # Attention: 2 * n^2 * d (QK^T and attn @ V)
        # Out: n * d * d
        attn_per_layer = 4 * n * d * d + 2 * n * n * d
        attn_flops = attn_per_layer * L

        # FFN: 2 * n * d * d * ff_expand (SwiGLU has two projections)
        ffn_per_layer = 2 * n * d * d * self.config.ff_expand
        ffn_flops = ffn_per_layer * L

        # LM head: n * d * v
        head_flops = n * d * v

        total = attn_flops + ffn_flops + head_flops

        return {
            "total": total,
            "attention": attn_flops,
            "ffn": ffn_flops,
            "lm_head": head_flops,
            "per_token": total // n,
        }


def create_transformer_small() -> BaselineTransformer:
    """Create small (125M) baseline transformer."""
    config = TransformerConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
    )
    return BaselineTransformer(config)


def create_transformer_medium() -> BaselineTransformer:
    """Create medium (350M) baseline transformer."""
    config = TransformerConfig(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
    )
    return BaselineTransformer(config)
