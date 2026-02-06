"""
SSH (Selective-State Hybrid) Architecture.

Novel architecture combining:
1. Mamba-style SSM layers for efficient sequence processing
2. Selective attention at strategic depths
3. Adaptive early-exit for simple tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .layers import (
    RMSNorm,
    MambaSSM,
    SelectiveAttention,
    AdaptiveDepthBlock,
    FeedForward,
)


@dataclass
class SSHConfig:
    """Configuration for SSH model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4    # SSM convolution kernel
    expand: int = 2    # SSM expansion factor
    ff_expand: int = 4
    dropout: float = 0.0
    max_seq_len: int = 1024

    # SSH-specific
    attention_layers: List[int] = None  # Which layers use attention (default: [3, 7, 11])
    use_adaptive_depth: bool = True
    early_exit_threshold: float = 0.9
    attention_gate_threshold: float = 0.5

    def __post_init__(self):
        if self.attention_layers is None:
            # Default: attention at every 4th layer
            self.attention_layers = [i for i in range(self.n_layers) if (i + 1) % 4 == 0]


class SSHBlock(nn.Module):
    """
    Single SSH block that can use either SSM or attention.
    """

    def __init__(
        self,
        config: SSHConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_attention = layer_idx in config.attention_layers

        # Normalization
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        # Main layer: SSM or Attention
        if self.use_attention:
            self.mixer = SelectiveAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                gate_threshold=config.attention_gate_threshold,
            )
        else:
            self.mixer = MambaSSM(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )

        # Feed-forward
        self.ff = FeedForward(
            d_model=config.d_model,
            expand=config.ff_expand,
            dropout=config.dropout,
        )

        # Adaptive depth (optional)
        if config.use_adaptive_depth:
            self.adaptive = AdaptiveDepthBlock(
                d_model=config.d_model,
                threshold=config.early_exit_threshold,
            )
        else:
            self.adaptive = None

    def forward(
        self,
        x: torch.Tensor,
        exit_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional early exit.

        Returns dict with:
            - output: processed tensor
            - exit_mask: updated exit mask
            - gate_values: attention gate values (if attention layer)
            - confidence: early-exit confidence values
        """
        result = {"gate_values": None, "confidence": None}

        # Check adaptive early exit
        if self.adaptive is not None and exit_mask is not None:
            # Tokens that have exited skip this layer
            _, confidence, exit_mask = self.adaptive(x, exit_mask)
            result["confidence"] = confidence
            result["exit_mask"] = exit_mask

            # For exited tokens, skip computation (use identity)
            if exit_mask.any():
                # Create mask for computation
                compute_mask = ~exit_mask.unsqueeze(-1)
                x_compute = x * compute_mask.float()
            else:
                x_compute = x
        else:
            x_compute = x
            result["exit_mask"] = exit_mask

        # Main mixer (SSM or Attention)
        normed = self.norm1(x_compute)
        if self.use_attention:
            mixed, gate_values = self.mixer(normed)
            result["gate_values"] = gate_values
        else:
            mixed = self.mixer(normed)

        x = x + mixed

        # Feed-forward
        x = x + self.ff(self.norm2(x))

        result["output"] = x
        return result


class SSH(nn.Module):
    """
    Selective-State Hybrid Language Model.

    Combines:
    - SSM layers for efficient O(n) sequence processing
    - Sparse attention layers for global information
    - Adaptive depth for variable computation
    """

    def __init__(self, config: SSHConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Layers
        self.layers = nn.ModuleList([
            SSHBlock(config, i) for i in range(config.n_layers)
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
            metrics: dict of efficiency metrics (if return_metrics)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Track metrics
        metrics = {
            "gate_values": [],
            "confidences": [],
            "exit_masks": [],
            "layers_computed": [],
        }

        exit_mask = None

        # Forward through layers
        for layer in self.layers:
            result = layer(x, exit_mask)
            x = result["output"]
            exit_mask = result.get("exit_mask")

            if return_metrics:
                if result.get("gate_values") is not None:
                    metrics["gate_values"].append(result["gate_values"])
                if result.get("confidence") is not None:
                    metrics["confidences"].append(result["confidence"])
                if exit_mask is not None:
                    metrics["exit_masks"].append(exit_mask.clone())

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_metrics:
            # Compute summary metrics
            if metrics["exit_masks"]:
                # Average layers computed per token
                exit_layers = torch.stack(metrics["exit_masks"], dim=0).float()
                # Count first layer where each token exits
                exited_at = exit_layers.argmax(dim=0).float()
                exited_at[~exit_layers.any(dim=0)] = self.config.n_layers
                metrics["avg_layers"] = exited_at.mean().item()
            else:
                metrics["avg_layers"] = self.config.n_layers

            if metrics["gate_values"]:
                # Average attention usage
                gates = torch.cat(metrics["gate_values"], dim=-1)
                metrics["attention_usage"] = gates.mean().item()

            return {"logits": logits, "metrics": metrics}

        return {"logits": logits}

    def count_flops(self, seq_len: int) -> Dict[str, int]:
        """
        Estimate FLOPs for a forward pass.
        """
        d = self.config.d_model
        n = seq_len
        v = self.config.vocab_size
        L = self.config.n_layers
        n_attn = len(self.config.attention_layers)
        n_ssm = L - n_attn

        # Embedding: lookup (negligible)
        embed_flops = 0

        # SSM layers: O(n * d * d_state * expand)
        ssm_per_layer = n * d * self.config.d_state * self.config.expand * 4
        ssm_flops = ssm_per_layer * n_ssm

        # Attention layers: O(n^2 * d)
        attn_per_layer = 2 * n * n * d + 4 * n * d * d
        attn_flops = attn_per_layer * n_attn

        # FFN: O(n * d * d * ff_expand)
        ffn_per_layer = 2 * n * d * d * self.config.ff_expand
        ffn_flops = ffn_per_layer * L

        # LM head: O(n * d * v)
        head_flops = n * d * v

        total = embed_flops + ssm_flops + attn_flops + ffn_flops + head_flops

        return {
            "total": total,
            "ssm": ssm_flops,
            "attention": attn_flops,
            "ffn": ffn_flops,
            "lm_head": head_flops,
            "per_token": total // n,
        }


def create_ssh_small() -> SSH:
    """Create small (125M) SSH model."""
    config = SSHConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        attention_layers=[3, 7, 11],  # 3 attention layers
        use_adaptive_depth=True,
    )
    return SSH(config)


def create_ssh_medium() -> SSH:
    """Create medium (350M) SSH model."""
    config = SSHConfig(
        vocab_size=50257,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        attention_layers=[5, 11, 17, 23],  # 4 attention layers
        use_adaptive_depth=True,
    )
    return SSH(config)
