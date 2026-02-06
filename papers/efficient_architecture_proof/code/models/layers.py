"""
Core layers for SSH (Selective-State Hybrid) architecture.

Implements:
- RMSNorm (efficient normalization)
- Mamba-style SSM layer
- Selective attention with gating
- Adaptive depth early-exit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MambaSSM(nn.Module):
    """
    Simplified Mamba-style State Space Model layer.

    Key features:
    - O(n) complexity in sequence length
    - Selective state updates based on input
    - Hardware-efficient parallel scan
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_conv)
        delta, B, C = x_dbl[..., :1], x_dbl[..., 1:self.d_state+1], x_dbl[..., self.d_state+1:]
        delta = F.softplus(delta)

        # Discretize A
        A = -torch.exp(self.A_log)

        # Selective scan (simplified - in production use parallel scan)
        y = self._selective_scan(x_conv, delta, A, B, C)

        # Output with skip connection
        y = y + x_conv * self.D
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Selective scan operation.

        For efficiency on MPS, we use a simple sequential implementation.
        Production code would use parallel scan for GPU.
        """
        batch, seq_len, d_inner = x.shape

        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h = exp(delta * A) * h + delta * B * x
            delta_t = delta[:, t]  # (batch, 1)
            x_t = x[:, t]  # (batch, d_inner)
            B_t = B[:, t]  # (batch, d_state)
            C_t = C[:, t]  # (batch, d_state)

            # Discretized state transition
            dA = torch.exp(delta_t.unsqueeze(-1) * A)  # (batch, 1, d_state)
            dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, 1, d_state)

            # Update state
            h = h * dA + x_t.unsqueeze(-1) * dB

            # Output
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (batch, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class SelectiveAttention(nn.Module):
    """
    Attention with learned gating to select which tokens need attention.

    Key insight: Not all tokens benefit from global attention.
    Simple tokens can skip attention entirely.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.gate_threshold = gate_threshold

        assert d_model % n_heads == 0

        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Gate to decide if token needs attention
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: (batch, seq_len, d_model)
            gate_values: (batch, seq_len, 1) - which tokens used attention
        """
        batch, seq_len, _ = x.shape

        # Compute gate values
        gate_values = self.gate(x)  # (batch, seq_len, 1)

        # During training, use soft gating
        # During inference, can hard-threshold for speedup
        if self.training:
            gate_mask = gate_values
        else:
            gate_mask = (gate_values > self.gate_threshold).float()

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)

        # Apply gate: blend attention output with skip connection
        out = gate_mask * out + (1 - gate_mask) * x

        return out, gate_values


class AdaptiveDepthBlock(nn.Module):
    """
    Block with learned early-exit based on confidence.

    If the model is confident about a token's representation,
    it can skip remaining layers for that token.
    """

    def __init__(
        self,
        d_model: int,
        threshold: float = 0.8,
    ):
        super().__init__()
        self.threshold = threshold

        # Confidence predictor
        self.confidence = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        exit_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            exit_mask: (batch, seq_len) - tokens that have already exited
        Returns:
            x: unchanged input (processing happens in parent)
            confidence: (batch, seq_len, 1)
            new_exit_mask: (batch, seq_len) - updated exit mask
        """
        batch, seq_len, _ = x.shape

        # Compute confidence
        confidence = self.confidence(x)  # (batch, seq_len, 1)

        # Update exit mask
        if exit_mask is None:
            exit_mask = torch.zeros(batch, seq_len, device=x.device, dtype=torch.bool)

        # Tokens exit if confident enough (and haven't already exited)
        new_exits = (confidence.squeeze(-1) > self.threshold) & ~exit_mask
        new_exit_mask = exit_mask | new_exits

        return x, confidence, new_exit_mask


class FeedForward(nn.Module):
    """Standard feed-forward layer with SwiGLU activation."""

    def __init__(
        self,
        d_model: int,
        expand: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_model * expand * 2 // 3  # SwiGLU uses 2/3 factor

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
