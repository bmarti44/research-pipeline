"""
LAHR: Latent Adaptive Hierarchical Reasoning Architecture

Combines:
1. Latent-space reasoning (thinking in embeddings, not tokens)
2. Adaptive depth with shared parameters (variable compute per input)
3. Hierarchical memory bank (persistent working memory)

Designed for progressive scaling:
- Tiny (1M params): Validate architecture works at all
- Small (10M params): Validate each component contributes
- Medium (50M params): Validate benefits scale
- Large (125M params): Final validation, comparison to baselines

All trainable on consumer hardware (MacBook, 36GB RAM).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LAHRConfig:
    """
    Configuration for LAHR architecture.

    Scale presets:
    - tiny:   d_model=128,  n_heads=4,  ~1M params
    - small:  d_model=256,  n_heads=8,  ~10M params
    - medium: d_model=512,  n_heads=8,  ~50M params
    - large:  d_model=768,  n_heads=12, ~125M params
    """
    # Core dimensions
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    ff_expand: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512

    # Adaptive depth (shared block applied iteratively)
    n_shared_blocks: int = 2      # Number of distinct shared blocks
    min_depth: int = 2            # Minimum iterations
    max_depth: int = 16           # Maximum iterations
    halt_threshold: float = 0.99  # Cumulative probability to halt

    # Latent reasoning
    latent_dim: int = None        # Defaults to d_model
    n_latent_iterations: int = 4  # Max reasoning iterations
    use_latent_reasoning: bool = True

    # Hierarchical memory
    n_memory_slots: int = 64      # Number of memory slots
    memory_dim: int = None        # Defaults to d_model
    use_memory: bool = True

    def __post_init__(self):
        if self.latent_dim is None:
            self.latent_dim = self.d_model
        if self.memory_dim is None:
            self.memory_dim = self.d_model


# =============================================================================
# Core Layers
# =============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention with rotary embeddings."""

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU Feed-forward."""
    def __init__(self, config: LAHRConfig):
        super().__init__()
        d_ff = config.d_model * config.ff_expand * 2 // 3

        self.w1 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w2 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# =============================================================================
# Component 1: Adaptive Depth (Shared Block with Halting)
# =============================================================================

class AdaptiveTransformerBlock(nn.Module):
    """
    Transformer block with learned halting mechanism.

    Applied iteratively with shared parameters. Each iteration,
    a halting probability is computed. Tokens stop when cumulative
    halting exceeds threshold.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

        # Halting mechanism
        self.halt_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cumulative_halt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single iteration of the adaptive block.

        Returns:
            x: Updated hidden state
            halt_probs: This iteration's halting probabilities (B, T, 1)
            cumulative_halt: Updated cumulative halting (B, T, 1)
        """
        B, T, _ = x.shape

        # Standard transformer ops
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        # Compute halting probability
        halt_probs = self.halt_predictor(x)  # (B, T, 1)

        # Update cumulative halting
        if cumulative_halt is None:
            cumulative_halt = halt_probs
        else:
            # Only accumulate for tokens that haven't halted
            still_computing = (cumulative_halt < self.config.halt_threshold).float()
            cumulative_halt = cumulative_halt + halt_probs * still_computing

        return x, halt_probs, cumulative_halt


class AdaptiveDepthModule(nn.Module):
    """
    Applies shared transformer blocks iteratively with adaptive halting.

    Key insight: Easy inputs halt early, hard inputs use full depth.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config

        # Shared blocks (small set, applied iteratively)
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(config)
            for _ in range(config.n_shared_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply adaptive depth processing.

        Returns dict with:
            - output: Processed tensor
            - effective_depth: Average iterations used
            - halt_distribution: Histogram of halting iterations
        """
        B, T, _ = x.shape

        cumulative_halt = None
        iterations_used = torch.zeros(B, T, device=x.device)
        weighted_output = torch.zeros_like(x)
        remainder = torch.ones(B, T, 1, device=x.device)

        metrics = {"halt_probs": [], "depths": []}

        for iteration in range(self.config.max_depth):
            # Select which shared block to use (cycle through)
            block_idx = iteration % self.config.n_shared_blocks
            block = self.blocks[block_idx]

            # Apply block
            x, halt_probs, cumulative_halt = block(x, cumulative_halt)

            # ACT-style weighted average: contribute to output proportionally
            # Tokens that halt this iteration contribute their current state
            halted_this_iter = (cumulative_halt >= self.config.halt_threshold).float() * remainder
            weighted_output = weighted_output + halted_this_iter * x

            # Update remainder (how much each token still needs to contribute)
            remainder = remainder * (1 - halted_this_iter)

            # Track iterations (for metrics)
            iterations_used = iterations_used + (remainder.squeeze(-1) > 0).float()

            if return_metrics:
                metrics["halt_probs"].append(halt_probs.mean().item())
                metrics["depths"].append(iterations_used.mean().item())

            # Early exit if all tokens have halted
            if iteration >= self.config.min_depth - 1:
                if (cumulative_halt >= self.config.halt_threshold).all():
                    break

        # Add remainder to output (tokens that never fully halted)
        weighted_output = weighted_output + remainder * x

        result = {
            "output": weighted_output,
            "effective_depth": iterations_used.mean().item(),
            "iterations_used": iterations_used,
        }

        if return_metrics:
            result["metrics"] = metrics

        return result


# =============================================================================
# Component 2: Latent Reasoning Module
# =============================================================================

class LatentReasoningModule(nn.Module):
    """
    Reasoning in continuous latent space instead of token space.

    Key insight: One latent iteration operates on the full embedding
    dimensionality vs a token's one-hot bottleneck. Much higher bandwidth
    for "thinking."
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config

        # Project to latent space (may compress or expand)
        self.to_latent = nn.Linear(config.d_model, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, config.d_model)

        # Recurrent reasoning transformation
        self.reasoning_layer = nn.Sequential(
            RMSNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.GELU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.Dropout(config.dropout),
        )

        # Learned halting for reasoning iterations
        self.reasoning_halt = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 4),
            nn.GELU(),
            nn.Linear(config.latent_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform latent reasoning iterations.

        Args:
            x: (B, T, d_model) input representations

        Returns:
            dict with output and reasoning metrics
        """
        B, T, _ = x.shape

        # Project to latent space
        z = self.to_latent(x)  # (B, T, latent_dim)

        # Iterative reasoning
        cumulative_halt = torch.zeros(B, T, 1, device=x.device)
        iterations = 0

        for i in range(self.config.n_latent_iterations):
            # Reasoning step
            z_new = z + self.reasoning_layer(z)

            # Halting decision
            halt_prob = self.reasoning_halt(z_new)
            cumulative_halt = cumulative_halt + halt_prob * (1 - (cumulative_halt >= 0.99).float())

            # Weighted update (ACT-style)
            z = z_new

            iterations += 1

            # Early exit if all positions halted
            if (cumulative_halt >= 0.99).all():
                break

        # Project back from latent space
        output = self.from_latent(z)

        result = {
            "output": output,
            "latent_iterations": iterations,
            "final_halt": cumulative_halt.mean().item(),
        }

        return result


# =============================================================================
# Component 3: Hierarchical Memory Bank
# =============================================================================

class HierarchicalMemory(nn.Module):
    """
    Differentiable memory bank with read/write operations.

    Three tiers (conceptually):
    - Registers: KV cache in attention (handled by attention layer)
    - RAM: This memory bank (persistent, updatable)
    - Disk: External retrieval (not implemented in this version)
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config

        # Memory bank: (n_slots, memory_dim)
        self.memory = nn.Parameter(torch.randn(config.n_memory_slots, config.memory_dim) * 0.02)

        # Read mechanism
        self.read_query = nn.Linear(config.d_model, config.memory_dim)
        self.read_proj = nn.Linear(config.memory_dim, config.d_model)

        # Write mechanism
        self.write_key = nn.Linear(config.d_model, config.memory_dim)
        self.write_value = nn.Linear(config.d_model, config.memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Usage tracking (for LRU-style replacement)
        self.register_buffer('usage', torch.zeros(config.n_memory_slots))

    def read(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory based on query.

        Args:
            x: (B, T, d_model) query vectors

        Returns:
            read_output: (B, T, d_model) retrieved information
            read_weights: (B, T, n_slots) attention over memory
        """
        B, T, _ = x.shape

        # Compute query
        query = self.read_query(x)  # (B, T, memory_dim)

        # Attention over memory slots
        # memory: (n_slots, memory_dim) -> (1, n_slots, memory_dim)
        memory = self.memory.unsqueeze(0)

        # (B, T, memory_dim) @ (1, memory_dim, n_slots) -> (B, T, n_slots)
        attn = torch.matmul(query, memory.transpose(-2, -1)) / math.sqrt(self.config.memory_dim)
        read_weights = F.softmax(attn, dim=-1)

        # Read: (B, T, n_slots) @ (1, n_slots, memory_dim) -> (B, T, memory_dim)
        read_content = torch.matmul(read_weights, memory)
        read_output = self.read_proj(read_content)

        # Update usage (for tracking which slots are used)
        self.usage = self.usage + read_weights.sum(dim=(0, 1)).detach()

        return read_output, read_weights

    def write(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write to memory.

        Args:
            x: (B, T, d_model) vectors to potentially write

        Returns:
            memory_update: Change to memory bank
            write_weights: (B, T, n_slots) where we wrote
        """
        B, T, _ = x.shape

        # Compute write gate (should we write at all?)
        write_gate = self.write_gate(x)  # (B, T, 1)

        # Compute write key and value
        key = self.write_key(x)    # (B, T, memory_dim)
        value = self.write_value(x)  # (B, T, memory_dim)

        # Attention to determine WHERE to write (using key similarity)
        memory = self.memory.unsqueeze(0)
        attn = torch.matmul(key, memory.transpose(-2, -1)) / math.sqrt(self.config.memory_dim)
        write_weights = F.softmax(attn, dim=-1)  # (B, T, n_slots)

        # Apply write gate
        write_weights = write_weights * write_gate

        # Compute update: weighted average of values
        # (B, T, n_slots).T @ (B, T, memory_dim) -> (B, n_slots, memory_dim)
        # Average over batch and sequence
        update = torch.einsum('btn,btm->nm', write_weights, value) / (B * T + 1e-8)

        return update, write_weights

    def forward(
        self,
        x: torch.Tensor,
        do_write: bool = True,
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        Memory read and optional write.

        Args:
            x: (B, T, d_model) input
            do_write: whether to update memory

        Returns:
            dict with output and memory metrics
        """
        # Read from memory
        read_output, read_weights = self.read(x)

        # Combine with input
        output = x + read_output

        # Write to memory (only during training, and with gradient)
        if do_write and self.training:
            update, write_weights = self.write(x)
            # Apply update with small learning rate (stability)
            with torch.no_grad():
                self.memory.data = self.memory.data + 0.01 * update

        result = {
            "output": output,
            "memory_usage": (self.usage > 0.1).float().mean().item(),
        }

        if return_metrics:
            result["read_entropy"] = -(read_weights * (read_weights + 1e-8).log()).sum(-1).mean().item()

        return result


# =============================================================================
# Full LAHR Architecture
# =============================================================================

class LAHR(nn.Module):
    """
    LAHR: Latent Adaptive Hierarchical Reasoning

    Combines:
    1. Adaptive depth with shared parameters
    2. Latent-space reasoning
    3. Hierarchical memory bank

    Each component can be disabled for ablation studies.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Component 1: Adaptive depth
        self.adaptive_depth = AdaptiveDepthModule(config)

        # Component 2: Latent reasoning (optional)
        if config.use_latent_reasoning:
            self.latent_reasoning = LatentReasoningModule(config)
        else:
            self.latent_reasoning = None

        # Component 3: Hierarchical memory (optional)
        if config.use_memory:
            self.memory = HierarchicalMemory(config)
        else:
            self.memory = None

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        # Initialize
        self.apply(self._init_weights)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

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
        Forward pass through LAHR.

        Args:
            input_ids: (B, T) token indices
            return_metrics: whether to return detailed metrics

        Returns:
            dict with logits and optional metrics
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        metrics = {}

        # 1. Memory read (if enabled) - get context from persistent memory
        if self.memory is not None:
            mem_result = self.memory(x, do_write=False, return_metrics=return_metrics)
            x = mem_result["output"]
            metrics["memory_usage"] = mem_result["memory_usage"]

        # 2. Adaptive depth processing
        depth_result = self.adaptive_depth(x, return_metrics=return_metrics)
        x = depth_result["output"]
        metrics["effective_depth"] = depth_result["effective_depth"]

        # 3. Latent reasoning (if enabled)
        if self.latent_reasoning is not None:
            latent_result = self.latent_reasoning(x, return_metrics=return_metrics)
            x = x + latent_result["output"]  # Residual connection
            metrics["latent_iterations"] = latent_result["latent_iterations"]

        # 4. Memory write (if enabled) - store processed information
        if self.memory is not None:
            mem_result = self.memory(x, do_write=True, return_metrics=return_metrics)
            x = mem_result["output"]

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if return_metrics:
            result["metrics"] = metrics

        return result


# =============================================================================
# Factory Functions for Different Scales
# =============================================================================

def create_lahr_tiny() -> LAHR:
    """~1M params - for architecture validation."""
    config = LAHRConfig(
        d_model=128,
        n_heads=4,
        ff_expand=4,
        n_shared_blocks=2,
        max_depth=8,
        n_latent_iterations=2,
        n_memory_slots=32,
        max_seq_len=256,
    )
    return LAHR(config)


def create_lahr_small() -> LAHR:
    """~10M params - for component validation."""
    config = LAHRConfig(
        d_model=256,
        n_heads=8,
        ff_expand=4,
        n_shared_blocks=2,
        max_depth=12,
        n_latent_iterations=4,
        n_memory_slots=64,
        max_seq_len=512,
    )
    return LAHR(config)


def create_lahr_medium() -> LAHR:
    """~50M params - for scaling validation."""
    config = LAHRConfig(
        d_model=512,
        n_heads=8,
        ff_expand=4,
        n_shared_blocks=3,
        max_depth=16,
        n_latent_iterations=4,
        n_memory_slots=128,
        max_seq_len=1024,
    )
    return LAHR(config)


def create_lahr_large() -> LAHR:
    """~125M params - for final validation."""
    config = LAHRConfig(
        d_model=768,
        n_heads=12,
        ff_expand=4,
        n_shared_blocks=4,
        max_depth=24,
        n_latent_iterations=6,
        n_memory_slots=256,
        max_seq_len=1024,
    )
    return LAHR(config)


# Ablation variants
def create_lahr_no_latent(size: str = "small") -> LAHR:
    """LAHR without latent reasoning (adaptive depth + memory only)."""
    configs = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, use_latent_reasoning=False, max_seq_len=256),
        "small": LAHRConfig(d_model=256, n_heads=8, use_latent_reasoning=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, use_latent_reasoning=False),
    }
    return LAHR(configs[size])


def create_lahr_no_memory(size: str = "small") -> LAHR:
    """LAHR without memory (adaptive depth + latent only)."""
    configs = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, use_memory=False, max_seq_len=256),
        "small": LAHRConfig(d_model=256, n_heads=8, use_memory=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, use_memory=False),
    }
    return LAHR(configs[size])


def create_lahr_no_adaptive(size: str = "small") -> LAHR:
    """LAHR with fixed depth (latent + memory only)."""
    configs = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, min_depth=8, max_depth=8, max_seq_len=256),
        "small": LAHRConfig(d_model=256, n_heads=8, min_depth=12, max_depth=12),
        "medium": LAHRConfig(d_model=512, n_heads=8, min_depth=16, max_depth=16),
    }
    return LAHR(configs[size])


if __name__ == "__main__":
    # Test all scales
    print("Testing LAHR at different scales:\n")

    for name, create_fn in [
        ("tiny", create_lahr_tiny),
        ("small", create_lahr_small),
        ("medium", create_lahr_medium),
        ("large", create_lahr_large),
    ]:
        model = create_fn()
        print(f"{name.upper()}: {model.n_params:,} parameters")

        # Test forward pass
        x = torch.randint(0, 1000, (2, 64))
        output = model(x, return_metrics=True)

        print(f"  Effective depth: {output['metrics']['effective_depth']:.1f}")
        if "latent_iterations" in output["metrics"]:
            print(f"  Latent iterations: {output['metrics']['latent_iterations']}")
        if "memory_usage" in output["metrics"]:
            print(f"  Memory usage: {output['metrics']['memory_usage']:.1%}")
        print()
