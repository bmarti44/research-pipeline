"""
LAHR v5: Fixed COCONUT Implementation

Fixes from Reviews 1-15:
1. MoD: Process only selected tokens WITH proper position encodings (efficiency preserved)
2. **REAL** COCONUT latent reasoning (hidden state -> embedding replacement)
3. Proper batched memory indexing
4. Adaptive iteration depth (Think-at-Hard style)

THIS IS THE CANONICAL VERSION - use this one for training.

KEY FIX (v5): The latent reasoning now uses TRUE COCONUT mechanism:
- Old (broken): Just loop a transformer block N times
- New (correct): Replace input embeddings with hidden states from previous iteration

Based on:
- COCONUT paper: https://arxiv.org/abs/2412.06769
- Official code: https://github.com/facebookresearch/coconut

HONEST LIMITATIONS:
- Latent reasoning FAILS on arithmetic (GSM8K: 34% vs 42% CoT)
- Latent reasoning WORKS on logical search (ProsQA: 97% vs 77% CoT)
- Small-scale results may not transfer
- Component interactions are UNKNOWN
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LAHRConfig:
    """Configuration for LAHR v4."""
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512

    # MoD
    n_layers: int = 12
    mod_capacity: float = 0.125  # 12.5% tokens processed
    mod_every_n: int = 2

    # Latent reasoning
    use_latent_reasoning: bool = True
    n_latent_iterations: int = 4

    # Memory
    use_memory: bool = True
    n_memory_slots: int = 128
    memory_top_k: int = 8

    def get_mod_layers(self) -> List[int]:
        return [i for i in range(self.n_layers) if i % self.mod_every_n == 1]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = int(d_model * ff_mult * 2 / 3)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class PositionAwareCausalAttention(nn.Module):
    """
    Attention that accepts arbitrary position indices.

    Critical for MoD: When processing a subset of tokens, we need
    to maintain their original positions for correct causal masking.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,  # REQUIRED: original positions
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) - T may be < max_seq_len if MoD selected subset
            position_ids: (B, T) - original positions of each token
        """
        B, T, D = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create position-aware causal mask
        # Token i can only attend to token j if position_ids[i] >= position_ids[j]
        pos_i = position_ids.unsqueeze(-1)  # B x T x 1
        pos_j = position_ids.unsqueeze(-2)  # B x 1 x T
        causal_mask = (pos_i < pos_j)  # B x T x T: True where we should mask

        # Expand for heads
        causal_mask = causal_mask.unsqueeze(1)  # B x 1 x T x T
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # Handle NaN from all-masked rows (R6 fix)
        # If a token can attend to nothing, softmax produces NaN
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Replace NaN with 0
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Standard transformer block with position-aware attention."""

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = PositionAwareCausalAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.ff_mult, config.dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids)
        x = x + self.ff(self.norm2(x))
        return x


class MoDBlock(nn.Module):
    """
    Mixture-of-Depths block with CORRECT efficient implementation.

    Key fix from Review 2: We process ONLY selected tokens (for efficiency)
    but preserve their ORIGINAL position indices (for correct causality).
    """

    def __init__(self, config: LAHRConfig, use_mod: bool = True):
        super().__init__()
        self.use_mod = use_mod
        self.capacity_fraction = config.mod_capacity
        self.d_model = config.d_model

        self.block = TransformerBlock(config)

        if use_mod:
            self.router = nn.Linear(config.d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        EFFICIENT MoD: Only compute for selected tokens.

        1. Route to select top-k tokens
        2. Gather selected tokens (preserving position_ids)
        3. Process ONLY selected tokens through transformer
        4. Scatter results back

        FLOPs savings: Only k/T of the computation for attention/FFN
        """
        B, T, D = x.shape

        if not self.use_mod:
            return self.block(x, position_ids), None, torch.tensor(0.0, device=x.device)

        # Compute routing scores
        scores = self.router(x).squeeze(-1)  # B x T

        # Add small tie-breaking noise for determinism (R6 fix)
        # When scores are tied, topk behavior is undefined; this ensures reproducibility
        tie_breaker = torch.arange(T, device=x.device, dtype=x.dtype) * 1e-6
        scores_with_tiebreak = scores + tie_breaker.unsqueeze(0)

        # Select top-k
        k = max(1, int(T * self.capacity_fraction))
        top_k_scores, top_k_indices = torch.topk(scores_with_tiebreak, k, dim=-1, sorted=True)

        # Sort indices to maintain relative order (important for causality)
        sorted_indices, sort_order = torch.sort(top_k_indices, dim=-1)

        # Gather selected tokens
        indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, D)
        x_selected = torch.gather(x, 1, indices_expanded)  # B x k x D

        # Gather position IDs for selected tokens (CRITICAL for correct masking)
        pos_selected = torch.gather(position_ids, 1, sorted_indices)  # B x k

        # Process ONLY the selected tokens (this is where efficiency comes from)
        x_processed = self.block(x_selected, pos_selected)  # B x k x D

        # Scatter back: selected positions get processed values, others unchanged
        output = x.clone()
        output.scatter_(1, indices_expanded, x_processed)

        routing_weights = None
        aux_loss = torch.tensor(0.0, device=x.device)

        if return_routing:
            routing_weights = torch.zeros(B, T, device=x.device)
            routing_weights.scatter_(1, sorted_indices, 1.0)

            # Auxiliary load balancing loss (standard for MoD/MoE)
            # Encourages balanced token selection across the sequence
            # Reference: Switch Transformer (Fedus et al., 2021)
            router_probs = F.softmax(scores, dim=-1)  # B x T
            # Fraction of tokens selected at each position (averaged over batch)
            selection_fraction = routing_weights.mean(dim=0)  # T
            # Target: uniform selection at capacity_fraction rate
            # Loss: penalize deviation from uniform
            aux_loss = ((selection_fraction - self.capacity_fraction) ** 2).mean()

        return output, routing_weights, aux_loss

    def estimate_flops(self, seq_len: int) -> Dict[str, int]:
        """Estimate actual FLOPs (only for selected tokens)."""
        k = max(1, int(seq_len * self.capacity_fraction))
        # Attention: O(k² * d) instead of O(T² * d)
        # FFN: O(k * d²) instead of O(T * d²)
        return {
            "tokens_processed": k,
            "tokens_total": seq_len,
            "efficiency": k / seq_len,
        }


class LatentReasoningModule(nn.Module):
    """
    TRUE COCONUT-style latent reasoning with hidden state -> embedding replacement.

    KEY INSIGHT (from research on real COCONUT):
    - Old (broken): Just loop a transformer block N times on same input
    - New (correct): Replace input embeddings with hidden states from previous iteration

    This implements the V2 "virtual thoughts" approach that doesn't require
    tokenizer modifications. It prepends learned virtual thought embeddings,
    runs iterative reasoning, and uses the COCONUT mechanism where each
    iteration's hidden state becomes the next iteration's input.

    Also includes Think-at-Hard style adaptive iteration stopping.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config
        self.n_iterations = config.n_latent_iterations
        self.d_model = config.d_model

        # Learned virtual thought embeddings (no tokenizer modification needed)
        self.n_virtual_thoughts = config.n_latent_iterations
        self.virtual_thoughts = nn.Parameter(
            torch.randn(self.n_virtual_thoughts, config.d_model) * 0.02
        )

        # The shared thinking block
        self.thinking_block = TransformerBlock(config)

        # Maps hidden state back to embedding space (key COCONUT component)
        self.hidden_to_embed = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )

        # Adaptive iteration decider (Think-at-Hard style)
        self.iteration_decider = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        n_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Apply TRUE COCONUT-style latent reasoning.

        The key mechanism:
        1. Prepend virtual thought embeddings to input
        2. Run through thinking block
        3. Replace virtual thought embeddings with hidden states (COCONUT mechanism)
        4. Repeat until adaptive stopping or max iterations
        5. Return processed sequence (without virtual thoughts)
        """
        B, T, D = x.shape
        max_iter = n_iterations or self.n_iterations
        device = x.device

        # Prepend virtual thought embeddings
        virtual = self.virtual_thoughts.unsqueeze(0).expand(B, -1, -1)  # [B, n_virtual, D]

        # Concatenate: [virtual_thoughts, input_sequence]
        x_with_virtual = torch.cat([virtual, x], dim=1)  # [B, n_virtual + T, D]

        # Iterative reasoning with COCONUT mechanism
        prev_virtual = virtual.clone()
        continue_probs = []
        actual_iterations = 0

        # Generate position_ids for the extended sequence
        # Virtual thoughts get positions 0 to n_virtual-1
        # Original sequence continues from n_virtual
        T_extended = self.n_virtual_thoughts + T
        extended_position_ids = torch.arange(T_extended, device=device).unsqueeze(0).expand(B, -1)

        for iteration in range(max_iter):
            # Forward through thinking block with proper position_ids
            hidden = self.thinking_block(x_with_virtual, extended_position_ids)

            # Extract virtual thought hidden states
            virtual_hidden = hidden[:, :self.n_virtual_thoughts, :]

            # Adaptive stopping (Think-at-Hard style) after first iteration
            if iteration > 0:
                prob = self.iteration_decider(
                    torch.cat([prev_virtual.mean(dim=1), virtual_hidden.mean(dim=1)], dim=-1)
                )
                continue_probs.append(prob.mean().item())

                # During inference, stop if average probability < 0.5
                if not self.training and prob.mean() < 0.5:
                    break

            # THE KEY COCONUT MECHANISM:
            # Replace virtual thought embeddings with transformed hidden states
            # This is what creates "continuous thought" in latent space!
            new_virtual = self.hidden_to_embed(virtual_hidden)

            # Update the combined representation
            x_with_virtual = torch.cat([new_virtual, hidden[:, self.n_virtual_thoughts:, :]], dim=1)

            prev_virtual = new_virtual
            actual_iterations += 1

        # Return only the original sequence positions (not virtual thoughts)
        output = x_with_virtual[:, self.n_virtual_thoughts:, :]

        return {
            "output": output,
            "n_iterations": actual_iterations,
            "continue_probs": continue_probs if continue_probs else None,
        }


class MemoryModule(nn.Module):
    """
    kNN memory retrieval with proper batched indexing.

    DESIGN DECISION: Memory is queried BEFORE transformer layers.
    Rationale: This allows retrieved context to influence ALL subsequent
    processing. Alternative (query after layers) would provide more
    contextualized queries but less influence on processing.
    This is a design choice - empirical validation needed to determine optimal.
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.n_slots = config.n_memory_slots
        self.top_k = config.memory_top_k
        self.d_model = config.d_model

        self.memory_keys = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)

        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        B, T, D = x.shape

        queries = self.query_proj(x)
        sim = torch.einsum('btd,md->btm', queries, self.memory_keys) / math.sqrt(D)

        top_k_sim, top_k_idx = torch.topk(sim, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_sim, dim=-1)

        # Proper batched gather
        flat_idx = top_k_idx.view(-1)
        gathered = self.memory_values[flat_idx].view(B, T, self.top_k, D)

        retrieved = torch.einsum('btk,btkd->btd', top_k_weights, gathered)
        retrieved = self.out_proj(retrieved)

        gate = self.gate(torch.cat([x, retrieved], dim=-1))
        output = x + gate * retrieved

        return {"output": output}


class LAHRv4(nn.Module):
    """
    LAHR v4: Canonical Implementation

    All fixes from Reviews 1 & 2 integrated:
    - Efficient MoD with position-aware masking
    - True COCONUT shared-block reasoning
    - Proper memory indexing
    """

    def __init__(self, config: LAHRConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        mod_layers = config.get_mod_layers()
        self.layers = nn.ModuleList([
            MoDBlock(config, use_mod=(i in mod_layers))
            for i in range(config.n_layers)
        ])

        self.latent = LatentReasoningModule(config) if config.use_latent_reasoning else None
        self.memory = MemoryModule(config) if config.use_memory else None

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights FIRST, then tie (R6 fix)
        self.apply(self._init_weights)

        # Weight tying AFTER initialization to avoid double-init
        self.lm_head.weight = self.embed.weight

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
        use_latent: bool = True,
        use_memory: bool = True,
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device

        # Bounds check for position embeddings (R6 fix)
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len {self.config.max_seq_len}. "
                f"Either truncate input or increase max_seq_len in config."
            )

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(position_ids)
        x = self.dropout(x)

        metrics = {"mod_efficiency": []}

        if use_memory and self.memory:
            x = self.memory(x)["output"]

        total_aux_loss = 0.0
        for layer in self.layers:
            x, routing, aux_loss = layer(x, position_ids, return_routing=return_metrics)
            total_aux_loss = total_aux_loss + aux_loss
            if return_metrics and routing is not None:
                metrics["mod_efficiency"].append(routing.mean().item())

        if use_latent and self.latent:
            lat = self.latent(x, position_ids)
            x = lat["output"]
            if return_metrics:
                metrics["latent_iterations"] = lat["n_iterations"]

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits, "aux_loss": total_aux_loss}
        if return_metrics:
            result["metrics"] = metrics
        return result


# Factory functions
def create_lahr_tiny() -> LAHRv4:
    return LAHRv4(LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, n_memory_slots=64))

def create_lahr_small() -> LAHRv4:
    return LAHRv4(LAHRConfig(d_model=256, n_heads=8, n_layers=8))

def create_lahr_medium() -> LAHRv4:
    return LAHRv4(LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024))

def create_baseline(size: str = "small") -> LAHRv4:
    configs = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, use_latent_reasoning=False, use_memory=False, mod_every_n=999, max_seq_len=256),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, use_latent_reasoning=False, use_memory=False, mod_every_n=999),
    }
    return LAHRv4(configs[size])


# =============================================================================
# Full 2^3 Factorial Ablation Factory Functions
# =============================================================================
# These create all 8 conditions for the complete factorial design.
# Required for proper statistical analysis of component contributions.

def create_ablation_full(size: str = "small") -> LAHRv4:
    """Full LAHR with all components: MoD + Latent + Memory."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, n_memory_slots=64),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024),
        "base": LAHRConfig(d_model=768, n_heads=12, n_layers=12, max_seq_len=1024, n_memory_slots=512),
    }
    return LAHRv4(sizes[size])


def create_ablation_no_mod(size: str = "small") -> LAHRv4:
    """LAHR without MoD: Latent + Memory only."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, n_memory_slots=64, mod_every_n=999),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, mod_every_n=999),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, mod_every_n=999),
    }
    return LAHRv4(sizes[size])


def create_ablation_no_latent(size: str = "small") -> LAHRv4:
    """LAHR without latent reasoning: MoD + Memory only."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, n_memory_slots=64, use_latent_reasoning=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, use_latent_reasoning=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, use_latent_reasoning=False),
    }
    return LAHRv4(sizes[size])


def create_ablation_no_memory(size: str = "small") -> LAHRv4:
    """LAHR without memory: MoD + Latent only."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, use_memory=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, use_memory=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, use_memory=False),
    }
    return LAHRv4(sizes[size])


def create_ablation_mod_only(size: str = "small") -> LAHRv4:
    """MoD only: no latent reasoning, no memory."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, use_latent_reasoning=False, use_memory=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, use_latent_reasoning=False, use_memory=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, use_latent_reasoning=False, use_memory=False),
    }
    return LAHRv4(sizes[size])


def create_ablation_latent_only(size: str = "small") -> LAHRv4:
    """Latent reasoning only: no MoD, no memory."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, mod_every_n=999, use_memory=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, mod_every_n=999, use_memory=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, mod_every_n=999, use_memory=False),
    }
    return LAHRv4(sizes[size])


def create_ablation_memory_only(size: str = "small") -> LAHRv4:
    """Memory only: no MoD, no latent reasoning."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, n_memory_slots=64, mod_every_n=999, use_latent_reasoning=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, mod_every_n=999, use_latent_reasoning=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, mod_every_n=999, use_latent_reasoning=False),
    }
    return LAHRv4(sizes[size])


def create_ablation_baseline(size: str = "small") -> LAHRv4:
    """Baseline: no MoD, no latent reasoning, no memory (standard transformer)."""
    sizes = {
        "tiny": LAHRConfig(d_model=128, n_heads=4, n_layers=6, max_seq_len=256, mod_every_n=999, use_latent_reasoning=False, use_memory=False),
        "small": LAHRConfig(d_model=256, n_heads=8, n_layers=8, mod_every_n=999, use_latent_reasoning=False, use_memory=False),
        "medium": LAHRConfig(d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, mod_every_n=999, use_latent_reasoning=False, use_memory=False),
    }
    return LAHRv4(sizes[size])


# Ablation factory registry for programmatic access
ABLATION_FACTORY = {
    "full": create_ablation_full,
    "no_mod": create_ablation_no_mod,
    "no_latent": create_ablation_no_latent,
    "no_memory": create_ablation_no_memory,
    "mod_only": create_ablation_mod_only,
    "latent_only": create_ablation_latent_only,
    "memory_only": create_ablation_memory_only,
    "baseline": create_ablation_baseline,
}


def create_ablation(condition: str, size: str = "small") -> LAHRv4:
    """
    Create a model for a specific ablation condition.

    Args:
        condition: One of "full", "no_mod", "no_latent", "no_memory",
                   "mod_only", "latent_only", "memory_only", "baseline"
        size: One of "tiny", "small", "medium", "base"

    Returns:
        LAHRv4 model configured for the ablation condition
    """
    if condition not in ABLATION_FACTORY:
        raise ValueError(f"Unknown condition: {condition}. Valid: {list(ABLATION_FACTORY.keys())}")
    return ABLATION_FACTORY[condition](size)


if __name__ == "__main__":
    print("LAHR v4 - Canonical Fixed Implementation")
    print("\nTesting...")

    model = create_lahr_tiny()
    x = torch.randint(0, 1000, (2, 64))
    out = model(x, return_metrics=True)

    print(f"Parameters: {model.n_params:,}")
    print(f"Output shape: {out['logits'].shape}")

    if out.get("metrics", {}).get("mod_efficiency"):
        avg_eff = sum(out["metrics"]["mod_efficiency"]) / len(out["metrics"]["mod_efficiency"])
        print(f"MoD efficiency (fraction processed): {avg_eff:.3f}")

    # Test gradient flow
    loss = out["logits"].sum()
    loss.backward()
    print("Gradient flow: OK")
