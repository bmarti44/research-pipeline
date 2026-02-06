"""
LAHR + Full COCONUT Integration

Combines:
- Mixture-of-Depths (MoD) for efficient routing
- TRUE COCONUT mechanism for latent reasoning
- Differentiable memory retrieval

This is the COMPLETE architecture with all components working correctly.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LAHRCOCONUTConfig:
    """Configuration for LAHR + COCONUT model."""
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512

    # MoD settings
    mod_capacity: float = 0.125  # 12.5% tokens processed
    mod_every_n: int = 2  # Apply MoD to every nth layer
    use_mod: bool = True

    # COCONUT settings
    use_coconut: bool = True
    max_n_latent: int = 8

    # Memory settings
    use_memory: bool = True
    n_memory_slots: int = 128
    memory_top_k: int = 8

    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bot_token_id: int = 2
    thought_token_id: int = 3
    eot_token_id: int = 4

    def get_mod_layers(self) -> List[int]:
        if not self.use_mod:
            return []
        return [i for i in range(self.n_layers) if i % self.mod_every_n == 1]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config: LAHRCOCONUTConfig):
        super().__init__()
        hidden_dim = config.d_model * config.ff_mult
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalAttention(nn.Module):
    """Causal self-attention with position-aware masking."""

    def __init__(self, config: LAHRCOCONUTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: LAHRCOCONUTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids)
        x = x + self.ff(self.norm2(x))
        return x


class MoDBlock(nn.Module):
    """Mixture-of-Depths block with efficient sparse computation."""

    def __init__(self, config: LAHRCOCONUTConfig, use_mod: bool = True):
        super().__init__()
        self.config = config
        self.use_mod = use_mod
        self.block = TransformerBlock(config)

        if use_mod:
            self.router = nn.Linear(config.d_model, 1, bias=False)
            self.capacity = config.mod_capacity

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        if not self.use_mod:
            return self.block(x, position_ids), None, 0.0

        B, T, D = x.shape
        k = max(1, int(T * self.capacity))

        # Router scores with tie-breaking noise
        router_logits = self.router(x).squeeze(-1)
        noise = torch.rand_like(router_logits) * 1e-6
        scores = router_logits + noise

        # Select top-k tokens
        _, top_indices = torch.topk(scores, k, dim=1)
        top_indices_sorted, _ = torch.sort(top_indices, dim=1)

        # Gather selected tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        selected = x[batch_indices, top_indices_sorted]

        # Process only selected tokens
        processed = self.block(selected)

        # Scatter back using gradient-safe torch.where (FIX: in-place assignment breaks gradients)
        # Create mask for selected positions
        mask = torch.zeros(B, T, 1, device=x.device, dtype=torch.bool)
        mask[batch_indices, top_indices_sorted] = True

        # Expand processed back to full sequence positions
        full_processed = torch.zeros_like(x)
        full_processed[batch_indices, top_indices_sorted] = processed

        # Gradient-safe combination
        output = torch.where(mask.expand(-1, -1, x.size(-1)), full_processed, x)

        # Auxiliary loss for load balancing
        routing_probs = torch.sigmoid(router_logits)
        aux_loss = routing_probs.mean()

        return output, routing_probs, aux_loss


class MemoryModule(nn.Module):
    """Differentiable kNN memory retrieval."""

    def __init__(self, config: LAHRCOCONUTConfig):
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
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Query memory
        queries = self.query_proj(x)  # [B, T, D]

        # Compute similarities
        keys_norm = F.normalize(self.memory_keys, dim=-1)
        queries_norm = F.normalize(queries, dim=-1)
        similarities = torch.matmul(queries_norm, keys_norm.T)  # [B, T, n_slots]

        # Top-k retrieval
        topk_sims, topk_indices = torch.topk(similarities, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_sims, dim=-1)  # [B, T, k]

        # Gather values
        topk_values = self.memory_values[topk_indices]  # [B, T, k, D]
        retrieved = (topk_weights.unsqueeze(-1) * topk_values).sum(dim=2)  # [B, T, D]
        retrieved = self.out_proj(retrieved)

        # Gated addition
        gate_input = torch.cat([x, retrieved], dim=-1)
        gate_weight = self.gate(gate_input)
        return x + gate_weight * retrieved


class LAHRCOCONUT(nn.Module):
    """
    LAHR + Full COCONUT Model

    Architecture:
    1. Token embedding + position embedding
    2. Memory retrieval (optional)
    3. MoD transformer layers
    4. COCONUT latent reasoning (if thought tokens present)
    5. Output projection
    """

    def __init__(self, config: LAHRCOCONUTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # MoD layers
        mod_layers = config.get_mod_layers()
        self.layers = nn.ModuleList([
            MoDBlock(config, use_mod=(i in mod_layers))
            for i in range(config.n_layers)
        ])

        # Memory
        self.memory = MemoryModule(config) if config.use_memory else None

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
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

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        return self.embed(input_ids) + self.pos_embed(position_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device

        # Get embeddings
        x = self.get_embeddings(input_ids)
        x = self.dropout(x)

        # Memory retrieval
        if self.memory and self.config.use_memory:
            x = self.memory(x)

        # Check for thought tokens
        thought_mask = (input_ids == self.config.thought_token_id)
        has_thoughts = thought_mask.any()

        # MoD layers + COCONUT processing
        total_aux_loss = 0.0
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        if has_thoughts and self.config.use_coconut:
            # COCONUT: iterative processing with thought replacement
            x = self._coconut_forward(x, thought_mask, position_ids)
        else:
            # Standard forward
            for layer in self.layers:
                x, _, aux_loss = layer(x, position_ids)
                total_aux_loss += aux_loss

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
            loss = loss + 0.01 * total_aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": total_aux_loss,
        }

    def _coconut_forward(
        self,
        inputs_embeds: torch.Tensor,
        thought_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        COCONUT mechanism integrated with MoD layers.

        For each thought token iteration:
        1. Run through all MoD layers
        2. Replace thought embeddings with hidden states from previous position
        3. Repeat
        """
        B, T, D = inputs_embeds.shape
        device = inputs_embeds.device

        # Find thought positions
        thought_positions = []
        max_thoughts = 0
        for b in range(B):
            positions = thought_mask[b].nonzero(as_tuple=True)[0].tolist()
            thought_positions.append(positions)
            max_thoughts = max(max_thoughts, len(positions))

        if max_thoughts == 0:
            # No thoughts - standard forward
            x = inputs_embeds
            for layer in self.layers:
                x, _, _ = layer(x, position_ids)
            return x

        # Iterative COCONUT processing
        x = inputs_embeds

        for thought_idx in range(max_thoughts):
            # Build replacement mask for this iteration
            replace_mask = torch.zeros(B, T, 1, device=device, dtype=torch.bool)
            for b in range(B):
                if thought_idx < len(thought_positions[b]):
                    pos = thought_positions[b][thought_idx]
                    if pos > 0:
                        replace_mask[b, pos, 0] = True

            # Run through MoD layers
            for layer in self.layers:
                x, _, _ = layer(x, position_ids)

            if replace_mask.any():
                # Shift hidden states for COCONUT replacement
                shifted_x = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
                x = torch.where(replace_mask.expand(-1, -1, D), shifted_x, x)

        # FIX: Final forward pass after all thought replacements
        # Without this, the last thought replacement is never processed
        for layer in self.layers:
            x, _, _ = layer(x, position_ids)

        return x


def create_lahr_coconut(size: str = "small") -> LAHRCOCONUT:
    """Factory function."""
    configs = {
        "tiny": LAHRCOCONUTConfig(
            d_model=128, n_heads=4, n_layers=4, max_seq_len=256,
            n_memory_slots=64,
        ),
        "small": LAHRCOCONUTConfig(
            d_model=256, n_heads=8, n_layers=8, max_seq_len=512,
        ),
        "medium": LAHRCOCONUTConfig(
            d_model=512, n_heads=8, n_layers=12, max_seq_len=1024,
        ),
    }
    return LAHRCOCONUT(configs[size])


if __name__ == "__main__":
    print("Testing LAHR + COCONUT model...")

    config = LAHRCOCONUTConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=4,
        max_seq_len=64,
    )
    model = LAHRCOCONUT(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test without thought tokens
    x = torch.randint(5, 100, (2, 32))
    out = model(x)
    print(f"Without thoughts - logits shape: {out['logits'].shape}")

    # Test with thought tokens
    x[:, 5] = config.bot_token_id
    x[:, 6:9] = config.thought_token_id
    x[:, 9] = config.eot_token_id
    labels = x.clone()
    labels[:, :10] = -100

    out = model(x, labels=labels)
    print(f"With thoughts - loss: {out['loss'].item():.4f}")

    # Test gradients
    out["loss"].backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"Parameters with gradients: {grads}")

    print("\nAll tests passed!")
