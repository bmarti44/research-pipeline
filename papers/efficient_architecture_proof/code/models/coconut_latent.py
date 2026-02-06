"""
Real COCONUT-style Latent Reasoning Implementation

Based on: "Training Large Language Models to Reason in a Continuous Latent Space"
(Hao et al., 2024) - https://arxiv.org/abs/2412.06769
Official code: https://github.com/facebookresearch/coconut

Key insight from research:
- Our BROKEN approach: Just loop a transformer block N times
- REAL COCONUT: Replace input embeddings with hidden states from previous iteration

This module implements the TRUE COCONUT mechanism where:
1. Special <thought> tokens act as placeholders in the input
2. After each forward pass, the hidden state at position i-1 replaces
   the embedding at thought position i
3. This allows "continuous thought" in latent space without decoding to tokens

Also includes Think-at-Hard style adaptive iteration (optional).
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class COCONUTConfig:
    """Configuration for COCONUT-style latent reasoning."""
    d_model: int = 256
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    # Special token IDs (must be added to tokenizer)
    bot_token_id: int = 50258  # <bot> - beginning of thought
    thought_token_id: int = 50259  # <thought> - continuous thought placeholder
    eot_token_id: int = 50260  # <eot> - end of thought

    # Latent reasoning params
    max_latent_iterations: int = 4
    use_adaptive_depth: bool = True  # Think-at-Hard style

    # Curriculum training
    curriculum_stage: int = 0  # Current training stage (0 = full CoT, k = k latent tokens)
    thoughts_per_stage: int = 1  # c_thought in paper


class TransformerBlock(nn.Module):
    """Standard transformer block for latent reasoning."""

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Attention
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

        # FFN (SwiGLU)
        d_ff = int(config.d_model * config.ff_mult * 2 / 3)
        self.w1 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w2 = nn.Linear(config.d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, config.d_model, bias=False)

        # Norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = residual + self.dropout(self.out(out))

        # FFN with residual (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        x = residual + self.dropout(x)

        return x


class AdaptiveIterationDecider(nn.Module):
    """
    Think-at-Hard style decider for adaptive latent iteration depth.

    Learns to predict whether continuing iteration would help.
    Trained separately after backbone training.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Takes concatenated representations from different "depths"
        # In practice, we use the hidden state at different iterations
        self.decider = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h_prev: torch.Tensor, h_curr: torch.Tensor) -> torch.Tensor:
        """
        Predict whether to continue iterating.

        Args:
            h_prev: Hidden state from previous iteration [B, T, D]
            h_curr: Hidden state from current iteration [B, T, D]

        Returns:
            continue_prob: Probability of continuing [B, T, 1]
        """
        concat = torch.cat([h_prev, h_curr], dim=-1)
        return self.decider(concat)


class COCONUTLatentReasoning(nn.Module):
    """
    TRUE COCONUT-style latent reasoning.

    The key mechanism (from the paper):
    1. Input contains <thought> placeholder tokens
    2. After forward pass, hidden state at position i-1 REPLACES
       the embedding at thought position i
    3. This creates "continuous thought" that encodes reasoning
       without decoding to discrete tokens

    This is fundamentally different from our broken 4x loop approach!

    Training uses curriculum:
    - Stage 0: Full CoT supervision (no latent tokens)
    - Stage k: k thought tokens replace first k CoT steps
    """

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # The shared thinking block (applied iteratively)
        self.thinking_block = TransformerBlock(config)

        # Adapter: maps hidden state to embedding space
        # (hidden states and embeddings may have different distributions)
        self.hidden_to_embed = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )

        # Optional: adaptive iteration depth (Think-at-Hard style)
        if config.use_adaptive_depth:
            self.iteration_decider = AdaptiveIterationDecider(config.d_model)
        else:
            self.iteration_decider = None

        # Track special token IDs
        self.bot_token_id = config.bot_token_id
        self.thought_token_id = config.thought_token_id
        self.eot_token_id = config.eot_token_id

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Apply COCONUT-style latent reasoning.

        Args:
            inputs_embeds: Token embeddings [B, T, D]
            input_ids: Original token IDs [B, T] - needed to find thought positions
            attention_mask: Optional attention mask [B, T]
            max_iterations: Override max iterations (default: config.max_latent_iterations)

        Returns:
            Dict with:
                - output: Processed embeddings [B, T, D]
                - n_iterations: Number of iterations performed
                - continue_probs: Iteration decision probabilities (if adaptive)
        """
        B, T, D = inputs_embeds.shape
        max_iter = max_iterations or self.config.max_latent_iterations

        # Find thought token positions
        thought_mask = (input_ids == self.thought_token_id)
        n_thoughts = thought_mask.sum(dim=1).max().item()

        if n_thoughts == 0:
            # No thought tokens - return unchanged
            return {
                "output": inputs_embeds,
                "n_iterations": 0,
                "continue_probs": None,
            }

        # Working copy of embeddings
        x = inputs_embeds.clone()

        # Track iteration decisions for training the decider
        continue_probs = []
        prev_hidden = x.clone()

        # Iterative latent reasoning
        actual_iterations = 0
        for iteration in range(max_iter):
            # Forward pass through thinking block
            hidden = self.thinking_block(x, attention_mask)

            # Adaptive stopping (Think-at-Hard style)
            if self.iteration_decider is not None and iteration > 0:
                prob = self.iteration_decider(prev_hidden, hidden)
                continue_probs.append(prob)

                # During inference, stop if average probability < 0.5
                if not self.training and prob.mean() < 0.5:
                    break

            # THE KEY COCONUT MECHANISM:
            # Replace thought token embeddings with hidden states from previous position
            # This is what makes "continuous thought" work!

            # Get hidden states at positions BEFORE each thought token
            # thought_mask[b, i] = True means position i is a thought token
            # We want hidden[b, i-1] to become the new embedding at position i

            thought_indices = thought_mask.nonzero(as_tuple=False)  # [N, 2] - (batch_idx, position)

            if len(thought_indices) > 0:
                for idx in thought_indices:
                    b, pos = idx[0].item(), idx[1].item()
                    if pos > 0:  # Can't use position -1
                        # Map hidden state to embedding space and replace
                        new_embed = self.hidden_to_embed(hidden[b, pos - 1])
                        x[b, pos] = new_embed

            prev_hidden = hidden.clone()
            actual_iterations += 1

        return {
            "output": x,
            "n_iterations": actual_iterations,
            "continue_probs": torch.stack(continue_probs, dim=0) if continue_probs else None,
        }

    def prepare_curriculum_data(
        self,
        question_ids: torch.Tensor,
        cot_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        stage: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data for curriculum learning.

        At stage k, replace first k CoT steps with k thought tokens.

        Args:
            question_ids: Tokenized question [T_q]
            cot_ids: Tokenized chain-of-thought [T_cot]
            answer_ids: Tokenized answer [T_a]
            stage: Current curriculum stage (0 = full CoT, k = k thought tokens)

        Returns:
            input_ids: Combined input with thought tokens
            labels: Labels with thought positions masked (-100)
        """
        device = question_ids.device
        n_thoughts = stage * self.config.thoughts_per_stage

        if n_thoughts == 0:
            # Stage 0: Full CoT, no thought tokens
            input_ids = torch.cat([question_ids, cot_ids, answer_ids])
            labels = torch.cat([
                torch.full_like(question_ids, -100),  # Mask question
                cot_ids,  # Supervise CoT
                answer_ids,  # Supervise answer
            ])
        else:
            # Stage k: Replace first k CoT tokens with k thought tokens
            thought_tokens = torch.full((n_thoughts,), self.thought_token_id, device=device)
            remaining_cot = cot_ids[n_thoughts:] if n_thoughts < len(cot_ids) else torch.tensor([], device=device, dtype=cot_ids.dtype)

            input_ids = torch.cat([
                question_ids,
                torch.tensor([self.bot_token_id], device=device),  # <bot>
                thought_tokens,  # <thought> * n_thoughts
                torch.tensor([self.eot_token_id], device=device),  # <eot>
                remaining_cot,
                answer_ids,
            ])

            labels = torch.cat([
                torch.full((len(question_ids) + 1 + n_thoughts + 1,), -100, device=device),  # Mask q + bot + thoughts + eot
                remaining_cot if len(remaining_cot) > 0 else torch.tensor([], device=device, dtype=torch.long),  # Supervise remaining CoT
                answer_ids,  # Supervise answer
            ])

        return input_ids, labels


class COCONUTLatentReasoningV2(nn.Module):
    """
    Simplified COCONUT that works without special tokens.

    For use when you can't modify the tokenizer.
    Uses learned "virtual" thought tokens instead.
    """

    def __init__(self, config: COCONUTConfig, n_virtual_thoughts: int = 4):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_virtual_thoughts = n_virtual_thoughts

        # Learned virtual thought embeddings (don't need tokenizer modification)
        self.virtual_thoughts = nn.Parameter(torch.randn(n_virtual_thoughts, config.d_model) * 0.02)

        # Thinking block
        self.thinking_block = TransformerBlock(config)

        # Maps hidden state back to embedding space
        self.hidden_to_embed = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
        )

        # Optional adaptive depth
        if config.use_adaptive_depth:
            self.iteration_decider = AdaptiveIterationDecider(config.d_model)
        else:
            self.iteration_decider = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Apply latent reasoning with virtual thought tokens.

        Prepends virtual thought embeddings to the sequence,
        runs iterative reasoning, then returns the processed representation.

        Args:
            x: Input embeddings [B, T, D]
            position_ids: Position IDs [B, T]
            max_iterations: Override max iterations

        Returns:
            Dict with output, n_iterations, continue_probs
        """
        B, T, D = x.shape
        max_iter = max_iterations or self.config.max_latent_iterations

        # Prepend virtual thought embeddings to each batch item
        virtual = self.virtual_thoughts.unsqueeze(0).expand(B, -1, -1)  # [B, n_virtual, D]

        # Extend position IDs for virtual thoughts (use negative positions)
        virtual_positions = torch.arange(-self.n_virtual_thoughts, 0, device=position_ids.device)
        virtual_positions = virtual_positions.unsqueeze(0).expand(B, -1)

        # Concatenate: [virtual_thoughts, input_sequence]
        x_with_virtual = torch.cat([virtual, x], dim=1)  # [B, n_virtual + T, D]
        pos_with_virtual = torch.cat([virtual_positions, position_ids], dim=1)

        # Iterative reasoning
        prev_virtual = virtual.clone()
        continue_probs = []
        actual_iterations = 0

        for iteration in range(max_iter):
            # Forward through thinking block
            hidden = self.thinking_block(x_with_virtual)

            # Adaptive stopping
            if self.iteration_decider is not None and iteration > 0:
                # Use virtual thought positions for decision
                virtual_hidden = hidden[:, :self.n_virtual_thoughts, :]
                prob = self.iteration_decider(prev_virtual, virtual_hidden)
                continue_probs.append(prob)

                if not self.training and prob.mean() < 0.5:
                    break

            # COCONUT mechanism: Replace virtual embeddings with hidden states
            # Each virtual thought i gets the hidden state from position i-1
            new_virtual = self.hidden_to_embed(hidden[:, :self.n_virtual_thoughts, :])

            # Shift: virtual[i] <- hidden[i-1]
            # First virtual thought gets a learned initialization
            new_virtual = torch.cat([
                new_virtual[:, :1, :],  # First stays from initial embedding
                new_virtual[:, :-1, :],  # Rest shift from previous positions
            ], dim=1)

            x_with_virtual = torch.cat([new_virtual, x_with_virtual[:, self.n_virtual_thoughts:, :]], dim=1)
            prev_virtual = new_virtual
            actual_iterations += 1

        # Return only the original sequence positions (not virtual thoughts)
        output = x_with_virtual[:, self.n_virtual_thoughts:, :]

        return {
            "output": output,
            "n_iterations": actual_iterations,
            "continue_probs": torch.stack(continue_probs, dim=0) if continue_probs else None,
            "final_thoughts": x_with_virtual[:, :self.n_virtual_thoughts, :],  # For analysis
        }


# Factory function to integrate with existing LAHR
def create_latent_module(config, use_virtual_tokens: bool = True):
    """
    Create the appropriate latent reasoning module.

    Args:
        config: LAHRConfig or COCONUTConfig
        use_virtual_tokens: If True, use V2 that doesn't need tokenizer mods

    Returns:
        Latent reasoning module
    """
    # Convert LAHRConfig to COCONUTConfig if needed
    coconut_config = COCONUTConfig(
        d_model=getattr(config, 'd_model', 256),
        n_heads=getattr(config, 'n_heads', 8),
        ff_mult=getattr(config, 'ff_mult', 4),
        dropout=getattr(config, 'dropout', 0.1),
        max_latent_iterations=getattr(config, 'n_latent_iterations', 4),
        use_adaptive_depth=True,
    )

    if use_virtual_tokens:
        return COCONUTLatentReasoningV2(
            coconut_config,
            n_virtual_thoughts=getattr(config, 'n_latent_iterations', 4)
        )
    else:
        return COCONUTLatentReasoning(coconut_config)
