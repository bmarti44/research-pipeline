# 3. Method

## 3.1 Architecture Overview

LAHR (Latent Adaptive Hierarchical Reasoning) combines three components:

1. **Mixture-of-Depths (MoD) Layers**: Selective token processing for efficiency
2. **Latent Reasoning Module**: Iterative processing in continuous space
3. **Memory Module**: Differentiable kNN retrieval from learned memory slots

The overall forward pass is:
```
Input → Embedding → Memory Retrieval → MoD Layers → Latent Reasoning → Output
```

## 3.2 Mixture-of-Depths Layers

Following Raposo et al. [2024], we implement MoD as top-k token selection:

1. A linear router predicts importance scores for each token
2. The top k tokens (k = capacity × T) are selected
3. Only selected tokens pass through the transformer block
4. Results are scattered back to original positions

**Critical Implementation Detail**: When processing the selected subset, we preserve
original position indices for correct causal masking. Token i can only attend to
token j if position(i) ≥ position(j), regardless of their indices in the subset.

We apply MoD to alternating layers (layers 1, 3, 5, ...) with 12.5% capacity,
meaning only 12.5% of tokens are processed at each MoD layer.

## 3.3 Latent Reasoning Module

Inspired by COCONUT [Hao et al., 2024], we implement latent reasoning as repeated
application of a shared transformer block:

```python
for i in range(n_iterations):
    x = transformer_block(x, position_ids)
```

This allows the model to perform additional computation without generating tokens.
We use n_iterations = 4 by default.

**Note**: This is a simplified version of COCONUT. We do not implement:
- Special `<bot>`/`<eot>` tokens marking reasoning boundaries
- Multi-stage curriculum training
- Breadth-first search over reasoning paths

We position latent reasoning AFTER all MoD layers, reasoning that "thinking" may
require global information that sparse MoD would disrupt.

## 3.4 Memory Module

We implement differentiable memory as a kNN retrieval module:

1. **Memory Bank**: Learned parameters M ∈ ℝ^(n_slots × d_model)
2. **Query**: Project input to query space
3. **Retrieval**: Select top-k most similar memory slots
4. **Integration**: Gate-weighted addition to input

The memory is queried BEFORE transformer layers, allowing retrieved context to
influence all subsequent processing.

## 3.5 Model Configurations

| Config | d_model | n_layers | n_heads | Parameters |
|--------|---------|----------|---------|------------|
| tiny   | 128     | 6        | 4       | ~1M        |
| small  | 256     | 8        | 8       | ~20M       |
| medium | 512     | 12       | 8       | ~50M       |

## 3.6 Training Details

- **Optimizer**: AdamW with β = (0.9, 0.95)
- **Learning Rate**: 3e-4 with warmup and cosine decay to 10%
- **Batch Size**: 32 (4 × 8 gradient accumulation)
- **Sequence Length**: 512 tokens
- **Data**: TinyStories dataset (~5M tokens)

## 3.7 Ablation Design

We conduct a full 2³ factorial experiment with 8 conditions:

| Condition | MoD | Latent | Memory |
|-----------|-----|--------|--------|
| full      | ✓   | ✓      | ✓      |
| no_mod    | ✗   | ✓      | ✓      |
| no_latent | ✓   | ✗      | ✓      |
| no_memory | ✓   | ✓      | ✗      |
| mod_only  | ✓   | ✗      | ✗      |
| latent_only | ✗ | ✓      | ✗      |
| memory_only | ✗ | ✗      | ✓      |
| baseline  | ✗   | ✗      | ✗      |

This design allows us to estimate:
- Main effects of each component
- Two-way interactions
- Three-way interaction
