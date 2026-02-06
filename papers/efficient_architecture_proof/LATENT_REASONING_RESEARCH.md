# Latent Reasoning Research Summary

## Executive Summary

Research agent found that true COCONUT implementation uses **hidden state → embedding replacement**, not simple transformer block looping. This has been fixed in lahr_v4.py (now v5).

## Key Architectures Compared

| Architecture | Key Mechanism | Best For | Limitation |
|-------------|---------------|----------|------------|
| **COCONUT** | Hidden state replaces input embedding | Logical reasoning (ProsQA: 97%) | Fails on arithmetic (GSM8K: 34%) |
| **Quiet-STaR** | Thoughts at every token position | General LM improvement | Expensive (thoughts everywhere) |
| **Think-at-Hard** | Adaptive iteration depth | Efficiency | Two-stage training required |
| **Recurrent Depth** | Variable-depth core recursion | Test-time scaling | Requires from-scratch pretraining |
| **Mixture-of-Recursions (MoR)** | Token-level adaptive recursion | MoD + latent combined | Newer, less validated |

## COCONUT Implementation Details

### The Correct Mechanism (from official repo)

```python
# Key step: Replace thought token embeddings with hidden states
for (batch_idx, token_idx) in thought_positions:
    inputs_embeds[batch_idx, token_idx] = hidden_states[batch_idx, token_idx - 1, :]
```

### Required Components
1. **Special tokens**: `<bot>` (begin thought), `<thought>` (placeholder), `<eot>` (end thought)
2. **Curriculum training**: Stage k replaces k CoT steps with k latent tokens
3. **Training data**: Must have chain-of-thought annotations

### Our Implementation (V2 - Virtual Thoughts)
Uses learned virtual thought embeddings instead of special tokens:
- No tokenizer modification needed
- Virtual thoughts prepended to sequence
- Hidden states replace virtual embeddings each iteration
- Adaptive stopping via Think-at-Hard style decider

## Path Forward for Novel Contribution

### Recommended: Hybrid COCONUT + Think-at-Hard + MoD

**Rationale**:
1. **MoD** provides throughput gains to fund extra latent compute
2. **COCONUT** provides correct latent reasoning mechanism
3. **Think-at-Hard** enables adaptive iteration depth (efficiency)

### Novel Research Questions
1. Does MoD routing correlate with latent reasoning needs?
2. Can memory retrieval reduce required latent iterations?
3. What's the efficiency-quality tradeoff of MoD + adaptive latent depth?

### Existing Gaps in Literature
| Paper | MoD | Latent | Adaptive Depth | Memory |
|-------|-----|--------|----------------|--------|
| COCONUT | No | Yes (static) | No | No |
| Think-at-Hard | No | Yes | Yes | No |
| MoR | Yes | Yes | Yes (router) | No |
| **LAHR** | Yes | Yes | Yes (decider) | **Yes** |

Memory-augmented latent reasoning is unique to our architecture.

## References

- [COCONUT Paper](https://arxiv.org/abs/2412.06769) - [Code](https://github.com/facebookresearch/coconut)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629) - [Code](https://github.com/ezelikman/quiet-star)
- [Think-at-Hard](https://arxiv.org/abs/2511.08577) - [Code](https://github.com/thu-nics/TaH)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [Recurrent Depth](https://huggingface.co/papers/2502.05171)
- [CODI](https://arxiv.org/abs/2502.21074)

## Next Steps

1. **For validation**: Test on ProsQA (where COCONUT excels)
2. **For curriculum training**: Implement stage progression with CoT data
3. **For novelty**: Investigate MoD routing → latent iteration correlation
4. **For scale**: Test at larger model sizes with proper compute budget
