# Research Synthesis: LAHR Architecture

## Executive Summary

Based on deep research into the three component innovations, here's what we know works and what's uncertain:

| Component | Validated At Small Scale? | Key Evidence | Critical Gap |
|-----------|--------------------------|--------------|--------------|
| Latent Reasoning (COCONUT) | **Yes** (GPT-2) | ProsQA: 97% vs 77.5% CoT | Underperforms on math (GSM8K) |
| Adaptive Depth (MoD) | **Yes** (1B scale) | 50% FLOP reduction | No evidence of combining with other innovations |
| Hierarchical Memory | **Partial** | Knowledge tasks improve | Training stability at scale unclear |

## Detailed Findings

### 1. Latent Reasoning - COCONUT (Meta, Dec 2024)

**Source**: [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)

**Key Results (using GPT-2)**:
- ProntoQA: 99.8% accuracy (vs 98.8% CoT) with 9 vectors vs 92.5 tokens
- ProsQA: **97.0% accuracy** (vs 77.5% CoT) with 14.2 vectors vs 49.4 tokens
- GSM8K: **34.1% accuracy** (vs 42.9% CoT) - **UNDERPERFORMS on math**

**Training Methodology**:
- Multi-stage curriculum: progressively replace CoT steps with continuous thoughts
- Hyperparameter `c`: number of continuous thoughts per language step
- Special tokens `<bot>` and `<eot>` mark latent reasoning boundaries
- Works with GPT-2 - validates small-scale approach

**Critical Insight**: Latent reasoning excels at logical/search tasks (BFS capability) but struggles with arithmetic. This may be because math requires precise symbolic manipulation that benefits from discrete token space.

**Code Available**: [facebookresearch/coconut](https://github.com/facebookresearch/coconut)

### 2. Adaptive Depth - Mixture of Depths (Google DeepMind, April 2024)

**Source**: [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)

**Key Results**:
- **50% FLOP reduction** with equivalent perplexity
- **66% faster training** up to 1B scale
- Memory savings at larger sizes
- Best configuration: 12.5% capacity blocks, every other block

**Mechanism**:
- Top-k routing: only k tokens participate in self-attention/MLP per layer
- Network learns which tokens need full computation
- Simple tokens skip layers entirely

**Combines Well With**: MoE layers for additional gains

**Key Insight**: Not all tokens need equal computation. This is validated at scale.

### 3. Hierarchical Memory - Multiple Lines of Work

**Key Sources**:
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913) (Google, 2022)
- [TNTM](https://medium.com/@MarxismLeninism/tntm-transformer-neural-turing-machine-giving-language-models-actual-memory-across-chats-95f138ced7fe) (2025)
- [MemLong](https://www.rohan-paul.com/p/state-of-memory-augmented-language) (2024)

**Key Results**:
- Memorizing Transformers: Performance improves up to 262K token memory
- 128B memory slots outperformed 2x larger dense transformer
- MemLong: Extended usable context from 4K to ~80K tokens

**Training Challenge**: Write operations are difficult to train end-to-end differentiably at scale.

### 4. Hybrid Architectures - Jamba (AI21, 2024)

**Source**: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

**Architecture**: 1:7 ratio of attention to Mamba layers

**Key Results**:
- 256K context length with small memory footprint
- Competitive with state-of-the-art at 52B params (12B active)
- **Finding**: Mamba-1 + Attention works better than Mamba-2 + Attention

**Implication**: Simple isn't always best - interaction effects matter.

### 5. Scaling Laws - Can Small Scale Predict Large?

**Source**: [Scaling Laws Under the Microscope](https://aclanthology.org/2022.findings-emnlp.544/) (EMNLP 2022)

**Key Finding**: Scaling laws emerge even with models as small as 10K parameters for some tasks.

**Caveat**: [Scaling Laws Are Unreliable for Downstream Tasks](https://aclanthology.org/2025.findings-emnlp.877.pdf) (2025) - context dependent.

**Implication**: Small-scale validation is meaningful but not guaranteed to transfer. We should:
1. Test on multiple task types
2. Look for consistent patterns across scales
3. Not over-claim from single experiments

### 6. Training Stability Considerations

**Key Issues**:
- Adam optimizer can cause loss spikes with adaptive preconditioners
- Gradient explosion in deep/recurrent networks
- Signal propagation problems at initialization

**Solutions**:
- Pre-normalization (RMSNorm before attention/FFN)
- Adaptive Gradient Clipping (AGC)
- Careful initialization (Xavier, scaled appropriately)

## Revised Architecture Decisions

Based on this research, here are the key decisions for LAHR:

### What to Include

1. **Latent Reasoning**: Yes, but with curriculum training following COCONUT methodology
   - Use `<bot>`/`<eot>` tokens
   - Multi-stage training
   - Don't expect math improvements

2. **Adaptive Depth**: Yes, use MoD-style top-k routing
   - 12.5% capacity is validated
   - Apply every other layer

3. **Memory**: Yes, but simplified
   - Read-only during training initially (more stable)
   - Write operations added later in curriculum

### What to Change from Original Design

1. **Remove**: Complex ACT-style halting probability accumulation
   - Use simpler top-k routing instead (validated)

2. **Add**: Proper curriculum learning for latent reasoning
   - Stage 1: Standard CoT training
   - Stage 2+: Progressive replacement with continuous thoughts

3. **Simplify**: Memory operations
   - Start with kNN retrieval (Memorizing Transformers style)
   - Add learned writes only after base model is stable

### Experimental Design

**Phase 1: Validate Components Individually (Tiny - 1M params)**
- Latent reasoning on ProntoQA/ProsQA
- Adaptive depth on perplexity
- Memory on knowledge retrieval

**Phase 2: Test Combinations (Small - 10M params)**
- Latent + Adaptive
- Adaptive + Memory
- Full combination

**Phase 3: Scale Validation (Medium - 50M params)**
- Verify patterns hold
- Compare to baseline transformer

## Honest Assessment of Success Probability

Given the research, I revise my estimate:

**Probability of meaningful contribution: 35-45%** (up from 15-25%)

Reasons for upgrade:
1. COCONUT validates latent reasoning works at GPT-2 scale
2. MoD validates adaptive depth at 1B scale
3. Memory augmentation is well-established
4. Clear curriculum learning methodology exists

Remaining risks:
1. Combination effects may be negative
2. Training three innovations together is complex
3. Small-scale results may not predict large-scale
4. Math/arithmetic tasks likely won't improve

## Sources

1. [COCONUT - arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
2. [Mixture-of-Depths - arXiv:2404.02258](https://arxiv.org/abs/2404.02258)
3. [Jamba - arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
4. [Memorizing Transformers - arXiv:2203.08913](https://arxiv.org/abs/2203.08913)
5. [Scaling Laws Under the Microscope - EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.544/)
6. [Memory-Augmented Transformers Survey](https://arxiv.org/html/2508.10824v1)
7. [PonderNet - arXiv:2107.05407](https://arxiv.org/abs/2107.05407)
8. [COCONUT GitHub](https://github.com/facebookresearch/coconut)
