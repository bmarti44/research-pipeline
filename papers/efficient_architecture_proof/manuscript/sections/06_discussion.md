# 5. Discussion

## 5.1 Key Findings

Our pilot study yields several findings relevant to future architecture research:

**1. MoD provides consistent throughput gains.** The Mixture-of-Depths routing
mechanism achieves the target 12.5% capacity and delivers 13% throughput improvement.
This validates the sparse computation approach at small scale.

**2. Component benefits may require longer training.** At 100 steps, all conditions
show similar loss (~10.4). The added complexity of latent reasoning and memory
retrieval does not provide immediate benefits, suggesting these components may
require the model to first learn basic patterns before they can contribute.

**3. Architectural overhead is measurable.** The full LAHR model runs 17% slower
than baseline due to latent reasoning iterations. This throughput cost must be
offset by quality gains that may only emerge with longer training.

**4. Ablation infrastructure works.** The 2³ factorial design successfully isolates
component contributions. The pipeline enables systematic comparison across
architectural variants, providing a template for future ablation studies.

**5. IMPORTANT: These are negative results.** The full LAHR model shows 1.7% higher
perplexity than baseline (21,867 vs 21,494). We found no evidence of complementary
benefits from combining MoD, latent reasoning, and memory. This negative finding is
an important contribution: it suggests that architectural combinations require
careful validation rather than assumed additive benefits.

## 5.2 Limitations

We acknowledge several important limitations of this work:

### Scale Limitations
Our experiments are conducted at ~20M parameters on TinyStories. This is
significantly smaller than the scales where MoD (1B), COCONUT (GPT-2), and
Memorizing Transformers were originally validated. Our results may not transfer
to larger scales.

### Statistical Power
With limited seeds per condition, our study is underpowered to detect small
effects. The exploratory nature of this work means effect sizes should be
interpreted with appropriate uncertainty.

### Implementation Simplifications
Our latent reasoning module is a simplified version of COCONUT:
- No special `<bot>`/`<eot>` tokens
- No curriculum training
- No breadth-first search over reasoning paths

A full COCONUT implementation might show different results.

### Task Limitations
We evaluate on language modeling perplexity only. Latent reasoning is known to
struggle with arithmetic tasks (GSM8K: 34% vs 42% for CoT). We do not claim
improvement on such tasks.

### Hardware-Specific Results
Results were obtained on Apple Silicon (MPS backend), which has different
numerical characteristics than CUDA. Reproducibility on other hardware may vary.

## 5.3 Future Work

1. **Scale up**: Test at 125M+ parameters with more compute
2. **Full COCONUT**: Implement curriculum training and special tokens
3. **Task-specific evaluation**: Evaluate on reasoning benchmarks
4. **Theoretical analysis**: Develop theory for component interactions
5. **Efficiency measurements**: Detailed FLOP accounting

## 5.4 Broader Impact

This work explores efficiency techniques for language models. If successful,
such techniques could:
- Reduce the computational cost of AI systems
- Enable deployment on consumer hardware
- Democratize AI research

We do not foresee direct negative applications of this architectural research.
