# 1. Introduction

Large language models have demonstrated remarkable capabilities across diverse tasks,
yet their computational efficiency remains a significant challenge. Standard transformer
architectures apply uniform computation to all tokens regardless of complexity, and
explicit chain-of-thought reasoning requires generating potentially redundant tokens.

Recent work has proposed three promising directions for improving efficiency:

1. **Mixture-of-Depths (MoD)** [Raposo et al., 2024]: Routes tokens through selective
   computation paths, achieving ~50% FLOP reduction at matched quality on language
   modeling tasks at the 1B parameter scale.

2. **Latent Reasoning** [Hao et al., 2024]: The COCONUT (Chain of Continuous Thought)
   approach performs reasoning in continuous latent space rather than discrete token
   space, achieving 97% accuracy on ProsQA (vs. 77.5% for chain-of-thought) at GPT-2
   scale. However, this approach struggles with arithmetic tasks (34% vs 42% for CoT
   on GSM8K).

3. **Memory-Augmented Networks** [Wu et al., 2022]: External memory banks provide
   working memory that persists across context, improving performance on tasks
   requiring long-range information.

These techniques address fundamentally different bottlenecks: MoD addresses computational
redundancy, latent reasoning addresses token serialization overhead, and memory addresses
context limitations. This raises a natural question: **can these approaches be combined
for complementary benefits?**

In this work, we propose LAHR (Latent Adaptive Hierarchical Reasoning), an architecture
that integrates all three innovations. We conduct an exploratory study at small scale
(~20M parameters) to investigate the feasibility and potential benefits of this combination.

## Contributions

1. We present LAHR, a unified architecture combining MoD, latent reasoning, and
   differentiable memory.

2. We conduct a full 2³ factorial ablation study to isolate the contribution of
   each component.

3. We release our implementation for reproducibility and future research.

## Limitations (Stated Upfront)

We acknowledge several important limitations:

- **Scale**: Our experiments are conducted at ~20M parameters on TinyStories.
  Results may not transfer to larger scales.

- **Statistical Power**: Our study is exploratory with limited seeds per condition.
  Effect sizes should be interpreted with appropriate uncertainty.

- **Simplified Implementation**: Our latent reasoning module is a simplified
  version of COCONUT, lacking the full curriculum training and special tokens.

- **Known Failure Modes**: Latent reasoning is known to struggle with arithmetic
  tasks; we do not claim improvement on such tasks.
