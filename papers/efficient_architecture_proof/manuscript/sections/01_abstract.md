# Abstract

We present LAHR (Latent Adaptive Hierarchical Reasoning), a neural architecture that
combines three recent innovations: Mixture-of-Depths adaptive computation, latent-space
reasoning inspired by COCONUT, and differentiable memory retrieval. We conduct an
exploratory study at small scale (~20M parameters) on the TinyStories dataset to
investigate whether these components provide complementary benefits.

Our pilot experiments (3 conditions, 100 training steps each) yield **negative results**:
the full LAHR model underperforms a standard transformer baseline (training perplexity
21,867 vs 21,494, a 1.7% degradation). Mixture-of-Depths improves throughput by 13%,
but the full model has 17% lower throughput than baseline due to latent reasoning
overhead. At this scale and training duration, we find no evidence of complementary
benefits from combining these techniques. Whether benefits emerge with longer training
or larger scale remains an open question requiring further study.

We release our implementation to enable future work on efficient reasoning architectures.
This work represents an exploratory design study; claims should be interpreted with
appropriate caution given the limited scale and statistical power.

**Keywords:** adaptive computation, latent reasoning, memory-augmented networks,
efficient transformers
