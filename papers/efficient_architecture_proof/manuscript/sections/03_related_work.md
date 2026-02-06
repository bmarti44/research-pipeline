# 2. Related Work

## 2.1 Adaptive Computation

The idea that different inputs require different amounts of computation dates back to
Adaptive Computation Time [Graves, 2016]. Universal Transformer [Dehghani et al., 2018]
applied this to transformers with shared-parameter blocks and learned halting.

More recently, Mixture-of-Depths [Raposo et al., 2024] demonstrated that simple top-k
routing can achieve significant efficiency gains. At the 1B parameter scale, MoD
achieved ~50% FLOP reduction while maintaining language modeling quality. The key
insight is that many tokens in a sequence can skip computation at certain layers
without degrading output quality.

## 2.2 Latent Reasoning

Chain-of-thought prompting [Wei et al., 2022] demonstrated that explicit reasoning
improves performance on complex tasks but requires generating potentially redundant
tokens. Several approaches have explored implicit reasoning:

- **Pause tokens** [Goyal et al., 2023]: Insert special tokens that allow additional
  computation without semantic content.

- **COCONUT** [Hao et al., 2024]: Chain of Continuous Thought performs reasoning in
  the model's continuous latent space. Hidden states are fed back through the model
  without decoding to vocabulary. This achieves 97% on ProsQA (a logical search task)
  vs. 77.5% for explicit CoT. However, COCONUT underperforms on arithmetic (GSM8K:
  34% vs. 42% for CoT), suggesting latent reasoning has different strengths than
  explicit reasoning.

## 2.3 Memory-Augmented Networks

External memory in neural networks has a long history:

- **Memory Networks** [Weston et al., 2014]: Introduced differentiable memory access.
- **Neural Turing Machines** [Graves et al., 2014]: Content and location-based addressing.
- **Memorizing Transformers** [Wu et al., 2022]: kNN retrieval from external memory
  improved language modeling with up to 262K tokens of context.

## 2.4 Hybrid Architectures

Recent work has explored combining multiple efficiency techniques:

- **Jamba** [AI21, 2024]: Combines attention and Mamba layers in a 1:7 ratio,
  achieving efficient inference at 52B parameters.
- **MoE + Adaptive Depth** [Various]: Several works combine mixture-of-experts
  routing with adaptive computation.

Our work differs in combining three distinct techniques (MoD, latent reasoning,
memory) rather than two, and in focusing on the interaction effects between components.
