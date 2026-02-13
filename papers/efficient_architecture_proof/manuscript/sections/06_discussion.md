## 5. Discussion

### 5.1 Convergent Evidence

Three independent experimental paradigms -- corruption analysis, representational probing, and out-of-distribution generalization -- produce a consistent picture. On every diagnostic where the reasoning hypothesis and the buffering hypothesis make divergent predictions, the data favor buffering. Table 5 summarizes the alignment.

| Evidence | Reasoning claim | Buffering claim | Our result |
|----------|:-:|:-:|:-:|
| Permutation sensitivity | Order matters | Order irrelevant | 0% flip rate for both M3 and M5 |
| Cross-transplant | Problem-specific states | Generic compute | Both tolerate foreign thoughts (M3: 97.0%, M5: 96.5%) |
| Corruption cliff | Gradual cascade | Threshold collapse | Identical cliff at position 4 for both models |
| Probing selectivity | Step-specific encoding | General broadcast | Selectivity = 0.0 for both M3 and M5 |
| Thought-vs-input advantage | Only COCONUT benefits | Equal benefit | M3 higher (10.5% vs. 4.0%), but unused |
| OOD generalization | COCONUT advantages | Equal or M5 advantages | M5 wins 3 of 4 test sets |

No single experiment is decisive in isolation. Permutation insensitivity could reflect redundant encoding; cross-transplant tolerance could indicate overlapping representations. But taken together, six independent diagnostics consistently fail to find evidence that COCONUT's recycled hidden states carry reasoning content that differs functionally from M5's learned pause vectors. The convergence across methods strengthens the conclusion beyond what any single test provides.

### 5.2 Information Without Function

The probing results reveal a dissociation between representational content and computational use. M3's thought-token positions encode 10.5% more decodable information about intermediate reasoning steps than its input positions, compared with 4.0% for M5. By this metric, the recycling mechanism has a measurable representational effect: it injects information into the thought positions that is absent from the pause baseline. Yet this additional information does not translate into a behavioral advantage. M5 matches M3 on the in-distribution test set (95.6% vs. 98.0%, p = 0.857 after Bonferroni correction) and outperforms it on three of four out-of-distribution benchmarks.

This dissociation is consistent with the distinction drawn by Ravichander et al. (2021): information that is linearly decodable from a model's representations is not necessarily used by the model's downstream computation. A probe can recover a signal that the classifier head never attends to. The recycling mechanism deposits intermediate-step information at layer 0 -- M3's peak probing accuracy occurs at the embedding layer, where the recycled hidden state is directly injected -- but this information does not propagate through the transformer's 12 subsequent layers in a way that improves output. M5 builds its (smaller) probing signal through the standard transformer computation, peaking at layer 12, yet reaches comparable or superior accuracy. The two models construct different representational pathways to the same behavioral outcome, and neither pathway encodes step-specific reasoning that exceeds what a control probe on random targets can achieve (selectivity = 0.0 for both).

### 5.3 The Sequential Bottleneck

COCONUT's hidden-state recycling imposes a sequential bottleneck: each thought position receives the final-layer hidden state of the previous position as its input embedding. Information must flow through a chain of forward passes, each dependent on the last. This architecture was motivated by the analogy to recurrent computation, where sequential state updates enable multi-step reasoning. But on ProsQA, this sequential dependency appears to be a liability rather than an asset.

M5's pause tokens occupy the same positions in the sequence but impose no such constraint. Each pause embedding is a fixed learned vector, and the model's self-attention mechanism is free to route information across all positions -- input tokens and pause tokens alike -- without forced sequential dependencies. This architectural freedom may explain M5's 7-9 percentage-point advantage on out-of-distribution test sets requiring longer reasoning chains (7-hop: +9.4pp, p = 0.007; 8-hop: +7.6pp, p = 0.050, Bonferroni-corrected). When the task demands generalization beyond training-distribution path lengths, the sequential bottleneck constrains the recycling model to a computation pattern that was optimized for shorter chains, while the pause model's standard self-attention can flexibly redistribute computation across the available positions.

### 5.4 Relation to Prior Work

Zhang et al. (2025) found that COCONUT's continuous thought tokens are largely causally inert on MMLU and HotpotQA when evaluated on LLaMA 7B and 8B models: shuffling, zeroing, or replacing thoughts with Gaussian noise produced minimal accuracy drops. Our results extend this finding to ProsQA -- the task where COCONUT achieves its strongest reported performance and where the theoretical case for latent reasoning is most compelling. The convergence across tasks (natural language QA, multi-hop retrieval, graph traversal) and scales (GPT-2 124M, LLaMA 7B/8B) strengthens the generality of the causal inertness finding, though the scale gap between our study and theirs remains a limitation.

Zhu et al. (2025) proved that continuous thought tokens are theoretically more expressive than discrete chain-of-thought tokens, capable of encoding superposition states that enable breadth-first search over graph structures. ProsQA was designed precisely to test this capability. Our probing analysis shows that the theoretical expressiveness is not realized in practice at GPT-2 124M scale: neither model exhibits step-specific encoding (selectivity = 0.0), and the recycling mechanism's additional representational content does not translate to a behavioral advantage. This does not refute the theoretical result -- expressiveness is an upper bound on what is possible, not a guarantee of what is learned -- but it does constrain the practical relevance of the expressiveness argument at the scale and training regime studied here.

Goyal et al. (2024) demonstrated that pause tokens can improve transformer performance by providing additional computation time, even when the tokens carry no task-relevant information. Our M5 baseline confirms and extends this finding: curriculum-trained pause tokens close 85% of the gap between chain-of-thought and COCONUT on ProsQA, and outperform COCONUT on out-of-distribution generalization. The curriculum, which progressively forces the model to internalize explicit reasoning, appears to be the active ingredient; the pause tokens provide the computational budget that the curriculum requires.

### 5.5 Practical Implications

The continuous thought mechanism introduces substantial architectural complexity. Hidden-state recycling requires multi-pass forward loops during both training and inference, roughly doubling VRAM consumption relative to a single-pass model with the same number of latent positions. Our results suggest that this complexity yields no measurable benefit on ProsQA: the pause baseline matches in-distribution accuracy and exceeds out-of-distribution accuracy with a simpler, single-pass architecture.

For researchers building on COCONUT's results, these findings suggest that investment in curriculum design -- the progressive removal of explicit reasoning tokens, the scheduling of thought-token introduction, the annealing of supervision -- is likely to produce larger returns than investment in the hidden-state recycling mechanism itself. The curriculum is the component that both M3 and M5 share, and it is the component that separates both models from the M1 chain-of-thought baseline by 14-15 percentage points on the in-distribution test set. Simpler architectures that exploit the same curriculum may achieve comparable performance with lower engineering and computational cost.
