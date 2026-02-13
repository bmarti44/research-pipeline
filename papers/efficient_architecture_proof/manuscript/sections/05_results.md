## 4. Results

### 4.1 Training Replication

Table 1 reports validation and test accuracy for all three models. M3 (COCONUT) achieves 98.0% test accuracy, replicating the ~97% reported by Hao et al. (2024) to within 1 percentage point. M1 (chain-of-thought) reaches 83.0%, consistent with the original baseline. M5 (pause) reaches 95.6% on the test set, closing 85% of the gap between M1 and M3. On the validation set, M5 matches M3 exactly at 97.3%; the 2.4 percentage-point test gap falls within single-seed variance and does not reach statistical significance.

**Table 1.** Accuracy by model on ProsQA validation (n = 300) and test (n = 500) sets.

| Model | Mechanism | Val Accuracy | Test Accuracy | Best Epoch |
|-------|-----------|:------------:|:-------------:|:----------:|
| M1 (CoT) | Explicit chain-of-thought | 79.67% | 83.0% | 44 |
| M3 (COCONUT) | Hidden-state recycling | 97.3% | 98.0% | 49 |
| M5 (Pause) | Learned pause embeddings | 97.3% | 95.6% | 43 |

Training curves for all three models are shown in Figure 5. M3 and M5 converge at comparable rates under the shared curriculum schedule, while M1 plateaus earlier at a lower asymptote.

### 4.2 Experiment 1: Corruption Ablation

We corrupted thought-token representations at each of the six latent positions (0--5) by replacing the hidden state with Gaussian noise, proceeding from position 0 forward. Table 2 reports accuracy as a function of the number of positions corrupted.

**Table 2.** Accuracy under progressive forward corruption by number of thought positions replaced with noise (n = 500 per condition).

| Positions Corrupted | M3 (COCONUT) | M5 (Pause) |
|:-------------------:|:------------:|:----------:|
| 0 (clean) | 97.0% | 96.6% |
| 1 | 96.8% | 96.4% |
| 2 | 96.8% | 96.2% |
| 3 | 96.8% | 95.8% |
| 4 | 57.4% | 57.2% |
| 5 | 15.6% | 15.6% |
| 6 | 2.4% | 2.2% |

Both models exhibit identical degradation profiles (Figure 3). Accuracy remains near ceiling through position 3, drops precipitously between positions 3 and 4 (from ~96% to ~57%), and collapses to near chance by position 6. The parallel trajectories indicate that the recycled hidden states in M3 do not confer robustness to corruption beyond what the learned pause embeddings in M5 provide.

**Permutation sensitivity.** We tested whether the ordering of thought tokens carries sequential information by permuting all latent positions and measuring the rate at which the model's prediction changes. Across 500 test samples with 10 random permutations each (5,000 permutation trials per model), neither M3 nor M5 produced a single prediction flip (flip rate = 0.0%). Partial permutation experiments, in which subsets of positions were permuted, likewise produced a 0.0% flip rate. Both models treat thought positions as an unordered bag of compute: the information distributed across latent tokens is order-invariant.

**Cross-problem transplantation.** To test whether thought representations encode problem-specific information, we transplanted the full set of thought-token activations from one problem into another and measured accuracy on the recipient problem. Across 200 donor--recipient pairs, M3 achieved 97.0% and M5 achieved 96.5%, matching clean-input performance. Thought representations are not problem-specific; they carry general computational state that functions equally well regardless of which problem generated them.

### 4.3 Experiment 2: Representation Probing

We trained linear probes on frozen hidden states at every (layer, position) cell to decode which intermediate reasoning step the model had reached. Each model has 13 layers and 6 thought positions, yielding 78 probed cells per model. Sample sizes vary by position because not all ProsQA problems require all six hops: n = 500 for positions 0--2, n = 298 for position 3, n = 81 for position 4, and n = 12 for position 5.

**Table 3.** Probing summary statistics for M3 and M5.

| Metric | M3 (COCONUT) | M5 (Pause) |
|--------|:------------:|:----------:|
| Peak probe accuracy | 55.4% | 57.1% |
| Peak location (layer, position) | (0, 3) | (12, 3) |
| Selectivity (all 78 cells) | 0.0 | 0.0 |
| Cells where MLP > linear | 0 / 78 | 0 / 78 |
| Mean thought-vs-input advantage | 10.5% | 4.0% |
| Max input position accuracy | 5.0% | 6.2% |

Two results are noteworthy. First, selectivity is 0.0 for every probed cell in both models. Following the framework of Ravichander et al. (2021), selectivity measures how much better a probe performs on the true target variable than on a random control variable. A selectivity of zero indicates that the probing accuracy does not exceed the control baseline, meaning that the representations do not encode step-specific information above chance. Every thought position decodes every reasoning step equally well, consistent with a general problem representation broadcast uniformly across positions rather than a sequential chain in which each position encodes a distinct step.

Second, the two models concentrate decodable information at different locations in the network. M3's peak probe accuracy occurs at layer 0, position 3. Because COCONUT recycles the final-layer hidden state back into the input embedding stream, the recycled representation arrives pre-processed at layer 0, making intermediate information linearly accessible from the earliest layer. M5 builds its representations through the transformer stack, with peak accuracy at layer 12 (the final layer). The diagonal peak layers for M3 are [8, 12, 12, 0] across positions 0--3; for M5 they are [8, 11, 12, 12]. These patterns reflect architectural differences in where information is injected, not differences in what information is encoded.

M3's higher thought-vs-input advantage (10.5% vs. 4.0%) shows that hidden-state recycling injects more task-relevant information into thought positions relative to input positions. However, this additional decodable information does not translate to a performance advantage: M5 matches M3 on in-distribution accuracy and exceeds it on most out-of-distribution tests (Section 4.4). The nonlinear probe advantage is zero for both models (no cell shows higher accuracy with an MLP probe than with a linear probe), indicating that the encoded information, such as it is, is linearly decodable. The probing heatmaps for both models are shown in Figure 1.

### 4.4 Experiment 3: Out-of-Distribution Generalization

We evaluated all three models on four out-of-distribution test sets that vary graph structure and path length beyond the training distribution: 7-hop paths, 8-hop paths, directed acyclic graphs (DAG), and dense graphs. Each test set contains 500 examples. Table 4 reports accuracy and pairwise comparisons between M3 and M5 using McNemar's test with Bonferroni correction across the four OOD comparisons.

**Table 4.** Out-of-distribution accuracy and M5 vs. M3 pairwise comparisons. Bonferroni correction applied across k = 5 tests.

| Test Set | M1 (CoT) | M3 (COCONUT) | M5 (Pause) | M5 -- M3 | McNemar $\chi^2$ | p (raw) | p (Bonferroni) | Sig. |
|----------|:---------:|:------------:|:----------:|:--------:|:-----------:|:-------:|:--------------:|:----:|
| ProsQA (ID) | 83.0% | 97.0% | 96.6% | --0.4 pp | 0.032 | 0.857 | 1.000 | No |
| 7-hop | 10.7% | 66.0% | 75.4% | +9.4 pp | 10.107 | 0.001 | 0.007 | Yes |
| 8-hop | 8.2% | 67.5% | 75.1% | +7.6 pp | 6.643 | 0.010 | 0.050 | Yes |
| DAG | 28.2% | 59.2% | 51.9% | --7.3 pp | 5.076 | 0.024 | 0.121 | No |
| Dense | 14.1% | 61.2% | 68.4% | +7.2 pp | 5.340 | 0.021 | 0.104 | No |

M5 outperforms M3 on three of four OOD test sets. The 7-hop advantage is statistically significant after Bonferroni correction ($\chi^2$ = 10.107, p = 0.007), and the 8-hop advantage is borderline significant (p = 0.050). M3 holds a 7.3 percentage-point advantage on DAG topology, but this difference does not survive correction (p = 0.121). On dense graphs, M5 leads by 7.2 points (p = 0.104, not significant after correction). The in-distribution comparison shows no meaningful difference between M3 and M5 (p = 1.000). The OOD accuracy pattern is shown in Figure 2.

The direction of these results is consistent with a sequential-bottleneck account of the recycling mechanism. COCONUT's hidden-state recycling forces each thought token to depend on the output of the previous step, creating a serial dependency chain. When problems require more hops than the training distribution contains, this chain must extrapolate sequentially, and errors compound across steps. Pause tokens impose no such dependency: each position attends freely to all previous positions through standard self-attention, allowing the model to distribute computation more flexibly. The advantage of M5 on 7-hop and 8-hop paths -- the test sets that most directly stress sequential extrapolation -- supports this interpretation. M3's advantage on DAG structures may reflect a case where the sequential inductive bias of recycling aligns with the topological ordering of directed acyclic graphs, though this effect is not statistically reliable.

M1 performs near chance on all OOD test sets (8.2%--28.2%), confirming that the curriculum-trained latent-reasoning approach, whether implemented via recycling or pause tokens, provides substantial generalization benefits over explicit chain-of-thought at this model scale.
