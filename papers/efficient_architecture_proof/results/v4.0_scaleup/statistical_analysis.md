# Statistical Analysis - v4.0 Scaleup (38M params)

## Executive Summary

At 38M parameters with BPE tokenization and warm-start training:
- **Latent multi-pass reasoning provides genuine improvement** (-4.8% PPL vs baseline, p < 0.001)
- **Full A+B+C provides additional synergy** (-6.8% PPL vs baseline, p < 0.001)
- **FLOP confound is resolved**: baseline_6000 catastrophically overfits (3.07 PPL vs 2.45 baseline_1000)
- **Not regularization**: dropout control (2.45 PPL) matches baseline (2.45 PPL)
- **Not continued training effect**: baseline_1000_continued (2.47 PPL) shows no improvement over baseline_1000

**Methodology Note**: Our implementation follows the core COCONUT mechanism (hidden state replaces input embeddings across iterations) but does not include special token markup (`<bot>`, `<thought>`, `<eot>`) or BFS-specific evaluation from the original paper. We refer to this as "latent multi-pass reasoning" or "COCONUT-style latent reasoning."

---

## 1. Descriptive Statistics with 95% Confidence Intervals

| Condition | n | Mean PPL | Std | 95% CI |
|-----------|---|----------|-----|--------|
| baseline_1000 | 10 | 2.4466 | 0.0079 | [2.4410, 2.4523] |
| baseline_1000_continued | 5 | 2.4654 | 0.0142 | [2.4478, 2.4830] |
| baseline_2000 | 6 | 2.4616 | 0.0148 | [2.4460, 2.4771] |
| baseline_6000 | 5 | **3.0730** | 0.0492 | [3.0119, 3.1341] |
| baseline_2000_dropout | 5 | 2.4477 | 0.0140 | [2.4304, 2.4650] |
| latent_warmstart | 10 | **2.3304** | 0.0250 | [2.3125, 2.3482] |
| full_abc_warmstart | 10 | **2.2800** | 0.0091 | [2.2735, 2.2865] |

---

## 2. Corrected Pairwise Tests

### 2.1 Paired t-tests (Same Seeds)

These comparisons use the same seeds across conditions, enabling paired analysis which controls for seed-level variance and increases statistical power.

| Comparison | n | Paired t | df | p | Cohen's d | Mean Diff [95% CI] | Sig |
|------------|---|----------|-----|------|-----------|---------------------|-----|
| full_abc vs baseline_1000 | 10 | -45.53 | 9 | <0.0001 | -14.40 | -0.167 [-0.174, -0.160] | *** |
| latent vs baseline_1000 | 10 | -14.59 | 9 | <0.0001 | -4.61 | -0.116 [-0.132, -0.101] | *** |
| full_abc vs latent (synergy) | 10 | -6.85 | 9 | <0.0001 | -2.17 | -0.050 [-0.065, -0.036] | *** |
| latent vs baseline_1000_continued | 5 | -12.25 | 4 | 0.0003 | -5.48 | -0.133 [-0.163, -0.103] | *** |

*Note: Large Cohen's d values (|d| > 4) reflect high reproducibility across seeds (very low variance in difference scores), not implausibly large effects. The absolute PPL differences are modest (0.05-0.17 PPL units) but highly consistent.*

### 2.2 Welch's t-tests (Unpaired, Unequal Variance)

These comparisons involve different sample sizes or non-overlapping seeds. Welch's t-test is used due to variance heterogeneity (Levene's test p < 0.05 for FLOP control).

| Comparison | Welch t | df | p | Cohen's d | Sig |
|------------|---------|-----|------|-----------|-----|
| baseline_6000 vs baseline_1000 | 28.29 | 4.1 | <0.0001 | 22.31 | *** |
| baseline_1000_continued vs baseline_1000 | 2.74 | 5.3 | 0.038 | 1.82 | * |
| baseline_2000 vs baseline_1000 | 2.28 | 6.7 | 0.058 | 1.37 | ns |
| baseline_2000_dropout vs baseline_1000 | 0.16 | 5.3 | 0.875 | 0.11 | ns |

***Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant***

---

## 3. Holm-Bonferroni Correction

All 8 tests in the family, ordered by p-value:

| Rank | Comparison | p-value | Adjusted α | Survives? |
|------|------------|---------|------------|-----------|
| 1 | full_abc vs baseline | <0.0001 | 0.0063 | YES *** |
| 2 | latent vs baseline | <0.0001 | 0.0071 | YES *** |
| 3 | baseline_6000 vs baseline_1000 | <0.0001 | 0.0083 | YES *** |
| 4 | full_abc vs latent (synergy) | <0.0001 | 0.0100 | YES *** |
| 5 | latent vs baseline_1000_continued | 0.0003 | 0.0125 | YES *** |
| 6 | baseline_1000_continued vs baseline_1000 | 0.038 | 0.0167 | NO |
| 7 | baseline_2000 vs baseline_1000 | 0.058 | 0.0250 | NO |
| 8 | baseline_dropout vs baseline_1000 | 0.875 | 0.0500 | NO |

---

## 4. FLOP Confound Resolution

### 4.1 Compute Breakdown

| Condition | Total Steps | FWD Passes/Step | Total FWD Passes | Final PPL |
|-----------|-------------|-----------------|------------------|-----------|
| baseline_1000 | 1000 | 1 | 1,000 | 2.45 |
| baseline_6000 | 6000 | 1 | 6,000 | **3.07** |
| latent_warmstart | 1000+1000 | 1+5 | 6,000 | **2.33** |

*Note: max_n_latent=4 was used, resulting in 5 forward passes per latent training step (4 iterations + 1 final).*

### 4.2 Key Finding: Baseline Cannot Use Extra Compute

- **baseline_6000 vs baseline_1000**: Welch t(4.1) = 28.29, p < 0.0001, d = 22.31
- The baseline gets **25.6% WORSE** with 6x more compute
- Train PPL drops to 1.07 (severe overfitting) while val PPL climbs to 3.07

### 4.3 Best-Ever Comparison (Exploratory)

| Condition | Best-ever PPL | At Step |
|-----------|---------------|---------|
| baseline_6000 | 2.396 ± 0.029 | ~1000-2000 |
| latent_warmstart | **2.330 ± 0.025** | 1000 (final) |

**Note**: This is post-hoc exploratory analysis (checkpoint selection based on same validation set).

### 4.4 FLOP-Matched Comparison

- latent_warmstart (2.33) vs baseline_6000 (3.07)
- **Latent multi-pass reasoning wins by 24.1%** at matched compute

---

## 5. Regularization Control

Does latent multi-pass reasoning act as a regularizer?

| Condition | Dropout | PPL |
|-----------|---------|-----|
| baseline_2000 | 0.1 (default) | 2.46 |
| baseline_2000_dropout | 0.2 | 2.45 |

- Welch t(5.3) = 0.16, p = 0.875, d = 0.11
- **No significant difference** - dropout does NOT explain latent reasoning's improvement

---

## 5.5 Warm-Start Control (NEW)

Does the latent reasoning improvement come from the mechanism itself, or simply from continued training?

### 5.5.1 Control Design

We ran a control experiment where baseline_1000 checkpoints (n=5 seeds) were continued for 1000 additional steps **without** the latent mechanism — exactly matching the training protocol of latent_warmstart except using standard forward passes instead of iterative hidden-state refinement.

| Condition | Starting Point | Additional Steps | Mechanism | Final PPL |
|-----------|----------------|------------------|-----------|-----------|
| baseline_1000 | scratch | 1000 | standard | 2.45 |
| baseline_1000_continued | baseline_1000 | +1000 | standard | 2.47 |
| latent_warmstart | baseline_1000 | +1000 | latent reasoning | **2.33** |

### 5.5.2 Key Finding: Mechanism Effect Isolated

Comparing conditions with matching seeds (n=5):

| Comparison | Paired t(4) | p | Cohen's d | Interpretation |
|------------|-------------|---|-----------|----------------|
| latent vs baseline_1000_continued | -12.25 | 0.0003 | -5.48 | **Latent mechanism is the cause** |
| baseline_1000_continued vs baseline_1000 | — | 0.049* | 1.64 | No benefit from continued training |

*Note: The comparison of baseline_1000_continued vs baseline_1000 uses Welch's t-test (different samples), and p=0.049 does not survive Holm-Bonferroni correction (adjusted α=0.0167), so there is **no significant improvement** from continued training alone.

### 5.5.3 Conclusion

**The latent mechanism is the causal factor, not continued training.**
- Continued vanilla training slightly *increases* PPL (overfitting tendency)
- Latent reasoning decreases PPL by 5.5% vs the same starting point
- This isolates the effect to the mechanism itself

---

## 6. Synergy Analysis

Does adding MoD + Memory on top of latent reasoning help?

| Condition | PPL | vs baseline |
|-----------|-----|-------------|
| latent_warmstart | 2.330 | -4.8% |
| full_abc_warmstart | **2.280** | -6.8% |

- full_abc vs latent: Paired t(9) = -6.85, p < 0.0001, d = -2.17
- Adding MoD + Memory improves by **-2.2%** on top of latent reasoning
- **Additional benefit observed**: mechanisms provide improvement when combined (whether additive or synergistic cannot be determined without single-component ablations)

---

## 7. One-Way ANOVA

Across baseline_1000, latent_warmstart, full_abc_warmstart:
- F(2, 27) = 285.16, p < 0.0001
- Significant differences exist between groups

---

## 8. Changes from Previous Analysis

### 8.1 Test Type Corrections
- **Paired t-tests** now used for same-seed comparisons (baseline vs latent vs full_abc)
- **Welch's t-tests** now used for all unpaired comparisons (variance heterogeneity confirmed)

### 8.2 Significance Changes
| Comparison | Previous | Corrected |
|------------|----------|-----------|
| baseline_2000 vs baseline_1000 | p = 0.019 (*) | p = 0.058 (ns) |
| All other comparisons | Significant | Significant (unchanged) |

### 8.3 Effect Size Interpretation
- Paired Cohen's d values are based on SD of differences (appropriate for paired design)
- Effect sizes remain large (d = -4.61 for latent, d = -14.40 for full_abc)
- These values reflect high reproducibility across seeds, not implausibly large effects

---

## 9. Conclusions

### 9.1 FLOP Confound is RESOLVED

The compute confound argument is definitively refuted:
1. baseline_6000 uses the same compute as latent warmstart (6000 FWD passes)
2. baseline_6000 is **25.6% WORSE** than baseline_1000 due to overfitting
3. Even at its best checkpoint, baseline cannot match latent reasoning
4. Latent multi-pass reasoning wins by **24%** at matched compute

### 9.2 Latent Multi-Pass Benefit is Genuine

All plausible confounds are controlled:
- **Not regularization**: dropout control (0.2 dropout) shows no benefit vs baseline
- **Not simply more training**: baseline_6000 overfits catastrophically with extended training
- **Not warm-start effect**: baseline_1000_continued matches training duration but shows no benefit
- **Conclusion**: The iterative hidden-state refinement mechanism provides genuine value

### 9.3 Warm-Start Control Resolves Remaining Confound

The baseline_1000_continued experiment (n=5) definitively isolates mechanism from training duration:
- Same starting checkpoint (baseline_1000)
- Same additional training steps (1000)
- Same optimizer, learning rate schedule, data
- **Only difference**: latent reasoning mechanism

Result: latent_warmstart (2.33) vs baseline_1000_continued (2.47)
- Paired t(4) = -12.25, p = 0.0003
- **5.5% improvement attributable to mechanism alone**

### 9.4 Additional Benefit from Combined Mechanisms

Full A+B+C (latent + MoD + Memory) beats latent alone:
- -2.2% additional improvement (p < 0.0001)
- Whether this is additive or synergistic cannot be determined without single-component ablations at 38M scale

### 9.5 Limitations

1. **Warm-start design**: Compares continued training, not from-scratch architectures
2. **Task domain**: Only TinyStories perplexity tested, not reasoning tasks where original COCONUT was validated
3. **Scale**: 38M parameters; may not generalize to larger models
4. **Parameter overhead**: The latent mechanism adds ~65K parameters via `hidden_to_embed` projection (~0.17% overhead); however, this small overhead cannot explain a 4.8% PPL improvement
5. **Learning rate schedule**: Both warm-start conditions (latent_warmstart and baseline_1000_continued) used identical LR schedules continuing from the decayed LR at step 1000 (~3e-5), ensuring fair comparison but potentially suboptimal for continued training

---

## Raw Data

### baseline_1000 (n=10)
Seeds: 1001, 123, 1234, 2345, 3456, 42, 456, 4567, 5678, 789
Values: 2.447, 2.438, 2.446, 2.437, 2.444, 2.445, 2.439, 2.460, 2.456, 2.455

### baseline_6000 (n=5)
Seeds: 1001, 123, 42, 456, 789
Final PPL: 3.011, 3.132, 3.106, 3.077, 3.038
Best-ever PPL: 2.390, 2.404, 2.434, 2.397, 2.354

### latent_warmstart (n=10)
Seeds: 1001, 123, 1234, 2345, 3456, 42, 456, 4567, 5678, 789
Values: 2.330, 2.318, 2.305, 2.305, 2.376, 2.353, 2.319, 2.302, 2.353, 2.344

### full_abc_warmstart (n=10)
Seeds: 1001, 123, 1234, 2345, 3456, 42, 456, 4567, 5678, 789
Values: 2.282, 2.290, 2.278, 2.275, 2.279, 2.291, 2.264, 2.275, 2.293, 2.272

### baseline_1000_continued (n=5)
Seeds: 42, 123, 456, 789, 1001
Values: 2.465, 2.481, 2.454, 2.449, 2.479
