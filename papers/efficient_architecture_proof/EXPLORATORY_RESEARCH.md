# Exploratory Research Findings

Interesting observations from the A+B+C Architecture Study that warrant follow-up investigation.

**Generated**: 2026-02-05
**Source Study**: A+B+C Architecture Combination Study v3.3

---

## 1. COCONUT with Subword Tokenization (HIGH PRIORITY)

### Observation
COCONUT was designed for BPE-tokenized text where latent tokens replace *reasoning steps* (words/phrases). Our character-level implementation replaces individual characters, defeating the mechanism.

### Why Interesting
The original COCONUT paper showed significant benefits (-7.7% PPL) on proper text. Our "null result" may simply reflect implementation mismatch, not architectural limitation.

### Proposed Follow-Up Study
**Title**: "COCONUT at Small Scale: Proper Tokenization Test"

**Design**:
- Use tiktoken BPE tokenizer (~50K vocab)
- Reformat multi-step arithmetic as BPE token sequences
- Test at 10M+ params (need capacity for large vocab)
- Compare: baseline vs COCONUT with curriculum learning

**Hypothesis**: With subword tokens, COCONUT will show measurable benefit even at small scale.

**Effort**: ~2 weeks (new tokenizer, data reformatting, training infrastructure)

---

## 2. MoD Capacity Sweet Spot for Dense Prediction

### Observation
MoD shows dose-response relationship:
| Capacity | Overhead |
|----------|----------|
| 12.5% | +9.3% |
| 50% | +9.3% |
| 90% | +2.6% |
| 95% | +1.66% |

The relationship appears non-linear (plateau at low capacity, then steep improvement at high capacity).

### Why Interesting
- MoD was designed for tasks with "unimportant" tokens (padding, stop words)
- Character-level LM has no such tokens - every position matters
- But MoD still might work if nearly all tokens are processed (>97% capacity?)

### Proposed Follow-Up Study
**Title**: "Mixture-of-Depths for Dense vs Sparse Token Prediction"

**Design**:
- Test MoD at 97%, 98%, 99% capacity
- Compare tasks: char-level LM (dense) vs classification (sparse output) vs seq2seq (mixed)
- Hypothesis: MoD benefit/harm depends on task density

**Effort**: ~1 week

---

## 3. Memory Variance Anomaly

### Observation
Memory condition shows 2.5x higher variance than baseline (0.030 vs 0.012 std at tiny scale). This persisted across seeds.

### Why Interesting
- Memory mechanism involves learned retrieval (top-k selection)
- Top-k is discrete and may be unstable early in training
- Higher variance suggests sensitivity to initialization

### Proposed Follow-Up Study
**Title**: "Stability of Differentiable Memory During Training"

**Design**:
- Track memory retrieval patterns across training
- Compare: soft attention vs hard top-k retrieval
- Measure variance across seeds at different training stages

**Hypothesis**: Hard top-k selection creates bifurcating optimization paths.

**Effort**: ~1 week (mostly analysis, minimal new code)

---

## 4. Throughput Anomaly in MoD

### Observation
MoD with `skip_compute=True` was FASTER than baseline (higher tok/s) despite processing fewer tokens. This seems impossible if the router adds overhead.

### Why Interesting
- May indicate router learned degenerate "skip everything" solution
- Or GPU kernel efficiency from smaller batch
- Could be measurement artifact

### Proposed Follow-Up Study
**Title**: "MoD Router Behavior Analysis"

**Design**:
- Log router decisions across training
- Check if router converges to fixed pattern
- Profile actual compute time per component

**Effort**: ~3 days (mostly logging/analysis)

---

## 5. Scale-Dependent Null Results

### Observation
Memory showed essentially identical null result at 1M and 8.5M params:
- 1M: -0.34%, p=0.44
- 8.5M: +0.10%, p=0.55

Effect didn't grow with scale.

### Why Interesting
- Memory should help more at larger scale (fixed overhead amortized)
- Null result at BOTH scales suggests fundamental limitation, not just underpowered

### Proposed Follow-Up Study
**Title**: "Memory Augmentation: Scale Threshold Detection"

**Design**:
- Test at 50M, 100M, 500M params
- Track: when does memory overhead become worthwhile?
- Consider: task complexity may matter more than model size

**Effort**: ~4-6 weeks (larger compute requirements)

---

## 6. Character vs Subword for Reasoning Tasks

### Observation
Character-level tokenization may fundamentally limit reasoning architectures:
- COCONUT: Can't compress reasoning steps
- Memory: Retrieves individual characters, not concepts
- MoD: Every character matters equally

### Why Interesting
This suggests tokenization choice may determine which efficiency mechanisms can work. No previous study has systematically tested this.

### Proposed Follow-Up Study
**Title**: "Tokenization-Architecture Interactions in Efficiency Mechanisms"

**Design**:
- Fixed task (multi-step arithmetic)
- Variable tokenization: char, BPE-small (1K), BPE-medium (10K), BPE-large (50K)
- Variable architecture: baseline, +Memory, +MoD, +COCONUT
- Full factorial design

**Effort**: ~4 weeks

---

## Priority Ranking

| # | Finding | Novelty | Effort | Impact | Priority |
|---|---------|---------|--------|--------|----------|
| 1 | COCONUT + subword | High | Medium | High | **1** |
| 6 | Tokenization interactions | Very High | Medium-High | Very High | **2** |
| 2 | MoD capacity sweep | Medium | Low | Medium | 3 |
| 4 | MoD router analysis | Medium | Low | Medium | 4 |
| 3 | Memory variance | Low | Low | Low | 5 |
| 5 | Memory scale threshold | Medium | Very High | Medium | 6 |

---

## Immediate Next Steps

1. **COCONUT Subword Test** - Highest priority, directly addresses study limitation
2. **MoD 97-99% Capacity** - Quick win, could validate/invalidate MoD entirely
3. **Router Analysis** - Quick diagnostic, informs MoD conclusions

---

*This document captures serendipitous findings from the A+B+C study. Each item represents a potential paper-worthy investigation that emerged from negative/null results.*
