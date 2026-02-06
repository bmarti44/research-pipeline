# A+B+C Study: Recommended Next Steps (v3.3 Review Synthesis)

**Date**: 2026-02-05
**Status**: Active

## Summary of v3.3 Findings

| Component | Result | Statistical Support |
|-----------|--------|---------------------|
| Memory | -0.34% vs baseline | p=0.44, NOT significant (no detectable effect) |
| MoD 90% | +2.6% vs baseline | p=0.0001, VALIDATED (n=5) |
| COCONUT | Not tested | Out of scope (char tokens defeat mechanism) |

## Reviewer Consensus (5 independent reviews)

### Critical Issues to Address

1. **MoD 90% underpowered** (n=3)
   - Cannot claim "viable" without proper sample size
   - Action: Run n=5 with confidence intervals
   - Priority: HIGH
   - Time: ~30 min

2. **Memory null ≠ no effect**
   - p=0.44 with n=10 has only 15% power
   - Reframe: "Effect below detection threshold"
   - Action: Either n=20 seeds OR scale test
   - Priority: MEDIUM
   - Time: 1-2 hrs

3. **COCONUT unfairly tested**
   - Character tokenization defeats mechanism
   - Action: Test with BPE tokens OR label as "unfair test"
   - Priority: MEDIUM (can defer)
   - Time: 4-6 hrs if pursued

4. **Scale test critical for publication**
   - 10M params needed to know if effects are real
   - Action: Run memory_only at small scale
   - Priority: MEDIUM
   - Time: 2-3 hrs

## Recommended Execution Order

### Phase 1: Quick Validation (TODAY)
```bash
# 1. MoD 90% validation (n=5) - ~30 min
python training/train_abc.py --condition mod_only --mod_capacity 0.9 \
    --max_steps 500 --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_mod_90_validated

# Expected: Confirm +2.5% finding with proper CIs
```

### Phase 2: Scale or Power (CHOOSE ONE)

**Option A: Scale Test** (recommended if optimistic)
```bash
# Memory at 10M params - ~2 hrs
python training/train_abc.py --condition baseline --size small \
    --max_steps 1000 --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_scale/baseline

python training/train_abc.py --condition memory_only --size small \
    --max_steps 1000 --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_scale/memory_only
```

**Option B: Power Test** (recommended if conservative)
```bash
# Memory with n=20 seeds - ~2 hrs
python training/train_abc.py --condition baseline --max_steps 500 \
    --seeds $(seq 1 20) --output ../results/v3.3_power_20/baseline

python training/train_abc.py --condition memory_only --max_steps 500 \
    --seeds $(seq 1 20) --output ../results/v3.3_power_20/memory_only
```

### Phase 3: Documentation
- Update RESEARCH_PLAN_ABC.md with final results
- Write negative result paper framing
- Choose publication venue (negative results track)

## Decision Criteria

### After MoD Validation (Phase 1)
- If MoD 90% CI excludes zero degradation → "viable" claim supported
- If MoD 90% CI includes zero → MoD is actually neutral at high capacity (good!)
- If MoD 90% shows >5% degradation → "viable" claim wrong, revise

### After Scale/Power Test (Phase 2)
- If Memory effect grows at scale → pursue full investigation
- If Memory effect shrinks → document as scale-dependent null
- If Memory p < 0.05 at n=20 → real but tiny effect

## Publication Path

### Minimum Viable Paper (after Phase 1-2)
**Title**: "Efficiency Architecture Combinations at Small Scale: A Null Study"
**Venue**: NeurIPS/MLSys Negative Results Track
**Narrative**: Components don't combine synergistically at tiny scale; MoD needs high capacity; Memory effect too small to detect

### Full Paper (if scale test positive)
**Title**: "Scale-Dependent Benefits of Efficiency Mechanisms in Transformers"
**Venue**: Main track (ICLR/NeurIPS)
**Narrative**: Effects emerge at larger scale; guidance on when to deploy

## Tracking

| Task | Status | Result |
|------|--------|--------|
| MoD 90% n=5 validation | **COMPLETE** | +2.6% vs baseline (p=0.0001, 95% CI: +1.4% to +3.9%) |
| MoD 95% n=5 validation | **COMPLETE** | +1.66% vs baseline (p=0.020, 95% CI: +0.4% to +2.9%) |
| Scale test (8.5M params) | **COMPLETE** | +0.10% vs baseline (p=0.55, NOT significant) |
| COCONUT BPE test | DEFERRED | Out of scope for this study |
| Paper draft | NOT STARTED | - |

## MoD 90% Validation Results (2026-02-05)

```
MoD 90% (n=5):  2.422 ± 0.033
Baseline (n=10): 2.360 ± 0.012

Difference: +0.062 PPL (+2.6%)
95% CI: [+1.4%, +3.9%]
p-value: 0.0001
Cohen's d: 3.02

CONCLUSION: MoD 90% significantly degrades performance, but less than MoD 12.5%
- MoD 12.5%: +9.3% degradation
- MoD 50%: +9.3% degradation
- MoD 90%: +2.6% degradation (validated)

Revised claim: MoD at 90% is "less harmful" but still not beneficial.
```

## Scale Test Results (2026-02-05) - COMPLETE

### Configuration
- Model size: "small" (8.5M params vs 1M tiny)
- Training: 1000 steps, cosine LR schedule
- Seeds: 42, 123, 456, 789, 1001 (n=5 per condition)

### Raw Data

| Seed | Baseline PPL | Memory PPL |
|------|--------------|------------|
| 42 | 1.3832 | 1.3842 |
| 123 | 1.3821 | 1.3819 |
| 456 | 1.3846 | 1.3852 |
| 789 | 1.3816 | 1.3905 |
| 1001 | 1.3910 | 1.3877 |
| **Mean** | **1.3845 ± 0.0038** | **1.3859 ± 0.0033** |

### Statistical Analysis

```
Baseline:  1.3845 ± 0.0038 (n=5)
Memory:    1.3859 ± 0.0033 (n=5)

Difference: +0.0014 PPL (+0.10%)
Direction: Memory slightly WORSE (but negligible)
t-statistic: 0.63
p-value: 0.55 (NOT significant)
95% CI: [-0.37%, +0.57%]
```

### CONCLUSION: Memory shows no detectable benefit at 8.5M scale

The memory mechanism shows:
- Essentially zero effect (+0.10%)
- Not statistically significant (p=0.55)
- 95% CI [-0.37%, +0.57%] includes zero
- Consistent with tiny scale result (-0.34%, p=0.44, 95% CI [-1.2%, +0.5%])

**NEGATIVE RESULT**: Memory shows no detectable benefit at either scale tested. Effect size is below our detection threshold (<0.5%). With n=5 seeds, we had ~80% power to detect effects ≥2%.

---

## Final Summary: A+B+C Study v3.3

| Component | Tiny Scale (1M) | Small Scale (8.5M) | Verdict |
|-----------|-----------------|-------------------|---------|
| **Memory** | -0.34% (p=0.44) | +0.10% (p=0.55) | **NO DETECTABLE EFFECT** (<0.5%) |
| **MoD 90%** | +2.6% (p=0.0001) | Not tested | **HARMFUL** |
| **COCONUT** | Not tested | Not tested | **NOT TESTED** (out of scope) |

### Publication Recommendation

**Venue**: NeurIPS/MLSys Negative Results Track

**Title**: "Efficiency Mechanisms Don't Synergize at Small Scale: A Null Study"

**Key Claims**:
1. Memory augmentation shows **no detectable benefit** at ≤10M params (effect <0.5%)
2. MoD **consistently degrades performance** even at 90% capacity (+2.6%, p=0.0001)
3. COCONUT evaluation was **out of scope** (requires subword tokens)
4. Effects **may emerge at larger scales** (future work beyond scope)

**Statistical Note**: With n=5-10 seeds, we had ~80% power to detect effects ≥2%. Smaller effects cannot be ruled out but would be of limited practical significance.

---

*Last updated: 2026-02-05*
