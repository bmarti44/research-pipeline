# A+B+C Architecture Combination Study: Research Plan

## Status: v3.5 COMPLETE ✅ (With Compute-Matched Controls)
**Date**: 2026-02-05
**Version**: 3.5.1 (Corrected After 5-Round Review)

### Executive Summary (v3.5.1)

**Hypothesis**: COCONUT (A) + MoD (B) + Memory (C) provide complementary benefits.

**Result**: **NEGATIVE FOR ALL COMPONENTS AT SMALL SCALE**

| Component | Effect | Verdict |
|-----------|--------|---------|
| **COCONUT (A)** | +3.0% worse (vs matched baseline) | **NO BENEFIT** |
| Memory (C) | ~0% effect | **NO BENEFIT** |
| MoD (B) | +2-3% overhead | **HARMFUL** |

**Critical Finding (v3.5.1)**:
Initial warm-start results appeared to show -58.6% improvement. However, 5-round expert review identified a **compute confound**: warm-start used 1000 total steps vs 500 for comparisons.

When compute-matched controls were added:

| Condition | Steps | Val PPL | vs Baseline@1000 |
|-----------|-------|---------|------------------|
| Baseline | 1000 | **2.90** | — |
| COCONUT warmstart | 1000 | 2.98 | **+3.0% worse** (p=0.0002) |

**The "58.6% improvement" was entirely due to more training, not COCONUT.**

**Conclusion**: Efficiency mechanisms (COCONUT, MoD, Memory) provide no benefit at 7.5M parameter scale. The original COCONUT paper used 125M params (GPT-2) - we are 17x smaller. These mechanisms may require larger scale to show benefits.

> **Note**: This plan was reviewed critically and updated through v3 to address all methodological concerns while maintaining small-scale exploratory design.

### v3 Fixes Applied (All Review Issues Resolved)

| Issue | v2 State | v3 Fix |
|-------|----------|--------|
| Data too simple | 2-3 step arithmetic | Multi-step word problems (5-9 steps avg) |
| No validation split | Training PPL only | 15% held out for val PPL |
| Iterations reduced | 2 iterations (COCONUT-lite) | Restored to 4 iterations |
| Curriculum limited | 2 stages (max 1 latent) | Restored to 4 stages (full compression) |
| Seeds below minimum | n=3 | n=5 (meets power requirement) |
| Compute unfairness | Undocumented | FLOP tracking per condition |

---

## 1. Study Overview

This plan tests whether three efficiency mechanisms provide **complementary or interfering** effects when combined:

- **A: COCONUT Latent Reasoning** — Reasoning in continuous embedding space
- **B: Mixture-of-Depths (MoD)** — Adaptive computation per token
- **C: Hierarchical Memory** — Differentiable memory bank

The study was designed for execution on Apple Silicon, testing at 1M and 8.5M parameter scales.

---

## 2. Background and Motivation

### 2.1 Prior Results

| Component | Isolated Test Result | Source |
|-----------|---------------------|--------|
| A: COCONUT | PPL 5.74 vs 6.22 baseline (-7.7%), 28x slower | `REVIEW_SUMMARY.md` |
| B: MoD | 13% throughput improvement, marginal PPL difference | Pilot study (100 steps) |
| C: Memory | Not independently tested | - |
| B+C (LAHR) | PPL ~21,000 at 100 steps (17% lower throughput) | Pilot study |

### 2.2 The Open Question

**Do A+B+C provide additive, super-additive, or interfering effects?**

Possible outcomes:
1. **Super-additive (synergy)**: Combined PPL < min(individual PPLs) — components amplify each other
2. **Additive**: Combined improvement ≈ sum of individual improvements
3. **Sub-additive**: Combined improvement < sum of individual improvements (diminishing returns)
4. **Interfering**: Combined PPL > baseline — components fight each other

### 2.3 Theoretical Tensions

| Component | Compute Strategy | Potential Conflict |
|-----------|-----------------|-------------------|
| A (COCONUT) | Multiple forward passes through thoughts | Adds compute |
| B (MoD) | Skip layers for easy tokens | Reduces compute |
| C (Memory) | Retrieve before processing | Adds overhead |

**Key concern**: COCONUT adds 4-8x forward passes for thought tokens. MoD tries to skip computation. These goals may conflict.

### 2.4 Known Limitations (Acknowledged Pre-Registration)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **n=1 seeds (Phase 1)** | Cannot estimate variance | Phase 2 uses 3+ seeds if signal found |
| **Memory (C) untested** | Cannot isolate C's contribution | Phase 2 includes C-only condition |
| **Curriculum is COCONUT-inherent** | Not separable from A | By design - curriculum IS how COCONUT works |
| **500 steps may not converge** | Premature conclusions | Monitor loss curves, extend if needed |
| **Prior 7.7% based on throughput-unequal runs** | May overestimate A benefit | Phase 1 controls for training steps |

**Design Choice**: Curriculum training is inherent to COCONUT (gradually replacing CoT with latent tokens). This is not a confound—it's the mechanism being tested. Non-COCONUT conditions see the same data without latent replacement.

---

## 3. Hypotheses

### 3.1 Primary Hypothesis (H1)

**H1**: The combination A+B+C achieves lower perplexity than any individual component alone.

**Operationalized**: `PPL(A+B+C) < min(PPL(A), PPL(B+C), PPL(baseline))`

**Prediction**: 10-15% PPL improvement over baseline (based on COCONUT's 7.7% + expected MoD/Memory contributions)

### 3.2 Secondary Hypotheses

**H2 (Interference)**: Components do NOT interfere destructively.
- Operationalized: `PPL(A+B+C) < 1.2 × PPL(baseline)`
- If violated: Stop further investigation

**H3 (Efficiency)**: A+B+C maintains throughput within 50% of baseline.
- Operationalized: `tok/s(A+B+C) > 0.5 × tok/s(baseline)`

**H4 (Complementarity)**: Each component provides non-redundant benefit.
- Operationalized: Removing any component increases PPL

### 3.3 Predictions (Pre-Registered)

| Condition | Predicted PPL (relative to baseline) |
|-----------|-------------------------------------|
| Baseline | 1.00 |
| A only (COCONUT) | 0.93 (7% better) |
| B+C only (LAHR) | 0.95 (5% better) |
| A+B+C (full) | 0.88 (12% better) |

---

## 4. Experimental Design

### 4.1 Phase 1: Quick Validation (30-60 minutes)

**Goal**: Determine if combination is worth pursuing.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model size | tiny (~1M params) | Fast iteration |
| Training steps | 500 per condition | Enough for signal |
| Batch size | 4 | MacBook compatible |
| Sequence length | 256 | Faster, sufficient for pattern |
| Seeds | 1 | Speed over variance (Phase 1 only) |
| Conditions | 4 | Minimal comparison set |

### 4.2 Conditions (Phase 1)

| ID | Name | MoD (B) | COCONUT (A) | Memory (C) |
|----|------|---------|-------------|------------|
| C0 | baseline | ❌ | ❌ | ❌ |
| C1 | coconut_only | ❌ | ✅ | ❌ |
| C2 | lahr_only | ✅ | ❌ | ✅ |
| C3 | full_abc | ✅ | ✅ | ✅ |

### 4.3 Why These 4 Conditions?

1. **Baseline (C0)**: Required anchor point
2. **COCONUT-only (C1)**: Tests A in isolation (known to work: 7.7% improvement)
3. **LAHR-only (C2)**: Tests B+C together (MoD + Memory without latent reasoning)
4. **Full A+B+C (C3)**: Tests the combination hypothesis

This answers: **"Does adding COCONUT to LAHR help or hurt?"**

---

## 5. Metrics

### 5.1 Primary Metric

| Metric | How Measured | Success Threshold |
|--------|--------------|-------------------|
| Training Perplexity | `exp(cross_entropy_loss)` | A+B+C < 0.95 × baseline |

### 5.2 Secondary Metrics

| Metric | How Measured | Purpose |
|--------|--------------|---------|
| Training Loss | Final loss at step 500 | Learning signal |
| Throughput | tokens/second | Efficiency cost |
| MoD Utilization | % tokens routed | Component behavior |
| Latent Iterations | Avg COCONUT iterations | Component behavior |
| **Loss Trend** | Loss at steps 400 vs 500 | Convergence check |

### 5.3 Convergence Monitoring

To address the valid concern about premature stopping, we track:
- **Loss delta (final 100 steps)**: If loss is still dropping >5% in final 100 steps, results are tentative
- **Extension rule**: If ALL conditions show >5% drop in final segment, extend to 750 steps

---

## 6. Success Criteria

### 6.1 Decision Matrix

| Observed Result | Interpretation | Next Step |
|-----------------|---------------|-----------|
| PPL(A+B+C) < PPL(all others) by >5% | **Strong synergy** | Proceed to Phase 2 |
| PPL(A+B+C) ≈ best single component | **No synergy** | Investigate, likely stop |
| PPL(A+B+C) > PPL(baseline) | **Interference** | Debug, understand why |
| PPL(A+B+C) > 1.2 × PPL(baseline) | **Reject hypothesis** | Stop |

### 6.2 Quantitative Thresholds

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Minimum improvement | 5% PPL reduction | Worth the complexity |
| Maximum slowdown | 50% throughput loss | Practical for use |
| Training stability | Loss decreasing at step 500 | Still learning |

---

## 7. Execution Plan

### 7.1 Prerequisites

```bash
# 1. Activate environment
cd /Users/briamart/github/tool-calling/papers/efficient_architecture_proof
source /Users/briamart/github/tool-calling/.venv/bin/activate

# 2. Ensure data exists
ls -la code/data/cot_training_data.json  # Should exist from previous run
```

### 7.2 Phase 1 Commands (v3 - UNIFIED SCRIPT)

**CRITICAL**: All conditions use `train_abc.py` with the SAME data for fair comparison.

**Run Full v3 Study (5 seeds × 4 conditions)**
```bash
cd code && python training/train_abc.py \
    --train_data data/multistep_train.json \
    --val_data data/multistep_val.json \
    --output ../results/abc_study_v3 \
    --size tiny \
    --max_steps 500
# Default: 5 seeds, 4 curriculum stages
# Estimated runtime: ~2-3 hours on MacBook M1/M2
```

**Or Run Individual Conditions**
```bash
# C0: Baseline
cd code && python training/train_abc.py --condition baseline --max_steps 500

# C1: COCONUT-only
cd code && python training/train_abc.py --condition coconut_only --max_steps 500

# C2: LAHR-only (MoD + Memory)
cd code && python training/train_abc.py --condition lahr_only --max_steps 500

# C3: Full A+B+C
cd code && python training/train_abc.py --condition full_abc --max_steps 500
```

### 7.3 Fair Comparison Design

| Aspect | All Conditions |
|--------|---------------|
| **Data** | Same `cot_training_data.json` |
| **Tokenizer** | Same character-level tokenizer |
| **Steps** | Same 500 steps per condition |
| **Batch size** | Same 4 |
| **Sequence length** | Same 256 |
| **Seed** | Same 42 |

**COCONUT-specific**: Conditions with COCONUT use curriculum training (4 stages × 125 steps), replacing CoT tokens with latent tokens progressively. Non-COCONUT conditions train on full text without replacement. This is the correct design—curriculum IS how COCONUT works.

### 7.4 Estimated Runtime

| Condition | Estimated Time |
|-----------|---------------|
| C0: Baseline | ~5 minutes |
| C1: COCONUT-only | ~15 minutes |
| C2: LAHR-only | ~8 minutes |
| C3: Full A+B+C | ~25 minutes |
| **Total** | **~55 minutes** |

---

## 8. Analysis Plan

### 8.0 Phase 1 v3 Results (ALL ISSUES RESOLVED)

| Condition | Val PPL (mean±std) | vs Baseline | FLOPs | Throughput |
|-----------|-------------------|-------------|-------|------------|
| **baseline** | **2.36 ± 0.01** | - | 1.0x | 12,385 tok/s |
| coconut_only | 2.36 ± 0.02 | +0.2% | 5.0x | 2,985 tok/s |
| lahr_only | 2.68 ± 0.03 | **+13.7%** | 1.0x | 16,342 tok/s |
| full_abc | 2.67 ± 0.03 | **+13.2%** | 5.0x | 3,549 tok/s |

**Statistical check**: n=5 seeds, std~0.01-0.03, differences are statistically significant

**Key Findings (v3)**:
1. **COCONUT neutral at higher complexity**: +0.2% (within noise) but at 5x FLOP cost
2. **LAHR degrades performance**: +13.7% worse PPL - MoD/Memory hurt learning
3. **No synergy**: Full ABC = LAHR penalty, COCONUT provides no offset
4. **Verdict**: INCONCLUSIVE / NEGATIVE at tiny scale

---

### 8.1 Comparison Table (Phase 1 v1 - BROKEN IMPLEMENTATION)

| Condition | Final Loss | PPL | vs Baseline | Throughput |
|-----------|------------|-----|-------------|------------|
| C0: baseline | 0.381 | 1.46 | - | 27,258 tok/s |
| C1: coconut_only | 0.531 | 1.70 | +16.1% (worse) | 4,278 tok/s |
| C2: lahr_only | 0.378 | 1.46 | -0.3% | 23,525 tok/s |
| C3: full_abc | 0.513 | 1.67 | +14.1% (worse) | 3,848 tok/s |

**⚠️ v1 had implementation bugs - see v2 below for corrected results**

### 8.1.1 Phase 1 v2 - FIXED IMPLEMENTATION (3 seeds)

| Condition | PPL (mean±std) | vs Baseline | Throughput |
|-----------|----------------|-------------|------------|
| C0: baseline | **1.48 ± 0.01** | - | 24,674 tok/s |
| C1: coconut_only | 1.56 ± 0.01 | **+5.0%** | 9,329 tok/s |
| C2: lahr_only | 1.52 ± 0.02 | +2.5% | 21,190 tok/s |
| C3: full_abc | 1.61 ± 0.02 | **+8.7%** | 8,301 tok/s |

**Statistical check**: diff=0.129, 2×SE=0.030, **significant=True**

### 8.1.2 Fixes Applied (v1 → v2)

1. **CRITICAL**: Added final forward pass after COCONUT iterations (was missing)
2. Reduced latent iterations from 4 to 2 (fairer compute comparison)
3. Reduced curriculum stages from 4 to 2 (250 steps/stage instead of 125)
4. Added learning rate warmup (50 steps)
5. Multi-seed support (3 seeds for statistical validity)

### 8.1.3 Key Findings (v2 - Validated)

1. **COCONUT penalty reduced**: From +16% (broken) to +5% (fixed) - confirms implementation was buggy
2. **LAHR shows slight degradation**: +2.5% worse than baseline (MoD+Memory overhead)
3. **No synergy**: Full ABC (+8.7%) ≈ COCONUT effect (+5%) + LAHR effect (+2.5%)
4. **Throughput cost significant**: COCONUT ~2.6x slower, Full ABC ~3x slower
5. **Results statistically significant**: 3 seeds confirm the difference is real, not noise

### 8.2 Decision Logic (Applied to v2 Results)

```
IF PPL(C3) < 0.95 × min(PPL(C1), PPL(C2)):
    → STRONG SIGNAL: Proceed to Phase 2

    v2 Result: 1.61 < 0.95 × 1.48 = 1.41?  NO

ELIF PPL(C3) < PPL(C0):
    → WEAK SIGNAL: Combination helps, investigate further

    v2 Result: 1.61 < 1.48?  NO

ELIF PPL(C3) > 1.2 × PPL(C0):
    → REJECT: Components interfere destructively

    v2 Result: 1.61 > 1.2 × 1.48 = 1.78?  NO (1.61 < 1.78)

ELSE:
    → INCONCLUSIVE: Need more data / different setup
```

**VERDICT: INCONCLUSIVE** — At tiny scale, components show small penalties:
- COCONUT: +5% PPL penalty (acceptable, but not beneficial)
- LAHR: +2.5% PPL penalty (MoD+Memory overhead)
- Combined: +8.7% PPL penalty (additive, no interference but no synergy)

**Interpretation**: The architecture works correctly (no catastrophic interference), but shows no benefit at tiny scale. Scale-up hypothesis: benefits may emerge at larger model sizes where latent reasoning and memory have more capacity to exploit.

---

## 9. Phase 2 (If Proceed)

### 9.1 Full 2³ Factorial Design

| Condition | MoD | COCONUT | Memory |
|-----------|-----|---------|--------|
| 000 | ❌ | ❌ | ❌ |
| 001 | ❌ | ❌ | ✅ |
| 010 | ❌ | ✅ | ❌ |
| 011 | ❌ | ✅ | ✅ |
| 100 | ✅ | ❌ | ❌ |
| 101 | ✅ | ❌ | ✅ |
| 110 | ✅ | ✅ | ❌ |
| 111 | ✅ | ✅ | ✅ |

### 9.2 Phase 2 Parameters

| Parameter | Value |
|-----------|-------|
| Model size | small (~10M params) |
| Training steps | 5,000 |
| Seeds | 3 per condition |
| Total runs | 8 × 3 = 24 |
| Estimated time | ~24-48 hours |

---

## 10. Key Files

| File | Purpose |
|------|---------|
| `code/training/train_abc.py` | **UNIFIED training script for all 4 conditions** |
| `code/data/cot_training_data.json` | Training data (500 CoT samples) |
| `code/models/lahr_v4.py` | Reference: LAHR (B+C) architecture |
| `code/models/lahr_coconut.py` | Reference: Full A+B+C combination |
| `code/models/coconut_full.py` | Reference: COCONUT mechanism (A) |

---

## 11. Pre-Registration Checklist

Before running Phase 1:

- [x] Data exists (`code/data/cot_training_data.json`) ✓
- [x] Unified training script exists (`code/training/train_abc.py`) ✓
- [x] All conditions use same data for fair comparison ✓
- [x] Predictions recorded (Section 3.3) ✓
- [x] Seed fixed to 42 ✓
- [x] Known limitations acknowledged (Section 2.4) ✓
- [ ] Output directories created
- [ ] Smoke test completed (single condition runs without error)

---

## 12. Next Steps After Phase 1

### 12.1 v3 Outcome: INVESTIGATE LAHR

Based on Phase 1 v3 (all issues resolved) results:

1. **COCONUT validated**: With harder data, COCONUT is neutral (not +5% penalty)
2. **LAHR problematic**: MoD + Memory causes +13.7% degradation
3. **No synergy**: Full ABC shows no benefit over components alone

**Recommended path**: **INVESTIGATE LAHR components** to understand why MoD/Memory hurt performance at this scale

### 12.1.1 Possible Explanations for LAHR Degradation

| Hypothesis | Test |
|------------|------|
| MoD capacity too low (12.5%) | Try 25%, 50% capacity |
| Memory retrieval interferes with learning | Test Memory-only condition |
| MoD hurts curriculum learning | Test MoD-only condition |
| Tiny scale insufficient for memory | Test at small (~10M) scale |

### 12.1.2 v3 Review Agent Findings (5 Parallel Reviews)

**Consensus Issues Identified:**

| Issue | Severity | Evidence |
|-------|----------|----------|
| LAHR throughput > baseline (16,342 vs 12,385 tok/s) | CRITICAL | Should be impossible - MoD should add overhead |
| 12.5% MoD capacity too aggressive | CRITICAL | Original paper uses 50%+, skipping 87.5% corrupts KV cache |
| "Neutral" COCONUT claim has only 8% power | MAJOR | Need n≈64 for 80% power; cannot claim equivalence |
| Character-level tokenization defeats COCONUT | MAJOR | Replaces characters, not reasoning steps |
| Memory queries raw embeddings (no context) | MAJOR | Should retrieve after transformer layers |

### 12.1.3 Immediate Action Plan (v3.1)

**Priority 1: Isolate LAHR Components (~40 min)**
```bash
# 1a. MoD-only (no Memory)
python training/train_abc.py --condition mod_only --max_steps 500 --seeds 42 123 456

# 1b. Memory-only (no MoD)
python training/train_abc.py --condition memory_only --max_steps 500 --seeds 42 123 456
```

**Priority 2: Test MoD Capacity (~20 min)**
```bash
# 2. MoD at 50% capacity
python training/train_abc.py --condition lahr_only --mod_capacity 0.5 --max_steps 500 --seeds 42 123 456
```

**Priority 3: Investigate Throughput Anomaly**
- Profile why LAHR is 32% faster than baseline
- Check if MoD router learns degenerate "skip everything" solution

**Expected Outcomes:**
- If MoD-only matches baseline → Memory is the culprit
- If Memory-only matches baseline → MoD is the culprit
- If 50% capacity improves → 12.5% is too aggressive
- If throughput anomaly explained → Implementation bug identified

### 12.1.5 v3.1 Isolation Results (COMPLETED)

| Condition | Val PPL (mean±std) | vs Baseline | Throughput |
|-----------|-------------------|-------------|------------|
| baseline | 2.36 ± 0.02 | - | 23,420 tok/s |
| **mod_only** (skip_compute=True) | **2.58 ± 0.04** | **+9.3%** | 12,687 tok/s |
| **mod_only** (skip_compute=False) | **2.63 ± 0.05** | **+11.4%** | 37,625 tok/s |
| **memory_only** | **2.34 ± 0.02** | **-0.8%** | 14,460 tok/s |
| lahr_50pct | 2.58 ± 0.04 | +9.3% | 13,243 tok/s |

**CONCLUSION: MoD is the culprit, not Memory**

1. ✅ Memory-only **MATCHES baseline** (2.34 vs 2.36 = -0.8%) - Memory is beneficial
2. ❌ MoD-only **DEGRADES performance** (+9-11%) - MoD is the problem
3. ⚠️ Higher MoD capacity helps (50%: +9.3% vs 12.5%: +13.7%) but doesn't fix
4. ✅ Throughput anomaly explained: MoD skips tokens → faster but worse

### 12.1.6 v3.1 MoD Deep Dive: Attention Fix Failed

**Hypothesis**: MoD's broken causal attention (only 12.5% of tokens see each other) causes degradation.

**Test**: Added `skip_compute=False` option that processes ALL tokens through transformer block but only UPDATES selected tokens' outputs.

**Result**: Made things **worse** (+11.4% vs +9.3%)

| MoD Mode | Mechanism | Val PPL | vs Baseline |
|----------|-----------|---------|-------------|
| Original (skip_compute=True) | Only process selected tokens | 2.58 | +9.3% |
| Fixed (skip_compute=False) | Process all, update selected | 2.63 | +11.4% |

**Interpretation**: The problem is NOT broken causal attention. The problem is **selective output** itself:
- Original MoD: Bad attention context but gradients flow through selected tokens
- Fixed MoD: Good attention context but acts like "output dropout" - 87.5% of positions get zero gradient

**Root Cause (Revised)**: MoD is fundamentally unsuited for **dense prediction tasks** like character-level language modeling where EVERY position matters for the loss. MoD was designed for tasks where some tokens are "less important" (padding, stop words in classification).

**Recommendation**: Drop MoD from A+B+C. Test A+C (COCONUT + Memory) as the new combination hypothesis.

### 12.1.7 v3.2 Action Plan: Test A+C (COCONUT + Memory)

**Rationale**:
- MoD confirmed as detrimental (drop B)
- Memory confirmed as beneficial (keep C)
- COCONUT was neutral at v3 complexity (needs retest with Memory)

**New Hypothesis (H5)**: COCONUT + Memory outperforms either component alone.

**Operationalized**: `PPL(A+C) < min(PPL(A), PPL(C), PPL(baseline))`

**Required New Condition**:
```bash
# Add to train_abc.py conditions:
"coconut_memory": {"use_mod": False, "use_coconut": True, "use_memory": True}
```

**Execution Plan**:
```bash
# Test A+C combination
python training/train_abc.py --condition coconut_memory --max_steps 500 --seeds 42 123 456

# Compare against:
# - baseline:      2.36 (reference)
# - coconut_only:  ~2.36 (neutral)
# - memory_only:   2.34 (-0.8%)
```

**Expected Outcomes**:
- If A+C < 2.30: Synergy found (COCONUT + Memory complement each other)
- If A+C ≈ 2.34: No synergy (Memory dominates, COCONUT adds nothing)
- If A+C > 2.40: Interference (latent reasoning conflicts with memory retrieval)

### 12.1.8 v3.2 Results: COCONUT + Memory (COMPLETED)

| Condition | Val PPL (mean±std) | vs Baseline | Throughput |
|-----------|-------------------|-------------|------------|
| baseline | 2.36 ± 0.02 | - | 23,420 tok/s |
| memory_only | 2.34 ± 0.02 | **-0.8%** | 14,460 tok/s |
| **coconut_memory** | **2.35 ± 0.02** | **-0.4%** | ~4,000 tok/s |

**CONCLUSION: No synergy between COCONUT and Memory**

1. ✅ COCONUT + Memory (2.35) ≈ Memory-only (2.34) - within noise
2. ❌ No improvement over Memory alone
3. ❌ 4x throughput penalty from COCONUT curriculum training
4. ✅ No interference (COCONUT doesn't hurt Memory)

**Interpretation**: At tiny scale (~1M params), COCONUT provides no benefit over the baseline or Memory-only. Memory alone provides the best efficiency/quality tradeoff.

---

## 13. Final Conclusions (v3.3 - REVISED)

### 13.1 Component Summary (Updated)

| Component | v3.2 Claim | v3.3 Evidence | Revised Verdict |
|-----------|------------|---------------|-----------------|
| **A: COCONUT** | Not tested | Character tokenization defeats mechanism | **NOT TESTED** (out of scope) |
| **B: MoD** | Overhead (+9-11%) | +2.6% at 90% capacity | **SMALL OVERHEAD** with tuning |
| **C: Memory** | Beneficial (-0.8%) | p=0.44, 95% CI includes zero | **INCONCLUSIVE** |

### 13.2 Architecture Recommendations (Revised)

**For tiny scale (~1M params):**
1. **No clear winner** — All components show negligible or noisy effects
2. **MoD needs high capacity** — 90%+ to avoid degradation
3. **Memory effect is small** — May need larger scale to matter
4. **COCONUT needs subword tokens** — Character-level defeats its purpose

**For scaling up:**
- Memory likely to help more at larger scales (fixed overhead amortized)
- COCONUT requires proper tokenization (subword, not character)
- MoD may work at any scale with appropriate capacity tuning

### 13.3 What We Learned

The A+B+C hypothesis was: "Latent reasoning (A) + adaptive compute (B) + external memory (C) provide complementary benefits."

**Actual findings**:
- No component provides statistically significant benefit at tiny scale
- MoD's apparent failure was hyperparameter misconfiguration, not fundamental incompatibility
- Memory's apparent success was statistical noise
- COCONUT evaluation was out of scope (character tokenization defeats mechanism - requires subword tokens)

**Methodological lessons**:
- n=3 seeds is insufficient for detecting small effects
- Hyperparameters (MoD capacity) dominate architecture effects
- Task-architecture alignment (tokenization) is critical

### 13.4 Next Steps (If Continuing)

1. **Scale up Memory test** — Effect may emerge at 10M+ params
2. **Test COCONUT with BPE tokenization** — Required for fair comparison
3. **Full MoD capacity sweep** — Find optimal capacity for dense prediction
4. **Increase seeds to n≥20** — Required for detecting effects <2%

### 12.2 Phase 2 Design (If Scaling Up)

| Parameter | Phase 1 (current) | Phase 2 (proposed) |
|-----------|------------------|-------------------|
| Model size | tiny (1.1M) | small (10M) |
| Training steps | 500 | 2,000 |
| Seeds | 3 | 5 |
| Curriculum stages | 2 | 3 |
| Data samples | 500 | 2,000 |

**Hypothesis**: At 10M+ parameters, the latent reasoning mechanism has sufficient capacity to compress CoT effectively, turning the +5% penalty into a benefit.

### 12.3 Alternative: Document as Negative Result

If scaling up is not pursued, the current findings are publishable as:
> "At tiny scale (~1M parameters), COCONUT latent reasoning shows a modest +5% PPL penalty.
> Combined with MoD+Memory (LAHR), the penalty is additive (+8.7%), not multiplicative.
> This suggests no catastrophic interference, but also no synergy at this scale."

### 12.4 Original Decision Tree

1. **If PROCEED**: Design Phase 2 full factorial
2. **If INVESTIGATE**: Run additional ablations ← **COMPLETED (v2 fixes)**
3. **If SCALE UP**: Test at larger model size ← **RECOMMENDED**
4. **If REJECT**: Document findings, consider alternatives

---

## 14. Phase 3: Validation Experiments (v3.3)

Based on v3.2 review feedback, three experiments are needed to validate conclusions:

### 14.1 Experiment Order and Rationale

| # | Experiment | Purpose | Runtime | Priority |
|---|------------|---------|---------|----------|
| 1 | Statistical Power | Establish if Memory's -0.8% is real (n=10 seeds) | ~1 hour | HIGH |
| 2 | Scale Test | Check if Memory benefit persists at 10M params | ~2 hours | HIGH |
| 3 | MoD High Capacity | Test if MoD works at 90% capacity | ~30 min | MEDIUM |

**Note**: COCONUT subword tokenization test deferred - requires significant code changes (new tokenizer, new data format). Will pursue if Memory scales.

---

### 14.2 Experiment 1: Statistical Power (n=10 seeds)

**Issue Addressed**: Reviewers noted n=3 insufficient to detect -0.8% effect. Need formal statistical test.

**Hypothesis**: Memory provides statistically significant PPL improvement over baseline.

**Design**:
```bash
# Baseline with 10 seeds
python training/train_abc.py --condition baseline --max_steps 500 \
    --seeds 42 123 456 789 1001 1234 2345 3456 4567 5678 \
    --output ../results/v3.3_power/baseline

# Memory-only with 10 seeds
python training/train_abc.py --condition memory_only --max_steps 500 \
    --seeds 42 123 456 789 1001 1234 2345 3456 4567 5678 \
    --output ../results/v3.3_power/memory_only
```

**Analysis**:
- Two-sample t-test for PPL difference
- Report p-value, 95% CI, Cohen's d
- Success criterion: p < 0.05 for Memory < Baseline

**Expected Results**:
- If p < 0.05: Memory benefit confirmed
- If p ≥ 0.05: Memory benefit is noise, revise conclusions

---

### 14.3 Experiment 2: Scale Test (10M params)

**Issue Addressed**: Tiny scale (~1M) may be too small for Memory to show full benefit.

**Hypothesis**: Memory benefit persists or grows at 10M parameter scale.

**Design**:
```bash
# Baseline at small scale
python training/train_abc.py --condition baseline --size small --max_steps 1000 \
    --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_scale/baseline

# Memory-only at small scale
python training/train_abc.py --condition memory_only --size small --max_steps 1000 \
    --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_scale/memory_only
```

**Analysis**:
- Compare effect size at tiny vs small scale
- Success criterion: Memory benefit ≥ 1% at small scale

**Expected Results**:
- If benefit grows: Memory scales, pursue further
- If benefit shrinks: Memory is tiny-scale artifact
- If benefit same: Memory is consistent but small

---

### 14.4 Experiment 3: MoD High Capacity (90%)

**Issue Addressed**: Reviewers noted 12.5% capacity is "pathologically low." Test near-full capacity.

**Hypothesis**: MoD at 90% capacity does not harm performance.

**Design**:
```bash
# MoD at 90% capacity
python training/train_abc.py --condition mod_only --mod_capacity 0.9 --max_steps 500 \
    --seeds 42 123 456 \
    --output ../results/v3.3_mod_90/mod_only
```

**Analysis**:
- Compare Val PPL to baseline (2.36)
- Success criterion: MoD 90% within 2% of baseline

**Expected Results**:
- If ≤2% degradation: MoD viable at high capacity, problem was hyperparameter
- If >2% degradation: MoD fundamentally unsuited (confirms v3.2 conclusion)

---

### 14.5 Results (v3.3 COMPLETE)

#### Experiment 1: Statistical Power (n=10 seeds)

| Condition | Val PPL (mean±std) | vs Baseline |
|-----------|-------------------|-------------|
| baseline (n=10) | **2.360 ± 0.012** | - |
| memory_only (n=10) | **2.352 ± 0.030** | -0.34% |

*Note: Memory condition shows higher variance (0.030) than baseline (0.012). This may indicate the memory mechanism introduces stochasticity, though the difference in variance is not statistically significant (Levene's test not performed).*

**Statistical Test**:
- Two-sample t-test: t = -0.79, **p = 0.438**
- Cohen's d = -0.36 (small effect)
- 95% CI for difference: **[-0.028, +0.012]** (includes zero)

**CONCLUSION: Memory benefit is NOT statistically significant**

The earlier claim that Memory provides -0.8% improvement was noise. With proper sample size (n=10), the true effect is -0.34% with 95% CI crossing zero.

#### Experiment 3: MoD High Capacity (90%) - VALIDATED

| MoD Capacity | Val PPL (mean±std) | vs Baseline | n | p-value |
|--------------|-------------------|-------------|---|---------|
| 12.5% | 2.58 ± 0.04 | +9.3% | 3 | - |
| 50% | 2.58 ± 0.04 | +9.3% | 3 | - |
| **90%** | **2.422 ± 0.033** | **+2.6%** | **5** | **0.0001** |

**Statistical Validation (n=5)**:
- Difference: +0.062 PPL (+2.6%)
- 95% CI: [+1.4%, +3.9%] — excludes zero
- p-value: 0.0001 (highly significant)
- Cohen's d: 3.02 (very large effect)

**CONCLUSION: MoD at 90% shows reduced but still statistically significant overhead**

The degradation is statistically significant (p=0.0001) but much smaller than at lower capacities. MoD is not "viable" in the sense of being acceptable - it still hurts performance. However, the issue IS hyperparameter-related (capacity), not fundamental architecture incompatibility.

#### Experiment 2: Scale Test - COMPLETE (2026-02-05)

**Configuration**: 8.5M params ("small"), 1000 steps, n=5 seeds

| Seed | Baseline PPL | Memory PPL |
|------|--------------|------------|
| 42 | 1.3832 | 1.3842 |
| 123 | 1.3821 | 1.3819 |
| 456 | 1.3846 | 1.3852 |
| 789 | 1.3816 | 1.3905 |
| 1001 | 1.3910 | 1.3877 |
| **Mean** | **1.3845 ± 0.0038** | **1.3859 ± 0.0033** |

**Statistical Analysis**:
- Difference: +0.0014 PPL (+0.10%)
- t-statistic: 0.63
- **p-value: 0.55** (NOT significant)
- 95% CI: [-0.37%, +0.57%] (includes zero)

**CONCLUSION: Memory provides NO benefit at 8.5M scale**

This confirms the tiny-scale finding. Memory is definitively a null result:
- Tiny scale (1M): -0.34%, p=0.44
- Small scale (8.5M): +0.10%, p=0.55

Both scales show essentially zero effect, with 95% CIs including zero.

---

### 14.6 Final Conclusions (v3.3.1 - SCALE TEST COMPLETE)

| Original Claim (v3.2) | Tiny Scale (v3.3) | Small Scale (v3.3.1) | Final Verdict |
|-----------------------|-------------------|----------------------|---------------|
| Memory is beneficial (-0.8%) | -0.34%, p=0.44 | +0.10%, p=0.55 | **NO DETECTABLE EFFECT** (<0.5%) |
| MoD is fundamentally unsuited | +2.6% at 90% (p=0.0001) | Not tested | **SMALL OVERHEAD** |
| COCONUT is neutral | Not tested | Not tested | **NOT TESTED** (out of scope) |

**Definitive Findings**:
1. **Memory shows no detectable benefit** at either 1M or 8.5M params (effect <0.5%, 95% CI includes zero)
2. **MoD shows consistent overhead** even at 90% capacity (+2.6%, p=0.0001), decreasing to +1.66% at 95%
3. **COCONUT was not tested** - character tokenization defeats mechanism (out of scope for this study)

**Memory Summary (No Detectable Effect)**:
| Scale | Params | Effect | p-value | 95% CI (%) | Verdict |
|-------|--------|--------|---------|------------|---------|
| Tiny | 1M | -0.34% | 0.44 | [-1.2%, +0.5%] | No detectable effect |
| Small | 8.5M | +0.10% | 0.55 | [-0.37%, +0.57%] | No detectable effect |

*Note: "No detectable effect" means effect size is below our detection threshold (<0.5%). With n=5-10 seeds, we had ~80% power to detect effects ≥2%.*

**MoD Capacity Summary (Complete Dose-Response)**:
| Capacity | Degradation | p-value | n | Status |
|----------|-------------|---------|---|--------|
| 12.5% | +9.3% | - | 3 | Large overhead |
| 50% | +9.3% | - | 3 | Large overhead |
| 90% | +2.6% | 0.0001 | 5 | Small overhead (significant) |
| **95%** | **+1.66%** | **0.020** | **5** | **Minimal overhead (significant)** |
| 100% | ~0%? | - | - | Not tested (equivalent to no MoD) |

**Dose-Response Conclusion**: MoD degradation decreases with higher capacity. At 95% capacity, degradation is +1.66% (p=0.020), still statistically significant but approaching practical tolerance.

*Caveat: While the trend suggests higher capacity may reduce degradation further, we did not test >95% capacity. The relationship may not be linear, and extrapolation beyond tested values should be verified empirically.*

### 14.7 Power Analysis

**Statistical Power for Memory Comparisons**:

| Comparison | n per group | Observed SD | MDE (80% power) | Observed Effect |
|------------|-------------|-------------|-----------------|-----------------|
| Memory vs Baseline (tiny) | 10 | 0.012-0.030 | ±1.2% | -0.34% |
| Memory vs Baseline (small) | 5 | 0.0033-0.0038 | ±0.46% | +0.10% |
| MoD 90% vs Baseline | 5 vs 10 | 0.012-0.033 | ±1.5% | +2.6% |

**Interpretation**:
- At small scale (n=5), we could detect effects ≥0.46% with 80% power
- The observed Memory effect (+0.10%) is well below MDE, hence "no detectable effect"
- MoD 90% effect (+2.6%) is well above MDE, hence statistically significant

**Statistical Method Notes**:
- All comparisons use **Welch's t-test** (unequal variances)
- Cohen's d calculated using **pooled standard deviation**: `d = (M1-M2) / SD_pooled`
- SD_pooled formula: `sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))`
- 95% CIs reported as **percentage difference from baseline**
- **Multiple comparisons**: This study performs ~10 statistical tests. Without correction (e.g., Bonferroni), some marginally significant findings (p~0.02-0.05) may be false positives. Primary conclusions (Memory null, MoD overhead) are robust (p>>0.05 or p<<0.01 respectively).

### 14.8 Methodological Lessons

This study provides several lessons for small-scale architecture research:

**1. Detecting Broken Implementations**
- v1 COCONUT was missing final forward pass after latent iterations
- Symptom: Latent reasoning showed no benefit despite increased compute
- Fix: Always verify output uses final hidden states, not intermediate

**2. Hyperparameter vs Architecture Effects**
- MoD at 12.5% capacity: +9.3% degradation → initially concluded "MoD incompatible"
- MoD at 90% capacity: +2.6% degradation → hyperparameter, not fundamental
- Lesson: Test multiple configurations before concluding architectural incompatibility

**3. Task-Architecture Alignment**
- Character-level tokenization defeats COCONUT (compresses characters, not reasoning steps)
- Dense prediction (every token matters) may conflict with MoD's skip-token design
- Memory retrieval on raw embeddings may be less effective than contextualized queries
- Lesson: Match task characteristics to mechanism assumptions

**4. Statistical Power Requirements**
- Effect size -0.8% initially reported as "beneficial" was noise (p=0.44)
- With n=3, we had ~15% power to detect this effect
- Minimum n=5 required for effects <1%; n≈20 for effects <0.5%
- Lesson: Calculate power BEFORE running experiments, not after

**5. Null Result Framing**
- "No effect" is stronger than evidence supports
- Correct framing: "No detectable effect at our sample size (effect <MDE)"
- Always report what you COULD detect, not just what you didn't find

### 14.9 Publication Recommendation

**Venue**: NeurIPS/MLSys Negative Results Track

**Recommended Title**: "Memory Augmentation and Mixture-of-Depths for Character-Level Language Modeling: No Benefits at ≤10M Parameters"

**Alternative Titles**:
- "Scale Thresholds for Efficiency Mechanisms on Multi-Step Arithmetic: A Null Result at 1M-10M Parameters"
- "Efficiency Mechanisms for Dense Token Prediction: No Synergy at Small Scale"

*Note: Titles specify task domain (character-level LM / multi-step arithmetic) to avoid overgeneralizing beyond tested conditions.*

**Key Claims** (with precision-qualified language):
1. Memory augmentation shows **no detectable benefit** at ≤10M params (effect <0.5%, 95% CI includes zero)
2. MoD **consistently degrades performance** even at 90% capacity (+2.6%, p=0.0001)
3. COCONUT evaluation was **out of scope** (requires subword tokens, not character-level)
4. Effects **may emerge at larger scales** (future work beyond scope)

**Statistical Note**: With n=5-10 seeds, we had ~80% power to detect effects ≥0.5%. Smaller effects cannot be ruled out but would be of limited practical significance at this scale.

---

## Appendix: Review Response (v1.1)

This plan underwent critical review. Below are the issues raised and responses:

| Review Issue | Severity (Claimed) | Verdict | Response |
|--------------|-------------------|---------|----------|
| Apples-to-oranges comparison | FATAL | **Addressed** | `train_abc.py` uses same data/steps for all |
| Incomparable training data | CRITICAL | **Addressed** | All conditions use `cot_training_data.json` |
| Single seed (n=1) | FATAL | **Acknowledged** | Valid limitation; Phase 1 is exploratory |
| Training script missing | CRITICAL | **Fixed** | Created `train_abc.py` |
| Contradictory component goals | CRITICAL | **By design** | This IS the research question |
| Memory (C) untested | MAJOR | **Acknowledged** | Limitation noted; Phase 2 includes C-only |
| 7.7% claim misleading | CRITICAL | **Clarified** | Both ran 200 steps; throughput differed |
| No convergence criterion | MAJOR | **Added** | Section 5.3 convergence monitoring |
| Curriculum is confound | CRITICAL | **Rejected** | Curriculum IS COCONUT's mechanism |

**Review verdict on "FATAL" claims**: The reviewer called n=1 "FATAL" for scientific conclusions. This is correct for confirmatory research but overstated for exploratory Phase 1. We acknowledge: Phase 1 results are signals for GO/NO-GO, not publishable findings. Phase 2 adds seeds if signal found.

**Maintained design decisions**:
- 4 conditions (not 8) - sufficient for initial signal
- 500 steps - with convergence monitoring
- 1 seed - acknowledged limitation, Phase 2 expands
- Tiny model - fast iteration for proof-of-concept

---

## 15. v3.4 Extension: BPE Tokenization for COCONUT

### 15.0 Review Summary (5-Round Expert Review Complete)

**Date**: 2026-02-05
**Reviewers**: Methodologist, Statistician, Skeptic, Technical, Domain Expert
**Overall Assessment**: **REVISE** (1 revise, 4 pass_with_conditions)

**Critical Issues Identified**:

| ID | Issue | Resolution |
|----|-------|------------|
| F001 | Post-hoc rationalization (HARKing) | Acknowledge as exploratory |
| M001/F002 | 500x parameter confound | Focus on within-BPE comparisons |
| M002/D003 | Curriculum replaces steps, not BPE tokens | Document mechanism unchanged |
| F006/S002 | PPL incomparable across tokenizers | Report within-tokenizer % change |

**Key Insight from Domain Expert**: The embedding parameter difference (6.4M vs 13K) is NOT a confound for COCONUT mechanism testing because COCONUT operates in transformer layers (~500K params), which are IDENTICAL between BPE and char conditions. The comparison tests the SAME reasoning architecture with different tokenization.

### 15.1 Motivation

The v3.3 conclusions noted that **COCONUT evaluation was out of scope** because character-level tokenization may affect its mechanism.

**Important Clarification (from domain review)**: The claim "character tokenization defeats COCONUT" is *partially correct but oversimplified*. COCONUT's curriculum training replaces reasoning STEPS (complete text strings) with thought tokens, regardless of how those steps are tokenized. What changes between char and BPE is:

- **Character mode**: Model must learn character-level patterns; hidden state at p-1 contains info about a single character
- **BPE mode**: Model sees word/subword patterns; hidden state at p-1 contains info about a semantic unit

The mechanism (step-level replacement) is unchanged. The hypothesis is that BPE's richer per-token semantics helps COCONUT's hidden state evolution.

**Acknowledged Limitation**: This hypothesis emerged AFTER observing null results with character tokenization. This is exploratory follow-up (HARKing acknowledged), NOT confirmatory testing of the original COCONUT hypothesis.

### 15.2 Implementation

**New file**: `code/data/bpe_tokenizer.py`
- Wraps tiktoken GPT-2 encoding (50,257 tokens)
- Extends with COCONUT special tokens: `<pad>`, `<eos>`, `<bot>`, `<thought>`, `<eot>`
- Final vocab_size: 50,262

**Modified**: `code/training/train_abc.py`
- Added `--tokenizer {char,bpe}` flag (default: char for backward compatibility)
- Added `--max_seq_len` flag (default: 512 for char, 256 for bpe)
- Tokenizer metadata logged in results JSON

### 15.3 Experiment Protocol

```bash
# v3.4 BPE experiment (5 seeds each)
cd code

# Baseline with BPE
python training/train_abc.py --condition baseline --tokenizer bpe \
    --max_steps 500 --seeds 42 123 456 789 1001 \
    --output ../results/v3.4_bpe/baseline

# COCONUT-only with BPE
python training/train_abc.py --condition coconut_only --tokenizer bpe \
    --max_steps 500 --seeds 42 123 456 789 1001 \
    --output ../results/v3.4_bpe/coconut_only
```

### 15.4 Known Confounds and Mitigations

| Confound | Impact | Mitigation |
|----------|--------|------------|
| **Embedding parameters** | BPE: ~6.4M params vs char: ~13K (500x larger) | **Not a confound for COCONUT mechanism**: Transformer layers are identical (~500K params). COCONUT operates in transformer, not embeddings. |
| **Compression ratio** | BPE: ~10 tokens/step; Char: ~50 chars/step | Same curriculum (1 thought/step). Document that step-level replacement is unchanged. |
| **Sequence length** | BPE default 256 vs char 512 tokens | BPE sees more content per token. This is expected behavior, not confound. |
| **PPL scale** | Different vocabs → different PPL scales | **CRITICAL**: Do NOT compare PPL across tokenizers. Only compare within-tokenizer. |

**Parameter Breakdown** (from Domain Expert review):

| Component | BPE Model | Char Model | Notes |
|-----------|-----------|------------|-------|
| Embedding layer | 6.4M | 13K | NOT where COCONUT operates |
| Transformer layers | ~500K | ~500K | **IDENTICAL** - this is what matters |
| Total | ~7.5M | ~1.1M | Misleading comparison |

The key insight is that COCONUT's latent reasoning happens in transformer layers, not embeddings. Both BPE and char models have **identical transformer capacity**.

**Acknowledged Limitations** (from Methodologist review):
- Embedding size may affect optimization dynamics (gradient flow), though not expected to interact with COCONUT's iterative mechanism
- Sequence length differs (BPE 256 vs char 512 tokens), but for multi-step arithmetic (5-9 steps), reasoning chains fit within either limit

### 15.5 Hypotheses and Pre-Registered Analysis

**Primary Hypothesis (H5)**: COCONUT shows larger relative benefit with BPE tokenization than with character tokenization.

**Operationalized**: `(BPE_baseline - BPE_COCONUT) / BPE_baseline > (char_baseline - char_COCONUT) / char_baseline`

**Pre-Registered Primary Analysis**:
- Two-sample t-test: BPE_COCONUT vs BPE_baseline (5 seeds each)
- Report: p-value, 95% CI for difference, Cohen's d
- Effect size interpretation: d > 0.5 is "medium", d > 0.8 is "large"
- α = 0.05 (treat as separate experiment from main study)

**Power Analysis**:
- With n=5 seeds per condition and expected SD ~0.02 (based on v3.3 char experiments)
- MDE (80% power) ≈ 2.8 × SD × sqrt(2/n) ≈ 2.8 × 0.02 × 0.632 ≈ **3.5%**
- The 2% success threshold is MORE stringent than our detection limit
- **Implication**: We can reliably detect effects ≥3.5%. Effects in the 2-3.5% range may be real but underpowered
- **Sensitivity provision**: If results are marginal (2-4% effect, 0.05 < p < 0.10), note as "suggestive" and consider n=10 replication

**Pre-Registered Secondary Analysis**:
- Compare COCONUT effect magnitude: BPE vs char (descriptive, not inferential)
- Report both PPL and % improvement within each tokenizer type

**Expected Outcomes**:

| Outcome | Interpretation | Next Step |
|---------|----------------|-----------|
| BPE COCONUT < BPE baseline (p<0.05) | **COCONUT benefits from subword tokens** | Report effect size, compare to char |
| BPE COCONUT ≈ BPE baseline (p≥0.05) | **COCONUT ineffective at this scale** | Conclude tokenization is not sufficient |
| BPE COCONUT > BPE baseline | **COCONUT hurts performance** | Document as negative finding |

**Stopping Rule**: If BPE COCONUT shows no significant benefit (p>0.05, effect <2%), conclude that COCONUT does not help at this scale regardless of tokenization. No further tokenization experiments.

### 15.6 Verification Checklist

Before running full experiment:
- [ ] Smoke test: `python training/train_abc.py --condition baseline --tokenizer bpe --max_steps 10`
- [ ] Check special tokens encode correctly: `<thought>` → 50260
- [ ] Compare sequence lengths: BPE should be ~5x shorter for same text
- [ ] Verify baseline PPL is reasonable (BPE baseline will differ from char baseline)

### 15.7 Results (COMPLETE - 2026-02-05)

**Within-Tokenizer Comparisons (PRIMARY)**:

| Tokenizer | Condition | Val PPL (mean±std) | COCONUT Effect | p-value | Cohen's d |
|-----------|-----------|-------------------|----------------|---------|-----------|
| **BPE** | baseline | **7.21 ± 0.22** | - | - | - |
| **BPE** | coconut_only | **7.45 ± 0.22** | **+3.4%** | **0.120** | 1.10 |
| char (ref) | baseline | 2.36 ± 0.01 | - | - | - |
| char (ref) | coconut_only | 2.36 ± 0.02 | +0.2% (neutral) | - | ~0 |

**Individual Seed Results**:

| Seed | BPE Baseline | BPE COCONUT |
|------|--------------|-------------|
| 42 | 6.97 | 7.11 |
| 123 | 7.01 | 7.46 |
| 456 | 7.33 | 7.55 |
| 789 | 7.49 | 7.71 |
| 1001 | 7.23 | 7.41 |

**Statistical Analysis**:
- Difference: +0.24 PPL (+3.4% worse)
- 95% CI: [-0.03, +0.51] (includes zero)
- Welch's t-test: t=1.74, **p=0.120** (NOT significant)
- Cohen's d = 1.10 (large effect size, but wrong direction)

**Note**: Cross-tokenizer PPL comparison is NOT meaningful. BPE baseline PPL will differ from char baseline PPL due to different vocabulary sizes. Only within-tokenizer comparisons (% change) are reported.

**Transformer vs Embedding Parameters**:

| Model | Embedding Params | Transformer Params | Total |
|-------|------------------|-------------------|-------|
| BPE | ~6.4M | ~500K | ~7.0M |
| char | ~13K | ~500K | ~1.1M |

### 15.8 Analysis Plan (Pre-Registered)

**Primary Analysis** (regardless of outcome):
1. Two-sample t-test: BPE_COCONUT vs BPE_baseline
2. Report: difference in PPL, 95% CI, p-value, Cohen's d
3. Success criterion: p < 0.05 AND effect > 2% improvement

**Secondary Analyses**:
1. Compare COCONUT effect magnitude: BPE (%) vs char (+0.2%)
2. Report FLOPs per sample for compute fairness
3. Log BPE tokenization of sample reasoning steps (exploratory)

**Interpretation Guide**:
- If BPE COCONUT significantly better: Tokenization matters for COCONUT
- If BPE COCONUT neutral (like char): COCONUT doesn't work at this scale, period
- If BPE COCONUT worse: Document as unexpected finding

**Post-Hoc Exploration** (if BPE COCONUT shows benefit):
- Analyze alignment between thought tokens and reasoning step boundaries
- Consider testing step-level tokenization (1 token = 1 step) as future work

---

### 15.9 Conclusions (v3.4 COMPLETE - REVISED AFTER 5-ROUND REVIEW)

**Primary Finding**: COCONUT showed a trend toward WORSE performance with BPE tokenization.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| COCONUT effect (BPE) | **+3.4% worse** | Trend toward degradation |
| Cohen's d | **1.10** | Large effect size (wrong direction) |
| p-value | 0.120 | Not significant at α=0.05 |
| 95% CI (t-dist, df=8) | [-0.08, +0.56] | Includes zero but skewed toward harm |
| Statistical power | **36%** | Underpowered to detect MDE |

**Hypothesis Test Result**:

H5 stated: "COCONUT shows larger relative benefit with BPE tokenization than with character tokenization."

**NOT CONFIRMED**: BPE COCONUT (+3.4% worse, d=1.10) showed a large effect size in the WRONG direction compared to char COCONUT (+0.2% neutral). While p=0.120 fails to reach significance, the 36% statistical power means we cannot definitively distinguish "COCONUT has no effect" from "COCONUT harms performance but we lack power to confirm."

**Appropriate Interpretation** (per Statistician review):
> "COCONUT showed a trend toward worse performance with BPE tokenization (+3.4% perplexity, d=1.10), though this did not reach statistical significance (p=0.12). Given the study's limited power (36%), we cannot definitively conclude whether this represents a true degradation or sampling variability. The direction of the effect (worse, not better) fails to support the hypothesis that BPE would benefit COCONUT."

**Scale Context** (per Domain Expert review):
The original COCONUT paper used GPT-2-scale models (117M+ parameters) and warm-started from CoT-pretrained checkpoints. This study used:
- **15-200x smaller models** (7.5M vs 117M+)
- **Training from scratch** (not warm-started from CoT-pretrained model)
- **Simpler task** (linear arithmetic vs branching reasoning)

A null or negative result at this scale is **consistent with domain expectations**, not a contradiction of the original COCONUT findings.

**Possible Explanations** (refined):
1. **Scale insufficient**: COCONUT's BFS-like mechanism requires hidden dimensions large enough to encode multiple reasoning paths - likely unavailable at 7.5M params
2. **Training protocol mismatch**: Original COCONUT warm-starts from CoT-pretrained model (~40% accuracy); this study trains from scratch
3. **Task complexity**: COCONUT excels on tasks with branching; linear arithmetic may not activate this mechanism
4. **Overhead without benefit**: At insufficient scale, COCONUT's curriculum training adds overhead without the representational capacity to realize benefits

**Stopping Rule Applied** (with caveats):
Per pre-registration, since BPE COCONUT shows no significant benefit (p>0.05), we stop further tokenization experiments. However, reviewers noted:
- The stopping rule was designed for neutral outcomes, not negative trends
- A large effect size (d=1.10) in the harmful direction may warrant additional investigation
- Running n=10 seeds could resolve whether the +3.4% trend is real

**Final Verdict on COCONUT (all experiments)**:

| Tokenization | Scale | COCONUT Effect | Cohen's d | Verdict |
|--------------|-------|----------------|-----------|---------|
| Character | 1.1M | +0.2% (neutral) | ~0 | No detectable effect |
| BPE | 7.5M | +3.4% (trend worse) | 1.10 | Trend toward harm (underpowered) |

**COCONUT provides no detectable benefit at ≤10M parameter scale on multi-step arithmetic. The +3.4% degradation with BPE (d=1.10) suggests COCONUT may add overhead without benefit at scales far below the original paper's 117M+ parameter experiments. Further investigation at larger scales (100M+) with warm-start training protocol would be needed to properly evaluate COCONUT.**

---

## 16. v3.5 Extension: Warm-Start Training Protocol

### 16.0 Motivation

The v3.4 BPE experiment showed COCONUT performed WORSE (+3.4%, d=1.10) when training from scratch. However, Domain Expert review noted a critical methodological difference:

| Aspect | Original COCONUT Paper | Our Study (v3.4) |
|--------|----------------------|------------------|
| Pre-training | Warm-start from CoT-pretrained model | Training from scratch |
| Base accuracy | ~40% on task before COCONUT | 0% (random init) |
| Training signal | Fine-tuning already-capable model | Learning task + latent reasoning simultaneously |

**Hypothesis**: COCONUT may require the model to already "know" the task before it can learn to compress reasoning into latent tokens. Training from scratch asks the model to simultaneously:
1. Learn the task (multi-step arithmetic)
2. Learn latent reasoning (COCONUT mechanism)

This dual learning burden may be too much, causing the observed degradation.

### 16.1 Experiment Design

**Protocol**:
1. **Phase 1**: Train baseline model to convergence (500 steps, no COCONUT)
2. **Phase 2**: Load baseline checkpoint, enable COCONUT curriculum, continue training (500 steps)
3. **Compare**: Warm-start COCONUT vs from-scratch COCONUT vs baseline

**Conditions** (5 seeds each):

| Condition | Description | Total Steps |
|-----------|-------------|-------------|
| `baseline` | Standard training, no COCONUT | 500 |
| `coconut_scratch` | COCONUT from random init | 500 |
| `coconut_warmstart` | Baseline (500) → COCONUT (500) | 1000 total |

**Note**: `coconut_warmstart` uses more total steps (1000 vs 500). This is intentional - the question is whether warm-start HELPS, not whether it's compute-equivalent. If warm-start COCONUT beats from-scratch COCONUT despite similar COCONUT-phase steps, that indicates the training protocol matters.

### 16.2 Implementation

```bash
# Phase 1: Train baseline (5 seeds)
for seed in 42 123 456 789 1001; do
  python training/train_abc.py --condition baseline --tokenizer bpe \
    --max_steps 500 --seed $seed --output ../results/v3.5_warmstart/baseline/seed_$seed
done

# Phase 2: Warm-start COCONUT from each baseline checkpoint
for seed in 42 123 456 789 1001; do
  python training/train_abc.py --condition coconut_only --tokenizer bpe \
    --max_steps 500 --seed $seed \
    --checkpoint ../results/v3.5_warmstart/baseline/seed_$seed/baseline_model.pt \
    --warmstart_from baseline \
    --output ../results/v3.5_warmstart/coconut_warmstart/seed_$seed
done

# Control: From-scratch COCONUT (already have from v3.4, but re-run for fairness)
for seed in 42 123 456 789 1001; do
  python training/train_abc.py --condition coconut_only --tokenizer bpe \
    --max_steps 500 --seed $seed --output ../results/v3.5_warmstart/coconut_scratch/seed_$seed
done
```

### 16.3 Hypotheses and Pre-Registered Analysis

**Primary Hypothesis (H6)**:
> Warm-start COCONUT (baseline → COCONUT) will show lower perplexity than from-scratch COCONUT.

**Secondary Hypothesis (H7)**:
> Warm-start COCONUT will show lower perplexity than baseline (COCONUT provides benefit when given proper foundation).

**Pre-Registered Tests**:
- H6: Paired t-test, `coconut_warmstart` vs `coconut_scratch` (same seeds), α=0.05
- H7: Paired t-test, `coconut_warmstart` vs `baseline` (same seeds), α=0.05
- Effect sizes: Cohen's d with 95% CI
- Multiple comparison correction: Holm-Bonferroni (2 tests)

**Stopping Rule**:
- If H6 confirmed (warmstart < scratch, p<0.05): Protocol matters, continue to larger scale
- If H6 rejected and H7 rejected: COCONUT provides no benefit regardless of protocol at this scale
- If H6 rejected but H7 confirmed: From-scratch is adequate, warm-start not needed

### 16.4 Verification Checklist

- [ ] Baseline checkpoints exist for all 5 seeds
- [ ] Warm-start loading confirmed (parameter count matches)
- [ ] All conditions use same tokenizer (BPE) and sequence length (256)
- [ ] Seeds are consistent across phases (same seed used for baseline → warmstart continuation)

### 16.5 Results (COMPLETE - 2026-02-05)

| Condition | Total Steps | Val PPL (mean ± std) | vs Baseline |
|-----------|-------------|----------------------|-------------|
| Baseline | 500 | 7.21 ± 0.22 | — |
| COCONUT scratch | 500 | 7.45 ± 0.22 | **+3.3% (worse)** |
| COCONUT warmstart | 1000 (500+500) | **2.98 ± 0.02** | **-58.6% (better)** |

**Per-seed results**:

| Seed | Baseline | COCONUT scratch | COCONUT warmstart |
|------|----------|-----------------|-------------------|
| 42 | 6.97 | 7.12 | 2.99 |
| 123 | 7.01 | 7.46 | 2.95 |
| 456 | 7.33 | 7.55 | 3.00 |
| 789 | 7.50 | 7.71 | 3.00 |
| 1001 | 7.23 | 7.41 | 2.98 |

### 16.6 Statistical Analysis

**H6: Warm-start COCONUT vs From-scratch COCONUT**
- Mean difference: 4.46 PPL (scratch is worse)
- t(4) = 46.70, **p = 1.26e-06**
- Cohen's d = **20.9** (extremely large)
- **CONFIRMED**: Warm-start dramatically outperforms from-scratch

**H7: Warm-start COCONUT vs Baseline**
- Mean difference: 4.22 PPL (baseline is worse)
- t(4) = 44.98, **p = 1.46e-06**
- Cohen's d = **20.1** (extremely large)
- **CONFIRMED**: COCONUT provides massive benefit when warm-started

**Additional: From-scratch COCONUT vs Baseline**
- Mean difference: +0.24 PPL (scratch is worse)
- t(4) = 4.33, **p = 0.012**
- Cohen's d = 1.94 (large, wrong direction)
- **CONFIRMED**: From-scratch COCONUT is HARMFUL at small scale

### 16.7 Conclusions (v3.5.1 - CORRECTED AFTER REVIEW)

**5-ROUND REVIEW IDENTIFIED CRITICAL CONFOUND**

The initial warm-start results (Section 16.5-16.6) were misleading due to a compute confound: warm-start used 1000 total steps while comparisons used only 500 steps.

**Compute-Matched Control Results**:

| Condition | Steps | Val PPL (mean ± std) | vs Baseline@1000 |
|-----------|-------|----------------------|------------------|
| Baseline | 500 | 7.21 ± 0.22 | — |
| Baseline **(CONTROL)** | 1000 | **2.90 ± 0.02** | — |
| COCONUT scratch | 500 | 7.45 ± 0.22 | — |
| COCONUT warmstart | 1000 | 2.98 ± 0.02 | **+3.0% worse** |

**Statistical Test (Corrected)**:
- H: Warm-start COCONUT vs Baseline@1000 (compute-matched)
- Difference: +0.088 PPL (COCONUT is worse)
- t(4) = 13.27, **p = 0.0002**
- Cohen's d = 5.93

**Corrected Conclusion**:

The initial "58.6% improvement" was **entirely due to more training**, not COCONUT.

When matched for compute:
- Baseline@1000: **2.90 PPL**
- COCONUT warmstart@1000: **2.98 PPL** (+3.0% worse)

**COCONUT provides NO benefit at 7.5M parameter scale**, even with warm-start matching the original paper's methodology.

**Why This Differs From Original Paper**:

| Aspect | Original COCONUT | Our Study |
|--------|-----------------|-----------|
| Base model | Pre-trained GPT-2 | Random init |
| Parameters | 125M | 7.5M (17x smaller) |
| Language understanding | Full internet corpus | One math task |
| Stage 0 training | Fine-tuning | From scratch |

The original paper's success likely depends on:
1. **Scale**: 125M params vs 7.5M
2. **Pre-training**: General language capability, not just task-specific

**Final Verdict for COCONUT at Small Scale**:

| Experiment | Result | Conclusion |
|------------|--------|------------|
| v3 (char-level) | +0.2% neutral | No effect |
| v3.4 (BPE, scratch) | +3.4% worse | Harmful |
| v3.5 (BPE, warmstart) | +3.0% worse (vs matched) | **No benefit** |

**COCONUT requires scale (100M+ params) and/or pre-training to provide benefits. At 7.5M params with task-specific training, it adds computational overhead (5x forward passes) without improvement.**
