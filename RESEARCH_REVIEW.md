# Research Review: Hook-Based Tool Call Validation

**Document Type:** Pre-Publication Research Assessment
**Date:** 2026-01-31
**Status:** Draft - Experiment In Progress

---

## 1. Research Question & Hypothesis

### Core Question
Can a rule-based validator using PreToolUse/PostToolUse hooks improve Claude's tool selection accuracy compared to unvalidated baseline behavior?

### Hypothesis
A lightweight validator combining semantic classification and deterministic rules can:
- Reduce unnecessary tool calls (F1: static knowledge, F4: memory queries)
- Prevent incorrect tool sequences (F8: location-dependent, F10: duplicates)
- Block hallucinated or invalid operations (F13: invented paths, F15: binary files)

### Success Criteria (as defined)
| Metric | Target | Rationale |
|--------|--------|-----------|
| Score improvement | >= 0.15 | Meaningful practical difference |
| p-value | < 0.05 | Statistical significance |
| Catch rate | >= 60% | Validator catches most violations |
| False positive rate | < 10% | Minimal blocking of valid actions |

---

## 2. Experimental Design Issues

### 2.1 Ordering Effects (Critical)

**Issue:** Baseline and validated trials run in fixed alternating order.

```python
for scenario in scenarios:
    for trial_num in range(n_trials):
        baseline_result = await run_baseline_trial(...)   # Always first
        validated_result = await run_validated_trial(...) # Always second
```

**Impact:**
- API-level caching could favor second trial
- Model "warm-up" effects not controlled
- Any time-based variation systematically biases one condition

**Recommended Fix:** Randomize trial order with counterbalancing (equal baseline-first and validated-first).

---

### 2.2 Data Leakage: Exemplar-Test Overlap (Critical)

**Issue:** Some test scenarios appear verbatim in training exemplars.

| Exemplar (exemplars.py) | Test Scenario (generator.py) |
|------------------------|------------------------------|
| "What is the capital of France?" | f1_001: "What is the capital of France?" |
| "Define recursion" | f1_002: "Explain recursion in programming" |
| "What is photosynthesis?" | f1_003: "What is photosynthesis?" |

**Impact:** Inflates classifier accuracy on F1 scenarios. The semantic classifier will show artificially high similarity scores for queries it was trained on.

**Recommended Fix:**
- Use held-out test scenarios not in exemplar set
- Or use cross-validation (train on subset, test on remainder)

---

### 2.3 Threshold Selection Without Calibration Data

**Issue:** Thresholds were iteratively adjusted based on test set performance.

```python
# Evolution during development:
# 0.60 → 0.50 → 0.40 → 0.35 (static_knowledge)
# 0.55 → 0.50 → 0.45 (memory_reference)
```

**Impact:** This is a form of overfitting to the evaluation set. Thresholds may not generalize to novel queries.

**Recommended Fix:**
- Create separate calibration dataset
- Use ROC curve analysis to select optimal threshold
- Report AUC and threshold sensitivity analysis

---

### 2.4 Scoring Function Validity

**Issue:** The scoring function assigns arbitrary point values without theoretical justification.

| Outcome | Score | Justification |
|---------|-------|---------------|
| Perfect | 3.0 | None provided |
| Validator helped | 2.5 | None provided |
| Partial | 1.5 | None provided |
| Wrong | 0.0 | None provided |

**Statistical Concern:**
- These are ordinal categories, not interval measurements
- Paired t-test assumes interval/ratio data with normal distribution
- Mean differences may not be meaningful (is 3.0→2.5 the same as 1.5→1.0?)

**Recommended Fix:**
- Use Wilcoxon signed-rank test (non-parametric, ordinal-appropriate)
- Or provide utility-theoretic justification for interval interpretation
- Report both parametric and non-parametric results

---

### 2.5 Sample Size and Power Analysis

**Current Design:**
- 41 scenarios × 5 trials × 2 conditions = 410 total trials
- 205 paired comparisons

**Concern:** No a priori power analysis. With high variance in LLM outputs, this sample size may be insufficient to detect small effects reliably.

**Recommended Fix:**
- Conduct post-hoc power analysis
- Report effect size (Cohen's d) alongside p-value
- Consider whether 5 trials per scenario is sufficient given variance

---

### 2.6 F10 Rule Implementation Bug

**Issue:** The F10 (duplicate search) rule cannot fire in baseline condition.

```python
# Baseline trial - search_queries is never populated
options = ClaudeAgentOptions(...)  # No state tracking
# ...
# Later, F10 checks ctx.search_queries which is empty
```

**Impact:** F10 scenarios are not properly tested. The "prior search" context is simulated via prompt text, not actual tracked state.

---

## 3. Methodological Gaps

### 3.1 No Control for Model Variability

**Issue:** LLM outputs are stochastic. The same prompt can produce different tool choices.

**Uncontrolled Variables:**
- API load and latency
- Server-side model updates
- Temperature/sampling parameters
- Time of day effects

**Recommended Fix:**
- Record timestamps and API response metadata
- Use fixed temperature if API allows
- Report date range of experiments
- Consider using multiple model checkpoints if available

---

### 3.2 No Inter-Annotator Agreement for Ground Truth

**Issue:** The "expected_behavior" labels were assigned by the implementer without validation.

Example ambiguities:
- Is "How do I reverse a string in Python?" really static knowledge? (Could argue for code search)
- Is "What's in README.md?" a valid file read or should it list first?

**Recommended Fix:**
- Have 2-3 independent annotators label expected behaviors
- Report Cohen's kappa or Fleiss' kappa for agreement
- Discuss disagreement cases

---

### 3.3 Limited Failure Mode Coverage

**Current Coverage:**
| Failure Mode | Scenarios | % of Total |
|--------------|-----------|------------|
| F1 (static knowledge) | 12 | 29% |
| F4 (memory reference) | 4 | 10% |
| F8 (missing location) | 3 | 7% |
| F10 (duplicate search) | 3 | 7% |
| F13 (hallucinated path) | 3 | 7% |
| F15 (binary file) | 4 | 10% |
| Valid (no failure) | 12 | 29% |

**Concern:** Heavy skew toward F1 (static knowledge). Results may not generalize to other failure modes.

---

### 3.4 No Ablation Study

**Issue:** All rules are tested together. Unknown which rules contribute most to improvement.

**Recommended Fix:**
- Test each rule in isolation
- Report per-rule catch rate and false positive rate
- Identify which rules provide most value

---

### 3.5 Single Model Testing

**Issue:** Only tested on Claude (via claude-agent-sdk). Unknown if findings generalize.

**Recommended Fix:**
- Test on multiple models (GPT-4, Gemini, open-source)
- Or explicitly limit claims to Claude-specific findings

---

## 4. Performance and Practical Caveats

### 4.1 Latency Overhead

| Component | Latency | Frequency |
|-----------|---------|-----------|
| Semantic embedding | 10-50ms | Per tool call |
| Rule evaluation | <1ms | Per tool call |
| Hook dispatch | 1-5ms | Per tool call |
| **Subtotal** | **~15-60ms** | **Per tool call** |

**Acceptable** for most use cases (~2-5% of model inference time).

### 4.2 Rejection Retry Cost

**Critical:** Each rejection triggers a full model retry.

| Rejections | Total Turns | Latency Multiplier |
|------------|-------------|-------------------|
| 0 | 1 | 1x |
| 1 | 2 | 2x |
| 2 | 3 | 3x |
| 3 | 4 | 4x |
| 5 (max) | 6 | 6x |

**Worst case:** 6x latency increase when model repeatedly violates rules.

**Mitigation:** Convergence control limits retries, but degraded responses may result.

---

### 4.3 False Positive Cost

**Issue:** False positives have asymmetric cost.

- False negative: Unnecessary tool call (minor cost)
- False positive: Valid action blocked, degraded response, user frustration

Current false positive rate of 0% is excellent but may not hold with:
- Lower thresholds (trading precision for recall)
- Novel query distributions
- Adversarial inputs

---

### 4.4 Rule Maintenance Burden

**Scalability Concern:** Each new tool or use case requires:
1. New validation rules
2. New semantic exemplars
3. Threshold recalibration
4. Testing for rule interactions

This is **O(n)** in tools/use-cases, vs. training which amortizes across all cases.

---

### 4.5 Coverage Limitations

**Issue:** Rules only catch anticipated failure modes.

First experiment showed 11% catch rate, meaning **89% of potential improvements were missed**.

Causes:
- Novel failure patterns not covered by rules
- Edge cases in rule logic
- Semantic classifier threshold misses

---

### 4.6 Adversarial Robustness

**Issue:** Rules can potentially be evaded.

| Rule | Evasion Vector |
|------|----------------|
| F1 (static) | "I need to verify..." framing |
| F13 (hallucinated) | Use relative paths |
| F15 (binary) | Rename file extension |

**Not a concern for benign use** but relevant for safety-critical deployments.

---

### 4.7 Semantic Classifier Limitations

**Model:** all-MiniLM-L6-v2 (22M parameters)

**Limitations:**
- English-only (no multilingual support)
- 256 token max sequence length
- May not capture domain-specific semantics
- Centroid-based classification is simplistic

**Alternative Approaches:**
- Few-shot classification with the main LLM
- Fine-tuned classifier on tool-use data
- Ensemble of multiple embedding models

---

## 5. Broader Research Implications

### 5.1 The "Verifier is Easier Than Generator" Principle

This research supports the hypothesis that inference-time verification is more tractable than training perfect generators.

**Related Work:**
- Constitutional AI (Anthropic)
- RLHF with reward models
- Code generation with test validation
- Retrieval-augmented generation

---

### 5.2 Neuro-Symbolic Hybrid Architectures

The validator acts as a symbolic layer over neural predictions. This suggests value in:
- Combining learned and rule-based components
- Using domain knowledge as runtime constraints
- Maintaining interpretable decision boundaries

---

### 5.3 Inference-Time Compute Scaling

Spending additional compute at inference (validation, retries) can improve accuracy. This aligns with:
- Chain-of-thought reasoning
- Self-consistency sampling
- Tree-of-thought search

**Tradeoff:** Latency vs. accuracy is configurable per-deployment.

---

### 5.4 Limitations of This Approach

**What This Approach Cannot Do:**
- Improve the model's underlying capabilities
- Catch failure modes not anticipated by rules
- Scale to arbitrary new domains without rule engineering
- Guarantee correctness (only reduce certain error classes)

**When This Approach Is Inappropriate:**
- Low-stakes interactions (overhead not justified)
- Novel domains without established rules
- Real-time latency-critical applications
- When false positives are very costly

---

## 6. Recommendations for Publication

### 6.1 Minimum Viable Fixes

1. **Randomize trial order** to eliminate ordering bias
2. **Remove exemplar-test overlap** or use cross-validation
3. **Add non-parametric test** (Wilcoxon) alongside t-test
4. **Fix F10 implementation** to actually track searches in baseline
5. **Report effect size** (Cohen's d) not just p-value

### 6.2 Strengthening Additions

1. **Ablation study** - test each rule independently
2. **Threshold sensitivity analysis** - how do results change with ±0.05 threshold?
3. **Inter-annotator agreement** for ground truth labels
4. **Power analysis** - report statistical power achieved
5. **Per-rule breakdown** - which rules contribute most?

### 6.3 Scope Limitations to Acknowledge

- Single model (Claude) tested
- English queries only
- Limited failure mode coverage
- Thresholds may not generalize
- No adversarial robustness testing

---

## 7. Summary of Issues by Severity

### Critical (Invalidates Results)
| Issue | Section |
|-------|---------|
| Ordering effects in trial design | 2.1 |
| Exemplar-test data leakage | 2.2 |
| Threshold tuning on test set | 2.3 |
| F10 rule implementation bug | 2.6 |

### High (Weakens Claims)
| Issue | Section |
|-------|---------|
| Scoring function validity for t-test | 2.4 |
| No ground truth validation | 3.2 |
| Limited failure mode coverage | 3.3 |
| No ablation study | 3.4 |

### Medium (Should Address)
| Issue | Section |
|-------|---------|
| No power analysis | 2.5 |
| Model variability uncontrolled | 3.1 |
| Single model testing | 3.5 |
| Rejection latency cost | 4.2 |

### Low (Nice to Have)
| Issue | Section |
|-------|---------|
| Rule maintenance scalability | 4.4 |
| Adversarial robustness | 4.6 |
| Semantic classifier limitations | 4.7 |

---

## 8. Conclusion

This research addresses a valuable question: can lightweight runtime validators improve agentic AI reliability? The experimental framework is reasonable, but significant methodological issues must be addressed before publication.

**Current State:** Promising prototype with interesting preliminary results, but not publication-ready.

**Path to Publication:**
1. Fix critical methodology issues (ordering, data leakage, F10 bug)
2. Add appropriate statistical tests (non-parametric)
3. Include ablation study and sensitivity analysis
4. Document limitations clearly
5. Position appropriately (Claude-specific, English-only, limited failure modes)

**Estimated Effort:** 2-3 additional experiment iterations with fixes, plus analysis write-up.

---

*This document is a self-assessment for research quality. External peer review is recommended before submission.*
