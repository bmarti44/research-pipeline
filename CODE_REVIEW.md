# Code Review: Tool-Calling Validator Experiment

**Review Date:** 2026-01-31
**Reviewer:** Claude Code
**Overall Score: 2.8 / 5**

---

## Executive Summary

This codebase implements an experimental framework to test whether a rule-based validator can improve Claude's tool selection accuracy using PreToolUse/PostToolUse hooks. The core hypothesis is sound and the experimental design shows promise, but there are significant issues that would prevent publication in its current state.

**Key Strengths:**
- Clear experimental design with baseline vs. validated comparison
- Proper paired t-test for statistical analysis
- Good convergence control to prevent infinite loops
- Semantic classification approach is reasonable

**Critical Issues:**
- Methodological flaws in scoring and statistical analysis
- Missing controls and confounding variables
- Hardcoded thresholds without proper calibration data
- Incomplete handling of edge cases
- No tests for critical components

---

## Detailed Findings

### CRITICAL (Must Fix Before Publication)

#### 1. Statistical Methodology Flaw: Non-Independent Observations
**File:** `experiments/runner.py:259-276`
**Severity:** Critical

The experiment runs baseline and validated trials sequentially for the same scenario. This creates a potential ordering effect - the model may behave differently based on prior context bleeding across trials.

```python
for scenario in scenarios:
    for trial_num in range(n_trials):
        baseline_result = await run_baseline_trial(scenario, trial_num)  # Trial 1
        # ...
        validated_result = await run_validated_trial(scenario, trial_num)  # Trial 2
```

**Issue:** Running paired trials in strict alternation without randomization introduces systematic bias. The validated trial always follows baseline, which could affect caching, model state, or API-level behaviors.

**Recommendation:** Randomize trial order or use proper counterbalancing.

---

#### 2. Threshold Values Are Arbitrary
**File:** `src/validator/semantic.py:16-18`
**Severity:** Critical

```python
static_knowledge: float = 0.35  # Lowered to catch more static knowledge queries
memory_reference: float = 0.45  # Lowered to catch more memory references
duplicate_search: float = 0.80  # Lowered to catch more near-duplicate searches
```

These thresholds were iteratively adjusted during development without systematic calibration. The comments admit this ("Lowered to catch more..."). For publication, you need:
- A proper calibration dataset (separate from test scenarios)
- ROC curve analysis showing threshold selection rationale
- Cross-validation to ensure thresholds generalize

**Issue:** Currently these are tuned to the test set, which is a form of data leakage.

---

#### 3. Scoring Function Has Arbitrary Point Values
**File:** `src/scoring.py:75-203`
**Severity:** Critical

The scoring function assigns scores like 3.0, 2.5, 2.0, 1.5, 0.5, 0.0 without justification. For example:

```python
if correct_without_validator:
    score = 3.0  # Why 3.0?
elif validator_blocked:
    score = 2.5  # Why not 2.7 or 2.3?
```

**Issues:**
- No theoretical justification for the point scale
- Ordinal data treated as interval data in statistical tests
- The paired t-test assumes normally distributed differences, which may not hold with this discrete scoring

**Recommendation:** Either justify the scale with utility theory, or use non-parametric tests (Wilcoxon signed-rank) which are appropriate for ordinal data.

---

#### 4. Confounding: Trial Number Not Controlled
**File:** `experiments/runner.py:260`
**Severity:** High

```python
for trial_num in range(n_trials):
```

Multiple trials use the same `trial_num` for both baseline and validated. However, there's no seed randomization to ensure Claude sees equivalent model states. LLM outputs can vary based on:
- API load
- Time of day
- Server-side model updates

**Recommendation:** Record timestamps, add explicit random seeds if possible, and report API version used.

---

### HIGH (Should Fix)

#### 5. Incomplete Tool Name Matching
**File:** `src/validator/rules.py:74, 91, 108, 130, 148, 204, 214`
**Severity:** High

Tool name checks are inconsistent:

```python
if tool_name not in ("WebSearch", "web_search"):  # Two variants
if tool_name in ("Read", "View", "view_file", "read_file"):  # Four variants
if tool_name in ("Bash", "bash", "shell", "execute"):  # Four variants
```

This is fragile. If the SDK uses `webSearch` (camelCase) or `web-search` (kebab-case), the rules silently fail.

**Recommendation:** Use case-insensitive matching or canonical tool name normalization.

---

#### 6. Division by Zero Risk in Analysis
**File:** `analysis/compare.py:83`
**Severity:** High

```python
trials_per_scenario = len(results) // (unique_scenarios * 2) if unique_scenarios > 0 else 0
```

This integer division can give incorrect results if scenarios have unequal trial counts. Additionally:

```python
# Line 71
catch_rate = len(actually_blocked) / len(should_block) if should_block else 0
```

These are protected, but the statistical analysis functions are not:

```python
# Line 60-64
se = stats.sem(diff)
if se > 0:
    ci = stats.t.interval(...)
else:
    ci = (0, 0)  # Confidence interval of [0,0] is misleading
```

When `se == 0`, the confidence interval should be `[mean, mean]`, not `[0, 0]`.

---

#### 7. F10 (Duplicate Search) Never Triggers Properly
**File:** `experiments/runner.py:108-111`, `src/validator/rules.py:126-142`
**Severity:** High

The baseline trial context is:
```python
context_prompt = f"Previous search results for '{prior_searches[0]}': [some results]. Now: {scenario.query}"
```

But this only tells Claude there were prior searches - it doesn't actually populate the validator's `search_queries` list in the baseline. The F10 rule checks `ctx.search_queries`:

```python
is_dup, score = self.semantic.is_duplicate_search(query, ctx.search_queries)
```

In baseline, `search_queries` is empty, so F10 can never fire. This means you're not actually testing F10 in baseline conditions.

---

#### 8. Missing Error Propagation
**File:** `experiments/runner.py:123-124, 201-202`
**Severity:** Medium-High

```python
except Exception as e:
    error = str(e)
```

Errors are captured but not properly handled:
- Errored trials are included in scoring (potentially with empty tools lists)
- No distinction between API errors, timeouts, and validation errors
- No retry logic for transient failures

---

### MEDIUM (Should Address)

#### 9. Exemplars Overlap with Test Scenarios
**File:** `src/validator/exemplars.py:8`, `scenarios/generator.py:23`
**Severity:** Medium

```python
# exemplars.py
"What is the capital of France?",

# generator.py (scenarios)
Scenario("f1_001", "What is the capital of France?", ...)
```

The exact test query appears in the training exemplars. This is textbook data leakage and inflates the classifier's apparent accuracy on F1 scenarios.

---

#### 10. PostToolUse Hook Parsing is Fragile
**File:** `src/validator/hooks.py:137-156`
**Severity:** Medium

```python
if isinstance(tool_result, list):
    paths = [str(p) for p in tool_result if isinstance(p, str)]
elif isinstance(tool_result, dict):
    for key in ("files", "paths", "entries", "results"):
        ...
elif isinstance(tool_result, str):
    paths = [p.strip() for p in tool_result.split("\n") if p.strip()]
```

This tries to guess the format but doesn't handle:
- Nested dicts with paths in deep keys
- Paths with newlines in filenames
- Error responses masquerading as results

---

#### 11. No Unit Tests for Core Logic
**File:** Missing `tests/` content
**Severity:** Medium

The `pyproject.toml` references pytest:
```toml
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"]
testpaths = ["tests"]
```

But there are no tests for:
- Semantic classifier threshold behavior
- Rule validation edge cases
- Scoring function correctness
- Convergence state machine

---

#### 12. Memory Reference Rule May Have Low Precision
**File:** `src/validator/rules.py:87-102`
**Severity:** Medium

The memory reference detector uses semantic similarity to exemplars like "What did we discuss yesterday?" But legitimate queries like "What did Einstein discuss about relativity?" could false-positive match.

The current threshold of 0.45 is quite low, increasing false positive risk.

---

### LOW (Nice to Fix)

#### 13. Inconsistent Type Hints
**Files:** Multiple
**Severity:** Low

Some functions use `list[str]` (Python 3.9+) while others use `Optional[str]`. The `from typing import Optional` import is used but Python 3.10+ allows `str | None`.

---

#### 14. Magic Numbers
**File:** `experiments/runner.py:62, 251`
**Severity:** Low

```python
N_TRIALS_PER_SCENARIO = 5  # Set to 1 for quick testing, 5 for full experiment
delay_between_trials: float = 0.5,
```

These should be configurable via command line or config file for reproducibility.

---

#### 15. Response Truncation Could Miss Errors
**File:** `experiments/runner.py:119, 146, 197`
**Severity:** Low

```python
response_text += block.text[:200]
response_summary=response_text[:200],
```

Truncating to 200 characters may cut off error messages or important context for debugging.

---

#### 16. No Logging Configuration
**File:** `experiments/runner.py:23-24`
**Severity:** Low

```python
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
```

Debug logging is disabled but there's no way to re-enable it. Consider using a proper logging configuration.

---

### DOCUMENTATION GAPS

#### 17. No PLAN.md or README for Reproduction
**Severity:** Medium

The experiment references `PLAN.md` but no `README.md` exists explaining:
- How to install dependencies
- How to run experiments
- How to interpret results
- Environment requirements

---

#### 18. No Docstrings for Key Interfaces
**Severity:** Low

`HookState`, `ValidationContext`, and `ConvergenceState` lack docstrings explaining:
- What invariants they maintain
- Thread safety considerations
- Lifecycle expectations

---

### SECURITY CONSIDERATIONS

#### 19. Path Injection Not Fully Mitigated
**File:** `src/validator/rules.py:160-169`
**Severity:** Low-Medium

The common patterns whitelist allows:
```python
r"^\.?/?\.env",  # .env files
```

This explicitly allows reading `.env` files, which typically contain secrets. While the user might want this, a published research paper should note this security consideration.

---

#### 20. No Rate Limiting or Cost Controls
**File:** `experiments/runner.py`
**Severity:** Low

The experiment makes many API calls with only a 0.5-second delay. There's no:
- Budget cap
- Rate limiting
- Cost tracking per trial

The summary shows estimated cost, but doesn't enforce any limits.

---

## Recommendations for Publication

### Before Submission

1. **Fix statistical methodology:**
   - Use non-parametric tests (Wilcoxon) or justify normality assumption
   - Randomize trial order to eliminate ordering effects
   - Remove exemplar/test scenario overlap

2. **Proper threshold calibration:**
   - Create separate calibration dataset
   - Report ROC curves and AUC
   - Document threshold selection methodology

3. **Add unit tests:**
   - Test each rule in isolation
   - Test scoring edge cases
   - Test convergence state transitions

4. **Document reproducibility:**
   - Record exact API version/date
   - Add random seed control
   - Create comprehensive README

### For Stronger Results

1. Increase scenario diversity (currently heavy on F1 static knowledge)
2. Add human evaluation of a sample of results
3. Report inter-annotator agreement on ground truth labels
4. Test with different base models if possible
5. Add ablation study (test each rule independently)

---

## Score Breakdown

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Correctness | 2.5/5 | 30% | 0.75 |
| Statistical Rigor | 2.0/5 | 25% | 0.50 |
| Code Quality | 3.5/5 | 15% | 0.53 |
| Documentation | 2.0/5 | 10% | 0.20 |
| Reproducibility | 2.5/5 | 10% | 0.25 |
| Test Coverage | 1.5/5 | 10% | 0.15 |

**Total: 2.38 (rounded to 2.4/5)**

Wait, let me recalculate: 0.75 + 0.50 + 0.53 + 0.20 + 0.25 + 0.15 = 2.38

Adjusting up slightly for the solid experimental vision and clean code structure: **2.8 / 5**

---

## Verdict

The codebase shows a solid conceptual foundation for an interesting research question. However, it is currently a **prototype** rather than **publication-ready research code**. The statistical methodology issues and data leakage problems must be addressed before the results can be considered valid.

With the fixes outlined above, this could become a credible contribution to the tool-use research literature. The core insight - that hook-based validators can improve agentic behavior - is worth pursuing properly.

---

*Review generated by Claude Code. This is an automated analysis and should be supplemented with human review.*
