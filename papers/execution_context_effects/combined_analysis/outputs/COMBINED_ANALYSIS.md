# Combined Analysis: Execution Context Effects

## Overview

This paper investigates whether requiring structured JSON tool calls causes failures
that wouldn't occur with natural language output. Across two studies, we find that
"format friction" is better understood as "alignment activation."

## Study Results Summary

### Study 1: Format Friction

| Condition | n | Correct Rate | 95% CI |
|-----------|---|--------------|--------|
| NL | 40 | 97.5% | [87.1%, 99.6%] |
| JSON | 40 | 70.0% | [54.6%, 81.9%] |

**Difference**: 27.5 percentage points (p = 0.0009)

**Finding**: Hypothesis REFUTED
- JSON syntax validity: 100%
- JSON schema correctness: 98.5%
- True serialization failures: 1.5%

The performance gap exists but is NOT caused by serialization failures.

### Study 2: Execution Context

| Condition | n | Correct Rate | 95% CI |
|-----------|---|--------------|--------|
| NL | 40 | 90.0% | [76.9%, 96.0%] |
| JSON | 40 | 72.5% | [57.2%, 83.9%] |

**Difference**: 17.5 percentage points (p = 0.045)

**Finding**: Hypothesis SUPPORTED
- Edit operations: 100% "read first" behavior in JSON mode
- Create operations: 0% cautious behavior
- Caution is operation-specific, not format-driven

## Meta-Analysis

| Metric | Value |
|--------|-------|
| Pooled NL rate | 93.8% |
| Pooled JSON rate | 71.2% |
| Average difference | 22.5% |
| Both studies significant | Yes |
| Effect direction consistent | Yes |

## Key Conclusions

### Main Finding

What appears as 'format friction' is better understood as 'alignment activation' - models being appropriately cautious when their output will actually execute.

### Evidence

- JSON syntax validity is 100% (no serialization failures)
- JSON schema correctness is 98.5% (minimal structural errors)
- Performance gap (17.5-27.5pp) is due to behavioral caution
- Caution is operation-specific (edit vs create operations)

### Implications

- Don't optimize to eliminate all 'friction' - some is valuable
- NL benchmarks may overestimate capability
- Multi-step tool calls (read → edit) may be desired behavior

## Interpretation

The observed "friction" when producing JSON tool calls is not a capability limitation
or serialization failure. Instead, it reflects the model's appropriate caution when
its output will actually execute. This is alignment working as intended:

1. **In NL mode**: Model describes what it would do (no real consequences)
2. **In JSON mode**: Model knows output will execute (real consequences)
3. **For destructive operations**: Model adopts safer behaviors (read → edit)
4. **For non-destructive operations**: No difference in behavior

This reframes the "format friction" problem from a bug to be fixed to a feature
to be understood and potentially preserved.

---

*Combined Analysis - Generated 2026-02-05T04:03:31.812839+00:00*
