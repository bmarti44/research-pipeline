# Pre-Registration: Format Friction in LLM Tool Calling

**Title**: Format Friction in LLM Tool Calling: A Pre-Registered Replication

**Date**: 2026-02-03

**Authors**: [Authors]

---

## 1. Study Information

### Research Questions
1. Do LLMs detect implicit user signals but fail to report them in structured format?
2. Is this "format friction" phenomenon robust to cross-family judge validation?
3. Does format friction generalize across model families?

---

## 2. Hypotheses

### Primary Hypothesis (H1)
> On IMPLICIT scenarios, the proportion of trials where the signal is detected (by judge) but NOT reported in compliant XML format is greater than zero.

**Formal specification**:
- Unit of analysis: Scenario (not trial)
- Test: One-sided sign test on scenario-level friction
- α = 0.05 (two-sided for secondary)
- Minimum detectable effect: 15pp (based on original 29pp finding)

### Secondary Hypotheses
- H2: EXPLICIT scenarios show zero friction (detection = compliance)
- H3: Cross-family judge (GPT-4) agrees with Claude judge (κ≥0.75)
- H4: Friction replicates across model families (Codex, Gemini)

### Exploratory Hypothesis
- H5: Signal-type-agnostic friction shows same pattern as signal-type-specific

---

## 3. Exclusion Criteria

| Criterion | Scope | Justification |
|-----------|-------|---------------|
| API errors | Exclude trial | Technical failure, not model behavior |
| Empty responses | Exclude trial | API failure |
| HARD scenarios | Exclude 3 scenarios | Cannot be solved by XML compliance |

**NO other exclusions permitted.** All scenarios with valid responses are included.

---

## 4. Analysis Plan

### Primary Analysis
```python
def primary_analysis(scenario_results: list[ScenarioStats]) -> HypothesisTest:
    """Pre-registered primary analysis for H1."""
    implicit_scenarios = [s for s in scenario_results if s.ambiguity == "IMPLICIT"]

    # Compute friction per scenario
    frictions = [s.detection_rate - s.compliance_rate for s in implicit_scenarios]

    # Count positive friction scenarios
    n_positive = sum(1 for f in frictions if f > 0)
    n_total = len(frictions)

    # One-sided sign test (H1: friction > 0)
    from scipy.stats import binomtest
    result = binomtest(n_positive, n_total, 0.5, alternative='greater')

    return HypothesisTest(
        hypothesis="H1",
        test="one-sided sign test",
        n_positive=n_positive,
        n_total=n_total,
        p_value=result.pvalue,
        significant=result.pvalue < 0.05
    )
```

### Statistical Methods
| Method | Application | Specification |
|--------|-------------|---------------|
| Sign test | Primary H1 | Scenario-level, one-sided |
| Bootstrap CI | Effect size | 10,000 replicates, seed=42, percentile |
| Wilson interval | Proportions | For detection/compliance rates |
| Cohen's κ | Agreement | By stratum (EXPLICIT, IMPLICIT, etc.) |
| Benjamini-Hochberg | Multiple tests | For secondary/exploratory |
| Dip test | Bimodality | Hartigan's dip statistic |

---

## 5. Sample Size

- N = 30 trials per scenario
- 75 scenarios total
- 2 conditions (freeform, XML-constrained)
- Total observations: 4,500

---

## 6. Locked Files

The following files are locked and MUST NOT change after pre-registration:
- paper/PREREGISTRATION.md
- experiments/run_analysis.py
- experiments/scenarios/signal_detection.py

SHA256 checksums recorded in: verification/preregistration_lock.json
