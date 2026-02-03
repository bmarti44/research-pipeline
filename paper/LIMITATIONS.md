# Limitations and Scope

This document outlines the known limitations of this research study on hook-based tool call validation for LLM agents.

## Model Limitations

### Single Model Testing
- **Limitation**: All experiments were conducted exclusively on Claude (Anthropic).
- **Impact**: Findings may not generalize to other LLMs (GPT-4, Gemini, LLaMA, etc.).
- **Mitigation**: Future work should validate on multiple models.

### Model Version Variability
- **Limitation**: Results may vary across Claude model versions and API updates.
- **Impact**: Reproducibility may be affected by model updates.
- **Mitigation**: Experiment metadata records timestamps; consider version pinning.

### Temperature and Sampling
- **Limitation**: The Claude Agent SDK does not expose temperature control.
- **Impact**: Cannot control for sampling randomness across trials.
- **Mitigation**: Multiple trials (n=5) per scenario provide some variance estimation.

## Dataset Limitations

### Limited Failure Modes
- **Limitation**: Only 6 failure modes tested (F1, F4, F8, F10, F13, F15).
- **Impact**: Many potential failure modes are not covered:
  - Hallucinated API endpoints
  - Incorrect tool parameter types
  - Tool sequencing errors beyond location-first
  - Context window overflow issues
  - Multi-step reasoning failures
- **Mitigation**: Framework is extensible; additional rules can be added.

### Scenario Count
- **Limitation**: 41 total scenarios (29 should-block, 12 valid control).
- **Impact**: Limited statistical power for per-rule analysis.
- **Mitigation**: 5 trials per scenario partially addresses this.

### English Only
- **Limitation**: All test scenarios are in English.
- **Impact**: Semantic classifier behavior on other languages is unknown.
- **Mitigation**: Future work should include multilingual scenarios.

### Ground Truth Validity
- **Limitation**: Expected behaviors were defined by a single annotator.
- **Impact**: Potential annotator bias; no inter-annotator agreement measured.
- **Recommendation**: For publication, obtain independent annotations and report Cohen's kappa.

## Classifier Limitations

### Threshold Selection
- **Limitation**: Thresholds were selected via cross-validation on a small calibration set (30 examples per category).
- **Impact**: Possible overfitting to calibration distribution.
- **Mitigation**: Sensitivity analysis shows stability across ±0.10 threshold changes.

### Semantic Model
- **Limitation**: Uses `all-MiniLM-L6-v2` (384-dimensional embeddings).
- **Impact**: Larger models might provide better semantic matching.
- **Trade-off**: Chosen for speed (real-time validation requirement).

### Category Coverage
- **Limitation**: Only two semantic categories (static_knowledge, memory_reference) + duplicate detection.
- **Impact**: Other semantic patterns (e.g., "opinion questions", "creative requests") not covered.
- **Extensibility**: Additional categories can be added with new exemplars.

## Experimental Design Limitations

### Simulated Context
- **Limitation**: F10 (duplicate search) scenarios inject prior searches via prompt, not actual tool calls.
- **Impact**: May not perfectly simulate real conversational context.
- **Mitigation**: Baseline context tracking added for fair comparison.

### Tool Environment
- **Limitation**: Tools are limited to a mock/sandbox environment.
- **Impact**: Real-world tool failures, latency, and error handling not tested.
- **Consideration**: Production deployment may reveal additional failure modes.

### Convergence Behavior
- **Limitation**: Forced termination after 5 rejections may affect score outcomes.
- **Impact**: Some scenarios may score differently due to early termination.
- **Trade-off**: Necessary to prevent infinite loops.

## Statistical Limitations

### Effect Size
- **Observed**: Cohen's d = 0.21 (small effect).
- **Interpretation**: While statistically significant, practical impact is modest.
- **Context**: Small effect is expected when baseline model is already competent.

### Catch Rate vs. Model Competence
- **Observed**: 9.7% catch rate (vs. 60% target).
- **Explanation**: Claude correctly handles F1/F4 scenarios without tool calls; nothing to catch.
- **Interpretation**: Low catch rate indicates model competence, not validator failure.

### Power Analysis
- **Achieved**: 84.1% power for observed effect size.
- **Note**: Adequate for detecting effects of d=0.21, but would need ~350 pairs for smaller effects.

## Recommendations for Future Work

1. **Multi-model validation**: Test on GPT-4, Gemini, open-source models.
2. **Additional failure modes**: Expand rule coverage to other tool-use errors.
3. **Multilingual testing**: Validate semantic classifier on non-English queries.
4. **Independent annotation**: Obtain ground truth labels from multiple annotators.
5. **Production evaluation**: Deploy in real agent systems to identify edge cases.
6. **Adaptive thresholds**: Explore dynamic threshold adjustment based on context.
7. **User studies**: Evaluate whether blocked tool calls improve user satisfaction.

## Conclusion

Despite these limitations, the study demonstrates:
- Statistically significant improvement with validation (+0.22 score, p<0.01)
- Low false positive rate (8.3%)
- Particular value for F10 (duplicate search) scenarios (+3.0 improvement)
- Claude's inherent competence on F1/F4 scenarios (no tool calls attempted)

The validator provides marginal but measurable improvement, with primary value in scenarios where the base model struggles (duplicate searches, missing location context).
