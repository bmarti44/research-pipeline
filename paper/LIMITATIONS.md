# Limitations and Scope

This document outlines the known limitations of the Format Friction study on structured output compliance in LLM tool-calling systems.

---

## Model Limitations

### Single Model Family
- **Limitation**: All experiments conducted exclusively on Claude Sonnet (claude-sonnet-4-5-20250929).
- **Impact**: Format friction may be model-specific. Models with different tool-use training (GPT-4, Gemini, Llama) may show different friction profiles.
- **Mitigation**: Future work should validate on multiple model families. Ollama-compatible models (Qwen 2.5, Llama 3.1) identified for local replication.

### Model Version Variability
- **Limitation**: Results tied to a specific model snapshot.
- **Impact**: Friction rates may change with model updates.
- **Mitigation**: Experiment metadata records model version and timestamps.

### Temperature and Sampling
- **Limitation**: Default temperature used; not explicitly controlled.
- **Impact**: Some response variability expected.
- **Mitigation**: 10 trials per scenario provide variance estimation. Key findings (100% EXPLICIT compliance) are at ceiling, unaffected by sampling.

---

## Task and Scenario Limitations

### Single Task Domain
- **Limitation**: Only signal detection (frustration/urgency/blocking) tested.
- **Impact**: Friction patterns may differ for other tool-calling tasks:
  - Memory operations
  - File system interactions
  - API calls with complex parameters
  - Multi-step tool sequences
- **Mitigation**: Task chosen for clean measurement (binary signal present/absent). Generalization requires follow-up studies.

### Scenario Difficulty Variation
- **Limitation**: Three IMPLICIT scenarios flagged as potentially too subtle during piloting (sig_implicit_frust_007, sig_implicit_block_001, sig_implicit_block_008).
- **Impact**: These "HARD" scenarios may inflate friction estimates.
- **Sensitivity Analysis**: Excluding HARD scenarios, friction drops from 12.2pp to 10.3pp. Main finding remains robust.

### Scenario Relabeling
- **Limitation**: Three scenarios relabeled from IMPLICIT to CONTROL during development after 0% detection in pilot.
- **Impact**: Post-hoc relabeling could affect estimates.
- **Transparency**: Relabeling documented in code with rationale. Original labels available in version history.

### English Only
- **Limitation**: All scenarios in English.
- **Impact**: Format friction behavior on other languages unknown.
- **Recommendation**: Future work should include multilingual scenarios.

---

## Measurement Limitations

### Judge-Human Agreement on IMPLICIT Signals
- **Limitation**: Overall κ = 0.81 (substantial), but IMPLICIT stratum shows κ = 0.41 (moderate).
- **Impact**: 24% disagreement on IMPLICIT scenarios where the main friction finding (18.4pp) originates.
- **Analysis**: Human annotators tended not to count "helpful without explicit acknowledgment" as detection. Judge is more permissive.
- **Sensitivity**: Excluding disagreement cases, IMPLICIT friction remains ~17pp. Finding is robust.

### LLM Judge (Same Model Family)
- **Limitation**: Judge model (Claude Sonnet) is same family as subject model.
- **Impact**: Potential systematic bias if both share similar detection patterns.
- **Mitigation**: Human validation subsample (n=150) achieved κ = 0.81. Consider different model family for judging in future work.

### Binary Detection Measure
- **Limitation**: Judge outputs YES/NO; cannot capture partial detection or detection quality.
- **Impact**: Nuanced signal acknowledgment may be lost.
- **Trade-off**: Binary measure enables clean statistical analysis and matches production behavior (tool fires or doesn't).

### Regex Measurement Asymmetry
- **Limitation**: Regex detection uses keyword patterns (NL) vs XML parsing (structured).
- **Impact**: Regex undercounts NL responses, inflating apparent format effect.
- **Mitigation**: Within-condition analysis (detection vs compliance) avoids cross-condition measurement asymmetry.

---

## Statistical Limitations

### Effect Size Interpretation
- **Detection gap**: 3.8pp (NL 89.4% vs ST 85.6%), p = 0.065
- **Compliance gap**: 10.3pp (ST detection 85.6% vs compliance 75.3%)
- **Context**: Effects are meaningful for production systems processing thousands of requests, but not catastrophic impairment.

### Confidence Intervals
Key estimates with 95% Wilson score CIs (primary analysis, excluding HARD scenarios, N=340):

| Metric | Estimate | 95% CI |
|--------|----------|--------|
| NL detection | 89.4% | [85.8%, 92.2%] |
| ST detection | 85.6% | [81.5%, 89.0%] |
| ST compliance | 75.3% | [70.4%, 79.6%] |
| Format friction | 10.3pp | [5.8pp, 14.8pp]* |
| IMPLICIT friction | 18.4pp | ~[12pp, 25pp]** |

*Bootstrap CI (10,000 replicates).
**Approximate; derived from component CIs.

### Trial Non-Independence
- **Limitation**: 10 trials per scenario are not fully independent (same prompt, similar responses).
- **Impact**: Standard errors may be underestimated.
- **Mitigation**: Sign test at scenario level provides conservative inference.

---

## Two-Pass Recovery Limitations

### Single Task Tested
- **Limitation**: Recovery rates (74% Sonnet, 50% Qwen-7B) only validated on signal detection.
- **Impact**: Recovery may differ for other tool-calling domains.
- **Recommendation**: Test on additional tasks before generalizing.

### No Fine-Tuning
- **Limitation**: Extraction models used off-the-shelf, not task-specific fine-tuned.
- **Impact**: Recovery rates likely underestimate potential with fine-tuning.
- **Opportunity**: SLOT paper shows 99.5% schema accuracy achievable with fine-tuning.

### Cost Analysis Incomplete
- **Limitation**: Two-pass cost estimates don't account for latency or infrastructure overhead.
- **Impact**: Production deployment may reveal hidden costs.
- **Trade-off**: Even with overhead, 16pp compliance improvement may justify costs.

---

## Scope Clarifications

### Applies To: Prompt-Based Tool Calling
- Systems parsing XML/JSON from free-form LLM output (LangChain, ReAct, custom agents)
- Subject to format friction

### Does NOT Apply To: Native Tool-Use APIs
- Constrained decoding systems (OpenAI function calling, Anthropic tool_use)
- Format friction eliminated by construction
- Different trade-offs (cannot reason in NL before committing)

---

## Recommendations for Future Work

1. **Cross-model validation**: Test on GPT-4, Gemini, Llama, Qwen families
2. **Multi-task validation**: Memory operations, API calls, code generation
3. **Multilingual scenarios**: Non-English signal detection
4. **Independent annotation**: Multiple annotators for IMPLICIT scenarios
5. **Production evaluation**: Deploy two-pass architecture in real systems
6. **Fine-tuned extraction**: Train small models specifically for structure recovery
7. **Mechanistic analysis**: Attention patterns during structured vs NL output

---

## Conclusion

Despite these limitations, the study establishes:

1. **Format friction exists**: 10.3pp compliance gap [95% CI: 5.8pp, 14.8pp] in structured condition (50 compliance gaps / 340 trials)
2. **Concentrates in uncertainty**: 0pp EXPLICIT vs 18.4pp IMPLICIT friction
3. **Not reasoning impairment**: Detection rates similar across conditions (89% vs 86%)
4. **Recoverable**: Two-pass extraction recovers 50-74% of compliance gaps

The primary limitation is single-model testing—format friction is established for Claude Sonnet but generalization to other model families requires validation.

---

*Last updated: 2026-02-03*
