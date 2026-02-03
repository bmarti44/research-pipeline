# Academic Review: "Format Friction: The Compliance Gap in Prompt-Based Tool Calling"

**Reviewer**: Automated Academic Review
**Date**: 2026-02-03
**Review Type**: Rigorous methodological and statistical audit

---

## Overall Score: 72/100 (B-)

**Verdict**: A methodologically sound contribution with a genuinely novel finding, undermined by single-model/single-task scope and presentation inconsistencies. The core insight—that detection and compliance diverge in structured output conditions—is valid and reproducible. Publication-ready with revisions.

---

## Executive Summary

### What the Paper Does Well
1. **Identifies a real confound in prior work**: The observation that Tam et al. and Johnson et al. conflate format effects with prompt asymmetries is insightful and validated.
2. **Clean within-condition methodology**: Measuring detection vs. compliance *within* the structured condition elegantly sidesteps measurement asymmetry.
3. **Transparent limitation acknowledgment**: The authors proactively disclose single-model scope, HARD scenario exclusion, and judge reliability issues.
4. **Reproducible**: Code, data, and analysis pipeline are complete and verifiable.

### Critical Weaknesses
1. **Single model, single task**: All findings are from Claude Sonnet on signal detection. Generalization is entirely unknown.
2. **Low reliability on key stratum**: κ = 0.41 on IMPLICIT scenarios—exactly where the 18.4pp friction manifests.
3. **Confusing metrics**: "50 compliance gaps" vs "10.3pp friction" measure different things; the text conflates them.
4. **Underpowered for some claims**: HARD scenario analysis (n=30) and FPR comparison (5/230 structured FPs) are exploratory at best.

---

## Detailed Evaluation

### 1. Research Question and Motivation (Score: 85/100)

**Strengths**:
- The paper correctly identifies that prior format-effect studies (Tam et al., Johnson et al.) contain a suppression confound that could explain their results.
- The research question—"When format is the only variable, what is the actual impact?"—is well-posed and addresses a genuine gap.
- The shift to within-condition analysis (detection vs. compliance) is methodologically sophisticated.

**Weaknesses**:
- The framing overstates novelty. The compliance gap is a known issue in prompt-based systems; this paper quantifies it for one model on one task.
- The title "Format Friction" is catchy but potentially misleading—this is compliance failure, not reasoning friction.

**Verdict**: Strong motivation; appropriately scoped research question.

---

### 2. Experimental Design (Score: 78/100)

#### 2.1 Task Design

**Strengths**:
- Signal detection is a reasonable proxy for "soft" tool-calling scenarios.
- The structurally parallel prompts (both conditions receive identical "when" guidance) successfully eliminate the suppression confound.
- 75 scenarios across 4 ambiguity levels provide reasonable coverage.

**Weaknesses**:
- **Single task limitation**: Signal detection may not generalize to other tool-calling domains (API calls, memory operations, multi-step chains).
- **Scenario construction**: Some IMPLICIT scenarios were relabeled post-hoc after 0% pilot detection (sig_implicit_frust_005, sig_implicit_urg_003, sig_implicit_block_005). This is disclosed but introduces potential bias.
- **HARD exclusion rationale**: The decision to exclude 3 scenarios based on pilot results is reasonable but the threshold is arbitrary. Why not include them with appropriate caveats?

#### 2.2 Sample Size

**Verified**:
- 75 scenarios × 10 trials × 2 conditions = 1,500 observations
- Primary analysis: 34 scenarios × 10 trials = 340 trials (excluding HARD)

**Concerns**:
- No power analysis reported. For the within-condition friction test, 340 trials provides adequate power for detecting 10pp differences, but marginal for the 3.8pp cross-condition effect.
- HARD scenario analysis (n=30) is underpowered for any meaningful inference.

#### 2.3 Measurement

**Judge Validation** (κ = 0.81):
- Overall agreement is acceptable.
- **Critical issue**: IMPLICIT stratum κ = 0.41 (76% agreement). This is "moderate" agreement per Landis & Koch, below the 0.60 threshold typically required for research instruments. The main finding (18.4pp IMPLICIT friction) rests on a measurement with questionable reliability.

**Recommendation**: The paper should either (a) report a sensitivity analysis excluding judge-human disagreement cases, or (b) acknowledge that IMPLICIT friction estimates have wide uncertainty due to measurement unreliability.

*Note: The authors do report sensitivity analysis (17.2pp excluding disagreements), which is commendable.*

---

### 3. Statistical Analysis (Score: 75/100)

#### 3.1 Verified Claims

I independently verified the following statistics from the primary data file:

| Claim | Paper | Verified | Status |
|-------|-------|----------|--------|
| N (excluding HARD) | 340 | 340 | ✓ |
| NL detection rate | 89.4% | 89.4% (304/340) | ✓ |
| ST detection rate | 85.6% | 85.6% (291/340) | ✓ |
| ST compliance rate | 75.3% | 75.3% (256/340) | ✓ |
| Format friction | 10.3pp | 10.3pp | ✓ |
| Compliance gaps | 50 | 50 | ✓ |
| EXPLICIT friction | 0.0pp | 0.0pp | ✓ |
| IMPLICIT friction | 18.4pp | 18.4pp (excluding HARD) | ✓ |

**All primary statistics verified.**

#### 3.2 Statistical Issues

**Issue 1: Compliance Gaps vs. Friction Rate Confusion**

The paper states "50 compliance gaps" but friction is 10.3pp = (291-256)/340 = 35/340. These differ because:
- 50 trials have detection without XML (compliance gaps)
- 15 trials have XML without detection (reverse gaps)
- Net difference: 35

This is mathematically correct but potentially confusing. The paper should clarify that "50 compliance gaps" is the gross count, while "10.3pp" is the net rate difference.

**Issue 2: McNemar vs. Sign Test Interpretation**

- Trial-level McNemar: p = 0.065 (not significant at α = 0.05)
- Scenario-level sign test: p = 0.0007 (highly significant)

The paper correctly notes that trial-level tests are inappropriate due to within-scenario correlation, but then reports both. The sign test is the appropriate primary test; McNemar should be relegated to a footnote or removed.

**Issue 3: Confidence Interval for Friction**

The paper reports friction CI as [5.8pp, 14.8pp] via bootstrap. This is reasonable, but the bootstrap should account for the paired structure of trials. Verification of bootstrap methodology not possible without access to the bootstrap code.

#### 3.3 False Positive Rate Paradox

The paper notes an "apparent contradiction": if structured output causes hedging (fewer TPs on uncertain signals), we'd expect fewer FPs too. Instead: 2.2% FPR structured vs. 0.0% NL.

**Analysis**:
- Structured FPs: 5/230 = 2.2%
- NL FPs: 0/230 = 0.0%

This is based on 5 false positives. With n=5, the 95% CI for structured FPR is [0.7%, 5.0%]. The "paradox" may be noise. The paper's speculative explanations (threshold shift, XML commitment dynamics) are reasonable but untestable with current data.

---

### 4. Two-Pass Recovery Analysis (Score: 70/100)

#### 4.1 Design

The two-pass recovery is tested on the 50 compliance gap trials—cases where detection occurred but no XML was produced. This is the appropriate test population.

**Verified from data**:
- Sonnet tested on compliance gaps: Recovery methodology appears sound
- Qwen 7B tested on same gaps: Provides cost comparison

#### 4.2 Concerns

1. **No baseline comparison**: What's the extraction success rate on trials where the original model *did* produce XML? Without this baseline, we can't assess whether recovery is model-specific.

2. **Type accuracy discrepancy**: Sonnet 81.1% vs. Qwen 7B 36.0% type accuracy. This large gap suggests the task may be sensitive to model capability in ways that matter for production deployment.

3. **Cost claims unverified**: "~1× + local" for Qwen 7B assumes local inference, but doesn't account for latency or infrastructure overhead.

4. **Effective compliance estimates**: The paper reports:
   - NL → Sonnet: ~94%
   - NL → 7B: ~88%

   These are upper bounds assuming perfect precision. Actual production rates would be lower due to false extractions.

---

### 5. Reproducibility (Score: 90/100)

**Strengths**:
- Complete code repository with clear structure
- Data manifest explicitly documents which files support which claims
- Git history preserves version control
- Failed experiments documented (not deleted)
- Human validation data available

**Weaknesses**:
- No requirements.txt or lockfile for exact dependency versions
- Temperature/sampling parameters not explicitly controlled
- Random seed documented (42) but not verified across all runs

**Verdict**: Highly reproducible by research standards.

---

### 6. Presentation Quality (Score: 68/100)

**Strengths**:
- Clear abstract with concrete numbers
- Good use of tables for statistical summaries
- Appendices provide necessary detail (prompts, judge validation)

**Weaknesses**:
- Figures reference filenames that don't match paper (fig1_ambiguity_interaction vs fig2 in code)
- Some redundancy between main text and limitations section
- The "format friction" terminology is introduced but never formally defined until deep in the paper
- DATA_MANIFEST.md reports different numbers (12.2pp, 60 failures) than paper (10.3pp, 50)—these are for different N values but could confuse readers reviewing the repository

---

### 7. Limitations Assessment (Score: 82/100)

The authors provide a thorough LIMITATIONS.md document covering:
- Single model family
- Single task domain
- Judge reliability issues
- HARD scenario exclusion
- Statistical caveats

This transparency is commendable and exceeds typical paper standards.

**Missing limitations**:
1. No discussion of prompt sensitivity—would different phrasing of the "how to flag" instructions change friction rates?
2. No analysis of response length effects—are compliance gaps correlated with response verbosity?
3. No examination of whether friction varies by signal type (frustration vs. urgency vs. blocking)

---

## Data Integrity Audit

### Verified Data Files

| File | Purpose | Integrity |
|------|---------|-----------|
| signal_detection_20260203_074411_judged.json | Primary results | ✓ Valid, 750 trials |
| signal_detection_20260203_121413.json | Raw experiment | ✓ Valid, 1500 observations |
| two_pass_sonnet_nl_20260203_125603.json | Recovery testing | ✓ Valid, 750 trials |
| two_pass_qwen7b_nl_20260203_131141.json | Recovery testing | ✓ Valid, 750 trials |

### Code Verification

| Component | Status |
|-----------|--------|
| Wilson CI implementation | ✓ Correct formula |
| McNemar test implementation | ✓ Correct |
| Sign test implementation | ✓ Uses scipy.binomtest |
| Judge prompt | ✓ Matches paper Appendix B |
| Scenario definitions | ✓ 75 scenarios, HARD flagged |

### Discrepancy Log

| Issue | Explanation | Severity |
|-------|-------------|----------|
| DATA_MANIFEST says 60 failures, paper says 50 | MANIFEST includes HARD; paper excludes | Low (documented) |
| DATA_MANIFEST says 12.2pp, paper says 10.3pp | Same reason | Low (documented) |
| Figure numbering mismatch (code vs paper) | Code generates fig2, fig3, fig4, fig5; paper references fig1-4 | Low (cosmetic) |

---

## Specific Corrections Required

### Errors

1. **Table 2 presentation**: "50 compliance gaps" appears alongside "10.3pp friction" without explaining these measure different things (gross count vs. net rate). Add a footnote clarifying: "50 trials showed detection without XML; the net rate difference (10.3pp) accounts for 15 trials with XML but no judged detection."

2. **Section 3.6 HARD scenarios**: The claim "33% detection but 0% XML compliance (+33pp friction)" is based on n=30 trials from 3 scenarios. This should be explicitly labeled as exploratory/underpowered, not just "small sample warrants caution."

### Clarifications Needed

1. Define "format friction" formally in Section 1 or 2, not just by example.

2. Explain why the sign test is the primary test and McNemar is secondary (within-scenario correlation).

3. Address whether compliance gaps are concentrated in specific scenarios or distributed uniformly across IMPLICIT trials.

### Missing Analyses

1. **Signal type breakdown**: Is friction higher for frustration vs. urgency vs. blocking_issue?

2. **Scenario-level variance**: Do some IMPLICIT scenarios show 0% friction while others show 40%+?

3. **Response length correlation**: Are longer responses more likely to show compliance gaps?

---

## Scoring Breakdown

| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| Research Question | 15% | 85 | 12.75 |
| Experimental Design | 20% | 78 | 15.60 |
| Statistical Analysis | 25% | 75 | 18.75 |
| Two-Pass Analysis | 10% | 70 | 7.00 |
| Reproducibility | 15% | 90 | 13.50 |
| Presentation | 10% | 68 | 6.80 |
| Limitations Handling | 5% | 82 | 4.10 |
| **Total** | **100%** | — | **72.50** |

---

## Recommendation

**Accept with Major Revisions**

The paper makes a valid contribution by:
1. Identifying and controlling for the suppression confound in prior work
2. Introducing within-condition analysis as a cleaner methodology
3. Quantifying format friction at 10.3pp with appropriate uncertainty bounds

However, significant revisions are needed:
1. Address the compliance gaps vs. friction rate confusion
2. Acknowledge measurement unreliability on IMPLICIT stratum more prominently
3. Reframe HARD scenario analysis as exploratory
4. Add missing analyses (signal type breakdown, scenario-level variance)

The single-model/single-task scope is a fundamental limitation that cannot be addressed without additional experiments. The paper should be more conservative in its claims about generalization.

---

## Comparison to Related Work

| Paper | Task | Models | Format Effect Claimed |
|-------|------|--------|----------------------|
| Tam et al. (2024) | Reasoning benchmarks | Multiple | 27.3pp (confounded) |
| Johnson et al. (2025) | Tool-calling | 10 models | 18.4pp (confounded) |
| **This paper** | Signal detection | 1 model | 10.3pp (compliance gap) |

The key contribution is methodological: showing that apparent format effects can be decomposed into detection impairment (small: ~4pp) and compliance failure (larger: ~10pp). This is a more nuanced picture than prior work, but applies only to the tested model and task.

---

## Final Assessment

This is a competent research contribution with a genuine insight: format requirements in prompt-based tool-calling create a compliance gap distinct from reasoning impairment. The methodology is sound, the statistics are verified, and the limitations are honestly acknowledged.

The paper falls short of excellence due to:
- Narrow scope (single model, single task)
- Measurement reliability concerns on the key stratum
- Presentation issues that obscure rather than clarify

**Score: 72/100 (B-)**

For a top-tier venue (NeurIPS, ICML), multi-model validation would be required. For a workshop or applications track, the current contribution is acceptable with revisions.

---

## Remediation Plan

This section provides a concrete plan for addressing each major issue identified in this review, **excluding multi-model validation** (which requires substantial new experimentation).

---

### Issue 1: Low Reliability on IMPLICIT Stratum (κ = 0.41)

**Problem**: The judge-human agreement on IMPLICIT scenarios is only 76%, which undermines confidence in the 18.4pp friction estimate where the main finding manifests.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **1a. Compute bounded estimates from existing data** | 1 hour | High |
| - Report friction under "optimistic" (judge) and "conservative" (human) interpretations | | |
| - Use existing validation data to compute bounds | | |
| **1b. Add uncertainty quantification to Table 3** | 30 min | Medium |
| - Show friction range: 18.4pp (judge) to ~14pp (conservative) | | |
| **1c. Expand sensitivity analysis in text** | 1 hour | Medium |
| - Already have 17.2pp excluding disagreement cases | | |
| - Add explicit statement: "IMPLICIT friction estimates have ±4pp uncertainty due to measurement reliability" | | |

**Implementation**:

```python
def compute_bounded_friction(results: list[dict], validation_data: dict) -> dict:
    """Compute friction bounds using existing human validation."""
    implicit = [r for r in results
                if r.get('ambiguity') == 'IMPLICIT'
                and r.get('expected_detection') is True]

    # Optimistic (judge-based)
    det_judge = sum(1 for r in implicit if r.get('st_judge_detected') is True)
    comp = sum(1 for r in implicit if r.get('st_regex_detected') is True)
    friction_optimistic = (det_judge - comp) / len(implicit) * 100

    # Conservative (discount by disagreement rate)
    # IMPLICIT disagreement rate = 24% from validation
    disagreement_rate = 0.24
    det_conservative = det_judge * (1 - disagreement_rate)
    friction_conservative = (det_conservative - comp) / len(implicit) * 100

    return {
        'friction_optimistic': friction_optimistic,
        'friction_conservative': friction_conservative,
        'uncertainty_pp': friction_optimistic - friction_conservative
    }
```

**Deliverable**: Updated Table 3 with bounded estimates, expanded sensitivity discussion.

**Timeline**: 2.5 hours

---

### Issue 2: Confusing Metrics (50 Gaps vs. 10.3pp Friction)

**Problem**: "50 compliance gaps" and "10.3pp friction" measure different things (gross count vs. net rate), but the paper presents them without clarification.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **2a. Add explanatory footnote to Table 2** | 30 min | High |
| - "50 trials showed detection without XML (compliance gaps). The net friction rate (10.3pp) differs because 15 trials showed XML without judged detection (reverse gaps)." | | |
| **2b. Add 2×2 contingency table** | 1 hour | High |
| - Show full breakdown: Both/Neither/Detection-only/XML-only | | |
| **2c. Discuss reverse gaps in §4.4** | 1 hour | Medium |
| - What do the 15 reverse gaps represent? False positive XML? Overly strict judge? | | |

**Deliverable**: Revised Table 2 with footnote, new Table 2a with contingency breakdown.

**Timeline**: 2 hours

**Implementation**:

```markdown
## New Table 2a: Detection × Compliance Contingency

|                    | XML Present | XML Absent |
|--------------------|-------------|------------|
| **Judge: Detected**    | 241         | 50 (compliance gaps) |
| **Judge: Not Detected**| 15 (reverse gaps) | 34         |

Note: Net friction = (291-256)/340 = 10.3pp. Gross compliance gaps = 50.
```

---

### Issue 3: Underpowered HARD Scenario Analysis

**Problem**: The claim "33% detection, 0% compliance (+33pp friction)" is based on n=30 trials from 3 scenarios—too small for reliable inference.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **3a. Reframe as exploratory observation** | 30 min | High |
| - Change "HARD scenarios show extreme friction" to "Preliminary observation: HARD scenarios (n=30) showed elevated friction, suggesting a potential dose-response relationship with ambiguity" | | |
| **3b. Add explicit power limitation** | 15 min | Medium |
| - "With n=30, the 95% CI for 33% detection is [17%, 52%]; inference is unreliable" | | |
| **3c. Compute and report wide confidence intervals** | 30 min | Medium |
| - Use Wilson score CI for HARD detection/compliance rates | | |
| - Make uncertainty visually clear in any HARD-related figures | | |

**Implementation**:

```python
def hard_scenario_uncertainty(results: list[dict]) -> dict:
    """Compute CIs for HARD scenarios to show uncertainty."""
    hard_ids = ['sig_implicit_frust_007', 'sig_implicit_block_001', 'sig_implicit_block_008']
    hard = [r for r in results if r.get('scenario_id') in hard_ids]

    det = sum(1 for r in hard if r.get('st_judge_detected') is True)
    comp = sum(1 for r in hard if r.get('st_regex_detected') is True)
    n = len(hard)

    det_ci = wilson_ci(det, n)
    comp_ci = wilson_ci(comp, n)

    return {
        'n': n,
        'detection': det / n,
        'detection_ci': det_ci,  # Expected: ~[17%, 52%]
        'compliance': comp / n,
        'compliance_ci': comp_ci,  # Expected: [0%, 12%]
        'note': 'Wide CIs reflect underpowered sample'
    }
```

**Timeline**: 1.25 hours

---

### Issue 4: Missing Analyses

**Problem**: Three analyses would strengthen the paper but are absent: signal type breakdown, scenario-level variance, response length correlation.

**Remediation Plan**:

#### 4a. Signal Type Breakdown

| Action | Effort | Impact |
|--------|--------|--------|
| Add analysis script section | 2 hours | Medium |
| Report friction by frustration/urgency/blocking_issue | | |

**Implementation** (add to `analyze_judged_results.py`):

```python
def analyze_friction_by_signal_type(results: list[dict]) -> dict:
    """Friction breakdown by signal type (frustration, urgency, blocking_issue)."""
    analysis = {}
    for signal_type in ['frustration', 'urgency', 'blocking_issue']:
        subset = [r for r in results
                  if r.get('signal_type') == signal_type
                  and r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]
        if not subset:
            continue
        detection = sum(1 for r in subset if r.get('st_judge_detected') is True)
        compliance = sum(1 for r in subset if r.get('st_regex_detected') is True)
        n = len(subset)
        analysis[signal_type] = {
            'n': n,
            'detection_rate': detection / n,
            'compliance_rate': compliance / n,
            'friction_pp': (detection - compliance) / n * 100
        }
    return analysis
```

**Expected output**: New Table showing whether friction is uniform across signal types or concentrated (e.g., blocking_issue might have higher friction than frustration).

#### 4b. Scenario-Level Variance

| Action | Effort | Impact |
|--------|--------|--------|
| Compute per-scenario friction rates | 1 hour | High |
| Identify high-friction scenarios | | |
| Report variance/IQR | | |

**Implementation**:

```python
def scenario_friction_distribution(results: list[dict]) -> dict:
    """Compute friction for each IMPLICIT scenario."""
    from collections import defaultdict
    by_scenario = defaultdict(list)

    for r in results:
        if r.get('ambiguity') == 'IMPLICIT' and r.get('expected_detection') is True:
            by_scenario[r['scenario_id']].append(r)

    frictions = []
    for scenario_id, trials in by_scenario.items():
        det = sum(1 for t in trials if t.get('st_judge_detected') is True)
        comp = sum(1 for t in trials if t.get('st_regex_detected') is True)
        n = len(trials)
        friction = (det - comp) / n * 100 if n > 0 else 0
        frictions.append({'scenario_id': scenario_id, 'friction_pp': friction, 'n': n})

    return {
        'scenarios': sorted(frictions, key=lambda x: -x['friction_pp']),
        'mean_friction': np.mean([f['friction_pp'] for f in frictions]),
        'std_friction': np.std([f['friction_pp'] for f in frictions]),
        'max_friction': max(f['friction_pp'] for f in frictions),
        'min_friction': min(f['friction_pp'] for f in frictions),
    }
```

**Expected insight**: Does friction concentrate in a few "pathological" scenarios, or is it broadly distributed?

#### 4c. Response Length Correlation

| Action | Effort | Impact |
|--------|--------|--------|
| Correlate response length with compliance gaps | 1 hour | Medium |

**Implementation**:

```python
def response_length_analysis(results: list[dict]) -> dict:
    """Analyze whether longer responses have more compliance gaps."""
    with_truth = [r for r in results
                  if r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']
                  and r.get('expected_detection') is True]

    # Get response lengths for compliance gaps vs successful compliance
    gap_lengths = [len(r.get('st_response_text', ''))
                   for r in with_truth
                   if r.get('st_judge_detected') is True
                   and r.get('st_regex_detected') is not True]

    success_lengths = [len(r.get('st_response_text', ''))
                       for r in with_truth
                       if r.get('st_regex_detected') is True]

    from scipy.stats import mannwhitneyu
    stat, pvalue = mannwhitneyu(gap_lengths, success_lengths, alternative='two-sided')

    return {
        'gap_mean_length': np.mean(gap_lengths),
        'success_mean_length': np.mean(success_lengths),
        'mann_whitney_p': pvalue,
    }
```

**Hypothesis**: Compliance gaps may be associated with longer, more elaborate responses where the model "talks around" the signal rather than tagging it.

**Timeline for all 4a-4c**: 4-6 hours

---

### Issue 5: Statistical Presentation (McNemar vs. Sign Test)

**Problem**: The paper reports both McNemar (p=0.065) and sign test (p=0.0007) without clearly explaining why sign test is primary.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **5a. Add explanatory paragraph** | 30 min | High |
| - "We report the scenario-level sign test as primary because trials within a scenario are correlated (same prompt, similar responses). McNemar's test assumes independence, making it anticonservative here. The sign test treats each scenario as a single observation, providing valid inference." | | |
| **5b. Move McNemar to footnote** | 15 min | Medium |
| - Or remove entirely from main text | | |

**Timeline**: 45 minutes

---

### Issue 6: Presentation Issues

**Problem**: Multiple presentation inconsistencies reduce clarity.

**Remediation Plan**:

| Issue | Action | Effort |
|-------|--------|--------|
| **6a. Formal definition of "format friction"** | Add to §1 or §2 | 30 min |
| "We define *format friction* as the within-condition gap between signal detection (as measured by an LLM judge) and format compliance (as measured by XML tag presence) in structured output responses." | | |
| **6b. Figure numbering** | Rename fig2→fig1, fig3→fig2, etc. in code and paper | 30 min |
| **6c. Sync DATA_MANIFEST with paper** | Add note clarifying that MANIFEST reports full dataset (N=370) while paper primary analysis excludes HARD (N=340) | 15 min |
| **6d. Remove redundancy** | Cut overlapping content between §4.6 Limitations and LIMITATIONS.md | 1 hour |

**Timeline**: 2-3 hours

---

### Issue 7: Two-Pass Recovery Baseline

**Problem**: Recovery rates (74%, 50%) lack context—what's the extraction success on trials that *did* produce XML?

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **7a. Run extraction on XML-present trials** | 2 hours | High |
| - Sample 50 trials where XML was produced | | |
| - Run Sonnet and Qwen extraction on original NL response | | |
| - Compare extraction success vs. original XML | | |
| **7b. Report baseline** | 30 min | Medium |
| - "On trials where the model did produce XML, extraction from NL achieved X% agreement, establishing a Y% baseline vs. Z% recovery rate" | | |

**Implementation**:

```python
# Add to two_pass_extraction.py
def baseline_extraction(results: list[dict], extraction_model: str) -> dict:
    """Extract from NL responses of trials that DID produce XML."""
    xml_present = [r for r in results
                   if r.get('st_regex_detected') is True
                   and r.get('ambiguity') in ['EXPLICIT', 'IMPLICIT']]

    # Sample 50 for comparison
    sample = random.sample(xml_present, min(50, len(xml_present)))

    # Run extraction on NL responses
    # Compare to original XML tag
    ...
```

**Timeline**: 3 hours

---

### Issue 8: Reproducibility Gaps

**Problem**: Missing requirements.txt, undocumented temperature settings.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **8a. Create requirements.txt** | 15 min | High |
| ```claude-agent-sdk>=0.1.20``` | | |
| ```sentence-transformers>=2.2.0``` | | |
| ```numpy>=1.24.0``` | | |
| ```scipy>=1.10.0``` | | |
| ```matplotlib>=3.7.0``` | | |
| **8b. Document temperature** | 30 min | Medium |
| - Add to experiment metadata: `"temperature": "default (1.0)"` | | |
| - Discuss in limitations: "Temperature was not explicitly controlled; default settings used" | | |
| **8c. Verify seed consistency** | 1 hour | Low |
| - Check that seed=42 propagates to all random operations | | |

**Timeline**: 2 hours

---

### Issue 9: FPR Paradox Resolution

**Problem**: 2.2% structured FPR vs 0% NL FPR is noted but unexplained. Sample (n=5 FPs) is too small for inference.

**Remediation Plan**:

| Action | Effort | Impact |
|--------|--------|--------|
| **9a. Reframe as observation with statistical context** | 30 min | Medium |
| - "We observed 5 false positives in structured vs. 0 in NL (p=0.06, Fisher's exact). With only 5 events, this may be noise; we note it as an observation for future investigation." | | |
| **9b. Add confidence intervals** | 15 min | Medium |
| - Report 95% CI for structured FPR: [0.7%, 5.0%] | | |
| - Report 95% CI for NL FPR: [0%, 1.6%] (one-sided) | | |
| **9c. Remove speculative explanations** | 15 min | Low |
| - Current §4.4 offers untestable hypotheses (threshold shift, XML commitment) | | |
| - Replace with: "The mechanism, if real, requires further investigation" | | |

**Implementation**:

```python
def fpr_with_uncertainty(results: list[dict]) -> dict:
    """Compute FPR with appropriate uncertainty for small samples."""
    control = [r for r in results if r.get('ambiguity') == 'CONTROL']
    n = len(control)

    nl_fp = sum(1 for r in control if r.get('nl_judge_detected') is True)
    st_fp = sum(1 for r in control if r.get('st_judge_detected') is True)

    from scipy.stats import fisher_exact
    table = [[nl_fp, n - nl_fp], [st_fp, n - st_fp]]
    odds_ratio, p_value = fisher_exact(table)

    return {
        'nl_fpr': nl_fp / n,
        'nl_fpr_ci': wilson_ci(nl_fp, n),
        'st_fpr': st_fp / n,
        'st_fpr_ci': wilson_ci(st_fp, n),
        'fisher_p': p_value,
        'interpretation': 'Not significant; treat as exploratory observation'
    }
```

**Timeline**: 1 hour

---

### Issue 10: Single Task Limitation (Partial Mitigation)

**Problem**: Signal detection may not generalize to other tool-calling domains.

**Remediation Plan** (text-based mitigations only):

| Action | Effort | Impact |
|--------|--------|--------|
| **10a. Add explicit scope statement** | 30 min | Medium |
| - "These findings are validated on signal detection only. Generalization to other tool-calling tasks (API calls, memory operations, code generation) requires future validation." | | |
| **10b. Conceptual analysis section** | 2 hours | Medium |
| - Add discussion: "Why might friction generalize (or not)?" | | |
| - Argument for: Friction is about output format commitment, not task semantics | | |
| - Argument against: Different tasks may have different ambiguity profiles | | |
| **10c. Connect to related work** | 1 hour | Medium |
| - Note that Wang et al. (SLOT) found format compliance issues across multiple tasks | | |
| - Tam et al. tested multiple reasoning benchmarks | | |
| - Our contribution is methodology (within-condition), not task breadth | | |

**Implementation** (add to Discussion section):

```markdown
### 4.X Generalization Considerations

Our findings are validated on signal detection only. We offer two perspectives on generalization:

**Arguments for generalization**: Format friction appears to reflect output commitment
under uncertainty—a general property of language model generation, not task-specific
reasoning. The model's reluctance to produce XML when uncertain about the signal
should manifest whenever structured output requires committing to uncertain claims.

**Arguments against generalization**: Signal detection involves recognizing subtle
emotional/situational cues—a task where uncertainty is inherent. Tool-calling tasks
with clearer triggering conditions (e.g., "save this to memory" vs. detecting implicit
frustration) may show less friction because ambiguity is lower.

Empirical validation across tasks remains necessary for strong generalization claims.
```

**Timeline**: 3.5 hours

---

## Remediation Summary

### Priority 1: Must-Fix (Before Submission)

| Issue | Action | Time |
|-------|--------|------|
| Metrics confusion | Add Table 2a, footnote | 2 hours |
| HARD underpowered | Reframe as exploratory | 1 hour |
| Statistical presentation | Clarify sign test primary | 45 min |
| Format friction definition | Add formal definition | 30 min |
| **Subtotal** | | **~4 hours** |

### Priority 2: Should-Fix (Strengthens Paper)

| Issue | Action | Time |
|-------|--------|------|
| Missing analyses | Signal type, scenario variance, response length | 5 hours |
| IMPLICIT reliability | Expand annotation, bounded estimates | 6 hours |
| Presentation cleanup | Figure numbers, MANIFEST sync | 2 hours |
| Reproducibility | requirements.txt, temperature docs | 2 hours |
| **Subtotal** | | **~15 hours** |

### Priority 3: Nice-to-Have (For Top Venues)

| Issue | Action | Time |
|-------|--------|------|
| Two-pass baseline | Run extraction on XML-present trials | 3 hours |
| Expanded CONTROL | More scenarios for FPR analysis | 1 week |
| Pilot second task | Memory persistence validation | 2 weeks |
| **Subtotal** | | **~3 weeks** |

---

## Expected Impact of Remediation

| Metric | Current | After Priority 1+2 | After All |
|--------|---------|---------------------|-----------|
| Review Score | 72/100 | 80-82/100 | 85-88/100 |
| Venue Target | Workshop | Main track (borderline) | Main track (solid) |
| Key Weakness | Presentation, reliability | Single task | (Addressed except multi-model) |

---

*Remediation plan completed: 2026-02-03*

---

*Review completed: 2026-02-03*
*Methodology: Full data audit, code verification, statistical replication*
