# Format Friction Phase Transition: Exploratory Research Plan

**Status**: Hypothesis stage - requires validation
**Potential venue**: NeurIPS 2026 / ICML 2027
**Estimated effort**: 6-8 weeks if validation succeeds
**Estimated cost**: $3-5K API costs

---

## Executive Summary

The current Format Friction paper documents a 12.2pp compliance gap on average. But buried in the HARD scenario data is a more striking finding: on maximally ambiguous signals, friction approaches **70 percentage points**—near-complete compliance collapse even when detection remains high.

**Core hypothesis**: Format friction scales non-linearly with task difficulty, exhibiting a phase transition. Below a difficulty threshold, friction is negligible; above it, compliance collapses catastrophically.

If validated, this explains the benchmark-production gap (benchmarks test easy cases), provides a scaling law, and motivates difficulty-aware architecture selection.

---

## 1. The Seed Finding

### Current Data (HARD Scenarios, n=30)

| Scenario | NL Detection | ST Detection | XML Compliance | Friction |
|----------|--------------|--------------|----------------|----------|
| sig_implicit_frust_007 | 0% | 0% | 0% | 0pp (too subtle for both) |
| sig_implicit_block_001 | 90% | 30% | 0% | +30pp |
| sig_implicit_block_008 | 90% | 70% | **0%** | **+70pp** |

**The striking observation**: On sig_implicit_block_008, the model detects the signal 70% of the time but produces XML **0% of the time**. This isn't friction—it's complete compliance collapse under ambiguity.

### Comparison Across Ambiguity Levels

| Ambiguity Level | N | Detection (ST) | Compliance (XML) | Friction |
|-----------------|---|----------------|------------------|----------|
| EXPLICIT | 150 | 100.0% | 100.0% | 0.0pp |
| IMPLICIT (non-HARD) | 190 | 73.2% | 54.7% | 18.5pp |
| IMPLICIT (HARD) | 30 | 33.3% | 0.0% | 33.3pp |

**Pattern**: Friction increases with ambiguity, possibly non-linearly.

---

## 2. The Hypothesis

### Primary Claim

> Format friction follows a **phase transition**: below a difficulty threshold, friction is negligible (0-5pp); above it, compliance collapses toward zero even when detection remains substantial.

### Formalization

Let:
- D(x) = P(model detects signal | input x)
- C(x) = P(model produces valid XML | input x)
- F(x) = D(x) - C(x) = format friction
- d(x) = difficulty/ambiguity of input x

**Hypothesis**: F(d) exhibits sigmoid-like behavior:

```
Friction
   |
70%|                          ___________
   |                        /
   |                      /
   |                    /
20%|              ____/
   |         ___/
 0%|________/
   +---------------------------------> Difficulty
        Easy    Medium    Hard    Extreme
```

### Alternative Hypotheses

1. **Linear scaling**: F(d) = α·d (less interesting, no phase transition)
2. **Threshold effect**: F(d) = 0 for d < τ, F(d) = constant for d ≥ τ
3. **No relationship**: F(d) is independent of d (current data is noise)

---

## 3. Why This Matters

### 3.1 Explains the Benchmark-Production Gap

**Known problem**: Practitioners report unreliable tool-calling in production, but benchmarks show high accuracy.

**Resolution**: Benchmarks test easy cases (explicit intents, clear parameters). Production involves hard cases (ambiguous intents, implicit parameters). If friction is ~0% on easy and ~70% on hard, both observations are correct.

### 3.2 Provides a Scaling Law

Prior work: "Format hurts performance" (binary claim)
This work: "Format hurts as f(difficulty), with predictable scaling" (quantitative law)

Scaling laws are highly valued because they:
- Enable prediction without exhaustive testing
- Reveal underlying mechanism
- Guide architectural decisions

### 3.3 Motivates Adaptive Architecture

**Implication**: Use direct structured output for easy tasks (cheap, sufficient), two-pass for hard tasks (expensive, necessary).

| Task Difficulty | Recommended Architecture | Rationale |
|-----------------|-------------------------|-----------|
| Easy | Direct XML | Friction ~0%, two-pass overhead unnecessary |
| Medium | Direct XML + fallback | Friction moderate, retry may suffice |
| Hard | Two-pass (NL → extraction) | Friction catastrophic, two-pass essential |

### 3.4 Testable Prediction

Any benchmark can be stratified by difficulty. The scaling law predicts:
- Format effects will be small on easy strata
- Format effects will be large on hard strata
- The ratio should follow the scaling function

This is falsifiable and generalizable.

---

## 4. Experimental Plan

### Phase 0: Cheap Validation (Week 1)

**Goal**: Determine if the phase transition is real before committing resources.

**Design**:
- 10 scenarios at each of 5 difficulty levels (50 scenarios)
- 10 trials per scenario (500 trials total)
- Single model (Claude Sonnet)
- Single task (signal detection)

**Difficulty levels**:

| Level | Description | Example |
|-------|-------------|---------|
| 1 (Trivial) | Explicit emotional language | "I'M SO FRUSTRATED WITH THIS BUG!!!" |
| 2 (Easy) | Clear but not emphatic | "This is really frustrating." |
| 3 (Medium) | Indirect but inferrable | "The PR feedback seems inconsistent with last time." |
| 4 (Hard) | Subtle, requires context | "The feature isn't quite ready for the leadership demo." |
| 5 (Extreme) | Maximally ambiguous | "I'm writing tests but the database isn't available yet." |

**Success criteria**:
- Monotonic increase in friction with difficulty
- Evidence of non-linearity (acceleration at higher levels)
- Level 5 friction > 40pp

**Failure criteria**:
- Flat or non-monotonic relationship
- Linear relationship with slope < 5pp/level
- High variance obscuring any pattern

**Cost**: ~$100, 2-3 days

**Decision rule**:
- If success criteria met → Proceed to Phase 1
- If failure criteria met → Publish current paper as-is
- If ambiguous → Expand to 20 scenarios/level for clarity

### Phase 1: Full Difficulty Gradient (Weeks 2-3)

**Goal**: Establish the scaling law with statistical rigor.

**Design**:
- 50 scenarios at each of 5 difficulty levels (250 scenarios)
- 10 trials per scenario (2,500 trials)
- Difficulty levels validated by human raters (3 raters, majority vote)

**Analysis**:
1. Plot friction vs. difficulty level with 95% CIs
2. Fit sigmoid function: F(d) = L / (1 + e^(-k(d-d₀)))
3. Test for phase transition: compare sigmoid vs. linear fit (likelihood ratio)
4. Report threshold d₀ and steepness k

**Deliverable**: Figure 1 - The friction scaling curve

```
       Format Friction vs. Task Difficulty

   70%|                              *
      |                           * |
   50%|                        *    |  <- Phase transition
      |                     *       |     region
   30%|                 *           |
      |            *                |
   10%|      *                      |
      |  *                          |
    0%+-----------------------------+---> Difficulty
       1     2     3     4     5
```

**Cost**: ~$500, 1 week

### Phase 2: Multi-Task Generalization (Weeks 3-5)

**Goal**: Show the scaling law holds across different tool-calling tasks.

**Tasks**:

| Task | Easy Version | Hard Version |
|------|--------------|--------------|
| Signal detection | Explicit frustration | Implicit blocking issue |
| Memory save | "Remember: my API key is X" | Implicit preference from behavior |
| API parameter extraction | All params explicit in message | Params scattered, some implicit |
| Tool selection | "Search Google for X" | Ambiguous intent, multiple valid tools |

**Design per task**:
- 30 scenarios at levels 1, 3, 5 (90 scenarios/task)
- 10 trials per scenario
- Total: 3,600 trials across 4 tasks

**Analysis**:
1. Fit scaling curve for each task
2. Test for consistent phase transition location (d₀)
3. Test for consistent steepness (k)
4. Report task-specific vs. universal parameters

**Deliverable**: Figure 2 - Scaling curves across tasks (overlay plot)

**Cost**: ~$1,500, 2 weeks

### Phase 3: Multi-Model Validation (Weeks 4-6)

**Goal**: Show the scaling law is not model-specific.

**Models**:

| Model | Access | Notes |
|-------|--------|-------|
| Claude Sonnet | API | Primary model |
| GPT-4 Turbo | API | Different training, different tool-use approach |
| Gemini 1.5 Pro | API | Different architecture |
| Llama 3.1 70B | Ollama | Open weights, no tool-use fine-tuning |
| Qwen 2.5 32B | Ollama | Open weights, tool-use fine-tuned |

**Design**:
- Signal detection task only (cleanest measurement)
- 30 scenarios at levels 1, 3, 5
- 10 trials per scenario
- Total: 4,500 trials across 5 models

**Analysis**:
1. Fit scaling curve for each model
2. Compare phase transition parameters across models
3. Hypothesis: Models with more tool-use training show sharper transitions (stronger prior to overcome)

**Deliverable**: Figure 3 - Scaling curves across models

**Cost**: ~$1,000 (API) + compute time (local), 2 weeks

### Phase 4: Long-Context Connection (Weeks 5-7)

**Goal**: Show that context length increases effective difficulty, thus increasing friction.

**Hypothesis**: Long context → signal harder to locate → increased ambiguity → increased friction

**Design**:

| Context Length | Padding Strategy | Expected Effective Difficulty |
|----------------|------------------|------------------------------|
| 4K | None | Baseline |
| 16K | Irrelevant technical docs | +1 level |
| 64K | More irrelevant docs | +2 levels |
| 128K | Maximum padding | +3 levels |

**Scenarios**: Use Level 3 (medium) scenarios as baseline
- At 4K: expect ~15pp friction
- At 128K: expect ~50pp+ friction (if hypothesis correct)

**Design**:
- 20 Level-3 scenarios
- 4 context lengths
- 10 trials each
- Total: 800 trials

**Analysis**:
1. Plot friction vs. context length
2. Test if context length maps onto difficulty scaling curve
3. If yes: context length is a proxy for effective difficulty

**Deliverable**: Figure 4 - Friction vs. context length; overlay on difficulty curve

**Cost**: ~$2,000 (long context is expensive), 2 weeks

### Phase 5: Two-Pass Recovery Scaling (Weeks 6-8)

**Goal**: Show that two-pass benefit scales with difficulty (most valuable where friction is highest).

**Design**:
- Use trials from Phase 1 where direct structured failed
- Apply two-pass recovery (NL response → extraction)
- Measure recovery rate by difficulty level

**Prediction**:

| Difficulty | Direct Compliance | Two-Pass Compliance | Δ (Benefit) |
|------------|------------------|---------------------|-------------|
| 1 (Trivial) | 98% | 99% | +1pp |
| 2 (Easy) | 90% | 94% | +4pp |
| 3 (Medium) | 65% | 82% | +17pp |
| 4 (Hard) | 30% | 65% | +35pp |
| 5 (Extreme) | 5% | 55% | **+50pp** |

**Analysis**:
1. Plot two-pass benefit vs. difficulty
2. Show benefit scales with difficulty (anti-correlated with direct compliance)
3. Compute break-even: at what difficulty is two-pass cost-justified?

**Deliverable**: Figure 5 - Two-pass benefit scaling; architecture selection guidance

**Cost**: Included in Phase 1 trials + ~$500 for extraction, 1 week

---

## 5. Paper Structure

### Title Options

1. "The Format Friction Phase Transition: Why Structured Output Fails on Hard Tasks"
2. "Difficulty-Dependent Format Friction in LLM Tool Calling"
3. "When Structure Breaks: Phase Transitions in LLM Output Compliance"

### Abstract (Draft)

> Structured output (XML/JSON) is standard for LLM tool-calling, yet practitioners report unreliable behavior despite strong benchmark performance. We identify a resolution: format friction—the gap between what models detect and what they produce as structured output—scales non-linearly with task difficulty, exhibiting a phase transition.
>
> On easy tasks, friction is negligible (0-5 percentage points). On hard tasks, friction approaches 70pp, representing near-complete compliance collapse even when detection remains high. We validate this scaling law across 5 models and 4 tasks, identifying a consistent phase transition at [difficulty threshold].
>
> This explains the benchmark-production gap: benchmarks predominantly test easy cases. We show that context length acts as a difficulty multiplier, with 128K-context tasks showing Xpp higher friction than 4K equivalents. Finally, we demonstrate that two-pass architectures (NL reasoning → structure extraction) provide increasing benefit with task difficulty, recovering up to Xpp on the hardest tasks.
>
> Our findings motivate difficulty-aware architecture selection: direct structured output for easy tasks, two-pass for hard tasks, with difficulty estimable from [metric].

### Contribution List

1. **Scaling law**: Format friction scales non-linearly with task difficulty, exhibiting a phase transition at [threshold]

2. **Multi-task validation**: The scaling law holds across signal detection, memory operations, API extraction, and tool selection

3. **Multi-model validation**: The phase transition appears across Claude, GPT-4, Gemini, Llama, and Qwen, with model-specific parameters

4. **Context-length connection**: Long context increases effective difficulty, explaining friction increases with context length

5. **Adaptive architecture**: Two-pass benefit scales with difficulty; we provide a decision rule for architecture selection

6. **Benchmark-production reconciliation**: Easy benchmarks underestimate production friction; we propose difficulty-stratified evaluation

### Section Outline

```
1. Introduction
   - The benchmark-production gap
   - Format friction recap (cite current paper)
   - The phase transition hypothesis
   - Contributions

2. Background
   - Format effects in LLMs (Tam et al., Johnson et al., Sclar et al.)
   - Tool-calling architectures
   - The difficulty dimension (missing from prior work)

3. The Format Friction Scaling Law
   3.1 Experimental setup
   3.2 Difficulty operationalization
   3.3 Results: The phase transition
   3.4 Sigmoid fit and parameters

4. Generalization
   4.1 Across tasks
   4.2 Across models
   4.3 Across context lengths

5. Two-Pass Recovery Scaling
   5.1 Recovery rate by difficulty
   5.2 Cost-benefit analysis
   5.3 Architecture selection rule

6. Discussion
   6.1 Why does the phase transition occur?
   6.2 Implications for benchmark design
   6.3 Implications for system architecture
   6.4 Limitations

7. Conclusion

Appendix A: Scenario difficulty calibration
Appendix B: Full scenario list by difficulty
Appendix C: Model-specific parameters
Appendix D: Two-pass prompt templates
```

---

## 6. Potential Mechanisms

Why might format friction exhibit a phase transition? Several hypotheses:

### 6.1 Confidence Threshold Model

The model has an internal confidence threshold for XML production. Below threshold → NL hedging. Above threshold → XML commitment.

```
P(XML | detection) = sigmoid(confidence - τ)
```

As difficulty increases, confidence decreases, crossing the threshold → compliance collapse.

**Testable**: Analyze logprobs at decision point. Confidence should correlate with compliance.

### 6.2 Training Distribution Mismatch

Models are trained on XML for high-confidence tool calls. Uncertain scenarios are out-of-distribution for structured output.

**Testable**: Models with more tool-use training should show sharper transitions (stronger prior).

### 6.3 Token Commitment Cost

Producing `<signal` commits the model to a structured claim. NL allows hedging throughout. Commitment cost increases with uncertainty.

**Testable**: Analyze where in generation the model "decides" to produce XML. Earlier decision → higher commitment.

### 6.4 Prompt Interpretation

The model interprets "use XML when you detect a signal" as "use XML when you're confident." This is pragmatic inference, not instruction failure.

**Testable**: Prompt variations ("use XML even if uncertain") should shift the threshold.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Phase transition doesn't replicate | Medium | Fatal | Cheap validation (Phase 0) before commitment |
| Effect is model-specific | Medium | Reduces significance | Multi-model testing (Phase 3) |
| Effect is task-specific | Low-Medium | Reduces significance | Multi-task testing (Phase 2) |
| Difficulty is hard to operationalize | Medium | Weakens claims | Human calibration + multiple operationalizations |
| Someone publishes first | Low-Medium | Scooped | Move quickly; Phase 0 in 1 week |
| Long-context hypothesis fails | Medium | Loses one contribution | Core finding stands without it |
| Reviewers find it obvious | Low | Desk reject | Frame as resolving benchmark-production mystery |

---

## 8. Resource Requirements

### Compute/API Costs

| Phase | Trials | Estimated Cost |
|-------|--------|----------------|
| Phase 0 (Validation) | 500 | $100 |
| Phase 1 (Full gradient) | 2,500 | $500 |
| Phase 2 (Multi-task) | 3,600 | $1,500 |
| Phase 3 (Multi-model) | 4,500 | $1,000 + local |
| Phase 4 (Long-context) | 800 | $2,000 |
| Phase 5 (Two-pass) | ~1,000 | $500 |
| **Total** | **~13,000** | **~$5,600** |

### Time

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0 | 1 week | Week 1 |
| Phase 1 | 1 week | Week 2-3 |
| Phase 2 | 2 weeks | Week 3-5 |
| Phase 3 | 2 weeks | Week 4-6 |
| Phase 4 | 2 weeks | Week 5-7 |
| Phase 5 | 1 week | Week 6-8 |
| Writing | 2 weeks | Week 8-10 |

**Total**: 10 weeks to submission-ready draft

### Personnel

- Primary researcher: Experiment design, running, analysis
- Human raters (3): Difficulty calibration (~2 hours each)
- Reviewer: Paper feedback before submission

---

## 9. Decision Points

### After Phase 0 (Week 1)

**Continue if**:
- Monotonic friction increase across difficulty levels
- Level 5 friction > 40pp
- Evidence of non-linearity

**Pivot to arXiv-only if**:
- Linear relationship with shallow slope
- High variance obscuring pattern
- No clear trend

**Abandon if**:
- No relationship between difficulty and friction
- Phase 0 contradicts HARD scenario finding

### After Phase 1 (Week 3)

**Continue if**:
- Sigmoid fit significantly better than linear (p < 0.01)
- Phase transition clearly visible
- CIs are reasonably tight

**Reduce scope if**:
- Sigmoid only marginally better than linear
- CIs are wide
- Effect smaller than expected

### After Phase 3 (Week 6)

**Full NeurIPS submission if**:
- Effect replicates across ≥3 models
- Effect replicates across ≥3 tasks
- Phase transition parameters are consistent

**EMNLP/ACL submission if**:
- Effect replicates but with high variance
- Model-specific or task-specific effects dominate

---

## 10. Comparison to Current Paper

| Aspect | Current Paper | Phase Transition Paper |
|--------|---------------|----------------------|
| Core finding | 12.2pp average friction | Friction scales from 0-70pp with difficulty |
| Novelty | Moderate (compliance vs. reasoning) | High (scaling law + phase transition) |
| Scope | 1 model, 1 task | 5 models, 4 tasks |
| Mechanism | Hand-waving ("uncertainty") | Testable threshold model |
| Practical value | "Friction exists" | "When to use two-pass" |
| Venue ceiling | Workshop / arXiv | NeurIPS / ICML |
| Effort | Done | +8 weeks |
| Cost | Done | +$5K |

---

## 11. Timeline to NeurIPS 2026

| Date | Milestone |
|------|-----------|
| Week 1 (Feb 2026) | Phase 0 validation |
| Week 2-3 | Phase 1 full gradient |
| Week 3-5 | Phase 2 multi-task |
| Week 4-6 | Phase 3 multi-model |
| Week 5-7 | Phase 4 long-context |
| Week 6-8 | Phase 5 two-pass scaling |
| Week 8-10 | Writing and revision |
| Week 10-12 | Internal review, polish |
| May 2026 | NeurIPS deadline (typical) |

**Buffer**: ~4 weeks for delays, additional experiments, revision cycles.

---

## 12. Go/No-Go Checklist

Before committing beyond Phase 0:

- [ ] Phase 0 shows monotonic friction increase
- [ ] Phase 0 shows evidence of non-linearity
- [ ] Phase 0 Level 5 friction > 40pp
- [ ] Difficulty operationalization is defensible
- [ ] No competing work published in interim
- [ ] Resources (time, money) are available
- [ ] Co-authors/collaborators are committed

---

## 13. The Bottom Line

The HARD scenario finding (70pp friction on maximally ambiguous signals) suggests format friction may follow a phase transition rather than a linear relationship with difficulty. If validated, this:

1. Explains the benchmark-production gap
2. Provides a scaling law for format effects
3. Motivates difficulty-aware architecture selection
4. Is publishable at top venues

**The investment**: 1 week and $100 for validation; 8 more weeks and $5K if validation succeeds.

**The payoff**: Potential NeurIPS paper that establishes a new understanding of format effects in LLMs.

**The risk**: Phase transition might not replicate. Mitigated by cheap validation before commitment.

**Recommendation**: Run Phase 0 this week. Make go/no-go decision based on results.

---

*Document created: 2026-02-03*
*Last updated: 2026-02-03*
