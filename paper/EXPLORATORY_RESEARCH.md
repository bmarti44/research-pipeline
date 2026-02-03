# Exploratory Research Directions

This document captures potential future research directions that extend the core findings from the Format Friction paper. These are **not required** for the current paper but may be valuable for follow-up work.

---

## 1. Familiarity Interaction Effect

### Current State

The paper reports an exploratory finding that the NL-vs-structured gap may increase on unfamiliar file paths:

| Familiarity | NL Recall | Structured Recall | Gap |
|-------------|-----------|-------------------|-----|
| High (common files) | 90% | 80% | +10pp |
| Low (uncommon files) | 88% | 62% | +26pp |

**Limitation**: With 50 observations per subgroup (10 scenarios × 5 trials), confidence intervals are wide (±12-15pp). The difference-in-differences (+16pp) is suggestive but within noise.

### How to Strengthen This Finding

#### Option 1: Increase Sample Size (Most Direct)

```bash
python -m experiments.natural_language_intent_experiment \
    --filepath-only \
    --trials 20
```

20 trials × 10 scenarios = 200 observations per subgroup → CIs shrink to ~±5pp.

#### Option 2: Proper 2×2 Factorial Design

To prove familiarity *causes* the gap increase (not just correlates):

```
Design: 2×2 factorial
- Factor A: Format (NL vs Structured)
- Factor B: Familiarity (High vs Low)

Prediction: Significant interaction effect (Format × Familiarity)
- Main effect of Format: NL > Structured
- Main effect of Familiarity: High > Low (both conditions)
- Interaction: Gap(Low) >> Gap(High)
```

Requires ~100 observations per cell (400 total) for adequate power to detect interaction.

#### Option 3: Difficulty Gradient (5-Level Scale)

Instead of binary high/low, create a 5-level familiarity scale:

| Level | Example | Predicted Gap |
|-------|---------|---------------|
| 1 (Very familiar) | `index.js` | +5pp |
| 2 | `routes.py` | +10pp |
| 3 | `bootstrap.php` | +15pp |
| 4 | `orchestrator.py` | +20pp |
| 5 (Very unfamiliar) | `reconciler.zig` | +25pp |

If gap increases monotonically with unfamiliarity, that's strong evidence for the cognitive load hypothesis.

---

## 2. Other Cognitive Load Manipulations

The core hypothesis is that format friction scales with cognitive load/uncertainty. Other ways to test this:

### 2.1 Context Length

**Hypothesis**: More preceding text increases cognitive load → gap widens.

**Method**: Add 500-1000 words of irrelevant project context before the query.

**Example**:
```
[500 words of project description, architecture notes, etc.]

User: "orchestrator.py is in ./core"
```

Compare gap with and without context padding.

### 2.2 Schema Complexity

**Hypothesis**: More complex XML schemas increase format friction.

| Complexity | Schema | Predicted Gap |
|------------|--------|---------------|
| Simple (current) | `<save-memory category="X">content</save-memory>` | +9pp |
| Medium | `<save><memory category="X" priority="high">content</memory></save>` | +15pp? |
| Complex | Nested structure with multiple required attributes | +20pp? |

### 2.3 Category Proliferation

**Hypothesis**: More categories = more decision friction for structured.

Current: 5 categories (codebase, user_preference, decision, constraint, other)

Test with 10+ categories:
- codebase, user_preference, decision, constraint, architecture, dependency, configuration, workflow, convention, security, performance, other

### 2.4 Multi-Fact Queries

**Hypothesis**: Compound information increases uncertainty → gap widens.

| Type | Example | Predicted Gap |
|------|---------|---------------|
| Single fact | "orchestrator.py is in ./core" | +10pp |
| Two facts | "orchestrator.py is in ./core and uses Redis" | +15pp? |
| Three facts | "orchestrator.py is in ./core, uses Redis, and handles async jobs" | +20pp? |

### 2.5 Ambiguous Categorization

**Hypothesis**: Scenarios that could fit multiple categories increase friction.

Examples:
- "We decided to use tabs" — preference? decision? codebase convention?
- "API rate limit is 100/min for security" — constraint? security? codebase?
- "I prefer PostgreSQL for new projects" — preference? decision?

---

## 3. Expanding Unfamiliarity Dimension

Current low-familiarity scenarios use uncommon file names. Could expand to:

### 3.1 Made-Up Extensions
- `coordinator.zyx`
- `handler.qwp`
- `processor.abc`

### 3.2 Domain-Specific Jargon
- `saga_orchestrator.ex` (Elixir + DDD)
- `cqrs_projector.fs` (F# + CQRS)
- `hexagonal_adapter.kt` (Kotlin + Hexagonal Architecture)

### 3.3 Non-ASCII Paths
- `配置.toml is in ./config`
- `конфиг.yaml is in ./settings`

### 3.4 Very Long Paths
- `./src/modules/auth/v2/internal/handlers/oauth/providers/google.ts`

---

## 4. Priority Assessment

| Research Direction | Effort | Expected Impact | Recommendation |
|--------------------|--------|-----------------|----------------|
| Larger sample size for familiarity | Low | Medium | Do first if pursuing |
| Context length manipulation | Medium | High (novel) | Strong candidate |
| Schema complexity | Medium | Medium | Interesting but niche |
| Category proliferation | Low | Low | Minor variation |
| Multi-fact queries | Medium | Medium | Worth exploring |
| Ambiguous categorization | Medium | Medium | Mechanistically interesting |

**Recommended path if pursuing**:
1. Replicate familiarity with 20 trials (tighten CIs)
2. Add context length experiment (novel cognitive load factor)
3. Write up as follow-up paper focused on "when does format friction matter most?"

---

## 5. Decision: Why We Deferred This

The core paper contribution is already strong:
- 9.4pp gap with p=0.001 (survives Bonferroni correction)
- Zero-cell phenomenon (NL is strict superset of structured)
- Verification detour mechanism (qualitatively observed, now quantified)

The familiarity interaction is decorative, not load-bearing. The paper doesn't need it to make its case. Pursuing it now would:
- Delay publication of the core finding
- Risk not replicating (it could be noise)
- Not change the main conclusion even if it does replicate

**Current approach**: Frame familiarity as exploratory in the paper, note it in Future Work, save deeper investigation for follow-up.

---

## 6. Two-Stage Architecture: Large NL Model → Small Schema Model

This is potentially the most impactful follow-up research. The paper proposes but doesn't validate a two-stage architecture.

### The Hypothesis

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Large Model (NL Intent)                           │
│  - Claude Opus / GPT-4 / Large open-source                  │
│  - No format constraints                                    │
│  - High recall on when to act                               │
│  - Output: Natural language with embedded intent            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Small Model (Schema Extraction)                   │
│  - Llama-3.2-1B / Mistral-7B / Fine-tuned small model       │
│  - Constrained decoding                                     │
│  - High precision on structure                              │
│  - Output: Valid JSON/XML schema                            │
└─────────────────────────────────────────────────────────────┘
```

### Why This Matters

1. **Cost efficiency**: Large model reasons, small model formats
2. **Latency**: Schema extraction can be parallelized or cached
3. **Separation of concerns**: Reasoning vs formatting are different skills
4. **Validated externally**: SLOT paper achieved 99.5% schema accuracy with this approach

### Experimental Design

| Condition | Stage 1 | Stage 2 | Metrics |
|-----------|---------|---------|---------|
| Baseline (current) | Claude Sonnet + XML | N/A | 82% recall |
| NL-only | Claude Sonnet + NL | N/A | 92% recall |
| Two-stage | Claude Sonnet + NL | Mistral-7B | Recall × Schema accuracy |
| Two-stage (small) | Claude Sonnet + NL | Llama-1B | Recall × Schema accuracy |

### Key Questions

1. **Does extraction accuracy hold?** Can a small model reliably extract structured data from NL intent?
2. **What's the end-to-end accuracy?** Recall × Extraction accuracy = final performance
3. **Cost/latency tradeoff?** Is the two-call overhead worth the accuracy gain?
4. **Fine-tuning requirements?** Does the small model need task-specific fine-tuning?

### Implementation Sketch

```python
# Stage 1: Large model generates NL intent
nl_response = large_model.generate(
    prompt=NL_PROMPT,
    user_message=user_query
)

# Stage 2: Small model extracts structure
extraction_prompt = f"""
Extract any save-memory intent from this response.
If the response indicates saving information, output JSON:
{{"action": "save", "category": "<category>", "content": "<content>"}}
If no save intent, output:
{{"action": "none"}}

Response: {nl_response}
"""

structured_output = small_model.generate(
    prompt=extraction_prompt,
    constrained_decoding=True  # Force valid JSON
)
```

---

## 7. Content-Dependent Format Recommendations

The paper found that **different content types favor different formats**:

| Content Type | Winner | Finding |
|--------------|--------|---------|
| General implicit | NL (+9.4pp recall) | Format friction suppresses action |
| Precision (exact values) | NL (100% fidelity) | Structured adds unwanted conversions |
| Multi-field technical | Tie (100% both) | No friction when confident |
| Negation/state changes | Structured (73% fidelity) | Better captures "what was" → "what is" |

### Research Question

Can we build a **format selector** that chooses NL vs Structured based on content type?

### Experimental Design

1. **Classify content type** from user message (precision, state-change, general, etc.)
2. **Route to appropriate format** (NL for general/precision, Structured for state-changes)
3. **Measure end-to-end accuracy** vs single-format baselines

---

## 8. Cross-Model Validation

The paper only tests Claude Sonnet. Critical question: **Is format friction model-specific?**

### Models to Test

| Model | Tool-calling Training | Hypothesis |
|-------|----------------------|------------|
| GPT-4 | Heavy (function calling) | Similar friction expected |
| GPT-3.5 | Moderate | May show larger gap |
| Gemini Pro | Heavy | Similar friction expected |
| Llama-3-70B | Light | May show smaller gap (less tool-use prior) |
| Mistral-7B | Minimal | May show minimal friction |

### Prediction

Models with more tool-calling fine-tuning may show:
- Higher structured baseline (better at format)
- But also higher friction (stronger format prior to overcome)

Models with less tool-calling training may show:
- Lower structured baseline
- But also lower friction (no prior to compete with)

---

## 9. TOOL_ATTEMPT Phenomenon Deep Dive

The review identified that 15-20% of NL false negatives show tool-calling behavior despite suppression instruction.

### Research Questions

1. **Which scenarios trigger tool attempts?** File paths? Implementation requests?
2. **Is this model-specific?** Do other models show same behavior?
3. **Can we suppress it better?** Different prompt engineering approaches
4. **Is it actually harmful?** Maybe TOOL_ATTEMPT responses are reasonable

### Data Already Available

The validation samples (`validation_samples_*.json`) contain examples. Could analyze:
- Which tools attempted (Read, Bash, Task?)
- Which scenarios triggered attempts
- Correlation with scenario type

---

## 10. Control Scenario Design Insights

From the validation run (2026-02-02):

| Control | Query | NL FP | Structured FP | Issue |
|---------|-------|-------|---------------|-------|
| ctrl_opinion_001 | "React has more GitHub stars than Vue" | 40% | 0% | Working better |
| ctrl_known_001 | "Did I already tell you we use React?" | 100% | 80% | Model extracts "we use React" from question |

### Insight

The model extracts factual content even from questions. True negatives need to be:
- Actually non-informative ("Did I tell you something already?")
- Or explicitly transient ("Just testing something")

### Better Control Candidates

```python
# Truly non-informative
"Did I mention something about our frontend?"
"Have we discussed the tech stack?"

# Explicitly transient
"Ignore this, just testing"
"This is hypothetical: what if we used React?"

# Meta-questions
"What have I told you so far?"
"Do you remember anything from earlier?"
```

---

## 11. Paradigm-Shift Research: Proving Format Friction is Fundamental

The current paper establishes an empirical observation: format friction exists and costs ~9pp on Claude Sonnet. But this is **confirmatory**, not **paradigm-shifting**. Prior work (Tam et al., Johnson et al.) already showed format degrades performance.

To reach the level of "the entire paradigm of structured LLM output is flawed," we would need to prove that format friction is **fundamental** — not a training artifact that better fine-tuning could eliminate.

### The Core Question

> Is format friction a FUNDAMENTAL property of how LLMs work, or is it a TRAINING ARTIFACT that could be eliminated with better fine-tuning?

If training artifact → the paradigm isn't flawed, we just need better training.
If fundamental → the paradigm is flawed, and two-stage architectures are necessary.

### Path 1: Prove Training Can't Fix It

**Approach**: Show that format friction persists or increases despite training interventions designed to eliminate it.

**Method**:
1. Get access to models at different training stages (base → instruct → RLHF → tool-tuned)
2. Show format friction *increases* with more tool-use training, or stays constant despite it
3. If a lab has optimized specifically for XML tool-calling and friction still exists, that's evidence training can't eliminate it

**Strongest version**: Partner with a lab. Train a model *specifically* to minimize format friction — optimize directly for "same decision regardless of output format." Show the model either fails to learn this, or trades off other capabilities to achieve it.

**Difficulty**: Medium (requires lab access or open-weight models at multiple training stages)
**Payoff**: High — directly addresses "is this fixable?"

### Path 2: Information-Theoretic Argument

**Approach**: Prove mathematically that structured output requires compute that necessarily competes with reasoning.

**Hypothesis**: Structured output requires the model to simultaneously:
- Decide WHAT to do (reasoning)
- Decide HOW to format it (syntax)

These compete for the same finite compute (attention, parameters, context).

**Method**:
1. Measure token-level entropy/perplexity during structured vs. NL output
2. Show the model allocates attention differently — more to format tokens, less to reasoning
3. Formalize: If a model has capacity C, and format requires F, reasoning gets C-F. When F > 0, reasoning is necessarily degraded.

**Strongest version**: Derive a theoretical bound:
> "For any transformer with attention capacity C, requiring structure S reduces effective reasoning capacity by f(S), where f is monotonically increasing with schema complexity."

Then validate empirically across models and schemas.

**Difficulty**: Hard (requires theory expertise)
**Payoff**: Very high — a formal bound would be citable for decades

### Path 3: Universality Across Architectures

**Approach**: If friction appears in every architecture, it's not implementation-specific — it's fundamental to the task.

**Method**:
1. Test transformers (GPT-4, Claude, Gemini, Llama)
2. Test state-space models (Mamba, RWKV)
3. Test hybrid architectures (Jamba)
4. Test models with explicit structure modules (Toolformer-style)
5. Test models with NO tool-use training (pure base models)

**Prediction**: If friction appears in ALL of them — including architectures designed to avoid it — that's strong evidence for fundamentality.

**Key insight**: If a model with zero tool-use training still shows friction, it's not about learned priors. If a model architecturally designed for structured output still shows friction, it's not about architecture.

**Difficulty**: Medium-hard (requires significant compute for multi-model evaluation)
**Payoff**: Medium-high — strong empirical case, but not a proof

### Path 4: Mechanistic Interpretability

**Approach**: Open the black box. Show exactly which circuits/attention heads cause friction and that the tradeoff is architecturally determined.

**Method**:
1. Use techniques from Anthropic's interpretability work (activation patching, circuit analysis)
2. Identify the "format compliance" circuits vs. "reasoning" circuits
3. Show they share resources (same heads, same layers, same parameters)
4. Show that activating format circuits *necessarily* suppresses reasoning circuits

**Strongest version**: Find a mathematical relationship:
> "Format circuit activation F correlates with reasoning circuit suppression R at r = -0.X, and this tradeoff is architecturally determined by weight sharing at layer N."

**Difficulty**: Very hard (frontier interpretability research)
**Payoff**: Very high — mechanistic explanation would be landmark paper

### Path 5: The Impossibility Result

**Approach**: A formal proof that no bounded-compute system can optimize both reasoning and format compliance simultaneously.

**Sketch of argument**:
1. Define "reasoning quality" R(x) for input x
2. Define "format compliance" F(x, s) for input x and schema s
3. Prove: For any bounded-compute system, max(R) and max(F) cannot be achieved simultaneously
4. The proof would likely involve showing format verification requires compute that scales with schema complexity, creating an unavoidable tradeoff

**Analogy**: Like the bias-variance tradeoff or the no-free-lunch theorem — a fundamental limit, not an engineering problem.

**Difficulty**: Extremely hard (may not even be true)
**Payoff**: Field-defining — this would be the "Attention Is All You Need" of structured output

### What We Can Do Now

| Path | Feasible with Current Resources? | Recommendation |
|------|----------------------------------|----------------|
| Path 1 (Training) | Partial — can test Llama base/instruct/tool-tuned | Do this |
| Path 2 (Information theory) | No — needs theory expertise | Defer or collaborate |
| Path 3 (Universality) | Yes — can test 4-5 open models | Do this |
| Path 4 (Mechanistic) | No — needs interpretability infrastructure | Defer or collaborate |
| Path 5 (Impossibility) | No — needs formal methods expertise | Long-term aspiration |

### Recommended Research Program

**Phase 1 (3-6 months)**: Universality
- Test format friction on GPT-4, Gemini, Llama-3-70B, Mistral, Mamba
- If friction appears across all architectures, publish as "Format Friction is Architecture-Independent"

**Phase 2 (6-12 months)**: Training Dynamics
- Partner with lab or use open-weight models
- Track friction across training stages
- Test if friction-minimization training succeeds or fails
- If friction persists despite targeted training, publish as "Format Friction Cannot Be Trained Away"

**Phase 3 (12-24 months)**: Mechanistic or Theoretical
- Either pursue interpretability (if Anthropic/collaborator access)
- Or pursue information-theoretic bound (if theory collaborator)
- Goal: Move from "friction exists" to "friction must exist"

### The Honest Assessment

The current paper contributes one data point to a growing pile of evidence. To prove format friction is fundamental would require a multi-year research program spanning empirical validation, training experiments, and either mechanistic or theoretical work.

The paradigm shift, if achievable, would be: **"Requiring structured output from LLMs is like requiring humans to do mental math while reciting the alphabet — the tasks compete for the same cognitive resources, and no amount of practice fully eliminates the interference."**

That's publishable in Nature. The current paper is not.

---

*Last updated: 2026-02-02*
