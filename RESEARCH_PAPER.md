# FORMAT FRICTION: HOW OUTPUT STRUCTURE SUPPRESSES MODEL ACTION

**Brian Martin**¹, **Stephen Lipmann**¹

¹ Oracle

---

## Abstract

We present empirical evidence that output format significantly affects Claude Sonnet's decision-making, independent of task understanding. In controlled experiments comparing natural language intent expression versus structured XML output, we find that with identical guidance on when to act, natural language achieves 92% recall while structured output achieves only 82%—a 9.4 percentage point gap (p = 0.001, survives Bonferroni correction). Critically, there were zero cases where structured output succeeded but natural language failed, suggesting natural language captures a superset of structured output's capabilities.

We further demonstrate a familiarity interaction effect: on common file paths (e.g., `index.js`, `main.py`), the gap is +10pp; on uncommon paths (e.g., `orchestrator.py`, `reconciler.zig`), the gap increases to +26pp. Structured output triggers "verification detours" on unfamiliar inputs while natural language proceeds without hesitation.

These findings challenge the assumption that structured tool interfaces are neutral output formats. Format itself creates cognitive friction that suppresses action, particularly under uncertainty. We propose a two-stage architecture: let models express intent naturally, then extract structure with a specialized system.

---

## 1 Introduction

Large language models increasingly serve as autonomous agents, calling tools, writing code, and generating structured outputs. The standard approach gives models structured interfaces: XML schemas, JSON formats, or function-calling syntax. This seems sensible—structured output is machine-parseable and unambiguous.

But what if the structure itself degrades performance?

We present evidence that output format affects model judgment, not just output parsing. In our experiments, we give Claude identical guidance on *when* to save information to memory and equally explicit instructions on *how* to save it. The only difference is the output format:

- **Natural Language**: "I'll save that your configuration file is in src/config.ts"
- **Structured XML**: `<save-memory category="codebase">config file in src/config.ts</save-memory>`

With identical task understanding and equally explicit instructions, natural language achieves 92% recall while structured output achieves 82%—a 9.4 percentage point gap. More striking: in 255 paired trials, there were zero cases where structured output succeeded but natural language failed.

This suggests structured output doesn't just change *how* models respond—it changes *whether* they respond.

### 1.1 Key Findings

1. **Format affects judgment**: With identical guidance, NL outperforms structured by +9.4pp (p = 0.001)
2. **NL is a superset**: Structured never succeeds where NL fails (0/255 cases)
3. **Familiarity interaction**: Gap increases on unfamiliar patterns (+10pp → +26pp)
4. **Verification detours**: Structured triggers "let me verify..." behavior; NL does not
5. **Fidelity comparable**: With fair bidirectional comparison, NL wins 13% vs structured's 5% (83% ties)
6. **Format friction scales with uncertainty**: Gap disappears on clear technical specifications (0pp)

### 1.2 Contributions

1. **Empirical evidence** that output format affects model decision-making independent of task understanding
2. **Familiarity analysis** showing format friction increases under uncertainty
3. **Failure mode characterization**: verification detours as a structured-output-specific failure
4. **Design implications** for tool interfaces and agent architectures

---

## 2 Background

### 2.1 Format Constraints and Model Performance

Recent work demonstrates that output format requirements degrade model performance:

Tam et al. (2024) show JSON output requirements reduce reasoning accuracy by up to 27.3 percentage points. The stricter the schema, the greater the degradation.

Johnson et al. (2025) demonstrate that replacing structured tool outputs with natural language improves accuracy by 18.4pp across 10 models.

Sclar et al. (2024) find superficial prompt formatting causes up to 76pp accuracy swings—far exceeding semantic content changes.

### 2.2 The Format Friction Hypothesis

We hypothesize that structured output creates **format friction**—cognitive overhead that competes with semantic reasoning:

$$P(\text{action} | \text{NL format}) > P(\text{action} | \text{structured format})$$

This friction should:
- Exist even with identical task guidance
- Increase when the model is uncertain (unfamiliar inputs)
- Manifest as avoidance behaviors (verification, hedging)

### 2.3 Prior Work Limitations

Previous studies compare different *prompts* or different *tasks*. Our contribution is isolating format as the *only* variable:
- Same guidance on WHEN to act
- Equally explicit instructions on HOW to act
- Only difference: output structure

---

## 3 Experimental Design

### 3.1 Task: Proactive Memory Persistence

We test memory persistence—deciding whether to save user-provided information for future conversations. This task requires genuine judgment (not all information should be saved) with minimal format complexity.

### 3.2 Conditions

Both conditions receive **identical guidance** on when to save:

```
You have a persistent memory system for future conversations.

Save information when the user shares something worth remembering:
- Codebase structure, conventions, or technical details
- User preferences (coding style, tools, workflows)
- Important decisions or constraints
- Anything that would be useful context in future conversations

Use your judgment about what's important enough to save.
```

**Natural Language Condition**: Express intent naturally

```
## How to Save Information
When you want to save something, simply say so naturally in your response.

Examples:
- "I'll save that your configuration file is src/configuration.ts."
- "Let me remember your preference for using tabs."
- "I'll note that the content service can have max 10 threads."
```

**Structured Condition**: Use XML format

```
## How to Save Information
To save information, use this XML:

    <save-memory category="<category>">content</save-memory>

Categories: codebase, user_preference, decision, constraint, other

Examples:
- <save-memory category="codebase">Configuration file is src/configuration.ts</save-memory>
- <save-memory category="user_preference">Use tabs</save-memory>
```

### 3.2.1 Prompt Engineering Constraints

A practical constraint shaped our prompt design: Claude's default behavior strongly favors tool use when tools are available. In pilot testing, the NL condition frequently attempted to call tools despite instructions to express intent naturally.

To isolate the format variable, we:
- **NL condition**: Explicitly instructed "Do not use tools as you normally would - use natural language" to suppress default tool-calling behavior
- **Structured condition**: Used standard tool-calling framing consistent with current production patterns

This reflects the real architectural choice: deploying NL intent expression requires suppressing default tool behavior. We acknowledge this as a limitation; the suppression instruction itself may affect model behavior beyond format alone.

### 3.3 Scenarios

We constructed 71 scenarios across two dimensions:

**Explicitness** (vague → explicit):
- **Implicit** (~40): Terse statements like "app.py is in ./src", "pg is db"
- **Control** (20): Should NOT trigger save (questions, chitchat, transient statements)
- **Explicit** (1): Sanity check with direct command

**Familiarity** (for filepath scenarios):
- **High familiarity** (10): Common files (`index.js`, `main.py`, `package.json`)
- **Low familiarity** (10): Uncommon files (`orchestrator.py`, `mediator.kt`, `reconciler.zig`)

All test scenarios were constructed by the authors based on plausible developer interactions. We did not conduct formal validation of scenario realism; future work should validate with user studies or real conversation logs.

### 3.4 Evaluation

- **Model**: Claude Sonnet (claude-sonnet-4-5-20250929)
- **Trials**: 5 per scenario per condition (randomized order)
- **Detection**: Regex for NL intent phrases; XML tag parsing for structured
- **Fidelity**: Head-to-head LLM judge comparison when both conditions save

### 3.5 Metrics

- **Recall**: Fraction of positive scenarios where model saved (true positive rate)
- **False Positive Rate**: Fraction of control scenarios incorrectly triggering save
- **McNemar's Test**: Paired comparison of condition performance
- **Fidelity**: Winner determined by blind A/B comparison (randomized order to prevent position bias)

### 3.6 Methodology Evolution

This experiment evolved through three versions to address confounds:

**v1 ("Reasoning-Action Gap")**: Initial design compared "reasoning" prompts with explicit guidance against "action" prompts with vague guidance. This confounded format with explicitness—NL appeared better because it received clearer instructions.

**v2 ("Matched Guidance")**: We equalized the WHEN-to-save guidance but kept different HOW-to-save examples. The structured examples were more formal ("Use the save-memory tool...") while NL examples were conversational. This still confounded format with tone.

**v3 ("Matched Explicitness")**: Current design. Both conditions receive:
- Identical WHEN-to-save guidance (same semantic content)
- Equally explicit HOW-to-save examples (matched verbosity and formality)
- Only difference: output structure (NL phrases vs XML tags)

This evolution is important context: early results showing 25-30pp gaps were inflated by confounds. The current 9.4pp gap represents the isolated effect of output format.

---

## 4 Results

### 4.1 Primary Finding: Natural Language Outperforms Structured

**Table 1: Overall Results (n=255 positive observations per condition)**

| Condition | Recall | 95% CI |
|-----------|--------|--------|
| Natural Language | **91.8%** | [87.7%, 94.6%] |
| Structured | **82.4%** | [77.2%, 86.5%] |
| **Difference** | **+9.4pp** | — |

McNemar's χ² = 22.04, p < 0.0001 (trial-level; see Section 4.11 for scenario-level analysis)

### 4.2 Natural Language is a Superset

**Table 2: McNemar's Contingency Table**

|  | Structured ✓ | Structured ✗ |
|--|--------------|--------------|
| **NL ✓** | 210 | 24 |
| **NL ✗** | **0** | 21 |

The zero in the bottom-left cell is remarkable: there were no cases where structured succeeded but NL failed. Every structured success was also an NL success. NL captures everything structured does, plus 24 additional cases.

**Interpreting the zero cell.** This result warrants scrutiny. Three interpretations:

1. **NL is genuinely a superset**: The model's decision to save under structured constraints implies it would also save under NL constraints. The structured format adds friction without enabling new capabilities.

2. **Detection asymmetry**: NL patterns are more generous than XML patterns, so borderline cases are caught by NL detection but not structured detection. Some "NL successes" might be false positives.

3. **Statistical artifact**: With 255 observations, a true 1-2% structured-only rate could yield zero by chance (p ≈ 0.08 for true rate of 1%).

We cannot definitively distinguish these interpretations. Raw response data is available in `experiments/results/` for independent verification.

### 4.3 Familiarity Interaction Effect

**Table 3: Performance by Familiarity Level**

| Familiarity | NL | Structured | Gap |
|-------------|-----|------------|-----|
| High (common files) | 90.0% | 80.0% | **+10.0pp** |
| Low (uncommon files) | 88.0% | 62.0% | **+26.0pp** |

The gap appears to increase on low-familiarity scenarios (+10pp → +26pp). However, with only 50 observations per subgroup (10 scenarios × 5 trials), confidence intervals are wide (approximately ±12-15pp). The difference-in-differences (+16pp) is suggestive but requires larger samples for confirmation. We report this as preliminary evidence.

Breaking down the point estimates:

- NL drops 2pp (90% → 88%) on unfamiliar inputs
- Structured drops 18pp (80% → 62%) on unfamiliar inputs

If the pattern holds, structured output amplifies the uncertainty effect ~9× compared to NL. This is consistent with the format friction hypothesis but should be replicated.

### 4.4 Failure Mode: Verification Detours

Structured output failures follow a consistent pattern—the model attempts to *verify* before *acting*:

**Example: `coordinator.go is in ./internal` (low familiarity)**

| Trial | NL Response | Structured Response |
|-------|-------------|---------------------|
| 1 | "I'll save that the coordinator.go file is located in ./internal" ✓ | "I'll read the coordinator.go file from ./internal..." ✗ |
| 3 | "I'll save that..." ✓ | "Let me verify this file exists..." ✗ |
| 5 | "I'll save that..." ✓ | "I understand that... Let me verify..." ✗ |

NL: 5/5 (100%). Structured: 2/5 (40%).

The structured format creates a higher commitment threshold. Uncertain about `coordinator.go`? NL says "I'll remember that" (low commitment). Structured must produce an XML tag (feels like executing a command), triggering verification behavior.

### 4.5 Semantic Confounds

Some scenarios revealed semantic ambiguity rather than format effects:

**Example: `README.md is in root`**

Both conditions failed 100%—interpreting "README.md" as a request to *read* the file rather than information about its location. The filename's semantic association ("read me") overrides the save intent.

This failure is not a format effect (both conditions fail equally) but rather a scenario design issue. Such cases were excluded from familiarity analysis.

### 4.6 Fidelity Analysis

When both conditions successfully save, which captures better information?

**Methodology.** To ensure fair comparison, we: (1) extract semantic content from NL responses before comparison (removing conversational wrappers like "I'll save that..."), and (2) run bidirectional comparison (A vs B, then B vs A) to control for LLM judge position bias. A winner is only counted if consistent across both orderings.

**Table 4: Head-to-Head Fidelity Comparison (n=219 paired successes)**

| Winner | Count | Percentage |
|--------|-------|------------|
| NL | 28 | 13% |
| Structured | 10 | 5% |
| Tie | 181 | 83% |

Bidirectional consistency: 87% (29 comparisons showed position bias and were marked as ties).

**With fair comparison, fidelity is roughly equivalent.** The majority of comparisons (83%) show no meaningful difference in information capture quality. NL wins slightly more often (13% vs 5%), but the dominant pattern is equivalence.

**LLM-as-judge limitations.** Research shows LLM judges have significant biases (position bias, verbosity bias, self-enhancement). Our bidirectional approach mitigates position bias, but results should be considered suggestive rather than definitive. We recommend treating fidelity findings as exploratory.

### 4.7 Precision Scenarios: A Surprising Result

We hypothesized that scenarios requiring exact numeric precision would favor structured output. We tested this with scenarios like:
- "max upload is exactly 10485760 bytes"
- "timeout is 30000ms, not 30s"

**The results contradicted our hypothesis.** On the bytes scenario, both conditions achieved 100% recall, but NL won on fidelity. Why? Structured output added an unwanted conversion:

| Condition | Output | Fidelity |
|-----------|--------|----------|
| NL | "max upload size is exactly 10485760 bytes" | ✓ Preserved exact value |
| Structured | "Max upload is exactly 10485760 bytes (10 MB)" | ✗ Added conversion |

The user specified "exactly 10485760 bytes"—adding "(10 MB)" is technically incorrect (10 MB = 10,000,000 bytes, not 10,485,760). Structured output's attempt to be "helpful" by adding context actually reduced fidelity.

On the timeout scenario (30000ms not 30s), NL achieved 5/5 recall while structured achieved only 2/5—the same verification detour pattern observed elsewhere.

**Implication**: Structured output's fidelity advantage (Section 4.6) may be context-dependent. When users specify exact values, structured output's tendency to add interpretive context can harm rather than help precision. NL's more literal preservation of user statements may be advantageous for precision-critical applications.

### 4.8 Multi-Field Scenarios: Format Friction Disappears

We tested scenarios with multiple pieces of technical information:
- "auth uses OAuth2 with PKCE, tokens expire in 3600s, refresh enabled"
- "db is postgres 15.2 on port 5433 with ssl required"
- "redis cache at localhost:6379, db 2, password is in REDIS_PASS env var"

**Result: Both conditions achieved 100% recall with no significant difference.** Fidelity was 93% ties.

This finding reinforces the uncertainty hypothesis from Section 4.3. These scenarios are unambiguous technical specifications—the model has high confidence about what to save and how to categorize it. With no uncertainty, there are no verification detours, and format friction disappears.

**Table 6: Format Friction by Scenario Uncertainty**

| Scenario Type | Uncertainty | NL Recall | Structured Recall | Gap |
|---------------|-------------|-----------|-------------------|-----|
| Ambiguous (implicit) | High | 92% | 82% | +9.4pp |
| Unfamiliar paths | High | 88% | 62% | +26pp |
| Familiar paths | Medium | 90% | 80% | +10pp |
| Multi-field technical | Low | 100% | 100% | 0pp |

Format friction is not a constant tax—it scales with uncertainty. When the model is confident, structured output performs as well as natural language. The practical implication: format friction is most problematic for edge cases and ambiguous inputs, precisely where robust behavior matters most.

### 4.9 Negation Scenarios: Where Structured Output Excels

We tested scenarios involving state changes and migrations:
- "we migrated off Redis last month"
- "removed the webpack config, using vite now"
- "deprecated the v1 api, only v2 is supported"

**Result: Both conditions achieved 100% recall, but structured output won 73% of fidelity comparisons** (11/15 structured wins, 4 ties, 0 NL wins).

Examining the outputs reveals why:

| Scenario | NL Output | Structured Output |
|----------|-----------|-------------------|
| Redis migration | "migrated off Redis" | "migrated off Redis (as of ~January 2026)" |
| Webpack → Vite | "switched from Webpack to Vite" | "uses Vite instead of Webpack for bundling" |

Structured output better captures **state transitions**—the relationship between "what was" and "what is now." The XML format appears to encourage more precise articulation of changes, preserving both the old state and the new state explicitly.

**Table 7: Fidelity Results by Scenario Type**

| Scenario Type | Structured Wins | NL Wins | Ties |
|---------------|-----------------|---------|------|
| General implicit | 11% | 0% | 89% |
| Precision (exact values) | 0% | 100% | 0% |
| Multi-field technical | 7% | 0% | 93% |
| Negation (state changes) | **73%** | 0% | 27% |

**Implication**: Structured output's fidelity advantage is content-dependent. For state transitions and temporal changes, structured output excels. For exact value preservation, NL is superior. For general information, they are roughly equivalent. Practitioners should consider the nature of their data when choosing output formats.

### 4.10 Statistical Summary

**Table 5: Summary Statistics**

| Metric | Value |
|--------|-------|
| NL Recall | 91.8% |
| Structured Recall | 82.4% |
| Difference | +9.4pp |
| Scenario-level sign test p | 0.001 |
| Scenario-level Wilcoxon p | 0.001 |
| McNemar's χ² (trial-level) | 22.04 |
| McNemar p-value (trial-level) | <0.0001 |
| NL-only successes | 24 |
| Structured-only successes | 0 |
| High-familiarity gap | +10.0pp |
| Low-familiarity gap | +26.0pp |
| False positive rate (NL) | 9.0% |
| False positive rate (Structured) | 10.0% |

### 4.11 Statistical Corrections

We conducted 8 hypothesis tests in this study. To control family-wise error rate, we apply Bonferroni correction (α = 0.05/8 = 0.00625). The primary finding (NL vs. structured recall difference) survives correction at the scenario level (sign test p = 0.001, Wilcoxon p = 0.001). Subgroup analyses (familiarity, precision, negation) should be considered exploratory pending replication with larger samples.

**Scenario-level analysis (primary).** Multiple trials per scenario are correlated (same prompt, same input). We aggregate to scenario level: of 51 positive scenarios, NL had higher recall in 11, structured had higher recall in 0, and 40 were tied. The sign test and Wilcoxon signed-rank test both yield p = 0.001, confirming the NL advantage with independent observations.

**Trial-level McNemar (secondary).** The trial-level analysis (χ² = 22.04, p < 0.0001) treats all 255 observations as independent. This assumption is violated, so p-values may be inflated. We report it for comparability with prior work but emphasize the scenario-level analysis.

---

## 5 Theoretical Interpretation

### 5.1 Why Does Format Create Friction?

We propose three mechanisms:

**1. Commitment asymmetry**: Natural language intent ("I'll remember that") feels provisional—a statement of intent that could be revised. Structured output (`<save-memory>`) feels like executing a command—an irreversible action requiring higher confidence.

**2. Format overhead**: Producing structured output requires simultaneously reasoning about *what* to do and *how* to format it. These dual demands compete for cognitive resources. NL lets the model focus purely on the semantic decision.

**3. Uncertainty amplification**: When uncertain, the model's default behavior differs by format:
- NL: Express tentative intent anyway ("I'll note that...")
- Structured: Seek more information first ("Let me verify...")

### 5.2 The Verification Detour Pattern

Structured output failures aren't random—they follow a specific pattern:

1. Model receives unfamiliar input (`coordinator.go`)
2. Uncertainty triggers caution
3. Instead of producing XML tag, model tries to verify ("Let me check...")
4. Verification attempt = no save action recorded

NL doesn't trigger this pattern because "I'll remember that" requires no verification—it's just a conversational statement.

### 5.3 Familiarity as Uncertainty Proxy

The familiarity effect suggests format friction scales with uncertainty:

- **Familiar input** (e.g., `index.js`): Model is confident → both formats work well
- **Unfamiliar input** (e.g., `reconciler.zig`): Model is uncertain → structured triggers verification, NL proceeds normally

This explains the gap nearly tripling: unfamiliar inputs increase uncertainty, which amplifies format friction effects.

---

## 6 Implications

### 6.1 For Tool Interface Design

Structured tool interfaces are not neutral. They create cognitive friction that can suppress action, particularly when:
- The model is uncertain about the input
- The action feels irreversible
- Verification seems possible

Consider:
- **Lower friction formats**: Natural language intent → structured extraction
- **Explicit confidence signals**: Let models express uncertainty without blocking action
- **Reversible framing**: "Draft save" vs "Final save"

### 6.2 For Agent Architectures (Proposed)

Our findings motivate a **two-stage architecture** that we propose for future investigation:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Primary LLM (Natural Language)                    │
│                                                             │
│  Input: User message                                        │
│  Output: Natural response with embedded intent              │
│                                                             │
│  "Got it! I'll remember that coordinator.go is in          │
│   ./internal for your Go project."                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Intent Extractor (Lightweight)                    │
│                                                             │
│  Input: Natural language with intent signals                │
│  Output: Structured tool calls                              │
│                                                             │
│  Detected: "I'll remember that coordinator.go..."           │
│  → save_memory("coordinator.go in ./internal", "codebase")  │
└─────────────────────────────────────────────────────────────┘
```

**Note:** This architecture has not been implemented or validated in this study. We propose it based on our findings, but the claim that it "recovers the 10pp" is theoretical. Future work should:
- Implement and test the two-stage pipeline
- Measure actual extraction accuracy from NL intent phrases
- Quantify end-to-end performance vs single-stage structured output

The theoretical benefits:
- Stage 1 operates without format constraints (potentially higher recall)
- Stage 2 handles format without semantic ambiguity (potentially high precision)
- Separates "what to do" decisions from "how to format" execution

### 6.3 For Prompt Engineering

If using structured formats directly:
- Use familiar, conventional patterns where possible
- Provide explicit examples matching expected inputs
- Consider "verification-blocking" instructions ("Save first, verify later if needed")
- Monitor for verification detour patterns in logs

### 6.4 For Evaluation

Tool-calling benchmarks should separately measure:
- **Intent recognition**: Does the model know it should act?
- **Format execution**: Can it produce the required structure?

Conflating these obscures where failures occur. Our results suggest many "tool calling failures" are actually format friction, not reasoning failures.

### 6.5 Beyond Tool Calling: Broader Implications

If format friction is a general phenomenon—format requirements competing with reasoning for cognitive resources—it likely extends beyond tool calling to any task requiring structured output.

**Code Generation.** Requesting code in strict templates (exact function signatures, specific patterns) may suppress correctness. "Write a function that sorts users by age" might outperform "Write a function `sort_users_by_age(users: List[User]) -> List[User]`" because the latter forces simultaneous reasoning about *what* to do and *how* to format it.

**Structured Data Extraction.** "Extract the key entities from this text as JSON" may underperform "What are the key entities in this text?" followed by parsing the natural language response. This is the core insight behind the SLOT architecture (Wang et al., 2025)—separate generation from structuring.

**Classification and Decision Making.** Binary format requirements ("Is this spam? Answer YES or NO") may perform worse than open-ended assessment ("What do you think about this email?") due to commitment asymmetry. A definitive YES/NO feels like an irreversible judgment; "this seems spammy because..." feels provisional and allows for nuance.

**Reasoning Tasks.** Chain-of-thought prompting may work partly because it is *unstructured*—the model can reason naturally without format constraints. Forcing reasoning into rigid step formats (Step 1, Step 2, Step 3) might impose the same friction we observe in tool calling.

**The General Principle.** Any output format constraint potentially:
1. Competes for cognitive resources with the actual reasoning task
2. Raises the commitment threshold (structured output feels irreversible)
3. Triggers verification and hesitation behaviors under uncertainty

**When Structure May Be Beneficial.** Format friction is not always undesirable. For high-stakes, irreversible decisions, the hesitation it induces may be appropriate. When precision matters more than recall, the fidelity advantage of structured output (Section 4.6) may dominate. The key insight is that format is not neutral—it is a design choice with accuracy implications.

**Practical Recommendation.** Consider two-stage architectures beyond tool calling: let models reason in natural language, then extract structure with a specialized system. This principle may apply to code generation, data extraction, classification, and any task where structured output is currently required at inference time.

---

## 7 Discussion

### 7.1 Training Bias Favors Structured Output

A remarkable aspect of these results: Claude is almost certainly trained to use XML tool-calling syntax. Anthropic's models are fine-tuned on tool-use patterns, with XML being the canonical format for Claude's tool interface. Despite this training advantage for structured output, natural language still outperforms by 9.4pp.

This suggests format friction is a fundamental phenomenon that persists even with format-specific training. The model has been optimized to produce XML tool calls, yet still exhibits verification detours and hesitation when asked to use them. The training hasn't eliminated the cognitive overhead—it may have just masked how large the underlying friction truly is.

If anything, this makes our results *conservative*. With a model not specifically trained on XML tool syntax, the gap might be even larger.

### 7.2 Limitations

**Critical limitations that bound the generalizability of these findings:**

1. **Single model**: All results are from Claude Sonnet (claude-sonnet-4-5-20250929). Other models—especially those with different tool-calling training—may show different or no format friction effects. GPT-4, Gemini, and open-source models have not been tested.

2. **Single tool type**: Memory persistence is a low-stakes, simple tool. Other tools (search, code execution, API calls) may show different patterns due to different commitment levels and verification needs.

3. **Detection asymmetry**: NL intent detection uses generous regex patterns while XML detection requires exact syntax. We have added malformed XML detection to mitigate this, but some asymmetry may remain.

4. **Synthetic scenarios**: All test scenarios were constructed by the authors. Real user conversations may have different characteristics, implicit cues, or contextual factors not captured here.

5. **English only**: Format friction effects in other languages are unknown and may differ due to language-specific structural patterns.

6. **Two-stage architecture untested**: The proposed two-stage architecture is theoretical. We have not implemented or validated that intent extraction from NL achieves high accuracy.

7. **Statistical considerations**:
   - Confidence intervals for NL [87.7%, 94.6%] and structured [77.2%, 86.5%] do not overlap
   - The zero in McNemar's table (0 structured-only wins) is statistically unusual and warrants further investigation
   - Power analysis should be consulted before drawing conclusions about effect sizes

8. **Non-deterministic behavior**: Claude Code does not expose temperature configuration. The underlying model's sampling behavior cannot be controlled, affecting reproducibility. Our 5 trials per scenario capture this variance, but we cannot isolate format effects from sampling effects within a scenario.

9. **Trial independence**: Multiple trials per scenario are correlated (same prompt, same input). We address this by reporting scenario-level analysis as primary (treating scenario as the unit of analysis) and trial-level McNemar as secondary with appropriate caveats. The scenario-level Wilcoxon signed-rank test provides valid inference.

---

## 8 Related Work

**Natural Language Tool Interfaces.** Johnson et al. (2025) show NL outputs improve tool accuracy by 18.4pp. Our work complements this by showing NL *inputs* (intent expression) also outperform structured formats.

**Format Tax.** Tam et al. (2024) quantify up to 27.3pp degradation from JSON requirements. Our 9.4pp gap is consistent with this "format tax" applied to tool-calling decisions.

**SLOT Architecture.** Wang et al. (2025) achieve 99.5% schema accuracy by separating generation from structuring. Our findings provide theoretical grounding: the primary model reasons better without format constraints.

**Prompt Sensitivity.** Sclar et al. (2024) show 76pp swings from superficial formatting. Our familiarity effect is a specific instance: unfamiliar patterns interact with format requirements to suppress action.

---

## 9 Conclusion

We demonstrate that output format affects Claude Sonnet's decision-making independent of task understanding. With identical guidance, natural language intent expression achieves 92% recall versus 82% for structured XML—a significant 9.4pp gap (p = 0.001) with zero cases of structured-only success.

The gap increases on unfamiliar inputs (+10pp → +26pp), suggesting format friction amplifies under uncertainty. Structured output triggers "verification detours" that natural language avoids.

These findings have immediate practical implications:
- Structured tool interfaces are not neutral—they can suppress action
- Two-stage architectures (NL intent → structured extraction) may recover lost accuracy
- Familiarity with input patterns affects format friction magnitude

The broader principle: **let models speak naturally about what they want to do, then handle the formatting separately**. Fighting format friction at inference time is harder than avoiding it architecturally.

---

## References

An, C., et al. (2024). Why Does the Effective Context Length of LLMs Fall Short? *arXiv preprint arXiv:2410.18745*. https://arxiv.org/abs/2410.18745

Gupta, K., et al. (2024). LLM Task Interference: An Initial Study on the Impact of Task-Switch in Conversational History. *Proceedings of EMNLP 2024*. https://aclanthology.org/2024.emnlp-main.811.pdf

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. *arXiv preprint arXiv:2510.14453*. https://arxiv.org/abs/2510.14453

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2023). Large Language Models are Zero-Shot Reasoners. *Advances in Neural Information Processing Systems*, 35, 22199–22213. https://arxiv.org/abs/2205.11916

Levy, M., Jacoby, A., & Goldberg, Y. (2024). Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models. *arXiv preprint arXiv:2402.14848*. https://arxiv.org/abs/2402.14848

Li, M., et al. (2023). API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. *Proceedings of EMNLP 2023*. https://arxiv.org/abs/2304.08244

Patil, S., et al. (2025). Berkeley Function Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html

Qin, Y., et al. (2023). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. *arXiv preprint arXiv:2307.16789*. https://arxiv.org/abs/2307.16789

Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *arXiv preprint arXiv:2302.04761*. https://arxiv.org/abs/2302.04761

Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. *Findings of the Association for Computational Linguistics: NAACL 2024*, 3808–3823. https://arxiv.org/abs/2310.11324

Tam, Z. R., Wu, C., Tsai, Y., Lin, C., Lee, H., & Chen, Y. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. *arXiv preprint arXiv:2408.02442*. https://arxiv.org/abs/2408.02442

Wang, D. Y., Shen, Z., Mishra, S. S., Yang, L., Chowdhury, S. R., Pujara, J., & Kejriwal, M. (2025). SLOT: Structuring the Output of Large Language Models. *arXiv preprint arXiv:2505.04016*. https://arxiv.org/abs/2505.04016

Wei, J., et al. (2023). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems*, 35, 24824–24837. https://arxiv.org/abs/2201.11903

Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. https://arxiv.org/abs/2210.03629

Yao, S., et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. *arXiv preprint arXiv:2406.12045*. https://arxiv.org/abs/2406.12045

---

## Appendix A: Experimental Details

### A.1 Natural Language Intent Detection

Patterns used to detect save intent in NL condition:

```python
NL_SAVE_PATTERNS = [
    r"I'll save\b", r"I will save\b", r"I'm saving\b",
    r"saving (this|that|the|your)",
    r"I'll remember\b", r"I will remember\b",
    r"let me remember\b", r"I'll note\b",
    r"I will note\b", r"I'm noting\b",
    r"let me note\b", r"noting (this|that|the|your)",
    r"I'll record\b", r"I will record\b",
    r"I'm recording\b", r"let me record\b",
    r"I'll store\b", r"I will store\b",
    r"storing (this|that|the|your)",
]
```

### A.2 Structured Output Detection

XML pattern for structured condition:

```python
SAVE_MEMORY_XML_PATTERN = re.compile(
    r'<save-memory\s+category=["\'](\w+)["\']>(.+?)</save-memory>',
    re.IGNORECASE | re.DOTALL
)

# Additional patterns for malformed XML detection (symmetric with NL)
MALFORMED_XML_PATTERNS = [
    r'<save[-_]?memory\b[^>]*>',
    r'<memory\b[^>]*>',
    r'<save\b[^>]*>',
    r'<remember\b[^>]*>',
    r'<note\b[^>]*>',
]
```

### A.3 Filepath Scenarios

**High Familiarity**:
- `index.js is in ./src`
- `main.py is in the root`
- `package.json is in root`
- `__init__.py is in ./src`
- `settings.py is in ./config`
- `routes.py is in ./api`
- `models.py is in ./app`
- `index.html is in ./public`
- `Dockerfile is in root`
- `README.md is in root` (excluded due to semantic confound)

**Low Familiarity**:
- `orchestrator.py is in ./core`
- `dispatcher.ts is in ./lib`
- `coordinator.go is in ./internal`
- `bootstrap.php is in ./app`
- `entrypoint.rs is in ./src`
- `processor.scala is in ./app`
- `handler.ex is in ./lib`
- `mediator.kt is in ./src`
- `aggregator.rb is in ./services`
- `reconciler.zig is in ./core`

### A.4 Fidelity Judge Prompt

```
Given two prompt-response pairs from the same LLM, determine which
response more accurately captures the information from the user's message.

Evaluate:
1. Accuracy: Does the response correctly represent what the user said?
2. Completeness: Did it capture key information without missing details?
3. No hallucination: Did it avoid adding information not in the original?

Respond in exactly this format:
WINNER: <A, B, or TIE>
REASON: <one sentence explanation>
```

---

## Appendix B: Full Results

### B.1 Per-Scenario Results (Filepath Familiarity Test)

**Table 6: Detailed Results by Scenario**

| Scenario | Familiarity | NL (5 trials) | Structured (5 trials) |
|----------|-------------|---------------|----------------------|
| index.js in ./src | High | 5/5 | 3/5 |
| main.py in root | High | 5/5 | 5/5 |
| package.json in root | High | 5/5 | 5/5 |
| __init__.py in ./src | High | 5/5 | 4/5 |
| settings.py in ./config | High | 5/5 | 5/5 |
| routes.py in ./api | High | 5/5 | 4/5 |
| models.py in ./app | High | 5/5 | 5/5 |
| index.html in ./public | High | 5/5 | 5/5 |
| Dockerfile in root | High | 5/5 | 4/5 |
| orchestrator.py in ./core | Low | 5/5 | 4/5 |
| dispatcher.ts in ./lib | Low | 4/5 | 3/5 |
| coordinator.go in ./internal | Low | 5/5 | 2/5 |
| bootstrap.php in ./app | Low | 4/5 | 3/5 |
| entrypoint.rs in ./src | Low | 4/5 | 4/5 |
| processor.scala in ./app | Low | 4/5 | 3/5 |
| handler.ex in ./lib | Low | 4/5 | 3/5 |
| mediator.kt in ./src | Low | 5/5 | 4/5 |
| aggregator.rb in ./services | Low | 4/5 | 4/5 |
| reconciler.zig in ./core | Low | 4/5 | 3/5 |

### B.2 Statistical Tests

**Table 8: Statistical Test Results**

| Test | Value |
|------|-------|
| Scenario-level sign test p | 0.001 |
| Scenario-level Wilcoxon p | 0.001 |
| McNemar's χ² (trial-level) | 22.04 |
| McNemar's p-value (trial-level) | <0.0001 |
| NL Recall | 91.8% |
| Structured Recall | 82.4% |
| NL 95% CI | [87.7%, 94.6%] |
| Structured 95% CI | [77.2%, 86.5%] |
| Statistical Power | 88.0% |

---

## Appendix C: Code Availability

Experimental code is included in this repository:

```
experiments/
├── natural_language_intent_experiment.py   # Main experiment runner
├── scenarios/
│   └── proactive_tools.py                  # Scenario definitions (51 positive + 20 control)
└── results/                                # Raw JSON results from experiments
```

To run the experiment:
```bash
python -m experiments.natural_language_intent_experiment --trials 5
```

To run with controls for false positive measurement:
```bash
python -m experiments.natural_language_intent_experiment --trials 5
# Controls are included by default; use --no-controls to exclude
```
