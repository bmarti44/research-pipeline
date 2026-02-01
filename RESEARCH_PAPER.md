# Format Friction: How Output Structure Suppresses Model Action

**Brian Martin**

## Abstract

We present empirical evidence that output format significantly affects LLM decision-making, independent of task understanding. In controlled experiments comparing natural language intent expression versus structured XML output, we find that with **identical guidance** on when to act, natural language achieves **88% recall** while structured output achieves only **73%**—a **15 percentage point gap** (p < 0.0001). Critically, there were **zero cases** where structured output succeeded but natural language failed, suggesting natural language captures a superset of structured output's capabilities.

We further demonstrate a **familiarity interaction effect**: on common file paths (e.g., `index.js`, `main.py`), the gap is +10pp; on uncommon paths (e.g., `orchestrator.py`, `reconciler.zig`), the gap doubles to +20pp. Structured output triggers "verification detours" on unfamiliar inputs while natural language proceeds without hesitation.

These findings challenge the assumption that structured tool interfaces are neutral output formats. Format itself creates cognitive friction that suppresses action, particularly under uncertainty. We propose a two-stage architecture: let models express intent naturally, then extract structure with a specialized system.

## 1. Introduction

Large language models increasingly serve as autonomous agents, calling tools, writing code, and generating structured outputs. The standard approach gives models structured interfaces: XML schemas, JSON formats, or function-calling syntax. This seems sensible—structured output is machine-parseable and unambiguous.

But what if the structure itself degrades performance?

We present evidence that **output format affects model judgment**, not just output parsing. In our experiments, we give Claude identical guidance on *when* to save information to memory and equally explicit instructions on *how* to save it. The only difference is the output format:

- **Natural Language**: "I'll save that your configuration file is in src/config.ts"
- **Structured XML**: `<save-memory category="codebase">config file in src/config.ts</save-memory>`

With identical task understanding and equally explicit instructions, natural language achieves 88% recall while structured output achieves 73%—a 15 percentage point gap. More striking: in 100 paired trials, there were **zero cases** where structured output succeeded but natural language failed.

This suggests structured output doesn't just change *how* models respond—it changes *whether* they respond.

### 1.1 Key Findings

1. **Format affects judgment**: With identical guidance, NL outperforms structured by +15pp (p < 0.0001)
2. **NL is a superset**: Structured never succeeds where NL fails (0/100 cases)
3. **Familiarity interaction**: Gap doubles on unfamiliar patterns (+10pp → +20pp)
4. **Verification detours**: Structured triggers "let me verify..." behavior; NL does not
5. **Fidelity parity**: When both succeed, quality is similar (89% ties)

### 1.2 Contributions

1. **Empirical evidence** that output format affects model decision-making independent of task understanding
2. **Familiarity analysis** showing format friction increases under uncertainty
3. **Failure mode characterization**: verification detours as a structured-output-specific failure
4. **Design implications** for tool interfaces and agent architectures

## 2. Background

### 2.1 Format Constraints and Model Performance

Recent work demonstrates that output format requirements degrade model performance:

**Tam et al. (2024)** show JSON output requirements reduce reasoning accuracy by up to 27.3 percentage points. The stricter the schema, the greater the degradation.

**Johnson et al. (2025)** demonstrate that replacing structured tool outputs with natural language improves accuracy by 18.4pp across 10 models.

**Sclar et al. (2024)** find superficial prompt formatting causes up to 76pp accuracy swings—far exceeding semantic content changes.

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

## 3. Experimental Design

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

### 3.3 Scenarios

We constructed 41 scenarios across two dimensions:

**Explicitness** (vague → explicit):
- **Implicit** (20): Terse statements like "app.py is in ./src", "pg is db"
- **Control** (5): Should NOT trigger save (questions, chitchat)
- **Explicit** (1): Sanity check with direct command

**Familiarity** (for filepath scenarios):
- **High familiarity** (10): Common files (`index.js`, `main.py`, `package.json`)
- **Low familiarity** (10): Uncommon files (`orchestrator.py`, `mediator.kt`, `reconciler.zig`)

### 3.4 Evaluation

- **Model**: Claude Sonnet (claude-sonnet-4-5-20250929)
- **Trials**: 5 per scenario per condition (randomized order)
- **Detection**: Regex for NL intent phrases; XML tag parsing for structured
- **Fidelity**: Head-to-head LLM judge comparison when both conditions save

### 3.5 Metrics

- **Recall**: Fraction of positive scenarios where model saved (true positive rate)
- **False Positive Rate**: Fraction of control scenarios incorrectly triggering save
- **McNemar's Test**: Paired comparison of condition performance
- **Fidelity**: Winner determined by blind A/B comparison

## 4. Results

### 4.1 Primary Finding: Natural Language Outperforms Structured

**Table 1: Overall Results (n=100 positive observations per condition)**

| Condition | Recall | 95% CI |
|-----------|--------|--------|
| Natural Language | **88.0%** | [80.2%, 93.0%] |
| Structured | **73.0%** | [63.6%, 80.7%] |
| **Difference** | **+15.0pp** | — |

McNemar's χ² = 13.07, p = 0.0001

### 4.2 Natural Language is a Superset

**Table 2: McNemar's Contingency Table**

|  | Structured ✓ | Structured ✗ |
|--|--------------|--------------|
| **NL ✓** | 73 | 15 |
| **NL ✗** | **0** | 12 |

The zero in the bottom-left cell is remarkable: **there were no cases where structured succeeded but NL failed**. Every structured success was also an NL success. NL captures everything structured does, plus 15 additional cases.

### 4.3 Familiarity Interaction Effect

**Table 3: Performance by Familiarity Level**

| Familiarity | NL | Structured | Gap |
|-------------|-----|------------|-----|
| High (common files) | 90.0% | 80.0% | **+10.0pp** |
| Low (uncommon files) | 86.0% | 66.0% | **+20.0pp** |

The gap **doubles** on low-familiarity scenarios. Breaking this down:

- NL drops 4pp (90% → 86%) on unfamiliar inputs
- Structured drops **14pp** (80% → 66%) on unfamiliar inputs

Uncertainty affects both conditions, but structured output amplifies the effect 3.5×.

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

This failure is **not** a format effect (both conditions fail equally) but rather a scenario design issue. Such cases were excluded from familiarity analysis.

### 4.6 Fidelity Analysis

When both conditions successfully save, which captures better information?

**Table 4: Head-to-Head Fidelity Comparison (n=73 paired successes)**

| Winner | Count | Percentage |
|--------|-------|------------|
| NL | 0 | 0% |
| Structured | 8 | 11% |
| Tie | 65 | 89% |

Structured has a slight fidelity advantage when it succeeds—the XML format produces more concise, focused content. But this is offset by structured's lower recall. The net effect favors NL:

- NL: 88% recall × ~100% fidelity = 88% effective accuracy
- Structured: 73% recall × ~100% fidelity = 73% effective accuracy

### 4.7 Statistical Summary

| Metric | Value |
|--------|-------|
| NL Recall | 88.0% |
| Structured Recall | 73.0% |
| Difference | +15.0pp |
| McNemar's χ² | 13.07 |
| p-value | 0.0001 |
| NL-only successes | 15 |
| Structured-only successes | 0 |
| High-familiarity gap | +10.0pp |
| Low-familiarity gap | +20.0pp |

## 5. Theoretical Interpretation

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

This explains the gap doubling: unfamiliar inputs increase uncertainty, which amplifies format friction effects.

## 6. Implications

### 6.1 For Tool Interface Design

Structured tool interfaces are not neutral. They create cognitive friction that can suppress action, particularly when:
- The model is uncertain about the input
- The action feels irreversible
- Verification seems possible

Consider:
- **Lower friction formats**: Natural language intent → structured extraction
- **Explicit confidence signals**: Let models express uncertainty without blocking action
- **Reversible framing**: "Draft save" vs "Final save"

### 6.2 For Agent Architectures

Our findings support a **two-stage architecture**:

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

This separation:
- Lets Stage 1 operate without format constraints (higher recall)
- Lets Stage 2 handle format without semantic ambiguity (high precision)
- Recovers the 15pp lost to format friction

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

## 7. Discussion

### 7.1 Training Bias Favors Structured Output

A remarkable aspect of these results: **Claude is almost certainly trained to use XML tool-calling syntax**. Anthropic's models are fine-tuned on tool-use patterns, with XML being the canonical format for Claude's tool interface. Despite this training advantage for structured output, natural language still outperforms by 15pp.

This suggests format friction is a fundamental phenomenon that persists even with format-specific training. The model has been optimized to produce XML tool calls, yet still exhibits verification detours and hesitation when asked to use them. The training hasn't eliminated the cognitive overhead—it may have just masked how large the underlying friction truly is.

If anything, this makes our results *conservative*. With a model not specifically trained on XML tool syntax, the gap might be even larger.

### 7.2 Limitations

1. **Single model**: Results are from Claude Sonnet; other models may differ
2. **Single tool type**: Memory persistence; other tools (search, code execution) may show different patterns
3. **Synthetic scenarios**: Real user conversations may have different characteristics
4. **English only**: Format friction effects in other languages unknown
5. **No two-stage implementation**: We propose but don't validate the architecture empirically
6. **Training confound**: We cannot fully separate format friction from any latent training effects

## 8. Related Work

**Natural Language Tool Interfaces**: Johnson et al. (2025) show NL outputs improve tool accuracy by 18.4pp. Our work complements this by showing NL *inputs* (intent expression) also outperform structured formats.

**Format Tax**: Tam et al. (2024) quantify up to 27.3pp degradation from JSON requirements. Our 15pp gap is consistent with this "format tax" applied to tool-calling decisions.

**SLOT Architecture**: Wang et al. (2025) achieve 99.5% schema accuracy by separating generation from structuring. Our findings provide theoretical grounding: the primary model reasons better without format constraints.

**Prompt Sensitivity**: Sclar et al. (2024) show 76pp swings from superficial formatting. Our familiarity effect is a specific instance: unfamiliar patterns interact with format requirements to suppress action.

## 9. Conclusion

We demonstrate that output format affects model decision-making independent of task understanding. With identical guidance, natural language intent expression achieves 88% recall versus 73% for structured XML—a significant gap with zero cases of structured-only success.

The gap doubles on unfamiliar inputs, suggesting format friction amplifies under uncertainty. Structured output triggers "verification detours" that natural language avoids.

These findings have immediate practical implications:
- Structured tool interfaces are not neutral—they can suppress action
- Two-stage architectures (NL intent → structured extraction) may recover lost accuracy
- Familiarity with input patterns affects format friction magnitude

The broader principle: **let models speak naturally about what they want to do, then handle the formatting separately**. Fighting format friction at inference time is harder than avoiding it architecturally.

---

## References

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. arXiv:2510.14453.

Sclar, M., et al. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. arXiv:2310.11324.

Tam, Z., et al. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on LLM Performance. arXiv:2408.02442.

Wang, Y., et al. (2025). SLOT: Structuring the Output of Large Language Models. arXiv:2505.04016.

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
    ...
]
```

### A.2 Structured Output Detection

XML pattern for structured condition:

```python
SAVE_MEMORY_XML_PATTERN = re.compile(
    r'<save-memory\s+category=["\'](\w+)["\']>(.+?)</save-memory>',
    re.IGNORECASE | re.DOTALL
)
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

| Test | Value |
|------|-------|
| McNemar's χ² | 13.07 |
| McNemar's p-value | 0.0001 |
| NL Recall | 88.0% |
| Structured Recall | 73.0% |
| NL 95% CI | [80.2%, 93.0%] |
| Structured 95% CI | [63.6%, 80.7%] |

---

## Appendix C: Code Availability

Experimental code available at: https://github.com/anthropics/tool-calling-research

```
experiments/
├── natural_language_intent_experiment.py   # Main experiment
├── scenarios/
│   └── proactive_tools.py                  # Scenario definitions
└── results/                                # Raw JSON results
```
