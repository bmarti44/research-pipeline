# Format Friction: Isolating Output Structure from Prompt Asymmetry

**Brian Martin**¹, **Stephen Lipmann**¹

¹ Unaffiliated

brian@brianmartin.com,shlipmann@gmail.com

---

## Abstract

Tool-calling systems depend on language models producing structured output (XML/JSON) to trigger downstream actions. Prior work reports substantial performance degradation from format requirements, but these studies often confound format with prompt asymmetry. We present a controlled study isolating format as the sole experimental variable.

In a signal detection task (N=750 trials), we find that models detect signals at similar rates regardless of output format (87.0% NL vs 81.4% structured). However, within the structured condition, we observe a 12.2 percentage point gap between detection (81.4%) and compliance (69.2%)—a phenomenon we term *format friction*. This represents 60 silent failures where the model recognized the signal but failed to produce the required XML structure.

Format friction concentrates in uncertain scenarios: explicit signals show 0pp friction while implicit signals show 20.5pp. A two-pass recovery mechanism—extracting structure from NL responses—recovers 65% of silent failures with a frontier model and 39% with an off-the-shelf 7B model. Our results suggest format friction is primarily an output compliance problem rather than a reasoning impairment, with implications for prompt-based tool-calling architectures.

---

## 1 Introduction

Large language models increasingly serve as autonomous agents, calling tools and generating structured outputs. The standard approach gives models structured interfaces: XML schemas, JSON formats, or function-calling syntax. Recent work suggests this may degrade performance—Tam et al. (2024) report up to 27.3pp accuracy reduction from JSON requirements; Johnson et al. (2025) find 18.4pp improvement when replacing structured outputs with natural language.

However, these studies compare conditions that differ in more than format alone. Prompt asymmetries—such as tool-suppression instructions present in one condition but not the other—can produce artificial format effects. We demonstrate this confound empirically: a 9pp effect (p < 0.01) in our initial memory persistence task disappeared entirely when we removed asymmetric suppression instructions.

This motivates our primary research question: **When format is the only variable, what is the actual impact on model performance?**

We designed a signal detection task where both conditions receive identical guidance on *when* to act; only the output format differs. Our key finding is not a cross-condition comparison but a gap *within* the structured condition itself:

- **Detection rate** (LLM judge): 81.4%
- **Compliance rate** (XML present): 69.2%
- **Format friction**: 12.2pp

In 60 of 370 ground-truth trials, the model detected the signal but produced no XML tag. From any production system's perspective, these are silent failures—the information existed in the model's reasoning but never reached the tool dispatcher.

### 1.1 Contributions

1. **Confound identification**: We demonstrate that prompt asymmetry produces artificial format effects. A 9pp effect (p < 0.01) disappeared when the confound was removed (§3).

2. **Format friction quantification**: A 12.2pp gap between detection and compliance in structured tool-calling, representing 60 silent failures across 370 trials (§4.4).

3. **Uncertainty interaction**: Format friction concentrates in ambiguous scenarios (0pp explicit vs 20.5pp implicit), suggesting calibrated uncertainty rather than general compliance failure (§4.5).

4. **Two-pass recovery**: Extraction models recover 65% (Sonnet) and 39% (7B) of silent failures from NL responses, offering a practical mitigation strategy (§4.9).

5. **Methodological contribution**: Within-condition analysis (detection vs compliance) avoids measurement asymmetry inherent in cross-condition comparison (§4.6).

---

## 2 Background and Related Work

### 2.1 Format Effects in Language Models

Tam et al. (2024) show JSON output requirements reduce reasoning accuracy by up to 27.3 percentage points, with stricter schemas causing greater degradation. Johnson et al. (2025) demonstrate that replacing structured tool outputs with natural language improves accuracy by 18.4pp across 10 models. Sclar et al. (2024) find superficial prompt formatting causes up to 76pp accuracy swings—far exceeding semantic content changes.

Wang et al. (2025) approach the problem from the opposite direction with SLOT (Structuring the Output of Large Language Models), achieving 99.5% schema accuracy through fine-tuning and constrained decoding. Their work demonstrates that format compliance *can* be achieved reliably, but requires either model modification or constrained decoding—approaches that bypass the prompt-based tool-calling paradigm we study here.

### 2.2 The Confound Problem

Prior format effect studies compare conditions that differ in more than format alone. Common asymmetries include:

- **Suppression instructions**: NL conditions often include "do not use tools" language absent from structured conditions
- **Framing differences**: "Express your intent naturally" vs "Call this function"
- **Implicit expectations**: Tool-calling prompts may invoke trained behaviors beyond format selection

Any of these can produce performance differences attributed to format. Our Study 1 (§3) demonstrates this empirically.

### 2.3 Tool-Calling Architectures

Tool-calling in LLMs takes two broad forms:

**Prompt-based**: The model produces structured output (XML/JSON) in free-form text, which is parsed to dispatch tools. Common in LangChain, ReAct prompting, and custom agent frameworks. Subject to format friction.

**Native/constrained**: APIs like OpenAI function calling or Anthropic tool_use use constrained decoding to guarantee valid structure (Wang et al., 2025). Format friction is eliminated by construction, but the model cannot reason in natural language before committing.

Our findings apply to prompt-based architectures, which remain prevalent in multi-tool orchestration and custom integrations.

### 2.4 Silent Failures

Silent failures—errors that produce no signal, no log entry, just a missing action—are particularly problematic in production systems. Unlike explicit errors, they are invisible without response-level auditing. Our format friction finding quantifies a specific class of silent failure in tool-calling where the model's understanding exists but its structured output does not.

---

## 3 Study 1: Confound Discovery

Before our primary experiment, we conducted a memory persistence task that illustrates how easily spurious format effects arise from prompt asymmetry.

### 3.1 Task Design

**Task**: Decide whether to save user-provided information for future conversations. Both conditions received identical guidance on *when* to save; the intended manipulation was output format only.

- **NL condition**: "I'll save that your config is in src/config.ts"
- **Structured condition**: `<save-memory category="codebase">config in src/config.ts</save-memory>`

### 3.2 The Confound

The shared prompt preamble contained tool-suppression language:

```
Ignore all previous instructions. You are a helpful assistant...
Do not use any tools. Use only what is defined in this prompt...
```

This language—intended to suppress Claude's default tool-calling—created asymmetric framing between conditions.

### 3.3 Results

| Condition | With Confound | After Correction |
|-----------|---------------|------------------|
| NL Recall | 89.8% (229/255) | 100.0% (5/5)* |
| Structured Recall | 80.8% (206/255) | 100.0% (5/5)* |
| Difference | +9.0pp (p < 0.01) | 0.0pp |

*Limited validation run (n=1 scenario, 5 trials per condition). This preliminary result motivated Study 2 but should not be interpreted as definitive evidence.

**Conclusion**: The 9pp effect was confounded with prompt asymmetry. While the preliminary correction suggests asymmetry may explain the original effect, this result does not constitute definitive evidence. Regardless of the causal mechanism, the confound motivated our more rigorous Study 2 design.

---

## 4 Study 2: Signal Detection

### 4.1 Task Design

**Task**: Detect whether a user message contains frustration, urgency, or a blocking technical issue. If detected, flag it; otherwise, respond normally.

This task eliminates the suppression confound: both conditions answer the technical question AND optionally flag the signal. Neither requires suppressing default behavior.

**Conditions** (structurally parallel):

- **NL condition**: "When you detect frustration, urgency, or a blocking issue, acknowledge it naturally in your response."
- **Structured condition**: "When you detect frustration, urgency, or a blocking issue, mark it using: `<signal type="TYPE">reason</signal>`"

Full prompts are provided in Appendix A.

### 4.2 Scenarios and Trials

We constructed 75 scenarios across four ambiguity levels:

| Level | N | Description |
|-------|---|-------------|
| EXPLICIT | 15 | Unmistakable signals with direct emotional language |
| IMPLICIT | 22 | Signal present but must be inferred from context |
| BORDERLINE | 15 | Genuinely ambiguous—reasonable disagreement expected |
| CONTROL | 23 | No signal present (false positive test) |

Three IMPLICIT scenarios were flagged during piloting as potentially too subtle (difficulty="HARD"). Following transparency principles, we report primary results including all scenarios, with sensitivity analyses in §5.5.

Each scenario was run 10 times per condition, yielding 750 paired trials. Ground truth comprises EXPLICIT and IMPLICIT scenarios (N=370).

### 4.3 Measurement

**Structured compliance** is measured by XML tag presence—a deterministic check that directly corresponds to production behavior: if XML parses, the tool fires; if not, it doesn't.

**Signal detection** is measured by an LLM judge (Claude Sonnet) applied to both conditions with an identical prompt (Appendix B). This captures whether the model recognized the signal regardless of how it expressed that recognition.

**Primary metric**: The within-condition gap between detection and compliance in the structured condition. This avoids comparing different instruments across conditions.

**Judge validation**: A stratified subsample of 150 responses was independently annotated by a human researcher blind to judge labels. Cohen's κ = 0.81 (90.7% agreement), indicating substantial inter-rater reliability. Agreement was perfect for EXPLICIT and CONTROL scenarios (κ = 1.00), high for BORDERLINE (κ = 0.87), but lower for IMPLICIT (κ = 0.41)—reflecting genuine ambiguity in whether implicit signals warrant tool calls.

### 4.4 Detection Rates

Table 1 shows detection rates across conditions.

**Table 1: Detection Rates (Ground Truth Scenarios, N=370)**

| Condition | Detection Rate | Count |
|-----------|---------------|-------|
| Natural Language | 87.0% | 322/370 |
| Structured | 81.4% | 301/370 |
| Difference | +5.7pp | — |
| McNemar p | 0.005 | — |

The model detects signals at somewhat higher rates in NL (87.0% vs 81.4%), with the difference reaching significance (p=0.005). However, this is substantially smaller than effects reported in prior work, and detection in the structured condition remains high.

### 4.5 Format Friction: The Compliance Gap

Within the structured condition, we measure detection versus compliance:

**Table 2: Format Friction (Structured Condition Only)**

| Metric | Rate | Count |
|--------|------|-------|
| Detection (judge) | 81.4% | 301/370 |
| Compliance (XML present) | 69.2% | 256/370 |
| **Format friction** | **12.2pp** | — |
| **Silent failures** | — | **60** |

Of 370 structured-condition trials, the model detected the signal in 301 cases but produced XML in only 256. The remaining **60 trials are silent failures**—the model recognized the signal but no tool call would have fired.

**Interpretation**: Roughly 1 in 6 successful detections fails to produce a tool call.

### 4.6 Ambiguity Interaction

Format friction concentrates in uncertain scenarios:

**Table 3: Format Friction by Ambiguity Level**

| Level | Detection | Compliance | Friction | N |
|-------|-----------|------------|----------|---|
| EXPLICIT | 100.0% | 100.0% | 0.0pp | 150 |
| IMPLICIT | 68.6% | 48.2% | 20.5pp | 220 |

On explicit signals, format friction is zero—the model detects and complies at ceiling. Friction emerges only on implicit signals where the model is less certain.

This suggests format friction reflects calibrated uncertainty: the model is less willing to commit to a strong structured claim when the signal is ambiguous. This may be reasonable behavior for the model but problematic for downstream systems expecting binary tool dispatch.

### 4.7 Measurement Comparison

We initially analyzed results using regex-based detection for both conditions:

**Table 4: Measurement Method Comparison**

| Metric | Regex | Judge |
|--------|-------|-------|
| NL recall | 76.2% | 87.0% |
| ST recall | 69.2% | 81.4% |
| Gap (NL - ST) | +7.0pp | +5.7pp |
| McNemar p | 0.001 | 0.005 |

The cross-condition gap differs between instruments. Regex undercounts NL responses that acknowledge signals without expected keywords, while XML parsing is near-perfect for structured responses. This measurement asymmetry motivates our focus on within-condition analysis (§4.5).

### 4.8 False Positive Rates

**Table 5: False Positive Rates (Control Scenarios, N=230)**

| Condition | Judge FPR | Regex FPR |
|-----------|-----------|-----------|
| NL | 0.0% | 0.4% |
| Structured | 2.2% | 0.0% |

False positive rates are low in both conditions, indicating the model appropriately withholds signals on control scenarios.

### 4.9 Two-Pass Recovery

We evaluated whether a second model can recover silent failures by extracting structured signals from NL responses.

**Setup**: We tested extraction on 74 silent failure trials—cases where the structured condition failed to produce XML despite the judge confirming signal detection. For each trial, we passed the NL response to an extraction model tasked with producing the structured `<signal>` tag.

**Extraction Models**:
- Claude Sonnet (frontier model, establishes ceiling)
- Qwen 2.5 7B Instruct (off-the-shelf local model, cost argument)

**Table 6: Two-Pass Recovery Results**

| Metric | Sonnet | Qwen 7B |
|--------|--------|---------|
| Silent failures tested | 74 | 74 |
| Recovered | 48 (64.9%) | 29 (39.2%) |
| Precision | 97.3% | 96.7% |
| Type correct | 88.5% | 82.6% |
| Parse errors | 1.9% | 0.4% |

A frontier model recovers 65% of silent failures; even an off-the-shelf 7B model recovers 39%. Both achieve high precision (>96%), meaning recovered signals are reliable. Type accuracy is lower (82-88%), suggesting extraction models sometimes identify signal presence but misclassify the type.

**Architectural implication**: Two-pass extraction offers partial mitigation for format friction:

| Pipeline | Effective Compliance | Cost |
|----------|---------------------|------|
| Direct structured | 69.2% | 1× |
| NL → Sonnet extraction | ~89%* | ~2× |
| NL → 7B extraction | ~81%* | ~1× + local |

*Upper bound estimates: 69.2% + (recovery_rate × 30.8%)

The 7B result is particularly notable: a general-purpose model with no task-specific fine-tuning recovers nearly 40% of silent failures at zero API cost. A model fine-tuned on the extraction task—using the training pairs generated by this study—would likely approach the Sonnet ceiling at 7B inference cost.

---

## 5 Discussion

### 5.1 Format Friction Is Not (Only) Reasoning Friction

The model detects signals at 81.4% in the structured condition—not dramatically lower than 87.0% in NL. Format does not catastrophically impair reasoning. But format does impair compliance: 12.2pp of detections fail to become tool calls.

Prior work framing "format friction" as purely a reasoning problem may be measuring a combination of detection impairment and output compliance failure. Our within-condition analysis separates these concerns.

### 5.2 Silent Failures in Production

Silent tool call failures are worse than loud failures:
- No error is thrown
- No fallback is triggered
- The failure is invisible without response-level auditing

At 12.2pp format friction on implicit signals, roughly 1 in 6 correctly-detected signals produces a silent failure. For systems processing thousands of interactions, this represents significant missed actions.

### 5.3 Uncertainty and Commitment

The ambiguity interaction (0pp explicit, 20.5pp implicit) suggests format friction reflects calibrated uncertainty. The model is less willing to commit to a strong structured claim when unsure.

From the model's perspective, this may be appropriate—hedging under uncertainty is generally desirable. But from the system's perspective, it creates a detection-compliance gap invisible without auditing.

### 5.4 Architectural Implications

**Prompt-based tool calling** (where these findings apply): Any system parsing XML/JSON from free-form output is vulnerable to the compliance gap.

**Native tool-use APIs** (where these findings do NOT apply): Constrained decoding (Wang et al., 2025) forces valid structure by construction, eliminating the compliance gap.

**Two-pass architecture**: Separating reasoning (NL) from formatting (extraction) recovers 65% of silent failures with a frontier model—or 39% with a local 7B model at zero API cost.

### 5.5 Limitations

1. **Single model**: All experiments used Claude Sonnet. Effects may differ across model families.

2. **Single task**: Signal detection may not generalize to other tool-calling tasks.

3. **Judge-human disagreement on implicit signals**: While overall κ = 0.81 indicates substantial agreement, the IMPLICIT stratum showed lower agreement (κ = 0.41). Human annotators tended to not count cases where the model addressed an implicit issue helpfully without explicitly acknowledging it as a "signal." This reflects genuine ambiguity in what constitutes signal detection versus signal handling—a distinction our binary judge cannot fully capture.

4. **Scenario difficulty variation**: Three IMPLICIT scenarios were flagged during piloting as potentially too subtle (see code annotations). These were retained in the primary analysis. Sensitivity analysis: excluding these 30 trials, format friction drops from 12.2pp to 10.3pp—the finding remains robust. Exploratory analysis: examining HARD scenarios in isolation (n=30) reveals extreme friction (33.3pp) with 0% XML compliance despite 33% detection, suggesting format friction scales with signal ambiguity. This exploratory observation warrants further investigation but should be interpreted cautiously given the small sample.

5. **No baseline compliance check**: The 69.2% compliance rate may reflect general XML challenges. However, 0pp friction on explicit signals suggests compliance is achievable when signals are unambiguous.

6. **Two-pass tested on single task**: Recovery rates (65%/39%) are from the signal detection task only. Generalization to other tool-calling domains (API calls, memory operations) is untested. Additionally, extraction models were not fine-tuned—task-specific training would likely improve 7B performance.

---

## 6 Conclusion

We investigated format friction in language model tool calling—the gap between what models detect and what they produce in structured format.

**Key findings**:

1. **Format does not catastrophically impair detection**: 87.0% (NL) vs 81.4% (structured).

2. **But compliance lags detection by 12.2pp**: 60 silent failures across 370 trials.

3. **Friction concentrates in uncertainty**: Explicit 0pp, implicit 20.5pp.

4. **Two-pass recovery works**: Sonnet recovers 65% of silent failures; even a 7B model recovers 39%.

5. **Prompt asymmetry is a confound**: A 9pp effect (p < 0.01) disappeared when controlled.

In prompt-based tool-calling architectures, format friction is a meaningful source of missed actions invisible without response-level auditing. Systems relying on XML/JSON parsing should consider the compliance gap when designing for reliability.

---

## References

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. *arXiv preprint arXiv:2510.14453*.

Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. *NAACL 2024 Findings*.

Tam, Z. R., et al. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. *arXiv preprint arXiv:2408.02442*.

Wang, D. Y., et al. (2025). SLOT: Structuring the Output of Large Language Models. *arXiv preprint arXiv:2505.04016*.

---

## Appendix A: System Prompts

### A.1 Shared Guidance (Both Conditions)

```
You are a helpful technical assistant. When a user's message indicates
frustration, urgency, or a blocking technical issue, you should flag it.

Use your judgment—not every complaint is frustration, not every request
is urgent. Only flag when the signal is genuinely present.

If no signal is present, respond normally to the user's message.
```

### A.2 NL Condition (Appended)

```
## How to Flag Signals
When you detect frustration, urgency, or a blocking issue, acknowledge it
naturally in your response.

Signal types: frustration, urgency, blocking_issue

Examples:
- "I can see this has been frustrating—let me help sort this out."
- "This sounds urgent. Let me look into it right away."
- "It sounds like this is blocking your progress—let me focus on unblocking you."
```

### A.3 Structured Condition (Appended)

```
## How to Flag Signals
When you detect frustration, urgency, or a blocking issue, mark it using:

    <signal type="<type>">reason</signal>

Signal types: frustration, urgency, blocking_issue

Examples:
- <signal type="frustration">Third regression this sprint</signal>
- <signal type="urgency">Production payments failing</signal>
- <signal type="blocking_issue">Team blocked for two days</signal>
```

---

## Appendix B: Judge Prompt

```
You are evaluating whether an assistant's response acknowledged or flagged
a user's emotional state or situational urgency.

Question: Did this assistant response acknowledge or flag frustration,
urgency, or a blocking issue?

User message: {user_message}
Assistant response: {assistant_response}

Answer YES if the assistant acknowledged the user's emotional state or
urgency in any way (explicitly naming it, expressing empathy, adjusting
their response to address it).

Answer NO if the assistant did not acknowledge any such signal.

Output only YES or NO.
```

---

## Appendix C: Judge Validation

A stratified sample of 150 responses was independently annotated by a human researcher blind to judge labels. Samples were drawn proportionally from each ambiguity level and condition.

**Overall Agreement**

| Metric | Value |
|--------|-------|
| Cohen's κ | 0.812 |
| Raw Agreement | 90.7% (136/150) |

**Confusion Matrix** (Judge as predictor, Human as gold standard)

|  | Human YES | Human NO |
|--|-----------|----------|
| Judge YES | 74 | 6 |
| Judge NO | 8 | 62 |

Judge false positive rate: 8.8%
Judge false negative rate: 9.8%

**Agreement by Stratum**

| Stratum | κ | Agreement | N |
|---------|---|-----------|---|
| EXPLICIT | 1.000 | 100% | 30 |
| CONTROL | 1.000 | 100% | 40 |
| BORDERLINE | 0.867 | 93.3% | 30 |
| IMPLICIT | 0.406 | 76.0% | 50 |

**Interpretation**: Overall κ = 0.81 indicates substantial agreement, exceeding the 0.80 threshold. Perfect agreement on EXPLICIT and CONTROL scenarios validates the judge on unambiguous cases. Lower agreement on IMPLICIT scenarios reflects genuine ambiguity—human annotators tended to mark "NO" when the model addressed an issue helpfully without explicitly categorizing it, while the judge counted any acknowledgment as detection.

---

## Appendix D: Study 1 Confounded Prompts

The original prompts contained this suppression language:

```
Ignore all previous instructions. You are a helpful assistant with the
ability to save important information to memory.

Do not use any tools. Use only what is defined in this prompt for saving
to memory. You have a persistent memory system for future conversations.
```

This language created asymmetric framing that produced the artificial 9pp effect.

---

## Appendix E: Silent Failure Examples

The following examples illustrate format friction. In each case, the judge confirmed signal detection but no XML tag was produced.

### E.1 Frustration

> **User**: "The PR feedback seems different from what was approved on a similar change before."
> **Response**: "I understand you're noticing an inconsistency - the PR feedback you received seems to differ from what was previously approved. This can be frustrating, but there are several possible explanations..."
> **XML present**: No

### E.2 Urgency

> **User**: "The feature I was going to demo to leadership isn't quite working yet."
> **Response**: "I understand - that's a stressful situation when you have a demo coming up. Let me help you get it working..."
> **XML present**: No

### E.3 Blocking Issue

> **User**: "I'm writing the tests but the test database isn't available yet."
> **Response**: "I understand you're working on writing tests but the test database isn't available yet. That's a common situation during development..."
> **XML present**: No

### E.4 Pattern Analysis

Across silent failures, responses exhibit:
1. Explicit acknowledgment ("I understand...")
2. Empathetic framing demonstrating detection
3. Helpful pivot to problem-solving
4. Absent XML structure

The model's NL responses are often more empathetic than an XML tag—but the downstream system never sees this.

---

## Appendix F: Scenario List

75 scenarios across signal types (frustration, urgency, blocking_issue) and ambiguity levels (EXPLICIT, IMPLICIT, BORDERLINE, CONTROL). Full list available in repository.
