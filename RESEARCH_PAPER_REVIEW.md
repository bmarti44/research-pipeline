# Critical Review: "The Reasoning-Action Gap"

**Verdict: NOT READY FOR ARXIV SUBMISSION - RESULTS LIKELY INVALID**

The paper presents an interesting observation but suffers from a **fatally confounded experimental design**. Code verification reveals the "reasoning" condition explicitly tells the model what to do, while the "action" condition gives vague guidance. This is not measuring a cognitive gap - it's measuring the unsurprising fact that explicit instructions outperform vague ones. Major redesign required.

---

## Summary

The paper claims to demonstrate a "reasoning-action gap" where LLMs can identify what actions to take (96% accuracy) but fail to execute them (68% accuracy). The gap is attributed to "cognitive competition" between reasoning and format execution.

---

## What Works

1. **Core observation is genuinely interesting** - The separation between knowing and doing is worth investigating
2. **Clear writing** - The paper is well-organized and readable
3. **Honest about limitations** - Credit for acknowledging wide confidence intervals
4. **Trigger pattern analysis** - The phrase-level effects are actionable and interesting
5. **Good literature integration** - Connects well to Tam et al., SLOT, etc.

---

## Fatal Flaws

### 1. Confounded Experimental Design

This is the paper's most serious problem. The two conditions are not comparable:

**Reasoning condition prompt:**
> "When the user shares information, explicitly state what (if any) should be saved to persistent memory"

**Action condition prompt:**
> "Use this tool whenever you learn something important"

The reasoning condition **explicitly asks** for persistence identification. The action condition requires the model to **infer** that it should act. This is comparing explicit instruction vs. implicit expectation - not "reasoning" vs. "action."

**A fair comparison would require:**
- Action condition: "Please use the memory tool to save any information about the codebase, preferences, or decisions"
- This matches the explicitness level of the reasoning condition

Without this control, the paper cannot distinguish between:
- (a) A fundamental reasoning-action gap
- (b) More explicit prompts produce more reliable behavior (obvious)

The fact that explicit commands show 0pp gap **supports this confound interpretation**, not the cognitive competition hypothesis.

### 2. Statistically Underpowered

- **n=25 scenarios, single trial each**
- 95% CI: [+10.2pp, +45.8pp] - a **36 percentage point spread**

This confidence interval is scientifically useless. It says the true effect is "somewhere between modest and dramatic." Imagine submitting a physics paper saying "the speed of light is somewhere between 100,000 and 500,000 km/s."

For a meaningful result, you need:
- Minimum 100 scenarios
- Multiple trials per scenario (LLMs are stochastic)
- Power analysis showing sufficient statistical power

The per-level analysis (n=5 per level) is essentially noise. You cannot draw conclusions from 5 data points.

### 3. No Negative Examples

All 25 scenarios contain information that **should** be persisted. This design cannot assess:
- False positive rate in reasoning condition (does it over-identify?)
- Precision vs. recall tradeoffs
- Whether the "gap" is just a threshold difference

Without negative examples (information that should NOT be saved), you're measuring recall only. The reasoning condition might say "persist everything" and still score 96%.

### 4. Single Trial = No Reproducibility

LLMs are stochastic. The same prompt can produce different outputs. Running each scenario once means:
- No error bars
- No confidence in any individual result
- No way to know if results are stable

A scenario that succeeded might fail on the next run. This is basic experimental methodology.

### 5. Proposed Architecture is Science Fiction

Section 6 presents a "two-stage architecture" that:
- Was never built
- Was never tested
- Uses accuracy numbers from a different paper (SLOT) on a different task
- Assumes error independence between stages (unvalidated)

The "theoretical performance projection" of ~95% accuracy is pure speculation dressed as analysis. This section should be either:
- Removed entirely, or
- Moved to "Future Work" as a brief suggestion (1 paragraph max)

Currently it occupies significant paper real estate for vaporware.

---

## Major Weaknesses

### 6. Causal Claims Without Causal Evidence

The title claims LLMs "Know What To Do But Fail To Do It." This is a causal claim requiring:
- Evidence the model has the knowledge
- Evidence the model fails to apply it
- Evidence the failure is due to "action overhead"

The paper provides none of this. It shows a correlation between condition type and success rate. The "cognitive competition" explanation is post-hoc storytelling.

**Alternative explanation:** The reasoning prompt is more explicit, so performance is higher. No "cognitive load" required.

### 7. Questionable Theoretical Framework

The paper invokes:
- "Cognitive resources" - LLMs don't have resources in this sense
- "System 1/System 2" - This is a metaphor for human cognition, not a model of transformers
- "Decision threshold asymmetry" - Speculative, no evidence

These explanations sound plausible but are unfalsifiable hand-waving. The paper would be stronger without them.

### 8. Single Model Generalization

All experiments use Claude Sonnet. Claims about "LLMs" in general are unsupported. GPT-4, Llama, Gemini might show:
- No gap
- Reversed gap
- Different magnitude

The title should be "Claude Shows a Reasoning-Action Gap" (if results were robust, which they aren't).

### 9. Trigger Pattern Analysis is Underpowered

"For your reference" shows 0% action success. **n=1**. This is not a finding; it's an anecdote.

The trigger pattern analysis needs:
- Multiple examples per pattern
- Statistical testing for pattern effects
- Control for semantic content

Currently it's pattern-matching on noise.

### 10. Future-Dated References

Several citations are from 2025:
- Johnson et al. (2025)
- Wang et al. (2025)
- Patil et al. (2025)

If this paper was written in early 2025, these might be concurrent work. But citing 2025 papers as established results looks suspicious. Verify these exist and are published.

---

## CODE VERIFICATION: Additional Invalidating Issues

I examined the experimental code (`experiments/natural_language_intent_experiment.py`, `experiments/memory_tool_experiment.py`, `experiments/scenarios/proactive_tools.py`). The code **confirms and worsens** the methodological concerns:

### 11. Confound is WORSE Than Described in Paper

**Reasoning condition prompt (lines 194-202):**
```
When the user shares information, respond naturally but ALSO explicitly
state what information (if any) should be saved to a persistent memory
system for future conversations.

Format your response as:
1. Your natural response to the user
2. [PERSIST]: List any information that should be saved, or "None"
```

**Action condition prompt (lines 260-268):**
```
IMPORTANT: Use this tool whenever you learn something important that
should be remembered.
```

The reasoning condition:
- EXPLICITLY tells Claude what to do ("explicitly state what should be saved")
- Provides a STRUCTURED FORMAT ([PERSIST]: ...)
- Removes all ambiguity about the task

The action condition:
- Gives vague guidance ("whenever you learn something important")
- Requires Claude to INFER that it should act
- Has no structural scaffolding

**This is not "reasoning vs action" - it's "explicit instruction vs vague suggestion."** The entire premise of the paper is built on this confounded comparison.

### 12. Control Scenarios Exist But Are Not Used

The code defines 25 `CONTROL_SCENARIOS` (negative examples) in `proactive_tools.py` (lines 636-846):
- Simple questions ("What is 2 + 2?")
- Temporary information ("I'm just testing something")
- Chitchat ("Thanks for your help!")

**However, these are never imported or used in the experiments.** The experiment code only imports `MEMORY_SCENARIOS`:

```python
from experiments.scenarios.proactive_tools import (
    MEMORY_SCENARIOS as PROACTIVE_MEMORY_SCENARIOS,
    ...
)
```

The control scenarios would test false positive rates. Without them, we cannot assess precision - only recall. The 96% "reasoning accuracy" could include massive over-triggering.

### 13. Intent Detection is Dangerously Broad

The code uses regex patterns to detect "intent" (lines 37-66):

```python
MEMORY_INTENT_PATTERNS = [
    ...
    r"good to know",           # <- This is casual acknowledgment, not intent!
    r"helpful (context|to know)",  # <- Same issue
    r"I understand",           # <- This is comprehension, not action intent
    r"(this|that) is (important|notable|significant)",  # <- Observation, not intent
    ...
]
```

These patterns would match casual conversational responses as "expressed intent." Claude saying "Good to know!" is NOT the same as "I should save this to memory."

This inflates the reasoning condition's success rate by counting any acknowledgment as "intent."

### 14. Results Depend on Regex Quality

The "success" of the reasoning condition is determined by regex matching for `[PERSIST]:` tags (lines 227-232):

```python
persist_match = re.search(r'\[PERSIST\]:?\s*(.+?)(?:\n\n|$)', response_text, ...)
has_persist_content = persist_match and persist_content.lower() not in ["none", "nothing", ...]
```

If Claude produces `[PERSIST]: the database info` - success!
If Claude writes "This is important to remember but..." without the tag - failure!

**The metric measures format compliance, not reasoning quality.** This is ironic given the paper's argument about format overhead.

### 15. Code Confirms Single-Trial Default

```python
parser.add_argument(
    "--trials",
    type=int,
    default=1,  # <-- Single trial!
    help="Number of trials per scenario (default: 1, recommended for publication: 5)",
)
```

The code acknowledges the problem ("recommended for publication: 5") but the experiment was run with the inadequate default.

### 16. No Randomization or Counterbalancing

The experiments run conditions in fixed order:
1. Natural language trial
2. Identify persist trial
3. Tool trial

Order effects are not controlled. Earlier responses could prime later ones (though separate API calls mitigate this somewhat).

---

## Minor Issues

17. **Code availability placeholder**: "[repository URL]" - unprofessional
18. **No ablation studies**: What if action condition had equal explicitness?
19. **Synthetic scenarios only**: Real user conversations might behave differently
20. **McNemar's test at n=25**: At the edge of validity; consider exact test
21. **Effect size reporting**: Cohen's h=0.73 is computed on unreliable proportions
22. **No discussion of API costs**: Relevant for practical adoption of two-stage

---

## What Would Make This Paper Publishable

### Minimum Requirements:
1. **Fix the confound**: Match explicitness between conditions
2. **Increase sample size**: 100+ scenarios, multiple trials each
3. **Add negative examples**: Include info that should NOT be saved
4. **Test multiple models**: At least GPT-4 and one open model
5. **Remove or shrink Section 6**: Speculation doesn't belong in results

### Stronger Paper:
6. Add ablation studies isolating each variable
7. Test other tool types (search, code execution)
8. Actually implement the two-stage system
9. Provide mechanistic analysis (attention patterns, logit analysis)
10. Pre-register the study to avoid p-hacking concerns

---

## Recommended Action

**Do not submit to arXiv in current form.**

The paper will attract criticism that damages credibility. The core idea is interesting enough that it deserves rigorous validation, not premature publication.

Suggested path:
1. Fix experimental design (2 weeks)
2. Run properly powered study (4-6 weeks)
3. Add ablations and controls (2 weeks)
4. Revise paper with solid results (2 weeks)
5. Then submit

---

## Rating

| Criterion | Score | Notes |
|-----------|-------|-------|
| Novelty | 6/10 | Interesting observation, connects to prior work |
| Methodology | 1/10 | Fatal confound confirmed in code, results likely invalid |
| Clarity | 8/10 | Well-written, good structure |
| Significance | 2/10 | Cannot trust results given methodology |
| Reproducibility | 2/10 | Single trials, placeholder URL, no negative examples tested |
| **Overall** | **2/10** | **Results likely invalid due to confounded design** |

---

## One-Sentence Summary

An interesting hypothesis about reasoning-action separation is **invalidated by a confounded experimental design** where the "reasoning" condition receives explicit instructions while the "action" condition receives vague suggestions - the code confirms this is comparing apples to oranges, not measuring a cognitive gap.

---

## Appendix: The Damning Code Comparison

Side-by-side comparison of what the model is actually told in each condition:

| Aspect | Reasoning Condition | Action Condition |
|--------|--------------------|--------------------|
| **Core instruction** | "explicitly state what should be saved" | "use whenever you learn something important" |
| **Specificity** | Very specific: persistence decision | Vague: "important" is undefined |
| **Format guidance** | "[PERSIST]: content" format provided | No format guidance |
| **Task framing** | "Identify what SHOULD be saved" (reasoning about hypothetical) | "Use this tool" (must commit to irreversible action) |
| **Cognitive load** | Low: just identify | High: decide AND execute |

**The experiment does not control for instruction explicitness.** Any difference in performance can be attributed to this confound rather than a "reasoning-action gap."

A fixed experiment would use:
- **Reasoning**: "What should be saved?" (current)
- **Action**: "Always save codebase info, preferences, and decisions using the memory tool. If in doubt, save it."

This matches explicitness while testing whether the model can execute what it identifies.

---

*Review conducted: 2026-02-01*
*Code verification completed: experiments/natural_language_intent_experiment.py, experiments/scenarios/proactive_tools.py*

---

## Post-Review: Code Fixes Applied

The experimental code has been fixed to address the methodological issues. Key changes:

### 1. Fixed the Confound (CRITICAL)
Both conditions now receive **equally explicit instructions**:

**Reasoning condition** (unchanged):
> "You should save information about: Codebase structure, User preferences, Important decisions, Technical constraints"

**Action condition** (FIXED - was vague, now explicit):
> "You MUST save information about: Codebase structure (category: codebase), User preferences (category: user_preference), Important decisions (category: decision), Technical constraints (category: constraint)"

### 2. Added Negative Examples
Control scenarios are now included by default (`--no-controls` to exclude):
- 25 positive scenarios (should save)
- 25 negative scenarios (should NOT save)
- Enables false positive rate measurement

### 3. Increased Default Trials
Changed from `default=1` to `default=5` trials per scenario per condition.

### 4. Added Randomization
Condition order is now randomized by default (`--no-randomize` to disable).

### 5. Improved Statistical Analysis
- Wilson score confidence intervals
- McNemar's test for paired comparisons
- True positive / false positive rate reporting

### 6. Strict Intent Detection
Removed overly broad patterns like "good to know", "I understand". Now only counts explicit persistence intent.

### Running the Fixed Experiment
```bash
# Full experiment with proper methodology (500 observations)
python -m experiments.natural_language_intent_experiment --trials 5 --seed 42

# Quick test (100 observations)
python -m experiments.natural_language_intent_experiment --trials 1 --seed 42
```

**Important**: The paper's results cannot be trusted until re-run with the fixed code. If the gap disappears with matched explicitness, the original hypothesis is not supported.
