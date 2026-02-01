# Paper Expansion Plan: Reasoning vs Action Gap

## Target Venue
- **Primary**: EMNLP 2025, ACL 2025, or NeurIPS 2025 (depending on timing)
- **Backup**: COLM 2025, ICLR 2026
- **Workshop**: NeurIPS Workshop on Foundation Models, ICML Workshop on LLM Agents

## Current State vs Requirements

| Aspect | Current | Required | Gap |
|--------|---------|----------|-----|
| Scenarios | 8 | 100+ | Need 90+ more |
| Models | 1 (Claude) | 4+ | Need GPT-4, Llama, Gemini |
| Tool types | 1 (memory) | 4+ | Need search, logging, file ops |
| Statistical tests | None | Full suite | Need power analysis, CI |
| Two-stage impl | Projection | Working system | Need to build it |
| Baselines | 1 | 3+ | Need comparisons |

---

## Phase 1: Scenario Expansion (Week 1-2)

### 1.1 Proactive Tool Types

**Memory/Persistence** (current)
- Save important information for future conversations

**Task Logging**
- Log completed actions, errors, milestones
- "I fixed the bug in auth.py" → should log

**Analytics/Telemetry**
- Track user preferences, usage patterns
- "I always use dark mode" → should record

**Proactive Search**
- Search when information would help (not explicitly asked)
- "I'm working on React 19" → could proactively fetch docs

**File Bookmarking**
- Mark files as important for later reference
- "This config file is crucial" → should bookmark

### 1.2 Scenario Categories (25 per tool type = 100 total)

For each tool type, create scenarios at 5 levels × 5 variations:

| Level | Description | Count |
|-------|-------------|-------|
| Implicit | No hint of tool need | 5 |
| Weak hint | Subtle future reference | 5 |
| Moderate hint | Clear but indirect | 5 |
| Strong hint | Direct suggestion | 5 |
| Explicit request | Direct command | 5 |

### 1.3 Trigger Pattern Variations

Test specific patterns systematically:
- "This is important: X" vs "X" vs "FYI: X" vs "Note: X"
- "Going forward..." vs "In the future..." vs "Later..."
- "Keep in mind..." vs "Remember..." vs "For reference..."
- Task completion: "Done with X" vs "Finished X" vs "Completed X"

### 1.4 Control Scenarios (25 total)
Scenarios where NO tool should be called:
- Simple questions
- Chitchat
- Already-known information
- Temporary/ephemeral content

---

## Phase 2: Multi-Model Evaluation (Week 2-3)

### 2.1 Models to Test

| Model | API | Priority |
|-------|-----|----------|
| Claude 3.5 Sonnet | Anthropic | ✓ Have |
| GPT-4o | OpenAI | High |
| GPT-4o-mini | OpenAI | Medium |
| Llama 3.1 70B | Together/Replicate | High |
| Llama 3.1 8B | Local | Medium |
| Gemini 1.5 Pro | Google | High |
| Mistral Large | Mistral | Medium |

### 2.2 Standardized Evaluation Harness

```python
class ToolCallingEvaluator:
    def __init__(self, model_name: str):
        self.model = load_model(model_name)

    async def evaluate_reasoning(self, scenario: Scenario) -> ReasoningResult:
        """Condition 1: Identify what should be done"""
        pass

    async def evaluate_action(self, scenario: Scenario) -> ActionResult:
        """Condition 2: Actually call the tool"""
        pass

    def compute_gap(self, reasoning: list, action: list) -> GapMetrics:
        """Compute reasoning-action gap with CI"""
        pass
```

### 2.3 API Cost Estimation

| Model | Cost/1K tokens | Scenarios | Estimated Cost |
|-------|---------------|-----------|----------------|
| Claude 3.5 | $0.003/$0.015 | 250 | ~$20 |
| GPT-4o | $0.005/$0.015 | 250 | ~$25 |
| GPT-4o-mini | $0.00015/$0.0006 | 250 | ~$2 |
| Gemini 1.5 | $0.00125/$0.005 | 250 | ~$10 |
| Llama (API) | $0.001/$0.001 | 250 | ~$5 |
| **Total** | | | **~$60-100** |

---

## Phase 3: Two-Stage Implementation (Week 3-4)

### 3.1 Stage 1: Intent Elicitation Prompts

Test multiple prompt strategies:

**Strategy A: Explicit tagging**
```
When responding, tag any actions you would take:
[SAVE: content] for memory
[SEARCH: query] for web search
[LOG: event] for logging
```

**Strategy B: Natural expression**
```
Express any actions naturally in your response, e.g.,
"I should save this to memory" or "Let me search for that"
```

**Strategy C: Structured section**
```
End your response with:
## Actions
- save_memory("content", "category")
- search_web("query")
```

### 3.2 Stage 2: Intent Extractor

**Option A: Fine-tuned small model**
- Base: Mistral-7B or Llama-3.1-8B
- Training data: 10K synthetic intent → tool call pairs
- Expected accuracy: 95-99%

**Option B: Few-shot extraction**
- Use GPT-4o-mini with few-shot examples
- No fine-tuning needed
- Expected accuracy: 90-95%

**Option C: Rule-based extraction**
- Regex patterns for [TAG: content] format
- Fast, no API cost
- Expected accuracy: 85-95% (depends on format compliance)

### 3.3 End-to-End Evaluation

Compare:
1. **Baseline**: Single-stage tool calling
2. **Two-stage A**: Intent elicitation + fine-tuned extractor
3. **Two-stage B**: Intent elicitation + few-shot extractor
4. **Two-stage C**: Intent elicitation + rule-based extractor

---

## Phase 4: Statistical Rigor (Throughout)

### 4.1 Power Analysis

For detecting 20pp difference (our observed 28.6pp) with:
- α = 0.05, power = 0.80
- Required n ≈ 50 per condition

For detecting 10pp difference:
- Required n ≈ 200 per condition

### 4.2 Statistical Tests

| Comparison | Test | Justification |
|------------|------|---------------|
| Reasoning vs Action | McNemar's test | Paired binary outcomes |
| Across models | Cochran's Q | Multiple related samples |
| Effect size | Cohen's h | Binary proportions |
| Confidence intervals | Wilson score | Better for proportions |

### 4.3 Multiple Comparisons Correction
- Bonferroni for primary hypotheses
- FDR (Benjamini-Hochberg) for exploratory analyses

---

## Phase 5: Additional Experiments (Week 4-5)

### 5.1 Ablation Studies

**Prompt ablations:**
- With vs without trigger phrases in system prompt
- Different instruction verbosity levels
- With vs without examples

**Tool ablations:**
- Single tool vs multiple tools available
- Tool description length
- Tool naming conventions

### 5.2 Robustness Tests

- Temperature variation (0, 0.3, 0.7, 1.0)
- Prompt paraphrasing (3 versions per scenario)
- Order effects (randomized scenario presentation)

### 5.3 Error Analysis

Categorize failures:
- False negatives: Should have called, didn't
- False positives: Shouldn't have called, did
- Wrong tool: Called wrong tool
- Wrong parameters: Right tool, wrong arguments

---

## Phase 6: Paper Writing (Week 5-6)

### 6.1 Expanded Sections

| Section | Current | Target |
|---------|---------|--------|
| Abstract | 150 words | 250 words |
| Introduction | 400 words | 800 words |
| Background | 400 words | 600 words |
| Methods | 500 words | 1000 words |
| Results | 600 words | 1500 words |
| Analysis | 200 words | 800 words |
| Related Work | 400 words | 600 words |
| Conclusion | 200 words | 400 words |
| **Total** | ~3000 words | ~6000 words |

### 6.2 New Figures

1. **Main result figure**: Reasoning vs Action gap across models (bar chart)
2. **Per-tool breakdown**: Gap by tool type (grouped bar)
3. **Trigger patterns heatmap**: Pattern × Model success rates
4. **Two-stage architecture diagram**: Polished version
5. **Error analysis**: Confusion matrix or Sankey diagram

### 6.3 New Tables

1. Full scenario inventory (appendix)
2. Per-model results with confidence intervals
3. Ablation results
4. Two-stage vs single-stage comparison
5. Statistical test results

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Scenario expansion | 100+ scenarios, validation framework |
| 2 | Multi-model setup | API integrations, evaluation harness |
| 3 | Run experiments | All models × all scenarios |
| 4 | Two-stage implementation | Working system, comparison results |
| 5 | Analysis & writing | Statistical tests, figures, draft |
| 6 | Polish & submit | Final paper, code release |

---

## Success Criteria

**Minimum viable paper:**
- [ ] 100+ scenarios across 4 tool types
- [ ] 3+ models evaluated
- [ ] Statistically significant gap (p < 0.01)
- [ ] Two-stage implementation showing improvement
- [ ] Proper confidence intervals on all metrics

**Strong paper:**
- [ ] 200+ scenarios
- [ ] 5+ models including open-source
- [ ] Ablation studies
- [ ] Error analysis
- [ ] Code and data release

---

## Immediate Next Steps

1. **Today**: Design scenario generation framework
2. **Tomorrow**: Generate 100 scenarios with GPT-4 assistance
3. **Day 3**: Set up multi-model evaluation harness
4. **Day 4-5**: Run Claude experiments on full scenario set
5. **Day 6-7**: Add GPT-4 and Llama evaluations

Start with scenario generation?
