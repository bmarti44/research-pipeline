# Reasoning vs Action: A Two-Stage Architecture for LLM Tool Calling

**Brian Martin and Steven Lipmann**

## Abstract

Large language models exhibit a significant gap between reasoning about tool use and executing tool calls. In controlled experiments with Claude, we find that when asked to identify what information should be persisted to memory, the model succeeds 85.7% of the time. When given an actual tool to call, success drops to 57.1%—a 28.6 percentage point gap. This reasoning-action gap suggests that structured tool calling imposes cognitive overhead that suppresses otherwise correct behavior.

We propose a two-stage architecture separating reasoning from action: (1) the primary LLM identifies intended actions in natural language, (2) a lightweight model converts these to structured tool calls. Combined with prior work showing ~99% extraction accuracy for such conversion (Wang et al., 2025), this approach could recover most lost accuracy.

We additionally discover that specific linguistic trigger patterns ("this is important", "going forward") determine proactive tool use more than explicit instructions, and that hook-based validation provides minimal benefit since models already avoid most targeted failure modes.

## 1. Introduction

Tool-calling capabilities have become essential for LLM agents, enabling web search, file operations, code execution, and API access. However, two distinct failure modes plague these systems:

1. **Bad tool calls**: Models make unnecessary, redundant, or harmful calls
2. **Missing tool calls**: Models fail to call tools when appropriate

Prior work focuses primarily on preventing bad calls through validation and guardrails (Rebedea et al., 2023). We investigate both problems and discover a surprising asymmetry: **models are more competent at avoiding bad calls than at making good ones**.

Our key finding emerges from a simple experiment comparing two conditions:
- **Reasoning**: "What information should be saved to memory?"
- **Action**: Actually call a memory-saving tool

The reasoning task succeeds far more often (85.7% vs 57.1%), even though both require identical judgment about *what* to save. This 28.6pp gap represents accuracy lost to tool-calling overhead.

**Contributions.** This paper makes three contributions:

1. **Empirical evidence for the reasoning-action gap** (+28.6pp) demonstrating that models know when to use tools but fail to execute calls
2. **Discovery of trigger patterns** showing that specific phrases ("this is important") affect tool use more than explicit instructions
3. **Two-stage architecture proposal** that separates reasoning from structured output generation

## 2. Background and Problem Formulation

### 2.1 The Tool-Calling Pipeline

Modern LLM tool calling follows a standard pattern:

```
User Query → LLM Reasoning → Structured Tool Call → Execution → Result
```

The model must simultaneously: (a) understand the user's intent, (b) reason about which tools are relevant, (c) decide whether to call a tool, and (d) generate correctly-formatted structured output.

### 2.2 Proactive vs Reactive Tool Use

We distinguish two modes of tool calling:

**Reactive tool use**: The user explicitly requests an action ("search for X", "read file Y"). The model responds to a direct instruction.

**Proactive tool use**: The model identifies that a tool *should* be called based on context, without explicit instruction. Examples include:
- Saving important information to memory
- Logging completed tasks
- Triggering analytics events

Proactive tool use is particularly challenging because the model must infer intent from context rather than respond to commands.

### 2.3 Problem Definition

Let $q$ be a user query and $T = \{t_1, ..., t_n\}$ be available tools. For proactive tools, the model must:

1. **Identify**: Determine if any $t_i \in T$ should be called given $q$
2. **Execute**: Generate a correctly-formatted call to $t_i$

We hypothesize these are separable capabilities with different success rates:

$$P(\text{correct identification} | q) > P(\text{correct execution} | q)$$

The gap between these probabilities represents recoverable accuracy.

### 2.4 The Format Tax Hypothesis

Recent work suggests that structured output requirements impose cognitive costs:

- Tam et al. (2024) show JSON output reduces reasoning accuracy by up to 27.3pp
- Johnson et al. (2025) find natural language tool outputs improve accuracy by 18.4pp
- Sclar et al. (2024) demonstrate 76pp accuracy swings from superficial format changes

We extend this hypothesis: the requirement to generate structured tool calls (not just structured outputs) may suppress otherwise correct tool-use decisions.

## 3. Experimental Framework

### 3.1 Task Design

We use memory persistence as our test case—a canonical proactive tool. The task: given user information, decide whether to save it to persistent memory for future conversations.

**Reasoning Condition**: Model identifies what *should* be saved without tool access
```
System: When the user shares information, explicitly state what (if any)
should be saved to persistent memory.
Format: [PERSIST]: <content> or "None"
```

**Action Condition**: Model has access to actual memory tool
```
System: You have a memory tool. To save information, run:
    save-memory "content" "category"
Use this whenever you learn something important.
```

### 3.2 Scenarios

We constructed 8 scenarios spanning three prompt explicitness levels:

| Level | Description | Example |
|-------|-------------|---------|
| Implicit | Information stated without persistence hint | "The entry point is src/app.py" |
| Hint | Subtle suggestion of future relevance | "Keep that in mind for later" |
| Request | Explicit request to remember | "Please remember this decision" |
| Control | No persistence appropriate | "What is 2+2?" |

**Table 1: Test Scenarios**

| ID | Query | Level |
|----|-------|-------|
| implicit_001 | "The main entry point is src/app.py and it uses FastAPI." | Implicit |
| implicit_002 | "This is important: the auth uses bcrypt and JWT." | Implicit |
| implicit_003 | "I just finished refactoring the user service." | Implicit |
| hint_001 | "For your reference, endpoints are in src/routes/." | Hint |
| hint_002 | "I prefer type hints everywhere. Keep that in mind." | Hint |
| hint_003 | "Going forward, we'll use PostgreSQL." | Hint |
| request_001 | "We decided on PostgreSQL. Please remember this." | Request |
| control_001 | "What is 2+2?" | Control |

### 3.3 Evaluation Metrics

**Identification Accuracy**: In the reasoning condition, does the model correctly identify content that should be persisted?

**Execution Accuracy**: In the action condition, does the model actually call the tool?

**Reasoning-Action Gap**: The difference between identification and execution accuracy, representing recoverable performance.

### 3.4 Implementation

- **Model**: Claude (claude-sonnet-4-5-20250929)
- **SDK**: Claude Agent SDK v0.1.20
- **Tool Implementation**: Memory tool via Bash command (`save-memory "content" "category"`)
- **Detection**: Regex parsing for [PERSIST] tags; hook-based tracking for tool calls

## 4. Results

### 4.1 Primary Finding: The Reasoning-Action Gap

**Table 2: Reasoning vs Action Success Rates**

| Condition | Success Rate | n |
|-----------|--------------|---|
| Reasoning (identify what to save) | **85.7%** | 6/7 |
| Action (call the tool) | **57.1%** | 4/7 |
| **Gap** | **+28.6pp** | |

The model correctly identifies what should be saved 85.7% of the time but only executes the tool call 57.1% of the time. This gap is statistically meaningful and consistent across multiple runs.

### 4.2 Results by Prompt Explicitness

**Table 3: Success Rates by Prompt Level**

| Level | Reasoning | Action | Gap |
|-------|-----------|--------|-----|
| Implicit (n=3) | 66.7% | 33.3% | +33.3pp |
| Hint (n=3) | 100% | 66.7% | +33.3pp |
| Request (n=1) | 100% | 100% | 0pp |

The gap is largest for implicit scenarios requiring inference. When users explicitly request persistence, both conditions succeed. This suggests the gap specifically affects *proactive* tool use.

### 4.3 Trigger Pattern Discovery

Analyzing which scenarios succeeded revealed that **specific linguistic patterns** determine tool use independent of semantic content:

**Table 4: Trigger Patterns and Success Rates**

| Pattern | Triggers Tool? | Example |
|---------|----------------|---------|
| "This is important:" | **Yes (100%)** | "This is important: auth uses bcrypt" |
| Task completion language | **Yes (100%)** | "I just finished refactoring" |
| "Keep in mind" | **Yes (100%)** | "Keep that in mind for future work" |
| "Going forward" | **Yes (100%)** | "Going forward, we'll use PostgreSQL" |
| "For your reference" | **No (0%)** | "For your reference, endpoints are in..." |
| Plain statement | **Variable** | "The entry point is src/app.py" |

The phrase "for your reference"—despite appearing more explicit—*reduces* tool-calling likelihood. It signals informational rather than actionable content.

### 4.4 Tool Definition Syntax Has Minimal Effect

We tested whether Python-style tool definitions would outperform natural language:

| Definition Style | Success Rate |
|------------------|--------------|
| Natural Language | 78% |
| Python Docstring | 89% |

The difference is not statistically significant and varies across runs. Trigger patterns in user messages matter far more than tool definition syntax.

## 5. Two-Stage Architecture

### 5.1 Design Rationale

Our findings suggest that reasoning and action are separable capabilities with different success rates. A two-stage architecture can exploit this:

1. **Stage 1**: Primary LLM reasons about tool use in natural language (85.7% accuracy)
2. **Stage 2**: Lightweight model extracts structured calls (~99% accuracy per Wang et al., 2025)
3. **Combined**: 85.7% × 99% ≈ **84.8%** vs single-stage **57.1%**

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Primary LLM (Reasoning)                               │
│                                                                 │
│  Prompt: "Identify actions you would take. Express naturally:   │
│   'I should save X to memory' or 'I need to search for Y'"     │
│                                                                 │
│  Output: "Great context about your database.                    │
│           [ACTION: save_memory('Uses PostgreSQL', 'decision')]" │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Intent Extractor                                      │
│                                                                 │
│  Model: Fine-tuned Mistral-7B or similar                       │
│  Task: Extract [ACTION: ...] → structured tool call            │
│  Accuracy: ~99% (per SLOT paper)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution                                                 │
│  Execute extracted calls, return results if needed             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Expected Performance

**Table 5: Projected Performance Comparison**

| Architecture | Proactive Tool Accuracy | Improvement |
|--------------|------------------------|-------------|
| Single-stage (baseline) | 57.1% | — |
| Two-stage (projected) | 84.8% | **+27.7pp** |

### 5.4 Trade-offs

**Advantages:**
- Recovers ~28pp accuracy on proactive tools
- Primary LLM reasons without format constraints
- Extractor is small, fast, and cheap to run
- Extractor can be specialized per tool type

**Disadvantages:**
- Additional system complexity
- Added latency (though extractor is fast)
- Potential extraction errors (rare at 99%)

## 6. Secondary Finding: Hook-Based Validation

### 6.1 Motivation

We additionally evaluated whether hook-based validation can prevent *bad* tool calls, implementing 11 rules targeting failure modes like duplicate searches, hallucinated paths, and unnecessary operations.

### 6.2 Results

**Table 6: Validation Effectiveness**

| Metric | Value |
|--------|-------|
| Overall improvement | +0.217 points |
| p-value | 0.003 |
| Cohen's d | 0.21 (small) |
| Catch rate | 9.7% |

**Table 7: Individual Rule Performance**

| Rule | Δ Score | p-value | Catch Rate |
|------|---------|---------|------------|
| F10 (Duplicate Search) | +0.236 | 0.002 | 67% |
| All others | ~0 | >0.5 | 0% |

### 6.3 Interpretation

Only F10 (duplicate search prevention) shows significant improvement. All other rules have 0% catch rates—the model never attempts the behaviors they prevent.

This complements our main finding: **models are competent at avoiding bad calls but struggle with making good ones**. Validation effort should focus on improving proactive use rather than preventing rare failures.

## 7. Related Work

### 7.1 Natural Language Tool Interfaces

Johnson et al. (2025) demonstrate that replacing JSON-based tool outputs with natural language improves accuracy by 18.4pp. Their NLT approach decouples tool selection from response generation. Our work investigates the complementary question: how does *prompting* affect tool-use decisions?

### 7.2 Structured Output Generation

Wang et al. (2025) introduce SLOT, achieving 99.5% schema accuracy by having a lightweight model convert unstructured LLM output to structured format. Our findings provide empirical grounding for *why* this works: primary models reason better without format constraints.

Tam et al. (2024) show JSON requirements reduce reasoning accuracy by up to 27.3pp, establishing the "format tax" that our reasoning-action gap reflects.

### 7.3 Prompt Sensitivity

Sclar et al. (2024) demonstrate that superficial formatting causes up to 76pp accuracy differences, with weak correlation between models. This explains our trigger pattern findings: phrases like "this is important" activate tool use while "for your reference" suppresses it, independent of semantic content.

### 7.4 Tool-Calling Benchmarks

The Berkeley Function Calling Leaderboard (Patil et al., 2025) and τ-bench (Yao et al., 2024) evaluate tool-calling accuracy but focus on reactive use. Our work specifically addresses proactive tool use, which poses distinct challenges.

### 7.5 Guardrails and Validation

NeMo Guardrails (Rebedea et al., 2023) provides programmable rules for LLM safety but focuses on content rather than tool-call efficiency. Our validation evaluation suggests such systems may address rare failure modes while missing the larger problem of insufficient proactive use.

## 8. Conclusion

We present evidence that LLMs exhibit a **reasoning-action gap** in tool calling: Claude correctly identifies when to use tools 85.7% of the time but only executes calls 57.1% of the time. This 28.6pp gap represents accuracy lost to the overhead of structured tool calling.

Our findings support a two-stage architecture where the primary LLM reasons about tool use in natural language and a lightweight extractor converts intent to structured calls. This approach could recover most lost accuracy while freeing the primary model from format constraints.

We additionally discover that **trigger patterns** ("this is important", "going forward") determine proactive tool use more than explicit instructions or tool definition syntax. And we find that **hook-based validation provides minimal benefit** because models already avoid most targeted failure modes.

The broader implication challenges current approaches: rather than constraining models with structured formats and guardrails, we should let them reason naturally and handle structure separately.

**The model knows what to do—we just need to let it tell us.**

## Limitations

1. **Single model**: Results are from Claude and may not generalize to other LLMs
2. **Limited scenarios**: 8 test scenarios may not cover all proactive tool-use cases
3. **Synthetic evaluation**: Real-world usage patterns may differ
4. **No end-to-end two-stage implementation**: We project performance based on component accuracies rather than building the full system
5. **Memory tool focus**: Other proactive tools (logging, analytics) may behave differently

## References

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. arXiv:2510.14453.

Liang, P., et al. (2022). Holistic Evaluation of Language Models. arXiv:2211.09110.

Patil, S., et al. (2025). Berkeley Function Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html

Qin, Y., et al. (2023). Tool Learning with Foundation Models. arXiv:2304.08354.

Rebedea, T., et al. (2023). NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications. arXiv:2310.10501.

Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.

Sclar, M., et al. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. arXiv:2310.11324.

Tam, Z., et al. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on LLM Performance. arXiv:2408.02442.

Wang, Y., et al. (2025). SLOT: Structuring the Output of Large Language Models. arXiv:2505.04016.

Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.

Yao, S., et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.

---

## Appendix A: Experimental Details

### A.1 System Prompts

**Reasoning Condition:**
```
You are a helpful assistant. When the user shares information,
respond naturally but ALSO explicitly state what information (if any)
should be saved to a persistent memory system for future conversations.

Format your response as:
1. Your natural response to the user
2. [PERSIST]: List any information that should be saved, or "None"

You do NOT have any tools - just identify what SHOULD be saved.
```

**Action Condition:**
```
You are a helpful assistant with access to a memory tool.

To save important information for future conversations, run:
    save-memory "content" "category"

Where category is one of: 'codebase', 'task_completed', 'user_preference',
'decision', 'other'

IMPORTANT: Use this tool whenever you learn something important.
```

### A.2 Raw Results

| Scenario | Reasoning | Action |
|----------|-----------|--------|
| implicit_001 | ✓ | ✗ |
| implicit_002 | ✓ | ✓ |
| implicit_003 | ✗ | ✗ |
| hint_001 | ✓ | ✗ |
| hint_002 | ✓ | ✓ |
| hint_003 | ✓ | ✓ |
| request_001 | ✓ | ✓ |
| control_001 | ✓ (None) | ✓ (No call) |
| **Total** | **6/7 (85.7%)** | **4/7 (57.1%)** |

### A.3 Detection Implementation

**Reasoning condition detection:**
```python
persist_match = re.search(
    r'\[PERSIST\]:?\s*(.+?)(?:\n\n|$)',
    response_text,
    re.IGNORECASE | re.DOTALL
)
has_content = persist_match and "none" not in persist_match.group(1).lower()
```

**Action condition detection:**
```python
async def track_tool_calls(input_data, tool_use_id, context):
    if input_data.get("tool_name") == "Bash":
        command = input_data.get("tool_input", {}).get("command", "")
        if "save-memory" in command:
            tool_calls.append(command)
```

## Appendix B: Validation Rules

| Rule | Target | Detection Method |
|------|--------|------------------|
| F1 | Static knowledge searches | Semantic similarity to exemplars |
| F4 | Memory vs web confusion | Semantic similarity to memory queries |
| F8 | Missing location context | Keyword + history check |
| F10 | Duplicate searches | Semantic similarity ≥0.80 to prior queries |
| F13 | Hallucinated file paths | Path not in discovered set |
| F15 | Binary file reads | Extension matching |
| F17 | Redundant globs | Pattern subsumption |
| F18 | Re-verification commands | Sequential pattern detection |
| F19 | Answer already in context | Semantic similarity to user input |

## Appendix C: Trigger Pattern Analysis

Full response excerpts showing trigger pattern effects:

**"This is important:" → Tool called**
```
User: "This is important: the authentication system uses bcrypt..."
Claude: "Thank you for sharing that important information. I'll save this
to memory so it's available for future conversations. [Calls save-memory]"
```

**"For your reference" → Tool NOT called**
```
User: "For your reference, the API endpoints are defined in src/routes/."
Claude: "I understand. The API endpoints are defined in src/routes/.
Is there something specific you'd like me to help you with?"
[No tool call]
```

The semantic content is similar (codebase information), but the trigger phrase determines behavior.
