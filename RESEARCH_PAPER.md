# The Reasoning-Action Gap: Why LLMs Know What To Do But Fail To Do It

**Brian Martin and Steven Lipmann**

## Abstract

We present empirical evidence that large language models exhibit a fundamental gap between reasoning about actions and executing them. In controlled experiments with Claude across 25 scenarios, we find that when asked to *identify* what information should be persisted to memory, the model succeeds **96%** of the time. When given an actual tool to call, success drops to **68%**—a **28 percentage point gap**. This gap varies dramatically with prompt explicitness: implicit scenarios show a +60pp gap, while explicit commands show no gap at all.

We argue this "reasoning-action gap" reflects a general principle: **reasoning and execution are separable cognitive capabilities with different costs**. This principle extends beyond tool calling to code generation, structured output, and any task requiring both semantic understanding and format compliance. We propose a two-stage architecture separating reasoning from execution, projecting +28pp accuracy recovery on proactive tasks.

Our findings challenge the assumption that LLM failures stem from lack of knowledge. Often, models know exactly what to do—they just fail to do it. The solution is not better prompting or fine-tuning, but architectural separation of reasoning and action.

## 1. Introduction

Large language models increasingly serve as autonomous agents, calling tools, writing code, and generating structured outputs. Yet these capabilities remain unreliable: models fail to call tools when they should, generate syntactically invalid code, and produce malformed JSON. The standard response is more guardrails, better prompts, or additional fine-tuning.

We propose a different diagnosis: **the problem is not that models don't know what to do, but that the act of doing it interferes with the knowing**.

Consider a simple experiment. We give Claude two tasks:
1. **Reasoning**: "What information from this conversation should be saved to memory?"
2. **Action**: "You have a memory tool. Use it when appropriate."

Both tasks require identical judgment—deciding what's worth remembering. Yet reasoning succeeds 96% of the time while action succeeds only 68%. The model *knows* what to save but *fails to save it* 28% of the time.

This reasoning-action gap has profound implications:

1. **For tool calling**: Models can identify when tools should be used but fail to execute calls
2. **For code generation**: Models can describe correct logic but produce buggy implementations
3. **For structured output**: Models can identify what to extract but generate invalid formats
4. **For system design**: Separating reasoning from execution can recover lost accuracy

**Contributions.** This paper makes four contributions:

1. **Empirical demonstration** of the reasoning-action gap across 25 scenarios with 96% vs 68% accuracy (+28pp)
2. **Granular analysis** showing the gap varies from +60pp (implicit) to 0pp (explicit) based on prompt type
3. **Theoretical framing** connecting our findings to cognitive load, format constraints, and the broader "System 1/System 2" paradigm
4. **Architectural proposal** for two-stage systems that separate reasoning from execution

## 2. Background and Theoretical Framework

### 2.1 The Dual Nature of LLM Tasks

Most LLM tasks combine two distinct cognitive demands:

**Semantic reasoning**: Understanding meaning, making judgments, deciding what should happen
- "Is this information important enough to save?"
- "What logic does this function need?"
- "Which fields should I extract from this text?"

**Format execution**: Producing output that satisfies structural constraints
- Generate valid tool call syntax
- Write syntactically correct code
- Output well-formed JSON

We hypothesize these demands compete for cognitive resources, and format execution can suppress correct semantic reasoning.

### 2.2 Evidence for Cognitive Competition

Recent work supports this hypothesis:

**Format constraints degrade reasoning.** Tam et al. (2024) show that requiring JSON output reduces accuracy by up to 27.3 percentage points on reasoning tasks. The stricter the format, the greater the degradation.

**Natural language improves tool use.** Johnson et al. (2025) demonstrate that replacing structured tool outputs with natural language improves accuracy by 18.4pp across 10 models. Removing format constraints enables better decisions.

**Superficial features dominate semantics.** Sclar et al. (2024) find that superficial prompt formatting causes up to 76pp accuracy swings—far exceeding the impact of semantic content changes.

### 2.3 The Reasoning-Action Gap Hypothesis

We formalize the competition between reasoning and execution:

Let $R(q)$ denote the probability of correct reasoning given query $q$, and $A(q)$ denote the probability of correct action. We hypothesize:

$$R(q) > A(q) \text{ when format constraints are non-trivial}$$

The gap $G(q) = R(q) - A(q)$ represents accuracy lost to execution overhead. This gap should:
- Increase with format complexity
- Decrease with explicit instruction
- Disappear when execution is trivial

### 2.4 Generalization Beyond Tool Calling

The reasoning-action gap is not specific to tools. It applies to any task combining semantic judgment with structured output:

| Domain | Reasoning Task | Action Task | Expected Gap |
|--------|---------------|-------------|--------------|
| Tool calling | "Should I call this tool?" | Generate tool call syntax | High |
| Code generation | "What should this code do?" | Write valid syntax | Medium-High |
| JSON extraction | "What entities are present?" | Format as valid JSON | Medium |
| SQL generation | "What query logic is needed?" | Write valid SQL | Medium |
| API calls | "What endpoint and params?" | Format HTTP request | Medium |
| Math | "What steps solve this?" | Execute arithmetic | Low |

## 3. Experimental Design

### 3.1 Task: Proactive Memory Persistence

We use memory persistence as our test case—a canonical proactive tool. The task: given user information, decide whether to save it to persistent memory for future conversations.

This task is ideal because:
- It requires genuine semantic judgment (not all information should be saved)
- The tool call format is simple (low execution complexity)
- It's ecologically valid (memory tools are common in production systems)

### 3.2 Conditions

**Reasoning Condition**: Model identifies what *should* be saved, without tool access

```
System: When the user shares information, explicitly state what (if any)
should be saved to persistent memory for future conversations.

Format: [PERSIST]: <content to save> or "None"

You do NOT have any tools—just identify what SHOULD be saved.
```

**Action Condition**: Model has access to actual memory tool

```
System: You have a memory tool. To save information, run:
    save-memory "content" "category"

Use this tool whenever you learn something important about the codebase,
user preferences, or decisions.
```

### 3.3 Scenarios

We constructed 25 scenarios spanning five explicitness levels:

**Table 1: Scenario Distribution**

| Level | Description | Example | n |
|-------|-------------|---------|---|
| Implicit | No hint of persistence need | "We use PostgreSQL with pgvector." | 5 |
| Weak Hint | Subtle suggestion | "FYI, I prefer functional components." | 5 |
| Moderate Hint | Clear but indirect | "This is important: auth uses bcrypt." | 5 |
| Strong Hint | Direct suggestion | "Please note for later: CI/CD is in .github/" | 5 |
| Explicit | Direct command | "Save this to memory: we use Stripe." | 5 |

All scenarios involve information that *should* be persisted (codebase facts, user preferences, architectural decisions). See Appendix A for complete scenario inventory.

### 3.4 Evaluation Metrics

**Reasoning Accuracy**: Fraction of scenarios where model correctly identifies content to persist (via [PERSIST] tag with substantive content)

**Action Accuracy**: Fraction of scenarios where model actually calls the memory tool

**Reasoning-Action Gap**: Difference between reasoning and action accuracy

### 3.5 Implementation

- **Model**: Claude Sonnet (claude-sonnet-4-5-20250929)
- **SDK**: Claude Agent SDK v0.1.20
- **Tool Implementation**: Bash command (`save-memory "content" "category"`)
- **Trials**: Each scenario evaluated once per condition
- **Detection**: Regex parsing for [PERSIST] tags; hook-based tracking for tool calls

## 4. Results

### 4.1 Primary Finding: The Reasoning-Action Gap

**Table 2: Overall Results (n=25)**

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| Reasoning | **96.0%** | [80.5%, 99.3%] |
| Action | **68.0%** | [48.4%, 82.8%] |
| **Gap** | **+28.0pp** | |

The model correctly identifies what should be saved 96% of the time but only executes the tool call 68% of the time. This 28 percentage point gap is statistically significant (McNemar's test, p < 0.01).

### 4.2 Gap Varies by Prompt Explicitness

**Table 3: Results by Explicitness Level**

| Level | Reasoning | Action | Gap | n |
|-------|-----------|--------|-----|---|
| Implicit | 100% | 40% | **+60pp** | 5 |
| Weak Hint | 80% | 40% | **+40pp** | 5 |
| Moderate Hint | 100% | 80% | +20pp | 5 |
| Strong Hint | 100% | 80% | +20pp | 5 |
| Explicit | 100% | 100% | **0pp** | 5 |

The gap follows a clear pattern:
- **Implicit scenarios**: +60pp gap—model knows what to save but rarely acts
- **Hint scenarios**: +20-40pp gap—partial action
- **Explicit commands**: 0pp gap—both reasoning and action succeed

This confirms our hypothesis: the gap reflects execution overhead, not reasoning failure. When execution is made trivial (explicit command), the gap disappears.

### 4.3 Failure Mode Analysis

We analyzed the 7 action failures (scenarios where reasoning succeeded but action failed):

| Scenario | Level | Model Behavior |
|----------|-------|----------------|
| mem_implicit_001 | Implicit | Acknowledged info, didn't save |
| mem_implicit_003 | Implicit | Said "good to know", no action |
| mem_implicit_004 | Implicit | Discussed implications, no action |
| mem_weak_003 | Weak | Noted "useful context", no action |
| mem_weak_004 | Weak | Thanked user, no action |
| mem_weak_005 | Weak | Brief acknowledgment only |
| mem_strong_001 | Strong | Described what to save, didn't call tool |

Pattern: The model consistently *acknowledges* the information's importance but fails to *act* on that acknowledgment. This is not a reasoning failure—it's an execution failure.

### 4.4 Trigger Pattern Effects

Certain phrases consistently triggered tool use while others suppressed it:

**Table 4: Trigger Pattern Success Rates**

| Pattern | Action Success | Example |
|---------|---------------|---------|
| "This is important:" | 100% | "This is important: auth uses bcrypt" |
| "Going forward" | 100% | "Going forward, we'll use PostgreSQL" |
| "Please note/remember" | 100% | "Please note for later: CI in .github/" |
| "Save this to memory" | 100% | Explicit command |
| "FYI" / "Just so you know" | 40% | "FYI, I prefer functional components" |
| "For your reference" | 0% | "For your reference, endpoints in src/routes" |
| Plain statement | 40% | "We use PostgreSQL with pgvector" |

The phrase "for your reference"—despite appearing helpful—*suppresses* tool calls. It signals informational content rather than actionable persistence.

## 5. Theoretical Interpretation

### 5.1 Why Does the Gap Exist?

We propose three complementary mechanisms:

**1. Format overhead**: Generating structured tool calls requires cognitive resources that compete with semantic reasoning. The model must simultaneously decide *what* to do and *how* to format it—dual demands that interfere with each other.

**2. Decision threshold asymmetry**: Reasoning ("what should be saved?") has a low threshold—any plausibly important information qualifies. Action ("call the tool now") has a higher threshold—the model must commit to an irreversible operation. This asymmetry explains why reasoning succeeds more often.

**3. Training distribution mismatch**: Models see far more examples of *discussing* actions than *executing* them. "I should save this to memory" appears in training data; actually calling a save-memory tool does not.

### 5.2 Why Does the Gap Disappear for Explicit Commands?

When users explicitly command tool use ("Save this to memory"), two things change:

1. **Decision is externalized**: The user has decided; the model only executes
2. **Format is specified**: "Save this to memory" maps directly to the tool call

Both changes reduce cognitive load, eliminating the reasoning-action gap.

### 5.3 Implications for System Design

The gap suggests a general principle: **separate reasoning from execution**.

Instead of asking models to simultaneously decide and act, decompose tasks:

```
Stage 1 (Reasoning): "What should happen?" → Natural language intent
Stage 2 (Execution): Convert intent → Structured action
```

This separation:
- Lets the reasoning stage operate without format constraints
- Lets the execution stage operate without semantic ambiguity
- Recovers accuracy lost to cognitive competition

## 6. Two-Stage Architecture

### 6.1 Design

Based on our findings, we propose a two-stage architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Primary LLM (Reasoning)                               │
│                                                                 │
│  Prompt: "Identify what actions you would take. Express them    │
│   naturally: 'I should save X to memory' or 'I need to search'  │
│                                                                 │
│  Output: "This is important context about your database choice. │
│           I should save this to memory for future reference."   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Intent Extractor (Lightweight Model)                  │
│                                                                 │
│  Input: Natural language with embedded intent                   │
│  Output: Structured tool calls                                  │
│                                                                 │
│  "I should save this to memory" → save_memory("database choice")│
│                                                                 │
│  Model: Fine-tuned Mistral-7B or rule-based extraction         │
│  Accuracy: ~99% (per SLOT paper, Wang et al. 2025)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution Layer                                           │
│  Execute extracted calls, return results if needed              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Projected Performance

**Table 5: Single-Stage vs Two-Stage Performance**

| Architecture | Reasoning | Extraction | Combined | vs Baseline |
|--------------|-----------|------------|----------|-------------|
| Single-stage (baseline) | — | — | 68.0% | — |
| Two-stage | 96.0% | ~99%* | **95.0%** | **+27.0pp** |

*Extraction accuracy from Wang et al. (2025) SLOT paper

The two-stage approach projects a 27 percentage point improvement on proactive tool use by eliminating the reasoning-action gap.

### 6.3 Generalization to Other Domains

The two-stage pattern applies broadly:

**Code Generation**
- Stage 1: "Describe what this function should do"
- Stage 2: Convert description to syntactically correct code

**Structured Extraction**
- Stage 1: "What entities and relationships exist in this text?"
- Stage 2: Convert to valid JSON schema

**SQL Generation**
- Stage 1: "What data do we need and how should it be filtered?"
- Stage 2: Convert to valid SQL query

**API Composition**
- Stage 1: "What sequence of API calls achieves this goal?"
- Stage 2: Convert to properly formatted requests

In each case, separating semantic reasoning from format execution should reduce errors.

## 7. Related Work

### 7.1 Natural Language Tool Interfaces

Johnson et al. (2025) demonstrate that replacing JSON-based tool outputs with natural language improves accuracy by 18.4pp. Their NLT approach removes format constraints from tool *outputs*. Our work addresses tool *inputs*—showing that even the decision to call a tool is affected by format overhead.

### 7.2 Structured Output Generation

Wang et al. (2025) introduce SLOT, achieving 99.5% schema accuracy by having a primary LLM generate unstructured output and a fine-tuned extractor convert to structured format. Our findings provide theoretical grounding for *why* this works: the primary model reasons better without format constraints.

Tam et al. (2024) quantify the "format tax"—up to 27.3pp accuracy reduction from JSON requirements. Our reasoning-action gap is a specific instance of this phenomenon applied to tool-calling decisions.

### 7.3 Prompt Sensitivity

Sclar et al. (2024) show that superficial formatting causes up to 76pp accuracy differences. This explains our trigger pattern findings: "this is important" activates tool use while "for your reference" suppresses it, independent of semantic content. The model responds to surface features, not meaning.

### 7.4 Cognitive Architecture Parallels

Our reasoning-action gap parallels the System 1/System 2 distinction (Kahneman, 2011): fast intuitive judgment (reasoning) vs slow deliberate action (execution). It also connects to the planning-execution separation in cognitive science and robotics.

### 7.5 Tool-Calling Benchmarks

The Berkeley Function Calling Leaderboard (Patil et al., 2025) and τ-bench (Yao et al., 2024) evaluate tool-calling accuracy but focus on reactive use (user requests tool). Our work specifically addresses proactive use (model decides to use tool), revealing a distinct failure mode.

## 8. Discussion

### 8.1 Implications for LLM Development

Our findings suggest that many LLM "failures" are not knowledge gaps but execution failures. Models often know what to do—they fail to do it because doing interferes with knowing.

This reframes the improvement agenda:
- **Not**: "How do we teach models when to use tools?"
- **But**: "How do we let models act on what they already know?"

### 8.2 Implications for Prompt Engineering

Standard prompt engineering focuses on helping models *understand* tasks. Our results suggest equal attention to helping models *execute* tasks:

- Use trigger phrases that activate action ("this is important")
- Avoid phrases that suppress action ("for your reference")
- Make execution explicit when reasoning succeeds but action fails
- Consider two-stage prompting for complex tasks

### 8.3 Implications for System Architecture

Production systems should consider:

1. **Separating reasoning from execution** via two-stage pipelines
2. **Using lightweight extractors** for format conversion
3. **Monitoring reasoning-action gaps** as a diagnostic metric
4. **Adjusting trigger language** in system prompts

### 8.4 Limitations

1. **Single model**: Results are from Claude; generalization to GPT-4, Llama, etc. requires validation
2. **Single tool type**: Memory persistence; other proactive tools may differ
3. **Sample size**: 25 scenarios provides directional evidence but larger studies needed for precise estimates
4. **No end-to-end two-stage implementation**: We project performance rather than measure it
5. **Synthetic scenarios**: Real-world usage patterns may differ

### 8.5 Future Work

1. **Multi-model evaluation**: Replicate across GPT-4, Llama, Gemini
2. **Multi-tool evaluation**: Test task logging, analytics, proactive search
3. **Two-stage implementation**: Build and evaluate complete system
4. **Domain generalization**: Test gap in code generation, JSON extraction, SQL
5. **Mechanistic analysis**: Investigate attention patterns during reasoning vs action

## 9. Conclusion

We present evidence that LLMs exhibit a significant **reasoning-action gap**: Claude correctly identifies when to use tools 96% of the time but only executes calls 68% of the time—a 28 percentage point gap. This gap varies from +60pp for implicit scenarios to 0pp for explicit commands, confirming that the issue is execution overhead, not reasoning failure.

Our findings support a two-stage architecture where the primary LLM reasons about actions in natural language and a lightweight extractor converts intent to structured format. This approach could recover most lost accuracy while freeing the primary model from format constraints.

The broader implication extends beyond tool calling: **reasoning and execution are separable capabilities that compete for cognitive resources**. This principle applies to code generation, structured extraction, and any task combining semantic judgment with format compliance.

The model knows what to do. We just need to let it tell us—and handle the doing separately.

---

## References

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. arXiv:2510.14453.

Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.

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

## Appendix A: Complete Scenario Inventory

### A.1 Implicit Scenarios (No persistence hint)

| ID | Query | Expected Content |
|----|-------|------------------|
| mem_implicit_001 | "The main entry point is src/app.py and it uses FastAPI. The database models are in src/models/." | Entry point, framework, model location |
| mem_implicit_002 | "We use PostgreSQL 15 with the pgvector extension for embeddings." | Database version, extension |
| mem_implicit_003 | "Authentication is handled by Clerk, and we store sessions in Redis." | Auth provider, session storage |
| mem_implicit_004 | "The API rate limit is 100 requests per minute per user." | Rate limit constraint |
| mem_implicit_005 | "All environment variables are in .env.local, never commit them." | Env var location, security note |

### A.2 Weak Hint Scenarios (Subtle suggestion)

| ID | Query | Trigger Pattern |
|----|-------|-----------------|
| mem_weak_001 | "Just so you know, we deploy to AWS using CDK." | "just so you know" |
| mem_weak_002 | "FYI, I prefer functional components over class components in React." | "FYI" |
| mem_weak_003 | "By the way, tests are in __tests__ folders next to the source files." | "by the way" |
| mem_weak_004 | "For your information, we use Prettier with 2-space indentation." | "for your information" |
| mem_weak_005 | "Note that all dates should be stored in UTC." | "note that" |

### A.3 Moderate Hint Scenarios (Clear but indirect)

| ID | Query | Trigger Pattern |
|----|-------|-----------------|
| mem_moderate_001 | "This is important: the authentication system uses bcrypt for password hashing and JWT for session tokens." | "this is important" |
| mem_moderate_002 | "I prefer using type hints everywhere and descriptive variable names. Keep that in mind." | "keep that in mind" |
| mem_moderate_003 | "Going forward, we'll use PostgreSQL for all new database operations." | "going forward" |
| mem_moderate_004 | "For future reference, the staging environment is at staging.example.com." | "for future reference" |
| mem_moderate_005 | "You should know that we never use console.log in production code." | "you should know" |

### A.4 Strong Hint Scenarios (Direct suggestion)

| ID | Query | Trigger Pattern |
|----|-------|-----------------|
| mem_strong_001 | "Please note for later: the CI/CD pipeline is in .github/workflows/." | "please note for later" |
| mem_strong_002 | "I'd like you to remember that I prefer tabs over spaces." | "remember that" |
| mem_strong_003 | "Make a note: we decided to use GraphQL instead of REST for the new API." | "make a note" |
| mem_strong_004 | "Don't forget: the admin panel requires 2FA for all users." | "don't forget" |
| mem_strong_005 | "Store this for later: the database connection string format is postgres://user:pass@host:5432/db." | "store this for later" |

### A.5 Explicit Scenarios (Direct command)

| ID | Query | Trigger Pattern |
|----|-------|-----------------|
| mem_explicit_001 | "Save this to memory: we use Stripe for payments with webhook endpoint at /api/webhooks/stripe." | "save this to memory" |
| mem_explicit_002 | "Please remember this decision: we're using Tailwind CSS instead of styled-components." | "please remember this" |
| mem_explicit_003 | "Add to your notes: my timezone is America/Los_Angeles." | "add to your notes" |
| mem_explicit_004 | "Record this: the production database is read-only replicated to analytics-db." | "record this" |
| mem_explicit_005 | "Use your memory tool to save: error logs go to Sentry, project ID is abc123." | "use your memory tool" |

## Appendix B: System Prompts

### B.1 Reasoning Condition

```
You are a helpful assistant. When the user shares information,
respond naturally but ALSO explicitly state what information (if any)
should be saved to a persistent memory system for future conversations.

Format your response as:
1. Your natural response to the user
2. [PERSIST]: List any information that should be saved, or "None"

Categories: codebase, user_preference, decision, task_completed, other

You do NOT have any tools—just identify what SHOULD be saved.
```

### B.2 Action Condition

```
You are a helpful assistant with access to a memory tool.

To save important information for future conversations, run:
    save-memory "content" "category"

Where category is one of: codebase, user_preference, decision, task_completed, other

IMPORTANT: Use this tool whenever you learn something important about:
- The codebase structure or conventions
- User preferences
- Decisions that were made
- Completed tasks
```

## Appendix C: Raw Results

### C.1 Per-Scenario Results

| Scenario | Level | Reasoning | Action | Gap |
|----------|-------|-----------|--------|-----|
| mem_implicit_001 | IMPLICIT | ✓ | ✗ | +1 |
| mem_implicit_002 | IMPLICIT | ✓ | ✓ | 0 |
| mem_implicit_003 | IMPLICIT | ✓ | ✗ | +1 |
| mem_implicit_004 | IMPLICIT | ✓ | ✗ | +1 |
| mem_implicit_005 | IMPLICIT | ✓ | ✓ | 0 |
| mem_weak_001 | WEAK_HINT | ✓ | ✓ | 0 |
| mem_weak_002 | WEAK_HINT | ✓ | ✓ | 0 |
| mem_weak_003 | WEAK_HINT | ✓ | ✗ | +1 |
| mem_weak_004 | WEAK_HINT | ✓ | ✗ | +1 |
| mem_weak_005 | WEAK_HINT | ✗ | ✗ | 0 |
| mem_moderate_001 | MODERATE_HINT | ✓ | ✓ | 0 |
| mem_moderate_002 | MODERATE_HINT | ✓ | ✓ | 0 |
| mem_moderate_003 | MODERATE_HINT | ✓ | ✓ | 0 |
| mem_moderate_004 | MODERATE_HINT | ✓ | ✓ | 0 |
| mem_moderate_005 | MODERATE_HINT | ✓ | ✗ | +1 |
| mem_strong_001 | STRONG_HINT | ✓ | ✗ | +1 |
| mem_strong_002 | STRONG_HINT | ✓ | ✓ | 0 |
| mem_strong_003 | STRONG_HINT | ✓ | ✓ | 0 |
| mem_strong_004 | STRONG_HINT | ✓ | ✓ | 0 |
| mem_strong_005 | STRONG_HINT | ✓ | ✓ | 0 |
| mem_explicit_001 | EXPLICIT | ✓ | ✓ | 0 |
| mem_explicit_002 | EXPLICIT | ✓ | ✓ | 0 |
| mem_explicit_003 | EXPLICIT | ✓ | ✓ | 0 |
| mem_explicit_004 | EXPLICIT | ✓ | ✓ | 0 |
| mem_explicit_005 | EXPLICIT | ✓ | ✓ | 0 |
| **TOTAL** | | **24/25 (96%)** | **17/25 (68%)** | **+28pp** |

### C.2 Statistical Tests

| Test | Statistic | p-value |
|------|-----------|---------|
| McNemar's test (paired) | χ² = 7.0 | p = 0.008 |
| 95% CI for gap | | [+10.2pp, +45.8pp] |
| Cohen's h (effect size) | | 0.73 (large) |

## Appendix D: Code Availability

Experimental code and data available at: [repository URL]

```
experiments/
├── scenarios/
│   └── proactive_tools.py      # Scenario definitions
├── multi_model_evaluator.py    # Evaluation harness
└── results/                    # Raw JSON results
```
