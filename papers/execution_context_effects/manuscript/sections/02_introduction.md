# Introduction

Large language models (LLMs) are increasingly deployed as autonomous agents that interact with external systems through tool calls. These interactions typically require structured output formats like JSON, leading to concerns about "format friction"—the hypothesis that models exhibit higher error rates when constrained to produce structured output compared to natural language.

Understanding format friction matters for several reasons. First, agentic systems rely on reliable tool execution, and understanding failure modes is critical for deployment. Second, the tension between capability and constraint informs API design choices. Third, distinguishing between true capability limitations and other sources of error is essential for proper evaluation.

## The Format Friction Hypothesis

The format friction hypothesis posits that requiring structured JSON output creates serialization burden that degrades model performance. This could manifest as:

1. **Syntax failures**: Invalid JSON (missing quotes, brackets)
2. **Schema violations**: Valid JSON that doesn't match expected structure
3. **Content errors**: Correct structure with wrong values

Previous work has documented performance gaps between natural language and structured output tasks, typically attributing these to the cognitive overhead of format compliance.

## Our Investigation

We conducted two preregistered studies to systematically investigate format friction:

**Study 1** tested whether JSON causes serialization failures. We presented identical tasks requiring either natural language descriptions or JSON tool calls, measuring both format compliance and task correctness.

**Study 2** investigated the behavioral differences observed in Study 1. Based on pilot observations showing operation-specific patterns, we tested whether execution context triggers differential caution for destructive versus non-destructive operations.

Our findings challenge the standard framing of format friction. We demonstrate that the observed performance gap is not caused by serialization failures but rather by execution-aware behavioral adaptation—what we term "alignment activation."
