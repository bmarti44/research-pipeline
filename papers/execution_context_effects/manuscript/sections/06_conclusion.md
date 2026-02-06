# Conclusion

We set out to study format friction in LLM tool-calling and found something more interesting: what looks like friction is actually alignment activation.

## Summary of Findings

**Study 1** refuted the hypothesis that JSON causes serialization failures. JSON syntax validity was 100%, and schema compliance was 98.5%. The observed performance gap (NL: 97.5% vs JSON: 70.0%) could not be attributed to format-related errors.

**Study 2** confirmed that execution context triggers operation-specific caution. When output will actually run (JSON mode), models adopt safer behaviors for destructive operations. This pattern was absolute: 100% cautious behavior for edit/delete operations, 0% for create/read operations.

## Key Contributions

1. **Empirical**: We demonstrate that format friction is largely a misnomer—JSON serialization works reliably
2. **Theoretical**: We propose "alignment activation" as a framework for understanding execution-aware behavioral changes
3. **Methodological**: We highlight the need for evaluation approaches that distinguish capability from alignment

## Implications

Our work suggests that some "friction" in agentic systems may be valuable rather than problematic. Before optimizing to eliminate all performance gaps between NL and structured output, we should consider:

- Whether the gap reflects safety behaviors we want to preserve
- Whether NL evaluation provides unrealistic performance expectations
- Whether multi-step interactions (read → edit) are preferable to single-step execution

## Closing Thought

The finding that models are "too cautious" in execution contexts is, in many ways, a success story for alignment. The challenge now is to preserve appropriate caution while enabling efficient task completion—a design problem rather than a capability problem.
