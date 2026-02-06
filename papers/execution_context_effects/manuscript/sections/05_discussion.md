# Discussion

## Reframing Format Friction

Our findings challenge the conventional understanding of format friction. The standard narrative holds that structured output requirements create serialization burden that degrades performance. Our data tells a different story:

**What we expected**: JSON mode would cause more syntax errors, schema violations, and format-related failures.

**What we found**: JSON mode triggers execution-aware caution for destructive operations, leading to safer but "incorrect" (from a task-completion standpoint) responses.

This distinction matters because it changes the intervention. If format friction were a capability limitation, we would optimize for better JSON generation. Instead, we find it reflects alignment working as intended—suggesting we should consider whether eliminating this "friction" is desirable.

## Alignment Activation

We propose the term "alignment activation" to describe what we observed: differential behavior based on execution context. Key features:

1. **Context-sensitivity**: The model distinguishes between describing actions (NL) and performing actions (JSON)
2. **Operation-specificity**: Caution applies to destructive operations (edit, delete) but not non-destructive ones (create, read)
3. **Appropriate magnitude**: Requests for verification, reading before writing, and confirmation are reasonable safety measures

This suggests the model has learned (likely from RLHF or constitutional AI training) that actions with real consequences warrant additional caution—particularly for operations that could cause harm if incorrect.

## Implications for Evaluation

Our findings have important implications for how we evaluate LLM tool use:

### NL Benchmarks May Overestimate Capability

If NL evaluation doesn't trigger execution-aware caution, it may provide inflated estimates of how the model will perform when its output actually runs. Benchmark performance may not transfer to deployment.

### "Errors" May Be Features

Behaviors counted as errors in our strict evaluation (requesting to read before edit) might be desirable in production. Evaluation metrics should distinguish between:
- **True errors**: Wrong tool, wrong arguments, format failures
- **Safety behaviors**: Verification requests, read-first patterns
- **Alignment expressions**: Refusing harmful requests, requesting confirmation

### Multi-Step Evaluation Needed

Single-turn evaluation may penalize appropriate multi-step behavior. If a model correctly requests to read a file before editing it, a single-turn evaluation marks this as wrong, but a multi-turn system would allow the operation to complete correctly and more safely.

## Limitations

Several limitations constrain our conclusions:

1. **Single model**: We tested only Claude claude-sonnet-4-20250514. Other models may show different patterns.
2. **Sample size**: 80 trials per study provides limited statistical power for subgroup analyses.
3. **Task selection**: Our 8 tasks may not represent the full distribution of tool-calling scenarios.
4. **Evaluation strictness**: Our intent-based evaluation may still miss valid alternative formulations.

## Future Directions

Our findings suggest several research directions:

1. **Cross-model replication**: Do other models show similar alignment activation?
2. **Training analysis**: Can we identify which training signals create operation-specific caution?
3. **Calibration study**: Is the model's caution appropriately calibrated to actual risk?
4. **User preference study**: Do users prefer cautious or direct tool execution?
5. **Longitudinal analysis**: Does alignment activation persist across model versions?
