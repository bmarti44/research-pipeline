# Results

## Study 1: Format Friction

### JSON Serialization Quality

Contrary to expectations, JSON serialization was highly reliable:

| Metric | Rate |
|--------|------|
| JSON syntax validity | 100% |
| JSON schema compliance | 98.5% |
| True serialization failures | 1.5% |

The model produced syntactically valid JSON in every trial. Schema violations were rare (missing fields or wrong types).

### Task Accuracy

Despite reliable serialization, we observed a significant performance gap:

| Condition | n | Correct | Rate | 95% CI |
|-----------|---|---------|------|--------|
| NL | 40 | 39 | 97.5% | [87.1%, 99.6%] |
| JSON | 40 | 28 | 70.0% | [54.6%, 81.9%] |

**Two-proportion z-test**: z = 3.33, p < 0.001

The 27.5 percentage point difference was statistically significant. However, this could not be attributed to serialization failures, which were minimal.

### Error Analysis

Examining the JSON errors revealed a pattern:

1. **edit_file tasks**: Model frequently requested to read the file first
2. **delete_file tasks**: Model requested confirmation
3. **create_file tasks**: No cautious behavior observed

This suggested the errors were behavioral (cautious responses) rather than capability-based (format failures).

## Study 2: Execution Context

### Task Accuracy

The performance gap persisted:

| Condition | n | Correct | Rate | 95% CI |
|-----------|---|---------|------|--------|
| NL | 40 | 36 | 90.0% | [76.9%, 96.0%] |
| JSON | 40 | 29 | 72.5% | [57.2%, 83.9%] |

**Two-proportion z-test**: z = 2.01, p = 0.045

The 17.5 percentage point difference was statistically significant.

### Operation-Specific Patterns

Breaking down by operation type revealed striking patterns:

| Operation Type | NL Cautious | JSON Cautious |
|----------------|-------------|---------------|
| Destructive (edit, delete) | 0% | 100% |
| Non-destructive (create, read) | 0% | 0% |

In JSON mode, the model exhibited cautious behavior (read-first, confirmation requests) for every destructive operation but never for non-destructive operations. This pattern was absent in NL mode.

### Cautious Behaviors Observed

In JSON mode for destructive operations, we observed:
- **Read-first requests**: "I'll first read the file to understand its structure"
- **Confirmation requests**: "Before I proceed, please confirm..."
- **Alternative suggestions**: "A safer approach would be to..."

These behaviors are consistent with execution-aware caution—the model recognizing that its output will actually run.

## Combined Analysis

### Meta-Analysis

Pooling across both studies:

| Metric | Value |
|--------|-------|
| Pooled NL rate | 93.8% |
| Pooled JSON rate | 71.3% |
| Average difference | 22.5% |
| Both studies significant | Yes |
| Effect direction consistent | Yes |

### Interpretation

The consistent, significant effect across both studies supports our interpretation:

1. The performance gap is real and substantial (~20+ percentage points)
2. It is NOT caused by serialization failures (which are rare)
3. It IS caused by execution-aware behavioral adaptation
4. This adaptation is operation-specific (destructive vs non-destructive)

We term this phenomenon "alignment activation"—the model appropriately modulating its behavior based on whether its output will have real consequences.
