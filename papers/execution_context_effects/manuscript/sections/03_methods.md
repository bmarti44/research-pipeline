# Methods

## General Procedure

Both studies used a within-subjects design with two conditions:
- **NL condition**: Model describes intended action in natural language
- **JSON condition**: Model produces structured JSON tool call

We used Claude claude-sonnet-4-20250514 (claude-sonnet-4-20250514) for all trials, with temperature=0 for reproducibility. Each study comprised 80 trials (40 per condition × 8 tasks × 5 repetitions).

## Preregistration

Both studies were preregistered with tamper-evident SHA-256 hashes computed before data collection. Preregistration documents specified hypotheses, sample sizes, analysis plans, and stopping rules.

## Study 1: Format Friction

### Task Design

We selected 8 representative tool-calling tasks spanning common operations:
- File operations: create_file, edit_file, delete_file
- Data operations: search_data, query_database
- System operations: send_email, schedule_meeting, http_request

Each task included a tool specification (name, parameters, types) and a natural language prompt requesting the operation.

### Conditions

In the **NL condition**, models received:
```
Please describe what tool you would call and with what arguments.
```

In the **JSON condition**, models received:
```
Please output a JSON tool call with "tool" and "args" fields.
```

### Evaluation

We measured three aspects:
1. **Syntax validity**: Does the output parse as valid JSON?
2. **Schema compliance**: Does it match the tool call schema?
3. **Task correctness**: Does it select the right tool with correct arguments?

Task correctness used intent-based evaluation: does the response reference the expected tool and include the required argument values, even if phrasing differs?

## Study 2: Execution Context

### Task Design

Based on Study 1 pilot observations, we designed tasks to isolate operation type:
- **Destructive operations**: edit_file, delete_file, update_record, overwrite_config
- **Non-destructive operations**: create_file, read_file, list_directory, search

### Evaluation

In addition to task correctness, we tracked:
- **Cautious behavior**: Did the model request additional information or verification?
- **Read-first pattern**: For file operations, did the model request to read before writing?
- **Confirmation requests**: Did the model ask for user confirmation?

## Statistical Analysis

### Primary Test

Two-proportion z-test comparing NL vs JSON accuracy rates.

### Secondary Analyses

- Wilson confidence intervals for each condition
- Chi-square tests for independence
- Bootstrap resampling for robustness checks

### Assumptions

Before running parametric tests, we verified:
- Minimum expected cell counts ≥ 5 for chi-square
- Independence of observations (between-trial)
- Sufficient sample size for normal approximation (np > 5, n(1-p) > 5)

## Reproducibility

All code, data, and analysis scripts are available at the study repository. Random seeds were fixed for reproducibility. The pipeline uses locked package versions and deterministic evaluation.
