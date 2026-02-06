# Preregistration: execution_context

## Locked At
2026-02-05T03:38:46.301374+00:00

## Hash
sha256:0e82369507e314181a51fbed2c8588e2b74b1a70260be6fa9b9af802f4c62e64

## Hypothesis
When LLM output will actually execute (JSON tool calls), the model adopts
safer behaviors for destructive operations. This is operation-specific
caution, not serialization failure.

PILOT OBSERVATION:
- edit_file tasks: 100% chose "read first" in JSON mode
- create_file tasks: 0% cautious behavior
- This is execution-aware alignment activation, not format friction


## Primary Dependent Variable


## Analysis Plan
- Not specified

## Exclusion Criteria
- None specified

## Power Analysis
- Target effect size: None
- Alpha: 0.05
- Power: 0.8
- Sample size: None

## Locked Files
- config.yaml: sha256:c26ae500310a0a7233d0bb0fcd412e866a5b0e47a47ae41e02d7730b0a94bcdc
- tasks.py: sha256:bf5cb1366d6428b65d60b8a19a3aec9ececa676d85f99d4199435f1fb310ec78
- evaluation.py: sha256:84d17bee41df8e07456080fc027c3478da17e6b848aa06f326e476e576fd39f3
- analysis.py: sha256:45188848781f9ca32504104167cd0c05c66328165a5706b2cfeb7acb3ad5097d

---
*This document was auto-generated and should not be modified after creation.*
