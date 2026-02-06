# Preregistration: format_friction

## Locked At
2026-02-05T03:57:11.808322+00:00

## Hash
sha256:b2584100f1d4ca9bd066af8e7eb079f7ca1598d05b5318b75f676da0750e11ff

## Hypothesis
LLMs exhibit higher error rates when producing structured JSON tool calls
compared to expressing the same intent in natural language.

STATUS: PILOT REFUTED THIS HYPOTHESIS
- JSON syntax validity: 100%
- JSON schema correctness: 98.5%
- True serialization failures: 1.5%

The "friction" observed was actually behavioral adaptation, not serialization failure.


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
- config.yaml: sha256:b22689e57e941edd848c74bac244712265c123226319c5edf48505e352afa3c5
- tasks.py: sha256:bf5cb1366d6428b65d60b8a19a3aec9ececa676d85f99d4199435f1fb310ec78
- evaluation.py: sha256:84d17bee41df8e07456080fc027c3478da17e6b848aa06f326e476e576fd39f3
- analysis.py: sha256:de182000495db868067107dcfa40a84976a5e819be7cc774dfef6b9c385c3986

---
*This document was auto-generated and should not be modified after creation.*
