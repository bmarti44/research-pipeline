# Archive

This directory contains superseded or diagnostic files that are no longer part of the main experiment pipeline but are kept for reference and reproducibility documentation.

## Files

### rescore_with_fixed_patterns.py
**Purpose:** Diagnostic tool created during regex pattern debugging for NL signal detection.

**Background:** During development of the signal detection experiment, we discovered that regex-based scoring produced measurement asymmetry between NL and structured conditions. This script was used to:
1. Test different regex pattern configurations
2. Compare conservative vs permissive thresholds
3. Identify false positives from patterns like "immediately" in technical contexts
4. Debug the "sounds like you" pattern that was matching conversational mirroring

**Current status:** Not part of the final pipeline. The LLM judge (judge_scoring.py) is the primary scoring method. Regex patterns are frozen and kept only for the regex-vs-judge comparison analysis in the paper.

**See also:**
- experiments/signal_detection_experiment.py - Contains the frozen regex patterns
- experiments/judge_scoring.py - The primary scoring method
- Paper Section 4.2 (Measurement Approach) for the methodological discussion

---

### two_stage_experiment.py
**Purpose:** Experiment comparing single-pass vs two-stage (reasoner + extractor) tool calling.

**Status:** FAILED - All API calls returned "Credit balance is too low" error. No valid data produced.

**Data:** `experiments/results/failed/two_stage_20260203_094518.json`

---

### tool_definitions.py
**Purpose:** Tool definitions (api_call, memory_save) used by two_stage_experiment.py.

**Status:** DEAD - Only imported by two_stage_experiment.py which failed.

---

### tool_calling.py (from scenarios/)
**Purpose:** Scenario definitions for the two-stage experiment.

**Status:** DEAD - Only imported by two_stage_experiment.py which failed.

---

*Last updated: 2026-02-03*
