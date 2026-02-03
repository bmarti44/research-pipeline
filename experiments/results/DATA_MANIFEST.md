# Data Manifest

This document describes all experimental data files, their validity for paper claims, and their organization.

**Last Updated**: 2026-02-03
**Reorganization Commit**: See CHANGELOG.md

---

## Directory Structure

```
experiments/results/
├── primary/          # Main evidence for paper claims
├── replication/      # Supporting runs that replicate findings
├── exploratory/      # Small-scale exploration runs
├── failed/           # API errors, incomplete runs
├── pilot/            # Marked pilot runs (not for claims)
├── validation/       # Human annotation data
├── legacy/           # Old nl_vs_structured runs (Study 1)
└── archive/          # Pre-existing archive of early development
```

---

## Primary Evidence (for Paper Claims)

These files contain the main experimental data supporting paper claims.

| File | N | Experiment | Key Metrics | Supports |
|------|---|------------|-------------|----------|
| `primary/signal_detection_20260203_074411_judged.json` | 750 trials (1500 obs) | Signal Detection + LLM Judge | NL 87.0%, ST 81.4%, Friction 12.2pp | Tables 1-5, §4.4-4.8 |
| `primary/signal_detection_20260203_121413.json` | 1500 observations | Signal Detection (raw, regex) | NL 76.2%, ST 69.2% | §4.7 Regex baseline |
| `primary/two_pass_sonnet_nl_20260203_125603.json` | 750 trials | Two-Pass Extraction (Sonnet) | 64.9% recovery, 97.3% precision | §4.9 Table 6 |
| `primary/two_pass_qwen7b_nl_20260203_131141.json` | 750 trials | Two-Pass Extraction (Qwen 7B) | 39.2% recovery, 96.7% precision | §4.9 Table 6 |

### Key Statistics Supported

From `signal_detection_20260203_074411_judged.json`:
- **Table 1**: Detection rates (NL 87.0%, ST 81.4%, McNemar p=0.018)
- **Table 2**: Format friction (12.2pp gap, 60 silent failures)
- **Table 3**: Ambiguity interaction (EXPLICIT 0pp, IMPLICIT 20.5pp)
- **Table 4**: Measurement comparison (regex vs judge)
- **Table 5**: False positive rates (NL 0.0%, ST 2.2%)

From `two_pass_sonnet_nl_20260203_125603.json` and `two_pass_qwen7b_nl_20260203_131141.json`:
- **Table 6**: Two-pass recovery rates (Sonnet 64.9%, Qwen 7B 39.2%)
- Silent failures tested: 74
- Sonnet recovered: 48/74
- Qwen 7B recovered: 29/74
- Type accuracy: Sonnet 88.5%, Qwen 7B 82.6%

---

## Replication Data

These runs replicate the primary signal detection experiment and can be used for robustness checks.

| File | N | Notes |
|------|---|-------|
| `replication/signal_detection_20260203_010117.json` | 1500 | Full replication run |
| `replication/signal_detection_20260203_024909.json` | 1500 | Full replication run |
| `replication/signal_detection_20260203_012453.json` | 1500 | Full replication run |
| `replication/signal_detection_20260203_012650.json` | 1500 | Full replication run |
| `replication/signal_detection_20260203_013054.json` | 1500 | Full replication run |
| `replication/signal_validation_*.json` | varies | Validation outputs for above |

---

## Exploratory Data

Small-scale runs used for exploration, debugging, or early development. **Not for paper claims**.

| File | Notes |
|------|-------|
| `exploratory/signal_detection_20260202_221456.json` | Early signal detection trial |
| `exploratory/signal_detection_20260202_221655.json` | Early signal detection trial |
| `exploratory/signal_validation_*.json` | Validation outputs |
| `exploratory/quick_recovery_test.json` | Quick recovery mechanism test |
| `exploratory/analysis_report_*.txt` | Intermediate analysis outputs |

---

## Failed/Unusable Data

These files contain runs that failed due to API errors or other issues. **NOT for paper claims**.

| File | Issue | Status |
|------|-------|--------|
| `failed/two_pass_sonnet_nl_20260203_113046.json` | API credit error ("Credit balance is too low" in all responses) | UNUSABLE |
| `failed/two_stage_20260203_094518.json` | API credit error ("Credit balance is too low" in all responses) | UNUSABLE |

---

## Pilot Data

Pilot runs marked for exploratory purposes only. **NOT for paper claims** without explicit caveats.

| File | N | Issue | Status |
|------|---|-------|--------|
| `pilot/two_pass_qwen7b_nl_20260203_113119.json` | 10 | Pilot run, `total_silent_failures: 0` in sample | PILOT ONLY |
| `pilot/two_pass_qwen7b_nl_20260203_114549.json` | 10 | Pilot run, `total_silent_failures: 0` in sample | PILOT ONLY |

### Important Note on Two-Pass Recovery

The pilot runs tested NL responses where the judge already detected the signal. Because the sampled trials had `total_silent_failures: 0`, there were no actual silent failures to recover. The Qwen 7B model successfully extracted signals from the NL responses (10/10 extracted, 9/10 type correct), demonstrating extraction viability but **not recovery of actual silent failures**.

**Paper Section 4.9 has been revised** to reflect this as preliminary/pilot work rather than validated findings.

---

## Validation Data

Human annotation data for judge validation. **COMPLETED** - κ = 0.812 (substantial agreement).

| File | Description | Status |
|------|-------------|--------|
| `validation/validation_annotation_20260203_074834.csv` | 150 stratified samples with human labels | COMPLETE |
| `validation/validation_key_20260203_074834.csv` | Judge/regex labels for comparison | COMPLETE |

### Agreement Results

| Stratum | κ | Agreement | N |
|---------|---|-----------|---|
| Overall | 0.812 | 90.7% | 150 |
| EXPLICIT | 1.000 | 100% | 30 |
| CONTROL | 1.000 | 100% | 40 |
| BORDERLINE | 0.867 | 93.3% | 30 |
| IMPLICIT | 0.406 | 76.0% | 50 |

Note: Lower IMPLICIT agreement reflects genuine ambiguity in what constitutes "signal detection" vs "signal handling."

---

## Legacy Data (Study 1)

Old `nl_vs_structured` experiment runs from Study 1 (memory persistence task). These demonstrate the confound discovery (§3) but use a different experimental paradigm.

| Directory | Contents |
|-----------|----------|
| `legacy/nl_vs_structured_*.json` | ~30 experiment runs |
| `legacy/validation_samples_*.json` | Corresponding validation samples |

---

## Archive

Pre-existing archive of early development files. See `archive/` subdirectory for:
- Early experiment iterations (`results_*.json`, `metadata_*.json`)
- Memory tool experiments (`memory_experiment_*.json`)
- Intermediate experiments (`intent_vs_tool_*.json`)
- Development `nl_vs_structured` runs
- Old analysis summaries (`analysis_*.md`)

---

## Claim-to-Evidence Mapping

| Paper Section | Claim | Evidence File | Status |
|---------------|-------|---------------|--------|
| §3.3 Study 1 | 9pp effect (p<0.01) disappeared after correction | `legacy/nl_vs_structured_*.json` | SUPPORTED |
| §3.3 Study 1 | "After Correction" 0pp difference | Limited validation (n=5) | WEAK |
| §4.4 Table 1 | Detection rates 87.0%/81.4% | `primary/signal_detection_judged.json` | SUPPORTED |
| §4.5 Table 2 | Format friction 12.2pp | `primary/signal_detection_judged.json` | SUPPORTED |
| §4.6 Table 3 | Ambiguity interaction | `primary/signal_detection_judged.json` | SUPPORTED |
| §4.7 Table 4 | Regex vs Judge comparison | `primary/signal_detection_*.json` | SUPPORTED |
| §4.8 Table 5 | False positive rates | `primary/signal_detection_judged.json` | SUPPORTED |
| §4.9 Table 6 | Two-pass recovery (Sonnet 64.9%, Qwen 39.2%) | `primary/two_pass_sonnet_nl_*.json`, `primary/two_pass_qwen7b_nl_*.json` | SUPPORTED |
| §4.3 | Judge validation κ = 0.81 | `validation/validation_annotation_*.csv` | SUPPORTED |

---

## Academic Integrity Notes

1. **No data deleted**: All files preserved and reorganized
2. **Failed experiments documented**: API errors clearly marked
3. **Pilot data distinguished**: Cannot support quantitative claims
4. **Provenance preserved**: Git history tracks all changes
5. **Judge validation completed**: Human annotation supports judge reliability (κ = 0.81)
5. **Claims aligned with evidence**: Paper revised where evidence insufficient
