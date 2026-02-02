# Ablation Study Results

**Date**: 2026-02-01
**Trials per scenario**: 3
**Total scenarios**: 41

## Summary

Each experiment tests ONE rule in isolation to determine its individual contribution.

| Rule | Baseline | Validated | Δ | p-value | Significant | Catch Rate |
|------|----------|-----------|---|---------|-------------|------------|
| F10 (Duplicate Search) | 1.890 | 2.126 | +0.236 | 0.0023 | **Yes** | 67% |
| F8 (Missing Location) | 2.061 | 2.114 | +0.053 | 0.3075 | No | 22% |
| F15 (Binary Files) | 1.947 | 1.972 | +0.024 | 0.7650 | No | 0% |

## Key Findings

### F10 (Duplicate Search) - **Primary Value Driver**
- Only rule with statistically significant improvement in isolation
- Improvement: +0.236 points (p=0.002)
- 67% catch rate - actively blocking duplicate searches
- **Recommendation**: This rule alone justifies the validation system

### F8 (Missing Location)
- Small positive improvement (+0.053) but not significant (p=0.31)
- 22% catch rate - detecting some but not all location-dependent queries
- **Recommendation**: Keep but consider threshold tuning

### F15 (Binary Files)
- Negligible improvement (+0.024), not significant (p=0.77)
- 0% catch rate - Claude rarely attempts to read binary files
- **Recommendation**: Keep for safety but doesn't contribute measurably

## Conclusion

The F10 duplicate search rule is the primary contributor to the validator's effectiveness. Without F10, the validation system would not show statistically significant improvement.

This suggests:
1. Focus development effort on rules that address actual model weaknesses
2. Rules for behaviors Claude already handles well (F1, F4, F15) provide minimal value
3. The validator's ROI comes from specific failure modes, not broad coverage
