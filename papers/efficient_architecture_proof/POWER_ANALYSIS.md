# LAHR Power Analysis

## Purpose

This document specifies the statistical power analysis for the LAHR ablation study,
determining the minimum sample size required to detect meaningful effects.

## Study Design

- **Design**: 2^3 factorial (8 conditions)
- **Conditions**: full, no_mod, no_latent, no_memory, mod_only, latent_only, memory_only, baseline
- **Primary Outcome**: Validation perplexity
- **Seeds per Condition**: 5 (minimum), 10 (recommended)

## Effect Size Estimation

### From Prior Literature

| Source | Comparison | Observed Effect | Cohen's d |
|--------|------------|-----------------|-----------|
| MoD (Raposo 2024) | MoD vs baseline at 1B | ~50% FLOP savings at matched quality | Large |
| COCONUT (Meta 2024) | Latent vs CoT on ProsQA | 97% vs 77.5% accuracy | 2.0+ |
| COCONUT (Meta 2024) | Latent vs CoT on GSM8K | 34% vs 42% accuracy | -0.3 |
| Memorizing Transformers | Memory vs no-memory | +3-5% perplexity | 0.4-0.6 |

### Conservative Estimates

For LAHR at small scale (10M-50M params), we use conservative estimates:

| Component | Expected Effect (perplexity) | Cohen's d |
|-----------|------------------------------|-----------|
| MoD | 5-10% reduction | 0.5 |
| Latent Reasoning | 0-5% change | 0.3 |
| Memory | 2-5% reduction | 0.4 |
| Full Combination | 10-20% reduction | 0.7 |

## Power Calculation

### Parameters
- **α (Type I error rate)**: 0.05
- **Power (1 - β)**: 0.80
- **Effect size (d)**: 0.5 (minimum of interest)
- **Test**: Two-tailed paired t-test

### Formula
For paired t-test:
```
n = 2 * ((z_α/2 + z_β) / d)²
n = 2 * ((1.96 + 0.84) / 0.5)²
n = 2 * (5.6)²
n = 62.7 ≈ 63
```

But with paired design (same data, same seed initialization strategy):
```
n = ((z_α/2 + z_β) / d)²
n = (2.8 / 0.5)²
n = 31.4 ≈ 32 per condition
```

### Sample Size Recommendation

| Scenario | Seeds per Condition | Total Runs | Power |
|----------|---------------------|------------|-------|
| Minimum viable | 5 | 40 | 0.60 for d=0.5 |
| Recommended | 10 | 80 | 0.80 for d=0.5 |
| High-powered | 20 | 160 | 0.95 for d=0.5 |

## Adjusted Alpha for Multiple Comparisons

### Primary Hypothesis (1 test)
- α = 0.05

### Secondary Hypotheses (3 tests)
Using Holm-Bonferroni correction:
- Most significant: α = 0.05 / 3 = 0.0167
- Second: α = 0.05 / 2 = 0.025
- Third: α = 0.05

### Exploratory Analyses
- No correction (clearly labeled as exploratory)
- Report effect sizes with 95% CI

## Minimum Detectable Effect

With our planned 5 seeds per condition:

| Metric | SD (estimated) | MDE at power=0.60 |
|--------|----------------|-------------------|
| Perplexity | 5% relative | 8% relative |
| Throughput | 10% relative | 16% relative |
| FLOPS/token | 5% relative | 8% relative |

With 10 seeds per condition:

| Metric | SD (estimated) | MDE at power=0.80 |
|--------|----------------|-------------------|
| Perplexity | 5% relative | 6.3% relative |
| Throughput | 10% relative | 12.6% relative |
| FLOPS/token | 5% relative | 6.3% relative |

## Variance Estimation Strategy

Before committing to full study:

1. **Pilot Phase**: Run 3 seeds of baseline and full conditions
2. **Estimate**: Calculate variance from pilot
3. **Adjust**: Update sample size if variance differs from estimates
4. **Proceed**: Run remaining seeds

## Stopping Rules

### Futility Stopping
After 5 seeds:
- If 95% CI for effect includes zero AND point estimate < 0.1d, stop for futility

### Success Stopping
- Not applicable (complete all planned seeds)

### Adaptive Sample Size
If pilot variance is:
- < 50% of estimated: Can reduce to 4 seeds per condition
- 50-150% of estimated: Continue with 5 seeds
- > 150% of estimated: Increase to 8 seeds

## Pre-Registration Commitment

This power analysis will be pre-registered before data collection:

1. Primary hypothesis: Full LAHR vs baseline
2. Secondary hypotheses: Each component ablation
3. Analysis plan: Paired t-tests with Holm-Bonferroni
4. Sample size: 5 seeds minimum, 10 recommended
5. Stopping rules: As specified above

## Limitations

1. **Effect size estimates are uncertain**: Based on different tasks, scales, implementations
2. **Variance estimates are approximate**: Actual variance may differ
3. **Small sample**: 5 seeds provides limited power for small effects
4. **Non-independence**: Same training data across conditions (addressed by paired design)

## Conclusion

- **Minimum viable**: 5 seeds per condition (40 total runs)
- **Recommended**: 10 seeds per condition (80 total runs)
- **MDE**: ~8% relative at 5 seeds, ~6% relative at 10 seeds
- **Runtime estimate**: ~2 hours per run on MacBook = 80-160 hours total

The study is powered to detect medium effects (d ≥ 0.5) with 5 seeds.
For small effects (d ≈ 0.3), 10+ seeds are needed.
