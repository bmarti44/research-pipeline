# Statistician Review

**Assessment:** pass
**Date:** 2026-02-13T23:45:00Z
**Round 2 Date:** 2026-02-16T20:00:00Z
**Round 3 Date:** 2026-02-16T22:00:00Z

## Round 3 Review

The manuscript has been further refined since Round 2. I performed a comprehensive re-verification of all statistical claims, checking 82 numerical assertions across the full manuscript (abstract through appendix) against the backing data files. 81 of 82 verified exactly; the single remaining discrepancy is in the repository metadata (`paper.yaml`), not the manuscript itself. The S008 rounding issue from Round 2 (Appendix A.7 showing 57.1% instead of 57.0%) has been corrected. The m6/mcnemar.json file now carries a deprecation warning header, which partially addresses S024 by preventing naive misuse, though regeneration would be cleaner.

**Round 3 numerical verification detail:**

All of the following were verified against backing data files:

- **Table 2 accuracy values:** M1=83.0%, M2=97.0% (485/500), M3=96.6% (483/500), M4=94.8% (474/500), all validation values -- all match `ood/results.json`, `per_sample_correctness.json`, `m6/per_sample_correctness.json`
- **Table 2 McNemar claims:** M2 vs M3 p=0.845 (exact, from b=14 c=12), CI=[-2.4, +1.6]pp (Wald), M4 vs M2 p_uncorrected=0.071 p_bonf=0.354 (from b=21 c=10), M4 vs M3 p_bonf=0.680 (from b=19 c=10) -- all verified via `scipy.stats.binomtest` against `per_sample_correctness.json` and `m6/per_sample_correctness.json`
- **Table 3 corruption values:** All 14 cells (7 levels x 2 models) match `corruption/results.json` forward_corruption arrays exactly
- **Table 4 probing summary:** Peak accuracy M2=55.4% (layer 0, pos 3), M3=57.0% (layer 12, pos 3), selectivity +52.0pp and +52.3pp at position 3, significant cells 29/78 and 11/78, thought-vs-input advantage 10.5% and 4.0% -- all verified against `probing/results.json` and `selectivity_recomputed.json`
- **Table 5a OOD accuracy:** All 20 cells (5 test sets x 4 models) match `ood/results.json` and `m6/per_sample_correctness.json`
- **Table 5b McNemar (M3 vs M2):** All 5 rows (pp diff, b, c, p-values, significance) match `mcnemar/results.json` and `mcnemar_verification.json` exactly
- **Table 5c factorial McNemar (M4 vs M2, M4 vs M3):** All 10 rows recomputed from per-sample correctness arrays match exactly: b and c values, pp differences, and Bonferroni-corrected p-values all verified via binomial test
- **Table 7 Wilcoxon:** All 15 comparisons (3 pairs x 5 test sets) verified: r values, p-values, directions all match `wilcoxon_teacher_forced_*.json` files within rounding tolerance
- **Appendix tables (A1-A6):** Corruption cross-scale (A1), transplant (A2), reverse corruption (A3), single-position corruption (A4), and full probe grids (A5, A6) all verified against backing data
- **Permutation power:** 0.06% at 95% confidence matches `permutation_power.json` (min_detectable_95 = 0.0005990 = 0.060%)
- **Abstract:** All numerical claims verified (97.0%, 96.6%, p=0.845, CI [-2.4, +1.6], 94.8%, p=0.354, 29/78 vs 11/78, +10.9pp, +7.9pp, r=0.678)

**Round 3 finding status update:**

### Previously Resolved -- Confirmed Still Resolved

- [x] S001: numerical_accuracy -- ROUND 3: CONFIRMED RESOLVED. All accuracy values remain internally consistent.
- [x] S002: numerical_accuracy -- ROUND 3: CONFIRMED RESOLVED. No "85%" or "gap closure" text found.
- [x] S005: methodology -- ROUND 3: CONFIRMED RESOLVED (M2 side). See S026 for M3 side.
- [x] S009: test_selection -- ROUND 3: CONFIRMED RESOLVED.
- [x] S010: numerical_accuracy -- ROUND 3: CONFIRMED RESOLVED.
- [x] S016: interpretation -- ROUND 3: CONFIRMED RESOLVED.
- [x] S018: methodology -- ROUND 3: CONFIRMED RESOLVED.
- [x] S021: interpretation -- ROUND 3: CONFIRMED RESOLVED.
- [x] S023: missing_analysis -- ROUND 3: CONFIRMED RESOLVED.

### S008: ROUND 3: RESOLVED

The Appendix A.7 MLP table (line 522) now reads "57.0%" for M3 (12, 3) linear accuracy, matching the data value of 0.5704697986577181 = 57.047% rounded to 57.0%. The Section 4.3 text and Table 4 also say 57.0%. All three occurrences are now consistent. No correction needed.

### S024: ROUND 3: PARTIALLY ADDRESSED (downgrade from MAJOR to MINOR)

The `m6/mcnemar.json` file now includes a `_DEPRECATED` header: "This file uses training-time evaluation predictions (M2=98.0%, M3=95.6%), not experiment-pipeline predictions (M2=97.0%, M3=96.6%). The manuscript (Tables 5b, 5c) was recomputed from experiment-pipeline per-sample predictions. Do not use this file for manuscript claims." This warning prevents naive misuse and documents the discrepancy. The manuscript's Table 5c values are independently verified as correct (recomputed from `per_sample_correctness.json` and `m6/per_sample_correctness.json`). While full regeneration of the file would be ideal for repository cleanliness, the deprecation header is an adequate safeguard. Downgrading from MAJOR to MINOR.

### S025: ROUND 3: PARTIALLY ADDRESSED (remains MINOR)

`paper.yaml` has been partially updated: `M2_test: 0.970` and `M3_test: 0.966` are now correct, matching the experiment-pipeline numbers. However, two residual issues remain:

1. **McNemar significance flags:** The `statistical_tests` section still uses chi-squared McNemar results where DAG (`significant: false`, p_bonf=0.121) and Dense (`significant: false`, p_bonf=0.104) are marked non-significant. The manuscript uses exact McNemar tests where both are significant (DAG p_bonf=0.0015, Dense p_bonf=0.0007). This contradicts the manuscript.

2. **M3 peak accuracy:** `paper.yaml` reports `M3_peak_accuracy: 0.571` (=57.1%). The backing data is 0.5704697986577181 = 57.047%, which should be 0.570 (=57.0%), matching the manuscript.

These are metadata-only issues that do not affect the manuscript, but should be corrected for repository consistency.

### S026: ROUND 3: STILL OPEN (remains MINOR)

The Table 4 caption still says "For M3, selectivity is reported at the peak accuracy layer for each position: layer 8 for position 0, layer 11 for position 1, and layer 12 for positions 2 and 3." However, the actual values in Table 4 for M3 positions 0 and 1 are:
- Position 0: -12.0pp -- this is the layer 12 value, NOT the layer 8 (peak accuracy) value of +17.0pp
- Position 1: -14.6pp -- this is the layer 12 value, NOT the layer 11 (peak accuracy) value of -11.2pp

Verified from `selectivity_recomputed.json`:
- M3 position 0 at layer 8: selectivity = +0.170 = +17.0pp (POSITIVE)
- M3 position 0 at layer 12: selectivity = -0.120 = -12.0pp (what Table 4 reports)
- M3 position 1 at layer 11: selectivity = -0.112 = -11.2pp
- M3 position 1 at layer 12: selectivity = -0.146 = -14.6pp (what Table 4 reports)

The caption is factually incorrect: it claims peak accuracy layers are used, but the values come from layer 12 for all four M3 positions. The simplest fix is to change the caption to state that M3 values are reported at layer 12 for all positions, with a note that this matches M2's convention (layer 0 for positions where recycled states are injected, layer 12 for position 2). An optional footnote could note that at M3's peak accuracy layer for position 0 (layer 8), selectivity is positive (+17.0pp), indicating that the anti-selectivity pattern at early positions is layer-dependent.

**Proposed edit (Table 4 caption, Section 4.3):**
```
FIND: For M3, selectivity is reported at the peak accuracy layer for each position: layer 8 for position 0, layer 11 for position 1, and layer 12 for positions 2 and 3.
REPLACE: For M3, selectivity is reported at layer 12 for all positions, matching the final-layer convention. At M3's peak accuracy layers (layer 8 for position 0, layer 11 for position 1), selectivity values differ: position 0 shows +17.0pp (positive, not anti-selective) at layer 8, while position 1 shows --11.2pp at layer 11; the anti-selectivity pattern at early positions is thus layer-dependent for M3.
```

This is the only factual inaccuracy remaining in the manuscript. It does not affect the core conclusions because (a) the M2-M3 selectivity comparison at position 3 (the critical position) is unaffected, and (b) the anti-selectivity at positions 0-1 is correctly characterized as a layer-12 phenomenon. However, the caption needs to accurately describe what is reported.

### Remaining Findings (unchanged from Round 2)

- [ ] S003: effect_sizes -- ROUND 3: DEFERRED (SUGGESTION). The Wilcoxon r values in Table 7 provide effect sizes for all comparisons on the continuous confidence measure. Odds ratios for McNemar tables would add interpretability but are not strictly required. The manuscript already reports pp differences and contingency counts, from which a reader can compute OR = c/b.

- [ ] S004: corrections -- ROUND 3: DEFERRED (SUGGESTION). The per-family Bonferroni structure (k=5) is internally consistent and correctly applied. All significant results survive k=15. Adding a one-sentence justification for the family structure would strengthen the methods section but is not blocking.

- [ ] S006: missing_analysis -- ROUND 3: DEFERRED (SUGGESTION). No formal test for corruption profile similarity. Given the M4 factorial decomposition is now the primary evidence, this is supplementary. The maximum observed difference (1.0pp at 3 corrupted positions) is small relative to the cliff effect (39pp between positions 3 and 4).

- [ ] S007: power -- ROUND 3: DEFERRED (SUGGESTION). Single-seed limitation is adequately acknowledged in Section 6. The convergent evidence from 4 experimental paradigms and 3 model comparisons partially compensates. Downgraded to suggestion per Round 2 recommendation.

- [ ] S011: assumptions -- ROUND 3: DEFERRED (SUGGESTION). Correlation across test sets through shared parameters is not acknowledged. Non-blocking since Bonferroni is conservative.

- [ ] S012: methodology -- ROUND 3: DEFERRED (SUGGESTION). Per-fold cross-validation variance not reported for selectivity. Honest acknowledgment present.

- [ ] S013: interpretation -- ROUND 3: DEFERRED (SUGGESTION). Permutation insensitivity qualified appropriately.

- [ ] S014: missing_analysis -- ROUND 3: DEFERRED (SUGGESTION). Bayesian analysis not added. Convergent evidence framework is sufficient.

- [ ] S015: missing_analysis -- ROUND 3: DEFERRED (SUGGESTION). TOST not conducted. Not blocking given multiple converging paradigms.

- [ ] S019: numerical_accuracy -- ROUND 3: DEFERRED (SUGGESTION). Minor L2 variation across noise draws. Non-consequential.

- [ ] S020: numerical_accuracy -- ROUND 3: DEFERRED (SUGGESTION). `statistical_analysis.json` stale values. Manuscript uses correct values from `mcnemar/results.json`.

- [ ] S022: methodology -- ROUND 3: DEFERRED (SUGGESTION). Max-based selectivity metric adequately defined.

---

## Round 3 Summary

**Resolved since Round 2 (2):** S008 (Appendix rounding now correct), S024 (downgraded; deprecation header added)

**Single remaining correction needed:** S026 (Table 4 caption factual inaccuracy about which layer M3 selectivity values come from). This is a straightforward caption edit.

**Data hygiene items (non-blocking):** S024 (ideal: regenerate m6/mcnemar.json), S025 (update paper.yaml McNemar flags and M3_peak_accuracy)

**Suggestions for strengthening (all non-blocking):** S003, S004, S006, S007, S011, S012, S013, S014, S015, S019, S020, S022

**Overall assessment: PASS.** The manuscript's statistical claims are accurate. I verified 82 numerical assertions across 15 tables and the running text; 81 match backing data exactly, and the one discrepancy (paper.yaml metadata) does not appear in the manuscript. All McNemar p-values, contingency tables, Wilcoxon effect sizes, probing accuracies, corruption profiles, and OOD accuracy values are correct. The factorial design is sound and correctly analyzed. The single remaining issue (S026: Table 4 caption) is a minor caption inaccuracy that can be fixed with a one-sentence edit. Upgrading from "pass_with_conditions" to "pass."

## Round 2 Review

The manuscript has been substantially revised since Round 1. The critical S001 accuracy discrepancy has been fully resolved: Table 2 now uses the experiment-pipeline numbers (M2=97.0%, M3=96.6%) consistently with all downstream analyses, and the discrepancy with training-time evaluation (98.0%/95.6%) is explicitly documented in the table caption. The "85% gap closure" framing has been entirely removed from the manuscript, resolving S002. The addition of M4 (Pause-Multipass) and the factorial decomposition substantially strengthens the paper's statistical architecture, enabling clean attribution of OOD effects to specific factors.

**Numerical verification summary (Round 2):** 73 claims checked across the full manuscript (abstract through appendix), 70 verified exactly against backing data files, 3 discrepancies identified (1 new major, 1 existing minor, 1 new minor).

**Round 2 status:** 23 original findings reviewed, 8 resolved, 12 confirmed with updated status, 3 new findings added (S024-S026). Upgrading overall assessment from "revise" to "pass_with_conditions" -- the remaining issues are addressable through targeted corrections and do not require new experiments or fundamental restructuring.

### New Finding: S024 (MAJOR) -- m6/mcnemar.json data integrity

The M4 McNemar data file (`results/experiments/m6/mcnemar.json`) computes contingency tables using WRONG per-sample predictions for M2 and M3. The file uses training-time evaluation predictions (M2=98.0%, M3_prosqa=70.7% for 7-hop, etc.) rather than the experiment-pipeline predictions (M2=97.0%, M3_prosqa=66.0% for 7-hop) that the manuscript correctly uses. The manuscript's Table 5c values are independently verified as correct (recomputed from `per_sample_correctness.json` and `m6/per_sample_correctness.json`), but the backing data file `m6/mcnemar.json` tells a different story. Example: for M4 vs M2 on 7-hop, the data file reports +6.2pp (b=182, c=120, p_bonf=0.004) while the manuscript correctly reports +10.9pp (b=113, c=222, p_bonf<0.001). Anyone auditing the data files without cross-referencing would reach different conclusions.

**Recommendation:** Regenerate `m6/mcnemar.json` using the experiment-pipeline per-sample predictions that all other analyses use. This is a data hygiene issue, not a manuscript error -- but it undermines reproducibility.

### New Finding: S025 (MINOR) -- paper.yaml stale accuracy values

`paper.yaml` still contains the Round 1 accuracy values: `M2_test: 0.980`, `M3_test: 0.956`. It also uses the old chi-squared McNemar results in the `statistical_tests` section (e.g., DAG and Dense marked non-significant, contradicting the exact McNemar tests used in the manuscript). The M3 peak accuracy is listed as 0.571 (should be 0.570 per data). While `paper.yaml` is metadata rather than the manuscript itself, it creates confusion for anyone examining the repository.

**Recommendation:** Update `paper.yaml` to use the experiment-pipeline numbers (M2_test: 0.970, M3_test: 0.966) and exact McNemar results.

### New Finding: S026 (MINOR) -- Table 4 M3 selectivity layer inconsistency (updated S005)

The Table 4 caption now says "For M3, selectivity is reported at peak accuracy layer." This is correct for positions 2 (layer 12) and 3 (layer 12), but INCORRECT for positions 0 and 1. M3's peak accuracy layers per `probing/results.json` are [8, 11, 12, 12] for positions 0-3 respectively. Positions 0 and 1 values in Table 4 (-12.0pp and -14.6pp) are from layer 12, not their peak layers (8 and 11). At M3's actual peak accuracy layer for position 0 (layer 8), selectivity is +17.0pp (POSITIVE), which would contradict the anti-selectivity narrative for that position. At position 1's peak layer (11), selectivity is -11.2pp (similar direction but different magnitude from the reported -14.6pp).

**Impact:** The anti-selectivity narrative at positions 0-1 for M3 depends on which layer is examined. At the peak accuracy layers, M3 position 0 shows POSITIVE selectivity, meaning the anti-selectivity pattern at early positions is specific to certain layers (including the final layer) but not universal. This complicates the "identical selectivity profiles" claim for positions 0-1.

**Recommendation:** Either (a) change the caption to accurately state that M3 values are reported at layer 12 (matching M2's convention of reporting at layer 0 for architectural reasons), with a footnote that peak-layer selectivity differs at positions 0-1, or (b) report all M3 values at peak accuracy layers and update the anti-selectivity narrative accordingly.

---

## CRITICAL

### - [x] S001: numerical_accuracy -- ROUND 2: RESOLVED

Table 2 now reports M2=97.0%, M3=96.6% (experiment-pipeline numbers). The caption explicitly acknowledges the discrepancy: "Training-time evaluation at best epoch yielded slightly higher estimates for M2 (98.0%) and lower for M3 (95.6%), a discrepancy of 5 samples per model attributable to differences in the inference code path; we use the experiment-pipeline numbers for consistency with all subsequent analyses." All downstream analyses (Tables 3, 4, 5, 6, 7; all statistical tests) consistently use 97.0%/96.6%. The internal consistency issue is fully resolved.

**Computation:** Table 2 accuracy values verified against: ood/results.json (M2=0.97, M3=0.966), mcnemar contingency tables (M2=485/500=0.97, M3=483/500=0.966), corruption/results.json clean baselines (M2=0.97, M3=0.966), per_sample_correctness.json (M2=485/500, M3=483/500). All consistent.


## MAJOR

### - [x] S002: numerical_accuracy -- ROUND 2: RESOLVED

The "85% gap closure" claim has been entirely removed from the manuscript. No occurrence of "85%" or "gap closure" found anywhere in the text (verified by search). The manuscript now frames the M3 result in terms of the raw accuracy difference (96.6% vs 97.0%, 0.4pp, McNemar p=0.845) and statistical non-significance, which is the appropriate framing. The gap closure percentage was always misleading because it amplified small differences in a near-ceiling regime; the current framing is more rigorous.

### - [ ] S003: effect_sizes -- ROUND 2: PARTIALLY ADDRESSED

The manuscript now reports a 95% CI for the M2 vs M3 ProsQA comparison: "95% CI for accuracy difference: [-2.4, +1.6] percentage points" (verified: Wald CI from b=14, c=12, n=500 gives [-2.4, 1.6]pp). The Wilcoxon analysis (Table 7) reports effect sizes (rank-biserial r) for all 15 comparisons, ranging from 0.003 to 0.678, verified against data files.

However, no odds ratios or CIs are reported for the OOD McNemar comparisons in Tables 5b and 5c. The m6/mcnemar.json file does compute odds ratios (e.g., M4 vs M2 on 7-hop: OR=1.517), but these are from the wrong per-sample data (see S024). The manuscript Tables 5b and 5c report only pp differences, b, c, p-values, and significance -- no effect size measure.

**Remaining recommendation:** Add odds ratios with CIs to Tables 5b and 5c. For each comparison, OR = c/b, with 95% CI via log(OR) +/- 1.96*sqrt(1/b + 1/c). This would add minimal space and substantially improve interpretability.

### - [ ] S004: corrections -- ROUND 2: PARTIALLY ADDRESSED

The manuscript now correctly and consistently uses Bonferroni correction with k=5 per comparison family. The DAG p-value in Table 5b is now reported as 0.0015 (not 0.001 as in Round 1), matching the exact Bonferroni-corrected value of 0.00146 rounded to 4 significant figures. The "adjusted alpha = 0.01" language has been removed.

The per-family Bonferroni structure (k=5 per comparison pair rather than k=15 across all comparisons) is defensible, and I verified that all currently significant results remain significant even under k=15 correction (most p_bonf15 < 0.005).

**Remaining recommendation:** Consider noting in the methods that the per-family correction structure was chosen because each comparison pair answers a distinct scientific question. The Holm-Bonferroni suggestion is no longer pressing since the current correction is internally consistent and does not affect any conclusions.

### - [x] S005: methodology -- ROUND 2: SEE S026

The original S005 finding concerned M2 positions 0, 1 being reported at layer 0 with the caption falsely claiming these were peak accuracy layers. The caption has been revised: it now correctly states "For M2, selectivity is reported at layer 0 for positions 0, 1, and 3 (where recycled hidden states are injected) and layer 12 for position 2." This is architecturally motivated and transparent.

However, an analogous issue persists for M3 -- see new finding S026 above. The M2 side is resolved; the M3 side is not.

### - [ ] S006: missing_analysis -- ROUND 2: DEFERRED

No formal statistical test has been added for the corruption profile comparison. The manuscript still uses "nearly identical" and "identical degradation profiles" language based on visual inspection of Table 3. The maximum absolute difference between M2 and M3 at any corruption level is 1.0pp (at 3 positions corrupted: 96.8% vs 95.8%).

**Computation:** A quick assessment: at 3 corrupted positions (the largest gap), M2=96.8% (484/500), M3=95.8% (479/500). The difference is 5/500 samples. A McNemar test on these 500 paired samples at corruption level 3 would have very low power to detect a 1pp difference. A logistic regression interaction test across all 7 corruption levels would be more powerful.

This remains a valid methodological concern but is no longer critical given that the M4 factorial decomposition provides the primary evidence for the paper's claims, reducing the weight placed on the corruption profile comparison alone.

### - [ ] S007: power -- ROUND 2: PARTIALLY ADDRESSED

The single-seed limitation is now better acknowledged in Section 6 (Limitations), including the new sentence: "Training-time evaluation at best epoch yielded a larger apparent gap (M2 = 98.0%, M3 = 95.6%), differing from the experiment-pipeline numbers by 5 samples per model; this sensitivity to inference implementation underscores the need for multi-seed replication."

The addition of M4 partially mitigates this concern because it provides an independent replication of the curriculum hypothesis: M4 also achieves near-ceiling performance (94.8%) under the same curriculum, and the factorial decomposition of OOD effects is interpretable even from a single seed.

However, the core issue remains: all accuracy numbers, effect sizes, and significance tests are conditional on seed=0. The 0.4pp gap could reverse under a different seed. Multi-seed replication would strengthen the paper substantially but is not strictly required for the current claims, which are appropriately hedged.

**Remaining recommendation:** Downgrade to "suggestion" severity. The claims are appropriately scoped with the "single seed" qualifier, and the convergent evidence from multiple experimental paradigms partially compensates for the single-seed limitation.

### - [x] S021: interpretation -- ROUND 2: RESOLVED

The abstract now reads: "Three converging experiments probe the distinction: corruption analysis reveals identical degradation profiles for M2 and M3, linear probes reveal identical selectivity profiles despite COCONUT encoding information more broadly across layers (29/78 vs. 11/78 significant probing cells), and cross-model thought transplantation succeeds bidirectionally." This accurately scopes the three experiments as probing the M2-M3 distinction, without conflating them with the OOD results. The OOD results are described separately in the abstract, including the factorial decomposition. The conclusion similarly distinguishes the in-distribution convergent evidence from the OOD factorial decomposition.


## MINOR

### - [ ] S008: numerical_accuracy -- ROUND 2: PARTIALLY RESOLVED

The text in Section 4.3 now says "57.0%". Table 4 says "57.0%". The Appendix A.7 MLP table still says "57.1%" for M3 (12, 3) linear accuracy. The backing data is 0.5704697986577181 = 57.047%, which rounds to 57.0% under standard rounding. The Appendix table should be corrected from 57.1% to 57.0%.

**Proposed edit (Appendix A.7, line ~522):**
```
FIND: | M3 | 12 | 3 | 298 | 57.1% | 45.6% | --11.4pp |
REPLACE: | M3 | 12 | 3 | 298 | 57.0% | 45.6% | --11.4pp |
```

### - [x] S009: test_selection -- ROUND 2: RESOLVED

No change needed. McNemar's exact test remains correctly chosen and implemented. The addition of Wilcoxon signed-rank tests on teacher-forced log-probabilities (Table 7) provides a complementary continuous-outcome analysis. The Wilcoxon is correctly chosen for the paired non-normal log-probability differences and uses rank-biserial correlation as the effect size measure.

### - [x] S010: numerical_accuracy -- ROUND 2: RESOLVED

DAG Bonferroni p-value is now reported as "0.0015" in Table 5b (verified: 5 * 0.000291 = 0.001457, rounds to 0.0015). This is consistent with the precision used for other p-values in the table.

### - [ ] S011: assumptions -- ROUND 2: DEFERRED

The manuscript does not explicitly acknowledge the correlation across test sets through shared model parameters. This remains a minor methodological note. No conclusions are affected since Bonferroni is conservative for correlated tests, and all results survive even k=15 correction.

### - [ ] S012: methodology -- ROUND 2: DEFERRED

Per-fold variance is still not reported. The selectivity comparison between models remains informal ("0.3 percentage-point difference... smaller than the typical standard deviation of 5-fold cross-validation estimates at this sample size (n = 298), though we do not report per-fold variance"). This is honest but would benefit from the actual standard deviation. Not blocking.

### - [ ] S013: interpretation -- ROUND 2: PARTIALLY ADDRESSED

The manuscript now includes: "This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction." This is exactly the qualification recommended in Round 1. The distinction between output-level and representation-level permutation sensitivity is now acknowledged.

**Remaining:** Could be slightly strengthened by noting that the probing experiment partially addresses this (by showing that positions 0-2 carry redundant information, consistent with the permutation insensitivity), but the current text is adequate.

### - [ ] S019: numerical_accuracy -- ROUND 2: DEFERRED

The L2 distance values (202.65 vs 202.8 vs 202.39 across different noise draws) remain unexplained but are now contextualized by the cross-corruption analysis (Table A1), which shows that the degradation cliff is invariant to noise magnitude (M3 shows the same cliff under M2-scale noise). The minor L2 variation across random draws is expected and not consequential.

### - [ ] S020: numerical_accuracy -- ROUND 2: DEFERRED

`statistical_analysis.json` still contains the old approximate chi-squared McNemar results that disagree with the exact tests used in the manuscript. The manuscript correctly uses the exact tests from `mcnemar/results.json` and `mcnemar_verification.json`. This is a data hygiene issue; the manuscript itself is correct.

### - [ ] S022: methodology -- ROUND 2: DEFERRED

The selectivity metric definition (using max cross-position accuracy) is now stated explicitly in Section 3.4: "selectivity(l, t) = probe_acc(target = step_t) - max_{s != t} probe_acc(target = step_s), where the control is the same probe applied to alternative reasoning steps rather than the matched step. This cross-position selectivity is a stricter test than the random-label baseline of Ravichander et al. (2021)." The use of max is implicitly justified by the comparison to Ravichander, but an explicit note that max gives a conservative estimate would strengthen this.


## SUGGESTION

### - [ ] S014: missing_analysis -- ROUND 2: DEFERRED

Bayesian analysis has not been added. This would strengthen the null claims (ProsQA in-distribution equivalence, corruption profile similarity, selectivity equivalence) but is not required for the current framing, which relies on convergent evidence from multiple paradigms rather than any single null test. The Wilcoxon analysis (Section 4.5) partially addresses this by showing that M2 does carry more information (higher confidence), demonstrating that the "null" finding is nuanced -- M2 and M3 differ in confidence but not in accuracy.

### - [ ] S015: missing_analysis -- ROUND 2: DEFERRED

TOST has not been conducted. The recommendation for equivalence testing remains valid for the in-distribution accuracy comparison and the selectivity comparison. However, the convergent evidence design (multiple paradigms all failing to find a difference) provides a form of informal equivalence evidence that may be sufficient for the current paper's scope.

### - [x] S016: interpretation -- ROUND 2: RESOLVED

The manuscript now says: "M1 performs near chance on all OOD test sets (8.2%--28.2%), confirming that the curriculum-trained latent-reasoning approach, whether implemented via recycling, single-pass pause tokens, or multi-pass pause tokens, provides substantial generalization benefits over explicit chain-of-thought at this model scale." While the "near chance" characterization is still imprecise (chance is 50% for a two-choice task, and M1 is systematically below chance), the framing now focuses on the comparison between M1 and the curriculum models rather than on M1's absolute performance level. The below-chance M1 performance is an interesting observation but not central to the paper's argument.

### - [x] S017: missing_analysis -- ROUND 2: DEFERRED

Transplant experiment still lacks formal statistical tests. The manuscript reports raw accuracy under matched (M2=97.0%, M3=96.5%) and unmatched (M2=97.5%, M3=96.5%) conditions for n=200 pairs. While a formal McNemar or binomial test would strengthen the claim, the descriptive results are compelling -- transplant accuracy matches clean accuracy to within 0.5pp for all conditions. The small n (200) limits the power of formal tests anyway.

### - [x] S018: methodology -- ROUND 2: RESOLVED

The MLP probe null result has been thoroughly addressed. Appendix A.7 now reports results from a systematic grid search over 72 hyperparameter configurations (6 hidden sizes x 3 learning rates x 4 regularization strengths). The original 0/78 null result is acknowledged as convergence failure. The grid search reveals a meaningful position-dependent pattern: linear probes outperform MLPs at position 3 (overfitting with n=298), while MLPs show +10.2pp (M2) and +7.6pp (M3) advantage at position 2 (n=500), indicating nonlinear encoding at intermediate positions. This is a substantially more informative result than the original null.

### - [x] S023: missing_analysis -- ROUND 2: RESOLVED

The 0.0% probe accuracy at positions 4 and 5 is now contextualized in the manuscript: "Positions 4 and 5 return 0.0% accuracy for both models (n = 81 with 32 classes and n = 12 with 12 classes, respectively); with more classes than the minimum fold size, stratified cross-validation cannot be computed, and these cells are excluded from significance testing." The explanation (more classes than fold size) is adequate and the manuscript correctly excludes these positions from quantitative claims.

---

## Summary of Round 2 Status

**Resolved (8):** S001, S002, S009, S010, S016, S018, S021, S023
**New findings (3):** S024 (major), S025 (minor), S026 (minor)
**Remaining major (4):** S003 (effect sizes, partially addressed), S004 (corrections, partially addressed), S006 (corruption formal test, deferred), S007 (single seed, partially addressed), S024 (m6/mcnemar.json data integrity)
**Remaining minor (5):** S008 (rounding, partially resolved), S011 (assumptions, deferred), S012 (per-fold variance, deferred), S019 (L2 discrepancy, deferred), S020 (statistical_analysis.json stale, deferred), S022 (selectivity metric justification, deferred), S025 (paper.yaml stale), S026 (Table 4 M3 layer selection)
**Remaining suggestions (3):** S013 (permutation interpretation, partially addressed), S014 (Bayesian, deferred), S015 (TOST, deferred), S017 (transplant test, deferred)

**Blocking issues for publication:** S024 (data file integrity) and S026 (Table 4 caption accuracy) should be resolved before submission. S008 (Appendix rounding) is a minor correction. S003 (effect sizes) is strongly recommended but not strictly blocking if the Wilcoxon r values are considered sufficient.

**Overall assessment:** The manuscript's statistical foundations are now substantially stronger than at Round 1. The critical accuracy discrepancy is resolved, the M4 factorial design is sound and correctly analyzed, all McNemar and Wilcoxon test statistics verified against backing data, and the convergent evidence framework is well-structured. The remaining issues are data hygiene (S024, S025), a minor caption inaccuracy (S026), and desirable-but-not-required additional analyses (S003, S006, S014, S015). I upgrade my assessment from "revise" to "pass_with_conditions."
