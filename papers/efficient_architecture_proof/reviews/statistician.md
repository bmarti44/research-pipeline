# Statistician Review

**Assessment:** pass
**Date:** 2026-02-13T23:45:00Z
**Round 2 Date:** 2026-02-16T20:00:00Z
**Round 3 Date:** 2026-02-16T22:00:00Z
**Round 4 Date:** 2026-02-17T18:00:00Z
**Round 5 Date:** 2026-02-17T22:30:00Z

## Round 5 Review

This is the final review round. I performed a comprehensive re-verification of every numerical claim in the manuscript against backing data files. All 85+ numerical assertions checked -- accuracy values, McNemar contingency tables and p-values, Wilcoxon effect sizes and p-values, probing accuracies, selectivity values, corruption profiles, permutation power, confidence intervals, and OOD accuracy -- are confirmed correct. Both prior open findings (S026 and S027) have been resolved in the current manuscript. No new statistical issues were found.

**Overall assessment: PASS.**

---

### Round 5 Full Numerical Verification

**Methodology:** Loaded and cross-referenced the following data files against the manuscript:
- `results/experiments/ood/results.json`
- `results/experiments/corruption/results.json`
- `results/experiments/probing/results.json`
- `results/experiments/mcnemar/results.json`
- `results/experiments/per_sample_correctness.json`
- `results/experiments/m6/per_sample_correctness.json`
- `results/selectivity_recomputed.json`
- `results/permutation_power.json`
- `results/mcnemar_verification.json`
- `results/experiments/wilcoxon_teacher_forced_m3_vs_m6.json` (M2 vs M4)
- `results/experiments/wilcoxon_teacher_forced_m3_vs_m5.json` (M2 vs M3)
- `results/experiments/wilcoxon_teacher_forced_m5_vs_m6.json` (M3 vs M4)
- `results/experiments/probing_corrected/m3_linear_perm.json` (M2 corrected significance)
- `results/experiments/probing_corrected/m5_linear_perm.json` (M3 corrected significance)

**Verification results (all verified against backing data):**

1. **Table 2 accuracy values:** M1=83.0%, M2=97.0% (485/500), M3=96.6% (483/500), M4=94.8% (474/500). All match.
2. **M2 vs M3 McNemar (ProsQA):** b=14, c=12, p=0.845019 (exact binomial). Manuscript: p=0.845. Match.
3. **M4 vs M2 McNemar (ProsQA):** b=10 (M4-only), c=21 (M2-only), p=0.070756, p_Bonf=0.353778. Manuscript: p=0.071, p_Bonf=0.354. Match.
4. **M4 vs M3 McNemar (ProsQA):** b=10, c=19, p=0.136046, p_Bonf=0.680230. Manuscript: p_Bonf=0.680. Match.
5. **95% CI for M2-M3 ID difference:** Wald CI = [-2.4, +1.6]pp. Manuscript: [-2.4, +1.6]pp. Match.
6. **Table 5 factorial decomposition (M4 vs M2):**
   - 7-hop: +10.9pp, b=222, c=113, p_Bonf<0.001. Match.
   - 8-hop: +7.7pp, b=188, c=111, p_Bonf<0.001. Match.
   - DAG: +0.6pp, b=182, c=176, p_Bonf=1.0. Match.
   - Dense: +3.6pp, b=186, c=150, p_Bonf=0.280. Match.
7. **Table 5 factorial decomposition (M4 vs M3):**
   - 7-hop: +1.5pp, b=139, c=124, p_Bonf=1.0. Match.
   - 8-hop: +0.1pp, b=141, c=140, p_Bonf=1.0. Match.
   - DAG: +7.9pp, b=235, c=156, p_Bonf<0.001. Match.
   - Dense: -3.6pp, b=157, c=193, p_Bonf=0.306. Match.
8. **Table A8 (M3 vs M2 OOD):** All 5 rows verified -- pp differences, b, c values, and corrected p-values all match.
9. **Table 6a Wilcoxon (M2 vs M4):** All 5 test sets verified -- r values (0.678, 0.109, 0.082, 0.073, 0.118), p_Bonf values, and directions all match.
10. **Table 6b Wilcoxon (M2 vs M3):** All 5 test sets verified -- r values (0.591, 0.006, 0.006, 0.003, 0.113), p_Bonf values, and directions all match.
11. **Table A10 Wilcoxon (M3 vs M4):** All 5 test sets verified -- r values (0.286, 0.142, 0.120, 0.136, 0.021), p_Bonf values, and directions all match.
12. **Table 3 probing summary:** Peak accuracy M2=55.4% (layer 0, pos 3), M3=57.0% (layer 12, pos 3). Match. Significant cells: 29/78 (M2), 11/78 (M3) from corrected permutation tests. Match.
13. **Table 3 selectivity:** M2 position 3: +52.0pp. M3 position 3: +52.3pp. M2 positions 0-1: -15.6pp, -10.6pp. M3 positions 0-1: -12.0pp, -14.6pp. All match.
14. **Thought-vs-input advantage:** M2=10.5%, M3=4.0%. Match.
15. **Table A1 corruption forward:** All 7 corruption levels for M2 and M3 match backing data exactly.
16. **Single-position corruption (Table A4):** M2 pos 3 = 57.6%, M3 pos 3 = 57.8%. Match.
17. **Transplant accuracy (Table A2):** M2=97.0%, M3=96.5%. Match.
18. **Permutation power:** min_detectable_95 = 0.0005990 = 0.060%. Manuscript: "excludes true flip rate >0.06% at 95% confidence". Match.
19. **M4 OOD accuracy (Table 4):** 7-hop=76.9%, 8-hop=75.2%, DAG=59.8%, Dense=64.8%. All match.
20. **M1 OOD accuracy (Table 4):** 7-hop=10.7%, 8-hop=8.2%, DAG=28.2%, Dense=14.1%. All match.
21. **Abstract numerical claims:** 97.0%, 96.6%, p=0.845, [-2.4, +1.6], 94.8%, p=0.354, 29/78, 11/78, +10.9pp, +7.9pp, r=0.678. All verified.
22. **Wilcoxon median probabilities (Section 4.5):** M2=99.998% (data: 0.999980509661182), M4=99.949% (data: 0.9994942125181384). Match.
23. **Wilcoxon 7-hop M2 vs M4:** r=0.109, p_Bonf=0.003 (data: 0.00287). Match within rounding.
24. **Wilcoxon 8-hop M2 vs M4:** r=0.082, p_Bonf=0.049 (data: 0.04949). Match.

---

### Previous Finding Status (S001--S027)

- [x] S001: numerical_accuracy -- ROUND 5: VERIFIED RESOLVED -- All accuracy values consistent across Tables 2, 4, 5, A1-A4, A8, abstract, and conclusion. Spot-checked 20+ accuracy values; all match backing data.

- [x] S002: numerical_accuracy -- ROUND 5: VERIFIED RESOLVED -- No "85%" or "gap closure" text found.

- [ ] S003: effect_sizes -- ROUND 5: DEFERRED (SUGGESTION) -- Wilcoxon r values provide continuous effect sizes for all 15 comparisons. McNemar tables report b and c from which OR = c/b is computable. Adding explicit OR columns remains a nice-to-have for reader convenience. Non-blocking.

- [ ] S004: corrections -- ROUND 5: DEFERRED (SUGGESTION) -- Per-family Bonferroni k=5 is internally consistent, correctly applied, and defensible. All significant results survive even k=15 correction. Non-blocking.

- [x] S005: methodology -- ROUND 5: VERIFIED RESOLVED -- M2 selectivity layer convention is correctly documented.

- [ ] S006: missing_analysis -- ROUND 5: DEFERRED (SUGGESTION) -- No formal corruption profile similarity test. The factorial decomposition via M4 is the primary evidence; corruption comparison is supporting. Non-blocking.

- [ ] S007: power -- ROUND 5: DEFERRED (SUGGESTION) -- Single-seed limitation acknowledged in Section 6. Convergent evidence from 4 paradigms partially compensates. Non-blocking.

- [x] S008: numerical_accuracy -- ROUND 5: VERIFIED RESOLVED -- All three occurrences of M3 peak accuracy (Section 4.3, Table 3, Appendix A.10) read 57.0%, consistent with backing data (0.5704697986577181 rounds to 57.0%).

- [x] S009: test_selection -- ROUND 5: VERIFIED RESOLVED -- McNemar and Wilcoxon correctly chosen and applied.

- [x] S010: numerical_accuracy -- ROUND 5: VERIFIED RESOLVED -- DAG p_Bonf=0.0015 verified (5 * 0.000291 = 0.001457, rounds to 0.0015).

- [ ] S011: assumptions -- ROUND 5: DEFERRED (SUGGESTION) -- Correlation across test sets through shared parameters not acknowledged. Non-blocking since Bonferroni is conservative.

- [ ] S012: methodology -- ROUND 5: DEFERRED (SUGGESTION) -- Per-fold cross-validation variance not reported for selectivity. Non-blocking.

- [ ] S013: interpretation -- ROUND 5: DEFERRED (SUGGESTION) -- Permutation insensitivity appropriately qualified with "This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction."

- [ ] S014: missing_analysis -- ROUND 5: DEFERRED (SUGGESTION) -- Bayesian analysis not added. Non-blocking given convergent evidence framework.

- [ ] S015: missing_analysis -- ROUND 5: DEFERRED (SUGGESTION) -- TOST not conducted. Non-blocking.

- [x] S016: interpretation -- ROUND 5: VERIFIED RESOLVED.

- [ ] S017: missing_analysis -- ROUND 5: DEFERRED (SUGGESTION) -- Transplant lacks formal test. Descriptive results compelling (accuracy within 0.5pp of clean). Non-blocking.

- [x] S018: methodology -- ROUND 5: VERIFIED RESOLVED -- MLP probe grid search adequately documented in A.10.

- [ ] S019: numerical_accuracy -- ROUND 5: DEFERRED (SUGGESTION) -- Minor L2 variation across noise draws. Non-consequential.

- [ ] S020: numerical_accuracy -- ROUND 5: DEFERRED (SUGGESTION) -- `statistical_analysis.json` stale but not referenced by manuscript. Non-blocking.

- [x] S021: interpretation -- ROUND 5: VERIFIED RESOLVED.

- [ ] S022: methodology -- ROUND 5: DEFERRED (SUGGESTION) -- Max-based selectivity metric adequately defined in Section 3.4.

- [x] S023: missing_analysis -- ROUND 5: VERIFIED RESOLVED -- Position 4-5 exclusion explained.

- [x] S024: data_hygiene -- ROUND 5: VERIFIED RESOLVED (downgraded in Round 3) -- `m6/mcnemar.json` carries deprecation header. Manuscript Table 5c values independently verified from per-sample predictions.

- [x] S025: data_hygiene -- ROUND 5: PARTIALLY ADDRESSED -- `paper.yaml` now has correct M2_test=0.970 and M3_test=0.966. Residual stale values (McNemar flags, M3 peak accuracy=0.571) remain in repository metadata but do not appear in the manuscript. Non-blocking for publication.

- [x] S026: caption_accuracy -- ROUND 5: VERIFIED RESOLVED -- The original problematic Table 3 caption claiming M3 selectivity at "peak accuracy layers" has been removed. The current Table 3 caption (line 92) makes no such claim. A.11 (line 407) now correctly states "Table 3 reports selectivity at layer 12 for all M3 positions" and documents the layer-dependent selectivity at alternative layers (+17.0pp at layer 8 for position 0, -11.2pp at layer 11 for position 1). This is accurate, transparent, and matches the backing data.

- [x] S027: interpretation -- ROUND 5: VERIFIED RESOLVED -- The dense result description (line 132) now reads "cancel additively, producing a near-zero net difference that masks opposing underlying forces rather than indicating an interaction." This replaces the previous "creating an interaction effect" framing and correctly characterizes the symmetric +3.6pp/-3.6pp pattern as additive cancellation.

---

### Round 5 Summary Table

| Finding | Severity | Category | Status |
|---------|----------|----------|--------|
| S001 | Critical | numerical_accuracy | RESOLVED (Round 2) |
| S002 | Major | numerical_accuracy | RESOLVED (Round 2) |
| S003 | Suggestion | effect_sizes | DEFERRED |
| S004 | Suggestion | corrections | DEFERRED |
| S005 | Major | methodology | RESOLVED (Round 2) |
| S006 | Suggestion | missing_analysis | DEFERRED |
| S007 | Suggestion | power | DEFERRED |
| S008 | Minor | numerical_accuracy | RESOLVED (Round 3) |
| S009 | Minor | test_selection | RESOLVED (Round 2) |
| S010 | Minor | numerical_accuracy | RESOLVED (Round 2) |
| S011 | Suggestion | assumptions | DEFERRED |
| S012 | Suggestion | methodology | DEFERRED |
| S013 | Suggestion | interpretation | DEFERRED |
| S014 | Suggestion | missing_analysis | DEFERRED |
| S015 | Suggestion | missing_analysis | DEFERRED |
| S016 | Suggestion | interpretation | RESOLVED (Round 2) |
| S017 | Suggestion | missing_analysis | DEFERRED |
| S018 | Suggestion | methodology | RESOLVED (Round 2) |
| S019 | Suggestion | numerical_accuracy | DEFERRED |
| S020 | Suggestion | numerical_accuracy | DEFERRED |
| S021 | Major | interpretation | RESOLVED (Round 2) |
| S022 | Suggestion | methodology | DEFERRED |
| S023 | Minor | missing_analysis | RESOLVED (Round 2) |
| S024 | Minor | data_hygiene | RESOLVED (Round 3, deprecation header) |
| S025 | Minor | data_hygiene | PARTIALLY ADDRESSED (non-blocking) |
| S026 | Suggestion | caption_accuracy | RESOLVED (Round 5) |
| S027 | Minor | interpretation | RESOLVED (Round 5) |

**Critical findings:** 0 open
**Major findings:** 0 open
**Minor findings:** 1 partially addressed (S025: paper.yaml residual stale values) -- non-blocking for manuscript
**Suggestions:** 12 open -- all non-blocking, representing possible future improvements

**Overall assessment: PASS.** The manuscript's statistical foundations are sound. Every numerical claim verified. All critical, major, and minor findings have been resolved or confirmed non-blocking. The 12 remaining suggestion-level items represent desirable enhancements (Bayesian analysis, TOST equivalence testing, odds ratios in McNemar tables, per-fold variance for selectivity) that would strengthen the paper but are not required for the current submission. The manuscript is statistically ready for publication.

---

## Round 4 Review

This is a fresh review incorporating external reviewer feedback on four specific items (#1, #2, #3, #8), plus a complete re-verification of all previous findings (S001--S026). All numerical claims were re-checked against backing data files.

**Overall assessment: PASS.**

The manuscript's statistical claims remain accurate. The external feedback items identify genuine nuance issues, but the manuscript already handles most of them adequately. One item (#1) is well-handled in Section 4.5 but could be slightly tightened in the abstract/conclusion; one item (#2) identifies a legitimate interpretive gap in the dense results; one item (#3) is already handled with appropriate transparency; and item #8's minor issues are largely non-problems or already addressed. I identify one new finding (S027) and update S026.

---

### External Feedback Item #1: Conclusion oversells ID Wilcoxon result

**Concern:** The conclusion says "miscalibrated confidence" but the ID confidence difference is trivial in absolute terms (M2 median=99.998%, M4 median=99.949%). The large r=0.678 reflects rank ordering, not practically meaningful absolute differences on ID data. The practical significance is in the OOD extrapolation pattern, not the ID magnitude.

**Assessment: ADEQUATELY HANDLED in Section 4.5; MINOR gap in abstract/conclusion.**

Section 4.5 (line 138) already contains the exact qualification the external reviewer wants: "though both achieve near-ceiling median probabilities (M2: 99.998%, M4: 99.949%). The large rank-biserial correlation reflects consistent paired rank ordering, not practically meaningful absolute differences." This is a textbook-quality contextualization of rank-based effect sizes in a ceiling regime.

The abstract (line 6) says "Recycled content also produces higher confidence that becomes miscalibrated on extended chains." The conclusion (line 178) says "Recycled content also produces miscalibrated confidence on extended chains." Neither the abstract nor the conclusion claims ID miscalibration -- both specifically scope "miscalibrated" to the OOD/extended-chain regime, which is the correct claim. On OOD chains, M2 IS simultaneously more confident and less accurate than M4 (7-hop: r=0.109, p_Bonf=0.003, but M2 accuracy 66.0% vs M4 76.9%), which is a genuine calibration failure.

Verified against data: `wilcoxon_teacher_forced_m3_vs_m6.json` confirms M2 median prob on ProsQA = 0.999980509661182 (99.998%), M4 = 0.9994942125181384 (99.949%), r = 0.678. The 0.005pp absolute difference is indeed trivial. On 7-hop: M2 median prob = 0.974039683283398, M4 = 0.966618355268511. The miscalibration claim is about the direction of the accuracy-confidence coupling reversing (M2 more confident but less accurate), not about the magnitude of the confidence gap.

**Verdict:** No manuscript edit required. The abstract/conclusion correctly scope the miscalibration claim to extended chains. Section 4.5 explicitly contextualizes the ID r=0.678 as rank ordering. If the author wishes to be maximally careful, a parenthetical in the conclusion like "(higher confidence accompanied by lower accuracy on 7-hop and 8-hop chains)" would make the scope explicit, but this is a suggestion, not a requirement.

---

### External Feedback Item #2: Table 5 dense result underinterpreted

**Concern:** M4-M2 = +3.6pp and M4-M3 = -3.6pp on dense are symmetric and suggest additive cancellation (recycled content helps topology but hurts chains, and dense requires both), not an interaction effect.

**Assessment: PARTIALLY ADDRESSED. The manuscript notes additivity but does not fully interpret dense.**

The manuscript (line 132) says: "Dense graphs may require both extended chains and richer topology simultaneously, creating an interaction effect that the additive factorial decomposition cannot capture." This is a reasonable hypothesis but the external reviewer's point is more specific and arguably more parsimonious: the symmetry (+3.6pp and -3.6pp) is exactly what you would expect from additive cancellation of two opposing effects, NOT an interaction. If recycled content hurts by ~3.6pp (chain-length penalty) and sequential processing helps by ~3.6pp (topological benefit), and dense requires both, the two effects cancel to produce the observed zero net difference for both M4 vs. M2 and M4 vs. M3 comparisons.

The distinction matters: "interaction effect" implies the factors combine non-additively (i.e., the effect of one factor depends on the level of the other). "Additive cancellation" implies the factors combine additively but in opposite directions, producing a net-zero. The data are more consistent with additive cancellation: the magnitudes are symmetric (both 3.6pp), and neither comparison reaches significance (p=0.280 and p=0.306). An interaction would typically produce asymmetric magnitudes.

The manuscript's current framing ("creating an interaction effect that the additive factorial decomposition cannot capture") is slightly misleading because the factorial decomposition IS capturing an additive pattern on dense -- it shows two opposing effects of equal magnitude that cancel. The decomposition works fine; it just yields a null net result because the effects offset.

**New Finding: S027 (MINOR) -- Dense interpretation imprecision**

**Proposed edit (line 132):**
```
FIND: Dense graphs may require both extended chains and richer topology simultaneously, creating an interaction effect that the additive factorial decomposition cannot capture.
REPLACE: Dense graphs show symmetric opposing effects: recycled content incurs the same ~3.6pp penalty as on chain-length tasks, while the absence of sequential processing incurs a similar ~3.6pp cost, producing additive cancellation that leaves neither factorial comparison significant.
```

This is a more precise characterization of the data pattern and avoids the potentially misleading "interaction" framing.

---

### External Feedback Item #3: M4 early plateau -- paper too generous with Bonferroni

**Concern:** M4 vs M2 uncorrected p = 0.071 is marginal. The paper uses Bonferroni to dismiss it (p_Bonf = 0.354). Check whether the paper is appropriately transparent.

**Assessment: ADEQUATELY HANDLED. The manuscript is transparent about both the uncorrected and corrected p-values.**

Line 71 reports: "M4 (pause-multipass) reaches 94.8%; the 2.2pp gap from M2 does not reach significance after Bonferroni correction (p = 0.071, p_Bonf = 0.354, 31 discordant pairs)." This explicitly reports the uncorrected p-value (0.071) alongside the corrected value (0.354), allowing the reader to form their own judgment.

Verified against data: From per-sample correctness files, M4 vs M2 on ProsQA ID has b=21 (M4-only correct), c=10 (M2-only correct). Exact McNemar: `binomtest(10, 31, 0.5)` gives p = 0.0708 (two-sided), matching the reported p = 0.071 within rounding. Bonferroni: 5 * 0.0708 = 0.354. All verified.

The manuscript goes further -- line 82 provides an extensive, honest treatment of the M4 plateau: "M4's best epoch (30) occurs 13--19 epochs earlier than M2 (49) and M3 (43). Whether this reflects an inherent capacity limit of the multi-pass fixed-embedding architecture, or indicates that M4 would benefit from different hyperparameters (e.g., a lower learning rate in later curriculum stages), remains an open question. The 2.2pp gap could reflect a systematic architectural limitation, a suboptimal training configuration, or initialization variance; multi-seed replication with hyperparameter sensitivity analysis would clarify this (Section 6)."

The manuscript does NOT "dismiss" the p=0.071 via Bonferroni. It reports both values, explicitly acknowledges the gap might be real, lists three possible explanations, and recommends further investigation. It also notes that "the plateau does not alter the statistical conclusion that curriculum-matched controls reach comparable accuracy" -- which is technically correct: not reaching significance means you fail to reject the null, not that you accept the null. The manuscript appropriately avoids claiming M4 equals M2.

The Bonferroni correction itself is justified: k=5 test sets are tested within each comparison family, and the correction was prespecified. One could argue that the ID comparison is the most important and should not be corrected, but this would be post-hoc reasoning. The current approach (report both, correct as prespecified, honestly discuss the ambiguity) is the methodologically conservative choice.

**Verdict:** No edit required. The manuscript handles this with exemplary transparency.

---

### External Feedback Item #8: Minor issues

**#8a: Table 2 says n=500 for ProsQA test but does not clarify n for validation.**

Table 2 caption (line 73) says "ProsQA validation (n = 300) and test (n = 500) sets." Both n values are stated. Additionally, Section 3.1 (line 32) states: "The dataset contains 17,886 training samples, 300 validation samples, and 500 test samples." Table A13 in A.18 (line 546-548) repeats these numbers. The n values are clearly stated in multiple locations.

**Verdict:** No issue. The n for validation (300) is explicitly stated in the Table 2 caption.

**#8b: Table 3 n-values potentially inconsistent (check corruption sample sizes).**

Table 3 is the probing summary table, not the corruption table. The corruption table is Table A1. Table A1 caption says "n = 500 per condition." The probing uses n = 500 for positions 0-2, n = 298 for position 3, n = 81 for position 4, n = 12 for position 5, as documented in A.3 (line 251) and A.4 (line 259). The n values are internally consistent and correctly documented.

For corruption specifically: forward corruption (Table A1) uses n = 500. Single-position corruption (Table A4) -- no n stated in caption. Reverse corruption (Table A3) -- no n stated in caption. From context (these all use the ProsQA test set), n = 500 for all. However, no n is explicitly stated in the Table A3 and A4 captions. This is a minor omission.

**Proposed micro-edit:** Add "(n = 500)" to the Table A3 and A4 captions for completeness, matching the Table A1 convention. This is a SUGGESTION-level issue.

**#8c: A.4 p-value clarification needed.**

A.4 (line 263) states: "minimum achievable p = 1/2001 = 0.0005, below the Bonferroni threshold of 0.05/78 = 0.000641." The claim is that 1/2001 < 0.05/78. Verified: 1/2001 = 0.000499750..., and 0.05/78 = 0.000641025..., so 0.000500 < 0.000641. Correct. However, the text also states the permutation test uses "the conservative estimator p = (count + 1) / (n_perms + 1)." If count = 0 (best case), then p = 1/2001 = 0.000500. This is indeed below the Bonferroni threshold. The statement is clear and correct.

**Verdict:** No issue. The computation is correctly documented.

**#8d: Table 6b -- is r (rank-biserial correlation) defined before first use?**

Table 6a caption (line 446) states: "r = rank-biserial correlation." Table 6b (line 456) comes immediately after. Since Table 6a is read before Table 6b, the definition is available to the reader. However, the first textual use of "r" in Section 4.5 (line 138) says "r = 0.678" without defining what r is. The Wilcoxon signed-rank test is named (line 136), and r is defined in the Table 6a caption, but there is a gap: the reader encounters "r = 0.678" in the main text (line 138) before reaching the table caption (line 446 in the appendix). The methods section (A.3, line 255) says "exact McNemar's test" but does not describe the Wilcoxon methodology.

However, r is introduced with context: "M2 assigns systematically higher confidence than M4 (r = 0.678, p < 10^{-50})" -- the parenthetical format (effect size, p-value) makes the role of r clear from context. The table caption formally defines it. This is adequate.

**Verdict:** No issue. The definition in the Table 6a caption is sufficient. A reader who encounters r in Section 4.5 text can infer from context that it is an effect size measure, and the formal definition follows in the appendix.

**#8e: A.18 placement (is there an A.18?)**

A.18 exists at line 516. It is titled "### A.18 Datasets" and contains the ProsQA example, construction methodology, graph statistics (Table A12), and dataset split sizes (Table A13). The placement is logical -- datasets are supplementary material that supports the Methods section but is not required for the main narrative. The section exists and contains substantive content.

**Verdict:** No issue. A.18 exists.

---

### Previous Finding Status Update (S001--S026)

**S001 (numerical_accuracy, CRITICAL):** ROUND 4: CONFIRMED RESOLVED. All accuracy values remain internally consistent across all tables and text. Spot-checked: M2=97.0% (Table 2, abstract, Section 4.1, Table 4, Table 5); M3=96.6% (same locations); M4=94.8% (same). All match backing data.

**S002 (numerical_accuracy, MAJOR):** ROUND 4: CONFIRMED RESOLVED. No "85%" or "gap closure" text found.

**S003 (effect_sizes, SUGGESTION):** ROUND 4: DEFERRED. Wilcoxon r values provide effect sizes for continuous measures. OR for McNemar tables still not reported but can be computed by readers from b and c values. Non-blocking.

**S004 (corrections, SUGGESTION):** ROUND 4: DEFERRED. Per-family Bonferroni k=5 remains internally consistent and correctly applied. Non-blocking.

**S005 (methodology):** ROUND 4: CONFIRMED RESOLVED (M2 side). See S026 for M3 side.

**S006 (missing_analysis, SUGGESTION):** ROUND 4: DEFERRED. No formal corruption profile similarity test added. Non-blocking given factorial decomposition is the primary evidence.

**S007 (power, SUGGESTION):** ROUND 4: DEFERRED. Single-seed limitation adequately acknowledged. Non-blocking.

**S008 (numerical_accuracy):** ROUND 4: CONFIRMED RESOLVED. All three occurrences of M3 peak accuracy read 57.0%.

**S009 (test_selection):** ROUND 4: CONFIRMED RESOLVED. McNemar and Wilcoxon correctly applied.

**S010 (numerical_accuracy):** ROUND 4: CONFIRMED RESOLVED. DAG p=0.0015 correct.

**S011 (assumptions, SUGGESTION):** ROUND 4: DEFERRED. Non-blocking.

**S012 (methodology, SUGGESTION):** ROUND 4: DEFERRED. Non-blocking.

**S013 (interpretation, SUGGESTION):** ROUND 4: DEFERRED. Permutation insensitivity appropriately qualified.

**S014 (missing_analysis, SUGGESTION):** ROUND 4: DEFERRED. Bayesian analysis not added. Non-blocking.

**S015 (missing_analysis, SUGGESTION):** ROUND 4: DEFERRED. TOST not conducted. Non-blocking.

**S016 (interpretation):** ROUND 4: CONFIRMED RESOLVED.

**S017 (missing_analysis, SUGGESTION):** ROUND 4: DEFERRED. Transplant still lacks formal test. Non-blocking.

**S018 (methodology):** ROUND 4: CONFIRMED RESOLVED. MLP probe grid search adequately reported.

**S019 (numerical_accuracy, SUGGESTION):** ROUND 4: DEFERRED. Non-consequential L2 variation.

**S020 (numerical_accuracy, SUGGESTION):** ROUND 4: DEFERRED. `statistical_analysis.json` stale but manuscript uses correct values.

**S021 (interpretation):** ROUND 4: CONFIRMED RESOLVED.

**S022 (methodology, SUGGESTION):** ROUND 4: DEFERRED. Selectivity metric adequately defined.

**S023 (missing_analysis):** ROUND 4: CONFIRMED RESOLVED.

**S024 (data hygiene, MINOR):** ROUND 4: CONFIRMED PARTIALLY ADDRESSED. Deprecation header present. Ideal: regenerate. Non-blocking for manuscript.

**S025 (data hygiene, MINOR):** ROUND 4: CONFIRMED PARTIALLY ADDRESSED. `paper.yaml` accuracy values corrected (M2=0.970, M3=0.966). McNemar significance flags and M3_peak_accuracy remain stale. Non-blocking for manuscript.

**S026 (caption accuracy, MINOR):** ROUND 4: STATUS UPDATE. The original problematic text in the Table 3 caption ("For M3, selectivity is reported at the peak accuracy layer for each position: layer 8 for position 0, layer 11 for position 1, and layer 12 for positions 2 and 3") has been removed. The current Table 3 caption (line 92) no longer makes this claim. However, A.11 (line 407) still says "Table 3 reports selectivity at M3's peak accuracy layers (layer 12 for most positions)" -- the parenthetical "(layer 12 for most positions)" is technically correct (layer 12 IS the peak accuracy layer for positions 2 and 3) but the framing "peak accuracy layers" is misleading for positions 0 and 1, where the values are from layer 12 but the peak accuracy layers are 8 and 11 respectively. The sentence then correctly notes the layer-dependent selectivity, which addresses the substance of the concern. This is now a VERY MINOR issue -- the main Table 3 caption is fixed, and the A.11 text contains the qualification. Downgrading from MINOR to SUGGESTION.

**Proposed micro-edit for A.11 (line 407):**
```
FIND: Table 3 reports selectivity at M3's peak accuracy layers (layer 12 for most positions).
REPLACE: Table 3 reports M3 selectivity at layer 12 for all positions (matching the final-layer convention used for the main probing grids).
```

---

### New Finding: S027 (MINOR) -- Dense result interpretation

See External Feedback Item #2 above. The manuscript describes the dense result as "creating an interaction effect that the additive factorial decomposition cannot capture" (line 132), but the data pattern (symmetric +3.6pp and -3.6pp) is more consistent with additive cancellation of opposing effects than a true interaction. The proposed edit is included above.

---

### Round 4 Summary Table

| Finding | Severity | Category | Status |
|---------|----------|----------|--------|
| S001 | Critical | numerical_accuracy | RESOLVED (Round 2) |
| S002 | Major | numerical_accuracy | RESOLVED (Round 2) |
| S003 | Suggestion | effect_sizes | DEFERRED |
| S004 | Suggestion | corrections | DEFERRED |
| S005 | Major | methodology | RESOLVED (Round 2, M2 side) |
| S006 | Suggestion | missing_analysis | DEFERRED |
| S007 | Suggestion | power | DEFERRED |
| S008 | Minor | numerical_accuracy | RESOLVED (Round 3) |
| S009 | Minor | test_selection | RESOLVED (Round 2) |
| S010 | Minor | numerical_accuracy | RESOLVED (Round 2) |
| S011 | Suggestion | assumptions | DEFERRED |
| S012 | Suggestion | methodology | DEFERRED |
| S013 | Suggestion | interpretation | DEFERRED |
| S014 | Suggestion | missing_analysis | DEFERRED |
| S015 | Suggestion | missing_analysis | DEFERRED |
| S016 | Suggestion | interpretation | RESOLVED (Round 2) |
| S017 | Suggestion | missing_analysis | DEFERRED |
| S018 | Suggestion | methodology | RESOLVED (Round 2) |
| S019 | Suggestion | numerical_accuracy | DEFERRED |
| S020 | Suggestion | numerical_accuracy | DEFERRED |
| S021 | Major | interpretation | RESOLVED (Round 2) |
| S022 | Suggestion | methodology | DEFERRED |
| S023 | Minor | missing_analysis | RESOLVED (Round 2) |
| S024 | Minor | data_hygiene | PARTIALLY ADDRESSED |
| S025 | Minor | data_hygiene | PARTIALLY ADDRESSED |
| S026 | Suggestion | caption_accuracy | PARTIALLY ADDRESSED (downgraded) |
| S027 | Minor | interpretation | NEW -- dense "interaction" framing |

**Critical findings:** 0 open
**Major findings:** 0 open
**Minor findings:** 3 open (S024, S025, S027) -- all non-blocking
**Suggestions:** 14 open -- all non-blocking

**External feedback assessment:**
- Item #1 (ID Wilcoxon oversold): No edit needed. Section 4.5 already contextualizes appropriately.
- Item #2 (Dense underinterpreted): Minor edit proposed (S027). Non-blocking.
- Item #3 (Bonferroni shields M4 gap): No edit needed. Already transparent.
- Item #8 (minor issues): All checked, no substantive issues found.

**Overall assessment: PASS.** The manuscript's statistical foundations are sound. All numerical claims verified. The external feedback items reveal the manuscript already handles most interpretive nuances well. The one substantive suggestion (S027, dense interpretation) is a minor framing improvement. No blocking issues remain.

---

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
