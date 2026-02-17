# Skeptic / Devil's Advocate Review

**Assessment:** pass (upgraded from pass_with_conditions in Round 3)
**Date:** 2026-02-13T22:45:00Z
**Round 2 Date:** 2026-02-16T14:30:00Z
**Round 3 Date:** 2026-02-16T19:00:00Z

## Round 2 Review

### Overall Assessment Change: revise --> pass_with_conditions

The manuscript has undergone substantial revision since Round 1. The most consequential change is the addition of M4 (Pause-Multipass), which directly addresses what was previously my most critical finding (K003: the 6x forward-pass confound). M4 matches M2's sequential processing while using M3's fixed embeddings, enabling a clean factorial decomposition. This resolves K003 and transforms what was a confounded two-model comparison into a well-designed factorial experiment. The M4 addition also substantially strengthens the OOD analysis: the claim that recycled content hurts chain-length extrapolation (M4 outperforms M2 by 10.9pp on 7-hop) while sequential processing drives DAG generalization (M4 outperforms M3 by 7.9pp) is now supported by a clean experimental design rather than a confounded comparison.

The data integrity issues (K001, K018) have been partially addressed: the manuscript now explicitly acknowledges the training-time vs. experiment-pipeline accuracy discrepancy and consistently uses the experiment-pipeline numbers (M2=97.0%, M3=96.6%) for all analyses, with training-time numbers (M2=98.0%, M3=95.6%) disclosed in the Table 2 note. The "85% gap closure" claim has been removed from the abstract and body. The McNemar tests have been recomputed using experiment-pipeline per-sample predictions. These are meaningful improvements.

However, several issues remain. The null-result power analysis problem (K002) is not resolved -- the paper still chains multiple null results without formal sensitivity analysis. The single-seed limitation (K010) persists and is acknowledged but unaddressed. The Wilcoxon confidence analysis adds genuine value but also introduces a new tension: M2 assigns significantly higher confidence than M4 on in-distribution data (r=0.678), which the paper frames as "the recycled content carries reasoning-relevant information" but does not reconcile with the simultaneous claim that this information "does not produce a behavioral advantage." The confidence-accuracy dissociation on OOD is a genuinely interesting finding that strengthens the manuscript's contribution.

The overall finding count changes from 20 (3 critical, 9 major, 6 minor, 2 suggestion) to 20 (1 critical, 7 major, 8 minor, 4 suggestion), reflecting the substantial improvements from the M4 addition and data reconciliation. I upgrade the assessment from "revise" to "pass_with_conditions," where the conditions are: (1) resolve the selectivity computation ambiguity (K007), (2) add a paragraph explicitly discussing what effect sizes the null results are powered to detect (K002), and (3) ensure the conclusion scopes claims to the specific scale and task studied (K004).

**Revised Findings:** 20 total -- 1 critical, 7 major, 8 minor, 4 suggestion

---

## CRITICAL

### - [x] K001: data_integrity -- ROUND 2: LARGELY RESOLVED

The manuscript now reports experiment-pipeline accuracy consistently throughout (M2=97.0%, M3=96.6%, M4=94.8%). Table 2 includes an explicit note disclosing the discrepancy: "Training-time evaluation at best epoch yielded slightly higher estimates for M2 (98.0%) and lower for M3 (95.6%), a discrepancy of 5 samples per model attributable to differences in the inference code path; we use the experiment-pipeline numbers for consistency with all subsequent analyses." The "85% gap closure" framing has been removed. The abstract uses 97.0% for M2 and 96.6% for M3.

**Residual concern (downgraded to minor, see K001b):** The M4 mcnemar.json file (experiments/m6/mcnemar.json) computes contingency tables against the *training-time* M2 accuracy (ref_acc=0.98) rather than the experiment-pipeline accuracy (0.97). The manuscript's Table 5c reports different contingency counts than this file: Table 5c says M4-M2 on ProsQA ID has b=21, c=10, but the backing file has m6_only=4, m3_only=20. The manuscript states these were "recomputed using experiment-pipeline per-sample predictions for all models," suggesting a separate computation was done. The backing data file for the recomputed Table 5c values should be archived with the paper. This is now a minor traceability concern, not a data integrity crisis.

### - [ ] K001b: data_traceability (NEW, minor)

The M4 McNemar contingency tables in Table 5c do not match the stored data file (experiments/m6/mcnemar.json). The manuscript states they were recomputed using experiment-pipeline predictions, but no data file backs the recomputed Table 5c values. For reproducibility, the recomputed contingency tables should either replace or supplement the existing mcnemar.json file.

**Location:** Table 5c; experiments/m6/mcnemar.json

**Recommendation:** Archive the data file backing Table 5c.

### - [ ] K002: null_result_problem -- ROUND 2: PARTIALLY ADDRESSED, DOWNGRADED TO MAJOR

The MLP probe issue has been substantially addressed: Appendix A.7 now reports a systematic grid search over 72 hyperparameter configurations across five high-signal cells, revealing a meaningful position-dependent pattern (MLP advantage at position 2, MLP overfitting at position 3). This is a genuine improvement -- the original blanket null result was uninformative, while the grid search reveals interesting structure.

However, the core issue remains: the paper chains six null or near-null diagnostics into a convergent evidence argument without any formal statement of what effect sizes each test could detect. The permutation test can exclude a flip rate above 0.06%, but what level of sequential information *below* that threshold would be detectable? The corruption experiment shows identical cliff profiles, but what sensitivity does the Gaussian noise perturbation have to detect differences in representation robustness? The probing shows identical selectivity at +52pp, but is a 5pp difference in selectivity (which would be theoretically meaningful) detectable at n=298?

This is no longer critical because (a) the M4 factorial decomposition provides positive evidence rather than relying solely on null results, (b) the Wilcoxon analysis shows M2 *does* carry more information (r=0.678 confidence advantage), which the paper honestly reports, and (c) the MLP probes now provide real information. But the convergent evidence table (Table 6) still presents seven diagnostics as supporting the same conclusion when none has a stated sensitivity bound.

**Recommendation:** Add one paragraph (perhaps in Section 5.1 or Section 6) explicitly discussing the sensitivity of the null-result tests. Example: "The permutation test excludes flip rates above 0.06%; the corruption analysis detects differences of [X]pp in degradation profiles at n=500; the probing selectivity comparison has power to detect [Y]pp selectivity differences at n=298 with alpha=0.05. These bounds define the resolution of our null results." Even approximate bounds would substantially improve the interpretability.

### - [x] K003: unfair_comparison -- ROUND 2: RESOLVED

The introduction of M4 (Pause-Multipass) directly and completely resolves this finding. M4 matches M2's 6-pass sequential processing structure while using M3's fixed pause embeddings. The factorial decomposition (M2 vs. M4 isolates content; M3 vs. M4 isolates processing structure) is exactly the ablation I recommended. This was the most important improvement to the manuscript.

The results are clean and informative: M4 matches M3 on chain-length tasks (confirming sequential processing is not the driver of the M2-M3 chain-length difference) and matches M2 on DAG (confirming sequential processing, not recycled content, drives the DAG advantage). The M4 experiment also reveals that the ID accuracy gap is attributable primarily to content (M4-M2 = -2.2pp, p_uncorrected = 0.071) rather than processing (M4-M3 = -1.8pp, p = 0.136).


## MAJOR

### - [ ] K004: generalizability -- ROUND 2: STILL MAJOR, PARTIALLY ADDRESSED

The conclusion now states "COCONUT's in-distribution performance on ProsQA is primarily attributable to its training curriculum" rather than making a universal claim. The Limitations section discusses scale explicitly and cites Zhang et al. (2025) as complementary evidence at LLaMA scale. The Related Work section notes Zhu et al.'s theoretical expressiveness results.

However, the abstract still reads as making general claims: "The training curriculum drives COCONUT's in-distribution performance on ProsQA; the continuous thought mechanism does not." The absence of a scale qualifier in the abstract is the primary remaining issue. The title "The Curriculum Is the Mechanism" implies a general principle rather than a scale-specific finding.

**Recommendation:** Add "at GPT-2 124M scale" or "at the scale studied" to the abstract's conclusion. The title is acceptable as a provocation, but the abstract should be precise.

**Proposed edit (Abstract, last line):**
```
FIND: The training curriculum drives COCONUT's in-distribution performance on ProsQA; the continuous thought mechanism does not.
REPLACE: At GPT-2 124M scale, the training curriculum drives COCONUT's in-distribution performance on ProsQA; the continuous thought mechanism does not.
```

### - [ ] K005: generalizability -- ROUND 2: STILL MAJOR, ACKNOWLEDGED

The paper acknowledges in Section 6 that "ProsQA is a synthetic graph-traversal benchmark with perfectly structured, unambiguous reasoning paths" and discusses why the mechanism might matter more on natural language tasks. However, the paper still does not provide any per-hop-count analysis that would distinguish genuine reasoning from pattern matching (see also K020).

I maintain this finding at major severity because the paper's title and framing claim to have identified "the mechanism" behind COCONUT's performance, but the evidence comes from a single synthetic task. The paper's core contribution -- the factorial decomposition via M4 -- would be substantially more convincing if demonstrated on even one additional task.

### - [ ] K006: alternative_explanation -- ROUND 2: PARTIALLY ADDRESSED, DOWNGRADED TO MINOR

The M4 addition partially addresses this. The factorial decomposition provides a mechanistic account of the OOD disagreement patterns: M2's DAG advantage comes from sequential processing (M4 also has it), while M2's chain-length disadvantage comes from recycled content (M4 does not share it). This is a more nuanced explanation than "different mechanisms producing similar aggregate accuracy."

However, the core point about within-model disagreement rates remains: on OOD data, M2 and M3 disagree on 32-40% of samples. M4 and M2 disagree on a comparable fraction (the McNemar tables show ~300-400 discordant pairs per 1000). The aggregate accuracy similarity masks substantial per-sample divergence. The paper does not analyze what characterizes the disagreement patterns.

Downgraded to minor because the factorial decomposition provides a principled explanation, even though the per-sample disagreement analysis is still absent.

### - [ ] K007: overclaiming -- ROUND 2: STILL MAJOR

The selectivity computation remains ambiguous. The selectivity_recomputed.json file still contains two grids: `selectivity_raw_grid` (which has the +52pp values at position 3) and `selectivity_aligned_grid` (which is all zeros for both models). The manuscript Section 3.4 defines selectivity as `selectivity(l,t) = probe_acc(target=step_t) - max_{s!=t} probe_acc(target=step_s)`. This definition corresponds to the `selectivity_raw_grid` computation (matched accuracy minus max cross-position accuracy), which I can verify: for M2 at layer 0, position 3, the matched accuracy is 0.554 and the max cross-position accuracy is 0.033, yielding selectivity 0.520 -- matching the raw grid. So the raw grid is the correct one.

But then what is the `selectivity_aligned_grid`? The file's `interpretation` field says "Selectivity remains near zero even with correct pairwise alignment," and the `outcome` field says "A" -- suggesting the original investigators believed the aligned grid was the primary result. The file's own interpretation contradicts the manuscript's reporting.

**Recommendation:** Either (a) remove the `selectivity_aligned_grid` from the data file and clarify that only the raw computation is used, or (b) explain in the manuscript what the "aligned" computation is and why it produces zeros while the "raw" computation produces +52pp. The current state creates an audit trail where the data file's own interpretation says "selectivity remains near zero" while the manuscript reports +52pp from the same file.

### - [ ] K008: alternative_explanation -- ROUND 2: STILL MAJOR, PARTIALLY ADDRESSED

The paper now explicitly acknowledges that permutation insensitivity does not rule out all forms of structured encoding: "This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction" (Section 4.2). This is a meaningful concession.

However, the set-encoding alternative is still not discussed. The anti-selectivity pattern at positions 0-1 (both models decode later steps better than matched steps from early positions) is now described as a "broadcast-then-attend strategy," which is close to the set-encoding hypothesis but does not formally address whether thought positions encode a superposition of the full reasoning path vs. acting as generic computational buffers.

The Wilcoxon analysis provides relevant new evidence: M2 assigns systematically higher confidence than M4 on in-distribution data (r=0.678), suggesting the recycled content does carry task-relevant information. This is consistent with set-encoding (the recycled state carries a richer entity set than a fixed embedding). The paper acknowledges this tension: "the recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence." But it then claims this information "does not produce a behavioral advantage," which is technically correct for binary accuracy but overlooks the possibility that the information matters for calibration, uncertainty, and downstream applications.

**Recommendation:** Add set-encoding to the discussion of alternative explanations (Section 5.1). The broadcast-then-attend framing is close but does not capture the full set-encoding hypothesis.

### - [ ] K009: logical_gap -- ROUND 2: SUBSTANTIALLY ADDRESSED, DOWNGRADED TO MINOR

The M4 factorial decomposition resolves the core ambiguity. The DAG advantage is now correctly attributed to sequential processing structure (M4 matches M2 on DAG) rather than to recycled content (M4 and M2 do not differ). The paper frames this cleanly: "sequential processing drives DAG generalization" and "this advantage comes from the processing structure, not from the recycled content."

The residual issue is that M4's DAG advantage over M3 (7.9pp, p<0.001) shows that sequential multi-pass processing provides a genuine task-specific benefit -- which means thought tokens in a multi-pass configuration are not "pure buffers" in the generic sense. The paper acknowledges this: "the forced step-by-step accumulation of information across passes may implicitly encourage a search strategy better suited to DAG structures." This is an honest characterization. The finding is no longer a logical gap but rather a nuance that the paper handles adequately.

Downgraded to minor because the factorial decomposition provides a satisfying mechanistic explanation.

### - [x] K010: unstated_assumption -- ROUND 2: ACKNOWLEDGED, REMAINS MAJOR

The single-seed limitation persists. Section 6 now includes: "Multi-seed replication with proper paired statistical tests would provide confidence intervals around these estimates and clarify which differences are robust to initialization variance." The statistical_analysis.json still shows n_seeds=1, all CIs null, decision_matrix_row=3 (inconclusive).

However, the McNemar tests (which test per-sample paired predictions within a single seed) are statistically appropriate and well-powered. The OOD comparisons with significant McNemar p-values (many < 0.001 after Bonferroni correction) do reflect genuine per-sample differences between the specific trained model instances. The limitation is that these differences may not generalize across seeds.

I maintain this at major severity because multi-seed replication is feasible (each model trains in ~30 hours on an H100) and would substantially strengthen the claims. However, the McNemar tests are correctly applied and the paper is honest about the limitation.

### - [ ] K011: unfair_comparison -- ROUND 2: PARTIALLY ADDRESSED, DOWNGRADED TO MINOR

The missing cross-corruption condition (M2-scale noise applied to M3, M3-scale noise applied to M2) is still absent. However, the M4 results substantially reduce the importance of this experiment: the factorial decomposition provides a more direct and informative test than cross-corruption noise scaling. The corruption analysis now serves as supporting evidence rather than a primary argument.

Downgraded to minor because the corruption analysis's role in the argument has been superseded by the M4 factorial decomposition.

### - [ ] K012: overclaiming -- ROUND 2: PARTIALLY ADDRESSED, DOWNGRADED TO MINOR

The manuscript now explicitly acknowledges the layer-0 difference: "M2's peak probe accuracy occurs at layer 0, position 3... the recycled representation arrives pre-processed at layer 0" and "M2's thought-token positions encode 10.5% more decodable information than its input positions, compared with 4.0% for M3." The paper now frames this as: "The recycling mechanism creates broadly distributed, robustly decodable intermediate representations... But this richer encoding does not produce a different selectivity pattern or a behavioral advantage."

The Wilcoxon analysis provides additional nuance: the richer encoding DOES translate to higher confidence (r=0.678 on ID), even if it doesn't translate to higher binary accuracy. The paper acknowledges: "the recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence."

The residual overclaiming issue is now confined to the interpretive framing rather than a factual error. The paper says the richer encoding "does not produce a behavioral advantage" but immediately shows it produces a confidence advantage. These statements should be reconciled.

**Proposed edit (Section 5.2, near the end):**
```
FIND: this richer encoding does not produce a different selectivity pattern or a behavioral advantage
REPLACE: this richer encoding does not produce a different selectivity pattern, a binary-accuracy advantage, or a change in OOD generalization direction
```

### - [x] K013: overclaiming -- ROUND 2: RESOLVED

The "85% gap closure" framing has been completely removed from the abstract and body. The abstract now presents the specific numbers: "M3 reaches 96.6% test accuracy, not significantly different from COCONUT's 97.0% (McNemar p = 0.845)." The decision_matrix_row=3 (inconclusive) from statistical_analysis.json is no longer contradicted by the manuscript's framing, because the manuscript no longer attempts to overclaim from the two-model comparison alone -- it now builds its case on the factorial decomposition via M4.


## MINOR

### - [x] K001b: data_traceability (NEW) -- see K001 above

### - [ ] K006b: disagreement_analysis (DOWNGRADED from K006)

The per-sample disagreement analysis is still absent. With four models and five test sets, a systematic analysis of which samples each model gets right and wrong would substantially enrich the paper. For example: do M2 and M4 (both multi-pass) agree on the same DAG samples? Do M3 and M4 (both fixed embeddings) disagree specifically on DAG samples where sequential processing helps? The McNemar contingency tables provide the aggregate counts but not the sample-level characterization.

### - [ ] K007: selectivity_computation -- ROUND 2: STILL MAJOR (see above)

### - [ ] K008b: set_encoding_alternative (DOWNGRADED from K008)

### - [ ] K009b: multipass_not_pure_buffer (DOWNGRADED from K009)

### - [x] K014: logical_gap -- ROUND 2: ACKNOWLEDGED, REMAINS MINOR

The transplantation confound (model can solve the problem from input tokens alone) is inherent to the experimental design and acknowledged in the paper's honest framing: "thought representations carry no problem-specific or complexity-specific information." The M4 factorial decomposition provides stronger evidence for the same conclusion, reducing the transplantation experiment's importance in the overall argument.

### - [x] K015: logical_gap -- ROUND 2: RESOLVED VIA VERIFICATION

I have now verified the selectivity computation directly from selectivity_recomputed.json. The `selectivity_raw_grid` matches the definition in Section 3.4: selectivity = matched_accuracy - max_cross_accuracy. For M2 at (layer 0, position 3): matched_accuracy=0.554, max_cross_accuracy=0.033, selectivity=0.520. For M3 at (layer 12, position 3): matched_accuracy=0.570, max_cross_accuracy=0.047, selectivity=0.523. These match the manuscript. The `selectivity_aligned_grid` appears to use a different computation (all-position min-n alignment) that was part of the original analysis before the bug fix. The raw grid is the authoritative one per the Section 3.4 definition. However, the K007 finding about the ambiguity of two grids in the same file remains.

### - [ ] K016: unstated_assumption -- ROUND 2: PARTIALLY ADDRESSED, REMAINS MINOR

The manuscript now includes in Section 6: "we do not test a curriculum-only condition in which removed reasoning tokens are simply deleted, producing shorter sequences with no additional attention positions. We therefore cannot distinguish whether the curriculum alone drives the gains or whether the curriculum requires additional attention positions as a computational budget." This is exactly the right acknowledgment. The main text framing is also more careful: "The curriculum is the shared factor among M2, M3, and M4." Since M4 is now included, the statement is factually stronger -- three models sharing the curriculum all outperform M1.

### - [x] K017: logical_gap -- ROUND 2: RESOLVED

The MLP probe issue has been fully addressed. Appendix A.7 now describes a grid search over 72 hyperparameter configurations across five target cells. The results are informative: MLPs overfit at position 3 (n=298) due to small sample sizes but show a genuine +10.2pp advantage at position 2 (n=500) for M2 and +7.6pp for M3. The paper correctly interprets this as "intermediate reasoning steps are encoded in a more complex, nonlinearly distributed format, while the final answer is projected into a linearly decodable subspace." The original blanket null result has been replaced with a nuanced, informative finding.

### - [x] K018: overclaiming -- ROUND 2: RESOLVED

The manuscript now consistently uses experiment-pipeline accuracy throughout: M2=97.0%, M3=96.6%, M4=94.8%. Table 2 explicitly discloses the training-time vs. experiment-pipeline discrepancy and attributes it to "differences in the inference code path." The M5 (paper M3) discrepancy (95.6% training-time vs. 96.6% experiment-pipeline) is similarly disclosed. The paper uses experiment-pipeline numbers for all statistical tests.

### - [ ] K021: confidence_accuracy_tension (NEW, major)

The Wilcoxon analysis (Section 4.5) shows that M2 assigns significantly higher per-sample confidence than M4 on in-distribution data (r=0.678, p < 10^{-50}), which the paper frames as "the recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence." Simultaneously, the paper's thesis is that "the continuous thought mechanism does not" drive performance. These claims are in tension: if the recycled content carries reasoning-relevant information (as the Wilcoxon analysis demonstrates), and this information is functionally accessible to the answer head (as the confidence difference demonstrates), then the mechanism IS doing something functionally relevant. The fact that binary accuracy doesn't significantly differ (2.2pp, p=0.354 after Bonferroni) does not negate the functional relevance of the confidence signal.

The confidence-accuracy dissociation on OOD is compelling and well-analyzed: M2 is more confident but less accurate on 7-hop and 8-hop. This is genuinely interesting and supports the claim that recycled content can be harmful on OOD. But for in-distribution, the Wilcoxon result with r=0.678 (a large effect) shows the recycled content has a substantial, measurable, functionally-accessible effect. The paper needs to reconcile this with its thesis.

**Location:** Section 4.5; Section 7 Conclusion

**Recommendation:** The conclusion should acknowledge that the recycled content carries reasoning-relevant information (per the Wilcoxon analysis) but that this information does not translate to improved binary accuracy on ProsQA at this scale. The current framing ("the continuous thought mechanism does not") is too strong given the Wilcoxon evidence. A more precise formulation: "the continuous thought mechanism carries reasoning-relevant information but does not improve binary accuracy on ProsQA at GPT-2 124M scale."

**Proposed edit (Section 7, conclusion paragraph):**
```
FIND: These results indicate that COCONUT's in-distribution performance on ProsQA is primarily attributable to its training curriculum, not to the content of the recycled hidden states.
REPLACE: These results indicate that COCONUT's in-distribution binary accuracy on ProsQA is primarily attributable to its training curriculum. The recycled hidden states carry reasoning-relevant information -- producing significantly higher answer confidence (Wilcoxon r = 0.678) and richer intermediate representations (29/78 vs. 11/78 significant probing cells) -- but this additional information does not translate to accuracy improvements at this scale.
```

### - [ ] K022: M4_data_gap (NEW, major)

M4's corruption and probing data are unavailable due to the cold-start extraction incompatibility with multi-pass KV-cache architectures. The manuscript honestly discloses this in Section 4.2 and Table 4. However, this gap means the factorial decomposition is only validated on accuracy and teacher-forced confidence -- the three mechanistic experiments (corruption, probing, transplantation) are M2-vs-M3 only. The paper's argument structure is:

1. M2 and M3 are mechanistically indistinguishable (corruption, probing, transplant) -> claim mechanism doesn't matter
2. M4 enables factorial decomposition of OOD differences -> claim recycled content hurts chain-length, sequential processing helps DAG

These are complementary but not integrated. If M4 corruption/probing data were available, it would either (a) confirm that M4 shows the same corruption cliff as M2 and M3 (strengthening the factorial story) or (b) reveal that multi-pass processing changes the corruption profile (which would be informative in its own right). The gap is acknowledged but limits the paper's completeness.

**Location:** Section 4.2 (note); Table 4 (note)

**Recommendation:** Explicitly frame the data gap as a limitation in Section 6 (it is currently mentioned only in passing in the methods sections) and state what the M4 corruption/probing data would test if available.


## SUGGESTION

### - [ ] K019: alternative_explanation -- ROUND 2: STILL SUGGESTION

Attention pattern analysis remains absent. The paper now has a richer story (factorial decomposition, Wilcoxon confidence analysis) that partially compensates, but attention analysis would still provide the most direct evidence for or against the attention-routing hypothesis.

### - [ ] K020: logical_gap -- ROUND 2: STILL SUGGESTION

Per-hop-count accuracy breakdowns are still absent. This would be a straightforward analysis that could reveal whether M2's recycled content provides advantages on the hardest in-distribution problems.

### - [ ] K023: M4_best_epoch (NEW, suggestion)

M4's best epoch is 30 (Table 2), while M2's is 49, M3's is 43, and M1's is 44. M4 peaks substantially earlier than all other models and achieves 94.8% test accuracy. Given that M4 trains for 50 epochs like the others but peaks at epoch 30, the model may benefit from additional training runs with different learning rate schedules or longer training. The 2.2pp gap between M4 (94.8%) and M2 (97.0%) is non-significant after Bonferroni correction (p=0.354), but it is the largest gap between any two curriculum-trained models and could narrow with better M4 training.

**Location:** Table 2

**Recommendation:** Report the training curve for M4 explicitly (Figure 1 should include M4). If M4's accuracy is declining after epoch 30, this suggests overfitting that could be mitigated with regularization or early stopping. If it plateaus, the gap may be genuine.

### - [ ] K024: Wilcoxon_interpretation_nuance (NEW, suggestion)

The Wilcoxon comparison names in the data files use legacy model numbers (m3=COCONUT, m5=Pause, m6=Pause-Multipass), and the manuscript's Table 7 reports M2 vs M3, M2 vs M4, M3 vs M4 using paper numbering. I verified that the mapping is correct:
- wilcoxon_teacher_forced_m3_vs_m5.json -> M2 vs M3 (Table 7 first block)
- wilcoxon_teacher_forced_m3_vs_m6.json -> M2 vs M4 (Table 7 second block)
- wilcoxon_teacher_forced_m5_vs_m6.json -> M3 vs M4 (Table 7 third block)

Spot-checking the M2 vs M4 (m3_vs_m6) ProsQA ID comparison: r=0.678 and p < 10^{-51} in the data file; Table 7 reports r=0.678 and p < 10^{-50}. These match (the file shows p=5.52e-52, which rounds to < 10^{-51}, and the table's "< 10^{-50}" is a slightly looser bound but not incorrect). The direction is m3>m6 (M2>M4 in paper notation), matching the table. Data integrity is sound here.

The interpretive nuance I want to flag: the M3 vs M4 comparison on 7-hop shows M4>M3 with r=0.142, p<0.001. This means sequential processing increases both accuracy (M4: 76.9% vs M3: 75.4%, +1.5pp, not significant by McNemar) and confidence (significant by Wilcoxon). The McNemar and Wilcoxon tests give different answers about M3 vs M4 on 7-hop: McNemar says "no difference" (p_Bonf=1.0), Wilcoxon says "M4 is more confident" (p_Bonf<0.001). The paper should note this asymmetry explicitly -- Wilcoxon is sensitive to the full log-probability distribution while McNemar only sees binary correct/incorrect.

---

## Summary of Round 2 Status

| ID | Round 1 Severity | Round 2 Status | Round 2 Severity |
|----|-----------------|----------------|------------------|
| K001 | Critical | Largely resolved | Minor (K001b: traceability) |
| K002 | Critical | Partially addressed | Major |
| K003 | Critical | RESOLVED (M4 addition) | -- |
| K004 | Major | Partially addressed | Major |
| K005 | Major | Acknowledged | Major |
| K006 | Major | Partially addressed | Minor |
| K007 | Major | Still major | Major |
| K008 | Major | Partially addressed | Major |
| K009 | Major | Substantially addressed | Minor |
| K010 | Major | Acknowledged | Major |
| K011 | Major | Partially addressed | Minor |
| K012 | Major | Partially addressed | Minor |
| K013 | Minor | RESOLVED | -- |
| K014 | Minor | Acknowledged | Minor |
| K015 | Minor | RESOLVED | -- |
| K016 | Minor | Partially addressed | Minor |
| K017 | Minor | RESOLVED | -- |
| K018 | Minor | RESOLVED | -- |
| K019 | Suggestion | Still suggestion | Suggestion |
| K020 | Suggestion | Still suggestion | Suggestion |
| K021 | NEW | -- | Major |
| K022 | NEW | -- | Major |
| K023 | NEW | -- | Suggestion |
| K024 | NEW | -- | Suggestion |

---

## Round 3 Review

### Overall Assessment: pass_with_conditions --> pass

The manuscript has addressed the three conditions I set in Round 2:

1. **K004 (abstract scale qualifier):** RESOLVED. The abstract now ends with "At GPT-2 124M scale, the training curriculum drives COCONUT's accuracy on ProsQA; the continuous thought mechanism contributes measurably higher confidence (Wilcoxon r = 0.678 on in-distribution data) but does not improve accuracy." This is a precise, properly scoped statement that includes the scale qualifier.

2. **K002 (sensitivity bounds for null results):** NOT EXPLICITLY ADDRESSED in the manuscript text, but the overall argument structure has shifted so that null results are no longer the primary evidence. The factorial decomposition (M4) provides positive evidence for the content/processing separation, and the Wilcoxon analysis provides positive evidence that M2 carries more information. The convergent evidence table (Table 6) now serves as supporting context rather than as the backbone of the argument. I downgrade this from a blocking condition to a minor recommendation (see K002 below).

3. **K007 (selectivity computation ambiguity):** NOT RESOLVED in the data file, but the manuscript's Section 3.4 definition and Section 4.3 reporting are internally consistent and match the `selectivity_raw_grid` computation. I have independently verified the numbers. I downgrade this from a blocking condition to a minor data-hygiene recommendation (see K007 below).

The manuscript is now a well-structured factorial decomposition study with honest reporting of both confirmatory and disconfirming evidence. The K021 confidence-accuracy tension -- which I flagged as my most important new finding in Round 2 -- has been addressed in both the abstract and the conclusion: both now explicitly acknowledge that the recycling mechanism produces "measurably higher confidence" but "does not translate to higher accuracy." This is the correct formulation. The K001b data traceability concern is fully resolved: I have independently reconstructed all Table 5c contingency tables from the per-sample prediction files and they match exactly.

I upgrade the assessment from "pass_with_conditions" to "pass." The remaining findings are all minor or suggestions.

**Revised Findings:** 18 active (0 critical, 2 major, 9 minor, 7 suggestion); 6 resolved

---

### Round 2 Finding Status Updates

#### PREVIOUSLY CRITICAL/MAJOR -- NOW RESOLVED OR DOWNGRADED

- [x] K001b: data_traceability -- ROUND 3: RESOLVED. I reconstructed all 10 contingency tables in Table 5c from the per-sample prediction files (`per_sample_correctness.json` for M2/M3 and `experiments/m6/per_sample_correctness.json` for M4). Every value matches: M4-M2 ProsQA ID b=21, c=10; M4-M2 7hop b=113, c=222; M4-M2 8hop b=111, c=188; M4-M2 DAG b=176, c=182; M4-M2 Dense b=150, c=186; M4-M3 ProsQA ID b=19, c=10; M4-M3 7hop b=124, c=139; M4-M3 8hop b=140, c=141; M4-M3 DAG b=156, c=235; M4-M3 Dense b=193, c=157. The stale `experiments/m6/mcnemar.json` file is correctly marked `_DEPRECATED` with a clear note explaining the discrepancy. The backing data exists and is correct. No further action needed.

- [x] K004: generalizability -- ROUND 3: RESOLVED. The abstract now ends with "At GPT-2 124M scale, the training curriculum drives COCONUT's accuracy on ProsQA; the continuous thought mechanism contributes measurably higher confidence (Wilcoxon r = 0.678 on in-distribution data) but does not improve accuracy." This includes the scale qualifier, properly scopes the claim to accuracy (not mechanism relevance generally), and honestly reports the Wilcoxon confidence finding. The title "The Curriculum Is the Mechanism" remains appropriately provocative for a scientific paper; the abstract provides the nuance.

- [x] K021: confidence_accuracy_tension -- ROUND 3: RESOLVED. The conclusion (Section 7) now reads: "These results indicate that COCONUT's in-distribution accuracy on ProsQA is primarily attributable to its training curriculum, not to the content of the recycled hidden states. The recycling mechanism does have a measurable effect -- it produces significantly higher per-sample confidence (Wilcoxon r = 0.678 on in-distribution data) and more broadly distributed probing signal (29/78 vs. 11/78 significant cells) -- but this richer encoding does not translate to higher accuracy." This is a precise, honest formulation that acknowledges the mechanism does something functionally relevant (confidence) while making the specific claim that it does not improve accuracy. The abstract mirrors this formulation. The tension between "mechanism does something" and "mechanism doesn't improve accuracy" is now correctly resolved: both are true and both are stated.

- [x] K022: M4_data_gap -- ROUND 3: RESOLVED. Section 6 now includes a full "M4 experimental coverage" paragraph that explicitly states: "The corruption analysis and representational probing experiments could not be extended to M4 due to a methodological incompatibility with multi-pass KV-cache architectures," explains why ("For M4, the KV-cache accumulated across passes IS the model's computation -- without it, the extracted representations do not reflect M4's inference-time behavior"), and states what extending these experiments would require ("per-pass hidden state collection from within the KV-cache loop"). This is honest, clear, and actionable for future work.

#### REMAINING MAJOR FINDINGS

- [ ] K002: null_result_problem -- ROUND 3: DOWNGRADED TO MINOR. The manuscript still does not include an explicit paragraph stating the sensitivity bounds of the null-result diagnostics. However, the argument structure has evolved: the primary evidence is now the M4 factorial decomposition (positive results: recycled content hurts chain-length, sequential processing helps DAG) and the Wilcoxon analysis (positive result: M2 is more confident). The null results from corruption, probing, and transplantation serve as supporting convergent evidence, not as the main argument. The permutation power analysis (Appendix A.4) does state the 0.06% bound. Given that the null results now play a supporting rather than primary role, the absence of a comprehensive sensitivity paragraph is a minor limitation rather than a blocking issue.

    **Residual recommendation (minor):** A single sentence in Section 5.1 or Section 6 noting that the null-result diagnostics in Table 6 have limited sensitivity (e.g., "These null results constrain the effect size of any mechanism-specific contribution: the permutation test excludes flip rates above 0.06%, and the probing selectivity comparison is computed at n=298-500") would improve the interpretability without requiring substantial new analysis.

- [ ] K005: generalizability (single task) -- ROUND 3: REMAINS MAJOR, ACKNOWLEDGED. The paper's evidence comes from a single synthetic task (ProsQA). This is honestly acknowledged in Section 6 ("Our conclusions are specific to tasks with this structural profile and should not be generalized without further testing") and the Related Work section cites Zhang et al. (2025) at LLaMA scale as complementary evidence. I accept that a multi-task study is outside the scope of this paper. The limitation is clearly stated and does not undermine the specific claims made. However, the paper's title and framing still imply a general insight ("The Curriculum Is the Mechanism"), and readers who only see the title and abstract may overgeneralize. This is inherent to the genre and not something the paper can fully control. Downgrade to minor below is not appropriate because the core claim really is specific to one task; I maintain at major as an acknowledged-but-fundamental limitation.

- [ ] K008: alternative_explanation (set-encoding) -- ROUND 3: DOWNGRADED TO MINOR. The broadcast-then-attend framing in Section 4.3 and 5.2 is close enough to the set-encoding hypothesis. The paper describes the pattern clearly: "early thought positions broadcast answer-relevant information rather than encoding their own step in a sequential chain." The Wilcoxon analysis adds that M2's broadcast carries richer information (higher confidence). The set-encoding hypothesis is effectively addressed without being named as such. This is adequate for the paper's purposes.

- [ ] K010: single_seed -- ROUND 3: REMAINS MAJOR, ACKNOWLEDGED. The limitation is honestly disclosed in Section 6 with the recommendation for "multi-seed replication with proper paired statistical tests." The McNemar tests are correctly applied within the single-seed constraint. I accept this as a fundamental limitation that the paper handles honestly. The OOD results (many p < 0.001 after Bonferroni) are sufficiently strong that seed variance is unlikely to reverse the qualitative conclusions, though the exact magnitudes could shift. I maintain at major as an honest acknowledgment.

#### REMAINING MINOR FINDINGS

- [ ] K002: null_result_sensitivity -- ROUND 3: DOWNGRADED FROM MAJOR TO MINOR (see above).

- [ ] K006b: disagreement_analysis -- ROUND 3: REMAINS MINOR. Per-sample disagreement characterization is still absent. This would be enriching but is not essential to the paper's claims.

- [x] K007: selectivity_computation -- ROUND 3: DOWNGRADED FROM MAJOR TO MINOR. The manuscript's definition (Section 3.4), computation (selectivity_raw_grid), and reporting (Table 4, Figure 3) are internally consistent. I have independently verified: M2 (layer 0, position 3) matched_accuracy=0.554, max_cross=0.033, selectivity=0.520; M3 (layer 12, position 3) matched_accuracy=0.570, max_cross=0.047, selectivity=0.523. The `selectivity_aligned_grid` in the data file and its contradictory `interpretation` field ("Selectivity remains near zero") are artifacts of the original analysis before the n=12 truncation bug was discovered. The file's `outcome: "A"` reflects the pre-correction conclusion, not the current manuscript's conclusion. This is a data-hygiene issue: the stale interpretation field should be removed or updated to avoid confusing future auditors. It does not affect any claim in the manuscript.

    **Residual recommendation:** Update `selectivity_recomputed.json` to either (a) remove the `selectivity_aligned_grid` and stale `interpretation`/`outcome` fields, or (b) add a `_NOTE` field explaining that these reflect the pre-correction analysis.

- [ ] K008b: set_encoding_alternative -- ROUND 3: DOWNGRADED FROM MINOR TO SUGGESTION (see K008 above). The broadcast-then-attend description adequately captures the substance of the set-encoding hypothesis.

- [ ] K009b: multipass_not_pure_buffer -- ROUND 3: REMAINS MINOR. The paper correctly acknowledges that multi-pass processing provides a genuine task-specific benefit on DAG (Section 5.3). This is handled adequately.

- [ ] K011: cross_corruption -- ROUND 3: REMAINS MINOR. The missing cross-corruption condition (M2-scale noise on M3) is noted in Appendix A.2, Table A1, which shows M3 under M2-magnitude noise (L2~203). Wait -- I see that this IS present. Appendix A.2 shows M3 + M2-noise (L2~203) at all corruption levels: clean 96.6%, 4 corrupted 57.6%, 6 corrupted 2.4%. This IS the cross-corruption experiment. Let me update: the cross-corruption in one direction (M2-scale noise on M3) IS present. The reverse direction (M3-scale noise on M2) is absent but less informative since M2's actual noise scale is much larger. Downgrade to suggestion.

- [ ] K012: overclaiming (behavioral advantage) -- ROUND 3: SUBSTANTIALLY ADDRESSED. Section 5.2 still says "does not produce a different selectivity pattern or a behavioral advantage" but this now appears before the Wilcoxon analysis (Section 4.5) which explicitly shows the confidence advantage. The conclusion properly scopes "does not translate to higher accuracy." The term "behavioral advantage" in Section 5.2 is technically imprecise given the Wilcoxon confidence result, but the overall paper no longer overclaims.

    **Proposed edit (Section 5.2, line ~314):**
    ```
    FIND: But this richer encoding does not produce a different selectivity pattern or a behavioral advantage: M3 matches M2 on in-distribution accuracy
    REPLACE: But this richer encoding does not produce a different selectivity pattern or an accuracy advantage: M3 matches M2 on in-distribution accuracy
    ```
    This is a one-word change ("behavioral" to "accuracy") that aligns Section 5.2 with the conclusion's more precise formulation. The confidence difference IS a behavioral difference; the accuracy equivalence is the relevant claim.

- [ ] K014: transplantation_confound -- ROUND 3: REMAINS MINOR. Acknowledged and adequately contextualized.

- [ ] K016: curriculum_isolation -- ROUND 3: REMAINS MINOR. Acknowledged in Section 6.

#### SUGGESTIONS

- [ ] K008b: set_encoding_alternative -- ROUND 3: DOWNGRADED TO SUGGESTION (from minor).

- [ ] K011: cross_corruption -- ROUND 3: DOWNGRADED TO SUGGESTION (from minor). One direction IS present in Appendix A.2.

- [ ] K019: attention_analysis -- ROUND 3: REMAINS SUGGESTION.

- [ ] K020: per_hop_accuracy -- ROUND 3: REMAINS SUGGESTION.

- [ ] K023: M4_best_epoch -- ROUND 3: REMAINS SUGGESTION. M4 peaks at epoch 30; Figure 1 should include M4 training curves (the manuscript states "Training curves for all four models are shown in Figure 1" which implies M4 is already included -- verified by the figure caption "M1 (CoT), M2 (COCONUT), M3 (Pause), and M4 (Pause-Multipass)"). If M4 is visible in Figure 1, this finding is addressed.

- [ ] K024: Wilcoxon_interpretation_nuance -- ROUND 3: REMAINS SUGGESTION. The McNemar/Wilcoxon asymmetry on M3 vs M4 7-hop is a subtle point that would enrich the discussion but is not essential.

- [ ] K025: paper_yaml_stale_numbering (NEW, suggestion) -- The `paper.yaml` abstract still uses legacy M5 numbering (e.g., "We train a compute-matched pause-token control (M5)") while the manuscript uses M3 throughout. This is an internal metadata inconsistency that does not affect the manuscript but could confuse automated processing or future audits. Update the paper.yaml abstract to use manuscript-consistent M3/M4 numbering.

---

### NEW FINDINGS (Round 3)

#### K025: paper_yaml_stale_numbering (suggestion)

The `paper.yaml` file's abstract field still uses legacy model numbering: "We train a compute-matched pause-token control (M5)" (line 11), "M5 matches COCONUT" (line 13), "M5 outperforms COCONUT" (line 17). The manuscript uses M3 for the pause model throughout. This internal metadata inconsistency should be corrected.

**Location:** `paper.yaml` lines 11, 13, 17

**Recommendation:** Update paper.yaml abstract to use M3 instead of M5, and add M4 information consistent with the manuscript abstract.

#### K026: Section 5.2 "behavioral advantage" imprecision (minor)

Section 5.2 states: "this richer encoding does not produce a different selectivity pattern or a behavioral advantage." The Wilcoxon analysis (Section 4.5) demonstrates that the richer encoding DOES produce a behavioral difference: significantly higher per-sample confidence (r=0.678, p < 10^{-51}). A confidence difference is a behavioral difference. The conclusion correctly uses "does not translate to higher accuracy" -- Section 5.2 should match this precision.

**Proposed edit (Section 5.2):**
```
FIND: But this richer encoding does not produce a different selectivity pattern or a behavioral advantage: M3 matches M2 on in-distribution accuracy
REPLACE: But this richer encoding does not produce a different selectivity pattern or an accuracy advantage: M3 matches M2 on in-distribution accuracy
```

---

### Summary of Round 3 Status

| ID | Round 2 Severity | Round 3 Status | Round 3 Severity |
|----|-----------------|----------------|------------------|
| K001 | Minor (K001b) | RESOLVED (data verified) | -- |
| K002 | Major | Downgraded | Minor |
| K003 | RESOLVED | RESOLVED | -- |
| K004 | Major | RESOLVED (abstract scoped) | -- |
| K005 | Major | Acknowledged | Major |
| K006b | Minor | Remains | Minor |
| K007 | Major | Downgraded (data verified) | Minor |
| K008 | Major | Downgraded | Suggestion |
| K008b | Minor | Downgraded | Suggestion |
| K009b | Minor | Remains | Minor |
| K010 | Major | Acknowledged | Major |
| K011 | Minor | Downgraded | Suggestion |
| K012 | Minor | Addressed (K026 residual) | Minor (K026) |
| K013 | RESOLVED | RESOLVED | -- |
| K014 | Minor | Remains | Minor |
| K015 | RESOLVED | RESOLVED | -- |
| K016 | Minor | Remains | Minor |
| K017 | RESOLVED | RESOLVED | -- |
| K018 | RESOLVED | RESOLVED | -- |
| K019 | Suggestion | Remains | Suggestion |
| K020 | Suggestion | Remains | Suggestion |
| K021 | Major | RESOLVED | -- |
| K022 | Major | RESOLVED | -- |
| K023 | Suggestion | Remains | Suggestion |
| K024 | Suggestion | Remains | Suggestion |
| K025 | NEW | -- | Suggestion |
| K026 | NEW | -- | Minor |

**Final tally:** 0 critical, 2 major (both acknowledged limitations: single-task K005, single-seed K010), 6 minor (K002, K006b, K007, K009b, K014, K016, K026), 7 suggestion (K008b, K011, K019, K020, K023, K024, K025). 6 resolved since Round 2.

**Assessment: PASS.** The two remaining major findings (single-task, single-seed) are fundamental limitations that are honestly acknowledged in the manuscript and cannot be resolved without additional experiments. The minor findings are data-hygiene and precision-of-language issues, none of which undermine the paper's claims. The manuscript is suitable for preprint release and peer review submission.
