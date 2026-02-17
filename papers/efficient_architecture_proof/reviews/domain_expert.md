# Domain Expert (LLM/Mechanistic Interpretability) Review

**Assessment:** pass_with_conditions
**Date:** 2026-02-13T22:30:00Z

---

## Round 2 Review

**Assessment:** pass
**Date:** 2026-02-16T21:00:00Z
**Reviewer:** Domain Expert (Mechanistic Interpretability / LLM Specialist)

### Round 2 Summary

The manuscript has undergone substantial and effective revision since Round 1. I have re-read the full manuscript and re-verified all statistical claims against the underlying data files. The most significant changes since Round 1 are:

1. **New title:** "The Curriculum Is the Mechanism" replaces the binary "reasoning vs. buffering" framing. This directly addresses D001 and D014.
2. **M4 (Pause-Multipass) addition:** The factorial decomposition via M4 resolves the forward-pass asymmetry confound (D004) and provides a clean causal attribution of OOD performance differences. This is a major methodological improvement.
3. **Framing overhaul:** The paper no longer uses "buffering" as its central framing. Table 6 now acknowledges structured encoding. Section 5.1 explicitly states the evidence "does not support pure 'buffering' in the sense of unstructured generic computation." This addresses D001.
4. **Nonlinear probe fix:** Appendix A.7 now reports a proper grid search over 72 hyperparameter configurations, producing genuine MLP results that reveal a position-dependent pattern (linear-dominant at position 3, MLP-advantage at position 2). This resolves D009.
5. **Related work expanded:** Deng et al. (2024) and Zelikman et al. (2024) are now cited and discussed in Section 2.1. This addresses D007 and D017.
6. **Teacher-forced confidence analysis (Section 4.5):** The Wilcoxon signed-rank analysis adds a valuable dimension, demonstrating that recycled content produces higher confidence that becomes miscalibrated on OOD tasks.
7. **Accuracy discrepancy explained:** Table 2 note now explicitly explains the 1pp discrepancy between training-time and experiment-pipeline evaluation (D020).

The paper has moved from a binary framing to a nuanced factorial analysis. The remaining issues are largely matters of emphasis and additional context rather than fundamental methodological or interpretive flaws.

### Data Verification (Round 2)

I re-verified the following data-manuscript alignments:

- **Table 2 accuracy values:** M1=83.0%, M2=97.0%, M3=96.6%, M4=94.8%. Verified against `ood/results.json` (m3=0.97 for M2, m5=0.966 for M3), `paper.yaml` (M1_test=0.830, M2_test=0.970, M3_test=0.966, M4_test=0.948). All match.
- **Table 3 corruption values:** Verified against `corruption/results.json`. M2 (m3 in file) clean=0.97, forward=[0.968, 0.968, 0.968, 0.574, 0.156, 0.024]. M3 (m5 in file) clean=0.966, forward=[0.964, 0.962, 0.958, 0.572, 0.156, 0.022]. All match manuscript.
- **Table 4 probing values:** M2 peak=55.4% at (0,3), verified from `probing/results.json` m3 linear_probe_accuracy[0][3]=0.5537. M3 peak=57.0% at (12,3), verified from m5 linear_probe_accuracy[12][3]=0.5705. All match.
- **Table 4 selectivity values:** M2 position 3 = +52.0pp, verified from `selectivity_recomputed.json` m3 selectivity_raw_grid[0][3]=0.520 (layer 0). M3 position 3 = +52.3pp, verified from m5 selectivity_raw_grid[12][3]=0.523 (layer 12). All match.
- **Table 5a OOD accuracy:** Verified against `ood/results.json` for M2 and M3. M4 values verified against `paper.yaml` factorial_decomposition data. All match.
- **Table 5b McNemar:** Verified against `mcnemar/results.json`. Contingency tables match. p-values match.
- **Table 5c factorial McNemar:** The manuscript states "McNemar comparisons recomputed using experiment-pipeline per-sample predictions for all models." The values are internally consistent (M2=97.0%, M3=96.6%, M4=94.8%). The file `m6/mcnemar.json` uses training-time evaluation for the reference model (M2=98.0%), which is explicitly deprecated in that file's header. The manuscript numbers are correct.
- **Table 7 Wilcoxon:** M2 vs M4 ID: r=0.678, verified from `wilcoxon_teacher_forced_m3_vs_m6.json` prosqa_id effect_size_r=0.678. Direction m3>m6, which maps to M2>M4. All match. M3 vs M4 ID: r=0.286, verified from `wilcoxon_teacher_forced_m5_vs_m6.json` prosqa_id effect_size_r=0.286. Direction m5>m6, which maps to M3>M4. All match. M2 vs M3 ID: r=0.591, verified from `wilcoxon_teacher_forced_m3_vs_m5.json` prosqa_id effect_size_r=0.591. Direction m3>m5, which maps to M2>M3. All match.
- **MLP grid search (Appendix A.7):** Verified against `mlp_probe_grid_search.json`. M2 (layer 0, pos 3): linear=55.4%, best MLP=46.0%, advantage=-9.4pp. M2 (layer 12, pos 2): linear=19.0%, best MLP=29.2%, advantage=+10.2pp. M3 (layer 12, pos 3): linear=57.0%, best MLP=45.6%, advantage=-11.4pp. M3 (layer 12, pos 2): linear=22.0%, best MLP=29.6%, advantage=+7.6pp. All match.
- **M4 corruption artifact:** `m6/corruption.json` shows 2.4% accuracy at all corruption levels including zero, with explicit artifact warning. The manuscript correctly excludes M4 from Table 3 and explains why.
- **M4 probing exclusion:** `probing/results.json` contains only m3 and m5 data (no m6). The manuscript correctly excludes M4 from Table 4 and explains the cold-start extraction methodology limitation.

All data-manuscript alignments verified. No discrepancies found.

### Round 2 Checklist Updates

#### MAJOR

- [x] D001: interpretation -- ROUND 2: SUBSTANTIALLY RESOLVED. The new title ("The Curriculum Is the Mechanism") and rewritten framing eliminate the binary "reasoning vs. buffering" dichotomy. Section 5.1 now explicitly acknowledges that "the evidence does not support pure 'buffering' in the sense of unstructured generic computation: the strong selectivity at position 3 (+52pp) and anti-selectivity at positions 0--1 reveal structured, position-specific encoding in both models." Table 6 is revised accordingly. The paper now correctly frames the finding as "curriculum-driven structured computation that does not require the recycling mechanism." Minor residual: the conclusion (Section 7, line ~368) still uses "serve primarily as curriculum-shaped computational scaffolding." While "scaffolding" is more accurate than "buffering," it could be read as implying content-free structure. The abstract avoids this phrasing and instead precisely states the factorial results. The conclusion could benefit from matching the abstract's precision, but this is a minor stylistic point that does not affect the paper's claims.

- [x] D002: interpretation -- ROUND 2: PARTIALLY RESOLVED. The paper now uses "broadcast-then-attend" language in Section 4.3 and describes positions 0-2 as broadcasting "answer-relevant (later-step) information to early thought positions, where it becomes accessible to all subsequent positions through causal self-attention." However, I verified that the corruption discussion (Section 4.2, line 167) still states "positions 0--2 carry redundant information" and Appendix A.5 (line 470) repeats "positions 0--2 are largely redundant." The word "redundant" does not capture the distributed-then-integrate strategy that the probing data supports. The information at positions 0-2 is mutually redundant (they encode similar answer-relevant content, as shown by the anti-selectivity pattern), so corrupting any single early position is compensated by the remaining copies. This is importantly different from "unnecessary" or "unused." The proposed edit below clarifies this distinction at both locations.

**Proposed edit (Section 4.2, line ~167):**
```
FIND: indicating that positions 0--2 carry redundant information
REPLACE: indicating that positions 0--2 carry mutually redundant information -- each encodes similar answer-relevant content (consistent with the anti-selectivity pattern in Section 4.3), so corrupting any individual early position is compensated by the remaining copies
```

**Proposed edit (Appendix A.5, line ~470):**
```
FIND: position 3 carries critical information while positions 0--2 are largely redundant
REPLACE: position 3 carries critical information while positions 0--2 carry mutually redundant copies of answer-relevant content
```

- [x] D003: interpretation -- ROUND 2: PARTIALLY RESOLVED. The paper now discusses the permutation result more carefully (Section 4.2, line ~171: "Both models treat thought positions as an unordered bag of compute with respect to final predictions: permuting latent tokens does not change the model's output. This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction."). The final sentence is a good caveat. However, the paper still does not explicitly discuss the tension between permutation insensitivity and single-position corruption criticality at position 3 -- which is the most theoretically interesting aspect and the most informative constraint on the computational mechanism. Position 3's activation can be moved to any slot without affecting output (0% flip rate across 5000 trials), yet destroying that activation collapses accuracy to ~57%. The most parsimonious explanation is content-based attention routing: the model attends to activations by their content (learned features), not by their absolute position. This interpretation is testable (e.g., by examining whether the attention weights from the answer-generation token track position 3's content regardless of slot) and would strengthen the "broadcast-then-attend" narrative. Adding two sentences would meaningfully tighten the mechanistic argument.

**Proposed edit (Section 4.2, after the permutation paragraph, line ~171):**
```
FIND: This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction.
REPLACE: This does not rule out order-sensitive internal representations that are ultimately redundant for the final prediction. The contrast with single-position corruption is informative: position 3 is critical to corrupt (Section A.5) but irrelevant to permute, suggesting that the model retrieves information from thought positions by content-based attention rather than position-indexed lookup. Moving position 3's activation to a different slot does not disrupt access because the attention mechanism locates it by content; destroying it does disrupt access because the content itself is lost.
```

- [x] D004: technical_accuracy -- ROUND 2: RESOLVED. The addition of M4 (Pause-Multipass) directly addresses this confound. M4 matches M2's sequential processing structure (6 passes via KV-cache incremental decoding) while using fixed pause embeddings, cleanly separating the content and processing factors. Section 3.2 now explicitly describes the factorial design and Section 5.3 provides the decomposition. The paper addresses the residual asymmetry in the Limitations section (Section 6, "Forward-pass asymmetry"): "M4 and M2 process the same number of sequential steps, but the information available at each step differs qualitatively (fixed embedding vs. a representation that reflects accumulated state). This is an inherent property of the recycling mechanism and cannot be further decomposed without more invasive interventions." This is a complete and honest resolution of the original concern.

- [x] D005: missing_experiments -- ROUND 2: DEFERRED (acknowledged limitation). The curriculum-only ablation remains missing, and the Limitations section (Section 6, "Curriculum isolation") continues to acknowledge this gap: "we do not test a curriculum-only condition in which removed reasoning tokens are simply deleted, producing shorter sequences with no additional attention positions." While this ablation would strengthen the paper, the M4 factorial design provides sufficient evidence for the paper's narrower claim (that the recycling mechanism is not the causal driver). The curriculum-only ablation would address a broader question (whether attention positions are needed at all) that is adjacent to but not central to the paper's contribution. The Pfau et al. (2024) citation (Section 2.3) appropriately contextualizes this gap by noting that even meaningless filler tokens expand the computational class. Acceptable as a stated limitation.

- [x] D011: missing_experiments -- ROUND 2: DEFERRED (acceptable). Attention pattern analysis would strengthen the interpretive claims but is no longer critical given the M4 factorial decomposition. The probing + corruption + M4 evidence is sufficient to support the paper's claims without attention visualization. The "broadcast-then-attend" interpretation (Section 4.3, Section 5.2) remains a plausible but unconfirmed hypothesis. The paper could soften this slightly by noting it is an interpretation consistent with the probing data rather than a demonstrated mechanism, but the current language ("consistent with a broadcast-then-attend strategy") is already appropriately hedged. This is a desirable addition for a future version but does not block the current paper.

#### MINOR

- [x] D006: related_work -- ROUND 2: PARTIALLY RESOLVED. The paper now cites Meng et al. (2022) for causal analysis, Ravichander et al. (2021) for probing, Zelikman et al. (2024) for Quiet-STaR, and Deng et al. (2024) for CoT distillation. The broader mechanistic interpretability toolkit (Olsson et al. 2022 on induction heads, Conmy et al. 2023 on ACDC) remains uncited, but this is now a minor gap rather than a significant omission. The paper's scope is not mechanistic interpretability per se but rather curriculum vs. mechanism attribution, so exhaustive coverage of the mech interp literature is not required. Acceptable as is.

- [x] D007: related_work -- ROUND 2: RESOLVED. Deng et al. (2024) is now cited in Section 2.1: "Deng et al. (2024) demonstrated that models can be distilled from explicit chain-of-thought into implicit reasoning through progressive removal of intermediate steps, suggesting that the training curriculum itself -- rather than any particular latent mechanism -- may be the key ingredient." This is exactly the context I recommended.

- [x] D008: technical_accuracy -- ROUND 2: RESOLVED. The abstract now explicitly states "M3 outperforms COCONUT on 3 of 4 out-of-distribution test sets" (the "out-of-distribution" qualifier is present in the abstract). The introduction (paragraph 4) enumerates the specific test sets: "M3 outperforms M2 on 7-hop, 8-hop, and dense graphs by 7--9 percentage points." No ambiguity remains.

- [x] D009: technical_accuracy -- ROUND 2: RESOLVED. Appendix A.7 now reports a proper grid search over 72 hyperparameter configurations (6 hidden sizes x 3 learning rates x 4 regularization strengths). I re-verified all five grid search results against `mlp_probe_grid_search.json`. Every value matches. The interpretation is sound: the position-dependent pattern (linear-dominant at position 3 due to overfitting with n=298, nonlinear advantage at position 2 with n=500) is consistent with a representational strategy that places the final answer in a linearly accessible format while using more complex encoding for intermediate steps. The original probing/results.json still shows 0.0 for all nonlinear_probe_accuracy cells (these are from the original broken default-hyperparameter run), but the manuscript correctly cites the grid search results rather than these stale values.

- [x] D010: technical_accuracy -- ROUND 2: PARTIALLY RESOLVED. Table 4 now states that selectivity is reported at the peak accuracy layer for each position, with explicit layer indices. I re-verified from `selectivity_recomputed.json`: M2 (m3 in file) position 3 selectivity ranges from 0.221 (layer 8) to 0.523 (layer 12); M3 (m5 in file) position 3 selectivity ranges from -0.007 (layer 8) to 0.523 (layer 12). The M3 data is particularly notable: selectivity at position 3 is near zero or negative for layers 0-10, then jumps sharply to 0.49 at layer 11 and 0.52 at layer 12. This late emergence is consistent with the paper's statement that "M3 builds its representations through the transformer stack, with peak accuracy at layer 12." The peak-layer reporting is standard practice for probing studies, and Table 4 now explicitly states which layers are used. No further action required; reporting the layer-selectivity range would add transparency but is not necessary for the claims made.

- [x] D012: novelty -- ROUND 2: RESOLVED. The paper now positions its contributions relative to Zhang et al. (2025) in Section 5.4 with appropriate precision. The M4 factorial decomposition substantially strengthens the novelty beyond Zhang et al.: (a) constructive alternative (curriculum-matched baselines) rather than ablation alone; (b) factorial decomposition of the confounded factors via M4; (c) demonstration that recycled content is not merely inert but actively harmful on OOD chain-length tasks; (d) Wilcoxon confidence-accuracy dissociation showing that recycled content produces miscalibrated overconfidence. These are genuinely distinct contributions that the paper articulates clearly. The introduction explicitly delineates three contributions (paragraph 5).

- [x] D013: interpretation -- ROUND 2: RESOLVED. The DAG result is now attributed to sequential processing rather than recycled content via the M4 factorial decomposition. Section 5.3 states: "M4 outperforms M3 by 7.9pp on DAG (p < 0.001), while M4 and M2 do not differ" and "The sequential structure provides an inductive bias that helps with novel graph topologies -- the forced step-by-step accumulation of information across passes may implicitly encourage a search strategy better suited to DAG structures than M3's parallel processing." The "may implicitly encourage" hedging is appropriate. The speculative "path convergence" language from Round 1 has been replaced with empirically grounded factorial attribution.

- [x] D014: framing -- ROUND 2: RESOLVED. The new title "The Curriculum Is the Mechanism: Dissecting COCONUT's Latent Thought Gains on ProsQA" captures the main finding accurately. The paper now frames the question as "curriculum vs. mechanism" rather than "reasoning vs. buffering." The introduction explicitly sets up the confound and explains how M3 and M4 resolve it.

- [x] D018: technical_accuracy -- ROUND 2: PARTIALLY RESOLVED. Section 3.2 now discusses the processing structure difference carefully. However, I re-verified that the manuscript still does not explicitly note the key connectivity difference: in M3's single-pass architecture, thought position k can attend to all prior thought positions 0..k-1 via standard causal attention, while in M2's multi-pass architecture, each thought position accesses only the recycled hidden state at its own position from the previous pass. M4's multi-pass architecture similarly processes each thought token in isolation (only attending to input tokens and prior KV-cache entries, not other thought positions' current-pass representations). The fact that M3's richer attention connectivity produces comparable or better results is itself informative and directly relevant to the M3 vs M4 comparison (M3 has richer connectivity but fewer passes; M4 has sequential passes but no inter-thought-token attention within a pass). This is worth a sentence because it constrains how sequential processing helps on DAG: it is the iterative state accumulation in the KV-cache, not inter-thought-position attention, that matters.

**Proposed edit (Section 3.2, after the M3 description, line ~76):**
```
FIND: The only position-distinguishing signal available to the model is GPT-2's learned positional encoding; the pause embeddings themselves carry no position-specific information.
REPLACE: The only position-distinguishing signal available to the model is GPT-2's learned positional encoding; the pause embeddings themselves carry no position-specific information. Notably, M3's thought positions can attend to all prior thought positions via standard causal self-attention, providing richer inter-position connectivity than M2's sequential recycling chain or M4's sequential multi-pass structure, where each thought position is processed in isolation with access only to the accumulated KV-cache from prior passes.
```

- [x] D020: technical_accuracy -- ROUND 2: RESOLVED. Table 2 now includes a detailed note explaining the discrepancy. The manuscript consistently uses experiment-pipeline numbers (M2: 97.0%, M3: 96.6%) throughout. Verified: corruption results.json shows clean_accuracy = 0.97 for M2 (m3) and 0.966 for M3 (m5), matching Table 3. The `m6/mcnemar.json` file's header explicitly marks it as deprecated ("This file uses training-time evaluation predictions... Do not use this file for manuscript claims."). No issues.

#### SUGGESTION

- [x] D015: missing_experiments -- ROUND 2: DEFERRED. Per-hop-count ID breakdown remains absent but is not critical. All curriculum models are near-ceiling on ID accuracy; the meaningful variation is in OOD generalization, which is thoroughly analyzed via the factorial design.

- [x] D016: missing_experiments -- ROUND 2: DEFERRED (acknowledged). Section 5.4 continues to acknowledge this limitation appropriately.

- [x] D017: concurrent_work -- ROUND 2: RESOLVED. Zelikman et al. (2024) Quiet-STaR is now cited in Section 2.1. The related work coverage is adequate for the paper's scope.

- [x] D019: impact -- ROUND 2: DEFERRED (acceptable). Section 5.5 recommends curriculum design as the higher-leverage investment but does not decompose which curriculum properties matter. This is a reasonable scope boundary.

- [x] D021: framing -- ROUND 2: PARTIALLY RESOLVED. The paper now better contextualizes M3 as a curriculum-matched control. Section 5.5 notes that "simpler architectures that exploit the same curriculum may achieve comparable performance with lower engineering and computational cost." The broader theoretical implication -- that standard transformer attention suffices for multi-hop reasoning when paired with appropriate curriculum design, without architectural modifications -- is implicit but not explicitly stated. This is a minor presentation point that does not affect the paper's validity.

### New Findings (Round 2)

#### D022 (new, minor): Wilcoxon interpretation precision

The Wilcoxon analysis (Section 4.5) states that M2 assigns "significantly higher confidence" than M4 on the ID test set (r = 0.678). I verified from `wilcoxon_teacher_forced_m3_vs_m6.json`: M2 (m3) median probability = 99.998%, M4 (m6) median probability = 99.949%. Both are essentially ceiling, differing by 0.049 percentage points in probability space. The Wilcoxon test is significant because it operates on paired rank differences in log-probability space, where even small absolute differences produce consistent rank orderings across 500 samples (z = 15.17). The interpretation "recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence" is technically correct but the practical magnitude is vanishingly small on ID data. The paper should note that the ID confidence difference, while statistically robust, is measured against a ceiling where all models assign >99.9% probability to the correct answer, and its practical significance lies not in the ID magnitude but in the OOD extrapolation pattern where the same mechanism produces miscalibration.

The paper partially addresses this by focusing the interpretation on the OOD confidence-accuracy dissociation (paragraphs 2-3 of Section 4.5), which is where the practical significance lies. But the in-distribution paragraph does not contextualize the absolute magnitude. A single sentence noting that all models assign >99.9% median probability on ID data would help readers calibrate the practical significance of the r = 0.678 effect size.

**Proposed edit (Section 4.5, in-distribution confidence hierarchy paragraph, line ~283):**
```
FIND: the recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence, even when both models achieve comparable binary accuracy (97.0% vs. 94.8%, McNemar p_Bonf = 0.354).
REPLACE: the recycled hidden states carry reasoning-relevant information that translates to measurably higher per-sample confidence, even when both models achieve comparable binary accuracy (97.0% vs. 94.8%, McNemar p_Bonf = 0.354). The absolute magnitude of this difference is small -- all three models assign median probabilities above 99.9% to the correct answer on ID data -- but the consistency of the paired differences produces a large rank-based effect size.
```

**Location:** Section 4.5 (in-distribution confidence hierarchy paragraph)

#### D023 (new, minor): M4 McNemar data file deprecation

The file `m6/mcnemar.json` uses training-time evaluation predictions for the reference model (M2=98.0% vs the experiment-pipeline value of 97.0%). The file already includes a deprecation header ("_DEPRECATED: This file uses training-time evaluation predictions... Do not use this file for manuscript claims."). The manuscript's Table 5c values are internally consistent and use experiment-pipeline predictions throughout. No manuscript change needed. The deprecation header is sufficient; if a repository README is added, a note pointing readers to Table 5c rather than this raw file would be helpful but is not required.

**Location:** `results/experiments/m6/mcnemar.json`

#### D024 (new, suggestion): Confidence-accuracy dissociation as standalone contribution

The confidence-accuracy dissociation on OOD tasks (Section 4.5) is the paper's most novel finding from a domain perspective. The observation that recycled content makes M2 simultaneously more confident and less accurate on out-of-range problems (M2 > M4 in confidence on 7-hop with r = 0.109, p = 0.003, while M2 < M4 in accuracy by 10.9pp, p < 0.001) has direct implications for safety and deployment of latent reasoning systems: hidden-state recycling may produce well-calibrated in-distribution predictions but confidently incorrect OOD predictions. This pattern -- where the mechanism is not merely inert but actively misleading -- goes beyond Zhang et al.'s (2025) "causal inertness" finding and could be positioned more prominently. The paper currently treats this as supporting evidence for the factorial decomposition rather than as a standalone result. In a future revision, this finding could be elevated to a named contribution in the introduction.

**Location:** Section 4.5, Section 5.3, Introduction (contribution list)

#### D025 (new, minor): paper.yaml abstract uses Lambda-era model numbering

The `paper.yaml` abstract still uses "M5" to refer to the pause baseline, while the manuscript consistently uses "M3." The abstract in paper.yaml states "We train a compute-matched pause-token control (M5)..." This does not affect the manuscript itself but creates an inconsistency in the repository metadata. This should be updated to use M3 for consistency with the manuscript, or a mapping note should be added to paper.yaml.

**Location:** `paper.yaml`, lines 11-24

#### D026 (new, suggestion): Terminology consistency -- "compute buffer" vs "computational scaffolding"

The manuscript uses three different phrases to describe the non-reasoning interpretation: "compute buffer" (Sections 3.4, Experiment 1 and 2 predictions), "computational scaffolding" (Section 7 conclusion), and "curriculum-driven computation" (Section 5.1, throughout). Of these, "curriculum-driven computation" is the most precise and best supported by the data, and it is the dominant framing. The residual use of "compute buffer" in the experiment prediction sections (lines 102, 113) is appropriate because it sets up the original binary hypothesis that the experiments then refine. The use of "scaffolding" in the conclusion (line 368) is potentially misleading because scaffolding implies temporary support that is removed after construction, whereas the thought positions are permanent computational resources. The paper could use "curriculum-driven computation" consistently in the conclusion to match the discussion section. This is a minor consistency issue.

### Round 2 Overall Assessment

**Assessment: pass**

The manuscript has addressed the critical and most of the major issues from Round 1. The M4 factorial design resolves the forward-pass asymmetry confound, the framing overhaul eliminates the false dichotomy, the nonlinear probe issue is fixed with proper grid search, and the related work is now adequate. All data-manuscript alignments verified with no discrepancies found across all tables (2, 3, 4, 5a-c, 7) and appendix tables (A1-A7).

The remaining actionable items from Round 1 are three minor wording improvements:
1. **D002**: Replace "redundant" with "mutually redundant" at two locations (Section 4.2 line ~167, Appendix A.5 line ~470)
2. **D003**: Add 2 sentences on the permutation-corruption tension and content-based attention routing (Section 4.2 line ~171)
3. **D018**: Add 1 sentence on inter-position attention connectivity (Section 3.2 line ~76)

The new findings (D022-D026) are minor or suggestion-level; none are blocking. The paper makes a clear, well-evidenced contribution to understanding COCONUT's latent reasoning mechanism. No critical or blocking issues remain.

---

## Summary

This paper makes a genuine and valuable contribution by constructing a well-controlled curriculum-matched baseline (M5) that isolates the role of COCONUT's hidden-state recycling mechanism from its training curriculum. The experimental methodology is thorough: the corruption, probing, and OOD generalization experiments converge on a consistent picture, and the statistical analysis is rigorous (McNemar's test with Bonferroni correction, independently verified). The single-seed limitation is honestly acknowledged, and the data files show meticulous verification.

However, the paper's central framing -- the binary 'reasoning vs buffering' dichotomy -- is not well-supported by its own evidence. The probing results show +52pp selectivity at position 3 in both models, which is not 'buffering' in any conventional sense. Both models learn structured, position-specific representational strategies driven by the shared curriculum. The correct conclusion is not that COCONUT 'buffers' but that its curriculum -- not its recycling mechanism -- drives a structured computational strategy that does not require inter-step hidden-state propagation. This is a more nuanced and arguably more interesting finding than the binary framing suggests. The nonlinear probe results (0.0% accuracy everywhere) indicate a clear implementation failure that should be fixed or removed. The missing curriculum-only ablation and attention pattern analysis would substantially strengthen the interpretive claims.

The paper's strongest contributions are: (1) the M5 control methodology, which is applicable beyond COCONUT to any architecture claiming gains from a mechanism confounded with a curriculum; (2) the OOD generalization finding that the recycling mechanism introduces a task-dependent tradeoff rather than a uniform benefit; and (3) the demonstration that curriculum design is the higher-leverage intervention. With the framing adjusted and the nonlinear probe issue resolved, this paper would make a solid contribution to the understanding of latent reasoning architectures.

**Findings:** 21 total -- 6 major, 10 minor, 5 suggestion (Round 1)
**Round 2 status:** 26 total -- 6 major (4 resolved, 2 deferred), 10 minor (6 resolved, 4 partially resolved), 5 suggestion (2 resolved, 3 deferred), 5 new (3 minor, 2 suggestion)
**Blocking issues:** 0
**Actionable edits proposed:** 5 (all minor wording changes, total ~8 sentences)

---

## 🟠 MAJOR

### - [ ] D001: interpretation

The selectivity values reported in Table 4 and Section 4.3 (+52.0pp and +52.3pp at position 3) genuinely indicate step-specific encoding, which is more consistent with a structured representational strategy than with pure 'buffering'. The paper frames this as arising from the curriculum, but the fact that position 3 specifically encodes the entity corresponding to the final hop -- with 52 percentage points of selectivity over the best control position -- suggests that these representations are not 'generic compute buffers' in any standard sense of the term. A compute buffer, by definition, carries no task-relevant structure; these positions clearly do. The paper's own probing results contradict the headline 'buffering' framing. The paper partially acknowledges this tension (Section 5.2) but does not fully resolve it: the convergent evidence table (Table 6) lists probing selectivity under 'general broadcast' for the buffering claim, but +52pp selectivity at a specific position with anti-selectivity at others is precisely the opposite of a broadcast pattern. This is a structured, position-specific encoding strategy, shared between M3 and M5. The correct conclusion is not 'buffering vs reasoning' but rather 'curriculum-driven structured computation that does not require the recycling mechanism.' The binary framing obscures what is actually a more interesting and nuanced finding.

**Location:** Section 4.3, Table 4, Table 6 (Section 5.1)

**Evidence:** Table 4: Position 3 selectivity = +52.0pp (M3), +52.3pp (M5). Table 6 labels this as 'General broadcast' under the buffering claim, but the data show position-specific encoding with a clear hierarchy (positions 0-1 anti-selective, position 2 mildly selective, position 3 strongly selective).

**Recommendation:** Reframe the paper away from the binary 'reasoning vs buffering' dichotomy. The more precise conclusion is that both models learn a structured representational strategy driven by the curriculum, with step-relevant information concentrated at specific positions, but this strategy does not require the hidden-state recycling mechanism. Consider replacing 'buffering' with 'curriculum-driven structured computation' or similar language that does not imply the absence of task-relevant organization.

### - [ ] D002: interpretation

The corruption cliff at position 4 does not straightforwardly mean that the first 3 positions (0-2) are 'redundant'. Combined with the probing data showing that positions 0-1 encode later-step (answer-relevant) information with anti-selectivity, and position 2 encodes mild step-specific content, these positions may serve a preparatory or routing function. In causal attention, positions 0-2 are visible to positions 3-5. If the model's strategy is to place answer-relevant information at early positions where it can be attended to from all later positions, then corrupting those positions should not matter -- until the model loses the positions that actually use that information (position 3+). This is not 'redundancy' but a deliberate two-phase computation: distribute information early, then integrate at position 3. The single-position corruption data supports this: corrupting position 3 alone causes the cliff, but corrupting positions 0-2 individually does not. The paper partially captures this with the 'broadcast-then-attend' language in Section 4.3 but then reverts to calling the positions 'redundant' in the corruption discussion.

**Location:** Section 4.2, Section 4.3

**Evidence:** Table A4: Corrupting position 0, 1, or 2 alone causes no accuracy drop (<1pp). Corrupting position 3 alone drops accuracy to ~57.6%. Positions 0-1 show anti-selectivity (encode step 2 better than their own matched step). This pattern is consistent with a distributed computation strategy, not redundancy.

**Recommendation:** Replace the language of 'redundancy' for positions 0-2 with a more precise characterization. These positions appear to serve a distributional role (broadcasting answer-relevant information across the sequence for causal attention access) while position 3 serves as the integration/decision point. This is a specific computational strategy, not mere redundancy.

### - [ ] D003: interpretation

The permutation insensitivity result (0% flip rate across 5000 trials) is presented as strong evidence against sequential encoding. However, the paper's own probing analysis shows that position-specific information is concentrated at position 3, with positions 0-2 encoding qualitatively similar content (later-step entities with anti-selectivity). Permuting positions that encode similar content would not produce prediction flips even if the model were performing position-sensitive computation, because the swapped activations carry approximately interchangeable information. The critical test would be to permute position 3 with a non-critical position (e.g., position 0), which should disrupt the computation if position 3 is functionally special. But this is exactly what the permutation test does (random permutations include such swaps), and the result is still zero flips. The paper should more carefully reason through why zero flips occurs even when position 3 (which is demonstrably critical, per single-position corruption) is moved. One possible explanation: the model attends based on activation content rather than absolute position, so moving position 3's activation to slot 0 does not matter because the attention mechanism can still find it by content matching. This would be consistent with 'structured computation that is position-agnostic in its attention routing' rather than 'buffering'.

**Location:** Section 4.2 (Permutation sensitivity)

**Evidence:** Permutation flip rate = 0.0% across 5000 trials for both M3 and M5. But single-position corruption at position 3 causes catastrophic failure (~57% accuracy). This creates a puzzle: position 3 is critical to corrupt but irrelevant to move. The resolution likely involves attention-based content retrieval rather than positional dependence, which is a more specific mechanism than 'buffering'.

**Recommendation:** Add a paragraph discussing the tension between permutation insensitivity and single-position corruption criticality. The most parsimonious explanation is that the model routes information via content-based attention rather than position-indexed lookup, making the physical position of the critical activation irrelevant. This is an important architectural insight that gets lost in the 'buffering' framing.

### - [ ] D004: technical_accuracy

The paper states that M5 performs 'a single forward pass over the entire sequence' (Section 3.1, bullet 3), while M3 performs 'six sequential forward passes'. This is a fundamental computational asymmetry that goes beyond the hidden-state recycling mechanism. M3 gets 6x the transformer depth (72 effective layers vs 12) for the thought-token region of the sequence. The paper acknowledges the FLOP difference (Section 3.2, final paragraph) but frames it as favoring the paper's argument. However, this confounds two separate claims: (1) M5 matches M3 with less compute (fair), and (2) M3's recycled hidden states do not carry useful information (the corruption/probing claims). For claim (2), the computational asymmetry is problematic: M3 may be using those 6 forward passes to build up the representation at position 3 that the probes detect, and the recycling may be essential for that build-up process even though the final result is phenomenologically similar to what M5 achieves in a single pass via a different route. The transplantation experiment partially addresses this (M5 can use M3's thought representations and vice versa), but the 6x depth difference makes the two models less directly comparable than the paper implies.

**Location:** Section 3.1, Section 3.2 (final paragraph)

**Evidence:** M3: 6 sequential forward passes of 12 layers each = 72 effective layers for thought computation. M5: 1 forward pass of 12 layers = 12 effective layers. Paper acknowledges 'approximately one-sixth the inference-time FLOPs' but does not discuss implications for the probing and corruption comparisons.

**Recommendation:** Discuss the depth asymmetry more explicitly in the methods and interpretation. Consider framing it as: 'M3 builds representations through 6x the computational depth but arrives at a functionally equivalent result, suggesting that the additional depth is not leveraged for qualitatively different computation.' Also note that the probing comparison at 'layer 0' for M3 (where recycled states are injected) versus 'layer 12' for M5 reflects this asymmetry -- M3's layer 0 states have already been through 12 layers of processing in the previous pass.

### - [ ] D005: missing_experiments

The paper's central claim -- that the curriculum drives performance rather than the recycling mechanism -- would be substantially strengthened by a 'curriculum-only' ablation that the Limitations section (Section 6) acknowledges is missing. Without this ablation, the paper cannot distinguish between two hypotheses: (a) the curriculum is sufficient (no extra attention positions needed), and (b) the curriculum requires additional attention positions to deposit intermediate computations, and pause tokens provide those positions. Hypothesis (b) is still consistent with 'buffering' but would mean that the number and presence of thought positions matters, which is a more specific claim than 'the curriculum drives everything.' This is a significant gap because the Pfau et al. (2024) result (filler tokens expand computational class) suggests that additional positions are likely necessary, not just the curriculum.

**Location:** Section 6 (Limitations, 'Curriculum isolation' paragraph)

**Evidence:** The paper states: 'we do not test a curriculum-only condition in which removed reasoning tokens are simply deleted, producing shorter sequences with no additional attention positions.' This is a major missing ablation that would resolve the paper's central ambiguity.

**Recommendation:** If feasible, run a 'curriculum-only' ablation where the CoT tokens are removed at each stage but no pause tokens are inserted (the sequence simply gets shorter). If this model performs comparably to M5, the curriculum alone is sufficient. If it performs much worse, the additional attention positions are a necessary computational resource, which would support a 'compute buffering' interpretation more precisely. At minimum, acknowledge this more prominently than in the limitations section -- it is a core interpretive ambiguity, not a peripheral limitation.

### - [ ] D011: missing_experiments

The paper lacks attention pattern analysis. Given that the key claim is about how information flows (or does not flow) through thought positions, examining attention weights would provide direct evidence. Specifically: (1) Does position 3 receive more attention from the answer-generating position than other thought positions? (2) Do early thought positions (0-2) attend heavily to input tokens, suggesting they are building representations from the input rather than from prior thought tokens? (3) In M3, does the sequential forward-pass structure produce different attention patterns than M5's single-pass architecture? Attention patterns would distinguish between the 'broadcast-then-attend' strategy the paper proposes and alternative information routing strategies. This is standard methodology in mechanistic interpretability and its absence is notable.

**Location:** Missing from the manuscript

**Evidence:** The paper proposes a 'broadcast-then-attend' strategy (Section 4.3, Section 5.2) but provides no attention analysis to support it. The probing results show where information is stored but not how it flows.

**Recommendation:** Add an attention pattern analysis for both M3 and M5. Visualize attention from the answer-generating position to thought positions, and from thought positions to input positions. This would directly test the 'broadcast-then-attend' interpretation and could reveal important differences between M3 and M5 that the current methodology misses.


## 🟡 MINOR

### - [ ] D006: related_work

The paper does not cite or discuss several relevant lines of work in mechanistic interpretability. Specifically: (1) Nostalgebraist's work on interpreting GPT-2 internals, which is directly relevant since the base model is GPT-2. (2) The information bottleneck perspective from Shwartz-Ziv and Tishby, which provides a theoretical framework for understanding why intermediate representations might compress information. (3) Olsson et al. (2022) on induction heads, which is relevant to how the model might be routing information across thought positions via attention. (4) Conmy et al. (2023) on automated circuit discovery (ACDC), which would provide a more targeted methodology for identifying which circuits use the thought positions. (5) Bills et al. (2023) on automated interpretability, relevant to understanding what the thought position representations encode. None of these are strictly required, but the paper's probing methodology would benefit from being situated in the broader mechanistic interpretability toolkit.

**Location:** Section 2.4, Section 2 generally

**Evidence:** The related work section cites Meng et al. (2022) for causal analysis and Ravichander et al. (2021) for probing, but does not engage with the broader mechanistic interpretability literature.

**Recommendation:** Add a brief paragraph in Section 2.4 situating the probing methodology within the broader mechanistic interpretability literature. Consider citing Olsson et al. (2022) on induction heads and attention-based information routing, and note that circuit-level analysis (e.g., via ACDC or path patching) would provide a more targeted decomposition than linear probing.

### - [ ] D007: related_work

The paper does not discuss Deng et al. (2024) 'Explicit CoT Training for Implicit CoT Reasoning' or related work on distilling chain-of-thought into implicit reasoning, which is closely related to the COCONUT curriculum. This line of work suggests that models can learn to perform implicit reasoning when trained with explicit supervision that is gradually removed -- which is exactly the curriculum mechanism that this paper identifies as the key driver.

**Location:** Section 2.1

**Evidence:** The paper discusses Wei et al. (2022) and the general question of whether verbalization is necessary, but does not cite the growing literature on explicit-to-implicit CoT distillation.

**Recommendation:** Add a brief mention of work on CoT distillation and implicit reasoning training, which supports the paper's thesis that progressive curriculum design (rather than the specific latent mechanism) drives performance.

### - [ ] D008: technical_accuracy

The manuscript's abstract states M5 outperforms on '3 of 4 OOD test sets', but the OOD section (4.4) and Table 5 report 5 comparisons (ProsQA ID + 4 OOD). The abstract should be consistent: M5 outperforms M3 on 3 of 4 OOD sets (correct -- 7-hop, 8-hop, dense), while the in-distribution comparison shows no difference. The body text is consistent but the abstract's '3 of 4' phrasing accurately reflects the 4 OOD-only comparisons. However, the introduction says '3 of 4 test sets' (paragraph 4) which is slightly ambiguous -- a reader could interpret this as 3 of 4 total comparisons. This is a minor wording issue.

**Location:** Abstract, Section 1 (paragraph 4)

**Evidence:** Abstract: 'M5 outperforms COCONUT on 3 of 4 test sets (all statistically significant).' Body: Table 5 shows 5 comparisons. 3 of 4 OOD comparisons favor M5. The phrasing is technically correct but potentially confusing since the total number of comparisons varies by context.

**Recommendation:** Clarify in the abstract and introduction: 'M5 outperforms COCONUT on 3 of 4 out-of-distribution test sets' to avoid ambiguity about whether the in-distribution comparison is included in the count.

### - [ ] D009: technical_accuracy

The nonlinear probe results (Appendix A.7) show 0.0 accuracy for ALL MLP probes across ALL 78 cells for both models. This is not a null result indicating 'MLP probes do not exceed linear probes' -- it indicates that the MLP probes completely failed to learn. An MLP probe that achieves 0.0% accuracy on a classification task (where chance is above 0% for most target distributions) has convergence or implementation issues. The paper acknowledges this in the final sentence of A.7 ('the MLP training procedure... may warrant further tuning to rule out convergence failure'), but this understatement obscures the severity: zero accuracy across 156 probing configurations is almost certainly a bug in the MLP training, not a genuine null finding. The paper should either fix the MLP probing or remove the claim that 'the encoded information is linearly decodable' (Section 4.3), since the nonlinear baseline is broken.

**Location:** Appendix A.7, Section 4.3 (Table 4, 'Cells where MLP > linear: 0 / 78')

**Evidence:** probing/results.json: nonlinear_probe_accuracy is 0.0 for all 78 cells in both M3 and M5. This is not a null result but a failure of the MLP training procedure.

**Recommendation:** Either (1) fix the MLP probe training (likely needs learning rate tuning, longer training, or proper cross-validation) and re-run, or (2) remove the nonlinear probe claims entirely and note that the linear-vs-nonlinear comparison was not successfully conducted. Do not present a broken experiment as a null result.

### - [ ] D010: technical_accuracy

The selectivity values reported in Table 4 are described as computed at each model's 'peak probe accuracy layer'. For M3, positions 0, 1, 3 have peak at layer 0, and position 2 at layer 12. For M5, all positions peak at layer 12. However, the selectivity_recomputed.json data shows selectivity values at ALL layers, not just the peak layer. The reported values in the manuscript (+52.0pp for M3 at position 3) correspond to the selectivity at the peak layer for that position. This is methodologically sound but should be stated more explicitly: the selectivity is measured at the single layer where the matched-step probe achieves its highest accuracy, which biases the selectivity estimate upward (cherry-picking the best layer). A more conservative approach would report the mean selectivity across all layers for each position.

**Location:** Section 4.3, Table 4

**Evidence:** selectivity_recomputed.json shows that M3 position 3 selectivity ranges from 0.22 (layer 8) to 0.52 (layers 0 and 12), and M5 position 3 selectivity ranges from -0.007 (layer 8) to 0.52 (layer 12). Reporting only the peak layer value overstates the consistency of the selectivity signal.

**Recommendation:** Add a note that selectivity is reported at the peak probing layer for each position. Consider also reporting the mean selectivity across layers for each position, which would give a more conservative estimate. For M5, position 3 selectivity is only high at layers 11-12, suggesting that the step-specific encoding is a late-stage phenomenon in M5.

### - [ ] D012: novelty

The paper's core finding -- that COCONUT's continuous thought tokens are largely inert -- has been independently established by Zhang et al. (2025) on different tasks (MMLU, HotpotQA) and at larger scale (LLaMA 7B/8B). The paper acknowledges this and positions itself as extending the finding to ProsQA (COCONUT's strongest task). However, the novelty is somewhat reduced by the concurrent work. The paper's unique contributions are: (a) the curriculum-matched M5 baseline, which provides a constructive alternative rather than just ablation; (b) the OOD generalization analysis showing a task-dependent tradeoff; and (c) the probing analysis showing identical selectivity patterns. Of these, (a) is the strongest novel contribution. The paper should be clearer about which findings are confirmatory (extending Zhang et al.) and which are genuinely new.

**Location:** Section 2.2, Section 5.4

**Evidence:** Zhang et al. (2025) found causal inertness of COCONUT thoughts on MMLU and HotpotQA. This paper finds the same on ProsQA. The novel contribution is the M5 baseline and the OOD tradeoff analysis.

**Recommendation:** Restructure the contributions to more clearly delineate: (1) confirmatory replication of Zhang et al.'s causal inertness finding on ProsQA (the strongest-case domain for COCONUT), (2) novel M5 curriculum-matched baseline methodology (applicable beyond COCONUT), and (3) novel finding that the recycling mechanism introduces a task-dependent generalization tradeoff rather than uniform benefit or deficit.

### - [ ] D013: interpretation

The DAG result (M3 outperforms M5 by 7.3pp) is interpreted as suggesting that 'sequential accumulation of state through recycling may provide a useful inductive bias for tracking path convergence.' This is speculative and the paper does not provide evidence for this specific mechanism. An equally plausible explanation is that M3's 6x computational depth simply provides more representational capacity for handling novel graph structures, and DAGs happen to require more computation than extended chains. Without targeted ablations (e.g., varying the number of forward passes in M3, or providing M5 with more attention positions for DAG problems), the 'path convergence' interpretation is underdetermined.

**Location:** Section 4.4, Section 5.3

**Evidence:** M3 outperforms M5 on DAG by 7.3pp (p = 0.001). The interpretation invokes 'path convergence' but this is post-hoc. DAGs also have other structural differences from trees (multiple parents, convergent paths, potential cycles in the general case) that could explain the advantage through different mechanisms.

**Recommendation:** Present the DAG advantage more cautiously. State that M3 outperforms on DAG structures but that the mechanistic explanation (path convergence benefiting from sequential state accumulation) is speculative. Note alternative explanations (computational depth, structural novelty).

### - [ ] D014: framing

The title 'Does COCONUT Reason or Buffer?' sets up a binary that the paper's own evidence complicates. The probing results show structured, step-specific encoding that is neither pure 'reasoning' (in the sense of sequential BFS) nor pure 'buffering' (in the sense of content-free computation). The evidence is most consistent with 'curriculum-driven structured computation' where both models learn a specific representational strategy for the task, but this strategy does not require the recycling mechanism. The binary framing may attract readers but risks oversimplifying the contribution.

**Location:** Title, throughout

**Evidence:** The paper's evidence shows: (1) thought positions are not content-free buffers (they encode step-specific entities with 52pp selectivity); (2) they are not sequential reasoning chains (permutation insensitive, transplant tolerant); (3) the representational strategy is identical between architectures, arising from the curriculum.

**Recommendation:** Consider a title that better captures the nuance, e.g., 'Dissecting Latent Thought Tokens: Curriculum, Not Mechanism, Drives COCONUT's Performance on ProsQA' or similar. Alternatively, keep the title but address the false dichotomy explicitly in the introduction.

### - [ ] D018: technical_accuracy

The paper states that M3 and M5 share 'the same number of attention positions occupied by thought tokens during both training and inference' (Section 3.2). However, the computational structure is fundamentally different. In M3, each thought position in forward pass k can only attend to (a) all input tokens and (b) the single recycled hidden state at the current position from the previous pass. In M5, each thought position can attend to all input tokens AND all other thought positions simultaneously via standard causal attention. M5's thought positions have access to a richer attention context (they can attend to each other), while M3's thought positions are informationally isolated from each other except through the sequential recycling chain. This is an important architectural difference that the paper should discuss more carefully, as it may explain why M5 can achieve comparable performance with 1/6 the FLOPs: M5's attention structure is strictly more connected than M3's for the thought-token region.

**Location:** Section 3.2

**Evidence:** M3: thought position k in pass k can attend to input tokens and the recycled hidden state at position k (from pass k-1). M5: thought position k can attend to input tokens and all thought positions 0 through k-1 via causal self-attention.

**Recommendation:** Add a discussion of the attention structure difference. M5's thought positions can attend to each other through standard causal attention, while M3's are informationally linked only through the sequential recycling chain. This connectivity difference is likely important for understanding the OOD generalization results.

### - [ ] D020: technical_accuracy

The paper reports M3's test accuracy as 98.0% (Table 2) but the corruption results (Table 3, results.json) show clean accuracy of 97.0%. This 1pp discrepancy likely reflects a difference between evaluation methods (the corruption experiment may use a different evaluation procedure or a different subset). The paper should note and explain this discrepancy.

**Location:** Table 2 vs Table 3

**Evidence:** Table 2: M3 test accuracy = 98.0%. Table 3: M3 clean accuracy = 97.0%. corruption/results.json: clean_accuracy = 0.97. The corruption experiments may use the trained model at a different checkpoint or with a different decoding strategy.

**Recommendation:** Explain the 1pp discrepancy between Table 2 and Table 3 for M3's accuracy. If the corruption experiments use a different evaluation procedure (e.g., greedy decoding vs. the evaluation used for Table 2), state this explicitly.


## 🔵 SUGGESTION

### - [ ] D015: missing_experiments

The paper would benefit from a per-hop-count breakdown of in-distribution accuracy for M3 vs M5. ProsQA contains paths of 3-6 hops, and the performance difference between models may concentrate at specific hop counts. If M3 outperforms M5 primarily on 5-6 hop problems (where the sequential pipeline has more room to contribute), this would provide additional evidence about when the recycling mechanism matters. The OOD results show this pattern extrapolates to 7-8 hops, but the ID breakdown would be informative.

**Location:** Section 4.1

**Evidence:** Table 2 reports aggregate accuracy only. The 2.4pp gap between M3 (98.0%) and M5 (95.6%) on the test set might concentrate at specific hop counts.

**Recommendation:** Add a per-hop-count accuracy breakdown for the in-distribution test set. This would connect the ID results to the OOD findings and clarify whether the recycling mechanism provides a marginal benefit even in-distribution for longer chains.

### - [ ] D016: missing_experiments

The probing analysis tests whether thought positions encode the entity at the corresponding step of the reasoning path. This tests for BFS-like sequential encoding. However, Zhu et al. (2025) proved that continuous thought tokens can encode superposition states representing multiple frontier nodes simultaneously. The paper's probes would not detect such superposition encoding because they classify a single entity label. A probe designed to decode whether a position contains information about multiple entities (e.g., a multi-label classification or a representational similarity analysis comparing thought-position activations to embeddings of all entities in the graph) would provide a more targeted test of the BFS superposition hypothesis. The paper briefly acknowledges this in Section 5.4 but does not pursue it.

**Location:** Section 5.4

**Evidence:** Section 5.4 states: 'A probe designed to decode multiple frontier nodes simultaneously would provide a more targeted test of the BFS hypothesis and could reveal representational differences between M3 and M5 that our current analysis does not capture.'

**Recommendation:** This is correctly identified as a limitation. Consider adding a representational similarity analysis (RSA) that compares thought-position activations to the full set of entity embeddings, which could detect superposition encoding without requiring explicit multi-label classification.

### - [ ] D017: concurrent_work

The paper should check for and discuss any concurrent work on COCONUT analysis that may have appeared since Zhang et al. (2025). The COCONUT paper (Hao et al., 2024) has generated significant interest, and there may be other concurrent analyses of its thought tokens from different research groups. Additionally, concurrent work on latent reasoning in other architectures (e.g., Quiet-STaR from Zelikman et al., 2024) may provide useful context for understanding when latent reasoning mechanisms do vs. do not provide benefits.

**Location:** Section 2.2, Section 5.4

**Evidence:** The paper cites Zhang et al. (2025) and Zhu et al. (2025) as concurrent/follow-up work on COCONUT. Other concurrent analyses may exist.

**Recommendation:** Conduct a thorough literature search for concurrent COCONUT analyses. Consider citing Zelikman et al. (2024) Quiet-STaR as an alternative approach to latent reasoning that provides a useful comparison point.

### - [ ] D019: impact

The paper's practical implications section (5.5) correctly identifies that curriculum design is the higher-leverage investment. However, it could strengthen its impact by providing more concrete guidance for researchers building latent reasoning systems. Specifically: (1) What properties of the curriculum drive the performance? Is it the progressive removal, the number of stages, the epoch budget per stage? (2) Would a 3-stage curriculum work as well as a 7-stage one? (3) Is the specific initialization of the pause embedding important, or would random initialization work? These are actionable research questions that follow directly from the paper's findings and would increase its practical value.

**Location:** Section 5.5

**Evidence:** The paper identifies the curriculum as the key ingredient but does not decompose which properties of the curriculum matter.

**Recommendation:** Add a brief discussion of which curriculum properties are likely to matter based on the results, and flag specific ablations (number of stages, epoch budget, initialization) as high-value follow-up experiments.

### - [ ] D021: framing

The paper's M5 baseline is a strong and well-motivated control. However, the paper could better contextualize what M5 represents theoretically. M5 is not just a 'pause token' model -- it is a model that performs the same curriculum-driven training as COCONUT but routes all inter-step computation through standard self-attention rather than through an explicit recurrence loop. This makes M5 architecturally similar to a standard transformer with extra padding tokens, trained with a specific curriculum. The fact that this achieves 95.6% accuracy on ProsQA is itself a significant result: it demonstrates that standard transformer attention is sufficient for multi-hop reasoning when the training curriculum is appropriate, without any architectural modifications beyond adding learnable tokens. This relates to the broader debate about whether transformers need architectural augmentation for reasoning tasks.

**Location:** Section 3.2, Section 5.5

**Evidence:** M5 = GPT-2 + 1 learned embedding + standard causal attention + COCONUT's 7-stage curriculum = 95.6% on ProsQA. This is itself a strong result that deserves more emphasis.

**Recommendation:** Add a brief discussion framing M5 as evidence that standard transformer attention, combined with appropriate curriculum design, is sufficient for multi-hop reasoning on ProsQA. This connects to the broader debate about whether transformers need architectural modifications for reasoning.
