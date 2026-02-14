# Phase 4: Experiments

> **Portability note:** This document is the original execution log from a Lambda Labs H100 server. All paths referencing `/lambda/nfs/experiment/` and SSH commands are specific to that server. To reproduce, substitute your own GPU server and working directory. See the paper's [README.md](README.md) for portable reproduction instructions.

**Prerequisites:** Phase 3 complete. Gates 3-5 passed (or Gate 4 failed → proceeding with negative-result framing).
**Where:** Lambda H100.
**Time estimate:** ~6-8 hours
**GPU hours:** ~5.5
**Gates:** None (but deterministic verifications per experiment)

---

## Checkpoint Locations (Seed 0)

All checkpoints live in `/lambda/nfs/experiment/results/v9_meta_fork/`:

| Model | Checkpoint Path | Type |
|-------|----------------|------|
| M1 | `prosqa-cot/checkpoint_50` | Plain GPT-2 |
| M2 | `prosqa-nocot/checkpoint_50` | Plain GPT-2 |
| M3 | `prosqa-coconut/checkpoint_50` | Coconut-wrapped |
| M4 | `prosqa-m4-frozen/checkpoint_50` | Coconut-wrapped |
| M4b | `prosqa-m4b-shared/checkpoint_50` | Coconut-wrapped |
| M5 | `prosqa-m5-pause/checkpoint_50` | Pause-curriculum (plain GPT-2 + pause embeddings) |

Experiment scripts auto-detect model type by checking state_dict keys for `base_causallm` prefix.

---

## Execution Order

1. **Exp 0: Causal Sanity Gate** — MUST run first, gates everything (~10 min)
2. **Exp 1: Corruption** — fastest diagnostic after gate (~45 min)
3. **Exp 2: Probing** — builds information surface (~35 min)
4. **Exp 3: OOD Generalization** — cleanest behavioral test (~25 min)
5. **Exp 4: Causal Tracing** — strongest evidence, most expensive (~1.5 hours)
6. **Exp 5: Zhang Replication** — perturbation sensitivity comparison (~30 min)
7. **Ablation 7.1: Token Count** — inference only (~10 min)

---

## Experiment 0: Causal Sanity Gate

**Question:** Does our activation patching implementation actually work? Gate everything on M1 validation.

**MUST RUN FIRST.** If this fails, do NOT proceed to Exp 4. Debug the patching implementation.

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_causal.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data data/prosqa_test.json \
    --num_samples 50 \
    --models m1 \
    --mode sanity \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/causal_sanity/ \
    --seed 0" | cut -c1-200
```

### Method

Same corrupt→restore→measure pipeline as Exp 4 (see below), but on M1's CoT tokens only, 50 samples. We already know M1's CoT tokens encode reasoning (they're literal text). If patching them back doesn't restore performance, our implementation is broken.

### Pass Criteria

- CE > 0.3 at ≥50% of CoT token positions
- logit_clean > logit_corrupt for >80% of samples
- Runs in <10 minutes

### If It Fails

Debug the patching implementation. Common issues: wrong hook names, wrong position indexing, activation shape mismatch. Do NOT run Exp 4 or trust any causal results until this passes.

---

## Experiment 1: Graduated Corruption Ablation

**Question:** Is information in thought tokens sequential (chain) or redundant (buffer)?

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_corruption.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data data/prosqa_test.json \
    --num_samples 500 \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/corruption/ \
    --seed 0" | cut -c1-200
```

### Corruption Modes

- **Forward:** Clean → Corrupt-1 → Corrupt-1:2 → ... → Corrupt-all
- **Reverse:** Corrupt-T → Corrupt-(T-1):T → ... → Corrupt-all
- **Single-position:** Corrupt-only-k for each k
- **Permutation:** Take M3's own thought tokens for the SAME problem, shuffle order. 10 random permutations per sample. If order doesn't matter → buffer. If order matters → sequential dependency. Cleaner test than forward/reverse which conflate "content important?" with "order important?"
- **Partial permutation:** Swap only adjacent pairs. Tests local vs global ordering sensitivity.
- **Cross-problem transplant:** 200 pairs, same hop count, different graphs — inject A's thoughts into B

### Deterministic Verification

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
r = json.load(open('/lambda/nfs/experiment/results/v9_meta_fork/experiments/corruption/results.json'))
# Check 1: random replacement L2 distance from original
assert r['verification']['replacement_l2_distance'] > 5.0, f'L2 too small: {r[\\\"verification\\\"][\\\"replacement_l2_distance\\\"]}'
# Check 2: clean condition accuracy matches test_id accuracy
assert abs(r['m3']['clean_accuracy'] - r['verification']['m3_test_id_accuracy']) < 0.02, 'Clean != test_id accuracy'
# Check 3: corruption applied at correct positions
assert r['verification']['position_check_passed'], 'Corruption positions wrong'
print('ALL CORRUPTION VERIFICATIONS PASSED')
\"" | cut -c1-200
```

### Interpretation

| Pattern | Conclusion |
|---------|------------|
| M3 cascades from early corruption; M4b AND M5 robust | Sequential reasoning chain in M3 |
| M3 AND M5 both cascade; M4b robust | Curriculum creates dependencies regardless of mechanism |
| All cascade similarly | Architectural property, not diagnostic |
| Cross-transplant fails for M3, works for M4b/M5 | M3 encodes problem-specific states (**strong evidence**) |
| Permutation destroys M3 accuracy; M5 unaffected | Order matters in M3 thoughts — sequential chain (**strong**) |
| Permutation no effect on M3 | Order irrelevant — bag not chain (supports buffer) |

**IMPORTANT:** Cascading alone ≠ sequential reasoning. Any autoregressive system cascades. The diagnostic power is in COMPARING cascade patterns across M3, M4b, **M5**. M5 is the primary comparison.

---

## Experiment 2: Representation Probing

**Question:** Do thought tokens encode structured intermediate reasoning states?

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_probing.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data data/prosqa_test.json \
    --num_samples 500 \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/probing/ \
    --seed 0" | cut -c1-200
```

### Method

1. Extract hidden states h(l,t) at every layer l ∈ {0,...,11} and thought position t ∈ {0,...,T-1}
2. Hidden states are float32 (Meta trains in fp32, `bf16: False`). No casting needed.
3. Probe target at thought t: identity of type_t in ground-truth path (multi-class)
4. Linear probe (logistic regression, C=1.0) + nonlinear probe (2-layer MLP, 256 hidden)
5. 5-fold CV on 500 samples (prosqa_test.json)
6. Permutation test: shuffle labels 100× per cell, Bonferroni correction: p < 0.05 / (13 × T)

### Additional Probing Analyses

- **Selectivity metric:** For each (l,t) cell, compute `selectivity(l,t) = probe_acc(l,t,target=step_t) - max_{s≠t} probe_acc(l,t,target=step_s)`. High selectivity = thought t specifically encodes step t's info. Low selectivity = general problem representation at every position (buffer).
- **Cross-position leakage test:** Train probe on h(l,t=0) to predict step 5. If this works well, the model front-loads all info into the first thought = compression/buffering.
- **Input-position control:** Probe hidden states at input token positions (graph fact tokens). If input-position probes decode step-t info as well as thought-t probes, thoughts add no representational content. This addresses Ravichander et al. (2021): probes can decode information present but not used by the model.

### Deterministic Verification

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
r = json.load(open('/lambda/nfs/experiment/results/v9_meta_fork/experiments/probing/results.json'))
# Check 1: probe labels match ground-truth paths
assert r['verification']['label_alignment_check'] == 'PASSED'
# Check 2: hidden states are 768-dim (GPT-2 124M), all finite
assert r['verification']['hidden_state_shape'] == [768]
assert r['verification']['all_finite'] == True
# Check 3: synthetic separable data → 100% probe accuracy
assert r['verification']['synthetic_probe_accuracy'] > 0.99
print('ALL PROBING VERIFICATIONS PASSED')
\"" | cut -c1-200
```

### Interpretation

| Pattern | Conclusion |
|---------|------------|
| M3 diagonal + high selectivity; M4b flat; M5 flat | **Strong reasoning evidence** |
| M3 diagonal + high selectivity; M5 also diagonal | Curriculum creates structure, continuous thoughts not required |
| M3 diagonal but LOW selectivity | Compressed buffer — info present but not step-specific |
| M3 flat everywhere | No structured info. Buffering. |
| Linear fails, nonlinear succeeds for M3 | Info present but nonlinearly encoded. Still reasoning. |
| Probes at input ≈ probes at thoughts | Thoughts don't add new info, model reasons from input directly (buffer) |

---

## Experiment 3: OOD Generalization

**Question:** Does COCONUT generalize to unseen reasoning structures?

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_ood.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data_dir data/ \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/ood/ \
    --seed 0" | cut -c1-200
```

### Test Sets

| Set | Novelty |
|-----|---------|
| prosqa_test | None (baseline) |
| ood_7hop | Hop count 7 (trained 3-6) |
| ood_8hop | Hop count 8 |
| ood_dag | DAGs (trained on trees) |
| ood_dense | Dense (branch 5-8) |

### Deterministic Verification

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
r = json.load(open('/lambda/nfs/experiment/results/v9_meta_fork/experiments/ood/results.json'))
# Check 1: M3/M4/M4b use correct number of thought tokens (= 6, max_latent_stage)
assert r['verification']['thought_token_count_correct'] == True
# Check 2: M1 generates CoT text (not thought tokens)
assert r['verification']['m1_generates_cot'] == True
# Check 3: answer parser verified on 50 outputs
assert r['verification']['parser_accuracy'] > 0.95
print('ALL OOD VERIFICATIONS PASSED')
\"" | cut -c1-200
```

### Interpretation

| Pattern | Conclusion |
|---------|------------|
| M3 generalizes to 7-8 hops; M4b AND M5 don't | **Strongest behavioral evidence** |
| M3 AND M5 both generalize; M4b doesn't | Curriculum + extra compute sufficient, continuous thoughts not required |
| Neither M3 nor M5 generalizes | Both pattern-match training distribution |
| M3 handles DAGs/dense; M5 doesn't | General graph traversal via continuous thoughts (**very strong**) |
| M3 handles DAGs/dense; M5 also handles | Curriculum teaches traversal, mechanism irrelevant |

---

## Experiment 4: Causal Tracing

**Question:** Which (layer, position) pairs are causally responsible for correct answer?

### Reference Implementation: ROME (Meng et al. 2022)

**Use the ROME causal tracing code as the primary guide for `exp_causal.py`.**

- **Repository:** `https://github.com/kmeng01/rome`
- **Key files:** `experiments/causal_trace.py` and `notebooks/causal_trace.ipynb`
- **Also available as:** `pip install causal-tracer` (cleaner API, batch support, works with GPT-2)
- **pyvene alternative:** `https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Causal_Tracing.html`

**ROME's methodology is exactly what we need:** corrupt input → run forward pass → restore clean activations at specific (layer, position) → measure output probability recovery. The adaptation from ROME's factual recall setting to our latent reasoning setting is:

| ROME (factual recall) | Our setting (latent reasoning) |
|---|---|
| Corrupt subject tokens with Gaussian noise | Corrupt thought token positions with matched-distribution random embeddings |
| Restore hidden state at (layer, subject_position) | Restore hidden state at (layer, thought_position) |
| Measure P(correct_fact) recovery | Measure P(correct_answer) recovery |
| Hook: `model.transformer.h[layer]` output | Hook: same (`model.transformer.h[layer]` or `model.base_causallm.model.transformer.h[layer]` for Coconut-wrapped) |

**Key implementation details from ROME to preserve:**
1. **Noise calibration:** ROME computes embedding-space σ from a corpus, then uses 3σ Gaussian. We use matched-distribution random embeddings (mean/std from M3's actual thought token hidden states across 500 samples). Either approach works; ours is more conservative.
2. **Hook mechanism:** ROME uses `register_forward_hook` on transformer blocks to intercept and replace activations. Copy this pattern directly — don't try to reimplement from scratch.
3. **Batch processing:** The `causal-tracer` pip package supports batched patching. Use it if possible to speed up the ~500 samples × 12 layers × T positions grid.
4. **CE formula:** ROME uses `(p_patched - p_corrupt) / (p_clean - p_corrupt)`. We use the same, on logits: `CE(l,p) = (logit_patched - logit_corrupt) / (logit_clean - logit_corrupt + 1e-6)`.

**Do NOT write the activation patching from scratch.** Read ROME's `trace_with_patch()` function first, understand its hook-based approach, then adapt for our Coconut models. This avoids the class of bugs that come from misunderstanding PyTorch hook semantics.

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_causal.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data data/prosqa_test.json \
    --num_samples 500 \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/causal/ \
    --seed 0" | cut -c1-200
```

### Method (Adapted from Meng et al. 2022 — ROME)

1. Clean run: record all activations + logit_clean
2. Corrupt run: replace all thought tokens with matched-distribution random embeddings → logit_corrupt
3. Patch: for each (l,p), run corrupt + swap in clean activation → logit_patched
4. CE(l,p) = (logit_patched - logit_corrupt) / (logit_clean - logit_corrupt + 1e-6)
5. Average over 500 samples
6. Run for M1 (validation), M3, M4b, M5

### MANDATORY Validation (Exp 0 must pass first)

M1 (Baseline-CoT) must show CE > 0.3 at ≥50% of CoT token positions. **If Exp 0 failed, do not run this. Debug the patching implementation first.**

### Deterministic Verification

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
r = json.load(open('/lambda/nfs/experiment/results/v9_meta_fork/experiments/causal/results.json'))
# Check 1: M1 validation
assert r['verification']['m1_cot_ce_above_03_fraction'] >= 0.5, 'M1 validation FAILED — patching is broken'
# Check 2: logit_clean > logit_corrupt for >80% of samples
assert r['verification']['clean_gt_corrupt_fraction'] > 0.8
# Check 3: CE values in [0,1] for >95% of entries
assert r['verification']['ce_in_range_fraction'] > 0.95
# Check 4: epsilon sensitivity
assert r['verification']['epsilon_sensitivity_pct'] < 1.0
print('ALL CAUSAL VERIFICATIONS PASSED')
\"" | cut -c1-200
```

### Interpretation

| Pattern | Conclusion |
|---------|------------|
| High CE at M3 thoughts; low at M4b AND M5 | **Strongest causal evidence for continuous thought reasoning** |
| High CE at M3 AND M5; low at M4b | Curriculum-trained compute causally important, not mechanism |
| CE only at fact positions | Model reasons from facts directly, thoughts not involved |

**Note on Zhang et al. discrepancy:** If our results differ from Zhang (we find high CE, they found low PSR), the likely explanation is: PSR measures sensitivity to *small* perturbations (staying on manifold); CE measures replacement with random (off manifold). Low PSR + high CE = "thoughts live on narrow manifold where small perturbations don't leave it, but complete replacement does." Exp 5 tests this directly.

---

## Experiment 5: Zhang et al. Perturbation Sensitivity Replication

**Question:** Are COCONUT's thought tokens sensitive to perturbation? Replicates Zhang et al. (2025) methodology on our setup.

**No reference code exists for this paper.** Zhang et al. did not release code. Implement from their methodology description (Section 3.1 of arXiv:2512.21711).

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_zhang_replication.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data data/prosqa_test.json \
    --num_samples 500 \
    --sigma_multipliers 0.1,0.5,1.0,2.0 \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/zhang_replication/ \
    --seed 0" | cut -c1-200
```

### Method

1. For each sample, extract hidden states at thought positions (M3) or CoT positions (M1)
2. Compute per-position standard deviation: `position_std = std(h(l,t))` across samples
3. Apply Gaussian perturbation: `h_perturbed = h_clean + N(0, σ × position_std)` for σ ∈ {0.1, 0.5, 1.0, 2.0}
4. Measure Perturbation Success Rate (PSR): fraction of samples where the perturbed output differs from the clean output
5. Run for M1 (CoT tokens), M3 (thought tokens), M5 (pause positions)

### Interpretation

| Pattern | Conclusion |
|---------|------------|
| M1 PSR >> M3 PSR (replicates Zhang) | Thoughts insensitive to small perturbations — consistent with buffer OR narrow-manifold reasoning |
| M1 PSR ≈ M3 PSR | Thoughts ARE sensitive — contradicts Zhang (likely due to scale difference: GPT-2 124M vs LLaMA 7B) |
| M3 PSR > M5 PSR | Continuous thoughts more sensitive than pause tokens — supports mechanism matters |
| M3 PSR ≈ M5 PSR | Both equally (in)sensitive — mechanism doesn't matter |

**Critical note:** Low PSR alone cannot distinguish buffer from reasoning. Its value is in (a) replicating/contradicting Zhang et al.'s finding on our setup, and (b) the M3-vs-M5 comparison. Combine with Exp 4 for full picture: low PSR + high CE = narrow manifold hypothesis.

---

## Ablation 7.1: Thought Token Count

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python exp_ood.py \
    --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
    --data_dir data/ \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/experiments/token_count/ \
    --seed 0 \
    --token_counts 2,3,4,5,6,8 \
    --models_only m3,m5" | cut -c1-200
```

**Run for BOTH M3 and M5.** This is critical — if both show the same scaling curve, the relationship is about compute not reasoning steps.

**Report as per-hop breakdown:** accuracy matrix (rows=hop count 3-6, cols=token count 2-12). Aggregate accuracy is too coarse. Buffer improves monotonically with tokens. Reasoning needs ≥k tokens for k-hop problems.

**Interpretation:**
- M3 plateaus at k tokens for k-hop; M5 monotonic → reasoning (uses exactly as many steps as needed)
- Both M3 and M5 show same scaling curve → compute, not reasoning
- M3 degrades sharply with fewer tokens; M5 gradual → sequential chain in M3

---

## Results Summary

After all experiments, generate summary:

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json, glob, os
results = {}
exp_dir = '/lambda/nfs/experiment/results/v9_meta_fork/experiments'
for subdir in os.listdir(exp_dir):
    rpath = os.path.join(exp_dir, subdir, 'results.json')
    if os.path.exists(rpath):
        results[subdir] = json.load(open(rpath))
json.dump(results, open(os.path.join(exp_dir, 'results_summary.json'), 'w'), indent=2)
print('Summary saved.')
for name, r in results.items():
    print(f'  {name}: {list(r.keys())[:5]}...')
\"" | cut -c1-200
```

---

## Red Flags

- All probe accuracies exactly 1/100 → hidden state extraction broken
- All probe accuracies exactly 1.0 → data leakage
- M2 > M1 on any test → training was broken (stop everything)
- Any model 100% on any OOD set → data leak
- All models ~50% everywhere → task broken or all guessing
- M1 causal validation fails (Exp 0) → patching implementation broken, fix before trusting anything
- M5 > M3 on in-distribution → something wrong with M3 training or M5 has a bug
- M3 > M5 on only 1 out of 6 experiments → cherry-picking territory, do not overclaim
- Probe selectivity high but causal effect zero → information present but not used
- OOD works but corruption shows no order-dependence → generalizes but not via sequential reasoning
- All results point one direction but only on GPT-2/ProsQA → scope limitation, not general COCONUT claim
- M5 = M3 on ALL experiments → paper's mechanism claim collapses (valid negative result, still publishable)
