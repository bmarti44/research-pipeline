# Phase 5: Multi-Seed Runs + Review + Figures

**Prerequisites:** Phase 4 complete. Seed 0 results in hand.
**Where:** Lambda H100.
**Time estimate:** ~3-4 days
**GPU hours:** ~65
**Gates:** None (publication quality checkpoint)

---

## Step 5.0: Create Per-Seed Config Files

Meta's `run.py` takes a single YAML config with no CLI override support. For seeds 1 and 2 (seed 0 is the default from Phase 2-4), create per-seed copies of every config.

```bash
ssh ubuntu@192.222.52.148 "cd /lambda/nfs/experiment/code/v9_meta_fork/args && \
  for SEED in 1 2; do
    for cfg in prosqa_cot prosqa_nocot prosqa_coconut_1gpu prosqa_m4_frozen prosqa_m4b_shared; do
      # Copy config, change seed and name (to avoid overwriting seed 0 checkpoints)
      sed -e \"s/^seed: 0/seed: \${SEED}/\" \
          -e \"s/^name: \(.*\)/name: \1-seed\${SEED}/\" \
          \${cfg}.yaml > \${cfg}_seed\${SEED}.yaml
    done
  done && ls args/*seed*" | cut -c1-200
```

This creates configs like `prosqa_cot_seed1.yaml` with `seed: 1` and `name: prosqa-cot-seed1` (so checkpoints go to a separate directory).

**Verify:** Each config has the correct seed and a unique name field.

```bash
ssh ubuntu@192.222.52.148 "cd /lambda/nfs/experiment/code/v9_meta_fork && \
  for f in args/*seed*.yaml; do
    echo \"\$f: seed=\$(grep '^seed:' \$f | awk '{print \$2}'), name=\$(grep '^name:' \$f | awk '{print \$2}')\"
  done" | cut -c1-200
```

---

## Step 5.1: Seeds 1 and 2 — Full Pipeline

All models start from `GPT2LMHeadModel.from_pretrained("openai-community/gpt2")`. No warm-start. 50 epochs each.

```bash
for SEED in 1 2; do
  # Create results directories
  ssh ubuntu@192.222.52.148 "mkdir -p /lambda/nfs/experiment/results/v9_meta_fork/logs"

  # M1 (CoT baseline)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_cot_seed${SEED}.yaml \
      > /lambda/nfs/experiment/results/v9_meta_fork/logs/m1_cot_seed${SEED}.log 2>&1"

  # M2 (Direct baseline)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_nocot_seed${SEED}.yaml \
      > /lambda/nfs/experiment/results/v9_meta_fork/logs/m2_nocot_seed${SEED}.log 2>&1"

  # M3 (COCONUT)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_coconut_1gpu_seed${SEED}.yaml \
      > /lambda/nfs/experiment/results/v9_meta_fork/logs/m3_coconut_seed${SEED}.log 2>&1"

  # M4 (Pause-frozen)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_m4_frozen_seed${SEED}.yaml \
      > /lambda/nfs/experiment/results/v9_meta_fork/logs/m4_frozen_seed${SEED}.log 2>&1"

  # M4b (Pause-learned-shared)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_m4b_shared_seed${SEED}.yaml \
      > /lambda/nfs/experiment/results/v9_meta_fork/logs/m4b_shared_seed${SEED}.log 2>&1"
done
```

**Run `verify_training.py` on every model for every seed.** All must pass, including the answer-token CE check for M4/M4b at stages 3+ (see MASTER_REFERENCE Section 7).

**Note:** All 5 models use 50 epochs. All start from pretrained GPT-2. No warm-start.

---

## Step 5.2: Run Experiments on All Seeds

Experiment scripts are standalone Python scripts with argparse. They auto-detect model type from checkpoint state_dict keys.

```bash
for SEED_DIR in prosqa-cot prosqa-cot-seed1 prosqa-cot-seed2; do
  SEED_SUFFIX="${SEED_DIR#prosqa-cot}"  # "" or "-seed1" or "-seed2"
  RESULTS_BASE="/lambda/nfs/experiment/results/v9_meta_fork"

  # OOD Generalization (most important for multi-seed)
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    python exp_ood.py \
      --checkpoint_dir ${RESULTS_BASE} \
      --name_suffix '${SEED_SUFFIX}' \
      --data_dir data/ \
      --output_dir ${RESULTS_BASE}/experiments${SEED_SUFFIX}/ood/" | cut -c1-200

  # Corruption Ablation
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    python exp_corruption.py \
      --checkpoint_dir ${RESULTS_BASE} \
      --name_suffix '${SEED_SUFFIX}' \
      --data data/prosqa_test.json --num_samples 2000 \
      --output_dir ${RESULTS_BASE}/experiments${SEED_SUFFIX}/corruption/" | cut -c1-200

  # Representation Probing
  ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
    cd /lambda/nfs/experiment/code/v9_meta_fork && \
    python exp_probing.py \
      --checkpoint_dir ${RESULTS_BASE} \
      --name_suffix '${SEED_SUFFIX}' \
      --data data/prosqa_test.json --num_samples 2000 \
      --output_dir ${RESULTS_BASE}/experiments${SEED_SUFFIX}/probing/" | cut -c1-200
done
```

**Note:** The `--name_suffix` flag tells the experiment script which checkpoint subdirectory to use (e.g., `prosqa-coconut-seed1` vs `prosqa-coconut`). If this approach is awkward, we can also pass explicit checkpoint paths — the experiment scripts accept both modes.

---

## Step 5.3: Statistical Analysis

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python statistical_analysis.py \
    --results_dirs \
      /lambda/nfs/experiment/results/v9_meta_fork/experiments/ \
      /lambda/nfs/experiment/results/v9_meta_fork/experiments-seed1/ \
      /lambda/nfs/experiment/results/v9_meta_fork/experiments-seed2/ \
    --output /lambda/nfs/experiment/results/v9_meta_fork/statistical_analysis.json" | cut -c1-200
```

The analysis script must compute:
- **Mean ± std** across 3 seeds for every metric
- **McNemar's test** for M3 vs M4b on each OOD set (primary comparison)
- **Cohen's d** effect sizes for all comparisons
- **Bonferroni correction** across all tests
- **Paired tests** where seeds match (M3 seed-0 vs M4b seed-0, etc.)
- **95% confidence intervals** for all condition means and all differences

---

## Step 5.4: Review Rounds (All on Lambda — Agents Read Results Files)

### Methodologist (5 rounds)

```
Read all results in /lambda/nfs/experiment/results/v9_meta_fork/.
Focus:
Round 1: Do results from different experiments converge on same conclusion?
Round 2: Any contradictions between experiments?
Round 3: Which Decision Matrix row do results support?
Round 4: Alternative explanations for positive results?
Round 5: What would a hostile ICML reviewer say?
BLOCKING: Contradictory results between experiments not explained.
```

### Statistician (5 rounds)

```
Read statistical_analysis.json.
Focus:
Round 1: All p-values computed correctly? Multiple comparisons corrected?
Round 2: Bootstrap vs parametric CIs appropriate?
Round 3: 3-seed sufficient for observed effect sizes?
Round 4: Cohen's d reported alongside significance?
Round 5: Any claim that wouldn't survive methods appendix review?
BLOCKING: Any statistical test applied incorrectly.
```

### Skeptic (5 rounds)

```
Attack every positive result.
Round 1: Could probing be memorization not reasoning?
Round 2: Could OOD generalization be distributional overlap?
Round 3: Could corruption sensitivity be autoregressive cascade artifact?
Round 4: Could causal tracing results be M4b weight adaptation, not thought content?
Round 5: Write most devastating 1-paragraph review. Which experiment addresses each point?
BLOCKING: Any attack that no experiment addresses.
```

### Domain Expert (5 rounds)

```
Round 1: Has anyone published similar results since plan was written?
Round 2: How do results compare to CODI, SoftCoT++, other COCONUT variants?
Round 3: Is framing fair to Hao et al. and Zhang et al.?
Round 4: Best venue given results?
Round 5: One-sentence contribution for abstract?
BLOCKING: Published paper with substantially overlapping results.
```

---

## Step 5.5: Fix BLOCKING Issues

Address all BLOCKING issues from reviews. Re-run affected experiments or analyses. Submit fixes back to reviewer for confirmation.

---

## Step 5.6: Generate Figures

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python generate_figures.py \
    --results_dir /lambda/nfs/experiment/results/v9_meta_fork/ \
    --output_dir /lambda/nfs/experiment/results/v9_meta_fork/figures/" | cut -c1-200
```

**7 Figures (per MASTER_REFERENCE Section 12):**
1. Information surface heatmap (4 panels)
2. OOD grouped bar chart (error bars from 3 seeds)
3. Corruption cascade curves
4. Causal effect heatmap (3 panels)
5. Curriculum divergence plot
6. Thought token count curve
7. Cross-problem transplant matrix (appendix)

### SCP Figures to Local

```bash
scp -r ubuntu@192.222.52.148:/lambda/nfs/experiment/results/v9_meta_fork/figures/ ./v9_figures/
```

---

## Step 5.7: Final Results Summary

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
stats = json.load(open('/lambda/nfs/experiment/results/v9_meta_fork/statistical_analysis.json'))
print('=== FINAL RESULTS SUMMARY ===')
print(f'Decision Matrix Row: {stats[\\\"decision_matrix_row\\\"]}')
print(f'Primary comparison (M3 vs M4b):')
for test_set in ['prosqa_test', 'ood_7hop', 'ood_8hop', 'ood_dag', 'ood_dense']:
    m3 = stats['ood'][test_set]['m3']
    m4b = stats['ood'][test_set]['m4b']
    p = stats['mcnemar'][test_set]['m3_vs_m4b']['p_value']
    d = stats['effect_sizes'][test_set]['m3_vs_m4b']
    print(f'  {test_set}: M3={m3[\\\"mean\\\"]:.3f}±{m3[\\\"std\\\"]:.3f} M4b={m4b[\\\"mean\\\"]:.3f}±{m4b[\\\"std\\\"]:.3f} p={p:.4f} d={d:.2f}')
\"" | cut -c1-200
```

---

## Overseer's Final Checklist

```
□ Every experiment has a deterministic verification that passed
□ No BLOCKING issues remain open from any reviewer
□ Decision Matrix row assignment justified by multiple experiments
□ Skeptic's "devastating review" paragraph is addressed
□ Figures are readable and clearly labeled
□ Statistical analysis complete with CIs, effect sizes, corrections
□ All 3 seeds show qualitatively similar results (no seed flips conclusion)
□ Results summary JSON complete and internally consistent
```

---

## Compute Budget Summary

| Phase | GPU Hours | Cost @ $2.50/hr |
|-------|-----------|-----------------|
| Phase 1 (setup/data) | 0 | $0 |
| Phase 2 (baselines, seed 0) | ~6 | $15 |
| Phase 3 (curriculum, seed 0) | ~9 | $23 |
| Phase 4 (experiments, seed 0) | ~3.5 | $9 |
| Phase 5 (seeds 1-2 + experiments) | ~50 | $125 |
| **Total** | **~69** | **~$172** |

Baselines are 50 epochs each. Curriculum models are 50 epochs each.
Earliest termination: Gate 1 at ~$3.
