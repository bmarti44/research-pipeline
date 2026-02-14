# COCONUT Reasoning Study — Overall Reference

> **Portability note:** This document references the original Lambda Labs H100 server used for the study. All paths under `/lambda/nfs/experiment/` and SSH commands to `192.222.52.148` are specific to that server. To reproduce, substitute your own GPU server and working directory. See the paper's [README.md](README.md) for portable reproduction instructions.

## CRITICAL INSTRUCTIONS

Always solve the root cause of a problem. Do not solve a symptom of a problem.
Do not gloss over issues and ignore them. Always solve an issue that pops up.

---

## Research Synopsis

Meta's COCONUT (Chain of Continuous Thought) replaces explicit chain-of-thought tokens with continuous hidden states recycled as input embeddings, claiming the model learns to reason in latent space — including emergent breadth-first search over graph structures. The core question this study asks: **does COCONUT actually encode sequential reasoning in its latent tokens, or do the latent tokens just provide extra forward passes (compute buffer) that help regardless of what's in them?**

This matters because the answer determines whether latent reasoning is a genuine new capability or an expensive way to add think-time. Zhang et al. (2025) found that perturbing COCONUT's latent tokens on LLaMA 7B/8B had virtually no effect on outputs — suggesting they're causally inert placeholders, not reasoning states. But their experiments used natural language tasks (MMLU, HotpotQA) where COCONUT already underperforms, and Zhu et al. (ICML 2025) proved theoretically that continuous thoughts *can* encode superposition states enabling BFS in graph traversal.

We test this on ProsQA (the task where COCONUT is strongest) with GPT-2 124M using five converging experiments: graduated corruption, representation probing, out-of-distribution generalization, causal tracing, and perturbation sensitivity. The linchpin is **M5 (Pause-Curriculum)** — a model trained with the identical curriculum and forward pass count as COCONUT but without the continuous thought mechanism. If M5 matches M3 (COCONUT) across experiments, the gains come from the curriculum and extra compute, not from latent reasoning. If M3 consistently outperforms M5 on diagnostics that specifically test sequential state (permutation sensitivity, step-specific probing, OOD hop generalization, causal effect at thought positions), that's convergent evidence for genuine reasoning.

Either outcome is publishable. Positive: "COCONUT encodes sequential reasoning on synthetic graph tasks." Negative: "COCONUT's gains on ProsQA are explained by curriculum training and extra compute, not by the continuous thought mechanism." The latter would be arguably more useful to the field.

---

## For All Phases

This document contains information needed across all phases. Each phase has its own document with phase-specific instructions. **Read this document first, then the relevant phase document.**

---

## 1. Lambda H100 Access

### Connection

```bash
# CRITICAL: Always pipe SSH output through cut -c1-200
# tqdm lines can be 10,000+ characters and overflow the CLI
ssh ubuntu@192.222.52.148 "COMMAND" | cut -c1-200
```

### Key Directories

| Path | Purpose |
|------|---------|
| `/lambda/nfs/experiment/.venv/` | Python virtual environment (transformers==4.46.2) |
| `/lambda/nfs/experiment/reference_repos/coconut/` | Meta's official COCONUT repo (untouched reference) |
| `/lambda/nfs/experiment/code/v9_meta_fork/` | **Working directory — fork of Meta's repo with minimal M4/M4b additions** |
| `/lambda/nfs/experiment/results/v9_meta_fork/` | **Results directory for this study** |
| `/lambda/nfs/experiment/code/v8_pure_prosqa/` | Previous custom rewrite — **abandoned, do NOT use** |

### Standard Command Prefix

Every command on Lambda should use this pattern:

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  COMMAND" | cut -c1-200
```

### GPU Health Check

```bash
ssh ubuntu@192.222.52.148 "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv" | cut -c1-200
```

### Previous Experiment Failures (Context)

**V7 failure:** Custom COCONUT implementation failed to replicate multi-pass forward. Degenerate model.

**V8 custom rewrite failure:** From-scratch train.py had 4+ bugs vs Meta's code: wrong epoch count, wrong optimizer reset schedule, wrong LR schedule, wrong precision, wrong eval-time latent count. Each bug was discovered one at a time, requiring painful debugging.

**V8 key lesson → V9 strategy:** Stop rewriting Meta's code. **Fork Meta's repo directly and make minimal additions.** This eliminates the class of "our code doesn't match Meta's code" bugs. The only changes to Meta's code are: (1) `feedback_mode` parameter in coconut.py for M4/M4b, (2) `feedback_mode` passthrough in run.py, (3) config files for single-GPU training.

---

## 2. Execution Principle

**Everything runs on the Lambda H100.** Code writing, data generation, training, evaluation, probing, figure generation — all on Lambda. The only local operations are reviewing agent outputs and editing plan documents.

---

## 3. Project Structure (on Lambda)

**Strategy: Fork Meta's code, add minimally.**

```
/lambda/nfs/experiment/code/v9_meta_fork/    # Fork of Meta's COCONUT repo
├── run.py                      # Meta's training script (1 line changed: feedback_mode passthrough)
├── coconut.py                  # Meta's COCONUT model (+M4/M4b/M5 feedback_mode parameter)
├── dataset.py                  # Meta's data loading (UNTOUCHED)
├── utils.py                    # Meta's utilities (UNTOUCHED)
├── args/
│   ├── prosqa_coconut.yaml     # Meta's original config (reference)
│   ├── prosqa_cot.yaml         # M1: CoT baseline (single GPU)
│   ├── prosqa_nocot.yaml       # M2: No-CoT baseline (single GPU)
│   ├── prosqa_coconut_1gpu.yaml # M3: COCONUT (single GPU)
│   ├── prosqa_m4_frozen.yaml   # M4: Frozen pause (single GPU)
│   ├── prosqa_m4b_shared.yaml  # M4b: Learned shared pause (single GPU)
│   └── prosqa_m5_pause.yaml    # M5: Pause-curriculum (single GPU)
├── data/                       # Meta's ProsQA data + OOD test sets
│   ├── prosqa_train.json       # 17886 samples (Meta's)
│   ├── prosqa_valid.json       # 300 samples (Meta's)
│   ├── prosqa_test.json        # 500 samples (Meta's)
│   ├── ood_7hop.json           # 1000 samples (generated, ProsQA vocab)
│   ├── ood_8hop.json           # 1000 samples (generated, ProsQA vocab)
│   ├── ood_dag.json            # 1000 samples (generated, ProsQA vocab)
│   └── ood_dense.json          # 1000 samples (generated, ProsQA vocab)
├── generate_ood_data.py        # OOD data generator (uses ProsQA vocabulary)
├── exp_utils.py                # Shared experiment utilities (model loading, inference)
├── exp_corruption.py           # Experiment 1: Graduated corruption ablation
├── exp_probing.py              # Experiment 2: Representation probing
├── exp_ood.py                  # Experiment 3: OOD generalization
├── exp_causal.py               # Experiment 0 (sanity) + Experiment 4: Causal tracing (use ROME as guide — github.com/kmeng01/rome)
├── exp_zhang_replication.py    # Experiment 5: Zhang et al. perturbation sensitivity replication (no reference code exists — implement from paper)
└── preprocessing/              # Meta's preprocessing scripts (reference)
```

Results:
```
/lambda/nfs/experiment/results/v9_meta_fork/
├── prosqa-cot/                 # M1 checkpoints (checkpoint_1, checkpoint_2, ...)
├── prosqa-nocot/               # M2 checkpoints
├── prosqa-coconut/             # M3 checkpoints
├── prosqa-m4-frozen/           # M4 checkpoints
├── prosqa-m4b-shared/          # M4b checkpoints
├── prosqa-m5-pause/            # M5 checkpoints
├── logs/                       # Training logs (m1_cot.log, m2_nocot.log, ..., m5_pause.log)
└── experiments/                # Experiment results
    ├── causal_sanity/          # Exp 0: M1 causal validation gate
    ├── corruption/             # Exp 1: Graduated corruption ablation
    ├── probing/                # Exp 2: Representation probing
    ├── ood/                    # Exp 3: OOD generalization
    ├── causal/                 # Exp 4: Full causal tracing
    ├── zhang_replication/      # Exp 5: Zhang et al. perturbation sensitivity
    ├── token_count/            # Ablation 7.1: Thought token count
    └── results_summary.json    # Aggregated results from all experiments
```

---

## 4. Model Specifications

### Base Architecture

```python
# Pretrained GPT-2 124M from HuggingFace
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
model.resize_token_embeddings(50260)  # +3 special tokens
# ~124M parameters, hidden_size=768, 12 layers, 12 heads
```

**Why pretrained GPT-2 (not random init):** Pretrained GPT-2 already knows in-context retrieval from pretraining on web text. A randomly initialized model cannot learn retrieval from ~18K ProsQA samples alone. Meta uses pretrained GPT-2 in their COCONUT paper. Our V8 random-init 350M model achieved only 45-50% (chance) despite 100% teacher-forced accuracy — the model memorized token patterns but couldn't do autoregressive retrieval.

### Special Token IDs

| Token | ID | Trainable | Init | Used by |
|-------|----|-----------|------|---------|
| `<bot>` | 50257 | Yes | From `<<` embedding | M3, M4, M4b, M5 |
| `<eot>` | 50258 | Yes | From `<<` embedding | M3, M4, M4b, M5 |
| `<pause>` | 50259 | M4: No (frozen), M4b/M5: Yes | From `<<` embedding | M4, M4b, M5 |

Special token embeddings are initialized by copying the `<<` token embedding BEFORE resizing. This gives them a meaningful starting point in the embedding space.

### Model Registry

| ID | Name | Config | Init | Curriculum | Purpose |
|----|------|--------|------|------------|---------|
| M1 | Baseline-CoT | prosqa_cot.yaml | `from_pretrained("gpt2")` | None (stage=0 for all 50 epochs) | Upper bound: explicit reasoning |
| M2 | Baseline-Direct | prosqa_nocot.yaml | `from_pretrained("gpt2")` | None (stage=0 for all 50 epochs) | Lower bound: no reasoning scaffold |
| M3 | COCONUT | prosqa_coconut_1gpu.yaml | `from_pretrained("gpt2")` | CoT → continuous thoughts (50 epochs, 7 stages) | Model under investigation |
| M4 | Pause-Frozen | prosqa_m4_frozen.yaml | `from_pretrained("gpt2")` | CoT → fixed pause (feedback_mode=frozen, 50 epochs) | Control: extra compute, zero learned content |
| M4b | Pause-Learned-Shared | prosqa_m4b_shared.yaml | `from_pretrained("gpt2")` | CoT → shared pause (feedback_mode=learned_shared, 50 epochs) | **Critical control**: learned compute vs problem-specific reasoning |
| M5 | **Pause-Curriculum** | prosqa_m5_pause.yaml | `from_pretrained("gpt2")` | CoT → fixed `<pause>` embedding (feedback_mode=pause_curriculum, 50 epochs) | **Critical compute-matched control**: same curriculum, same forward passes, NO continuous thought mechanism |

> **M5 replaces the old "Random-Embed" concept.** The old M5 (random embeddings at inference from M3 checkpoint) was redundant with M4. The new M5 is a *trained model* that goes through the identical multi-stage curriculum as M3, but inserts a single fixed learned `<pause>` embedding at every thought position instead of recycling hidden states. This is the "pause as thought" baseline from Hao et al.'s own ablations, but trained with our exact setup for a fair comparison. **M5 is the single most important control in the study.** Without it, every positive M3 result has the alternative explanation "extra forward passes help regardless of mechanism."

**CRITICAL: No warm-start.** ALL models (including M3/M4/M4b/M5) start from `GPT2LMHeadModel.from_pretrained("openai-community/gpt2")`. For curriculum models (M3/M4/M4b/M5), stage 0 (epochs 0-4) IS the built-in CoT warm-up — they train on full CoT just like M1 during these epochs. Meta's config has `load_model_path: None`.

### Why M4 AND M4b

M4 (frozen) isolates "extra forward passes help." But M4's model adapts to route around frozen noise — confounding. M4b learns a single shared `<pause>` embedding (same vector at every thought position, every problem), so the model can co-adapt. The M3 vs M4b comparison isolates whether the model needs **problem-specific sequential state**, not just **learned extra compute.**

M5 (pause-curriculum) is the decisive control. Unlike M4/M4b which use COCONUT's hidden-state-recycling architecture, M5 uses a plain GPT-2 with fixed pause embeddings at thought positions. It gets the SAME curriculum, SAME number of forward passes, but thought positions receive a learned `<pause>` embedding instead of the previous step's hidden state. M5 isolates whether **the continuous thought mechanism itself** contributes anything beyond the curriculum + extra compute.

**Comparison hierarchy:**
1. **M3 vs M5** — Does the continuous thought mechanism matter? (Primary question)
2. **M3 vs M4b** — Does problem-specific state matter vs shared learned compute?
3. **M5 vs M4b** — Does curriculum with pause tokens beat curriculum with shared pause in COCONUT wrapper?
4. **M3 vs M4** — Does ANY content in thoughts matter vs frozen noise?
5. **M1 vs M3** — Does latent reasoning match explicit CoT?

**M3 vs M5 is the primary comparison in every experiment.** M3 vs M4b is secondary.

---

## 5. Training Hyperparameters

**Directly from Meta's `prosqa_coconut.yaml` — we run Meta's code, not a reimplementation.**

### All Models (shared settings from Meta's config)

| Param | Value | Source |
|-------|-------|--------|
| Optimizer | AdamW | Meta run.py |
| Learning rate | 1e-4 (constant, **no scheduler**) | Meta config |
| Weight decay | 0.01 | Meta config |
| Batch size | 32 per GPU | Meta config |
| Gradient accumulation | 4 (single GPU) → effective 128 | Adapted from Meta's 4-GPU setup |
| Precision | **fp32** (no bf16) | Meta config: `bf16: False` |
| Optimizer reset | **Every epoch** (see note below) | Meta run.py: `reset_optimizer: True` |
| Total epochs | **50** for ALL models | Meta config: `num_epochs: 50` |
| Parallelism | FSDP (even on 1 GPU, it's Meta's code) | Meta run.py |
| Checkpoint | Every epoch (`checkpoint_{epoch}`) | Meta run.py |

**Key difference from Meta's setup:** Meta uses 4 GPUs with `gradient_accumulation_steps: 1`, giving effective batch 128. We use 1 GPU with `gradient_accumulation_steps: 4` for the same effective batch. FSDP overhead on 1 GPU is wasteful (~69 GB VRAM for GPT-2 124M) but we don't modify the distributed strategy to avoid introducing bugs.

**Optimizer reset behavior (verified in run.py lines 313-320):** AdamW is recreated (`del optimizer; optimizer = optim.AdamW(...)`) every epoch — NOT at stage boundaries. This is inside the epoch loop with no stage check. Consequence: first and second moment estimates never accumulate past a single epoch. For curriculum models (M3/M4/M4b) this prevents stale momentum across stage transitions. For baselines (M1/M2) at constant stage 0 for 50 epochs, it's actively harmful to convergence — each epoch starts with cold momentum, explaining why (a) M1 convergence is slow, and (b) 50 epochs are needed for baselines. This is Meta's deliberate design choice; we do not modify it.

### Stage Schedule (curriculum models only: M3, M4, M4b)

For M1/M2: `stage = 0` for all 50 epochs (Meta run.py: `0 if (configs.cot or configs.no_cot)`).
For M3/M4/M4b/M5: `stage = epoch // epochs_per_stage`, capped at `max_latent_stage=6`.

### Curriculum Schedule

| Stage | Epochs | Thought tokens | Explicit CoT remaining |
|-------|--------|---------------|------------------------|
| 0 | 0-4 | 0 | All |
| 1 | 5-9 | 1 (last step) | All except last |
| 2 | 10-14 | 2 | All except last 2 |
| 3 | 15-19 | 3 | All except last 3 |
| 4 | 20-24 | 4 | All except last 4 |
| 5 | 25-29 | 5 | All except last 5 |
| 6 | 30-49 | All | None |

Stage is computed as `epoch // EPOCHS_PER_STAGE`, capped at `MAX_STAGE=6`. Stage 6 runs from epoch 30 through 49 (20 epochs of fully latent training).

### M5 (Pause-Curriculum) — Trained Model

M5 uses Meta's `run.py` with a new `feedback_mode=pause_curriculum`. Implementation: at each thought position, instead of feeding back the hidden state (M3 behavior), insert a single learned `<pause>` embedding (shared across all positions and all samples). The embedding is an `nn.Parameter` initialized from the `<<` token embedding, same as M4b's pause. The difference from M4b: M5 is a **plain GPT-2** (not wrapped in `Coconut` class), so there is no hidden-state recycling machinery at all. The model simply sees extra input positions with pause embeddings and attends over them normally.

**Training:** Same 50 epochs, same 7-stage curriculum schedule, same optimizer settings. The only difference from M3 is what goes into thought positions.

**Config:** `prosqa_m5_pause.yaml` — identical to `prosqa_coconut_1gpu.yaml` except `feedback_mode: pause_curriculum`.

---

## 6. Data Specification

### Primary Data: Meta's ProsQA (Used for All Training + ID Evaluation)

We use Meta's official ProsQA dataset directly, without conversion or reformatting.

| Dataset | Samples | Hops | Source |
|---------|---------|------|--------|
| `prosqa_train.json` | 17,886 | 3-6 | Meta's COCONUT repo |
| `prosqa_valid.json` | 300 | 3-6 | Meta's COCONUT repo |
| `prosqa_test.json` | 500 | 3-6 | Meta's COCONUT repo |

**Location on Lambda:** `/lambda/nfs/experiment/reference_repos/coconut/data/`

**Key properties of Meta's ProsQA:**
- **Two-choice format** — question asks "Is X a Y or Z?" where Y is the reachable target and Z is unreachable. The answer is always a statement naming the reachable one ("X is a Y."). The model must identify WHICH option is correct, not just say "yes." **Verify that target/neg_target order is randomized in the question** — if the correct answer is always the first option, the model can cheat with position bias.
- **Hop distribution:** 3-6 hops (harder than our V8 custom data which used 2-5)
- **Expected M1 CoT accuracy:** ~77.5% (per Meta's paper)
- **Graph metadata preserved:** `idx_to_symbol`, `edges`, `root`, `target`, `neg_target`

### Sample Format (Meta's ProsQA)

```json
{
  "question": "Alex is a jompus. Every jompus is a zhorpus. Every zhorpus is a tompus. Every tompus is a daxil. Is Alex a daxil or a brimpus?",
  "steps": [
    "Alex is a jompus.",
    "Every jompus is a zhorpus, so Alex is a zhorpus.",
    "Every zhorpus is a tompus, so Alex is a tompus.",
    "Every tompus is a daxil, so Alex is a daxil."
  ],
  "answer": "Alex is a daxil.",
  "idx_to_symbol": {"0": "jompus", "1": "zhorpus", "2": "tompus", "3": "daxil", "4": "brimpus"},
  "edges": [[0, 1], [1, 2], [2, 3]],
  "root": 0,
  "target": 3,
  "neg_target": 4
}
```

**Key format differences from V8 custom data:**
- `steps` array (not `cot` string)
- `answer` is a statement (not "Yes"/"No")
- `question` embeds facts inline (not separate `facts` array)
- No `hops` field — infer from `len(steps)`

### OOD Data: Custom Generator (Future Use)

The OOD test set generator (`code/generate_ood_data.py`) produces:

| Dataset | Samples | Hops | Graph | Notes |
|---------|---------|------|-------|-------|
| `ood_7hop.json` | 1K | 7 | Trees | Novel hop count beyond ProsQA's 3-6 |
| `ood_8hop.json` | 1K | 8 | Trees | Novel hop count |
| `ood_dag.json` | 1K | 4 | DAGs (convergent) | Novel structure |
| `ood_dense.json` | 1K | 4 | Dense (branch 5-8) | Novel density |

These OOD sets will be adapted to match Meta's ProsQA format (statement answers, inline facts) so the model sees a consistent input format.

---

## 7. Verification

Meta's run.py provides built-in logging (per-epoch accuracy, loss, checkpoints). But our novel M4/M4b models need additional verification — these are the most likely to produce degenerate behavior.

### Built-in Verification (Meta's run.py)

| Check | How | Expected |
|-------|-----|----------|
| Val accuracy | `Accuracy on validation set: X / 300` in log | M1: ~75-80%, M3: ~97% (Meta Table 1) |
| Loss convergence | Loss reported per batch in training log | Decreasing trend |
| Checkpoints | `checkpoint_{epoch}` files in save_dir | 50 files per model |

### Required Post-Training Checks (ALL models, especially M4/M4b/M5)

| # | Check | Fail if | Why |
|---|-------|---------|-----|
| 1 | **Answer-token CE loss** | > 0.693 at stages 3+ | Above binary chance = degenerate model. **Critical for M4/M4b/M5** at stages 3+ (epochs 15+), when most explicit CoT is replaced by thought tokens. Early stages still have explicit CoT to anchor the loss, so check stages where the model must rely on thought tokens. |
| 2 | Checkpoint integrity | Load + forward pass fails | Catch corruption |
| 3 | Val accuracy plausible | M1 < 70% or M4/M4b/M5 < 40% | Degenerate or broken training |
| 4 | M1 matches Meta | M1 < 74% at epoch 50 | Environment issue |
| 5 | Stage progression (M3/M4/M4b/M5) | Wrong stage at wrong epoch | Config error |
| 6 | No NaN/Inf in loss log | Any present | Training diverged |
| 7 | M4b embedding trained | Embedding unchanged from init | Gradient not flowing to M4b's pause parameter |
| 8 | **M5 embedding trained** | Embedding unchanged from init | Gradient not flowing to M5's pause parameter |
| 9 | **M5 uses plain GPT-2 forward** | Hidden state recycling detected | M5 should NOT recycle hidden states — if it does, it's just M3 with a different label |
| 10 | **M5 ≤ M3 on prosqa_test** | M5 > M3 by ≥3pp | If M5 beats M3, something is wrong with M3 training (or M5 implementation has a bug giving it an unfair advantage) |

**Checks 1, 9, 10 are CRITICAL for M5.** M5's entire purpose is to be a compute-matched control. If it secretly recycles hidden states (check 9), or if it outperforms M3 (check 10), the control is broken and all experimental comparisons are invalid.

### OOD Data Verification (Already completed)

| Check | Result |
|-------|--------|
| All OOD species names from ProsQA vocabulary | PASS (38 species, 17 person names) |
| Token lengths < 1024 | PASS (max 587 tokens) |
| Reasoning steps trace valid path | PASS (0 issues across 4000 samples) |
| Full graph structure preserved | PASS (edges, root, target, neg_target) |

---

## 8. Fail-Fast Gates

| Gate | When | Key Criteria | Fail Action |
|------|------|-------------|-------------|
| **1** | M1 @ epoch 10 | Val accuracy ≥ 70% overall | Debug; 3 attempts then terminate |
| **2** | M1+M2 @ epoch 50 | M1 75-80%; M2 ≈ M1 (Meta §4.4: "CoT does not offer notable improvement over No-CoT" on ProsQA) | Check data loading |
| **3** | M3 @ stage 2 (epoch 14) | Val acc >58% on 300 val samples | Halve LR; audit implementation |
| **3b** | M5 @ stage 2 (epoch 14) | Val acc >45% (M5 should train, just possibly worse than M3) | Check pause_curriculum implementation; verify embeddings receive gradients |
| **4** | M3+M4+M4b+M5 done | M3 > M5 by ≥5pp OR M3 sens > M5 by ≥10pp on any experiment | Proceed anyway if fails (negative paper) |
| **4b** | M5 done | M5 ≤ M3 on in-distribution (prosqa_test) | If M5 > M3: something wrong with M3 training; debug before experiments |
| **5** | Before experiments (Exp 0) | M1 causal sanity check passes (CE > 0.3 at ≥50% CoT positions) | Debug patching implementation; do NOT run Exp 4 until this passes |

**Expected M1 accuracy:** ~75-80% (Meta's reported CoT accuracy on ProsQA). Our result: 79.67%. Since we run Meta's exact code, any significant deviation indicates an environment issue (batch size, version mismatch), not a code bug.

**Expected M3 accuracy:** ~97% (Meta Table 1 reports COCONUT at ~97% on ProsQA). Our result: 98.0% (490/500 on prosqa_test). This is a near-perfect replication (+1pp, within noise), NOT an anomalous finding. The ~75% figure is the CoT baseline (M1), not COCONUT.

**Expected M5 accuracy:** Unknown (novel model), but should be somewhere between M4b and M3. If M5 < M4b, the pause_curriculum implementation is likely broken. If M5 > M3, M3 training is broken. Hao et al.'s own "pause as thought" ablation shows near-parity with COCONUT on ProsQA, so M5 in the 90-97% range would be consistent with Row 2 (curriculum sufficient).

---

## 9. Specialist Agent Roster

| Agent | Role | Deploy When |
|-------|------|-------------|
| **Methodologist** | Design, controls, confounds | Before experiments; after results |
| **Statistician** | Tests, power, CIs, corrections | After data collection |
| **Tech Code Auditor** | Code, CUDA, gradients, checkpoints | Before training; on errors |
| **Skeptic** | Adversarial review, alternatives | After experiments; before paper |
| **Domain Expert** | COCONUT specifics, Meta comparison | Start; synthesis; submission |

5 rounds each. Issues tagged BLOCKING or ADVISORY. New BLOCKING in round 5 → escalate.

---

## 10. Decision Matrix

**The core question is: does M3 outperform M5?** Every row below is defined by the M3-vs-M5 relationship. M4b results provide secondary confirmation.

| # | M3 vs M5 | Corruption (Exp 1) | Probing (Exp 2) | OOD (Exp 3) | Causal (Exp 4) | Zhang PSR (Exp 5) | Token Count (7.1) | Verdict |
|---|----------|-------------------|-----------------|-------------|----------------|-------------------|-------------------|---------|
| 1 | M3 >> M5 | M3 permutation-sensitive, M5 not; cross-transplant fails M3 only | M3 diagonal + selective; M5 flat | M3 generalizes; M5 doesn't | CE at M3 thoughts only | M3 PSR > M5 PSR or both low (see note) | M3 plateaus at k-hops; M5 monotonic | **COCONUT genuinely reasons via continuous thoughts** |
| 2 | M3 ≈ M5 | Both permutation-sensitive equally | Both diagonal | Both generalize | CE at both | Both similar PSR | Both plateau at k | **Curriculum + extra compute sufficient; continuous thoughts not required.** Publish as negative result for mechanism, positive for curriculum. |
| 3 | M3 ≈ M5 ≈ M4b | All cascade similarly | All flat | None generalize | No CE anywhere | All low PSR | All monotonic | **Nothing reasons. Buffering only.** Strongest negative result. |
| 4a | M3 > M5 on some | M3 permutation-sensitive; M5 partially | M3 diagonal; M5 partial diagonal | M3 generalizes hops; both fail structure | CE at M3 > M5 | Mixed | M3 plateaus; M5 partial | **Partial: continuous thoughts help for certain reasoning types (hop extension) but not structural generalization** |
| 4b | M3 > M5 on probing/causal, M3 ≈ M5 on OOD | M3 cross-transplant fails; M5 also fails | M3 diagonal; M5 flat | Both generalize equally | CE at M3 only | M3 PSR > M5 | Both plateau | **Representations differ but behavior equivalent.** Information present but not functionally necessary. Weaker claim. |

**Row 2 is the most likely outcome** based on Hao et al.'s own "pause as thought" ablation showing near-parity with full COCONUT on ProsQA.

**Note on Exp 5 interpretation:** Low PSR for M3 is compatible with either buffer or reasoning (see PHASE_4_EXPERIMENTS.md Exp 5 interpretation). PSR alone cannot distinguish. Its value is in replicating/contradicting Zhang et al. and in the M3-vs-M5 comparison.

---

## 11. Recovery Hypotheses

### If Row 3 (nothing reasons):

| Order | Hypothesis | Fix | GPU hrs |
|-------|-----------|-----|---------|
| 1 | Insufficient signal | Probe loss on thoughts | ~4 |
| 2 | Curriculum too aggressive | 5-8 epochs/stage | ~6 |
| 3 | Position encoding mismatch | Separate thought position embeds | ~8 |
| 4 | Architecture bottleneck | Multi-vector thoughts | ~15 |
| 5 | Scale insufficient | 1B+ (separate paper) | 100+ |
| 6 | From-scratch limitation | Fine-tune GPT-2 1.5B or Llama 3B with COCONUT curriculum on natural language multi-hop QA (HotpotQA, MuSiQue) | 50+ |

### If Row 2 (M3 ≈ M5 — curriculum sufficient, mechanism irrelevant):

This is not a failure — it's a publishable finding. Recovery means pivoting the paper framing:

| Action | Description |
|--------|-------------|
| **Reframe paper** | "The training curriculum, not the continuous thought mechanism, drives COCONUT's gains on ProsQA" |
| **Test curriculum variants** | Does the same curriculum improve CoT models? (Train M1 with progressive CoT truncation) |
| **Test at scale** | Does M3 vs M5 gap emerge at larger scale? (GPT-2 774M or larger) |
| **Test on harder tasks** | ProsQA may be too easy for the mechanism to matter. Try longer chains, harder graph structures |

### If Row 4a/4b (partial results):

Report honestly as partial. Identify which specific capability comes from the mechanism vs the curriculum. Don't overclaim.

---

## 12. Paper Figures

1. Information surface heatmap (Exp 2) — 5 panels: M3|M4|M4b|M5|input-positions + selectivity overlay
2. OOD grouped bar chart (Exp 3) — M1|M3|M4b|M5, error bars from 3 seeds
3. Corruption cascade curves (Exp 1) — per model including M5, with permutation results highlighted
4. Causal effect heatmap (Exp 4) — 4 panels: M1(validation)|M3|M4b|M5
5. Zhang PSR comparison (Exp 5) — M1 CoT vs M3 thoughts vs M5 pauses, across sigma values
6. Thought token count matrix (Ablation 7.1) — rows=hops, cols=tokens, for M3 AND M5 side-by-side
7. Curriculum divergence plot (Ablation 7.3) — M3 vs M4 vs M4b vs M5
8. Cross-problem transplant matrix (Exp 1 sub) — M3 vs M5 comparison — appendix
9. **Summary figure** — convergent evidence diagram showing which experiments point to which conclusion

---

## 13. Probing Technical Note

Training uses fp32 (Meta's config: `bf16: False`). Hidden states are already float32, so no casting needed for sklearn. Hidden states are 768-dim (GPT-2 124M). Model has 12 layers, 12 heads.

**Selectivity metric (NEW):** For each (layer, position) cell, selectivity = probe_acc(target=step_t) - max(probe_acc(target=step_s≠t)). High selectivity = step-specific encoding. Low selectivity = general problem encoding at every position (buffer). See PHASE_4_EXPERIMENTS.md Exp 2 for full specification.

**Input-position control (NEW):** Also probe hidden states at input token positions (graph fact tokens). If input positions decode step-t info as well as thought-t positions, thoughts add no representational content. See PHASE_4_EXPERIMENTS.md Exp 2.

## 13b. Reference Implementations for Experiments

| Experiment | Reference Code | What to Use | What to Write from Scratch |
|---|---|---|---|
| **Exp 0/4 (Causal tracing)** | **ROME** — `github.com/kmeng01/rome` (`experiments/causal_trace.py`, `notebooks/causal_trace.ipynb`). Also: `pip install causal-tracer` (PyPI) or Stanford `pyvene` library. | The corrupt→restore→measure loop. Hook-based activation patching via `register_forward_hook`. CE formula. Noise calibration. **Read ROME's `trace_with_patch()` first, then adapt.** Do NOT write activation patching from scratch. | Adaptation to Coconut-wrapped models (hook path differs: `model.base_causallm.model.transformer.h[layer]`). Corruption uses matched-distribution random embeddings instead of ROME's Gaussian noise on subject tokens. |
| **Exp 1 (Corruption)** | No reference. Novel design. | — | Everything. Thought token extraction, replacement, permutation, cross-transplant. |
| **Exp 2 (Probing)** | Standard sklearn. No specialized library needed. | `LogisticRegression` and `MLPClassifier` from sklearn. | Hidden state extraction hooks, selectivity metric, input-position control. |
| **Exp 3 (OOD)** | No reference. Just inference on new data. | — | Inference loop with variable thought token counts. |
| **Exp 5 (Zhang replication)** | **No code released.** Zhang et al. (arXiv:2512.21711) did not publish a repository. | — | Everything. Implement from Section 3.1 methodology: Gaussian perturbation at varying σ, PSR measurement. Simpler than Exp 4. |
| **Training (all models)** | **Meta's COCONUT** — `github.com/facebookresearch/coconut` | Already using as v9_meta_fork base. | M4/M4b/M5 `feedback_mode` additions to `coconut.py` and `run.py`. |

## 14. Changes to Meta's Code (Complete List)

Only 3 modifications to Meta's codebase (same count as before — M5 uses the existing `feedback_mode` mechanism):

1. **coconut.py** — Added `feedback_mode` parameter to `Coconut.__init__()` (default "continuous" preserves Meta's behavior). M4 uses `register_buffer` for frozen pause embedding. M4b uses `nn.Parameter` for learned shared embedding. M5 uses `nn.Parameter` for learned pause embedding with `pause_curriculum` mode (same as M4b mechanically, but critically: M5 should be instantiated as plain GPT-2 with pause embeddings injected at input, NOT as a Coconut-wrapped model — see Section 4 M5 description). Feedback loop conditionally uses pause embedding instead of hidden states.

2. **run.py** — Added 2 lines to read `feedback_mode` from config and pass to `Coconut()` constructor. Default "continuous" means M3 behaves identically to Meta's original. For M5 (`pause_curriculum`), the training loop inserts pause embeddings at thought positions instead of recycling hidden states.

3. **Config files** — Created 6 new YAML configs (prosqa_cot, prosqa_nocot, prosqa_coconut_1gpu, prosqa_m4_frozen, prosqa_m4b_shared, **prosqa_m5_pause**). Only changes from Meta's original: `gradient_accumulation_steps: 4` (single GPU), `debug: False`, and model-specific flags.

**Everything else is Meta's code, untouched.**

---

## 15. Known Confounds and Limitations

Documenting these upfront prevents post-hoc rationalization.

1. **Scale:** GPT-2 124M on ProsQA is a toy setup. Zhang et al.'s negative results used LLaMA 7B/8B on MMLU/HotpotQA. Positive results here may not generalize to larger models or natural language tasks. Acknowledge this limitation explicitly in any paper.

2. **Synthetic domain:** ProsQA is a perfectly structured graph traversal task. Real-world reasoning involves noisy, ambiguous, multi-modal information. Strong results on ProsQA prove COCONUT *can* reason in a narrow domain, not that it *does* reason in general.

3. **Curriculum confound:** If M5 ≈ M3, we can't distinguish "curriculum is sufficient" from "pause tokens are secretly doing latent reasoning too." The pause embeddings do get attended to; the model could in principle learn to use those attention operations for reasoning. This is a weaker confound but worth noting.

4. **Probing ≠ using:** High probe accuracy shows information is *present* in a representation, not that the model *uses* it (Ravichander et al., 2021). Exp 4 (causal tracing) partially addresses this, but causal tracing has its own assumptions (linear restoration of causal effect).

5. **Statistical power:** 500 test samples with 5-fold CV. Some effects may be too small to detect. Report confidence intervals for all comparisons, not just point estimates.
