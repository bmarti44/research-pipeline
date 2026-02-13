# Phase 3: COCONUT + Pause-Token Training

**Prerequisites:** Phase 2 complete. Gate 2 passed.
**Where:** Lambda H100.
**Time estimate:** ~12 hours (4 models sequential — FSDP uses ~69 GB, can't parallel)
  - M3 (COCONUT): ~3 hrs
  - M4 (frozen pause): ~3 hrs
  - M4b (learned shared pause): ~3 hrs
  - M5 (pause-curriculum, single pass): ~2 hrs (faster — no multi-pass loop)
**GPU hours:** ~11
**Gates:** Gate 3, Gate 3b, Gate 4, Gate 4b

---

## Strategy: Fork Meta's Code, Minimal Additions

All four models (M3, M4, M4b, M5) use Meta's run.py with the Coconut class. The only code changes are:
- `coconut.py`: `feedback_mode` parameter (default "continuous" = Meta's original M3 behavior)
  - `"continuous"` (M3): hidden-state recycling (Meta's default)
  - `"frozen"` (M4): fixed pause embedding, no gradient
  - `"learned_shared"` (M4b): single learned nn.Parameter pause embedding, multi-pass loop
  - `"pause_curriculum"` (M5): single learned nn.Parameter pause embedding, **single forward pass** (no multi-pass loop)
- `run.py`: passes `feedback_mode` from config to Coconut constructor

**No warm-start.** All models initialize from `GPT2LMHeadModel.from_pretrained("openai-community/gpt2")`. Stage 0 (epochs 0-4) IS the built-in CoT warm-up.

**M3 vs M5 is the PRIMARY comparison in every experiment.** M3 vs M4b is secondary. See REFERENCE.md §4 for full comparison hierarchy.

---

## Step 3.1: M3 COCONUT Training (50 Epochs)

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  nohup env WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_coconut_1gpu.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m3_coconut.log 2>&1 &"
```

This is Meta's exact COCONUT training with `feedback_mode=continuous` (the default). Curriculum:

| Stage | Epochs | Thought tokens | Explicit CoT remaining |
|-------|--------|---------------|------------------------|
| 0 | 0-4 | 0 | All (full CoT, same as M1) |
| 1 | 5-9 | 1 | All except last step |
| 2 | 10-14 | 2 | All except last 2 |
| 3 | 15-19 | 3 | All except last 3 |
| 4 | 20-24 | 4 | All except last 4 |
| 5 | 25-29 | 5 | All except last 5 |
| 6 | 30-49 | All | None (fully latent) |

### Gate 3: Check After Stage 2 (Epoch 14)

```bash
ssh ubuntu@192.222.52.148 "grep 'Accuracy on validation' /lambda/nfs/experiment/results/v9_meta_fork/logs/m3_coconut.log | head -15" | cut -c1-200
```

**Gate 3 Checklist:**
```
□ M3 val accuracy at epoch 14 (stage 2) > 58%
□ Loss not diverging
```

**If Gate 3 fails:** M3 is Meta's exact code, so failure likely means:
1. Environment mismatch (transformers version, CUDA)
2. Effective batch size wrong
3. Data issue

---

## Step 3.2: M4 Pause-Frozen Training (50 Epochs)

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  nohup env WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_m4_frozen.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m4_frozen.log 2>&1 &"
```

Same curriculum as M3, but `feedback_mode=frozen`: thought token embeddings are a fixed vector (initialized from `<<` token embedding, `register_buffer` so no gradient flows to it).

---

## Step 3.3: M4b Pause-Learned-Shared Training (50 Epochs)

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  nohup env WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_m4b_shared.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m4b_shared.log 2>&1 &"
```

Same curriculum as M3, but `feedback_mode=learned_shared`: all thought positions share a single `nn.Parameter` embedding. The model can co-adapt to this vector during training, but cannot encode problem-specific sequential state.

---

## Step 3.4: M5 Pause-Curriculum Training (50 Epochs)

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  nohup env WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_m5_pause.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m5_pause.log 2>&1 &"
```

Same curriculum as M3, but `feedback_mode=pause_curriculum`: at every thought position, inserts a single learned `<pause>` embedding (`nn.Parameter`, shared across all positions and all samples). **Single forward pass** — no hidden-state recycling, no multi-pass loop. M5 gets the same curriculum schedule and the same number of forward passes as M3 during training, but thought positions receive a learned pause embedding instead of the previous step's hidden state.

**M5 is the most important control in the entire study.** M3 vs M5 isolates whether the continuous thought mechanism itself contributes anything beyond curriculum + extra compute.

### Gate 3b: Check M5 After Stage 2 (Epoch 14)

```bash
ssh ubuntu@192.222.52.148 "grep 'Accuracy on validation' /lambda/nfs/experiment/results/v9_meta_fork/logs/m5_pause.log | head -15" | cut -c1-200
```

**Gate 3b Checklist:**
```
□ M5 val accuracy at epoch 14 (stage 2) > 45%
□ Loss not diverging
□ M5 pause_embedding has changed from init (gradient is flowing)
```

**If Gate 3b fails:** Check that `pause_curriculum` mode in `coconut.py` is correctly inserting pause embeddings and computing loss. Verify `nn.Parameter` receives gradient.

---

## Step 3.5: Gate 4 — Separation Signal

After all 4 models finish:

```bash
# Compare final val accuracies
ssh ubuntu@192.222.52.148 "for log in m3_coconut m4_frozen m4b_shared m5_pause; do \
  echo \"=== \$log ===\"
  grep 'Accuracy on validation' /lambda/nfs/experiment/results/v9_meta_fork/logs/\${log}.log | tail -1
done" | cut -c1-200
```

**Gate 4 Checklist:**
```
□ M3 accuracy > M5 accuracy by ≥ 5pp
  OR M3 corruption sensitivity > M5 by ≥ 10pp (run abbreviated corruption test)

If BOTH fail: Strong evidence for curriculum/compute hypothesis (Row 2).
PROCEED ANYWAY — run all experiments for negative-result paper.

□ M5 ≤ M3 on in-distribution (prosqa_test)
  If M5 > M3 by ≥ 3pp: something is wrong with M3 training. Debug before experiments.
```

### Gate 4b: M5 Verification (Critical)

```bash
# Verify M5 pause embedding was trained (changed from init)
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python3 -c \"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load init embedding
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens('<|start-latent|>')
tokenizer.add_tokens('<|end-latent|>')
tokenizer.add_tokens('<|latent|>')
base = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
base.resize_token_embeddings(50260)
init_embed = base.transformer.wte.weight.data[50259].clone()

# Load M5 checkpoint
ckpt = torch.load('../../results/v9_meta_fork/prosqa-m5-pause/checkpoint_50', map_location='cpu')
trained_embed = ckpt['pause_embedding']

# Compare
diff = (trained_embed - init_embed).norm().item()
cos_sim = torch.nn.functional.cosine_similarity(init_embed.unsqueeze(0), trained_embed.unsqueeze(0)).item()
print(f'L2 distance from init: {diff:.6f}')
print(f'Cosine similarity to init: {cos_sim:.6f}')
print(f'PASS: embedding trained' if diff > 0.01 else 'FAIL: embedding unchanged')
\"" | cut -c1-200
```

---

## Red Flags

- M3 loss spikes at stage transitions → check optimizer reset (should reset every epoch)
- M3 accuracy drops to chance after stage 3 → curriculum too aggressive (unlikely with Meta's code)
- M4b accuracy equals M3 → may not be a bug — this IS a result (Row 2 of Decision Matrix)
- M5 accuracy equals M3 → may not be a bug — this IS a result (Row 2, strongest version)
- M5 > M3 by ≥3pp → BUG. Debug M3 training or M5 implementation before experiments.
- Any model at 100% → data leak
- Training takes >4 hours per model → check GPU utilization (M5 may be faster due to single pass)
- M5 pause embedding unchanged → gradient not flowing, training is broken
