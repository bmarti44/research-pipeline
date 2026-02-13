# Phase 1: Setup + Data Preparation

**Prerequisites:** Read MASTER_REFERENCE.md first.
**Where:** Everything on Lambda H100.
**Time estimate:** ~1-2 hours
**GPU hours:** 0 (CPU only)
**Gates:** Gate 0

---

## Strategy: Fork Meta's COCONUT Repo Directly

We use Meta's COCONUT codebase as-is, with exactly 3 surgical modifications:

1. **coconut.py**: Added `feedback_mode` parameter to `Coconut.__init__()` for M4/M4b pause token variants
2. **coconut.py**: Modified thought token feedback loop to use `self.pause_embedding` when `feedback_mode != "continuous"`
3. **run.py**: 2 lines to read `feedback_mode` from config and pass to `Coconut` constructor

This avoids the bugs that plagued the from-scratch v8 reimplementation.

---

## Step 1.1: Clone and Set Up v9_meta_fork

```bash
ssh ubuntu@192.222.52.148 "mkdir -p /lambda/nfs/experiment/code && \
  cp -r /lambda/nfs/experiment/reference_repos/coconut /lambda/nfs/experiment/code/v9_meta_fork" | cut -c1-200
```

**Install dependencies (Meta's pinned versions):**
```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  pip install transformers==4.46.2 torch==2.5.1 numpy==2.1.3 datasets==3.1.0 scikit-learn" | cut -c1-200
```

**Verify:**
```bash
ssh ubuntu@192.222.52.148 "ls /lambda/nfs/experiment/code/v9_meta_fork/" | cut -c1-200
# Expected: args/  coconut.py  dataset.py  preprocessing/  run.py  utils.py  data/
```

---

## Step 1.2: Verify Data

Meta's ProsQA data should already be in the `data/` directory of the repo.

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
for f in ['data/prosqa_train.json', 'data/prosqa_valid.json', 'data/prosqa_test.json']:
    d = json.load(open(f))
    hops = [len(s['steps']) for s in d]
    print(f'{f}: {len(d)} samples, hops {min(hops)}-{max(hops)}')
\"" | cut -c1-200
# Expected:
#   prosqa_train.json: 17886 samples, hops 3-6
#   prosqa_valid.json: 300 samples, hops 3-6
#   prosqa_test.json: 500 samples, hops 3-6
```

---

## Step 1.3: Apply M4/M4b Modifications to coconut.py

Apply the 3 surgical changes. See MASTER_REFERENCE Section 14 for exact diffs.

**Patch 1 (coconut.py `__init__`)**: Add `feedback_mode="continuous"` parameter. Add pause embedding initialization for "frozen" and "learned_shared" modes.

**Patch 2 (coconut.py feedback loop)**: In the thought token replacement loop, branch on `self.feedback_mode`:
- `"continuous"` (M3): Meta's original hidden-state feedback
- `"frozen"` (M4): use `self.pause_embedding` (registered as buffer, no gradient)
- `"learned_shared"` (M4b): use `self.pause_embedding` (nn.Parameter, gradient flows)

**Patch 3 (run.py)**: Two lines to pass feedback_mode from config to Coconut:
```python
if configs.coconut:
    feedback_mode = getattr(configs, "feedback_mode", "continuous")
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, feedback_mode=feedback_mode)
```

**Smoke test all 3 feedback modes:**
```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut

tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens('<|start-latent|>')
tokenizer.add_tokens('<|end-latent|>')
tokenizer.add_tokens('<|latent|>')
latent_id = tokenizer.convert_tokens_to_ids('<|latent|>')
start_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
end_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')

for mode in ['continuous', 'frozen', 'learned_shared']:
    model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids('<<')
    for tid in [latent_id, start_id, end_id]:
        embeddings.weight.data[tid] = embeddings.weight.data[target_id].clone()
    cmodel = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id, feedback_mode=mode)
    n_params = sum(p.numel() for p in cmodel.parameters())
    n_trainable = sum(p.numel() for p in cmodel.parameters() if p.requires_grad)
    print(f'{mode}: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable')
    del cmodel, model
print('ALL SMOKE TESTS PASSED')
\"" | cut -c1-200
# Expected: continuous and learned_shared have same trainable count.
# frozen has slightly fewer trainable params (pause embedding is a buffer).
```

---

## Step 1.4: Create Training Configs

Create single-GPU YAML configs for all 5 models. Key parameters:
- `batch_size_training: 32`, `gradient_accumulation_steps: 4` → effective batch 128 (matches Meta's 4-GPU setup)
- `num_epochs: 50` for ALL models
- `seed: 0` (seeds 1, 2 created in Phase 5)
- `bf16: False` (fp32 training)

```bash
# M1: CoT baseline
# args/prosqa_cot.yaml — cot: True, coconut: False

# M2: Direct baseline
# args/prosqa_nocot.yaml — no_cot: True, coconut: False

# M3: COCONUT
# args/prosqa_coconut_1gpu.yaml — coconut: True

# M4: Pause-frozen
# args/prosqa_m4_frozen.yaml — coconut: True, feedback_mode: frozen

# M4b: Pause-learned-shared
# args/prosqa_m4b_shared.yaml — coconut: True, feedback_mode: learned_shared
```

**Verify all configs:**
```bash
ssh ubuntu@192.222.52.148 "cd /lambda/nfs/experiment/code/v9_meta_fork && \
  for f in args/prosqa_*.yaml; do
    echo \"=== \$f ===\"
    grep -E 'seed:|num_epochs:|batch_size|gradient_accum|coconut:|cot:|no_cot:|feedback_mode:|name:' \$f
  done" | cut -c1-200
```

---

## Step 1.5: Generate OOD Test Data

Generate 4 out-of-distribution test sets using **exact ProsQA vocabulary** (17 person names, 38 species names):

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python generate_ood_data.py" | cut -c1-200
```

This creates:
- `data/ood_7hop.json` (1000 samples, 7-hop reasoning chains)
- `data/ood_8hop.json` (1000 samples, 8-hop chains)
- `data/ood_dag.json` (1000 samples, DAG structure instead of trees)
- `data/ood_dense.json` (1000 samples, dense branching factor 5-8)

**Verify OOD data:**
```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  python -c \"
import json
for f in ['data/ood_7hop.json', 'data/ood_8hop.json', 'data/ood_dag.json', 'data/ood_dense.json']:
    d = json.load(open(f))
    hops = [len(s['steps']) for s in d]
    print(f'{f}: {len(d)} samples, hops {min(hops)}-{max(hops)}, has steps: {all(\\\"steps\\\" in s for s in d)}')
\"" | cut -c1-200
```

---

## Step 1.6: Understanding Meta's Code (Reference)

Key things verified about Meta's implementation:

1. **Multi-pass forward**: `coconut.py` processes thought tokens one at a time. Each pass feeds back the hidden state from position t-1 as the embedding for position t.

2. **Curriculum**: Stage = epoch // epochs_per_stage (capped at max_latent_stage=6). Stage 0 = full CoT. Stage 6+ = all latent.

3. **Optimizer reset**: AdamW optimizer is recreated (`del optimizer; optimizer = optim.AdamW(...)`) **every epoch**. This means first/second moment estimates never accumulate past a single epoch. For curriculum models this prevents stale momentum across stage transitions, but for baselines at constant stage 0 it slows convergence — explaining why 50 epochs are needed.

4. **No LR scheduler**: Constant LR 1e-4 throughout.

5. **fp32 training**: `bf16: False` in config. FSDP wraps but doesn't shard GPT-2 blocks (only LlamaDecoderLayer in auto-wrap policy), so it effectively acts as DDP.

6. **Answer format**: `"### " + answer` with `eos_token_id` appended. Answer is a statement like "Tom is a zhorpus." not Yes/No.

---

## Gate 0 Checklist

```
□ v9_meta_fork directory exists with Meta's code + M4/M4b patches
□ All 3 feedback modes pass smoke test
□ Meta's ProsQA data verified (17886/300/500 samples)
□ 5 YAML configs created and verified
□ OOD data generated with correct ProsQA vocabulary
□ All OOD samples have steps field (for probing/corruption experiments)
□ transformers==4.46.2 installed (Meta's pinned version)
```

**If any check fails:** Fix and re-run. Do not proceed to Phase 2.
