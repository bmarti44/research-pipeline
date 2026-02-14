# Phase 2: Baseline Training

> **Portability note:** This document is the original execution log from a Lambda Labs H100 server. All paths referencing `/lambda/nfs/experiment/` and SSH commands are specific to that server. To reproduce, substitute your own GPU server and working directory. See the paper's [README.md](README.md) for portable reproduction instructions.

**Prerequisites:** v9_meta_fork set up with Meta's code + minimal M4/M4b additions.
**Where:** Lambda H100.
**Time estimate:** ~6 hours (2 models × 50 epochs × ~3 hrs each, sequential)
**GPU hours:** ~6
**Gates:** Gate 1 (epoch 10), Gate 2 (epoch 50)

---

## Step 2.1: M1 Training (CoT Baseline, 50 Epochs)

Train M1 using Meta's run.py with the CoT config. All 50 epochs at stage=0 (full CoT).

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  mkdir -p /lambda/nfs/experiment/results/v9_meta_fork/logs && \
  nohup env WANDB_MODE=disabled torchrun --nproc_per_node=1 run.py args/prosqa_cot.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m1_cot.log 2>&1 &"
```

**Monitor:**
```bash
ssh ubuntu@192.222.52.148 "grep 'Accuracy on validation' /lambda/nfs/experiment/results/v9_meta_fork/logs/m1_cot.log" | cut -c1-200
```

### Gate 1 Check (After Epoch 10)

**Pass criteria:**
| Check | Required | Rationale |
|-------|----------|-----------|
| Val accuracy at epoch 10 | ≥ 70% | Pretrained GPT-2 should learn ProsQA with CoT (Meta reports ~77.5%) |

**If Gate 1 fails:**
1. Check transformers version matches Meta's (4.46.2)
2. Verify data loaded correctly (17886 train samples)
3. Check effective batch size (should be 128)
4. After 3 attempts: terminate. Fundamental setup issue.

---

## Step 2.2: M2 Training (Direct Baseline, 50 Epochs)

M2 trains on the same data but without CoT steps (`no_cot: True`). **Cannot run in parallel with M1** — FSDP uses ~69 GB VRAM on single GPU.

```bash
ssh ubuntu@192.222.52.148 "source /lambda/nfs/experiment/.venv/bin/activate && \
  cd /lambda/nfs/experiment/code/v9_meta_fork && \
  nohup env WANDB_MODE=disabled MASTER_PORT=29501 torchrun --nproc_per_node=1 --master_port=29501 \
    run.py args/prosqa_nocot.yaml \
    > /lambda/nfs/experiment/results/v9_meta_fork/logs/m2_nocot.log 2>&1 &"
```

---

## Step 2.3: Gate 2 Evaluation

After both M1 and M2 finish 50 epochs, evaluate on test set.

Meta's run.py already evaluates on val set every epoch via generation. For test set evaluation, use Meta's `only_eval` mode or check the epoch 50 val accuracy from the log.

### Gate 2 Checklist

```
□ M1 val accuracy ~75-80% (matching Meta's ~77.5%)
□ M2 val accuracy ≈ M1 (Meta §4.4: "CoT does not offer notable improvement over No-CoT")
□ Neither model at 100% (no data leak)
□ 50 checkpoint files exist for each model
```

**M2 ≈ M1 is expected.** Meta's paper (arXiv 2412.06769, Section 4.4) shows No-CoT ≈ CoT ≈ 77-80% on ProsQA. The improvement comes from COCONUT's latent reasoning (M3 reaches ~97%), not from explicit CoT. ProsQA questions contain all graph edges in the question text, so GPT-2 can memorize question→answer mappings from the 500-sample training set over 50 epochs.

**If M1 accuracy is 2-3pp below Meta's 77.5%:** Likely due to single-GPU batch size (128 vs Meta's 128 with 4-GPU DDP — should match, but FSDP vs DDP could cause minor differences). Proceed if ≥ 74%.

**If M1 accuracy is > 5pp below Meta's:** Check:
- Effective batch size actually 128 (not 32)
- Optimizer resets every epoch (check log for "reset" messages)
- Data loaded correctly
- Loss converges

---

## Red Flags (Stop Everything)

- NaN in loss → check FSDP config
- Val accuracy 100% → data leak
- M2 ≥ M1 after 50 epochs → something fundamentally wrong
- No checkpoints saved → check `debug: False` in config
- CUDA OOM → cannot reduce batch without affecting effective batch size; would need to modify code
