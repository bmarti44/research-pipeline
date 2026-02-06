# A+B+C Architecture Study - Code

Reproduction code for the efficiency mechanisms study testing Memory, MoD, and COCONUT combinations.

## Requirements

```bash
# Python 3.12.11
pip install -r requirements.lock
```

See `requirements.lock` for pinned versions:
- torch==2.10.0
- numpy==2.4.2
- scipy==1.17.0
- statsmodels==0.14.6

## Reproduction Commands

### Main Results (v3.3)

```bash
# Activate environment
source /path/to/.venv/bin/activate
cd code

# Baseline (n=10 seeds)
python training/train_abc.py --condition baseline --max_steps 500 \
    --seeds 42 123 456 789 1001 1234 2345 3456 4567 5678 \
    --output ../results/v3.3_power/baseline

# Memory-only (n=10 seeds)
python training/train_abc.py --condition memory_only --max_steps 500 \
    --seeds 42 123 456 789 1001 1234 2345 3456 4567 5678 \
    --output ../results/v3.3_power/memory_only

# MoD at various capacities
python training/train_abc.py --condition mod_only --mod_capacity 0.9 \
    --max_steps 500 --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_mod_90

python training/train_abc.py --condition mod_only --mod_capacity 0.95 \
    --max_steps 500 --seeds 42 123 456 789 1001 \
    --output ../results/v3.3_mod_95

# Scale test (8.5M params)
python training/train_abc.py --condition baseline --size small --max_steps 1000 \
    --seeds 42 123 456 789 1001 --output ../results/v3.3_scale/baseline

python training/train_abc.py --condition memory_only --size small --max_steps 1000 \
    --seeds 42 123 456 789 1001 --output ../results/v3.3_scale/memory_only
```

### Statistical Analysis

```bash
python analysis/analyze_results.py \
    --baseline ../results/v3.3_power/baseline \
    --treatment ../results/v3.3_power/memory_only \
    --metric final_val_ppl \
    --baseline-name "Baseline" \
    --treatment-name "Memory"
```

## Key Results

| Condition | Tiny (1M) | Small (8.5M) | Verdict |
|-----------|-----------|--------------|---------|
| Memory | -0.34% (p=0.44) | +0.10% (p=0.55) | No detectable effect |
| MoD 90% | +2.6% (p=0.0001) | - | Small overhead |
| MoD 95% | +1.66% (p=0.020) | - | Minimal overhead |
| COCONUT | - | - | Out of scope (requires subword tokens) |

## Directory Structure

```
code/
├── training/
│   └── train_abc.py      # Main training script
├── models/
│   ├── lahr_v4.py        # MoD + Memory
│   ├── coconut_full.py   # COCONUT mechanism
│   └── lahr_coconut.py   # Full A+B+C
├── data/
│   ├── multistep_train.json
│   └── multistep_val.json
├── analysis/
│   └── analyze_results.py # Statistical analysis
└── requirements.lock     # Pinned dependencies
```

## Hardware

Tested on:
- Apple M1/M2 MacBook (MPS backend)
- ~1-2 hours for full reproduction

## Citation

See `../RESEARCH_PLAN_ABC.md` for full methodology and results.
