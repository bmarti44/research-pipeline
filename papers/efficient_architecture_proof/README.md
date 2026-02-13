# Does COCONUT Reason or Buffer?

Meta's COCONUT replaces chain-of-thought with continuous hidden states recycled as input embeddings, achieving 97% on ProsQA (graph traversal). We train a compute-matched control — same GPT-2 124M, same 7-stage curriculum, same number of thought positions — but with fixed learned pause embeddings instead of recycled hidden states. The pause baseline matches COCONUT in-distribution and outperforms it on 3 of 4 out-of-distribution test sets. The curriculum drives the gains; the mechanism does not.

## Quick Result

| Test Set | M1 (CoT) | M3 (COCONUT) | M5 (Pause) | M5 - M3 |
|----------|------:|------:|------:|--------:|
| ProsQA (in-dist) | 83.0% | 97.0% | 96.6% | -0.4pp |
| 7-hop | 10.7% | 66.0% | **75.4%** | **+9.4pp** |
| 8-hop | 8.2% | 67.5% | **75.1%** | **+7.6pp** |
| DAG | 28.2% | **59.2%** | 51.9% | -7.3pp |
| Dense | 14.1% | 61.2% | **68.4%** | **+7.2pp** |

Both models show identical corruption profiles (zero permutation sensitivity, successful cross-problem transplant) and diagonal probing patterns. Neither performs sequential reasoning — both buffer compute.

Full analysis with figures and statistical tests: **[RESULTS.md](RESULTS.md)**

## Reproduce

### Setup

```bash
git clone <this-repo> && cd <this-repo>
pip install -r requirements.txt
```

Requires Python 3.10+, CUDA 12.x, and an NVIDIA GPU with >= 40GB VRAM (H100/A100). See `requirements.txt` for package versions.

### Data

Download and generate ProsQA data from Meta's COCONUT repo (see [`data/README.md`](data/README.md)). Place the JSON files in `code/data/` — training configs and experiment scripts reference this path. Then generate OOD test sets:

```bash
cd code
python generate_ood_data.py    # writes to code/data/ood_{7hop,8hop,dag,dense}.json
```

### Train

```bash
cd code

# M1 (CoT baseline) — ~8h on H100
torchrun --nproc_per_node=1 run.py args/prosqa_cot.yaml

# M3 (COCONUT) — ~28h on H100
torchrun --nproc_per_node=1 run.py args/prosqa_coconut_1gpu.yaml

# M5 (Pause-Curriculum control) — ~28h on H100
torchrun --nproc_per_node=1 run.py args/prosqa_m5_pause.yaml
```

Checkpoints save to `results/prosqa-{cot,coconut,m5-pause}/checkpoint_<epoch>`.

### Experiments

```bash
cd code

# Sanity gate (must pass first)
python exp_causal.py --mode sanity --checkpoint_dir ../results \
    --data data/prosqa_test.json --output_dir ../results/experiments/causal_sanity \
    --models m1 --num_samples 50

# Core experiments
python exp_corruption.py --checkpoint_dir ../results \
    --data data/prosqa_test.json --output_dir ../results/experiments/corruption \
    --num_samples 500

python exp_probing.py --checkpoint_dir ../results \
    --data data/prosqa_test.json --output_dir ../results/experiments/probing \
    --num_samples 500

python exp_ood.py --checkpoint_dir ../results \
    --data_dir data/ --output_dir ../results/experiments/ood
```

### Figures and Stats

```bash
python generate_figures.py --results_dir ../results --output_dir ../results/figures
python statistical_analysis.py --results_dirs ../results/experiments --output ../results/statistical_analysis.json
```

## Code Changes from Meta's COCONUT

This repo is a fork of Meta's official COCONUT codebase with 3 modifications:

1. **`coconut.py`** — Added `feedback_mode` parameter. `"pause_curriculum"` (M5) replaces hidden-state recycling with a single learned pause embedding and runs a single forward pass.
2. **`run.py`** — 2 lines to read `feedback_mode` from config and pass it to `Coconut`.
3. **`args/prosqa_m5_pause.yaml`** — M5 training config (identical to M3 except `feedback_mode: pause_curriculum`).

`dataset.py` and `utils.py` are unmodified.

## Repo Structure

```
├── README.md                          # This file
├── RESULTS.md                         # Full technical writeup with figures
├── requirements.txt                   # Python dependencies
├── manuscript/
│   ├── figures/                       # Publication-quality PNGs
├── code/
│   ├── run.py                         # Meta's training script (+feedback_mode passthrough)
│   ├── coconut.py                     # Meta's model (+feedback_mode parameter)
│   ├── dataset.py                     # Meta's data loading (unmodified)
│   ├── utils.py                       # Meta's utilities (unmodified)
│   ├── exp_utils.py                   # Shared experiment utilities
│   ├── exp_causal.py                  # Exp 0: Causal tracing sanity gate
│   ├── exp_corruption.py              # Exp 1: Corruption ablation
│   ├── exp_probing.py                 # Exp 2: Representation probing
│   ├── exp_ood.py                     # Exp 3: OOD generalization
│   ├── generate_ood_data.py           # OOD test set generator
│   ├── generate_figures.py            # Figure generation
│   ├── statistical_analysis.py        # Statistical tests
│   ├── find_best_epoch.py             # Checkpoint selection utility
│   └── args/                          # Training configs (YAML)
│       ├── prosqa_cot.yaml            # M1 (CoT baseline)
│       ├── prosqa_nocot.yaml          # M2 (direct answer, not used in paper)
│       ├── prosqa_coconut_1gpu.yaml   # M3 (COCONUT)
│       └── prosqa_m5_pause.yaml       # M5 (Pause-Curriculum)
├── results/
│   ├── experiments/                   # Experiment output JSONs
│   ├── statistical_analysis.json      # Full statistical analysis
│   ├── m3_test_eval.json              # M3 test set accuracy
│   └── m5_test_eval.json              # M5 test set accuracy
└── data/
    └── README.md                      # Data download instructions
```

## References

- Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- Zhang et al. (2025). [On the Causal Role of Continuous Thought Tokens](https://arxiv.org/abs/2512.21711)
