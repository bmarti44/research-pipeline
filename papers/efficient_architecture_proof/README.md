# The Curriculum Is the Mechanism

Meta's COCONUT model replaces human-readable chain-of-thought with a novel architecture: it recycles hidden states between reasoning steps, creating a "continuous latent space" for thinking. On ProsQA (a graph-traversal benchmark), COCONUT achieves 97% accuracy — far above chain-of-thought baselines. The natural conclusion is that this recycling mechanism enables a new form of reasoning.

We show it doesn't. The training curriculum does all the work.

## The Idea

COCONUT is trained with a 7-stage curriculum that progressively removes explicit reasoning tokens, forcing the model to internalize multi-hop reasoning. But two things change simultaneously during this training: the model learns the curriculum *and* it uses the recycling mechanism. Which one drives the result?

We built two controls to find out:

- **M3 (Pause):** Same architecture, same curriculum, same number of thought positions — but instead of recycling hidden states between passes, every thought token gets the same fixed learned embedding. One forward pass. No information flows between reasoning steps.
- **M4 (Pause-Multipass):** Same as M3, but processes thought tokens sequentially across 6 passes, exactly like COCONUT — just without the recycled content. This isolates whether COCONUT's advantage comes from the recycled information or from the sequential processing structure.

## The Result

| Model | What it does | Thought content | Processing | ProsQA Accuracy |
|-------|-------------|----------------|------------|:---------:|
| M1 (CoT) | Explicit reasoning tokens | Human-readable | Single pass | 83.0% |
| M2 (COCONUT) | Recycles hidden states | Rich, information-carrying | 6 sequential passes | 97.0% |
| M3 (Pause) | Fixed learned embedding | Empty | Single pass | 96.6% |
| M4 (Pause-Multipass) | Fixed learned embedding | Empty | 6 sequential passes | *training* |

M3 matches COCONUT despite having no information flow between reasoning steps *and* using 1/6th the compute. Three independent experiments (corruption analysis, probing, cross-model transplant) fail to distinguish the two models on any diagnostic.

On out-of-distribution tests, M3 actually outperforms COCONUT on 3 of 4 test sets:

| Test Set | M1 (CoT) | M2 (COCONUT) | M3 (Pause) |
|----------|------:|------:|------:|
| ProsQA (in-dist) | 83.0% | 97.0% | 96.6% |
| 7-hop | 10.7% | 66.0% | **75.4%** |
| 8-hop | 8.2% | 67.5% | **75.1%** |
| DAG | 28.2% | **59.2%** | 51.9% |
| Dense | 14.1% | 61.2% | **68.4%** |

## What This Means

This is *not* a story about "models just buffer compute." Both M2 and M3 develop structured, position-specific representations — the final thought position encodes the answer entity with +52 percentage points of selectivity over controls. That's real learned structure, not padding.

But that structure comes entirely from the curriculum. The 7-stage progressive removal of chain-of-thought tokens teaches the model to internalize reasoning into whatever computational substrate is available — recycled hidden states, empty sequential passes, or parallel empty tokens. The thought tokens provide a computational budget (extra attention positions). The curriculum teaches the model how to use it. What's *in* the tokens doesn't matter.

**For the latent reasoning community:** Stop optimizing mechanisms. Start optimizing curricula.

## Data

This study uses Meta's ProsQA dataset. All data files are tracked in `code/data/` and available after cloning. If they're missing, run the setup script to repopulate from the git submodule:

```bash
git submodule update --init --recursive
bash setup_data.sh
```

**Expected files in `code/data/`:**

| File | Samples | Description |
|------|---------|-------------|
| `prosqa_train.json` | 17,886 | Training set (~29 MB) |
| `prosqa_valid.json` | 300 | Validation set |
| `prosqa_test.json` | 500 | In-distribution test set |
| `ood_7hop.json` | 1,000 | 7-hop chains (training uses 3-6 hops) |
| `ood_8hop.json` | 1,000 | 8-hop chains |
| `ood_dag.json` | 1,000 | DAG topology (training uses trees) |
| `ood_dense.json` | 1,000 | Dense graphs (higher connectivity) |

Each sample is a dict with keys: `question`, `answer`, `steps` (CoT), `edges`, `root`, `target`, `neg_target`. ProsQA data is subject to Meta's COCONUT repository license (MIT).

## Quick Start

```bash
# From pretrained checkpoints (~2h on 1 GPU):
python reproduce.py --from-checkpoints

# Full reproduction from scratch (~120 GPU hours):
python reproduce.py --full

# Dry run (print steps without executing):
python reproduce.py --from-checkpoints --dry-run
```

Requires Python 3.10+, CUDA 12.x, and an NVIDIA GPU with 40GB+ VRAM (H100/A100).

## Reproduce

### Option A: From Pretrained Checkpoints

Download pretrained model checkpoints from HuggingFace, then run experiments only. See [CHECKPOINTS.md](CHECKPOINTS.md) for checkpoint details.

```bash
python reproduce.py --from-checkpoints
```

This will: install deps, set up data, download checkpoints from HuggingFace, run all experiments (corruption, probing, OOD, stats), and generate figures.

### Option B: Full Reproduction

Train all models from scratch, then run experiments.

```bash
python reproduce.py --full
```

### Manual Steps

If you prefer to run each step individually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up data (copies ProsQA from submodule or checkpoints/data/, generates OOD sets)
bash setup_data.sh

# 3. Train models
cd code
torchrun --nproc_per_node=1 run.py args/prosqa_cot.yaml            # M1 CoT (~8h)
torchrun --nproc_per_node=1 run.py args/prosqa_coconut_1gpu.yaml    # M2 COCONUT (~28h)
torchrun --nproc_per_node=1 run.py args/prosqa_m5_pause.yaml        # M3 Pause (~28h)
torchrun --nproc_per_node=1 run.py args/prosqa_m6_pause_multipass.yaml  # M4 Pause-Multipass (~40h)

# 4. Find best checkpoints and create symlinks
python find_best_epoch.py --all --log-dir ../logs --results-dir ../results --link

# 5. Run experiments
python exp_corruption.py --checkpoint_dir ../results \
    --data data/prosqa_test.json --output_dir ../results/experiments/corruption \
    --num_samples 500

python exp_probing.py --checkpoint_dir ../results \
    --data data/prosqa_test.json --output_dir ../results/experiments/probing \
    --num_samples 500

python exp_ood.py --checkpoint_dir ../results \
    --data_dir data/ --output_dir ../results/experiments/ood

# 6. Run M4-specific experiment suite
python run_all_m6.py --checkpoint ../results/pause-multipass/checkpoint_best \
    --feedback-mode pause_multipass --name pause-multipass

# 7. Statistics and figures
python statistical_analysis.py --results_dirs ../results/experiments \
    --output ../results/statistical_analysis.json
python generate_figures.py --results_dir ../results --output_dir ../results/figures
```

Checkpoints save to `results/{cot-baseline,coconut,pause-curriculum,pause-multipass}/checkpoint_<epoch>`.
Use `find_best_epoch.py --link` to create `checkpoint_best` symlinks to peak-validation epochs.

## Code Changes from Meta's COCONUT

This repo forks Meta's official COCONUT codebase with minimal modifications:

1. **`coconut.py`** — Added `feedback_mode` parameter. `"pause_curriculum"` (M3) uses a single learned pause embedding in a single forward pass. `"pause_multipass"` (M4) uses the same pause embedding but processes thought tokens sequentially across 6 passes, matching COCONUT's processing structure.
2. **`run.py`** — 2 lines to read `feedback_mode` from config and pass it to `Coconut`.
3. **`args/prosqa_m5_pause.yaml`** — M3 Pause config (identical to M2 except `feedback_mode: pause_curriculum`).
4. **`args/prosqa_m6_pause_multipass.yaml`** — M4 Pause-Multipass config (identical to M2 except `feedback_mode: pause_multipass`).

`dataset.py` and `utils.py` are unmodified.

## Repo Structure

```
├── README.md                          # This file
├── reproduce.py                       # One-command reproduction script
├── requirements.txt                   # Python dependencies
├── requirements-lock.txt              # Pinned transitive dependencies
├── setup_data.sh                      # Data setup helper
├── CHECKPOINTS.md                     # Checkpoint download & loading guide
├── manuscript/                        # Paper manuscript and figures
├── code/
│   ├── run.py                         # Training script (+feedback_mode)
│   ├── coconut.py                     # Model (+feedback_mode parameter)
│   ├── dataset.py                     # Data loading (unmodified from Meta)
│   ├── utils.py                       # Utilities (unmodified from Meta)
│   ├── exp_utils.py                   # Shared experiment utilities
│   ├── exp_causal.py                  # Exp 0: Causal tracing sanity gate
│   ├── exp_corruption.py              # Exp 1: Corruption ablation
│   ├── exp_probing.py                 # Exp 2: Representation probing
│   ├── exp_ood.py                     # Exp 3: OOD generalization
│   ├── run_all_m6.py                  # Unified M4 (Pause-Multipass) experiment pipeline
│   ├── wilcoxon_teacher_forced.py     # Wilcoxon species token analysis
│   ├── mlp_probe_grid_search.py       # MLP probe hyperparameter search
│   ├── find_best_epoch.py             # Best checkpoint finder
│   ├── generate_ood_data.py           # OOD test set generator
│   ├── generate_figures.py            # Figure generation
│   ├── statistical_analysis.py        # Statistical tests
│   ├── data/                          # ProsQA + OOD test data
│   └── args/                          # Training configs
│       ├── prosqa_cot.yaml            # M1 (CoT)
│       ├── prosqa_coconut_1gpu.yaml   # M2 (COCONUT)
│       ├── prosqa_m5_pause.yaml       # M3 (Pause)
│       └── prosqa_m6_pause_multipass.yaml  # M4 (Pause-Multipass)
└── results/
    ├── experiments/                    # Experiment output JSONs
    └── statistical_analysis.json      # Statistical tests
```

## References

- Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- Zhang et al. (2025). [On the Causal Role of Continuous Thought Tokens](https://arxiv.org/abs/2512.21711)
