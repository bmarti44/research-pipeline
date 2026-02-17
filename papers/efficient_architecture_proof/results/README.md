# Results Directory

All experiment outputs, analysis results, and training logs for "The Curriculum Is the Mechanism."

## Directory Structure

```
results/
├── experiments/                          # Per-experiment outputs
│   ├── causal_sanity/                    # Causal intervention sanity checks
│   │   ├── exp0_sanity_result.json       #   Baseline sanity check
│   │   └── m1_causal.json               #   M1 (CoT) causal intervention data
│   ├── corruption/                       # Thought corruption analysis
│   │   ├── results.json                  #   Aggregated corruption results (M2/M3)
│   │   ├── m3_corruption.json            #   M3 raw per-position corruption
│   │   └── m5_corruption.json            #   M5 raw per-position corruption
│   ├── m6/                               # M4 (Pause-Multipass) full experiment suite
│   │   ├── accuracy.json                 #   ID + OOD accuracy
│   │   ├── corruption.json               #   Corruption analysis
│   │   ├── mcnemar.json                  #   McNemar tests vs M2, M3
│   │   ├── m6_labels.json                #   Per-sample ground truth labels
│   │   ├── m6_linear_perm.json           #   Permutation test results
│   │   ├── m6_transplant.json            #   Cross-model thought transplant
│   │   ├── per_sample_correctness.json   #   Per-trial correct/incorrect
│   │   ├── summary.json                  #   Aggregated summary
│   │   └── m6_hidden_states.npz          #   Cached hidden states (not in git, ~72MB)
│   ├── m6_epoch39/                       # M4 at epoch 39 (comparison checkpoint)
│   │   ├── accuracy.json
│   │   ├── per_sample_correctness.json
│   │   └── summary.json
│   ├── mcnemar/
│   │   └── results.json                  # M2 vs M3 McNemar test (pre-M4)
│   ├── ood/
│   │   ├── results.json                  # Aggregated OOD results (M1/M2/M3)
│   │   └── log.txt                       # OOD experiment log
│   ├── ood_persample/                    # Per-sample OOD outputs (for Wilcoxon)
│   │   ├── m1_ood_{7hop,8hop,dag,dense}.json
│   │   ├── m1_prosqa_test.json
│   │   ├── m3_ood_{7hop,8hop,dag,dense}.json
│   │   ├── m3_prosqa_test.json
│   │   ├── m5_ood_{7hop,8hop,dag,dense}.json
│   │   └── m5_prosqa_test.json
│   ├── probing/                          # Linear probing experiments
│   │   ├── results.json                  #   Aggregated probing results
│   │   ├── m3_probing.json               #   M3 raw probing data
│   │   ├── m5_probing.json               #   M5 raw probing data
│   │   └── log.txt                       #   Probing experiment log
│   ├── probing_corrected/                # Corrected probing with permutation tests
│   │   ├── m3_hidden_states.npz          #   M3 cached hidden states (~67MB, not in git)
│   │   ├── m5_hidden_states.npz          #   M5 cached hidden states (~67MB, not in git)
│   │   ├── m3_labels.json                #   M3 per-sample labels
│   │   ├── m5_labels.json                #   M5 per-sample labels
│   │   ├── m3_linear_perm.json           #   M3 permutation test results
│   │   ├── m5_linear_perm.json           #   M5 permutation test results
│   │   └── results_linear_perm.json      #   Combined permutation results
│   ├── per_sample_correctness.json       # M3/M5 reference correctness (used by run_all_m6.py)
│   ├── per_sample_species_logprobs_*.json # Pairwise species logprobs for Wilcoxon tests
│   ├── mlp_probe_grid_search.json        # MLP probe hyperparameter search results
│   ├── wilcoxon_teacher_forced_*.json    # Pairwise Wilcoxon signed-rank test results
│   ├── wilcoxon_diagnostics.json         # Wilcoxon diagnostic information
│   └── wilcoxon_sensitivity{,_v3}.json   # Sensitivity analysis for Wilcoxon tests
│
├── logs/                                 # Training logs (needed for figure regeneration)
│   ├── m1_cot.log                        #   M1 (CoT) training log
│   ├── m2_nocot.log                      #   M2 (NoCot baseline) training log
│   ├── m3_coconut.log                    #   M2 (COCONUT) training log
│   ├── m4_training.log                   #   M4 (Pause-Multipass) training log
│   └── m5_pause.log                      #   M3 (Pause-Curriculum) training log
│
├── statistical_analysis.json             # Aggregated stats used by generate_figures.py
├── selectivity_recomputed.json           # Probing selectivity (used by plot_selectivity_bars.py)
├── appendix_data.json                    # Supplementary/appendix data
├── cross_corruption.json                 # Cross-model corruption analysis
├── unmatched_transplant.json             # Unmatched thought transplant results
├── mcnemar_verification.json             # McNemar test verification
├── permutation_power.json                # Permutation test power analysis
├── pause_embedding_architecture.txt      # Pause embedding architecture notes
├── probing_m3_log.txt                    # M3 probing run log
└── probing_m5_log.txt                    # M5 probing run log
```

## Model Naming

| Paper Name | Descriptive Name | Code Directory | Feedback Mode |
|-----------|-----------------|---------------|---------------|
| M1 | CoT (baseline) | `prosqa-cot` | `cot` |
| M2 | COCONUT | `prosqa-coconut` | `continuous` |
| M3 | Pause-Curriculum | `prosqa-m5-pause` | `pause_curriculum` |
| M4 | Pause-Multipass | `pause-multipass` | `pause_multipass` |

## Model Checkpoints

Checkpoints are stored on HuggingFace at [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints) and are **not kept locally**. Download with:

```bash
python reproduce.py --from-checkpoints
```

See `CHECKPOINTS.md` for details.

## Hidden States (.npz files)

The `*_hidden_states.npz` files are large (~67-72 MB each) cached intermediate outputs from probing experiments. They are `.gitignore`d and can be regenerated by running the probing pipeline with the model checkpoints:

```bash
cd code
python rerun_probes.py --checkpoint_dir ../results --output_dir ../results/experiments/probing_corrected
python run_all_m6.py --checkpoint ../results/pause-multipass/checkpoint_best --feedback-mode pause_multipass --name m6 --stages probing
```

## Key Code Relationships

| Script | Reads From | Writes To |
|--------|-----------|----------|
| `generate_figures.py` | `experiments/*/results.json`, `statistical_analysis.json` | `manuscript/figures/` |
| `run_all_m6.py` | `experiments/per_sample_correctness.json`, `code/data/` | `experiments/m6/` |
| `wilcoxon_teacher_forced.py` | `code/data/` | `experiments/wilcoxon_*.json`, `experiments/per_sample_species_*.json` |
| `plot_selectivity_bars.py` | `selectivity_recomputed.json` | `manuscript/figures/` |
| `statistical_analysis.py` | `experiments/*/results.json` | `statistical_analysis.json` |
| `mlp_probe_grid_search.py` | `experiments/probing_corrected/` | `experiments/mlp_probe_grid_search.json` |
