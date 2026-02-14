# Data

This study uses Meta's ProsQA dataset from the COCONUT codebase. We do not redistribute it, but it is available via the Git submodule at `reference_repos/coconut/`.

**Important:** Training configs and experiment scripts reference data as `data/prosqa_train.json` relative to the `code/` directory. All data files must be placed in `code/data/`, not this top-level `data/` directory.

## Quick Setup

Run the setup script from the paper root:

```bash
cd papers/efficient_architecture_proof

# Initialize the COCONUT submodule (if not already done)
git submodule update --init --recursive

# Copy data and generate OOD test sets
bash setup_data.sh
```

## Manual Setup

### 1. Get ProsQA data

The ProsQA data files are included in Meta's COCONUT repository, which is available as a Git submodule at `reference_repos/coconut/data/`. Initialize the submodule, then copy the data:

```bash
git submodule update --init --recursive
cp reference_repos/coconut/data/prosqa_*.json code/data/
```

Alternatively, clone Meta's repository directly:

```bash
git clone https://github.com/facebookresearch/coconut.git /tmp/coconut
cp /tmp/coconut/data/prosqa_*.json code/data/
```

### 2. Expected files in code/data/

```
code/data/
├── prosqa_train.json      # 17,886 samples (~29 MB)
├── prosqa_valid.json      # 300 samples
└── prosqa_test.json       # 500 samples
```

Each sample is a dict with keys:
- `question`: graph traversal question with nonsense entities
- `answer`: single entity name (the target node)
- `steps`: list of intermediate reasoning steps (CoT)
- `edges`: graph edge list
- `root`, `target`, `neg_target`: graph metadata

The training YAML configs reference `train_path: data/prosqa_train.json` and `val_path: data/prosqa_valid.json` (relative to the `code/` working directory). The experiment scripts reference `data/prosqa_test.json` the same way.

### 3. Generate OOD test sets

After setting up the base ProsQA data, generate the out-of-distribution test sets:

```bash
cd code
python generate_ood_data.py
```

This creates 4 OOD test files in `code/data/` (1,000 samples each, seed 42):
- `data/ood_7hop.json` — 7-hop chains (training uses 3-6 hops)
- `data/ood_8hop.json` — 8-hop chains
- `data/ood_dag.json` — DAG topology (training uses trees)
- `data/ood_dense.json` — Dense graphs (higher connectivity)

The script has no CLI arguments — output path and sample counts are hardcoded.

## License

ProsQA data is subject to Meta's COCONUT repository license (MIT).
