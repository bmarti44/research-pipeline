# Data

This study uses Meta's ProsQA dataset from the COCONUT codebase. We do not redistribute it.

**Important:** Training configs and experiment scripts reference data as `data/prosqa_train.json` relative to the `code/` directory. All data files must be placed in `code/data/`, not this top-level `data/` directory.

## Setup

### 1. Generate ProsQA data

Clone Meta's COCONUT repository and generate the ProsQA dataset:

```bash
git clone https://github.com/facebookresearch/coconut.git
cd coconut/preprocessing
python generate_data.py --task prosqa
```

This produces JSON files. Each sample is a dict with keys:
- `question`: graph traversal question with nonsense entities
- `answer`: single entity name (the target node)
- `steps`: list of intermediate reasoning steps (CoT)
- `edges`: graph edge list
- `root`, `target`, `neg_target`: graph metadata

### 2. Place files in code/data/

```
code/data/
├── prosqa_train.json      # 17,886 samples
├── prosqa_valid.json      # 300 samples
└── prosqa_test.json       # 500 samples
```

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
