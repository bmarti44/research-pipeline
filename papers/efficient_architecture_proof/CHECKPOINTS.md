# Model Checkpoints

Pretrained model checkpoints for "The Curriculum Is the Mechanism"

## HuggingFace Hub

**Repository**: [bmarti44/coconut-curriculum-checkpoints](https://huggingface.co/bmarti44/coconut-curriculum-checkpoints) *(upload pending)*

| Model | Name | Feedback Mode | Best Epoch | Checkpoint | ProsQA Accuracy |
|-------|------|--------------|:----------:|------------|:---------------:|
| M2 | COCONUT | `continuous` | 49 | `coconut/checkpoint_best` | 97.0% |
| M3 | Pause-Curriculum | `pause_curriculum` | 43 | `pause-curriculum/checkpoint_best` | 96.6% |
| M4 | Pause-Multipass | `pause_multipass` | TBD | `pause-multipass/checkpoint_best` | *training* |

**Note**: Best epoch is determined by peak validation accuracy via `find_best_epoch.py`.
The `checkpoint_best` symlink points to the peak-validation epoch.
`load_model_by_name()` searches for `checkpoint_best` first, then falls back to
`checkpoint_50` (final epoch), then the highest-numbered checkpoint.

## Download

```bash
# Automatic (via reproduce.py)
python reproduce.py --from-checkpoints

# Manual (via huggingface_hub)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('bmarti44/coconut-curriculum-checkpoints', local_dir='results/')
"
```

## Loading

```python
from code.exp_utils import load_model_by_name

# Load any model by descriptive name
model, tokenizer, info = load_model_by_name("coconut", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("pause-curriculum", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("pause-multipass", "results/", device="cuda")
```

## Upload (maintainer only)

```bash
# After training completes, upload checkpoints:
python upload_checkpoints.py --repo bmarti44/coconut-curriculum-checkpoints

# Dry run first:
python upload_checkpoints.py --repo bmarti44/coconut-curriculum-checkpoints --dry-run
```

## Checkpoint Format

Each checkpoint is a PyTorch `state_dict` saved with `torch.save()`. The state dict contains:

- `base_causallm.*` — GPT-2 weights (124M parameters)
- `pause_embedding` — Learned pause embedding (pause-curriculum and pause-multipass only)

All models use `openai-community/gpt2` as the base and add 3 special tokens (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`), giving a vocab size of 50260.
