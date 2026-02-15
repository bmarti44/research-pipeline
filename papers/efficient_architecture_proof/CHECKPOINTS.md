# Model Checkpoints

Pretrained model checkpoints for "The Curriculum Is the Mechanism"

## HuggingFace Hub

**Repository**: [briamart/coconut-curriculum-checkpoints](https://huggingface.co/briamart/coconut-curriculum-checkpoints) *(upload pending)*

| Model | Feedback Mode | Best Epoch | Checkpoint | ProsQA Accuracy |
|-------|--------------|:----------:|------------|:---------------:|
| M3 (COCONUT) | `continuous` | 49 | `prosqa-coconut/checkpoint_best` | 97.0% |
| M5 (Pause) | `pause_curriculum` | 43 | `prosqa-m5-pause/checkpoint_best` | 96.6% |
| M6 (Pause-Multipass) | `pause_multipass` | TBD | `prosqa-m6-pause-multipass/checkpoint_best` | *training* |

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
snapshot_download('briamart/coconut-curriculum-checkpoints', local_dir='results/')
"
```

## Loading

```python
from code.exp_utils import load_model_by_name

# Load any model by name
model, tokenizer, info = load_model_by_name("m3", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("m5", "results/", device="cuda")
model, tokenizer, info = load_model_by_name("m6", "results/", device="cuda")
```

## Upload (maintainer only)

```bash
# After training completes, upload checkpoints:
python upload_checkpoints.py --repo briamart/coconut-curriculum-checkpoints

# Dry run first:
python upload_checkpoints.py --repo briamart/coconut-curriculum-checkpoints --dry-run
```

## Checkpoint Format

Each checkpoint is a PyTorch `state_dict` saved with `torch.save()`. The state dict contains:

- `base_causallm.*` — GPT-2 weights (124M parameters)
- `pause_embedding` — Learned pause embedding (M5, M6 only)

All models use `openai-community/gpt2` as the base and add 3 special tokens (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`), giving a vocab size of 50260.
