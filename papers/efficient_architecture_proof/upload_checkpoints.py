#!/usr/bin/env python3
"""Upload trained model checkpoints to HuggingFace Hub.

Reads best checkpoint for each model (M2 COCONUT, M3 Pause, M4 Pause-Multipass),
uploads with model cards. Includes config.yaml so loaders know the feedback_mode.

Usage:
    # Upload all models
    python upload_checkpoints.py --repo bmarti44/coconut-curriculum-checkpoints

    # Upload just pause-multipass
    python upload_checkpoints.py --repo bmarti44/coconut-curriculum-checkpoints --models pause-multipass

    # Dry run (show what would be uploaded)
    python upload_checkpoints.py --repo bmarti44/coconut-curriculum-checkpoints --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
CODE_DIR = REPO_ROOT / "code"

MODELS = {
    "coconut": {
        "source_subdir": "prosqa-coconut",
        "upload_subdir": "coconut",
        "feedback_mode": "continuous",
        "description": "COCONUT (continuous thought recycling) — Meta's original architecture",
    },
    "pause-curriculum": {
        "source_subdir": "prosqa-m5-pause",
        "upload_subdir": "pause-curriculum",
        "feedback_mode": "pause_curriculum",
        "description": "Pause-Curriculum (single learned embedding, single forward pass)",
    },
    "pause-multipass": {
        "source_subdir": "prosqa-m6-pause-multipass",
        "upload_subdir": "pause-multipass",
        "feedback_mode": "pause_multipass",
        "description": "Pause-Multipass (single learned embedding, 6 sequential passes)",
    },
}


def find_best_checkpoint(model_dir):
    """Find the best checkpoint in a model directory.

    Prefers checkpoint_best symlink, then checkpoint_50 (final epoch),
    then the highest-numbered checkpoint.
    """
    model_dir = Path(model_dir)

    # Prefer explicit best
    best = model_dir / "checkpoint_best"
    if best.exists():
        return best

    # Then final epoch
    final = model_dir / "checkpoint_50"
    if final.exists():
        return final

    # Then highest numbered
    checkpoints = sorted(model_dir.glob("checkpoint_*"), key=lambda p: p.name)
    if checkpoints:
        return checkpoints[-1]

    return None


def create_model_card(name, cfg, checkpoint_path):
    """Create a model card for a checkpoint."""
    return f"""---
language: en
tags:
  - coconut
  - latent-reasoning
  - gpt2
  - prosqa
license: mit
---

# {name.upper()}: {cfg['description']}

Model checkpoint from "The Curriculum Is the Mechanism"

## Model Details

- **Base model**: GPT-2 (124M parameters)
- **Feedback mode**: `{cfg['feedback_mode']}`
- **Training data**: ProsQA (17,886 graph-traversal problems)
- **Training**: 50 epochs, 7-stage curriculum, seed=0
- **Checkpoint**: `checkpoint_best` (originally `{checkpoint_path.name}`)

## Usage

```python
from exp_utils import load_model

model, tokenizer, model_info = load_model(
    "path/to/checkpoint_best",
    device="cuda",
    feedback_mode="{cfg['feedback_mode']}",
)
```

## Loading with `load_model_by_name`

```python
from exp_utils import load_model_by_name

model, tokenizer, model_info = load_model_by_name("{name}", "path/to/results/")
```
"""


def _dir_size_mb(path):
    """Total size of a directory in MB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def upload(repo_id, models_to_upload, dry_run=False):
    """Upload specified models to HuggingFace Hub.

    Each checkpoint is a directory (pytorch_model.bin, optimizer state, etc.).
    We upload the entire directory as checkpoint_best/ so load_model_by_name
    finds it regardless of the original epoch number.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    if not dry_run:
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"  Note: {e}")

    for name in models_to_upload:
        cfg = MODELS[name]
        # Source: read from Lambda-era directory names on disk
        model_dir = RESULTS_DIR / cfg["source_subdir"]
        # Destination: upload under clean descriptive names
        hf_subdir = cfg["upload_subdir"]

        if not model_dir.exists():
            print(f"  SKIP {name}: {model_dir} does not exist")
            continue

        ckpt = find_best_checkpoint(model_dir)
        if ckpt is None:
            print(f"  SKIP {name}: no checkpoint found in {model_dir}")
            continue

        # Always upload as checkpoint_best/ so load_model_by_name finds it
        upload_name = "checkpoint_best"
        print(f"\n  Uploading {name}: {ckpt.name}/ -> {hf_subdir}/{upload_name}/")

        # Training config to include
        config_name = {
            "coconut": "prosqa_coconut_1gpu.yaml",
            "pause-curriculum": "prosqa_m5_pause.yaml",
            "pause-multipass": "prosqa_m6_pause_multipass.yaml",
        }.get(name)
        config_path = CODE_DIR / "args" / config_name

        if dry_run:
            if ckpt.is_dir():
                size_mb = _dir_size_mb(ckpt)
                files = list(ckpt.iterdir())
                print(f"    Would upload directory to {repo_id}/{hf_subdir}/{upload_name}/:")
                for f in sorted(files):
                    fsize = f.stat().st_size / (1024 * 1024)
                    print(f"      {f.name} ({fsize:.1f} MB)")
                print(f"      Total: {size_mb:.1f} MB")
            else:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"    Would upload file to {repo_id}/{hf_subdir}/{upload_name}:")
                print(f"      {ckpt.name} ({size_mb:.1f} MB)")
            if config_path.exists():
                print(f"      + {config_name}")
            model_card = create_model_card(name, cfg, ckpt)
            print(f"    Would create README.md ({len(model_card)} chars)")
        else:
            # Upload checkpoint directory (or file) AS checkpoint_best/
            if ckpt.is_dir():
                api.upload_folder(
                    folder_path=str(ckpt),
                    path_in_repo=f"{hf_subdir}/{upload_name}",
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"    Uploaded {ckpt.name}/ -> {hf_subdir}/{upload_name}/")
            else:
                # Single-file checkpoint (legacy format)
                api.upload_file(
                    path_or_fileobj=str(ckpt),
                    path_in_repo=f"{hf_subdir}/{upload_name}",
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"    Uploaded {ckpt.name} as {upload_name}")

            # Upload training config
            if config_path.exists():
                api.upload_file(
                    path_or_fileobj=str(config_path),
                    path_in_repo=f"{hf_subdir}/{config_name}",
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"    Uploaded {config_name}")

            # Upload model card
            model_card = create_model_card(name, cfg, ckpt)
            api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo=f"{hf_subdir}/README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"    Uploaded README.md")

            # Upload config metadata (records original checkpoint name for provenance)
            config_meta = {
                "model_name": name,
                "feedback_mode": cfg["feedback_mode"],
                "original_checkpoint": ckpt.name,
                "uploaded_as": upload_name,
                "base_model": "openai-community/gpt2",
            }
            api.upload_file(
                path_or_fileobj=json.dumps(config_meta, indent=2).encode(),
                path_in_repo=f"{hf_subdir}/config.json",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"    Uploaded config.json")

    print("\n  Done.")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace Hub")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument(
        "--models",
        type=str,
        default="coconut,pause-curriculum,pause-multipass",
        help="Comma-separated model names to upload (default: coconut,pause-curriculum,pause-multipass)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    models_to_upload = args.models.split(",")
    for m in models_to_upload:
        if m not in MODELS:
            print(f"ERROR: Unknown model '{m}'. Expected one of {list(MODELS.keys())}")
            sys.exit(1)

    print("=" * 60)
    print("  Upload Checkpoints to HuggingFace Hub")
    print("=" * 60)
    print(f"  Repo: {args.repo}")
    print(f"  Models: {models_to_upload}")
    print(f"  Dry run: {args.dry_run}")

    upload(args.repo, models_to_upload, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
