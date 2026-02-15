#!/usr/bin/env python3
"""Unified reproduction script for 'The Curriculum Is the Mechanism'

Two modes:
  --from-checkpoints  Download pretrained from HuggingFace, run experiments (~2h on 1 GPU)
  --full              Train from scratch + run experiments (~120 GPU hours)

Usage:
  python reproduce.py --from-checkpoints
  python reproduce.py --full
  python reproduce.py --from-checkpoints --dry-run   # Print steps without executing
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve paths relative to this script
REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DATA = REPO_ROOT / "checkpoints" / "data"
REF_DATA = REPO_ROOT / "reference_repos" / "coconut" / "data"

# HuggingFace Hub repo
HF_REPO = "bmarti44/coconut-curriculum-checkpoints"

# Models to train/evaluate
MODELS = {
    "cot-baseline": {
        "config": "args/prosqa_cot.yaml",
        "subdir": "cot-baseline",
        "feedback_mode": None,
        "train_hours": "~8h on H100",
    },
    "coconut": {
        "config": "args/prosqa_coconut_1gpu.yaml",
        "subdir": "coconut",
        "feedback_mode": "continuous",
        "train_hours": "~28h on H100",
    },
    "pause-curriculum": {
        "config": "args/prosqa_m5_pause.yaml",
        "subdir": "pause-curriculum",
        "feedback_mode": "pause_curriculum",
        "train_hours": "~28h on H100",
    },
    "pause-multipass": {
        "config": "args/prosqa_m6_pause_multipass.yaml",
        "subdir": "pause-multipass",
        "feedback_mode": "pause_multipass",
        "train_hours": "~40h on H100",
    },
}


def run(cmd, dry_run=False, cwd=None):
    """Run a shell command, printing it first."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    print(f"\n  $ {display}")
    if dry_run:
        print("    [dry-run] skipped")
        return True
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd)
    if result.returncode != 0:
        print(f"    [FAILED] exit code {result.returncode}")
        return False
    return True


def step(msg):
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


# -------------------------------------------------------------------------
# Step 1: Install dependencies
# -------------------------------------------------------------------------

def install_deps(dry_run=False):
    step("Step 1: Install dependencies")
    return run(
        [sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")],
        dry_run=dry_run,
    )


# -------------------------------------------------------------------------
# Step 2: Set up data
# -------------------------------------------------------------------------

def setup_data(dry_run=False):
    step("Step 2: Set up data")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    core_files = ["prosqa_train.json", "prosqa_valid.json", "prosqa_test.json"]
    ood_files = ["ood_7hop.json", "ood_8hop.json", "ood_dag.json", "ood_dense.json"]

    # Copy core ProsQA data
    for f in core_files:
        dest = DATA_DIR / f
        if dest.exists():
            print(f"  {f} already exists, skipping")
            continue
        # Try checkpoints/data/ first, then reference_repos/
        src = CKPT_DATA / f
        if not src.exists():
            src = REF_DATA / f
        if not src.exists():
            print(f"  ERROR: {f} not found in checkpoints/data/ or reference_repos/coconut/data/")
            print(f"  Run: git submodule update --init --recursive")
            return False
        if not dry_run:
            shutil.copy2(src, dest)
        print(f"  Copied {f} from {src.parent.name}/")

    # Copy OOD data if available, otherwise generate
    ood_missing = [f for f in ood_files if not (DATA_DIR / f).exists()]
    if ood_missing:
        # Try copying from checkpoints/data/
        copied = False
        for f in ood_missing:
            src = CKPT_DATA / f
            if src.exists():
                if not dry_run:
                    shutil.copy2(src, DATA_DIR / f)
                print(f"  Copied {f} from checkpoints/data/")
                copied = True

        # If any still missing, generate them
        still_missing = [f for f in ood_files if not (DATA_DIR / f).exists()]
        if still_missing and not dry_run:
            print("  Generating OOD test sets...")
            run([sys.executable, "generate_ood_data.py"], dry_run=dry_run, cwd=str(CODE_DIR))
    else:
        print("  OOD test sets already exist, skipping")

    # Verify
    all_files = core_files + ood_files
    for f in all_files:
        p = DATA_DIR / f
        if not dry_run and not p.exists():
            print(f"  MISSING: {p}")
            return False
    print(f"  All {len(all_files)} data files present.")
    return True


# -------------------------------------------------------------------------
# Step 3: Download checkpoints from HuggingFace
# -------------------------------------------------------------------------

def download_checkpoints(dry_run=False):
    step("Step 3: Download pretrained checkpoints from HuggingFace")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  Installing huggingface_hub...")
        run([sys.executable, "-m", "pip", "install", "huggingface_hub"], dry_run=dry_run)
        if not dry_run:
            from huggingface_hub import snapshot_download

    for name, cfg in MODELS.items():
        if name == "cot-baseline":
            continue  # CoT baseline not needed for core experiments
        dest = RESULTS_DIR / cfg["subdir"]
        if dest.exists() and any(dest.iterdir()):
            print(f"  {name} ({cfg['subdir']}) already exists, skipping")
            continue
        print(f"  Downloading {name} ({cfg['subdir']})...")
        if not dry_run:
            try:
                snapshot_download(
                    repo_id=HF_REPO,
                    allow_patterns=[f"{cfg['subdir']}/*"],
                    local_dir=str(RESULTS_DIR),
                )
            except Exception as e:
                print(f"  WARNING: Download failed: {e}")
                print(f"  You may need to train {name} from scratch (--full mode)")
                return False

    print("  Checkpoints downloaded.")
    return True


# -------------------------------------------------------------------------
# Step 4: Train models from scratch
# -------------------------------------------------------------------------

def train_models(dry_run=False):
    step("Step 4: Train all models from scratch")

    for name, cfg in MODELS.items():
        ckpt_dir = RESULTS_DIR / cfg["subdir"]
        if ckpt_dir.exists() and list(ckpt_dir.glob("checkpoint_*")):
            print(f"  {name} checkpoints already exist in {ckpt_dir}, skipping")
            continue
        print(f"\n  Training {name} ({cfg['train_hours']})...")
        ok = run(
            ["torchrun", "--nproc_per_node=1", "run.py", cfg["config"]],
            dry_run=dry_run,
            cwd=str(CODE_DIR),
        )
        if not ok:
            print(f"  ERROR: Training {name} failed")
            return False

    print("\n  All models trained.")
    return True


# -------------------------------------------------------------------------
# Step 5: Run experiments
# -------------------------------------------------------------------------

def run_experiments(dry_run=False):
    step("Step 5: Run experiments")

    exp_dir = RESULTS_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Corruption analysis
    print("\n  Running corruption analysis...")
    run(
        [
            sys.executable, "exp_corruption.py",
            "--checkpoint_dir", str(RESULTS_DIR),
            "--data", str(DATA_DIR / "prosqa_test.json"),
            "--output_dir", str(exp_dir / "corruption"),
            "--num_samples", "500",
        ],
        dry_run=dry_run,
        cwd=str(CODE_DIR),
    )

    # Probing analysis
    print("\n  Running probing analysis...")
    run(
        [
            sys.executable, "exp_probing.py",
            "--checkpoint_dir", str(RESULTS_DIR),
            "--data", str(DATA_DIR / "prosqa_test.json"),
            "--output_dir", str(exp_dir / "probing"),
            "--num_samples", "500",
        ],
        dry_run=dry_run,
        cwd=str(CODE_DIR),
    )

    # OOD generalization
    print("\n  Running OOD generalization tests...")
    run(
        [
            sys.executable, "exp_ood.py",
            "--checkpoint_dir", str(RESULTS_DIR),
            "--data_dir", str(DATA_DIR),
            "--output_dir", str(exp_dir / "ood"),
        ],
        dry_run=dry_run,
        cwd=str(CODE_DIR),
    )

    # Statistical analysis
    print("\n  Running statistical analysis...")
    run(
        [
            sys.executable, "statistical_analysis.py",
            "--results_dirs", str(exp_dir),
            "--output", str(RESULTS_DIR / "statistical_analysis.json"),
        ],
        dry_run=dry_run,
        cwd=str(CODE_DIR),
    )

    return True


# -------------------------------------------------------------------------
# Step 6: Generate figures
# -------------------------------------------------------------------------

def generate_figures(dry_run=False):
    step("Step 6: Generate figures")

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    return run(
        [
            sys.executable, "generate_figures.py",
            "--results_dir", str(RESULTS_DIR),
            "--output_dir", str(figures_dir),
        ],
        dry_run=dry_run,
        cwd=str(CODE_DIR),
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce 'The Curriculum Is the Mechanism'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce.py --from-checkpoints            # ~2h on 1 GPU
  python reproduce.py --full                         # ~120 GPU hours
  python reproduce.py --from-checkpoints --dry-run   # Print steps only
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-checkpoints",
        action="store_true",
        help="Download pretrained checkpoints from HuggingFace, run experiments (~2h on 1 GPU)",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Train from scratch + run experiments (~120 GPU hours)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps without executing",
    )
    args = parser.parse_args()
    dry = args.dry_run

    print("=" * 60)
    print("  REPRODUCE: The Curriculum Is the Mechanism")
    print("=" * 60)
    mode = "from-checkpoints" if args.from_checkpoints else "full"
    print(f"  Mode: {mode}")
    print(f"  Dry run: {dry}")
    print(f"  Repo root: {REPO_ROOT}")

    # Step 1: Install deps
    if not install_deps(dry_run=dry):
        sys.exit(1)

    # Step 2: Set up data
    if not setup_data(dry_run=dry):
        sys.exit(1)

    if args.from_checkpoints:
        # Step 3: Download pretrained checkpoints
        if not download_checkpoints(dry_run=dry):
            sys.exit(1)
    else:
        # Step 4: Train from scratch
        if not train_models(dry_run=dry):
            sys.exit(1)

    # Step 5: Run experiments
    if not run_experiments(dry_run=dry):
        sys.exit(1)

    # Step 6: Generate figures
    if not generate_figures(dry_run=dry):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  REPRODUCTION COMPLETE")
    print("=" * 60)
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {RESULTS_DIR / 'figures'}")
    print(f"  Statistics: {RESULTS_DIR / 'statistical_analysis.json'}")


if __name__ == "__main__":
    main()
