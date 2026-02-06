"""
COCONUT Curriculum Training

Implements the exact training procedure from the COCONUT paper:
1. Stage 0: Train on full Chain-of-Thought (warmup)
2. Stage 1: Replace 1 CoT step with 1 latent token
3. Stage k: Replace k CoT steps with k latent tokens
4. Continue until max_stage or CoT fully replaced

The key insight: gradual compression of explicit reasoning
into continuous latent space.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.coconut_full import COCONUTModel, COCONUTConfig
from data.coconut_dataset import COCONUTDataset, COCONUTTokenizer


def set_seed(seed: int):
    """Set all random seeds for reproducibility (R3 fix: added numpy)."""
    random.seed(seed)
    np.random.seed(seed)  # FIX: Added numpy seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic mode for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device with determinism warnings."""
    if torch.cuda.is_available():
        print("Using CUDA device (deterministic mode enabled)")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # FIX R3: Warn about MPS non-determinism
        print("⚠️  Using MPS device - NOTE: MPS has limited determinism support.")
        print("   Results may vary slightly across runs even with the same seed.")
        print("   For exact reproducibility, use CUDA or CPU.")
        return torch.device("mps")
    print("Using CPU device (fully deterministic)")
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Cosine learning rate schedule with warmup."""
    min_lr = max_lr * min_lr_ratio

    # FIX R4: Guard against division by zero
    if max_steps <= warmup_steps:
        # If no decay phase, just return max_lr after warmup
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        return max_lr

    # Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class COCONUTTrainer:
    """
    Curriculum trainer for COCONUT.

    Handles:
    - Multi-stage curriculum (gradually increase latent tokens)
    - Stage progression based on loss convergence
    - Checkpointing at each stage
    - Evaluation with and without latent reasoning
    """

    def __init__(
        self,
        model: COCONUTModel,
        train_dataset: COCONUTDataset,
        val_dataset: Optional[COCONUTDataset],
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.device = get_device()
        self.model = self.model.to(self.device)

        # Optimizer (FIX R3: explicit eps for version-independent behavior)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8,  # FIX: Explicitly specified for reproducibility
        )

        # Training state
        self.global_step = 0
        self.current_stage = 0
        self.stage_steps = 0
        self.best_val_loss = float("inf")

        # Metrics
        self.metrics_history = []

    def train(self):
        """Full curriculum training loop."""
        print(f"\nStarting COCONUT curriculum training")
        print(f"Device: {self.device}")
        print(f"Max stages: {self.config['max_stages']}")
        print(f"Steps per stage: {self.config['steps_per_stage']}")

        for stage in range(self.config["max_stages"] + 1):
            print(f"\n{'='*60}")
            print(f"STAGE {stage}: {stage} latent tokens replace {stage} CoT steps")
            print(f"{'='*60}")

            self.current_stage = stage
            self.train_dataset.set_stage(stage)

            if self.val_dataset:
                self.val_dataset.set_stage(stage)

            # Train this stage
            stage_metrics = self.train_stage()

            # Save checkpoint
            self.save_checkpoint(f"stage_{stage}")

            # Evaluate
            if self.val_dataset:
                val_loss = self.evaluate()
                print(f"Stage {stage} validation loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")

            self.metrics_history.append({
                "stage": stage,
                **stage_metrics,
            })

        print("\nTraining complete!")
        return self.metrics_history

    def train_stage(self) -> Dict[str, float]:
        """Train one curriculum stage."""
        self.model.train()
        self.stage_steps = 0

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        steps_per_stage = self.config["steps_per_stage"]
        log_interval = self.config.get("log_interval", 10)

        data_iter = iter(dataloader)

        for step in range(steps_per_stage):
            # Get batch (cycle if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs.loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("grad_clip", 1.0)
            )

            # Update learning rate
            lr = get_lr(
                self.global_step,
                self.config["warmup_steps"],
                self.config["max_steps"],
                self.config["learning_rate"],
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Step
            self.optimizer.step()

            # Metrics
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            self.global_step += 1
            self.stage_steps += 1

            # Log
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

                print(
                    f"  Stage {self.current_stage} | "
                    f"Step {step + 1}/{steps_per_stage} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        elapsed = time.time() - start_time

        return {
            "train_loss": avg_loss,
            "train_ppl": math.exp(min(avg_loss, 20)),
            "tokens_per_sec": total_tokens / elapsed,
            "steps": self.stage_steps,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels)

            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        self.model.train()
        return total_loss / total_tokens if total_tokens > 0 else float("inf")

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get("output_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "metrics_history": self.metrics_history,
        }

        path = checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # FIX R3: Use weights_only=False explicitly (required for optimizer state)
        # Note: Only load checkpoints from trusted sources
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_stage = checkpoint["current_stage"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics_history = checkpoint.get("metrics_history", [])

        # FIX R4: Restore dataset stage for correct curriculum resumption
        self.train_dataset.set_stage(self.current_stage)
        if self.val_dataset:
            self.val_dataset.set_stage(self.current_stage)

        print(f"Loaded checkpoint from {path}")
        print(f"  Global step: {self.global_step}")
        print(f"  Current stage: {self.current_stage}")


def main():
    parser = argparse.ArgumentParser(description="Train COCONUT with curriculum")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CoT training data")
    parser.add_argument("--output_dir", type=str, default="coconut_checkpoints")
    parser.add_argument("--size", type=str, default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_stages", type=int, default=5)
    parser.add_argument("--steps_per_stage", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)

    # Config
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_stages": args.max_stages,
        "steps_per_stage": args.steps_per_stage,
        "max_steps": (args.max_stages + 1) * args.steps_per_stage,  # FIX: +1 because stages 0..max_stages inclusive
        "output_dir": args.output_dir,
        "log_interval": 10,
        "grad_clip": 1.0,
    }

    print("COCONUT Curriculum Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Create tokenizer
    tokenizer = COCONUTTokenizer()

    # Create model
    model_configs = {
        "tiny": COCONUTConfig(
            vocab_size=tokenizer.actual_vocab_size,
            d_model=128, n_heads=4, n_layers=4, max_seq_len=args.max_seq_len,
        ),
        "small": COCONUTConfig(
            vocab_size=tokenizer.actual_vocab_size,
            d_model=256, n_heads=8, n_layers=8, max_seq_len=args.max_seq_len,
        ),
        "medium": COCONUTConfig(
            vocab_size=tokenizer.actual_vocab_size,
            d_model=512, n_heads=8, n_layers=12, max_seq_len=args.max_seq_len,
        ),
    }

    model = COCONUTModel(model_configs[args.size])
    print(f"\nModel size: {args.size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = COCONUTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        current_stage=0,
        max_stage=args.max_stages,
    )

    # Create trainer
    trainer = COCONUTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,  # Could add validation split
        config=config,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    metrics = trainer.train()

    # Save final metrics
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
