"""
Training script for LAHR architecture experiments.

Designed to run on MacBook with Apple Silicon (MPS backend).
Supports full 2^3 factorial ablation study.

Usage:
    # Train full LAHR
    python train_lahr.py --condition full --size small

    # Run ablation condition
    python train_lahr.py --condition no_latent --size small --seed 42

    # Run full factorial study (all 8 conditions, 5 seeds each)
    python train_lahr.py --ablation_study --size tiny --max_steps 5000
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int):
    """Set all random seeds for reproducibility (R9 fix)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: MPS does not have a deterministic mode

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    create_ablation,
    ABLATION_FACTORY,
    LAHRv4,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "seed": 42,
    "seq_len": 512,
    "batch_size": 4,  # Small for MacBook
    "gradient_accumulation": 8,  # Effective batch = 32
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 1000,
    "max_steps": 50000,
    "eval_interval": 1000,
    "save_interval": 5000,
    "log_interval": 100,
    "grad_clip": 1.0,
    # Device selection: CUDA > MPS > CPU (R9 fix)
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "dtype": "float32",  # MPS doesn't support bf16 well yet
    "aux_loss_weight": 0.01,  # Weight for MoD auxiliary loss
    "min_lr_ratio": 0.1,  # Minimum LR as fraction of peak (R9 fix)
}


# ============================================================================
# Data Loading
# ============================================================================

class TextDataset(Dataset):
    """Simple text dataset from tokenized files."""

    def __init__(
        self,
        data_path: str,
        seq_len: int,
    ):
        self.seq_len = seq_len

        # Load pre-tokenized data or create synthetic
        if os.path.exists(data_path):
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            print(f"WARNING: Data not found at {data_path}")
            print("Run prepare_data.py first for real experiments!")
            print("Using synthetic data - results will be MEANINGLESS")
            self.tokens = np.random.randint(0, 50257, (10_000_000,), dtype=np.uint16)

        # Fix off-by-one: need seq_len + 1 tokens for each sample (R9 fix)
        self.n_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = torch.tensor(self.tokens[start:end].astype(int), dtype=torch.long)

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


def create_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader for training."""
    dataset = TextDataset(data_path, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# ============================================================================
# Training Loop
# ============================================================================

class LAHRTrainer:
    """Training manager for LAHR models."""

    def __init__(
        self,
        model: LAHRv4,
        config: Dict[str, Any],
        output_dir: str,
        condition: str = "full",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.condition = condition

        # Move model to device
        self.device = config["device"]
        self.model = self.model.to(self.device)

        # Count parameters
        self.n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {self.n_params:,}")

        # Log model configuration
        self._log_model_config()

        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics tracking
        self.metrics_history = []
        self.step = 0
        self.best_val_loss = float("inf")

    def _log_model_config(self):
        """Log model configuration for reproducibility."""
        config_log = {
            "condition": self.condition,
            "n_params": self.n_params,
            "model_config": {
                "d_model": self.model.config.d_model,
                "n_layers": self.model.config.n_layers,
                "n_heads": self.model.config.n_heads,
                "use_latent_reasoning": self.model.config.use_latent_reasoning,
                "use_memory": self.model.config.use_memory,
                "mod_capacity": self.model.config.mod_capacity,
                "mod_every_n": self.model.config.mod_every_n,
                "n_latent_iterations": self.model.config.n_latent_iterations,
                "n_memory_slots": self.model.config.n_memory_slots,
            },
            "training_config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_log, f, indent=2)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(step):
            warmup = self.config["warmup_steps"]
            max_steps = self.config["max_steps"]
            min_lr_ratio = self.config.get("min_lr_ratio", 0.1)

            if step < warmup:
                # Fix: start at 1/warmup, not 0 (R9 fix)
                return (step + 1) / warmup
            else:
                progress = (step - warmup) / (max_steps - warmup)
                # Decay to min_lr_ratio, not 0 (R9 fix)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()
        outputs = self.model(x, return_metrics=True)
        logits = outputs["logits"]

        # Main loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # Auxiliary loss (MoD router load balancing)
        aux_loss = outputs.get("aux_loss", 0.0)
        if isinstance(aux_loss, torch.Tensor):
            total_loss = ce_loss + self.config["aux_loss_weight"] * aux_loss
        else:
            total_loss = ce_loss

        # Backward
        loss_scaled = total_loss / self.config["gradient_accumulation"]
        loss_scaled.backward()

        metrics = {
            "loss": ce_loss.item(),
            "perplexity": math.exp(min(ce_loss.item(), 10)),
        }

        if isinstance(aux_loss, torch.Tensor):
            metrics["aux_loss"] = aux_loss.item()

        # Model-specific metrics
        if "metrics" in outputs:
            model_metrics = outputs["metrics"]
            if model_metrics.get("mod_efficiency"):
                metrics["mod_efficiency"] = sum(model_metrics["mod_efficiency"]) / len(model_metrics["mod_efficiency"])
            if "latent_iterations" in model_metrics:
                metrics["latent_iterations"] = model_metrics["latent_iterations"]

        return metrics

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["grad_clip"],
        )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        n_batches = 0

        for batch in dataloader:
            if n_batches >= max_batches:
                break

            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            outputs = self.model(x, return_metrics=True)
            logits = outputs["logits"]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += y.numel()
            n_batches += 1

        avg_loss = total_loss / total_tokens
        return {
            "val_loss": avg_loss,
            "val_perplexity": math.exp(min(avg_loss, 10)),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"\nTraining LAHR ({self.condition}) on {self.device}")
        print(f"Batch size: {self.config['batch_size']} x {self.config['gradient_accumulation']} = {self.config['batch_size'] * self.config['gradient_accumulation']}")

        running_metrics = {}
        start_time = time.time()
        tokens_processed = 0

        data_iter = iter(train_loader)
        pbar = tqdm(range(self.config["max_steps"]), desc="Training")

        for step in pbar:
            self.step = step

            # Gradient accumulation
            for _ in range(self.config["gradient_accumulation"]):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v

                tokens_processed += batch[0].numel()

            self.optimizer_step()

            # Logging
            if step % self.config["log_interval"] == 0 and step > 0:
                n = self.config["log_interval"] * self.config["gradient_accumulation"]
                avg_metrics = {k: v / n for k, v in running_metrics.items()}

                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed

                pbar.set_postfix({
                    "loss": f"{avg_metrics['loss']:.4f}",
                    "ppl": f"{avg_metrics['perplexity']:.2f}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                })

                self.metrics_history.append({
                    "step": step,
                    "elapsed": elapsed,
                    "tokens": tokens_processed,
                    **avg_metrics,
                })

                running_metrics = {}

            # Evaluation
            if val_loader and step % self.config["eval_interval"] == 0 and step > 0:
                val_metrics = self.evaluate(val_loader)
                print(f"\nStep {step}: val_loss={val_metrics['val_loss']:.4f}, val_ppl={val_metrics['val_perplexity']:.2f}")

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best.pt")

            # Save checkpoint
            if step % self.config["save_interval"] == 0 and step > 0:
                self.save_checkpoint(f"step_{step}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        self.save_metrics()

        return self.best_val_loss

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "condition": self.condition,
            "best_val_loss": self.best_val_loss,
        }, path)

    def save_metrics(self):
        """Save training metrics."""
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump({
                "condition": self.condition,
                "n_params": self.n_params,
                "best_val_loss": self.best_val_loss,
                "history": self.metrics_history,
            }, f, indent=2)


# ============================================================================
# Ablation Study Runner
# ============================================================================

def run_ablation_study(
    size: str,
    data_path: str,
    val_data_path: str,
    output_base: str,
    seeds: List[int],
    max_steps: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run full 2^3 factorial ablation study.

    Args:
        size: Model size ("tiny", "small", "medium")
        seeds: List of random seeds to use
        ...

    Returns:
        Results dictionary with all conditions and seeds
    """
    conditions = list(ABLATION_FACTORY.keys())
    results = {}

    print(f"\n{'='*60}")
    print(f"LAHR Ablation Study")
    print(f"Conditions: {len(conditions)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total runs: {len(conditions) * len(seeds)}")
    print(f"{'='*60}\n")

    for condition in conditions:
        results[condition] = {}

        for seed in seeds:
            print(f"\n>>> Running: {condition} (seed={seed})")

            # Set all seeds for reproducibility (R9 fix)
            set_seed(seed)

            # Create model
            model = create_ablation(condition, size)

            # Output directory
            output_dir = f"{output_base}/{condition}_seed{seed}"

            # Update config with seed
            run_config = config.copy()
            run_config["seed"] = seed
            run_config["max_steps"] = max_steps

            # Create dataloaders
            train_loader = create_dataloader(
                data_path,
                run_config["seq_len"],
                run_config["batch_size"],
            )

            val_loader = None
            if os.path.exists(val_data_path):
                val_loader = create_dataloader(
                    val_data_path,
                    run_config["seq_len"],
                    run_config["batch_size"],
                    shuffle=False,
                )

            # Train
            trainer = LAHRTrainer(model, run_config, output_dir, condition)
            best_val_loss = trainer.train(train_loader, val_loader)

            results[condition][seed] = {
                "best_val_loss": best_val_loss,
                "n_params": trainer.n_params,
            }

    # Save aggregated results
    with open(f"{output_base}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LAHR models")
    parser.add_argument("--condition", type=str, choices=list(ABLATION_FACTORY.keys()), default="full")
    parser.add_argument("--size", type=str, choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--data", type=str, default="data/train.bin")
    parser.add_argument("--val_data", type=str, default="data/val.bin")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Ablation study mode
    parser.add_argument("--ablation_study", action="store_true", help="Run full 2^3 factorial")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1000])

    args = parser.parse_args()

    # Config
    config = DEFAULT_CONFIG.copy()
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.batch_size:
        config["batch_size"] = args.batch_size

    if args.ablation_study:
        # Run full factorial study
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{args.output}/ablation_{args.size}_{timestamp}"

        run_ablation_study(
            size=args.size,
            data_path=args.data,
            val_data_path=args.val_data,
            output_base=output_base,
            seeds=args.seeds,
            max_steps=config["max_steps"],
            config=config,
        )
    else:
        # Single run
        set_seed(args.seed)  # Use comprehensive seed setting (R9 fix)

        model = create_ablation(args.condition, args.size)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output}/lahr_{args.condition}_{args.size}_{timestamp}"

        train_loader = create_dataloader(
            args.data,
            config["seq_len"],
            config["batch_size"],
        )

        val_loader = None
        if os.path.exists(args.val_data):
            val_loader = create_dataloader(
                args.val_data,
                config["seq_len"],
                config["batch_size"],
                shuffle=False,
            )

        trainer = LAHRTrainer(model, config, output_dir, args.condition)

        # Resume from checkpoint if specified (R9 fix)
        if args.resume and os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            trainer.step = checkpoint["step"]
            trainer.best_val_loss = checkpoint["best_val_loss"]
            print(f"Resumed from step {trainer.step}")

        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
