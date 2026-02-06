"""
Training script for SSH architecture study.

Designed to run on MacBook with Apple Silicon (MPS backend).

Usage:
    python train.py --model ssh --size small --data openwebtext
    python train.py --model transformer --size small --data openwebtext
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    create_ssh_small,
    create_ssh_medium,
    create_transformer_small,
    create_transformer_medium,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "seed": 42,
    "seq_len": 1024,
    "batch_size": 4,  # Small for MacBook
    "gradient_accumulation": 8,  # Effective batch = 32
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 1000,
    "max_steps": 50000,  # ~10B tokens
    "eval_interval": 1000,
    "save_interval": 5000,
    "log_interval": 100,
    "grad_clip": 1.0,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "dtype": "float32",  # MPS doesn't support bf16 well yet
    "compile": False,  # torch.compile not fully supported on MPS
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
        tokenizer_name: str = "gpt2",
    ):
        self.seq_len = seq_len

        # Load pre-tokenized data or tokenize on the fly
        if os.path.exists(data_path):
            # Load pre-tokenized numpy array
            import numpy as np
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            # Create synthetic data for testing
            print(f"Data not found at {data_path}, using synthetic data")
            self.tokens = torch.randint(0, 50257, (10_000_000,)).numpy()

        self.n_samples = len(self.tokens) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target

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
        num_workers=0,  # MPS works better with 0 workers
        pin_memory=False,
    )


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Training manager for SSH/Transformer models."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: str,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.device = config["device"]
        self.model = self.model.to(self.device)

        # Count parameters
        self.n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {self.n_params:,}")

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Gradient scaler (for mixed precision)
        self.scaler = GradScaler() if config["dtype"] == "float16" else None

        # Metrics tracking
        self.metrics_history = []
        self.step = 0
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate weight decay for different param groups
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

            if step < warmup:
                return step / warmup
            else:
                progress = (step - warmup) / (max_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        self.model.train()
        outputs = self.model(x, return_metrics=True)
        logits = outputs["logits"]

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # Backward pass
        loss_scaled = loss / self.config["gradient_accumulation"]
        loss_scaled.backward()

        metrics = {
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 10)),
        }

        # Add model-specific metrics
        if "metrics" in outputs:
            model_metrics = outputs["metrics"]
            if "avg_layers" in model_metrics:
                metrics["avg_layers"] = model_metrics["avg_layers"]
            if "attention_usage" in model_metrics:
                metrics["attention_usage"] = model_metrics["attention_usage"]

        return metrics

    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["grad_clip"],
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        total_layers = 0
        total_attn = 0
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

            if "metrics" in outputs:
                total_layers += outputs["metrics"].get("avg_layers", 12)
                total_attn += outputs["metrics"].get("attention_usage", 1.0)

        avg_loss = total_loss / total_tokens
        return {
            "val_loss": avg_loss,
            "val_perplexity": math.exp(min(avg_loss, 10)),
            "avg_layers": total_layers / n_batches if n_batches > 0 else 0,
            "attention_usage": total_attn / n_batches if n_batches > 0 else 0,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"Training on {self.device}")
        print(f"Batch size: {self.config['batch_size']} x {self.config['gradient_accumulation']} = {self.config['batch_size'] * self.config['gradient_accumulation']}")

        # Training state
        running_loss = 0
        running_metrics = {}
        start_time = time.time()
        tokens_processed = 0

        # Data iterator
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
                running_loss += metrics["loss"]

                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v

                tokens_processed += batch[0].numel()

            # Optimizer step
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
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                })

                # Save metrics
                self.metrics_history.append({
                    "step": step,
                    "elapsed": elapsed,
                    "tokens": tokens_processed,
                    **avg_metrics,
                })

                running_loss = 0
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

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"Saved checkpoint to {path}")

    def save_metrics(self):
        """Save training metrics."""
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump({
                "config": self.config,
                "n_params": self.n_params,
                "history": self.metrics_history,
            }, f, indent=2)
        print(f"Saved metrics to {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SSH/Transformer models")
    parser.add_argument("--model", type=str, choices=["ssh", "transformer"], default="ssh")
    parser.add_argument("--size", type=str, choices=["small", "medium"], default="small")
    parser.add_argument("--data", type=str, default="data/train.bin")
    parser.add_argument("--val_data", type=str, default="data/val.bin")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Config
    config = DEFAULT_CONFIG.copy()
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.batch_size:
        config["batch_size"] = args.batch_size

    # Create model
    if args.model == "ssh":
        if args.size == "small":
            model = create_ssh_small()
        else:
            model = create_ssh_medium()
    else:
        if args.size == "small":
            model = create_transformer_small()
        else:
            model = create_transformer_medium()

    print(f"\nModel: {args.model}-{args.size}")
    print(f"Device: {config['device']}")

    # Create dataloaders
    train_loader = create_dataloader(
        args.data,
        config["seq_len"],
        config["batch_size"],
        shuffle=True,
    )

    val_loader = None
    if os.path.exists(args.val_data):
        val_loader = create_dataloader(
            args.val_data,
            config["seq_len"],
            config["batch_size"],
            shuffle=False,
        )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}/{args.model}_{args.size}_{timestamp}"

    # Train
    trainer = Trainer(model, config, output_dir)
    trainer.train(train_loader, val_loader)

    # Print final FLOP counts
    flops = model.count_flops(config["seq_len"])
    print(f"\nFLOPs per forward pass (seq_len={config['seq_len']}):")
    for k, v in flops.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
