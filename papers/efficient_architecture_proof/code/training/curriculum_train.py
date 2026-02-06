"""
Curriculum Learning Training for LAHR v2

Based on COCONUT's validated multi-stage curriculum:
1. Stage 0: Standard language modeling (warm-up)
2. Stage 1: Standard CoT training (if using reasoning data)
3. Stage 2+: Progressive replacement of CoT with latent thoughts

Reference: arxiv.org/abs/2412.06769 (COCONUT)

Designed for MacBook Pro with 36GB unified memory.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lahr_v2 import (
    LAHRv2, LAHRConfig,
    create_lahr_tiny, create_lahr_small, create_lahr_medium,
    create_baseline_transformer,
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with MacBook-friendly defaults."""
    # Hardware
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: str = "float32"  # MPS doesn't support bf16 well

    # Batch settings (small for MacBook memory)
    batch_size: int = 4
    gradient_accumulation: int = 8  # Effective batch = 32
    max_seq_len: int = 512

    # Learning rate
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Curriculum stages
    n_stages: int = 4
    steps_per_stage: int = 5000
    reset_optimizer_per_stage: bool = False  # COCONUT tests both

    # Evaluation
    eval_interval: int = 500
    save_interval: int = 2000
    log_interval: int = 50

    # Reproducibility
    seed: int = 42


from dataclasses import dataclass


# =============================================================================
# Dataset with Curriculum Support
# =============================================================================

class CurriculumDataset(Dataset):
    """
    Dataset supporting curriculum learning for latent reasoning.

    In COCONUT's curriculum:
    - Stage 0: Standard text
    - Stage k (k>0): First k reasoning steps replaced with latent tokens

    For simplicity, we use synthetic data that simulates reasoning chains.
    In practice, you'd use GSM8K, ProntoQA, or similar datasets.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        seq_len: int = 512,
        stage: int = 0,
        latent_per_step: int = 2,  # COCONUT's 'c' hyperparameter
        bot_token: int = 50255,
        eot_token: int = 50256,
    ):
        self.seq_len = seq_len
        self.stage = stage
        self.latent_per_step = latent_per_step
        self.bot_token = bot_token
        self.eot_token = eot_token

        # Load data or generate synthetic
        if data_path and os.path.exists(data_path):
            import numpy as np
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            # Synthetic data for testing
            print("Using synthetic data (replace with real data for actual training)")
            self.tokens = torch.randint(0, 50254, (5_000_000,)).numpy()

        self.n_samples = len(self.tokens) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        tokens = torch.tensor(self.tokens[start:end].astype(int), dtype=torch.long)

        x = tokens[:-1]
        y = tokens[1:]

        # In curriculum stage k > 0, we would inject latent tokens here
        # For now, we just return standard tokens
        # Full implementation would replace first k reasoning steps with <bot>...<eot>

        return x, y

    def set_stage(self, stage: int):
        """Update curriculum stage."""
        self.stage = stage


# =============================================================================
# Trainer with Curriculum Learning
# =============================================================================

class CurriculumTrainer:
    """
    Trainer implementing COCONUT-style curriculum learning.

    Key features:
    - Multi-stage curriculum
    - Per-stage optimizer reset option
    - Component-specific metrics tracking
    - Ablation support
    """

    def __init__(
        self,
        model: LAHRv2,
        config: TrainingConfig,
        output_dir: str,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.device = config.device
        self.model = self.model.to(self.device)

        # Parameter count
        self.n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {self.n_params:,} parameters")

        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Created per stage

        # Metrics
        self.metrics_history = []
        self.stage_metrics = {}
        self.global_step = 0
        self.current_stage = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW with weight decay groups."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.learning_rate, betas=(0.9, 0.95))

    def _create_scheduler(self, total_steps: int):
        """Create cosine scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        batch: tuple,
        use_latent: bool = True,
        use_memory: bool = True,
    ) -> Dict[str, float]:
        """Single training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()

        # Forward pass
        output = self.model(
            x,
            use_latent=use_latent,
            use_memory=use_memory,
            return_metrics=True,
        )
        logits = output["logits"]

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # Backward
        loss_scaled = loss / self.config.gradient_accumulation
        loss_scaled.backward()

        metrics = {
            "loss": loss.item(),
            "perplexity": math.exp(min(loss.item(), 10)),
        }

        # Model-specific metrics
        if "metrics" in output and output["metrics"]:
            m = output["metrics"]
            if m.get("mod_routing"):
                metrics["mod_active"] = sum(m["mod_routing"]) / len(m["mod_routing"])
            if m.get("memory_usage") is not None:
                metrics["memory_usage"] = m["memory_usage"]
            if m.get("latent_iterations") is not None:
                metrics["latent_iters"] = m["latent_iterations"]

        return metrics

    def optimizer_step(self):
        """Gradient clipping and optimizer step."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
        )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: int = 50,
        use_latent: bool = True,
        use_memory: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x, use_latent=use_latent, use_memory=use_memory)
            logits = output["logits"]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += y.numel()

        return {
            "val_loss": total_loss / total_tokens,
            "val_perplexity": math.exp(min(total_loss / total_tokens, 10)),
        }

    def train_stage(
        self,
        stage: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        use_latent: bool = True,
        use_memory: bool = True,
    ):
        """Train one curriculum stage."""
        self.current_stage = stage
        steps = self.config.steps_per_stage

        # Reset optimizer if configured
        if self.config.reset_optimizer_per_stage and stage > 0:
            self.optimizer = self._create_optimizer()

        # Create scheduler for this stage
        self.scheduler = self._create_scheduler(steps)

        print(f"\n{'='*50}")
        print(f"Stage {stage}: {steps} steps")
        print(f"  use_latent={use_latent}, use_memory={use_memory}")
        print(f"{'='*50}\n")

        # Training loop
        data_iter = iter(train_loader)
        running_metrics = {}
        start_time = time.time()
        tokens_processed = 0

        pbar = tqdm(range(steps), desc=f"Stage {stage}")

        for step in pbar:
            # Gradient accumulation
            for _ in range(self.config.gradient_accumulation):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                metrics = self.train_step(batch, use_latent=use_latent, use_memory=use_memory)

                for k, v in metrics.items():
                    running_metrics[k] = running_metrics.get(k, 0) + v

                tokens_processed += batch[0].numel()

            # Optimizer step
            self.optimizer_step()
            self.global_step += 1

            # Logging
            if step % self.config.log_interval == 0 and step > 0:
                n = self.config.log_interval * self.config.gradient_accumulation
                avg = {k: v / n for k, v in running_metrics.items()}

                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed

                pbar.set_postfix({
                    "loss": f"{avg['loss']:.4f}",
                    "ppl": f"{avg['perplexity']:.1f}",
                    "tok/s": f"{tok_per_sec:.0f}",
                })

                self.metrics_history.append({
                    "stage": stage,
                    "step": self.global_step,
                    "local_step": step,
                    **avg,
                })

                running_metrics = {}

            # Evaluation
            if val_loader and step % self.config.eval_interval == 0 and step > 0:
                val_metrics = self.evaluate(val_loader, use_latent=use_latent, use_memory=use_memory)
                print(f"\n  Step {step}: val_loss={val_metrics['val_loss']:.4f}, val_ppl={val_metrics['val_perplexity']:.1f}")

            # Save checkpoint
            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(f"stage{stage}_step{step}.pt")

        # Save stage checkpoint
        self.save_checkpoint(f"stage{stage}_final.pt")

        # Log stage summary
        self.stage_metrics[stage] = {
            "final_loss": self.metrics_history[-1]["loss"] if self.metrics_history else None,
            "steps": steps,
            "time": time.time() - start_time,
        }

    def train_curriculum(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Full curriculum training.

        Stage progression:
        - Stage 0: Full model (baseline metrics)
        - Stage 1: Latent reasoning only (test component)
        - Stage 2: Memory only (test component)
        - Stage 3: Full model (final training)
        """
        print("\n" + "="*60)
        print("CURRICULUM TRAINING")
        print("="*60)
        print(f"Model: {self.n_params:,} parameters")
        print(f"Device: {self.device}")
        print(f"Stages: {self.config.n_stages}")
        print(f"Steps per stage: {self.config.steps_per_stage}")
        print("="*60 + "\n")

        # Stage 0: Warm-up with full model
        self.train_stage(0, train_loader, val_loader, use_latent=True, use_memory=True)

        # Stage 1: Latent reasoning focus (reduce memory use)
        self.train_stage(1, train_loader, val_loader, use_latent=True, use_memory=False)

        # Stage 2: Memory focus (reduce latent use)
        self.train_stage(2, train_loader, val_loader, use_latent=False, use_memory=True)

        # Stage 3: Final full training
        self.train_stage(3, train_loader, val_loader, use_latent=True, use_memory=True)

        # Final save
        self.save_checkpoint("final.pt")
        self.save_metrics()

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        for stage, metrics in self.stage_metrics.items():
            print(f"Stage {stage}: loss={metrics.get('final_loss', 'N/A'):.4f}, time={metrics.get('time', 0)/60:.1f}min")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "stage_metrics": self.stage_metrics,
        }, path)
        print(f"  Saved: {path}")

    def save_metrics(self):
        """Save training metrics."""
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump({
                "n_params": self.n_params,
                "config": vars(self.config),
                "stage_metrics": self.stage_metrics,
                "history": self.metrics_history,
            }, f, indent=2, default=str)
        print(f"  Metrics saved: {path}")


# =============================================================================
# Ablation Runner
# =============================================================================

def run_ablation(
    output_dir: str,
    size: str = "tiny",
    config: Optional[TrainingConfig] = None,
):
    """
    Run ablation study comparing variants.

    Variants:
    1. Full LAHR (all components)
    2. No latent (MoD + Memory only)
    3. No memory (MoD + Latent only)
    4. No MoD (Latent + Memory only)
    5. Baseline (no innovations)
    """
    if config is None:
        config = TrainingConfig()
        config.steps_per_stage = 1000  # Faster for ablation

    variants = {
        "full": lambda: create_lahr_tiny() if size == "tiny" else create_lahr_small(),
        "no_latent": lambda: create_lahr_no_latent(size),
        "no_memory": lambda: create_lahr_no_memory(size),
        "no_mod": lambda: create_lahr_no_mod(size),
        "baseline": lambda: create_baseline_transformer(size),
    }

    results = {}

    for name, create_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {name}")
        print(f"{'='*60}")

        model = create_fn()
        variant_dir = f"{output_dir}/{name}"

        trainer = CurriculumTrainer(model, config, variant_dir)

        # Create simple data loader
        dataset = CurriculumDataset(seq_len=config.max_seq_len)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # Train (single stage for ablation)
        trainer.train_stage(0, loader)

        results[name] = {
            "n_params": trainer.n_params,
            "final_loss": trainer.metrics_history[-1]["loss"] if trainer.metrics_history else None,
        }

    # Save ablation results
    with open(f"{output_dir}/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    for name, r in results.items():
        print(f"{name:15s}: params={r['n_params']:,}, loss={r.get('final_loss', 'N/A')}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LAHR Curriculum Training")
    parser.add_argument("--size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--output", type=str, default="checkpoints/lahr")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--steps_per_stage", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Config
    config = TrainingConfig()
    config.steps_per_stage = args.steps_per_stage

    if args.ablation:
        run_ablation(args.output, size=args.size, config=config)
    else:
        # Create model
        model_fns = {
            "tiny": create_lahr_tiny,
            "small": create_lahr_small,
            "medium": create_lahr_medium,
        }
        model = model_fns[args.size]()

        # Create data
        dataset = CurriculumDataset(data_path=args.data, seq_len=config.max_seq_len)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # Train
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output}_{args.size}_{timestamp}"

        trainer = CurriculumTrainer(model, config, output_dir)
        trainer.train_curriculum(train_loader)


if __name__ == "__main__":
    main()
