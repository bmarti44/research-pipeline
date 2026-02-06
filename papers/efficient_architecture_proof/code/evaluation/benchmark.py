"""
Benchmarking script for SSH vs Transformer comparison.

Measures:
- Perplexity on standard benchmarks
- Inference throughput (tokens/second)
- FLOPs per token
- Memory usage
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    create_ssh_small,
    create_ssh_medium,
    create_transformer_small,
    create_transformer_medium,
)


# ============================================================================
# Benchmarking Functions
# ============================================================================

def measure_perplexity(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int = 4,
    device: str = "mps",
) -> Dict[str, float]:
    """
    Measure perplexity on a dataset.

    Args:
        model: Language model
        data: Token tensor (1D)
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        dict with perplexity and loss
    """
    model.eval()
    model = model.to(device)

    seq_len = 1024
    total_loss = 0
    total_tokens = 0

    # Process in chunks
    n_chunks = len(data) // (seq_len + 1)

    with torch.no_grad():
        for i in tqdm(range(0, n_chunks, batch_size), desc="Perplexity"):
            # Prepare batch
            batch_x = []
            batch_y = []

            for j in range(batch_size):
                if i + j >= n_chunks:
                    break
                start = (i + j) * (seq_len + 1)
                chunk = data[start:start + seq_len + 1]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])

            if not batch_x:
                break

            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)

            # Forward
            outputs = model(x)
            logits = outputs["logits"]

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_evaluated": total_tokens,
    }


def measure_throughput(
    model: torch.nn.Module,
    seq_len: int = 1024,
    batch_sizes: List[int] = [1, 4, 8, 16],
    n_iterations: int = 50,
    warmup: int = 10,
    device: str = "mps",
) -> Dict[int, Dict[str, float]]:
    """
    Measure inference throughput at different batch sizes.

    Returns tokens per second for each batch size.
    """
    model.eval()
    model = model.to(device)

    results = {}

    for batch_size in batch_sizes:
        # Create dummy input
        x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)

        # Synchronize
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # Time iterations
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(n_iterations):
                _ = model(x)

        # Synchronize
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        tokens_processed = batch_size * seq_len * n_iterations
        tokens_per_second = tokens_processed / elapsed

        results[batch_size] = {
            "tokens_per_second": tokens_per_second,
            "ms_per_batch": (elapsed / n_iterations) * 1000,
            "ms_per_token": (elapsed / tokens_processed) * 1000,
        }

        print(f"Batch {batch_size}: {tokens_per_second:.0f} tok/s, {results[batch_size]['ms_per_batch']:.1f} ms/batch")

    return results


def measure_memory(
    model: torch.nn.Module,
    seq_len: int = 1024,
    batch_size: int = 1,
    device: str = "mps",
) -> Dict[str, float]:
    """
    Measure peak memory usage during inference.
    """
    model = model.to(device)

    # Clear cache
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Get memory stats
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3
    else:
        # MPS doesn't have detailed memory tracking
        # Estimate from model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        peak_memory = param_memory * 2  # Rough estimate including activations
        current_memory = param_memory

    return {
        "peak_memory_gb": peak_memory,
        "current_memory_gb": current_memory,
    }


def measure_adaptive_behavior(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int = 4,
    device: str = "mps",
) -> Dict[str, Any]:
    """
    Measure adaptive computation behavior (for SSH model).

    Analyzes:
    - Average layers used per token
    - Attention gate activation rates
    - Variation in computation across different text types
    """
    model.eval()
    model = model.to(device)

    seq_len = 1024
    all_layers = []
    all_attention = []

    n_chunks = min(100, len(data) // (seq_len + 1))

    with torch.no_grad():
        for i in tqdm(range(0, n_chunks, batch_size), desc="Adaptive analysis"):
            batch_x = []

            for j in range(batch_size):
                if i + j >= n_chunks:
                    break
                start = (i + j) * (seq_len + 1)
                chunk = data[start:start + seq_len]
                batch_x.append(chunk)

            if not batch_x:
                break

            x = torch.stack(batch_x).to(device)

            outputs = model(x, return_metrics=True)

            if "metrics" in outputs:
                metrics = outputs["metrics"]
                if "avg_layers" in metrics:
                    all_layers.append(metrics["avg_layers"])
                if "attention_usage" in metrics:
                    all_attention.append(metrics["attention_usage"])

    results = {}

    if all_layers:
        results["avg_layers_used"] = sum(all_layers) / len(all_layers)
        results["layers_std"] = (
            sum((x - results["avg_layers_used"])**2 for x in all_layers) / len(all_layers)
        ) ** 0.5

    if all_attention:
        results["avg_attention_usage"] = sum(all_attention) / len(all_attention)
        results["attention_std"] = (
            sum((x - results["avg_attention_usage"])**2 for x in all_attention) / len(all_attention)
        ) ** 0.5

    return results


# ============================================================================
# Comparison Runner
# ============================================================================

def run_comparison(
    ssh_checkpoint: str = None,
    transformer_checkpoint: str = None,
    size: str = "small",
    eval_data_path: str = "data/test.bin",
    output_path: str = "benchmark_results.json",
    device: str = "mps",
):
    """
    Run full comparison between SSH and Transformer.
    """
    print(f"Running comparison on {device}")

    # Create models
    print("\nCreating models...")
    if size == "small":
        ssh_model = create_ssh_small()
        transformer_model = create_transformer_small()
    else:
        ssh_model = create_ssh_medium()
        transformer_model = create_transformer_medium()

    # Load checkpoints if provided
    if ssh_checkpoint:
        state = torch.load(ssh_checkpoint, map_location="cpu")
        ssh_model.load_state_dict(state["model_state_dict"])
        print(f"Loaded SSH checkpoint from {ssh_checkpoint}")

    if transformer_checkpoint:
        state = torch.load(transformer_checkpoint, map_location="cpu")
        transformer_model.load_state_dict(state["model_state_dict"])
        print(f"Loaded Transformer checkpoint from {transformer_checkpoint}")

    # Load evaluation data
    import numpy as np
    if Path(eval_data_path).exists():
        data = torch.tensor(np.memmap(eval_data_path, dtype=np.uint16, mode='r').astype(int))
    else:
        print("Using synthetic data for benchmarking")
        data = torch.randint(0, 50257, (1_000_000,))

    results = {
        "size": size,
        "device": device,
        "ssh": {},
        "transformer": {},
    }

    # FLOPs comparison
    print("\n=== FLOP Counts ===")
    ssh_flops = ssh_model.count_flops(1024)
    transformer_flops = transformer_model.count_flops(1024)

    results["ssh"]["flops"] = ssh_flops
    results["transformer"]["flops"] = transformer_flops

    print(f"SSH FLOPs per token: {ssh_flops['per_token']:,}")
    print(f"Transformer FLOPs per token: {transformer_flops['per_token']:,}")
    print(f"SSH/Transformer ratio: {ssh_flops['per_token'] / transformer_flops['per_token']:.2%}")

    # Throughput comparison
    print("\n=== Throughput ===")
    print("\nSSH:")
    results["ssh"]["throughput"] = measure_throughput(ssh_model, device=device)

    print("\nTransformer:")
    results["transformer"]["throughput"] = measure_throughput(transformer_model, device=device)

    # Compare at batch_size=1 (most common for generation)
    ssh_tps = results["ssh"]["throughput"][1]["tokens_per_second"]
    trans_tps = results["transformer"]["throughput"][1]["tokens_per_second"]
    print(f"\nThroughput ratio (batch=1): SSH/Transformer = {ssh_tps/trans_tps:.2f}x")

    # Perplexity comparison (if checkpoints provided)
    if ssh_checkpoint and transformer_checkpoint:
        print("\n=== Perplexity ===")
        print("\nSSH:")
        results["ssh"]["perplexity"] = measure_perplexity(ssh_model, data, device=device)

        print("\nTransformer:")
        results["transformer"]["perplexity"] = measure_perplexity(transformer_model, data, device=device)

        print(f"\nSSH perplexity: {results['ssh']['perplexity']['perplexity']:.2f}")
        print(f"Transformer perplexity: {results['transformer']['perplexity']['perplexity']:.2f}")

    # Adaptive behavior (SSH only)
    print("\n=== Adaptive Behavior (SSH) ===")
    results["ssh"]["adaptive"] = measure_adaptive_behavior(ssh_model, data, device=device)
    if results["ssh"]["adaptive"]:
        print(f"Average layers used: {results['ssh']['adaptive'].get('avg_layers_used', 'N/A')}")
        print(f"Attention usage: {results['ssh']['adaptive'].get('avg_attention_usage', 'N/A')}")

    # Memory comparison
    print("\n=== Memory Usage ===")
    results["ssh"]["memory"] = measure_memory(ssh_model, device=device)
    results["transformer"]["memory"] = measure_memory(transformer_model, device=device)

    print(f"SSH peak memory: {results['ssh']['memory']['peak_memory_gb']:.2f} GB")
    print(f"Transformer peak memory: {results['transformer']['memory']['peak_memory_gb']:.2f} GB")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    flop_ratio = ssh_flops['per_token'] / transformer_flops['per_token']
    throughput_ratio = ssh_tps / trans_tps

    print(f"FLOPs reduction: {(1-flop_ratio)*100:.1f}%")
    print(f"Throughput improvement: {(throughput_ratio-1)*100:.1f}%")

    if "perplexity" in results["ssh"] and "perplexity" in results["transformer"]:
        ppl_diff = results["ssh"]["perplexity"]["perplexity"] - results["transformer"]["perplexity"]["perplexity"]
        print(f"Perplexity difference: {ppl_diff:+.2f}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark SSH vs Transformer")
    parser.add_argument("--ssh_checkpoint", type=str, default=None)
    parser.add_argument("--transformer_checkpoint", type=str, default=None)
    parser.add_argument("--size", type=str, choices=["small", "medium"], default="small")
    parser.add_argument("--data", type=str, default="data/test.bin")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")

    args = parser.parse_args()

    run_comparison(
        ssh_checkpoint=args.ssh_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        size=args.size,
        eval_data_path=args.data,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
