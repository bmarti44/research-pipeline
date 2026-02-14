"""
Task 4: Cross-corruption (M3-magnitude noise on M5)
Task 5: Unmatched cross-problem transplant

Uses the same corruption API as exp_corruption.py.

Usage:
    cd code
    python task4_5_gpu.py \
        --checkpoint_dir ../results \
        --data data/prosqa_test.json \
        --output_dir ../results/experiments/revision \
        --seed 0
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import numpy as np

from exp_utils import (
    load_model_by_name,
    prepare_input,
    run_inference,
    extract_answer,
    get_ground_truth,
    get_special_ids,
    get_processed_embeds,
    load_data,
    set_seed,
)
from exp_corruption import check_answer_from_corrupted

NUM_THOUGHTS = 6


# =========================================================================
# TASK 4: Cross-Corruption
# =========================================================================

@torch.no_grad()
def task4_cross_corruption(model_m3, tokenizer, model_info_m3,
                           model_m5, model_info_m5,
                           data, num_samples, device, output_dir):
    """
    Apply M3-magnitude noise to M5's thought positions.
    Three conditions:
      1. M3 + M3-scale noise (existing, verify)
      2. M5 + M5-scale noise (existing, verify)
      3. M5 + M3-scale noise (NEW)
    """
    print("\n" + "=" * 70)
    print("TASK 4: Cross-Corruption (M3-magnitude noise on M5)")
    print("=" * 70)

    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    T = NUM_THOUGHTS

    # Step 1: Collect M3's thought-token embedding statistics
    print("\n  Collecting M3 thought-token statistics from processed embeddings...")
    m3_thought_vecs = []
    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        if len(sample.get("steps", [])) == 0:
            continue
        input_ids = prepare_input(sample, tokenizer, model_info_m3,
                                  num_thoughts=T, device=device)
        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
        if not thought_positions:
            continue
        clean_embeds = get_processed_embeds(model_m3, input_ids)
        for pos in thought_positions:
            m3_thought_vecs.append(clean_embeds[0, pos, :].cpu())
    m3_thought_vecs = torch.stack(m3_thought_vecs)
    m3_noise_std = m3_thought_vecs.std().item()  # scalar std for isotropic noise
    m3_noise_mean_l2 = torch.norm(m3_thought_vecs, dim=1).mean().item()
    print(f"  M3 thought stats: std={m3_noise_std:.4f}, mean_L2={m3_noise_mean_l2:.2f}")

    # Step 2: Collect M5's thought-token embedding statistics
    print("  Collecting M5 thought-token statistics...")
    m5_thought_vecs = []
    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        if len(sample.get("steps", [])) == 0:
            continue
        input_ids = prepare_input(sample, tokenizer, model_info_m5,
                                  num_thoughts=T, device=device)
        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
        if not thought_positions:
            continue
        clean_embeds = get_processed_embeds(model_m5, input_ids)
        for pos in thought_positions:
            m5_thought_vecs.append(clean_embeds[0, pos, :].cpu())
    m5_thought_vecs = torch.stack(m5_thought_vecs)
    m5_noise_std = m5_thought_vecs.std().item()
    m5_noise_mean_l2 = torch.norm(m5_thought_vecs, dim=1).mean().item()
    print(f"  M5 thought stats: std={m5_noise_std:.4f}, mean_L2={m5_noise_mean_l2:.2f}")

    # Step 3: Run progressive forward corruption under three conditions
    conditions = [
        ("m3_m3noise", model_m3, model_info_m3, m3_noise_std, "M3 + M3-scale noise"),
        ("m5_m5noise", model_m5, model_info_m5, m5_noise_std, "M5 + M5-scale noise"),
        ("m5_m3noise", model_m5, model_info_m5, m3_noise_std, "M5 + M3-scale noise"),
    ]

    results = {}
    for cond_name, model, model_info, noise_std, label in conditions:
        print(f"\n  Running: {label} (noise_std={noise_std:.4f})...")
        accuracies = []

        # Clean accuracy (0 positions corrupted)
        clean_correct = 0
        forward_correct = [0] * T
        total = 0

        for idx in range(min(num_samples, len(data))):
            sample = data[idx]
            if len(sample.get("steps", [])) == 0:
                continue
            gt = get_ground_truth(sample)
            input_ids = prepare_input(sample, tokenizer, model_info,
                                      num_thoughts=T, device=device)
            tokens = input_ids[0].tolist()
            thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
            if not thought_positions:
                continue

            clean_embeds = get_processed_embeds(model, input_ids)
            actual_T = len(thought_positions)
            hidden_dim = clean_embeds.shape[-1]

            # Clean answer
            clean_answer = check_answer_from_corrupted(
                model, clean_embeds, [], torch.zeros(0, hidden_dim, device=device), tokenizer
            )
            clean_correct += int(clean_answer == gt)
            total += 1

            # Generate noise at the specified scale
            all_noise = torch.randn(actual_T, hidden_dim, device=device) * noise_std

            # Progressive forward corruption
            for k in range(min(actual_T, T)):
                forward_positions = thought_positions[:k + 1]
                fwd_answer = check_answer_from_corrupted(
                    model, clean_embeds, forward_positions, all_noise[:k + 1], tokenizer
                )
                forward_correct[k] += int(fwd_answer == gt)

            if (idx + 1) % 100 == 0:
                print(f"    Progress: {idx+1}/{min(num_samples, len(data))}")

        clean_acc = clean_correct / total if total > 0 else 0
        forward_acc = [forward_correct[k] / total if total > 0 else 0 for k in range(T)]
        accuracies = [round(clean_acc, 4)] + [round(a, 4) for a in forward_acc]

        # Compute L2 distance of noise vectors
        sample_noise = torch.randn(1000, 768) * noise_std
        noise_l2 = torch.norm(sample_noise, dim=1).mean().item()

        results[cond_name] = {
            "label": label,
            "noise_std": noise_std,
            "noise_l2_mean": round(noise_l2, 2),
            "accuracies": accuracies,  # [clean, 1_corrupt, 2_corrupt, ..., 6_corrupt]
            "total_samples": total,
        }

        print(f"    Clean: {clean_acc:.3f}")
        for k in range(T):
            print(f"    Corrupt {k+1}: {forward_acc[k]:.3f}")

    # Save
    out_path = os.path.join(output_dir, "cross_corruption.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return results


# =========================================================================
# TASK 5: Unmatched Cross-Problem Transplant
# =========================================================================

@torch.no_grad()
def task5_unmatched_transplant(model_m3, tokenizer, model_info_m3,
                                model_m5, model_info_m5,
                                data, num_samples, device, output_dir):
    """
    Fully random (unmatched) thought token transplant.
    200 random donor-recipient pairs with no hop-count matching.
    """
    print("\n" + "=" * 70)
    print("TASK 5: Unmatched Cross-Problem Transplant")
    print("=" * 70)

    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    T = NUM_THOUGHTS
    n_pairs = 200
    rng = np.random.RandomState(42)

    results = {}

    for model_name, model, model_info in [("m3", model_m3, model_info_m3),
                                           ("m5", model_m5, model_info_m5)]:
        print(f"\n  Processing {model_name}...")

        # Extract processed embeddings for all samples
        print(f"    Extracting processed embeddings...")
        embeds_store = []
        for idx in range(min(num_samples, len(data))):
            sample = data[idx]
            if len(sample.get("steps", [])) == 0:
                continue
            gt = get_ground_truth(sample)
            input_ids = prepare_input(sample, tokenizer, model_info,
                                      num_thoughts=T, device=device)
            tokens = input_ids[0].tolist()
            thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
            if not thought_positions:
                continue
            clean_embeds = get_processed_embeds(model, input_ids)

            embeds_store.append({
                "embeds": clean_embeds.cpu(),
                "thought_positions": thought_positions,
                "input_ids": input_ids.cpu(),
                "gt": gt,
                "n_hops": len(sample.get("steps", [])),
                "sample_idx": idx,
            })

            if (idx + 1) % 100 == 0:
                print(f"      Extracted {idx+1}/{min(num_samples, len(data))}")

        print(f"    Total extracted: {len(embeds_store)}")

        # Create random (unmatched) pairs
        indices = list(range(len(embeds_store)))
        donor_indices = rng.choice(indices, size=n_pairs, replace=True).tolist()
        recipient_indices = rng.choice(indices, size=n_pairs, replace=True).tolist()
        for i in range(n_pairs):
            while recipient_indices[i] == donor_indices[i]:
                recipient_indices[i] = rng.choice(indices)

        # Run transplant
        print(f"    Running {n_pairs} unmatched transplant pairs...")
        correct = 0
        total = 0
        mismatch_results = defaultdict(lambda: [0, 0])

        for pair_idx in range(n_pairs):
            d_idx = donor_indices[pair_idx]
            r_idx = recipient_indices[pair_idx]

            donor = embeds_store[d_idx]
            recipient = embeds_store[r_idx]

            # Create transplanted embeddings
            transplanted = recipient["embeds"].clone().to(device)
            d_embeds = donor["embeds"].to(device)

            # Replace recipient's thought positions with donor's
            for t_k in range(min(len(recipient["thought_positions"]),
                                  len(donor["thought_positions"]))):
                r_pos = recipient["thought_positions"][t_k]
                d_pos = donor["thought_positions"][t_k]
                transplanted[0, r_pos, :] = d_embeds[0, d_pos, :]

            # Run inference with transplanted embeddings
            pred = check_answer_from_corrupted(
                model, transplanted, [], torch.zeros(0, 768, device=device), tokenizer
            )

            gt = recipient["gt"]
            is_correct = (pred == gt)
            correct += int(is_correct)
            total += 1

            mismatch = abs(donor["n_hops"] - recipient["n_hops"])
            mismatch_results[mismatch][0] += int(is_correct)
            mismatch_results[mismatch][1] += 1

        acc = correct / total if total > 0 else 0
        print(f"    Unmatched transplant accuracy: {acc:.3f} ({correct}/{total})")

        by_mismatch = {}
        for mm in sorted(mismatch_results.keys()):
            c, t = mismatch_results[mm]
            by_mismatch[str(mm)] = {
                "correct": c, "total": t,
                "accuracy": round(c / t, 4) if t > 0 else 0
            }
            print(f"      Mismatch {mm}: {c}/{t} = {c/t:.3f}" if t > 0 else f"      Mismatch {mm}: no pairs")

        results[model_name] = {
            "unmatched_accuracy": round(acc, 4),
            "unmatched_correct": correct,
            "unmatched_total": total,
            "by_mismatch": by_mismatch,
        }

    # Add reference values
    results["reference"] = {
        "matched_m3_accuracy": 0.970,
        "matched_m5_accuracy": 0.965,
        "matched_pairs": 200,
        "note": "From original exp_corruption.py (hop-count matched)"
    }

    out_path = os.path.join(output_dir, "unmatched_transplant.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="4,5")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [int(t) for t in args.tasks.split(",")]

    print("=" * 70)
    print(f"REVISION TASKS 4+5 | Device: {device}")
    print("=" * 70)

    data = load_data(args.data)
    print(f"Loaded {len(data)} samples")

    print("\nLoading M3 (COCONUT)...")
    model_m3, tokenizer, model_info_m3 = load_model_by_name(
        "m3", args.checkpoint_dir, device=device)

    print("Loading M5 (Pause)...")
    model_m5, _, model_info_m5 = load_model_by_name(
        "m5", args.checkpoint_dir, device=device)

    if 4 in tasks:
        task4_cross_corruption(
            model_m3, tokenizer, model_info_m3,
            model_m5, model_info_m5,
            data, args.num_samples, device, args.output_dir)

    if 5 in tasks:
        task5_unmatched_transplant(
            model_m3, tokenizer, model_info_m3,
            model_m5, model_info_m5,
            data, args.num_samples, device, args.output_dir)

    print("\nAll tasks complete!")


if __name__ == "__main__":
    main()
