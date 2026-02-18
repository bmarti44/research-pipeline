"""
Revision Tasks: GPU experiments for paper revision.

Task 2: Recompute selectivity with pairwise position alignment
Task 4: Cross-corruption (M3-magnitude noise applied to M5)
Task 5: Unmatched cross-problem transplant

Usage:
    cd code
    python revision_tasks.py \
        --checkpoint_dir ../results \
        --data data/prosqa_test.json \
        --output_dir ../results/experiments/revision
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from exp_utils import (
    load_model_by_name,
    prepare_input,
    get_hidden_states,
    get_special_ids,
    get_step_types,
    load_data,
    set_seed,
    run_inference,
)


NUM_THOUGHTS = 6
NUM_LAYERS = 13
HIDDEN_DIM = 768
N_FOLDS = 5


def run_linear_probe(X, y, n_folds=N_FOLDS):
    """Train RidgeClassifier probe, return 5-fold CV accuracy."""
    if len(np.unique(y)) < 2:
        return 0.0
    min_class_count = min(np.bincount(y))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0

    from sklearn.model_selection import cross_val_score
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores))


# =========================================================================
# TASK 2: Pairwise Selectivity
# =========================================================================

@torch.no_grad()
def task2_selectivity(model_m3, tokenizer, model_info_m3,
                      model_m5, model_info_m5,
                      data, num_samples, device, output_dir):
    """
    Recompute selectivity with pairwise position alignment.

    Instead of aligning ALL positions to n_common (limited by pos 5 n=12),
    compute selectivity for each (t, s) pair using min(n_t, n_s) samples.
    """
    print("\n" + "=" * 70)
    print("TASK 2: Pairwise Selectivity Computation")
    print("=" * 70)

    results = {}

    for model_name, model, model_info in [("m3", model_m3, model_info_m3),
                                           ("m5", model_m5, model_info_m5)]:
        print(f"\n--- Processing {model_name} ---")
        special_ids = get_special_ids(tokenizer)
        latent_id = special_ids["latent_id"]

        # Extract hidden states at thought positions
        hidden_by_pos = {t: [] for t in range(NUM_THOUGHTS)}
        labels_by_pos = {t: [] for t in range(NUM_THOUGHTS)}

        for idx in range(min(num_samples, len(data))):
            sample = data[idx]
            steps = sample.get("steps", [])
            step_types = get_step_types(sample)

            if len(steps) == 0:
                continue

            input_ids = prepare_input(sample, tokenizer, model_info,
                                      num_thoughts=NUM_THOUGHTS, device=device)
            tokens = input_ids[0].tolist()
            thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]

            if len(thought_positions) == 0:
                continue

            all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)

            if not torch.isfinite(all_hidden).all():
                continue

            actual_T = len(thought_positions)
            for t in range(min(actual_T, NUM_THOUGHTS)):
                pos = thought_positions[t]
                h = all_hidden[:, pos, :].cpu().float().numpy()

                if t < len(step_types):
                    label = step_types[t]
                else:
                    answer = sample.get("answer", "")
                    answer_clean = answer.rstrip().rstrip(".")
                    words = answer_clean.split()
                    label = words[-1] if words else "UNKNOWN"

                hidden_by_pos[t].append(h)
                labels_by_pos[t].append(label)

            if (idx + 1) % 100 == 0:
                print(f"  Extracted {idx + 1}/{min(num_samples, len(data))}")

        # Convert to numpy
        for t in range(NUM_THOUGHTS):
            if len(hidden_by_pos[t]) > 0:
                hidden_by_pos[t] = np.stack(hidden_by_pos[t], axis=0)
            else:
                hidden_by_pos[t] = None

        n_per_pos = {t: (hidden_by_pos[t].shape[0] if hidden_by_pos[t] is not None else 0)
                     for t in range(NUM_THOUGHTS)}
        print(f"  Samples per position: {n_per_pos}")

        # Compute PAIRWISE selectivity: for each position t, compare matched vs cross
        # using min(n_t, n_s) alignment for each pair
        selectivity_matrix = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
        matched_acc_matrix = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
        max_cross_acc_matrix = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
        cross_detail = {}  # (t, s) -> [layer accuracies]

        valid_positions = [t for t in range(NUM_THOUGHTS)
                          if hidden_by_pos[t] is not None and n_per_pos[t] >= 20]
        print(f"  Valid positions (n >= 20): {valid_positions}")

        for t in valid_positions:
            n_t = n_per_pos[t]
            le_t = LabelEncoder()
            y_t = le_t.fit_transform(labels_by_pos[t][:n_t])

            # Matched probe: accuracy at position t predicting step t
            for layer in range(NUM_LAYERS):
                X = hidden_by_pos[t][:n_t, layer, :]
                acc = run_linear_probe(X, y_t)
                matched_acc_matrix[layer, t] = acc

            # Cross-position probes: accuracy at position t predicting step s
            best_cross_per_layer = np.zeros(NUM_LAYERS)

            for s in valid_positions:
                if s == t:
                    continue

                n_common = min(n_t, n_per_pos[s])
                if n_common < 20:
                    continue

                le_s = LabelEncoder()
                y_s = le_s.fit_transform(labels_by_pos[s][:n_common])

                if len(np.unique(y_s)) < 2:
                    continue

                layer_accs = np.zeros(NUM_LAYERS)
                for layer in range(NUM_LAYERS):
                    X = hidden_by_pos[t][:n_common, layer, :]
                    acc = run_linear_probe(X, y_s)
                    layer_accs[layer] = acc

                cross_detail[(t, s)] = layer_accs.tolist()
                best_cross_per_layer = np.maximum(best_cross_per_layer, layer_accs)

            max_cross_acc_matrix[:, t] = best_cross_per_layer
            selectivity_matrix[:, t] = matched_acc_matrix[:, t] - best_cross_per_layer

            print(f"  Pos {t}: matched peak = {matched_acc_matrix[:, t].max():.4f}, "
                  f"max cross = {best_cross_per_layer.max():.4f}, "
                  f"selectivity range = [{selectivity_matrix[:, t].min():.4f}, "
                  f"{selectivity_matrix[:, t].max():.4f}]")

        # Summary statistics
        all_sel = selectivity_matrix[:, valid_positions].flatten()
        model_results = {
            "selectivity_grid": selectivity_matrix.tolist(),
            "matched_accuracy_grid": matched_acc_matrix.tolist(),
            "max_cross_accuracy_grid": max_cross_acc_matrix.tolist(),
            "cross_detail": {f"{t}_{s}": v for (t, s), v in cross_detail.items()},
            "n_per_position": n_per_pos,
            "valid_positions": valid_positions,
            "summary": {
                "mean_selectivity": float(np.mean(all_sel)),
                "std_selectivity": float(np.std(all_sel)),
                "max_selectivity": float(np.max(all_sel)) if len(all_sel) > 0 else 0.0,
                "min_selectivity": float(np.min(all_sel)) if len(all_sel) > 0 else 0.0,
                "cells_above_0001": int(np.sum(all_sel > 0.001)),
                "cells_above_001": int(np.sum(all_sel > 0.01)),
                "cells_above_005": int(np.sum(all_sel > 0.05)),
                "n_cells_evaluated": len(all_sel),
            }
        }
        results[model_name] = model_results

        print(f"\n  {model_name} selectivity summary:")
        for k, v in model_results["summary"].items():
            print(f"    {k}: {v}")

    # Save
    out_path = os.path.join(output_dir, "selectivity_detailed.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved selectivity results to {out_path}")
    return results


# =========================================================================
# TASK 4: Cross-Corruption (M3-magnitude noise on M5)
# =========================================================================

@torch.no_grad()
def task4_cross_corruption(model_m3, tokenizer, model_info_m3,
                           model_m5, model_info_m5,
                           data, num_samples, device, output_dir):
    """
    Apply M3-magnitude noise to M5's thought positions.
    Compare against M5's own-magnitude corruption and M3's corruption.
    """
    print("\n" + "=" * 70)
    print("TASK 4: Cross-Corruption (M3-magnitude noise on M5)")
    print("=" * 70)

    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

    # First, collect M3's thought token statistics (mean, std) for noise generation
    print("  Collecting M3 thought token statistics...")
    m3_thought_vecs = []
    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        if len(sample.get("steps", [])) == 0:
            continue
        input_ids = prepare_input(sample, tokenizer, model_info_m3,
                                  num_thoughts=NUM_THOUGHTS, device=device)
        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
        if not thought_positions:
            continue
        all_hidden = get_hidden_states(model_m3, tokenizer, input_ids, model_info_m3)
        if not torch.isfinite(all_hidden).all():
            continue
        for pos in thought_positions:
            # Use last layer hidden state (layer -1) for noise calibration
            h = all_hidden[-1, pos, :].cpu().float().numpy()
            m3_thought_vecs.append(h)
    m3_thought_vecs = np.stack(m3_thought_vecs)
    m3_mean = m3_thought_vecs.mean(axis=0)
    m3_std = m3_thought_vecs.std(axis=0)
    print(f"  M3 thought stats: mean L2={np.linalg.norm(m3_mean):.2f}, "
          f"std mean={m3_std.mean():.4f}")

    # Also collect M5's thought token statistics
    print("  Collecting M5 thought token statistics...")
    m5_thought_vecs = []
    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        if len(sample.get("steps", [])) == 0:
            continue
        input_ids = prepare_input(sample, tokenizer, model_info_m5,
                                  num_thoughts=NUM_THOUGHTS, device=device)
        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
        if not thought_positions:
            continue
        all_hidden = get_hidden_states(model_m5, tokenizer, input_ids, model_info_m5)
        if not torch.isfinite(all_hidden).all():
            continue
        for pos in thought_positions:
            h = all_hidden[-1, pos, :].cpu().float().numpy()
            m5_thought_vecs.append(h)
    m5_thought_vecs = np.stack(m5_thought_vecs)
    m5_mean = m5_thought_vecs.mean(axis=0)
    m5_std = m5_thought_vecs.std(axis=0)
    print(f"  M5 thought stats: mean L2={np.linalg.norm(m5_mean):.2f}, "
          f"std mean={m5_std.mean():.4f}")

    # Generate noise samples: M3-scale and M5-scale
    rng = np.random.RandomState(42)

    def generate_noise(mean, std, n_positions, n_samples):
        """Generate noise matching the given distribution."""
        noise = rng.randn(n_samples, n_positions, HIDDEN_DIM) * std[None, None, :] + mean[None, None, :]
        return noise

    # Run progressive forward corruption on M5 with three noise conditions
    conditions = {
        "m5_own_noise": {"model": model_m5, "model_info": model_info_m5,
                         "noise_mean": m5_mean, "noise_std": m5_std, "label": "M5 + M5-noise"},
        "m5_m3_noise": {"model": model_m5, "model_info": model_info_m5,
                        "noise_mean": m3_mean, "noise_std": m3_std, "label": "M5 + M3-noise"},
        "m3_own_noise": {"model": model_m3, "model_info": model_info_m3,
                         "noise_mean": m3_mean, "noise_std": m3_std, "label": "M3 + M3-noise"},
    }

    results = {}
    for cond_name, cond in conditions.items():
        print(f"\n  Running corruption: {cond['label']}...")
        model = cond["model"]
        model_info = cond["model_info"]
        noise_mean = cond["noise_mean"]
        noise_std = cond["noise_std"]

        # Progressive forward corruption: corrupt 0, 1, ..., 6 positions
        accuracies = []
        for n_corrupt in range(NUM_THOUGHTS + 1):
            correct = 0
            total = 0
            for idx in range(min(num_samples, len(data))):
                sample = data[idx]
                if len(sample.get("steps", [])) == 0:
                    continue
                input_ids = prepare_input(sample, tokenizer, model_info,
                                          num_thoughts=NUM_THOUGHTS, device=device)
                tokens = input_ids[0].tolist()
                thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
                if not thought_positions:
                    continue

                # Get hidden states
                all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)
                if not torch.isfinite(all_hidden).all():
                    continue

                # Corrupt first n_corrupt positions at the LAST layer
                if n_corrupt > 0:
                    noise = torch.tensor(
                        rng.randn(n_corrupt, HIDDEN_DIM) * noise_std + noise_mean,
                        dtype=all_hidden.dtype, device=device
                    )
                    for k in range(min(n_corrupt, len(thought_positions))):
                        pos = thought_positions[k]
                        all_hidden[-1, pos, :] = noise[k]

                # Run inference from corrupted hidden states
                # We need to get the model's prediction using the corrupted last-layer states
                # The simplest approach: run the model with hooks that replace thought positions
                pred = run_inference_with_corruption(
                    model, tokenizer, input_ids, model_info,
                    thought_positions, all_hidden[-1], n_corrupt, device
                )

                answer = sample.get("answer", "").strip().rstrip(".")
                if pred is not None and answer.lower() in pred.lower():
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0.0
            accuracies.append(acc)
            print(f"    Corrupt {n_corrupt}: {acc:.3f} ({correct}/{total})")

        # Compute L2 distance of noise
        noise_sample = rng.randn(1000, HIDDEN_DIM) * noise_std + noise_mean
        l2_dist = np.mean(np.linalg.norm(noise_sample, axis=1))

        results[cond_name] = {
            "label": cond["label"],
            "accuracies": accuracies,
            "noise_l2_mean": float(l2_dist),
        }

    # Save
    out_path = os.path.join(output_dir, "cross_corruption.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cross-corruption results to {out_path}")
    return results


@torch.no_grad()
def run_inference_with_corruption(model, tokenizer, input_ids, model_info,
                                  thought_positions, corrupted_last_layer,
                                  n_corrupt, device):
    """
    Run model inference, replacing the thought-position hidden states at the
    last layer with corrupted values for the first n_corrupt positions.

    Uses a forward hook on the last transformer layer to inject corrupted states.
    """
    if n_corrupt == 0:
        # No corruption, just run normal inference
        return run_inference(model, tokenizer, input_ids, model_info)

    # Register hook to corrupt last-layer outputs
    positions_to_corrupt = thought_positions[:n_corrupt]
    corrupted_vectors = corrupted_last_layer[thought_positions[:n_corrupt]]

    if model_info["type"] == "coconut":
        # For Coconut models, we need to hook into the base model's last layer
        base_model = model.base_causallm
        target_layer = base_model.transformer.h[-1]  # Last transformer block
    else:
        target_layer = model.transformer.h[-1]

    hook_handle = None

    def corruption_hook(module, input, output):
        # output is a tuple; first element is the hidden states
        hidden_states = output[0]
        for k, pos in enumerate(positions_to_corrupt):
            if pos < hidden_states.shape[1]:
                hidden_states[0, pos, :] = corrupted_vectors[k]
        # Return modified output
        return (hidden_states,) + output[1:]

    hook_handle = target_layer.register_forward_hook(corruption_hook)

    try:
        result = run_inference(model, tokenizer, input_ids, model_info)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return result


# =========================================================================
# TASK 5: Unmatched Cross-Problem Transplant
# =========================================================================

@torch.no_grad()
def task5_unmatched_transplant(model_m3, tokenizer, model_info_m3,
                                model_m5, model_info_m5,
                                data, num_samples, device, output_dir):
    """
    Run fully random (unmatched) thought token transplantation.
    200 random donor-recipient pairs with NO hop-count matching.
    """
    print("\n" + "=" * 70)
    print("TASK 5: Unmatched Cross-Problem Transplant")
    print("=" * 70)

    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    rng = np.random.RandomState(42)

    n_pairs = 200
    samples = data[:min(num_samples, len(data))]

    # Create random pairs (no hop-count matching)
    indices = list(range(len(samples)))
    donor_indices = rng.choice(indices, size=n_pairs, replace=True)
    recipient_indices = rng.choice(indices, size=n_pairs, replace=True)
    # Ensure no self-transplant
    for i in range(n_pairs):
        while recipient_indices[i] == donor_indices[i]:
            recipient_indices[i] = rng.choice(indices)

    results = {}
    for model_name, model, model_info in [("m3", model_m3, model_info_m3),
                                           ("m5", model_m5, model_info_m5)]:
        print(f"\n  Running unmatched transplant for {model_name}...")

        # First, extract thought representations for all samples
        print(f"    Extracting thought representations...")
        thought_reps = {}  # idx -> tensor of shape [n_thoughts, hidden_dim]
        for idx in range(len(samples)):
            sample = samples[idx]
            if len(sample.get("steps", [])) == 0:
                continue
            input_ids = prepare_input(sample, tokenizer, model_info,
                                      num_thoughts=NUM_THOUGHTS, device=device)
            tokens = input_ids[0].tolist()
            thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
            if not thought_positions:
                continue
            all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)
            if not torch.isfinite(all_hidden).all():
                continue
            # Store last-layer hidden states at thought positions
            reps = []
            for pos in thought_positions:
                reps.append(all_hidden[-1, pos, :].clone())
            thought_reps[idx] = torch.stack(reps)

            if (idx + 1) % 100 == 0:
                print(f"      Extracted {idx + 1}/{len(samples)}")

        # Run transplant
        correct = 0
        total = 0
        hop_mismatch_results = {}  # mismatch_magnitude -> (correct, total)

        for pair_idx in range(n_pairs):
            d_idx = int(donor_indices[pair_idx])
            r_idx = int(recipient_indices[pair_idx])

            if d_idx not in thought_reps or r_idx not in thought_reps:
                continue

            donor_reps = thought_reps[d_idx]
            recipient_sample = samples[r_idx]

            # Get hop counts for mismatch tracking
            d_hops = len(samples[d_idx].get("steps", []))
            r_hops = len(recipient_sample.get("steps", []))
            mismatch = abs(d_hops - r_hops)

            # Prepare recipient input
            input_ids = prepare_input(recipient_sample, tokenizer, model_info,
                                      num_thoughts=NUM_THOUGHTS, device=device)
            tokens = input_ids[0].tolist()
            thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]

            if not thought_positions:
                continue

            # Run inference with donor's thought representations injected
            n_to_transplant = min(len(thought_positions), donor_reps.shape[0])
            pred = run_inference_with_transplant(
                model, tokenizer, input_ids, model_info,
                thought_positions, donor_reps, n_to_transplant, device
            )

            answer = recipient_sample.get("answer", "").strip().rstrip(".")
            is_correct = pred is not None and answer.lower() in pred.lower()
            if is_correct:
                correct += 1
            total += 1

            if mismatch not in hop_mismatch_results:
                hop_mismatch_results[mismatch] = [0, 0]
            hop_mismatch_results[mismatch][1] += 1
            if is_correct:
                hop_mismatch_results[mismatch][0] += 1

        acc = correct / total if total > 0 else 0.0
        print(f"    Unmatched transplant accuracy: {acc:.3f} ({correct}/{total})")

        # Breakdown by hop-count mismatch
        mismatch_breakdown = {}
        for mm, (c, t) in sorted(hop_mismatch_results.items()):
            mismatch_breakdown[str(mm)] = {
                "correct": c, "total": t,
                "accuracy": c / t if t > 0 else 0.0
            }
            print(f"      Mismatch {mm} hops: {c}/{t} = {c/t:.3f}" if t > 0 else f"      Mismatch {mm}: no pairs")

        results[model_name] = {
            "unmatched_accuracy": acc,
            "unmatched_correct": correct,
            "unmatched_total": total,
            "mismatch_breakdown": mismatch_breakdown,
        }

    # Add matched results from existing data
    results["reference"] = {
        "matched_m3_accuracy": 0.970,
        "matched_m5_accuracy": 0.965,
        "matched_pairs": 200,
        "note": "From original corruption experiment (hop-count matched)"
    }

    # Save
    out_path = os.path.join(output_dir, "unmatched_transplant.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved unmatched transplant results to {out_path}")
    return results


@torch.no_grad()
def run_inference_with_transplant(model, tokenizer, input_ids, model_info,
                                   thought_positions, donor_reps, n_transplant, device):
    """
    Run inference, replacing thought-position hidden states with donor representations.
    """
    positions_to_replace = thought_positions[:n_transplant]
    replacement_vectors = donor_reps[:n_transplant].to(device)

    if model_info["type"] == "coconut":
        base_model = model.base_causallm
        target_layer = base_model.transformer.h[-1]
    else:
        target_layer = model.transformer.h[-1]

    def transplant_hook(module, input, output):
        hidden_states = output[0]
        for k, pos in enumerate(positions_to_replace):
            if pos < hidden_states.shape[1] and k < replacement_vectors.shape[0]:
                hidden_states[0, pos, :] = replacement_vectors[k]
        return (hidden_states,) + output[1:]

    hook_handle = target_layer.register_forward_hook(transplant_hook)
    try:
        result = run_inference(model, tokenizer, input_ids, model_info)
    finally:
        hook_handle.remove()

    return result


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Revision experiment tasks")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="2,4,5",
                        help="Comma-separated task numbers to run")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [int(t) for t in args.tasks.split(",")]

    print("=" * 70)
    print("REVISION TASKS")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_data(args.data)
    print(f"  Loaded {len(data)} samples")

    # Load models
    print("\nLoading M3 (COCONUT)...")
    model_m3, tokenizer, model_info_m3 = load_model_by_name(
        "m3", args.checkpoint_dir, device=device)

    print("\nLoading M5 (Pause)...")
    model_m5, _, model_info_m5 = load_model_by_name(
        "m5", args.checkpoint_dir, device=device)

    all_results = {}

    if 2 in tasks:
        sel_results = task2_selectivity(
            model_m3, tokenizer, model_info_m3,
            model_m5, model_info_m5,
            data, args.num_samples, device, args.output_dir
        )
        all_results["task2"] = sel_results

    if 4 in tasks:
        corr_results = task4_cross_corruption(
            model_m3, tokenizer, model_info_m3,
            model_m5, model_info_m5,
            data, args.num_samples, device, args.output_dir
        )
        all_results["task4"] = corr_results

    if 5 in tasks:
        trans_results = task5_unmatched_transplant(
            model_m3, tokenizer, model_info_m3,
            model_m5, model_info_m5,
            data, args.num_samples, device, args.output_dir
        )
        all_results["task5"] = trans_results

    # Save combined results
    combined_path = os.path.join(args.output_dir, "revision_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
