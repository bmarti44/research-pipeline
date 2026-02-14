"""
Task 2: Recompute selectivity with pairwise position alignment.

Bug: Original compute_selectivity() used n_common = min(samples across ALL positions) = 12
(limited by position 5 with n=12). This truncated ALL positions to 12 samples, producing
trivially 0.0 selectivity because RidgeClassifier can't learn with 12 samples and 38+ classes.

Fix: Use pairwise alignment. For each (t, s) cross-position comparison, use
min(n_t, n_s) samples instead of global min.

Usage:
    cd /lambda/nfs/experiment/code/v9_meta_fork
    python task2_selectivity_fix.py \
        --checkpoint_dir /lambda/nfs/experiment/results/v9_meta_fork \
        --data data/prosqa_test.json \
        --output_dir /lambda/nfs/experiment/experiments/revision
"""

import argparse
import json
import os
import time
import warnings

import torch
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(alpha=1.0, random_state=42)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores))


@torch.no_grad()
def extract_thought_hidden_states(model, tokenizer, model_info, data, num_samples, device):
    """Extract hidden states at thought positions for all samples."""
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

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
            print(f"    Extracted {idx + 1}/{min(num_samples, len(data))}")

    # Convert to numpy
    for t in range(NUM_THOUGHTS):
        if len(hidden_by_pos[t]) > 0:
            hidden_by_pos[t] = np.stack(hidden_by_pos[t], axis=0)
        else:
            hidden_by_pos[t] = None

    return hidden_by_pos, labels_by_pos


def compute_pairwise_selectivity(hidden_by_pos, labels_by_pos):
    """
    Compute selectivity with pairwise position alignment (NOT global n_common).

    For each position t and layer l:
        selectivity(l, t) = probe_acc(h(l,t), target=step_t)
                          - max_{s != t} probe_acc(h(l,t), target=step_s)

    Cross-position comparisons use min(n_t, n_s) samples for each pair.
    """
    n_per_pos = {}
    for t in range(NUM_THOUGHTS):
        if hidden_by_pos[t] is not None:
            n_per_pos[t] = hidden_by_pos[t].shape[0]
        else:
            n_per_pos[t] = 0

    print(f"  Samples per position: {n_per_pos}")

    # Which positions have enough data for meaningful probing?
    valid_positions = [t for t in range(NUM_THOUGHTS) if n_per_pos[t] >= 20]
    print(f"  Valid positions (n >= 20): {valid_positions}")

    # Compute matched probe accuracy using FULL per-position sample sizes
    matched_acc = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
    encoded_labels = {}

    for t in valid_positions:
        le = LabelEncoder()
        y = le.fit_transform(labels_by_pos[t])
        encoded_labels[t] = y
        n_classes = len(le.classes_)

        print(f"  Position {t}: n={n_per_pos[t]}, classes={n_classes}")

        for layer in range(NUM_LAYERS):
            X = hidden_by_pos[t][:, layer, :]
            acc = run_linear_probe(X, y)
            matched_acc[layer, t] = acc

    # Compute cross-position probe accuracy with PAIRWISE alignment
    # For each (t, s) pair, use min(n_t, n_s) samples
    cross_acc = {}  # (t, s) -> np.array of shape [NUM_LAYERS]

    for t in valid_positions:
        for s in valid_positions:
            if s == t:
                continue

            n_common = min(n_per_pos[t], n_per_pos[s])
            if n_common < 20:
                print(f"    Pair ({t},{s}): n_common={n_common} < 20, skipping")
                continue

            # Use labels from position s applied to hidden states from position t
            # Both truncated to n_common (samples are aligned by index)
            le_s = LabelEncoder()
            y_s = le_s.fit_transform(labels_by_pos[s][:n_common])

            if len(np.unique(y_s)) < 2:
                cross_acc[(t, s)] = np.zeros(NUM_LAYERS)
                continue

            layer_accs = np.zeros(NUM_LAYERS)
            for layer in range(NUM_LAYERS):
                X = hidden_by_pos[t][:n_common, layer, :]
                acc = run_linear_probe(X, y_s)
                layer_accs[layer] = acc

            cross_acc[(t, s)] = layer_accs

    # Compute selectivity: matched - max(cross) for each (layer, position)
    selectivity = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
    max_cross = np.zeros((NUM_LAYERS, NUM_THOUGHTS))

    for t in valid_positions:
        best_cross_per_layer = np.zeros(NUM_LAYERS)
        for s in valid_positions:
            if s == t:
                continue
            if (t, s) in cross_acc:
                best_cross_per_layer = np.maximum(best_cross_per_layer, cross_acc[(t, s)])

        max_cross[:, t] = best_cross_per_layer

        # For selectivity, use the matched accuracy computed with the SAME
        # sample size as the cross-position probes (for fair comparison)
        # Recompute matched acc at the pairwise n to ensure apples-to-apples
        n_for_cross = min(n_per_pos[s2] for s2 in valid_positions if s2 != t) \
                      if any(s2 != t for s2 in valid_positions) else n_per_pos[t]
        n_for_cross = min(n_for_cross, n_per_pos[t])

        # Actually, let's compute two versions:
        # 1. "raw" selectivity: matched_full - max_cross_pairwise
        #    (unfair because matched uses more data)
        # 2. "aligned" selectivity: matched_pairwise - max_cross_pairwise
        #    (fair comparison at same sample size)

        selectivity[:, t] = matched_acc[:, t] - best_cross_per_layer

    # Also compute "aligned" selectivity where matched uses same n as cross
    selectivity_aligned = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
    for t in valid_positions:
        # Find the largest n_common used in cross comparisons for this t
        cross_ns = [min(n_per_pos[t], n_per_pos[s])
                    for s in valid_positions if s != t and (t, s) in cross_acc]
        if not cross_ns:
            continue

        # Use the MINIMUM cross n for the fairest comparison
        n_align = min(cross_ns)

        le_t_align = LabelEncoder()
        y_t_align = le_t_align.fit_transform(labels_by_pos[t][:n_align])

        if len(np.unique(y_t_align)) < 2:
            continue

        best_cross_aligned = np.zeros(NUM_LAYERS)
        for s in valid_positions:
            if s == t or (t, s) not in cross_acc:
                continue
            n_cs = min(n_per_pos[t], n_per_pos[s])
            # Recompute cross at n_align for alignment
            le_s_a = LabelEncoder()
            y_s_a = le_s_a.fit_transform(labels_by_pos[s][:n_align])
            if len(np.unique(y_s_a)) < 2:
                continue
            for layer in range(NUM_LAYERS):
                X = hidden_by_pos[t][:n_align, layer, :]
                acc = run_linear_probe(X, y_s_a)
                best_cross_aligned[layer] = max(best_cross_aligned[layer], acc)

        for layer in range(NUM_LAYERS):
            X = hidden_by_pos[t][:n_align, layer, :]
            matched_a = run_linear_probe(X, y_t_align)
            selectivity_aligned[layer, t] = matched_a - best_cross_aligned[layer]

    return {
        "selectivity_raw": selectivity,  # matched_full - max_cross_pairwise
        "selectivity_aligned": selectivity_aligned,  # matched_aligned - max_cross_aligned
        "matched_accuracy": matched_acc,
        "max_cross_accuracy": max_cross,
        "cross_accuracies": {f"{t}_{s}": v.tolist() for (t, s), v in cross_acc.items()},
        "n_per_position": n_per_pos,
        "valid_positions": valid_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 2: Recompute selectivity")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TASK 2: Selectivity Recomputation (pairwise alignment)")
    print(f"Device: {device}")
    print("=" * 70)

    data = load_data(args.data)
    print(f"Loaded {len(data)} samples")

    results = {
        "bug_description": "Original selectivity used n_common=min(all positions)=12. "
                          "Positions 0-2 with 500 samples were truncated to 12, making "
                          "probes unable to learn with 38+ classes.",
        "original_selectivity": {"m3": "0.0 all 78 cells", "m5": "0.0 all 78 cells"},
        "heatmap_affected": False,
        "thought_vs_input_affected": False,
    }

    for model_name in ["m3", "m5"]:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        model, tokenizer, model_info = load_model_by_name(
            model_name, args.checkpoint_dir, device=device)

        print("  Extracting hidden states...")
        t0 = time.time()
        hidden_by_pos, labels_by_pos = extract_thought_hidden_states(
            model, tokenizer, model_info, data, args.num_samples, device)
        print(f"  Extraction done in {time.time() - t0:.1f}s")

        print("  Computing pairwise selectivity...")
        t0 = time.time()
        sel_data = compute_pairwise_selectivity(hidden_by_pos, labels_by_pos)
        print(f"  Selectivity done in {time.time() - t0:.1f}s")

        # Summary
        sel_raw = sel_data["selectivity_raw"]
        sel_aligned = sel_data["selectivity_aligned"]
        valid = sel_data["valid_positions"]

        # Only summarize over valid positions
        raw_vals = sel_raw[:, valid].flatten()
        aligned_vals = sel_aligned[:, valid].flatten()

        model_result = {
            "selectivity_raw_grid": sel_raw.tolist(),
            "selectivity_aligned_grid": sel_aligned.tolist(),
            "matched_accuracy_grid": sel_data["matched_accuracy"].tolist(),
            "max_cross_accuracy_grid": sel_data["max_cross_accuracy"].tolist(),
            "cross_accuracies": sel_data["cross_accuracies"],
            "n_per_position": sel_data["n_per_position"],
            "valid_positions": valid,
            "summary_raw": {
                "mean": float(np.mean(raw_vals)),
                "std": float(np.std(raw_vals)),
                "max": float(np.max(raw_vals)),
                "min": float(np.min(raw_vals)),
                "median": float(np.median(raw_vals)),
                "cells_above_0001": int(np.sum(raw_vals > 0.001)),
                "cells_above_001": int(np.sum(raw_vals > 0.01)),
                "cells_above_005": int(np.sum(raw_vals > 0.05)),
                "cells_above_010": int(np.sum(raw_vals > 0.10)),
                "n_cells": len(raw_vals),
            },
            "summary_aligned": {
                "mean": float(np.mean(aligned_vals)),
                "std": float(np.std(aligned_vals)),
                "max": float(np.max(aligned_vals)),
                "min": float(np.min(aligned_vals)),
                "median": float(np.median(aligned_vals)),
                "cells_above_0001": int(np.sum(aligned_vals > 0.001)),
                "cells_above_001": int(np.sum(aligned_vals > 0.01)),
                "cells_above_005": int(np.sum(aligned_vals > 0.05)),
                "cells_above_010": int(np.sum(aligned_vals > 0.10)),
                "n_cells": len(aligned_vals),
            },
        }

        results[model_name] = model_result

        print(f"\n  {model_name} selectivity summary (RAW: matched_full - max_cross_pairwise):")
        for k, v in model_result["summary_raw"].items():
            print(f"    {k}: {v}")
        print(f"\n  {model_name} selectivity summary (ALIGNED: same sample size):")
        for k, v in model_result["summary_aligned"].items():
            print(f"    {k}: {v}")

        # Print the selectivity grids
        print(f"\n  RAW selectivity grid (3 decimal places):")
        for layer in range(NUM_LAYERS):
            row = [f"{sel_raw[layer, t]:+.3f}" for t in range(NUM_THOUGHTS)]
            print(f"    L{layer:2d}: {row}")

        print(f"\n  ALIGNED selectivity grid (3 decimal places):")
        for layer in range(NUM_LAYERS):
            row = [f"{sel_aligned[layer, t]:+.3f}" for t in range(NUM_THOUGHTS)]
            print(f"    L{layer:2d}: {row}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Interpretation
    m3_raw_max = results["m3"]["summary_raw"]["max"]
    m5_raw_max = results["m5"]["summary_raw"]["max"]
    m3_aligned_max = results["m3"]["summary_aligned"]["max"]
    m5_aligned_max = results["m5"]["summary_aligned"]["max"]

    if max(m3_aligned_max, m5_aligned_max) < 0.05:
        results["outcome"] = "A"
        results["interpretation"] = (
            "Selectivity remains near zero even with correct pairwise alignment. "
            "The original finding holds: thought positions broadcast a general problem "
            "representation rather than encoding step-specific reasoning information. "
            "The n_common=12 bug existed but did not affect the conclusion."
        )
    else:
        results["outcome"] = "B"
        results["interpretation"] = (
            f"Selectivity is meaningfully positive (max aligned: M3={m3_aligned_max:.3f}, "
            f"M5={m5_aligned_max:.3f}). The paper's probing argument needs revision. "
            "STOP and report to human before editing manuscript."
        )

    # Save
    out_path = os.path.join(args.output_dir, "selectivity_recomputed.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nOUTCOME: {results['outcome']}")
    print(results["interpretation"])


if __name__ == "__main__":
    main()
