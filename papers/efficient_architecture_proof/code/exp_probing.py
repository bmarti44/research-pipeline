"""
Experiment: Representation Probing

Probes hidden states at each (layer, thought_position) to predict the
intermediate reasoning step. Tests whether COCONUT models encode structured
intermediate reasoning states aligned with the ground-truth reasoning path.

Method:
  1. Extract hidden states at every layer (0-12) and thought position (0..T-1)
  2. Probe target: type mentioned in steps[t] (multi-class classification)
  3. Linear probe: LogisticRegression with 5-fold CV
  4. Permutation test: 10000 shuffles, Bonferroni correction

Usage:
    python exp_probing.py \
        --checkpoint_dir /path/to/results/v9_meta_fork \
        --data data/prosqa_test.json \
        --num_samples 500 \
        --output_dir /path/to/experiments/probing/ \
        --seed 0

Deployed to: /lambda/nfs/experiment/code/v9_meta_fork/
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COCONUT_MODELS = ["m3", "m4", "m4b", "m5"]
NUM_THOUGHTS = 6
NUM_LAYERS = 13  # GPT-2: 12 transformer layers + 1 embedding layer (indices 0-12)
HIDDEN_DIM = 768
N_FOLDS = 5
N_PERMUTATIONS = 10000
PROBE_C = 1.0
PROBE_MAX_ITER = 1000


# ---------------------------------------------------------------------------
# Extract hidden states and labels for all samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_probing_data(model, tokenizer, model_info, data, num_samples,
                          num_thoughts, device):
    """
    Extract hidden states at thought positions for all samples.

    Returns:
        hidden_states_by_pos: dict mapping thought_position -> np.array [n_samples, n_layers, hidden_dim]
        labels_by_pos: dict mapping thought_position -> list of str labels
        sample_indices: list of sample indices that were successfully processed
    """
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

    # Collect hidden states per thought position
    # Each entry: [n_samples, n_layers, hidden_dim]
    hidden_by_pos = {t: [] for t in range(num_thoughts)}
    labels_by_pos = {t: [] for t in range(num_thoughts)}
    sample_indices = []

    n_processed = 0
    n_skipped = 0

    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        steps = sample.get("steps", [])
        step_types = get_step_types(sample)

        if len(steps) == 0:
            n_skipped += 1
            continue

        input_ids = prepare_input(
            sample, tokenizer, model_info,
            num_thoughts=num_thoughts,
            device=device,
        )

        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]

        if len(thought_positions) == 0:
            n_skipped += 1
            continue

        # Get hidden states: [n_layers+1, seq_len, hidden_dim]
        all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)

        # Check for NaN/Inf
        if not torch.isfinite(all_hidden).all():
            print(f"    WARNING: Non-finite hidden states at sample {idx}, skipping")
            n_skipped += 1
            continue

        # Extract at each thought position
        actual_T = len(thought_positions)
        for t in range(min(actual_T, num_thoughts)):
            pos = thought_positions[t]

            # Hidden state at this position: [n_layers, hidden_dim]
            h = all_hidden[:, pos, :].cpu().float().numpy()  # Cast to float32 for sklearn

            # Label: type at step t, or answer type if t >= len(steps)
            if t < len(step_types):
                label = step_types[t]
            else:
                # Use the answer type (last word of answer)
                answer = sample.get("answer", "")
                answer_clean = answer.rstrip().rstrip(".")
                words = answer_clean.split()
                label = words[-1] if words else "UNKNOWN"

            hidden_by_pos[t].append(h)
            labels_by_pos[t].append(label)

        sample_indices.append(idx)
        n_processed += 1

        if (n_processed) % 100 == 0:
            print(f"    Extracted: {n_processed}/{min(num_samples, len(data))} "
                  f"(skipped {n_skipped})")

    print(f"  Total extracted: {n_processed}, skipped: {n_skipped}")

    # Convert to numpy arrays
    for t in range(num_thoughts):
        if len(hidden_by_pos[t]) > 0:
            hidden_by_pos[t] = np.stack(hidden_by_pos[t], axis=0)  # [n_samples, n_layers, hidden_dim]
        else:
            hidden_by_pos[t] = None

    return hidden_by_pos, labels_by_pos, sample_indices


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def run_linear_probe(X, y, n_folds=N_FOLDS, C=PROBE_C, max_iter=PROBE_MAX_ITER):
    """
    Train a linear probe and return cross-validated accuracy.

    Uses RidgeClassifier (closed-form solution) for speed. With 38+ classes
    and 768 features, iterative solvers (LogisticRegression/lbfgs) take
    10-30s per fit, making the experiment intractable. RidgeClassifier
    gives equivalent linear-separability results in milliseconds.
    """
    if len(np.unique(y)) < 2:
        return 0.0

    # Ensure enough samples per class for stratified CV
    min_class_count = min(np.bincount(y))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(alpha=1.0 / C, random_state=42)),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    return float(np.mean(scores))


def run_nonlinear_probe(X, y, n_folds=N_FOLDS, max_iter=PROBE_MAX_ITER):
    """
    Train a nonlinear probe (2-layer MLP: 768 → 256 → num_classes, ReLU)
    and return cross-validated accuracy.

    Uses StandardScaler + MLPClassifier pipeline for stable training.
    """
    if len(np.unique(y)) < 2:
        return 0.0

    min_class_count = min(np.bincount(y))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256,),
            activation='relu',
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    return float(np.mean(scores))


def run_permutation_test(X, y, actual_accuracy, n_permutations=N_PERMUTATIONS,
                          n_folds=N_FOLDS, C=PROBE_C, max_iter=PROBE_MAX_ITER):
    """
    Permutation test: shuffle labels and re-run probe.
    Returns p-value (fraction of permuted accuracies >= actual).

    Uses a single 80/20 stratified split (not full CV) per permutation for
    computational tractability: 10000 perms × 5-fold CV × 78 cells × 3 models
    would require ~11.7M model fits. Single-split reduces this by 5×.
    The actual accuracy is still computed with full 5-fold CV.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if len(np.unique(y)) < 2:
        return 1.0

    count_geq = 0
    rng = np.random.RandomState(42)

    # Use a fixed single split for all permutations (ensures comparability)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    scaler = StandardScaler().fit(X[train_idx])
    X_train_s = scaler.transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)

        clf = RidgeClassifier(alpha=1.0 / C, random_state=42)

        # Check if test set has all classes represented
        if len(np.unique(y_perm[train_idx])) < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_s, y_perm[train_idx])
            perm_acc = clf.score(X_test_s, y_perm[test_idx])

        if perm_acc >= actual_accuracy:
            count_geq += 1

    # Add 1 to both numerator and denominator for conservative estimate
    p_value = (count_geq + 1) / (n_permutations + 1)
    return p_value


# ---------------------------------------------------------------------------
# Selectivity metric
# ---------------------------------------------------------------------------

def compute_selectivity(hidden_by_pos, labels_by_pos, num_layers, num_thoughts):
    """
    Compute selectivity metric: how specifically does position t encode step t vs other steps?

    For each (layer, thought_pos t):
      1. Train probe on h(l,t) to predict step_t labels -> acc_match
      2. For each s != t: train probe on h(l,t) to predict step_s labels -> acc_cross
      3. selectivity(l,t) = acc_match - max(acc_cross for s != t)

    Only uses samples that appear in BOTH position t's and position s's data,
    so labels are aligned across positions.

    Returns:
        selectivity_matrix: np.array [num_layers, num_thoughts]
        cross_accuracies: dict mapping (t, s) -> np.array [num_layers] of probe accuracies
    """
    # Build per-position sample index sets for alignment.
    # labels_by_pos[t] has one entry per sample that was successfully extracted
    # at position t. The i-th entry corresponds to the i-th sample in
    # hidden_by_pos[t]. Since extract_probing_data appends in order, sample i
    # in position t is the same underlying data point as sample i in position s
    # *when* both positions have at least i+1 samples.  We align by taking the
    # common prefix (minimum length).

    # Determine valid positions
    valid_positions = [
        t for t in range(num_thoughts)
        if hidden_by_pos[t] is not None and len(labels_by_pos[t]) >= 10
    ]

    # Find the common sample count across ALL valid positions
    if len(valid_positions) < 2:
        # Not enough positions to compute cross-position metrics
        return np.zeros((num_layers, num_thoughts)), {}

    n_common = min(len(labels_by_pos[t]) for t in valid_positions)
    if n_common < 10:
        return np.zeros((num_layers, num_thoughts)), {}

    # Pre-encode labels for each position using only the common samples
    encoded_labels = {}
    label_encoders = {}
    for t in valid_positions:
        le = LabelEncoder()
        encoded_labels[t] = le.fit_transform(labels_by_pos[t][:n_common])
        label_encoders[t] = le

    # Compute cross-position probe accuracies
    cross_accuracies = {}  # (t, s) -> np.array [num_layers]

    for t in valid_positions:
        for s in valid_positions:
            if t == s:
                continue

            accs = np.zeros(num_layers)
            y_cross = encoded_labels[s]  # Labels from position s

            # Need at least 2 classes in the cross labels
            if len(np.unique(y_cross)) < 2:
                cross_accuracies[(t, s)] = accs
                continue

            for layer in range(num_layers):
                X = hidden_by_pos[t][:n_common, layer, :]
                accs[layer] = run_linear_probe(X, y_cross)

            cross_accuracies[(t, s)] = accs

    # Compute selectivity matrix
    selectivity_matrix = np.zeros((num_layers, num_thoughts))

    for t in valid_positions:
        # Matched accuracy: probe h(l,t) to predict step_t labels
        y_match = encoded_labels[t]
        if len(np.unique(y_match)) < 2:
            continue

        for layer in range(num_layers):
            X = hidden_by_pos[t][:n_common, layer, :]
            acc_match = run_linear_probe(X, y_match)

            # Max cross-position accuracy for this (layer, t)
            max_cross = 0.0
            for s in valid_positions:
                if s == t and (t, s) not in cross_accuracies:
                    continue
                if (t, s) in cross_accuracies:
                    max_cross = max(max_cross, cross_accuracies[(t, s)][layer])

            selectivity_matrix[layer, t] = acc_match - max_cross

    return selectivity_matrix, cross_accuracies


# ---------------------------------------------------------------------------
# Input-position control probing
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_input_probing_data(model, tokenizer, model_info, data, num_samples,
                                num_thoughts, device):
    """
    Extract hidden states at INPUT token positions (before thought tokens)
    for comparison with thought-position probes.

    Input positions are the graph fact tokens in the question (everything before
    <|start-latent|>).  We select up to `num_thoughts` evenly-spaced input
    positions so the resulting arrays are directly comparable to the
    thought-position arrays.

    Returns:
        input_hidden_by_pos: dict mapping pos_idx -> np.array [n_samples, n_layers, hidden_dim]
            where pos_idx in range(num_thoughts)
        sample_count: int, number of samples successfully extracted
    """
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

    input_hidden_by_pos = {t: [] for t in range(num_thoughts)}
    n_processed = 0
    n_skipped = 0

    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        steps = sample.get("steps", [])

        if len(steps) == 0:
            n_skipped += 1
            continue

        input_ids = prepare_input(
            sample, tokenizer, model_info,
            num_thoughts=num_thoughts,
            device=device,
        )

        tokens = input_ids[0].tolist()

        # Find input token region: everything before the first latent token
        first_latent = None
        for i, t in enumerate(tokens):
            if t == latent_id:
                first_latent = i
                break

        if first_latent is None or first_latent < num_thoughts:
            # Not enough input tokens to select num_thoughts positions
            n_skipped += 1
            continue

        # Select num_thoughts evenly-spaced positions from input region
        # Skip position 0 (typically BOS/padding) by starting at 1
        input_region = list(range(1, first_latent))
        if len(input_region) < num_thoughts:
            n_skipped += 1
            continue

        indices = np.linspace(0, len(input_region) - 1, num_thoughts, dtype=int)
        selected_positions = [input_region[i] for i in indices]

        # Get hidden states: [n_layers, seq_len, hidden_dim]
        all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)

        if not torch.isfinite(all_hidden).all():
            n_skipped += 1
            continue

        for t_idx, pos in enumerate(selected_positions):
            h = all_hidden[:, pos, :].cpu().float().numpy()  # [n_layers, hidden_dim]
            input_hidden_by_pos[t_idx].append(h)

        n_processed += 1

    print(f"  Input-position extraction: {n_processed} samples, skipped {n_skipped}")

    # Convert to numpy arrays
    for t in range(num_thoughts):
        if len(input_hidden_by_pos[t]) > 0:
            input_hidden_by_pos[t] = np.stack(input_hidden_by_pos[t], axis=0)
        else:
            input_hidden_by_pos[t] = None

    return input_hidden_by_pos, n_processed


# ---------------------------------------------------------------------------
# Synthetic verification test
# ---------------------------------------------------------------------------

def synthetic_probe_test():
    """
    Create random separable data in 768-dim, train probe, expect >99% accuracy.
    This verifies the probing pipeline is working correctly.
    """
    rng = np.random.RandomState(42)
    n_classes = 10
    n_per_class = 50
    dim = HIDDEN_DIM

    # Create well-separated clusters
    X = []
    y = []
    for c in range(n_classes):
        center = rng.randn(dim) * 10  # Well-separated centers
        points = center + rng.randn(n_per_class, dim) * 0.1  # Tight clusters
        X.append(points)
        y.extend([c] * n_per_class)

    X = np.vstack(X)
    y = np.array(y)

    acc = run_linear_probe(X, y, n_folds=5)
    print(f"  Synthetic probe test: accuracy = {acc:.4f} (expect > 0.99)")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Representation Probing Experiment")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="Path to prosqa_test.json")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_thoughts", type=int, default=NUM_THOUGHTS)
    parser.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all in COCONUT_MODELS)")
    parser.add_argument("--linear_only", action="store_true",
                        help="Skip nonlinear (MLP) probes for faster execution")
    args = parser.parse_args()

    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
    else:
        models_to_run = list(COCONUT_MODELS)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("REPRESENTATION PROBING EXPERIMENT")
    print("=" * 70)
    print(f"Checkpoint dir:  {args.checkpoint_dir}")
    print(f"Data:            {args.data}")
    print(f"Num samples:     {args.num_samples}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Seed:            {args.seed}")
    print(f"Num thoughts:    {args.num_thoughts}")
    print(f"N permutations:  {args.n_permutations}")
    print()

    # Load data
    print("Loading data...")
    data = load_data(args.data)
    print(f"  Loaded {len(data)} samples")

    # Run synthetic verification first
    print("\nRunning synthetic probe verification...")
    synthetic_acc = synthetic_probe_test()

    # Bonferroni threshold: p < 0.05 / (num_layers * T)
    T = args.num_thoughts
    bonferroni_threshold = 0.05 / (NUM_LAYERS * T)
    print(f"Bonferroni threshold: p < {bonferroni_threshold:.6f}")

    results = {}
    verification = {
        "synthetic_probe_accuracy": round(synthetic_acc, 4),
        "hidden_state_shape": [HIDDEN_DIM],
        "all_finite": True,  # Will be updated if any non-finite found
        "label_alignment_check": "PASSED",  # Will be updated if checks fail
    }

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer, model_info = load_model_by_name(
                model_name, args.checkpoint_dir, device=device
            )
        except FileNotFoundError as e:
            print(f"  SKIPPING {model_name}: {e}")
            continue

        # Extract hidden states
        print(f"  Extracting hidden states from {args.num_samples} samples...")
        t0 = time.time()

        hidden_by_pos, labels_by_pos, sample_indices = extract_probing_data(
            model, tokenizer, model_info, data,
            num_samples=args.num_samples,
            num_thoughts=args.num_thoughts,
            device=device,
        )

        elapsed = time.time() - t0
        print(f"  Extraction done in {elapsed:.1f}s")

        # Verify hidden state dimensions and finiteness
        for t in range(T):
            if hidden_by_pos[t] is not None:
                shape = hidden_by_pos[t].shape
                if shape[-1] != HIDDEN_DIM:
                    print(f"    WARNING: Hidden dim mismatch at pos {t}: {shape[-1]} != {HIDDEN_DIM}")
                    verification["hidden_state_shape"] = [int(shape[-1])]
                if not np.isfinite(hidden_by_pos[t]).all():
                    print(f"    WARNING: Non-finite values at pos {t}")
                    verification["all_finite"] = False

        # Verify label alignment: check that probe labels match ground-truth paths
        label_alignment_ok = True
        for t in range(min(T, 3)):  # Check first 3 positions
            if len(labels_by_pos[t]) > 0:
                # Verify against raw data for first few samples
                for check_idx in range(min(5, len(sample_indices))):
                    sample_idx = sample_indices[check_idx]
                    sample = data[sample_idx]
                    step_types = get_step_types_safe(sample)
                    if t < len(step_types):
                        expected = step_types[t]
                        actual = labels_by_pos[t][check_idx] if check_idx < len(labels_by_pos[t]) else None
                        if actual is not None and actual != expected:
                            print(f"    WARNING: Label mismatch at pos {t}, sample {check_idx}: "
                                  f"expected '{expected}', got '{actual}'")
                            label_alignment_ok = False

        if not label_alignment_ok:
            verification["label_alignment_check"] = "FAILED"

        # Run probing at each (layer, position) cell
        print(f"  Running linear + nonlinear probes ({NUM_LAYERS} layers x {T} positions)...")
        t0 = time.time()

        probe_accuracy = np.zeros((NUM_LAYERS, T))
        nonlinear_accuracy = np.zeros((NUM_LAYERS, T))
        p_values = np.ones((NUM_LAYERS, T))
        significant = np.zeros((NUM_LAYERS, T), dtype=bool)
        n_samples_per_pos = {}

        for t in range(T):
            if hidden_by_pos[t] is None or len(labels_by_pos[t]) < 10:
                print(f"    Position {t}: insufficient data, skipping")
                n_samples_per_pos[t] = 0 if hidden_by_pos[t] is None else len(labels_by_pos[t])
                continue

            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(labels_by_pos[t])
            n_classes = len(le.classes_)
            n_samples = len(y)
            n_samples_per_pos[t] = n_samples

            print(f"    Position {t}: {n_samples} samples, {n_classes} classes")

            for layer in range(NUM_LAYERS):
                # X: [n_samples, hidden_dim]
                X = hidden_by_pos[t][:, layer, :]

                # Linear probe
                acc = run_linear_probe(X, y)
                probe_accuracy[layer, t] = acc

                # Nonlinear probe (MLP: 768 → 256 → num_classes)
                if not args.linear_only:
                    nl_acc = run_nonlinear_probe(X, y)
                    nonlinear_accuracy[layer, t] = nl_acc

                # Permutation test (on linear probe — the primary measure)
                if acc > 0:
                    p_val = run_permutation_test(
                        X, y, acc,
                        n_permutations=args.n_permutations,
                    )
                    p_values[layer, t] = p_val
                    significant[layer, t] = (p_val < bonferroni_threshold)

            # Print progress per position
            best_layer = np.argmax(probe_accuracy[:, t])
            best_acc = probe_accuracy[best_layer, t]
            best_nl_layer = np.argmax(nonlinear_accuracy[:, t])
            best_nl_acc = nonlinear_accuracy[best_nl_layer, t]
            print(f"      Linear:    layer {best_layer}, acc={best_acc:.4f}, "
                  f"sig={significant[best_layer, t]}")
            print(f"      Nonlinear: layer {best_nl_layer}, acc={best_nl_acc:.4f}")

        elapsed = time.time() - t0
        print(f"  Probing done in {elapsed:.1f}s")

        # ----- Selectivity analysis -----
        print(f"  Computing selectivity metric...")
        t0_sel = time.time()
        selectivity, cross_accs = compute_selectivity(
            hidden_by_pos, labels_by_pos, NUM_LAYERS, T
        )
        elapsed_sel = time.time() - t0_sel
        print(f"  Selectivity done in {elapsed_sel:.1f}s")

        positive_sel = selectivity[selectivity > 0]
        mean_sel = float(np.mean(positive_sel)) if len(positive_sel) > 0 else 0.0

        # Cross-position leakage: can h(l, t=0) predict the last step?
        last_valid_t = max(
            (t for t in range(T)
             if hidden_by_pos[t] is not None and len(labels_by_pos[t]) >= 10),
            default=None,
        )
        max_cross_pos_acc = 0.0
        cross_leakage_front_to_back = 0.0
        if last_valid_t is not None and last_valid_t > 0:
            if (0, last_valid_t) in cross_accs:
                cross_leakage_front_to_back = float(np.max(cross_accs[(0, last_valid_t)]))
            # Overall max cross-position accuracy across all (t, s) pairs
            for key, acc_arr in cross_accs.items():
                max_cross_pos_acc = max(max_cross_pos_acc, float(np.max(acc_arr)))

        print(f"    Mean selectivity (positive cells): {mean_sel:.4f}")
        print(f"    Max cross-position probe acc:      {max_cross_pos_acc:.4f}")
        print(f"    Front-to-back leakage (t=0->t={last_valid_t}): {cross_leakage_front_to_back:.4f}")

        # ----- Input-position control -----
        print(f"  Running input-position control probes...")
        t0_inp = time.time()
        input_hidden_by_pos, input_n_samples = extract_input_probing_data(
            model, tokenizer, model_info, data,
            num_samples=args.num_samples,
            num_thoughts=args.num_thoughts,
            device=device,
        )

        # Run linear probes at input positions to predict each step's labels
        input_probe_acc = np.zeros((NUM_LAYERS, T))
        thought_minus_input = np.zeros((NUM_LAYERS, T))

        for t in range(T):
            if input_hidden_by_pos[t] is None:
                continue
            if hidden_by_pos[t] is None or len(labels_by_pos[t]) < 10:
                continue

            # Align sample counts: input extraction may have different count
            n_input = input_hidden_by_pos[t].shape[0]
            n_thought = len(labels_by_pos[t])
            n_aligned = min(n_input, n_thought)
            if n_aligned < 10:
                continue

            le_inp = LabelEncoder()
            y_inp = le_inp.fit_transform(labels_by_pos[t][:n_aligned])

            if len(np.unique(y_inp)) < 2:
                continue

            for layer in range(NUM_LAYERS):
                X_inp = input_hidden_by_pos[t][:n_aligned, layer, :]
                acc_inp = run_linear_probe(X_inp, y_inp)
                input_probe_acc[layer, t] = acc_inp
                thought_minus_input[layer, t] = probe_accuracy[layer, t] - acc_inp

        elapsed_inp = time.time() - t0_inp
        print(f"  Input-position control done in {elapsed_inp:.1f}s")

        # Summarize input-position comparison
        mean_thought_advantage = float(np.mean(thought_minus_input))
        max_input_acc = float(np.max(input_probe_acc))
        print(f"    Mean thought-over-input advantage: {mean_thought_advantage:.4f}")
        print(f"    Max input-position probe acc:      {max_input_acc:.4f}")

        # Find peak
        peak_idx = np.unravel_index(np.argmax(probe_accuracy), probe_accuracy.shape)
        peak_layer = int(peak_idx[0])
        peak_position = int(peak_idx[1])
        peak_acc = float(probe_accuracy[peak_layer, peak_position])

        # Check for diagonal pattern: peak accuracy at (layer, t) where layer
        # increases with t (information flows through layers progressively)
        diagonal_peaks = []
        for t in range(T):
            best_layer_t = int(np.argmax(probe_accuracy[:, t]))
            best_acc_t = float(probe_accuracy[best_layer_t, t])
            if best_acc_t > 0.05:  # Above chance
                diagonal_peaks.append(best_layer_t)

        diagonal_pattern = False
        if len(diagonal_peaks) >= 3:
            # Check if peak layers generally increase or concentrate in later layers
            diffs = [diagonal_peaks[i+1] - diagonal_peaks[i] for i in range(len(diagonal_peaks)-1)]
            diagonal_pattern = sum(d >= 0 for d in diffs) > len(diffs) * 0.5

        # Nonlinear probe peak
        nl_peak_idx = np.unravel_index(np.argmax(nonlinear_accuracy), nonlinear_accuracy.shape)
        nl_peak_layer = int(nl_peak_idx[0])
        nl_peak_position = int(nl_peak_idx[1])
        nl_peak_acc = float(nonlinear_accuracy[nl_peak_layer, nl_peak_position])

        # Detect "linear fails, nonlinear succeeds" pattern
        # For each cell, check if nonlinear > linear by a meaningful margin
        nl_advantage = nonlinear_accuracy - probe_accuracy
        nl_wins = np.sum(nl_advantage > 0.1)  # >10pp advantage

        model_results = {
            "linear_probe_accuracy": probe_accuracy.tolist(),
            "nonlinear_probe_accuracy": nonlinear_accuracy.tolist(),
            "permutation_p_values": p_values.tolist(),
            "significant_cells": significant.tolist(),
            "peak_layer": peak_layer,
            "peak_position": peak_position,
            "peak_accuracy": round(peak_acc, 4),
            "nonlinear_peak_layer": nl_peak_layer,
            "nonlinear_peak_position": nl_peak_position,
            "nonlinear_peak_accuracy": round(nl_peak_acc, 4),
            "nonlinear_advantage_cells": int(nl_wins),
            "diagonal_pattern": diagonal_pattern,
            "diagonal_peak_layers": diagonal_peaks,
            "n_samples_per_position": n_samples_per_pos,
            "n_significant_cells": int(significant.sum()),
            "total_cells": NUM_LAYERS * T,
            # Selectivity analysis
            "selectivity": selectivity.tolist(),
            "mean_selectivity": round(mean_sel, 4),
            "max_cross_position_acc": round(max_cross_pos_acc, 4),
            "cross_leakage_front_to_back": round(cross_leakage_front_to_back, 4),
            # Input-position control
            "input_position_probe_acc": input_probe_acc.tolist(),
            "thought_minus_input_advantage": thought_minus_input.tolist(),
            "mean_thought_advantage": round(mean_thought_advantage, 4),
            "max_input_position_acc": round(max_input_acc, 4),
        }

        results[model_name] = model_results

        # Save intermediate results
        intermediate_path = os.path.join(args.output_dir, f"{model_name}_probing.json")
        with open(intermediate_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved intermediate results to {intermediate_path}")

        # Print summary heatmap (text-based)
        print(f"\n  Probe accuracy heatmap for {model_name}:")
        print(f"  {'Layer':<8}" + "".join(f"{'t='+str(t):<8}" for t in range(T)))
        for layer in range(NUM_LAYERS):
            row = f"  {layer:<8}"
            for t in range(T):
                val = probe_accuracy[layer, t]
                marker = "*" if significant[layer, t] else " "
                row += f"{val:.3f}{marker} "
            print(row)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Assemble final results
    final_results = {**results, "verification": verification}

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PROBING RESULTS SUMMARY")
    print("=" * 70)

    for model_name in models_to_run:
        if model_name not in results:
            continue
        r = results[model_name]
        print(f"\n{model_name}:")
        print(f"  Linear peak:    layer {r['peak_layer']}, pos {r['peak_position']}, "
              f"acc {r['peak_accuracy']:.4f}")
        print(f"  Nonlinear peak: layer {r['nonlinear_peak_layer']}, pos {r['nonlinear_peak_position']}, "
              f"acc {r['nonlinear_peak_accuracy']:.4f}")
        print(f"  NL>Linear by >10pp: {r['nonlinear_advantage_cells']} cells")
        print(f"  Diagonal pattern: {r['diagonal_pattern']}")
        print(f"  Significant cells: {r['n_significant_cells']}/{r['total_cells']}")
        print(f"  Diagonal peak layers: {r['diagonal_peak_layers']}")
        # Selectivity
        print(f"  Mean selectivity:  {r['mean_selectivity']:.4f}")
        print(f"  Max cross-pos acc: {r['max_cross_position_acc']:.4f}")
        print(f"  Front-to-back leakage: {r['cross_leakage_front_to_back']:.4f}")
        # Input-position control
        print(f"  Mean thought-over-input advantage: {r['mean_thought_advantage']:.4f}")
        print(f"  Max input-position acc: {r['max_input_position_acc']:.4f}")

    print("\nVerification:")
    for k, v in verification.items():
        print(f"  {k}: {v}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Helper: safe version of get_step_types for verification
# ---------------------------------------------------------------------------

def get_step_types_safe(sample):
    """Same as get_step_types from exp_utils, inline for verification."""
    types = []
    for step in sample.get("steps", []):
        step_clean = step.rstrip().rstrip(".")
        words = step_clean.split()
        if words:
            types.append(words[-1])
        else:
            types.append("UNKNOWN")
    return types


if __name__ == "__main__":
    main()
