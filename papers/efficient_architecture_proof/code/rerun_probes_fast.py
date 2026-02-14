"""
Fast corrected probing: linear probes + permutation tests only.
MLP probes deferred to targeted follow-up.

Saves:
  - Corrected permutation p-values (2,000 perms per cell)
  - Linear probe accuracy grid
  - Complete results JSON
"""
import sys, os, json, time, warnings
sys.path.insert(0, "/lambda/nfs/experiment/code/v9_meta_fork")

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from exp_utils import (
    load_model_by_name, prepare_input, get_hidden_states,
    get_special_ids, get_step_types, load_data, set_seed,
)

# Config
NUM_THOUGHTS = 6
NUM_LAYERS = 13
N_FOLDS = 5
N_PERMUTATIONS = 2000  # min p = 1/2001 = 0.0005, below Bonferroni 0.05/78 = 0.000641
MODELS = ["m5"]  # m3 already done
CHECKPOINT_DIR = "/lambda/nfs/experiment/results/v9_meta_fork"
DATA_PATH = "/lambda/nfs/experiment/code/v9_meta_fork/data/prosqa_test.json"
OUTPUT_DIR = "/lambda/nfs/experiment/results/v9_meta_fork/experiments/probing_corrected"
NUM_SAMPLES = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_linear_probe(X, y, n_folds=N_FOLDS):
    if len(np.unique(y)) < 2:
        return 0.0
    min_class_count = min(np.bincount(y))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=1.0, random_state=42)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


def run_permutation_test(X, y, actual_accuracy, n_permutations=N_PERMUTATIONS):
    """Optimized permutation test using precomputed Cholesky decomposition.

    Since X is fixed across permutations, we precompute (X^T X + alpha*I)^{-1}
    via Cholesky factorization and reuse it for each permuted Y.
    """
    if len(np.unique(y)) < 2:
        return 1.0

    from scipy.linalg import cho_factor, cho_solve

    count_geq = 0
    rng = np.random.RandomState(42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    # Precompute: (X^T X + alpha*I) Cholesky — constant across perms
    XtX = X_train.T @ X_train  # (768, 768)
    XtX += np.eye(XtX.shape[0])  # alpha = 1.0
    cho = cho_factor(XtX)
    Xt = X_train.T  # (768, n_train)

    n_classes = len(np.unique(y))
    n_train = X_train.shape[0]

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        y_train = y_perm[train_idx]
        y_test = y_perm[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        # One-hot encode targets
        Y_train = np.zeros((n_train, n_classes))
        for i, cls in enumerate(y_train):
            Y_train[i, cls] = 1.0

        # Solve: W = (X^T X + aI)^{-1} X^T Y
        XtY = Xt @ Y_train  # (768, n_classes)
        W = cho_solve(cho, XtY)  # (768, n_classes)

        # Predict
        scores = X_test @ W  # (n_test, n_classes)
        preds = np.argmax(scores, axis=1)
        perm_acc = np.mean(preds == y_test)

        if perm_acc >= actual_accuracy:
            count_geq += 1

    return (count_geq + 1) / (n_permutations + 1)


# ---- Main ----
if __name__ == "__main__":
    set_seed(0)
    device = "cuda"
    data = load_data(DATA_PATH)
    print("Loaded %d samples" % len(data))

    bonferroni_threshold = 0.05 / (NUM_LAYERS * NUM_THOUGHTS)
    print("Bonferroni threshold: %.6f" % bonferroni_threshold)

    all_results = {}

    for model_name in MODELS:
        print("\n" + "=" * 60)
        print("Model: %s" % model_name)
        print("=" * 60)

        # Check for cached hidden states
        cache_path = os.path.join(OUTPUT_DIR, "%s_hidden_states.npz" % model_name)
        labels_cache = os.path.join(OUTPUT_DIR, "%s_labels.json" % model_name)

        if os.path.exists(cache_path) and os.path.exists(labels_cache):
            print("  Loading cached hidden states...")
            cached = np.load(cache_path, allow_pickle=True)
            hidden_by_pos = {}
            for k in cached.files:
                arr = cached[k]
                if arr.shape == ():
                    hidden_by_pos[int(k)] = None
                else:
                    hidden_by_pos[int(k)] = arr
            with open(labels_cache) as f:
                labels_by_pos = {int(k): v for k, v in json.load(f).items()}
        else:
            print("  Extracting hidden states (GPU)...")
            model, tokenizer, model_info = load_model_by_name(
                model_name, CHECKPOINT_DIR, device=device)
            special_ids = get_special_ids(tokenizer)
            latent_id = special_ids["latent_id"]

            hidden_by_pos = {t: [] for t in range(NUM_THOUGHTS)}
            labels_by_pos = {t: [] for t in range(NUM_THOUGHTS)}

            t0 = time.time()
            for idx in range(min(NUM_SAMPLES, len(data))):
                sample = data[idx]
                steps = sample.get("steps", [])
                step_types = get_step_types(sample)
                if len(steps) == 0:
                    continue
                input_ids = prepare_input(
                    sample, tokenizer, model_info,
                    num_thoughts=NUM_THOUGHTS, device=device)
                tokens = input_ids[0].tolist()
                thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
                if len(thought_positions) == 0:
                    continue
                with torch.no_grad():
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
                        words = answer.rstrip().rstrip(".").split()
                        label = words[-1] if words else "UNKNOWN"
                    hidden_by_pos[t].append(h)
                    labels_by_pos[t].append(label)
                if (idx + 1) % 100 == 0:
                    print("    %d/%d" % (idx + 1, NUM_SAMPLES))

            print("  Extraction done in %.1fs" % (time.time() - t0))

            # Convert and save
            save_dict = {}
            for t in range(NUM_THOUGHTS):
                if len(hidden_by_pos[t]) > 0:
                    hidden_by_pos[t] = np.stack(hidden_by_pos[t], axis=0)
                    save_dict[str(t)] = hidden_by_pos[t]
                else:
                    hidden_by_pos[t] = None
                    save_dict[str(t)] = np.array(None)
            np.savez_compressed(cache_path, **save_dict)
            with open(labels_cache, "w") as f:
                json.dump({str(k): v for k, v in labels_by_pos.items()}, f)
            print("  Saved hidden states to %s" % cache_path)

            del model
            torch.cuda.empty_cache()

        # ---- Run linear probes + permutation tests ----
        print("  Running linear probes + permutation tests...")
        t0_all = time.time()
        probe_accuracy = np.zeros((NUM_LAYERS, NUM_THOUGHTS))
        p_values = np.ones((NUM_LAYERS, NUM_THOUGHTS))
        significant = np.zeros((NUM_LAYERS, NUM_THOUGHTS), dtype=bool)
        n_samples_per_pos = {}

        for t in range(NUM_THOUGHTS):
            if hidden_by_pos[t] is None or len(labels_by_pos[t]) < 10:
                n_samples_per_pos[t] = 0
                print("    Position %d: insufficient data, skipping" % t)
                continue

            le = LabelEncoder()
            y = le.fit_transform(labels_by_pos[t])
            n_classes = len(le.classes_)
            n_samples = len(y)
            n_samples_per_pos[t] = n_samples
            print("    Position %d: n=%d, %d classes" % (t, n_samples, n_classes))

            for layer in range(NUM_LAYERS):
                t0_cell = time.time()
                X = hidden_by_pos[t][:, layer, :]

                # Linear probe (5-fold CV)
                acc = run_linear_probe(X, y)
                probe_accuracy[layer, t] = acc

                # Permutation test (2,000 perms with Cholesky optimization)
                if acc > 0:
                    p_val = run_permutation_test(X, y, acc)
                    p_values[layer, t] = p_val
                    significant[layer, t] = (p_val < bonferroni_threshold)

                elapsed_cell = time.time() - t0_cell
                sig_str = " ***" if significant[layer, t] else ""
                print("      L%d: acc=%.4f, p=%.6f (%.1fs)%s" % (
                    layer, acc, p_values[layer, t], elapsed_cell, sig_str))

            best_layer = int(np.argmax(probe_accuracy[:, t]))
            best_p = p_values[best_layer, t]
            n_sig_t = int(significant[:, t].sum())
            print("    -> Peak: layer %d, acc=%.4f, p=%.6f, sig=%d/%d" % (
                best_layer, probe_accuracy[best_layer, t], best_p, n_sig_t, NUM_LAYERS))

        elapsed_all = time.time() - t0_all
        print("  All probes done in %.1fs (%.1f min)" % (elapsed_all, elapsed_all / 60))

        model_results = {
            "linear_probe_accuracy": probe_accuracy.tolist(),
            "permutation_p_values": p_values.tolist(),
            "significant_cells": significant.tolist(),
            "n_significant_cells": int(significant.sum()),
            "bonferroni_threshold": bonferroni_threshold,
            "n_samples_per_position": n_samples_per_pos,
            "n_permutations": N_PERMUTATIONS,
            "peak_layer": int(np.unravel_index(
                np.argmax(probe_accuracy), probe_accuracy.shape)[0]),
            "peak_position": int(np.unravel_index(
                np.argmax(probe_accuracy), probe_accuracy.shape)[1]),
            "peak_accuracy": round(float(np.max(probe_accuracy)), 4),
        }

        all_results[model_name] = model_results

        # Save per-model
        outpath = os.path.join(OUTPUT_DIR, "%s_linear_perm.json" % model_name)
        with open(outpath, "w") as f:
            json.dump(model_results, f, indent=2)
        print("  Saved %s results to %s" % (model_name, outpath))

    # Save combined
    combined_path = os.path.join(OUTPUT_DIR, "results_linear_perm.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("CORRECTED LINEAR PROBE + PERMUTATION RESULTS")
    print("=" * 70)
    for mn in MODELS:
        if mn not in all_results:
            continue
        r = all_results[mn]
        print("\n%s:" % mn)
        print("  Peak linear: layer %d, pos %d, acc=%.4f" % (
            r["peak_layer"], r["peak_position"], r["peak_accuracy"]))
        print("  Significant cells (Bonferroni): %d/%d" % (
            r["n_significant_cells"], NUM_LAYERS * NUM_THOUGHTS))

        pvals = np.array(r["permutation_p_values"])
        sig = np.array(r["significant_cells"])
        if sig.any():
            print("  Significant cell locations:")
            for idx in zip(*np.where(sig)):
                layer, pos = idx
                print("    layer=%d, pos=%d: acc=%.4f, p=%.6f" % (
                    layer, pos,
                    r["linear_probe_accuracy"][layer][pos],
                    pvals[layer][pos]))
        else:
            print("  No significant cells after Bonferroni correction")

    print("\nDone.")
