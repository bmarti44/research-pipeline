#!/usr/bin/env python3
"""MLP probe grid search on cached hidden states.

Uses pre-extracted hidden states (no GPU needed).
Runs grid search over learning rate, hidden size, and regularization
for the top 5 cells where MLP probes previously showed 0/78 significant.

If MLPs still show no advantage over linear at tuned hyperparams,
that's a definitive null result for Appendix A.7.
"""

import json
import os
import numpy as np
import warnings
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBING_DIR = os.path.join(SCRIPT_DIR, "..", "results", "experiments", "probing_corrected")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "experiments")

# Top 5 cells to probe (highest linear probe accuracy)
TARGET_CELLS = [
    ("m3", 0, 3),   # M3, layer 0, position 3: linear acc 55.4%
    ("m3", 12, 2),  # M3, layer 12, position 2: strong mid-layer signal
    ("m5", 12, 3),  # M5, layer 12, position 3: linear acc 57.0%
    ("m5", 8, 0),   # M5, layer 8, position 0: distributed signal
    ("m5", 12, 2),  # M5, layer 12, position 2: cross-position control
]

# Grid search parameters
GRID = {
    "hidden_sizes": [(64,), (128,), (256,), (512,), (128, 64), (256, 128)],
    "learning_rates": [1e-4, 1e-3, 1e-2],
    "alphas": [1e-4, 1e-3, 1e-2, 1e-1],  # L2 regularization
}

N_FOLDS = 5


def load_probing_data(model_name):
    """Load cached hidden states and labels."""
    hs = np.load(f"{PROBING_DIR}/{model_name}_hidden_states.npz")
    labels = json.load(open(f"{PROBING_DIR}/{model_name}_labels.json"))
    return hs, labels


def get_cell_data(hs, labels, layer, position):
    """Extract X, y for a specific (layer, position) cell."""
    pos_key = str(position)
    if pos_key not in hs:
        return None, None
    
    X_all = hs[pos_key]  # shape: [n_samples, n_layers, hidden_dim]
    y_raw = labels[pos_key]
    
    if X_all is None or len(y_raw) == 0:
        return None, None
    
    X = X_all[:, layer, :]  # [n_samples, hidden_dim]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    return X, y


def run_linear_baseline(X, y):
    """Run linear probe (RidgeClassifier) as baseline."""
    if len(np.unique(y)) < 2:
        return 0.0
    
    min_class_count = min(np.bincount(y))
    actual_folds = min(N_FOLDS, min_class_count)
    if actual_folds < 2:
        return 0.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=1.0, random_state=42)),
    ])
    
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    
    return float(np.mean(scores))


def run_mlp_config(X, y, hidden_sizes, lr, alpha):
    """Run a single MLP configuration with cross-validation."""
    if len(np.unique(y)) < 2:
        return 0.0
    
    min_class_count = min(np.bincount(y))
    actual_folds = min(N_FOLDS, min_class_count)
    if actual_folds < 2:
        return 0.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            alpha=alpha,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            batch_size=min(64, len(y)),
        )),
    ])
    
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    
    return float(np.mean(scores))


def main():
    print("=" * 70)
    print("MLP PROBE GRID SEARCH")
    print("=" * 70)
    print(f"Target cells: {len(TARGET_CELLS)}")
    print(f"Grid size per cell: {len(GRID['hidden_sizes'])} x {len(GRID['learning_rates'])} x {len(GRID['alphas'])} = "
          f"{len(GRID['hidden_sizes']) * len(GRID['learning_rates']) * len(GRID['alphas'])} configs")
    print()
    
    # Load data
    print("Loading cached hidden states...")
    m3_hs, m3_labels = load_probing_data("m3")
    m5_hs, m5_labels = load_probing_data("m5")
    data = {"m3": (m3_hs, m3_labels), "m5": (m5_hs, m5_labels)}
    print("  Loaded.")
    
    results = []
    
    for cell_idx, (model_name, layer, position) in enumerate(TARGET_CELLS):
        print(f"\n{'='*60}")
        print(f"Cell {cell_idx+1}/{len(TARGET_CELLS)}: {model_name} layer={layer} pos={position}")
        print(f"{'='*60}")
        
        hs, labels = data[model_name]
        X, y = get_cell_data(hs, labels, layer, position)
        
        if X is None:
            print(f"  SKIP: no data for this cell")
            results.append({
                "model": model_name, "layer": layer, "position": position,
                "error": "no data"
            })
            continue
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        print(f"  n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}")
        
        # Linear baseline
        linear_acc = run_linear_baseline(X, y)
        print(f"  Linear baseline: {linear_acc:.4f}")
        
        # Grid search
        best_mlp_acc = 0.0
        best_config = None
        all_configs = []
        
        configs = list(product(
            GRID["hidden_sizes"],
            GRID["learning_rates"],
            GRID["alphas"]
        ))
        
        for i, (hs_size, lr, alpha) in enumerate(configs):
            if (i + 1) % 20 == 0:
                print(f"    Config {i+1}/{len(configs)}...")
            
            mlp_acc = run_mlp_config(X, y, hs_size, lr, alpha)
            
            config_result = {
                "hidden_sizes": list(hs_size),
                "learning_rate": lr,
                "alpha": alpha,
                "accuracy": mlp_acc,
            }
            all_configs.append(config_result)
            
            if mlp_acc > best_mlp_acc:
                best_mlp_acc = mlp_acc
                best_config = config_result
        
        # Sort by accuracy
        all_configs.sort(key=lambda x: x["accuracy"], reverse=True)
        
        advantage = best_mlp_acc - linear_acc
        
        print(f"\n  Results:")
        print(f"    Linear:     {linear_acc:.4f}")
        print(f"    Best MLP:   {best_mlp_acc:.4f} (config: {best_config})")
        print(f"    Advantage:  {advantage:+.4f} ({advantage*100:+.1f}pp)")
        print(f"    Top 5 configs:")
        for cfg in all_configs[:5]:
            print(f"      hs={cfg['hidden_sizes']}, lr={cfg['learning_rate']}, "
                  f"alpha={cfg['alpha']}: {cfg['accuracy']:.4f}")
        
        cell_result = {
            "model": model_name,
            "layer": layer,
            "position": position,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "linear_accuracy": linear_acc,
            "best_mlp_accuracy": best_mlp_acc,
            "mlp_advantage_pp": round(advantage * 100, 2),
            "best_config": best_config,
            "top_5_configs": all_configs[:5],
            "all_configs": all_configs,
        }
        results.append(cell_result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<6} {'Layer':>5} {'Pos':>4} {'N':>5} {'Linear':>8} {'Best MLP':>9} {'Adv (pp)':>9}")
    print("-" * 52)
    
    any_advantage = False
    for r in results:
        if "error" in r:
            print(f"{r['model']:<6} {r['layer']:>5} {r['position']:>4}   ERROR")
            continue
        
        adv = r["mlp_advantage_pp"]
        flag = " ***" if adv > 2.0 else ""
        if adv > 2.0:
            any_advantage = True
        
        print(f"{r['model']:<6} {r['layer']:>5} {r['position']:>4} {r['n_samples']:>5} "
              f"{r['linear_accuracy']:>8.4f} {r['best_mlp_accuracy']:>9.4f} {adv:>+8.1f}{flag}")
    
    print()
    if any_advantage:
        print("*** FINDING: Some cells show >2pp MLP advantage. Paper A.7 needs updating.")
    else:
        print("FINDING: No cell shows meaningful MLP advantage over linear probes.")
        print("Appendix A.7 null result is CONFIRMED with tuned hyperparameters.")
    
    # Save
    output_path = f"{OUTPUT_DIR}/mlp_probe_grid_search.json"
    with open(output_path, "w") as f:
        json.dump({
            "target_cells": [{"model": m, "layer": l, "position": p} for m, l, p in TARGET_CELLS],
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals] 
                      for k, vals in GRID.items()},
            "n_folds": N_FOLDS,
            "results": results,
            "conclusion": "mlp_advantage" if any_advantage else "linear_sufficient",
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
