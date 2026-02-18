#!/usr/bin/env python3
"""
Find the best checkpoint epoch for each trained model.

Parses training logs to extract per-epoch validation accuracy,
identifies the peak, and optionally creates a symlink `checkpoint_best`.

Usage:
    python find_best_epoch.py --log /path/to/m1_cot.log --ckpt-dir /path/to/prosqa-cot/
    python find_best_epoch.py --all --log-dir /path/to/logs/ --results-dir /path/to/results/

Output:
    Prints best epoch and accuracy for each model.
    With --link, creates checkpoint_best -> checkpoint_N symlink.
"""

import argparse
import os
import re
import sys


# Maps model name to (log filename, checkpoint subdir)
MODEL_REGISTRY = {
    "m1": ("m1_cot.log", "prosqa-cot"),
    "m2": ("m2_nocot.log", "prosqa-nocot"),
    "m3": ("m3_coconut.log", "prosqa-coconut"),
    "m4": ("m4_frozen.log", "prosqa-m4-frozen"),
    "m4b": ("m4b_shared.log", "prosqa-m4b-shared"),
}

# Regex for Meta's training output format
ACC_PATTERN = re.compile(r"Accuracy on validation set:\s+(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)")


def parse_log(log_path):
    """
    Parse a training log and extract per-epoch validation accuracy.

    Returns:
        list of (epoch, correct, total, accuracy) tuples, 1-indexed epochs.
    """
    results = []
    epoch = 0
    with open(log_path, "r") as f:
        for line in f:
            m = ACC_PATTERN.search(line)
            if m:
                epoch += 1
                correct = int(m.group(1))
                total = int(m.group(2))
                acc = float(m.group(3))
                results.append((epoch, correct, total, acc))
    return results


def find_best(results):
    """
    Find the epoch with the highest validation accuracy.

    Returns:
        (best_epoch, best_acc, final_epoch, final_acc)
    """
    if not results:
        return None, None, None, None

    best = max(results, key=lambda x: x[3])
    final = results[-1]
    return best[0], best[3], final[0], final[3]


def create_symlink(ckpt_dir, best_epoch):
    """Create checkpoint_best -> checkpoint_N symlink."""
    best_path = os.path.join(ckpt_dir, f"checkpoint_{best_epoch}")
    link_path = os.path.join(ckpt_dir, "checkpoint_best")

    if not os.path.exists(best_path):
        print(f"  WARNING: {best_path} does not exist, skipping symlink")
        return False

    if os.path.islink(link_path):
        os.unlink(link_path)
    elif os.path.exists(link_path):
        print(f"  WARNING: {link_path} exists and is not a symlink, skipping")
        return False

    os.symlink(f"checkpoint_{best_epoch}", link_path)
    print(f"  Created: checkpoint_best -> checkpoint_{best_epoch}")
    return True


def process_model(model_name, log_path, ckpt_dir=None, create_link=False):
    """Process a single model's log and optionally create symlink."""
    if not os.path.exists(log_path):
        print(f"  [{model_name}] Log not found: {log_path}")
        return None

    results = parse_log(log_path)
    if not results:
        print(f"  [{model_name}] No accuracy lines found in log")
        return None

    best_epoch, best_acc, final_epoch, final_acc = find_best(results)
    gap = best_acc - final_acc

    print(f"  [{model_name}] {final_epoch} epochs completed")
    print(f"    Best:  epoch {best_epoch}, acc = {best_acc:.4f} ({int(best_acc * results[0][2])}/{results[0][2]})")
    print(f"    Final: epoch {final_epoch}, acc = {final_acc:.4f} ({int(final_acc * results[0][2])}/{results[0][2]})")
    print(f"    Gap:   {gap:+.4f} ({gap*100:+.1f}pp)")

    if gap > 0.05:
        print(f"    *** WARNING: >5pp gap — use checkpoint_{best_epoch}, not checkpoint_{final_epoch} ***")

    if create_link and ckpt_dir:
        create_symlink(ckpt_dir, best_epoch)

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_epoch": final_epoch,
        "final_acc": final_acc,
        "gap_pp": gap * 100,
        "all_accs": [(r[0], r[3]) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Find best checkpoint epoch from training logs")
    parser.add_argument("--log", type=str, help="Path to a single training log file")
    parser.add_argument("--ckpt-dir", type=str, help="Checkpoint directory (for symlink creation)")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()),
                        help="Model name (used with --log)")
    parser.add_argument("--all", action="store_true", help="Process all models")
    parser.add_argument("--log-dir", type=str, help="Directory containing all log files (with --all)")
    parser.add_argument("--results-dir", type=str, help="Results directory containing checkpoint subdirs (with --all)")
    parser.add_argument("--link", action="store_true", help="Create checkpoint_best symlinks")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if not args.log and not args.all:
        parser.error("Must specify either --log or --all")

    results = []

    if args.all:
        log_dir = args.log_dir or "."
        results_dir = args.results_dir or "."

        print("=" * 60)
        print("Best Epoch Finder — All Models")
        print("=" * 60)

        for model_name, (log_file, ckpt_subdir) in MODEL_REGISTRY.items():
            log_path = os.path.join(log_dir, log_file)
            ckpt_dir = os.path.join(results_dir, ckpt_subdir) if args.link else None
            print(f"\n--- {model_name.upper()} ---")
            r = process_model(model_name, log_path, ckpt_dir, args.link)
            if r:
                results.append(r)

        # Summary table
        if results:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"{'Model':<6} {'Best Epoch':>10} {'Best Acc':>10} {'Final Acc':>10} {'Gap':>8}")
            print("-" * 48)
            for r in results:
                print(f"{r['model']:<6} {r['best_epoch']:>10} {r['best_acc']:>10.4f} {r['final_acc']:>10.4f} {r['gap_pp']:>+7.1f}pp")
    else:
        model_name = args.model or "unknown"
        print(f"\n--- {model_name.upper()} ---")
        r = process_model(model_name, args.log, args.ckpt_dir, args.link)
        if r:
            results.append(r)

    if args.json:
        import json
        # Strip all_accs for compact output
        for r in results:
            del r["all_accs"]
        print("\n" + json.dumps(results, indent=2))

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
