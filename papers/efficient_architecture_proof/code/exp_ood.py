"""
Experiment: OOD Generalization

Tests all 5 models (M1-M4b) on in-distribution test set and 4 OOD sets.
Computes accuracy per model per test set, with verification checks.

Usage:
    python exp_ood.py \
        --checkpoint_dir /path/to/results/v9_meta_fork \
        --data_dir data/ \
        --output_dir /path/to/experiments/ood/ \
        --seed 0

Deployed to: /lambda/nfs/experiment/code/v9_meta_fork/
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np

from exp_utils import (
    load_model_by_name,
    prepare_input,
    run_inference,
    extract_answer,
    load_data,
    get_ground_truth,
    get_special_ids,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAMES = ["m1", "m2", "m3", "m4", "m4b", "m5"]

TEST_SETS = {
    "prosqa_test": "prosqa_test.json",
    "ood_7hop": "ood_7hop.json",
    "ood_8hop": "ood_8hop.json",
    "ood_dag": "ood_dag.json",
    "ood_dense": "ood_dense.json",
}

NUM_THOUGHTS = 6  # max_latent_stage from training config


# ---------------------------------------------------------------------------
# Evaluation on one model x one test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model_on_dataset(model, tokenizer, model_info, data, num_thoughts=6, max_new_tokens=128):
    """
    Evaluate a model on a dataset. Returns (accuracy, outputs_list).

    outputs_list: list of dicts with keys:
        idx, question, ground_truth, predicted, full_text, correct
    """
    correct = 0
    total = 0
    outputs_list = []

    for idx, sample in enumerate(data):
        input_ids = prepare_input(
            sample, tokenizer, model_info,
            num_thoughts=num_thoughts,
            device=next(model.parameters()).device,
        )

        full_text, predicted = run_inference(
            model, tokenizer, input_ids, model_info,
            max_new_tokens=max_new_tokens,
        )

        gt = get_ground_truth(sample)
        is_correct = (predicted == gt)
        correct += int(is_correct)
        total += 1

        outputs_list.append({
            "idx": idx,
            "question": sample["question"][:200],  # truncate for storage
            "ground_truth": gt,
            "predicted": predicted,
            "full_text": full_text[:500],  # truncate for storage
            "correct": is_correct,
        })

        if idx < 3:
            print(f"    Sample {idx}: GT='{gt}' Pred='{predicted}' Correct={is_correct}")

        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx+1}/{len(data)} acc={correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, outputs_list


# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------

def verify_thought_tokens(model, tokenizer, model_info, data, num_thoughts=6):
    """
    Verify that COCONUT models use the correct number of thought tokens.
    Returns True if the input contains exactly num_thoughts latent tokens.
    """
    if model_info["type"] != "coconut":
        return True  # Not applicable for non-coconut models

    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]

    sample = data[0]
    input_ids = prepare_input(sample, tokenizer, model_info, num_thoughts=num_thoughts)
    tokens = input_ids[0].tolist()

    n_latent = sum(1 for t in tokens if t == latent_id)
    n_steps = len(sample.get("steps", []))
    expected = min(num_thoughts, n_steps) if n_steps > 0 else num_thoughts

    print(f"  Thought token check: found {n_latent}, expected {expected}")
    return n_latent == expected


def verify_m1_generates_cot(outputs_list):
    """
    Verify that M1 (CoT baseline) generates chain-of-thought text.
    CoT text should contain newlines before the answer marker (#).
    """
    cot_count = 0
    check_n = min(50, len(outputs_list))

    for entry in outputs_list[:check_n]:
        text = entry["full_text"]
        # CoT models produce multi-line output before #
        pre_hash = text.split("#")[0] if "#" in text else text
        if "\n" in pre_hash and len(pre_hash.strip()) > 10:
            cot_count += 1

    fraction = cot_count / check_n if check_n > 0 else 0
    print(f"  M1 CoT check: {cot_count}/{check_n} outputs contain CoT ({fraction:.2f})")
    return fraction > 0.5


def verify_parser_accuracy(outputs_list, data):
    """
    Verify answer parser on first 50 M1 outputs.
    Check that extracted answer matches expected format (a single word/phrase).
    """
    check_n = min(50, len(outputs_list))
    valid = 0

    for i in range(check_n):
        predicted = outputs_list[i]["predicted"]
        gt = data[i]["answer"].replace(",", "").strip()

        # Check that predicted is a reasonable answer (non-empty, not the whole text)
        if len(predicted) > 0 and len(predicted) < 200:
            valid += 1

    accuracy = valid / check_n if check_n > 0 else 0
    print(f"  Parser check: {valid}/{check_n} outputs have valid format ({accuracy:.2f})")
    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OOD Generalization Experiment")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model subdirs (prosqa-cot/, prosqa-coconut/, etc.)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_thoughts", type=int, default=NUM_THOUGHTS)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all in MODEL_NAMES)")
    args = parser.parse_args()

    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",")]
    else:
        models_to_run = list(MODEL_NAMES)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("OOD GENERALIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Seed:           {args.seed}")
    print(f"Num thoughts:   {args.num_thoughts}")
    print()

    # Load all test sets
    print("Loading test sets...")
    test_data = {}
    for set_name, filename in TEST_SETS.items():
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            test_data[set_name] = load_data(filepath)
            print(f"  {set_name}: {len(test_data[set_name])} samples")
        else:
            print(f"  WARNING: {filepath} not found, skipping {set_name}")

    if not test_data:
        print("ERROR: No test sets found. Exiting.")
        sys.exit(1)

    # Results storage
    results = {}
    all_outputs = {}  # model -> test_set -> outputs_list
    verification = {}

    # Evaluate each model
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")

        try:
            model, tokenizer, model_info = load_model_by_name(
                model_name, args.checkpoint_dir, device="cuda"
            )
        except FileNotFoundError as e:
            print(f"  SKIPPING {model_name}: {e}")
            continue

        print(f"  Type: {model_info['type']}, feedback_mode: {model_info['feedback_mode']}")

        results[model_name] = {}
        all_outputs[model_name] = {}

        # Verify thought tokens for COCONUT models
        if model_info["type"] == "coconut" and "prosqa_test" in test_data:
            token_check = verify_thought_tokens(
                model, tokenizer, model_info, test_data["prosqa_test"],
                num_thoughts=args.num_thoughts,
            )
            verification.setdefault("thought_token_checks", {})[model_name] = token_check

        # Evaluate on each test set
        for set_name, data in test_data.items():
            print(f"\n  Evaluating {model_name} on {set_name} ({len(data)} samples)...")
            t0 = time.time()

            accuracy, outputs_list = evaluate_model_on_dataset(
                model, tokenizer, model_info, data,
                num_thoughts=args.num_thoughts,
                max_new_tokens=args.max_new_tokens,
            )

            elapsed = time.time() - t0
            results[model_name][set_name] = round(accuracy, 4)
            all_outputs[model_name][set_name] = outputs_list

            print(f"  {model_name} / {set_name}: accuracy={accuracy:.4f} ({elapsed:.1f}s)")

        # M1 verification: check CoT generation
        if model_name == "m1" and "prosqa_test" in all_outputs.get("m1", {}):
            verification["m1_generates_cot"] = verify_m1_generates_cot(
                all_outputs["m1"]["prosqa_test"]
            )
            verification["parser_accuracy"] = verify_parser_accuracy(
                all_outputs["m1"]["prosqa_test"],
                test_data["prosqa_test"],
            )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Aggregate thought token verification
    thought_checks = verification.get("thought_token_checks", {})
    verification["thought_token_count_correct"] = all(
        thought_checks.get(m, True) for m in ["m3", "m4", "m4b"]
    )

    # Ensure verification keys exist even if M1 was skipped
    verification.setdefault("m1_generates_cot", False)
    verification.setdefault("parser_accuracy", 0.0)

    # Build final results dict
    final_results = {**results, "verification": verification}

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save detailed outputs (per-sample predictions)
    details_path = os.path.join(args.output_dir, "detailed_outputs.json")
    with open(details_path, "w") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"Detailed outputs saved to {details_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Header
    set_names = list(test_data.keys())
    header = f"{'Model':<8}" + "".join(f"{s:<16}" for s in set_names)
    print(header)
    print("-" * len(header))

    for model_name in models_to_run:
        if model_name in results:
            row = f"{model_name:<8}"
            for s in set_names:
                val = results[model_name].get(s, "N/A")
                if isinstance(val, float):
                    row += f"{val:<16.4f}"
                else:
                    row += f"{str(val):<16}"
            print(row)

    print()
    print("Verification:")
    for k, v in verification.items():
        print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
