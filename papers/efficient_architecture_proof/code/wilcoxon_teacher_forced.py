#!/usr/bin/env python3
"""Wilcoxon with teacher-forced species token extraction.

Previous Wilcoxon versions extracted the first answer token (person name),
which is predicted at prob~1.0 by all models (copied from context).
This version teacher-forces through "### [Name] is a" and extracts
log P(species_token), which requires multi-hop reasoning.

For M3 vs M5: validates the pipeline (different architectures, expect significant).
For M3 vs M6: the key test (matched architecture, tests recycling contribution).
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wilcoxon, norm
from tqdm import tqdm

# Add this script's directory to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_utils import load_model, prepare_input, load_data, get_special_ids, setup_tokenizer, find_checkpoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "experiments")

TEST_SETS = {
    "prosqa_id": f"{DATA_DIR}/prosqa_test.json",
    "7hop": f"{DATA_DIR}/ood_7hop.json",
    "8hop": f"{DATA_DIR}/ood_8hop.json",
    "dag": f"{DATA_DIR}/ood_dag.json",
    "dense": f"{DATA_DIR}/ood_dense.json",
}


def _resolve_model(subdir):
    """Resolve model checkpoint, returning path or placeholder if missing."""
    model_dir = os.path.join(CHECKPOINT_DIR, subdir)
    try:
        return find_checkpoint(model_dir)
    except FileNotFoundError:
        return os.path.join(model_dir, "checkpoint_best")


# Model configs: (checkpoint_path, feedback_mode)
MODEL_CONFIGS = {
    "m3": (_resolve_model("prosqa-coconut"), "continuous"),
    "m5": (_resolve_model("prosqa-m5-pause"), "pause_curriculum"),
    "m6": (_resolve_model("prosqa-m6-pause-multipass"), "pause_multipass"),
}


def find_species_token_index(tokenizer, answer_text):
    """Find the index of the species token in the tokenized answer.
    
    ProsQA answers are always "[Name] is a [species]."
    We want the token index of [species] in the tokenized answer.
    
    Returns:
        (prefix_token_ids, species_token_id, species_text)
        prefix_token_ids: tokens for " [Name] is a"
        species_token_id: first token of species name
        species_text: decoded species token
    """
    # Tokenize full answer with leading space (matching generation format)
    full_tokens = tokenizer.encode(" " + answer_text, add_special_tokens=False)
    
    # Parse: "[Name] is a [species]."
    # Find "is a" in the answer to locate the split point
    parts = answer_text.split(" is a ")
    if len(parts) != 2:
        return None, None, None
    
    name_part = parts[0]  # e.g. "Sally"
    species_part = parts[1].rstrip(".")  # e.g. "sterpus"
    
    # Tokenize the prefix: " [Name] is a"
    prefix_text = " " + name_part + " is a"
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    
    # Tokenize the species with leading space: " sterpus"
    species_text = " " + species_part
    species_tokens = tokenizer.encode(species_text, add_special_tokens=False)
    
    if len(species_tokens) == 0:
        return None, None, None
    
    return prefix_tokens, species_tokens[0], species_text


@torch.no_grad()
def get_species_logprob(model, tokenizer, sample, model_info, device="cpu"):
    """Extract log P(species_token | input + thoughts + ### + [Name] is a).
    
    Teacher-forces through the answer prefix to reach the discriminative
    species token position.
    
    Returns:
        (species_logprob, species_text, diagnostics_dict)
    """
    answer_text = sample["answer"].replace(",", "").strip()
    
    # Find species token
    prefix_tokens, species_token_id, species_text = find_species_token_index(
        tokenizer, answer_text
    )
    if prefix_tokens is None:
        return float("nan"), None, {"error": "parse_failed"}
    
    # Build input: question + thoughts + ### + answer_prefix
    input_ids = prepare_input(sample, tokenizer, model_info, num_thoughts=6, device=device)
    
    # Append ### separator
    separator_ids = tokenizer.encode("###", add_special_tokens=False)
    sep_tensor = torch.tensor([separator_ids], device=device)
    
    # Append answer prefix tokens
    prefix_tensor = torch.tensor([prefix_tokens], device=device)
    
    # Concatenate: input + ### + prefix
    full_input = torch.cat([input_ids, sep_tensor, prefix_tensor], dim=1)
    
    # Forward pass
    attention_mask = torch.ones_like(full_input, device=device)
    position_ids = torch.arange(full_input.shape[1], device=device).unsqueeze(0)
    labels = full_input.clone()
    
    if hasattr(model, "base_causallm"):
        # Coconut model — use forward() for multi-pass processing
        outputs = model.forward(full_input, attention_mask, labels, position_ids)
    else:
        outputs = model(full_input)
    
    # Extract logits at the last position (predicting species token)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs[2]
    
    last_logits = logits[0, -1]
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    species_lp = log_probs[species_token_id].item()
    species_prob = np.exp(species_lp)
    
    # Top prediction for diagnostics
    top_id = log_probs.argmax().item()
    top_text = tokenizer.decode([top_id])
    top_lp = log_probs[top_id].item()
    
    # Rank of species token
    rank = (log_probs >= species_lp).sum().item()
    
    diagnostics = {
        "species_text": species_text,
        "species_lp": species_lp,
        "species_prob": species_prob,
        "species_rank": rank,
        "top_prediction": top_text,
        "top_lp": top_lp,
        "n_prefix_tokens": len(prefix_tokens),
        "answer": answer_text,
    }
    
    return species_lp, species_text, diagnostics


def verify_extraction(models_to_run, device="cpu"):
    """Phase 1: Verify on 5 samples per model."""
    print("=" * 60)
    print("PHASE 1: VERIFICATION (5 samples per model)")
    print("=" * 60)
    
    tokenizer = setup_tokenizer()
    test_path = TEST_SETS["prosqa_id"]
    data = load_data(test_path)
    
    all_ok = True
    for model_name in models_to_run:
        ckpt_path, fm = MODEL_CONFIGS[model_name]
        print(f"\nLoading {model_name} from {ckpt_path}...")
        
        try:
            model, _, model_info = load_model(ckpt_path, device=device, feedback_mode=fm)
        except FileNotFoundError:
            print(f"  SKIP: checkpoint not found")
            continue
        
        for i in range(5):
            sample = data[i]
            lp, species_text, diag = get_species_logprob(
                model, tokenizer, sample, model_info, device=device
            )
            
            if diag.get("error"):
                print(f"  [{i}] {model_name}: ERROR - {diag['error']}")
                all_ok = False
                continue
            
            prob = np.exp(lp) if not np.isnan(lp) else 0.0
            print(f"  [{i}] {model_name}: species='{species_text}', lp={lp:.4f}, "
                  f"prob={prob:.6f}, rank={diag['species_rank']}, "
                  f"top='{diag['top_prediction']}' (lp={diag['top_lp']:.4f}), "
                  f"answer='{diag['answer'][:40]}'")
            
            # Check that log-prob is in reasonable range
            if lp < -20:
                print(f"    WARNING: very low log-prob, extraction may be wrong")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return all_ok


def full_run(model_a, model_b, test_sets_to_run=None, device="cpu"):
    """Phase 2: Full Wilcoxon on specified test sets."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: WILCOXON {model_a.upper()} vs {model_b.upper()} (teacher-forced species token)")
    print(f"{'='*60}")
    
    tokenizer = setup_tokenizer()
    
    if test_sets_to_run is None:
        test_sets_to_run = TEST_SETS
    
    all_logprobs = {}
    all_diagnostics = {}
    
    for model_name in [model_a, model_b]:
        ckpt_path, fm = MODEL_CONFIGS[model_name]
        print(f"\nLoading {model_name} from {ckpt_path}...")
        model, _, model_info = load_model(ckpt_path, device=device, feedback_mode=fm)
        
        for test_name, test_path in test_sets_to_run.items():
            print(f"\n  Processing {model_name}/{test_name}...")
            data = load_data(test_path)
            
            logprobs = []
            diagnostics_list = []
            n_errors = 0
            
            for i, sample in enumerate(tqdm(data, desc=f"  {model_name}/{test_name}")):
                try:
                    lp, _, diag = get_species_logprob(
                        model, tokenizer, sample, model_info, device=device
                    )
                    logprobs.append(lp)
                    diagnostics_list.append(diag)
                except Exception as e:
                    if n_errors < 5:
                        print(f"  ERROR on sample {i}: {e}")
                    n_errors += 1
                    logprobs.append(float("nan"))
                    diagnostics_list.append({"error": str(e)})
                
                if i < 3:
                    p = np.exp(logprobs[-1]) if not np.isnan(logprobs[-1]) else 0.0
                    print(f"    [{i}] lp={logprobs[-1]:.4f}, prob={p:.6f}, "
                          f"answer='{sample['answer'][:30]}'")
            
            key = f"{model_name}_{test_name}"
            all_logprobs[key] = logprobs
            all_diagnostics[key] = diagnostics_list
            
            valid = [x for x in logprobs if not np.isnan(x)]
            if valid:
                print(f"  {key}: n_valid={len(valid)}/{len(logprobs)}, "
                      f"median_lp={np.median(valid):.4f}, "
                      f"median_prob={np.exp(np.median(valid)):.6f}")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Wilcoxon tests
    print(f"\n{'='*60}")
    print(f"WILCOXON SIGNED-RANK: {model_a.upper()} vs {model_b.upper()} (species token)")
    print("=" * 60)
    
    n_tests = len(test_sets_to_run)
    wilcoxon_results = {}
    
    for test_name in test_sets_to_run.keys():
        key_a = f"{model_a}_{test_name}"
        key_b = f"{model_b}_{test_name}"
        
        lp_a = np.array(all_logprobs[key_a])
        lp_b = np.array(all_logprobs[key_b])
        
        valid_mask = ~(np.isnan(lp_a) | np.isnan(lp_b))
        a_valid = lp_a[valid_mask]
        b_valid = lp_b[valid_mask]
        n = len(a_valid)
        
        if n < 10:
            wilcoxon_results[test_name] = {"n": n, "error": "insufficient valid pairs"}
            continue
        
        differences = b_valid - a_valid
        
        # Remove zero differences for Wilcoxon
        nonzero_mask = differences != 0
        n_ties = n - nonzero_mask.sum()
        
        if nonzero_mask.sum() < 10:
            wilcoxon_results[test_name] = {
                "n": n, "n_ties": int(n_ties),
                "error": "too many ties (identical predictions)"
            }
            continue
        
        try:
            stat, p_value = wilcoxon(b_valid, a_valid, alternative="two-sided")
        except Exception as e:
            wilcoxon_results[test_name] = {"n": n, "error": str(e)}
            continue
        
        if 0 < p_value < 1:
            z_score = abs(norm.ppf(p_value / 2))
        else:
            z_score = 0.0
        effect_size_r = z_score / np.sqrt(n) if n > 0 else 0.0
        p_bonferroni = min(p_value * n_tests, 1.0)
        
        median_diff = float(np.median(differences))
        mean_diff = float(np.mean(differences))
        direction = f"{model_b}>" + model_a if median_diff > 0 else (f"{model_a}>{model_b}" if median_diff < 0 else "equal")
        
        wilcoxon_results[test_name] = {
            "n": int(n),
            "n_ties": int(n_ties),
            "n_nonzero": int(nonzero_mask.sum()),
            "median_diff": median_diff,
            "mean_diff": mean_diff,
            "std_diff": float(np.std(differences)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p_value),
            "p_bonferroni": float(p_bonferroni),
            "effect_size_r": float(effect_size_r),
            "z_score": float(z_score),
            f"{model_a}_median_lp": float(np.median(a_valid)),
            f"{model_b}_median_lp": float(np.median(b_valid)),
            f"{model_a}_mean_lp": float(np.mean(a_valid)),
            f"{model_b}_mean_lp": float(np.mean(b_valid)),
            f"{model_a}_prob_median": float(np.exp(np.median(a_valid))),
            f"{model_b}_prob_median": float(np.exp(np.median(b_valid))),
            "direction": direction,
        }
        
        sig = "YES" if p_bonferroni < 0.01 else "No"
        print(f"\n  {test_name:12s}: n={n:4d} (ties={n_ties}), W={stat:12.1f}, "
              f"p={p_value:.2e}, p_bonf={p_bonferroni:.2e}, r={effect_size_r:.3f}, "
              f"dir={direction}, sig={sig}")
        print(f"    {model_a}: median_lp={np.median(a_valid):.4f} "
              f"(prob={np.exp(np.median(a_valid)):.6f})")
        print(f"    {model_b}: median_lp={np.median(b_valid):.4f} "
              f"(prob={np.exp(np.median(b_valid)):.6f})")
        print(f"    Median diff ({model_b}-{model_a}) = {median_diff:+.4f} nats")
    
    # Save results
    output = {
        "method": "Wilcoxon signed-rank on teacher-forced species token log-probabilities",
        "description": (
            f"Teacher-force through '### [Name] is a' prefix, then extract "
            f"log P(species_token). Species token is the discriminative one "
            f"requiring multi-hop reasoning. Comparison: {model_a} vs {model_b}."
        ),
        "comparison": f"{model_a}_vs_{model_b}",
        "bonferroni_k": n_tests,
        "adjusted_alpha": 0.05 / n_tests,
        "results": wilcoxon_results,
    }
    
    output_path = f"{OUTPUT_DIR}/wilcoxon_teacher_forced_{model_a}_vs_{model_b}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Also save per-sample log-probs
    lp_path = f"{OUTPUT_DIR}/per_sample_species_logprobs_{model_a}_vs_{model_b}.json"
    with open(lp_path, "w") as f:
        json.dump(all_logprobs, f)
    print(f"Per-sample log-probs saved to {lp_path}")
    
    return wilcoxon_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="m3,m5",
                        help="Two model names separated by comma (default: m3,m5)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-sets", type=str, default=None,
                        help="Comma-separated test set names (default: all)")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    
    models = args.models.split(",")
    assert len(models) == 2, "Must specify exactly 2 models"
    
    test_sets = None
    if args.test_sets:
        test_sets = {k: TEST_SETS[k] for k in args.test_sets.split(",")}
    
    if not args.skip_verify:
        ok = verify_extraction(models, device=args.device)
        if not ok:
            print("\nVerification issues detected. Proceeding anyway...")
    
    full_run(models[0], models[1], test_sets_to_run=test_sets or TEST_SETS, device=args.device)


if __name__ == "__main__":
    main()
