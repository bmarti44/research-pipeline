#!/usr/bin/env python3
"""Unified experiment pipeline for M6 (and any model).

Takes a checkpoint path + feedback_mode, runs the full experiment suite:
1. ID accuracy (prosqa_test, n=500)
2. OOD accuracy (7-hop, 8-hop, DAG, dense; n=1000 each)
3. McNemar tests (vs M3 paired, vs M5 paired)
4. Corruption analysis (forward, reverse, single-position)
5. Probing (extract hidden states, linear probes, permutation tests)
6. Transplant (cross-model thought transplant)
7. Wilcoxon teacher-forced species token analysis

Usage:
    # Run everything on M6 after training:
    python run_all_m6.py --checkpoint /path/to/checkpoint --feedback-mode pause_multipass --name m6

    # Run just accuracy + McNemar (fast):
    python run_all_m6.py --checkpoint /path/to/checkpoint --feedback-mode pause_multipass --name m6 --stages accuracy,mcnemar

    # Test on M3:
    python run_all_m6.py --checkpoint /path/to/prosqa-coconut/checkpoint_49 --feedback-mode continuous --name m3_test --stages accuracy
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wilcoxon, norm
from tqdm import tqdm

# Add this script's directory to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_utils import (
    load_model, setup_tokenizer, get_special_ids,
    prepare_input, run_inference, get_ground_truth,
    get_hidden_states, get_processed_embeds, load_data, set_seed,
    find_checkpoint,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def _resolve_ref_model(subdir):
    """Resolve reference model checkpoint, returning path or placeholder if missing."""
    model_dir = os.path.join(CHECKPOINT_DIR, subdir)
    try:
        return find_checkpoint(model_dir)
    except FileNotFoundError:
        return os.path.join(model_dir, "checkpoint_best")


# Reference models for comparative analyses
REF_MODELS = {
    "m3": (_resolve_ref_model("prosqa-coconut"), "continuous"),
    "m5": (_resolve_ref_model("prosqa-m5-pause"), "pause_curriculum"),
}

TEST_SETS = {
    "prosqa_test": f"{DATA_DIR}/prosqa_test.json",
    "ood_7hop": f"{DATA_DIR}/ood_7hop.json",
    "ood_8hop": f"{DATA_DIR}/ood_8hop.json",
    "ood_dag": f"{DATA_DIR}/ood_dag.json",
    "ood_dense": f"{DATA_DIR}/ood_dense.json",
}

ALL_STAGES = ["accuracy", "mcnemar", "corruption", "probing", "transplant", "wilcoxon"]


# ===========================================================================
# Stage 1: Accuracy (ID + OOD)
# ===========================================================================

@torch.no_grad()
def run_accuracy(model, tokenizer, model_info, output_dir, device):
    """Evaluate on all test sets. Returns dict of per-sample correctness."""
    print("\n" + "="*60)
    print("STAGE 1: ACCURACY (ID + OOD)")
    print("="*60)
    
    results = {}
    per_sample = {}
    
    for set_name, set_path in TEST_SETS.items():
        if not os.path.exists(set_path):
            print(f"  SKIP {set_name}: file not found")
            continue
        
        data = load_data(set_path)
        correct = 0
        correctness = []
        
        for i, sample in enumerate(tqdm(data, desc=f"  {set_name}")):
            input_ids = prepare_input(sample, tokenizer, model_info, num_thoughts=6, device=device)
            _, predicted = run_inference(model, tokenizer, input_ids, model_info)
            gt = get_ground_truth(sample)
            is_correct = (predicted == gt)
            correct += int(is_correct)
            correctness.append(int(is_correct))
            
            if i < 3:
                print(f"    [{i}] GT='{gt}' Pred='{predicted}' {'OK' if is_correct else 'WRONG'}")
        
        acc = correct / len(data)
        results[set_name] = {"accuracy": round(acc, 4), "correct": correct, "total": len(data)}
        per_sample[set_name] = correctness
        print(f"  {set_name}: {correct}/{len(data)} = {acc:.4f}")
    
    # Save
    with open(os.path.join(output_dir, "accuracy.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "per_sample_correctness.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    
    return results, per_sample


# ===========================================================================
# Stage 2: McNemar tests
# ===========================================================================

def run_mcnemar(target_per_sample, target_name, output_dir, device):
    """Run McNemar's exact test vs M3 and M5."""
    from scipy.stats import binom_test
    
    print("\n" + "="*60)
    print(f"STAGE 2: McNEMAR ({target_name} vs M3, {target_name} vs M5)")
    print("="*60)
    
    # Load reference per-sample correctness
    ref_correctness = {}
    for ref_name in ["m3", "m5"]:
        ref_path = f"{CHECKPOINT_DIR}/experiments/per_sample_correctness.json"
        if os.path.exists(ref_path):
            ref_data = json.load(open(ref_path))
            # The existing file has model-level keys
            for set_name in TEST_SETS:
                key = f"{ref_name}_{set_name}"
                if key in ref_data:
                    ref_correctness.setdefault(ref_name, {})[set_name] = ref_data[key]
    
    # If reference data not in expected format, load from OOD results
    if not ref_correctness:
        print("  Loading reference models for paired comparison...")
        tokenizer = setup_tokenizer()
        for ref_name, (ref_ckpt, ref_fm) in REF_MODELS.items():
            print(f"  Loading {ref_name}...")
            ref_model, _, ref_info = load_model(ref_ckpt, device=device, feedback_mode=ref_fm)
            ref_correctness[ref_name] = {}
            
            for set_name, set_path in TEST_SETS.items():
                if set_name not in target_per_sample:
                    continue
                data = load_data(set_path)
                correctness = []
                for i, sample in enumerate(tqdm(data, desc=f"  {ref_name}/{set_name}")):
                    input_ids = prepare_input(sample, tokenizer, ref_info, num_thoughts=6, device=device)
                    _, predicted = run_inference(ref_model, tokenizer, input_ids, ref_info)
                    gt = get_ground_truth(sample)
                    correctness.append(int(predicted == gt))
                ref_correctness[ref_name][set_name] = correctness
            
            del ref_model
            torch.cuda.empty_cache()
    
    # Run McNemar tests
    mcnemar_results = {}
    n_tests = 0
    
    for ref_name in ["m3", "m5"]:
        if ref_name not in ref_correctness:
            continue
        
        for set_name in TEST_SETS:
            if set_name not in target_per_sample or set_name not in ref_correctness.get(ref_name, {}):
                continue
            
            target_c = np.array(target_per_sample[set_name])
            ref_c = np.array(ref_correctness[ref_name][set_name])
            
            n = min(len(target_c), len(ref_c))
            target_c = target_c[:n]
            ref_c = ref_c[:n]
            
            # 2x2 contingency table
            both_correct = int(np.sum((target_c == 1) & (ref_c == 1)))
            target_only = int(np.sum((target_c == 1) & (ref_c == 0)))
            ref_only = int(np.sum((target_c == 0) & (ref_c == 1)))
            both_wrong = int(np.sum((target_c == 0) & (ref_c == 0)))
            
            # McNemar's exact test (binomial)
            b = target_only
            c = ref_only
            n_discordant = b + c
            
            if n_discordant == 0:
                p_value = 1.0
            else:
                p_value = 2 * min(
                    sum(binom_test(b, n_discordant, 0.5, alternative="two-sided")),
                    1.0
                ) if False else binom_test(b, n_discordant, 0.5)  # Removed deprecated call
                # Use scipy exact test
                from scipy.stats import binomtest
                result = binomtest(b, n_discordant, 0.5, alternative="two-sided")
                p_value = result.pvalue
            
            n_tests += 1
            
            # Effect size: odds ratio
            odds_ratio = (b / c) if c > 0 else float("inf")
            
            # Accuracy difference with CI
            target_acc = target_c.mean()
            ref_acc = ref_c.mean()
            diff = target_acc - ref_acc
            
            # Wilson CI for difference
            se = np.sqrt((b + c - (b - c)**2 / n) / n**2) if n > 0 else 0
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
            
            key = f"{target_name}_vs_{ref_name}_{set_name}"
            mcnemar_results[key] = {
                "n": n,
                "target_acc": round(float(target_acc), 4),
                "ref_acc": round(float(ref_acc), 4),
                "diff_pp": round(diff * 100, 2),
                "ci_95": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
                "contingency": {
                    "both_correct": both_correct,
                    f"{target_name}_only": target_only,
                    f"{ref_name}_only": ref_only,
                    "both_wrong": both_wrong,
                },
                "discordant": n_discordant,
                "p_value": float(p_value),
                "odds_ratio": float(odds_ratio) if odds_ratio != float("inf") else "inf",
            }
            
            print(f"  {key}: p={p_value:.4f}, diff={diff*100:+.1f}pp, "
                  f"CI=[{ci_low*100:+.1f}, {ci_high*100:+.1f}], "
                  f"discordant={n_discordant}")
    
    # Apply Bonferroni correction
    for key, res in mcnemar_results.items():
        res["p_bonferroni"] = min(res["p_value"] * n_tests, 1.0)
    
    with open(os.path.join(output_dir, "mcnemar.json"), "w") as f:
        json.dump({"n_tests": n_tests, "results": mcnemar_results}, f, indent=2)
    
    return mcnemar_results


# ===========================================================================
# Stage 3: Corruption
# ===========================================================================

@torch.no_grad()
def run_corruption(model, tokenizer, model_info, output_dir, device):
    """Run graduated corruption analysis."""
    print("\n" + "="*60)
    print("STAGE 3: CORRUPTION ANALYSIS")
    print("="*60)
    
    data = load_data(TEST_SETS["prosqa_test"])
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    
    results = {"forward": {}, "reverse": {}, "single_position": {}}
    n_samples = len(data)
    
    # For each corruption mode and level
    for corrupt_mode in ["forward", "reverse", "single_position"]:
        for k in range(6):  # 6 thought positions
            correct = 0
            
            for i, sample in enumerate(tqdm(data, desc=f"  {corrupt_mode} k={k}", leave=False)):
                input_ids = prepare_input(sample, tokenizer, model_info, num_thoughts=6, device=device)
                
                # Get clean embeddings
                processed = get_processed_embeds(model, input_ids)
                
                # Corrupt
                tokens = input_ids[0].tolist()
                thought_positions = [j for j, t in enumerate(tokens) if t == latent_id]
                
                if len(thought_positions) < 6:
                    continue
                
                # Determine which positions to corrupt
                if corrupt_mode == "forward":
                    corrupt_positions = thought_positions[:k+1]
                elif corrupt_mode == "reverse":
                    corrupt_positions = thought_positions[5-k:]
                else:  # single_position
                    corrupt_positions = [thought_positions[k]]
                
                # Replace with random noise (same norm)
                corrupted = processed.clone()
                for pos in corrupt_positions:
                    orig_norm = corrupted[0, pos].norm()
                    noise = torch.randn_like(corrupted[0, pos])
                    noise = noise / noise.norm() * orig_norm
                    corrupted[0, pos] = noise
                
                # Generate with corrupted embeddings
                new_inputs_embeds = corrupted
                outputs = model.base_causallm(inputs_embeds=new_inputs_embeds)
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                generated = [next_token]
                
                for _ in range(63):
                    new_embed = model.embedding(torch.tensor(generated[-1], device=device)).view(1, 1, -1)
                    new_inputs_embeds = torch.cat([new_inputs_embeds, new_embed], dim=1)
                    outputs = model.base_causallm(inputs_embeds=new_inputs_embeds)
                    next_token = torch.argmax(outputs.logits[0, -1]).item()
                    if next_token == tokenizer.eos_token_id:
                        break
                    generated.append(next_token)
                
                full_text = tokenizer.decode(input_ids[0].tolist() + generated, skip_special_tokens=True)
                predicted = full_text.split("#")[-1].replace(",", "").strip()
                gt = get_ground_truth(sample)
                correct += int(predicted == gt)
            
            acc = correct / n_samples
            results[corrupt_mode][str(k)] = round(acc, 4)
            print(f"  {corrupt_mode} k={k}: {correct}/{n_samples} = {acc:.4f}")
    
    with open(os.path.join(output_dir, "corruption.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===========================================================================
# Stage 4: Probing (extract hidden states + linear probes + permutation tests)
# ===========================================================================

@torch.no_grad()
def run_probing(model, tokenizer, model_info, name, output_dir, device):
    """Extract hidden states and run probing analysis."""
    print("\n" + "="*60)
    print("STAGE 4: PROBING")
    print("="*60)
    
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    
    data = load_data(TEST_SETS["prosqa_test"])
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    
    from exp_utils import get_step_types
    
    # Extract hidden states
    print("  Extracting hidden states...")
    hidden_by_pos = {t: [] for t in range(6)}
    labels_by_pos = {t: [] for t in range(6)}
    
    for idx, sample in enumerate(tqdm(data, desc="  Extracting")):
        steps = sample.get("steps", [])
        step_types = get_step_types(sample)
        if len(steps) == 0:
            continue
        
        input_ids = prepare_input(sample, tokenizer, model_info, num_thoughts=6, device=device)
        tokens = input_ids[0].tolist()
        thought_positions = [i for i, t in enumerate(tokens) if t == latent_id]
        
        all_hidden = get_hidden_states(model, tokenizer, input_ids, model_info)
        
        if not torch.isfinite(all_hidden).all():
            continue
        
        for t in range(min(len(thought_positions), 6)):
            pos = thought_positions[t]
            h = all_hidden[:, pos, :].cpu().float().numpy()
            label = step_types[t] if t < len(step_types) else "UNKNOWN"
            hidden_by_pos[t].append(h)
            labels_by_pos[t].append(label)
    
    # Convert to numpy
    for t in range(6):
        if hidden_by_pos[t]:
            hidden_by_pos[t] = np.stack(hidden_by_pos[t])
    
    # Save cached hidden states
    np.savez(os.path.join(output_dir, f"{name}_hidden_states.npz"),
             **{str(t): hidden_by_pos[t] for t in range(6) if isinstance(hidden_by_pos[t], np.ndarray)})
    with open(os.path.join(output_dir, f"{name}_labels.json"), "w") as f:
        json.dump({str(t): labels_by_pos[t] for t in range(6)}, f)
    
    print(f"  Samples per position: {[len(labels_by_pos[t]) for t in range(6)]}")
    
    # Run linear probes + permutation tests (2000 perms)
    print("  Running probes + permutation tests...")
    N_PERMS = 2000
    BONFERRONI_THRESHOLD = 0.05 / 78
    
    probe_accuracy = np.zeros((13, 6))
    perm_p_values = np.ones((13, 6))
    significant_cells = []
    
    for t in range(6):
        if not isinstance(hidden_by_pos[t], np.ndarray) or len(labels_by_pos[t]) < 10:
            continue
        
        le = LabelEncoder()
        y = le.fit_transform(labels_by_pos[t])
        
        if len(np.unique(y)) < 2:
            continue
        
        for layer in range(13):
            X = hidden_by_pos[t][:, layer, :]
            
            # Cross-validated accuracy
            min_class = min(np.bincount(y))
            n_folds = min(5, min_class)
            if n_folds < 2:
                continue
            
            clf = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeClassifier(alpha=1.0, random_state=42))])
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            
            acc = float(np.mean(scores))
            probe_accuracy[layer, t] = acc
            
            # Permutation test (single split for speed)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(sss.split(X, y))
            
            scaler = StandardScaler().fit(X[train_idx])
            X_train_s = scaler.transform(X[train_idx])
            X_test_s = scaler.transform(X[test_idx])
            
            rng = np.random.RandomState(42)
            count_geq = 0
            
            for _ in range(N_PERMS):
                y_perm = rng.permutation(y)
                if len(np.unique(y_perm[train_idx])) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    perm_clf = RidgeClassifier(alpha=1.0, random_state=42)
                    perm_clf.fit(X_train_s, y_perm[train_idx])
                    perm_acc = perm_clf.score(X_test_s, y_perm[test_idx])
                if perm_acc >= acc:
                    count_geq += 1
            
            p_val = (count_geq + 1) / (N_PERMS + 1)
            perm_p_values[layer, t] = p_val
            
            if p_val < BONFERRONI_THRESHOLD:
                significant_cells.append({"layer": layer, "position": t, "accuracy": acc, "p_value": p_val})
        
        print(f"    Position {t}: best layer={np.argmax(probe_accuracy[:, t])}, "
              f"best acc={probe_accuracy[:, t].max():.4f}")
    
    n_sig = len(significant_cells)
    print(f"  Significant cells (Bonferroni {BONFERRONI_THRESHOLD:.6f}): {n_sig}/78")
    
    # Save
    probe_results = {
        "linear_probe_accuracy": probe_accuracy.tolist(),
        "permutation_p_values": perm_p_values.tolist(),
        "significant_cells": significant_cells,
        "n_significant_cells": n_sig,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "n_samples_per_position": {str(t): len(labels_by_pos[t]) for t in range(6)},
        "n_permutations": N_PERMS,
        "peak_layer": int(np.unravel_index(probe_accuracy.argmax(), probe_accuracy.shape)[0]),
        "peak_position": int(np.unravel_index(probe_accuracy.argmax(), probe_accuracy.shape)[1]),
        "peak_accuracy": float(probe_accuracy.max()),
    }
    
    with open(os.path.join(output_dir, f"{name}_linear_perm.json"), "w") as f:
        json.dump(probe_results, f, indent=2)
    
    return probe_results


# ===========================================================================
# Stage 5: Transplant
# ===========================================================================

@torch.no_grad()
def run_transplant(model, tokenizer, model_info, name, output_dir, device):
    """Run cross-problem thought transplant experiment."""
    print("\n" + "="*60)
    print("STAGE 5: TRANSPLANT")
    print("="*60)
    
    data = load_data(TEST_SETS["prosqa_test"])
    special_ids = get_special_ids(tokenizer)
    latent_id = special_ids["latent_id"]
    
    N_PAIRS = 200
    rng = np.random.RandomState(42)
    
    # Generate random pairs
    indices = list(range(len(data)))
    pairs = [(rng.choice(indices), rng.choice(indices)) for _ in range(N_PAIRS)]
    # Ensure no self-pairs
    pairs = [(a, b) for a, b in pairs if a != b][:N_PAIRS]
    
    results = {"matched": 0, "unmatched": 0, "total": 0}
    
    for source_idx, target_idx in tqdm(pairs, desc="  Transplant"):
        source_sample = data[source_idx]
        target_sample = data[target_idx]
        
        # Get source embeddings
        source_input = prepare_input(source_sample, tokenizer, model_info, num_thoughts=6, device=device)
        source_embeds = get_processed_embeds(model, source_input)
        
        # Get target embeddings
        target_input = prepare_input(target_sample, tokenizer, model_info, num_thoughts=6, device=device)
        target_embeds = get_processed_embeds(model, target_input)
        
        # Find thought positions
        source_tokens = source_input[0].tolist()
        target_tokens = target_input[0].tolist()
        source_thought_pos = [i for i, t in enumerate(source_tokens) if t == latent_id]
        target_thought_pos = [i for i, t in enumerate(target_tokens) if t == latent_id]
        
        if len(source_thought_pos) < 6 or len(target_thought_pos) < 6:
            continue
        
        # Transplant: inject source thoughts into target
        transplanted = target_embeds.clone()
        for s_pos, t_pos in zip(source_thought_pos, target_thought_pos):
            transplanted[0, t_pos] = source_embeds[0, s_pos]
        
        # Generate with transplanted embeddings
        outputs = model.base_causallm(inputs_embeds=transplanted)
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        generated = [next_token]
        
        new_inputs_embeds = transplanted
        for _ in range(63):
            new_embed = model.embedding(torch.tensor(generated[-1], device=device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat([new_inputs_embeds, new_embed], dim=1)
            outputs = model.base_causallm(inputs_embeds=new_inputs_embeds)
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)
        
        full_text = tokenizer.decode(target_tokens + generated, skip_special_tokens=True)
        predicted = full_text.split("#")[-1].replace(",", "").strip()
        
        gt_target = get_ground_truth(target_sample)
        gt_source = get_ground_truth(source_sample)
        
        results["total"] += 1
        if predicted == gt_target:
            results["matched"] += 1  # Solved target with source thoughts (unexpected)
        if predicted == gt_source:
            results["unmatched"] += 1  # Solved source problem (thoughts carried info)
    
    results["matched_rate"] = round(results["matched"] / max(results["total"], 1), 4)
    results["unmatched_rate"] = round(results["unmatched"] / max(results["total"], 1), 4)
    
    print(f"  Matched (target correct with source thoughts): {results['matched']}/{results['total']} = {results['matched_rate']}")
    print(f"  Unmatched (source correct from thoughts): {results['unmatched']}/{results['total']} = {results['unmatched_rate']}")
    
    with open(os.path.join(output_dir, f"{name}_transplant.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ===========================================================================
# Stage 6: Wilcoxon teacher-forced species token
# ===========================================================================

@torch.no_grad()
def run_wilcoxon(model, tokenizer, model_info, name, output_dir, device):
    """Extract teacher-forced species token log-probs for Wilcoxon analysis."""
    print("\n" + "="*60)
    print(f"STAGE 6: WILCOXON SPECIES TOKEN EXTRACTION ({name})")
    print("="*60)
    
    from wilcoxon_teacher_forced import find_species_token_index, get_species_logprob
    
    all_logprobs = {}
    
    for test_name, test_path in TEST_SETS.items():
        data = load_data(test_path)
        logprobs = []
        
        for i, sample in enumerate(tqdm(data, desc=f"  {test_name}")):
            try:
                lp, _, _ = get_species_logprob(model, tokenizer, sample, model_info, device=device)
                logprobs.append(lp)
            except Exception as e:
                logprobs.append(float("nan"))
        
        all_logprobs[f"{name}_{test_name}"] = logprobs
        valid = [x for x in logprobs if not np.isnan(x)]
        if valid:
            print(f"  {test_name}: median_lp={np.median(valid):.4f}, "
                  f"median_prob={np.exp(np.median(valid)):.6f}")
    
    with open(os.path.join(output_dir, f"{name}_species_logprobs.json"), "w") as f:
        json.dump(all_logprobs, f)
    
    return all_logprobs


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified experiment pipeline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feedback-mode", type=str, required=True,
                        choices=["continuous", "pause_curriculum", "pause_multipass", "frozen", "learned_shared"])
    parser.add_argument("--name", type=str, required=True, help="Model name for output files")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stages", type=str, default="all",
                        help=f"Comma-separated stages to run (default: all). Options: {','.join(ALL_STAGES)}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")
    
    output_dir = args.output_dir or f"{CHECKPOINT_DIR}/experiments/{args.name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"EXPERIMENT PIPELINE: {args.name}")
    print("=" * 70)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Feedback mode: {args.feedback_mode}")
    print(f"Stages:        {stages}")
    print(f"Output:        {output_dir}")
    print(f"Device:        {args.device}")
    
    # Load model
    print(f"\nLoading model...")
    model, tokenizer, model_info = load_model(
        args.checkpoint, device=args.device, feedback_mode=args.feedback_mode
    )
    
    # Run stages
    all_results = {}
    
    if "accuracy" in stages:
        acc_results, per_sample = run_accuracy(model, tokenizer, model_info, output_dir, args.device)
        all_results["accuracy"] = acc_results
    
    if "mcnemar" in stages:
        if "accuracy" not in stages:
            # Need per_sample from accuracy stage
            print("  McNemar requires accuracy stage. Running accuracy first...")
            _, per_sample = run_accuracy(model, tokenizer, model_info, output_dir, args.device)
        mcnemar_results = run_mcnemar(per_sample, args.name, output_dir, args.device)
        all_results["mcnemar"] = mcnemar_results
    
    if "corruption" in stages:
        corruption_results = run_corruption(model, tokenizer, model_info, output_dir, args.device)
        all_results["corruption"] = corruption_results
    
    if "probing" in stages:
        probing_results = run_probing(model, tokenizer, model_info, args.name, output_dir, args.device)
        all_results["probing"] = probing_results
    
    if "transplant" in stages:
        transplant_results = run_transplant(model, tokenizer, model_info, args.name, output_dir, args.device)
        all_results["transplant"] = transplant_results
    
    if "wilcoxon" in stages:
        wilcoxon_results = run_wilcoxon(model, tokenizer, model_info, args.name, output_dir, args.device)
        all_results["wilcoxon"] = wilcoxon_results
    
    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "model": args.name,
            "checkpoint": args.checkpoint,
            "feedback_mode": args.feedback_mode,
            "stages_run": stages,
            "results": {k: v for k, v in all_results.items() if k != "wilcoxon"},
        }, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to {output_dir}/")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
